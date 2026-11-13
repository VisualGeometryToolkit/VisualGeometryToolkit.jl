# =============================================================================
# vgio/capture_settings/v2.0, and a calibration_record's `device`
# =============================================================================

@testitem "vgio/capture_settings/v2.0" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: CaptureSettings, ExposureSettings, LensSettings, DeviceIdentity, Attributed
    using JSON: JSON
    s = parse_unit("s")
    e1 = ExposureSettings(Float64(Float32(1 / 60)) * s, 4.0, 100)
    e2 = ExposureSettings(Float64(Float32(1 / 60)) * s, 4.0, 200)
    cs = CaptureSettings(
        Dict(
            1 => (exposure = e1, lens_settings = LensSettings(14.0mm)),
            4 => (exposure = e2, lens_settings = LensSettings(14.0mm, VgtkIO.read_length(1200.0, mm))),
        );
        body = DeviceIdentity(; make = "Sony", model = "ILME-FX3"),
        lens = DeviceIdentity(; make = "SIGMA", model = "14mm F1.8 DG HSM | Art"),
    )
    x = Attributed(cs, (; identity_sources = (body = "exif", lens = "config"), exif_lens = DeviceIdentity(; model = "DT 14mm F2.8 SAM"), sensor = "cam1"))
    fixture = joinpath(@__DIR__, "fixtures", "capture_settings", "capture_settings.json")

    mktempdir() do dir
        out = joinpath(dir, "cs.json")
        save(out, x)
        @test read(out, String) == read(fixture, String)               # the fixture is what the writer writes
        @test VgtkIO.validate(out) == "capture_settings"
        y = load(CaptureSettings, fixture)
        @test getfield(y, :value) == cs
        @test y.identity_sources == (body = "exif", lens = "config") && y.exif_lens == DeviceIdentity(; model = "DT 14mm F2.8 SAM")
        @test y.sensor == "cam1" && getfield(y, :attributes) == getfield(x, :attributes)
        d = JSON.parsefile(fixture)
        @test d["schema"] == "vgio/capture_settings/v2.0" && d["sensor"] == "cam1" && d["units"]["length"] == "mm" && d["units"]["time"] == "s"
        @test d["device"] == Dict("body" => Dict("make" => "Sony", "model" => "ILME-FX3"), "lens" => Dict("make" => "SIGMA", "model" => "14mm F1.8 DG HSM | Art"))
        @test d["exif_lens"] == Dict("model" => "DT 14mm F2.8 SAM") && [f["frame_index"] for f in d["frames"]] == [1, 4]
        save(joinpath(dir, "again.json"), y)
        @test read(joinpath(dir, "again.json"), String) == read(fixture, String)
        # in another length unit: the same value
        save(joinpath(dir, "m.json"), x; length_unit = m)
        @test getfield(load(CaptureSettings, joinpath(dir, "m.json")), :value) == cs
        # an EXIF-only lens keeps no claim beside it
        z = Attributed(cs, (; identity_sources = (body = "exif", lens = "exif")))
        save(joinpath(dir, "z.json"), z)
        @test keys(getfield(load(CaptureSettings, joinpath(dir, "z.json")), :attributes)) == (:identity_sources,)
        @test !haskey(JSON.parsefile(joinpath(dir, "z.json")), "exif_lens") && !haskey(JSON.parsefile(joinpath(dir, "z.json")), "sensor")

        # refusals: no sources, a source of neither kind, an overridden claim on an EXIF lens, an attribute of no kind's
        @test_throws ArgumentError save(joinpath(dir, "bare.json"), cs)
        @test_throws ArgumentError save(joinpath(dir, "bad.json"), Attributed(cs, (; identity_sources = (body = "exif", lens = "guess"))))
        @test_throws ArgumentError save(
            joinpath(dir, "bad.json"),
            Attributed(cs, (; identity_sources = (body = "exif", lens = "exif"), exif_lens = DeviceIdentity(; model = "x"))),
        )
        @test_throws ArgumentError save(joinpath(dir, "bad.json"), Attributed(cs, (; identity_sources = (body = "exif", lens = "exif"), camera = "cam1")))
        @test !isfile(joinpath(dir, "bad.json"))
        # on read: a duplicate frame index, an unknown source, a missing required key or envelope key, `exif_lens`
        # beside an EXIF lens, an empty sensor name
        doc = JSON.parsefile(fixture)
        function refused(edit!)
            d = deepcopy(doc); edit!(d)
            p = joinpath(dir, "edited.json"); open(io -> JSON.print(io, d), p, "w")
            return try load(CaptureSettings, p); nothing catch err; err end
        end
        @test refused(d -> push!(d["frames"], d["frames"][1])) isa ArgumentError
        @test refused(d -> (d["identity_sources"]["body"] = "guess")) isa ArgumentError
        @test refused(d -> delete!(d["frames"][1], "iso")) isa ArgumentError
        @test refused(d -> delete!(d["units"], "time")) isa ArgumentError
        @test refused(d -> (d["units"]["time"] = "ms")) isa ArgumentError
        @test refused(d -> (d["identity_sources"]["lens"] = "exif")) isa ArgumentError
        @test refused(d -> (d["sensor"] = "")) isa ArgumentError
        @test refused(d -> (d["device"]["body"]["source"] = "exif")) isa ArgumentError
        @test refused(identity) === nothing
        @test refused(d -> (d["frames"][1]["exposure_time"] = 0)) isa ArgumentError   # Core's positivity
    end
end

@testitem "vgio/cameras/v6.0: a calibration_record's device" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, Attributed, DeviceIdentity
    using JSON: JSON
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(2400.0, 0, 0, 0, 2400.0, 0, 2120.0, 1416.0, 1))
    bc = BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap), Size2(4240, 2832))
    device = (body = DeviceIdentity(; make = "Sony", model = "ILME-FX3", serial = "1234"), lens = DeviceIdentity(; model = "14mm F1.8 DG HSM | Art"))
    mktempdir() do dir
        p = joinpath(dir, "cameras.json")
        save(p, Dict(:cam => Attributed(bc, (; calibration_record = (; tool = "t", device)))))
        c = load(Dict{Symbol, BoundedCamera}, p)[:cam]
        @test c.calibration_record == (; tool = "t", device)
        @test VgtkIO.validate(p) == "cameras"
        # the ONE device-identity encoding: the camera's `calibration_record.device` is capture_settings' `device`
        @test JSON.parsefile(p)["cameras"][1]["calibration_record"]["device"] ==
              Dict("body" => Dict("make" => "Sony", "model" => "ILME-FX3", "serial" => "1234"), "lens" => Dict("model" => "14mm F1.8 DG HSM | Art"))
        # a block without a device (every file before it) reads as it did
        save(p, Dict(:cam => Attributed(bc, (; calibration_record = (; tool = "t")))))
        @test load(Dict{Symbol, BoundedCamera}, p)[:cam].calibration_record == (; tool = "t")
        @test_throws ArgumentError save(p, Dict(:cam => Attributed(bc, (; calibration_record = (; tool = "t", device = (; body = "Sony"))))))
    end
end
