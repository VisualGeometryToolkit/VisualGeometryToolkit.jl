# =============================================================================
# IO 0.10's envelope, on a file of every kind: every key always written, a
# file missing any refused by name, a higher minor tolerated, an unknown key at
# the reader's own minor refused, a later major refused, the generator naming
# the calling package (`with_caller`); and every kind's round trip, exact by
# type and byte for byte on a rewrite
# =============================================================================

@testsetup module EnvelopeCases
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: load, save
    using VisualGeometryToolkitCore: mm, m, px, CameraModel, CameraCalibrationMat, KannalaBrandt, Size2, EuclideanMap,
        BoundedCamera, PosedCamera, Rig, PosedTarget, PlanarTarget, PointTarget, AbstractTarget, Trajectory, TargetPoses,
        TargetHomographies, HomographyMap, HomographyMat, TargetObservations, CaptureSettings, ExposureSettings, LensSettings,
        DeviceIdentity, Attributed, AbstractVignetting, RadialVignetting, LogVignetting, EnvironmentLight, SphericalHarmonicMap,
        GaussianBlur, PixelFrame, WorldFrame, CameraFrame, BodyFrame, TargetFrame, ViewFrame, Uncertain, CovMat, relabel,
        correspondences, withattrs, with_control_points, with_unit, sign_normalize, parse_unit
    using StaticArrays: SVector, SMatrix
    using Rotations: QuatRotation
    using JSON: JSON
    export KINDS, write_every_kind, tomutable, refusal

    ml(t) = SVector{length(t)}(map(x -> VgtkIO.read_length(x, mm), Tuple(t)))
    pose(t) = EuclideanMap(QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false), ml(t))

    # one value of every kind, with the type `load` takes
    function values_()
        h = repeat("ab", 32)
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 320.0, 240.0, 1))
        cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap), Size2(640, 480))
        ident = EuclideanMap(one(QuatRotation{Float64}), ml((0.0, 0.0, 0.0)))
        pts = [ml((0.0, 0.0)), ml((62.8, 0.0)), ml((0.0, 40.25))]
        sec = parse_unit("s")
        H = HomographyMap(sign_normalize(HomographyMat{Float64}((812.3, 37.9, 0.0001, -41.7, 805.1, -0.0002, 1210.25, 640.5, 0.6))))
        return [
            "transforms" => (Vector{EuclideanMap}, [relabel(BodyFrame{:camera_rig}, CameraFrame, pose((251.9, -144.4, 904.1)))]),
            "cameras" => (Dict{Symbol, BoundedCamera}, Dict(:cam => cam)),
            "rig" => (Rig, Rig{ViewFrame{:a}}((BoundedCamera(PosedCamera(model_of(cam), relabel(ViewFrame{:a}, ViewFrame{:a}, ident)), Size2(640, 480)),))),
            "target" => (AbstractTarget, PointTarget([ml((1.0, 2.0, 3.0)), ml((-13.9, 0.0, 5.5))])),
            "trajectory" => (Trajectory, Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => pose((1.0, 2.0, 900.0)));
                images = Dict(1 => "a.png"), times = Dict(1 => 0.5), time_origin = "2026-05-27T09:30:00+02:00")),
            "target_poses" => (TargetPoses, TargetPoses{CameraFrame}(Dict(1 => Dict(:a => pose((1.0, 2.0, 900.0)))); images = Dict(1 => "a.png"), sensors = "cam")),
            "target_homographies" => (TargetHomographies, TargetHomographies{PixelFrame{:colmap}}(Dict(3 => Dict(:a => H)); images = Dict(3 => "a.png"), sensors = "cam", targets = [h])),
            "observations" => (TargetObservations, TargetObservations{PixelFrame{:julia}}(
                Dict(2 => Dict(:a => withattrs(correspondences([1, 3], [SVector(640.25, 480.5), SVector(1.0, 2.0)]); weights = [1.0, 0.5])));
                images = Dict(2 => "a.png"), sensors = "cam", targets = [h])),
            "capture_settings" => (CaptureSettings, Attributed(
                CaptureSettings(Dict(1 => (exposure = ExposureSettings(0.01 * sec, 4.0, 100), lens_settings = LensSettings(VgtkIO.read_length(14.0, mm))));
                    body = DeviceIdentity(; make = "Sony", model = "ILME-FX3"), lens = DeviceIdentity(; model = "14mm")),
                (; identity_sources = (body = "exif", lens = "exif"), sensor = "cam"))),
            "vignetting" => (AbstractVignetting, LogVignetting(with_control_points(RadialVignetting(6, SVector(320.0, 240.0), 400.0), [0.5 + 0.01i for i in 1:6]))),
            "environment_light" => (EnvironmentLight, with_unit(EnvironmentLight(SphericalHarmonicMap(ones(3, 4), 1), SVector(0.1, -0.2, 0.3), 1.5), m)),
            "blur" => (Dict{Symbol, GaussianBlur}, Dict(:R => GaussianBlur(relabel(PixelFrame{:colmap}, CovMat(SMatrix{2, 2, Float64}(0.5, 0.1, 0.1, 0.4)))))),
        ]
    end
    model_of(c) = VgtkCore.model(c)
    const KINDS = first.(values_())

    # Write one file of every kind into `dir`: kind => (path, type, value).
    function write_every_kind(dir)
        out = Dict{String, Tuple{String, Any, Any}}()
        for (kind, (T, v)) in values_()
            p = joinpath(dir, "$(kind).json")
            save(p, v)
            out[kind] = (p, T, v)
        end
        return out
    end

    tomutable(x::AbstractDict) = Dict{String, Any}(String(k) => tomutable(v) for (k, v) in x)
    tomutable(x::AbstractVector) = Any[tomutable(v) for v in x]
    tomutable(x) = x

    # `nothing` if `load(T, path)` reads the edited tree `t`, else the refusal's message (an ArgumentError, required)
    function refusal(dir, name, T, t)
        p = joinpath(dir, "edited_$(name).json")
        write(p, JSON.json(t))
        return try
            load(T, p)
            nothing
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
    end
end

@testitem "envelope: every kind writes every key, in order, the file's own" setup = [VGIoFixtures, EnvelopeCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        files = write_every_kind(dir)
        @test sort(collect(keys(files))) == sort(KINDS)
        # every kind of a VGC value has a file here (not: units, the COLMAP sidecar's; calibration, the manifest of a
        # directory, calibration_manifest_tests.jl; the kinds of io10-kinds' own tests: detection, sensor_noise, panorama)
        @test setdiff(keys(VgtkIO._SCHEMA_VERSION), ["units", "calibration", "detection", "sensor_noise", "panorama"]) ⊆ KINDS
        for (kind, (p, _, _)) in files
            d = JSON.parse(read(p, String))                     # (key order kept)
            @test collect(keys(d))[1:4] == ["schema", "generator", "units", "conventions"]
            @test d["schema"] == "vgio/$(kind)/v$(VgtkIO._SCHEMA_VERSION[kind][1]).$(VgtkIO._SCHEMA_VERSION[kind][2])"
            @test d["generator"] == Dict("name" => "VisualGeometryToolkitIO", "version" => string(pkgversion(VgtkIO)))
            @test collect(keys(d["units"])) == ["length", "time", "angle", "pixel", "density", "radiance", "length_roles"]
            @test d["units"]["time"] == "s" && d["units"]["angle"] == "rad" && d["units"]["density"] == "dpi" &&
                  d["units"]["radiance"] == "relative" && d["units"]["length_roles"] == Dict()
            @test d["conventions"] == Dict("rotation_encoding" => "quaternion_wxyz_hamilton", "transform_convention" => "x_to = R * x_from + t",
                                           "matrix_layout" => "rows", "grid_layout" => "row_major", "index_base" => 1)
            @test VgtkIO.validate(p) == kind
        end
        # the declared pixel frame is the value's (observations in julia pixels, homographies in colmap)
        @test JSON.parse(read(files["observations"][1], String))["units"]["pixel"] == "julia"
    end
end

@testitem "envelope: a file missing any envelope key, or with a wrong token, is refused by name" setup = [VGIoFixtures, EnvelopeCases] begin
    mktempdir() do dir
        for (kind, (p, T, _)) in write_every_kind(dir)
            d = tomutable(JSON.parse(read(p, String)))
            for k in ("schema", "generator", "units", "conventions")
                t = deepcopy(d); delete!(t, k)
                @test occursin(k, something(refusal(dir, "$(kind)_$(k)", T, t), ""))
            end
            for (obj, k) in (("generator", "name"), ("generator", "version"), ("units", "length"), ("units", "time"), ("units", "angle"),
                             ("units", "pixel"), ("units", "density"), ("units", "radiance"), ("units", "length_roles"),
                             ("conventions", "rotation_encoding"), ("conventions", "transform_convention"), ("conventions", "matrix_layout"),
                             ("conventions", "grid_layout"), ("conventions", "index_base"))
                t = deepcopy(d); delete!(t[obj], k)
                @test occursin("$(obj).$(k)", something(refusal(dir, "$(kind)_$(obj)_$(k)", T, t), ""))
            end
            # a declaration is a token: anything else is refused, never reinterpreted
            for (obj, k, v) in (("conventions", "index_base", 0), ("conventions", "rotation_encoding", "quaternion_xyzw"),
                                ("conventions", "matrix_layout", "columns"), ("conventions", "grid_layout", "column_major"),
                                ("conventions", "transform_convention", "x_from = R * x_to + t"), ("units", "time", "ms"),
                                ("units", "angle", "deg"), ("units", "radiance", "absolute"), ("units", "length", "px"),
                                ("units", "pixel", "lower_left"), ("units", "density", "mm"), ("generator", "version", "new"))
                t = deepcopy(d); t[obj][k] = v
                @test occursin("$(obj).$(k)", something(refusal(dir, "$(kind)_$(obj)_$(k)_bad", T, t), ""))
            end
            # a length role is a snake_case name other than the base unit's, its unit a length
            for (r, u) in (("length", "mm"), ("Pattern", "pt"), ("pattern", "px"))
                t = deepcopy(d); t["units"]["length_roles"][r] = u
                @test refusal(dir, "$(kind)_role", T, t) !== nothing
            end
            # the generator's caller is a package id; nothing else is admitted there
            t = deepcopy(d); t["generator"]["caller"] = Dict("name" => "X")
            @test occursin("generator.caller.version", something(refusal(dir, "$(kind)_caller", T, t), ""))
            t = deepcopy(d); t["generator"]["host"] = "dgx01"
            @test occursin("generator.host", something(refusal(dir, "$(kind)_host", T, t), ""))
        end
    end
end

@testitem "envelope: a higher minor is read with its added keys; at the reader's minor an unknown key is refused; a later major is refused" setup = [VGIoFixtures, EnvelopeCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        for (kind, (p, T, v)) in write_every_kind(dir)
            d = tomutable(JSON.parse(read(p, String)))
            M, n = VgtkIO._SCHEMA_VERSION[kind]
            # the same minor: an unknown key, at the top and in the units object, refused by name
            t = deepcopy(d); t["added_in_a_later_minor"] = 1
            @test occursin("added_in_a_later_minor", something(refusal(dir, "$(kind)_unknown", T, t), ""))
            t = deepcopy(d); t["units"]["luminance"] = "cd"
            @test occursin("units.luminance", something(refusal(dir, "$(kind)_unknown_unit", T, t), ""))
            # a higher minor: its added (optional) keys tolerated, the value read as it is
            t = deepcopy(d); t["schema"] = "vgio/$(kind)/v$(M).$(n + 1)"; t["added_in_a_later_minor"] = Dict("x" => 1)
            @test refusal(dir, "$(kind)_minor", T, t) === nothing
            @test VgtkIO.validate(joinpath(dir, "edited_$(kind)_minor.json")) == kind
            back = load(T, joinpath(dir, "edited_$(kind)_minor.json"))
            @test back == load(T, p)
            # …but a higher minor still holds every key its major requires
            delete!(t, "units")
            @test occursin("units", something(refusal(dir, "$(kind)_minor_nounits", T, t), ""))
            # a later major: refused, naming the update
            t = deepcopy(d); t["schema"] = "vgio/$(kind)/v$(M + 1).0"
            @test occursin("later major", something(refusal(dir, "$(kind)_major", T, t), ""))
            # a malformed version string is not the kind's
            t = deepcopy(d); t["schema"] = "vgio/$(kind)/v$(M).x"
            @test refusal(dir, "$(kind)_malformed", T, t) !== nothing
            # a pre-0.10 string without a minor is minor 0 of its major: the current major's `vM` reads as `vM.0`
            t = deepcopy(d); t["schema"] = "vgio/$(kind)/v$(M)"
            @test refusal(dir, "$(kind)_nominor", T, t) === nothing
        end
    end
end

@testitem "envelope: the generator names the calling package (with_caller)" setup = [VGIoFixtures, EnvelopeCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    mktempdir() do dir
        files = VgtkIO.with_caller(() -> write_every_kind(dir), VgtkCore)
        for (kind, (p, T, v)) in files
            g = JSON.parse(read(p, String))["generator"]
            @test g == Dict("name" => "VisualGeometryToolkitIO", "version" => string(pkgversion(VgtkIO)),
                            "caller" => Dict("name" => "VisualGeometryToolkitCore", "version" => string(pkgversion(VgtkCore))))
            @test load(T, p) == v                                      # the generator is the file's, not the value's
        end
        # scoped: outside the block the caller is gone; IO naming itself adds no caller
        p = joinpath(dir, "after.json")
        save(p, last(files["transforms"]))
        @test !haskey(JSON.parse(read(p, String))["generator"], "caller")
        VgtkIO.with_caller(() -> save(p, last(files["transforms"])), VgtkIO)
        @test !haskey(JSON.parse(read(p, String))["generator"], "caller")
        # write_record (another package's kind) writes the same envelope
        q = joinpath(dir, "owner.json")
        VgtkIO.with_caller(() -> VgtkIO.write_record(q, (schema = "rc/thing/v1.0", value = 1)), VgtkCore)
        r = JSON.parse(read(q, String))
        @test collect(keys(r)) == ["schema", "generator", "units", "conventions", "value"]
        @test r["generator"]["caller"]["name"] == "VisualGeometryToolkitCore"
        @test_throws ArgumentError VgtkIO.write_record(q, (schema = "rc/thing/v1.0", units = Dict("length" => "mm")))
        @test_throws ArgumentError VgtkIO.write_record(q, (; value = 1))
    end
end

@testitem "every kind round-trips exactly by type, and a rewrite is byte for byte" setup = [VGIoFixtures, EnvelopeCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        for (kind, (p, T, v)) in write_every_kind(dir)
            back = load(T, p)
            @test back == v
            @test typeof(back) == typeof(v) || (v isa AbstractVector && eltype(back) <: eltype(v))
            q = joinpath(dir, "again_$(kind).json")
            save(q, back)
            @test read(q, String) == read(p, String)
        end
    end
end

@testitem "index base 1: declared once, every index 1-based on disk, no ±1 in the reader" setup = [VGIoFixtures, EnvelopeCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: TargetObservations, PixelFrame, correspondences
    h = repeat("ab", 32)
    mktempdir() do dir
        # an observation's point index is the target's point's own position, 1-based, as stored
        x = TargetObservations{PixelFrame{:colmap}}(Dict(1 => Dict(:a => correspondences([1, 7], [SVector(1.0, 2.0), SVector(3.0, 4.0)])));
                                                     images = Dict(1 => "a.png"), sensors = "cam", targets = [h])
        p = joinpath(dir, "obs.json")
        save(p, x)
        d = JSON.parse(read(p, String))
        @test d["conventions"]["index_base"] == 1 && d["frames"][1]["observations"][1]["index"] == [1, 7]
        @test VisualGeometryToolkitCore.sources(load(TargetObservations, p)[1][:a]) == [1, 7]
        # a zero index is not 1-based: refused
        t = tomutable(d); t["frames"][1]["observations"][1]["index"] = [0, 7]
        @test refusal(dir, "zero", TargetObservations, t) !== nothing
        # a lattice's runs: rows and columns 1-based, the mask's own indices
        P = trues(4, 5); P[1, 1] = false; P[4, 5] = false
        runs = VgtkIO._present_runs(P)
        @test runs == Any[Any[1, Any[[2, 5]]], Any[2, Any[[1, 5]]], Any[3, Any[[1, 5]]], Any[4, Any[[1, 4]]]]
        @test VgtkIO._present_from_runs(runs, 5, 4) == P
        @test_throws ArgumentError VgtkIO._validate(VgtkIO._RowRuns(), Any[Any[0, Any[[1, 5]]]], "f", "present")
        @test_throws ArgumentError VgtkIO._validate(VgtkIO._RowRuns(), Any[Any[1, Any[[0, 5]]]], "f", "present")
    end
end
