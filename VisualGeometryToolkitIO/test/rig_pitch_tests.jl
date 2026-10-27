# =============================================================================
# rig/v5: a sensor bounded by its physical Sensor carries the pixel pitch
# =============================================================================
#
# The camera body of a rig sensor gains `"pitch": [width, height]`, a length per
# pixel in the rig's `length_unit`, iff the camera is bounded by a `Sensor`;
# absent, the body and the identity are the bare extent's, byte for byte (every
# other fixture and content_id pin covers that). Additive in rig/v5, unreleased.

@testitem "rig/v5: the pitch travels with the camera body" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, BoundedCamera, PosedCamera, StereoRig, Sensor, sensor, model, μm, CMOS_SENSORS, image_circle
    using VisualGeometryToolkitIO: content_id

    src = joinpath(@__DIR__, "fixtures", "units", "rig_pitch.json")
    rig = load(Rig, src)

    @testset "the reader restores the Sensor bound" begin
        s = sensor(rig[1])
        @test s isa Sensor
        @test extent(rig[1]) == Size2(2448, 2048)
        @test ustrip(μm, s.pitch.width) ≈ 3.45 && ustrip(μm, s.pitch.height) ≈ 3.45
        @test ustrip(m, s.pitch.width) === ustrip(m, VgtkIO.read_length(0.00345, mm))   # the metre value nearest the decimal
        @test sensor(rig[2]) === nothing                                 # a bare extent stays one
        @test image_circle(rig[1], 6.0mm).a ≈ 6.0e-3 / (2 * 3.45e-6)     # the datasheet route reads the loaded pitch
    end

    @testset "load(save(rig)): the bound, the bytes and the identity" begin
        mktempdir() do dir
            out = joinpath(dir, "rig.json")
            save(out, rig)
            @test read(out, String) == read(src, String)
            back = load(Rig, out)
            @test sensor(back[1]) == sensor(rig[1])
            @test content_id(back) == content_id(rig) == content_id(src)
            # a value built in code (the CMOS_SENSORS pitch, 3.45μm) writes and hashes as its file
            fish = BoundedCamera(rig[1].camera, CMOS_SENSORS["Sony"]["IMX250"])
            built = StereoRig(fish, rig[2]; ids = (:fisheye, :cam))
            p2 = joinpath(dir, "built.json")
            save(p2, built)
            @test read(p2, String) == read(src, String)
            @test content_id(built) == content_id(src)
            # the identity covers the pitch: the same rig bounded by its bare extent differs
            bare = StereoRig(BoundedCamera(rig[1].camera, extent(rig[1])), rig[2]; ids = (:fisheye, :cam))
            @test content_id(bare) != content_id(rig)
            p3 = joinpath(dir, "bare.json")
            save(p3, bare)
            @test !occursin("pitch", read(p3, String))
            @test sensor(load(Rig, p3)[1]) === nothing
        end
    end

    @testset "a pitch that is not two positive lengths per pixel is refused" begin
        text = read(src, String)
        mktempdir() do dir
            for (bad, why) in (("[\n                    -0.00345,", "negative"), ("[\n                    \"3.45um\",", "a string"))
                p = joinpath(dir, "bad.json")
                write(p, replace(text, "[\n                    0.00345," => bad; count = 1))
                @test_throws ArgumentError load(Rig, p)
            end
            p = joinpath(dir, "short.json")
            write(p, replace(text, "0.00345,\n                    0.00345" => "0.00345"; count = 1))
            @test_throws ArgumentError load(Rig, p)
            # the error names the key, as every refusal does
            msg = try
                load(Rig, joinpath(dir, "bad.json")); ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("pitch", msg)
        end
    end
end

@testitem "rig/v5: a generic grid sensor keeps its pitch" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedCamera, BoundedCamera, ViewFrame, Vec3, Sensor, sensor, relabel, model, μm
    using VisualGeometryToolkitIO: content_id
    using Rotations: RotXYZ

    dirs = [Vec3{Float64}(0.01 * gx, 0.02 * gy, 1.0) for gy in 1:4, gx in 1:5]
    gc = CentralGenericModel(64, 48, 0, 0, 63, 47, dirs)
    Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
    s = Sensor(Size2(width = 64px, height = 48px), 3.45μm)
    pitched = Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(gc, Xl), s),))
    bare = Rig{ViewFrame{:g}}((PosedCamera(gc, Xl),))
    mktempdir() do dir
        p = joinpath(dir, "rig.json"); save(p, pitched)
        d = parse_json(read(p, String))
        @test d.sensors[1].camera.pitch == [0.00345, 0.00345]
        back = load(Rig, p)
        @test model(back[1]) isa CentralGenericModel
        @test ustrip(μm, sensor(back[1]).pitch.width) ≈ 3.45
        @test content_id(back) == content_id(pitched) == content_id(p)
        @test content_id(pitched) != content_id(bare)
        pb = joinpath(dir, "bare.json"); save(pb, bare)
        @test !occursin("pitch", read(pb, String))
        @test sensor(load(Rig, pb)[1]) === nothing
    end
end

@testitem "cameras/v3: the store keeps a camera's pitch" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: BoundedCamera, Sensor, sensor, model, μm, CMOS_SENSORS

    m = CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(1270.0, 0, 0, 0, 1270.0, 0, 1224.0, 1024.0, 1)); lens = KannalaBrandt(-0.1351))
    s = CMOS_SENSORS["Sony"]["IMX250"]
    mktempdir() do dir
        p = joinpath(dir, "cameras.json")
        save(p, Dict(:fish => BoundedCamera(m, s), :plain => BoundedCamera(m, Size2(640, 480))))
        d = parse_json(read(p, String))
        @test d.length_unit == "mm"                                       # declared iff a camera carries a pitch
        @test d.cameras[1].pitch == [0.00345, 0.00345] && !haskey(d.cameras[2], :pitch)
        store = load(Dict{Symbol, BoundedCamera}, p)
        @test ustrip(μm, sensor(store[:fish]).pitch.width) ≈ 3.45 && ustrip(μm, sensor(store[:fish]).pitch.height) ≈ 3.45
        @test extent(store[:fish]) == Size2(2448, 2048)
        @test sensor(store[:plain]) === nothing
        p2 = joinpath(dir, "again.json"); save(p2, store)
        @test read(p2, String) == read(p, String)
        # a store with no pitch declares no length unit: today's bytes
        p3 = joinpath(dir, "plain.json"); save(p3, Dict(:plain => BoundedCamera(m, Size2(640, 480))))
        @test !occursin("length_unit", read(p3, String)) && !occursin("pitch", read(p3, String))
        # a pitch without a declared unit, or a unit without a pitch, is refused
        raw = read(p, String)
        bad = joinpath(dir, "bad.json")
        write(bad, replace(raw, "\"length_unit\": \"mm\",\n" => ""; count = 1))
        @test_throws ArgumentError load(Dict{Symbol, BoundedCamera}, bad)
    end
end
