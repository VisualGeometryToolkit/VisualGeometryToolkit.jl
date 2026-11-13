# =============================================================================
# rig/v10.0 and cameras/v6.0: a sensor bounded by its physical Sensor carries the pixel pitch
# =============================================================================
#
# The camera body of a rig sensor (and of a stored camera) carries `"pitch":
# {"width", "height"}` (IO 0.10: a Size2 object), a length per pixel in the
# file's `units.length`, iff the camera is bounded by a `Sensor`; absent, the
# body and the identity are the bare extent's, byte for byte (every other
# fixture and content_id pin covers that).

@testitem "rig/v10.0: the pitch travels with the camera body" setup = [VGIoFixtures] begin
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
        d = JSON.parse(read(src, String); dicttype = Dict{String, Any})
        i = findfirst(s -> haskey(s["camera"], "pitch"), d["sensors"])
        @test d["sensors"][i]["camera"]["pitch"] == Dict("width" => 0.00345, "height" => 0.00345)
        mktempdir() do dir
            function refusal(edit!)
                t = deepcopy(d); edit!(t["sensors"][i]["camera"])
                p = joinpath(dir, "bad.json"); write(p, JSON.json(t))
                return try load(Rig, p); "" catch e; e isa ArgumentError ? sprint(showerror, e) : rethrow() end
            end
            for bad! in (c -> (c["pitch"]["width"] = -0.00345), c -> (c["pitch"]["height"] = "3.45um"), c -> delete!(c["pitch"], "height"),
                         c -> (c["pitch"] = [0.00345, 0.00345]), c -> (c["pitch"]["width"] = 0.0))
                msg = refusal(bad!)
                @test occursin("pitch", msg)                     # the error names the key, as every refusal does
            end
            @test refusal(identity) == ""
        end
    end
end

@testitem "rig/v10.0: a grid sensor keeps its pitch" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedCamera, BoundedCamera, ViewFrame, Sensor, sensor, relabel, model, μm
    using VisualGeometryToolkitIO: content_id
    using Rotations: RotXYZ

    gc = CameraModel(SplineLens, CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-0.02)), Size2(64, 48); grid_size = (5, 4))
    Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
    s = Sensor(Size2(width = 64px, height = 48px), 3.45μm)
    pitched = Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(gc, Xl), s),))
    bare = Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(gc, Xl), Size2(64, 48)),))
    mktempdir() do dir
        p = joinpath(dir, "rig.json"); save(p, pitched)
        d = parse_json(read(p, String))
        @test d.sensors[1].camera.pitch == Dict(:width => 0.00345, :height => 0.00345)
        @test d.sensors[1].camera.image_size == Dict(:width => 64, :height => 48)
        back = load(Rig, p)
        @test model(back[1]).lens isa SplineLens
        @test ustrip(μm, sensor(back[1]).pitch.width) ≈ 3.45
        @test content_id(back) == content_id(pitched) == content_id(p)
        @test content_id(pitched) != content_id(bare)
        pb = joinpath(dir, "bare.json"); save(pb, bare)
        @test !occursin("pitch", read(pb, String))
        @test sensor(load(Rig, pb)[1]) === nothing
    end
end

@testitem "cameras/v6.0: the store keeps a camera's pitch" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: BoundedCamera, Sensor, sensor, model, μm, CMOS_SENSORS

    m = CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(1270.0, 0, 0, 0, 1270.0, 0, 1224.0, 1024.0, 1)); lens = KannalaBrandt(-0.1351))
    s = CMOS_SENSORS["Sony"]["IMX250"]
    mktempdir() do dir
        p = joinpath(dir, "cameras.json")
        save(p, Dict(:fish => BoundedCamera(m, s), :plain => BoundedCamera(m, Size2(640, 480))))
        d = parse_json(read(p, String))
        @test d.units.length == "mm"                                      # every units key, always
        @test d.cameras[1].pitch == Dict(:width => 0.00345, :height => 0.00345) && !haskey(d.cameras[2], :pitch)
        store = load(Dict{Symbol, BoundedCamera}, p)
        @test ustrip(μm, sensor(store[:fish]).pitch.width) ≈ 3.45 && ustrip(μm, sensor(store[:fish]).pitch.height) ≈ 3.45
        @test extent(store[:fish]) == Size2(2448, 2048)
        @test sensor(store[:plain]) === nothing
        p2 = joinpath(dir, "again.json"); save(p2, store)
        @test read(p2, String) == read(p, String)
        # a store with no pitch still declares its units, whole (nothing defaults); it writes no pitch
        p3 = joinpath(dir, "plain.json"); save(p3, Dict(:plain => BoundedCamera(m, Size2(640, 480))))
        @test parse_json(read(p3, String)).units.length == "mm" && !occursin("pitch", read(p3, String))
        # a pitch in m reads to the same Sensor
        p4 = joinpath(dir, "m.json"); save(p4, store; length_unit = VisualGeometryToolkitCore.m)
        @test parse_json(read(p4, String)).cameras[1].pitch == Dict(:width => 3.45e-6, :height => 3.45e-6)
        @test sensor(load(Dict{Symbol, BoundedCamera}, p4)[:fish]) == sensor(store[:fish])
        # a file without its length unit is refused
        t = JSON.parse(read(p, String); dicttype = Dict{String, Any}); delete!(t["units"], "length")
        bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
        @test occursin("units.length", try load(Dict{Symbol, BoundedCamera}, bad); "" catch e; e.msg end)
    end
end
