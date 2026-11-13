# vgio/panorama/v1 (panorama_schema.jl): a panorama's results — the bounded camera body, the rotations as the
# trajectory body with zero translations, the vignetting body and the frames' gains — every field round trip
# (Float64 and Float32 where a field holds numbers of either), the table's refusals, and the composition: each part's
# bytes are the bytes its own kind writes.
@testitem "panorama: every field round trips, Float64 and Float32" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: save_panorama, load_panorama, validate, camera_json
    using Rotations: params
    using VisualGeometryToolkitCore: Trajectory, CameraFrame, WorldFrame, AbstractVignetting, TensorProductVignetting, RadialVignetting,
        LogVignetting, with_control_points, control_points, knot_vectors, Rect, Circle, Point2, Sensor, lens_coverage, model, sensor, μm
    w(i, j) = 0.5 + 0.0371 * i + 0.0193 * j * j
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(1210.5, 0, 0, 0, 1209.25, 0, 2120.375, 1415.625, 1))
    cams = (
        BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap), Size2(4240, 2832)),
        BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap),
                      Sensor(Size2(width = 4240px, height = 2832px), 5.94μm); lens_coverage = Circle(Point2(2120.5, 1416.0), 1900.25)),
    )
    z = [0.0, 0.0, 0.0] .* m
    q(a, b, c, d) = (n = sqrt(a^2 + b^2 + c^2 + d^2); QuatRotation(a / n, b / n, c / n, d / n, false))
    R = Dict(1 => QuatRotation(1.0, 0.0, 0.0, 0.0, false), 2 => q(0.9, 0.1, -0.3, 0.05), 3 => q(0.7, -0.2, 0.6, 0.1))
    rots = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(r, z) for (k, r) in R);
                      images = Dict(k => "DSC0000$(k).ARW" for k in 1:3), times = Dict(k => 0.5k for k in 1:3), time_origin = "2026-11-13T10:00:00")
    tv = with_control_points(TensorProductVignetting((5, 4), Rect(0.0, 0.0, 4240.0, 2832.0); degree = (2, 3)), [w(i, j) for i in 1:5, j in 1:4])
    rv = with_control_points(RadialVignetting(7, SVector(2120.3, 1416.7), 2551.25; chart = :projective), [w(i, 1) for i in 1:7])
    G = [1.0 1.0 1.0; 1.0213 0.9987 1.0441; 0.9712 0.9655 0.9801]
    mktempdir() do dir
        p = joinpath(dir, "panorama.json")
        for cam in cams, v in (tv, rv, LogVignetting(tv), LogVignetting(rv)), gains in (G, nothing)
            save_panorama(p, cam, rots, v, gains)
            @test validate(p) == "panorama"
            b = load_panorama(p)
            @test keys(b) == (:camera, :rotations, :vignetting, :gains)
            # the camera: its body, bound and lens coverage exactly
            @test camera_json(b.camera; length_unit = mm) == camera_json(cam; length_unit = mm) && model(b.camera) == model(cam)
            @test lens_coverage(b.camera) == lens_coverage(cam) && sensor(b.camera) == sensor(cam)
            # the rotations: the stored quaternions, the stamps, zero translations, labeled camera → world
            @test b.rotations == rots && b.rotations isa Trajectory{CameraFrame, WorldFrame}
            # the vignetting, bit for bit
            @test typeof(b.vignetting) == typeof(v) && control_points(b.vignetting) == control_points(v) && knot_vectors(b.vignetting) == knot_vectors(v)
            @test b.vignetting(SVector(100.5, 200.25)) == v(SVector(100.5, 200.25))
            # the gains
            @test b.gains == gains && (gains === nothing || b.gains isa Matrix{Float64})
            d = JSON.parsefile(p)
            @test haskey(d, "gains") == (gains !== nothing) && d["pixel_frame"] == "colmap" && d["length_unit"] == "mm"
        end
        # Float32: rotations, gains and vignetting control points of Float32 are written exactly and read back as the
        # same numbers in Float64
        R32 = Dict(k => QuatRotation{Float32}(params(r)..., false) for (k, r) in R)
        r32 = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(r, z) for (k, r) in R32))
        G32 = Float32.(G)
        v32 = with_control_points(RadialVignetting(7, SVector(2120.3, 1416.7), 2551.25), Float32[w(i, 1) for i in 1:7])
        save_panorama(p, cams[1], r32, v32, G32)
        b = load_panorama(p)
        @test all(Float32.(params(b.rotations[k].R)) == params(R32[k]) && params(b.rotations[k].R) == Float64.(params(R32[k])) for k in 1:3)
        @test b.gains isa Matrix{Float64} && b.gains == Float64.(G32) && Float32.(b.gains) == G32
        @test Float32.(control_points(b.vignetting)) == control_points(v32)
        # a rotation matrix is written as its quaternion (any rotation converted on write), read back as a QuatRotation
        rm = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(RotMatrix{3, Float64}(r), z) for (k, r) in R))
        save_panorama(p, cams[1], rm, tv, nothing)
        @test all(isapprox(RotMatrix(load_panorama(p).rotations[k].R), RotMatrix(R[k]); atol = 1e-15) for k in 1:3)
    end
end

@testitem "panorama: composed of the camera, trajectory and vignetting bodies, byte for byte" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: save_panorama
    using VisualGeometryToolkitCore: Trajectory, CameraFrame, WorldFrame, AbstractVignetting, RadialVignetting, with_control_points
    cam = BoundedCamera(CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1)); lens = KannalaBrandt(0.1, 0.0, 0.0, 0.0), pixel_frame = :colmap), Size2(640, 480))
    rots = Trajectory(CameraFrame, WorldFrame, Dict(1 => EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), [0.0, 0.0, 0.0] .* m)))
    v = with_control_points(RadialVignetting(5, SVector(320.0, 240.0), 400.0), [1.0, 0.98, 0.93, 0.85, 0.74])
    mktempdir() do dir
        save_panorama(joinpath(dir, "p.json"), cam, rots, v, [1.0 1.0 1.0])
        save(joinpath(dir, "c.json"), Dict(:c => cam)); save(joinpath(dir, "t.json"), rots); save(joinpath(dir, "v.json"), v)
        P = JSON.parsefile(joinpath(dir, "p.json"))
        c = only(JSON.parsefile(joinpath(dir, "c.json"))["cameras"]); delete!(c, "name")
        t = JSON.parsefile(joinpath(dir, "t.json"))
        @test JSON.json(P["camera"]) == JSON.json(c)
        @test collect(keys(P["rotations"])) == ["from", "to", "frames"] && all(JSON.json(P["rotations"][k]) == JSON.json(t[k]) for k in ("from", "to", "frames"))
        @test JSON.json(P["vignetting"]) == JSON.json(JSON.parsefile(joinpath(dir, "v.json"))["vignetting"])
        @test all(P[k] == t[k] for k in ("length_unit", "rotation_encoding", "transform_convention"))
        @test P["gains"] == [Dict("t" => 1, "gain" => [1.0, 1.0, 1.0])]
    end
end

@testitem "panorama: the table's refusals, on write and on read" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: save_panorama, load_panorama, validate
    using VisualGeometryToolkitCore: Trajectory, CameraFrame, WorldFrame, RadialVignetting, with_control_points
    cam = BoundedCamera(CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 800.0, 0, 320.5, 240.5, 1)); lens = KannalaBrandt(0.1, 0.0, 0.0, 0.0), pixel_frame = :colmap), Size2(640, 480))
    I4 = QuatRotation(1.0, 0.0, 0.0, 0.0, false)
    rots = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(I4, [0.0, 0.0, 0.0] .* m) for k in 1:2))
    v = with_control_points(RadialVignetting(5, SVector(320.0, 240.0), 400.0), [1.0, 0.98, 0.93, 0.85, 0.74])
    mktempdir() do dir
        p = joinpath(dir, "p.json")
        # the camera only rotates: a translation is refused, and so is a plain one (no unit)
        moved = Trajectory(CameraFrame, WorldFrame, Dict(1 => EuclideanMap(I4, [0.0, 0.0, 1.0] .* mm), 2 => EuclideanMap(I4, [0.0, 0.0, 0.0] .* m)))
        @test_throws ArgumentError save_panorama(p, cam, moved, v, nothing)
        plain = Trajectory(CameraFrame, WorldFrame, Dict(k => EuclideanMap(I4, [0.0, 0.0, 0.0]) for k in 1:2))
        @test_throws ArgumentError save_panorama(p, cam, plain, v, nothing)
        # gains: one row per frame index 1:K, every gain positive; required, never defaulted
        @test_throws ArgumentError save_panorama(p, cam, rots, v, ones(3, 3))
        @test_throws ArgumentError save_panorama(p, cam, rots, v, [1.0 1.0 1.0; 1.0 0.0 1.0])
        @test_throws MethodError save_panorama(p, cam, rots, v)                       # the gains are stated, `nothing` or a matrix
        # the rotations map the camera to the world
        inv_rots = Trajectory(WorldFrame, CameraFrame, Dict(k => EuclideanMap(I4, [0.0, 0.0, 0.0] .* m) for k in 1:2))
        @test_throws ArgumentError save_panorama(p, cam, inv_rots, v, nothing)
        isfile(p) && rm(p)
        save_panorama(p, cam, rots, v, [1.0 1.0 1.0; 1.1 1.0 0.9])
        good = JSON.parsefile(p)
        bad(f) = (d = deepcopy(good); f(d); q = joinpath(dir, "bad.json"); write(q, JSON.json(d)); q)
        for f in (d -> (d["rotations"]["frames"][2]["pose"]["translation"] = [0.0, 1.0, 0.0]),   # a translation
                  d -> pop!(d["gains"]),                                                          # a frame with no gains
                  d -> (d["gains"][2]["t"] = 1),                                                  # a frame index twice
                  d -> (d["gains"][2]["gain"] = [1.0, 1.0]),                                      # another number of planes
                  d -> (d["gains"][2]["gain"][1] = -1.0),                                          # a gain not positive
                  d -> (d["pixel_frame"] = "opencv"),                                             # the vignetting body's frame is colmap
                  d -> (d["rotations"]["to"] = "view:pano"),                                      # camera → world only
                  d -> (foreach(f -> (f["t"] += 1), d["rotations"]["frames"]); foreach(g -> (g["t"] += 1), d["gains"])),   # gains of t ≠ 1:K
                  d -> delete!(d, "rotation_encoding"),                                           # a declaration missing
                  d -> delete!(d, "vignetting"), d -> delete!(d, "camera"), d -> delete!(d, "rotations"),
                  d -> (d["extra"] = 1))                                                          # an unknown key
            q = bad(f)
            @test_throws ArgumentError validate(q)
            @test_throws ArgumentError load_panorama(q)
        end
        # the gains are optional: a file without them reads `nothing`
        @test load_panorama(bad(d -> delete!(d, "gains"))).gains === nothing
        # a file of another kind is refused by name
        save(joinpath(dir, "t.json"), rots)
        @test_throws ArgumentError load_panorama(joinpath(dir, "t.json"))
    end
end
