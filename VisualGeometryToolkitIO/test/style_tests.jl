# VgioStyle (src/style.jl): the records every embedded VGC value is lowered to
# and lifted from — the edges a code review of the generic serializer found.

@testitem "VgioStyle: a -0.0 principal point is written 0.0, as the camera bodies always wrote it" setup = [VGIoFixtures] begin
    using Test: @inferred
    using VisualGeometryToolkitCore: BoundedCamera
    import VisualGeometryToolkitIO as VgtkIO
    K =CameraCalibrationMat(SMatrix{3, 3, Float64}(500.0, 0, 0, 0, 500.0, 0, -0.0, -0.0, 1))
    r = @inferred VgtkIO._k_record(K, VgtkIO.CANONICAL_PIXEL_FRAME)       # type-stable
    @test r.cx === 0.0 && r.cy === 0.0
    @test VgtkIO._k_values(K)[3] === -0.0          # the nominal pinhole and the identity read K as stored
    mktempdir() do dir
        p = joinpath(dir, "cameras.json")
        save(p, Dict(:c => BoundedCamera(CameraModel(K), Size2(640, 480))))
        txt = read(p, String)
        @test occursin("\"cx\": 0.0", txt) && !occursin("-0.0", txt)
    end
end

@testitem "VgioStyle: a lift reads a parsed object or a NamedTuple" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    K =CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 640.0, 480.0, 1))
    cm = CameraModel(K)
    back, ext = VgtkIO.camera_from_json(VgtkIO.camera_json(cm, Size2(1280, 960)), VgtkIO.CANONICAL_PIXEL_FRAME)
    @test back == cm && ext == Size2(1280, 960)
    X = _mkpose()
    pj = VgtkIO.pose_json(X, mm)
    rec = (rotation = Vector{Any}(pj.rotation), translation = Vector{Any}(pj.translation))
    @test VgtkIO.pose_from_json(rec, "mm") == VgtkIO.pose_from_json(JSON.parse(JSON.json(pj)), "mm")
    # a Dict has no property access: a TypeError at the call, not a FieldError inside
    @test_throws TypeError VgtkIO.pose_from_json(Dict{String, Any}("rotation" => rec.rotation, "translation" => rec.translation), "mm")
end

@testitem "VgioStyle: a parametric record holds exactly its kind's parameter count" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    s =VgtkIO.VgioStyle(; length_unit = mm)
    for (rec, n) in (("{\"type\":\"scale\",\"parameters\":[1.5,2.0]}", 2), ("{\"type\":\"scale\"}", 0), ("{\"type\":\"euclidean\",\"parameters\":[0.1,1.0]}", 2))
        e = try VgtkIO._target_calibration_from_json(s, JSON.parse(rec)); nothing catch err; err end
        @test e isa ArgumentError && occursin("\"$(JSON.parse(rec).type)\"", e.msg) && occursin("got $n", e.msg)
    end
    @test VgtkIO._target_calibration_from_json(s, JSON.parse("{\"type\":\"scale\",\"parameters\":[1.5]}")) == VisualGeometryToolkitCore.Scale(1.5)
end

@testitem "VgioStyle: a target's records — target_json, its lift, its calibration" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: PointTarget, PosedTarget, AbstractTarget
    import VisualGeometryToolkitIO as VgtkIO
    markers = PointTarget([mm_lengths(SVector(0.0, 0.0, 0.0)), mm_lengths(SVector(30.0, 0.0, 5.0))])
    @test_throws MethodError VgtkIO.target_json(PosedTarget(markers, _mkpose()))   # only an unposed target has a body
    s = VgtkIO.VgioStyle(; length_unit = mm)
    rec = JSON.parse(JSON.json(VgtkIO.target_json(markers, mm)))
    @test JSON.lift(s, AbstractTarget, rec) == VgtkIO.target_from_json(rec, "mm") == markers
    # a planar map left in a body is not a target calibration: no `lower` claims VGC's TargetCalibration union
    conv = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv})
    txt = JSON.json((x = conv,); style = s)
    @test !occursin("euclidean", txt) && !occursin("parameters", txt)
end

@testitem "VgioStyle: every writer serializes on its file's style (a rig in metres)" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedTarget, PointTarget, BoundedCamera, PosedCamera, BodyFrame, ViewFrame, relabel, ids,
        extrinsics, pose, model
    using VisualGeometryToolkitIO: content_id
    K = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 640.0, 480.0, 1)
    Xc = relabel(BodyFrame{:camera_rig}, ViewFrame{:cam}, _mkpose())
    cam = BoundedCamera(PosedCamera(CameraModel(CameraCalibrationMat(K)), Xc), Size2(1280, 960))
    markers = PointTarget([mm_lengths(SVector(0.0, 0.0, 0.0)), mm_lengths(SVector(30.0, 0.0, 5.0))])
    rig = Rig{BodyFrame{:camera_rig}}((cam,), Dict(:markers => PosedTarget(markers, _mkpose())))
    mktempdir() do dir
        p = joinpath(dir, "rig.json")
        save(p, rig; length_unit = m)
        d = parse_json(read(p, String))
        @test d.length_unit == "m" && d.targets[1].target.points[2][1] ≈ 0.03
        back = load(Rig, p)
        @test ids(back) == [:cam, :markers] && model(back[:cam]) == model(cam)
        @test back[:markers].target == markers && pose(back, :markers) == pose(rig, :markers)   # decimal-exact in metres too
        @test content_id(back) == content_id(p)
    end
end

@testitem "VgioStyle: a generic model's unstorable domain names its caller" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    struct _Disc end
    e(s) = try VgtkIO._refuse_domain(s, _Disc()); nothing catch err; err end
    w, i = e(VgtkIO.VgioStyle(; length_unit = mm)), e(VgtkIO._ID_STYLE)
    @test w isa ArgumentError && startswith(w.msg, "save: a generic model's domain is stored as an Ellipse")
    @test i isa ArgumentError && startswith(i.msg, "content_id: a generic model's domain is an Ellipse")
end
