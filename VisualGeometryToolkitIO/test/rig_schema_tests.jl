# =============================================================================
# rig/v4, target/v1, trajectory/v3, target_poses/v1, target_points2d/v1,
# target_points3d/v1 round trips, and the correspondences export
# =============================================================================
#
# EXACTNESS BY TYPE: every kind reads back as the VGC value it was written
# from, type included. A rotation is stored as a quaternion chosen to read back
# to the same matrix whenever one does (`_qvec`); a rotation computed from a
# matrix that no quaternion reads back to exactly agrees to an ulp — that is
# the one bound below (POSE_ULP), and a value READ from a file round-trips `==`.

@testitem "frame-name convention" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: frame_name, frame_from_name
    using VisualGeometryToolkitCore: WorldFrame, ViewFrame, TargetFrame, UnlabeledFrame, BodyFrame

    # The fallback: VGC has many more Frame subtypes than this schema handles,
    # and forgetting the id parameter (`ViewFrame`, not `ViewFrame{:cam}`) is
    # the likely way to reach it — so the message names that case first.
    @test_throws ArgumentError frame_name(ViewFrame)
    @test_throws ArgumentError frame_name(UnlabeledFrame)
    @test frame_name(WorldFrame) == "world"
    @test frame_name(ViewFrame{:cam1}) == "view:cam1"
    @test frame_name(TargetFrame{:b76e}) == "target:b76e"
    @test frame_from_name("world") === WorldFrame
    @test frame_from_name("view:left") === ViewFrame{:left}
    @test frame_from_name("target:b76e") === TargetFrame{:b76e}
    @test frame_from_name("target:1") === TargetFrame{Symbol("1")}   # an id is an identity, never a number
    @test frame_name(BodyFrame{:board_rig}) == "body:board_rig"   # a rigid body: a tracked or mounted rig
    @test frame_from_name("body:camera_rig") === BodyFrame{:camera_rig}
    @test_throws ArgumentError frame_from_name("body:")
    @test_throws ArgumentError frame_from_name("banana")
    @test_throws ArgumentError frame_from_name("view:")            # a prefix is not a name
end

@testitem "rig/v4: a sensor rig" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, StereoRig, BoundedCamera, PosedCamera, ViewFrame,
        ids, extrinsics, model, source, destination
    using Rotations: RotXYZ

    const POSE_ULP = 1.0e-15

    K1 = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 2120.0, 1416.0, 1)
    K2 = SMatrix{3, 3, Float64}(700.0, 0, 0, 0, 700.0, 0, 330.0, 250.0, 1)
    R = RotMatrix{3, Float64}(RotXYZ(0.08, -0.15, 0.05))
    t = mm_lengths(SVector(120.0, -15.0, 10.0))
    cam1 = PosedCamera(CameraModel(CameraCalibrationMat(K1)), EuclideanMap(one(RotMatrix{3, Float64}), mm_lengths(SVector(0.0, 0.0, 0.0))))
    cam2 = PosedCamera(CameraModel(CameraCalibrationMat(K2)), EuclideanMap(R, t))
    rig = StereoRig(
        BoundedCamera(cam1, Size2(4240, 2832)),
        BoundedCamera(cam2, Size2(640, 480)); ids = (:left, :right)
    )
    mktempdir() do dir
        path = joinpath(dir, "rig.json")
        save(path, rig; provenance = (name = "test-rig",))
        d = parse_json(read(path, String))
        @test d.schema == "vgio/rig/v4" && d.pixel_frame == "colmap" && d.length_unit == "mm"
        @test d.frame == "view:left" && d.provenance.name == "test-rig"
        @test !haskey(d, :targets)                                       # omitted, not empty
        # the extrinsics' frames are where they sit — never restated beside them
        @test !haskey(d.sensors[2].extrinsics, :from) && !haskey(d.sensors[2].camera, :name)
        @test d.sensors[2].extrinsics == parse_json(JSON.json(VgtkIO.pose_json(extrinsics(rig))))
        back = load(Rig, path)
        @test back isa Rig{ViewFrame{:left}, <:Tuple}
        @test ids(back) == (:left, :right)
        @test extent(source(back)) == Size2(4240, 2832)
        @test extent(destination(back)) == Size2(640, 480)
        @test model(source(back)) == model(source(rig)) && model(destination(back)) == model(destination(rig))
        Xa, Xb = extrinsics(rig), extrinsics(back)
        @test Xb.t == Xa.t
        # a loaded translation is lengths in metres
        @test Xb.t isa SVector{3, Quantity{Float64, Dimensions{FixRat32}}}
        @test VgtkIO._pose_frames(Xb) == VgtkIO._pose_frames(Xa)
        @test maximum(abs, Matrix(Xb.R) - Matrix(Xa.R)) <= POSE_ULP
        # a rig READ from a file round-trips exactly
        p2 = joinpath(dir, "again.json")
        save(p2, back)
        @test extrinsics(load(Rig, p2)) == extrinsics(back)
        # a bare PosedCamera sensor (no extent) is refused
        rig_bare = StereoRig(cam1, cam2; ids = (:a, :b))
        @test_throws ArgumentError save(path, rig_bare)
    end
end

@testitem "generic camera round trip (vgio/generic_camera/v2)" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, NoncentralGenericModel, Vec3, relabel, ids, model
    using Rotations: RotXYZ

    gh, gw = 4, 5
    dirs = [Vec3{Float64}(0.01 * gx, 0.02 * gy, 1.0) for gy in 1:gh, gx in 1:gw]
    pts = [Vec3{Float64}(0.1gx, -0.1gy, 0.05) for gy in 1:gh, gx in 1:gw]
    gc = CentralGenericModel(64, 48, 0, 0, 63, 47, dirs)
    gn = NoncentralGenericModel(64, 48, 0, 0, 63, 47, dirs, pts)
    mktempdir() do dir
        pc = joinpath(dir, "central.json"); save(pc, gc)
        dc = parse_json(read(pc, String))
        @test dc.pixel_frame == "colmap" && dc.grid_layout == "row_major" && !haskey(dc, :length_unit)
        back = load(CentralGenericModel, pc)
        @test back.grid == gc.grid && back.calib_max_x == 63 && back.width == 64
        pn = joinpath(dir, "noncentral.json"); save(pn, gn)
        @test parse_json(read(pn, String)).length_unit == "mm"     # the point grid's unit, declared
        backn = load(NoncentralGenericModel, pn)
        @test backn.directions == gn.directions && backn.points == gn.points
        # typed load is an assertion of intent
        @test_throws ArgumentError load(NoncentralGenericModel, pc)
        @test_throws ArgumentError load(CentralGenericModel, pn)
        # a rig with a GENERIC sensor round-trips (no BoundedCamera needed:
        # the record carries its own extent)
        Xl = relabel(
            ViewFrame{:g}, ViewFrame{:g},
            EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm])
        )
        rigg = Rig{ViewFrame{:g}}((PosedCamera(gc, Xl),))
        pr = joinpath(dir, "rig_generic.json"); save(pr, rigg)
        backr = load(Rig, pr)
        @test model(backr[1]) isa CentralGenericModel
        @test model(backr[1]).grid == gc.grid
        @test ids(backr) == (:g,)
        @test extent(backr[1]) == Size2(64, 48)
        # a generic sensor lives in the COLMAP pixel frame; a rig declaring
        # another could not hold one
        raw = copy(parse_json(read(pr, String)))
        raw[:pixel_frame] = "opencv"
        bad = joinpath(dir, "rig_generic_opencv.json"); write(bad, JSON.json(raw))
        @test_throws ArgumentError load(Rig, bad)
    end
end

@testitem "rig/v4: a rig of targets" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, PointTarget, is_resolved, resolve,
        WorldFrame, TargetFrame, ids, pose, points, reference_frame
    using Rotations: RotXYZ

    const POSE_ULP = 1.0e-15

    Xa = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, 0.2, -0.3)), [1.0mm, 2.0mm, 3.0mm])
    Xb = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(-0.2, 0.05, 0.4)), [-100.0mm, 50.0mm, 200.0mm])
    ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
    pts = [mm_lengths(p) for p in (SVector(0.0, 0.0), SVector(10.0, 0.0), SVector(0.0, 5.0))]
    # gauged on :a1f2 (explicit design points, at the identity); :b76e a
    # referenced design, resolved; :c9 a referenced design, unresolved
    tr = Rig{TargetFrame{:a1f2}}(
        Dict(
            :a1f2 => PosedTarget(PlanarTarget(pts), ident), :b76e => PosedTarget(PlanarTarget(:b76e; nominal = pts), Xa),
            :c9 => PosedTarget(PlanarTarget(:c9), Xb),
        )
    )
    designs = Dict(:b76e => pts, :c9 => pts)
    mktempdir() do dir
        path = joinpath(dir, "targets.json")
        save(path, tr)
        d = parse_json(read(path, String))
        @test d.schema == "vgio/rig/v4" && !haskey(d, :pixel_frame) && !haskey(d, :sensors)   # pixel_frame iff sensors
        @test [t.id for t in d.targets] == ["a1f2", "b76e", "c9"]      # ids sorted
        @test d.targets[1].target.type == "planar" && haskey(d.targets[1].target, :nominal)
        # a referenced design is written by its reference alone: its points are its owner's
        @test d.targets[2].target.design == "b76e" && !haskey(d.targets[2].target, :nominal)
        @test d.targets[2].target.warp.type == "rigid"
        back = load(Rig, path)
        @test back isa Rig{TargetFrame{:a1f2}, <:AbstractDict}
        @test reference_frame(back) === TargetFrame{:a1f2}
        @test ids(back) == [:a1f2, :b76e, :c9]
        Pa = pose(back, :b76e)
        @test Pa isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:b76e}, TargetFrame{:a1f2}}
        @test Pa.t == pose(tr, :b76e).t
        @test maximum(abs, Matrix(Pa.R) - Matrix(pose(tr, :b76e).R)) <= POSE_ULP
        # explicit points survive; a referenced design comes back unresolved…
        @test back[:a1f2].target == tr[:a1f2].target
        @test !is_resolved(back[:b76e].target) && back[:b76e].target.design === :b76e
        @test_throws ArgumentError points(back, :c9)
        # …and resolved when the design's owner supplies its resolver: exact by type
        backr = load(Rig, path; resolve = id -> designs[id])
        @test backr[:b76e].target == tr[:b76e].target && points(backr[:b76e]) == pts
        @test backr[:c9].target == resolve(tr[:c9].target, pts)
        # a rig READ from a file round-trips `==`
        p2 = joinpath(dir, "again.json")
        save(p2, backr)
        @test load(Rig, p2; resolve = id -> designs[id]).members == backr.members
        # a target rig posed in an external frame round-trips the same way
        pw = joinpath(dir, "targets_world.json")
        save(pw, Rig{WorldFrame}(Dict(:b76e => PosedTarget(PlanarTarget(:b76e), Xa))))
        backw = load(Rig, pw)
        @test backw isa Rig{WorldFrame} && ids(backw) == [:b76e]

        # The file's declared length unit is honoured, not assumed: the same
        # numbers declared in metres are a thousand times longer; a non-length
        # unit is refused.
        raw = copy(parse_json(read(path, String)))
        function with_unit(u)
            q = joinpath(dir, "targets_$(u).json")
            write(q, JSON.json(merge(raw, Dict(:length_unit => u))))
            return q
        end
        @test [ustrip.(m, p) for p in points(load(Rig, with_unit("m"))[:a1f2])] ≈ [1000 .* ustrip.(m, p) for p in pts]
        @test_throws ArgumentError load(Rig, with_unit("px"))

        # The content identity measurements name a rig by: the same for the
        # file and for the rig it was written from (resolved or not), and blind
        # to provenance.
        @test content_id(path) == content_id(tr) == content_id(back) == content_id(backr)
        pp = joinpath(dir, "targets_prov.json")
        save(pp, tr; provenance = (note = "relabeled",))
        @test content_id(pp) == content_id(path)
        @test content_id(pw) != content_id(path)
    end
end

@testitem "rig/v4: sensors and targets — the camera rig with its constellation" setup = [VGIoFixtures] begin
    using LinearAlgebra: LinearAlgebra
    using VisualGeometryToolkitCore: Rig, PosedTarget, PointTarget, PlanarTarget, AbstractTarget, Attributed, BoundedCamera, PosedCamera,
        BodyFrame, ViewFrame, Uncertain, CovMat, relabel, ids, pose, extrinsics, model, points,
        sensor_rig, target_rig, exact, is_resolved, Affine
    using Statistics: cov
    K = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 640.0, 480.0, 1)
    Xc = relabel(BodyFrame{:camera_rig}, ViewFrame{:cam}, _mkpose())
    cam = BoundedCamera(PosedCamera(CameraModel(CameraCalibrationMat(K)), Xc), Size2(1280, 960))
    markers = PointTarget([exact_mm(SVector(0.0, 0.0, 0.0)), exact_mm(SVector(30.0, 0.0, 5.0)), exact_mm(SVector(0.0, 40.0, -2.0))])
    ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
    Σ = CovMat(SMatrix{6, 6, Float64}(LinearAlgebra.I) * 1.0e-4)
    rig = Rig{BodyFrame{:camera_rig}}((cam,), Dict(:markers => PosedTarget(markers, Uncertain(ident, Σ))))
    mktempdir() do dir
        path = joinpath(dir, "camera_rig.json")
        save(path, rig)
        d = parse_json(read(path, String))
        @test d.pixel_frame == "colmap" && length(d.sensors) == 1 && length(d.targets) == 1
        @test d.targets[1].target.type == "points" && haskey(d.targets[1].target, :point_covariances)
        @test haskey(d.targets[1].pose, :covariance)
        back = load(Rig, path)
        @test back isa Rig{BodyFrame{:camera_rig}, <:NamedTuple}
        @test ids(back) == [:cam, :markers]
        @test extrinsics(back, :cam) == Xc                                  # a quaternion-born rotation: exact
        @test model(back[:cam]) == model(cam) && extent(back[:cam]) == extent(cam)
        @test back[:markers].target == markers                              # exact points, exactly
        @test pose(back, :markers) isa Uncertain && cov(pose(back, :markers)) == Σ
        @test ids(sensor_rig(back)) == (:cam,) && ids(target_rig(back)) == [:markers]
        @test content_id(back) == content_id(path)
        # the same constellation as a lone target, with its calibration block (target/v1)
        tp = joinpath(dir, "constellation.json")
        cal = (; tool = "mean over tracked frames", n_images = 120)
        save(tp, Attributed(markers, (; calibration = cal)))
        t = load(AbstractTarget, tp)
        @test t isa Attributed && t.value == markers && t.calibration === cal
        save(tp, markers)
        @test load(AbstractTarget, tp) == markers
        dt = parse_json(read(tp, String))
        @test dt.schema == "vgio/target/v1" && !haskey(dt, :calibration)
        @test_throws ArgumentError save(tp, Attributed(markers, (; calibration = (; date = "May 2026"))))
        # a posed target is not a target file: refused at save, naming the model and the rig writer, nothing written
        posed = PosedTarget(markers, ident)
        tq = joinpath(dir, "posed.json")
        for x in (posed, Attributed(posed, (; calibration = cal)))
            e = try save(tq, x); nothing catch err; err end
            @test e isa ArgumentError && occursin("model(pt)", e.msg) && occursin("rig writer", e.msg)
        end
        @test !isfile(tq)
        save(tq, model(posed))                                        # its model is
        @test load(AbstractTarget, tq) == markers
        # a target file holding a referenced design reads back unresolved unless resolved
        save(tp, PlanarTarget(:b76e))
        @test !is_resolved(load(AbstractTarget, tp))
        @test points(load(AbstractTarget, tp; resolve = _ -> [mm_lengths(SVector(1.0, 2.0))])) == [mm_lengths(SVector(1.0, 2.0))]
        # the v1 record's warp is rigid: any other target calibration is refused at save, nothing written
        ta = joinpath(dir, "affine.json")
        affine = Affine([1.001 0.0; 0.0 1.0], [0.1, 0.0] .* mm)
        e = try save(ta, PlanarTarget([mm_lengths(SVector(1.0, 2.0))]; calibration = affine)); nothing catch err; err end
        @test e isa ArgumentError && occursin("Affine", e.msg) && occursin("vgio/target/v1", e.msg)
        @test !isfile(ta)
    end
end

@testitem "rig/v4: the validator" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, PointTarget, WorldFrame, exact
    pts = [mm_lengths(SVector(0.0, 0.0)), mm_lengths(SVector(10.0, 0.0))]
    X = EuclideanMap(one(RotMatrix{3, Float64}), [10.0mm, 0.0mm, 0.0mm])
    mktempdir() do dir
        path = joinpath(dir, "r.json")
        save(path, Rig{WorldFrame}(Dict(:a => PosedTarget(PlanarTarget(pts), X))))
        good = JSON.parse(read(path, String); dicttype = Dict{String, Any})
        function refused(f, key)
            dd = deepcopy(good)
            f(dd)
            q = joinpath(dir, "bad.json")
            write(q, JSON.json(dd))
            e = try
                load(Rig, q); nothing
            catch e
                e
            end
            return e isa ArgumentError && occursin(key, e.msg)
        end
        @test refused(dd -> (dd["pixel_frame"] = "colmap"), "pixel_frame")             # declared iff sensors
        @test refused(dd -> (dd["targets"] = []), "omitted, not empty")
        @test refused(dd -> delete!(dd, "targets"), "sensors, targets, or both")
        @test refused(dd -> (dd["targets"][1]["target"]["design"] = "a"), "not both")     # design and nominal
        @test refused(dd -> (dd["targets"][1]["target"]["type"] = "spline"), "unknown target type")
        @test refused(dd -> (dd["targets"][1]["target"]["warp"] = Dict("type" => "affine")), "rigid")
        @test refused(dd -> delete!(dd["targets"][1]["target"], "warp"), "warp")
        @test refused(dd -> (dd["targets"][1]["target"] = Dict("type" => "points", "points" => [[1, 2], [1, 2, 3]])), "2-vectors")
        @test refused(dd -> push!(dd["targets"], deepcopy(dd["targets"][1])), "unique")
        # plain and uncertain points are not mixed in one target
        @test_throws ArgumentError save(joinpath(dir, "mixed.json"), Rig{WorldFrame}(Dict(:a => PosedTarget(PointTarget([mm_lengths(SVector(1.0, 2.0)), exact_mm(SVector(3.0, 4.0))]), X))))
    end
end

@testitem "retired kinds name their successor" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, TargetPoints2D, TargetPoints3D
    mktempdir() do dir
        for (schema, T, successor) in (
                ("vgio/observations/v2", TargetPoints2D, "target_points2d/v1"),
                ("vgio/point_observations/v1", TargetPoints3D, "target_points3d/v1"),
                ("vgio/targets/v2", Rig, "rig/v4"),
                ("vgio/rig/v3", Rig, "rig/v4"),
                ("vgio/session/v3", Rig, "calibration pipeline's own session kind"),
            )
            p = joinpath(dir, "old.json")
            write(p, JSON.json(Dict("schema" => schema)))
            e = try
                load(T, p); nothing
            catch e
                e
            end
            @test e isa ArgumentError && occursin(successor, e.msg) && occursin("retired", e.msg)
            @test_throws ArgumentError VgtkIO.validate(p)
        end
    end
end

@testitem "trajectory round trip" setup = [VGIoFixtures] begin
    using LinearAlgebra: LinearAlgebra
    using VisualGeometryToolkitCore: Trajectory, WorldFrame, BodyFrame, Uncertain, CovMat, geom
    using Rotations: RotXYZ
    using Statistics: cov

    const POSE_ULP = 1.0e-15

    traj = Dict(
        3 => EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, 0.0, 0.2)), [10.0mm, 20.0mm, 900.0mm]),
        1 => EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.3, -0.1)), [-5.0mm, 0.0mm, 800.0mm]),
    )
    Σ = CovMat(SMatrix{6, 6, Float64}(LinearAlgebra.I) * 1.0e-4)
    mktempdir() do dir
        tr = Trajectory(
            BodyFrame{:board_rig}, WorldFrame, Dict(1 => traj[1], 3 => Uncertain(traj[3], Σ));
            images = Dict(1 => "DSC0001.JPG", 3 => "DSC0003.JPG"), times = Dict(1 => 0.5, 3 => 2.25),
            time_origin = "2026-05-27T00:00:00", recovered = Set([3]),
        )
        path = joinpath(dir, "trajectory.json")
        save(path, tr)
        back = load(Trajectory, path)
        # the direction is part of the data, typed, and it comes back with the poses
        @test back isa Trajectory{BodyFrame{:board_rig}, WorldFrame}
        @test geom(back[1]) isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:board_rig}, WorldFrame}
        @test sort(collect(keys(back))) == [1, 3]
        for t in (1, 3)
            @test geom(back[t]).t == traj[t].t
            @test maximum(abs, Matrix(geom(back[t]).R) - Matrix(traj[t].R)) <= POSE_ULP
        end
        @test back[1] isa EuclideanMap && back[3] isa Uncertain && cov(back[3]) == Σ
        @test back.images == tr.images && back.times == tr.times && back.time_origin == tr.time_origin
        @test back.recovered == Set([3])
        d = parse_json(read(path, String))
        @test d.schema == "vgio/trajectory/v3" && (d.from, d.to) == ("body:board_rig", "world")
        @test !haskey(d.frames[1], :recovered) && d.frames[2].recovered == true      # written only when true
        # read → write → read is exact
        p2 = joinpath(dir, "again.json")
        save(p2, back)
        @test load(Trajectory, p2) == back
        # a quaternion-born trajectory round-trips `==` from the start
        qb = Trajectory(BodyFrame{:a}, WorldFrame, Dict(1 => _mkpose()))
        save(p2, qb)
        @test load(Trajectory, p2) == qb
    end
end

@testitem "target poses round trip (vgio/target_poses/v1)" setup = [VGIoFixtures] begin
    using LinearAlgebra: LinearAlgebra
    using VisualGeometryToolkitCore: TargetPoses, CameraFrame, TargetFrame, Uncertain, CovMat, pose
    using Rotations: RotXYZ
    using Statistics: cov
    X(z) = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3)), [1.0mm, 2.0mm, z])
    Σ = CovMat(SMatrix{6, 6, Float64}(LinearAlgebra.I) * 1.0e-3)
    mktempdir() do dir
        tp = TargetPoses{CameraFrame}(
            Dict(4 => Dict(:b89 => X(900.0mm), Symbol("006") => Uncertain(X(1200.0mm), Σ)), 7 => Dict{Symbol, Any}());
            images = Dict(4 => "DSC04523.JPG", 7 => "DSC04526.JPG"), sensors = "sony-fx3", times = Dict(4 => 1.5),
            time_origin = "2026-05-27T00:00:00",
        )
        path = joinpath(dir, "target_poses.json")
        save(path, tp)
        back = load(TargetPoses, path)
        @test sort(collect(keys(back))) == [4, 7] && isempty(back[7])          # a frame with no detection stays a frame
        @test back.images == tp.images && back.sensors == Dict(4 => "sony-fx3", 7 => "sony-fx3") && back.times == tp.times
        P = pose(back, 4, :b89)
        @test P isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:b89}, CameraFrame} && P.t == X(900.0mm).t
        U = pose(back, 4, Symbol("006"))
        @test U isa Uncertain && cov(U) == Σ
        d = parse_json(read(path, String))
        @test d.schema == "vgio/target_poses/v1" && d.to == "camera_optical"
        @test [p.target for p in d.frames[1].poses] == ["006", "b89"]           # ids sorted; an undetected target absent
        # read → write → read is exact
        p2 = joinpath(dir, "again.json")
        save(p2, back)
        @test load(TargetPoses, p2) == back
        # every frame names its image and its sensor: the writer refuses a value without them
        @test_throws ArgumentError save(p2, TargetPoses{CameraFrame}(Dict(1 => Dict(:a => X(1.0mm))); sensors = "c"))
        @test_throws ArgumentError save(p2, TargetPoses{CameraFrame}(Dict(1 => Dict(:a => X(1.0mm))); images = Dict(1 => "a")))
    end
end

@testitem "target points 3D round trip (vgio/target_points3d/v1)" setup = [VGIoFixtures] begin
    using LinearAlgebra: LinearAlgebra
    using VisualGeometryToolkitCore: TargetPoints3D, WorldFrame, Uncertain, CovMat, geom, exact
    using Statistics: cov
    mktempdir() do dir
        obs = TargetPoints3D{WorldFrame}(
            Dict(
                1 => [(target = :board_rig, index = 1, xyz = mm_lengths(SVector(1140.24, 327.55, 1614.39))), (target = :camera_rig, index = 3, xyz = exact_mm(SVector(2113.1, 24.0, 1319.0)))],
                2 => [(target = :board_rig, index = 2, xyz = Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), CovMat(SMatrix{3, 3, Float64}(LinearAlgebra.I) * 2.5e-7)))],
            );
            sensors = "mocap", images = Dict(1 => "DSC1.JPG"), times = Dict(1 => 10.0, 2 => 10.5), time_origin = "2026-05-27T00:00:00",
        )
        path = joinpath(dir, "markers.json")
        save(path, obs)
        back = load(TargetPoints3D, path)
        @test back == obs
        @test back[1][1].xyz isa SVector && back[1][2].xyz isa Uncertain && iszero(cov(back[1][2].xyz))
        d = parse_json(read(path, String))
        @test d.schema == "vgio/target_points3d/v1" && d.frame == "world" && !haskey(d, :targets)
        @test !haskey(d.frames[1].points[1], :covariance) && haskey(d.frames[1].points[2], :covariance)
        # a file in metres reads into the same lengths, its covariance (m²) as it is
        dm = JSON.parse(read(path, String); dicttype = Dict{String, Any})
        dm["length_unit"] = "m"
        dm["frames"][2]["points"][1]["xyz"] = [0.001, 0.002, 0.003]
        dm["frames"][2]["points"][1]["covariance"] = [[2.5e-7, 0, 0], [0, 2.5e-7, 0], [0, 0, 2.5e-7]]
        pm = joinpath(dir, "metres.json")
        write(pm, JSON.json(dm))
        u = load(TargetPoints3D, pm)[2][1].xyz
        @test geom(u) == mm_lengths(SVector(1.0, 2.0, 3.0)) && cov(u) == SMatrix{3, 3, Float64}(LinearAlgebra.I) * 2.5e-7
        # every frame names its sensor
        @test_throws ArgumentError save(joinpath(dir, "x.json"), TargetPoints3D{WorldFrame}(Dict(1 => [(target = :a, index = 1, xyz = mm_lengths(SVector(1.0, 2.0, 3.0)))])))
    end
end

@testitem "exporters: TUM, EuRoC, BOP, TRC read back through their text" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Trajectory, TargetPoses, TargetPoints3D, WorldFrame, BodyFrame, CameraFrame, geom, BoundedCamera
    using Rotations: RotXYZ, QuatRotation
    poses = Dict(t => EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1t, -0.05t, 0.02t)), [100.0t * mm, -20.0mm, 1500.0mm]) for t in 1:4)
    tr = Trajectory(BodyFrame{:board_rig}, WorldFrame, poses; times = Dict(t => 46800.0 + 12.5t for t in 1:4), time_origin = "2026-05-27T00:00:00")
    mktempdir() do dir
        # TUM: `t tx ty tz qx qy qz qw`, metres, comments skipped (evo file_interface.py:121-146)
        p = save_tum_trajectory(joinpath(dir, "tum.txt"), tr)
        rows = [parse.(Float64, split(l)) for l in eachline(p) if !startswith(l, "#")]
        @test length(rows) == 4 && all(length.(rows) .== 8)
        for (t, r) in zip(1:4, rows)
            @test r[1] == tr.times[t]
            @test SVector(r[2], r[3], r[4]) == ustrip.(m, geom(tr[t]).t)       # metres, exactly
            @test RotMatrix(QuatRotation(r[8], r[5], r[6], r[7])) ≈ RotMatrix(geom(tr[t]).R) atol = 1.0e-12
        end
        # EuRoC: `#timestamp [ns], p_RS_R_*, q_RS_w…`, comma-separated, ≥ 8 columns (file_interface.py:224-243)
        p = save_euroc_trajectory(joinpath(dir, "data.csv"), tr)
        lines = readlines(p)
        @test startswith(lines[1], "#timestamp [ns]")
        for (t, l) in zip(1:4, lines[2:end])
            v = split(l, ',')
            @test length(v) == 8 && parse(Int64, v[1]) == round(Int64, tr.times[t] * 1.0e9)
            q = parse.(Float64, v[5:8])
            @test RotMatrix(QuatRotation(q...)) ≈ RotMatrix(geom(tr[t]).R) atol = 1.0e-12
        end
        @test_throws ArgumentError save_tum_trajectory(joinpath(dir, "x.txt"), Trajectory(BodyFrame{:a}, WorldFrame, poses))
        save_tum_trajectory(joinpath(dir, "idx.txt"), Trajectory(BodyFrame{:a}, WorldFrame, poses); stamp = :index)
        # BOP: scene_gt (cam_R_m2c row-major, cam_t_m2c mm, integer obj_id) + scene_camera (cam_K row-major)
        tp = TargetPoses{CameraFrame}(Dict(2 => Dict(:b89 => poses[2], :a01 => poses[3])); images = Dict(2 => "DSC2.JPG"), sensors = "cam")
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 901.0, 0, 640.5, 480.5, 1))
        cams = Dict(:cam => BoundedCamera(CameraModel(K; lens = opencv_lens([-0.1, 0.01, 0.0, 0.0]), image_origin = :colmap), Size2(1280, 960)))
        @test_throws ArgumentError save_bop_scene(PixelFrame{:opencv}, joinpath(dir, "bop"), tp, cams)   # a lens, not a pinhole
        out = save_bop_scene(PixelFrame{:opencv}, joinpath(dir, "bop"), tp, cams; undistorted = true)
        gt = parse_json(read(out.scene_gt, String))
        e = only(o for o in gt[Symbol("2")] if o.obj_id == out.obj_ids[:b89])
        @test permutedims(reshape(collect(Float64, e.cam_R_m2c), 3, 3)) ≈ RotMatrix(poses[2].R) atol = 1.0e-15
        @test collect(Float64, e.cam_t_m2c) == [200.0, -20.0, 1500.0]
        sc = parse_json(read(out.scene_camera, String))
        @test collect(Float64, sc[Symbol("2")].cam_K) == [900.0, 0.0, 640.0, 0.0, 901.0, 480.0, 0.0, 0.0, 1.0]   # OpenCV pixel frame
        @test out.obj_ids == Dict(:a01 => 1, :b89 => 2) && out.images == Dict(2 => "DSC2.JPG")
        @test_throws ArgumentError save_bop_scene(PixelFrame{:opencv}, joinpath(dir, "bop2"), tp, Dict(:other => cams[:cam]); undistorted = true)
        # TRC: header lines, tab-separated with a trailing tab, missing markers as empty fields (TRCFileAdapter.cpp)
        obs = TargetPoints3D{WorldFrame}(
            Dict(1 => [(target = :rig, index = 1, xyz = mm_lengths(SVector(1.0, 2.0, 3.0))), (target = :rig, index = 2, xyz = mm_lengths(SVector(4.0, 5.0, 6.0)))],
                 2 => [(target = :rig, index = 2, xyz = mm_lengths(SVector(7.0, 8.0, 9.0)))]);
            sensors = "mocap", times = Dict(1 => 0.0, 2 => 0.5), time_origin = "2026-05-27T00:00:00",
        )
        r = save_trc(joinpath(dir, "markers.trc"), obs)
        L = readlines(r.path)
        @test startswith(L[1], "PathFileType\t4\t(X/Y/Z)\t")
        @test split(L[2], '\t')[1:8] == ["DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units", "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames"]
        @test split(L[3], '\t')[3:5] == ["2", "2", "mm"]
        @test split(L[4], '\t')[1:3] == ["Frame#", "Time", "rig_1"] && r.markers == ["rig_1", "rig_2"]
        @test isempty(L[6])
        f2 = split(L[8], '\t')
        @test f2[1:2] == ["2", "0.5"] && all(isempty, f2[3:5]) && parse.(Float64, f2[6:8]) == [7.0, 8.0, 9.0]
    end
end

@testitem "target points 2D and the correspondences export" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, TargetPoints2D, Trajectory, TargetFrame, WorldFrame,
        Uncertain, CovMat, convert_frame
    using Statistics: cov

    raw = Dict(
        3 => NamedTuple[(; target = :a1f2, index = 7, xy = SVector(100.0, 50.0), weight = 1.0), (; target = :b0c1, index = 1, xy = SVector(1.5, 2.5), weight = 0.5)],
        1 => NamedTuple[(; target = :a1f2, index = 2, xy = SVector(640.0, 480.0), weight = 1.0)],
    )
    pts = [mm_lengths(SVector(10.0i, 5.0i)) for i in 1:7]
    at(z) = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, z])
    targets = Rig{TargetFrame{:a1f2}}(Dict(:a1f2 => PosedTarget(PlanarTarget(pts), at(0.0mm)), :b0c1 => PosedTarget(PlanarTarget(pts), at(100.0mm))))
    images = Dict(1 => "DSC0001.png", 3 => "DSC0003.png")
    mktempdir() do dir
        targets_path = joinpath(dir, "targets.json")
        save(targets_path, targets)
        # the producer attaches the frame its pixels are in — here Julia's — and names the targets
        obs = TargetPoints2D{PixelFrame{:julia}}(raw; images, sensors = "cam", targets = [content_id(targets_path)])
        path = joinpath(dir, "target_points.json")
        # the file declares the frame the TYPE carries, and no pixel is converted
        save(path, obs)
        d = parse_json(read(path, String))
        @test d.schema == "vgio/target_points2d/v1" && d.pixel_frame == "julia" && d.point_index_base == 1
        @test d.frames[1].points[1].xy == [640.0, 480.0]
        @test d.targets == [content_id(targets)]
        @test d.frames[2].image == "DSC0003.png" && d.frames[2].sensor == "cam"
        # …and comes back typed in it, exactly
        back = load(TargetPoints2D, path)
        @test back isa TargetPoints2D{PixelFrame{:julia}} && back == obs
        @test back[3][2] == (target = :b0c1, index = 1, xy = SVector(1.5, 2.5), weight = 0.5)
        # a frame changes only with its numbers (VGC's convert_frame)
        canon = convert_frame(PixelFrame{:colmap}, back)
        @test canon isa TargetPoints2D{PixelFrame{:colmap}} && canon[1][1].xy == SVector(639.5, 479.5)
        @test convert_frame(PixelFrame{:julia}, canon) == obs
        # an uncertain pixel keeps its 2×2 covariance through the file
        C = CovMat(SMatrix{2, 2, Float64}(0.25, 0.0, 0.0, 0.25))
        uo = TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(target = :a1f2, index = 1, xy = Uncertain(SVector(640.0, 480.0), C), weight = 1.0)]);
            images = Dict(1 => "a.png"), sensors = "cam", targets = [content_id(targets)])
        up = joinpath(dir, "u.json")
        save(up, uo)
        ub = load(TargetPoints2D, up)
        @test ub == uo && ub[1][1].xy isa Uncertain && cov(ub[1][1].xy) == C

        # everything that makes the file sufficient is required
        p = joinpath(dir, "o.json")
        @test_throws ArgumentError save(p, TargetPoints2D{PixelFrame{:julia}}(raw; images, sensors = "cam"))                   # targets
        @test_throws ArgumentError save(p, TargetPoints2D{PixelFrame{:julia}}(raw; images = Dict(1 => "a.png"), sensors = "cam", targets = obs.targets))
        @test_throws ArgumentError save(p, TargetPoints2D{PixelFrame{:julia}}(raw; images, targets = obs.targets))           # sensors
        save(p, TargetPoints2D{PixelFrame{:julia}}(raw; images, sensors = Dict(1 => "left", 3 => "right"), targets = obs.targets))
        @test [f.sensor for f in parse_json(read(p, String)).frames] == ["left", "right"]

        # the export: one row per (frame, target), in an external calibrator's frame —
        # OpenCV's puts the pixel Julia's calls (640, 480) at (639, 479)
        cpath = joinpath(dir, "correspondences.json")
        @test save_correspondences(PixelFrame{:opencv}, cpath, obs, targets_path) == cpath
        c = parse_json(read(cpath, String))
        @test c.schema == "vgio/correspondences/v2" && c.pixel_frame == "opencv" && c.length_unit == "mm"
        @test [(r.t, r.target) for r in c.rows] == [(1, "a1f2"), (3, "a1f2"), (3, "b0c1")]      # frame 3's two boards stay apart
        @test c.rows[1].image == "DSC0001.png" && c.rows[1].sensor == "cam"
        @test c.rows[1].image_points[1] == [639.0, 479.0] && c.rows[1].object_points[1] == [20.0, 10.0, 0.0]
        @test c.rows[3].image_points[1] == [0.5, 1.5] && c.rows[3].object_points[1] == [10.0, 5.0, 0.0]
        # the private reader (the migration's verification) sees the rows as written
        f, rows = VgtkIO._read_correspondences(cpath)
        @test f === PixelFrame{:opencv} && length(rows) == 3
        # default: the measurements' own frame
        jpath = joinpath(dir, "julia_rows.json")
        save_correspondences(jpath, obs, targets)
        @test parse_json(read(jpath, String)).pixel_frame == "julia"
        # measurements resolve only against the targets they name
        other = joinpath(dir, "other_targets.json")
        save(other, Rig{TargetFrame{:a1f2}}(Dict(:a1f2 => PosedTarget(PlanarTarget(pts), at(0.0mm)), :b0c1 => PosedTarget(PlanarTarget(pts), at(50.0mm)))))
        @test_throws ArgumentError save_correspondences(joinpath(dir, "x.json"), obs, other)
        # every (target, index) must resolve
        bad = TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(; target = :a1f2, index = 8, xy = SVector(1.0, 2.0), weight = 1.0)]); images = Dict(1 => "a.png"), sensors = "cam")
        @test_throws ArgumentError save_correspondences(joinpath(dir, "x.json"), bad, targets)
        nob = TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(; target = :zz, index = 1, xy = SVector(1.0, 2.0), weight = 1.0)]); images = Dict(1 => "a.png"), sensors = "cam")
        @test_throws ArgumentError save_correspondences(joinpath(dir, "x.json"), nob, targets)
        # unresolved designs need their resolver
        refd = Rig{TargetFrame{:a1f2}}(Dict(:a1f2 => PosedTarget(PlanarTarget(:a1f2; nominal = pts), at(0.0mm)), :b0c1 => PosedTarget(PlanarTarget(:b0c1; nominal = pts), at(100.0mm))))
        rp = joinpath(dir, "refd.json")
        save(rp, refd)
        o2 = TargetPoints2D{PixelFrame{:julia}}(raw; images, sensors = "cam", targets = [content_id(rp)])
        @test_throws ArgumentError save_correspondences(joinpath(dir, "x.json"), o2, rp)
        save_correspondences(joinpath(dir, "y.json"), o2, rp; resolve = _ -> pts)
        @test parse_json(read(joinpath(dir, "y.json"), String)).rows[1].object_points[1] == [20.0, 10.0, 0.0]
        # a trajectory file is not target points
        tpath = joinpath(dir, "trajectory.json")
        save(tpath, Trajectory(TargetFrame{:a1f2}, WorldFrame, Dict(1 => at(900.0mm))))
        @test_throws ArgumentError load(TargetPoints2D, tpath)
    end
end
