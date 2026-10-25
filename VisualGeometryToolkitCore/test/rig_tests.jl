@testitem "rig" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model, reference_frame, points
    using StaticArrays
    using LinearAlgebra: LinearAlgebra, norm
    using Statistics: cov
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore.Synthetic: imc_pose_error

    # =============================================================================
    # Rig — one type for sensors and targets; the frame is the gauge
    # =============================================================================
    #
    # Covers: the Rig{F} construction invariants (rig-relative labels, ViewFrame
    # destinations, distinct ids, the gauge sensor at the identity), the 1-sensor
    # rig (a moving camera's "rig of one"), per-sensor label-preserving accessors,
    # StereoRig as the 2-tuple alias (fields gone, accessors derived from the
    # stored per-camera maps), BoundedCamera wrapper preservation through
    # re-expression, and the rig of targets: dict storage, labels minted
    # (TargetFrame{id} → F), the gauge target at the identity, world points.

    @testset "Rig — sensors and targets" begin
        K1 = @SMatrix [600.0 0.0 320.0; 0.0 600.0 240.0; 0.0 0.0 1.0]
        K2 = @SMatrix [700.0 0.0 330.0; 0.0 700.0 250.0; 0.0 0.0 1.0]
        R = RotMatrix{3, Float64}(RotXYZ(0.08, -0.15, 0.05))
        t = SVector(0.4, -0.15, 0.1)

        @testset "construction invariants" begin
            cam = PosedCamera(K1, one(RotMatrix{3, Float64}), zero(Vec3{Float64}))
            # unlabeled extrinsics: Src is not the rig frame
            @test_throws ArgumentError Rig{WorldFrame}((cam,))
            # labeled but destination is not a ViewFrame
            wrong = PosedCamera(model(cam), relabel(WorldFrame, CameraFrame, cam.extrinsics))
            @test_throws ArgumentError Rig{WorldFrame}((wrong,))
            # duplicate view ids
            c1 = PosedCamera(model(cam), relabel(WorldFrame, ViewFrame{:a}, cam.extrinsics))
            c2 = PosedCamera(
                model(cam), relabel(WorldFrame, ViewFrame{:a}, EuclideanMap(R, t))
            )
            @test_throws ArgumentError Rig{WorldFrame}((c1, c2))
            # empty
            @test_throws ArgumentError Rig{WorldFrame}(())
        end

        @testset "rig of one — the moving-camera case" begin
            cam = PosedCamera(K1, R, t)
            c = PosedCamera(model(cam), relabel(WorldFrame, ViewFrame{:cam}, cam.extrinsics))
            rig = Rig{WorldFrame}((c,))
            @test length(rig) == 1
            @test ids(rig) == (:cam,)
            @test view_frames(rig) == (ViewFrame{:cam},)
            @test image_frames(rig) == (ImageFrame{:cam},)
            # label-preserving per-sensor accessors (NOT the WorldFrame→CameraFrame
            # relabel of extrinsics(::PosedCamera))
            X = extrinsics(rig, 1)
            @test X isa EuclideanMap{3, Float64, <:Any, <:Any, WorldFrame, ViewFrame{:cam}}
            P = pose(rig, 1)
            @test P isa EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:cam}, WorldFrame}
            Xw = SVector(0.3, -0.2, 4.0)
            @test P(X(Xw)) ≈ Xw atol = 1.0e-12
            # frame deduction from the first sensor
            @test Rig((c,)) isa Rig{WorldFrame}
        end

        @testset "StereoRig is the 2-tuple Rig" begin
            rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t; ids = (:left, :right))
            @test rig isa Rig
            @test rig isa StereoRig
            @test rig isa Rig{ViewFrame{:left}}          # reference frame = view 1
            @test fieldnames(typeof(rig)) == (:members,)
            @test length(rig) == 2
            @test ids(rig) == (:left, :right)
            # camera 1 stored rig-relative: the identity on the rig frame
            X1 = extrinsics(rig, 1)
            @test X1 isa
                EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:left}, ViewFrame{:left}}
            @test norm(X1.t) == 0.0
            # camera 2 stored rig-relative: the labeled relative transform, and
            # the derived stereo accessor equals it exactly
            X2 = extrinsics(rig, 2)
            @test X2 isa
                EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:left}, ViewFrame{:right}}
            Xrel = extrinsics(rig)
            @test Xrel.t == X2.t && RotMatrix(Xrel.R) == RotMatrix(X2.R)
            # the reference view is the gauge: a rig gauged on a view whose
            # extrinsics is not the identity is refused
            cam2 = PosedCamera(K2, R, t)
            moved = PosedCamera(model(cam2), relabel(ViewFrame{:left}, ViewFrame{:left}, EuclideanMap(R, t)))
            @test_throws ArgumentError Rig{ViewFrame{:left}}((moved,))
            # a 1-sensor rig has no stereo surface
            cam = PosedCamera(K1, R, t)
            c = PosedCamera(model(cam), relabel(WorldFrame, ViewFrame{:cam}, cam.extrinsics))
            mono = Rig{WorldFrame}((c,))
            @test_throws MethodError source(mono)
            @test_throws MethodError extrinsics(mono)
        end

        @testset "BoundedCamera survives re-expression" begin
            cam1 = PosedCamera(K1, one(RotMatrix{3, Float64}), zero(Vec3{Float64}))
            cam2 = PosedCamera(K2, R, Vec3{Float64}(t))
            b1 = BoundedCamera(cam1, Size2(640, 480))
            rig = StereoRig(b1, cam2; ids = (:l, :r))
            @test source(rig) isa BoundedCamera
            @test extent(source(rig)) == Size2(640, 480)
            @test target(rig) isa PosedCamera
            @test ids(rig) == (:l, :r)
        end

        @testset "an unposed member: check = false admits it, everything else refuses" begin
            cm = CameraModel(CameraCalibrationMat(K1))
            u = BoundedCamera(cm, Size2(640, 480))
            @test_throws ArgumentError pose(u)
            @test_throws ArgumentError extrinsics(u)
            E = EuclideanMap(R, Vec3{Float64}(t))
            rig = StereoRig(u, u, E; check = false)        # the rig poses its members
            @test rig isa StereoRig
            @test VgtkCore.has_extrinsics(source(rig)) && VgtkCore.has_extrinsics(target(rig))
            @test extrinsics(rig).t ≈ t && Matrix(extrinsics(rig).R) ≈ Matrix(R)   # the given transform survives
            @test_throws ArgumentError StereoRig(u, u, E)   # checked: members must be posed
            @test_throws ArgumentError StereoRig(u, u)       # computed from the members
        end

        @testset "a rig of targets — dict storage, labels minted, the frame is the gauge" begin
            Xa = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, 0.2, -0.3)), SVector(1.0, 2.0, 3.0))
            Xb = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(-0.2, 0.05, 0.4)), SVector(-1.0, 0.5, 2.0))
            pts = [SVector(0.0, 0.0), SVector(10.0, 0.0), SVector(0.0, 5.0)]
            # posed in an external frame: no gauge member, any pose goes; pose-only boards
            # are unresolved planar targets (a design reference, no points yet)
            tr = Rig{WorldFrame}(Dict(:b => Mounted(PlanarTarget(:b), Xb), :a => Mounted(PlanarTarget(:a), Xa)))
            @test tr isa Rig && length(tr) == 2
            @test haskey(tr, :a) && !haskey(tr, :z)
            @test ids(tr) == [:a, :b]                     # sorted, deterministic
            @test target_frames(tr) == [TargetFrame{:a}, TargetFrame{:b}]
            Pa = pose(tr, :a)
            @test Pa isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:a}, WorldFrame}
            @test Pa.t == Xa.t
            Ea = extrinsics(tr, :a)
            @test Ea isa EuclideanMap{3, Float64, <:Any, <:Any, WorldFrame, TargetFrame{:a}}
            Xt = SVector(0.1, 0.2, 0.0)
            @test Ea(Pa(Xt)) ≈ Xt atol = 1.0e-12
            @test tr[:a] isa Mounted && pose(tr[:a]).t == Xa.t && !is_resolved(tr[:a].target)
            @test_throws ArgumentError points(tr, :a)      # unresolved: no pattern geometry
            @test isempty(points(tr))
            # gauged on a member: that member's frame IS the rig frame, its pose the identity
            ident = EuclideanMap(one(RotMatrix{3, Float64}), zero(SVector{3, Float64}))
            g = Rig{TargetFrame{:a}}(Dict(:a => Mounted(PlanarTarget(pts), ident), :b => Mounted(PlanarTarget(:b; nominal = pts), Xb)))
            @test reference_frame(g) === TargetFrame{:a}
            @test pose(g, :b) isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:b}, TargetFrame{:a}}
            @test norm(pose(g, :a).t) == 0.0
            # world points: the pattern points through each target's pose, lifted to z = 0
            @test points(g, :a) == [SVector(0.0, 0.0, 0.0), SVector(10.0, 0.0, 0.0), SVector(0.0, 5.0, 0.0)]
            @test points(g, :b)[2] ≈ Xb(SVector(10.0, 0.0, 0.0)) atol = 1.0e-12
            @test Set(keys(points(g))) == Set([:a, :b])
            # the gauge invariants: the gauge member exists, and sits at the identity
            @test_throws ArgumentError Rig{TargetFrame{:a}}(Dict(:a => Mounted(PlanarTarget(:a), Xa)))
            @test_throws ArgumentError Rig{TargetFrame{:z}}(Dict(:a => Mounted(PlanarTarget(:a), ident)))
            # uncertain members (conventions/persisted-geometry.md): an estimated mount on a
            # body, its covariance kept through the minted labels; estimated points a PointTarget
            Σ = CovMat(SMatrix{6, 6, Float64}(LinearAlgebra.I) * 1.0e-4)
            ub = Rig{BodyFrame{:board_rig}}(Dict(:a => Mounted(PointTarget([exact(p) for p in pts]), Uncertain(Xa, Σ))))
            Pu = pose(ub, :a)
            @test Pu isa Uncertain
            @test geom(Pu) isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:a}, BodyFrame{:board_rig}}
            @test cov(Pu) == Σ && geom(Pu).t == Xa.t
            @test points(ub, :a)[2] ≈ Xa(SVector(10.0, 0.0, 0.0)) atol = 1.0e-12
            @test_throws ArgumentError extrinsics(ub, :a)          # a covariance does not invert by relabeling
            @test_throws ArgumentError Rig{WorldFrame}(Dict{Symbol, Mounted}())
            @test_throws ArgumentError Rig{WorldFrame}(Dict(:a => PlanarTarget(pts)))   # a member is Mounted
            # a member's geometry is independent of its mount: the same target in two rigs
            board = PlanarTarget(:a; nominal = pts)
            r1 = Rig{WorldFrame}(Dict(:a => Mounted(board, Xa)))
            r2 = Rig{BodyFrame{:board_rig}}(Dict(:a => Mounted(board, Xb)))
            @test r1[:a].target === r2[:a].target && points(r1[:a]) == points(r2[:a]) == pts
        end

        @testset "a rig of sensors and targets — the camera rig with its constellation" begin
            ident = EuclideanMap(one(RotMatrix{3, Float64}), zero(SVector{3, Float64}))
            Xc = relabel(BodyFrame{:camera_rig}, ViewFrame{:cam}, EuclideanMap(RotMatrix{3, Float64}(R), t))
            cam = BoundedCamera(PosedCamera(CameraModel(CameraCalibrationMat(K1)), Xc), Size2(640, 480))
            markers = PointTarget([SVector(0.0, 0.0, 0.0), SVector(30.0, 0.0, 5.0), SVector(0.0, 40.0, -2.0)])
            rig = Rig{BodyFrame{:camera_rig}}((cam,), Dict(:markers => Mounted(markers, ident)))
            @test ids(rig) == [:cam, :markers] && length(rig) == 2
            @test haskey(rig, :cam) && haskey(rig, :markers) && !haskey(rig, :z)
            @test rig[:cam] === cam && rig[1] === cam && rig[:markers].target === markers
            @test extrinsics(rig, :cam) === Xc && pose(rig, 1) isa EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:cam}, BodyFrame{:camera_rig}}
            @test pose(rig, :markers) isa EuclideanMap{3, Float64, <:Any, <:Any, TargetFrame{:markers}, BodyFrame{:camera_rig}}
            @test points(rig, :markers) == points(markers)
            @test view_frames(rig) == (ViewFrame{:cam},) && target_frames(rig) == [TargetFrame{:markers}]
            @test sensor_rig(rig) isa Rig{BodyFrame{:camera_rig}, <:Tuple} && ids(sensor_rig(rig)) == (:cam,)
            @test target_rig(rig) isa Rig{BodyFrame{:camera_rig}, <:AbstractDict} && ids(target_rig(rig)) == [:markers]
            @test_throws ArgumentError Rig{BodyFrame{:camera_rig}}((cam,), Dict(:cam => Mounted(markers, ident)))   # ids distinct
        end

        @testset "targets — the model, the warp, the calibratable protocol" begin
            pts = [SVector(0.0, 0.0), SVector(10.0, 0.0), SVector(0.0, 5.0)]
            p = PlanarTarget(:b76e; nominal = pts)
            @test points(p) == pts && p.warp === Rigid() && is_resolved(p)
            u = PlanarTarget(:b76e)
            @test !is_resolved(u) && resolve(u, pts) == p
            @test_throws ArgumentError points(u)
            @test_throws ArgumentError PlanarTarget(nothing, nothing, Rigid())
            # Rigid: no parameters, no regularizer, a zero-dimensional manifold
            @test isempty(parameter_vector(p)) && parameter_names(p) == () && parameters(p) == (;)
            @test isempty(regularizer(p)) && with_parameters(p, Float64[]) == p
            ch = VgtkCore.minimal_manifold(p)
            @test ch isa VgtkCore.TargetManifold && VgtkCore.tangent_size(ch) == 0 && model(ch) === p
            @test size(VgtkCore.action_jacobian(ch, pts[2])) == (2, 0)
            @test_throws ArgumentError with_parameters(p, [1.0])
            # PointTarget: explicit points, one dimension, plain or Uncertain
            c = PointTarget([SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)])
            @test points(c) == [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
            @test_throws ArgumentError PointTarget([SVector(1.0, 2.0), SVector(1.0, 2.0, 3.0)])
        end

        @testset "with_model — same geometry, new camera model" begin
            rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t; ids = (:left, :right))
            m2 = CameraModel(CameraCalibrationMat(K2); lens = VgtkCore.Division(-0.2))
            rig2 = with_model(rig, 2 => m2)
            @test ids(rig2) == (:left, :right)
            @test model(target(rig2)) === m2
            @test model(source(rig2)) === model(source(rig))
            X0, X1 = extrinsics(rig), extrinsics(rig2)          # geometry untouched
            @test X1.t == X0.t && RotMatrix(X1.R) == RotMatrix(X0.R)
            @test typeof(X1) == typeof(X0)                       # labels preserved
            # BoundedCamera wrapper survives
            cam1 = PosedCamera(K1, one(RotMatrix{3, Float64}), zero(Vec3{Float64}))
            cam2 = PosedCamera(K2, R, Vec3{Float64}(t))
            rb = StereoRig(BoundedCamera(cam1, Size2(640, 480)), cam2; ids = (:l, :r))
            rb2 = with_model(rb, 1 => m2)
            @test source(rb2) isa BoundedCamera && extent(source(rb2)) == Size2(640, 480)
            # a GENERIC model rides a rig (PosedCamera's model is any unposed AbstractCamera)
            gen = CentralGenericModel(m2, (640, 480); grid_size = (6, 5))
            rg = with_model(rb, 1 => gen)
            @test model(source(rg)) isa CentralGenericModel
            @test ids(rg) == (:l, :r)
            # single-sensor convenience + guards
            cammono = PosedCamera(model(cam1), relabel(WorldFrame, ViewFrame{:cam}, cam1.extrinsics))
            mono = Rig{WorldFrame}((cammono,))
            @test model(with_model(mono, m2)[1]) === m2
            @test_throws ArgumentError with_model(rb, m2)        # 2 sensors, no index
            @test_throws ArgumentError with_model(rb, 5 => m2)   # out of range
        end
    end

    # =============================================================================
    # Synthetic.imc_pose_error(F, rig): the rig overload reaches `source` and
    # `extrinsics` through the Synthetic module's explicit import list. It shipped
    # in 0.16 without those two imports (an UndefVarError on first call, reported
    # by JET, unreachable by every other test), so this pins the overload against
    # the (F, K, R, t) form on a rig whose F is exact.
    # =============================================================================

    @testset "Synthetic.imc_pose_error(F, rig)" begin
        K = SMatrix{3, 3, Float64}(800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 320.0, 240.0, 1.0)
        R = SMatrix{3, 3, Float64}(RotXYZ(0.05, -0.1, 0.02))
        t = SVector(0.3, -0.1, 0.05)
        rig = StereoRig(K, K, R, t; ids = (:left, :right))
        F = FundMat(EssentialMat(R, t), K)
        # Both are rounding (≈ 8e-15°) since `rotation_angle` stopped taking the acos of
        # the trace (2026-09-30), which quantized every near-identity angle to ~1e-6°;
        # the two routes round differently, so they agree absolutely, not relatively
        # (measured 8.118e-15 vs 8.189e-15).
        @test imc_pose_error(F, rig) ≈ imc_pose_error(F, K, K, R, t) atol = 1.0e-12
        @test imc_pose_error(F, rig) < 1.0e-4   # degrees; SVD-limited
    end
end
