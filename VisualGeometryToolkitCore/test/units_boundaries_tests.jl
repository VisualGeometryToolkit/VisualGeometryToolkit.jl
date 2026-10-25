# =============================================================================
# Core's readers of physical poses: the inner loop takes plain maps only (a
# unitful map is a MethodError there), and every public function that takes a
# physical pose strips at entry in metres, computes the plain call and returns
# in the input's form — so the unitful call equals the plain call on
# `ustrip(m, ·)` bit for bit, and a plain input behaves exactly as before.
# =============================================================================

@testitem "units boundaries" begin
    using Test
    using StaticArrays, Rotations, LinearAlgebra
    using FlexUnits: Quantity
    using Statistics: cov
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: EuclideanMap, m, mm, ustrip, DimensionError, translation, rotation,
        relabel, ViewFrame, has_units, with_unit, PosedCamera, CameraModel, CameraCalibrationMat, StereoRig,
        extrinsics, pose, essential_form, EpipolarForm, EssentialMat, rectify, lookat, lookat_pose,
        transfer, Attributed, Point2, translation_angle, log_se3, minimal_manifold, RigidExpManifold6,
        LeftRigidManifold6, triangulate, is_in_front, unproject, Point3, BoundedCamera, Size2,
        transfer_ellipse, Circle, Ellipse, HomographyMat, Vec3, project
    using VisualGeometryToolkitCore.Synthetic: pose_error, imc_pose_error

    R1 = RotMatrix{3, Float64}(RotXYZ(0.01, -0.02, 0.005))
    R2 = RotMatrix{3, Float64}(RotXYZ(-0.015, 0.03, 0.01))
    t1 = SVector(0.012, -0.004, 0.0021)
    t2 = SVector(-0.118, 0.0031, 0.0094)
    E1u = EuclideanMap(R1, t1 .* m)
    E2u = EuclideanMap(R2, t2 .* m)
    model = CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(800, 0, 0, 0, 810, 0, 320, 240, 1)))
    C1u, C2u = PosedCamera(model, E1u), PosedCamera(model, E2u)
    C1, C2 = ustrip(m, C1u), ustrip(m, C2u)

    @testset "inner loop: plain maps only" begin
        @test_throws MethodError RigidExpManifold6(E1u)
        @test_throws MethodError LeftRigidManifold6(E1u)
        @test_throws MethodError minimal_manifold(E1u)
        @test_throws MethodError log_se3(E1u)
        f = normalize(SVector(0.1, 0.0, 1.0))
        @test_throws MethodError triangulate(E1u, f, f)
        @test_throws MethodError VgtkCore.constraint_with_jacobians(E1u, SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 1.0))
        # the plain forms are unchanged
        @test RigidExpManifold6(ustrip(m, E1u)) isa RigidExpManifold6
        @test log_se3(ustrip(m, E1u)) isa Tuple
    end

    @testset "stereo: StereoRig, its forms and transfer" begin
        rig_u = StereoRig(C1u, C2u; ids = (:l, :r))
        rig = StereoRig(C1, C2; ids = (:l, :r))
        @test has_units(rig_u) && !has_units(rig)
        @test translation(extrinsics(ustrip(m, rig_u))) === translation(extrinsics(rig))
        @test rotation(extrinsics(ustrip(m, rig_u))) === rotation(extrinsics(rig))
        # the 3-argument form, checked in metres
        X = extrinsics(C2u) ∘ pose(C1u)
        rig3_u = StereoRig(C1u, C2u, X; ids = (:l, :r))
        @test translation(extrinsics(ustrip(m, rig3_u))) === translation(extrinsics(StereoRig(C1, C2, ustrip(m, X); ids = (:l, :r))))
        Xbad = EuclideanMap(rotation(X), translation(X) .+ SVector(0.01, 0.0, 0.0) .* m)
        @test_throws ArgumentError StereoRig(C1u, C2u, Xbad)
        @test_throws ArgumentError StereoRig(C1u, C2)        # one physical camera, one plain
        @test_throws DimensionError StereoRig(C1u, C2u, ustrip(m, X))   # a plain relative pose for physical cameras
        # the epipolar forms are unit-free: equal to the plain rig's
        @test SMatrix{3, 3}(EssentialMat(X)) == SMatrix{3, 3}(EssentialMat(ustrip(m, X)))
        @test essential_form(rig_u) == essential_form(rig)
        @test EpipolarForm(rig_u) == EpipolarForm(rig)
        # depth-attributed transfer: depths in metres, as for the plain rig in metres
        p = Attributed(Point2(300.0, 250.0), 2.5)
        @test transfer(rig_u, p) == transfer(rig, p)
        F = SMatrix{3, 3, Float64}(1.0e-7, 3.0e-6, -2.0e-3, -2.5e-6, 4.0e-7, 1.0e-2, 1.5e-3, -9.0e-3, 0.4)
        @test imc_pose_error(F, rig_u) == imc_pose_error(F, rig)
    end

    @testset "rectify: the rectified rig in the input's form" begin
        rig_u = StereoRig(C1u, C2u; ids = (:l, :r))
        rig = StereoRig(C1, C2; ids = (:l, :r))
        r_u = rectify(rig_u)
        r = rectify(rig)
        @test has_units(r_u.rig) && !has_units(r.rig)
        @test translation(extrinsics(ustrip(m, r_u.rig))) === translation(extrinsics(r.rig))
        @test r_u.R == r.R && r_u.K == r.K
    end

    @testset "camera utilities: lookat, is_in_front, unproject" begin
        eye, tgt, up = SVector(0.1, -0.2, -1.5), SVector(0.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0)
        L_u = lookat(eye .* m, tgt .* m, up)
        @test has_units(L_u)
        @test L_u isa VgtkCore.CameraExtrinsics
        P = ustrip(m, L_u)
        L = lookat(eye, tgt, up)
        @test translation(P) === translation(L) && rotation(P) === rotation(L)
        @test ustrip(mm, L_u) isa VgtkCore.CameraExtrinsics
        Lp_u = lookat_pose(eye .* m, tgt .* m, up)
        @test translation(ustrip(m, Lp_u)) === translation(lookat_pose(eye, tgt, up))
        @test_throws DimensionError lookat(eye .* m, tgt, up)
        # is_in_front and unproject on a physical camera
        X = Point3(0.0, 0.0, 3.0)
        @test is_in_front(C1u, Point3(X .* m)) == is_in_front(C1, X)
        @test_throws DimensionError is_in_front(C1u, X)
        u = SVector(330.0, 250.0)
        Xu = unproject(C1u, u, 2.0m)
        @test has_units(Xu)
        @test ustrip.(m, Xu) == unproject(C1, u, 2.0)
        B = BoundedCamera(C1u, Size2(640, 480))
        @test is_in_front(B, Point3(X .* m)) == is_in_front(C1, X)
    end

    @testset "plane-coordinate verbs refuse a physical camera, naming the fix" begin
        circ = Circle(Point2(0.01, 0.02), 0.005)
        ell = Ellipse(Point2(0.01, 0.02), 0.006, 0.004, 0.3)
        for f in (() -> transfer_ellipse(C1u, circ), () -> transfer_ellipse(C1u, ell),
                  () -> project(C1u, circ), () -> project(C1u, ell), () -> HomographyMat(C1u),
                  () -> HomographyMat(C1u, Point3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)))
            err = try f(); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("ustrip(mm, camera)", sprint(showerror, err))
        end
        # a bounded physical camera forwards to the same refusal
        @test_throws ArgumentError project(BoundedCamera(C1u, Size2(640, 480)), circ)
        # the plain camera in the plane's unit is the fix
        @test HomographyMat(ustrip(mm, C1u)) isa HomographyMat
        @test transfer_ellipse(ustrip(m, C1u), circ) isa Union{Ellipse, Nothing}
    end

    @testset "a plain camera of a non-central grid model is refused, naming the fix" begin
        dirs = [Vec3{Float64}(0.01 * gx, 0.02 * gy, 1.0) for gy in 1:4, gx in 1:5]
        gn = VgtkCore.NoncentralGenericModel(64, 48, 0, 0, 63, 47, dirs, [Vec3{Float64}(1.0e-4gx, -1.0e-4gy, 5.0e-5) for gy in 1:4, gx in 1:5])
        msg(f) = (err = try f(); nothing catch e; e end; err isa ArgumentError ? sprint(showerror, err) : err)
        # the grid is metres; a plain map is "in the caller's unit": ambiguous, refused
        for f in (() -> PosedCamera(gn, EuclideanMap(R1, t1 .* 1000)), () -> PosedCamera(gn, pose(PosedCamera(gn, E1u)) |> X -> ustrip(m, X)))
            s = msg(f)
            @test s isa String && occursin("t .* m", s) && occursin("with_unit(extrinsics, mm)", s)
        end
        # the physical camera is the plain-metres camera's call, and stripping it in another unit is refused
        Cu = PosedCamera(gn, E1u)
        @test has_units(Cu)
        @test ustrip(m, Cu).extrinsics === ustrip(m, E1u)
        @test has_units(with_unit(ustrip(m, Cu), m))
        @test msg(() -> ustrip(mm, Cu)) isa String
        @test msg(() -> with_unit(ustrip(m, Cu), mm)) isa String
        @test with_unit(EuclideanMap(R1, t1), m) == E1u     # a plain map takes its unit
        @test with_unit(EuclideanMap(R1, t1 .* 1000), mm).R === R1
    end

    @testset "the stereo consistency check reads a plain rig in no fixed unit" begin
        # the same 0.1 m baseline, perturbed by 5 ppm of it, in metres and in mm:
        # one answer for both (a fixed 1-unit floor accepted the metres one)
        I3 = one(RotMatrix{3, Float64})
        for s in (1.0, 1000.0)
            tb = SVector(-0.1, 0.002, 0.001) .* s
            A, B = PosedCamera(model, EuclideanMap(I3, zero(SVector{3, Float64}))), PosedCamera(model, EuclideanMap(R2, tb))
            @test StereoRig(A, B, EuclideanMap(R2, tb)) isa StereoRig
            @test_throws ArgumentError StereoRig(A, B, EuclideanMap(R2, tb + SVector(5.0e-7 * s, 0.0, 0.0)))
        end
    end

    @testset "physical forms of the raw-matrix camera, the arrow projection and kabsch" begin
        K = SMatrix{3, 3, Float64}(800, 0, 0, 0, 810, 0, 320, 240, 1)
        Cr = PosedCamera(K, SMatrix{3, 3, Float64}(R1), t1 .* m)
        @test has_units(Cr) && ustrip(m, Cr).extrinsics == ustrip(m, E1u)
        o, d = SVector(0.01, -0.02, 2.0), SVector(0.0, 0.05, 0.0)
        @test project(C1u, Point3(o .* m), Vec3(d .* m)) == project(C1, Point3(o...), Vec3(d...))
        @test_throws DimensionError project(C1u, Point3(o...), Vec3(d...))
        Xs = [SVector(1.0, 2, 3), SVector(-1.0, 0.5, 2), SVector(0.3, -2, 1)]
        e = VgtkCore.exact(SVector(1.0, 2.0, 3.0) .* mm)       # a physical point known exactly
        @test VgtkCore.geom(e) == SVector(0.001, 0.002, 0.003) .* m && iszero(Matrix(cov(e)))
        @test_throws DimensionError VgtkCore.exact(SVector(1.0, 2.0, 3.0) .* VgtkCore.px)
        @test VgtkCore.kabsch([x .* mm => (R2 * x) .* mm for x in Xs]) == VgtkCore.kabsch([ustrip.(m, x .* mm) => ustrip.(m, (R2 * x) .* mm) for x in Xs])
    end

    @testset "pose error: translations in metres, angles out" begin
        @test translation_angle(t1 .* m, t2 .* m) == translation_angle(t1, t2)
        @test pose_error(R1, t1 .* m, R2, t2 .* m) == pose_error(R1, t1, R2, t2)
        @test_throws DimensionError translation_angle(t1 .* m, t2)
    end
end
