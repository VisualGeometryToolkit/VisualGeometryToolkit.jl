# A failed RANSAC has no estimate: when no hypothesis survives, the result is the
# `:no_model` RansacEstimate whose estimate is `nothing` — never a `zero(M)`
# placeholder a forgotten `converged` test would read as a model.

@testitem "ransac: no model" setup = [CircleScene] begin
    using Test, Random, StaticArrays
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, MsacScoring, csponds,
        estimate, converged, retcode, iterations
    using VisualGeometryToolkitEstimation: CircleProblem, LineFittingProblem, RansacEstimate,
        ransac, find_circle, find_ellipse, find_homography, find_fundmat

    function no_model(r)
        @test retcode(r) === :no_model
        @test estimate(r) === nothing
        @test !converged(r)
        @test !any(r.inlier_mask)
        @test iterations(r) > 0
        @test r.quality == -Inf
        @test isnan(r.sigma)
        @test r.dof == 0
        @test r.lo_retcode === :not_run
        return r
    end

    collinear = [Point2(Float64(i), 2.0 * i + 1) for i in 1:10]

    r = no_model(find_circle(collinear; rng = Xoshiro(1)))
    @test r isa RansacEstimate{Circle{Float64}, Float32}
    @test (@inferred Union{Circle{Float64}, Nothing} estimate(r)) === nothing
    @test Base.return_types(estimate, (typeof(r),)) == [Union{Circle{Float64}, Nothing}]

    # the benchmark path: `ransac` called directly spends its whole budget
    p = CircleProblem(collinear)
    r = no_model(ransac(p, MsacScoring(p; threshold = 1.0)))
    @test r isa RansacEstimate{Circle{Float64}, Float32}

    pts, _ = circle_scene(Xoshiro(3), 200; σ = 0.5, outl = 0.2)
    ok = find_circle(pts; rng = Xoshiro(4))
    @test converged(ok)
    @test ok isa RansacEstimate{Circle{Float64}, Float32}
    @test estimate(ok) isa Circle{Float64}

    no_model(find_ellipse(collinear; rng = Xoshiro(1)))

    # every sample of one repeated point fails the line's `test_sample`
    same = fill(Point2(1.0, 2.0), 10)
    q = LineFittingProblem(same)
    no_model(ransac(q, MsacScoring(q; threshold = 1.0)))

    # correspondences whose sources are all one point: no sample is a homography or an F
    rng = Xoshiro(5)
    src = fill(Point2(10.0, 20.0), 20)
    dst = [Point2(100 * rand(rng), 100 * rand(rng)) for _ in 1:20]
    no_model(find_homography(csponds(src, dst); rng = Xoshiro(6)))
    no_model(find_fundmat(csponds(src, dst); rng = Xoshiro(7)))
end

@testitem "ransac: no model through the metre and frame re-wrappers" begin
    using Test, Random, StaticArrays
    using Rotations: RotMatrix, RotX, RotZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, EuclideanMap, project, m, estimate, converged, retcode, iterations
    using VisualGeometryToolkitEstimation: find_absolute_pose, find_hand_eye

    function no_model(r)
        @test retcode(r) === :no_model
        @test estimate(r) === nothing
        @test !converged(r)
        @test !any(r.inlier_mask)
        return r
    end

    # hand-eye: every rig motion turns about z, so every sample's axes are parallel
    A = [EuclideanMap(RotMatrix(RotZ(0.35i)), SVector(0.1i, 0.05i^2, 0.2)) for i in 1:8]
    B = [EuclideanMap(RotMatrix(RotX(0.35i)), SVector(0.2, 0.1i, -0.03i)) for i in 1:8]
    plain = no_model(find_hand_eye(A .=> B; rng = Xoshiro(1)))
    phy = no_model(find_hand_eye(VgtkCore.with_unit.(A, m) .=> VgtkCore.with_unit.(B, m); rng = Xoshiro(1)))
    @test iterations(phy) == iterations(plain) > 0
    @test phy isa VgtkEstimation.RansacEstimate

    # absolute pose: three collinear world points hold no pose
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    X = [SVector(0.1i, 0.0, 2.0) for i in 1:3]
    u = [project(cam, x) for x in X]
    no_model(find_absolute_pose(cam, X .=> u; rng = Xoshiro(2)))
    no_model(find_absolute_pose(cam, [x .* m for x in X] .=> u; rng = Xoshiro(2)))

    # the frame re-wrapper of refine_absolute_pose keeps a failed result
    r = find_absolute_pose(cam, X .=> u; rng = Xoshiro(2))
    pose = VgtkCore.Extrinsics3(RotMatrix{3, Float64}(RotZ(0.1)), VgtkCore.Vec3{Float64}(0.0, 0.0, 1.0))
    framed = no_model(VgtkEstimation._with_frames(r, pose))
    @test iterations(framed) == iterations(r)
    @test VgtkEstimation._map_estimate(identity, r).metadata === r.metadata
end
