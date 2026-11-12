# A failed RANSAC has no estimate: when no hypothesis survives, the result is the
# `:no_model` RansacEstimate whose estimate is `nothing` — never a `zero(M)`
# placeholder a forgotten `converged` test would read as a model.

@testitem "ransac: no model" setup = [ConicScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitCore: inlier_mask, noise_scale, scale_source, ScaleNone, ReturnCode   # the robust contract's accessors
    using Test, Random, StaticArrays
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, MsacScoring, csponds,
        estimate, converged, retcode, iterations
    using VisualGeometryToolkitEstimation: CircleProblem, LineFittingProblem, RansacEstimate,
        ransac, find_circle, find_ellipse, find_homography, find_fundmat

    function no_model(r)
        @test retcode_symbol(r) === :no_model
        @test estimate(r) === nothing
        @test !converged(r)
        @test !any(inlier_mask(r))
        @test iterations(r) > 0
        @test r.quality == -Inf
        @test scale_source(r) == ScaleNone && isnan(r.scale)
        @test_throws ArgumentError noise_scale(r)
        @test r.dof == 0
        @test retcode_symbol(r.lo_retcode, r.lo_stop_rule) === :not_run && r.lo_retcode == ReturnCode.Default
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
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitCore: inlier_mask   # the robust contract's accessors
    using Test, Random, StaticArrays
    using Rotations: RotMatrix, RotX, RotZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, EuclideanMap, project, m, estimate, converged, retcode, iterations
    using VisualGeometryToolkitEstimation: find_absolute_pose, find_hand_eye

    function no_model(r)
        @test retcode_symbol(r) === :no_model
        @test estimate(r) === nothing
        @test !converged(r)
        @test !any(inlier_mask(r))
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

@testitem "ransac: too few points is the no-model result" setup = [TooFewData] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitCore: inlier_mask, scale_source, ScaleNone   # the robust contract's accessors
    using Test, Random, StaticArrays, LinearAlgebra
    using Rotations: RotMatrix
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, CameraModel, Csponds, csponds, HomographyMat, FundMat,
        EssentialMat, RadialHomography, RadialCamera, Extrinsics3, Vec3, m,
        estimate, converged, retcode, iterations
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacWorkspace, CircleProblem,
        find_homography, refine_homography, find_fundmat, refine_fundmat, find_essential, refine_essential,
        find_rotation, refine_rotation, find_circle, refine_circle, find_ellipse, refine_ellipse,
        find_radial_homography, refine_radial_homography, find_radial_camera, refine_radial_camera,
        find_absolute_pose, refine_absolute_pose, RefineMetadata

    # a refine_* on too few data: the guess back unrefined, its result type a refine's (the refine contract)
    function too_few_refine(r, guess, n)
        @test r isa VgtkCore.FitResult{<:Any, <:RefineMetadata}
        @test retcode_symbol(r) === :degenerate && iterations(r) == 0 && r.n == 0 && isnan(r.rms)
        θ = estimate(r)
        @test guess isa VgtkCore.EuclideanMap ?
            (VgtkCore.rotation(θ) == VgtkCore.rotation(guess) && VgtkCore.translation(θ) == VgtkCore.translation(guess)) : θ == guess
        @test length(r.residuals) == n && all(isnan, r.residuals)
        return r
    end
    function too_few(r, n)
        @test r isa RansacEstimate
        @test retcode_symbol(r) === :no_model
        @test iterations(r) == 0
        @test estimate(r) === nothing
        @test !converged(r)
        @test length(inlier_mask(r)) == n && !any(inlier_mask(r))
        @test isnan(r.sample_acceptance_rate)
        @test r.quality == -Inf && scale_source(r) == ScaleNone && isnan(r.scale) && isnan(r.rms)
        return r
    end

    pts, cs = few_points, few_csponds
    I3 = SMatrix{3, 3, Float64}(I)
    H = HomographyMat{Float64}(Tuple(I3))
    X3 = SMatrix{3, 3, Float64}(0, 0, 0, 0, 0, -1, 0, 1, 0)
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))

    too_few(find_homography(cs(3)), 3)
    too_few_refine(refine_homography(H, cs(3); scale = 1.0), H, 3)
    too_few(find_fundmat(cs(6)), 6)
    too_few_refine(refine_fundmat(FundMat{Float64}(Tuple(X3)), cs(6); scale = 1.0), FundMat{Float64}(Tuple(X3)), 6)
    too_few(find_essential(cam, cam, cs(4)), 4)
    too_few_refine(refine_essential(EssentialMat{Float64}(Tuple(X3)), cam, cam, cs(4); scale = 1.0), EssentialMat{Float64}(Tuple(X3)), 4)
    too_few(find_rotation(cam, cam, cs(1)), 1)
    too_few_refine(refine_rotation(one(RotMatrix{3, Float64}), cam, cam, cs(1); scale = 1.0), one(RotMatrix{3, Float64}), 1)
    b = Csponds([SA[0.0, 0.0, 1.0]], [SA[0.0, 0.0, 1.0]])
    too_few(find_rotation(b), 1)
    too_few_refine(refine_rotation(one(RotMatrix{3, Float64}), b; scale = 1.0), one(RotMatrix{3, Float64}), 1)
    too_few(find_circle(pts(2)), 2)
    too_few_refine(refine_circle(Circle(Point2(0.0, 0.0), 1.0), pts(2); scale = 1.0), Circle(Point2(0.0, 0.0), 1.0), 2)
    too_few(find_ellipse(pts(4)), 4)
    too_few_refine(refine_ellipse(Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0), pts(4); scale = 1.0), Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0), 4)
    c = Point2(500.0, 400.0)
    too_few(find_radial_homography(cs(4); center = c), 4)
    too_few(find_radial_homography(cs(3); center = c, minimal = :homography), 3)
    too_few_refine(refine_radial_homography(RadialHomography{Float64}(H, 0.0), cs(4); scale = 1.0), RadialHomography{Float64}(H, 0.0), 4)
    too_few(find_radial_camera(cs(4); center = c), 4)
    too_few_refine(refine_radial_camera(RadialCamera(H, c), cs(4); scale = 1.0), RadialCamera(H, c), 4)

    X = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    u = [SA[400.0, 300.0], SA[500.0, 350.0]]
    pose = Extrinsics3(RotMatrix{3, Float64}(I3), Vec3{Float64}(0.0, 0.0, 0.0))
    too_few(find_absolute_pose(cam, X .=> u), 2)
    too_few(find_absolute_pose(cam, [x .* m for x in X] .=> u), 2)
    too_few_refine(refine_absolute_pose(pose, cam, X .=> u; scale = 1.0), pose, 2)
    too_few_refine(refine_absolute_pose(VgtkCore.with_unit(pose, m), cam, [x .* m for x in X] .=> u; scale = 1.0), VgtkCore.with_unit(pose, m), 2)
    f = [normalize(x) for x in X]
    too_few_refine(refine_absolute_pose(pose, Csponds(X, f); scale = 1.0), pose, 2)
    too_few_refine(refine_absolute_pose(VgtkCore.with_unit(pose, m), Csponds([x .* m for x in X], f); scale = 1.0), VgtkCore.with_unit(pose, m), 2)

    # the supplied workspace's storage eltype
    r = too_few(find_circle(pts(2); workspace = RansacWorkspace(Float64, CircleProblem, 2)), 2)
    @test r.residuals isa Vector{Float64} && r isa RansacEstimate{Circle{Float64}, Float64}
    @test too_few(find_circle(pts(2)), 2) isa RansacEstimate{Circle{Float64}, Float32}
    # the too-few result is the same type as a search's
    @test typeof(find_homography(cs(3))) === typeof(find_homography(cs(8)))
end

@testitem "ransac: too few points refuses the keywords enough points refuse" setup = [TooFewData] begin
    # The too-few path is the pipeline's: a retired, a width or an unknown keyword
    # raises the same error below the minimal sample as above it, never a :no_model.
    using Test, StaticArrays, LinearAlgebra
    using Rotations: RotMatrix
    using VisualGeometryToolkitCore: Point2, Circle, CameraModel, Csponds, csponds, HomographyMat, Extrinsics3, Vec3,
        TLSShape
    using VisualGeometryToolkitEstimation: find_homography, refine_homography, find_circle, refine_circle, find_absolute_pose, refine_absolute_pose
    using VisualGeometryToolkitCore: Irls

    caught(f) = try f(); nothing catch e; e end
    # the same refusal: the same exception type and, for an ArgumentError, the same message
    same(a, b) = a !== nothing && typeof(a) === typeof(b) && (!(a isa ArgumentError) || a.msg == b.msg)

    pts, cs = few_points, few_csponds
    H = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    Xs(n) = [SA[0.1i, 0.2 - 0.05i^2, 2.0 + 0.1i] for i in 1:n]
    us(n) = [SA[400.0 + 20i, 300.0 + 3i^2] for i in 1:n]
    pose = Extrinsics3(RotMatrix{3, Float64}(I), Vec3{Float64}(0.0, 0.0, 0.0))
    lo = Irls{TLSShape}()

    find_kw = ((; local_optimization = lo), (; threshold = 2.0), (; outlier_halfwidth = 20.0), (; no_such_keyword = 1))
    refine_kw = ((; local_optimization = lo), (; local_optimization_strategy = lo), (; threshold = 2.0),
                 (; no_such_keyword = 1))
    calls = (
        (find_kw, n -> kw -> find_homography(cs(n); kw...), 3, 8),
        (refine_kw, n -> kw -> refine_homography(H, cs(n); kw...), 3, 8),
        (find_kw, n -> kw -> find_circle(pts(n); kw...), 2, 8),
        (refine_kw, n -> kw -> refine_circle(Circle(Point2(0.0, 0.0), 1.0), pts(n); kw...), 2, 8),
        (find_kw, n -> kw -> find_absolute_pose(cam, Xs(n) .=> us(n); kw...), 2, 8),
        (refine_kw, n -> kw -> refine_absolute_pose(pose, cam, Xs(n) .=> us(n); kw...), 2, 8),
        (refine_kw, n -> kw -> refine_absolute_pose(pose, Csponds(Xs(n), normalize.(Xs(n))); kw...), 2, 8),
    )
    for (kws, call, few, enough) in calls, kw in kws
        enough_err = caught(() -> call(enough)(kw))
        @test enough_err isa Union{ArgumentError, MethodError}
        @test same(caught(() -> call(few)(kw)), enough_err)
    end
end
