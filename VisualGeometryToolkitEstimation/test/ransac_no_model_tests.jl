# A failed RANSAC has no estimate: when no hypothesis survives, the result is the
# `:no_model` RansacEstimate whose estimate is `nothing` — never a `zero(M)`
# placeholder a forgotten `converged` test would read as a model.

@testitem "ransac: no model" setup = [ConicScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: dof
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem, FundMatProblem, EllipseProblem
    
    using VisualGeometryToolkitCore: inlier_mask, noise_scale, scale_source, ScaleNone, ReturnCode
    using Test, Random, StaticArrays
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, MsacScoring, correspondences
    using VisualGeometryToolkitEstimation: CircleProblem, LineFittingProblem, RansacEstimate

    function no_model(r)
        @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget)
        @test r.u === nothing
        @test !successful_retcode(r)
        @test !any(inlier_mask(r))
        @test r.stats.nsteps > 0
        @test r.metadata.quality == -Inf
        @test scale_source(r) == ScaleNone
        @test_throws ArgumentError noise_scale(r)
        @test dof(r) == 0
        @test (r.metadata.lo_retcode, r.metadata.lo_stop_rule) === (VgtkCore.ReturnCode.Default, VgtkCore.NoRule) && r.metadata.lo_retcode == ReturnCode.Default
        return r
    end

    collinear = [Point2(Float64(i), 2.0 * i + 1) for i in 1:10]

    r = no_model(solve(EstimationProblem(CircleProblem(collinear)), Ransac(); rng = Xoshiro(1)))
    @test r isa RansacEstimate{Circle{Float64}, Float32}
    @test (@inferred Union{Circle{Float64}, Nothing} getproperty(r, :u)) === nothing
    @test fieldtype(typeof(r), :u) == Union{Circle{Float64}, Nothing}

    # the benchmark path: `ransac` called directly spends its whole budget
    p = CircleProblem(collinear)
    r = no_model(solve(EstimationProblem(p), Ransac(; scoring = MsacScoring(p; threshold = 1.0), local_optimization_strategy = nothing)))
    @test r isa RansacEstimate{Circle{Float64}, Float32}

    pts, _ = circle_scene(Xoshiro(3), 200; σ = 0.5, outl = 0.2)
    ok = solve(EstimationProblem(CircleProblem(pts)), Ransac(); rng = Xoshiro(4))
    @test successful_retcode(ok)
    @test ok isa RansacEstimate{Circle{Float64}, Float32}
    @test ok.u isa Circle{Float64}

    no_model(solve(EstimationProblem(EllipseProblem(collinear)), Ransac(); rng = Xoshiro(1)))

    # every sample of one repeated point fails the line's `test_sample`
    same = fill(Point2(1.0, 2.0), 10)
    q = LineFittingProblem(same)
    no_model(solve(EstimationProblem(q), Ransac(; scoring = MsacScoring(q; threshold = 1.0), local_optimization_strategy = nothing)))

    # correspondences whose sources are all one point: no sample is a homography or an F
    rng = Xoshiro(5)
    src = fill(Point2(10.0, 20.0), 20)
    dst = [Point2(100 * rand(rng), 100 * rand(rng)) for _ in 1:20]
    no_model(solve(EstimationProblem(HomographyProblem(correspondences(src, dst))), Ransac(); rng = Xoshiro(6)))
    no_model(solve(EstimationProblem(FundMatProblem(correspondences(src, dst))), Ransac(); rng = Xoshiro(7)))
end

@testitem "ransac: no model through the metre and frame re-wrappers" begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac, AbsolutePoseProblem, HandEyeProblem
    
    using VisualGeometryToolkitCore: inlier_mask
    using Test, Random, StaticArrays
    using Rotations: RotMatrix, RotX, RotZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, EuclideanMap, project, m
    

    function no_model(r)
        @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget)
        @test r.u === nothing
        @test !successful_retcode(r)
        @test !any(inlier_mask(r))
        return r
    end

    # hand-eye: every rig motion turns about z, so every sample's axes are parallel
    A = [EuclideanMap(RotMatrix(RotZ(0.35i)), SVector(0.1i, 0.05i^2, 0.2)) for i in 1:8]
    B = [EuclideanMap(RotMatrix(RotX(0.35i)), SVector(0.2, 0.1i, -0.03i)) for i in 1:8]
    plain = no_model(solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = Xoshiro(1)))
    phy = no_model(solve(EstimationProblem(HandEyeProblem(VgtkCore.with_unit.(A, m) .=> VgtkCore.with_unit.(B, m))), Ransac(); rng = Xoshiro(1)))
    @test phy.stats.nsteps == plain.stats.nsteps > 0
    @test phy isa VgtkEstimation.RansacEstimate

    # absolute pose: three collinear world points hold no pose
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    X = [SVector(0.1i, 0.0, 2.0) for i in 1:3]
    u = [project(cam, x) for x in X]
    no_model(solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u)), Ransac(); rng = Xoshiro(2)))
    no_model(solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u)), Ransac(); rng = Xoshiro(2)))

    # a re-expressed failed result keeps its record and metadata
    r = solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u)), Ransac(); rng = Xoshiro(2))
    mapped = no_model(VgtkCore.map_estimate(identity, r))
    @test mapped.stats.nsteps == r.stats.nsteps
    @test mapped.metadata === r.metadata
end

@testitem "ransac: too few points is the no-model result" setup = [TooFewData] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, successful_retcode, RefineMetadata
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem, BearingPoseProblem, RadialHomographyProblem, RadialCameraProblem, RotationProblem, EllipseProblem
    
    using VisualGeometryToolkitCore: inlier_mask, scale_source, ScaleNone
    using Test, Random, StaticArrays, LinearAlgebra
    using Rotations: RotMatrix
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, CameraModel, Correspondences, correspondences, HomographyMat, FundMat, EssentialMat, RadialHomography, RadialCamera, CameraExtrinsics, Vec3, m, EuclideanMap
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacWorkspace, CircleProblem

    # a refine_* on too few data: the guess back unrefined, its result type a refine's (the refine contract)
    function too_few_refine(r, guess, n)
        @test r isa VgtkCore.FitResult{<:Any, <:RefineMetadata}
        @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule) && r.stats.nsteps == 0 && r.n == 0 && isnan(r.rms)
        θ = r.u
        @test guess isa VgtkCore.EuclideanMap ?
            (VgtkCore.rotation(θ) == VgtkCore.rotation(guess) && VgtkCore.translation(θ) == VgtkCore.translation(guess)) : θ == guess
        @test length(r.resid) == n && all(isnan, r.resid)
        return r
    end
    function too_few(r, n)
        @test r isa RansacEstimate
        @test (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget)
        @test r.stats.nsteps == 0
        @test r.u === nothing
        @test !successful_retcode(r)
        @test length(inlier_mask(r)) == n && !any(inlier_mask(r))
        @test isnan(r.metadata.sample_acceptance_rate)
        @test r.metadata.quality == -Inf && scale_source(r) == ScaleNone && isnan(r.rms)
        return r
    end

    pts, cs = few_points, few_correspondences
    I3 = SMatrix{3, 3, Float64}(I)
    H = HomographyMat{Float64}(Tuple(I3))
    X3 = SMatrix{3, 3, Float64}(0, 0, 0, 0, 0, -1, 0, 1, 0)
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))

    too_few(solve(EstimationProblem(HomographyProblem(cs(3))), Ransac()), 3)
    too_few_refine(solve(RefineProblem(HomographyProblem(cs(3)), H), LevenbergMarquardt(); scale = 1.0), H, 3)
    too_few(solve(EstimationProblem(FundMatProblem(cs(6))), Ransac()), 6)
    too_few_refine(solve(RefineProblem(FundMatProblem(cs(6)), FundMat{Float64}(Tuple(X3))), LevenbergMarquardt(); scale = 1.0), FundMat{Float64}(Tuple(X3)), 6)
    too_few(solve(EstimationProblem(EssentialProblem(cam, cam, cs(4))), Ransac()), 4)
    too_few_refine(solve(RefineProblem(EssentialProblem(cam, cam, cs(4)), EssentialMat{Float64}(Tuple(X3))), LevenbergMarquardt(); scale = 1.0), EssentialMat{Float64}(Tuple(X3)), 4)
    too_few(solve(EstimationProblem(RotationProblem(cam, cam, cs(1))), Ransac()), 1)
    too_few_refine(solve(RefineProblem(RotationProblem(cam, cam, cs(1)), one(RotMatrix{3, Float64})), LevenbergMarquardt(); scale = 1.0), one(RotMatrix{3, Float64}), 1)
    too_few(solve(EstimationProblem(CircleProblem(pts(2))), Ransac()), 2)
    too_few_refine(solve(RefineProblem(CircleProblem(pts(2)), Circle(Point2(0.0, 0.0), 1.0)), LevenbergMarquardt(); scale = 1.0), Circle(Point2(0.0, 0.0), 1.0), 2)
    too_few(solve(EstimationProblem(EllipseProblem(pts(4))), Ransac()), 4)
    too_few_refine(solve(RefineProblem(EllipseProblem(pts(4)), Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)), LevenbergMarquardt(); scale = 1.0), Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0), 4)
    c = Point2(500.0, 400.0)
    too_few(solve(EstimationProblem(RadialHomographyProblem(cs(4); center = c)), Ransac()), 4)
    too_few(solve(EstimationProblem(RadialHomographyProblem(cs(3); center = c, minimal = :homography)), Ransac()), 3)
    too_few_refine(solve(RefineProblem(RadialHomographyProblem(cs(4); center = VgtkCore.center(RadialHomography{Float64}(H, 0.0))), RadialHomography{Float64}(H, 0.0)), LevenbergMarquardt(); scale = 1.0), RadialHomography{Float64}(H, 0.0), 4)
    too_few(solve(EstimationProblem(RadialCameraProblem(cs(4); center = c)), Ransac()), 4)
    too_few_refine(solve(RefineProblem(RadialCameraProblem(cs(4); center = VgtkCore.center(RadialCamera(H, c))), RadialCamera(H, c)), LevenbergMarquardt(); scale = 1.0), RadialCamera(H, c), 4)

    X = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    u = [SA[400.0, 300.0], SA[500.0, 350.0]]
    pose = CameraExtrinsics(EuclideanMap(RotMatrix{3, Float64}(I3), Vec3{Float64}(0.0, 0.0, 0.0)))
    too_few(solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u)), Ransac()), 2)
    too_few(solve(EstimationProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u)), Ransac()), 2)
    too_few_refine(solve(RefineProblem(AbsolutePoseProblem(cam, X .=> u), pose), LevenbergMarquardt(); scale = 1.0), pose, 2)
    too_few_refine(solve(RefineProblem(AbsolutePoseProblem(cam, [x .* m for x in X] .=> u), VgtkCore.with_unit(pose, m)), LevenbergMarquardt(); scale = 1.0), VgtkCore.with_unit(pose, m), 2)
    f = [normalize(x) for x in X]
    too_few_refine(solve(RefineProblem(BearingPoseProblem(Correspondences(X, f)), pose), LevenbergMarquardt(); scale = 1.0), pose, 2)
    too_few_refine(solve(RefineProblem(BearingPoseProblem(Correspondences([x .* m for x in X], f)), VgtkCore.with_unit(pose, m)), LevenbergMarquardt(); scale = 1.0), VgtkCore.with_unit(pose, m), 2)

    # the supplied workspace's storage eltype
    r = too_few(solve(EstimationProblem(CircleProblem(pts(2))), Ransac(Float64)), 2)
    @test r.resid isa Vector{Float64} && r isa RansacEstimate{Circle{Float64}, Float64}
    @test too_few(solve(EstimationProblem(CircleProblem(pts(2))), Ransac()), 2) isa RansacEstimate{Circle{Float64}, Float32}
    # the too-few result is the same type as a search's
    @test typeof(solve(EstimationProblem(HomographyProblem(cs(3))), Ransac())) === typeof(solve(EstimationProblem(HomographyProblem(cs(8))), Ransac()))
end

@testitem "ransac: too few points refuses the keywords enough points refuse" setup = [TooFewData] begin
    # The too-few path is the pipeline's: a retired, a width or an unknown keyword
    # raises the same error below the minimal sample as above it, never a :no_model.
    using Test, StaticArrays, LinearAlgebra
    using Rotations: RotMatrix
    using VisualGeometryToolkitCore: Point2, Circle, CameraModel, Correspondences, correspondences, HomographyMat, CameraExtrinsics, Vec3, TLSShape, EuclideanMap
    
    using VisualGeometryToolkitCore: Irls, solve, RefineProblem, LevenbergMarquardt
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem, CircleProblem, AbsolutePoseProblem, BearingPoseProblem

    caught(f) = try f(); nothing catch e; e end
    # the same refusal: the same exception type and, for an ArgumentError, the same message
    same(a, b) = a !== nothing && typeof(a) === typeof(b) && (!(a isa ArgumentError) || a.msg == b.msg)

    pts, cs = few_points, few_correspondences
    H = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    Xs(n) = [SA[0.1i, 0.2 - 0.05i^2, 2.0 + 0.1i] for i in 1:n]
    us(n) = [SA[400.0 + 20i, 300.0 + 3i^2] for i in 1:n]
    pose = CameraExtrinsics(EuclideanMap(RotMatrix{3, Float64}(I), Vec3{Float64}(0.0, 0.0, 0.0)))
    lo = Irls{TLSShape}()

    # a find's configuration is its algorithm's, refused at construction whatever the data
    for kw in ((; local_optimization = lo), (; threshold = 2.0), (; outlier_halfwidth = 20.0), (; no_such_keyword = 1))
        @test caught(() -> Ransac(; kw...)) isa Union{ArgumentError, MethodError}
    end
    # a refine's run settings are its solve keywords: one it does not take is refused alike on too few and enough data
    refine_kw = ((; local_optimization = lo), (; local_optimization_strategy = lo), (; threshold = 2.0), (; no_such_keyword = 1))
    lm = LevenbergMarquardt()
    calls = (
        (n -> RefineProblem(HomographyProblem(cs(n)), H), 3, 8),
        (n -> RefineProblem(CircleProblem(pts(n)), Circle(Point2(0.0, 0.0), 1.0)), 2, 8),
        (n -> RefineProblem(AbsolutePoseProblem(cam, Xs(n) .=> us(n)), pose), 2, 8),
        (n -> RefineProblem(BearingPoseProblem(Correspondences(Xs(n), normalize.(Xs(n)))), pose), 2, 8),
    )
    for (prob, few, enough) in calls, kw in refine_kw
        enough_err = caught(() -> solve(prob(enough), lm; scale = 1.0, kw...))
        @test enough_err isa MethodError
        @test same(caught(() -> solve(prob(few), lm; scale = 1.0, kw...)), enough_err)
    end
end

@testitem "with_extent: the problem itself without an extent, a BoundedProblem with one" begin
    using VisualGeometryToolkitEstimation, StaticArrays
    using VisualGeometryToolkitEstimation: BoundedProblem, HomographyProblem, with_extent
    import VisualGeometryToolkitCore as VgtkCore
    pairs = [SVector(Float64(i), Float64(i^2 % 7)) => SVector(2.0 * i, Float64(i % 5)) for i in 1:8]
    p = HomographyProblem(pairs)
    @test with_extent(p, nothing) === p
    rect = VgtkCore.Rect(0.0, 0.0, 200.0, 150.0)
    b = with_extent(p, rect)
    @test b isa BoundedProblem && b.inner === p && b.extent == rect
end
