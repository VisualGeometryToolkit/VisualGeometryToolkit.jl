# R22: contract test — the circle problem's chart rows and refit step (`residual_rows!`, `residual_jacobian_rows!`, `refine!`)
# CircleProblem / Ransac on a CircleProblem — the 2D circle as one more model of the RANSAC
# family: the minimal solver through the problem, the residual against the
# geometric distance, the chart rows against ForwardDiff, the chart-LM refit,
# the recovery under 40% outliers (a straight run and clutter), and the
# keyword set shared with the other `find_*` pipelines.

@testitem "CircleProblem: minimal solver, residual, chart rows, refit" setup = [ConicScenes] begin
    using VisualGeometryToolkitCore: solve
    import VisualGeometryToolkitCore as VgtkCore; using VisualGeometryToolkitCore: warm_allocated
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random, LinearAlgebra, StaticArrays
    import ForwardDiff
    using VisualGeometryToolkitCore: Point2, Circle, residual_rows!, residual_jacobian_rows!, whitened_residuals!
    using VisualGeometryToolkitEstimation: CircleProblem
    import VisualGeometryToolkitEstimation as VgtkEstimation
    center_xy(c) = SVector(c.center[1], c.center[2])
    rng = Xoshiro(31)
    pts, _ = circle_scene(rng, 200; σ = 0.0, outl = 0.0)
    p = CircleProblem(pts)
    @test VgtkEstimation.codimension(p) == 1 && VgtkEstimation.minimal_sample_size(p) == 3 && VgtkEstimation.chart_dof(p) == 3
    sol = VgtkEstimation.sample_models(p, [1, 2, 3])
    @test sol.count == 1 && norm(center_xy(first(sol)) - C₀) < 1.0e-9 && abs(first(sol).r - R₀) < 1.0e-9
    r = zeros(200); VgtkEstimation.mahalanobis_distances!(r, p, first(sol))
    @test maximum(abs, r) < 1.0e-9                                       # pixels
    @test warm_allocated(VgtkEstimation.mahalanobis_distances!, r, p, first(sol)) == 0
    # collinear and repeated points: no circle
    q = CircleProblem([Point2(0.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 2.0), Point2(0.0, 0.0)])
    @test VgtkEstimation.sample_models(q, [1, 2, 3]).count == 0 && VgtkEstimation.sample_models(q, [1, 2, 4]).count == 0
    @test !VgtkEstimation.test_sample(q, [1, 2, 3]) && !VgtkEstimation.test_sample(q, [1, 2, 4])   # the gate refuses them first
    @test VgtkEstimation.test_sample(p, [1, 2, 3])
    # the signed residual is the Sampson residual (d² − r²)/(2d): the geometric signed distance d − r to
    # first order, short of it by exactly (d − r)²/(2d); the scoring distance is its magnitude
    c = Circle(Point2(C₀...), R₀)
    off = [Point2{Float64}(C₀ + (R₀ + δ) * SVector(cos(t), sin(t))) for (δ, t) in zip(range(-5, 5, length = 21), range(0, 6, length = 21))]
    po = CircleProblem(off)
    ro = zeros(21); whitened_residuals!(ro, po, c)
    d = [norm(SVector(x...) - C₀) for x in off]
    @test ro ≈ d .- R₀ .- (d .- R₀) .^ 2 ./ (2 .* d) atol = 1.0e-10
    @test maximum(abs, ro .- (d .- R₀)) < 25 / (2 * (R₀ - 5)) + 1.0e-12   # 5 px off a 300 px circle: 0.04 px
    @test VgtkEstimation.mahalanobis_distances!(zeros(21), po, c) == abs.(ro)
    # the chart rows against ForwardDiff, at random weights
    pn, _ = circle_scene(rng, 100; σ = 1.0, outl = 0.0)
    pp = CircleProblem(pn)
    c₀ = Circle(Point2(C₀ .+ SVector(2.0, -1.5)...), R₀ + 3)
    lm = VgtkEstimation._lm_state(pp, VgtkEstimation._chart(pp, c₀), rand(rng, 100))
    a0 = SVector(0.3, -0.2, 1.0e-3)
    J = zeros(100, 3); residual_jacobian_rows!(J, lm, a0)
    rows(a) = (out = zeros(eltype(a), 100); residual_rows!(out, lm, SVector{3}(a)); out)
    @test J ≈ ForwardDiff.jacobian(rows, Vector(a0)) rtol = 1.0e-8
    # the chart-LM refit (refine!) from a perturbed circle lands at the noise floor
    ws = VgtkEstimation.refit_workspace(pp)
    c₁ = VgtkCore.refine!(ws, pp, c₀, ones(100))
    @test c₁ isa Circle{Float64}
    @test norm(center_xy(c₁) - C₀) < 0.3 && abs(c₁.r - R₀) < 0.3            # σ = 1 px on 100 points: σ/√50 ≈ 0.14 px
    @test warm_allocated(VgtkCore.refine!, ws, pp, c₀, ones(100)) == 0
end

@testitem "CircleProblem: Ransac on 40% outliers (a straight run and clutter)" setup = [ConicScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac, CircleProblem
    
    using VisualGeometryToolkitCore: inlier_mask
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: Point2, Circle
    
    center_xy(c) = SVector(c.center[1], c.center[2])
    rng = Xoshiro(32)
    for trial in 1:3
        pts, bad = circle_scene(rng, 1000; σ = 0.5, outl = 0.4)
        res = solve(EstimationProblem(CircleProblem(pts)), Ransac(); rng = Xoshiro(trial))
        @test successful_retcode(res)
        c = res.u
        @test c isa Circle{Float64}
        @test norm(center_xy(c) - C₀) < 0.2 && abs(c.r - R₀) < 0.2          # σ = 0.5 px on 600 inliers: σ/√300 ≈ 0.03 px
        inl = inlier_mask(res)
        good = setdiff(1:1000, bad)
        @test count(inl[good]) > 0.9 * length(good)                       # 1 px MSAC on a 0.5 px normal residual: P(|z| < 2) = 0.95
        @test count(inl[bad]) < 0.05 * length(bad)
    end
    @test let r = solve(EstimationProblem(CircleProblem([Point2(0.0, 0.0), Point2(1.0, 0.0)])), Ransac())
        (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget) && r.stats.nsteps == 0 && r.u === nothing
    end
    # no sample gives a circle (collinear points): a :no_model result whose estimate is nothing,
    # not a MethodError (2026-11-03: `zero(::Type{Circle{Float64}})` did not exist)
    none = solve(EstimationProblem(CircleProblem([Point2(Float64(i), 2.0 * i + 1) for i in 1:10])), Ransac(); rng = Xoshiro(1))
    @test !successful_retcode(none) && (none.retcode, none.stop_rule) === (VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget) && none.u === nothing && !any(inlier_mask(none))
end

@testitem "Ransac: the configuration of every find, and its refusals" setup = [ConicScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac
    using Test, Random
    using VisualGeometryToolkitCore: MsacScoring
    using VisualGeometryToolkitEstimation: CircleProblem, EllipseProblem, RansacConfig
    # The find's configuration is `Ransac`'s fields, the same for every family problem; each is accepted with the meaning
    # it has elsewhere (the trial bounds in its `RansacConfig`, the storage precision its optional type argument).
    pts, _ = circle_scene(Xoshiro(33), 300; outl = 0.3)
    p = CircleProblem(pts)
    @test Set(fieldnames(Ransac)) == Set([:scoring, :local_optimization_strategy, :config, :score_guard, :on_improvement, :on_hypothesis])
    for alg in (Ransac(; local_optimization_strategy = nothing), Ransac(; config = RansacConfig(max_trials = 500)),
                Ransac(; config = RansacConfig(min_trials = 50)), Ransac(; scoring = MsacScoring(p; threshold = 1.0)),
                Ransac(; config = RansacConfig(max_trials = 400, min_trials = 20)), Ransac(Float64))
        @test successful_retcode(solve(EstimationProblem(p), alg; rng = Xoshiro(2)))
    end
    @test successful_retcode(solve(EstimationProblem(p), Ransac(); rng = Xoshiro(1)))
    # a bare width keyword is refused, alone or beside a scoring, as for every find_*
    @test_throws ArgumentError solve(EstimationProblem(CircleProblem(pts)), Ransac(; outlier_halfwidth = 5.0))
    @test_throws ArgumentError solve(EstimationProblem(CircleProblem(pts)), Ransac(; threshold = 2.0))
    @test_throws ArgumentError solve(EstimationProblem(CircleProblem(pts)), Ransac(; threshold = 2.0, scoring = MsacScoring(p; threshold = 1.0)))
    # an extent context needs a model class with an extent check; a circle and an ellipse have none
    @test_throws ArgumentError solve(EstimationProblem(CircleProblem(pts), (; extent = nothing)), Ransac())
    @test_throws ArgumentError solve(EstimationProblem(EllipseProblem(pts), (; extent = nothing)), Ransac())
end
