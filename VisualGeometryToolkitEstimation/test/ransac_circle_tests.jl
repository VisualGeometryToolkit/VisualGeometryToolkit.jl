# CircleProblem / find_circle — the 2D circle as one more model of the RANSAC
# family: the minimal solver through the problem, the residual against the
# geometric distance, the chart rows against ForwardDiff, the chart-LM refit,
# the recovery under 40% outliers (a straight run and clutter), and the
# keyword set shared with the other `find_*` pipelines.

@testsetup module CircleScene
using Random: randn, rand, randperm
using StaticArrays: SVector
using VisualGeometryToolkitCore: Point2
export circle_scene, C₀, R₀

const C₀ = SVector(512.3, 401.7)
const R₀ = 300.0

# n points on the circle (C₀, R₀) with isotropic noise σ, then a fraction `outl`
# replaced: half by a straight run (a floor edge), half by uniform clutter.
function circle_scene(rng, n; σ = 0.5, outl = 0.4)
    pts = [Point2{Float64}(C₀ + R₀ * SVector(cos(t), sin(t)) + σ * randn(rng, SVector{2, Float64})) for t in 2π .* rand(rng, n)]
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for (k, i) in enumerate(bad)
        pts[i] = isodd(k) ? Point2{Float64}(250.0 + 500 * rand(rng), 760.0 + randn(rng)) :
            Point2{Float64}(1024 * rand(rng), 800 * rand(rng))
    end
    return pts, bad
end
end

@testitem "CircleProblem: minimal solver, residual, chart rows, refit" setup = [CircleScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import ForwardDiff
    using VisualGeometryToolkitCore: Point2, Circle, residual_rows!, residual_jacobian_rows!
    using VisualGeometryToolkitEstimation: CircleProblem
    import VisualGeometryToolkitEstimation as VgtkEstimation
    centre(c) = SVector(c.center[1], c.center[2])
    rng = Xoshiro(31)
    pts, _ = circle_scene(rng, 200; σ = 0.0, outl = 0.0)
    p = CircleProblem(pts)
    @test VgtkEstimation.codimension(p) == 1 && VgtkEstimation.minimal_sample_size(p) == 3 && VgtkEstimation.chart_dof(p) == 3
    sol = VgtkEstimation.solve(p, [1, 2, 3])
    @test sol.count == 1 && norm(centre(first(sol)) - C₀) < 1.0e-9 && abs(first(sol).r - R₀) < 1.0e-9
    r = zeros(200); VgtkEstimation.residuals!(r, p, first(sol))
    @test maximum(abs, r) < 1.0e-9                                       # pixels
    allocs(r, p, θ) = (VgtkEstimation.residuals!(r, p, θ); @allocated VgtkEstimation.residuals!(r, p, θ))
    @test allocs(r, p, first(sol)) == 0
    # collinear and repeated points: no circle
    q = CircleProblem([Point2(0.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 2.0), Point2(0.0, 0.0)])
    @test VgtkEstimation.solve(q, [1, 2, 3]).count == 0 && VgtkEstimation.solve(q, [1, 2, 4]).count == 0
    @test !VgtkEstimation.test_sample(q, [1, 2, 3]) && !VgtkEstimation.test_sample(q, [1, 2, 4])   # the gate refuses them first
    @test VgtkEstimation.test_sample(p, [1, 2, 3])
    # the residual is the Sampson distance (d² − r²)/(2d): the geometric distance d − r to first order,
    # short of it by exactly (d − r)²/(2d)
    c = Circle(Point2(C₀...), R₀)
    off = [Point2{Float64}(C₀ + (R₀ + δ) * SVector(cos(t), sin(t))) for (δ, t) in zip(range(-5, 5, length = 21), range(0, 6, length = 21))]
    po = CircleProblem(off)
    ro = zeros(21); VgtkEstimation.residuals!(ro, po, c)
    d = [norm(SVector(x...) - C₀) for x in off]
    @test ro ≈ d .- R₀ .- (d .- R₀) .^ 2 ./ (2 .* d) atol = 1.0e-10
    @test maximum(abs, ro .- (d .- R₀)) < 25 / (2 * (R₀ - 5)) + 1.0e-12   # 5 px off a 300 px circle: 0.04 px
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
    ws = VgtkEstimation.refine_workspace(typeof(pp), 100)
    c₁ = VgtkEstimation.refine!(ws, pp, c₀, ones(100))
    @test c₁ isa Circle{Float64}
    @test norm(centre(c₁) - C₀) < 0.3 && abs(c₁.r - R₀) < 0.3            # σ = 1 px on 100 points: σ/√50 ≈ 0.14 px
    refit_allocs(ws, p, θ, w) = (VgtkEstimation.refine!(ws, p, θ, w); @allocated VgtkEstimation.refine!(ws, p, θ, w))
    @test refit_allocs(ws, pp, c₀, ones(100)) == 0
end

@testitem "find_circle: 40% outliers (a straight run and clutter)" setup = [CircleScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: Point2, Circle, estimate, converged
    using VisualGeometryToolkitEstimation: find_circle
    centre(c) = SVector(c.center[1], c.center[2])
    rng = Xoshiro(32)
    for trial in 1:3
        pts, bad = circle_scene(rng, 1000; σ = 0.5, outl = 0.4)
        res = find_circle(pts; rng = Xoshiro(trial))
        @test converged(res)
        c = estimate(res)
        @test c isa Circle{Float64}
        @test norm(centre(c) - C₀) < 0.2 && abs(c.r - R₀) < 0.2          # σ = 0.5 px on 600 inliers: σ/√300 ≈ 0.03 px
        inl = res.inlier_mask
        good = setdiff(1:1000, bad)
        @test count(inl[good]) > 0.9 * length(good)                       # 1 px MSAC on a 0.5 px normal residual: P(|z| < 2) = 0.95
        @test count(inl[bad]) < 0.05 * length(bad)
    end
    @test find_circle([Point2(0.0, 0.0), Point2(1.0, 0.0)]) === nothing
end

@testitem "find_circle: the keywords of every find_* pipeline" setup = [CircleScene] begin
    using Test, Random
    using VisualGeometryToolkitCore: MsacScoring, estimate, converged
    using VisualGeometryToolkitEstimation: find_circle, find_fundmat, find_rotation, find_essential, CircleProblem,
        RansacConfig, RansacWorkspace, NoLocalOptimization
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # The shared keyword set is the one pipeline's (`_ransac_pipeline`); find_circle
    # declares no keyword of its own and forwards them all, as find_fundmat,
    # find_essential and find_rotation do (find_homography adds `extent`,
    # find_radial_camera `center`, find_hand_eye `motion_pairs`).
    shared = Set(Base.kwarg_decl(only(methods(VgtkEstimation._ransac_pipeline))))
    forward = [Symbol("kwargs...")]
    @test Base.kwarg_decl(only(methods(find_circle))) == forward
    @test Base.kwarg_decl(only(methods(find_fundmat))) == forward
    @test all(m -> Base.kwarg_decl(m) == forward, methods(find_rotation))
    # every shared keyword is accepted by find_circle, with the meaning it has elsewhere
    pts, _ = circle_scene(Xoshiro(33), 300; outl = 0.3)
    p = CircleProblem(pts)
    supplied = Dict(
        :threshold => 2.0,
        :local_optimization => NoLocalOptimization(),
        :max_trials => 500,
        :min_trials => 50,
        :scoring => MsacScoring(p; threshold = 1.0),
        :config => RansacConfig(max_trials = 400, min_trials = 20),
        :workspace => RansacWorkspace(CircleProblem, 300),
        :rng => Xoshiro(1),
    )
    @test Set(keys(supplied)) ∪ Set([:outlier_halfwidth]) == shared
    for (k, v) in supplied
        res = find_circle(pts; (; k => v)..., (k == :rng ? (;) : (; rng = Xoshiro(2)))...)
        @test converged(res)
    end
    # the retired half-width and a threshold beside a scoring are refused, as for every find_*
    @test_throws ArgumentError find_circle(pts; outlier_halfwidth = 5.0)
    @test_throws ArgumentError find_circle(pts; threshold = 2.0, scoring = MsacScoring(p; threshold = 1.0))
    @test_throws MethodError find_circle(pts; extent = nothing)            # no circle-specific keyword
end
