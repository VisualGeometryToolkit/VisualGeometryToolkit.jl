# EllipseProblem / find_ellipse — the 2D ellipse as one more model of the RANSAC
# family, as the circle (ransac_circle_tests.jl): the minimal solver through the
# problem, the residual against the geometric distance, the chart rows against
# ForwardDiff, the chart-LM refit, the recovery under 40% outliers (a straight
# run and clutter), and the keyword set shared with the other `find_*` pipelines.

@testsetup module EllipseScene
using Random: randn, rand, randperm
using StaticArrays: SVector
using VisualGeometryToolkitCore: Point2, Ellipse, boundary_point
export ellipse_scene, E₀

# axes ratio 0.6, rotated
const E₀ = Ellipse(Point2(512.3, 401.7), 300.0, 180.0, 0.6)

# n points on E₀ with isotropic noise σ, then a fraction `outl` replaced: half by
# a straight run (a floor edge), half by uniform clutter.
function ellipse_scene(rng, n; σ = 0.5, outl = 0.4)
    pts = [Point2{Float64}(SVector(boundary_point(E₀, t)...) + σ * randn(rng, SVector{2, Float64})) for t in 2π .* rand(rng, n)]
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for (k, i) in enumerate(bad)
        pts[i] = isodd(k) ? Point2{Float64}(250.0 + 500 * rand(rng), 760.0 + randn(rng)) :
            Point2{Float64}(1024 * rand(rng), 800 * rand(rng))
    end
    return pts, bad
end
end

@testitem "EllipseProblem: minimal solver, residual, chart rows, refit" setup = [EllipseScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    import ForwardDiff
    using VisualGeometryToolkitCore: Point2, Ellipse, residual_rows!, residual_jacobian_rows!, ambient_coordinates, boundary_point
    using VisualGeometryToolkitEstimation: EllipseProblem
    import VisualGeometryToolkitEstimation as VgtkEstimation
    centre(e) = SVector(e.center[1], e.center[2])
    rng = Xoshiro(31)
    pts, _ = ellipse_scene(rng, 200; σ = 0.0, outl = 0.0)
    p = EllipseProblem(pts)
    @test VgtkEstimation.codimension(p) == 1 && VgtkEstimation.minimal_sample_size(p) == 5 && VgtkEstimation.chart_dof(p) == 5
    sol = VgtkEstimation.solve(p, [1, 2, 3, 4, 5])
    e = first(sol)
    @test sol.count == 1 && norm(centre(e) - centre(E₀)) < 1.0e-8 && abs(e.a - E₀.a) < 1.0e-8 && abs(e.b - E₀.b) < 1.0e-8 &&
        abs(sin(e.θ - E₀.θ)) < 1.0e-10
    r = zeros(200); VgtkEstimation.residuals!(r, p, e)
    @test maximum(abs, r) < 1.0e-8                                       # pixels
    allocs(r, p, θ) = (VgtkEstimation.residuals!(r, p, θ); @allocated VgtkEstimation.residuals!(r, p, θ))
    @test allocs(r, p, e) == 0
    # a collinear triple and a repeated point: no ellipse
    q = EllipseProblem([Point2(0.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 2.0), Point2(0.0, 3.0), Point2(3.0, 0.5), Point2(0.0, 0.0)])
    @test VgtkEstimation.solve(q, [1, 2, 3, 4, 5]).count == 0
    @test !VgtkEstimation.test_sample(q, [1, 2, 3, 4, 5]) && !VgtkEstimation.test_sample(q, [1, 4, 5, 2, 6])   # the gate refuses them first
    @test VgtkEstimation.test_sample(p, [1, 2, 3, 4, 5])
    # the residual is the Sampson distance: the geometric distance to first order.
    # Along the major axis (normal there) a point δ off is (δ − δ²/(2(a + δ)))…
    off = [Point2{Float64}(centre(E₀) + (E₀.a + δ) * SVector(cos(E₀.θ), sin(E₀.θ))) for δ in range(-5, 5, length = 21)]
    po = EllipseProblem(off)
    ro = zeros(21); VgtkEstimation.residuals!(ro, po, E₀)
    δs = collect(range(-5, 5, length = 21))
    @test ro ≈ δs .- δs .^ 2 ./ (2 .* (E₀.a .+ δs)) atol = 1.0e-10
    # …and anywhere on the curve, a point δ along the normal is δ to first order
    nrm(t) = normalize(SMatrix{2, 2}(cos(E₀.θ), sin(E₀.θ), -sin(E₀.θ), cos(E₀.θ)) * SVector(E₀.b * cos(t), E₀.a * sin(t)))   # outward
    ts = range(0, 2π, length = 13)
    near = [Point2{Float64}(SVector(boundary_point(E₀, t)...) + s * 0.5 * nrm(t)) for t in ts for s in (1, -1)]
    rn = zeros(26); VgtkEstimation.residuals!(rn, EllipseProblem(near), E₀)
    @test maximum(abs, rn .- repeat([0.5, -0.5], 13)) < 0.5^2 * E₀.a / E₀.b^2   # second order in δ, the curvature at most a/b²
    # the chart rows against ForwardDiff, at random weights
    pn, _ = ellipse_scene(rng, 100; σ = 1.0, outl = 0.0)
    pp = EllipseProblem(pn)
    e₀ = Ellipse(Point2((centre(E₀) .+ SVector(2.0, -1.5))...), E₀.a + 3, E₀.b - 2, E₀.θ + 0.01)
    lm = VgtkEstimation._lm_state(pp, VgtkEstimation._chart(pp, e₀), rand(rng, 100))
    a0 = SVector(0.3, -0.2, 1.0e-3, -2.0e-3, 5.0e-4)
    J = zeros(100, 5); residual_jacobian_rows!(J, lm, a0)
    rows(a) = (out = zeros(eltype(a), 100); residual_rows!(out, lm, SVector{5}(a)); out)
    @test J ≈ ForwardDiff.jacobian(rows, Vector(a0)) rtol = 1.0e-8
    # the chart-LM refit (refine!) from a perturbed ellipse lands at the noise floor
    ws = VgtkEstimation.refine_workspace(typeof(pp), 100)
    e₁ = VgtkEstimation.refine!(ws, pp, e₀, ones(100))
    @test e₁ isa Ellipse{Float64}
    @test norm(centre(e₁) - centre(E₀)) < 0.5 && abs(e₁.a - E₀.a) < 0.5 && abs(e₁.b - E₀.b) < 0.5 && abs(e₁.θ - E₀.θ) < 0.01   # σ = 1 px on 100 points
    refit_allocs(ws, p, θ, w) = (VgtkEstimation.refine!(ws, p, θ, w); @allocated VgtkEstimation.refine!(ws, p, θ, w))
    @test refit_allocs(ws, pp, e₀, ones(100)) == 0
end

@testitem "find_ellipse: 40% outliers (a straight run and clutter)" setup = [EllipseScene] begin
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: Point2, Ellipse, estimate, converged
    using VisualGeometryToolkitEstimation: find_ellipse
    centre(e) = SVector(e.center[1], e.center[2])
    rng = Xoshiro(32)
    for trial in 1:3
        pts, bad = ellipse_scene(rng, 1000; σ = 0.5, outl = 0.4)
        res = find_ellipse(pts; rng = Xoshiro(trial))
        @test converged(res)
        e = estimate(res)
        @test e isa Ellipse{Float64}
        # σ = 0.5 px on 600 inliers: centre and axes σ/√300 ≈ 0.03 px, the angle ≈ 0.03/(a − b) rad
        @test norm(centre(e) - centre(E₀)) < 0.2 && abs(e.a - E₀.a) < 0.2 && abs(e.b - E₀.b) < 0.2 && abs(e.θ - E₀.θ) < 2.0e-3
        inl = res.inlier_mask
        good = setdiff(1:1000, bad)
        @test count(inl[good]) > 0.9 * length(good)                       # 1 px MSAC on a 0.5 px normal residual: P(|z| < 2) = 0.95
        @test count(inl[bad]) < 0.05 * length(bad)
    end
    @test find_ellipse([Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0), Point2(0.0, 1.0)]) === nothing
end

@testitem "find_ellipse: the keywords of every find_* pipeline" setup = [EllipseScene] begin
    using Test, Random
    using VisualGeometryToolkitCore: MsacScoring, estimate, converged
    using VisualGeometryToolkitEstimation: find_ellipse, find_circle, EllipseProblem,
        RansacConfig, RansacWorkspace, NoLocalOptimization
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # The shared keyword set is the one pipeline's (`_ransac_pipeline`); find_ellipse
    # declares no keyword of its own and forwards them all, as find_circle does.
    shared = Set(Base.kwarg_decl(only(methods(VgtkEstimation._ransac_pipeline))))
    forward = [Symbol("kwargs...")]
    @test Base.kwarg_decl(only(methods(find_ellipse))) == forward == Base.kwarg_decl(only(methods(find_circle)))
    # every shared keyword is accepted by find_ellipse, with the meaning it has elsewhere
    pts, _ = ellipse_scene(Xoshiro(33), 300; outl = 0.3)
    p = EllipseProblem(pts)
    supplied = Dict(
        :threshold => 2.0,
        :local_optimization => NoLocalOptimization(),
        :max_trials => 500,
        :min_trials => 50,
        :scoring => MsacScoring(p; threshold = 1.0),
        :config => RansacConfig(max_trials = 400, min_trials = 20),
        :workspace => RansacWorkspace(EllipseProblem, 300),
        :rng => Xoshiro(1),
    )
    @test Set(keys(supplied)) ∪ Set([:outlier_halfwidth]) == shared
    for (k, v) in supplied
        res = find_ellipse(pts; (; k => v)..., (k == :rng ? (;) : (; rng = Xoshiro(2)))...)
        @test converged(res)
    end
    # the retired half-width and a threshold beside a scoring are refused, as for every find_*
    @test_throws ArgumentError find_ellipse(pts; outlier_halfwidth = 5.0)
    @test_throws ArgumentError find_ellipse(pts; threshold = 2.0, scoring = MsacScoring(p; threshold = 1.0))
    @test_throws MethodError find_ellipse(pts; extent = nothing)            # no ellipse-specific keyword
end
