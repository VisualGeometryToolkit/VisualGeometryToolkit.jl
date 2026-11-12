# EllipseProblem / find_ellipse — the 2D ellipse as one more model of the RANSAC
# family, as the circle (ransac_circle_tests.jl): the minimal solver through the
# problem, the residual against the geometric distance, the chart rows (the
# orthogonal distances) against finite differences, the chart-LM refit, the recovery under 40% outliers (a straight
# run and clutter), and the keyword set shared with the other `find_*` pipelines.

@testitem "EllipseProblem: minimal solver, residual, chart rows, refit" setup = [ConicScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: Point2, Ellipse, residual_rows!, residual_jacobian_rows!, ambient_coordinates, boundary_point,
        whitened_residuals!
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
    r = zeros(200); VgtkEstimation.mahalanobis_distances!(r, p, e)
    @test maximum(abs, r) < 1.0e-8                                       # pixels
    allocs(r, p, θ) = (VgtkEstimation.mahalanobis_distances!(r, p, θ); @allocated VgtkEstimation.mahalanobis_distances!(r, p, θ))
    @test allocs(r, p, e) == 0
    # a collinear triple and a repeated point: no ellipse
    q = EllipseProblem([Point2(0.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 2.0), Point2(0.0, 3.0), Point2(3.0, 0.5), Point2(0.0, 0.0)])
    @test VgtkEstimation.solve(q, [1, 2, 3, 4, 5]).count == 0
    @test !VgtkEstimation.test_sample(q, [1, 2, 3, 4, 5]) && !VgtkEstimation.test_sample(q, [1, 4, 5, 2, 6])   # the gate refuses them first
    @test VgtkEstimation.test_sample(p, [1, 2, 3, 4, 5])
    # the signed residual is the Sampson residual: the geometric signed distance to first
    # order, and the scoring distance its magnitude.
    # Along the major axis (normal there) a point δ off is (δ − δ²/(2(a + δ)))…
    off = [Point2{Float64}(centre(E₀) + (E₀.a + δ) * SVector(cos(E₀.θ), sin(E₀.θ))) for δ in range(-5, 5, length = 21)]
    po = EllipseProblem(off)
    ro = zeros(21); whitened_residuals!(ro, po, E₀)
    δs = collect(range(-5, 5, length = 21))
    @test ro ≈ δs .- δs .^ 2 ./ (2 .* (E₀.a .+ δs)) atol = 1.0e-10
    @test VgtkEstimation.mahalanobis_distances!(zeros(21), po, E₀) == abs.(ro)
    # …and anywhere on the curve, a point δ along the normal is δ to first order
    nrm(t) = normalize(SMatrix{2, 2}(cos(E₀.θ), sin(E₀.θ), -sin(E₀.θ), cos(E₀.θ)) * SVector(E₀.b * cos(t), E₀.a * sin(t)))   # outward
    ts = range(0, 2π, length = 13)
    near = [Point2{Float64}(SVector(boundary_point(E₀, t)...) + s * 0.5 * nrm(t)) for t in ts for s in (1, -1)]
    rn = zeros(26); whitened_residuals!(rn, EllipseProblem(near), E₀)
    @test maximum(abs, rn .- repeat([0.5, -0.5], 13)) < 0.5^2 * E₀.a / E₀.b^2   # second order in δ, the curvature at most a/b²
    # the chart rows — the orthogonal distances, whose foot point is a root solve
    # AD must not see — against central differences, at random weights
    pn, _ = ellipse_scene(rng, 100; σ = 1.0, outl = 0.0)
    pp = EllipseProblem(pn)
    e₀ = Ellipse(Point2((centre(E₀) .+ SVector(2.0, -1.5))...), E₀.a + 3, E₀.b - 2, E₀.θ + 0.01)
    lm = VgtkEstimation._lm_state(pp, VgtkEstimation._chart(pp, e₀), rand(rng, 100))
    a0 = SVector(0.3, -0.2, 1.0e-3, -2.0e-3, 5.0e-4)
    J = zeros(100, 5); residual_jacobian_rows!(J, lm, a0)
    rows(a) = residual_rows!(zeros(100), lm, SVector{5}(a))
    h = 1.0e-6
    Jfd = reduce(hcat, [(rows(a0 + h * SVector{5}(k .== 1:5)) - rows(a0 - h * SVector{5}(k .== 1:5))) / 2h for k in 1:5])
    @test J ≈ Jfd rtol = 1.0e-5
    # the rows ARE the orthogonal distances (√w-weighted) at the anchor
    @test abs(rows(zero(a0))[7] / lm.sqrtw[7]) ≈ minimum(φ -> norm(SVector(pn[7]...) - SVector(boundary_point(e₀, φ)...)), range(0, 2π, 200_001)) atol = 1.0e-5
    # the chart-LM refit (refine!) from a perturbed ellipse lands at the noise floor
    ws = VgtkEstimation.refine_workspace(typeof(pp), 100)
    e₁ = VgtkCore.refine!(ws, pp, e₀, ones(100))
    @test e₁ isa Ellipse{Float64}
    @test norm(centre(e₁) - centre(E₀)) < 0.5 && abs(e₁.a - E₀.a) < 0.5 && abs(e₁.b - E₀.b) < 0.5 && abs(e₁.θ - E₀.θ) < 0.01   # σ = 1 px on 100 points
    refit_allocs(ws, p, θ, w) = (VgtkCore.refine!(ws, p, θ, w); @allocated VgtkCore.refine!(ws, p, θ, w))
    @test refit_allocs(ws, pp, e₀, ones(100)) == 0
end

@testitem "find_ellipse: 40% outliers (a straight run and clutter)" setup = [ConicScenes] begin
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitCore: inlier_mask   # the robust contract's accessors
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore: Point2, Ellipse, estimate, converged, retcode, iterations
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
        inl = inlier_mask(res)
        good = setdiff(1:1000, bad)
        @test count(inl[good]) > 0.9 * length(good)                       # 1 px MSAC on a 0.5 px normal residual: P(|z| < 2) = 0.95
        @test count(inl[bad]) < 0.05 * length(bad)
    end
    @test let r = find_ellipse([Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0), Point2(0.0, 1.0)])
        retcode_symbol(r) === :no_model && iterations(r) == 0 && estimate(r) === nothing
    end
    # no sample gives an ellipse (collinear points): a :no_model result whose estimate is nothing,
    # not a MethodError (2026-11-03: `zero(::Type{Ellipse{Float64}})` did not exist)
    none = find_ellipse([Point2(Float64(i), 2.0 * i + 1) for i in 1:10]; rng = Xoshiro(1))
    @test !converged(none) && retcode_symbol(none) === :no_model && estimate(none) === nothing && !any(inlier_mask(none))
end

@testitem "find_ellipse: the keywords of every find_* pipeline" setup = [ConicScenes] begin
    using Test, Random
    using VisualGeometryToolkitCore: MsacScoring, estimate, converged
    using VisualGeometryToolkitEstimation: find_ellipse, find_circle, EllipseProblem,
        RansacConfig, RansacWorkspace
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
        :local_optimization_strategy => nothing,
        :max_trials => 500,
        :min_trials => 50,
        :scoring => MsacScoring(p; threshold = 1.0),
        :config => RansacConfig(max_trials = 400, min_trials = 20),
        :workspace => RansacWorkspace(EllipseProblem, 300),
        :rng => Xoshiro(1),
    )
    @test Set(keys(supplied)) ∪ Set([:threshold, :outlier_halfwidth, :local_optimization]) == shared   # the two widths and the retired keyword: declared to be refused
    for (k, v) in supplied
        res = find_ellipse(pts; (; k => v)..., (k == :rng ? (;) : (; rng = Xoshiro(2)))...)
        @test converged(res)
    end
    # a bare width keyword is refused, alone or beside a scoring, as for every find_*
    @test_throws ArgumentError find_ellipse(pts; outlier_halfwidth = 5.0)
    @test_throws ArgumentError find_ellipse(pts; threshold = 2.0)
    @test_throws ArgumentError find_ellipse(pts; threshold = 2.0, scoring = MsacScoring(p; threshold = 1.0))
    @test_throws MethodError find_ellipse(pts; extent = nothing)            # no ellipse-specific keyword
end
