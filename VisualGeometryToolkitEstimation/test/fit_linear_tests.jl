# fit_linear — the robust coefficients of a linear model: exact on clean rows,
# robust to gross outliers, the prior weights, the reference loop it was
# hoisted from, and the refusals.

@testitem "fit_linear: exact, robust, prior weights" begin
    using Test, Random, LinearAlgebra, Statistics
    using VisualGeometryToolkitEstimation: fit_linear, Irls, optimization_strategy_from_toml
    using VisualGeometryToolkitCore: HuberShape, TLSShape, GemanMcclureShape, estimate, converged, decision_width

    rng = Xoshiro(11)
    m, n = 400, 5
    A = randn(rng, m, n); x0 = randn(rng, n)
    @testset "clean rows: the exact coefficients, converged" begin
        r = fit_linear(A, A * x0)
        @test converged(r)
        @test estimate(r) ≈ x0 rtol = 1.0e-12
        @test r.sigma < 1.0e-12
    end
    @testset "20% gross outliers: at the inliers' noise floor" begin
        y = A * x0 .+ 1.0e-3 .* randn(rng, m)
        bad = randperm(rng, m)[1:80]
        y[bad] .+= 5 .* randn(rng, 80)
        r = fit_linear(A, y)
        ls = A \ y
        @test norm(estimate(r) - x0) < 2.0e-4
        @test norm(ls - x0) > 1.0e-2
        @test 0.5e-3 < r.sigma < 2.0e-3                         # the MAD scale of the inliers
        @test all(r.weights[bad] .< 1)
        @test length(r.residuals) == m
    end
    @testset "a zero prior weight drops its row" begin
        y = A * x0; y[1:10] .+= 100
        p = ones(m); p[1:10] .= 0
        r = fit_linear(A, y, p)
        @test estimate(r) ≈ x0 rtol = 1.0e-12
        @test all(iszero, r.weights[1:10])
    end
    @testset "the hoisted loop: Huber at 1.345 × 1.4826·median|r|, k + 1 solves" begin
        y = A * x0 .+ 0.01 .* randn(rng, m); y[1:40] .+= 1
        function reference(A, y, w0, iters)            # rotation_vignetting.py irls_fit
            w = copy(w0); x = nothing
            for _ in 1:iters
                sw = sqrt.(w); x = (A .* sw) \ (y .* sw)
                ρ = y - A * x; s = max(1.4826 * median(abs.(ρ[w .> 0])), 1e-9)
                w = w0 .* min.(1.0, 1.345 * s ./ max.(abs.(ρ), 1e-15))
            end
            return x, w
        end
        xr, wr = reference(A, y, ones(m), 8)
        r = fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 7), tol = 0)
        @test estimate(r) ≈ xr rtol = 1.0e-8     # column-pivoted QR and √chisq_median(1) against lstsq and 1.4826
        @test r.weights ≈ wr rtol = 1.0e-6
        cnt = Float64.(rand(rng, 0:2, m))
        xb, _ = reference(A, y, cnt .* r.weights, 3)
        @test estimate(fit_linear(A, y, cnt .* r.weights; optimization_strategy = Irls{HuberShape}(max_iter = 2), tol = 0)) ≈ xb rtol = 1.0e-8
    end
    @testset "refusals and the degenerate case" begin
        @test_throws DimensionMismatch fit_linear(A, ones(m - 1))
        @test_throws DimensionMismatch fit_linear(A, ones(m), ones(3))
        @test_throws ArgumentError fit_linear(A, A * x0; optimization_strategy = Irls{TLSShape}())
        @test fit_linear(A, A * x0; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0)).retcode in (:converged, :max_iterations)
        p = zeros(m); p[1:3] .= 1
        @test fit_linear(A, A * x0, p).retcode === :degenerate
        @test_throws ArgumentError fit_linear(A, A * x0, fill(-1.0, m))
    end
end

@testitem "fit_linear: the robust estimate is the stack's Irls strategy" begin
    using Test, Random, LinearAlgebra
    using VisualGeometryToolkitEstimation: fit_linear, Irls, optimization_strategy_from_toml
    using VisualGeometryToolkitEstimation: _dflt
    using VisualGeometryToolkitCore: HuberShape, WelschShape, GemanMcclureShape, estimate, decision_width
    import VisualGeometryToolkitCore as VgtkCore

    rng = Xoshiro(5)
    m, n = 300, 4
    A = randn(rng, m, n); x0 = randn(rng, n)
    y = A * x0 .+ 1.0e-3 .* randn(rng, m); y[1:60] .+= 3 .* randn(rng, 60)
    # The default IS the file's strategy table, read by the one strategy reader.
    lo = optimization_strategy_from_toml(_dflt("fit_linear", "optimization_strategy"); class = :fit_linear)
    @test lo == Irls{HuberShape}(max_iter = 50)
    @test VgtkCore.lo_width(lo) === nothing
    r = fit_linear(A, y)
    @test estimate(r) == estimate(fit_linear(A, y; optimization_strategy = lo))
    # The strategy's max_iter caps the iterations; no private keyword remains.
    @test fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 3), tol = 0).iterations == 3
    @test_throws MethodError fit_linear(A, y; loss = HuberShape())
    @test_throws MethodError fit_linear(A, y; efficiency_constant = 2.0)
    @test_throws MethodError fit_linear(A, y; max_iter = 3)
    # A width mode resolves through Core's loss_at at the fit's σ̂: Welsch at 2.5·σ̂ ...
    rw = fit_linear(A, y; optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5))
    c² = (2.5 * rw.sigma)^2
    @test rw.weights ≈ [exp(-x^2 / c²) for x in rw.residuals] rtol = 1.0e-12
    # ... the plain multiple for every loss: a Geman–McClure multiple m puts the
    # width at m·σ̂ (3.1075479480600743 = decision_width(GM, 2.0), the half-weight
    # depth 2.0 of the retired currency, at full precision).
    rg = fit_linear(A, y; optimization_strategy = Irls{GemanMcclureShape}(efficiency_constant = 3.1075479480600743))
    cg² = (3.1075479480600743 * rg.sigma)^2
    @test rg.weights ≈ [(cg² / (cg² + x^2))^2 for x in rg.residuals] rtol = 1.0e-12
    # An absolute width: the loss at w whatever σ̂.
    rl = fit_linear(A, y; optimization_strategy = Irls{HuberShape}(loss_width = 0.004))
    @test rl.weights ≈ [abs(x) <= 0.004 ? 1.0 : 0.004 / abs(x) for x in rl.residuals] rtol = 1.0e-12
end
