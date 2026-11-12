# fit_linear — the robust coefficients of a linear model: exact on clean rows,
# robust to gross outliers at a GIVEN scale, the prior weights, the reference
# loop it was hoisted from, and the refusals (the robust-estimation contract).

@testitem "fit_linear: exact, robust, prior weights" begin
    using Test, Random, LinearAlgebra, Statistics
    using VisualGeometryToolkitEstimation: fit_linear
    using VisualGeometryToolkitCore: Irls
    using VisualGeometryToolkitCore: HuberShape, TLSShape, estimate, converged, noise_scale, scale_source, ScaleGiven
    import VisualGeometryToolkitCore as VgtkCore

    rng = Xoshiro(11)
    m, n = 400, 5
    A = randn(rng, m, n); x0 = randn(rng, n)
    @testset "clean rows: the exact coefficients, converged" begin
        r = fit_linear(A, A * x0; scale = 1.0e-3)
        @test converged(r)
        @test estimate(r) ≈ x0 rtol = 1.0e-12
        @test all(==(1.0), r.weights)                            # every residual inside the width
        r2 = fit_linear(A, A * x0; optimization_strategy = nothing)
        @test estimate(r2) ≈ x0 rtol = 1.0e-12
        @test r2.retcode === :solved
    end
    @testset "20% gross outliers: at the inliers' noise floor" begin
        y = A * x0 .+ 1.0e-3 .* randn(rng, m)
        bad = randperm(rng, m)[1:80]
        y[bad] .+= 5 .* randn(rng, 80)
        r = fit_linear(A, y; scale = 1.0e-3)
        ls = A \ y
        @test norm(estimate(r) - x0) < 2.0e-4
        @test norm(ls - x0) > 1.0e-2
        @test noise_scale(r) == 1.0e-3 && scale_source(r) == ScaleGiven
        @test r.loss_width ≈ 1.345e-3                             # Huber's constant at codimension 1 × the scale
        @test all(r.weights[bad] .< 1)
        @test length(r.residuals) == m
        @test r.n == m                                            # Huber never zeroes a weight
    end
    @testset "a zero prior weight drops its row" begin
        y = A * x0; y[1:10] .+= 100
        p = ones(m); p[1:10] .= 0
        r = fit_linear(A, y, p; scale = 1.0e-3)
        @test estimate(r) ≈ x0 rtol = 1.0e-12
        @test r.n == m - 10
    end
    @testset "the hoisted loop at a FIXED width: Huber at 1.345 × σ, k + 1 solves" begin
        y = A * x0 .+ 0.01 .* randn(rng, m); y[1:40] .+= 1
        function reference(A, y, w0, iters, s)         # rotation_vignetting.py irls_fit, the scale held at s
            w = copy(w0); x = nothing
            for _ in 1:iters
                sw = sqrt.(w); x = (A .* sw) \ (y .* sw)
                ρ = y - A * x
                w = w0 .* min.(1.0, 1.345 * s ./ max.(abs.(ρ), 1e-15))
            end
            return x, w
        end
        xr, wr = reference(A, y, ones(m), 8, 0.01)
        r = fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 7), scale = 0.01, tol = 0)
        @test estimate(r) ≈ xr rtol = 1.0e-8
        @test r.weights ≈ wr rtol = 1.0e-6
        cnt = Float64.(rand(rng, 0:2, m))
        xb, _ = reference(A, y, cnt .* r.weights, 3, 0.01)
        @test estimate(fit_linear(A, y, cnt .* r.weights; optimization_strategy = Irls{HuberShape}(max_iter = 2),
                                  scale = 0.01, tol = 0)) ≈ xb rtol = 1.0e-8
    end
    @testset "refusals and the degenerate case" begin
        @test_throws DimensionMismatch fit_linear(A, ones(m - 1); scale = 1.0)
        @test_throws DimensionMismatch fit_linear(A, ones(m), ones(3); scale = 1.0)
        @test_throws ArgumentError fit_linear(A, A * x0; optimization_strategy = Irls{TLSShape}(), scale = 1.0)
        @test fit_linear(A, A * x0; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 1.0e-3).retcode === :converged
        @test fit_linear(A, A * x0; optimization_strategy = Irls{TLSShape}(), width = 3.0e-3).retcode === :converged
        p = zeros(m); p[1:3] .= 1
        @test fit_linear(A, A * x0, p; scale = 1.0).retcode === :degenerate
        @test_throws ArgumentError fit_linear(A, A * x0, fill(-1.0, m); scale = 1.0)
    end
end

@testitem "fit_linear: the robust estimate is the stack's Irls strategy" begin
    using Test, Random, LinearAlgebra
    using VisualGeometryToolkitEstimation: fit_linear, optimization_strategy_from_toml
    using VisualGeometryToolkitCore: Irls
    using VisualGeometryToolkitEstimation: _dflt
    using VisualGeometryToolkitCore: HuberShape, WelschShape, GemanMcclureShape, estimate
    import VisualGeometryToolkitCore as VgtkCore

    rng = Xoshiro(5)
    m, n = 300, 4
    A = randn(rng, m, n); x0 = randn(rng, n)
    y = A * x0 .+ 1.0e-3 .* randn(rng, m); y[1:60] .+= 3 .* randn(rng, 60)
    # The default IS the file's strategy table (Huber, the owner's choice for accuracy), read by the one strategy reader.
    lo = optimization_strategy_from_toml(_dflt("fit_linear", "optimization_strategy"); class = :fit_linear)
    @test lo == Irls{HuberShape}(max_iter = 50)
    @test VgtkCore.lo_width(lo) === nothing
    r = fit_linear(A, y; scale = 1.0e-3)
    @test estimate(r) == estimate(fit_linear(A, y; optimization_strategy = lo, scale = 1.0e-3))
    # The strategy's max_iter caps the iterations; no private keyword remains.
    @test fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 3), scale = 1.0e-3, tol = 0).iterations == 3
    @test_throws MethodError fit_linear(A, y; loss = HuberShape(), scale = 1.0e-3)
    @test_throws MethodError fit_linear(A, y; efficiency_constant = 2.0, scale = 1.0e-3)
    @test_throws MethodError fit_linear(A, y; max_iter = 3, scale = 1.0e-3)
    @test_throws MethodError fit_linear(A, y; irls = Irls{HuberShape}(), scale = 1.0e-3)
    @test_throws ArgumentError fit_linear(A, y; local_optimization = Irls{HuberShape}(), scale = 1.0e-3)   # retired, naming optimization_strategy
    # The strategy's efficiency constant is the PLAIN multiple for every shape: width = c · scale.
    rw = fit_linear(A, y; optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5), scale = 1.0e-3)
    @test rw.loss_width ≈ 2.5e-3
    @test rw.weights ≈ [exp(-x^2 / rw.loss_width^2) for x in rw.residuals] rtol = 1.0e-12
    rg = fit_linear(A, y; optimization_strategy = Irls{GemanMcclureShape}(efficiency_constant = 2.0), scale = 1.0e-3)
    cg² = rg.loss_width^2
    @test rg.loss_width ≈ 2.0e-3
    @test rg.weights ≈ [(cg² / (cg² + x^2))^2 for x in rg.residuals] rtol = 1.0e-12
    # An absolute width: the loss at w; the result records no scale.
    rl = fit_linear(A, y; width = 0.004)
    @test rl.weights ≈ [abs(x) <= 0.004 ? 1.0 : 0.004 / abs(x) for x in rl.residuals] rtol = 1.0e-12
    @test VgtkCore.scale_source(rl) == VgtkCore.ScaleNone && isnan(rl.scale) && rl.loss_width == 0.004
    @test_throws ArgumentError VgtkCore.noise_scale(rl)
end
