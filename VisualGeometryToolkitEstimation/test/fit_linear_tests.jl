# fit_linear — the robust coefficients of a linear model: exact on clean rows,
# robust to gross outliers, the prior weights, the reference loop it was
# hoisted from, and the refusals.

@testitem "fit_linear: exact, robust, prior weights" begin
    using Test, Random, LinearAlgebra, Statistics
    using VisualGeometryToolkitEstimation: fit_linear
    using VisualGeometryToolkitCore: HuberShape, TLSShape, estimate, converged

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
        r = fit_linear(A, y; max_iter = 7, tol = 0)
        @test estimate(r) ≈ xr rtol = 1.0e-8     # column-pivoted QR and √chisq_median(1) against lstsq and 1.4826
        @test r.weights ≈ wr rtol = 1.0e-6
        cnt = Float64.(rand(rng, 0:2, m))
        xb, _ = reference(A, y, cnt .* r.weights, 3)
        @test estimate(fit_linear(A, y, cnt .* r.weights; max_iter = 2, tol = 0)) ≈ xb rtol = 1.0e-8
    end
    @testset "refusals and the degenerate case" begin
        @test_throws DimensionMismatch fit_linear(A, ones(m - 1))
        @test_throws DimensionMismatch fit_linear(A, ones(m), ones(3))
        @test_throws ArgumentError fit_linear(A, A * x0; loss = TLSShape())
        @test fit_linear(A, A * x0; loss = TLSShape(), scale_multiple = 3).retcode in (:converged, :max_iterations)
        p = zeros(m); p[1:3] .= 1
        @test fit_linear(A, A * x0, p).retcode === :degenerate
        @test_throws ArgumentError fit_linear(A, A * x0, fill(-1.0, m))
    end
end
