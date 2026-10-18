# The robust layer's noise scale and loss width against Monte Carlo:
# `noise_sigma` (the median-of-norms scale, Rayleigh-median at d = 2) through the
# calibration harness, and `efficiency_constant`'s defining claim — 95% Gaussian
# efficiency of the loss's M-estimate — by paired simulation.
@testitem "noise_scale" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Statistics
    using Distributions: Chi, pdf, quantile
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: noise_sigma, chisq_median, efficiency_constant, weight,
        WelschShape, CauchyShape, GemanMcclureShape
    const CHar = VgtkCore.Synthetic

    @testset "noise_sigma: calibrated against its asymptotic variance (d = 1, 2, 6)" begin
        # σ̂ = median‖r‖/√q, q = chisq_median(d): the sample median of σ·χ_d norms has
        # asymptotic variance 1/(4n·f(m)²), so Var(σ̂) = σ²/(4n·q·f_χd(√q)²)
        predicted_var(σ, n, d) = (q = chisq_median(d); f = pdf(Chi(d), sqrt(q)); σ^2 / (4 * n * q * f^2))
        sigma_trial(d, n, σ) = rng -> begin
            norms = [norm(σ * randn(rng, d)) for _ in 1:n]
            ([noise_sigma(norms, d) - σ], fill(predicted_var(σ, n, d), 1, 1))
        end
        for d in (1, 2, 6)
            r = CHar.run_calibration(sigma_trial(d, 200, 0.7), MersenneTwister(40 + d), 4000, 1)
            @test CHar.is_calibrated(r)
        end
    end

    @testset "noise_sigma: gross outliers move it to a predicted quantile" begin
        # A fraction ε of gross errors (above every clean norm) puts the median at the
        # clean distribution's 0.5/(1 − ε) quantile: σ̂/σ = F⁻¹(0.5/(1 − ε))/F⁻¹(½).
        # Robust (50% breakdown), not unbiased — at ε = 0.2, d = 2 it reads 1.19σ.
        ε, σ, n, N = 0.2, 0.7, 400, 2000
        for d in (1, 2, 6)
            rng = MersenneTwister(50 + d)
            est = map(1:N) do _
                norms = [norm(σ * randn(rng, d)) for _ in 1:n]
                k = round(Int, ε * n)
                norms[1:k] .= 10σ .+ 100σ .* rand(rng, k)
                noise_sigma(norms, d)
            end
            predicted = σ * quantile(Chi(d), 0.5 / (1 - ε)) / quantile(Chi(d), 0.5)
            @test abs(mean(est) - predicted) < 4 * std(est) / sqrt(N)
        end
    end

    @testset "efficiency_constant: 95% Gaussian efficiency of the M-estimate" begin
        # The constant's definition: at width scale_multiple·σ, scale_multiple = efficiency_constant(shape, d),
        # the loss's M-estimate of a d-dim location has 95% of the sample mean's
        # efficiency under N(μ, σ²I). Paired: the same samples give both estimates.
        function mestimate(xs, S, c2)
            μ = sum(xs) / length(xs)
            for _ in 1:100
                w = [weight(S, sum(abs2, x - μ), c2) for x in xs]
                μn = sum(w .* xs) / sum(w)
                done = norm(μn - μ) < 1.0e-12
                μ = μn
                done && break
            end
            return μ
        end
        for (S, d) in ((WelschShape(), 1), (WelschShape(), 2), (CauchyShape(), 1), (GemanMcclureShape(), 2))
            scale_multiple = efficiency_constant(typeof(S), d)
            rng = MersenneTwister(60 + d)
            n, N = 400, 3000
            e_mean = zeros(N); e_m = zeros(N)
            for t in 1:N
                xs = [SVector{d}(randn(rng, d)) for _ in 1:n]
                e_mean[t] = sum(abs2, sum(xs) / n)
                e_m[t] = sum(abs2, mestimate(xs, S, scale_multiple^2))
            end
            # measured 2026-09-22: 0.952 / 0.948 / 0.952 / 0.948, bootstrap SE 0.006–0.008
            @test abs(mean(e_mean) / mean(e_m) - 0.95) < 0.025
        end
    end
end
