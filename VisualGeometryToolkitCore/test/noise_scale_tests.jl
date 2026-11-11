# The robust layer's noise scale and loss width against Monte Carlo:
# `mad_scale` (the median-of-norms scale, Rayleigh-median at d = 2) through the
# calibration harness, and `efficiency_constant`'s defining claim — 95% Gaussian
# efficiency of the loss's M-estimate — by paired simulation.
@testitem "noise_scale" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Statistics
    using Distributions: Chi, Normal, pdf, quantile
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: mad_scale, chisq_median, efficiency_constant, weight,
        WelschShape, CauchyShape, GemanMcclureShape
    const CHar = VgtkCore.Synthetic

    @testset "mad_scale: calibrated against its asymptotic variance (d = 1, 2, 6)" begin
        # σ̂ = median‖r‖/√q, q = chisq_median(d): the sample median of σ·χ_d norms has
        # asymptotic variance 1/(4n·f(m)²), so Var(σ̂) = σ²/(4n·q·f_χd(√q)²)
        predicted_var(σ, n, d) = (q = chisq_median(d); f = pdf(Chi(d), sqrt(q)); σ^2 / (4 * n * q * f^2))
        sigma_trial(d, n, σ) = rng -> begin
            residuals = [SVector{d}(σ * randn(rng, d)) for _ in 1:n]
            ([mad_scale(residuals) - σ], fill(predicted_var(σ, n, d), 1, 1))
        end
        for d in (1, 2, 6)
            r = CHar.run_calibration(sigma_trial(d, 200, 0.7), MersenneTwister(40 + d), 4000, 1)
            @test CHar.is_calibrated(r)
        end
    end

    @testset "mad_scale: gross outliers move it to a predicted quantile" begin
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
                VgtkCore._mad_scale_norms(; norms, codimension = d)
            end
            predicted = σ * quantile(Chi(d), 0.5 / (1 - ε)) / quantile(Chi(d), 0.5)
            @test abs(mean(est) - predicted) < 4 * std(est) / sqrt(N)
        end
    end

    @testset "mad_scale: residual vectors, centred on the model" begin
        rs = [SVector(3.0, 4.0), SVector(0.0, 1.0), SVector(6.0, 8.0)]          # norms 5, 1, 10
        @test @inferred(mad_scale(rs)) == 5.0 / sqrt(chisq_median(2))
        @test mad_scale(rs) === VgtkCore._mad_scale_norms(; norms = [5.0, 1.0, 10.0], codimension = 2)   # the internal norms form
        @test_throws MethodError mad_scale(; norms = [5.0, 1.0, 10.0], codimension = 2)                   # no public norms form
        @test mad_scale(SVector{1}.([-2.0, 1.0, 3.0])) == 2.0 / sqrt(chisq_median(1))   # d = 1: |r|, no centring
        @test mad_scale([SVector(1.0f0, 0.0f0), SVector(0.0f0, 2.0f0), SVector(0.0f0, 3.0f0)]) ≈ 2 / sqrt(chisq_median(2))
        x = 0.37 .* randn(MersenneTwister(7), 20_001) .+ 0.9
        @test mad_scale(SVector{1}.(x)) > 3 * 0.37                                  # a bias counts as noise
        # no scalar form: the retired centred MAD's call shape is refused, not silently re-meant
        @test_throws MethodError mad_scale([1.0, 2.0, 3.0])
        @test_throws MethodError mad_scale([5.0, 1.0, 10.0], 2)
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
