# The IRLS M-step's scale, `update_scale!`, is the ECM step of the d_g-variate
# Student-t with scale ŝ²·I:  ŝ² ← Σπᵢuᵢrᵢ² / (d_g·Σπᵢ) = ŝ²·Σwᵢrᵢ² / (d_g·Σπᵢ).
#
# `sq_residuals` holds squared NORMS of d_g-vectors and the E-step reads ŝ² per
# component. The step was Σwᵢrᵢ²/Σwᵢ, whose fixed point is d_g·σ² — ŝ ≈ 1.42σ at
# d_g = 2 and 2.44σ at d_g = 6 on Gaussian data — with the inverse-gamma prior's
# pseudo-counts added in the units of the precision-scaled weights.
#
# Monte Carlo on the EM alone (fixed residuals, E-step ⇄ M-step, outliers made
# negligible by a huge outlier half-width): ŝ/σ ≈ 1 at d_g = 1, 2, 6 on Gaussian
# data, and ν₀ = 4 with ŝ recovered on t₄ data.

@testitem "irls_scale" setup = [ScaleOracles] begin
    using Test, Random
    using VisualGeometryCore
    using VisualGeometryCore: PosteriorWeightState, IGPrior, update_weights!, update_scale!, update_nu!,
        next_nu, weighted_rss
    const _t_sq = t_sq_residuals

    # Squared norms of d-variate Gaussian residuals, scale σ.
    _gauss_sq(rng, n, d, σ) = [σ^2 * sum(abs2, randn(rng, d)) for _ in 1:n]

    # The EM on fixed residuals, from a deliberately wrong start (ν = 30, ŝ² = 3).
    function _em(sq, d; ν0 = 30.0, s2_0 = 3.0, outlier_halfwidth = 1e6, iters = 200, prior = IGPrior())
        n = length(sq)
        pws = PosteriorWeightState(zeros(n), zeros(n), copy(sq), ν0, s2_0, log(2 * outlier_halfwidth), d, prior)
        update_weights!(pws, n)
        for _ in 1:iters
            update_scale!(pws, n)
            update_nu!(pws, n)
            update_weights!(pws, n)
        end
        return pws
    end

    @testset "IRLS M-step: ŝ² per component (÷ d_g)" begin
        @testset "update_scale! is ŝ²·Σwr² / (d_g·Σπ), then update_nu! (d = $d)" for d in (1, 2, 6)
            n = 7
            sq = [0.01, 0.04, 0.09, 0.64, 9.0, 16.0, 1.0e9]     # the last is past the prefix
            for prior in (IGPrior(), IGPrior(1.0, 1.0), IGPrior(2.5, 0.125), IGPrior(0.0, 3.0))
                pws = PosteriorWeightState(zeros(n), zeros(n), copy(sq), 5.0, 1.7, log(2.0 * 8), d, prior)
                update_weights!(pws, n)
                w, π_in, s2_old, ν_old = copy(pws.w), copy(pws.inlier_posterior), pws.s_hat_sq, pws.ν
                update_scale!(pws, n - 1)                       # n is a PREFIX length
                @test pws.ν === ν_old                           # the scale block only
                update_nu!(pws, n - 1)
                m = n - 1
                sum_wr2 = 0.0; sum_π = 0.0
                for i in 1:m
                    sum_wr2 += w[i] * sq[i]; sum_π += π_in[i]
                end
                @test pws.s_hat_sq ≈ map_scale_oracle(prior, s2_old * sum_wr2, d * sum_π) rtol = 1e-7
                @test pws.ν === next_nu(ν_old, sq, pws.s_hat_sq, π_in, m; d_g = d)
            end
        end

        @testset "scale-equivariant under an inverse-gamma prior (d = $d)" for d in (1, 2)
            # r² → c²r², ŝ² → c²ŝ², outlier half-width → c·(outlier half-width), β₀ → c²β₀: one M-step scales ŝ² by c²
            # exactly (up to rounding) and leaves ν and the weights' posteriors fixed.
            rng = MersenneTwister(5 + d)
            n = 400
            sq = _t_sq(rng, n, d, 4.0, 1.0)
            c = 37.0
            states = map((1.0, c)) do k
                pws = PosteriorWeightState(zeros(n), zeros(n), k^2 .* sq, 8.0, k^2 * 1.5,
                                           log(2 * 20k), d, IGPrior(2.0, k^2 * 0.3))
                update_weights!(pws, n)
                update_scale!(pws, n)
                update_nu!(pws, n)
                pws
            end
            @test states[2].s_hat_sq ≈ c^2 * states[1].s_hat_sq rtol = 1e-12
            @test states[2].ν ≈ states[1].ν rtol = 1e-10
        end

        @testset "Gaussian data: ŝ/σ ≈ 1 (d = $d)" for d in (1, 2, 6)
            rng = MersenneTwister(100 + d)
            σ = 0.7
            pws = _em(_gauss_sq(rng, 20000, d, σ), d)
            @test abs(sqrt(pws.s_hat_sq) / σ - 1) < 0.02      # was √d_g: 1.42 at d = 2, 2.44 at d = 6
            @test pws.ν > 50                                   # near-Gaussian
        end

        @testset "t₄ data: ν₀ = 4 and ŝ recovered (d = $d)" for d in (1, 2, 6)
            rng = MersenneTwister(200 + d)
            s = 1.3
            pws = _em(_t_sq(rng, 20000, d, 4.0, s), d)
            @test abs(sqrt(pws.s_hat_sq) / s - 1) < 0.03
            @test 3.5 < pws.ν < 4.6
        end
    end

    @testset "a zero-weight point with an infinite residual leaves ŝ² finite" begin
        n = 6
        sq = [0.01, 0.04, 0.02, 0.03, Inf, 0.05]     # point 5: a degenerate (Sampson) point
        w = [1.0, 0.5, 1.0, 0.25, 0.0, 1.0]
        @test weighted_rss(w, sq, n) == 0.01 + 0.5 * 0.04 + 0.02 + 0.25 * 0.03 + 0.05
        @test weighted_rss(w, sq, 4) == 0.01 + 0.5 * 0.04 + 0.02 + 0.25 * 0.03
        pws = PosteriorWeightState(copy(w), [1.0, 1.0, 1.0, 1.0, 0.0, 1.0], sq, 5.0, 0.03,
                                   log(20.0), 1, IGPrior())
        @test isfinite(update_scale!(pws, n))
    end
end
