# Tests for `fit_scale` — the inverse-gamma MAP squared scale — and the
# partition seed of the posterior-weight state built on it.
#
# ONE estimator, one method, per component: `(rss + 2β₀)/(dof + 2α₀)` with the
# dof counting components — `ν_data = (k − model_dof)·d_g` for a partition,
# `d_g·Σπ` in the IRLS M-step (irls_scale_tests.jl pins that one).
#
# The estimate is pinned to an independent oracle, `map_scale_oracle` (the
# posterior mode found numerically from Distributions' inverse-gamma density);
# the floors are pinned to their literal values.

@testitem "fit_scale" setup = [ScaleOracles] begin
    using Test
    using Statistics: median
    using VisualGeometryCore
    using VisualGeometryCore: IGPrior, JeffreysPrior, fit_scale, partition_student_t, partition_rss,
        PosteriorWeightState, posterior_weights!, posterior_inlier_probs!, median_seeded_posterior,
        soft_posterior_weights!, chisq_median, s_hat_sq, student_t_dof

    const _PRIORS = (IGPrior(0.0, 0.0),      # Jeffreys
                     IGPrior(1.0, 1.0),
                     IGPrior(2.5, 0.125),
                     IGPrior(0.0, 3.0),      # β₀ only — the low-count floor
                     IGPrior(4.0, 0.0))      # α₀ only

    @testset "fit_scale" begin
        # The counted-dof form. `ν_data` subtracts the model dof and so cannot be
        # recovered from a weight vector — the form every partition seed uses.
        @testset "3-arg counted-dof form is the inverse-gamma MAP" begin
            RSS = 7.5
            for k in (4, 10, 50), model_dof in (4, 7), dg in (1, 2)
                ν_data = Float64((k - model_dof) * dg)
                for prior in _PRIORS
                    if ν_data + 2 * prior.alpha0 > 0            # a proper posterior
                        @test fit_scale(prior, RSS, ν_data) ≈ map_scale_oracle(prior, RSS, ν_data) rtol = 1e-7
                    else                                        # k ≤ model_dof: the floor
                        @test fit_scale(prior, RSS, ν_data) >= eps()
                    end
                end
            end
            # Integer sums are accepted and promote (the `Real` signature).
            @test fit_scale(IGPrior(1.0, 1.0), 8, 4) === fit_scale(IGPrior(1.0, 1.0), 8.0, 4.0)
        end

        # Jeffreys (α₀ = β₀ = 0) is the reduction the docstring promises.
        @testset "Jeffreys reduces to rss/dof" begin
            jeffreys = IGPrior(0.0, 0.0)
            @test jeffreys === IGPrior()          # the zero-arg form agrees
            @test jeffreys === JeffreysPrior()    # a function returning the value, not a type
            @test JeffreysPrior() isa IGPrior
            for (rss, dof) in ((7.5, 6.0), (0.3, 2.5), (1.0e3, 7.0))
                @test fit_scale(jeffreys, rss, dof) === rss / dof   # no floor engaged here
            end
        end

        # The eps() floor: under Jeffreys with no weight mass the result is
        # eps(), NOT a NaN and NOT the caller's previous scale.
        @testset "eps() floor holds" begin
            jeffreys = IGPrior(0.0, 0.0)
            n = 3
            z = fit_scale(jeffreys, 0.0, 0.0)
            @test z === eps()
            @test !isnan(z)
            @test fit_scale(jeffreys, 0.0, Float64(n)) === eps()
            @test fit_scale(jeffreys, 1.0e-300, 1.0) === eps()
            # The floor engages on the QUOTIENT, so a huge dof floors too.
            @test fit_scale(jeffreys, 1.0, 1.0e300) === eps()
            # With β₀ > 0 the prior itself lifts it off the floor.
            p = IGPrior(0.0, 3.0)
            @test fit_scale(p, 0.0, 0.0) === 6.0 / eps()   # (0 + 2·3)/max(0, eps())
            @test fit_scale(p, 0.0, 0.0) > eps()
            for prior in _PRIORS
                for (rss, dof) in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0e-320, 1.0e300))
                    s2 = fit_scale(prior, rss, dof)
                    @test isfinite(s2)
                    @test s2 >= eps()
                end
            end
        end
    end

    # The partition seed: `PosteriorWeightState(sq, n, model_dof, k; order, prior, …)`
    # is the partition's Student-t (`partition_student_t`) with the E-step run
    # once — the same mixture `posterior_weights!`/`posterior_inlier_probs!`
    # weigh the partition with, and the mixture from its definition.
    @testset "the partition seed is the partition's Student-t" begin
        n = 12
        sq_r = Float64[0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                       0.07, 0.08, 9.0, 16.0, 25.0, 36.0]
        r = sqrt.(sq_r)
        order = [3, 1, 2, 5, 4, 8, 6, 7, 9, 11, 10, 12]     # any order whose prefix is the partition
        k = 8
        model_dof, halfwidth = 4, 10.0
        log_two_outlier_halfwidth = log(2 * halfwidth)
        w_jeff = nothing
        for dg in (1, 2), prior in _PRIORS
            pws = PosteriorWeightState(sq_r, n, model_dof, k; order, prior, w = zeros(n),
                                       inlier_posterior = zeros(n), sq_r = zeros(n),
                                       log_two_outlier_halfwidth, codimension = dg)
            rss = partition_rss(order, r, k)
            @test rss == sum(sq_r[order[j]] for j in 1:k)
            ν, s2 = partition_student_t(prior, rss, k; model_dof, codimension = dg)
            @test student_t_dof(pws) === ν && s_hat_sq(pws) === s2
            @test ν == max((k - model_dof) * dg + 2 * prior.alpha0, 2)
            @test isfinite(s2) && s2 >= eps()
            @test pws.prior === prior

            w = zeros(n); p = zeros(n)
            posterior_weights!(w, sq_r, order, k; model_dof, log_two_outlier_halfwidth, codimension = dg, prior)
            posterior_inlier_probs!(p, sq_r, order, k; model_dof, log_two_outlier_halfwidth, codimension = dg, prior)
            @test p ≈ pws.inlier_posterior rtol = 1e-12
            @test w ≈ pws.w rtol = 1e-12
            # against the mixture from its definition, at the MAP scale
            ν_data = (k - model_dof) * dg
            s2_ref = map_scale_oracle(prior, rss, ν_data)
            ref = [student_t_mixture_oracle(x, ν, s2_ref, log_two_outlier_halfwidth, dg, 0.5) for x in sq_r]
            @test p ≈ getfield.(ref, :p) rtol = 1e-6
            @test w ≈ getfield.(ref, :w) rtol = 1e-5
            if dg == 1
                if prior.alpha0 == prior.beta0 == 0
                    w_jeff = w
                else
                    @test w != w_jeff       # a proper prior moves the weights
                end
            end
        end
    end

    # `median_seeded_posterior` moved here from RobustVisualGeometry verbatim; the
    # state it builds must equal RVG's copy, field for field.
    @testset "median_seeded_posterior == RVG's verbatim copy" begin
        function _rvg_copy(sq, n; outlier_halfwidth, codimension = 2, ν = 1.0e4, prior = IGPrior(0.0, 0.0))
            m_d = chisq_median(codimension)
            sq_r = Vector{Float64}(sq[1:n])
            s2 = max(median(sq_r) / m_d, eps())
            w = Vector{Float64}(undef, n); ip = Vector{Float64}(undef, n)
            log_two_outlier_halfwidth = log(2 * Float64(outlier_halfwidth))
            soft_posterior_weights!(w, ip, sq_r, Float64(ν), s2, log_two_outlier_halfwidth, n, codimension)
            PosteriorWeightState(w, ip, sq_r, Float64(ν), s2, log_two_outlier_halfwidth, codimension, prior)
        end
        sq = [0.01, 0.04, 0.02, 0.09, 0.03, 4.0, 9.0, 0.05, 16.0, 0.06, 99.0]
        for dg in (1, 2), n in (5, 11), prior in (IGPrior(), IGPrior(2.0, 0.1))
            a = median_seeded_posterior(sq, n; outlier_halfwidth = 3.0, codimension = dg, prior)
            b = _rvg_copy(sq, n; outlier_halfwidth = 3.0, codimension = dg, prior)
            for f in fieldnames(PosteriorWeightState)
                @test getfield(a, f) == getfield(b, f)
            end
            @test a.sq_residuals !== sq                  # its own buffers
        end
        @test_throws ArgumentError median_seeded_posterior(sq, 5; outlier_halfwidth = 1.0, codimension = 3)
    end
end
