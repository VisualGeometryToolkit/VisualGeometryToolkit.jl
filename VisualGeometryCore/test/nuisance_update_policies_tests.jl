# The nuisance parts' update policies (`Estimated`, `FixedS2`, `FixedPrior`,
# `FixedNu`, `ConsensusDof`) on a `PosteriorWeightState`.

@testitem "nuisance_update_policies" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: PosteriorWeightState, IGPrior, s_hat_sq, student_t_dof, inlier_prior,
        update_scale!, update_nu!, update_inlier_prior!, Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof

    sq = Float64[0.01, 0.04, 0.02, 0.09, 0.03, 25.0, 0.05, 49.0, 0.06, 0.02]
    n = length(sq)
    fresh() = (p = PosteriorWeightState(zeros(n), zeros(n), copy(sq), 6.0, 0.05, log(20.0), 2, IGPrior(0.0, 0.0));
               update_weights!(p, n); p)

    # Estimated() IS the estimating block, bit for bit
    a, b = fresh(), fresh()
    @test update_scale!(a, n, Estimated()) === update_scale!(b, n)
    @test update_nu!(a, n, Estimated()) === update_nu!(b, n)
    @test update_inlier_prior!(a, n, Estimated()) === update_inlier_prior!(b, n)
    @test (s_hat_sq(a), student_t_dof(a), inlier_prior(a)) === (s_hat_sq(b), student_t_dof(b), inlier_prior(b))

    # a held part writes its value and nothing else
    p = fresh()
    ν0, π0 = student_t_dof(p), inlier_prior(p)
    @test update_scale!(p, n, FixedS2(0.7)) === 0.7
    @test s_hat_sq(p) === 0.7 && student_t_dof(p) === ν0 && inlier_prior(p) === π0
    @test update_nu!(p, n, FixedNu(3.0)) === 3.0
    @test student_t_dof(p) === 3.0 && s_hat_sq(p) === 0.7
    @test update_inlier_prior!(p, n, FixedPrior(0.8)) === 0.8
    @test inlier_prior(p) === 0.8
    # the one clamp applies to a held prior too
    @test update_inlier_prior!(p, n, FixedPrior(1.0e-6)) === 1.0e-3

    # refused values
    @test_throws ArgumentError FixedS2(-1.0)
    @test_throws ArgumentError FixedS2(0.0)
    @test_throws ArgumentError FixedS2(Inf)
    @test_throws ArgumentError FixedPrior(0.0)
    @test_throws ArgumentError FixedPrior(1.0)
    @test_throws ArgumentError FixedNu(0.0)
    @test_throws ArgumentError FixedNu(Inf)

    # ConsensusDof: the seed's ν IS the consensus set's posterior-predictive dof,
    # 2α₀ + (k − model_dof)·d_g, and the ν block holds it
    for prior in (IGPrior(0.0, 0.0), IGPrior(1.5, 0.25))
        order = sortperm(sq); k, model_dof = 8, 5
        q = PosteriorWeightState(sq, n, model_dof, k; order, prior, w = zeros(n), inlier_posterior = zeros(n),
                                 sq_r = zeros(n), log_two_outlier_halfwidth = log(20.0), codimension = 2)
        ν0 = student_t_dof(q)
        @test ν0 == 2 * prior.alpha0 + (k - model_dof) * 2
        for _ in 1:5
            update_scale!(q, n); update_inlier_prior!(q, n)
            @test update_nu!(q, n, ConsensusDof()) === ν0
            update_weights!(q, n)
        end
        @test student_t_dof(q) === ν0
    end
end
