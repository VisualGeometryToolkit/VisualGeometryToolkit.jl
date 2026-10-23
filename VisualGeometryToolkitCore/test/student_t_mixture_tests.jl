# `StudentTMixture` — the scalar E-step: one Student-t + uniform mixture, one
# squared residual. `inlier_posterior`, `posterior_weight` and `mixture_nll` are
# plain allocation-free functions (callable inside a KernelAbstractions kernel),
# and every vector path — `soft_posterior_weights!`, `posterior_weights!`,
# `posterior_inlier_probs!` — calls the same per-point kernel.

@testitem "student_t_mixture" setup = [ScaleOracles] begin
    using Test, Random
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: StudentTMixture, inlier_posterior, posterior_weight, mixture_nll,
        posterior_weight_derivative, soft_posterior_weights!, GauInlier, decision_threshold,
        PosteriorWeightState, JeffreysPrior, update_inlier_prior!

    # The mixture from its definition in probability space (Distributions' Student-t).
    const _ref_mixture = student_t_mixture_oracle

    @kernel function _weights_kernel!(w, @Const(sq_r), m)
        i = @index(Global)
        @inbounds w[i] = posterior_weight(m, sq_r[i])
    end

    @testset "StudentTMixture: the scalar E-step" begin
        sq_r = [0.0, 1e-4, 0.01, 0.2, 1.0, 4.0, 25.0, 400.0]
        for (ν, s2, halfwidth, dg, πin) in ((5.0, 0.04, 2.0, 1, 0.5), (12.0, 1.3, 30.0, 2, 0.3),
                                    (1e4, 0.01, 0.5, 2, 0.8), (2.0, 2.5, 100.0, 1, 0.5),
                                    (Inf, 0.04, 2.0, 1, 0.5), (Inf, 1.3, 30.0, 2, 0.3))
            m = StudentTMixture(ν, s2, log(2 * halfwidth), dg, πin)
            @test isbits(m)
            for x in sq_r
                ref = _ref_mixture(x, ν, s2, log(2 * halfwidth), dg, πin)
                @test inlier_posterior(m, x) ≈ ref.p rtol = 1e-10 atol = 1e-300
                @test posterior_weight(m, x) ≈ ref.w rtol = 1e-6 atol = 1e-300
                @test mixture_nll(m, x) ≈ ref.nll rtol = 1e-10
            end
            # an infinite residual is the outlier term's: P = 0, w = 0, and a finite
            # NLL, −log((1−π)·p_out) (the naive probability form gives it too, but
            # would overflow long before)
            @test inlier_posterior(m, Inf) == 0.0
            @test posterior_weight(m, Inf) == 0.0
            @test mixture_nll(m, Inf) ≈ -(log1p(-πin) - dg * log(2 * halfwidth)) rtol = 1e-14

            # the vector E-step is the scalar one, bit for bit
            n = length(sq_r); w = zeros(n); p = zeros(n)
            soft_posterior_weights!(w, p, sq_r, ν, s2, log(2 * halfwidth), n, dg, πin)
            @test w == posterior_weight.(Ref(m), sq_r)
            @test p == inlier_posterior.(Ref(m), sq_r)

            # Float32 constants for a Float32 kernel, type-preserving
            m32 = StudentTMixture{Float32}(m)
            for x in Float32.(sq_r)
                @test posterior_weight(m32, x) isa Float32
                @test posterior_weight(m32, x) ≈ posterior_weight(m, Float64(x)) rtol = 1f-4 atol = 1f-12
            end

            # callable inside a KernelAbstractions kernel (CPU() here; the same code
            # runs on a GPU backend)
            wk = zeros(n)
            _weights_kernel!(KernelAbstractions.CPU())(wk, sq_r, m; ndrange = n)
            KernelAbstractions.synchronize(KernelAbstractions.CPU())
            @test wk == posterior_weight.(Ref(m), sq_r)
        end

        # ν = Inf is the Gaussian inlier, the limit GauInlier also reads: the
        # finite forms are NaN there (Inf·0, Inf/Inf), and the MAP boundary of the
        # E-step's mixture is GauInlier's crossover, at every ν.
        for ν in (Inf, 7.0), dg in (1, 2)
            s2 = 0.3; halfwidth = 4.0
            m = StudentTMixture(ν, s2, log(2 * halfwidth), dg)
            @test all(isfinite, (inlier_posterior(m, 0.5), posterior_weight(m, 0.5),
                                 mixture_nll(m, 0.5), posterior_weight_derivative(m, 0.5)))
            crossover = decision_threshold(GauInlier(s2; nu = ν, d_g = dg), halfwidth)
            @test inlier_posterior(m, crossover^2) ≈ 0.5 rtol = 1e-12
        end
        @test posterior_weight(StudentTMixture(Inf, 0.3, log(8.0), 1), 0.0) ≈
              inlier_posterior(StudentTMixture(Inf, 0.3, log(8.0), 1), 0.0) / 0.3 rtol = 1e-14
    end

    @testset "posterior_weight_derivative: the weight's slope in r²" begin
        for (ν, s2, halfwidth, dg, πin) in ((5.0, 0.04, 2.0, 1, 0.5), (12.0, 1.3, 30.0, 2, 0.3),
                                    (2.0, 2.5, 100.0, 1, 0.9), (Inf, 0.04, 2.0, 1, 0.5))
            m = StudentTMixture(ν, s2, log(2 * halfwidth), dg, πin)
            for x in (0.0, 0.01, 0.2, 1.0, 4.0, 25.0)
                h = 1e-6 * (1 + x)
                fd = (posterior_weight(m, x + h) - posterior_weight(m, max(x - h, 0.0))) /
                     (x + h - max(x - h, 0.0))
                @test posterior_weight_derivative(m, x) ≈ fd rtol = 1e-5
                @test posterior_weight_derivative(m, x) < 0
            end
        end
    end

    @testset "update_inlier_prior!: π ← mean posterior, kept inside (0, 1)" begin
        rng = MersenneTwister(5)
        n = 40
        sq = 0.5 .* rand(rng, n) .^ 2
        sq[1:10] .= 900.0                                   # a quarter far outliers
        pws = PosteriorWeightState(zeros(n), zeros(n), sq, 6.0, 0.1, log(20.0), 1,
                                   JeffreysPrior(), 0.5)
        update_weights!(pws, n)
        # the state's mixture IS the one its E-step evaluated
        m = StudentTMixture(pws)
        @test pws.w == posterior_weight.(Ref(m), sq)
        # foldl, not sum: the M-step adds left to right, and `sum`'s pairwise order
        # differs by an ulp on some Julia versions (1.13.2)
        @test update_inlier_prior!(pws, n) == foldl(+, pws.inlier_posterior) / n
        @test StudentTMixture(pws) != m                       # π moved, and the mixture with it
        @test 0.6 < pws.inlier_prior <= 0.75 + 1e-6            # outliers weigh ~0; inliers at most 1
        fill!(pws.inlier_posterior, 1.0)
        @test update_inlier_prior!(pws, n) == 1 - 1e-3
        fill!(pws.inlier_posterior, 0.0)
        @test update_inlier_prior!(pws, n) == 1e-3
    end
end
