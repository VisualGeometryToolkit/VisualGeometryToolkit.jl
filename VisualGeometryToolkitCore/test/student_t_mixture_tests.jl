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
        posterior_weight_derivative, soft_posterior_weights!, GauInlier, decision_boundary,
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
            crossover = decision_boundary(GauInlier(s2; nu = ν, d_g = dg), halfwidth)
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

# A non-finite squared residual (Inf, NaN) — an out-of-reach point, an infinite
# Sampson distance, a residual that turned NaN — is the outlier's: inlier
# posterior 0 and weight 0, so it is no inlier, and nothing read from the E-step
# (ŝ², ν, π, the cost Σwᵢrᵢ²) sees it. An infinite residual gives it by the
# formulas; a NaN one by the kernel's guard. A finite residual's pair is the
# unguarded formula's, bit for bit.
@testitem "student_t_mixture: non-finite residuals" begin
    using Test, Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: StudentTMixture, inlier_posterior, posterior_weight, soft_posterior_weights!,
        PosteriorWeightState, JeffreysPrior, update_weights!, update_scale!, update_nu!, update_inlier_prior!,
        posterior_weights!, posterior_inlier_probs!, median_seeded_posterior, is_map_inlier, mixture_nll
    const VGC = VisualGeometryToolkitCore

    mixtures = [StudentTMixture(ν, s2, log(2 * hw), dg, πin)
                for (ν, s2, hw, dg, πin) in ((5.0, 0.04, 2.0, 1, 0.5), (12.0, 1.3, 30.0, 2, 0.3),
                                             (1e4, 0.01, 0.5, 2, 0.8), (Inf, 0.04, 2.0, 1, 0.5))]

    @testset "the kernel: (0, 0) for Inf and NaN, no inlier" begin
        for m in mixtures, T in (Float64, Float32)
            mT = StudentTMixture{T}(m)
            for x in (T(Inf), T(NaN), -T(NaN))
                @test VGC._posterior_pair(mT, x) === (zero(T), zero(T))
                @test inlier_posterior(mT, x) === zero(T)
                @test posterior_weight(mT, x) === zero(T)
                @test !is_map_inlier(inlier_posterior(mT, x))
            end
        end
        # unconditionally: a mixture whose constants make the formulas NaN at Inf
        # (an inlier prior of 1, log_odds_out = −Inf; an infinite scale)
        for m in (StudentTMixture(5.0, 0.04, log(4.0), 1, 1.0), StudentTMixture(5.0, Inf, log(4.0), 1)), x in (Inf, NaN)
            @test VGC._posterior_pair(m, x) === (0.0, 0.0)
        end
    end

    # the pair as the kernel wrote it before the guard, from its unchanged parts
    _unguarded(m::StudentTMixture{T}, x) where {T} =
        (p = one(T) / (one(T) + exp(m.log_odds_out - VGC._log_inlier_density(m, x)));
         (p, p * VGC._student_t_weight(m.νdg, m.νs2, m.s2, x)))

    @testset "finite residuals: bit for bit the unguarded kernel" begin
        rng = MersenneTwister(11)
        for m in mixtures, T in (Float64, Float32)
            mT = StudentTMixture{T}(m)
            xs = T[0, nextfloat(zero(T)), floatmin(T), eps(T), 1, floatmax(T), prevfloat(floatmax(T))]
            append!(xs, T.(exp10.(range(-12, 12; length = 400))), rand(rng, T, 400) .* 50)
            @test all(x -> VGC._posterior_pair(mT, x) === _unguarded(mT, x), xs)
            @test all(x -> mixture_nll(mT, x) ===
                           -(mT.log_mix_out + VGC.log1pexp(VGC._log_inlier_density(mT, x) - mT.log_odds_out)), xs)
        end
    end

    # The E-step over a buffer and every M-step block read from it: a NaN row is an
    # Inf row, bit for bit (the zero-weight term is exactly 0 in Σwᵢrᵢ², and ν's
    # sums skip it), and every other row is its finite value.
    @testset "the E-step and the M-step blocks: a NaN row is an Inf row" begin
        rng = MersenneTwister(3)
        n = 30
        base = 0.2 .* rand(rng, n) .^ 2; base[1:6] .= 400.0
        for bad in (NaN, Inf), dg in (1, 2)
            sq = copy(base); sq[[4, 17]] .= bad
            w = fill(-1.0, n); p = fill(-1.0, n)
            soft_posterior_weights!(w, p, sq, 6.0, 0.05, log(20.0), n, dg)
            @test w[[4, 17]] == [0.0, 0.0] && p[[4, 17]] == [0.0, 0.0]
            m = StudentTMixture(6.0, 0.05, log(20.0), dg)
            keep = setdiff(1:n, [4, 17])
            @test w[keep] == posterior_weight.(Ref(m), sq[keep])
            @test p[keep] == inlier_posterior.(Ref(m), sq[keep])
        end
        sq_nan = copy(base); sq_nan[[4, 17]] .= NaN
        sq_inf = copy(base); sq_inf[[4, 17]] .= Inf
        for dg in (1, 2)
            states = map((sq_nan, sq_inf)) do s
                pws = PosteriorWeightState(zeros(n), zeros(n), copy(s), 6.0, 0.05, log(20.0), dg, JeffreysPrior(), 0.5)
                update_weights!(pws, n)
                (update_scale!(pws, n), update_nu!(pws, n), update_inlier_prior!(pws, n), copy(pws.w), copy(pws.inlier_posterior))
            end
            @test all(isfinite, states[1][1:3])
            @test states[1] == states[2]
        end
    end

    # The partition mixture's weights: a datum outside the partition with a NaN
    # residual weighs 0, as an Inf one does.
    @testset "posterior_weights!, posterior_inlier_probs!" begin
        sq = [0.01, 0.04, 0.02, 0.03, 0.05, 0.015, NaN, Inf, 9.0]
        order = [1, 6, 3, 4, 2, 5, 9, 8, 7]
        for f! in (posterior_weights!, posterior_inlier_probs!)
            w = f!(fill(-1.0, 9), sq, order, 6; model_dof = 2, log_two_outlier_halfwidth = log(20.0),
                   prior = JeffreysPrior())
            @test w[7] === 0.0 && w[8] === 0.0
            @test all(isfinite, w) && all(>(0), w[[1:6; 9]])
        end
    end

    # The median seed (a fit with no partition): the median runs over the residuals
    # that are not NaN, so NaN residuals neither move the seed scale nor make it
    # NaN, even when they are most of the data, and the E-step gives them 0. An Inf
    # residual stays in the median (an outlier's large norm). All NaN: ŝ² is NaN.
    @testset "median_seeded_posterior: NaN residuals" begin
        rng = MersenneTwister(8)
        sq = 0.1 .* rand(rng, 21) .^ 2
        kw = (; outlier_halfwidth = 10.0, ν = 50.0, prior = JeffreysPrior(), codimension = 2)
        ref = median_seeded_posterior(sq, 21; kw...)
        for k in (1, 30)                                  # one NaN; NaN the majority
            sq_nan = [sq[1:10]; fill(NaN, k); sq[11:21]]
            pws = median_seeded_posterior(sq_nan, 21 + k; kw...)
            bad = 11:(10 + k); good = setdiff(1:(21 + k), bad)
            @test pws.s_hat_sq === ref.s_hat_sq
            @test all(iszero, pws.w[bad]) && all(iszero, pws.inlier_posterior[bad])
            @test pws.w[good] == ref.w && pws.inlier_posterior[good] == ref.inlier_posterior
        end
        sq_inf = [sq[1:10]; Inf; sq[11:21]]
        pinf = median_seeded_posterior(sq_inf, 22; kw...)
        @test isfinite(pinf.s_hat_sq) && pinf.w[11] === 0.0
        @test pinf.s_hat_sq == VGC._mad_scale_norms(; norms = sqrt.(sq_inf), codimension = 2)^2
        @test isnan(median_seeded_posterior(fill(NaN, 5), 5; kw...).s_hat_sq)
    end
end
