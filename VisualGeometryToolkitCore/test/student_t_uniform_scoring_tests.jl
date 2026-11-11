# =============================================================================
# StudentTUniformScoring (moved from RANSACScoringDoneRight, 2026-11-11): the
# sort-and-sweep's best partition is the best of ALL 2^n partitions under the
# score written out here (the sweep lemma, every scale and label prior); the
# one prefix kernel on the CPU backend is the sweep; the radix sort is
# `sortperm!`; the predictive parametric sweep is the brute-force best of its
# objective; the scorer fulfils the scoring contract without allocating; the
# priors are required (the stack's default is a configuration's).
# =============================================================================

@testitem "student_t_uniform_scoring: the sweep is the best of every partition" begin
    using Test
    using Random: MersenneTwister
    using SpecialFunctions: loggamma, logbeta
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: StudentTUniformScoring, Predictive, JeffreysPrior, IGPrior, FlatLabelPrior,
        ProfiledLabelPrior, BetaLabelPrior, alloc_scoring_workspace, score_residuals!, sweep!, sweep_order,
        penalties, sort_key

    # The score of the partition `I` (a bit mask over the points), written out: the inliers' Gaussian with the scale
    # integrated out against IG(α₀, β₀), the outliers' uniform box, the label prior's term at |I| = k.
    function partition_value(q, I, m, dg, h, prior, label, pen = zeros(length(q)))
        n = length(q); k = count(I)
        k > m || return -Inf
        ν2 = dg * (k - m) / 2
        rss = max(sum(q[I]), eps())
        S = loggamma(ν2 + prior.alpha0) - ν2 * log(2π) - (ν2 + prior.alpha0) * log(rss / 2 + prior.beta0) -
            (n - k) * dg * log(2h) - sum(pen[I]) / 2
        label isa ProfiledLabelPrior && k < n && (S += k * log(k / n) + (n - k) * log((n - k) / n))
        label isa BetaLabelPrior && (S += logbeta(k + label.alpha, n - k + label.beta) - logbeta(label.alpha, label.beta))
        return S
    end
    brute(q, m, dg, h, prior, label, pen = zeros(length(q))) =
        maximum(partition_value(q, BitVector(digits(b; base = 2, pad = length(q))), m, dg, h, prior, label, pen)
                for b in 0:(2^length(q) - 1))

    rng = MersenneTwister(11)
    for prior in (JeffreysPrior(), IGPrior(2.0, 0.5)), label in (FlatLabelPrior(), ProfiledLabelPrior(), BetaLabelPrior(0.5, 0.5)),
        dg in (1, 2), trial in 1:4
        n = 11; m = 2
        r = [rand(rng) < 0.6 ? 0.3randn(rng) : 20 * (2rand(rng) - 1) for _ in 1:n]
        h = 5.0 + 10rand(rng)
        s = StudentTUniformScoring(m, h; codimension = dg, scale_prior = prior, label_prior = label)
        sws = alloc_scoring_workspace(n, dg, s)
        S, k = score_residuals!(sws, zeros(n), s, r, n)
        @test S ≈ brute(r .^ 2, m, dg, h, prior, label) rtol = 1e-12
        # the support is the k smallest residuals, and its score is S
        I = falses(n); I[sweep_order(sws)[1:k]] .= true
        @test partition_value(r .^ 2, I, m, dg, h, prior, label) ≈ S rtol = 1e-12
        @test all(abs.(r[I]) .<= minimum(abs.(r[.!I]); init = Inf))
    end

    # the predictive score: heterogeneous per-point penalties, the exact parametric sweep
    for prior in (JeffreysPrior(), IGPrior(2.0, 0.5)), trial in 1:5
        n = 10; m = 2
        q = [rand(rng) < 0.6 ? 0.1rand(rng) : 50rand(rng) for _ in 1:n]
        s = StudentTUniformScoring{Predictive}(m, 8.0; scale_prior = prior, label_prior = FlatLabelPrior())
        sws = alloc_scoring_workspace(n, 1, s)
        pen = penalties(sws); key = sort_key(sws)
        for i in 1:n
            pen[i] = 2rand(rng); key[i] = q[i]
        end
        S, k = sweep!(sws, s, copy(q), n)
        I = falses(n); I[sweep_order(sws)[1:k]] .= true
        @test S ≈ brute(q, m, 1, 8.0, prior, FlatLabelPrior(), pen) rtol = 1e-12
        @test partition_value(q, I, m, 1, 8.0, prior, FlatLabelPrior(), pen) ≈ S rtol = 1e-12
    end
end

@testitem "student_t_uniform_scoring: one kernel, the radix sort, the contract, required priors" begin
    using Test
    using Random: MersenneTwister, shuffle
    import KernelAbstractions
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, IGPrior, FlatLabelPrior, BetaLabelPrior,
        FixedHalfwidth, OutlierHalfwidth, alloc_scoring_workspace, default_scoring_workspace, score_residuals!, sweep!,
        sweep_order, ig_sweep_kernel!, log_gamma_table, label_prior_table, partition_score, RadixPermBuffers,
        radix_sortperm!, radix_key, test_scoring_contract, scoring_workspace, score!, with_width, outlier_halfwidth,
        width_axis, scale_prior, label_prior, outlier_model, minimal_sample_size, codimension, SCORING_TAGS,
        LABEL_PRIOR_TAGS, check_workspace

    rng = MersenneTwister(12)
    # the radix permutation is sortperm!'s, ties, signed zeros, infinities and NaN included, at both key widths
    for K in (UInt32, UInt64), n in (1, 7, 300)
        v = [rand(rng, (0.0, -0.0, 1.5, Inf, -Inf, NaN, randn(rng), round(randn(rng); digits = 1))) for _ in 1:n]
        perm = zeros(Int, n)
        radix_sortperm!(perm, v, RadixPermBuffers{K}(n), n)
        @test K === UInt64 ? perm == sortperm(v) : issorted(Float32.(v[perm]); lt = isless)
    end
    @test radix_key(-0.0) < radix_key(0.0) && radix_key(Inf) < radix_key(NaN)

    # the sweep is the one kernel: a pool layout of three columns scored on the CPU backend gives each column's sweep
    cols = [abs.(randn(rng, n)) .* (rand(rng, n) .< 0.7) .+ 9 .* rand(rng, n) for n in (40, 40, 40)]
    flat = Float32.(vcat((sort(c .^ 2) for c in cols)...))
    lg = log_gamma_table(40, 1); h = [log(2 * 6.0)]
    out = fill(-Inf, 1, 3, 1); bk = zeros(Int32, 1, 3, 1)
    ig_sweep_kernel!(KernelAbstractions.CPU())(out, flat, Int64[0], Int32[40], Int32[3], lg, h, Int32(2), Int32(1),
                                               JeffreysPrior(), Int32(0), nothing, nothing, bk; ndrange = (1, 3, 1))
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    for j in 1:3
        sws = default_scoring_workspace(40; dg = 1, scale_prior = JeffreysPrior())
        @test (out[1, j, 1], Int(bk[1, j, 1])) ==
              sweep!(sws, Float64.(flat[(j - 1) * 40 + 1:j * 40]), 40; model_dof = 2, log_two_outlier_halfwidth = h[1])
    end
    # partition_score is the formula, in the RSS's precision
    @test partition_score(IGPrior(1.0, 2.0), 0.5, 3.0, 4.0, 0.25) ≈ 0.5 - 3log(2π) - 4log(4.0) - 0.25
    @test partition_score(JeffreysPrior(), 0.5f0, 3.0f0, 4.0f0, 0.25f0) isa Float32

    # the scoring contract, allocation-free (the kernel's launch buffers are the workspace's), every label prior
    r = [rand(rng) < 0.7 ? 0.4randn(rng) : 30rand(rng) for _ in 1:200]
    for label in (FlatLabelPrior(), BetaLabelPrior(2.0, 3.0)), prior in (JeffreysPrior(), IGPrior(2.0, 1.0))
        s = StudentTUniformScoring(2, 25.0; scale_prior = prior, label_prior = label)
        test_scoring_contract(s, r; sample = [3, 9])
        @test scale_prior(s) === prior && label_prior(s) === label && outlier_model(s) === FixedHalfwidth(25.0)
        @test minimal_sample_size(s) == 2 && codimension(s) == 1 && outlier_halfwidth(s) == 25.0
        @test with_width(s, OutlierHalfwidth(4.0)) == StudentTUniformScoring(2, 4.0; scale_prior = prior, label_prior = label)
        @test width_axis(s) === OutlierHalfwidth()
    end
    # a workspace of another prior is refused
    a = StudentTUniformScoring(2, 25.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    b = StudentTUniformScoring(2, 25.0; scale_prior = IGPrior(2.0, 1.0), label_prior = FlatLabelPrior())
    @test_throws ArgumentError check_workspace(b, scoring_workspace(a, 10, 1))
    # the priors are required: nothing is defaulted in code
    @test_throws UndefKeywordError StudentTUniformScoring(2, 25.0)
    @test_throws UndefKeywordError StudentTUniformScoring(2, 25.0; scale_prior = JeffreysPrior())
    @test_throws UndefKeywordError StudentTUniformScoring(2, 25.0; label_prior = FlatLabelPrior())
    @test_throws UndefKeywordError default_scoring_workspace(10)
    @test_throws MethodError BetaLabelPrior()
    # the registries: the scorer's tag, the label priors' tags
    @test SCORING_TAGS["student_t_uniform"] === StudentTUniformScoring
    @test LABEL_PRIOR_TAGS["beta"] === BetaLabelPrior && VgtkCore.tag_of(LABEL_PRIOR_TAGS, FlatLabelPrior()) == "flat"
    @test isempty(label_prior_table(5, FlatLabelPrior())) && label_prior_table(4, BetaLabelPrior(1.0, 1.0))[4] ≈ -log(5)
end
