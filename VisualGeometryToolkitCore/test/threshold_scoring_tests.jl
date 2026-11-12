# =============================================================================
# AbstractThresholdScoring: the one threshold loop. Its score is pinned, bit for
# bit, against the sum written out here; a family of the test's own (a Cauchy
# cost, whose quality is unbounded below and so states no score bound) inherits
# the whole scoring contract from the abstract type; the early exit never
# changes which model a running best selects.
# =============================================================================

@testitem "threshold_scoring: one loop for every family" begin
    using Test
    using Random: MersenneTwister, randperm
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractThresholdScoring, RansacScoring, MsacScoring, ThresholdScoringWorkspace,
        CauchyShape, residual_quality, scoring_workspace, score!, inlier_count, consensus_order, decision_boundary,
        has_decision_boundary, score_bound, test_scoring_contract

    # A family of the test's own: the Cauchy cost as a quality, 1 − ρ/τ², which
    # goes below 0 past the threshold and so has no score bound.
    struct CauchyCostScoring{T} <: AbstractThresholdScoring{T}
        threshold::T
    end
    VgtkCore.residual_quality(::CauchyCostScoring, d2::Float64, w::Float64) =
        1.0 - VgtkCore.ρ(CauchyShape(), d2, w * w) / (w * w)

    # The score written out: Σ quality in index order, a non-finite square an
    # outlier (quality 0), then each sample point whose square is finite corrected
    # to zero residual (a non-finite one stays an outlier: `sample_point_held`).
    held(d) = isfinite(Float64(d * d))
    function reference_score(sc, τ, r, sample)
        q(d) = (d2 = Float64(d * d); isfinite(d2) ? residual_quality(sc, d2, τ) : 0.0)
        s = 0.0
        for d in r
            s += q(d)
        end
        if sample !== nothing
            for i in sample
                held(r[i]) && (s += residual_quality(sc, 0.0, τ) - q(r[i]))
            end
        end
        return s
    end

    rng = MersenneTwister(11)
    n = 80
    for T in (Float32, Float64), τ in (0.3, 1.0, 4.0)
        r = T.(abs.(randn(rng, n)) .* 3)
        r[3] = T(NaN); r[7] = T(Inf); r[11] = T(-Inf)
        r[13] = T === Float32 ? 1.0f30 : 1.0e200           # finite, its square overflows
        r[17] = T(0.0)
        for sc in (RansacScoring(1; threshold = τ), MsacScoring(1; threshold = τ), CauchyCostScoring(τ))
            W = scoring_workspace(sc, n, 1)
            @test W isa ThresholdScoringWorkspace
            for sample in (nothing, [2, 5, 9, 20], [3, 13, 21, 40], [7, 2, 13, 5])   # the last two with a NaN, an Inf and an overflowing point
                scores = zeros(n)
                S = score!(W, sc, scores, r, n; sample)
                @test S === reference_score(sc, τ, r, sample)
                k = inlier_count(sc, W)
                inl = Set(i for i in 1:n if (sample !== nothing && i in sample && held(r[i])) ||
                                               (isfinite(r[i]) && Float64(r[i] * r[i]) < τ^2))
                @test k == length(inl) && Set(consensus_order(sc, W)[1:k]) == inl
                @test sort(consensus_order(sc, W)) == 1:n
            end
        end
    end

    # The inherited contract: the threshold through `decision_boundary`, the trait,
    # no score bound unless one is stated (and so a finite bound never stops it).
    sc = CauchyCostScoring(2.0)
    @test decision_boundary(sc, nothing) === 2.0 && has_decision_boundary(sc)
    @test score_bound(sc) === nothing
    @test score_bound(MsacScoring(1; threshold = 2.0)) === 1.0 === score_bound(RansacScoring(1; threshold = 2.0))
    r = abs.(randn(rng, n)) .* 3
    W = scoring_workspace(sc, n, 1)
    S = score!(W, sc, zeros(n), r, n)
    @test score!(W, sc, zeros(n), r, n; bound = S + 1000.0) === S
    test_scoring_contract(sc, r; sample = [1, 2, 3])
    test_scoring_contract(MsacScoring(1; threshold = 2.0), r; sample = [1, 2, 3])

    # The width-free form implements no scoring contract, as before.
    for F in (RansacScoring, MsacScoring)
        @test F() isa AbstractThresholdScoring{Nothing}
        @test !has_decision_boundary(F()) && score_bound(F()) === nothing
        @test_throws MethodError scoring_workspace(F(), n, 1)
    end

    # The early exit never changes the selection: a running best scored against
    # its own bound picks the full scores' first strict best.
    for sc in (RansacScoring(1; threshold = 1.5), MsacScoring(1; threshold = 1.5))
        R = abs.(randn(rng, n, 200)) .* (1 .+ 4 .* rand(rng, 1, 200))
        smp = reduce(hcat, [randperm(rng, n)[1:4] for _ in 1:200])
        W = scoring_workspace(sc, n, 1)
        full = [score!(W, sc, zeros(n), view(R, :, j), n; sample = view(smp, :, j)) for j in 1:200]
        best, best_j = -Inf, 0
        for j in 1:200
            s = score!(W, sc, zeros(n), view(R, :, j), n; sample = view(smp, :, j), bound = best)
            s > best && (best = s; best_j = j)
        end
        @test best_j == findfirst(==(maximum(full)), full) && best === maximum(full)
    end
end

@testitem "threshold_scoring, student_t_uniform, zero_sample!: a minimal-sample point with a non-finite residual is not held" begin
    # A spurious sample point: a detection the final model does not reach (RadiometricCamera's grid index, a lattice
    # pose leaving it unassigned), its distance Inf. Forced into the consensus set it made the set's scale Inf
    # (`consensus_s2`: the RSS of its plain residuals) and counted a dof it does not have.
    using Test
    using Random: MersenneTwister
    using VisualGeometryToolkitCore: RansacScoring, MsacScoring, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
        scoring_workspace, score!, inlier_count, consensus_order, consensus_s2, zero_sample!, sample_point_held
    rng = MersenneTwister(4)
    n = 60
    r = 0.3 .* abs.(randn(rng, n)); r[[17, 41]] .= Inf      # two points no model reaches; 17 drawn into the sample
    sample = [3, 17, 29]
    for sc in (RansacScoring(1; threshold = 8.0), MsacScoring(1; threshold = 8.0),
               StudentTUniformScoring(2, 25.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()))
        W = scoring_workspace(sc, n, 1)
        score!(W, sc, zeros(n), r, n; sample)
        k = inlier_count(sc, W)
        set = consensus_order(sc, W)[1:k]
        @test k == n - 2 && 17 ∉ set && 41 ∉ set && 3 in set && 29 in set
        s2 = consensus_s2(sc, r, consensus_order(sc, W), k; model_dof = 2, d_g = 1)
        @test isfinite(s2) && sqrt(s2) < 1.0
    end
    # the one rule, and zero_sample! by it: a held point zeroed, a non-finite one left (an invalid index still throws)
    @test sample_point_held(0.4) && !sample_point_held(Inf) && !sample_point_held(NaN)
    v = [1.0, Inf, 2.0, NaN]
    @test isequal(zero_sample!(copy(v), [1, 2, 4]), [0.0, Inf, 2.0, NaN])
    @test_throws BoundsError zero_sample!(copy(v), [5])
end
