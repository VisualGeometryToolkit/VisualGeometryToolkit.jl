# =============================================================================
# parametric_sweep_tests.jl — the generalized-sweep-lemma gates
# (uncertain_evidence 35ae25c §"The generalized sweep").
#
# b. PARAMETRIC SWEEP EXACTNESS: brute-force oracle over ALL subsets on
#    small heterogeneous instances (Jeffreys + IG priors, with and without the
#    profiled-γ prior table), including integer-grid instances whose crossings
#    COINCIDE — the kinetic walk's re-sort branch — plus property tests
#    (crossing bound, homoscedastic degradation), and a pin against the
#    retired re-sort-per-interval implementation on a fixed seed grid.
# a. FAST-PATH BIT-IDENTITY: equal gains must take the historical
#    single-sort path and reproduce it bit-for-bit.
# c. SCALE-EQUIVARIANCE PIN (the multiplicative-convention motivation):
#    scaling all residuals by c leaves the selected partition invariant and
#    shifts the profiled scale by c. This holds because q_i and the gains
#    are s-free under Ṽ_i = s²·V_i; under the OLD additive reading
#    (Ṽ = s² + leverage frozen at a reference scale) q_i and |Ṽ_i| would be
#    s-dependent, the lemma separation fails, and this pin CANNOT hold —
#    that latent violation is the motivation for the convention change.
# =============================================================================

@testitem "parametric_sweep" begin
    using Test, Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: default_scoring_workspace, partition_score, sweep!,
        log_gamma_table, _alpha0, label_prior_table
    using VisualGeometryToolkitCore: IGPrior, JeffreysPrior, FlatLabelPrior, ProfiledLabelPrior

    # Reference (brute-force) score of an explicit support I — the same formula
    # sweep! evaluates, applied to an arbitrary subset.
    function _brute_best(q, pen, n, m, dg, log_two_outlier_halfwidth, prior::IGPrior, prior_table = nothing)
        lg = log_gamma_table(n, dg; alpha0 = prior.alpha0)
        best = -Inf; bestI = Int[]
        for mask in 1:(2^n - 1)                       # exhaustive subset enumeration
            k = count_ones(mask)
            k <= m && continue
            RSS = 0.0; P = 0.0
            for i in 1:n
                if (mask >> (i - 1)) & 1 == 1
                    RSS += q[i]; P += pen[i]
                end
            end
            RSS = max(RSS, eps())
            ν½ = dg * (k - m) / 2
            S = partition_score(prior, lg[k - m], ν½, RSS, (n - k) * dg * log_two_outlier_halfwidth) - P / 2
            prior_table !== nothing && (S += prior_table[k])
            if S > best
                best = S
                bestI = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
            end
        end
        return best, bestI
    end

    # The score of one explicit support I (the brute force's formula).
    function _subset_score(q, pen, I, n, m, dg, log_two_outlier_halfwidth, prior::IGPrior, prior_table)
        k = length(I)
        lg = log_gamma_table(n, dg; alpha0 = prior.alpha0)
        S = partition_score(prior, lg[k - m], dg * (k - m) / 2, max(sum(q[I]), eps()),
                         (n - k) * dg * log_two_outlier_halfwidth) - sum(pen[I]) / 2
        return prior_table === nothing ? S : S + prior_table[k]
    end

    @testset "parametric sweep == brute force (oracle, heterogeneous gains)" begin
        for seed in 1:200
            rng = MersenneTwister(seed)
            n = rand(rng, 5:12)
            m = rand(rng, 1:3)
            dg = rand(rng, 1:2)
            log_two_outlier_halfwidth = log(2 * (0.5 + rand(rng)))
            q = rand(rng, n) .^ 2 .* 10 .+ 1e-6
            # heterogeneous penalties with deliberate CONFLICTED pairs (large gain
            # with large residual): correlate some penalties negatively with q
            pen = 0.5 .* randn(rng, n) .+ (rand(rng, n) .< 0.5) .* (-0.3 .* q)
            prior = seed % 2 == 0 ? JeffreysPrior() : IGPrior(1.3, 0.7)
            sws = default_scoring_workspace(n; dg, scale_prior = prior, label_prior = FlatLabelPrior())
            S, k = sweep!(sws.perm, sws.lg_table, m, log_two_outlier_halfwidth, dg, q, n;
                          scratch = sws.scratch, penalties = pen, scale_prior = prior)
            Sb, Ib = _brute_best(q, pen, n, m, dg, log_two_outlier_halfwidth, prior)
            @test isapprox(S, Sb; rtol = 0, atol = 1e-9)
            @test k == length(Ib)
            @test sort(sws.perm[1:k]) == sort(Ib)      # the materialized support IS the argmax
        end
    end

    @testset "kinetic walk: coincident crossings and the profiled prior == brute force" begin
        # Residuals and penalties on small integer grids make many pairwise
        # crossings μ_ij COINCIDE exactly (and repeat points outright), so groups
        # of several crossings — the walk's re-sort branch — occur alongside lone
        # adjacent swaps. The profiled-γ table enters every prefix score.
        for seed in 1:150
            rng = MersenneTwister(1000 + seed)
            n = rand(rng, 5:12)
            m = rand(rng, 1:3)
            dg = rand(rng, 1:2)
            log_two_outlier_halfwidth = log(2 * (0.5 + rand(rng)))
            q = Float64.(rand(rng, 1:4, n)) .+ (seed % 3 == 0 ? 0.0 : 1e-3 .* rand(rng, n))
            pen = 0.5 .* Float64.(rand(rng, -2:2, n))
            prior = seed % 2 == 0 ? JeffreysPrior() : IGPrior(1.3, 0.7)
            pt = seed % 4 < 2 ? label_prior_table(n, ProfiledLabelPrior()) : nothing
            sws = default_scoring_workspace(n; dg, scale_prior = prior, label_prior = FlatLabelPrior())
            S, k = sweep!(sws.perm, sws.lg_table, m, log_two_outlier_halfwidth, dg, q, n;
                          scratch = sws.scratch, penalties = pen, prior_table = pt,
                          scale_prior = prior)
            Sb, Ib = _brute_best(q, pen, n, m, dg, log_two_outlier_halfwidth, prior, pt)
            @test isapprox(S, Sb; rtol = 0, atol = 1e-9)
            # The materialized support attains the optimum. (Integer grids can tie
            # two subsets exactly, and either is then an argmax, so the support is
            # pinned by its score rather than as a set.)
            @test isapprox(_subset_score(q, pen, sws.perm[1:k], n, m, dg, log_two_outlier_halfwidth, prior, pt), Sb;
                           rtol = 0, atol = 1e-9)
        end
    end

    # The retired implementation (re-sort and re-score every prefix at every
    # interval's representative μ), verbatim — the reference the kinetic walk
    # replaced. The two enumerate the same candidate family; they may differ only
    # in the summation order of carried-forward prefix sums.
    function _retired_parametric_sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                                model_dof::Int, log_two_outlier_halfwidth::Float64, dg::Int,
                                scores::AbstractVector{T}, penalties::AbstractVector{Float64},
                                prior_table::Union{Nothing,AbstractVector{Float64}},
                                scale_prior::IGPrior, n::Int) where T
        # all positive pairwise crossings (conflicted pairs only)
        cross = Float64[]
        @inbounds for i in 1:(n - 1), j in (i + 1):n
            dq = Float64(scores[i]) - Float64(scores[j])
            dq == 0.0 && continue
            μ = ((penalties[j] - penalties[i]) / 2) / dq     # (g_i - g_j)/(q_i - q_j)
            μ > 0.0 && push!(cross, μ)
        end
        sort!(cross)
        # representative μ per ordering interval: below the first crossing,
        # midpoints of consecutive distinct crossings, above the last
        μs = Float64[]
        if isempty(cross)
            push!(μs, 1.0)
        else
            push!(μs, cross[1] / 2)
            prev = cross[1]
            @inbounds for c in cross
                if c > prev * (1 + 1e-13) + 1e-300
                    push!(μs, (prev + c) / 2)
                    prev = c
                end
            end
            push!(μs, cross[end] * 2 + 1.0)
        end

        m = model_dof
        dg_f = Float64(dg)
        ord = Vector{Int}(undef, n)
        bestord = Vector{Int}(undef, n)
        best_S = -Inf; best_k = 0
        for μ in μs
            # descending t(μ); deterministic ties by (ascending q, index)
            sortperm!(ord, 1:n;
                      by = i -> (-( -penalties[i] / 2 - μ * Float64(scores[i]) ),
                                 Float64(scores[i]), i))
            RSS = 0.0; pens = 0.0   # Float64 accumulation regardless of storage eltype
            @inbounds for k in 1:n
                idx = ord[k]
                RSS += Float64(scores[idx])
                pens += Float64(penalties[idx])
                k <= m && continue
                RSSc = max(RSS, eps(Float64))
                ν½ = dg_f * (k - m) / 2
                S = partition_score(scale_prior, lg_table[k - m], ν½, RSSc,
                                 (n - k) * dg_f * log_two_outlier_halfwidth) - pens / 2
                prior_table !== nothing && (S += Float64(prior_table[k]))
                if S > best_S
                    best_S = S; best_k = k
                    copyto!(bestord, ord)
                end
            end
        end
        # perm[1:k*] IS the selected support (ConsensusSet/mask contract)
        if length(perm) == n
            copyto!(perm, bestord)
        else
            @inbounds for k in 1:n; perm[k] = bestord[k]; end
        end
        return best_S, best_k
    end


    @testset "kinetic walk == the retired re-sort-per-interval sweep (seed grid)" begin
        # predictive-scoring-like instances: χ²_dg inliers, wide outliers, penalties
        # log V_i ≥ 0, the m sample points exact (q = 0, penalty 0)
        function instance(rng, n, dg; m = 4)
            q = [rand(rng) < 0.7 ? sum(abs2, randn(rng, dg)) : 400 * rand(rng) for _ in 1:n]
            pen = [log1p(0.05 * randexp(rng) * (1 + 3 * rand(rng))) for _ in 1:n]
            q[1:m] .= 0.0; pen[1:m] .= 0.0
            return q, pen
        end
        m = 4
        for seed in 1:8, n in (20, 60), dg in (1, 2),
            prior in (JeffreysPrior(), IGPrior(1.3, 0.7)), profiled in (false, true)
            rng = MersenneTwister(seed)
            q, pen = instance(rng, n, dg)
            log_two_outlier_halfwidth = log(2 * (5 + 30 * rand(rng)))
            pt = profiled ? label_prior_table(n, ProfiledLabelPrior()) : nothing
            lg = log_gamma_table(n, dg; alpha0 = prior.alpha0)
            perm_new = zeros(Int, n); perm_old = zeros(Int, n)
            S, k = sweep!(perm_new, copy(lg), m, log_two_outlier_halfwidth, dg, q, n;
                          penalties = pen, prior_table = pt, scale_prior = prior)
            So, ko = _retired_parametric_sweep!(perm_old, copy(lg), m, log_two_outlier_halfwidth, dg, q, pen,
                                                pt, prior, n)
            @test k == ko
            @test sort(perm_new[1:k]) == sort(perm_old[1:ko])
            @test S ≈ So rtol = 1e-14
        end
    end

    @testset "properties: crossing bound + homoscedastic degradation" begin
        rng = MersenneTwister(42)
        n = 40; m = 2; dg = 1; log_two_outlier_halfwidth = log(0.4)
        q = rand(rng, n) .* 5
        # homoscedastic degradation: constant penalties reduce to the prefix
        # family of the single q-sort (bit-identical fast path)
        penc = fill(1.234, n)
        sws1 = default_scoring_workspace(n; dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
        sws2 = default_scoring_workspace(n; dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
        S1, k1 = sweep!(sws1.perm, sws1.lg_table, m, log_two_outlier_halfwidth, dg, q, n;
                        scratch = sws1.scratch, penalties = penc, scale_prior = JeffreysPrior())
        q2 = copy(q)
        S2, k2 = sweep!(sws2.perm, sws2.lg_table, m, log_two_outlier_halfwidth, dg, q2, n;
                        scratch = sws2.scratch, penalties = penc, sort_key = copy(q), scale_prior = JeffreysPrior())
        @test S1 === S2 && k1 === k2 && sws1.perm == sws2.perm
        # crossing bound: the enumeration terminates within C(n,2) events and its
        # optimum dominates every single-μ prefix family (spot check μ grid)
        pen = randn(rng, n)
        sws = default_scoring_workspace(n; dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
        S, k = sweep!(sws.perm, sws.lg_table, m, log_two_outlier_halfwidth, dg, q, n;
                      scratch = sws.scratch, penalties = pen, scale_prior = JeffreysPrior())
        lg = log_gamma_table(n, dg)
        for μ in (1e-3, 0.1, 1.0, 10.0, 1e3)
            ordμ = sort(1:n; by = i -> -(-pen[i] / 2 - μ * q[i]))
            RSS = 0.0; pens = 0.0
            for kk in 1:n
                RSS += q[ordμ[kk]]; pens += pen[ordμ[kk]]
                kk <= m && continue
                ν½ = dg * (kk - m) / 2
                Sμ = partition_score(IGPrior(), lg[kk - m], ν½, max(RSS, eps()),
                                  (n - kk) * dg * log_two_outlier_halfwidth) - pens / 2
                @test S >= Sμ - 1e-9
            end
        end
    end

    @testset "scale-equivariance pin (multiplicative convention)" begin
        rng = MersenneTwister(7)
        n = 30; m = 3; dg = 1; log_two_outlier_halfwidth = log(1.7)
        q = rand(rng, n) .* 3 .+ 1e-4          # s-free unit-scale q_i = r_i' V_i^{-1} r_i
        pen = randn(rng, n) .* 0.7             # s-free log|V_i|
        for c in (0.25, 1.0, 8.0)
            # the scale-family statement: residuals scale by c AND the raw-domain
            # floor half-width co-scales (by c) — then the total score shifts
            # by the k-independent constant −(nd − n_θ)·log c and the argmax is
            # invariant. (With a FIXED floor the −ν_k·log c tilt legitimately
            # moves the boundary — that is a different experiment, not this pin.)
            sws = default_scoring_workspace(n; dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
            S, k = sweep!(sws.perm, sws.lg_table, m, log_two_outlier_halfwidth + log(c), dg, c^2 .* q, n;
                          scratch = sws.scratch, penalties = pen, scale_prior = JeffreysPrior())
            sws0 = default_scoring_workspace(n; dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
            S0, k0 = sweep!(sws0.perm, sws0.lg_table, m, log_two_outlier_halfwidth, dg, q, n;
                            scratch = sws0.scratch, penalties = pen, scale_prior = JeffreysPrior())
            @test k == k0                                   # selected partition invariant
            @test sort(sws.perm[1:k]) == sort(sws0.perm[1:k0])
            # profiled scale shifts by exactly c: ŝ² = Q_I/ν_k scales by c²
            ν = dg * (k - m)
            @test sqrt(sum(c^2 .* q[sws.perm[1:k]]) / ν) ≈ c * sqrt(sum(q[sws0.perm[1:k0]]) / ν) rtol = 1e-12
        end
    end

    @testset "the log Γ table: lg[k] = log Γ(k·d_g/2 + α₀)" begin
        lg = log_gamma_table(100, 1)
        @test length(lg) == 100
        @test lg[1] ≈ 0.5 * log(π) atol = 1.0e-12   # log Γ(1/2)
        @test lg[2] ≈ 0.0 atol = 1.0e-12             # log Γ(1)
        @test lg[4] ≈ 0.0 atol = 1.0e-12             # log Γ(2)
        @test lg[6] ≈ log(2.0) atol = 1.0e-12        # log Γ(3)
        @test log_gamma_table(3, 2)[3] ≈ log(2.0) atol = 1.0e-12             # d_g = 2: log Γ(3)
        @test log_gamma_table(2, 1; alpha0 = 1.0)[2] ≈ 0.0 atol = 1.0e-12    # log Γ(1 + 1)
    end
end

# The prefix sweep is one kernel (the owner's ruling on the twin audit's V2): `sweep!` launches `ig_sweep_kernel!` on the
# CPU backend over its sorted column, with the per-point penalties, the prior table and the best k the kernel now carries.
# Its (S*, k*) are the plain prefix scan's, bit for bit (the scan is the test's oracle).
@testitem "sweep!: the one prefix-sweep kernel, bit for bit the plain scan" begin
    using Test, Random
    using KernelAbstractions: KernelAbstractions
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sweep!, ig_sweep_kernel!, IGPrior, log_gamma_table, _table_partition_score
    rng = Xoshiro(23)
    function scan(lg, m, l2h, dg, s, n, pen, pt, prior)
        p = sortperm(s[1:n]); RSS = 0.0; ps = 0.0; bS = -Inf; bk = 0
        for k in 1:n
            RSS += Float64(s[p[k]]); pen === nothing || (ps += Float64(pen[p[k]]))
            k <= m && continue
            S = _table_partition_score(prior, lg, RSS, k, m, Float64(dg), n, l2h)
            pen === nothing || (S -= ps / 2)
            pt === nothing || (S += pt[k])
            S > bS && (bS = S; bk = k)
        end
        return bS, bk
    end
    for n in (3, 8, 120, 900), T in (Float32, Float64), pen in (false, true), pt in (false, true)
        s = T.(abs2.(randn(rng, n) .* ifelse.(rand(rng, n) .< 0.6, 0.3, 8.0)))
        pens = pen ? fill(0.4, n) : nothing                   # equal gains: the sort-and-sweep path
        prior = pt ? randn(rng, n) : nothing
        lg = log_gamma_table(n, 1; alpha0 = IGPrior().alpha0)
        @test sweep!(Vector{Int}(undef, n), lg, 7, log(9.0), 1, s, n; penalties = pens, prior_table = prior, scale_prior = IGPrior()) ===
              scan(lg, 7, log(9.0), 1, s, n, pens, prior, IGPrior())
    end
    # the kernel's best k: written where asked, the grid's 0-size array left alone
    s = sort(abs2.(randn(rng, 50))); lg = log_gamma_table(50, 1; alpha0 = IGPrior().alpha0)
    best = fill(-Inf, 1, 1, 1); k = zeros(Int32, 1, 1, 1); cpu = KernelAbstractions.CPU()
    ig_sweep_kernel!(cpu, (1, 1, 1))(best, s, Int64[0], Int32[50], Int32[1], lg, [log(9.0)], Int32(7), Int32(1), IGPrior(),
                                     Int32(0), nothing, nothing, k; ndrange = (1, 1, 1))
    @test (best[1], Int(k[1])) === scan(lg, 7, log(9.0), 1, s, 50, nothing, nothing, IGPrior())
end

# The prefix sweep's loop body is device-callable (the owner's ruling on the BlobBoards duplication audit, ask 5): a
# kernel that holds its sorted column in local memory (BlobBoards' per-blob sweep: one workgroup per blob, the column
# sorted in `@localmem`) calls `prefix_sweep` itself. In Float64 it is the grid kernel's and `sweep!`'s (S*, k*), bit for
# bit; in Float32 it sweeps in the kernel's own precision; the same kernel runs on the CPU backend and on a GPU.
@testitem "prefix_sweep: the one sweep's loop body inside a kernel, on every backend" begin
    using Test, Random
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @Const
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: prefix_sweep, sweep!, IGPrior, log_gamma_table
    @kernel function local_sweep!(S_out, k_out, @Const(sq), @Const(cnt), @Const(lg), @Const(l2h), model_dof::Int32)
        blob = @index(Group, Linear)
        tid = @index(Local, Linear)
        smem = @localmem eltype(S_out) (64,)
        n = Int(cnt[blob])
        tid <= n && (smem[tid] = sq[tid, blob])
        @synchronize()
        if tid == 1
            S, k = prefix_sweep(eltype(S_out), smem, 0, n, lg, model_dof, 1, IGPrior(0.0, 0.0), l2h[blob], nothing, nothing)
            S_out[blob] = S
            k_out[blob] = Int32(k)
        end
    end
    rng = Xoshiro(5)
    nb = 6; cnt = Int32[64, 40, 17, 9, 3, 2]; m = 2
    sq = zeros(Float64, 64, nb)
    for b in 1:nb
        n = cnt[b]
        sq[1:n, b] = sort(abs2.(randn(rng, n) .* ifelse.(rand(rng, n) .< 0.7, 0.5, 9.0)))
    end
    l2h = log.(2 .* (1.0 .+ rand(rng, nb)))
    lg = log_gamma_table(64, 1; alpha0 = 0.0)
    function launch(backend, T)
        up(x) = (d = KA.allocate(backend, eltype(x), size(x)); copyto!(d, x); d)
        S = KA.allocate(backend, T, nb); k = KA.allocate(backend, Int32, nb)
        local_sweep!(backend, 64)(S, k, up(T.(sq)), up(cnt), up(T.(lg)), up(T.(l2h)), Int32(m); ndrange = 64nb)
        KA.synchronize(backend)
        return Array(S), Int.(Array(k))
    end
    S64, k64 = launch(KA.CPU(), Float64)
    for b in 1:nb
        n = Int(cnt[b])
        if n > m
            @test (S64[b], k64[b]) === sweep!(Vector{Int}(undef, n), lg[1:n], m, l2h[b], 1, sq[1:n, b], n; scale_prior = IGPrior(0.0, 0.0))
        else
            @test (S64[b], k64[b]) === (-Inf, 0)              # no partition above the model's dof
        end
    end
    S32, k32 = launch(KA.CPU(), Float32)
    @test eltype(S32) === Float32
    @test k32 == k64 && isapprox(S32, S64; rtol = 1.0e-4)
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
        Sg, kg = Base.invokelatest(launch, gpu, Float32)
        @test kg == k32 && isapprox(Sg, S32; rtol = 1.0e-5)
        Sg64, kg64 = Base.invokelatest(launch, gpu, Float64)
        @test kg64 == k64 && isapprox(Sg64, S64; rtol = 1.0e-12)
    end
end
