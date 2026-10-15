# =============================================================================
# evaluation/engine/budget.jl — the sample-budget selection + LO-checkpoint engine.
# Per pair: walk the pool's hypothesis stream, have every selector
# track its running best, record it at batch checkpoints, and refit it with the
# configured LO strategy at LO checkpoints. Per-pair geometry flows through ONE
# PairCtx (build_pair_ctx) shared by selection scoring and every LO refit;
# selection reads the pool's residual matrices, the refits the light records.
# =============================================================================

"""
    SampleBudgetConfig

Configuration for the sample-budget selection protocol.

# Fields (the schedule and the LO are REQUIRED: a sample-budget experiment
# states its protocol of record, e.g. a paper's sample_budget_protocol.toml)
- `sample_cap::Int` — total minimal samples drawn per pair
- `batch_size::Int` — selection (non-LO) checkpoint stride
- `lo_every::Int`   — LO checkpoint stride (after the first batch)
- `lo_strategy::AbstractLocalOptimization` — LO method
- `budget_unit::Symbol` — budget axis: `:models` counts essential
  matrices inspected (Shekhovtsov's unit); `:samples` counts minimal subsets.

# Checkpoint schedule (both curves share one cumulative-sample x-axis)
- **Selection** (running-best, no LO): every `batch_size` — `100, 200, …, cap`.
- **LO** (refit the running best): after the first batch, then the `lo_every`
  grid — `{batch_size} ∪ {lo_every, 2·lo_every, …, cap}` = `100, 500, 1000, …`.
"""
Base.@kwdef struct SampleBudgetConfig
    sample_cap::Int
    batch_size::Int
    lo_every::Int
    lo_strategy::AbstractLocalOptimization
    budget_unit::Symbol
    # PROTOCOL: the tuned refinement native of an arm whose width parameter is
    # native (outlier half-width, loss width), from the tuning sidecar
    # (deployed_arm_settings). `nothing` ⇒ the arm's refit runs at the SELECTION
    # native, which is only correct for partition-anchored scale-multiple arms
    # (width lives in the LO object).
    nat_ref::Union{Nothing,Float64} = nothing
    # AXIS the incoming native lives on (a `NativeAxis`); `lo_error!` routes it
    # through `consumed_native` to the refiner's consumed axis.
    # `OutlierHalfwidth()` (default) is identity for the IRLS-family consumers
    # (Stu-tU/Welsch/TLS-HQ/GM-HQ and, since the 2026-07-18 port, the GNC-* arms)
    # and outlier half-width → loss width for the truncated consumers (TLS-LM/LZ,
    # +TLSadapt), exactly as the basin ladder and the tuning feed them.
    # `LossWidth()` is for IG-seeded truncated arms whose sidecar ref is already
    # a loss width (TLS-LM/LZ). An arm's kind is its axis (`deployed_budget_settings`).
    nat_axis::WidthAxis = OutlierHalfwidth()
    # `budget_unit` is a CHOICE, validated here and not at the one `===` that
    # reads it: `use_models = cfg.budget_unit === :models` (below) made
    # `budget_unit = :modles` run the `:samples` protocol to completion and
    # report success, and the value arrives from a TOML string in production
    # (`accv2026/configs/sample_budget_protocol.toml`). `nat_axis` must be a
    # native axis: a scale multiple has no native to route.
    function SampleBudgetConfig(sample_cap, batch_size, lo_every, lo_strategy, budget_unit,
                                nat_ref, nat_axis)
        budget_unit in (:models, :samples) ||
            throw(ArgumentError("SampleBudgetConfig: budget_unit must be :models or :samples; " *
                                "got :$budget_unit"))
        nat_axis isa NativeAxis ||
            throw(ArgumentError("SampleBudgetConfig: nat_axis must be a native axis, OutlierHalfwidth() " *
                                "or LossWidth(); got $nat_axis"))
        return new(sample_cap, batch_size, lo_every, lo_strategy, budget_unit, nat_ref, nat_axis)
    end
end

"Number of selection checkpoints of `cfg`: `sample_cap ÷ batch_size`."
n_batches(cfg::SampleBudgetConfig) = cfg.sample_cap ÷ cfg.batch_size

"The cumulative sample counts of the selection checkpoints: `batch_size:batch_size:sample_cap`."
batch_sample_counts(cfg::SampleBudgetConfig) =
    collect(cfg.batch_size:cfg.batch_size:cfg.sample_cap)

# `sort!(unique!(...))` collapses coinciding points.
"""
    lo_sample_counts(cfg::SampleBudgetConfig) -> Vector{Int}

The cumulative sample counts of the LO checkpoints: an early point after the
FIRST batch, then the regular `lo_every` grid up to `sample_cap`.
"""
lo_sample_counts(cfg::SampleBudgetConfig) =
    sort!(unique!(vcat(cfg.batch_size, collect(cfg.lo_every:cfg.lo_every:cfg.sample_cap))))

"Number of LO checkpoints of `cfg` (see [`lo_sample_counts`](@ref))."
n_los(cfg::SampleBudgetConfig) = length(lo_sample_counts(cfg))

"""
    SampleBudgetResults

Per-pair error curves for each method column, plus LO-refined error at each LO
checkpoint and the LO wall-clock accounting (see [`summarize_lo_timing`](@ref)).

# Fields
- `methods::Vector{Symbol}` — method labels (length M)
- `sample_counts::Vector{Int}` — cumulative-sample axis for batch curves (B)
- `lo_counts::Vector{Int}`     — cumulative-sample axis for LO checkpoints (L)
- `pair_ids::Vector{String}`   — per-pair identifiers (P)
- `err_batch::Array{Float32,3}` — (P, M, B) running-best imc [deg]
- `best_idx_batch::Array{Int32,3}` — (P, M, B) pool index of running best
- `err_lo::Array{Float32,3}` — (P, M, L) LO-refined imc [deg]
- `lo_ns_total/lo_calls_total/lo_startup_ns::Vector{Int64}` — per-method warm
  LO wall-clock [ns] / call count / cold first-call [ns] (JIT + one call)
"""
struct SampleBudgetResults
    methods::Vector{Symbol}
    sample_counts::Vector{Int}
    lo_counts::Vector{Int}
    pair_ids::Vector{String}
    err_batch::Array{Float32,3}
    best_idx_batch::Array{Int32,3}
    err_lo::Array{Float32,3}
    lo_ns_total::Vector{Int64}
    lo_calls_total::Vector{Int64}
    lo_startup_ns::Vector{Int64}
end

# =============================================================================
# Per-pair worker — walks the pool, fills the per-pair slice of the results
# =============================================================================
"""
    evaluate_sample_budget_pair!(out_err_batch, out_best_idx_batch, out_err_lo,
                                 out_lo_ns, out_lo_calls, selectors::Tuple,
                                 problem_type, pair, R, cfg; refit_scoring = nothing,
                                 sample_starts = nothing)

The per-pair worker of [`evaluate_sample_budget`](@ref): walk one pair's pool
and fill its slice of the results.

`pair` is the light pool record, `R` its residual matrix (num_points ×
num_models, Float32, the pool frame). `sample_starts` is the pair's
minimal-sample starts (VisualGeometryEval `sample_starts`), required when the
budget counts `:samples`. The selectors — a `Tuple`, so every per-model call is
statically dispatched — are mutated too (reset per pair, then folded), so the
problem type follows them. `refit_scoring` is the per-pair context's refit
template — the one every refit's scoring is built from at its own native — so
it must be the selectors' own.
"""
function evaluate_sample_budget_pair!(
    out_err_batch::AbstractMatrix{Float32},       # (n_sel, B)  imc pose error [deg]
    out_best_idx_batch::AbstractMatrix{Int32},
    out_err_lo::AbstractMatrix{Float32},          # (n_sel, L)
    out_lo_ns::AbstractVector{Int64},             # (n_sel,)  accumulated LO wall-clock [ns]
    out_lo_calls::AbstractVector{Int64},          # (n_sel,)  accumulated LO call count
    selectors::Tuple,
    problem_type::Type,
    pair::PoolPair,
    R::AbstractMatrix{Float32},
    cfg::SampleBudgetConfig;
    refit_scoring = nothing,
    sample_starts::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    nb = n_batches(cfg)
    nl = n_los(cfg)
    lo_counts = lo_sample_counts(cfg)   # explicit (non-uniform) LO checkpoint schedule

    n  = num_points(pair)
    M  = num_models(pair)
    sidx = pair.sample_indices
    size(R) == (n, M) || throw(DimensionMismatch(
        "evaluate_sample_budget_pair!: residual matrix $(size(R)) for a pair of $n points × $M models"))

    use_models = cfg.budget_unit === :models
    # The sample of each model: VisualGeometryEval's exact boundaries, never
    # rebuilt here (a column-change rule merges repeated draws).
    ids = use_models ? Int[] :
          sample_starts === nothing ?
              throw(ArgumentError("evaluate_sample_budget_pair!: a :samples budget needs the pair's sample_starts")) :
              sample_ids(sample_starts, M)

    foreach(reset!, selectors)   # fresh running-best state per pair

    samples_seen   = 0
    next_batch_pt  = cfg.batch_size
    next_batch_ix  = 1
    next_lo_ix     = 1

    # ONE per-pair context: pool-frame EssentialProblem + calibration +
    # GT pose + pool-frame points — shared by selection scoring and every refit.
    ctx  = build_pair_ctx(problem_type, pair; scoring = refit_scoring)
    geom = ctx.geom

    j = 1
    @inbounds while j <= M && (use_models ? j <= cfg.sample_cap : ids[j] <= cfg.sample_cap)
        col_j = view(sidx, :, j)
        # Samples COMPLETED once model j is seen: a checkpoint at B samples
        # sees every sibling of sample B.
        use_models || (samples_seen = j == M || ids[j+1] != ids[j] ? ids[j] : ids[j] - 1)

        foreach(sel -> update!(sel, geom, pair, R, j, col_j, n), selectors)

        # Budget counter: models inspected (`j`) or minimal samples (`samples_seen`).
        bgt = use_models ? j : samples_seen

        # Drain batch checkpoints — each selector reports its running best
        while bgt >= next_batch_pt && next_batch_ix <= nb
            _record_batch!(out_err_batch, out_best_idx_batch, selectors, geom, pair, next_batch_ix:next_batch_ix)
            next_batch_ix += 1
            next_batch_pt += cfg.batch_size
        end

        # Drain LO checkpoints — every selector refits its running best.
        while next_lo_ix <= nl && bgt >= lo_counts[next_lo_ix]
            _record_lo!(out_err_lo, out_lo_ns, out_lo_calls, ctx, selectors, cfg, next_lo_ix:next_lo_ix)
            next_lo_ix += 1
        end

        j += 1
    end

    # Pool exhausted before sample_cap — fill remaining batch checkpoints with
    # the last running best (flat tail rather than NaN).
    next_batch_ix <= nb &&
        _record_batch!(out_err_batch, out_best_idx_batch, selectors, geom, pair, next_batch_ix:nb)
    # LO tail: once the pool is exhausted the running best is FROZEN, so its LO
    # refit is constant across the remaining budgets. Compute it once and
    # flat-fill the tail (mirrors the batch flat-fill) instead of leaving NaN.
    # On pools ≫ sample_cap (PhotoTourism) this never runs; on pools sized near
    # sample_cap (ScanNet SPLG) it stops top-budget LO points from being
    # silently dropped as NaN.
    next_lo_ix <= nl && _record_lo!(out_err_lo, out_lo_ns, out_lo_calls, ctx, selectors, cfg, next_lo_ix:nl)
    return nothing
end

# `f(m, selectors[m])` over a tuple of selectors, unrolled (each call statically
# dispatched), in column order.
@inline _each_selector(f::F, t::Tuple, m::Int = 1) where {F} =
    isempty(t) ? nothing : (f(m, first(t)); _each_selector(f, Base.tail(t), m + 1))

# One batch checkpoint: each selector's running best, into column(s) `cols`.
function _record_batch!(out_err_batch, out_best_idx_batch, selectors::Tuple, geom, pair, cols)
    _each_selector(selectors) do m, sel
        er, bi = checkpoint(sel, geom, pair)
        out_err_batch[m, cols]      .= er
        out_best_idx_batch[m, cols] .= bi
    end
end

# One LO checkpoint: each selector's refit running best, into column(s) `cols`,
# and its LO wall-clock (ns == 0 ⇒ no model to refine, no call made).
function _record_lo!(out_err_lo, out_lo_ns, out_lo_calls, ctx, selectors::Tuple, cfg, cols)
    _each_selector(selectors) do m, sel
        er, ns = lo_error!(ctx, sel, cfg)
        out_err_lo[m, cols] .= er
        if ns > 0
            out_lo_ns[m]    += ns
            out_lo_calls[m] += 1
        end
    end
end

# =============================================================================
# Top-level driver — dispatch per-pair, gather results
# =============================================================================

"""
    SampleBudgetMethodSpec(name::Symbol, scoring)

A scoring-method column: a label and the scoring it selects with, at its
validated width (a Student-t + uniform marginal at its prior and outlier
half-width). Its running best is refit by the config's LO at the config's
native, else at the scoring's outlier half-width. `name` is opaque to the
evaluator.
"""
struct SampleBudgetMethodSpec{S}
    name::Symbol
    scoring::S
end

"""
    MsacSpec(name, loss_width, lo; refinement_loss_width = nothing)

One MSAC column of [`evaluate_sample_budget`](@ref): MSAC selection
(VisualGeometryCore's `MsacScoring` at threshold `loss_width`, pool frame), its
running best refit by its own refiner `lo` (a paper's truncated refiner — an
`ExternalRefit` strategy such as PoseLib's TLS-LM) at `refinement_loss_width`,
the arm's tuned refinement loss width; `nothing` refits at the selection loss
width.
"""
struct MsacSpec{L}
    name::Symbol
    loss_width::Float64
    lo::L
    refinement_loss_width::Union{Nothing,Float64}
end
MsacSpec(name::Symbol, loss_width::Real, lo::AbstractLocalOptimization;
         refinement_loss_width::Union{Nothing,Real} = nothing) =
    MsacSpec(name, Float64(loss_width), lo,
             refinement_loss_width === nothing ? nothing : Float64(refinement_loss_width))

# The selection width of a method column: its scoring's outlier half-width.
function _method_width(m::SampleBudgetMethodSpec)
    a = outlier_halfwidth(m.scoring)
    a === nothing && throw(ArgumentError(
        "evaluate_sample_budget: method $(m.name) scores with a $(nameof(typeof(m.scoring))), which has no " *
            "outlier half-width; a threshold column is an MsacSpec"))
    return Float64(a)
end

# The scoring template of a column at unit width — two columns refit under one
# template only if theirs agree.
_template(m::SampleBudgetMethodSpec) = with_width(m.scoring, OutlierHalfwidth(1.0))

"""
    evaluate_sample_budget(problem_type, methods::Vector{<:SampleBudgetMethodSpec}, pairs, residuals;
                           pair_ids, msac_specs::Vector{<:MsacSpec}, cfg::SampleBudgetConfig,
                           sample_starts = nothing, refit_scoring = methods[1].scoring)
        -> SampleBudgetResults

Run the sample-budget protocol over the pool pairs `pairs`.
The pairs are a flat vector of light pool records (`PoolPair`) with their residual matrices `residuals` (`residuals[i]` is pair
`i`'s; VisualGeometryEval `load_pool_residuals`, `residual_matrix`).
A `:samples` budget (`cfg.budget_unit`) counts minimal samples through
`sample_starts[i]`, pair `i`'s minimal-sample starts — VisualGeometryEval
`sample_starts(pool)`, the one reconstruction of sample boundaries; it is
required then and ignored for a `:models` budget.
Columns: the scoring methods, then one MSAC column per [`MsacSpec`](@ref),
then the oracle (last). Every native refit is seeded under `refit_scoring` (a
scoring TEMPLATE, rebuilt at each refit's own outlier half-width) — the
methods' own: they must share it.
Parallel over pairs; per-method scoring workspaces preallocated per thread.
"""
function evaluate_sample_budget(problem_type::Type,
                                methods::AbstractVector{<:SampleBudgetMethodSpec},
                                pairs::AbstractVector{<:PoolPair},
                                residuals::AbstractVector{<:AbstractMatrix{Float32}};
                                pair_ids::AbstractVector{<:AbstractString} = String[],
                                msac_specs::AbstractVector{<:MsacSpec} = MsacSpec[],
                                cfg::SampleBudgetConfig,
                                sample_starts::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}} = nothing,
                                refit_scoring = isempty(methods) ? nothing : methods[1].scoring)
    P  = length(pairs)
    length(residuals) == P || throw(DimensionMismatch(
        "evaluate_sample_budget: $P pairs and $(length(residuals)) residual matrices"))
    if cfg.budget_unit === :samples
        sample_starts === nothing && throw(ArgumentError(
            "evaluate_sample_budget: a :samples budget needs `sample_starts` (VisualGeometryEval `sample_starts(pool)`)"))
        length(sample_starts) == P || throw(DimensionMismatch(
            "evaluate_sample_budget: $P pairs and $(length(sample_starts)) sample-start vectors"))
    end
    # ONE refit template: the per-pair context carries it, and every refit
    # builds its scoring from it at its own native (as the miscal evaluation's
    # `apply_lo_refits` does) — a method selecting under another template (a
    # scale prior) would refit under the first method's.
    foreach(_method_width, methods)   # a threshold scorer is refused first: it is an MsacSpec
    isempty(methods) || all(m -> _template(m) == _template(methods[1]), methods) ||
        throw(ArgumentError("evaluate_sample_budget: the methods select under different scoring templates " *
                            "($(unique([_template(m) for m in methods]))); the per-pair context and every " *
                            "refit carry one — run one evaluation per template"))
    M  = length(methods)
    Mm = length(msac_specs)             # MSAC selectors (truncated select + LO)
    B  = n_batches(cfg)
    L  = n_los(cfg)
    nsel = M + Mm + 1            # scoring methods + MSAC + the oracle (last column)

    err_batch      = Array{Float32}(undef, P, nsel, B)
    best_idx_batch = Array{Int32}(undef,   P, nsel, B)
    err_lo         = Array{Float32}(undef, P, nsel, L)
    lo_ns_pp       = zeros(Int64, P, nsel)   # per-pair LO wall-clock [ns]  (race-free: one row per pair)
    lo_calls_pp    = zeros(Int64, P, nsel)   # per-pair LO call count

    # One selector tuple per thread (each ScoreSelector owns its scoring +
    # scratch workspace); the OracleSelector is appended last.
    N_max = maximum(num_points, pairs)
    dg    = codimension(problem_type)
    nt    = Threads.maxthreadid()
    _scorer(name, sc, width; kw...) =
        ScoreSelector(name, sc, width, scoring_workspace(sc, N_max, dg), Vector{Float64}(undef, N_max); kw...)
    _thread_tuple() = (
        Tuple(_scorer(m.name, m.scoring, _method_width(m)) for m in methods)...,
        # MSAC selection + its own refiner: one selector per spec.
        Tuple(_scorer(s.name, MsacScoring(dg; threshold = s.loss_width), s.loss_width;
                      lo = s.lo, refinement_width = s.refinement_loss_width) for s in msac_specs)...,
        # Oracle refines its GT-best model at a representative outlier
        # half-width (the first method's).
        OracleSelector(:oracle, _method_width(methods[1])),
    )
    thread_selectors = [_thread_tuple() for _ in 1:nt]

    # Startup accounting: warm every active LO path single-threaded BEFORE the
    # loop (first call pays JIT). Makes the per-call numbers pure steady-state.
    lo_startup_ns = _warmup_lo_startup(problem_type, methods, pairs, msac_specs, cfg, nsel, refit_scoring)

    Threads.@threads for i in 1:P
        tid = Threads.threadid()
        evaluate_sample_budget_pair!(
            view(err_batch,      i, :, :),
            view(best_idx_batch, i, :, :),
            view(err_lo,         i, :, :),
            view(lo_ns_pp,       i, :),
            view(lo_calls_pp,    i, :),
            thread_selectors[tid],
            problem_type,
            pairs[i],
            residuals[i],
            cfg;
            refit_scoring,
            sample_starts = cfg.budget_unit === :samples ? sample_starts[i] : nothing,
        )
    end

    ids = isempty(pair_ids) ? [string(i) for i in 1:P] : collect(pair_ids)
    length(ids) == P || error("pair_ids length $(length(ids)) ≠ $P pairs")

    return SampleBudgetResults(
        vcat(Symbol[m.name for m in methods],          # scoring methods …
             Symbol[spec.name for spec in msac_specs],     # … then MSAC …
             [:oracle]),                               # … oracle last
        batch_sample_counts(cfg),
        lo_sample_counts(cfg),
        ids,
        err_batch, best_idx_batch, err_lo,
        vec(sum(lo_ns_pp;    dims = 1)),               # per-method total warm LO [ns]
        vec(sum(lo_calls_pp; dims = 1)),               # per-method warm LO call count
        lo_startup_ns,                                 # per-method cold first-call [ns]
    )
end

# Warm every active LO path once, single-threaded, on the first pair's first
# model — compiles the path and records the cold cost (JIT + one call). Each
# distinct path (the IRLS-family `cfg.lo_strategy`, and the truncated
# refiner of the MSAC selectors) is measured once and broadcast to its columns.
function _warmup_lo_startup(problem_type, methods, pairs, msac_specs, cfg, nsel, refit_scoring)
    startup = zeros(Int64, nsel)
    isempty(pairs) && return startup
    M = length(methods)
    ctx = build_pair_ctx(problem_type, pairs[1]; scoring = refit_scoring)
    seed_E = refit_seed(ctx, ctx.pair.models[1])   # problem-family seed decode
    # Active-LO path (any cfg.lo_strategy: IRLS-family, the E-track lifts and
    # GNC arms — all native locally_optimize since the 2026-07-18 port; shared
    # by all scoring columns + the oracle column). Route the native through
    # `consumed_native` at cfg.nat_axis — identity for the :ig IRLS/GNC consumers,
    # outlier half-width → loss width for the truncated consumers — so the warm
    # call compiles and times the
    # SAME path the loop takes.
    if !(cfg.lo_strategy isa NoLocalOptimization)
        nat_w = consumed_native(cfg.lo_strategy, cfg.nat_axis, something(cfg.nat_ref, _method_width(methods[1]));
                             d_g = codimension(problem_type))
        t  = time_ns()
        refit_imc!(ctx, cfg.lo_strategy, seed_E, nat_w)
        cold = Int64(time_ns() - t)
        for m in 1:M; startup[m] = cold; end
        startup[nsel] = cold                       # oracle (last column)
    end
    # Truncated path (shared by all MSAC columns).
    if !isempty(msac_specs)
        s1 = first(msac_specs)
        t  = time_ns()
        refit_imc!(ctx, s1.lo, seed_E, s1.loss_width)
        cold = Int64(time_ns() - t)
        startup[M .+ eachindex(msac_specs)] .= cold
    end
    return startup
end

"""
    summarize_lo_timing(res::SampleBudgetResults) -> Vector{NamedTuple}

Per-method wall-clock summary: mean warm LO call [µs] (Render's
`lo_us_per_call`), total warm calls, and
the cold startup cost [ms]. Per-call numbers are measured in-situ under the
parallel run, so they include thread contention (run single-threaded for
isolated per-call cost).
"""
function summarize_lo_timing(res::SampleBudgetResults)
    [ (method     = res.methods[m],
       calls      = res.lo_calls_total[m],
       mean_us    = lo_us_per_call(res, m),
       total_ms   = res.lo_ns_total[m] / 1e6,
       startup_ms = res.lo_startup_ns[m] / 1e6)
      for m in eachindex(res.methods) ]
end
