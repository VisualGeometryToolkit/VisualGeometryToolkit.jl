# =============================================================================
# evaluate_val.jl — the bootstrap tuning harness: how well a hyperparameter
# tuned on a validation draw of `n` pairs performs on the test pool, as a
# function of `n`.
#
# Per bootstrap trial: take the protocol's fixed val/test split
# (`FlatProtocol`), tune on `n` val pairs (`tune`), evaluate the tuned column on
# the test pool (`evaluate`). `bootstrap_trials` is method-agnostic;
# `compute_method_rows` runs it over every `n` and summarizes with BCa CIs.
#
# Consumers: LocalOptimizationDoneRight's LO-sensitivity renderer (FlatProtocol +
# WindowedGridMethod through `compute_method_rows`) and the standardized
# window (`window_columns`). Every random draw is seeded by the run's
# `BudgetSampling` (`budget_rng`).
# =============================================================================

# =============================================================================
# Method abstraction — tune/evaluate dispatch
# =============================================================================

"""
    BootstrapMethod

A tuner the bootstrap harness runs ([`bootstrap_trials`](@ref)): it implements
[`tune`](@ref) and [`evaluate`](@ref). [`WindowedGridMethod`](@ref) is the one
in this package.
"""
abstract type BootstrapMethod end

"""
    tune(rng, method::BootstrapMethod, val_pairs, n) -> params

Tune `method` on `n` validation pairs drawn by `rng` from `val_pairs` (the
protocol's `(scene, pair)` indices); `params` is method-specific and is handed
back to [`evaluate`](@ref).
"""
function tune end

# =============================================================================
# The protocol — a fixed val/test pair split
# =============================================================================

"""
    FlatProtocol(val_pair_idx, test_pair_idx)

Fixed deterministic pair-level split, bootstrap pairs only. Trial-level
variance reflects pair-sampling variance only (no scene component).
Used for ScanNet1500 and other "uniform sample of pairs" datasets.
"""
struct FlatProtocol
    val_pair_idx::Vector{Tuple{Int,Int}}
    test_pair_idx::Vector{Tuple{Int,Int}}
end


"""
    evaluate(method::BootstrapMethod, test_pairs, params) -> Vector{Float64}

The test errors of `method` at the tuned `params` on `test_pairs` (the
protocol's `(scene, pair)` indices).
"""
function evaluate end

# =============================================================================
# Generic bootstrap loop — tune and evaluate are explicit, separate steps
# =============================================================================

"""
    bootstrap_trials(method, proto::FlatProtocol, budget::BudgetSampling, n;
                     T = maa_threshold())
        -> (trial_meds, trial_maas, trial_idxs)

The per-draw bootstrap: each of the `budget.n_draws` draws tunes the
hyperparameter on `n` val pairs of `proto` (`tune`) and evaluates the tuned
column on its test pairs (`evaluate`): the draw's median test error (a pair
without a model counts as `Inf`) and its mAA@`T` (the eligibility
rule, `eligible_maa`). Draw `r` uses `budget_rng(budget, n, r)`.
"""
function bootstrap_trials(method::BootstrapMethod, proto::FlatProtocol, budget::BudgetSampling, n::Int;
                           T::Real = maa_threshold())
    n_bootstrap = budget.n_draws
    trial_meds = Vector{Float64}(undef, n_bootstrap)
    trial_maas = Vector{Float64}(undef, n_bootstrap)
    trial_idxs = Vector{Int}(undef, n_bootstrap)
    # The test statistics depend on the tuned column only: computed once per column.
    test_stats = Dict{Int,NTuple{2,Float64}}()
    for b in 1:n_bootstrap
        col = tune(budget_rng(budget, n, b), method, proto.val_pair_idx, n)
        trial_idxs[b] = col
        trial_meds[b], trial_maas[b] = get!(test_stats, col) do
            errors = evaluate(method, proto.test_pair_idx, col)
            isempty(errors) ? (NaN, NaN) : (tune_objective(MedianError(), errors), eligible_maa(errors; T))
        end
    end
    return trial_meds, trial_maas, trial_idxs
end


# =============================================================================
# The standardized-window column selector
# =============================================================================

"""
    window_columns(window::ScaleMultipleWindow, grid, tonat, scale) -> Vector{Int}

Column indices of the native `grid` inside the standardized window
`[tonat(window.lo, scale), tonat(window.hi, scale)]` — the candidate set of
every windowed free argmin (the bootstrap tuner here, the miscalibration
calibration tune of LocalOptimizationDoneRight). Falls back to the EDGE CLAMP (the grid boundary the window crossed)
when the window slid entirely off the grid (extreme miscalibration on an
unwidened grid); never a snapped candidate set.
"""
function window_columns(window::ScaleMultipleWindow, grid::AbstractVector{<:Real}, tonat, scale::Real)
    lo, hi = tonat(window.lo, scale), tonat(window.hi, scale)
    lo > hi && ((lo, hi) = (hi, lo))
    cols = [c for c in eachindex(grid) if lo <= grid[c] <= hi]
    return isempty(cols) ?
        [log(grid[end]) < 0.5 * (log(lo) + log(hi)) ?
         lastindex(grid) : firstindex(grid)] : cols
end

# =============================================================================
# WindowedGridMethod — scale-multiple tuning with cardinality-tied scale
# =============================================================================

"""
    WindowedGridMethod(label, scene_errors, scene_sigma2, native_grid, window, to_native,
                       criterion)

Tunes on the **standardized scale-multiple axis** (accv §common-cutoff,
eq. 28) instead of the raw native grid, with the reference scale **estimated
from the validation draw** so it tracks the validation cardinality `n` — the
protocol of the `windowed_bootstrap` tuning scheme.

- `scene_errors[i]` — precomputed (pairs × |`native_grid`|) post-LO error matrix
  on a fixed grid of native hyperparameter values (`native_grid`).
- `scene_sigma2[i][j]` — pair `(i,j)`'s noise variance `σ̂²` (e.g. `gt_variance`),
  used to form the cardinality-tied scale `σ_med(n)`.
- `to_native(scale_multiple, scale) -> native` — the eq-(28) map for this
  scoring, a closure over the caller's `native_map(axis; d_g)` so this package stays
  decoupled from the scoring-spec types.
- `window` — the spec's [`ScaleMultipleWindow`](@ref); `criterion` — its
  objective (no default: both are parameters of the tuning spec).

Per trial, `tune` draws `n` val pairs, forms `σ_med(n)=√median σ̂²` over them,
and takes the FREE ARGMIN of the val objective over every `native_grid` column
inside the standardized window `[to_native(lo, σ_med), to_native(hi, σ_med)]`
(the window's covered range at the drawn supplied scale — corrupting the
scale slides the window, which is the miscalibration mechanism). `evaluate`
reads the tuned column on the test pairs.
"""
struct WindowedGridMethod{F,C<:TuningCriterion} <: BootstrapMethod
    label::String
    scene_errors::Vector{Matrix{Float64}}
    scene_sigma2::Vector{Vector{Float64}}
    native_grid::Vector{Float64}
    window::ScaleMultipleWindow
    to_native::F
    criterion::C
    WindowedGridMethod(label, scene_errors, scene_sigma2, native_grid, window::ScaleMultipleWindow,
                       to_native::F, criterion::C) where {F,C<:TuningCriterion} =
        new{F,C}(label, scene_errors, scene_sigma2, collect(Float64, native_grid), window, to_native, criterion)
end

function tune(rng::AbstractRNG, m::WindowedGridMethod, val_pairs, n)
    isempty(val_pairs) && return 1
    drawn = val_pairs[_sample_indices(rng, length(val_pairs), 1.0, n)]          # the draw of the budget sampling
    # s0(n) = √median σ̂² over the drawn val pairs — scale tied to cardinality n.
    s2 = Float64[]; sizehint!(s2, length(drawn))
    for (i, j) in drawn
        v = m.scene_sigma2[i][j]; isfinite(v) && push!(s2, v)
    end
    isempty(s2) && return 1
    # FREE ARGMIN within the standardized window at the drawn pairs' reference
    # scale: every native-grid column inside it is a candidate, and the val
    # objective is minimized freely over them. The WINDOW anchoring (what a
    # scale corruption slides) is the whole mechanism.
    cols = window_columns(m.window, m.native_grid, m.to_native, reference_scale(s2))
    errbuf = Vector{Float64}(undef, length(drawn))
    best_obj = Inf; best_col = first(cols)
    @inbounds for col in cols
        nfin = 0
        for (i, j) in drawn
            v = m.scene_errors[i][j, col]; isfinite(v) && (nfin += 1; errbuf[nfin] = v)
        end
        nfin == 0 && continue
        o = tune_objective(m.criterion, view(errbuf, 1:nfin))
        o < best_obj && (best_obj = o; best_col = col)
    end
    return best_col
end

# The test error at the tuned column is a precomputed lookup `scene_errors[i][j, col]`.
evaluate(m::WindowedGridMethod, test_pairs, col::Int) =
    Float64[m.scene_errors[i][j, col] for (i, j) in test_pairs]
# =============================================================================
# Result reporting
# =============================================================================

tuning_criterion_label(m::WindowedGridMethod) = tuning_criterion_label(m.criterion)

method_name(m::WindowedGridMethod) = m.label

_grid_T(m::WindowedGridMethod) = isempty(m.scene_errors) ? 0 : size(m.scene_errors[1], 2)


"""
    _summarize_trials(rng, meds, maas, idxs, T) -> NamedTuple

BCa-bootstrap mean/std + 95% CI per criterion, plus per-trial argmin
diagnostics (`idx_mean`, `idx_std`, `edge_frac`; NaN when `idxs` is empty
or `T == 0`).
"""
function _summarize_trials(rng::AbstractRNG, meds::AbstractVector,
                           maas::AbstractVector, idxs::AbstractVector, T::Int)
    isempty(meds) && return (mean_med=NaN, lo95_mean_med=NaN, hi95_mean_med=NaN,
                              std_med=NaN,  lo95_std_med=NaN,  hi95_std_med=NaN,
                              mean_mAA=NaN, lo95_mean_mAA=NaN, hi95_mean_mAA=NaN,
                              std_mAA=NaN,  lo95_std_mAA=NaN,  hi95_std_mAA=NaN,
                              idx_mean=NaN, idx_std=NaN, edge_frac=NaN, T=T)
    mm = mean(meds); lo_m, hi_m   = bootstrap_ci(mean, rng, meds)
    sm = std(meds);  lo_s, hi_s   = bootstrap_ci(std,  rng, meds)
    ma = mean(maas); lo_a, hi_a   = bootstrap_ci(mean, rng, maas)
    sa = std(maas);  lo_sa, hi_sa = bootstrap_ci(std,  rng, maas)
    if isempty(idxs) || T == 0
        idx_mean, idx_std, edge_frac = NaN, NaN, NaN
    else
        idx_mean  = mean(idxs)
        idx_std   = std(idxs)
        edge_frac = count(i -> i <= 2 || i >= T - 1, idxs) / length(idxs)
    end
    (mean_med=mm, lo95_mean_med=lo_m, hi95_mean_med=hi_m,
     std_med=sm,  lo95_std_med=lo_s,  hi95_std_med=hi_s,
     mean_mAA=ma, lo95_mean_mAA=lo_a, hi95_mean_mAA=hi_a,
     std_mAA=sa,  lo95_std_mAA=lo_sa, hi95_std_mAA=hi_sa,
     idx_mean=idx_mean, idx_std=idx_std, edge_frac=edge_frac, T=T)
end

"""
    compute_method_rows(method, budget::BudgetSampling; protocol, sizes = budget.sizes)
        -> (summary::Vector{NamedTuple}, trials::Vector{NamedTuple})

Per-method, per-`n` evaluation loop over the validation budgets `sizes` (the
schedule's own by default; a miscal evaluation supplies its one fixed budget):
`budget.n_draws` draws per budget ([`bootstrap_trials`](@ref)), the BCa CIs
resampled with `resampling_rng(budget, n)`. Returns two row vectors in canonical
DataFrame-friendly shape:

- `summary`: one row per `n` carrying the BCa CIs and per-trial argmin
  diagnostics that were the columns of the per-cell `summary` DataFrame.
- `trials` : one row per (n, trial) carrying the raw `(trial_med, trial_mAA,
  trial_idx)` so CIs can be replayed at any α downstream.

`protocol` is the val/test split ([`FlatProtocol`](@ref)). (The `produce_or_load` cache and the
decoupled `tune_protocol` path were pruned 2026-09-21 with their only
caller, `evaluate_val`.)
"""
function compute_method_rows(method::BootstrapMethod, budget::BudgetSampling;
                              protocol::FlatProtocol, sizes = budget.sizes)
    mname = method_name(method)
    tname = tuning_criterion_label(method)
    T = _grid_T(method)

    summary = NamedTuple[]
    trials  = NamedTuple[]
    for n in sizes
        rng = resampling_rng(budget, n)
        trial_meds, trial_maas, trial_idxs = bootstrap_trials(method, protocol, budget, n)
        for b in eachindex(trial_meds)
            push!(trials, (method=mname, tuning_criterion=tname, n=n, trial=b,
                           trial_med=trial_meds[b],
                           trial_maa=trial_maas[b],
                           trial_idx=trial_idxs[b]))
        end
        valid = .!isnan.(trial_meds)
        vm = trial_meds[valid]; va = trial_maas[valid]
        stats = _summarize_trials(rng, vm, va, trial_idxs[valid], T)
        push!(summary, (method=mname, tuning_criterion=tname, n=n, stats...))
    end

    @printf("%-12s [%s]", mname, tname)
    for r in summary
        if isnan(r.mean_med)
            @printf("  n=%-5d: NaN", r.n)
        else
            @printf("  n=%-5d: med=%.2f±%.2f mAA=%.3f±%.3f  idx=%.0f±%.0f/%d edge=%.2f",
                    r.n, r.mean_med, r.std_med, r.mean_mAA, r.std_mAA,
                    r.idx_mean, r.idx_std, T, r.edge_frac)
        end
    end
    println()
    flush(stdout)
    return summary, trials
end
