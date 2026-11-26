# =============================================================================
# evaluate_val.jl — the bootstrap tuning harness: how well a hyperparameter
# tuned on a validation draw of `n` pairs performs on the test pool, as a
# function of `n`.
#
# Per bootstrap trial: take the protocol's fixed val/test split
# (`FlatProtocol`), select on `n` drawn val pairs (`solve(EstimationProblem(Hyperparameter, errors, (; pairs,
# to_native)), scheme::WindowedBootstrap)`), read the selected column's errors on the test pool.
# `compute_method_rows` runs `bootstrap_trials` over every `n` and summarizes with BCa CIs.
#
# Consumers: LocalOptimizationDoneRight's LO-sensitivity renderer (FlatProtocol + ValidationErrors through
# `compute_method_rows`) and the standardized window (`window_columns`). Every random draw is seeded by the run's
# `BudgetSampling` (`budget_rng`).
# =============================================================================

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


# =============================================================================
# Generic bootstrap loop — tune and evaluate are explicit, separate steps
# =============================================================================

"""
    ValidationErrors(label, scene_errors, scene_sigma2, native_grid)

The data of a windowed hyperparameter selection: precomputed post-LO validation errors on a fixed grid of native
hyperparameter values.

- `scene_errors[i]` — the (pairs × |`native_grid`|) error matrix of scene `i`;
- `scene_sigma2[i][j]` — pair `(i, j)`'s noise variance `σ̂²` (e.g. `gt_variance`), which forms the
  cardinality-tied scale `σ_med(n)`;
- `label` — the method's name in the harness's rows.

`solve(EstimationProblem(Hyperparameter, errors, (; pairs, to_native)), scheme::WindowedBootstrap)` selects on the
drawn val `pairs` (`(scene, pair)` indices). `to_native(efficiency_constant, scale) -> native` is the eq-(28) map for
this scoring, a closure over the caller's `native_map(axis; d_g)`, so this package stays decoupled from the
scoring-spec types.
"""
struct ValidationErrors
    label::String
    scene_errors::Vector{Matrix{Float64}}
    scene_sigma2::Vector{Vector{Float64}}
    native_grid::Vector{Float64}
    ValidationErrors(label, scene_errors, scene_sigma2, native_grid) =
        new(String(label), scene_errors, scene_sigma2, collect(Float64, native_grid))
end

"""
    bootstrap_trials(errors::ValidationErrors, scheme::WindowedBootstrap, proto::FlatProtocol, n;
                     to_native, budget = scheme.budget_sampling, max_error = maa_threshold())
        -> (trial_meds, trial_maas, trial_idxs)

The per-draw bootstrap: each of the `budget.n_draws` draws selects the hyperparameter's column on `n` val pairs of
`proto` (drawn by `budget_rng(budget, n, r)`, then `solve(EstimationProblem(Hyperparameter, errors, (; pairs,
to_native)), scheme)`) and evaluates that column on its test pairs: the draw's median test error (a pair without a
model counts as `Inf`) and its mAA@`max_error` (the eligibility rule, `eligible_maa`). A draw with no finite σ̂² (or
no val pairs) selects the first column.
"""
function bootstrap_trials(errors::ValidationErrors, scheme::WindowedBootstrap, proto::FlatProtocol, n::Int;
                          to_native, budget::BudgetSampling = scheme.budget_sampling, max_error::Real = maa_threshold())
    n_bootstrap = budget.n_draws
    trial_meds = Vector{Float64}(undef, n_bootstrap)
    trial_maas = Vector{Float64}(undef, n_bootstrap)
    trial_idxs = Vector{Int}(undef, n_bootstrap)
    # The test statistics depend on the selected column only: computed once per column.
    test_stats = Dict{Int,NTuple{2,Float64}}()
    val = proto.val_pair_idx
    for b in 1:n_bootstrap
        col = if isempty(val)
            1
        else
            drawn = val[_sample_indices(budget_rng(budget, n, b), length(val), 1.0, n)]   # the draw of the budget sampling
            r = solve(EstimationProblem(Hyperparameter, errors, (; pairs = drawn, to_native)), scheme)
            something(r.u, 1)
        end
        trial_idxs[b] = col
        trial_meds[b], trial_maas[b] = get!(test_stats, col) do
            e = _test_errors(errors, proto.test_pair_idx, col)
            isempty(e) ? (NaN, NaN) : (tune_objective(MedianError(), e), eligible_maa(e; max_error))
        end
    end
    return trial_meds, trial_maas, trial_idxs
end


# =============================================================================
# The standardized-window column selector
# =============================================================================

"""
    window_columns(window::EfficiencyConstantWindow, grid, tonat, scale) -> Vector{Int}

Column indices of the native `grid` inside the standardized window
`[tonat(window.lo, scale), tonat(window.hi, scale)]` — the candidate set of
every windowed free argmin (the bootstrap tuner here, the miscalibration
calibration tune of LocalOptimizationDoneRight). Falls back to the EDGE CLAMP (the grid boundary the window crossed)
when the window slid entirely off the grid (extreme miscalibration on an
unwidened grid); never a snapped candidate set.
"""
function window_columns(window::EfficiencyConstantWindow, grid::AbstractVector{<:Real}, tonat, scale::Real)
    lo, hi = tonat(window.lo, scale), tonat(window.hi, scale)
    lo > hi && ((lo, hi) = (hi, lo))
    cols = [c for c in eachindex(grid) if lo <= grid[c] <= hi]
    return isempty(cols) ?
        [log(grid[end]) < 0.5 * (log(lo) + log(hi)) ?
         lastindex(grid) : firstindex(grid)] : cols
end

# =============================================================================
# ValidationErrors and the windowed selection (WindowedBootstrap's solve)
# =============================================================================


"""
    solve(EstimationProblem(Hyperparameter, errors::ValidationErrors, (; pairs, to_native)), scheme::WindowedBootstrap)

One draw of the bootstrap tuning harness (accv §common-cutoff, eq. 28): on the **standardized efficiency-constant
axis** instead of the raw native grid, with the reference scale **estimated from the drawn validation pairs**, so
it tracks the validation cardinality. `σ_med = √median σ̂²` over the drawn pairs; the FREE ARGMIN of the scheme's
`criterion` over every `native_grid` column inside the standardized window `[to_native(lo, σ_med),
to_native(hi, σ_med)]` (the scheme's `efficiency_constant_window`; corrupting the scale slides the window, which is
the miscalibration mechanism). `u` is the selected column, `n` the pairs with a finite σ̂², `(Success, ClosedForm)`;
no finite σ̂² is degenerate, `u = nothing` with `(InitialFailure, NoRule)`.
"""
function __solve(prob::EstimationProblem{Hyperparameter, ValidationErrors, <:NamedTuple{(:pairs, :to_native)}},
                 scheme::WindowedBootstrap)
    m = prob.f; drawn = prob.p.pairs
    s2 = Float64[]; sizehint!(s2, length(drawn))
    for (i, j) in drawn
        v = m.scene_sigma2[i][j]; isfinite(v) && push!(s2, v)
    end
    isempty(s2) && return FitResult(nothing, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, nothing)
    # FREE ARGMIN within the standardized window at the drawn pairs' reference scale: every native-grid column inside
    # it is a candidate. The WINDOW anchoring (what a scale corruption slides) is the whole mechanism.
    cols = window_columns(scheme.efficiency_constant_window, m.native_grid, prob.p.to_native, reference_scale(s2))
    errbuf = Vector{Float64}(undef, length(drawn))
    best_obj = Inf; best_col = first(cols)
    @inbounds for col in cols
        nfin = 0
        for (i, j) in drawn
            v = m.scene_errors[i][j, col]; isfinite(v) && (nfin += 1; errbuf[nfin] = v)
        end
        nfin == 0 && continue
        o = tune_objective(scheme.criterion, view(errbuf, 1:nfin))
        o < best_obj && (best_obj = o; best_col = col)
    end
    return FitResult(best_col, ReturnCode.Success, ClosedForm, SolveStats(), NaN, length(s2), nothing)
end

# The test errors at the selected column: a precomputed lookup `scene_errors[i][j, col]`.
_test_errors(m::ValidationErrors, test_pairs, col::Int) = Float64[m.scene_errors[i][j, col] for (i, j) in test_pairs]
# =============================================================================
# Result reporting
# =============================================================================

_grid_T(m::ValidationErrors) = isempty(m.scene_errors) ? 0 : size(m.scene_errors[1], 2)


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
    compute_method_rows(errors::ValidationErrors, scheme::WindowedBootstrap; protocol, to_native,
                        budget = scheme.budget_sampling, sizes = budget.sizes)
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
function compute_method_rows(errors::ValidationErrors, scheme::WindowedBootstrap; protocol::FlatProtocol, to_native,
                             budget::BudgetSampling = scheme.budget_sampling, sizes = budget.sizes)
    mname = errors.label
    tname = tuning_criterion_label(scheme.criterion)
    T = _grid_T(errors)

    summary = NamedTuple[]
    trials  = NamedTuple[]
    for n in sizes
        rng = resampling_rng(budget, n)
        trial_meds, trial_maas, trial_idxs = bootstrap_trials(errors, scheme, protocol, n; to_native, budget)
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
