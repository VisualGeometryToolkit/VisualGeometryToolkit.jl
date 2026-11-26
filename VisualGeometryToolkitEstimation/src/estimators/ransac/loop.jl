# =============================================================================
# RANSAC Loop — the generic loop on VisualGeometryToolkitCore's contracts
# =============================================================================
#
# The loop calls ONLY contract verbs (VisualGeometryToolkitCore src/robust/
# scoring_contract.jl and lo_contract.jl): `scored_model`, `score_model!`
# (residuals → score! → consensus mask), `inlier_count`, `consensus_order`,
# `consensus_set`, `consensus_s2`, `inlier_probabilities!`, `is_predictive`,
# and `locally_optimize!`. Every dispatch is static: the workspace
# `RansacWorkspace{K, M, T, W, L}` carries the scorer's workspace type `W` and
# the local optimizer's `L`, so `_ransac_loop` compiles once per (problem,
# scorer, local optimizer).
#
# Per trial (`_try_model!`):
#   Phase 1  scored_model → score_model! (mask deferred) → gate S > S_gl
#            → consensus mask → test_consensus
#   Phase 2  consensus_set → locally_optimize!
#   Phase 3  the refined model re-scored (its residuals reused when the
#            optimizer left them current) → gate S > S_lo → test_consensus
# =============================================================================

"""
    _ransac!(result::RansacResultBuffers, ws::RansacWorkspace, problem, scoring;
             local_optimization_strategy, config, on_improvement, on_hypothesis, score_guard, rng) -> RansacEstimate

RANSAC on VisualGeometryToolkitCore's scoring and local-optimization contracts — the body of [`Ransac`](@ref)'s
`solve!`: the loop calls only the contract verbs, so any scorer composes with any local optimizer and any problem. On
the preallocated workspace `ws` (trimmed to the problem's data size), its result written into the preallocated
`result` buffers — the returned estimate's `resid` and `weights` ARE those buffers.

For each minimal sample, fits θₜ, scores it, and takes the scorer's consensus set. Two independent best-so-far scores
are maintained: S_gl gates entry into local optimization; S_lo tracks the best locally optimized score. The adaptive
trial count uses the hypergeometric distribution (the stopping criterion).
"""
function _ransac!(
        result::RansacResultBuffers, ws::RansacWorkspace, problem::AbstractRansacProblem,
        scoring::AbstractScoring;
        local_optimization_strategy::Union{Nothing, AbstractOptimizationStrategy},
        config::RansacConfig,
        on_improvement = nothing,
        on_hypothesis = nothing,
        score_guard::Bool = false,
        rng::Random.AbstractRNG
    )
    _check_scoring(problem, scoring)
    _entry_workspace(ws, problem, scoring, local_optimization_strategy)
    _set_n!(result, data_size(problem))
    return _ransac_placed(
        lo_placement(local_optimization_strategy),
        ws, problem, scoring, local_optimization_strategy, config, on_improvement,
        score_guard, rng, on_hypothesis, result
    )
end

# The strategy's placement (`lo_placement`), dispatched statically: in-loop is
# the loop itself; terminal is the loop with no LO, then one seeded LO of its
# incumbent (`TerminalLO`).
@inline _ransac_placed(::InLoopPlacement, args...) = _ransac_loop(args...)

function _ransac_placed(
        ::TerminalPlacement, ws::RansacWorkspace, problem, scoring,
        local_optimization_strategy, config, on_improvement,
        score_guard::Bool, rng::Random.AbstractRNG, on_hypothesis, result
    )
    score_guard && throw(
        ArgumentError(
            "score_guard is a per-candidate acceptance rule; a terminal local optimization " *
                "($(nameof(typeof(local_optimization_strategy)))) runs on no candidate, only on the final incumbent"
        )
    )
    raw = _ransac_loop(
        ws, problem, scoring, nothing, config, on_improvement,
        false, rng, on_hypothesis, result
    )
    found_model(raw) || return raw
    # `has_best` still holds the raw incumbent: a seed the scorer cannot score
    # finalizes as it, at its race score. The search's termination record is
    # the result's, unless the terminal LO did not succeed (`_terminal_record`).
    q = raw.metadata.quality
    return _locally_optimize_seed!(
        ws, problem, scoring, local_optimization_strategy, raw.u;
        trial = raw.stats.nsteps, sar = raw.metadata.sample_acceptance_rate, result, unscored = (q, q),
        search = (raw.retcode, raw.stop_rule)
    )
end

# The workspace every run is on — a find's and a seeded local optimization's
# (`RansacCache`, solve.jl). A shared
# workspace may be sized for a larger problem, so every internal buffer is
# trimmed to `data_size(problem)`. And the previous call's incumbent is cleared:
# a run that never accepts a candidate would otherwise `_finalize` the stale
# `best_model` as :converged. The composition is checked once, here: a scorer's
# workspace built for another scorer or configuration, and a local optimizer that follows
# a scorer with no threshold.
function _entry_workspace(ws::RansacWorkspace, problem, scoring, lo)
    set_n!(ws, data_size(problem))
    ws.has_best = false
    ws.best_lo_retcode, ws.best_lo_stop_rule = _LO_NOT_RUN
    check_workspace(scoring, ws.buf.scoring_ws)
    check_composition(scoring, lo)
    return ws
end

_result_buffers(ws::RansacWorkspace{<:Any, <:Any, T}, problem) where {T} = RansacResultBuffers{T}(data_size(problem))

# Function barrier: ws has fully concrete type {K,M,T,W,L} after dispatch
function _ransac_loop(
        ws::RansacWorkspace{K}, problem, scoring,
        local_optimization_strategy, config, on_improvement,
        score_guard::Bool, rng::Random.AbstractRNG, on_hypothesis, result
    ) where {K}
    best = (-Inf, -Inf)
    # The sampler owns the schedule (`trial_budget`, `adapts`): `max_trials`,
    # re-armed adaptively, for a random sampler; every sample exactly once,
    # never re-armed, for the `ExhaustiveSampler`.
    s = sampler(problem)
    needed = trial_budget(s, config)
    # Whether `needed` is the adaptive count of a reached confidence (it then
    # ended the loop: `AdaptiveCount`) rather than the trial budget.
    reached = false
    trial = 0
    effective = 0

    _prepare!(problem, config.max_trials)

    # `needed` is initialized to the budget and, for an adaptive sampler,
    # clamped at `max_trials` by `_adaptive_count`, so a separate
    # `trial < max_trials` guard would be dead. The single bound is sufficient.
    while trial < needed
        trial += 1

        old_best = best
        accepted, best = _score_trial!(
            rng, ws, problem, scoring, local_optimization_strategy, best,
            score_guard, on_hypothesis, trial
        )
        accepted || continue
        effective += 1

        if best[2] > old_best[2] && ws.has_best
            # The adaptive count from the incumbent's consensus size k* (the
            # count of its mask, read without a bit count).
            adapts(s) && ((needed, reached) = _adaptive_count(ws.best_k, length(ws.best_mask), K, config))
            # Anytime-trace hook: fires only on acceptance of a new tracked
            # best (rare); the `nothing` default is specialized away entirely.
            on_improvement === nothing || on_improvement(trial, best[2], ws)
        end
    end

    sar = trial > 0 ? Float64(effective) / Float64(trial) : 1.0
    return _finalize(scoring, ws, problem, best, sar, trial, result, _search_record(s, reached))
end

# Why a search that found a model stopped (its termination record): an
# exhaustive sampler drew every sample (`Success`, `TrialBudget`); the adaptive
# count of a reached confidence ended it (`Success`, `AdaptiveCount`); or the
# trial budget `max_trials` was drawn without reaching the confidence
# (`MaxIters`, `TrialBudget`). A search with no model is `_no_model_result`'s.
_search_record(s, reached::Bool) =
    !adapts(s) ? (ReturnCode.Success, TrialBudget) :
    reached ? (ReturnCode.Success, AdaptiveCount) : (ReturnCode.MaxIters, TrialBudget)

# =============================================================================
# _draw_trial! — ONE minimal-sample trial (shared by `ransac` and `_draw_hypotheses`)
# =============================================================================

"""
    _draw_trial!(f, rng, indices, problem, acc) -> (accepted::Bool, n_solutions::Int, acc)

One minimal-sample trial: `draw_sample!` into `indices`, the problem's
`test_sample`, `sample_models`, then per solution `sample_hypothesis` and `test_model`.
Every surviving model is folded into `acc` as `acc = f(acc, model)`, in solver
order. `accepted` is whether `test_sample` accepted the sample; `n_solutions` the
number of solver solutions (0 when it did not). The one trial of the stack:
`ransac` folds its incumbent through it, `_draw_hypotheses` the hypotheses it
keeps. A fold, not a closure over mutated locals, so the hot loop boxes nothing.
"""
@inline function _draw_trial!(f::F, rng::Random.AbstractRNG, indices::AbstractVector{Int}, problem, acc) where {F}
    draw_sample!(rng, indices, problem)
    test_sample(problem, indices) || return (false, 0, acc)
    solutions = sample_models(problem, indices)
    for solution in solutions
        model = sample_hypothesis(problem, solution, indices)
        model === nothing && continue
        test_model(problem, model, indices) || continue
        acc = f(acc, model)
    end
    return (true, length(solutions), acc)
end

# One RANSAC trial: the shared trial, each surviving model raced against the
# two-level incumbent `best` (`_try_model!`).
function _score_trial!(
        rng, ws, problem, scoring::AbstractScoring,
        local_optimization_strategy, best::Tuple{Float64, Float64},
        score_guard::Bool, on_hypothesis, trial::Int
    )
    accepted, _, best = _draw_trial!(rng, ws.sample_indices, problem, best) do (best_g, best_l), model
        on_hypothesis === nothing || on_hypothesis(trial, model)
        _try_model!(
            ws, problem, scoring, local_optimization_strategy,
            model, best_g, best_l, score_guard
        )
    end
    return accepted, best
end

# =============================================================================
# Two-level gating
# =============================================================================

"""
    _track_best!(ws, scoring, model, (lo_code, lo_rule) = (ReturnCode.Default, NoRule))

Copy the scoring buffers (residuals, per-point scores, consensus mask and
order, k*) into the incumbent's, so `_finalize` reads the incumbent's scale and
soft weights with no re-score; the model's local-optimization record goes with
it (SciMLBase's `Default`, not run, when none ran).
"""
function _track_best!(ws::RansacWorkspace, scoring, model, lo::Tuple{ReturnCode.T, StopRule} = _LO_NOT_RUN)
    buf = ws.buf
    ws.best_model = model
    ws.has_best = true
    ws.best_lo_retcode, ws.best_lo_stop_rule = lo
    copyto!(ws.best_residuals, buf.residuals)
    copyto!(ws.best_scores, buf.scores)
    copyto!(ws.best_mask, buf.mask)
    copyto!(ws.best_order, consensus_order(scoring, buf.scoring_ws))
    ws.best_k = inlier_count(scoring, buf.scoring_ws)
    return nothing
end

"""
    _try_model!(ws, problem, scoring, local_optimization_strategy, model, best_g, best_l, score_guard)

Two-level gating with S_gl and S_lo.

Phase 1: the scored form of the model (`scored_model`: an `Uncertain` under a
predictive scorer, `nothing` for a degenerate sample — skipped like a failed
solve), scored (`score_model!`, its consensus mask deferred until the gate
succeeds; a bounded-sum scorer may stop early against `best_g`), gated on
S > S_gl and on `test_consensus`.

Phase 2: local optimization on the consensus set.

Phase 3: the refined model re-scored under the same scorer, gated on S_lo.
"""
function _try_model!(
        ws::RansacWorkspace{<:Any, M}, problem,
        scoring::AbstractScoring, local_optimization_strategy::LO,
        model::M, best_g::Float64, best_l::Float64,
        score_guard::Bool = false
    ) where {M, LO}
    buf = ws.buf
    model_scored = scored_model(scoring, problem, model, ws.sample_indices)
    model_scored === nothing && return (best_g, best_l)
    score_g, k_star = score_model!(
        buf, scoring, problem, model_scored;
        sample = ws.sample_indices, mask = false, bound = best_g
    )
    (!_holds_consensus(problem, k_star) || score_g <= best_g) && return (best_g, best_l)
    consensus_mask!(buf, scoring)

    test_consensus(problem, model, buf.mask) || return (best_g, best_l)

    # Phases 2–3: local optimization on the consensus set, the refined model
    # re-scored under the same scorer (`_lo_and_rescore!`). `nothing`: the
    # refined model has no covariance on its refit set — dropped, as a failed
    # refit is.
    out = _lo_and_rescore!(
        ws, problem, scoring, local_optimization_strategy, model, model_scored, k_star, ws.sample_indices, score_g
    )
    out === nothing && return (best_g, best_l)
    lo_est, score_l = out
    lo_model = lo_est.u

    # ScoreGuarded (per-candidate; kwarg `score_guard`): the refined model
    # scores below its own seed under the SAME scorer — demote the candidate
    # to its RAW hypothesis at score_g. The raw model already succeeded at
    # Phase 1's test_consensus on this identical set; re-score to refresh the
    # buffers for `_track_best!`.
    if score_guard && score_l < score_g
        score_g <= best_l && return (best_g, best_l)
        score_model!(buf, scoring, problem, model_scored; sample = ws.sample_indices)
        _track_best!(ws, scoring, model, (lo_est.retcode, SeedKept))
        return (score_g, score_g)
    end

    score_l <= best_l && return (best_g, best_l)

    # Re-validate the refined model on its new consensus set: a local
    # optimization can pull the model onto a wider consensus that breaks the
    # per-problem feasibility predicate (a bow-tie homography; for a
    # `BoundedProblem` the extent geometry check).
    test_consensus(problem, lo_model, buf.mask) || return (best_g, best_l)

    _track_best!(ws, scoring, lo_model, (lo_est.retcode, lo_est.stop_rule))
    return (score_g, score_l)
end

# The Phase-1 support gate of every candidate and seed: a consensus set no
# larger than the minimal sample holds no consensus beyond the points that
# made it. One point more than the minimal sample: `_holds_minimal_sample` at m + 1.
_holds_consensus(problem, k::Int) = _holds_minimal_sample(minimal_sample_size(problem) + 1, k)

# Local optimization of `model` from the consensus set of the score that just
# ran on it (k* = `k`) — the Phase-2 step of `_try_model!` and of
# `_locally_optimize_seed!`. A set of k* = m + 1 points has ν = (k* − (m + 1))·d_g
# = 0 residual dof: no scale to anchor a loss at, so the model comes back
# unrefined (`:thin_partition`). Every consensus set an optimizer receives is
# stamped here from the score that just ran, so it is fresh by construction.
function _lo_on_partition!(ws, problem, model, scoring, k::Int, lo, score::Real = NaN)
    _holds_stable_partition(problem, k) || return lo_estimate(model, ReturnCode.InitialFailure, NoRule)
    buf = ws.buf
    cs = consensus_set(
        scoring, buf.scoring_ws, buf.residuals;
        model_dof = minimal_sample_size(problem) + 1, d_g = codimension(problem), epoch = buf.epoch, score
    )
    return locally_optimize!(ws.lo_ws, lo, problem, model, cs, buf, scoring)
end

"""
    rescore_and_optimize!(L, lo, problem, θ, buf::ScoreBuffers, scoring) -> FitResult

The loop's Phase-2 step from a score under ANOTHER scoring: re-score `θ` under
`scoring` on the buffers `buf` (`score_model!`), then — when its consensus set
is stable (more than `m + 1` points; otherwise `θ` comes back unrefined,
`:thin_partition`) — run `lo` on its workspace `L` from that consensus set. The
building block of a strategy that refits at a scoring of its own inside a loop
(VisualGeometryToolkitEval's `RefitHalfwidth`).
"""
function rescore_and_optimize!(L, lo, problem, θ, buf::ScoreBuffers, scoring)
    score, k = score_model!(buf, scoring, problem, θ)
    return _lo_on_partition!((; buf, lo_ws = L), problem, θ, scoring, k, lo, score)   # the loop's own Phase-2 step
end

# Phases 2–3 of a model whose Phase-1 score (k* = `k`) is in the buffers — THE
# path of `_try_model!` (a minimal-sample hypothesis) and of
# `_locally_optimize_seed!` (a caller's seed); only their acceptance rules
# differ. Phase 3 re-scores the refined model — under a predictive scorer WITH
# its refit covariance on the consensus set it was refit on (the mask). A model
# handed back UNREFINED was never refit on that set, so it is scored exactly as
# Phase 1 scored it — `model_scored`, with the minimal sample `sample_idx`
# (`nothing` for a seed) zeroed. An optimizer that left the refined model's
# residuals in the buffers (`residuals_current`) saves the recomputation.
#
# Returns `(lo_est, score_l)`, or `nothing` when the refined model has no
# covariance on its refit set (a degenerate consensus): it is dropped.
function _lo_and_rescore!(ws, problem, scoring, lo, model, model_scored, k::Int, sample_idx, score::Real)
    lo_est = _lo_on_partition!(ws, problem, model, scoring, k, lo, score)
    lo_model = lo_est.u
    unrefined = lo_model === model
    lo_scored = unrefined ? model_scored : _covariance_scored(scoring, problem, lo_model, ws.buf.mask)
    lo_scored === nothing && return nothing
    score_l, _ = score_model!(
        ws.buf, scoring, problem, lo_scored;
        sample = unrefined ? sample_idx : nothing,
        reuse_residuals = lo_est.metadata.residuals_current
    )
    return (lo_est, score_l)
end

# A model's scored form on its consensus mask: the plain model (converted into
# the workspace's model type — the storage-boundary labeling, an unlabeled map
# into a labeled alias) under a model-certain scorer; under a predictive one the
# model WITH the constraint covariance on `mask` (`scored_model` on its
# indices), or `nothing` when there is none.
_covariance_scored(scoring, p, θ, mask::BitVector) =
    is_predictive(scoring) ? scored_model(scoring, p, convert(model_type(p), θ), findall(mask)) :
    convert(model_type(p), θ)

# =============================================================================
# LocalOptimization — seeded refine, no sampling
# =============================================================================

# A SEED (`LocalOptimization`'s, each `FixedModels` candidate) has no minimal
# sample. Under a predictive scorer it is scored with the covariance it brings
# (an `Uncertain`, unit scale) or, as a plain model, with the constraint
# covariance on its own consensus set: scored model-certain first to find that
# set. `nothing`: no covariance on the consensus — the seed is not scored. Local
# optimization, consensus and the incumbent always work on the plain model.
_seed_scored!(ws, p, scoring, θ) = is_predictive(scoring) ? _seed_scored_predictive!(ws, p, scoring, θ) : geom(θ)
_seed_scored_predictive!(_ws, _p, _scoring, θ::Uncertain) = θ   # it brings its own covariance
function _seed_scored_predictive!(ws, p, scoring, θ)
    score_model!(ws.buf, scoring, p, θ)
    return _covariance_scored(scoring, p, θ, ws.buf.mask)
end
# The storage-boundary `convert` (labels an unlabeled map into the workspace's
# model type), applied to an uncertain seed's geometry.
# A seed as the inner loop's model: the problem's refine contract reads it (`guess_model`: its type, its frames, a
# physical guess stripped in metres), an `Uncertain` seed's covariance kept.
_seed_model(problem, θ) = guess_model(problem, θ)
_seed_model(problem, θ::Uncertain) = Uncertain(guess_model(problem, geom(θ)), cov(θ))

# Score every candidate against the consensus set, keep the single best, then run
# the (single-model) local optimization on that winner only.
function _select_then_optimize!(
        ws::RansacWorkspace{K, M, T}, problem,
        scoring::AbstractScoring, lo::Union{Nothing, AbstractOptimizationStrategy},
        models::FixedModels; result::RansacResultBuffers = _result_buffers(ws, problem)
    ) where {K, M, T}
    _prepare!(problem)
    best_θ = nothing
    best_score = T(-Inf)
    for θ in models
        scored = _seed_scored!(ws, problem, scoring, θ)
        scored === nothing && continue
        score_g, k_star = score_model!(ws.buf, scoring, problem, scored)
        if _holds_consensus(problem, k_star) && test_consensus(problem, geom(θ), ws.buf.mask) && score_g > best_score
            best_score = score_g
            best_θ = θ
        end
    end
    # Nothing cleared consensus → no_model (ws.has_best is still false).
    best_θ === nothing &&
        return _finalize(scoring, ws, problem, (best_score, best_score), 1.0, 1, result, _CLOSED_FORM)
    # Locally optimize the chosen winner only (it re-scores the seed internally).
    return _locally_optimize_seed!(ws, problem, scoring, lo, best_θ; result)
end

# `trial`, `sar`, `result`, `unscored` and `search` are a terminal placement's
# (`_ransac_placed`): the raw race's trial count and acceptance rate, its result
# buffers, the score its incumbent finalizes at when the seed cannot be
# scored (its `has_best` is then still the race's), and its termination record.
# A seeded refine's own are one trial, every sample accepted, fresh buffers, no
# incumbent and no search (`nothing`).
#
# The termination record of a seeded refine: the kept refined model's local
# optimization's record (the LO result's core: an iterative optimizer's verdict,
# `(Success, ClosedForm)` for no LO); a refit that did not run (`InitialFailure`:
# a consensus too thin to refine) handed the seed back, and the seed kept is
# `(Success, ClosedForm)`. A terminal
# placement keeps its search's record unless that record is not successful
# (`_terminal_record`).
function _locally_optimize_seed!(
        ws::RansacWorkspace{K, M, T}, problem,
        scoring::AbstractScoring, lo::Union{Nothing, AbstractOptimizationStrategy}, seed::Union{M, Uncertain{M}};
        score_guard::Bool = false, trial::Int = 1, sar::Float64 = 1.0,
        result::RansacResultBuffers = _result_buffers(ws, problem),
        unscored::Tuple = (T(-Inf), T(-Inf)), search = nothing
    ) where {K, M, T}
    _prepare!(problem)
    θ₀ = geom(seed)
    scored = _seed_scored!(ws, problem, scoring, seed)
    # No covariance on the seed's consensus: not scored (`has_best` is false, or
    # a terminal placement's raw incumbent).
    scored === nothing && return _finalize(scoring, ws, problem, unscored, sar, trial, result, _terminal_record(search, _CLOSED_FORM))
    score_g, k_star = score_model!(ws.buf, scoring, problem, scored)
    if _holds_consensus(problem, k_star) && test_consensus(problem, θ₀, ws.buf.mask)
        # Phases 2–3 exactly as a RANSAC candidate runs them; only the
        # acceptance differs: the seed is kept whenever the refined model is not.
        out = _lo_and_rescore!(ws, problem, scoring, lo, θ₀, scored, k_star, nothing, score_g)
        if out !== nothing
            lo_est, score_l = out
            lo_model = lo_est.u
            # score_guard: keep the refined model only if it does not score
            # below the seed under the same scorer.
            if test_consensus(problem, lo_model, ws.buf.mask) && (!score_guard || score_l >= score_g)
                _track_best!(ws, scoring, lo_model, (lo_est.retcode, lo_est.stop_rule))
                record = _terminal_record(search, _refit_record(lo_est))
                return _finalize(scoring, ws, problem, (score_g, score_l), sar, trial, result, record)
            end
        end
        score_model!(ws.buf, scoring, problem, scored)
    end
    _track_best!(ws, scoring, θ₀)
    return _finalize(scoring, ws, problem, (score_g, score_g), sar, trial, result, _terminal_record(search, _CLOSED_FORM))
end

const _CLOSED_FORM = (ReturnCode.Success, ClosedForm)

# The record a kept refit gives its result: its own, or — a refit that did not run
# (`InitialFailure`, the seed handed back) — the seed's.
_refit_record(lo_est) = lo_est.retcode == ReturnCode.InitialFailure ? _CLOSED_FORM : (lo_est.retcode, lo_est.stop_rule)

# A seeded refine's record is its own (`search === nothing`); a terminal
# placement keeps its search's, unless its LO did not succeed (the more severe).
_terminal_record(::Nothing, record::Tuple{ReturnCode.T, StopRule}) = record
_terminal_record(search::Tuple{ReturnCode.T, StopRule}, record::Tuple{ReturnCode.T, StopRule}) =
    VgtkCore.successful_retcode(record[1]) ? search : worst_retcode(search, record)

# =============================================================================
# _finalize — the result from the incumbent's buffers
# =============================================================================

# The result of a search with no model — every sample failed, or (with
# `trials = 0`, `sar = NaN`) too few data for one sample: no estimate, zero
# residuals and weights (an all-false inlier mask), no scale; its record
# `(Failure, TrialBudget)` (the legacy `:no_model`).
function _no_model_result(::Type{M}, scoring::S, result::RansacResultBuffers{T}, trials::Int, sar::Float64) where {M, S, T}
    fill!(result.residuals, zero(T)); fill!(result.weights, zero(T))
    return FitResult{Union{M, Nothing}}(
        nothing, ReturnCode.Failure, TrialBudget, SolveStats(; nsteps = trials), NaN, 0,
        RansacMetadata{T, S}(result.weights, -Inf, NaN, ScaleNone, 0, sar, _LO_NOT_RUN...); resid = result.residuals
    )
end

function _finalize(
        scoring::S, ws, problem::AbstractRansacProblem, best::Tuple, sar, trial, result::RansacResultBuffers,
        (code, rule)::Tuple{ReturnCode.T, StopRule}
    ) where {S}
    M = model_type(problem)
    T = eltype(ws.best_residuals)   # attributes come back in the workspace's storage precision
    p = minimal_sample_size(problem)
    best_score = best[2]

    ws.has_best || return _no_model_result(M, scoring, result, trial, sar)

    model = ws.best_model
    d_g = codimension(problem)
    # The incumbent's consensus set is best_order[1:n_in] (its mask).
    n_in = ws.best_k
    nu = residual_dof(problem, n_in)       # (n_in − m)·d_g, 0 on a thin set
    # The scorer ORIGINATES the scale: the consensus set's Student-t scale under
    # its prior at the model's own dof (`consensus_s2` at m): √(RSS/ν) under
    # Jeffreys, floored at √eps. There is no set to scale at ν ≤ 0: ScaleNone.
    # An estimated scale is finite: a non-finite one (an overflowing RSS) is
    # dropped to ScaleNone and recorded (`nonfinite_scale`).
    s2 = consensus_s2(scoring, ws.best_residuals, ws.best_order, n_in; model_dof = p, d_g)
    scale, source, nonfinite_scale = _estimated_scale(nu, s2)

    # The weights wᵢ ∈ [0, 1], of the scorer's kind: its posterior, or 0/1.
    inlier_probabilities!(result.weights, scoring, ws.best_scores, ws.best_order, n_in; model_dof = p, d_g)
    copyto!(result.residuals, ws.best_residuals)
    n, rms = _support_rms!(VgtkCore.weight_kind(S), result.weights, result.residuals, ws.best_order, ws.best_residuals, n_in)
    # The explicit model type: the outer constructor would infer it from the model
    # and lose the union a failed search needs.
    return FitResult{Union{M, Nothing}}(
        model, code, rule, SolveStats(; nsteps = trial), rms, n,
        RansacMetadata{T, S}(
            result.weights, best_score, scale, source, nu, sar, ws.best_lo_retcode, ws.best_lo_stop_rule, nonfinite_scale
        ); resid = result.residuals
    )
end

# The scale a result records from its partition's `s2` at `nu` dof: `(scale, source, nonfinite_scale)` —
# `(√s2, ScaleEstimated, false)`, or `(NaN, ScaleNone, false)` with no set to scale (ν ≤ 0), or
# `(NaN, ScaleNone, true)` where `√s2` is not finite: an estimated scale is finite.
function _estimated_scale(nu::Integer, s2::Real)
    nu > 0 || return NaN, ScaleNone, false
    σ = sqrt(Float64(s2))
    return isfinite(σ) ? (σ, ScaleEstimated, false) : (NaN, ScaleNone, true)
end

# The FitResult's `n` and `rms` from the weights (the contract, §12): `n` the
# support `count(>(0), w)` and `rms = √(Σ wᵢ rᵢ² / Σ wᵢ)`. Binary weights are the
# consensus membership, so these are the consensus set's size and rms — computed
# over `order[1:k]` exactly as before the weights carried them (bit for bit).
function _support_rms!(::BinaryWeights, _w, _r, order, residuals, k::Int)
    return k, (k > 0 ? sqrt(partition_rss(order, residuals, k) / k) : NaN)
end
# Posterior weights: the one support-and-rms rule of every posterior-weighted result.
_support_rms!(::PosteriorWeights, w::AbstractVector, r::AbstractVector, _order, _residuals, _k::Int) =
    _posterior_support_rms!(w, (abs2(Float64(x)) for x in r))

# The `n` and `rms` of every weighted result is Core's `weighted_support_rms` (robust/refine.jl).
# Posterior weights: a posterior below `eps(T)` is first stored in `w` as an exact 0
# (out of the support), then the one rule.
function _posterior_support_rms!(w::AbstractVector{T}, sq) where {T}
    @inbounds for i in eachindex(w)
        w[i] < eps(T) && (w[i] = zero(T))
    end
    return weighted_support_rms(w, sq)
end

# =============================================================================
# Adaptive Trial Count (Hypergeometric, exact)
# =============================================================================

"""
    _p_all_inliers(n_inliers, n_total, k) -> Float64

Probability that all `k` draws are inliers when sampling without replacement
from `n_total` items with `n_inliers` successes.
"""
function _p_all_inliers(n_inliers::Int, n_total::Int, k::Int)
    n_inliers < k && return 0.0
    p = 1.0
    @inbounds for i in 0:(k - 1)
        p *= (n_inliers - i) / (n_total - i)
    end
    return p
end

# The adaptive trial count and whether the confidence `success_prob` is reached
# within the budget: `(count, reached)`. `reached` is whether the confidence count
# — the trials the confidence needs, before `min_trials` raises it — is at most
# `max_trials` (a count equal to the budget reached it AT the cap); it is false when
# the count is beyond the budget or undefined, the budget then drawn without it.
function _adaptive_count(n_inliers::Int, n_total::Int, k::Int, config::RansacConfig)
    p_success = _p_all_inliers(n_inliers, n_total, k)
    p_success ≈ 0 && return (config.max_trials, false)
    p_failure = 1 - p_success
    p_failure ≈ 0 && return (config.min_trials, true)
    # log(1 − η)/log(p_failure) in Float64 BEFORE any Int conversion: it is +Inf
    # for η = 1 (certainty needs every trial) and ±Inf/NaN when p_failure rounds
    # to 1. `ceil(Int, ±Inf)` would throw an InexactError; any count beyond the
    # budget, or undefined, is the budget.
    trials = log1p(-config.success_prob) / log(p_failure)
    0 <= trials <= config.max_trials || return (config.max_trials, false)
    return (clamp(ceil(Int, trials), config.min_trials, config.max_trials), true)
end
