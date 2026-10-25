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
    ransac(problem, scoring; local_optimization, config, workspace, rng) -> RansacEstimate

RANSAC on VisualGeometryToolkitCore's scoring and local-optimization contracts: the
loop calls only the contract verbs, so any scorer composes with any local
optimizer and any problem. The allocating convenience of [`ransac!`](@ref)
(fresh result buffers, and a fresh workspace unless one is supplied).

For each minimal sample, fits θₜ, scores it, and takes the scorer's consensus
set.
Two independent best-so-far scores are maintained: S_gl gates entry into local
optimization; S_lo tracks the best locally optimized score. The adaptive trial
count uses the hypergeometric distribution (the stopping criterion).

# Arguments
- `problem::AbstractRansacProblem`: Problem definition (data, solver, residuals)
- `scoring::AbstractScoring`: any scorer of the contract — VisualGeometryToolkitCore's
  `MsacScoring`/`RansacScoring` at a threshold, or a scorer from another package
  (a paper's: RANSACScoringDoneRight's Student-t + uniform marginal)

# Keyword Arguments
- `local_optimization::AbstractLocalOptimization=default_local_optimization(scoring)`:
  the LO-RANSAC refinement strategy (default `NoLocalOptimization()`); dispatched
  in the loop's Phase 2 via `locally_optimize!`, or — a [`TerminalLO`](@ref)
  ([`lo_placement`](@ref)) — once on the final incumbent of a raw race
- `config::RansacConfig=RansacConfig()`: max_trials, success_prob η, min_trials
- `workspace::Union{Nothing,RansacWorkspace}=nothing`: Pre-allocated workspace
- `rng::Random.AbstractRNG=Random.default_rng()`: source of every random draw
  the loop makes (sample selection through `draw_sample!`). Supply a seeded rng
  (e.g. `Xoshiro(seed)`) for a bit-reproducible run that is independent of
  ambient RNG state; the default preserves the historical task-local behavior.
- `on_improvement=nothing`: optional anytime-trace hook, called as
  `on_improvement(trial::Int, score, ws::RansacWorkspace)` exactly when the
  tracked best (the acceptance that also re-arms the adaptive
  trial count) is replaced — read `ws.best_model` / `ws.best_lo_retcode`
  inside the callback. Improvements are rare (≈ log-many per run), so a cheap
  callback keeps overhead negligible; `nothing` (production default) is
  specialized away at compile time.
- `score_guard::Bool=false`: per-candidate ScoreGuarded acceptance — the
  in-loop analogue of `locally_optimize(...; score_guard=true)` (the
  review-requested LO-RANSAC score-monotone variant). When the refined model
  of a gate-satisfying candidate scores BELOW its own seed under the same
  scorer, the candidate re-enters the incumbent race as the RAW hypothesis
  (at its Phase-1 score) instead of the refined model; the race itself stays
  the standard monotone best-so-far comparison. A raw hypothesis kept this
  way is stamped `lo_retcode = :guard_kept_seed` on the result.

# Returns
A [`RansacEstimate`](@ref); `iterations(result)` is the number of trials
(hypotheses drawn), whether the adaptive stop or `max_trials` ended the loop —
both return `:converged`. Test it with VisualGeometryToolkitCore's `converged(result)`
before reading the model: `retcode(result) === :no_model` (`converged` false)
means NO hypothesis survived — there is no estimate, and `estimate(result)` is a
`zero(M)` placeholder kept only for type stability, never a model (with
`quality = -Inf`, an all-false `inlier_mask`, `rms = sigma = NaN`).

# Examples
```julia
result = ransac(problem, MsacScoring(problem; threshold = 2.0); local_optimization = Irls{TLSShape}())
```
"""
function ransac(
        problem::AbstractRansacProblem,
        scoring::AbstractScoring;
        local_optimization::AbstractLocalOptimization = default_local_optimization(scoring),
        config::RansacConfig = RansacConfig(),
        workspace = nothing,
        on_improvement = nothing,
        # Per-hypothesis trace hook, called as `on_hypothesis(trial::Int, model)`
        # for EVERY candidate model that succeeds at `test_model` (before scoring).
        on_hypothesis = nothing,
        score_guard::Bool = false,
        # Every random draw the loop makes goes through `rng`, so
        # `rng = Xoshiro(seed)` makes the run bit-reproducible.
        rng::Random.AbstractRNG = Random.default_rng()
    )
    _check_scoring(problem, scoring)
    ws = _entry_workspace(workspace, problem, scoring, local_optimization)
    return ransac!(
        _result_buffers(ws, problem), ws, problem, scoring;
        local_optimization, config, on_improvement, on_hypothesis, score_guard, rng
    )
end

"""
    ransac!(result::RansacResultBuffers, ws::RansacWorkspace, problem, scoring;
            local_optimization, config, on_improvement, on_hypothesis, score_guard, rng) -> RansacEstimate

The zero-allocation form of [`ransac`](@ref): RANSAC on the preallocated
workspace `ws` (trimmed to the problem's data size), its result written into the
preallocated `result` buffers — the returned estimate's `inlier_mask`,
`residuals` and `weights` ARE those buffers.
"""
function ransac!(
        result::RansacResultBuffers, ws::RansacWorkspace, problem::AbstractRansacProblem,
        scoring::AbstractScoring;
        local_optimization::AbstractLocalOptimization = default_local_optimization(scoring),
        config::RansacConfig = RansacConfig(),
        on_improvement = nothing,
        on_hypothesis = nothing,
        score_guard::Bool = false,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    _check_scoring(problem, scoring)
    _entry_workspace(ws, problem, scoring, local_optimization)
    _set_n!(result, data_size(problem))
    return _ransac_placed(
        lo_placement(local_optimization),
        ws, problem, scoring, local_optimization, config, on_improvement,
        score_guard, rng, on_hypothesis, result
    )
end

# The strategy's placement (`lo_placement`), dispatched statically: in-loop is
# the loop itself; terminal is the loop with no LO, then one seeded LO of its
# incumbent (`TerminalLO`).
@inline _ransac_placed(::InLoopPlacement, args...) = _ransac_loop(args...)

function _ransac_placed(
        ::TerminalPlacement, ws::RansacWorkspace, problem, scoring,
        local_optimization, config, on_improvement,
        score_guard::Bool, rng::Random.AbstractRNG, on_hypothesis, result
    )
    score_guard && throw(
        ArgumentError(
            "score_guard is a per-candidate acceptance rule; a terminal local optimization " *
                "($(nameof(typeof(local_optimization)))) runs on no candidate, only on the final incumbent"
        )
    )
    raw = _ransac_loop(
        ws, problem, scoring, NoLocalOptimization(), config, on_improvement,
        false, rng, on_hypothesis, result
    )
    converged(raw) || return raw
    # `has_best` still holds the raw incumbent: a seed the scorer cannot score
    # finalizes as it, at its race score.
    q = raw.quality
    return _locally_optimize_seed!(
        ws, problem, scoring, terminal_strategy(local_optimization), estimate(raw);
        trial = iterations(raw), sar = raw.sample_acceptance_rate, result, unscored = (q, q)
    )
end

# The workspace every entry runs on — `ransac` and both `locally_optimize`
# methods. The caller's, or a fresh one built for the problem, the scorer and
# the local optimizer (NOTE: `something(workspace, RansacWorkspace(…))` would
# build one every call — Julia evaluates both arguments eagerly). A shared
# workspace may be sized for a larger problem, so every internal buffer is
# trimmed to `data_size(problem)`. And the previous call's incumbent is cleared:
# a run that never accepts a candidate would otherwise `_finalize` the stale
# `best_model` as :converged. The composition is checked once, here: a scorer's
# workspace built for another scorer or configuration, and a local optimizer that follows
# a scorer with no threshold.
function _entry_workspace(workspace, problem, scoring, lo)
    ws = workspace === nothing ? RansacWorkspace(problem, scoring, lo) : workspace
    set_n!(ws, data_size(problem))
    ws.has_best = false
    ws.best_lo_retcode = :not_run
    check_workspace(scoring, ws.buf.scoring_ws)
    check_composition(scoring, lo)
    return ws
end

_result_buffers(ws::RansacWorkspace{<:Any, <:Any, T}, problem) where {T} = RansacResultBuffers{T}(data_size(problem))

# Function barrier: ws has fully concrete type {K,M,T,W,L} after dispatch
function _ransac_loop(
        ws::RansacWorkspace{K}, problem, scoring,
        local_optimization, config, on_improvement,
        score_guard::Bool, rng::Random.AbstractRNG, on_hypothesis, result
    ) where {K}
    best = (-Inf, -Inf)
    # The sampler owns the schedule (`trial_budget`, `adapts`): `max_trials`,
    # re-armed adaptively, for a random sampler; every sample exactly once,
    # never re-armed, for the `ExhaustiveSampler`.
    s = sampler(problem)
    needed = trial_budget(s, config)
    trial = 0
    effective = 0

    _prepare!(problem, config.max_trials)

    # `needed` is initialized to the budget and, for an adaptive sampler,
    # clamped at `max_trials` by `_adaptive_trials`, so a separate
    # `trial < max_trials` guard would be dead. The single bound is sufficient.
    while trial < needed
        trial += 1

        old_best = best
        accepted, best = _score_trial!(
            rng, ws, problem, scoring, local_optimization, best,
            score_guard, on_hypothesis, trial
        )
        accepted || continue
        effective += 1

        if best[2] > old_best[2] && ws.has_best
            # The adaptive count from the incumbent's consensus size k* (the
            # count of its mask, read without a bit count).
            adapts(s) && (needed = _adaptive_trials(ws.best_k, length(ws.best_mask), K, config))
            # Anytime-trace hook: fires only on acceptance of a new tracked
            # best (rare); the `nothing` default is specialized away entirely.
            on_improvement === nothing || on_improvement(trial, best[2], ws)
        end
    end

    sar = trial > 0 ? Float64(effective) / Float64(trial) : 1.0
    return _finalize(scoring, ws, problem, best, sar, trial, result)
end

# =============================================================================
# _draw_trial! — ONE minimal-sample trial (shared by `ransac` and `draw_hypotheses`)
# =============================================================================

"""
    _draw_trial!(f, rng, indices, problem, acc) -> (accepted::Bool, n_solutions::Int, acc)

One minimal-sample trial: `draw_sample!` into `indices`, the problem's
`test_sample`, `solve`, then per solution `fit_hypothesis` and `test_model`.
Every surviving model is folded into `acc` as `acc = f(acc, model)`, in solver
order. `accepted` is whether `test_sample` accepted the sample; `n_solutions` the
number of solver solutions (0 when it did not). The one trial of the stack:
`ransac` folds its incumbent through it, `draw_hypotheses` the hypotheses it
keeps. A fold, not a closure over mutated locals, so the hot loop boxes nothing.
"""
@inline function _draw_trial!(f::F, rng::Random.AbstractRNG, indices::AbstractVector{Int}, problem, acc) where {F}
    draw_sample!(rng, indices, problem)
    test_sample(problem, indices) || return (false, 0, acc)
    solutions = solve(problem, indices)
    for solution in solutions
        model = fit_hypothesis(problem, solution, indices)
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
        local_optimization, best::Tuple{Float64, Float64},
        score_guard::Bool, on_hypothesis, trial::Int
    )
    accepted, _, best = _draw_trial!(rng, ws.sample_indices, problem, best) do (best_g, best_l), model
        on_hypothesis === nothing || on_hypothesis(trial, model)
        _try_model!(
            ws, problem, scoring, local_optimization,
            model, best_g, best_l, score_guard
        )
    end
    return accepted, best
end

# =============================================================================
# Two-level gating
# =============================================================================

"""
    _track_best!(ws, scoring, model, lo_retcode)

Copy the scoring buffers (residuals, per-point scores, consensus mask and
order, k*) into the incumbent's, so `_finalize` reads the incumbent's scale and
soft weights with no re-score.
"""
function _track_best!(ws::RansacWorkspace, scoring, model, lo_stop_reason::Symbol = :not_run)
    buf = ws.buf
    ws.best_model = model
    ws.has_best = true
    ws.best_lo_retcode = lo_stop_reason
    copyto!(ws.best_residuals, buf.residuals)
    copyto!(ws.best_scores, buf.scores)
    copyto!(ws.best_mask, buf.mask)
    copyto!(ws.best_order, consensus_order(scoring, buf.scoring_ws))
    ws.best_k = inlier_count(scoring, buf.scoring_ws)
    return nothing
end

"""
    _try_model!(ws, problem, scoring, local_optimization, model, best_g, best_l, score_guard)

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
        scoring::AbstractScoring, local_optimization::LO,
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
        ws, problem, scoring, local_optimization, model, model_scored, k_star, ws.sample_indices, score_g
    )
    out === nothing && return (best_g, best_l)
    lo_est, score_l = out
    lo_model = estimate(lo_est)

    # ScoreGuarded (per-candidate; kwarg `score_guard`): the refined model
    # scores below its own seed under the SAME scorer — demote the candidate
    # to its RAW hypothesis at score_g. The raw model already succeeded at
    # Phase 1's test_consensus on this identical set; re-score to refresh the
    # buffers for `_track_best!`.
    if score_guard && score_l < score_g
        score_g <= best_l && return (best_g, best_l)
        score_model!(buf, scoring, problem, model_scored; sample = ws.sample_indices)
        _track_best!(ws, scoring, model, :guard_kept_seed)
        return (score_g, score_g)
    end

    score_l <= best_l && return (best_g, best_l)

    # Re-validate the refined model on its new consensus set: a local
    # optimization can pull the model onto a wider consensus that breaks the
    # per-problem feasibility predicate (a bow-tie homography; for a
    # `BoundedProblem` the extent geometry check).
    test_consensus(problem, lo_model, buf.mask) || return (best_g, best_l)

    _track_best!(ws, scoring, lo_model, retcode(lo_est))
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
    _holds_stable_partition(problem, k) || return lo_estimate(model, :thin_partition)
    buf = ws.buf
    cs = consensus_set(
        scoring, buf.scoring_ws, buf.residuals;
        model_dof = minimal_sample_size(problem) + 1, d_g = codimension(problem), epoch = buf.epoch, score
    )
    return locally_optimize!(ws.lo_ws, lo, problem, model, cs, buf, scoring)
end

"""
    locally_optimize_rescored!(L, lo, problem, θ, buf::ScoreBuffers, scoring) -> FitResult

The loop's Phase-2 step from a score under ANOTHER scoring: re-score `θ` under
`scoring` on the buffers `buf` (`score_model!`), then — when its consensus set
is stable (more than `m + 1` points; otherwise `θ` comes back unrefined,
`:thin_partition`) — run `lo` on its workspace `L` from that consensus set. The
building block of a strategy that refits at a scoring of its own inside a loop
(VisualGeometryToolkitEval's `RefitHalfwidth`).
"""
function locally_optimize_rescored!(L, lo, problem, θ, buf::ScoreBuffers, scoring)
    score, k = score_model!(buf, scoring, problem, θ)
    _holds_stable_partition(problem, k) || return lo_estimate(θ, :thin_partition)
    cs = consensus_set(
        scoring, buf.scoring_ws, buf.residuals;
        model_dof = minimal_sample_size(problem) + 1, d_g = codimension(problem), epoch = buf.epoch, score
    )
    return locally_optimize!(L, lo, problem, θ, cs, buf, scoring)
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
    lo_model = estimate(lo_est)
    unrefined = lo_model === model
    lo_scored = unrefined ? model_scored : _covariance_scored(scoring, problem, lo_model, ws.buf.mask)
    lo_scored === nothing && return nothing
    score_l, _ = score_model!(
        ws.buf, scoring, problem, lo_scored;
        sample = unrefined ? sample_idx : nothing,
        reuse_residuals = lo_est.residuals_current
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
# locally_optimize — seeded refine, no sampling
# =============================================================================

# `threshold` is a parameter of the DEFAULT scoring (MSAC at that threshold) and
# nothing else: a caller that hands in its own `scoring` has already chosen its
# width. Accepting both let the two disagree with one silently ignored, so both
# is an error, and the threshold is required only when it is used. A marginal
# scorer's outlier half-width is that scorer's: `outlier_halfwidth` is refused
# with the way to say it.
function _default_scoring(problem, scoring, threshold, outlier_halfwidth = nothing)
    outlier_halfwidth === nothing || throw(
        ArgumentError(
            "`outlier_halfwidth` is no keyword of the core entries: the default scorer is MSAC at " *
                "`threshold`. A Student-t + uniform marginal's outlier half-width belongs to that " *
                "scoring — give `scoring = StudentTUniformScoring(problem, outlier_halfwidth)` " *
                "(RANSACScoringDoneRight)"
        )
    )
    if scoring !== nothing
        threshold === nothing || throw(
            ArgumentError(
                "supply `threshold` or `scoring`, not both: the threshold only " *
                    "builds the default scoring, and `scoring` carries its own width"
            )
        )
        _check_scoring(problem, scoring)
        return scoring
    end
    threshold === nothing && throw(
        ArgumentError("`threshold` is required when no `scoring` is given")
    )
    sc = _default_scoring_at(problem, Float64(threshold))
    _check_scoring(problem, sc)
    return sc
end

# A problem's refusal of a scoring it cannot mean — the ONE check every entry
# (`ransac`, both `locally_optimize` methods, every `find_*` through
# `_default_scoring`) runs on the scoring it is about to use, whoever built it.
# No-op by default; the essential problem refuses a sub-detector half-width
# (problems/essential.jl), the leftover of its calibrated-unit era.
_check_scoring(_problem, _scoring) = nothing

"""
    locally_optimize(problem, θ₀; local_optimization, threshold | scoring,
                     workspace=nothing, score_guard=false) -> RansacEstimate

Refine a caller-supplied seed model `θ₀` with one local-optimization run — the
same `score → locally_optimize! → re-score → test_consensus → _finalize`
machinery a RANSAC trial runs in `_try_model!`, driven by `θ₀` instead of a
minimal sample (no `draw_sample!`, no consensus search). Generic over the
problem type. `local_optimization` is REQUIRED (no default: the strategy is the
experiment's decision; the settled formulations live in the `find_*` pipelines).
If the refined model fails `test_consensus` the seed is returned.
Give exactly one of `threshold` (the default scoring's: MSAC at that threshold,
in the problem's residual unit) and `scoring`; both is an `ArgumentError`.
When `score_guard=true` the refined model is only accepted if it does not lower
the seed's score (the seed is kept otherwise).

The local optimizer receives its loss width through its width mode (VGC
`loss_at`): a consensus-anchored strategy (`ScaleMultiple(m)`) reads the
consensus set's ŝ², an absolute one (`LossWidth`, see `with_width`) runs at the
caller's width with the scoring untouched — the role-isolated validation
controls (solo CV, oracle-scale) deploy the strategy that way — and a strategy
with no width follows the scorer's `inlier_threshold`.
"""
function locally_optimize(
        problem, θ₀;
        threshold::Union{Real, Nothing} = nothing,
        outlier_halfwidth::Union{Real, Nothing} = nothing,   # refused, with the way to say it
        local_optimization::AbstractLocalOptimization,   # REQUIRED — no default
        scoring::Union{AbstractScoring, Nothing} = nothing,
        workspace = nothing,
        score_guard::Bool = false
    )
    sc = _default_scoring(problem, scoring, threshold, outlier_halfwidth)
    ws = _entry_workspace(workspace, problem, sc, local_optimization)
    # Storage-boundary labeling: the workspace model type M may be a labeled
    # alias (`Extrinsics3` for the absolute-pose family) while callers hand in
    # an unlabeled seed — `convert` LABELS an unlabeled map into the typed slot.
    return _locally_optimize_seed!(
        ws, problem, sc, local_optimization,
        _seed_convert(model_type(problem), θ₀); score_guard
    )
end

# A SEED (`locally_optimize`, each `FixedModels` candidate) has no minimal
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
_seed_convert(::Type{M}, θ) where {M} = convert(M, θ)
_seed_convert(::Type{M}, θ::Uncertain) where {M} = Uncertain(convert(M, geom(θ)), cov(θ))

"""
    locally_optimize(problem, models::FixedModels; local_optimization,
                     threshold | scoring, workspace=nothing) -> RansacEstimate

Entry for solvers that return several candidate models (e.g. the two planar-pose
ambiguity solutions). Each candidate is scored against the consensus set; the
single highest-scoring one that succeeds at `test_consensus` is kept, and **only that
one** is locally optimized. The candidates are never all optimized — selection
is by consensus score, optimization is on the winner alone. Returns a `:no_model`
result (`converged` false, no estimate) if no candidate clears consensus. `threshold` and `scoring`
are exclusive, as in the single-seed method.
"""
function locally_optimize(
        problem, models::FixedModels;
        threshold::Union{Real, Nothing} = nothing,
        outlier_halfwidth::Union{Real, Nothing} = nothing,   # refused, with the way to say it
        local_optimization::AbstractLocalOptimization,   # REQUIRED — no default
        scoring::Union{AbstractScoring, Nothing} = nothing,
        workspace = nothing
    )
    sc = _default_scoring(problem, scoring, threshold, outlier_halfwidth)
    # The no-consensus exit of `_select_then_optimize!` relies on `has_best`
    # being false, which `_entry_workspace` guarantees for a reused workspace too.
    ws = _entry_workspace(workspace, problem, sc, local_optimization)
    return _select_then_optimize!(ws, problem, sc, local_optimization, models)
end

# Score every candidate against the consensus set, keep the single best, then run
# the (single-model) local optimization on that winner only.
function _select_then_optimize!(
        ws::RansacWorkspace{K, M, T}, problem,
        scoring::AbstractScoring, lo::AbstractLocalOptimization,
        models::FixedModels
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
        return _finalize(scoring, ws, problem, (best_score, best_score), 1.0, 1, _result_buffers(ws, problem))
    # Locally optimize the chosen winner only (it re-scores the seed internally).
    return _locally_optimize_seed!(ws, problem, scoring, lo, best_θ)
end

# `trial`, `sar`, `result` and `unscored` are a terminal placement's
# (`_ransac_placed`): the raw race's trial count and acceptance rate, its result
# buffers, and the score its incumbent finalizes at when the seed cannot be
# scored (its `has_best` is then still the race's). A seeded refine's own are
# one trial, every sample accepted, fresh buffers and no incumbent.
function _locally_optimize_seed!(
        ws::RansacWorkspace{K, M, T}, problem,
        scoring::AbstractScoring, lo::AbstractLocalOptimization, seed::Union{M, Uncertain{M}};
        score_guard::Bool = false, trial::Int = 1, sar::Float64 = 1.0,
        result::RansacResultBuffers = _result_buffers(ws, problem),
        unscored::Tuple = (T(-Inf), T(-Inf))
    ) where {K, M, T}
    _prepare!(problem)
    θ₀ = geom(seed)
    scored = _seed_scored!(ws, problem, scoring, seed)
    # No covariance on the seed's consensus: not scored (`has_best` is false, or
    # a terminal placement's raw incumbent).
    scored === nothing && return _finalize(scoring, ws, problem, unscored, sar, trial, result)
    score_g, k_star = score_model!(ws.buf, scoring, problem, scored)
    if _holds_consensus(problem, k_star) && test_consensus(problem, θ₀, ws.buf.mask)
        # Phases 2–3 exactly as a RANSAC candidate runs them; only the
        # acceptance differs: the seed is kept whenever the refined model is not.
        out = _lo_and_rescore!(ws, problem, scoring, lo, θ₀, scored, k_star, nothing, score_g)
        if out !== nothing
            lo_est, score_l = out
            lo_model = estimate(lo_est)
            # score_guard: keep the refined model only if it does not score
            # below the seed under the same scorer.
            if test_consensus(problem, lo_model, ws.buf.mask) && (!score_guard || score_l >= score_g)
                _track_best!(ws, scoring, lo_model, retcode(lo_est))
                return _finalize(scoring, ws, problem, (score_g, score_l), sar, trial, result)
            end
        end
        score_model!(ws.buf, scoring, problem, scored)
    end
    _track_best!(ws, scoring, θ₀)
    return _finalize(scoring, ws, problem, (score_g, score_g), sar, trial, result)
end

# =============================================================================
# _finalize — the result from the incumbent's buffers
# =============================================================================

function _finalize(scoring, ws, problem, best::Tuple, sar, trial, result::RansacResultBuffers)
    M = model_type(problem)
    T = eltype(ws.best_residuals)   # attributes come back in the workspace's storage precision
    p = minimal_sample_size(problem)
    best_score = best[2]

    if !ws.has_best
        fill!(result.inlier_mask, false); fill!(result.residuals, zero(T)); fill!(result.weights, zero(T))
        return FitResult(
            zero(M), :no_model, trial, NaN, 0,
            RansacMetadata{T}(result.inlier_mask, result.residuals, result.weights, -Inf, NaN, 0, sar, :not_run)
        )
    end

    model = ws.best_model
    d_g = codimension(problem)
    # The incumbent's consensus set is best_order[1:n_in] (its mask).
    n_in = ws.best_k
    nu = residual_dof(problem, n_in)       # (n_in − m)·d_g, 0 on a thin set
    RSS = partition_rss(ws.best_order, ws.best_residuals, n_in)
    # σ̂ is the consensus set's Student-t scale under the scorer's prior at the
    # model's own dof (`consensus_s2` at m): √(RSS/ν) under Jeffreys, floored at
    # √eps; there is no set to scale at ν ≤ 0.
    s2 = consensus_s2(scoring, ws.best_residuals, ws.best_order, n_in; model_dof = p, d_g)
    sigma = nu > 0 ? sqrt(s2) : NaN

    # Soft inlier probabilities wᵢ ∈ [0, 1] (the scorer's posterior, or 0/1).
    inlier_probabilities!(result.weights, scoring, ws.best_scores, ws.best_order, n_in; model_dof = p, d_g)
    copyto!(result.inlier_mask, ws.best_mask)
    copyto!(result.residuals, ws.best_residuals)
    rms_in = n_in > 0 ? sqrt(RSS / n_in) : NaN
    return FitResult(
        model, :converged, trial, rms_in, n_in,
        RansacMetadata{T}(
            result.inlier_mask, result.residuals, result.weights,
            best_score, sigma, nu, sar, ws.best_lo_retcode
        )
    )
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

function _adaptive_trials(n_inliers::Int, n_total::Int, k::Int, config::RansacConfig)
    p_success = _p_all_inliers(n_inliers, n_total, k)
    p_success ≈ 0 && return config.max_trials
    p_failure = 1 - p_success
    p_failure ≈ 0 && return config.min_trials
    # log(1 − η)/log(p_failure) in Float64 BEFORE any Int conversion: it is +Inf
    # for η = 1 (certainty needs every trial) and ±Inf/NaN when p_failure rounds
    # to 1. `ceil(Int, ±Inf)` would throw an InexactError; any count at or
    # beyond the budget, or undefined, is the budget.
    trials = log1p(-config.success_prob) / log(p_failure)
    0 <= trials < config.max_trials || return config.max_trials
    return clamp(ceil(Int, trials), config.min_trials, config.max_trials)
end
