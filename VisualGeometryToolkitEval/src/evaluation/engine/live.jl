# =============================================================================
# evaluation/engine/live.jl — LIVE (end-to-end) RANSAC on stored pool pairs.
#
# The pool protocol re-scores a stored hypothesis pool; a live run instead runs
# the production `ransac` loop (real minimal-sample resampling, adaptive
# termination, in-loop LO) on the same pairs, in the same pool frame, on the
# problem the caller builds for each pair (a paper's: the pool-frame problem
# with the pool generator's own sampler, for sampler parity).
#
#   live_runs   one live run per pair (threaded, seeded per pair,
#               wall-bracketed, JIT warmed up outside every bracket)
#   live_row    the per-run summary: median error, mAA (the rule),
#               outcome counts, median trials and wall
# =============================================================================

"""
    live_runs(scoring, lo, ctxs; problem_of, seed_of, config = RansacConfig(), warmup = true,
              lo_of = nothing, on_improvement_of = nothing, score_guard = false,
              cancelled = () -> false, finalize = nothing)
        -> (; err, trials, wall_s)            # plus `extras` when `finalize !== nothing`

Run one live production `ransac` per pair context `ctxs[j]`, threaded over pairs.
Each run uses the caller's problem of the pair, `problem_of(ctxs[j].pair)`, the
selection `scoring`, the LO strategy `lo`, adaptive termination under `config`. Each pair's stream is `StableRNG(seed_of(j))`,
handed to `ransac` explicitly, so a run is independent of the thread schedule,
identical across calls that share `seed_of` (`seed_of` should be a `stable_seed`). Per pair: `err[j]` the imc pose error
[deg] of the final model, `Inf` when the run yielded no model (`converged`
false) or the pair was cancelled; `trials[j]` the trials drawn; `wall_s[j]` the wall clock of the
`ransac` call alone. `warmup` first runs one short solve on pair 1 (result
discarded) so no bracket pays JIT.

The keywords below default so that the result is exactly the plain run's.
- `lo_of(j)` returns the LO strategy of pair `j`, overriding `lo`. The warmup
  uses `lo_of(1)` (an LO holds no recorded state).
- `on_improvement_of(j)` returns the `ransac` improvement callback
  (`on_improvement(trial, score, ws)`, fired when the incumbent improves). It is
  called with `j = 0` once for the warmup (return a throwaway callback of the same
  type as for real pairs, writing to scratch storage, so the loop compiles once
  and the warmup records nothing) and with `j = 1:n` for the pairs.
- `score_guard` is forwarded to `ransac`: per-candidate ScoreGuarded acceptance.
- `cancelled()` is polled before each pair; a cancelled pair is not run and keeps
  `err = Inf`, `trials = 0`, `wall_s = NaN`. Pairs are threaded, so a pair already
  started finishes. [`Experiments.cancelled`](@ref) is the kaimon-aware poll.
- `finalize(j, r)` runs after pair `j`'s `ransac` result `r`, on the pair's
  thread and outside its wall bracket; its values come back as
  `extras::Vector{Any}` (`nothing` for a pair not run). Without `finalize` the
  result has no `extras` field.
"""
function live_runs(scoring, lo::AbstractLocalOptimization, ctxs::AbstractVector{<:PairCtx};
                   problem_of, seed_of, config::RansacConfig = RansacConfig(),
                   warmup::Bool = true, lo_of = nothing, on_improvement_of = nothing,
                   score_guard::Bool = false, cancelled = () -> false, finalize = nothing)
    n = length(ctxs)
    problem_at(j) = problem_of(ctxs[j].pair)
    lo_at(j) = lo_of === nothing ? lo : lo_of(j)
    cb_at(j) = on_improvement_of === nothing ? nothing : on_improvement_of(j)
    if warmup && n > 0
        ransac(problem_at(1), scoring; local_optimization = lo_at(1),
               config = RansacConfig(max_trials = 100, min_trials = 1),
               on_improvement = cb_at(0), score_guard)
    end
    err = fill(Inf, n); trials = zeros(Int, n); wall_s = fill(NaN, n)
    extras = Any[nothing for _ in 1:n]
    Threads.@threads for j in 1:n
        cancelled() && continue
        prob = problem_at(j)
        rng = StableRNG(seed_of(j))
        loj = lo_at(j); cb = cb_at(j)
        t0 = time_ns()
        r = ransac(prob, scoring; local_optimization = loj, config,
                   on_improvement = cb, score_guard, rng)
        wall_s[j] = (time_ns() - t0) / 1e9
        trials[j] = iterations(r)
        err[j] = converged(r) ? Float64(model_imc(estimate(r), ctxs[j])) : Inf
        finalize === nothing || (extras[j] = finalize(j, r))
    end
    return finalize === nothing ? (; err, trials, wall_s) : (; err, trials, wall_s, extras)
end

"""
    live_row(runs; T) -> NamedTuple

The summary of one [`live_runs`](@ref) result.
Its fields are `med` the finite median error
[deg], `maa` the mAA@`T` under the eligibility rule (VisualGeometryToolkitEval
`eligible_maa`: every pool pair is eligible, a run without a model scores 0;
`T` defaults to its `maa_threshold()`), `med_trials` the median
trials, `med_wall_ms` the median wall [ms], and the outcome counts
(`n_eligible`, `n_failed`, `n_skipped`).
"""
live_row(runs; kwargs...) =
    (; med = finite_median(runs.err), maa = eligible_maa(runs.err; kwargs...),
       med_trials = median(runs.trials), med_wall_ms = 1e3 * finite_median(runs.wall_s),
       pair_outcomes(runs.err)...)
