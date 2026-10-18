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
#   live_row    the per-run summary: median error, mAA (THE rule),
#               outcome counts, median trials and wall
# =============================================================================

"""
    live_runs(scoring, lo, ctxs; problem_of, seed_of, config = RansacConfig(), warmup = true)
        -> (; err, trials, wall_s)

Run one live production `ransac` per pair context `ctxs[j]`, threaded over pairs.
Each run uses the caller's problem of the pair, `problem_of(ctxs[j].pair)`, the
selection `scoring`, the LO strategy `lo`, adaptive termination under `config`. Each pair's stream is `StableRNG(seed_of(j))`,
handed to `ransac` explicitly, so a run is independent of the thread schedule,
identical across calls that share `seed_of`, and fixed across Julia versions
(`seed_of` should be a `stable_seed`). Per pair: `err[j]` the imc pose error
[deg] of the final model, `Inf` when the run yielded no model (`converged`
false); `trials[j]` the trials drawn; `wall_s[j]` the wall clock of the
`ransac` call alone. `warmup` first runs one short solve on pair 1 (result
discarded) so no bracket pays JIT.
"""
function live_runs(scoring, lo::AbstractLocalOptimization, ctxs::AbstractVector{<:PairCtx};
                   problem_of, seed_of, config::RansacConfig = RansacConfig(),
                   warmup::Bool = true)
    n = length(ctxs)
    problem_at(j) = problem_of(ctxs[j].pair)
    if warmup && n > 0
        ransac(problem_at(1), scoring; local_optimization = lo,
               config = RansacConfig(max_trials = 100, min_trials = 1))
    end
    err = fill(Inf, n); trials = zeros(Int, n); wall_s = fill(NaN, n)
    Threads.@threads for j in 1:n
        prob = problem_at(j)
        rng = StableRNG(seed_of(j))
        t0 = time_ns()
        r = ransac(prob, scoring; local_optimization = lo, config, rng)
        wall_s[j] = (time_ns() - t0) / 1e9
        trials[j] = iterations(r)
        err[j] = converged(r) ? Float64(model_imc(estimate(r), ctxs[j])) : Inf
    end
    return (; err, trials, wall_s)
end

"""
    live_row(runs; T) -> NamedTuple

The summary of one [`live_runs`](@ref) result.
Its fields are `med` the finite median error
[deg], `maa` the mAA@`T` under THE eligibility rule (VisualGeometryToolkitEval
`eligible_maa`: every pool pair is eligible, a run without a model scores 0;
`T` defaults to its `maa_threshold()`), `med_trials` the median
trials, `med_wall_ms` the median wall [ms], and the outcome counts
(`n_eligible`, `n_failed`, `n_skipped`).
"""
live_row(runs; kwargs...) =
    (; med = finite_median(runs.err), maa = eligible_maa(runs.err; kwargs...),
       med_trials = median(runs.trials), med_wall_ms = 1e3 * finite_median(runs.wall_s),
       pair_outcomes(runs.err)...)
