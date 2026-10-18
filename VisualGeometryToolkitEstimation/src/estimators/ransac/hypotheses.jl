# =============================================================================
# draw_hypotheses — minimal-sample hypotheses of a problem, no scoring
# =============================================================================
#
# The hypothesis generator of a benchmark pool: the RANSAC trial (`_draw_trial!`,
# loop.jl — the same draw → test_sample → solve → fit_hypothesis → test_model
# that `ransac` runs) repeated until enough samples have produced a model. It
# verifies no consensus and selects nothing, so it is not a `find_*`; a random
# draw is a trial, and `draw_sample!` names one trial's sample.
# =============================================================================

"""
    Hypotheses{M}

The minimal-sample hypotheses [`draw_hypotheses`](@ref) drew, in the problem's
model frame:

- `models::Vector{M}`: every solution the problem's `test_model` accepted,
  sibling solutions of one sample adjacent and in solver order;
- `sample_indices::Matrix{Int}`: `(m, length(models))`, the minimal sample
  each model was solved from (repeated across siblings);
- `trials::Int`: draws made;
- `samples_accepted::Int`: draws that produced at least one model;
- `n_solutions::Int`: solver solutions before `fit_hypothesis`/`test_model`.
"""
struct Hypotheses{M}
    models::Vector{M}
    sample_indices::Matrix{Int}
    trials::Int
    samples_accepted::Int
    n_solutions::Int
end

"""
    draw_hypotheses(rng, problem; target, max_trials) -> Hypotheses

Draw minimal samples with the problem's own sampler until `target` samples have
each produced at least one model the problem's `test_model` accepts, or
`max_trials` draws have been made. Every sibling solution of an accepted sample
is kept, the last accepted sample's included. No scoring, no consensus, no
local optimization.

Each trial is the RANSAC trial, the one code path: `draw_sample!`, the
problem's `test_sample`, `solve`, then per solution `fit_hypothesis` and
`test_model`. The sampler is prepared for the run as `ransac` prepares it
(`reset!(sampler, max_trials)`: a PROSAC schedule spans `max_trials`). Every
draw comes from `rng`, so a seeded `rng` makes the result a pure function of
the seed and the problem.

!!! note "Against the retired pool generator (pool format ≤ 3)"
    Same stopping rule, sibling layout and `sample_indices` columns; the one
    difference is the model gate. The old generator applied `test_model` to H
    only, while here every family is gated by its problem's own: F's oriented
    epipolar gate now applies (measured on a PhotoTourism pair under a uniform
    sampler, it accepts 76 of 510 F solutions), E has no post-solve gate (an E
    stream is unchanged), H is unchanged. A pool drawn here therefore differs
    from an F pool drawn there (accepted with pool format 4).
    A PROSAC sampler's growth horizon is now the run's `max_trials` (it was
    the generator's own `prosac_T_N`).
"""
function draw_hypotheses(rng::Random.AbstractRNG, problem::AbstractRansacProblem;
                         target::Integer, max_trials::Integer)
    target >= 0 || throw(ArgumentError("target must be non-negative, got $target"))
    max_trials >= 0 || throw(ArgumentError("max_trials must be non-negative, got $max_trials"))
    M = model_type(problem)
    m = minimal_sample_size(problem)
    idx = Vector{Int}(undef, m)
    models = M[]
    flat = Int[]                      # the sample indices, column-major (m per model)
    trials = 0; accepted = 0; n_solutions = 0
    _prepare!(problem, Int(max_trials))
    while accepted < target && trials < max_trials
        trials += 1
        _, n, kept = _draw_trial!(rng, idx, problem, 0) do kept, model
            push!(models, model)
            append!(flat, idx)
            kept + 1
        end
        n_solutions += n
        kept > 0 && (accepted += 1)
    end
    return Hypotheses{M}(models, reshape(flat, m, length(models)), trials, accepted, n_solutions)
end
