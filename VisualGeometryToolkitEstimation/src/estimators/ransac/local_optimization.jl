# =============================================================================
# The local-optimization strategies' contract methods — VisualGeometryToolkitCore's
# `locally_optimize!(L, lo, problem, θ, cs, buf, scoring)` for the strategies
# the core hosts — the null strategy and the static M-estimator `Irls{K}` — and
# the posterior-weight iteration they and LocalOptimizationDoneRight's
# posterior-IRLS arms (the Student-t + uniform IRLS, Barron-adaptive) run on.
# =============================================================================
#
# On entry (the contract): `residuals(buf)` holds θ's residuals and
# `consensus_mask(buf)` the consensus set `cs` of the score that just ran on
# `buf`; `consensus_order(scoring, scoring_workspace(buf))[1:inlier_count(cs)]`
# is that set and `scale_s2(cs)` its Student-t scale. A method may overwrite
# `buf`'s residuals, scores and mask; on exit the mask is the set the returned
# model was refit on. It refines only — the loop re-scores and keeps the
# incumbent.
# =============================================================================

# No local optimization is `nothing`: VisualGeometryToolkitCore's
# `locally_optimize!(L, ::Nothing, …)` returns the model unchanged (`:no_lo`).

# =============================================================================
# The posterior-weight iteration via the generic block_coordinate_descent! loop
# =============================================================================

# The model block of a `PosteriorIrlsMethod` is a bound, named callable —
# `model_update(refine_ws, problem, θ, weights) -> θ′` (`nothing` = the solve
# failed) — so WHAT the iteration does to the model is a value of the method,
# not a second method type: the weighted refit of every posterior-IRLS LO arm,
# or the model held while the noise model is fitted to its fixed residuals.

"""
    WeightedRefine()

The model update of a [`PosteriorIrlsMethod`](@ref) that REFITS the model under
the current weights: `refine!(refine_ws, problem, θ, weights)` — the problem's own
weighted solve (its chart LM). The RANSAC local optimization's, and the default.
"""
struct WeightedRefine end
(::WeightedRefine)(refine_ws, problem, θ, weights) = refine!(refine_ws, problem, θ, weights)

"""
    KeepModel(tol)

The model update of a [`PosteriorIrlsMethod`](@ref) that HOLDS the model: it
returns `θ` unchanged, so an iteration moves only the noise model (the
nuisance, then the E-step) — EM of the Student-t + uniform mixture on the
model's fixed residuals. The iteration's cost is then the mixture negative
log-likelihood (`mixture_nll`), which that EM never raises, and the loop
converges once an iteration lowers it by no more than `tol` relative.
"""
struct KeepModel
    tol::Float64
    function KeepModel(tol::Real)
        (isfinite(tol) && tol >= 0) || throw(ArgumentError("KeepModel: tol must be finite and ≥ 0, got $tol"))
        return new(Float64(tol))
    end
end
(::KeepModel)(_refine_ws, _problem, θ, _weights) = θ

# The stop rule of a `PosteriorIrlsMethod` is a value of the method too, selected in
# the EM's table (`stop`, [`STOP_TAGS`](@ref)): the incumbent's cost, or the change
# of the parameters.

"""
    NoImprovementStop()

The stop rule of a [`PosteriorIrlsMethod`](@ref) that ends the iteration once
an iteration fails to lower the INCUMBENT's cost
([`StopWhenNoImprovement`](@ref): from the second iteration under
[`WeightedRefine`](@ref), the cost Σ wᵢrᵢ²; from the first under
[`KeepModel`](@ref), the mixture NLL at its `tol`), and returns the incumbent.
The RANSAC local optimization's rule, and `[mixture_em]`'s. TOML tag
`"no_improvement"`.
"""
struct NoImprovementStop end

"""
    ParameterChangeStop(rtol)

The stop rule of a [`PosteriorIrlsMethod`](@ref) that ends the iteration once
an iteration moves the parameters by less than `rtol` RELATIVE,
`‖p − p_old‖ / ‖p_old‖ < rtol` with `p` the problem's
[`change_parameters`](@ref) of the iterate, and returns the LAST iterate. For a
fixed-point iteration with no fixed objective to track — one that re-linearizes
its residuals at every iterate (`fit_division_camera`), where the cost of one
iteration is not comparable to the next. TOML tag `"parameter_change"`.
"""
struct ParameterChangeStop
    rtol::Float64
    function ParameterChangeStop(rtol::Real)
        (isfinite(rtol) && rtol >= 0) || throw(ArgumentError("ParameterChangeStop: rtol must be finite and ≥ 0, got $rtol"))
        return new(Float64(rtol))
    end
end

"""
    change_parameters(problem, θ) -> AbstractVector

The parameters of the model `θ` that [`ParameterChangeStop`](@ref) compares
between iterations — a problem's own measure of how far an iteration moved its
model (`fit_division_camera`'s: the focal length and the division coefficients,
pixels). The default is the model's `parameter_vector`.
"""
change_parameters(_problem, θ) = parameter_vector(θ)

"""
    StopWhenParameterChangeLess(rtol)

`:converged` once the last iteration moved the problem's
[`change_parameters`](@ref) by less than `rtol` relative to their previous
value. The criterion of [`ParameterChangeStop`](@ref).
"""
struct StopWhenParameterChangeLess <: StoppingCriterion
    rtol::Float64
end
function stop_reason(c::StopWhenParameterChangeLess, state, method, prob, θ, θ_old, iter)
    p, p_old = change_parameters(prob, θ), change_parameters(prob, θ_old)
    Δ = sqrt(sum(abs2(a - b) for (a, b) in zip(p, p_old))) / max(sqrt(sum(abs2, p_old)), eps())
    return Δ < c.rtol ? :converged : :continue
end
# A failed model block (θ = `nothing`) is `StopWhenSolveFailed`'s to end.
stop_reason(::StopWhenParameterChangeLess, state, method, prob, ::Nothing, θ_old, iter) = :continue

"""
    PosteriorIrlsMethod(refine_ws, model_update = WeightedRefine(), stop = NoImprovementStop())

The block-coordinate method of the posterior-weight iteration (VGC
`block_coordinate_descent!`): the model block is `model_update(refine_ws, problem,
θ, robust_weight(state.rws))` ([`WeightedRefine`](@ref) or
[`KeepModel`](@ref)); the residuals, nuisance and weight blocks are the
likelihood's; `stop` the stop rule and the result ([`NoImprovementStop`](@ref):
the incumbent, the RANSAC local optimization's; [`ParameterChangeStop`](@ref):
the last iterate). `refine_ws` is the workspace the model update hands `refine!`
(`nothing` when the model is held). The state is a `PosteriorIrlsState` built
from a seeded posterior and its residual buffers — by the consensus seed
([`posterior_irls!`](@ref)) or by the caller. A [`MixtureEmConfig`](@ref)
states `model_update` and `stop` in its table.
"""
struct PosteriorIrlsMethod{W, U, S} <: AbstractBlockCoordinateMethod
    lo_ws::W
    model_update::U
    stop::S
end
PosteriorIrlsMethod(refine_ws) = PosteriorIrlsMethod(refine_ws, WeightedRefine())
PosteriorIrlsMethod(refine_ws, model_update) = PosteriorIrlsMethod(refine_ws, model_update, NoImprovementStop())

mutable struct PosteriorIrlsState{M, T}
    rws::PosteriorWeightState
    residuals::Vector{T}
    # Residuals corresponding to the incumbent — saved by `post_iteration!` whenever
    # the cost drops, so `locally_optimize!` can hand them off to the scoring
    # buffers at IRLS exit and let Phase 3 skip a redundant mahalanobis_distances!() sweep.
    best_residuals::Vector{T}
    # The incumbent, `Attributed(θ, (; cost))` (VGC `keep_best`), and the current
    # cost. Costs are Float64 regardless of the workspace storage T: the stop rule
    # compares running sums Σwᵢrᵢ², whose resolution must not degrade with Float32 storage.
    best::Attributed{M, @NamedTuple{cost::Float64}}
    current_cost::Float64
    # Whether the LAST iteration lowered the incumbent cost — written by
    # `post_iteration!`, read by the stopping criterion (`StopWhenNoImprovement`).
    improved::Bool
    PosteriorIrlsState{M, T}(rws, residuals, best_residuals, best, current_cost, improved) where {M, T} =
        new{M, T}(rws, residuals, best_residuals, best, current_cost, improved)
    # The state a local optimizer's workspace holds (`LoBuffers`): buffers
    # bound, the incumbent set by every seeding (`_seeded_irls_state!`).
    PosteriorIrlsState{M, T}(rws, residuals, best_residuals) where {M, T} =
        new{M, T}(rws, residuals, best_residuals)
end

"""
    PosteriorIrlsState(rws, residuals, best_residuals, θ₀, cost)

The state of a [`PosteriorIrlsMethod`](@ref) run: the seeded posterior `rws`,
the residual buffer `residuals` (the model's residuals at `θ₀`, refilled by
`update_residuals!`) and the incumbent's `best_residuals`, with `θ₀` the
incumbent at `cost`. The buffers are the caller's — the local optimizer's
workspace ([`posterior_irls!`](@ref)) or any caller's own.
"""
PosteriorIrlsState(rws::PosteriorWeightState, residuals::Vector{T}, best_residuals::Vector{T},
                   θ₀::M, cost::Real) where {M, T} =
    PosteriorIrlsState{M, T}(rws, residuals, best_residuals, _with_cost(M, θ₀, cost), Float64(cost), false)

# The incumbent: the model `θ` (held as an `M`) with its cost, a Float64.
_with_cost(::Type{M}, θ, cost::Real) where {M} = Attributed{M, @NamedTuple{cost::Float64}}(θ, (; cost = Float64(cost)))

"""
    consensus_posterior(L::LoBuffers, problem, cs::ConsensusSet, buf::ScoreBuffers, scoring, a::Float64)
        -> PosteriorWeightState

The Student-t + uniform posterior of the consensus set `cs`, on the local
optimizer's buffers `L` (its `posterior` state, re-seeded in place), at the uniform outlier half-width `a` — the ONE
seeding of every local optimization that starts from the posterior (the
posterior-weight iteration, [`posterior_irls!`](@ref); LocalOptimizationDoneRight's
mixture lifts). Freshness contract: `cs` must come from the most recent score
on `buf`; its consensus order is read from the scorer's workspace. The seed
model's residuals are already in `buf`, so squaring them into the posterior
buffer avoids a redundant mahalanobis_distances!() sweep. The Student-t scale contributes
one extra DOF on top of the minimal sample size (ν = (k − m)·dg).
"""
function consensus_posterior(L::LoBuffers, problem, cs::ConsensusSet, buf::ScoreBuffers, scoring, a::Float64)
    check_fresh(cs, buf)
    n = data_size(problem)
    square_distances!(L.posterior_sq_r, buf.residuals, n)
    return seed_posterior!(
        L.posterior, L.posterior_sq_r, n, minimal_sample_size(problem) + 1, inlier_count(cs);
        order = consensus_order(scoring, buf.scoring_ws),
        prior = scale_prior(scoring),
        log_two_outlier_halfwidth = log(2 * a),
        codimension = codimension(problem)
    )
end

"""
    required_outlier_halfwidth(scoring) -> Float64

The uniform outlier model a posterior-weight fit needs: the scorer's
`outlier_halfwidth`, refused (`ArgumentError`) when the scorer has none.
"""
required_outlier_halfwidth(scoring) = required_outlier_halfwidth(outlier_halfwidth(scoring), scoring)
required_outlier_halfwidth(a::Float64, _scoring) = a
required_outlier_halfwidth(::Nothing, scoring) = throw(
    ArgumentError(
        "$(nameof(typeof(scoring))) has no uniform outlier model (outlier_halfwidth); a " *
            "posterior-weight local optimization needs one — score with a scorer that has it"
    )
)

# The consensus seed of a `PosteriorIrlsMethod` run: the set's posterior on the
# optimizer's buffers, at the cost Σ wᵢ·rᵢ² of its weights, a zero-weight term
# exactly 0 (`weighted_rss`: an excluded point may carry a ±Inf residual, and
# 0·Inf = NaN would poison the cost, the best-tracking and the stop rule). The
# residuals are copied into the optimizer's state buffers: `update_residuals!`
# mutates `state.residuals` in place each iteration, so `lo_residuals` is the
# scratch and `lo_best_residuals` the incumbent — no vector per call.
function _seeded_irls_state!(L::LoBuffers, problem, θ₀, cs::ConsensusSet, buf::ScoreBuffers, scoring, a::Float64)
    rws = consensus_posterior(L, problem, cs, buf, scoring, a)
    cost = weighted_rss(robust_weight(rws), L.posterior_sq_r, data_size(problem))
    copyto!(L.lo_residuals, buf.residuals)
    copyto!(L.lo_best_residuals, buf.residuals)
    return _reset_irls_state!(L.irls, rws, L, θ₀, cost)
end

# The workspace's state, reset (its buffers are `L`'s), when it holds the
# seed's model type; any other seed type builds its own state, as before.
function _reset_irls_state!(state::PosteriorIrlsState{M}, _rws, _L, θ₀::M, cost) where {M}
    state.best = _with_cost(M, θ₀, cost)
    state.current_cost = Float64(cost)
    state.improved = false
    return state
end
_reset_irls_state!(_state, rws, L, θ₀, cost) = PosteriorIrlsState(rws, L.lo_residuals, L.lo_best_residuals, θ₀, cost)

update_model!(state::PosteriorIrlsState, method::PosteriorIrlsMethod, problem, θ) =
    method.model_update(method.lo_ws, problem, θ, robust_weight(state.rws))

function update_residuals!(state::PosteriorIrlsState{M, T}, method::PosteriorIrlsMethod, problem, θ) where {M, T}
    n = data_size(problem)
    mahalanobis_distances!(state.residuals, problem, θ)
    rws = state.rws
    square_distances!(sq_distances(rws), state.residuals, n)
    state.current_cost = weighted_rss(robust_weight(rws), sq_distances(rws), n)
    return nothing
end

function post_iteration!(state::PosteriorIrlsState, method::PosteriorIrlsMethod, prob, θ, iter)
    state.best, state.improved = keep_best(state.best, θ, state.current_cost)
    state.improved && copyto!(state.best_residuals, state.residuals)
    return nothing
end

# Converged when an iteration fails to lower the incumbent cost (the first
# iteration always runs a second). A pure query of the verdict
# `post_iteration!` stored.
stopping_criterion(::PosteriorIrlsMethod) = StopWhenNoImprovement(1, 2)
current_cost(state::PosteriorIrlsState) = state.current_cost

# The model HELD: the cost is the mixture NLL at the noise model the iteration
# just reached (after its nuisance and E-step), the incumbent is the held model
# at that cost, and the verdict is whether the NLL fell by more than `tol`
# relative. The residuals never move, so there is nothing to copy.
function post_iteration!(state::PosteriorIrlsState{M}, method::PosteriorIrlsMethod{<:Any, KeepModel}, prob, θ, iter) where {M}
    nll = mixture_nll(state.rws, state.residuals, data_size(prob))
    previous = state.best.cost
    state.improved = previous - nll > method.model_update.tol * abs(previous)
    state.current_cost = nll
    state.best = _with_cost(M, θ, nll)
    return nothing
end
# (the incumbent's rule: a parameter change, held model or not, is the method below — one method each, no ambiguity)
stopping_criterion(::PosteriorIrlsMethod{<:Any, KeepModel, NoImprovementStop}) = StopWhenNoImprovement(1, 1)

# The termination record's rule (§14) of a run its stop rule ended `:converged`, by the
# configured stop and model update — the ONE map: the parameter change; the incumbent's
# cost not falling under a refit; the mixture NLL falling by ≤ `tol` under a held model
# (its `StopWhenNoImprovement` is a cost-decrease test). The run's Symbol reason becomes
# its record through `retcode_from_symbol` (`lo_estimate`'s Symbol form).
_stop_rule(::ParameterChangeStop, _model_update) = ParameterChange
_stop_rule(::NoImprovementStop, ::WeightedRefine) = NoImprovement
_stop_rule(::NoImprovementStop, ::KeepModel) = CostDecrease
_stop_rule(m::PosteriorIrlsMethod) = _stop_rule(m.stop, m.model_update)

function block_coordinate_result(state::PosteriorIrlsState{M, T}, method::PosteriorIrlsMethod, prob, θ, stop_reason, final_iter) where {M, T}
    return lo_estimate(state.best.value, stop_reason, _stop_rule(method); iterations = final_iter)
end

# The parameter-change rule: its criterion, and the LAST iterate as the result (on
# `:solve_failed`, the last model the model block produced).
stopping_criterion(m::PosteriorIrlsMethod{<:Any, <:Any, ParameterChangeStop}) = StopWhenParameterChangeLess(m.stop.rtol)
block_coordinate_result(::PosteriorIrlsState, method::PosteriorIrlsMethod{<:Any, <:Any, ParameterChangeStop}, prob, θ, stop_reason, final_iter) =
    lo_estimate(θ, stop_reason, _stop_rule(method); iterations = final_iter)

"""
    posterior_irls!(likelihood, L::LoBuffers, problem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring,
                    a::Float64; max_iter) -> FitResult

THE posterior-weight iteration of every posterior-IRLS local optimization:
seeded from the consensus set `cs` of the score that just ran on `buf` (its
order reused, no fresh sort) at the uniform outlier half-width `a`, then
VisualGeometryToolkitCore's `block_coordinate_descent!` with a
[`PosteriorIrlsMethod`](@ref) on the problem's weighted refit. A strategy
contributes only its weight policy, `likelihood(rws, n)` on the seeded
posterior state `rws`:

- [`PosteriorLikelihood`](@ref): the Student-t + uniform posterior,
  self-scaling — its seed weights are the posterior's own;
- any other policy (a frozen-width loss, [`LossPlainWeights`](@ref)) REPLACES
  the seed weights before the first solve: its `update_nuisance!` then
  `update_weights!`, and the cost re-seeded under those weights.

The best iterate's residuals are handed back to `buf`
(`residuals_current = true`), so the loop skips a redundant mahalanobis_distances!() sweep.
"""
function posterior_irls!(
        likelihood::F, L::LoBuffers, problem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring,
        a::Float64; max_iter::Int
    ) where {F}
    method = PosteriorIrlsMethod(L.refine)
    state = _seeded_irls_state!(L, problem, model, cs, buf, scoring, a)
    lh = likelihood(state.rws, data_size(problem))
    _seed_policy_weights!(state, lh, data_size(problem))
    result = block_coordinate_descent!(state, lh, method, problem, model; max_iter)
    copyto!(buf.residuals, state.best_residuals)
    return lo_estimate(estimate(result), retcode(result), stop_rule(result); iterations = iterations(result), residuals_current = true)
end

"""
    locally_optimize!(L, lo::Irls, problem, model, cs, buf, scoring) -> FitResult

Static M-estimation on the posterior-weight iteration ([`posterior_irls!`](@ref)):
VGC's `weight` on the loss the width mode deploys (`lo_loss`), frozen
([`LossPlainWeights`](@ref)). The seed state's outlier model is the scorer's
when it has one; any finite value serves otherwise (MSAC), since the frozen
weights replace the seed's before the first solve.
"""
function locally_optimize!(
        L::LoBuffers, lo::Irls, problem, model,
        cs::ConsensusSet, buf::ScoreBuffers, scoring
    )
    loss = lo_loss(lo, cs, scoring, buf)
    return posterior_irls!(L, problem, model, cs, buf, scoring, something(outlier_halfwidth(scoring), 1.0);
                           max_iter = lo.max_iter) do rws, n
        LossPlainWeights(rws, n, loss)
    end
end

# The frozen-width policies' seed: `update_nuisance!` first (a no-op for a
# frozen loss), then `update_weights!`, and the cost re-seeded under those
# weights so the best-tracking/stop rule compares like with like. The posterior
# keeps the seed state's own weights.
_seed_policy_weights!(::PosteriorIrlsState, ::PosteriorLikelihood, ::Int) = nothing
function _seed_policy_weights!(state::PosteriorIrlsState{M}, lh, n::Int) where {M}
    update_nuisance!(lh)
    update_weights!(lh)
    cost = weighted_rss(robust_weight(state.rws), sq_distances(state.rws), n)
    state.best = _with_cost(M, state.best.value, cost)
    state.current_cost = cost
    return nothing
end

# =============================================================================
# CountedLO — instrumentation around any local-optimization strategy
# =============================================================================

"""
    CountedLO(inner::AbstractOptimizationStrategy)

Instrumentation wrapper: counts the `locally_optimize!` invocations (`calls[]`)
and the iterations they report (`iterations[]`, the sum of
`iterations(result)`), and records each invocation's termination in
`call_records`, a `Vector` of `(; iterations, retcode, stop_rule)` (the call's
`iterations(result)`, its `ReturnCode` and its `StopRule`), one entry per call.
It answers every hook the RANSAC loop, the scorer
composition check, the reporting and an external refit read on a strategy
exactly as `inner` does — its `RefitFamily`, placement and external-refit
bracket included — so a run with `CountedLO(inner)` returns the model a run
with `inner` returns. An external refit (`refine_external!`) is forwarded
uncounted. It is not a registered strategy (no TOML tag): it carries counters,
not configuration. A counter is shared by every run that is handed the same
wrapper — the counters are `Threads.Atomic{Int}` and `call_records` is appended
under a lock, so runs on several threads (a threaded sweep over pairs) add up
exactly (the records then in the threads' order); build a fresh one per
experiment, and read `call_records` once its runs have returned.
"""
struct CountedLO{L <: Union{Nothing, AbstractOptimizationStrategy}} <: AbstractOptimizationStrategy
    inner::L
    calls::Threads.Atomic{Int}
    iterations::Threads.Atomic{Int}
    call_records::Vector{@NamedTuple{iterations::Int, retcode::ReturnCode.T, stop_rule::StopRule}}
    call_records_lock::ReentrantLock
end
CountedLO(inner::Union{Nothing, AbstractOptimizationStrategy}) =
    CountedLO(inner, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
              @NamedTuple{iterations::Int, retcode::ReturnCode.T, stop_rule::StopRule}[], ReentrantLock())

function locally_optimize!(L, lo::CountedLO, problem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring)
    r = locally_optimize!(L, lo.inner, problem, model, cs, buf, scoring)
    Threads.atomic_add!(lo.calls, 1)
    Threads.atomic_add!(lo.iterations, VgtkCore.iterations(r))
    @lock lo.call_records_lock push!(lo.call_records, (; iterations = VgtkCore.iterations(r), retcode = VgtkCore.retcode(r), stop_rule = stop_rule(r)))
    return r
end

VgtkCore.with_width(lo::CountedLO, w) =
    CountedLO(VgtkCore.with_width(lo.inner, w), lo.calls, lo.iterations, lo.call_records, lo.call_records_lock)
VgtkCore.lo_label(lo::CountedLO) = VgtkCore.lo_label(lo.inner)
# A counter around a terminal placement is terminal, and counts the terminal call (its `locally_optimize!` counts, then forwards).
lo_placement(lo::CountedLO) = lo_placement(lo.inner)

# =============================================================================
# TerminalLO — the terminal placement (scoring_glue.jl); every hook is inner's
# =============================================================================
#
# The loop never calls `locally_optimize!` with a `TerminalLO` (its race runs
# `nothing`, its terminal call `inner`); a seeded `locally_optimize`
# does, and runs `inner`.

locally_optimize!(L, lo::TerminalLO, problem, model, cs::ConsensusSet, buf::ScoreBuffers, scoring) =
    locally_optimize!(L, lo.inner, problem, model, cs, buf, scoring)
VgtkCore.with_width(lo::TerminalLO, w) = TerminalLO(VgtkCore.with_width(lo.inner, w))
VgtkCore.lo_label(lo::TerminalLO) = VgtkCore.lo_label(lo.inner) * " (terminal)"

# =============================================================================
# The two wrappers' identity hooks — every one is `inner`'s
# =============================================================================
#
# What the loop, the composition check and the reporting read on a strategy
# (workspace, width, scale interface, loss family, path, marker), answered for
# `CountedLO` and `TerminalLO` alike by their `inner`; each wrapper's own
# methods above (`locally_optimize!`, `with_width`, `lo_label`, `lo_placement`)
# are where they differ.
const _WrappedLO = Union{CountedLO, TerminalLO}
lo_workspace(::Type{P}, ::Type{T}, lo::_WrappedLO, n::Int) where {P <: AbstractRansacProblem, T} =
    lo_workspace(P, T, lo.inner, n)
lo_width(lo::_WrappedLO) = lo_width(lo.inner)
strategy_loss(::Type{<:CountedLO{L}}) where {L} = strategy_loss(L)
strategy_loss(::Type{<:TerminalLO{L}}) where {L} = strategy_loss(L)
VgtkCore.optimization_path(lo::_WrappedLO) = VgtkCore.optimization_path(lo.inner)
VgtkCore.scale_interface(lo::_WrappedLO) = VgtkCore.scale_interface(lo.inner)
VgtkCore.loss_family(lo::_WrappedLO) = VgtkCore.loss_family(lo.inner)
VgtkCore.lo_marker(lo::_WrappedLO) = VgtkCore.lo_marker(lo.inner)
# The refit family and the external-refit bracket (VGC `lo_contract.jl`): an
# external strategy (PoseLib's LM/LZ arms) wrapped is routed, prepared and
# refined as `inner` is, so the wrapper returns the model `inner` returns.
# `CountedLO` counts `locally_optimize!` only: an external refit is forwarded
# uncounted. (A fixed noise scale reaches the `LossWidth` method through VGC's
# `FixedS2` default.)
VgtkCore.RefitFamily(lo::_WrappedLO) = VgtkCore.RefitFamily(lo.inner)
VgtkCore.external_workspace(lo::_WrappedLO, problem, frame, scoring) =
    VgtkCore.external_workspace(lo.inner, problem, frame, scoring)
VgtkCore.prepare_external!(ws, lo::_WrappedLO, problem, θ, width, scoring) =
    VgtkCore.prepare_external!(ws, lo.inner, problem, θ, width, scoring)
VgtkCore.refine_external!(ws, lo::_WrappedLO, problem, θ, width::LossWidth, scoring) =
    VgtkCore.refine_external!(ws, lo.inner, problem, θ, width, scoring)
