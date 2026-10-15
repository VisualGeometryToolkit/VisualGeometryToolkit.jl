# =============================================================================
# Generic block-coordinate-descent loop
# =============================================================================
#
# Provides a generic block-coordinate-descent loop with extension points.
#
# Block order: model → residuals → nuisance → weights → post_iteration! → stop.
# The loop is written in Manopt.jl's vocabulary — `initialize_solver!`,
# `step_solver!` (one BCD cycle), `stop_solver!` and `StoppingCriterion` values
# composed with `|` — without depending on Manopt: the names are `public`, not
# exported, so a later `using Manopt` does not clash and a bridge extends
# Manopt's functions instead.
#
# The likelihood model (nuisance + weight blocks) is separated from the
# problem method (model block). The `likelihood` argument to
# `block_coordinate_descent!` controls how squared residuals are converted to
# weights. RVG's `Irls{K}` is one LO STRATEGY that runs under this loop; the
# loop itself is not "IRLS".
# =============================================================================

"""
    AbstractBlockCoordinateMethod

Abstract type for the problem side of [`block_coordinate_descent!`](@ref): the
model block, its residuals, and the per-iteration bookkeeping.

Each concrete subtype defines extension points:
- `init_state!(method, prob, θ₀)` — build the state and compute the initial
  r/σ/w. It carries `!` because a method may fill buffers its own workspace
  holds (RobustVisualGeometry's posterior IRLS seeds `method.ws`'s residual
  buffers, which the iterations then read); the method is the mutated argument
- `update_model!(state, method, prob, θ)` — the model block: one weighted
  update of θ (weighted least squares, LM, Newton), or θ unchanged; `nothing`
  when there is no update to make (the run stops with `:solve_failed`)
- `update_residuals!(state, method, prob, θ)` — compute residuals + sq_residuals from θ
- `post_iteration!(state, method, prob, θ, iter)` — per-iteration bookkeeping
  (best-tracking, counters the stopping criteria read)
- [`stopping_criterion`](@ref)`(method)` — the method's own
  [`StoppingCriterion`](@ref) (default [`StopNever`](@ref))
- `block_coordinate_result(state, method, prob, θ, stop_reason, final_iter)` — package result
"""
abstract type AbstractBlockCoordinateMethod end

# Extension-point generic functions. The framework only calls them; concrete
# methods live downstream (RobustVisualGeometry problems, BlobBoards adapters).
function init_state! end
function update_model! end
function update_residuals! end
function block_coordinate_result end

# Default no-op for post_iteration!
post_iteration!(state, ::AbstractBlockCoordinateMethod, prob, θ, iter) = nothing

# Trust region is the OPTIMIZER's responsibility, not the framework's: the
# `update_model!` returns an accepted update (DLT is the exact minimizer; FLM owns its
# LM damping). So there is no trust-region hook — the loop takes the solver's
# result directly.

# =============================================================================
# Stopping criteria
# =============================================================================

"""
    StoppingCriterion

When a [`block_coordinate_descent!`](@ref) run ends, and why. A criterion is an
immutable, concretely typed value; it answers through
[`stop_reason`](@ref)`(criterion, state, method, prob, θ, θ_old, iter)`, which
returns `:continue` or the reason the run ends (`:converged`,
`:max_iterations`, `:solve_failed`, or any instance's own Symbol such as
`:left_basin`). Nothing is stored in the criterion: the counters a criterion
reads live in the solver state, written by `post_iteration!`.

Criteria compose with `|` into a [`StopWhenAny`](@ref), evaluated left to right;
the first reason other than `:continue` wins. The tuple is part of the type, so
the composed check compiles to the chain of branches it stands for.

The Manopt.jl correspondence: Manopt's criteria are mutable functors that
store their reason; these are pure, and the reason is the return value.
"""
abstract type StoppingCriterion end

"""
    stop_reason(criterion, state, method, prob, θ, θ_old, iter) -> Symbol

`:continue`, or the reason the run ends after iteration `iter`, which moved
`θ_old` to `θ` (`θ` is `nothing` when the model block failed). Pure: it reads
the state and writes nothing. Implement it for a new [`StoppingCriterion`](@ref).
"""
function stop_reason end

"""
    StopAfterIteration(n)

Stops with `:max_iterations` once `n` iterations have run (before the first,
when `n ≤ 0`). [`block_coordinate_descent!`](@ref) adds it, LAST, from its
`max_iter`, so on the final iteration a method's own reason wins.
"""
struct StopAfterIteration <: StoppingCriterion
    n::Int
end
@inline stop_reason(c::StopAfterIteration, state, method, prob, θ, θ_old, iter) =
    iter >= c.n ? :max_iterations : :continue

"""
    StopWhenSolveFailed()

Stops with `:solve_failed` when the model block returned `nothing` (a singular
or non-finite solve, an underdetermined refit): checked right after
`update_model!`, before any other block runs. [`block_coordinate_descent!`](@ref)
adds it FIRST.
"""
struct StopWhenSolveFailed <: StoppingCriterion end
@inline stop_reason(::StopWhenSolveFailed, state, method, prob, θ, θ_old, iter) =
    θ === nothing ? :solve_failed : :continue

"""
    StopNever()

Never stops: the default [`stopping_criterion`](@ref) of a method that runs its
whole budget.
"""
struct StopNever <: StoppingCriterion end
@inline stop_reason(::StopNever, state, method, prob, θ, θ_old, iter) = :continue

"""
    StopWhenAny(criteria::Tuple)
    a | b

Stops at the first of `criteria`, in order, whose [`stop_reason`](@ref) is
not `:continue`, with that reason. `a | b` builds one and flattens nested ones.
"""
struct StopWhenAny{C <: Tuple} <: StoppingCriterion
    criteria::C
end
@inline stop_reason(c::StopWhenAny, state, method, prob, θ, θ_old, iter) =
    _first_reason(c.criteria, state, method, prob, θ, θ_old, iter)

@inline _first_reason(::Tuple{}, state, method, prob, θ, θ_old, iter) = :continue
@inline function _first_reason(cs::Tuple, state, method, prob, θ, θ_old, iter)
    reason = stop_reason(first(cs), state, method, prob, θ, θ_old, iter)
    reason === :continue || return reason
    return _first_reason(Base.tail(cs), state, method, prob, θ, θ_old, iter)
end

Base.:|(a::StoppingCriterion, b::StoppingCriterion) = StopWhenAny((a, b))
Base.:|(a::StopWhenAny, b::StoppingCriterion) = StopWhenAny((a.criteria..., b))
Base.:|(a::StoppingCriterion, b::StopWhenAny) = StopWhenAny((a, b.criteria...))
Base.:|(a::StopWhenAny, b::StopWhenAny) = StopWhenAny((a.criteria..., b.criteria...))

# The criteria that can end a run before its first iteration: only the budget.
@inline _reason_at_start(c::StopAfterIteration) = c.n <= 0 ? :max_iterations : :continue
@inline _reason_at_start(::StoppingCriterion) = :continue
@inline _reason_at_start(c::StopWhenAny) = _first_reason_at_start(c.criteria)
@inline _first_reason_at_start(::Tuple{}) = :continue
@inline function _first_reason_at_start(cs::Tuple)
    reason = _reason_at_start(first(cs))
    reason === :continue || return reason
    return _first_reason_at_start(Base.tail(cs))
end

"""
    stopping_criterion(method::AbstractBlockCoordinateMethod) -> StoppingCriterion

The method's own termination test, evaluated after `post_iteration!` and
before the iteration budget ([`StopAfterIteration`](@ref)). Default
[`StopNever`](@ref): the run ends on the budget or a failed solve only.
"""
stopping_criterion(::AbstractBlockCoordinateMethod) = StopNever()

# =============================================================================
# The recorder
# =============================================================================

"""
    Recorded(state)

A [`block_coordinate_descent!`](@ref) state that also keeps a per-iteration
trace: `iterates` (the model each iteration produced), `cost` (the state's
[`current_cost`](@ref) after `post_iteration!`), `s2` (the likelihood's
[`current_s2`](@ref), the squared scale, `NaN` where it holds none) and
`reason` (a `Ref` to the stop reason). Give it in place of the state; the
driver runs every block and every criterion on the wrapped `state`, so a
recorded run is the bare run, iterate for iterate. The bare state never
records — off by default, nothing to branch on.
"""
struct Recorded{S}
    state::S
    iterates::Vector{Any}
    cost::Vector{Float64}
    s2::Vector{Float64}
    reason::Base.RefValue{Symbol}
end
Recorded(state) = Recorded(state, Any[], Float64[], Float64[], Ref(:continue))

"""
    current_cost(state) -> Float64

The objective a state tracks at its current iterate, for [`Recorded`](@ref).
`NaN` unless the state's type implements it.
"""
current_cost(_state) = NaN

"""
    current_s2(likelihood) -> Float64

The squared scale ŝ² a weight policy currently holds, for [`Recorded`](@ref).
`NaN` unless the policy's type implements it.
"""
current_s2(_likelihood) = NaN

# The state the blocks and criteria see.
@inline _bare(state) = state
@inline _bare(r::Recorded) = r.state

@inline _record_iteration!(_state, _likelihood, θ) = nothing
function _record_iteration!(r::Recorded, likelihood, θ)
    push!(r.iterates, θ)
    push!(r.cost, Float64(current_cost(r.state)))
    push!(r.s2, Float64(current_s2(likelihood)))
    return nothing
end

@inline _record_reason!(_state, reason::Symbol) = nothing
@inline _record_reason!(r::Recorded, reason::Symbol) = (r.reason[] = reason; nothing)

# =============================================================================
# The solver steps (Manopt names) and block_coordinate_descent!
# =============================================================================

"""
    initialize_solver!(state, likelihood, method, prob, θ₀, criterion) -> Symbol

Start a [`block_coordinate_descent!`](@ref) run: clears a [`Recorded`](@ref)
trace, and returns the reason the run ends before its first iteration
(`:max_iterations` for a non-positive budget) or `:continue`. The state itself
is built beforehand — by `init_state!` or the caller.
"""
function initialize_solver!(state, likelihood, method, prob, θ₀, criterion::StoppingCriterion)
    _clear_record!(state)
    reason = _reason_at_start(criterion)
    _record_reason!(state, reason)
    return reason
end
@inline _clear_record!(_state) = nothing
function _clear_record!(r::Recorded)
    empty!(r.iterates); empty!(r.cost); empty!(r.s2)
    r.reason[] = :continue
    return nothing
end

"""
    step_solver!(state, likelihood, method, prob, θ, iter) -> θ′ or nothing

One iteration — one cycle through the blocks, in the frozen order:
`update_model!` (the model block; `nothing` ends the cycle there, with no
other block run), `update_residuals!`, `update_nuisance!(likelihood)`,
`update_weights!(likelihood)`, `post_iteration!`. Returns the new model.
"""
@inline function step_solver!(state, likelihood, method, prob, θ, iter::Int)
    s = _bare(state)
    θ_new = update_model!(s, method, prob, θ)
    isnothing(θ_new) && return nothing
    update_residuals!(s, method, prob, θ_new)
    update_nuisance!(likelihood)
    update_weights!(likelihood)
    post_iteration!(s, method, prob, θ_new, iter)
    _record_iteration!(state, likelihood, θ_new)
    return θ_new
end

"""
    stop_solver!(state, criterion, method, prob, θ, θ_old, iter) -> Symbol

The run's [`stop_reason`](@ref) after iteration `iter` (`θ` is `nothing` when
its model block failed), recorded on a [`Recorded`](@ref) state.
"""
@inline function stop_solver!(state, criterion::StoppingCriterion, method, prob, θ, θ_old, iter::Int)
    reason = stop_reason(criterion, _bare(state), method, prob, θ, θ_old, iter)
    _record_reason!(state, reason)
    return reason
end

"""
    block_coordinate_descent!(likelihood, method, prob, θ₀; max_iter) -> result

Block coordinate descent on a negative log-likelihood, over three blocks:

- model — `update_model!` (the `method`): weighted least squares, LM, Newton,
  or nothing at all (a fixed-model method such as a seed EM, which returns θ
  unchanged);
- nuisance — [`update_nuisance!`](@ref) (the `likelihood`): scale, mixing
  proportion, ν, Barron α, or nothing;
- weights — [`update_weights!`](@ref) (the `likelihood`): EM responsibilities,
  loss weights, or GNC surrogate weights.

`update_residuals!` is bookkeeping between the model and nuisance blocks: it
recomputes the residuals block from the new model and descends on nothing.
One iteration ([`step_solver!`](@ref)) is one cycle through every block, in
the order model → residuals → nuisance → weights, then `post_iteration!`. See
[`update_nuisance!`](@ref) for why the nuisance block is not named "scale".

# Stopping

The run's criterion is

    StopWhenSolveFailed() | stopping_criterion(method) | StopAfterIteration(max_iter)

evaluated in that order by [`stop_solver!`](@ref) after every iteration: a
failed model block ends the run at once (`:solve_failed`, no other block run);
otherwise the method's own criterion is read after `post_iteration!`, and only
then the budget — on the last iteration a method's reason wins over
`:max_iterations`. `max_iter ≤ 0` runs no iteration. The result is
`block_coordinate_result(state, method, prob, θ, reason, iterations)` with `θ`
the last model the model block produced (θ₀ if none).

# Monotonicity

Instances: EM (under the Neal–Hinton free-energy view EM is coordinate ascent
on the free energy, i.e. descent on its negative); IRLS / half-quadratic
(RobustVisualGeometry's `Irls{K}` LO strategy is one — the strategy, not this
loop); Barron-adaptive; GNC.

"Descent" is on the negative log-likelihood, and monotonicity is each
INSTANCE's claim, not the loop's: the best iterate is kept (a method tracks it
in `post_iteration!` with [`keep_best`](@ref) and returns it from
`block_coordinate_result`), but the loop never rejects a step. An instance
proves monotonicity through its `post_iteration!` trace. GNC is the declared
exception — it changes its objective between iterations.

# The Manopt.jl names

The loop is Manopt's `solve!` loop; the names are `public`, not exported, so a
future bridge extends Manopt's functions rather than clashing with them:

| Manopt.jl | here |
|:--|:--|
| `AbstractManoptSolverState` | the method's state (any type; RVG's `*State`) |
| `initialize_solver!` | [`initialize_solver!`](@ref) |
| `step_solver!` | [`step_solver!`](@ref): one BCD cycle, the four blocks + `post_iteration!` |
| `stop_solver!` | [`stop_solver!`](@ref) |
| `StoppingCriterion`, `get_reason` | [`StoppingCriterion`](@ref), [`stop_reason`](@ref) (pure; the reason is returned) |
| `StopAfterIteration`, `StopWhenAny` (criteria or-ed) | [`StopAfterIteration`](@ref), [`StopWhenAny`](@ref) (criteria or-ed) |
| `RecordSolverState` | [`Recorded`](@ref) |
| `get_solver_result` | `block_coordinate_result` |

Argument order is Julia Base's: the inputs the loop mutates come first
(`state`, in the method below, then `likelihood`), then the inputs it only
reads (the method, the problem, the seed).
"""
function block_coordinate_descent!(
        likelihood::AbstractWeightPolicy, method::AbstractBlockCoordinateMethod, prob, θ₀;
        max_iter::Int
    )
    state = init_state!(method, prob, θ₀)
    return block_coordinate_descent!(state, likelihood, method, prob, θ₀; max_iter)
end

"""
    block_coordinate_descent!(state, likelihood, method, prob, θ₀; max_iter) -> result

Block coordinate descent with pre-allocated state (skips `init_state!`).
`state` may be a [`Recorded`](@ref) one.
"""
function block_coordinate_descent!(
        state, likelihood::AbstractWeightPolicy, method::AbstractBlockCoordinateMethod,
        prob, θ₀; max_iter::Int
    )
    criterion = StopWhenSolveFailed() | stopping_criterion(method) | StopAfterIteration(max_iter)
    θ = θ₀
    iter = 0
    reason = initialize_solver!(state, likelihood, method, prob, θ₀, criterion)
    while reason === :continue
        iter += 1
        θ_old = θ
        θ_new = step_solver!(state, likelihood, method, prob, θ, iter)
        if isnothing(θ_new)
            # A failed model block ends the run on the last model; the reason is
            # StopWhenSolveFailed's (first in the criterion).
            reason = stop_solver!(state, criterion, method, prob, nothing, θ_old, iter)
            break
        end
        θ = θ_new
        reason = stop_solver!(state, criterion, method, prob, θ, θ_old, iter)
    end
    return block_coordinate_result(_bare(state), method, prob, θ, reason, iter)
end

# =============================================================================
# The incumbent — best-tracking of an instance that claims descent
# =============================================================================

"""
    Incumbent(θ, cost)

The best iterate of a [`block_coordinate_descent!`](@ref) run and its cost. The
loop never rejects an iterate, so an instance that claims descent keeps one: its
`post_iteration!` offers every iterate to [`keep_best`](@ref), and its
`block_coordinate_result` returns `incumbent.θ`. Immutable, so a state holds it
inline and replacing it allocates nothing.
"""
struct Incumbent{M}
    θ::M
    cost::Float64
end

"""
    keep_best(incumbent, θ, cost) -> (incumbent′, improved::Bool)

`(Incumbent(θ, cost), true)` when `cost` is STRICTLY below the incumbent's, and
`(incumbent, false)` otherwise. Pure: the caller stores the result.
"""
@inline function keep_best(inc::Incumbent{M}, θ, cost::Real) where {M}
    improved = cost < inc.cost
    return (improved ? Incumbent{M}(θ, Float64(cost)) : inc), improved
end
