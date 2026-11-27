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
# `BlockCoordinateDescent` controls how squared residuals are converted to
# weights. RVG's `Irls{K}` is one LO STRATEGY that runs under this loop; the
# loop itself is not "IRLS".
# =============================================================================

"""
    AbstractBlockCoordinateMethod

Abstract type for the problem side of [`BlockCoordinateDescent`](@ref): the
model block, its residuals, and the per-iteration bookkeeping.

Each concrete subtype defines extension points:
- `init_state!(method, prob, θ₀)` — build the state and compute the initial
  r/σ/w. It carries `!` because a method may fill buffers its own workspace
  holds (VisualGeometryToolkitEstimation's posterior IRLS seeds `method.ws`'s residual
  buffers, which the iterations then read); the method is the mutated argument
- `update_model!(state, method, prob, θ)` — the model block: one weighted
  update of θ (weighted least squares, LM, Newton), or θ unchanged; `nothing`
  when there is no update to make (the run stops with `(Failure, NonFinite)`)
- `update_residuals!(state, method, prob, θ)` — compute residuals + sq_distances from θ
- `post_iteration!(state, method, prob, θ, iter)` — per-iteration bookkeeping
  (best-tracking, counters the stopping criteria read)
- [`stopping_criterion`](@ref)`(method)` — the method's own
  [`StoppingCriterion`](@ref) (default [`StopNever`](@ref))
- `block_coordinate_result(state, method, prob, θ, code, final_iter, rule)` — package result, from the run's
  termination record `(code::ReturnCode.T, rule::StopRule)`
"""
abstract type AbstractBlockCoordinateMethod end

# Extension-point generic functions. The framework only calls them; concrete
# methods live downstream (VisualGeometryToolkitEstimation problems, BlobBoards adapters).
"""
    init_state!(method::AbstractBlockCoordinateMethod, prob, θ₀) -> state

Build the solver state of [`BlockCoordinateDescent`](@ref) and compute the
initial residuals, scale and weights at `θ₀`. A method may fill buffers its own
workspace holds, hence the `!`. An extension point: the method's package adds
the method.
"""
function init_state! end
init_state!(method::AbstractBlockCoordinateMethod, _prob, _θ₀) =
    _no_method(method, init_state!, "BlockCoordinateDescent builds its state with it")

"""
    init_state!(state, method::AbstractBlockCoordinateMethod, prob, θ₀) -> state′

Re-seed a [`BlockCoordinateDescent`](@ref) cache's kept `state` for a run of `method` on `prob` from `θ₀`: what
`reinit!(cache, θ₀; f, alg)` calls. A method whose state is reusable overwrites `state` in place (its buffers, its
incumbent, its counters) and returns it, so the cache's reuse allocates nothing; the returned state has the cache's
state type. The default builds a new one, `init_state!(method, prob, θ₀)`.
"""
init_state!(_state, method::AbstractBlockCoordinateMethod, prob, θ₀) = init_state!(method, prob, θ₀)
"""
    update_model!(state, method::AbstractBlockCoordinateMethod, prob, θ) -> θ′ | nothing

The model block of [`BlockCoordinateDescent`](@ref): one weighted update of
`θ` at fixed weights and nuisance (a weighted least-squares solve, an LM step,
Newton), or `θ` unchanged. `nothing` means there is no update to make, and the
run stops with `(Failure, NonFinite)`. A fit `fit_<fit>`'s method calls its
`<fit>_step` here.
"""
function update_model! end
"""
    update_residuals!(state, method::AbstractBlockCoordinateMethod, prob, θ) -> state

The residual block of [`BlockCoordinateDescent`](@ref): writes the residuals
and squared residuals of `θ` into the state, and nothing else.
"""
function update_residuals! end
"""
    block_coordinate_result(state, method::AbstractBlockCoordinateMethod, prob, θ, code::ReturnCode.T, final_iter, rule::StopRule)

Package the run of [`BlockCoordinateDescent`](@ref) as the method's result
(a [`FitResult`](@ref) for the stack's fits), from the final state, model,
termination record `(code, rule)` (the [`stop_reason`](@ref) of the criterion that fired) and iteration count.
"""
function block_coordinate_result end

"""
    post_iteration!(state, method::AbstractBlockCoordinateMethod, prob, θ, iter) -> nothing

Per-iteration bookkeeping of [`BlockCoordinateDescent`](@ref), after the
weight block: best-model tracking and the counters a
[`StoppingCriterion`](@ref) reads. The default does nothing.
"""
post_iteration!(state, ::AbstractBlockCoordinateMethod, prob, θ, iter) = nothing

# Trust region is the OPTIMIZER's responsibility, not the framework's: the
# `update_model!` returns an accepted update (DLT is the exact minimizer; NLLSsolver owns its
# LM damping). So there is no trust-region hook — the loop takes the solver's
# result directly.

# =============================================================================
# Stopping criteria
# =============================================================================

"""
    StoppingCriterion

When a [`BlockCoordinateDescent`](@ref) run ends, and why. A criterion is an
immutable, concretely typed value; it answers through
[`stop_reason`](@ref)`(criterion, state, method, prob, θ, θ_old, iter)`, which
returns the termination record the run ends with, `(code::ReturnCode.T, rule::StopRule)` — `(Success, rule)` a
convergence, `(MaxIters, IterationCap)` the budget, `(Failure, NonFinite)` a failed model block, `(Terminated,
LeftBasin)` an instance's own — or [`CONTINUE`](@ref) `= (ReturnCode.Default, NoRule)` while the run goes on.
Nothing is stored in the criterion: the counters a criterion reads live in the solver state, written by
`post_iteration!`.

Criteria compose with `|` into a [`StopWhenAny`](@ref), evaluated left to right;
the first record other than `CONTINUE` wins. The tuple is part of the type, so
the composed check compiles to the chain of branches it stands for.

The Manopt.jl correspondence: Manopt's criteria are mutable functors that
store their reason; these are pure, and the reason is the return value.
"""
abstract type StoppingCriterion end

"""
    stop_reason(criterion, state, method, prob, θ, θ_old, iter) -> (ReturnCode.T, StopRule)

[`CONTINUE`](@ref), or the termination record the run ends with after iteration `iter`, which moved `θ_old` to `θ`
(`θ` is `nothing` when the model block failed). Pure: it reads the state and writes nothing. Implement it for a new
[`StoppingCriterion`](@ref).
"""
function stop_reason end

"""
    CONTINUE

`(ReturnCode.Default, NoRule)`: what [`stop_reason`](@ref) returns while a run goes on (SciML's `Default`: the solve
is not determined yet).
"""
const CONTINUE = (ReturnCode.Default, NoRule)
@inline _continues(rr::Tuple{ReturnCode.T, StopRule}) = first(rr) === ReturnCode.Default

"""
    StopAfterIteration(n)

Stops with `(MaxIters, IterationCap)` once `n` iterations have run (before the first,
when `n ≤ 0`). [`BlockCoordinateDescent`](@ref) adds it, LAST, from its
`max_iter`, so on the final iteration a method's own reason wins.
"""
struct StopAfterIteration <: StoppingCriterion
    n::Int
end
@inline stop_reason(c::StopAfterIteration, state, method, prob, θ, θ_old, iter) =
    iter >= c.n ? (ReturnCode.MaxIters, IterationCap) : CONTINUE

"""
    StopWhenSolveFailed()

Stops with `(Failure, NonFinite)` when the model block returned `nothing` (a singular
or non-finite solve, an underdetermined refit): checked right after
`update_model!`, before any other block runs. [`BlockCoordinateDescent`](@ref)
adds it FIRST.
"""
struct StopWhenSolveFailed <: StoppingCriterion end
@inline stop_reason(::StopWhenSolveFailed, state, method, prob, θ, θ_old, iter) =
    θ === nothing ? (ReturnCode.Failure, NonFinite) : CONTINUE

"""
    StopNever()

Never stops: the default [`stopping_criterion`](@ref) of a method that runs its
whole budget.
"""
struct StopNever <: StoppingCriterion end
@inline stop_reason(::StopNever, state, method, prob, θ, θ_old, iter) = CONTINUE

"""
    StopWhenIncrementLess(rtol)

Stops with `(Success, IncrementSize)` once the last iteration's increment, relative to the NEW model, is strictly
below `rtol`:

    sqrt(Σⱼ (θⱼ − θ_oldⱼ)²) / (sqrt(Σⱼ θⱼ²) + 1e-300) < rtol

`θ` and `θ_old` are indexable vectors of reals of one length (an `SVector`, a view of a parameter column), summed in
index order in their element type `T` (the guard `T(1e-300)`), compared with `rtol` as given. Isbits and pure, so
a device kernel evaluates it on its own parameter column (`stop_reason(c, nothing, nothing, nothing, θ, θ_old,
iter)`: the state, method and problem are not read). A failed model block (`θ = nothing`) is
[`StopWhenSolveFailed`](@ref)'s to end. Manopt.jl's `StopWhenChangeLess` measures the change against the old
iterate; this is the reweighted fits' rule (BlobBoards' per-blob IRLS).
"""
struct StopWhenIncrementLess{T <: Real} <: StoppingCriterion
    rtol::T
end
@inline function stop_reason(c::StopWhenIncrementLess, state, method, prob, θ, θ_old, iter)
    T = eltype(θ)
    diff_sq = zero(T)
    norm_sq = zero(T)
    for j in eachindex(θ, θ_old)
        d = θ[j] - θ_old[j]
        diff_sq += d * d
        norm_sq += θ[j] * θ[j]
    end
    return sqrt(diff_sq) / (sqrt(norm_sq) + T(1.0e-300)) < c.rtol ? (ReturnCode.Success, IncrementSize) : CONTINUE
end
@inline stop_reason(::StopWhenIncrementLess, state, method, prob, ::Nothing, θ_old, iter) = CONTINUE

"""
    StopWhenAny(criteria::Tuple)
    a | b

Stops at the first of `criteria`, in order, whose [`stop_reason`](@ref) is
not [`CONTINUE`](@ref), with that record. `a | b` builds one and flattens nested ones.
"""
struct StopWhenAny{C <: Tuple} <: StoppingCriterion
    criteria::C
end
@inline stop_reason(c::StopWhenAny, state, method, prob, θ, θ_old, iter) =
    _first_reason(c.criteria, state, method, prob, θ, θ_old, iter)

@inline _first_reason(::Tuple{}, state, method, prob, θ, θ_old, iter) = CONTINUE
@inline function _first_reason(cs::Tuple, state, method, prob, θ, θ_old, iter)
    rr = stop_reason(first(cs), state, method, prob, θ, θ_old, iter)
    _continues(rr) || return rr
    return _first_reason(Base.tail(cs), state, method, prob, θ, θ_old, iter)
end

Base.:|(a::StoppingCriterion, b::StoppingCriterion) = StopWhenAny((a, b))
Base.:|(a::StopWhenAny, b::StoppingCriterion) = StopWhenAny((a.criteria..., b))
Base.:|(a::StoppingCriterion, b::StopWhenAny) = StopWhenAny((a, b.criteria...))
Base.:|(a::StopWhenAny, b::StopWhenAny) = StopWhenAny((a.criteria..., b.criteria...))

# The criteria that can end a run before its first iteration: only the budget.
@inline _reason_at_start(c::StopAfterIteration, method) = c.n <= 0 ? (ReturnCode.MaxIters, IterationCap) : CONTINUE
@inline _reason_at_start(::StoppingCriterion, method) = CONTINUE
@inline _reason_at_start(c::StopWhenAny, method) = _first_reason_at_start(c.criteria, method)
@inline _first_reason_at_start(::Tuple{}, method) = CONTINUE
@inline function _first_reason_at_start(cs::Tuple, method)
    rr = _reason_at_start(first(cs), method)
    _continues(rr) || return rr
    return _first_reason_at_start(Base.tail(cs), method)
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

A [`BlockCoordinateDescent`](@ref) state that also keeps a per-iteration
trace: `iterates` (the model each iteration produced), `cost` (the state's
[`current_cost`](@ref) after `post_iteration!`), `s2` (the likelihood's
[`current_s2`](@ref), the squared scale, `NaN` where it holds none) and
`reason` (a `Ref` to the termination record, `(code, rule)`). Give it in place of the state; the
driver runs every block and every criterion on the wrapped `state`, so a
recorded run is the bare run, iterate for iterate. The bare state never
records — off by default, nothing to branch on.
"""
struct Recorded{S}
    state::S
    iterates::Vector{Any}
    cost::Vector{Float64}
    s2::Vector{Float64}
    reason::Base.RefValue{Tuple{ReturnCode.T, StopRule}}
end
Recorded(state) = Recorded(state, Any[], Float64[], Float64[], Ref(CONTINUE))

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

@inline _record_reason!(_state, rr::Tuple{ReturnCode.T, StopRule}) = nothing
@inline _record_reason!(r::Recorded, rr::Tuple{ReturnCode.T, StopRule}) = (r.reason[] = rr; nothing)

# =============================================================================
# The solver steps (Manopt names)
# =============================================================================

"""
    initialize_solver!(state, likelihood, method, prob, θ₀, criterion) -> (code::ReturnCode.T, rule::StopRule)

Start a [`BlockCoordinateDescent`](@ref) run: clears a [`Recorded`](@ref)
trace, and returns the record the run ends with before its first iteration
(`(MaxIters, IterationCap)` for a non-positive budget) or [`CONTINUE`](@ref). The state itself
is built beforehand — by `init_state!` or the caller.
"""
function initialize_solver!(state, likelihood, method, prob, θ₀, criterion::StoppingCriterion)
    _clear_record!(state)
    rr = _reason_at_start(criterion, method)
    _record_reason!(state, rr)
    return rr
end
@inline _clear_record!(_state) = nothing
function _clear_record!(r::Recorded)
    empty!(r.iterates); empty!(r.cost); empty!(r.s2)
    r.reason[] = CONTINUE
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
    stop_solver!(state, criterion, method, prob, θ, θ_old, iter) -> (code::ReturnCode.T, rule::StopRule)

The run's [`stop_reason`](@ref) after iteration `iter` (`θ` is `nothing` when
its model block failed): the record of the criterion that fired, [`CONTINUE`](@ref) while the run goes on;
recorded on a [`Recorded`](@ref) state.
"""
@inline function stop_solver!(state, criterion::StoppingCriterion, method, prob, θ, θ_old, iter::Int)
    rr = stop_reason(criterion, _bare(state), method, prob, θ, θ_old, iter)
    _record_reason!(state, rr)
    return rr
end

# =============================================================================
# The best iterate — best-tracking of an instance that claims descent
# =============================================================================

"""
    keep_best(best, θ, cost) -> (best′, improved::Bool)

Best-tracking for a [`BlockCoordinateDescent`](@ref) instance that claims
descent. The loop never rejects an iterate, so such an instance keeps its best
one as `best = Attributed(θ, (; cost))` (`cost` a `Float64`): its
`post_iteration!` offers every iterate here, and its `block_coordinate_result`
returns `best.value`. Returns `(Attributed(θ, (; cost)), true)` when `cost` is
STRICTLY below `best.cost`, and `(best, false)` otherwise. Pure and
allocation-free: the caller stores the result.
"""
@inline function keep_best(best::Attributed{M, @NamedTuple{cost::Float64}}, θ, cost::Real) where {M}
    improved = cost < best.cost
    return (improved ? Attributed{M, @NamedTuple{cost::Float64}}(θ, (; cost = Float64(cost))) : best), improved
end

# =============================================================================
# The front end — BlockCoordinateDescent, solved through solve/init
# =============================================================================

"""
    BlockCoordinateDescent(method::AbstractBlockCoordinateMethod; likelihood = FrozenNuisance())

Block coordinate descent as an algorithm, a [`RefineKind`](@ref): it refines the guess `θ₀` of a problem `prob`
(`RefineProblem(prob, θ₀)`) by the loop below — the `method`'s model block
and bookkeeping, the `likelihood`'s nuisance and weight blocks:

    solve(RefineProblem(prob, θ₀), BlockCoordinateDescent(method; likelihood); stops = StopAfterIteration(max_iter))
    c = init(RefineProblem(prob, θ₀), BlockCoordinateDescent(method; likelihood); stops)   # the state built once
    solve!(c); solve!(reinit!(c, θ₀′; f = prob′, alg = BlockCoordinateDescent(method′; likelihood′)))

`stops` (a [`StoppingCriterion`](@ref), required: the budget, `StopAfterIteration(max_iter)`, or a composition) runs
after the method's own [`stopping_criterion`](@ref): the run's criterion is
`StopWhenSolveFailed() | stopping_criterion(method) | stops`. The result is the method's `block_coordinate_result`.

The STATE is the run's workspace, so it is never the problem's context (`p` is refused): the one-shot `solve` reads
the method's [`init_state!`](@ref)`(method, prob, θ₀)` and builds no cache — a method that holds its buffers (its
workspace, a field of the method) seeds them there and returns its state, a [`Recorded`](@ref) one to record the run;
a cache (`init`) OWNS the state `init_state!` built, and `reinit!(c, θ₀′; f, alg, stops)` re-seeds it in place
through `init_state!(c's state, method′, prob′, θ₀′)` (default: a new state), so `solve!(reinit!(c, θ₀′; alg))`
allocates nothing for a method whose re-seed is in place, and `solve!` is inferred. The per-run seed a method's
state starts from (a consensus set, a seed score) is the method's own field, given anew through `alg`.

# The loop

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
One iteration (`step_solver!`) is one cycle through every block, in
the order model → residuals → nuisance → weights, then `post_iteration!`. See
[`update_nuisance!`](@ref) for why the nuisance block is not named "scale".

# Stopping

The run's criterion is

    StopWhenSolveFailed() | stopping_criterion(method) | stops

evaluated in that order by `stop_solver!` after every iteration: a
failed model block ends the run at once (`(Failure, NonFinite)`, no other block run);
otherwise the method's own criterion is read after `post_iteration!`, and only
then the budget — on the last iteration a method's record wins over
`(MaxIters, IterationCap)`. `StopAfterIteration(max_iter)` at `max_iter ≤ 0` runs no iteration. The result is
`block_coordinate_result(state, method, prob, θ, code, iterations, rule)` with `θ`
the last model the model block produced (θ₀ if none).

# Monotonicity

Instances: EM (under the Neal–Hinton free-energy view EM is coordinate ascent
on the free energy, i.e. descent on its negative); IRLS / half-quadratic
(VisualGeometryToolkitEstimation's `Irls{K}` LO strategy is one — the strategy, not this
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
| `initialize_solver!` | `initialize_solver!` |
| `step_solver!` | `step_solver!`: one BCD cycle, the four blocks + `post_iteration!` |
| `stop_solver!` | `stop_solver!` |
| `StoppingCriterion`, `get_reason` | [`StoppingCriterion`](@ref), [`stop_reason`](@ref) (pure; the record is returned) |
| `StopAfterIteration`, `StopWhenAny` (criteria or-ed) | [`StopAfterIteration`](@ref), [`StopWhenAny`](@ref) (criteria or-ed) |
| `RecordSolverState` | [`Recorded`](@ref) |
| `get_solver_result` | `block_coordinate_result` |

Argument order is Julia Base's: the inputs the loop mutates come first
(`state`, in the method below, then `likelihood`), then the inputs it only
reads (the method, the problem, the seed).
"""
struct BlockCoordinateDescent{M <: AbstractBlockCoordinateMethod, L <: AbstractWeightPolicy} <: AbstractEstimationAlgorithm
    method::M
    likelihood::L
end
BlockCoordinateDescent(method::AbstractBlockCoordinateMethod; likelihood::AbstractWeightPolicy = FrozenNuisance()) =
    BlockCoordinateDescent(method, likelihood)
estimator_kind(::BlockCoordinateDescent) = RefineKind()

mutable struct _BcdCache{F, U, A <: BlockCoordinateDescent, C <: StoppingCriterion, S}
    f::F
    u0::U
    alg::A
    stops::C
    state::S
end
__init(prob::RefineProblem, alg::BlockCoordinateDescent; stops::StoppingCriterion) = _bcd_cache(prob, alg, stops)
# The one-shot solve runs the loop on the run's state directly: no cache is built.
function __solve(prob::RefineProblem, alg::BlockCoordinateDescent; stops::StoppingCriterion)
    _refuse_box(prob, :BlockCoordinateDescent)
    return _block_coordinate_descent!(_bcd_state(alg.method, prob.f, prob.u0, prob.p), alg.likelihood, alg.method,
                                      prob.f, prob.u0, stops)
end
# The run's state: the method's `init_state!`. The state is the run's workspace, never the problem's context.
_bcd_state(method, f, u0, ::NullParameters) = init_state!(method, f, u0)
_bcd_state(_method, _f, _u0, p) = _refuse_bcd_context(p)
const _BCD_NO_CONTEXT = "BlockCoordinateDescent reads no context `p`: its state is the run's workspace — the method's " *
    "`init_state!(method, prob, θ₀)` builds it (a method holds its own buffers), a cache owns it (`init`) and " *
    "`reinit!(c, θ₀; alg)` re-seeds it"
_refuse_bcd_context(_p) = throw(ArgumentError(_BCD_NO_CONTEXT))
# The cache: it OWNS the state the method's `init_state!` built.
function _bcd_cache(prob::RefineProblem, alg::BlockCoordinateDescent, stops::StoppingCriterion)
    _refuse_box(prob, :BlockCoordinateDescent)
    return _BcdCache(prob.f, prob.u0, alg, stops, _bcd_state(alg.method, prob.f, prob.u0, prob.p))
end
solve!(c::_BcdCache) = _block_coordinate_descent!(c.state, c.alg.likelihood, c.alg.method, c.f, c.u0, c.stops)
# A new guess, data, algorithm configuration (of the cache's type: a new method value carries a new seed) or stops; the
# cache's state re-seeded through the method's `init_state!(state, …)` (in place for a reusable state).
function reinit!(c::_BcdCache{F, U, A, C, S}, u0 = c.u0; f = c.f, p = NullParameters(), alg::Union{Nothing, A} = nothing,
        stops::C = c.stops) where {F, U, A, C, S}
    p isa NullParameters || _refuse_bcd_context(p)
    alg === nothing || (c.alg = alg)
    c.f = f; c.u0 = u0; c.stops = stops
    c.state = init_state!(c.state, c.alg.method, f, u0)::S
    return c
end

# THE loop, on a criterion `stops` after the method's own (the front end's).
function _block_coordinate_descent!(state, likelihood::AbstractWeightPolicy, method::AbstractBlockCoordinateMethod,
        prob, θ₀, stops::StoppingCriterion)
    criterion = StopWhenSolveFailed() | stopping_criterion(method) | stops
    θ = θ₀
    iter = 0
    code, rule = initialize_solver!(state, likelihood, method, prob, θ₀, criterion)
    while code === ReturnCode.Default
        iter += 1
        θ_old = θ
        θ_new = step_solver!(state, likelihood, method, prob, θ, iter)
        if isnothing(θ_new)
            # A failed model block ends the run on the last model; the record is
            # StopWhenSolveFailed's (first in the criterion).
            code, rule = stop_solver!(state, criterion, method, prob, nothing, θ_old, iter)
            break
        end
        θ = θ_new
        code, rule = stop_solver!(state, criterion, method, prob, θ, θ_old, iter)
    end
    return block_coordinate_result(_bare(state), method, prob, θ, code, iter, rule)
end

public BlockCoordinateDescent, CONTINUE
