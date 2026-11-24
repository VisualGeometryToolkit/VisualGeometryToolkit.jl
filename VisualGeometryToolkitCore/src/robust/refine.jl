# =============================================================================
# The refine — Core's ONE host Levenberg–Marquardt solving a RefineProblem (decision 3; spec §4)
# =============================================================================
#
#     solve(RefineProblem(f, u0), LevenbergMarquardt(; optimization_strategy, damping); scale | width, stops)
#
# Moved from VisualGeometryToolkitEstimation 0.12 (fitting/_refine.jl, problems/manifold_providers.jl): one generic body
# over the interface of the problem's `f`, which its author implements on its own type (the design record's option A):
#
# - a CHART problem (an `AbstractConstraintProblem`: the RANSAC families) answers the chart contract — the chart at a
#   model (`model_chart`), the √w-weighted rows the LM descends at a chart (`chart_rows`, VGC's row protocol), the
#   model a chart stands for (`chart_model`) and the LM's size (`chart_size`) — plus the constraint protocol the
#   result is measured by (`data_size`, `codimension`, `minimal_sample_size`, `mahalanobis_distances!`);
# - a MULTI-BLOCK problem (an `AbstractBlockProblem`: the bundles) answers the block contract — its residual blocks
#   and variables at the guess (`bundle_blocks`) and its result from the solved blocks (`bundle_result`), solved by the
#   one bundle LM (`block_lm`, over Core's `LmProblem`).
#
# A refine starts from the caller's model and NEVER estimates a scale: it runs no scorer, forms no consensus set and
# has no thin-partition guard. Its robust loss runs at a width the caller states — `scale = σ` (or an upstream result,
# its `noise_scale`) times the strategy's efficiency constant, or `width = w` — and the result records the scale it ran
# with (`ScaleGiven`, `ScaleUpstream`, or `ScaleNone` for a width alone and for plain least squares). The strategies:
#
#   Triggs{K}  ONE robustified chart LM (`loss_robustifier` handed to NLLSsolver) — the class default of record and
#              the ONE fixed-scale robust path: no reweighting loop, the termination record the LM's
#   nothing    plain least squares over every datum (one chart LM), no scale
#
# An `Irls{K}` is refused, naming `Triggs{K}`: a refine runs one LM, and an Irls's cycles are a fit's and a RANSAC local
# optimization's. The result is `FitResult{M, RefineMetadata}`: the refined model, the per-datum whitened distances at
# it (`resid`), the weights and the scale record. The RANSAC-family seeded refine — a scorer, its consensus set, the
# strategy anchored on it — is VisualGeometryToolkitEstimation's `LocalOptimization`, not this.
# =============================================================================

# -----------------------------------------------------------------------------
# The metadata root: one weight kind per producer
# -----------------------------------------------------------------------------

"""
    AbstractFitMetadata{L}

The parent of a result metadata whose per-datum weights are its producer's: `L` is the PRODUCER (a robust strategy
`Irls{K}`/`Triggs{K}`, `nothing` for least squares, a scorer, `PosteriorWeightState` for the posterior-weight
iteration), and the weights' kind is the producer's ([`weight_kind`](@ref)`(L)`, the one method every such metadata
answers by). [`RefineMetadata`](@ref) is one; VisualGeometryToolkitEstimation's fit and RANSAC metadata are others.
"""
abstract type AbstractFitMetadata{L} end
weight_kind(::Type{<:AbstractFitMetadata{L}}) where {L} = weight_kind(L)
# The posterior-weight iteration produces posterior weights (the EM fits' metadata name it as their producer).
weight_kind(::Type{<:PosteriorWeightState}) = PosteriorWeights()

"""
    RefineMetadata{L, T, W} <: AbstractFitMetadata{L}

The metadata of a [`LevenbergMarquardt`](@ref) refine of a chart problem (`FitResult{M, RefineMetadata}`), read through
the accessors — [`weights`](@ref), [`weight_kind`](@ref), [`noise_scale`](@ref), [`scale_source`](@ref) — or by field.
The per-datum residual distances at the estimate are the result's `resid`.

- `weights::W`: one per datum at the estimate. For a robust refine (`L` a `Triggs` strategy) the loss weight
  `ρ′(r²)/ρ′(0)`, maximum 1, exactly 0 below `eps` ([`IrlsWeights`](@ref): no cut of its own,
  `inlier_mask(result, cut)`); for plain least squares (`L === Nothing`) a lazy all-ones vector
  ([`BinaryWeights`](@ref)).
- `scale::T`, `scale_source::ScaleSource`: the scale the refine ran with, as it was given (`ScaleGiven`,
  `ScaleUpstream`); `NaN` and `ScaleNone` for a width given alone and for plain least squares. A refine never
  estimates one.
- `loss_width::T`: the width the loss ran at — `width`, or the efficiency constant times `scale`; `NaN` for plain least
  squares.
- `residual_scale::T`: a DIAGNOSTIC, the median-norm scale of the residual norms at the problem's codimension
  (`median(‖r‖)/√χ²_d(½)`). It is never handed on: the hand-off is `noise_scale`, the scale the refine was given.

`L` is the strategy type (the producer, read by `weight_kind`), `T` the float type, `W` the weights' vector type.
"""
struct RefineMetadata{L, T <: AbstractFloat, W <: AbstractVector{T}} <: AbstractFitMetadata{L}
    weights::W
    scale::T
    scale_source::ScaleSource
    loss_width::T
    residual_scale::T
end
RefineMetadata{L}(weights::W, scale::Real, source::ScaleSource, loss_width::Real, residual_scale::Real) where {L, T, W <: AbstractVector{T}} =
    RefineMetadata{L, T, W}(weights, T(scale), source, T(loss_width), T(residual_scale))

# -----------------------------------------------------------------------------
# The chart contract — implemented by a problem's author on its own type
# -----------------------------------------------------------------------------

"""
    model_chart(f, model) -> chart

The chart a chart problem `f` refines on, anchored at `model` (a model of `model_type(f)`, or one `convert` labels into
it): the ONE seeding of every chart LM of `f` — a refine's, and VisualGeometryToolkitEstimation's local optimizations
and lifts. A contract generic of the [`LevenbergMarquardt`](@ref) refine (with [`chart_rows`](@ref),
[`chart_model`](@ref), [`chart_size`](@ref)).
"""
function model_chart end
model_chart(f, _model) = _no_chart_contract(f, model_chart)

"""
    chart_rows(f, chart, sqrtw) -> state

The linearization state whose √w-weighted rows (`sqrtw` per row) the chart LM descends at `chart`: a value answering
the row protocol ([`residual_rows!`](@ref), [`residual_jacobian_rows!`](@ref)), its per-datum norm the residual's own
unit — the unit a refine's loss width is in (the problem's `mahalanobis_distance`).
"""
function chart_rows end
chart_rows(f, _chart, _sqrtw) = _no_chart_contract(f, chart_rows)

"""
    chart_model(f, chart) -> model

The model of `model_type(f)` the chart stands for at its origin, in the problem's frame and type.
"""
function chart_model end
chart_model(f, _chart) = _no_chart_contract(f, chart_model)

"""
    chart_size(f) -> (rows, Val(dof))

The chart LM's size on `f`: its residual row count (`codimension(f)` rows per datum) and the chart's dof.
"""
function chart_size end
chart_size(f) = _no_chart_contract(f, chart_size)

"""
    guess_model(f, u0) -> model
    solution_model(f, θ, u0) -> model

The guess `u0` of a refine of `f` as a model of `model_type(f)`, and the refined model `θ` as the caller's form of the
guess: the one place a refine strips and reattaches what the inner loop does not carry — a physical guess's unit
(metres stripped at entry, reattached at exit) and a labelled guess's frames. Defaults: `convert(model_type(f), u0)`
(which labels an unlabeled map into the model type and refuses a relabelling) and `θ` itself. Contract generics of the
[`LevenbergMarquardt`](@ref) refine of a chart problem.
"""
guess_model(f, u0) = convert(model_type(f), u0)
@doc (@doc guess_model) solution_model(_f, θ, _u0) = θ

"""
    default_optimization_strategy(f) -> strategy

The optimization strategy of record of `f`'s problem class — what [`ClassDefault`](@ref)`()` resolves to in a
[`LevenbergMarquardt`](@ref) refine (the class's configuration of record: VisualGeometryToolkitEstimation's
`configs/default.toml` `[refine.<class>]` and the bundles' tables). A contract generic: a problem's author declares
its class's.
"""
function default_optimization_strategy end
default_optimization_strategy(f) = throw(ArgumentError(string("a ", nameof(typeof(f)), " has no class of record: give ",
    "LevenbergMarquardt(; optimization_strategy = …) (a Triggs{Shape} or nothing)")))
# A problem that implements no part of the chart contract: the one refusal, naming the missing method.
@noinline _no_chart_contract(f, g) = throw(ArgumentError(string("a ", nameof(typeof(f)), " implements no `", nameof(g),
    "` method: a LevenbergMarquardt refine of it needs the chart contract (model_chart, chart_rows, chart_model, chart_size)")))

# -----------------------------------------------------------------------------
# The chart refit — the one chart LM of a chart problem's model, and its zero-weight row rule
# -----------------------------------------------------------------------------

"""
    lm_workspace(ws, rows, Val(N)) -> ManifoldLmWorkspace{N}

The chart LM's workspace under the ONE reuse policy: `ws` when it is a `ManifoldLmWorkspace{N}` of `rows` rows,
otherwise a task-local workspace keyed by its type and size — built once per (task, size), reused by every later
chart LM of that size on that task (VGC's `task_buffers`; threaded callers never share).
"""
@inline function lm_workspace(ws, rows::Int, v::Val{N}) where {N}
    W = lm_workspace_type(v)
    ws isa W && length(sqrt_weights(ws)) == rows && return ws
    return task_buffers(W, () -> ManifoldLmWorkspace(rows, v), (:rvg_lm_ws, rows))
end

"""
    chart_refit(ws, f, chart, rows, weights; robustifier = nothing, stops = LmStops(), damping = MarquardtDamping())
        -> (model | nothing, (retcode, stop_rule), iterations)

THE chart refit — every chart LM of a chart problem's model (a [`LevenbergMarquardt`](@ref) refine's least squares and
its Triggs, VisualGeometryToolkitEstimation's local-optimization refits): from `chart`, the rows `rows(f, chart, √w)`
(a function of the [`chart_rows`](@ref) form; `chart_rows` itself, or a problem's own refit state) at √`weights`
(`nothing`: unit weight) in the workspace from `ws` ([`lm_workspace`](@ref)), ONE [`lmsolve_chart!`](@ref) at `stops`
and `damping`, robustified when a `robustifier` is given; the model re-anchored at the solution by the chart's
`recenter` and read by [`chart_model`](@ref). The model is `nothing` when the solve went non-finite.

A ZERO-WEIGHT ROW IS EXACTLY 0: a row whose √w is 0 (a datum the caller or the problem's own state weighs 0) and its
Jacobian row are set to 0 whatever the problem's row is, so a datum whose residual is Inf or NaN cannot reach the LM as
0·Inf = NaN. Every run is wrapped — ONE state type per workspace, so the kept NLLSsolver run (keyed by the state type)
is never rebuilt when a refit's weights gain or lose a zero — and a row at a nonzero √w is the problem's own, bit for
bit.
"""
function chart_refit(ws, f, ch, rows::R, weights; robustifier = nothing, stops::LmStops = LmStops(),
        damping::LmDamping = MarquardtDamping()) where {R}
    nrows, dof = chart_size(f)
    ews = lm_workspace(ws, nrows, dof)
    weights === nothing ? fill!(sqrt_weights(ews), 1.0) :
        fill_sqrt_weights!(sqrt_weights(ews), weights, data_size(f), nrows ÷ data_size(f))
    # after the state: a problem's own state may zero √w (PnP's out-of-reach anchor points, bearing pose's caller weights)
    lm = _ZeroWeightRows(rows(f, ch, sqrt_weights(ews)), sqrt_weights(ews), nrows)
    x, _, _, n_iter, record = robustifier === nothing ? lmsolve_chart!(ews, lm, stops, damping) :
        lmsolve_chart!(ews, lm, robustifier, Val(codimension(f)), stops, damping)
    any(isnan, x) && return nothing, (ReturnCode.Failure, NonFinite), n_iter
    return chart_model(f, recenter(ch, x)), record, n_iter
end

"""
    triggs_refit(ws, shape, f, model, loss; weights = nothing, stops = LmStops(), damping = MarquardtDamping())
        -> FitResult (an `lo_estimate`)

THE robustified chart LM — a [`LevenbergMarquardt`](@ref) refine's `Triggs{K}` and the Triggs local optimization of
VisualGeometryToolkitEstimation: `shape` at the deployed `loss`, frozen, as [`loss_robustifier`](@ref), and ONE
[`chart_refit`](@ref) from `model` over every datum at `weights` (`nothing`: unit weight), on `f`'s own chart and rows
([`model_chart`](@ref), [`chart_rows`](@ref)). A non-finite solve hands `model` back, `(Failure, NonFinite)`.
"""
function triggs_refit(ws, shape::AbstractLossShape, f, model, loss::DeployedLoss; weights = nothing,
        stops::LmStops = LmStops(), damping::LmDamping = MarquardtDamping())
    robustifier = loss_robustifier(shape, sqrt(sq_loss_width(loss)))
    m, (code, rule), n_iter = chart_refit(ws, f, model_chart(f, model), chart_rows, weights; robustifier, stops, damping)
    m === nothing && return lo_estimate(model, ReturnCode.Failure, NonFinite)
    return lo_estimate(m, code, rule; iterations = n_iter)
end

# A chart-LM state whose rows (and Jacobian rows) at a zero √w are exactly 0: the inner state's rows, then every row `j`
# with `sqrtw[j] == 0` overwritten with 0. The rows at a nonzero √w are the inner state's, untouched. Whether any row is
# at a zero √w is found once, at construction (`any_zero`: a refit with none skips the overwrite; nothing is
# allocated). The cost and normal-equation reductions are their defaults over these rows: a state that overrides them (a
# device state, VisualGeometryToolkitEstimation's camera refine) runs its own LM, not `chart_refit`.
struct _ZeroWeightRows{L}
    lm::L
    sqrtw::Vector{Float64}
    any_zero::Bool
end
_ZeroWeightRows(lm, sqrtw::Vector{Float64}, rows::Int) = _ZeroWeightRows(lm, sqrtw, any(iszero, view(sqrtw, 1:rows)))

function residual_rows!(r, s::_ZeroWeightRows, x)
    residual_rows!(r, s.lm, x)
    s.any_zero || return r
    sw = s.sqrtw
    @inbounds for j in eachindex(r)
        iszero(sw[j]) && (r[j] = zero(eltype(r)))
    end
    return r
end
function residual_jacobian_rows!(J, s::_ZeroWeightRows, x)
    residual_jacobian_rows!(J, s.lm, x)
    s.any_zero || return J
    sw = s.sqrtw
    @inbounds for k in axes(J, 2), j in axes(J, 1)
        iszero(sw[j]) && (J[j, k] = zero(eltype(J)))
    end
    return J
end

# -----------------------------------------------------------------------------
# The weights and the support of a weighted result
# -----------------------------------------------------------------------------

"""
    normalized_loss_weight(shape, s, c²) -> Float64

THE normalized loss weight `ρ′(s)/ρ′(0)` ∈ [0, 1] at the squared norm `s` and squared loss width `c²` ([`IrlsWeights`](@ref):
a maximum of 1) — every robust refine's and every fixed-scale reweighted fit's. ρ′(0) > 0 for every shipped shape at a
positive width; a zero (or NaN) width gives 0/0 = NaN (the strategies refuse such a width before a weight is formed).
"""
@inline normalized_loss_weight(shape, s, c²) = weight(shape, s, c²) / weight(shape, zero(s), c²)

"""
    loss_weights(shape, c², sq) -> Vector{Float64}

THE loss weights of a robust refine (the chart refines and the bundles) from the squared residual norms `sq`:
[`normalized_loss_weight`](@ref) at the squared width `c²`, and a weight below `eps` an exact 0 (out of the support).
"""
loss_weights(shape, c², sq) =
    [(x = Float64(normalized_loss_weight(shape, Float64(s), Float64(c²))); x < eps(Float64) ? 0.0 : x) for s in sq]

"""
    weighted_support(w, sq) -> (n, Σw, Σw·sq)

`n`, `Σ wᵢ` and `Σ wᵢ sqᵢ` over the support (every weight but an exact 0; weights are non-negative, so a NaN counts and
makes the rms NaN), in index order and in Float64; `sq` holds the squared residuals, one per weight. Pure: reads `w`.
The sums [`weighted_support_rms`](@ref) reduces, for a result that records them.
"""
function weighted_support(w::AbstractVector, sq)
    length(sq) == length(w) || throw(DimensionMismatch("$(length(w)) weights, $(length(sq)) squared residuals"))
    n = 0
    sw = 0.0
    swsq = 0.0
    @inbounds for (i, s) in zip(eachindex(w), sq)
        wi = w[i]
        iszero(wi) && continue
        n += 1
        sw += Float64(wi)
        swsq += Float64(wi) * s
    end
    return n, sw, swsq
end

"""
    weighted_support_rms(w, sq) -> (n, rms)

THE `n` and `rms` of every weighted result (the robust-estimation contract, §12): `n` the support `count(w > 0)` and
`rms = √(Σ wᵢ sqᵢ / Σ wᵢ)` (NaN without support), over the squared residuals `sq` (one per weight), in index order and in
Float64 — every refine, every reweighted and EM fit, and a RANSAC result under a posterior scorer.
"""
function weighted_support_rms(w::AbstractVector, sq)
    n, sw, swsq = weighted_support(w, sq)
    return n, (sw > 0 ? sqrt(swsq / sw) : NaN)
end

# -----------------------------------------------------------------------------
# The robust loss from the strategy and its scale input
# -----------------------------------------------------------------------------

# A refine runs `Triggs{K}` or `nothing` — nothing else hosts a refine without a consensus set. An `Irls{K}` is refused,
# naming the Triggs it becomes (its shape and width mode kept): a refine's one fixed-scale robust path is ONE robustified
# LM, with no reweighting cycles to count.
_refine_strategy(lo::Nothing) = lo
function _refine_strategy(lo::Union{Irls, Triggs})
    strategy_loss(lo) === GaussianShape && throw(ArgumentError(
        "$(lo_label(lo)) is plain least squares: a refine spells it `optimization_strategy = nothing`"))
    lo_width(lo) isa LossWidth && throw(ArgumentError(
        "a refine's loss width is its solve keyword `width = w`, not the strategy's `loss_width`"))
    lo isa Irls && throw(ArgumentError(
        "$(lo_label(lo)) reweights in cycles, and a refine runs ONE robustified LM: give " *
            "`optimization_strategy = Triggs{$(nameof(strategy_loss(lo)))}(…)`, the same loss at the same efficiency constant"))
    return lo
end
_refine_strategy(lo) = throw(ArgumentError(
    "a LevenbergMarquardt refine takes `optimization_strategy = Triggs{Shape}(…)` or `nothing` (plain least " *
        "squares), got $(nameof(typeof(lo))): it needs a scorer's consensus set, which a refine has none — " *
        "run it from the seed with VisualGeometryToolkitEstimation's `LocalOptimization`"))

# The loss a robust refine (or bundle) deploys from its scale input at codimension `d`, frozen: the shape at
# `c × scale` (`c` the strategy's efficiency constant), or at the caller's `width` — factored as the RANSAC local
# optimization deploys a multiple of its scale and an absolute width (`deploy(shape, s2; efficiency_constant)`), so the
# same width is the same loss.
function _refine_loss(lo::Union{Irls, Triggs}, input::ScaleInput, d::Integer)
    shape = strategy_loss(lo)()
    isnan(input.width) &&
        return deploy(shape, max(Float64(input.scale)^2, eps()); efficiency_constant = efficiency_constant(lo, d))
    lo_width(lo) === nothing || throw(ArgumentError(
        "`width = $(input.width)` is the loss width itself: the strategy's efficiency constant would be ignored — " *
            "give the strategy none, or give `scale`"))
    return deploy(shape, max(Float64(input.width)^2, eps()); efficiency_constant = 1.0)
end

"""
    bundle_loss(optimization_strategy; scale | width, codimension = 2, caller = "bundle_loss")
        -> (; strategy, input, loss)

The robust loss a refine deploys from its strategy and scale input, by the refine contract (the
[`LevenbergMarquardt`](@ref) refine of a chart problem or a bundle, and a downstream bundle of its own):
`optimization_strategy` a `Triggs{K}` (the one fixed-scale robust path, one robustified LM) or `nothing` (plain least
squares), refused otherwise — an `Irls{K}` naming its `Triggs{K}`, a `GaussianShape` strategy (least squares is
`nothing`) and a strategy holding an absolute `loss_width` (that is the `width` keyword); and EXACTLY ONE of `scale = σ`
(the observations' noise scale, or an upstream result; [`scale_input`](@ref)) or `width = w` for a robust strategy,
NEITHER for least squares (a refine estimates no scale). The loss is the strategy's shape deployed at the residual's
`codimension` (2 for a pixel reprojection error): at `c × scale`, `c` the strategy's efficiency constant (the one it
states, the plain multiple, or `efficiency_constant(K, codimension)`; a truncation states one), or at `width` — which
refuses a strategy that also states a constant. `caller` names the refine in the messages.

Returns `strategy` (the strategy, or `nothing`), `input` (the [`ScaleInput`](@ref), or `nothing`) and `loss` (the
[`DeployedLoss`](@ref), or `nothing`): a bundle robustifies its rows with `loss_robustifier(strategy_loss(strategy)(),
sqrt(sq_loss_width(loss)))` and records the scale it was given through the stack's accessors: `scale_source(input)`
(`ScaleNone` for least squares, `input === nothing`, and for a `width`) and, when that is not `ScaleNone`,
`noise_scale(input)`.
"""
function bundle_loss(lo; scale = nothing, width = nothing, codimension::Integer = 2,
                     caller::AbstractString = "bundle_loss")
    lo = _refine_strategy(lo)
    if lo === nothing
        (scale === nothing && width === nothing) || throw(ArgumentError(
            "$caller: plain least squares (optimization_strategy = nothing) reads no scale; got " *
                (scale === nothing ? "width = $width" : "scale = $(repr(scale))")))
        return (; strategy = lo, input = nothing, loss = nothing)
    end
    input = scale_input(scale, width)
    return (; strategy = lo, input, loss = _refine_loss(lo, input, codimension))
end

# The parsed refine is `bundle_loss`'s `(; strategy, input, loss)` (input and loss `nothing` for L2), on every path.
# No deployed loss (least squares): no width, NaN — what a refine's or a fit's metadata records for it.
sq_loss_width(::Nothing) = NaN

# The strategy the algorithm states, or (`ClassDefault()`) the problem class's of record.
_resolved_strategy(f, ::ClassDefault) = default_optimization_strategy(f)
_resolved_strategy(_f, strategy) = strategy

_refine_spec(f, alg::LevenbergMarquardt; scale, width) = bundle_loss(_resolved_strategy(f, alg.optimization_strategy);
    scale, width, codimension = codimension(f), caller = "LevenbergMarquardt")

"""
    recorded_scale(input) -> Float64

The scale a refine or a fit records from its parsed scale input (a [`ScaleInput`](@ref), or `nothing` for least
squares): the given scale, `NaN` for least squares and for a `width` (its scale is `NaN`); its source is
`scale_source(input)`.
"""
recorded_scale(input) = input === nothing ? NaN : Float64(input.scale)

"""
    scale_record(parsed) -> (; scale, scale_source, loss_width)

What a refine or a bundle records of its parsed refine (`parsed` with fields `input` and `loss`, as
[`bundle_loss`](@ref) returns): the scale it was given ([`recorded_scale`](@ref)), its source (`scale_source(input)`,
`ScaleNone` for least squares) and the deployed loss's width (`NaN` for least squares). Splat it, in this order, into a
metadata constructor.
"""
scale_record(parsed) = (; scale = recorded_scale(parsed.input), scale_source = scale_source(parsed.input),
                          loss_width = sqrt(sq_loss_width(parsed.loss)))

# -----------------------------------------------------------------------------
# The LevenbergMarquardt refine of a chart problem
# -----------------------------------------------------------------------------

default_algorithm(::RefineProblem{<:AbstractConstraintProblem}) = LevenbergMarquardt()

"""
    solve(RefineProblem(f, u0), LevenbergMarquardt(; optimization_strategy, damping); scale | width, stops = default_stops(f))
        -> FitResult{M, RefineMetadata}

The refine of a chart problem `f` (a VisualGeometryToolkitEstimation family problem: `HomographyProblem(cs)`, …) from
the guess `u0`: ONE chart LM ([`chart_refit`](@ref)) over every datum — robustified by a `Triggs{K}` strategy at the
loss its scale input deploys ([`bundle_loss`](@ref): `scale = σ` or an upstream result, or `width = w`), or plain
least squares (`optimization_strategy = nothing`, neither keyword) — at `stops` and the algorithm's `damping`.
`ClassDefault()` is the class's strategy of record ([`default_optimization_strategy`](@ref)).

The result: `u` the refined model (the guess when the solve went non-finite), the LM's termination record and accepted
steps (`stats.nsteps`), `resid` each datum's whitened distance at it (`mahalanobis_distances!`), `rms` the weighted
residual rms `√(Σ wᵢrᵢ² / Σ wᵢ)` and `n` the support (`count(w > 0)`), and [`RefineMetadata`](@ref). Fewer data than a
minimal sample: the guess back unrefined, `(InitialFailure, NoRule)`, `n` 0 and NaN distances. A box (`lb`, `ub`) is
refused: the host LM is unbounded. A refine reads no context: its `p` is `NullParameters` (a stray `p` has no method, as
any setting the solve does not take; a seeded local optimization's `p = (; extent)` is VisualGeometryToolkitEstimation's
`LocalOptimization`).
"""
function __solve(prob::RefineProblem{<:AbstractConstraintProblem, <:Any, NullParameters}, alg::LevenbergMarquardt;
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f))
    _refuse_box(prob)
    f = prob.f
    spec = _refine_spec(f, alg; scale, width)
    n = data_size(f)
    n >= minimal_sample_size(f) || return _refine_too_few(f, spec, prob.u0, n)
    return _refine_pipeline(f, spec, prob.u0, stops, alg.damping)
end

_refuse_box(prob::RefineProblem) = (prob.lb === nothing && prob.ub === nothing) || throw(ArgumentError(
    "LevenbergMarquardt is unbounded: a RefineProblem's `lb`/`ub` is StaticLevenbergMarquardt's (the device LM)"))

function _refine_pipeline(f, spec::NamedTuple, θ₀, stops::LmStops, damping::LmDamping)
    L = typeof(spec.strategy)
    M = model_type(f)
    θ = guess_model(f, θ₀)
    θ̂, (code, rule), iters = _refine_run(spec.strategy, f, θ, spec, stops, damping)
    θ̂ = convert(M, θ̂)::M   # the problem's model type on every path (a failed LM returns its seed), so the result is inferred
    n = data_size(f)
    r = Vector{Float64}(undef, n)
    mahalanobis_distances!(r, f, θ̂)
    sq = [abs2(x) for x in r]
    w = _refine_weights(spec.strategy, spec, sq)
    support, rms = weighted_support_rms(w, sq)   # n the support, rms weighted
    rs = _mad_scale_norms(; norms = r, codimension = codimension(f))   # the residual norms (mahalanobis_distances!)
    md = RefineMetadata{L}(w, scale_record(spec)..., rs)
    return FitResult(solution_model(f, θ̂, θ₀), code, rule, SolveStats(; nsteps = iters), rms, support, md; resid = r)
end

# The weights at the estimate from the squared residuals `sq`: the loss's, or L2's lazy all ones.
_refine_weights(::Nothing, _spec, sq) = unit_weights(Float64, length(sq))
_refine_weights(lo, spec, sq) = loss_weights(strategy_loss(lo)(), sq_loss_width(spec.loss), sq)

# Too few data for a minimal sample: the guess, unrefined, with no residual.
function _refine_too_few(f, spec::NamedTuple, θ₀, n::Int)
    L = typeof(spec.strategy)
    w = L === Nothing ? unit_weights(Float64, n) : zeros(n)   # L2's weights are its unit weights; nothing was fitted (n = 0)
    md = RefineMetadata{L}(w, scale_record(spec)..., NaN)
    return FitResult(solution_model(f, guess_model(f, θ₀), θ₀), ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, md; resid = fill(NaN, n))
end

# Plain least squares: ONE chart LM from θ over every datum at unit weight, on the rows Triggs robustifies
# (`chart_rows`), so L2 and Triggs at an unbounded width minimise the same cost; θ back on a non-finite solve.
function _refine_run(::Nothing, f, θ, _spec, stops::LmStops, damping::LmDamping)
    m, record, n_iter = chart_refit(nothing, f, model_chart(f, θ), chart_rows, nothing; stops, damping)
    return something(m, θ), record, n_iter
end
# Triggs: ONE robustified chart LM at the frozen width (`triggs_refit`, the RANSAC local optimization's own).
function _refine_run(lo::Triggs, f, θ, spec, stops::LmStops, damping::LmDamping)
    r = triggs_refit(nothing, strategy_loss(lo)(), f, θ, spec.loss; stops, damping)
    return r.u, (r.retcode, r.stop_rule), r.stats.nsteps
end

# -----------------------------------------------------------------------------
# The LevenbergMarquardt refine of a bundle — the block contract over `LmProblem`
# -----------------------------------------------------------------------------

"""
    default_stops(f) -> LmStops

The stop settings a [`LevenbergMarquardt`](@ref) refine of `f` runs at when its solve names none: `LmStops()` (Core's
`[lm]`) unless `f`'s class has its own configuration of record (VisualGeometryToolkitEstimation's camera-and-poses
bundles: `[CameraPosesProblem refine]`). A contract generic.
"""
default_stops(_f) = LmStops()

"""
    AbstractBlockProblem

The parent of a multi-block refine's `f` (a bundle adjustment, a camera regression: heterogeneous variables and residual blocks, solved
over Core's [`LmProblem`](@ref)). Its author implements the block contract: [`block_refit`](@ref) — the blocks and
variables at the guess, solved by [`block_lm`](@ref), the result read back — and, for each residual block type,
[`with_robustifier`](@ref). Solved by `solve(RefineProblem(f, u0), LevenbergMarquardt(); scale | width, stops)`, the
one host LM; `ClassDefault()` is `default_optimization_strategy(f)`, the default stops `default_stops(f)`.
"""
abstract type AbstractBlockProblem end

"""
    block_refit(f, u0, loss, stops, damping) -> FitResult

The block contract's one hook: the refine of the bundle `f` from the guess `u0` — its residual blocks and variables,
solved by [`block_lm`](@ref) (once, or per stage), the estimate and its metadata assembled into a `FitResult`.
`loss` is the parsed refine, `(; strategy, input, loss)` ([`bundle_loss`](@ref) at `codimension(f)`); `stops` and
`damping` the solve's and the algorithm's.
"""
function block_refit end
# Every pair the contract has no method for (a problem implementing none, or a guess of another form) is the pair's
# `MethodError`, stated once here.
block_refit(f, u0, loss, stops, damping) = throw(MethodError(block_refit, (f, u0, loss, stops, damping)))

"""
    with_robustifier(block, sqrt_weight, robustifier) -> block

The residual block `block` with its rows scaled by `sqrt_weight` and robustified by `robustifier` (a
[`loss_robustifier`](@ref), or NLLSsolver's `NoRobust()`): what [`block_lm`](@ref) runs a robust refine on. A
contract generic: a bundle's author adds it for each of its block types.
"""
function with_robustifier end
with_robustifier(block, _sqrt_weight, _robustifier) = throw(ArgumentError(string("a residual block ", nameof(typeof(block)),
    " implements no `with_robustifier` method: a robust bundle refine needs it")))

codimension(::AbstractBlockProblem) = 2   # a bundle's residual is a pixel reprojection error unless its author says otherwise

default_algorithm(::RefineProblem{<:AbstractBlockProblem}) = LevenbergMarquardt()

"""
    solve(RefineProblem(f::AbstractBlockProblem, u0), LevenbergMarquardt(; optimization_strategy, damping);
          scale | width, stops = default_stops(f)) -> FitResult

The refine of a bundle: the strategy and its scale input parsed by the refine contract ([`bundle_loss`](@ref): a
`Triggs{K}` at `scale = σ` or `width = w`, or `nothing`, plain least squares), then the bundle's own
[`block_refit`](@ref), which solves through [`block_lm`](@ref). A box is refused (the host LM is unbounded).
"""
function __solve(prob::RefineProblem{<:AbstractBlockProblem, <:Any, NullParameters}, alg::LevenbergMarquardt;
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f))
    _refuse_box(prob)
    f = prob.f
    loss = bundle_loss(_resolved_strategy(f, alg.optimization_strategy); scale, width, codimension = codimension(f),
                       caller = string(nameof(typeof(f))))
    return block_refit(f, prob.u0, loss, stops, alg.damping)
end

# Observation `k`'s squared norm: its `d` rows' squares summed in order.
@inline function _block_sq(r, d, k)
    o = d * (k - 1)
    s = r[o + 1]^2
    for j in 2:d
        s += r[o + j]^2
    end
    return s
end

"""
    block_lm(blocks, variables, unknowns, loss, stops; damping = MarquardtDamping())
        -> (; variables, code, rule, iterations, residuals, S, ν, rms, n, weights)

THE bundle solve: the residual `blocks` (one per observation, each with the same number of rows, its residual's
codimension: two for a pixel reprojection error; each answering [`with_robustifier`](@ref))
on `variables` (`unknowns` free coordinates), Core's one multi-block LM ([`LmProblem`](@ref) + [`lmsolve_chart!`](@ref))
at `stops` and `damping`. `loss` is the parsed refine (`(; strategy, input, loss)`, [`bundle_loss`](@ref)): plain
least squares is the whole solve for `strategy = nothing`; a robust refine has NO least-squares stage — ONE robustified
LM from the start at the frozen loss (Triggs). Returns the variables at the estimate, the termination record, the
accepted steps, the rows themselves, their sum of squares `S`, `ν = rows − unknowns`, the per-observation loss weights
(ρ′/ρ′(0), all ones under least squares), and `n`, `rms` their support and weighted rms ([`weighted_support_rms`](@ref)).
"""
function block_lm(blocks::AbstractVector, vars0::AbstractVector, p::Integer, loss, stops::LmStops;
        damping::LmDamping = MarquardtDamping())
    lo = loss.strategy
    costs = convert(Vector{_element_union(blocks)}, blocks)
    n = length(costs)
    if lo === nothing
        lp = LmProblem(copy(vars0), costs; damping)
    else
        shape = strategy_loss(lo)()
        lp = LmProblem(copy(vars0), [with_robustifier(c, 1.0, loss_robustifier(shape, sqrt(sq_loss_width(loss.loss)))) for c in costs]; damping)
    end
    sol = lmsolve_chart!(lp, stops)
    variables = lp.variables
    r = nlls_rows(costs, variables)   # the rows themselves
    d = length(r) ÷ max(n, 1)         # each block's rows: its residual's codimension
    sq = [_block_sq(r, d, k) for k in 1:n]   # each observation's squared norm, once
    weights = lo === nothing ? unit_weights(Float64, n) : loss_weights(shape, sq_loss_width(loss.loss), sq)   # ρ′/ρ′(0) at the estimate
    S = sum(abs2, r)
    support, rms = weighted_support_rms(weights, sq)
    return (; variables, sol.code, sol.rule, sol.iterations, residuals = r, S, ν = length(r) - p, rms, n = support, weights)
end
