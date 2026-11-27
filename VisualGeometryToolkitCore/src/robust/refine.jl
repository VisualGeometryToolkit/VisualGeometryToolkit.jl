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
# - a MULTI-BLOCK problem (an `AbstractBlockProblem`: the bundles) answers the block contract — its variables and
#   residual blocks at the guess (`block_structure`: `block_variables`, `residual_blocks`; `fixed_variables`) and its
#   result from the solved variables (`residual_summary`, `block_result`), solved by Core's one multi-block LM.
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
model_chart(f, _model) = _no_method(f, model_chart, _CHART_CONTRACT)

"""
    chart_rows(f, chart, sqrtw) -> state

The linearization state whose √w-weighted rows (`sqrtw` per row) the chart LM descends at `chart`: a value answering
the row protocol ([`residual_rows!`](@ref), [`residual_jacobian_rows!`](@ref)), its per-datum norm the residual's own
unit — the unit a refine's loss width is in (the problem's `mahalanobis_distance`).
"""
function chart_rows end
chart_rows(f, _chart, _sqrtw) = _no_method(f, chart_rows, _CHART_CONTRACT)

"""
    chart_model(f, chart) -> model

The model of `model_type(f)` the chart stands for at its origin, in the problem's frame and type.
"""
function chart_model end
chart_model(f, _chart) = _no_method(f, chart_model, _CHART_CONTRACT)

"""
    chart_size(f) -> (rows, Val(dof))
    chart_size(f, u0) -> (rows, Val(dof))

The chart LM's size on `f`: its residual row count (`codimension(f)` rows per datum) and the chart's dof. The
two-argument form is the size at the guess `u0`, for a chart problem whose dof depends on it (a camera regression: the
guess's family, its lens size, what it holds); its default is `chart_size(f)`. A chart-problem refine reads it once,
at `init` (`chart_size(f, u0)`): when the dof is not inferable from `typeof(u0)`, that is one function barrier per
`init`, never on a `solve!`/`reinit!` path.
"""
function chart_size end
chart_size(f) = _no_method(f, chart_size, _CHART_CONTRACT)
chart_size(f, _u0) = chart_size(f)

"""
    refit_chart(f, model, weights) -> chart | nothing
    refit_rows(f, chart, sqrtw) -> state

The chart and the rows of a local optimization's least-squares refit of `f` (the summary-free refine, `summary =
nothing`, without a robust strategy; a `Triggs{K}` refit reads `model_chart` and `chart_rows`): the chart
at `model` for the per-datum `weights` (`nothing`: unit weight) — `nothing` when the refit declines (an
underdetermined weighting, a model with no chart; the result's `u` is then `nothing`, `(InitialFailure, NoRule)`) —
and the √w-weighted rows the LM descends there. Contract generics; the defaults are the refine's own,
[`model_chart`](@ref)`(f, model)` and [`chart_rows`](@ref)`(f, chart, sqrtw)`. A family overrides them where its refit
seeds or whitens otherwise (a weight-aware seed chart, rows in another unit than its distances).
"""
function refit_chart end
refit_chart(f, model, _weights) = model_chart(f, model)
@doc (@doc refit_chart) function refit_rows end
refit_rows(f, ch, sqrtw) = chart_rows(f, ch, sqrtw)

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
const _CHART_CONTRACT = "a LevenbergMarquardt refine of it needs the chart contract (model_chart, chart_rows, chart_model, chart_size)"

# -----------------------------------------------------------------------------
# The chart refit — the one chart LM of a chart problem's model, and its zero-weight row rule
# -----------------------------------------------------------------------------

"""
    _lm_workspace(ws, rows, Val(N)) -> _ManifoldLmWorkspace{N}

The chart LM's workspace under the ONE reuse policy: `ws` when it is a `_ManifoldLmWorkspace{N}` of `rows` rows,
otherwise a task-local workspace keyed by its type and size — built once per (task, size), reused by every later
chart LM of that size on that task (VGC's `task_buffers`; threaded callers never share).
"""
@inline function _lm_workspace(ws, rows::Int, v::Val{N}) where {N}
    W = _lm_workspace_type(v)
    ws isa W && length(sqrt_weights(ws)) == rows && return ws
    return task_buffers(W, () -> _ManifoldLmWorkspace(rows, v), (:rvg_lm_ws, rows))
end

"""
    _chart_refit(ws, f, chart, rows, weights; robustifier = nothing, stops = LmStops(), damping = MarquardtDamping())
        -> FitResult

THE chart refit — every chart LM of a chart problem's model (a [`LevenbergMarquardt`](@ref) refine's least squares and
its Triggs, VisualGeometryToolkitEstimation's local-optimization refits): from `chart`, the rows `rows(f, chart, √w)`
(a function of the [`chart_rows`](@ref) form; `chart_rows` itself, or a problem's own refit state) at √`weights`
(`nothing`: unit weight) in the workspace from `ws` (`_lm_workspace`), ONE `_lmsolve_chart!` at `stops`
and `damping`, robustified when a `robustifier` is given; the model re-anchored at the solution by the chart's
`recenter` and read by [`chart_model`](@ref). The result's `u` is that model, its termination record and
`stats.nsteps` the LM's; `u` is `nothing`, `(Failure, NonFinite)`, when the solve went non-finite.

A ZERO-WEIGHT ROW IS EXACTLY 0: a row whose √w is 0 (a datum the caller or the problem's own state weighs 0) and its
Jacobian row are set to 0 whatever the problem's row is, so a datum whose residual is Inf or NaN cannot reach the LM as
0·Inf = NaN. Every run is wrapped — ONE state type per workspace, so the kept NLLSsolver run (keyed by the state type)
is never rebuilt when a refit's weights gain or lose a zero — and a row at a nonzero √w is the problem's own, bit for
bit.
"""
function _chart_refit(ws, f, ch, rows::R, weights; robustifier = nothing, stops::LmStops = LmStops(),
        damping::LmDamping = MarquardtDamping(), callback = nothing) where {R}
    nrows, dof = chart_size(f)
    ews = _lm_workspace(ws, nrows, dof)
    weights === nothing ? fill!(sqrt_weights(ews), 1.0) :
        _fill_sqrt_weights!(sqrt_weights(ews), weights, data_size(f), nrows ÷ data_size(f))
    # after the state: a problem's own state may zero √w (PnP's out-of-reach anchor points, bearing pose's caller weights)
    lm = _ZeroWeightRows(rows(f, ch, sqrt_weights(ews)), sqrt_weights(ews), nrows)
    r = robustifier === nothing ? _lmsolve_chart!(ews, lm, stops, damping; callback) :
        _lmsolve_chart!(ews, lm, robustifier, Val(codimension(f)), stops, damping; callback)
    x = r.u
    stats = SolveStats(; nsteps = r.stats.nsteps)
    any(isnan, x) && return FitResult(nothing, ReturnCode.Failure, NonFinite, stats, NaN, 0, nothing)
    return FitResult(chart_model(f, recenter(ch, x)), r.retcode, r.stop_rule, stats, NaN, 0, nothing)
end

"""
    _triggs_refit(ws, shape, f, model, loss; weights = nothing, stops = LmStops(), damping = MarquardtDamping())
        -> FitResult (an `lo_estimate`)

THE robustified chart LM — a [`LevenbergMarquardt`](@ref) refine's `Triggs{K}` and the Triggs local optimization of
VisualGeometryToolkitEstimation: `shape` at the deployed `loss`, frozen, as [`loss_robustifier`](@ref), and ONE
`_chart_refit` from `model` over every datum at `weights` (`nothing`: unit weight), on `f`'s own chart and rows
([`model_chart`](@ref), [`chart_rows`](@ref)). A non-finite solve hands `model` back, `(Failure, NonFinite)`.
"""
function _triggs_refit(ws, shape::AbstractLossShape, f, model, loss::DeployedLoss; weights = nothing,
        stops::LmStops = LmStops(), damping::LmDamping = MarquardtDamping(), callback = nothing)
    robustifier = loss_robustifier(shape, sqrt(sq_loss_width(loss)))
    r = _chart_refit(ws, f, model_chart(f, model), chart_rows, weights; robustifier, stops, damping, callback)
    r.u === nothing && return lo_estimate(model, ReturnCode.Failure, NonFinite)
    return lo_estimate(r.u, r.retcode, r.stop_rule; iterations = r.stats.nsteps)
end

# A chart-LM state whose rows (and Jacobian rows) at a zero √w are exactly 0: the inner state's rows, then every row `j`
# with `sqrtw[j] == 0` overwritten with 0. The rows at a nonzero √w are the inner state's, untouched. Whether any row is
# at a zero √w is found once, at construction (`any_zero`: a refit with none skips the overwrite; nothing is
# allocated). The cost and normal-equation reductions are their defaults over these rows: a state that overrides them (a
# device state, VisualGeometryToolkitEstimation's camera refine) runs its own LM, not `_chart_refit`.
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
NEITHER for least squares (a refine estimates no scale). A `width` that is a [`DeployedLoss`](@ref) of the
strategy's shape runs as deployed, its squared width unchanged — under any `Triggs{K}`, `Triggs{GaussianShape}` at its
deployed width included (a RANSAC local optimization's frozen loss); the shape-only spelling of a Gaussian strategy is
still refused. The loss is the strategy's shape deployed at the residual's
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
    # a deployed loss as the width runs as deployed under its Triggs strategy, a Gaussian one included (the RANSAC
    # local optimization's ℓ2-Triggs at its deployed width: the robustified LM it always ran)
    lo isa Triggs && width isa DeployedLoss && return _deployed_refine(lo, scale, width)
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
# A `width` that is a loss already deployed (a RANSAC local optimization's frozen loss): used as it is, so its squared
# width is the caller's bit for bit; its shape must be the strategy's.
function _deployed_refine(lo::Triggs, scale, loss::DeployedLoss)
    loss.value isa strategy_loss(lo) || throw(ArgumentError(string("`width` is a deployed ", nameof(typeof(loss.value)),
        " loss, the strategy's shape is ", nameof(strategy_loss(lo)))))
    lo_width(lo) === nothing || throw(ArgumentError(
        "`width` is the loss itself: the strategy's efficiency constant would be ignored — give the strategy none"))
    scale === nothing || throw(ArgumentError("give `scale` or `width`, not both (`width` is a deployed loss)"))
    # its width as given, unbounded included (ℓ2's deployment at s2 = Inf): no positivity re-check of a deployed width
    input = ScaleInput(NaN, sqrt(sq_loss_width(loss)), ScaleNone)
    return (; strategy = lo, input, loss)
end

# The parsed refine is `bundle_loss`'s `(; strategy, input, loss)` (input and loss `nothing` for L2), on every path.
# No deployed loss (least squares): no width, NaN — what a refine's or a fit's metadata records for it.
sq_loss_width(::Nothing) = NaN

# The strategy the algorithm states, or (`ClassDefault()`) the problem class's of record.
_resolved_strategy(f, ::ClassDefault) = default_optimization_strategy(f)
_resolved_strategy(_f, strategy) = strategy

_refine_spec(f, alg::LevenbergMarquardt{NativeSolver}; scale, width) = bundle_loss(_resolved_strategy(f, alg.optimization_strategy);
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
    CodimensionRows()
    RowsWithJacobian()

The residual summary a bundle asks for ([`residual_summary`](@ref)), formed at the solution and handed to its
[`block_result`](@ref):

- `CodimensionRows()` (the default): `(; residuals, S, ν, rms, n, weights, loss)` — the rows
  themselves (each block's own rows, in block order), their sum of squares `S`, the residual dof `ν = rows − unknowns`
  (`unknowns` the FREE coordinates: a variable [`fixed_variables`](@ref) holds is no unknown, so with nothing held it is
  every coordinate), per observation (block) its loss weight ρ′/ρ′(0) (all ones under least squares) with their
  support `n` and weighted rms ([`weighted_support_rms`](@ref)), and `loss` the parsed refine ([`bundle_loss`](@ref)'s
  `(; strategy, input, loss)`: its strategy, and through [`scale_record`](@ref) the scale it was given, its source and
  the deployed width);
- `RowsWithJacobian()`: the same, and `jacobian`, the rows' dense Jacobian in the variables' chart coordinates (each
  variable's columns in variable order), every block's rows scaled by √ρ′ under a robust strategy — the weighted
  problem's linearization a covariance reads ([`implicit_covariance`](@ref)).
"""
struct CodimensionRows end
@doc (@doc CodimensionRows) struct RowsWithJacobian end

"""
    solve(RefineProblem(f, u0), LevenbergMarquardt(; optimization_strategy, damping); scale | width, stops = default_stops(f))
        -> FitResult{M, RefineMetadata}

The refine of a chart problem `f` (a VisualGeometryToolkitEstimation family problem: `HomographyProblem(cs)`, …) from
the guess `u0`: ONE chart LM (`_chart_refit`) over every datum — robustified by a `Triggs{K}` strategy at the
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

The one-shot solve runs the cache on the task's kept chart workspace (a repeated solve rebuilds no NLLSsolver problem);
`init(RefineProblem(f, u0), LevenbergMarquardt(…); stops, scale | width, weights, callback, summary)` gives the cache a
workspace of its own (`reinit!(c, u0′; f, stops, weights)` re-points it). `summary = nothing` (the default is
`CodimensionRows()`, the refine above) is a LOCAL OPTIMIZATION's refit: under least squares the chart
[`refit_chart`](@ref)`(f, θ, weights)` (`nothing`: the refit declines, `u = nothing`, `(InitialFailure, NoRule)`) and
ONE chart LM on [`refit_rows`](@ref); under a `Triggs{K}` strategy ONE robustified chart LM on the refine's own
[`model_chart`](@ref) and [`chart_rows`](@ref) (the unit its loss width is in); either at the weights, and the model only — `rms` NaN, `n` 0, no metadata, no residual sweep; a warm
`solve!(reinit!(c, θ; weights))` allocates nothing. Its `θ` is the problem's own model (`convert(model_type(f), θ)`:
a RANSAC hypothesis, in the problem's frame and units), where the summary refine reads a user's guess through
[`guess_model`](@ref) (a physical pose stripped, a labelled one relabelled). The weights' type is the one `init` was given (`nothing`, a
`Vector` or a view), fixed: a caller that means unit weight in a weighted cache gives ones. A RANSAC builds the cache once per solve from the `zero(M)`
placeholder guess. `width` may be a [`DeployedLoss`](@ref) (a local optimization's frozen loss, of the strategy's
shape): it runs as deployed, its squared width unchanged.
"""
__solve(prob::RefineProblem{<:AbstractConstraintProblem, <:Any, NullParameters}, alg::LevenbergMarquardt{NativeSolver};
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f), callback = nothing, weights = nothing,
        summary::Union{Nothing, CodimensionRows} = CodimensionRows()) =
    solve!(_chart_refine_cache(prob, alg, true; scale, width, stops, callback, weights, summary))
# A native refine reads no context: a stray `p` has no method (the pair's MethodError), as before the backend route.
__solve(prob::RefineProblem{<:AbstractConstraintProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    throw(MethodError(__solve, (prob, alg)))
__init(prob::RefineProblem{<:AbstractConstraintProblem, <:Any, NullParameters}, alg::LevenbergMarquardt{NativeSolver};
        kwargs...) = _chart_refine_cache(prob, alg, false; kwargs...)


function _refine_pipeline(f, spec::NamedTuple, θ₀, stops::LmStops, damping::LmDamping, ews, weights, callback)
    L = typeof(spec.strategy)
    M = model_type(f)
    θ = guess_model(f, θ₀)
    θ̂, (code, rule), iters = _refine_run(spec.strategy, f, θ, spec, stops, damping, ews, weights, callback)
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
function _refine_run(::Nothing, f, θ, _spec, stops::LmStops, damping::LmDamping, ews, weights, callback)
    r = _chart_refit(ews, f, model_chart(f, θ), chart_rows, weights; stops, damping, callback)
    return something(r.u, θ), (r.retcode, r.stop_rule), r.stats.nsteps
end
# Triggs: ONE robustified chart LM at the frozen width (`_triggs_refit`, the RANSAC local optimization's own).
function _refine_run(lo::Triggs, f, θ, spec, stops::LmStops, damping::LmDamping, ews, weights, callback)
    r = _triggs_refit(ews, strategy_loss(lo)(), f, θ, spec.loss; weights, stops, damping, callback)
    return r.u, (r.retcode, r.stop_rule), r.stats.nsteps
end

# -----------------------------------------------------------------------------
# The LevenbergMarquardt refine of a bundle — the block contract over `_LmProblem`
# -----------------------------------------------------------------------------

"""
    default_stops(f) -> LmStops

The stop settings a [`LevenbergMarquardt`](@ref) refine of `f` runs at when its solve names none: `LmStops()` (Core's
`[levenberg_marquardt.stops]`) unless `f`'s class has its own configuration of record (VisualGeometryToolkitEstimation's camera-and-poses
bundles: `[CameraPosesProblem refine]`). A contract generic.
"""
default_stops(_f) = LmStops()

"""
    AbstractBlockProblem

The parent of a multi-block refine's `f` (a bundle adjustment, a camera regression: heterogeneous variables and residual blocks, solved
over Core's multi-block LM). Its author implements the block contract — [`block_variables`](@ref),
[`residual_blocks`](@ref), [`fixed_variables`](@ref), [`residual_summary`](@ref) and [`block_result`](@ref): the
problem stated at the guess, the result assembled from the solution — and, for each residual block type,
[`with_robustifier`](@ref). Solved by `solve(RefineProblem(f, u0), LevenbergMarquardt(); scale | width, stops)`, the
one host LM; `ClassDefault()` is `default_optimization_strategy(f)`, the default stops `default_stops(f)`.
"""
abstract type AbstractBlockProblem end

"""
    with_robustifier(block, sqrt_weight, robustifier) -> block

The residual block `block` with its rows scaled by `sqrt_weight` and robustified by `robustifier` (a
[`loss_robustifier`](@ref), or NLLSsolver's `NoRobust()`): what the multi-block LM runs a robust refine on. A
contract generic: a bundle's author adds it for each of its block types.
"""
function with_robustifier end
with_robustifier(block, _sqrt_weight, _robustifier) = _no_method(block, with_robustifier, "a robust bundle refine needs it")

codimension(::AbstractBlockProblem) = 2   # a bundle's residual is a pixel reprojection error unless its author says otherwise

default_algorithm(::RefineProblem{<:AbstractBlockProblem}) = LevenbergMarquardt()

# -----------------------------------------------------------------------------
# The LevenbergMarquardt refine of raw data by a solver backend (PoseLib's)
# -----------------------------------------------------------------------------

"""
    lm_backend_refine(backend, u0, data, p, alg; stops, scale, width) -> FitResult

The Levenberg–Marquardt refine of the guess `u0` on the raw `data` (a correspondence vector) with the context `p`, run
by the solver `backend` (the `backend` of `alg`, a [`LevenbergMarquardt`](@ref)): what
`solve(RefineProblem(data, u0[, p]), LevenbergMarquardt(backend; optimization_strategy); stops, scale | width)` runs.
The guess's type and the data's shape select the backend's residual. A backend contract generic, as the minimal
solvers' kernels and [`essential_8pt`](@ref): a backend's package implements it (PoseLib.jl, for `PoseLibSolver()`);
a backend without one is an `ArgumentError`.
"""
function lm_backend_refine end
lm_backend_refine(backend::AbstractSolverBackend, u0, data, p, alg; kw...) = throw(_no_backend_lm(backend))

"""
    solve(RefineProblem(data, u0[, p]), LevenbergMarquardt(backend; optimization_strategy, damping); stops, scale | width)
        -> FitResult

The refine of the guess `u0` on raw data (a correspondence vector, not a problem), run by a solver backend:
[`lm_backend_refine`](@ref)`(backend, u0, data, p, alg; stops, scale, width)`, which the backend's package implements
(PoseLib.jl: `LevenbergMarquardt(PoseLibSolver())`). `stops` defaults to `LmStops()`; a box (`lb`, `ub`) is refused, as
by every host LM.
"""
function __solve(prob::RefineProblem{<:AbstractVector}, alg::LevenbergMarquardt;
        scale = nothing, width = nothing, stops::LmStops = LmStops())
    _refuse_box(prob, :LevenbergMarquardt)
    check_honored_stops(alg.backend, stops)
    return lm_backend_refine(alg.backend, prob.u0, prob.f, prob.p, alg; stops, scale, width)
end

"""
    solve(RefineProblem(f::AbstractConstraintProblem, u0[, p]), LevenbergMarquardt(backend; optimization_strategy, damping);
          stops, scale | width) -> FitResult

The refine of a family problem `f` by a solver backend other than the native one: [`lm_backend_refine`](@ref)`(backend,
u0, f, p, alg; stops, scale, width)`, the backend's own residual on the problem's data (its package, or the package that
owns both the backend's binding and the problem, implements the method). `stops` must state only fields the backend
honours ([`honored_stops`](@ref)).
"""
function __solve(prob::RefineProblem{<:AbstractConstraintProblem}, alg::LevenbergMarquardt;
        scale = nothing, width = nothing, stops::LmStops = LmStops())
    _refuse_box(prob, :LevenbergMarquardt)
    check_honored_stops(alg.backend, stops)
    return lm_backend_refine(alg.backend, prob.u0, prob.f, prob.p, alg; stops, scale, width)
end

"""
    honored_stops(backend::AbstractSolverBackend) -> Tuple{Vararg{Symbol}}

The [`LmStops`](@ref) fields a Levenberg–Marquardt `backend` honours: every field for the native one (the default); a
backend's package states its library's (PoseLib.jl: `(:max_iter,)`). A contract generic.
[`check_honored_stops`](@ref) refuses stops that set any other field away from its configuration of record.
"""
honored_stops(::AbstractSolverBackend) = fieldnames(LmStops)

"""
    check_honored_stops(backend, stops::LmStops) -> nothing

The one refusal of stop settings a Levenberg–Marquardt `backend` does not honour: an `ArgumentError` naming the backend
and every field of `stops` that differs from `LmStops()` (the configuration of record) outside
[`honored_stops`](@ref)`(backend)`.
"""
function check_honored_stops(backend::AbstractSolverBackend, stops::LmStops)
    honored = honored_stops(backend)
    d = LmStops()
    bad = Symbol[k for k in fieldnames(LmStops) if !(k in honored) && getfield(stops, k) != getfield(d, k)]
    isempty(bad) && return nothing
    throw(ArgumentError(string(nameof(typeof(backend)), " honours only ", join(honored, ", "), " of LmStops; ",
        "the stops also set ", join(bad, ", "))))
end

"""
    BackendRefit(alg, stops::LmStops; cheirality_vote_points = 256)

The external local optimization of a RANSAC by a refine algorithm `alg` (an algorithm whose
[`estimator_kind`](@ref) is [`RefineKind`](@ref): `LevenbergMarquardt(PoseLibSolver(); optimization_strategy)`, or a
package's own refine) at its `stops`: the ONE `AbstractOptimizationStrategy` of the [`ExternalRefit`](@ref) family that
runs a refine algorithm. Its refit of a hypothesis `θ` of `problem` at the absolute loss width `w` is
`solve(RefineProblem(problem, θ), alg; stops, width = w)` ([`external_refit!`](@ref)), its `FitResult` (the external
refit's contract, Core 0.14): its `u` the refined model whatever the record — a refine that spends its budget
(`(MaxIters, IterationCap)` at `stops.max_iter`) is a normal budgeted stop, and the local optimization's own scoring
decides whether the model is kept. An algorithm of another kind is an
`ArgumentError`.

`cheirality_vote_points` is the size of the strided subsample on which a refit that starts from a relative pose
decomposes an essential hypothesis (`cheiral_pose_fast`; PoseLib.jl's essential route), about that many points; its
default, 256, is LocalOptimizationDoneRight's configuration of record (`[optimization_strategy.poselib]`).
"""
struct BackendRefit{A <: AbstractEstimationAlgorithm} <: AbstractOptimizationStrategy
    alg::A
    stops::LmStops
    cheirality_vote_points::Int
    function BackendRefit(alg::A, stops::LmStops; cheirality_vote_points::Integer = 256) where {A <: AbstractEstimationAlgorithm}
        estimator_kind(alg) === RefineKind() || throw(ArgumentError(string("BackendRefit runs a refine algorithm (",
            "estimator_kind RefineKind()); a ", nameof(A), " is ", nameof(typeof(estimator_kind(alg))))))
        cheirality_vote_points >= 1 || throw(ArgumentError("BackendRefit: cheirality_vote_points must be ≥ 1, got $cheirality_vote_points"))
        return new{A}(alg, stops, Int(cheirality_vote_points))
    end
end
RefitFamily(::BackendRefit) = ExternalRefit()
# Its identity (the evaluation's labels and markers, the scale it consumes) is its algorithm's; a Levenberg–Marquardt's
# is its `optimization_strategy`'s, its scale `FixedScale()` (the refit is given its loss width).
for trait in (:optimization_path, :scale_interface, :loss_family, :lo_label, :lo_marker)
    @eval $trait(lo::BackendRefit) = $trait(lo.alg)
end
for trait in (:optimization_path, :loss_family, :lo_label, :lo_marker)
    @eval $trait(lo::BackendRefit{<:LevenbergMarquardt}) = $trait(lo.alg.optimization_strategy)
end
scale_interface(::BackendRefit{<:LevenbergMarquardt}) = FixedScale()
external_refit!(_ws, lo::BackendRefit, problem, θ, width::LossWidth, _scoring) =
    solve(RefineProblem(problem, θ), lo.alg; stops = lo.stops, width = loss_width(width))

public honored_stops, check_honored_stops, BackendRefit

# =============================================================================
# The refine caches — `init`, `solve!` and `reinit!` of the host LevenbergMarquardt
# =============================================================================
#
#     c = init(RefineProblem(f, u0), LevenbergMarquardt(…); stops, scale | width, callback)
#     r = solve!(c)                  # a FitResult; it may alias the cache's buffers until the next solve!
#     reinit!(c, u0′; f, stops, …)   # a new guess, data or run setting, the workspace kept
#
# The cache owns its LM workspace and its kept NLLSsolver problem, so a chunked refinement (a solve!, a step of the
# caller's own between solves, a reinit! from the last iterate) and an LO's per-iteration refit rebuild neither. The
# one-shot `solve` runs the same cache on the task's kept workspace. Three chart forms:
#
# - a constraint problem `f` (an `AbstractConstraintProblem`: the RANSAC families): the chart refine above, its
#   per-datum `weights` a cache setting (`init`/`reinit!` keyword; `nothing`, unit weight, is the default);
# - a row problem `f::AbstractRowProblem{N}`: `f` IS the linearization state (the row protocol, or the two reductions
#   of a device state), solved from `u0::SVector{N}`;
# - a chart problem `f::AbstractChartProblem`: the chart contract with any row state (`chart_rows` may return a
#   reduction-only state), its result assembled by the author's `chart_result`.
# =============================================================================

"""
    AbstractRowProblem{N}

The parent of a linearization state the host [`LevenbergMarquardt`](@ref) solves directly: `f` in
`solve(RefineProblem(f, u0::SVector{N}), LevenbergMarquardt(; damping); stops, callback)`, minimizing its rows from the
chart point `u0` (the zero chart, typically). Its author implements the row protocol — [`residual_rows!`](@ref) and
[`residual_jacobian_rows!`](@ref), or, for a state whose rows live on a device, the two reductions
[`residual_cost!`](@ref) and [`residual_normal_equations!`](@ref) (through [`reduce_cost`](@ref) and
[`reduce_normal_equations`](@ref)) — and [`chart_size`](@ref)`(f) -> (rows, Val(N))`, the row count the workspace holds
(`1` for a reduction-only state: its rows never reach the host). Plain least squares by default
(`default_optimization_strategy(f) = nothing`); a `Triggs{K}` strategy robustifies per datum at
[`codimension`](@ref)`(f)` rows, at the solve's `scale` or `width`.

The result's `u` is the solution chart point (an `SVector{N}`), its record the LM's; `metadata = (; cost, stationary)`
the final cost and `stationary()`, a call that runs the chart LM's stationarity postcondition on the plain rows at `u`
([`is_stationary`](@ref); the chart-problem refine's form); no residual summary (`rms` NaN, `n` 0).
"""
abstract type AbstractRowProblem{N} end

"""
    AbstractChartProblem

The parent of a refine's `f` that implements the chart contract — [`model_chart`](@ref), [`chart_rows`](@ref),
[`chart_model`](@ref), [`chart_size`](@ref) (and [`guess_model`](@ref)/[`solution_model`](@ref) where a guess carries
units or labels) — without being a RANSAC constraint problem: its rows may be any row state, a reduction-only device
state among them. Solved by `solve(RefineProblem(f, u0), LevenbergMarquardt(; damping); stops, callback)`: the chart at
the guess, ONE chart LM from its origin, the model at the solution; the result assembled by its author's
[`chart_result`](@ref). A chart of no coordinates (`chart_size` `(rows, Val(0))`: every parameter held) has nothing
to solve: the model at the guess, `(Success, ClosedForm)`, no step, `cost` the rows' `‖r‖²` there, `stationary()` true.
"""
abstract type AbstractChartProblem end

"""
    chart_result(f::AbstractChartProblem, u0, r::FitResult) -> FitResult

The result of a refine of the chart problem `f` from the guess `u0`, from the chart LM's: `r.u` the refined model (the
caller's form, [`solution_model`](@ref); the guess when the solve went non-finite, `(Failure, NonFinite)`), `r`'s record
and `stats.nsteps` the LM's, and `r.metadata = (; cost, chart, x, rows, stationary)`: the final `‖r‖²`, the chart and
its solution point `x`, the row state the LM descended, and `stationary()`, a call that runs the chart LM's
stationarity postcondition at `x` ([`is_stationary`](@ref)). A contract generic: the default returns `r` with its
metadata `(; cost)`; a family returns its own (a residual summary, its validity gate).
"""
function chart_result end
chart_result(_f, _u0, r::FitResult) =
    FitResult(r.u, r.retcode, r.stop_rule, r.stats, r.rms, r.n, (; cost = r.metadata.cost))

default_optimization_strategy(::AbstractRowProblem) = nothing
default_algorithm(::RefineProblem{<:AbstractRowProblem}) = LevenbergMarquardt()
default_algorithm(::RefineProblem{<:AbstractChartProblem}) = LevenbergMarquardt()

# The LM's robustifier of a parsed refine, at `d` rows per datum: `nothing` for least squares.
_spec_robustifier(spec) = spec.strategy === nothing ? nothing :
    loss_robustifier(strategy_loss(spec.strategy)(), sqrt(sq_loss_width(spec.loss)))


# --- a constraint problem -------------------------------------------------------

mutable struct _ChartRefineCache{F <: AbstractConstraintProblem, U, S, D <: LmDamping, N, W, C, Y <: Union{Nothing, CodimensionRows}}
    f::F
    u0::U
    spec::S
    stops::LmStops
    damping::D
    const ews::_ManifoldLmWorkspace{N}
    weights::W   # the type `init` was given, fixed (a Union field would box an immutable array: a view)
    const callback::C
    const summary::Y
end
function _chart_refine_cache(prob::RefineProblem, alg::LevenbergMarquardt{NativeSolver}, shared::Bool;
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f), weights = nothing, callback = nothing,
        summary::Union{Nothing, CodimensionRows} = CodimensionRows())
    # Core's refines run the native backend (`LevenbergMarquardt{NativeSolver}`); another backend's package solves its
    # own problems, so another backend here has no method.
    _refuse_box(prob, :LevenbergMarquardt)
    f = prob.f
    spec = _refine_spec(f, alg; scale, width)
    rows, dof = chart_size(f)
    ews = shared ? _lm_workspace(nothing, rows, dof) : _ManifoldLmWorkspace(rows, dof)
    return _ChartRefineCache(f, prob.u0, spec, stops, alg.damping, ews, weights, callback, summary)
end
solve!(c::_ChartRefineCache) = _chart_solve!(c.summary, c)
function _chart_solve!(::CodimensionRows, c::_ChartRefineCache)
    _too_few(c.f) && return _refine_too_few(c.f, c.spec, c.u0, data_size(c.f))
    return _refine_pipeline(c.f, c.spec, c.u0, c.stops, c.damping, c.ews, c.weights, c.callback)
end
# The summary-free refine, a local optimization's refit: the problem's refit chart at the weights (none: declined), ONE
# chart LM on its refit rows — robustified at the cache's loss under a `Triggs{K}` strategy — and the model at the
# solution, `nothing` on a non-finite solve (`(Failure, NonFinite)`). No residual sweep: `rms` NaN, `n` 0, no metadata.
function _chart_solve!(::Nothing, c::_ChartRefineCache)
    f = c.f
    M = model_type(f)
    U = Union{Nothing, M}
    θ = convert(M, c.u0)::M   # the problem's own model (a hypothesis of the LO), never a user's guess (`guess_model`)
    k = _spec_robustifier(c.spec)
    if k === nothing
        # the plain refit: the problem's refit chart and rows (a weight-aware seed, rows in the refit's own unit)
        ch = refit_chart(f, θ, c.weights)
        ch === nothing && return FitResult{U}(nothing, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, nothing)
        r = _chart_refit(c.ews, f, ch, refit_rows, c.weights; stops = c.stops, damping = c.damping, callback = c.callback)
    else
        # the robustified refit (Triggs): the refine's own chart and rows, the unit its loss width is in
        r = _chart_refit(c.ews, f, model_chart(f, θ), chart_rows, c.weights; robustifier = k, stops = c.stops,
                         damping = c.damping, callback = c.callback)
    end
    return FitResult{U}(r.u, r.retcode, r.stop_rule, r.stats, NaN, 0, nothing)
end
function reinit!(c::_ChartRefineCache, u0 = c.u0; f = c.f, p = NullParameters(), stops::LmStops = c.stops,
        weights = c.weights, alg = nothing, scale = nothing, width = nothing)
    p isa NullParameters || throw(ArgumentError("reinit!: a LevenbergMarquardt refine reads no context `p`"))
    rows, dof = chart_size(f)
    rows == length(sqrt_weights(c.ews)) || throw(DimensionMismatch(string("reinit!: the cache holds ",
        length(sqrt_weights(c.ews)), " chart-LM rows, the new problem ", rows, " (init a new cache for data of another size)")))
    if alg !== nothing || scale !== nothing || width !== nothing
        a = something(alg, LevenbergMarquardt(c.spec.strategy, c.damping, NativeSolver()))
        c.spec = _refine_spec(f, a; scale, width)
        c.damping = a.damping
    end
    c.f = f; c.u0 = u0; c.stops = stops; c.weights = weights
    return c
end

# --- a row problem --------------------------------------------------------------

mutable struct _RowRefineCache{F, N, S, D <: LmDamping, C}
    f::F
    u0::SVector{N, Float64}
    spec::S
    stops::LmStops
    damping::D
    const ews::_ManifoldLmWorkspace{N}
    const callback::C
end
function _row_spec(f, alg::LevenbergMarquardt{NativeSolver}; scale, width)
    lo = _resolved_strategy(f, alg.optimization_strategy)
    return bundle_loss(lo; scale, width, codimension = lo === nothing ? 1 : codimension(f), caller = "LevenbergMarquardt")
end
function _row_refine_cache(prob::RefineProblem{<:AbstractRowProblem{N}}, alg::LevenbergMarquardt{NativeSolver}, shared::Bool;
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f), callback = nothing) where {N}
    _refuse_box(prob, :LevenbergMarquardt)
    prob.p isa NullParameters || throw(ArgumentError("a LevenbergMarquardt refine of a row problem reads no context `p`"))
    f = prob.f
    spec = _row_spec(f, alg; scale, width)
    rows, dof = chart_size(f)
    dof === Val(N) || throw(ArgumentError(string("a ", nameof(typeof(f)), " <: AbstractRowProblem{", N, "} states a chart of ",
        dof, " in chart_size")))
    ews = shared ? _lm_workspace(nothing, rows, dof) : _ManifoldLmWorkspace(rows, dof)
    return _RowRefineCache(f, SVector{N, Float64}(prob.u0), spec, stops, alg.damping, ews, callback)
end
__solve(prob::RefineProblem{<:AbstractRowProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    solve!(_row_refine_cache(prob, alg, true; kwargs...))
__init(prob::RefineProblem{<:AbstractRowProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    _row_refine_cache(prob, alg, false; kwargs...)
function solve!(c::_RowRefineCache)
    k = _spec_robustifier(c.spec)
    r = k === nothing ? _lmsolve_chart!(c.ews, c.f, c.stops, c.damping; callback = c.callback, u0 = c.u0) :
        _lmsolve_chart!(c.ews, c.f, k, Val(codimension(c.f)), c.stops, c.damping; callback = c.callback, u0 = c.u0)
    ews, f, x, stops = c.ews, c.f, r.u, c.stops
    stationary = () -> is_stationary(ews, f, x, stops)   # the chart-problem cache's form: the postcondition on demand
    return FitResult(r.u, r.retcode, r.stop_rule, r.stats, r.rms, r.n, (; cost = r.metadata.cost, stationary))
end
function reinit!(c::_RowRefineCache{F, N}, u0 = c.u0; f = c.f, p = NullParameters(), stops::LmStops = c.stops,
        alg = nothing, scale = nothing, width = nothing) where {F, N}
    p isa NullParameters || throw(ArgumentError("reinit!: a LevenbergMarquardt refine reads no context `p`"))
    if alg !== nothing || scale !== nothing || width !== nothing
        a = something(alg, LevenbergMarquardt(c.spec.strategy, c.damping, NativeSolver()))
        c.spec = _row_spec(f, a; scale, width)
        c.damping = a.damping
    end
    c.f = f; c.u0 = SVector{N, Float64}(u0); c.stops = stops
    return c
end

# --- a chart problem ------------------------------------------------------------

mutable struct _ChartProblemCache{F, U, D <: LmDamping, N, C}
    f::F
    u0::U
    stops::LmStops
    damping::D
    const ews::_ManifoldLmWorkspace{N}
    const callback::C
end
function _chart_problem_cache(prob::RefineProblem{<:AbstractChartProblem}, alg::LevenbergMarquardt{NativeSolver}, shared::Bool;
        stops::LmStops = default_stops(prob.f), callback = nothing)
    _refuse_box(prob, :LevenbergMarquardt)
    prob.p isa NullParameters || throw(ArgumentError("a LevenbergMarquardt refine of a chart problem reads no context `p`"))
    _resolved_strategy(prob.f, alg.optimization_strategy) === nothing || throw(ArgumentError(string("a ",
        nameof(typeof(prob.f)), " is refined by plain least squares: LevenbergMarquardt(; optimization_strategy = nothing)")))
    rows, dof = chart_size(prob.f, prob.u0)
    ews = shared ? _lm_workspace(nothing, rows, dof) : _ManifoldLmWorkspace(rows, dof)
    return _ChartProblemCache(prob.f, prob.u0, stops, alg.damping, ews, callback)
end
__solve(prob::RefineProblem{<:AbstractChartProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    solve!(_chart_problem_cache(prob, alg, true; kwargs...))
__init(prob::RefineProblem{<:AbstractChartProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    _chart_problem_cache(prob, alg, false; kwargs...)
function solve!(c::_ChartProblemCache)
    f, ews, stops = c.f, c.ews, c.stops
    θ = guess_model(f, c.u0)
    ch = model_chart(f, θ)
    fill!(sqrt_weights(ews), 1.0)
    lm = chart_rows(f, ch, sqrt_weights(ews))
    r = _chart_problem_lm(ews, lm, stops, c.damping, c.callback)
    x = r.u
    finite = !any(isnan, x)
    model = finite ? chart_model(f, recenter(ch, x)) : θ
    code, rule = finite ? (r.retcode, r.stop_rule) : (ReturnCode.Failure, NonFinite)
    stationary = _chart_problem_stationary(ews, lm, x, stops)
    res = FitResult(solution_model(f, model, c.u0), code, rule, r.stats, NaN, 0,
                    (; cost = r.metadata.cost, chart = ch, x, rows = lm, stationary))
    return chart_result(f, c.u0, res)
end
# The chart LM of a chart problem; a ZERO-dimensional chart (every parameter held: `chart_size` `(rows, Val(0))`) has
# nothing to solve: its one point, the empty chart vector, is the solution, `(Success, ClosedForm)`, no step, the cost
# `‖r‖²` there, stationary by construction (NLLSsolver takes no variable of no coordinates).
_chart_problem_lm(ews::_ManifoldLmWorkspace, lm, stops, damping, callback) =
    _lmsolve_chart!(ews, lm, stops, damping; callback)
function _chart_problem_lm(ews::_ManifoldLmWorkspace{0}, lm, _stops, _damping, _callback)
    x = SVector{0, Float64}()
    return FitResult(x, ReturnCode.Success, ClosedForm, SolveStats(; nsteps = 0), NaN, 0,
                     (; cost = 2 * residual_cost!(ews.r, lm, x)))
end
_chart_problem_stationary(ews::_ManifoldLmWorkspace, lm, x, stops) = () -> is_stationary(ews, lm, x, stops)
_chart_problem_stationary(::_ManifoldLmWorkspace{0}, _lm, _x, _stops) = () -> true

function reinit!(c::_ChartProblemCache, u0 = c.u0; f = c.f, p = NullParameters(), stops::LmStops = c.stops, alg = nothing)
    p isa NullParameters || throw(ArgumentError("reinit!: a LevenbergMarquardt refine reads no context `p`"))
    alg === nothing || (c.damping = alg.damping)
    c.f = f; c.u0 = u0; c.stops = stops
    return c
end

# --- a block problem (a bundle) --------------------------------------------------
#
# The block contract: the author states the problem at the guess — its
# variables, its residual blocks, the variables held — and assembles the result from the solved variables and the
# residual summary it asked for; Core builds the multi-block LM once, in the cache, and solves it.

"""
    block_variables(f::AbstractBlockProblem, u0) -> AbstractVector
    residual_blocks(f::AbstractBlockProblem, u0) -> AbstractVector
    fixed_variables(f::AbstractBlockProblem) -> indices | nothing

The block contract of a bundle `f` at the guess `u0`: its NLLSsolver variables (any of them — Core
[`Manifold`](@ref)s, `SVector`s, [`Eliminable`](@ref)s, of one type or several; the Schur system eliminates the
`Eliminable{true}` ones), its residual blocks (NLLSsolver `AbstractResidual`s, one per observation, each answering
[`with_robustifier`](@ref); their row counts may differ), and the indices of the variables the solve holds (a gauge,
an unconstrained track: NLLSsolver's unfixed mask), `nothing` (the default) for none. Contract generics.
"""
function block_variables end
@doc (@doc block_variables) function residual_blocks end
@doc (@doc block_variables) fixed_variables(_f) = nothing
# A bundle that states no block contract: the one refusal.
block_variables(f::AbstractBlockProblem, _u0) = _no_method(f, block_variables, _BLOCK_CONTRACT)
residual_blocks(f::AbstractBlockProblem, _u0) = _no_method(f, residual_blocks, _BLOCK_CONTRACT)
const _BLOCK_CONTRACT = "a LevenbergMarquardt refine of a bundle needs the block contract (block_variables, residual_blocks, block_result)"

"""
    block_structure(f::AbstractBlockProblem, u0) -> (vars, blocks, stated)

The bundle `f` stated at the guess `u0` ONCE: its variables and residual blocks (as [`block_variables`](@ref) and
[`residual_blocks`](@ref) give them) and `stated`, what its author keeps of the statement for the result: Core reads
it once per solve (the one-shot `solve`, `init`, `reinit!`) and hands `stated` to [`block_result`](@ref) as the
summary's field `stated` (the summary `(; stated)` when the bundle asks for none), so the result is assembled from it
and the bundle is never stated twice. The default is `(block_variables(f, u0), residual_blocks(f, u0), nothing)`, and
a `nothing` adds no field (and a `reinit!` that keeps the blocks then reads `block_variables` alone). A bundle whose statement is one pipeline (its blocks, variables and the assembly of its
result from one reading of its data) implements this method in place of the two; `stated` is then a concrete value
(the refine stays inferred). A contract generic.
"""
block_structure(f::AbstractBlockProblem, u0) = (block_variables(f, u0), residual_blocks(f, u0), nothing)


"""
    residual_summary(f::AbstractBlockProblem) -> CodimensionRows() | RowsWithJacobian() | nothing

The residual summary a refine of the bundle `f` forms at the solution ([`CodimensionRows`](@ref), the default), or
`nothing`: no sweep over the rows at all (a bundle whose rows live on a device and whose result reads none). A contract
generic.
"""
residual_summary(_f) = CodimensionRows()

"""
    block_result(f::AbstractBlockProblem, u0, r::FitResult, summary) -> FitResult

The result of a refine of the bundle `f` from `u0`: `r.u` the solved variables (the cache's problem vector, valid until
the next `solve!`: copy or convert what the result keeps), `r`'s termination record and `stats.nsteps` the LM's (no
solve at `stops.max_iter == 0`: `(MaxIters, IterationCap)`, `nsteps` 0, `r.u` the guess's variables), `summary` the
[`residual_summary`](@ref) `f` asked for (or `nothing`), with the field `stated` when [`block_structure`](@ref) kept
one. The author's own assembly; a contract generic.
"""
function block_result end
block_result(f::AbstractBlockProblem, _u0, _r, _summary) = _no_method(f, block_result, _BLOCK_CONTRACT)

# THE too-few-data guard of every refine: fewer data than a minimal sample (a bundle states none by default: 0).
_too_few(f::Union{AbstractConstraintProblem, AbstractBlockProblem}) = (m = minimal_sample_size(f); m > 0 && data_size(f) < m)
minimal_sample_size(::AbstractBlockProblem) = 0
data_size(f::AbstractBlockProblem) = _no_method(f, data_size, "a bundle that states a minimal_sample_size states its data_size")

mutable struct _BlockRefineCache{F, U, S, P, B, C, D <: LmDamping, T}
    f::F
    u0::U
    loss::S
    stops::LmStops
    const damping::D
    lp::P
    costs::B          # the blocks without their robustifiers: the rows the summary reads
    const callback::C
    stated::T         # `block_structure`'s `stated` at `u0`, the result's (`nothing`: none kept)
end

# The multi-block LM of `f` from its guess's variables `vars` and blocks `blocks`: the blocks robustified at the parsed
# loss (none under least squares), its held variables fixed (NLLSsolver's unfixed mask).
function _block_lm_problem(f::AbstractBlockProblem, vars, blocks, loss, damping::LmDamping)
    costs = convert(Vector{_element_union(blocks)}, blocks)
    rob = loss.strategy === nothing ? costs : [_robustified_block(c, loss) for c in costs]
    return _LmProblem(copy(vars), rob; fixed = fixed_variables(f), damping), costs
end
# Block `c` robustified at the parsed robust `loss` (its frozen loss_robustifier at unit √weight).
_robustified_block(c, loss) =
    with_robustifier(c, 1.0, loss_robustifier(strategy_loss(loss.strategy)(), sqrt(sq_loss_width(loss.loss))))

# The task's kept multi-block LM of the ONE-SHOT refine (`solve`), under the one reuse policy (`task_buffers`): one per
# task and kind — the bundle's type, its variables' and blocks' element types, the parsed loss's and the damping's
# types — whatever the sizes. A call of the kept structure (the same variable types and sizes in order, the same held
# variables, the same block types, `varindices` and row counts in order: what NLLSsolver's kept `OptimizeCache` and
# variable–cost map are built for) re-seeds it in place: the guess's variables into the problem, the call's blocks
# over the kept ones (NLLSsolver's costs and the summary's), rebuilding neither the problem nor its linear system.
# Another structure releases the old problem and builds the new one in its place. It holds the last call's blocks (and
# what they read) until the next call of its kind on the task, never the bundle itself or its guess. A one-shot refine
# started from inside a running one of its kind on the task (a callback) builds a problem of its own (`busy`).
mutable struct _KeptBlockLm{P, C}
    lp::Union{Nothing, P}         # the kept `_LmProblem`, or `nothing`
    costs::Union{Nothing, C}      # its blocks without their robustifiers (the summary's rows)
    busy::Bool
end
struct _KeptBlockLmSlot{F, V, B, S, D} end   # the kind: the bundle's, variables', blocks', loss's and damping's types
# The task's kept problem of the kind of (`f`, `vars`, `blocks`, `loss`, `damping`), typed by the problem
# `_block_lm_problem` builds for it (inferred: the solve on it stays type-stable).
function _kept_block_lm(f, vars, blocks, loss, damping)
    H = _kept_block_lm_type(Core.Compiler.return_type(_block_lm_problem,
        Tuple{typeof(f), typeof(vars), typeof(blocks), typeof(loss), typeof(damping)}))
    slot = _KeptBlockLmSlot{typeof(f), typeof(vars), typeof(blocks), typeof(loss), typeof(damping)}()
    return task_buffers(() -> H(nothing, nothing, false), slot)::H
end
_kept_block_lm_type(::Type{Tuple{P, C}}) where {P, C} = _KeptBlockLm{P, C}
_kept_block_lm_type(::Type) = _KeptBlockLm{Any, Any}

"""
    task_kept_blocks(F::Type) -> Vector{AbstractVector}

The residual blocks (without their robustifiers) of each multi-block problem the current task keeps for one-shot
refines of bundles of type `F` (`<: F`): one vector per kept problem, none before a refine of the kind has run. A
query of the reuse policy of `solve(RefineProblem(f::AbstractBlockProblem, u0), LevenbergMarquardt())` — one kept
problem per task and kind, whatever the sizes, holding the last call's blocks, never the bundle or its guess — for a
test to read; nothing solves through it.
"""
function task_kept_blocks(::Type{F}) where {F}
    out = AbstractVector[]
    for (k, v) in task_local_storage()
        k isa _KeptBlockLmSlot && _slot_bundle_type(k) <: F && v.costs !== nothing && push!(out, v.costs)
    end
    return out
end
_slot_bundle_type(::_KeptBlockLmSlot{F}) where {F} = F

# Re-seed the kept problem `lp` (and its unrobustified `costs`) with the call's `vars` and `blocks` when they have its
# structure (see `_KeptBlockLm`): true, the problem re-seeded; false, nothing changed. Every check precedes the first
# write; O(N), allocation-free.
function _reseed_block_lm!(lp::_LmProblem, costs::AbstractVector, f, vars, blocks, loss)
    v = lp.variables
    (length(v) == length(vars) && length(costs) == length(blocks)) || return false
    @inbounds for i in eachindex(v, vars)
        a = v[i]; b = vars[i]
        (typeof(a) === typeof(b) && NLLSsolver.dynamic(NLLSsolver.nvars(a)) == NLLSsolver.dynamic(NLLSsolver.nvars(b))) ||
            return false
    end
    _same_fixed(lp.free, lp.unfixed, fixed_variables(f)) || return false
    rob = lp.blocks
    R = eltype(rob)
    @inbounds for k in eachindex(costs, blocks)
        _same_block_structure(costs, k, blocks[k], loss, R) || return false
    end
    copyto!(v, vars)
    copyto!(costs, blocks)
    if rob !== costs
        if loss.strategy === nothing
            copyto!(rob, costs)
        else
            @inbounds for k in eachindex(rob, costs)
                rob[k] = _robustified_block(costs[k], loss)
            end
        end
    end
    foreach(r -> _refill_costs!(r, rob), values(lp.problem.costs))   # NLLSsolver's per-type cost vectors, in order
    return true
end
# The call's block `b` has the structure of the kept block `costs[k]` (its type, `varindices` and row count) and the
# kept problem's block type `R` (plain, or robustified at this `loss`): one call per block behind a function barrier,
# the kept block read inside it, so a bundle whose blocks are an abstract vector (blocks of several types) dispatches
# once per block on its own value and gets a Bool back, nothing boxed.
function _same_block_structure(costs, k, b, loss, ::Type{R}) where {R}
    a = @inbounds costs[k]
    return typeof(a) === typeof(b) && NLLSsolver.varindices(a) == NLLSsolver.varindices(b) &&
        NLLSsolver.dynamic(NLLSsolver.nres(a)) == NLLSsolver.dynamic(NLLSsolver.nres(b)) &&
        (loss.strategy === nothing ? b isa R : _robustified_block(b, loss) isa R)
end
# The held variables `fixed` (`fixed_variables`) are the kept problem's (its free mask; `unfixed === nothing`: none).
_same_fixed(::BitVector, unfixed, ::Nothing) = unfixed === nothing
function _same_fixed(free::BitVector, unfixed, fixed)
    unfixed === nothing && return false
    n = 0
    for i in fixed
        (i isa Integer && 1 <= i <= length(free) && !free[i]) || return false
        n += 1
    end
    return n == count(!, free)
end
# NLLSsolver's cost vector of block type `T`: the problem's blocks of that type, in the order they were added.
function _refill_costs!(r::Vector{T}, blocks) where {T}
    j = 0
    @inbounds for b in blocks
        if b isa T
            j += 1
            r[j] = b
        end
    end
    return r
end

function _block_refine_cache(prob::RefineProblem{<:AbstractBlockProblem}, alg::LevenbergMarquardt{NativeSolver};
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f), callback = nothing)
    f = prob.f
    loss = _block_refine_loss(prob, alg; scale, width)
    _too_few(f) && return _BlockRefineCache(f, prob.u0, loss, stops, alg.damping, nothing, nothing, callback, nothing)
    vars, blocks, stated = block_structure(f, prob.u0)
    lp, costs = _block_lm_problem(f, vars, blocks, loss, alg.damping)
    return _BlockRefineCache(f, prob.u0, loss, stops, alg.damping, lp, costs, callback, stated)
end
# The refine's checks (no box, no context) and its parsed loss: the strategy and its scale input (`bundle_loss`).
function _block_refine_loss(prob::RefineProblem{<:AbstractBlockProblem}, alg::LevenbergMarquardt; scale, width)
    _refuse_box(prob, :LevenbergMarquardt)
    prob.p isa NullParameters || throw(ArgumentError("a LevenbergMarquardt refine of a bundle reads no context `p`"))
    f = prob.f
    return bundle_loss(_resolved_strategy(f, alg.optimization_strategy); scale, width, codimension = codimension(f),
                       caller = string(nameof(typeof(f))))
end
# Too few data for a minimal sample (no LM is built): no estimate, as a chart refine's.
_too_few_block_result(f) = FitResult(nothing, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, data_size(f), nothing)

function solve!(c::_BlockRefineCache)
    lp = c.lp
    # too few data for a minimal sample (the cache built no LM): no estimate, as a chart refine's
    lp === nothing && return _too_few_block_result(c.f)
    return _block_solve(c.f, c.u0, c.loss, c.stops, lp, c.costs, c.callback, c.stated)
end
# THE block solve, the cache's and the one-shot refine's: the multi-block LM `lp` (its unrobustified blocks `costs`) at
# `stops`, the summary `f` asks for with `block_structure`'s `stated`, the author's result.
function _block_solve(f, u0, loss, stops::LmStops, lp::_LmProblem, costs, callback, stated)
    r = stops.max_iter == 0 ?
        FitResult(lp.variables, ReturnCode.MaxIters, IterationCap, SolveStats(), NaN, 0, (; cost = NaN)) :
        _lmsolve_chart!(lp, stops, _nlls_callback(callback))
    return block_result(f, u0, r, _with_stated(_residual_summary(residual_summary(f), loss, costs, lp), stated))
end
# The summary with the bundle's `stated` (`block_structure`): no field for `nothing`.
_with_stated(summary, ::Nothing) = summary
_with_stated(summary, stated) = _summary_with(summary, stated)
_summary_with(summary::NamedTuple, stated) = (; summary..., stated)
_summary_with(::Nothing, stated) = (; stated)

_residual_summary(::Nothing, _loss, _costs, _lp) = nothing
_residual_summary(::CodimensionRows, loss, costs, lp) = _block_summary(loss, costs, lp)
function _residual_summary(::RowsWithJacobian, loss, costs, lp)
    s = _block_summary(loss, costs, lp)
    lo = loss.strategy
    blocks = costs
    stationarity = lo === nothing ? blocks :
        [with_robustifier(b, sqrt(s.weights[k]), NLLSsolver.NoRobust()) for (k, b) in enumerate(blocks)]
    jacobian = Matrix(last(_nlls_rows_and_jacobian(stationarity, collect(lp.variables))))
    return (; s..., jacobian)
end
function _block_summary(loss, costs, lp)
    n = length(costs)
    variables = lp.variables
    r = nlls_rows(costs, variables)
    sq = Vector{Float64}(undef, n)
    o = 0
    for k in 1:n
        d = Int(NLLSsolver.dynamic(NLLSsolver.nres(costs[k])))
        sq[k] = _block_sq(r, o, d)
        o += d
    end
    lo = loss.strategy
    weights = lo === nothing ? unit_weights(Float64, n) : loss_weights(strategy_loss(lo)(), sq_loss_width(loss.loss), sq)
    support, rms = weighted_support_rms(weights, sq)
    unknowns = lp.offsets[end]   # the FREE coordinates: a held variable (fixed_variables) is no unknown
    return (; residuals = r, S = sum(abs2, r), ν = length(r) - unknowns, rms, n = support, weights, loss)
end
# Observation `k`'s squared norm: its `d` rows from offset `o`, their squares summed in order (THE one sum).
@inline function _block_sq(r, o, d)
    s = r[o + 1]^2
    for j in 2:d
        s += r[o + j]^2
    end
    return s
end

"""
    solve(RefineProblem(f::AbstractBlockProblem, u0), LevenbergMarquardt(; optimization_strategy, damping);
          scale | width, stops = default_stops(f)) -> FitResult

The refine of a bundle: the strategy and its scale input parsed by the refine contract ([`bundle_loss`](@ref): a
`Triggs{K}` at `scale = σ` or `width = w`, or `nothing`, plain least squares), then the block contract
([`block_variables`](@ref), [`residual_blocks`](@ref), [`fixed_variables`](@ref), [`residual_summary`](@ref),
[`block_result`](@ref)) solved by Core's multi-block LM, built once in the cache (`init`; `reinit!(c, u0′)` keeps it).
The one-shot `solve` runs the same solve on the task's kept problem of the bundle's kind: a later call of the same
structure (the variables' types and sizes, the held variables, the blocks' types, `varindices` and row counts, in
order) re-seeds it in place, so a repeated refine rebuilds neither the NLLSsolver problem nor its linear system;
another structure replaces it (one kept problem per task and kind, whatever the sizes). `r.u` in `block_result` is
then that problem's vector, valid until the next refine of the kind on the task. A bundle with fewer data than its `minimal_sample_size` (0 by default: no guard) is not solved: the result is
`u = nothing`, `(InitialFailure, NoRule)`, `n` its `data_size`, before any block is built. A box is refused (the host
LM is unbounded).
"""
function __solve(prob::RefineProblem{<:AbstractBlockProblem}, alg::LevenbergMarquardt{NativeSolver};
        scale = nothing, width = nothing, stops::LmStops = default_stops(prob.f), callback = nothing)
    f = prob.f
    loss = _block_refine_loss(prob, alg; scale, width)
    _too_few(f) && return _too_few_block_result(f)
    vars, blocks, stated = block_structure(f, prob.u0)
    kept = _kept_block_lm(f, vars, blocks, loss, alg.damping)
    kept.busy && return _block_solve(f, prob.u0, loss, stops, _block_lm_problem(f, vars, blocks, loss, alg.damping)..., callback, stated)
    kept.busy = true
    try
        lp, costs = kept.lp, kept.costs
        if lp === nothing || costs === nothing || !_reseed_block_lm!(lp, costs, f, vars, blocks, loss)
            kept.lp = kept.costs = nothing          # the old structure released before the new is built
            nlp, ncosts = _block_lm_problem(f, vars, blocks, loss, alg.damping)
            kept.lp = nlp; kept.costs = ncosts
            return _block_solve(f, prob.u0, loss, stops, nlp, ncosts, callback, stated)
        end
        return _block_solve(f, prob.u0, loss, stops, lp, costs, callback, stated)
    finally
        kept.busy = false
    end
end
__init(prob::RefineProblem{<:AbstractBlockProblem}, alg::LevenbergMarquardt{NativeSolver}; kwargs...) =
    _block_refine_cache(prob, alg; kwargs...)

function reinit!(c::_BlockRefineCache, u0 = c.u0; f = c.f, p = NullParameters(), stops::LmStops = c.stops,
        alg = nothing, scale = nothing, width = nothing)
    p isa NullParameters || throw(ArgumentError("reinit!: a LevenbergMarquardt refine of a bundle reads no context `p`"))
    rebuild = f !== c.f || alg !== nothing || scale !== nothing || width !== nothing
    if alg !== nothing || scale !== nothing || width !== nothing
        a = something(alg, LevenbergMarquardt(c.loss.strategy, c.damping, NativeSolver()))
        c.loss = bundle_loss(_resolved_strategy(f, a.optimization_strategy); scale, width, codimension = codimension(f),
                             caller = string(nameof(typeof(f))))
    end
    c.f = f; c.u0 = u0; c.stops = stops
    c.lp === nothing && throw(ArgumentError("reinit!: the cache was built on too few data (no LM): init a new one"))
    if rebuild
        vars, blocks, c.stated = block_structure(f, u0)
        c.lp, c.costs = _block_lm_problem(f, vars, blocks, c.loss, c.damping)
    else
        # the same structure: the guess's variables into the kept problem, its blocks and NLLSsolver cache kept
        vars = _reinit_variables!(c, f, u0)
        length(vars) == length(c.lp.variables) || throw(DimensionMismatch(
            "reinit!: the guess has $(length(vars)) variables, the cache's bundle $(length(c.lp.variables))"))
        copyto!(c.lp.variables, vars)
    end
    return c
end
# The guess's variables for a `reinit!` that keeps the blocks: a bundle that keeps no `stated` (`block_structure`'s
# default) answers `block_variables` alone; one that keeps one is stated once, its `stated` the result's.
_reinit_variables!(c::_BlockRefineCache, f, u0) = _reinit_variables!(c, c.stated, f, u0)
_reinit_variables!(::_BlockRefineCache, ::Nothing, f, u0) = block_variables(f, u0)
function _reinit_variables!(c::_BlockRefineCache, _stated, f, u0)
    vars, _, c.stated = block_structure(f, u0)
    return vars
end

public block_structure, block_variables, residual_blocks, fixed_variables, residual_summary, block_result, CodimensionRows, RowsWithJacobian
public task_kept_blocks

public AbstractRowProblem, AbstractChartProblem, chart_result, LmIterationState
