# =============================================================================
# The FIXED-SCALE reweighted fits — `ReweightedLeastSquares` on a linear model
# (codimension 1), a rotation between direction pairs (2), a rigid alignment (3)
# and a robot-world hand-eye calibration (6, the whitened pose residual), and
# `RotationAveraging` (3). The robust-estimation contract (§1, §2, §12, §13):
#
#   * A fit's objective is its own (a linear least squares, a Kabsch, …). Its
#     only robustness is `optimization_strategy = Irls{K}(…)`, an M-estimator
#     weighting at ONE width, set once from the scale input and never
#     re-estimated; `optimization_strategy = nothing` is the plain least
#     squares. A refine's Triggs is refused (a fit has no guess to polish).
#   * The scale input — the solve keywords — is EXACTLY ONE of `scale = σ`,
#     `scale = <an upstream robust result>` (its `noise_scale`) or `width = w`
#     (Core's `scale_input`); anything else is an ArgumentError. A
#     least-squares fit takes neither and refuses one it would ignore.
#   * The width is `w`, or `c · σ` with `c` the strategy's efficiency
#     constant, by default `efficiency_constant(K, codimension)` (Core's
#     solved table: Huber 1.345 at d = 1, 1.5011 at d = 2, 1.628 at d = 3, …).
#     A shape with no efficiency constant (TLS) needs the strategy's constant
#     or a `width`.
#   * The result records `scale` and `scale_source` (ScaleGiven, ScaleUpstream,
#     or ScaleNone for a width-only or least-squares fit, scale NaN) and the
#     loss width it ran at (`loss_width`, NaN for least squares); its per-datum `weights` are the
#     loss weights ρ′(r²) normalized to w_max = 1 (`IrlsWeights`), or a lazy
#     all-ones `Fill` (`BinaryWeights`) for least squares. A caller's prior
#     weights (bootstrap counts, edge weights) multiply the loss weights in
#     the solve; they are the caller's, not repeated in the metadata.
# =============================================================================

"""
    ReweightedLeastSquares(; optimization_strategy = ClassDefault(), tol = ClassDefault())

A [`FitKind`](@extref VisualGeometryToolkitCore.FitKind): the fixed-scale reweighted least squares of a
problem's own one-step solve, seeded by that solve. It iterates the residuals, the loss weights of
`optimization_strategy` (an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}`: its loss, its efficiency
constant and its `max_iter`; `nothing`: the plain least squares, one solve) at ONE loss width, and the weighted
solve, until an iteration moves the estimate by less than `tol` or the strategy's cap. The width comes from the
solve's scale input, `scale = σ` (or an upstream result) or `width = w`, and is never re-estimated.
`ClassDefault()` is the problem class's configuration of record (configs/default.toml
`[reweighted_least_squares.<class>]`; the rigid alignment's and the robot-world calibration's strategy is
`nothing`, the latter's `tol` its `[robot_world_hand_eye]` table's).

| problem | the one-step solve | residual (codimension) |
|---|---|---|
| `EstimationProblem(LinearModel, (A, y[, prior]))` | the (prior-)weighted least squares (column-pivoted QR) | `yᵢ − aᵢᵀx` (1) |
| `EstimationProblem(RotMatrix{3}, cs)` (direction pairs) | the weighted Kabsch/Wahba | `∠(R·aᵢ, bᵢ)` (2) |
| `EstimationProblem(EuclideanMap{3}, cs)` (point pairs) | the weighted Kabsch of the centred points | `‖bᵢ − T(aᵢ)‖` (3) |
| `EstimationProblem(RobotWorldHandEyeCalibration, markers, p)` | the staged linear solve | the whitened pose residual (6) |

The result is a `FitResult` with [`FitMetadata`](@ref) (the robot-world calibration's:
[`RobotWorldHandEyeMetadata`](@ref)); its termination record is [`irls_termination`](@ref)'s,
`stats.nsteps` the reweighted iterations, `n` the support (data of positive solve weight) and `rms` the
weight-weighted root mean square residual.
"""
struct ReweightedLeastSquares{S, T} <: AbstractEstimationAlgorithm
    optimization_strategy::S
    tol::T
end
ReweightedLeastSquares(; optimization_strategy = ClassDefault(), tol = ClassDefault()) =
    ReweightedLeastSquares(optimization_strategy, tol)
estimator_kind(::ReweightedLeastSquares) = FitKind()

# The algorithm's settings on a problem class: the caller's, or (`ClassDefault()`) the class's of record — its
# configs/default.toml table `section` (`strategy = false`: the class's default strategy is `nothing`).
function _reweighted_settings(alg::ReweightedLeastSquares, section::AbstractString, who::AbstractString; strategy::Bool = true)
    lo = alg.optimization_strategy isa ClassDefault ?
        (strategy ? _check_irls_strategy(who, _dflt_strategy(section, Symbol(section))) : nothing) :
        alg.optimization_strategy
    tol = alg.tol isa ClassDefault ? _dflt(section, "tol")::Float64 : alg.tol
    lo isa Union{Nothing, AbstractOptimizationStrategy} || throw(ArgumentError(
        "$who: `optimization_strategy` must be an Irls{K} strategy or nothing, got $(typeof(lo))"))
    return lo, tol
end

"""
    FitIrls{S <: VgtkCore.AbstractLossShape}

The resolved robust weighting of one fit call (internal): the deployed loss
`shape`, its squared loss width, and the scale record. Built once by
`_fit_irls` from the algorithm's `optimization_strategy` and the solve's `scale` and `width` keywords.
"""
struct FitIrls{S <: VgtkCore.AbstractLossShape}
    shape::S
    sq_loss_width::Float64
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end

_check_irls_strategy(_fit, ::Nothing) = nothing
function _check_irls_strategy(fit::AbstractString, lo::AbstractOptimizationStrategy)
    lo isa Triggs && throw(ArgumentError(
        "$fit: optimization_strategy = $(VgtkCore.lo_label(lo)) is a robustified LM, which needs a starting guess: " *
        "a fit takes `optimization_strategy = Irls{K}(…)` or `nothing` (a LevenbergMarquardt refine takes Triggs: refine the fit's result instead)"))
    lo isa Irls || throw(ArgumentError(
        "$fit: `optimization_strategy` must be an Irls{K} strategy (a fixed-scale M-estimator weighting) or nothing, got $(typeof(lo))"))
    lo isa Irls{GaussianShape} && throw(ArgumentError(
        "$fit: Irls{GaussianShape} is the plain least squares; give `optimization_strategy = nothing`"))
    lo_width(lo) isa LossWidth && throw(ArgumentError(
        "$fit: an absolute width is the fit's own `width = w` keyword, not the strategy's loss_width"))
    return lo
end

"""
    _fit_irls(fit, irls, scale, width, codim) -> FitIrls | nothing

The fit's robust weighting at its scale input: `nothing` for least squares
(`optimization_strategy = nothing`; a `scale` or `width` it would ignore is refused), otherwise
the strategy's loss at the width `w` (`width = w`, recording `ScaleNone`) or
`c · σ` (`scale = σ` or an upstream result, Core's `scale_input`; `c` the
strategy's efficiency constant or `efficiency_constant(K, codim)`).
"""
function _fit_irls(fit::AbstractString, ::Nothing, scale, width, _codim::Integer)
    (scale === nothing && width === nothing) || throw(ArgumentError(
        "$fit: the least-squares fit (optimization_strategy = nothing) takes no `scale` or `width`; " *
        "give an Irls{K} strategy to use one"))
    return nothing
end
function _fit_irls(fit::AbstractString, strategy::AbstractOptimizationStrategy, scale, width, codim::Integer)
    _check_irls_strategy(fit, strategy)
    input = try
        VgtkCore.scale_input(scale, width)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$fit: $(e.msg) (or `optimization_strategy = nothing` for the plain least squares)"))
    end
    return _fit_irls(fit, strategy, input, codim)
end
# ... at a parsed scale input (a fit whose data carry their scale supplies its own).
function _fit_irls(fit::AbstractString, strategy::AbstractOptimizationStrategy, input::VgtkCore.ScaleInput, codim::Integer)
    lo = _check_irls_strategy(fit, strategy)
    K = strategy_loss(lo)
    shape = K()
    loss_width = isfinite(input.width) ? Float64(input.width) :
        Float64(VgtkCore.efficiency_constant(lo, codim)) * Float64(input.scale)   # the strategy's constant, else Core's table
    sq = loss_width^2
    return FitIrls(shape, sq, Float64(input.scale), VgtkCore.scale_source(input), loss_width)
end

# The fit's per-datum loss weights at the residual norms `norms` into `w`
# (least squares: nothing to do).
_irls_weights!(w, ::Nothing, _norms) = w
function _irls_weights!(w, f::FitIrls, norms)
    @inbounds for i in eachindex(w, norms)
        w[i] = normalized_loss_weight(f.shape, Float64(norms[i])^2, f.sq_loss_width)
    end
    return w
end

# The solve weights: the prior times the loss weight.
_solve_weights(prior, w) = prior .* w

# The metadata's weights of a fit of `n` data: least squares, a lazy all-ones
# Fill (BinaryWeights); IRLS, the loss weights.
_fit_weights_storage(::Nothing, n::Integer) = VgtkCore.unit_weights(n)
_fit_weights_storage(::FitIrls, n::Integer) = ones(n)

# The scale record of the result: (scale, scale_source, loss_width).
_fit_scale_record(::Nothing) = (NaN, VgtkCore.ScaleNone, NaN)
_fit_scale_record(f::FitIrls) = (f.scale, f.scale_source, f.loss_width)

"""
    FitMetadata{L, W}

`FitResult` metadata of [`ReweightedLeastSquares`](@ref) on a linear model, a rotation and a rigid alignment,
all at the returned estimate (`L` the `optimization_strategy`'s type, `Nothing` for least squares: the producer
of the weights, `AbstractFitMetadata{L}`):

- `weights::W` — the loss weight `ρ′(r²)` of every datum, normalized to a
  maximum of 1 (`IrlsWeights`); a lazy all-ones vector for least squares
  (`BinaryWeights`). The solve's weight of a datum is its prior times this.
- `residuals` — every datum's residual at the estimate, in the fit's unit:
  `y − A·x` (a linear model: signed), the angle `∠(R·aᵢ, bᵢ)` in radians
  (a rotation), the distance `‖bᵢ − (R·aᵢ + t)‖` in the points' unit
  (a rigid alignment). They stay in the metadata, not the result's `resid`: a
  linear model's are signed residuals, not distances.
- `scale`, `scale_source` — the scale record (`noise_scale(result)`,
  `scale_source(result)`), in the residuals' unit: the scale given
  (`ScaleGiven`, `ScaleUpstream`), or `NaN` and `ScaleNone` for least squares
  or a `width`-only fit.
- `loss_width` — the loss width the weights were computed at (`NaN` for
  least squares).
"""
struct FitMetadata{L, W <: AbstractVector{Float64}} <: AbstractFitMetadata{L}
    weights::W
    residuals::Vector{Float64}
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end
FitMetadata{L}(w::W, r, s, src, lw) where {L, W} = FitMetadata{L, W}(w, r, s, src, lw)

# The IRLS strategy's iteration cap (least squares: one solve).
_fit_max_iter(::Nothing) = 0
_fit_max_iter(lo::Irls) = lo.max_iter

# The failed solve of a reweighted fit (one owner for every problem `ReweightedLeastSquares` solves): the problem's
# null estimate `u`, the record — before the loop a degenerate start `(InitialFailure, NoRule)`, inside it a weighted
# solve that could not be solved `(InternalLinearSolveFailed, NoRule)` — no rms, no support, and the n weights `w`
# with NaN residuals.
_reweighted_failure(u, strategy, f, n, w, record::Tuple{ReturnCode.T, StopRule} = (ReturnCode.InitialFailure, NoRule)) =
    FitResult(u, record..., SolveStats(), NaN, 0, FitMetadata{typeof(strategy)}(w, fill(NaN, n), _fit_scale_record(f)...))

"""
    irls_termination(converged, closed_form; linear_solve_failed = false) -> (ReturnCode.T, StopRule)

THE termination record of the reweighted fits' loop ([`ReweightedLeastSquares`](@ref) on every problem it solves,
[`RotationAveraging`](@ref)): its increment tolerance met, `(Success, IncrementSize)`;
otherwise a least-squares run (`closed_form`) is a closed form, `(Success, ClosedForm)`, and a robust one reached its
cap, `(MaxIters, IterationCap)`; a weighted solve that could not be solved inside the loop (too few rows of positive
weight left) is SciMLBase's `(InternalLinearSolveFailed, NoRule)`.

Public (as Core's `lmsolve_chart!` records an LM's) so that a downstream reweighted loop records its stops in the
same words: BlobBoards' per-blob IRLS (its relative parameter step below `rtol` is `IncrementSize`; a singular LU is
`linear_solve_failed`). A stop the loop adds of its own (BlobBoards' shape gate, `ShapeGate`) stays its own.
"""
function irls_termination(converged::Bool, closed_form::Bool; linear_solve_failed::Bool = false)
    linear_solve_failed && return (ReturnCode.InternalLinearSolveFailed, NoRule)
    converged && return (ReturnCode.Success, IncrementSize)
    return closed_form ? (ReturnCode.Success, ClosedForm) : (ReturnCode.MaxIters, IterationCap)
end
