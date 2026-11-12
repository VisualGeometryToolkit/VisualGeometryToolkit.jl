# =============================================================================
# The FIXED-SCALE IRLS of every reweighted fit — fit_linear (codimension 1),
# fit_rotation (2), fit_rotation_averaging (3), fit_rigid_alignment (3) and
# fit_robot_world_hand_eye (6, the whitened pose residual). The robust-
# estimation contract (§1, §2, §12, §13):
#
#   * A fit's objective is its own (a linear least squares, a Kabsch, …). Its
#     only robustness is `optimization_strategy = Irls{K}(…)`, an M-estimator
#     weighting at ONE width, set once from the scale input and never
#     re-estimated; `optimization_strategy = nothing` is the plain least
#     squares. A refine's Triggs is refused (a fit has no guess to polish).
#   * The scale input is EXACTLY ONE of `scale = σ`, `scale = <an upstream
#     robust result>` (its `noise_scale`) or `width = w` (Core's
#     `scale_input`); anything else is an ArgumentError. A least-squares fit
#     takes neither and refuses one it would ignore.
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
    FitIrls{S <: VgtkCore.AbstractLossShape}

The resolved robust weighting of one fit call (internal): the deployed loss
`shape`, its squared loss width, and the scale record. Built once by
`_fit_irls` from the fit's `optimization_strategy`, `scale` and `width` keywords.
"""
struct FitIrls{S <: VgtkCore.AbstractLossShape}
    shape::S
    sq_loss_width::Float64
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end

# A fit's default strategy: the `optimization_strategy` table of its
# configs/default.toml table `fit`, read by the one strategy reader and checked
# an Irls.
_fit_strategy_default(fit::AbstractString) = _check_irls_strategy(fit, _dflt_strategy(fit, Symbol(fit)))   # fitting/_refine.jl

_check_irls_strategy(_fit, ::Nothing) = nothing
function _check_irls_strategy(fit::AbstractString, lo::AbstractOptimizationStrategy)
    lo isa Triggs && throw(ArgumentError(
        "$fit: optimization_strategy = $(VgtkCore.lo_label(lo)) is a robustified LM, which needs a starting guess: " *
        "a fit takes `optimization_strategy = Irls{K}(…)` or `nothing` (refine_* takes Triggs: refine the fit's result instead)"))
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

# THE normalized loss weight ρ′(s)/ρ′(0) ∈ [0, 1] at the squared norm `s` and squared width `c²` (Core's
# `IrlsWeights`: a maximum of 1): every fit's (`_irls_weights!`) and every refine's (`_loss_weights`, _refine.jl).
# ρ′(0) > 0 for every shipped shape at a positive width; a zero (or NaN) width gives 0/0 = NaN, as the former division by
# the stored `w0` did — the strategies refuse such a width before a weight is formed.
@inline _irls_weight(shape, s, c²) = weight(shape, s, c²) / weight(shape, zero(s), c²)

# The fit's per-datum loss weights at the residual norms `norms` into `w`
# (least squares: nothing to do).
_irls_weights!(w, ::Nothing, _norms) = w
function _irls_weights!(w, f::FitIrls, norms)
    @inbounds for i in eachindex(w, norms)
        w[i] = _irls_weight(f.shape, Float64(norms[i])^2, f.sq_loss_width)
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

# The IRLS strategy's iteration cap (least squares: one solve).
_fit_max_iter(::Nothing) = 0
_fit_max_iter(lo::Irls) = lo.max_iter

"""
    irls_termination(converged, closed_form; solve_failed = false) -> (ReturnCode.T, StopRule)

THE termination record of the reweighted fits' loop (`fit_linear`, `fit_rotation`, `fit_rigid_alignment`,
`fit_rotation_averaging`, `fit_robot_world_hand_eye`): its increment tolerance met, `(Success, IncrementSize)`;
otherwise a least-squares run (`closed_form`) is a closed form, `(Success, ClosedForm)`, and a robust one reached its
cap, `(MaxIters, IterationCap)`; a weighted solve that could not be solved inside the loop (too few rows of positive
weight left) is SciMLBase's `(InternalLinearSolveFailed, NoRule)`.

Public (as Core's `nlls_termination` is for an LM) so that a downstream reweighted loop records its stops in the
same words: BlobBoards' per-blob IRLS (its relative parameter step below `rtol` is `IncrementSize`; a singular LU is
`solve_failed`). A stop the loop adds of its own (BlobBoards' shape gate, `ShapeGate`) stays its own.
"""
function irls_termination(converged::Bool, closed_form::Bool; solve_failed::Bool = false)
    solve_failed && return (ReturnCode.InternalLinearSolveFailed, NoRule)
    converged && return (ReturnCode.Success, IncrementSize)
    return closed_form ? (ReturnCode.Success, ClosedForm) : (ReturnCode.MaxIters, IterationCap)
end
