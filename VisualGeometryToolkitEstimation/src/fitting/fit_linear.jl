# =============================================================================
# fit_linear — a robust linear least-squares fit
# =============================================================================
#
# The coefficients x of y ≈ A·x with a fraction of the rows gross outliers
# (a log-ratio model of image values, a regression with contaminated rows).
# The scalar-residual sibling of `fit_rotation`, the same reweighted loop:
#
#   seed      the prior-weighted least squares (column-pivoted QR of √w·A);
#   repeat    the residuals rᵢ = yᵢ − aᵢᵀx; the IRLS weights of the
#             `Irls{K}` strategy's loss at ONE loss width, set once from
#             the caller's scale input (`scale = σ` or an upstream result:
#             the efficiency constant × σ, Huber's 1.345 at codimension 1;
#             or `width = w`) and never re-estimated (irls_weights.jl);
#             those weights times the prior; the weighted least squares.
#   stop      when an iteration moves x by less than `tol` (max-norm,
#             relative to max(1, ‖x‖∞)), or at the strategy's `max_iter`.
#   optimization_strategy = nothing: the prior-weighted least squares, one solve.
#
# Hoisted from RadiometricCamera's rooftop vignetting fit
# (experiments/vignetting/panorama_stitching/rotation_vignetting.py
# `irls_fit`: Huber at 1.345 × the MAD scale, prior weights for a bootstrap),
# 2026-10-28, with the stack's loss shapes in place of the hard-wired Huber.
# Since the robust-estimation contract (0.9) the scale is the caller's, fixed:
# the MAD scale it re-estimated every iteration is gone.
# =============================================================================

"""
    LinearFitMetadata{L, W}

`FitResult` metadata of [`fit_linear`](@ref), all at the returned
coefficients (`L` the `optimization_strategy`'s type, `Nothing` for least squares):

- `weights::W` — the loss weight `ρ′(r²)` of every row, normalized to a
  maximum of 1 (`IrlsWeights`); a lazy all-ones vector for least squares
  (`BinaryWeights`). The solve's weight of a row is its prior times this.
- `residuals` — `y − A·x`.
- `scale`, `scale_source` — the scale record (`noise_scale(result)`,
  `scale_source(result)`): the scale given (`ScaleGiven`, `ScaleUpstream`),
  or `NaN` and `ScaleNone` for least squares or a `width`-only fit.
- `loss_width` — the loss width the weights were computed at (`NaN` for
  least squares).
"""
struct LinearFitMetadata{L, W <: AbstractVector{Float64}}
    weights::W
    residuals::Vector{Float64}
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end
LinearFitMetadata{L}(w::W, r, s, src, lw) where {L, W} = LinearFitMetadata{L, W}(w, r, s, src, lw)
VgtkCore.weight_kind(::Type{<:LinearFitMetadata{L}}) where {L} = VgtkCore.weight_kind(L)   # Core's trait of the strategy (Irls: IrlsWeights; nothing: binary)

"""
    fit_linear(A, y[, prior]; optimization_strategy, scale | width, tol)
        -> FitResult{Vector{Float64}, LinearFitMetadata}

The robust coefficients `x` of `y ≈ A·x` (`A` an `m × n` matrix, `y` an
`m`-vector), with a fraction of the rows gross outliers. A fit — it seeds
itself with the prior-weighted least squares — iterating a reweighted least
squares at ONE loss width: residuals `rᵢ = yᵢ − aᵢᵀx`, the weights of the
strategy's loss at that width, times the prior, and the weighted least
squares. The width is fixed by the caller's scale input and never
re-estimated.

- `prior` — fixed non-negative row weights (a bootstrap's resampling counts, a
  previous fit's weights); a zero weight drops its row. Default all ones.
- `optimization_strategy` — the robust weighting, an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}` strategy (its loss
  `K`, its efficiency constant and its `max_iter`; default
  configs/default.toml `[fit_linear] optimization_strategy`, Huber), or `nothing` for the
  plain prior-weighted least squares. At most `max_iter` iterations after the
  seed: `Irls{HuberShape}(max_iter = k)` makes `k + 1` weighted solves.
- `scale` | `width` — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the residuals' noise scale, in `y`'s units), `scale = result` (an upstream
  robust result's `noise_scale`), or `width = w` (the loss width itself).
  The width is `w`, or `c · σ` with `c` the strategy's efficiency constant,
  by default the shape's at codimension 1 (`efficiency_constant(K, 1)`:
  Huber 1.345); a shape with none (TLS) needs the strategy's or a `width`.
  Neither, both, or one given with `optimization_strategy = nothing` is an `ArgumentError`.
- `tol` — converged when an iteration moves `x` by less than
  `tol · max(1, ‖x‖∞)` in the max-norm (default configs/default.toml
  [fit_linear]).

The least squares is a column-pivoted QR, so a column that no row with weight
reaches (an unsupported basis function) gets a basic solution's value, not an
error; drop such columns before the call when their value matters.

`retcode_symbol(result)` is `:converged`, `:max_iterations`, `:solved` (least squares), or
`:degenerate` when fewer rows keep a positive weight than there are columns.
`n` is the support (the rows of positive solve weight) and `rms` the
weight-weighted root mean square residual over the rows,
`√(Σ pᵢwᵢrᵢ² / Σ pᵢwᵢ)`. The metadata ([`LinearFitMetadata`](@ref)) carries
the weights, the residuals, the scale record and the loss width.
"""
function fit_linear(
        A::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        prior::AbstractVector{<:Real} = ones(length(y));
        optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = _fit_strategy_default("fit_linear"),
        local_optimization = RetiredKeyword(),   # refused, naming its successor
        scale = nothing, width::Union{Nothing, Real} = nothing,
        tol::Real = _dflt("fit_linear", "tol"),
    )
    m, n = size(A)
    length(y) == m || throw(DimensionMismatch("fit_linear: $m rows of A, $(length(y)) values of y"))
    length(prior) == m || throw(DimensionMismatch("fit_linear: $m rows, $(length(prior)) prior weights"))
    refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    f = _fit_irls("fit_linear", optimization_strategy, scale, width, 1)
    p = Float64.(VgtkCore.check_weights(collect(prior)))
    Af = Matrix{Float64}(A); yf = Vector{Float64}(y)
    w = _fit_weights_storage(f, m)
    x = _weighted_lstsq(Af, yf, p)
    x === nothing && return _linear_degenerate(optimization_strategy, f, n, m, w)
    r = yf .- Af * x
    tol_met = false
    iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        _irls_weights!(w, f, r)
        x_new = _weighted_lstsq(Af, yf, _solve_weights(p, w))
        x_new === nothing && return _linear_degenerate(optimization_strategy, f, n, m, w, _irls_record(false, false; solve_failed = true))
        Δ = maximum(abs, x_new .- x; init = 0.0)
        x = x_new
        r = yf .- Af * x
        if Δ < tol * max(1.0, maximum(abs, x; init = 0.0))
            tol_met = true
            break
        end
    end
    _irls_weights!(w, f, r)                   # the metadata at the returned coefficients
    sw = _solve_weights(p, w)
    used, rms = _weighted_support_rms(sw, (abs2(x) for x in r))
    return FitResult(x, _irls_record(tol_met, f === nothing)..., iter, rms, used, LinearFitMetadata{typeof(optimization_strategy)}(w, r, _fit_scale_record(f)...))
end

# The weighted least squares on the rows of positive weight: the column-pivoted
# QR of √w·A; `nothing` when fewer such rows than columns remain.
function _weighted_lstsq(A, y, w)
    keep = findall(>(0), w)
    length(keep) >= size(A, 2) || return nothing
    s = sqrt.(w[keep])
    return LinearAlgebra.qr(s .* A[keep, :], LinearAlgebra.ColumnNorm()) \ (s .* y[keep])
end

# A solve that failed: before the loop, a degenerate start `(InitialFailure, NoRule)`; inside it, a weighted
# solve that could not be solved `(InternalLinearSolveFailed, NoRule)`.
_linear_degenerate(strategy, f, n, m, w, record::Tuple{ReturnCode.T, StopRule} = (ReturnCode.InitialFailure, NoRule)) =
    FitResult(fill(NaN, n), record..., 0, NaN, 0, LinearFitMetadata{typeof(strategy)}(w, fill(NaN, m), _fit_scale_record(f)...))
