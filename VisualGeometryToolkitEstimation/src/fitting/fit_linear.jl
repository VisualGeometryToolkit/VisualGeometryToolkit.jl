# =============================================================================
# fit_linear — a robust linear least-squares fit
# =============================================================================
#
# The coefficients x of y ≈ A·x with a fraction of the rows gross outliers
# (a log-ratio model of image values, a regression with contaminated rows).
# The scalar-residual sibling of `fit_rotation`, the same reweighted loop:
#
#   seed      the prior-weighted least squares (column-pivoted QR of √w·A);
#   repeat    the residuals rᵢ = yᵢ − aᵢᵀx; the noise scale σ̂ from their
#             median absolute value (`median|r| / √chisq_median(1)`, the
#             normal MAD scale 1.4826·median|r|) over the rows of positive
#             prior weight; the loss at width `scale_multiple · σ̂` (`deploy`,
#             default multiple the shape's efficiency constant at codimension
#             1 — Huber's 1.345); its IRLS weights times the prior; the
#             weighted least squares.
#   stop      when an iteration moves x by less than `tol` (max-norm,
#             relative to max(1, ‖x‖∞)), or at `max_iter`.
#
# Hoisted from RadiometricCamera's rooftop vignetting fit
# (experiments/vignetting/panorama_stitching/rotation_vignetting.py
# `irls_fit`: Huber at 1.345 × the MAD scale, prior weights for a bootstrap),
# 2026-10-28, with the stack's loss shapes in place of the hard-wired Huber.
# =============================================================================

"""
    LinearFitMetadata

`FitResult` metadata of [`fit_linear`](@ref), all at the returned
coefficients: `weights`, the loss's IRLS weight of every row times its prior
weight; `residuals`, `y − A·x`; `sigma`, the robust noise scale σ̂ the final
weights were computed at. They forward as properties: `result.weights`,
`result.residuals`, `result.sigma`.
"""
struct LinearFitMetadata
    weights::Vector{Float64}
    residuals::Vector{Float64}
    sigma::Float64
end

"""
    fit_linear(A, y[, prior]; loss, scale_multiple = nothing, max_iter, tol)
        -> FitResult{Vector{Float64}, LinearFitMetadata}

The robust coefficients `x` of `y ≈ A·x` (`A` an `m × n` matrix, `y` an
`m`-vector), with a fraction of the rows gross outliers. A fit — it seeds
itself with the prior-weighted least squares — iterating a reweighted least
squares: residuals `rᵢ = yᵢ − aᵢᵀx`, the robust scale
`σ̂ = median|r| / √chisq_median(1)` (the normal MAD scale, over the rows of
positive prior weight), the weights of `loss` (an `AbstractLossShape`, e.g.
`HuberShape()`) at width `scale_multiple · σ̂`, times the prior, and the
weighted least squares.

- `prior` — fixed non-negative row weights (a bootstrap's resampling counts, a
  previous fit's weights); a zero weight drops its row. Default all ones.
- `loss` — the robust loss shape (default the TOML's, Huber).
- `scale_multiple` — the width as a multiple of σ̂; `nothing` takes the
  shape's efficiency constant at codimension 1 (`deploy`'s default; Huber's
  1.345), which only an M-estimator (`AbstractMEstimator`) has: any other
  shape requires it (`ArgumentError`).
- `max_iter`, `tol` — at most `max_iter` iterations after the seed; converged
  when one moves `x` by less than `tol · max(1, ‖x‖∞)` in the max-norm
  (defaults: configs/default.toml [fit_linear]). `max_iter = k` makes `k + 1`
  weighted solves.

The least squares is a column-pivoted QR, so a column that no row with weight
reaches (an unsupported basis function) gets a basic solution's value, not an
error; drop such columns before the call when their value matters.

`retcode` is `:converged`, `:max_iterations`, or `:degenerate` when fewer rows
keep a positive weight than there are columns; on exact data (every residual
zero) the seed is returned, `:converged`. `rms` is the root mean square
residual over the rows of positive prior weight; the metadata carries the
weights, the residuals and σ̂ at the returned coefficients.
"""
function fit_linear(
        A::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
        prior::AbstractVector{<:Real} = ones(length(y));
        loss::VgtkCore.AbstractLossShape = LOSS_TAGS[_dflt("fit_linear", "loss")](),
        scale_multiple::Union{Nothing, Real} = nothing,
        max_iter::Integer = _dflt("fit_linear", "max_iter"),
        tol::Real = _dflt("fit_linear", "tol"),
    )
    m, n = size(A)
    length(y) == m || throw(DimensionMismatch("fit_linear: $m rows of A, $(length(y)) values of y"))
    length(prior) == m || throw(DimensionMismatch("fit_linear: $m rows, $(length(prior)) prior weights"))
    loss isa VgtkCore.AbstractMEstimator || scale_multiple !== nothing || throw(
        ArgumentError("fit_linear: the loss $(nameof(typeof(loss))) has no efficiency default; give its width as `scale_multiple`")
    )
    p = Float64.(VgtkCore.check_weights(collect(prior)))
    Af = Matrix{Float64}(A); yf = Vector{Float64}(y)
    x = _weighted_lstsq(Af, yf, p)
    x === nothing && return _linear_degenerate(n, m, p)
    r = yf .- Af * x
    w = copy(p)
    retcode = :max_iterations
    iter = 0
    while iter < max_iter
        iter += 1
        σ = _linear_weights!(w, r, p, loss, scale_multiple)
        if iszero(σ)                        # an exact fit: the weights are the prior's, x stands
            retcode = :converged
            break
        end
        x_new = _weighted_lstsq(Af, yf, w)
        x_new === nothing && return _linear_degenerate(n, m, w)
        Δ = maximum(abs, x_new .- x; init = 0.0)
        x = x_new
        r = yf .- Af * x
        if Δ < tol * max(1.0, maximum(abs, x; init = 0.0))
            retcode = :converged
            break
        end
    end
    σ = _linear_weights!(w, r, p, loss, scale_multiple)   # the metadata at the returned coefficients
    used = count(>(0), p)
    rms = sqrt(sum(abs2(r[i]) for i in eachindex(r) if p[i] > 0; init = 0.0) / max(used, 1))
    return FitResult(x, retcode, iter, rms, used, LinearFitMetadata(w, r, σ))
end

# σ̂ at residuals `r` — the MAD scale over the rows the caller weights (prior >
# 0) — and the loss's IRLS weights times the prior into `w`; the prior itself
# on an exact fit (σ̂ = 0). Returns σ̂.
function _linear_weights!(w, r, prior, loss, scale_multiple)
    σ = Statistics.median(abs(r[i]) for i in eachindex(r) if prior[i] > 0) / sqrt(VgtkCore.chisq_median(1))
    if iszero(σ)
        w .= prior
        return σ
    end
    c² = VgtkCore.sq_loss_width(_linear_loss(loss, σ^2, scale_multiple))
    @inbounds for i in eachindex(w, r, prior)
        w[i] = prior[i] * weight(loss, r[i]^2, c²)
    end
    return σ
end

_linear_loss(loss::VgtkCore.AbstractMEstimator, s2, ::Nothing) = deploy(loss, s2; codim = 1)
_linear_loss(loss::VgtkCore.AbstractMEstimator, s2, scale_multiple::Real) = deploy(loss, s2; codim = 1, scale_multiple)
_linear_loss(loss::VgtkCore.AbstractLossShape, s2, scale_multiple::Real) = deploy(loss, s2; scale_multiple)

# The weighted least squares on the rows of positive weight: the column-pivoted
# QR of √w·A; `nothing` when fewer such rows than columns remain.
function _weighted_lstsq(A, y, w)
    keep = findall(>(0), w)
    length(keep) >= size(A, 2) || return nothing
    s = sqrt.(w[keep])
    return LinearAlgebra.qr(s .* A[keep, :], LinearAlgebra.ColumnNorm()) \ (s .* y[keep])
end

_linear_degenerate(n, m, w) =
    FitResult(fill(NaN, n), :degenerate, 0, NaN, count(>(0), w), LinearFitMetadata(collect(Float64, w), fill(NaN, m), NaN))
