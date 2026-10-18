# =============================================================================
# tls_line — the total-least-squares line (one closed-form step)
# =============================================================================
#
# Moved up from VisualGeometryToolkitEstimation's unweighted `fit_line` (taxonomy v2: one
# deterministic step is an algorithm, and algorithms live here). The weighted
# form (FIXED weights) is the same one step, so it lives here too; the line fit
# that REWEIGHTS by per-point covariances along the normal stays
# VisualGeometryToolkitEstimation's `fit_line`, built on `tls_line_statistics`.

"""
    tls_line(points) -> Uncertain{Line2D{T}}
    tls_line(points, weights) -> Uncertain{Line2D{T}}

Total least squares (orthogonal distance regression) line through `points`
(2-vectors): the centroid, and the scatter matrix's smallest-eigenvalue
eigenvector as the normal — one closed-form step. The line is in Hesse form
`n⊤p + d = 0`, `n = (cos φ, sin φ)`, and carries the covariance of
`θ = (φ, d)`: `s² (JᵀWJ)⁻¹`, with `s² = Σwᵢrᵢ²/(n₊−2)` the residual variance,
`n₊` the number of points with positive weight and `J` the residual Jacobian —
calibrated against Monte Carlo.

`weights` are FIXED non-negative per-point weights (`wᵢ = 1/σᵢ²` for known
isotropic variances): the weighted centroid, the weighted scatter, and a zero
weight drops the point. Without them every point weighs 1. Throws
`ArgumentError` for fewer than 3 points of positive weight, a negative or
non-finite weight, or a `weights` whose length differs from `points`'.

For points carrying their own covariances, VisualGeometryToolkitEstimation's
`fit_line` (an iterative weighted fit) applies.
"""
function tls_line(points::AbstractVector{<:StaticVector{2}})
    T = float(eltype(eltype(points)))
    return _tls_line(points, _ -> one(T), length(points), T)
end

function tls_line(points::AbstractVector{<:StaticVector{2}}, weights::AbstractVector{<:Real})
    check_weights(weights, length(points))
    T = float(promote_type(eltype(eltype(points)), eltype(weights)))
    return _tls_line(points, i -> T(weights[i]), count(>(0), weights), T)
end

function _tls_line(points, weight::F, n₊::Int, ::Type{T}) where {F, T}
    n₊ >= 3 || throw(ArgumentError("tls_line requires at least 3 points of positive weight, got $n₊"))
    line, S, JtWJ = _tls_line_statistics(points, weight, nothing, T)
    return Uncertain(line, CovMat(S / (n₊ - 2) * inv(JtWJ)))
end

"""
    tls_line_statistics(points, weights[, normal]) -> (line, S, JᵀWJ)

The weighted total-least-squares line of [`tls_line`](@ref)`(points, weights)`
with the two statistics its covariance is made of, both at the returned line:
the weighted residual sum `S = Σwᵢrᵢ²` and the `(φ, d)` information `JᵀWJ`
(`J`'s rows `(tᵀpᵢ, 1)`, `t` the line's direction). The residual variance is
`S/(n₊ − 2)` and the covariance `S/(n₊ − 2)·(JᵀWJ)⁻¹`; a caller that owns its
own variance model reads the pieces instead.

With `normal` given (a 2-vector, taken as is), the step estimates the offset
only: the line with that normal through the weighted centroid — the least-squares
offset for a known normal.

`weights` as for `tls_line`: FIXED and non-negative, a zero weight drops the
point; there is no minimum count here (the covariance's is the caller's), but
weights that sum to zero throw `ArgumentError`.
"""
function tls_line_statistics(
        points::AbstractVector{<:StaticVector{2}}, weights::AbstractVector{<:Real},
        normal::Union{Nothing, StaticVector{2}} = nothing
    )
    check_weights(weights, length(points))
    T = float(promote_type(eltype(eltype(points)), eltype(weights)))
    return _tls_line_statistics(points, i -> T(weights[i]), normal, T)
end

# The one step, on a weight FUNCTION `weight(i)` (unit weights allocate nothing).
function _tls_line_statistics(points, weight::F, nv_given, ::Type{T}) where {F, T}
    # Weighted centroid and scatter about it.
    W = zero(T); mx = zero(T); my = zero(T)
    @inbounds for (i, p) in enumerate(points)
        w = weight(i)
        iszero(w) && continue
        W += w; mx += w * T(p[1]); my += w * T(p[2])
    end
    W > 0 || throw(ArgumentError("tls_line_statistics: the weights sum to zero; no point supports the line"))
    mx /= W; my /= W

    # The normal: the scatter's smallest-eigenvalue eigenvector — or the one
    # given, when only the offset is estimated; d = −n⊤p̄.
    nv = if nv_given === nothing
        m11 = zero(T); m12 = zero(T); m22 = zero(T)
        @inbounds for (i, p) in enumerate(points)
            w = weight(i)
            iszero(w) && continue
            dx = T(p[1]) - mx; dy = T(p[2]) - my
            m11 += w * dx * dx; m12 += w * dx * dy; m22 += w * dy * dy
        end
        E = eigen(Symmetric(SMatrix{2, 2, T}(m11, m12, m12, m22)))
        SVector{2, T}(E.vectors[:, argmin(E.values)])
    else
        SVector{2, T}(nv_given)
    end
    line = Line2D{T}(SVector{3, T}(nv[1], nv[2], -(nv[1] * mx + nv[2] * my)))
    nv = normal(line); d = coeffs(line)[3]
    tv = SVector{2, T}(-nv[2], nv[1])

    # Weighted residuals and the (φ, d) information JᵀWJ.
    S = zero(T); j11 = zero(T); j12 = zero(T); j22 = zero(T)
    @inbounds for (i, p) in enumerate(points)
        w = weight(i)
        iszero(w) && continue
        ri = nv[1] * T(p[1]) + nv[2] * T(p[2]) + d
        ti = tv[1] * T(p[1]) + tv[2] * T(p[2])
        S += w * ri * ri
        j11 += w * ti * ti; j12 += w * ti; j22 += w
    end
    return (line, S, SMatrix{2, 2, T}(j11, j12, j12, j22))
end
