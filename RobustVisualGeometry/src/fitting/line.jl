# =============================================================================
# Line Fitting — Orthogonal distance with optional per-point covariances
# =============================================================================
#
# Implements the weighted orthogonal-distance fit:
#   - fit_line(uncertain_points)        — weighted by projected covariances
# (The unweighted total least squares is VisualGeometryCore's `tls_line`: one
# algebraic step, so an algorithm name, not a fit — estimator taxonomy v2.)
#
# Line parametrization: n⊤p + d = 0, n = (cos φ, sin φ)
# Parameter vector θ = (φ, d), covariance Cov(θ̂) = s² · (J⊤WJ)⁻¹
#
# The covariance-weighted step `_weighted_line` is shared with the
# LineFittingProblem's LO-RANSAC `refine!` (problems/line.jl), which runs it
# under the local optimizer's robust weights — so a soft-weighting LO fits the
# line it weighs, not every point with positive weight at full weight.
#
# =============================================================================

# Dependencies: StaticArrays (SVector, SMatrix)
#               VGC: Line2D, Point2, Uncertain, normal, tls_line, tls_line_statistics

# =============================================================================
# Result Type
# =============================================================================

"""
    LineFitMetadata{T}

`FitResult` metadata of [`fit_line`](@ref). The core carries the fitted
line — `estimate` is an `Uncertain{Line2D{T}, <:CovMat{2,T}}` with the
(φ, d) parameter covariance — `retcode = :solved`, `iterations = 1` (the one
covariance reweight-and-solve cycle after the total-least-squares seed), and
`rms`, the root-mean-square orthogonal residual over the `n` points. Here:

- `s²::T`: estimated residual variance of the whitened residuals, `S/(n-2)`
- `ν::Int`: degrees of freedom, `n - 2`

Both forward as properties: `result.s²`, `result.ν`.
"""
struct LineFitMetadata{T}
    s²::T
    ν::Int
end

# =============================================================================
# The covariance-weighted line step (shared by fit_line and the LO refit)
# =============================================================================

# One reweight: `w[i] = prior[i]/σᵢ²` at the normal `nv` (a zero prior drops the
# point). `false` when a point with positive prior has no variance along the
# normal (`σᵢ² ≤ 0`): its weight would be infinite.
function _normal_weights!(w::AbstractVector, cov_shapes, prior::AbstractVector, nv)
    @inbounds for i in eachindex(w, cov_shapes, prior)
        pw = prior[i]
        if iszero(pw)
            w[i] = zero(eltype(w))
            continue
        end
        σ² = propagate(nv', cov_shapes[i])          # nᵀΣn: the point's variance along the normal
        σ² > zero(σ²) || return false
        w[i] = pw / σ²
    end
    return true
end

"""
    _weighted_line(points, cov_shapes, prior, nv₀) -> (line, S, JᵀWJ) | nothing

The covariance-weighted line of `points` from the seed normal `nv₀`: point
`i` weighs `prior[i]/σᵢ²`, `σᵢ² = nᵀΣᵢn` its variance along the current
normal, `prior[i] ≥ 0` a caller's weight (`1` for [`fit_line`](@ref), the LO's
robust weight for `refine!(…, ::LineFittingProblem, …)`; `0` drops the point).
One reweight-and-solve cycle, each solve VisualGeometryCore's fixed-weight
total-least-squares step (`tls_line_statistics`): weights at `nv₀` → weighted
TLS normal; weights at that normal → the offset through the weighted centroid.
Returns the line, the weighted residual sum `S = Σwᵢrᵢ²` and the (φ, d)
information `JᵀWJ`, all at the final normal.

`nothing` when a point with positive prior weight has no variance along the
normal (`σᵢ² ≤ 0`): its weight would be infinite. That is the stack's one
Sampson degeneracy rule (VGC `whiten`), which the scorer reads as an exact or
impossible point — never an `eps` floor, which would weigh the point by an
arbitrary 1/eps instead.
"""
function _weighted_line(points::AbstractVector, cov_shapes::AbstractVector, prior::AbstractVector, nv₀)
    w = similar(prior)
    _normal_weights!(w, cov_shapes, prior, nv₀) || return nothing
    nv = normal(first(tls_line_statistics(points, w)))
    _normal_weights!(w, cov_shapes, prior, nv) || return nothing
    return tls_line_statistics(points, w, nv)
end

# =============================================================================
# fit_line (per-point covariances)
# =============================================================================

"""
    fit_line(points::AbstractVector{<:Uncertain{<:Point2}}) -> FitResult{Uncertain{Line2D}, LineFitMetadata}

Fit a 2D line with per-point covariance weighting.

Each point carries a 2×2 covariance matrix Σᵢ. The projected variance along the
line normal is σᵢ² = n⊤Σᵢn, giving weight wᵢ = 1/σᵢ².

Seeds the normal with VisualGeometryCore's unweighted [`tls_line`](@ref), then
runs one reweight-and-solve cycle: weights at the seed normal → weighted
scatter → refined normal → weights recomputed → weighted offset. The (φ, d)
covariance is `s²·(JᵀWJ)⁻¹` at the final normal. Throws `ArgumentError` for
fewer than 3 points, or when a point has no variance along the normal.
"""
function fit_line(points::AbstractVector{<:Uncertain{<:Point2}})
    n = length(points)
    n >= 3 || throw(ArgumentError("fit_line requires at least 3 points, got $n"))

    # The problem's constructor is the one conversion of uncertain points into
    # (points, covariance shapes); the fit reads its storage.
    lp = LineFittingProblem(points)
    T = eltype(eltype(lp.points))
    nv₀ = SVector{2, T}(normal(geom(tls_line(lp.points))))
    out = _weighted_line(lp.points, lp.cov_shapes, ones(T, n), nv₀)
    out === nothing && throw(
        ArgumentError("fit_line: a point's covariance has zero variance along the line normal")
    )
    line, S, JtWJ = out

    ν = residual_dof(LineFittingProblem, n)   # (n − 2)·1
    s² = S / ν
    return FitResult(Uncertain(line, CovMat(s² * inv(JtWJ))), :solved, 1, sqrt(S / n), n, LineFitMetadata(s², ν))
end
