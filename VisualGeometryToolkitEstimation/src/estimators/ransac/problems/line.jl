# =============================================================================
# RANSAC Line Fitting — LineFittingProblem
# =============================================================================
#
# Implements AbstractRansacProblem for 2D line estimation from points with
# optional per-point covariances. Uses standardized residuals for comparable
# scoring across heterogeneous noise levels.
#
# =============================================================================

# =============================================================================
# LineFittingProblem — Plain RANSAC (no local optimization)
# =============================================================================

"""
    LineFittingProblem{T} <: AbstractRansacProblem

Plain RANSAC problem for estimating a 2D line from points with per-point covariances.

Residuals are **standardized**: `r_std[i] = (n⊤pᵢ + d) / √(n⊤Σᵢn)`, so all
points contribute on a comparable scale regardless of their covariance shape.

Supports LO-RANSAC via `refine!(ws, problem, θ, weights)` (θ unread: a linear
problem's step): the covariance-weighted line of `fit_line` with each point's
weight `wᵢ/σᵢ²` — the local optimizer's robust weight times its inverse
variance along the normal — seeded by the `w`-weighted total-least-squares
normal. A zero weight drops the point.

# Constructors
    LineFittingProblem(pts::AbstractVector{<:Uncertain{<:Point2}})
    LineFittingProblem(pts::AbstractVector{<:Point2})  # identity covariances

# Solver Details
- Minimal sample: 2 points (returns `FixedModels{1, Line2D{T}}`)
- Model type: `Line2D{T}` (Hesse normal form)
- Residual: Standardized orthogonal distance
- LO-RANSAC: `refine!(ws, p, θ, weights)` — the `weights`-weighted `fit_line` step, θ unread
"""
struct LineFittingProblem{T} <: AbstractRansacProblem{2}
    points::Vector{Point2{T}}
    cov_shapes::Vector{SMatrix{2, 2, T, 4}}
end

# ONE construction: each point's position and covariance shape — the
# `Uncertain` point's own, or the identity for a bare point.
function LineFittingProblem(pts::AbstractVector{<:Union{Point2, Uncertain{<:Point2}}})
    T = _line_eltype(pts)
    return LineFittingProblem{T}(
        [Point2{T}(_line_point(p)...) for p in pts], [_line_shape(T, p) for p in pts]
    )
end
_line_eltype(pts::AbstractVector{<:Uncertain}) = float(eltype(cov(first(pts))))
_line_eltype(pts::AbstractVector{<:Point2}) = float(eltype(eltype(pts)))
_line_point(p::Uncertain) = geom(p)
_line_point(p::Point2) = p
_line_shape(::Type{T}, p::Uncertain) where {T} = SMatrix{2, 2, T, 4}(cov(p))
_line_shape(::Type{T}, ::Point2) where {T} = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

# =============================================================================
# AbstractRansacProblem Interface — LineFittingProblem
# =============================================================================

codimension(::Type{<:LineFittingProblem}) = 1  # d_g = 1: signed distance is scalar
data_size(p::LineFittingProblem) = length(p.points)
model_type(::Type{<:LineFittingProblem{T}}) where {T} = Line2D{T}
function solve(p::LineFittingProblem, idx::AbstractVector{Int})
    line = @inbounds Line2D(p.points[idx[1]], p.points[idx[2]])
    return FixedModels{1, typeof(line)}(1, (line,))
end

# `residuals!` is VisualGeometryToolkitCore's default: the whitened constraint from
# `constraint_with_jacobians` below through VGC's `whiten` — the stack's one
# Sampson degeneracy rule (a vanishing projected variance nᵀΣn gives 0 or Inf),
# not the `max(σᵢ, eps)` floor this problem carried until 2026-09-22.

function test_sample(p::LineFittingProblem{T}, idx::AbstractVector{Int}) where {T}
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
        dx = T(p1[1]) - T(p2[1])
        dy = T(p1[2]) - T(p2[2])
        return dx * dx + dy * dy > eps(T)
    end
end

# =============================================================================
# refine! — the line problem's local-optimization step: the covariance-weighted
# line (`_weighted_line`, fitting/line.jl) under the LO's weights (a linear
# problem: θ is not read)
# =============================================================================

function refine!(_lo_ws, p::LineFittingProblem{T}, _θ, weights::AbstractVector) where {T}
    _underdetermined(p, weights) && return nothing
    # VGC's one weights contract (finite, non-negative — a violation is the
    # caller's error, as in every LM refit's row scaling).
    VgtkCore.check_weights(weights)
    prior = T[T(weights[i]) for i in 1:data_size(p)]
    # Seed normal: the prior-weighted total-least-squares direction.
    nv₀ = normal(first(tls_line_statistics(p.points, prior)))
    out = _weighted_line(p.points, p.cov_shapes, prior, nv₀)
    return out === nothing ? nothing : first(out)
end

# --- constraint_with_jacobians for LineFittingProblem (heteroscedastic) ---
#
# VGC's point-on-line constraint (`constraint_with_jacobians(::Line2D, x)`: g = n·x + d,
# G_x = n) with the model row in the line's Hesse chart (φ, d) — VGC's
# `action_jacobian(Line2DManifold(l), x)` = (t·x, 1) — plus the point's
# covariance shape as the 4th value, so whiten(splat...) weighs by it.
@inline function constraint_with_jacobians(
        p::LineFittingProblem{T},
        model::Line2D{T}, i::Int
    ) where {T}
    @inbounds begin
        pt = p.points[i]
        Σ̃_x = p.cov_shapes[i]
    end
    x = SVector{2, T}(T(pt[1]), T(pt[2]))
    g, G_x, _ = VgtkCore.constraint_with_jacobians(model, x)
    ∂_θg = SVector{2, T}(action_jacobian(Line2DManifold(model), x))
    return (g, Vec2{T}(G_x), ∂_θg, Σ̃_x)
end
provides_solver_covariance(::Type{<:LineFittingProblem}) = true   # predictive scoring (covariance.jl)

# =============================================================================
# InhomLineFittingProblem — Simplest possible RANSAC problem (y = a + bx)
# =============================================================================

"""
    InhomLineFittingProblem{T} <: AbstractRansacProblem

Inhomogeneous line fitting: `y = a + b·x`, 2 parameters, scalar residual.

The simplest possible RANSAC problem for debugging and testing the
covariance-aware pipeline. No standardization, no per-point covariances.

# Model
- `SVector{2,T}`: `(a, b)` where `y = a + b·x`
- Residual: `r_i = y_i - a - b·x_i`  (scalar, raw)
- Residual DOF: 1
- Model DOF: 2
"""
struct InhomLineFittingProblem{T} <: AbstractRansacProblem{2}
    points::Vector{Point2{T}}
end

function InhomLineFittingProblem(pts::AbstractVector)
    T = float(eltype(eltype(pts)))
    return InhomLineFittingProblem{T}([Point2{T}(p[1], p[2]) for p in pts])
end

codimension(::Type{<:InhomLineFittingProblem}) = 1  # d_g = 1: signed distance is scalar
data_size(p::InhomLineFittingProblem) = length(p.points)
model_type(::Type{<:InhomLineFittingProblem{T}}) where {T} = SVector{2, T}
function solve(p::InhomLineFittingProblem{T}, idx::AbstractVector{Int}) where {T}
    M = SVector{2, T}
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
        dx = p2[1] - p1[1]
        abs(dx) < eps(T) && return FixedModels{1, M}(0, (zero(M),))
        b = (p2[2] - p1[2]) / dx
        a = p1[2] - b * p1[1]
    end
    return FixedModels{1, M}(1, (M(a, b),))
end

# `residuals!` is VisualGeometryToolkitCore's default through
# `constraint_with_jacobians` below (∂ₓg = [1], so the whitened residual is g
# itself, bit-for-bit).

function test_sample(p::InhomLineFittingProblem{T}, idx::AbstractVector{Int}) where {T}
    @inbounds begin
        dx = p.points[idx[2]][1] - p.points[idx[1]][1]
        return abs(dx) > eps(T)
    end
end

# --- constraint_with_jacobians: g = yᵢ - a - bxᵢ, ∂ₓg = [1], ∂_θg = [-1, -xᵢ] ---

@inline function constraint_with_jacobians(
        p::InhomLineFittingProblem{T},
        model::SVector{2, T}, i::Int
    ) where {T}
    @inbounds begin
        pt = p.points[i]
        a, b = model
        g = pt[2] - a - b * pt[1]
        ∂ₓg = SA[one(T)]
        ∂_θg = SA[-one(T), -pt[1]]
    end
    return (g, ∂ₓg, ∂_θg)
end
provides_solver_covariance(::Type{<:InhomLineFittingProblem}) = true   # predictive scoring (covariance.jl)
