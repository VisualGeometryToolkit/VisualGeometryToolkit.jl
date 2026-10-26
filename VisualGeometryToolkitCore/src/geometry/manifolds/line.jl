# =============================================================================
# Line2DManifold — the chart of a 2D line n·x + d = 0 (unit normal): the angle
# of its normal and its offset.
# =============================================================================

"""
    Line2DManifold(l::Line2D) <: Manifold

The 2-dof chart of a 2D line `n·x + d = 0` (`|n| = 1`): `v = (φ, δd)` rotates
the normal by `φ` and moves the offset, `retract(ch, v) = Line2D(R(φ)·n, d + δd)`.
The ambient space is the line's coefficients `(n₁, n₂, d)`, `retract_jacobian`
`[R(φ)·t  0; 0  1]` with `t = (−n₂, n₁)` the line's direction. A line acts on a
point by its signed distance ([`constraint_with_jacobians`](@ref)`(l, x)`), so
`action_jacobian(ch, x) = (t·x, 1)` — the point-on-line constraint's chart row.
`local_coordinates` is closed form and reads `(n, d)` and `(−n, −d)` as the same
line. `minimal_manifold(l::Line2D)` is this chart.
"""
struct Line2DManifold{T} <: Manifold
    line::Line2D{T}
end

tangent_size(::Line2DManifold) = 2
ambient_coordinates(l::Line2D) = coeffs(l)
model(ch::Line2DManifold) = ch.line

function retract(ch::Line2DManifold, v)
    n = normal(ch.line)
    c, s = cos(v[1]), sin(v[1])
    return Hyperplane(SVector(c * n[1] - s * n[2], s * n[1] + c * n[2], coeffs(ch.line)[3] + v[2]))
end
recenter(ch::Line2DManifold, v) = Line2DManifold(retract(ch, v))

function retract_jacobian(ch::Line2DManifold{T}, v::StaticVector = zero(SVector{2, T})) where {T}
    n = normal(ch.line)
    c, s = cos(v[1]), sin(v[1])
    z = zero(c)
    return SMatrix{3, 2}(-s * n[1] - c * n[2], c * n[1] - s * n[2], z, z, z, one(c))
end

function action_jacobian(ch::Line2DManifold, x::AbstractVector)
    n = normal(ch.line)
    return SMatrix{1, 2}(-n[2] * x[1] + n[1] * x[2], one(eltype(n)))
end

function local_coordinates(ch::Line2DManifold, l::Line2D)
    n0 = normal(ch.line)
    n, d = normal(l), coeffs(l)[3]
    if dot(n0, n) < 0                      # (−n, −d) is the same line
        n, d = -n, -d
    end
    return SVector(atan(_cross2d(n0, n), dot(n0, n)), d - coeffs(ch.line)[3])
end
