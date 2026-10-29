# =============================================================================
# CircleManifold — the chart of a 2D circle ‖x − c‖ = r (GeometryBasics'
# `Circle`): the centre free, the radius on a log scale.
# =============================================================================

"""
    CircleManifold(c::Circle) <: Manifold

The 3-dof chart of a 2D circle with centre `c` and radius `r > 0`:
`v = (δc₁, δc₂, δlog r)` moves the centre and scales the radius,
`retract(ch, v) = Circle(c + (δc₁, δc₂), r·exp(δlog r))`, so every point of the
chart is a circle with a positive radius. The ambient space is `(c₁, c₂, r)`
([`ambient_coordinates`](@ref)); `retract_jacobian` is `diag(1, 1, r·exp(δlog r))`.
A circle acts on a point by the point-on-circle constraint
([`constraint_with_jacobians`](@ref)`(c, x)`), so `action_jacobian(ch, x) =
(−2(x − c)ᵀ, −2r²)` is that constraint's chart row. `local_coordinates` is closed
form. `minimal_manifold(c::Circle)` is this chart.
"""
struct CircleManifold{T} <: Manifold
    circle::Circle{T}
end

tangent_size(::CircleManifold) = 3
ambient_coordinates(c::Circle) = (o = GeometryBasics.origin(c); SVector(o[1], o[2], radius(c)))
model(ch::CircleManifold) = ch.circle

function retract(ch::CircleManifold, v)
    o = GeometryBasics.origin(ch.circle)
    c₁ = o[1] + v[1]; c₂ = o[2] + v[2]; r = radius(ch.circle) * exp(v[3])
    T = promote_type(typeof(c₁), typeof(c₂), typeof(r))
    return Circle(Point2{T}(c₁, c₂), T(r))
end
recenter(ch::CircleManifold, v) = CircleManifold(retract(ch, v))

function retract_jacobian(ch::CircleManifold{T}, v::StaticVector = zero(SVector{3, T})) where {T}
    s = radius(ch.circle) * exp(v[3])
    z = zero(s); o = one(s)
    return SMatrix{3, 3}(o, z, z, z, o, z, z, z, s)
end

function action_jacobian(ch::CircleManifold, x::AbstractVector)
    _, _, G_θ = constraint_with_jacobians(ch.circle, SVector(x[1], x[2]))
    return transpose(G_θ) * retract_jacobian(ch)
end

function local_coordinates(ch::CircleManifold, c::Circle)
    o₀ = GeometryBasics.origin(ch.circle); o = GeometryBasics.origin(c)
    return SVector(o[1] - o₀[1], o[2] - o₀[2], log(radius(c) / radius(ch.circle)))
end

"""
    constraint_with_jacobians(c::Circle, x::StaticVector{2}) -> (g, G_x::SVector{2}, G_θ::SVector{3})

The point-on-circle constraint `g = ‖x − c‖² − r²` with `G_x = 2(x − c)` and
`G_θ = ∂g/∂(c₁, c₂, r) = (−2(x − c), −2r)`, the circle's ambient coordinates
([`CircleManifold`](@ref)'s). Its Sampson distance `g/‖G_x‖ = (d² − r²)/(2d)`,
`d = ‖x − c‖`, is the geometric distance `d − r` to first order (the ratio is
`(d + r)/(2d)`). A point at the centre has `G_x = 0` — the one Sampson
degeneracy (VisualGeometryToolkitCore's `whiten`).
"""
@inline function constraint_with_jacobians(c::Circle, x::StaticVector{2})
    o = GeometryBasics.origin(c); r = radius(c)
    e = SVector(x[1] - o[1], x[2] - o[2])
    g = e[1]^2 + e[2]^2 - r^2
    return g, 2 * e, SVector(-2 * e[1], -2 * e[2], -2 * r)
end
minimal_manifold(c::Circle) = CircleManifold(c)
