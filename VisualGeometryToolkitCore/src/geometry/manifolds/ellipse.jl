# =============================================================================
# EllipseManifold — the chart of a 2D ellipse (x − c)ᵀ Q (x − c) = 1: the
# centre free, the shape on the log-shape chart (`ellipse_log_shape`).
# =============================================================================

"""
    EllipseManifold(e::Ellipse) <: Manifold

The 5-dof chart of a 2D ellipse: VisualGeometryToolkitCore's log-shape chart
([`ellipse_log_shape`](@ref)) taken as local coordinates,
`v = (δc₁, δc₂, δS₁₁, δS₁₂, δS₂₂)`, `retract(ch, v) =
ellipse_from_log_shape(ellipse_log_shape(e) + v)`, so every point of the chart is
a real ellipse (the exponential of every symmetric `S` is positive definite) and
the chart is smooth at a circle. The ambient space is the centre and the
quadratic form `Q = exp(−2S) = R·diag(a⁻², b⁻²)·Rᵀ` of the ellipse,
`(c₁, c₂, Q₁₁, Q₁₂, Q₂₂)` ([`ambient_coordinates`](@ref)), in which the
point-on-ellipse constraint ([`constraint_with_jacobians`](@ref)`(e, x)`) is
polynomial; `retract_jacobian` is `diag(I₂, ∂Q/∂S)`, the derivative of the
symmetric matrix exponential in closed form (Daleckii–Krein).
`action_jacobian(ch, x)` is that constraint's chart row. `local_coordinates`
is closed form. `minimal_manifold(e::Ellipse)` is this chart.
"""
struct EllipseManifold{T} <: Manifold
    ellipse::Ellipse{T}
end

tangent_size(::EllipseManifold) = 5
function ambient_coordinates(e::Ellipse)
    Q = _ellipse_form(e)
    return SVector(e.center[1], e.center[2], Q[1, 1], Q[1, 2], Q[2, 2])
end
model(ch::EllipseManifold) = ch.ellipse

retract(ch::EllipseManifold, v) = ellipse_from_log_shape(ellipse_log_shape(ch.ellipse) + SVector{5}(v[1], v[2], v[3], v[4], v[5]))
recenter(ch::EllipseManifold, v) = EllipseManifold(retract(ch, v))

# The quadratic form Q = R·diag(a⁻², b⁻²)·Rᵀ of `e` (R the rotation by θ).
@inline function _ellipse_form(e::Ellipse)
    s, c = sincos(e.θ)
    ia², ib² = inv(e.a^2), inv(e.b^2)
    q₁₂ = s * c * (ia² - ib²)
    return SMatrix{2, 2}(c^2 * ia² + s^2 * ib², q₁₂, q₁₂, s^2 * ia² + c^2 * ib²)
end

# ∂(Q₁₁, Q₁₂, Q₂₂)/∂(S₁₁, S₁₂, S₂₂) of Q = exp(−2S), S = R·diag(log a, log b)·Rᵀ:
# the Daleckii–Krein derivative dQ = R·(Φ ∘ (Rᵀ·(−2dS)·R))·Rᵀ with Φ the divided
# differences of exp at μ = −2(log a, log b), its off-diagonal the limit eᵘ at a
# circle.
function _ellipse_form_jacobian(e::Ellipse)
    s, c = sincos(e.θ)
    R = SMatrix{2, 2}(c, s, -s, c)
    ia², ib² = inv(e.a^2), inv(e.b^2)
    δ = -2 * log(e.a / e.b)
    φ = iszero(δ) ? ib² : ib² * expm1(δ) / δ
    Φ = SMatrix{2, 2}(ia², φ, φ, ib²)
    E = (SMatrix{2, 2}(1.0, 0.0, 0.0, 0.0), SMatrix{2, 2}(0.0, 1.0, 1.0, 0.0), SMatrix{2, 2}(0.0, 0.0, 0.0, 1.0))
    cols = map(E) do Eₖ
        dQ = R * (Φ .* (R' * (-2 * Eₖ) * R)) * R'
        SVector(dQ[1, 1], dQ[1, 2], dQ[2, 2])
    end
    return hcat(cols...)
end

function retract_jacobian(ch::EllipseManifold{T}, v::AbstractVector = zero(SVector{5, T})) where {T}
    e = iszero(v) ? ch.ellipse : retract(ch, v)
    J = _ellipse_form_jacobian(e)
    z = zero(eltype(J)); o = one(eltype(J))
    return SMatrix{5, 5}(
        o, z, z, z, z,
        z, o, z, z, z,
        z, z, J[1, 1], J[2, 1], J[3, 1],
        z, z, J[1, 2], J[2, 2], J[3, 2],
        z, z, J[1, 3], J[2, 3], J[3, 3],
    )
end

function action_jacobian(ch::EllipseManifold, x::AbstractVector)
    _, _, G_θ = constraint_with_jacobians(ch.ellipse, SVector(x[1], x[2]))
    return transpose(G_θ) * retract_jacobian(ch)
end

local_coordinates(ch::EllipseManifold, e::Ellipse) = ellipse_log_shape(e) - ellipse_log_shape(ch.ellipse)

"""
    constraint_with_jacobians(e::Ellipse, x::StaticVector{2}) -> (g, G_x::SVector{2}, G_θ::SVector{5})

The point-on-ellipse constraint, the implicit conic `g = (x − c)ᵀ Q (x − c) − 1`
(`Q = R·diag(a⁻², b⁻²)·Rᵀ`, so `g = x̃ᵀ C x̃` for the ellipse's conic `C`
scaled to `−1` at the centre), with `G_x = 2Q(x − c)` and
`G_θ = ∂g/∂(c₁, c₂, Q₁₁, Q₁₂, Q₂₂) = (−2Q(x − c), d₁², 2d₁d₂, d₂²)`,
`d = x − c`, the ellipse's ambient coordinates ([`EllipseManifold`](@ref)'s).
Its Sampson distance `g/‖G_x‖` is the geometric distance to the ellipse to
first order, and at a circle of radius `r` the circle's `(‖d‖² − r²)/(2‖d‖)`
(VisualGeometryToolkitCore's `constraint_with_jacobians(::Circle, x)`). A
point at the centre has `G_x = 0` — the one Sampson degeneracy
(VisualGeometryToolkitCore's `whiten`).
"""
@inline function constraint_with_jacobians(e::Ellipse, x::StaticVector{2})
    Q = _ellipse_form(e)
    d = SVector(x[1] - e.center[1], x[2] - e.center[2])
    Qd = Q * d
    g = dot(d, Qd) - 1
    return g, 2 * Qd, SVector(-2 * Qd[1], -2 * Qd[2], d[1]^2, 2 * d[1] * d[2], d[2]^2)
end
minimal_manifold(e::Ellipse) = EllipseManifold(e)
