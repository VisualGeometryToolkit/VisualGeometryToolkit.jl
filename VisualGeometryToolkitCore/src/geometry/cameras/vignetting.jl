# =============================================================================
# Vignetting — a camera's relative illumination V: sensor → ℝ, a B-spline in its coefficients
# =============================================================================
#
# Two fields, both LINEAR in their coefficients (`coefficient_jacobian(v, u) ⋅ vec(control_points(v)) == v(u)`) and both
# evaluated by the stack's one isbits Cox–de Boor basis (`bspline_basis`) on clamped uniform knots (`clamped_knots`):
#
#   TensorVignetting   V(u) = Σᵢ Σⱼ Nᵢ,ₚ(u₁) Mⱼ,ₚ(u₂) Pᵢⱼ                 a tensor surface over a box of the sensor
#   RadialVignetting   V(u) = Σᵢ Nᵢ,ₚ(‖u − c‖ / r) Pᵢ                     a profile in the radius about a centre
#
# Positions are sensor points in COLMAP pixels; outside the knots the basis is clamped (a constant continuation). Both are
# isbits (kernel-lawful in `pointwise!`). The estimators that fit them are their owners' (a radiometric calibration, a
# panorama's self-calibration); the model, its basis and the regular subspace of a radial profile are here.
# =============================================================================

"""
    AbstractVignetting

A vignetting field `V` on the sensor, linear in its control points:
`coefficient_jacobian(v, u) ⋅ vec(control_points(v)) == v(u)`. Callable,
`v(u)` at a sensor point `u` (COLMAP pixels), and `v(u, f)` with the point's
ray `f` (not read: the form a panorama's render takes). Implementations:
[`TensorVignetting`](@ref) and [`RadialVignetting`](@ref).
"""
abstract type AbstractVignetting end

# -----------------------------------------------------------------------------
# The tensor surface
# -----------------------------------------------------------------------------

"""
    TensorVignetting(n, box::Rect2; degree = 3, T = Float64)

The tensor-product B-spline vignetting surface of `n = (nx, ny)` control
points over the sensor box `box` (COLMAP pixels): degree `degree` along both
axes (or a pair), clamped uniform knots ([`clamped_knots`](@ref)) over the
box's two ranges, control points `P ≡ 1` (the partition of unity: `V ≡ 1`).
`V(u) = Σᵢⱼ Nᵢ(u₁) Mⱼ(u₂) Pᵢⱼ`, a point outside the box clamped onto it.
"""
struct TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ} <: AbstractVignetting
    knots_x::SVector{KX, T}
    knots_y::SVector{KY, T}
    P::SMatrix{NX, NY, T, NXY}
end
function TensorVignetting(n::NTuple{2, Integer}, box::GeometryBasics.HyperRectangle{2}; degree = 3, T = Float64)
    p, q = degree isa Integer ? (Int(degree), Int(degree)) : (Int(degree[1]), Int(degree[2]))
    o = GeometryBasics.origin(box); w = GeometryBasics.widths(box)
    kx = clamped_knots(n[1], p, o[1], o[1] + w[1])
    ky = clamped_knots(n[2], q, o[2], o[2] + w[2])
    NX, NY = Int(n[1]), Int(n[2])
    return TensorVignetting{T, NX, NY, length(kx), length(ky), NX * NY, p, q}(SVector{length(kx), T}(kx), SVector{length(ky), T}(ky),
                                                                               ones(SMatrix{NX, NY, T, NX * NY}))
end

@inline function (v::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ})(u::StaticVector{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    Nx = bspline_basis(v.knots_x, Val(DP), u[1])
    My = bspline_basis(v.knots_y, Val(DQ), u[2])
    return dot(Nx, v.P * My)
end

"""
    coefficient_jacobian(v::AbstractVignetting, u) -> SVector

`∂V(u)/∂vec(P)`: the basis functions at the sensor point `u`, ordered like
`vec(control_points(v))` (column-major; for a tensor surface
`Nᵢ(u₁) Mⱼ(u₂)` at `i + (j − 1)·nx`), so `V(u) = coefficient_jacobian(v, u) ⋅ vec(P)`.
V is linear in `P`: the Jacobian does not depend on `P`.
"""
@inline function coefficient_jacobian(v::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, u::StaticVector{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    Nx = bspline_basis(v.knots_x, Val(DP), u[1])
    My = bspline_basis(v.knots_y, Val(DQ), u[2])
    return SVector{NX * NY}(ntuple(k -> Nx[(k - 1) % NX + 1] * My[(k - 1) ÷ NX + 1], Val(NX * NY)))
end

# The first of the p + 1 consecutive basis functions that can be nonzero at a point: the first nonzero one, kept p + 1 from the
# end (at a knot one of the p + 1 is zero). Kernel-lawful (a loop over the static basis).
@inline function _first_active(B::SVector{N}, p::Integer) where {N}
    i0 = 0
    @inbounds for i in 1:N
        (i0 == 0 && !iszero(B[i])) && (i0 = i)
    end
    return clamp(i0, 1, N - p)
end

"""
    active_coefficients(v::AbstractVignetting, u) -> (idx::NTuple, w::NTuple)

The control points whose basis functions can be nonzero at the sensor point
`u` and their values there: `(p + 1)` of a radial profile, `(p + 1)(q + 1)`
of a tensor surface (linear indices of `vec(control_points(v))`), the entries
of [`coefficient_jacobian`](@ref)`(v, u)` at `idx`, every other entry zero.
The sparse form a scatter-add over many points reads. Kernel-lawful.
"""
@inline function active_coefficients(v::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, u::StaticVector{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    Nx = bspline_basis(v.knots_x, Val(DP), u[1])
    My = bspline_basis(v.knots_y, Val(DQ), u[2])
    ix = _first_active(Nx, DP); iy = _first_active(My, DQ)
    idx = ntuple(k -> (iy + (k - 1) ÷ (DP + 1) - 1) * NX + ix + (k - 1) % (DP + 1), Val((DP + 1) * (DQ + 1)))
    w = ntuple(k -> Nx[ix + (k - 1) % (DP + 1)] * My[iy + (k - 1) ÷ (DP + 1)], Val((DP + 1) * (DQ + 1)))
    return idx, w
end

"""
    control_points(v::AbstractVignetting) -> SMatrix

The control points: `nx × ny` for a tensor surface, `N × 1` for a radial
profile, so `vec(control_points(v))` is the coefficient vector of either field
(the order of [`coefficient_jacobian`](@ref)).
"""
control_points(v::TensorVignetting) = v.P
"""
    n_control_points(v::AbstractVignetting) -> Int

The number of control points, `length(vec(control_points(v)))`.
"""
n_control_points(::TensorVignetting{T, NX, NY}) where {T, NX, NY} = NX * NY
"""
    knot_vectors(v::AbstractVignetting) -> Tuple

The clamped knot vectors, one per axis: `(knots_x, knots_y)` for a tensor
surface (camera pixels), `(knots,)` over `[0, 1]` for a radial profile.
"""
knot_vectors(v::TensorVignetting) = (v.knots_x, v.knots_y)
"""
    spline_degree(v::AbstractVignetting) -> Tuple

The B-spline degree per axis: `(p, q)` for a tensor surface, `(p,)` for a radial profile.
"""
spline_degree(::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}) where {T, NX, NY, KX, KY, NXY, DP, DQ} = (DP, DQ)
"""
    field_center(v::AbstractVignetting) -> SVector{2}

The field's centre (COLMAP pixels): the middle of a tensor surface's box, a
radial profile's centre of symmetry.
"""
field_center(v::TensorVignetting) = SVector((first(v.knots_x) + last(v.knots_x)) / 2, (first(v.knots_y) + last(v.knots_y)) / 2)

"""
    with_control_points(v::AbstractVignetting, P) -> AbstractVignetting

The same field (knots, degree, centre) with the control points `P`: a matrix
shaped like `control_points(v)` or its `vec`; the element type is `P`'s.
"""
function with_control_points(v::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, P::AbstractMatrix{S}) where {T, NX, NY, KX, KY, NXY, DP, DQ, S}
    return TensorVignetting{S, NX, NY, KX, KY, NXY, DP, DQ}(v.knots_x, v.knots_y, SMatrix{NX, NY, S, NXY}(P))
end
with_control_points(v::TensorVignetting{T, NX, NY}, P::AbstractVector) where {T, NX, NY} =
    with_control_points(v, reshape(collect(P), NX, NY))

# `T` any real type: a Dual for a device Jacobian, as well as a float.
convert_eltype(::Type{T}, v::TensorVignetting{S, NX, NY, KX, KY, NXY, DP, DQ}) where {T <: Real, S, NX, NY, KX, KY, NXY, DP, DQ} =
    TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}(SVector{KX, T}(v.knots_x), SVector{KY, T}(v.knots_y), SMatrix{NX, NY, T, NXY}(v.P))

# The surface's control points as a lattice of the support protocol (support.jl): control point (i, j) influences the knot span
# [kxᵢ, kxᵢ₊ₚ₊₁] × [kyⱼ, kyⱼ₊q₊₁], so `free_nodes` / `node_mask` over an imaged region give the points the region constrains and
# `BoundaryRule` the dependent ones.
support_trait(::Type{<:TensorVignetting}) = LocalSupport()
nodes(::TensorVignetting{T, NX, NY}) where {T, NX, NY} = CartesianIndices((NX, NY))
function support(v::TensorVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, i::CartesianIndex{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    x0, x1 = Float64(v.knots_x[i[1]]), Float64(v.knots_x[i[1] + DP + 1])
    y0, y1 = Float64(v.knots_y[i[2]]), Float64(v.knots_y[i[2] + DQ + 1])
    return Rect2{Float64}(x0, y0, x1 - x0, y1 - y0)
end

# -----------------------------------------------------------------------------
# The radial profile
# -----------------------------------------------------------------------------

"""
    RadialVignetting(n, center, radius; degree = 3, T = Float64)

The radial B-spline vignetting profile: `n` control points of degree `degree`
on clamped uniform knots over the normalized radius `ρ = ‖u − center‖ / radius
∈ [0, 1]` (`center` and `radius` in COLMAP pixels), `P ≡ 1` (`V ≡ 1`).
`V(u) = Σᵢ Nᵢ(ρ(u)) Pᵢ`, constant beyond `radius`. Radially symmetric by
construction: a field about one centre. [`pole_regular_basis`](@ref) is the
subspace of the profiles that are smooth at the centre.
"""
struct RadialVignetting{T, N, K, DP} <: AbstractVignetting
    center::SVector{2, T}
    radius::T
    knots::SVector{K, T}
    P::SVector{N, T}
end
function RadialVignetting(n::Integer, center, radius::Real; degree::Integer = 3, T = Float64)
    knots = clamped_knots(n, degree, 0.0, 1.0)
    return RadialVignetting{T, Int(n), length(knots), Int(degree)}(SVector{2, T}(center[1], center[2]), T(radius),
                                                                   SVector{length(knots), T}(knots), ones(SVector{Int(n), T}))
end

"""
    radial_coordinate(v::RadialVignetting, u) -> ρ

The normalized radius `‖u − center‖ / radius` of the sensor point `u` (the profile's argument).
"""
@inline function radial_coordinate(v::RadialVignetting, u::StaticVector{2})
    dx = u[1] - v.center[1]
    dy = u[2] - v.center[2]
    return sqrt(dx * dx + dy * dy) / v.radius
end

@inline coefficient_jacobian(v::RadialVignetting{T, N, K, DP}, u::StaticVector{2}) where {T, N, K, DP} =
    bspline_basis(v.knots, Val(DP), radial_coordinate(v, u))
@inline (v::RadialVignetting)(u::StaticVector{2}) = dot(coefficient_jacobian(v, u), v.P)
# A vignetting at a sensor point and its ray, `v(u, f)`, the form a panorama's render and sampling take (a field that reads the
# ray's direction is another callable): a field on the sensor reads `u` only.
@inline (v::AbstractVignetting)(u, f) = v(u)
@inline function active_coefficients(v::RadialVignetting{T, N, K, DP}, u::StaticVector{2}) where {T, N, K, DP}
    B = coefficient_jacobian(v, u)
    i0 = _first_active(B, DP)
    return ntuple(k -> i0 + k - 1, Val(DP + 1)), ntuple(k -> B[i0 + k - 1], Val(DP + 1))
end

control_points(v::RadialVignetting{T, N}) where {T, N} = SMatrix{N, 1, T, N}(v.P)
n_control_points(::RadialVignetting{T, N}) where {T, N} = N
knot_vectors(v::RadialVignetting) = (v.knots,)
spline_degree(::RadialVignetting{T, N, K, DP}) where {T, N, K, DP} = (DP,)
field_center(v::RadialVignetting) = v.center
"""The radius (COLMAP pixels) at which a radial profile's normalized radius is 1."""
field_radius(v::RadialVignetting) = v.radius

with_control_points(v::RadialVignetting{T, N, K, DP}, P::AbstractVector{S}) where {T, N, K, DP, S} =
    RadialVignetting{S, N, K, DP}(v.center, v.radius, v.knots, SVector{N, S}(P))
with_control_points(v::RadialVignetting, P::AbstractMatrix) = with_control_points(v, vec(collect(P)))

convert_eltype(::Type{T}, v::RadialVignetting{S, N, K, DP}) where {T <: Real, S, N, K, DP} =
    RadialVignetting{T, N, K, DP}(SVector{2, T}(v.center), T(v.radius), SVector{K, T}(v.knots), SVector{N, T}(v.P))

"""
    pole_regular_basis(v::RadialVignetting) -> Matrix{Float64}

A basis `Z` (`N × (N − 1)`) of the radial profiles that are smooth at the
centre, `vec(P) = vec(P₀) + Z y`: the first two control points tied,
`P₁ = P₂`. A field of revolution `V(‖u − c‖)` is differentiable at its centre
only when its generatrix meets the axis at a right angle, `V′(0) = 0`
(Piegl & Tiller 1997, §8.5); on clamped knots `V′(0) = p/k_{p+2} · (P₂ − P₁)`
(their eq. 3.7), so the tie is that condition, and the field is then as smooth
at the centre as at its knots. Column 1 moves `P₁` and `P₂` together, column
`j > 1` moves `P_{j+1}`.
"""
function pole_regular_basis(v::RadialVignetting)
    N = n_control_points(v)
    return vcat([1.0 zeros(1, N - 2)], Matrix{Float64}(LinearAlgebra.I, N - 1, N - 1))
end

# -----------------------------------------------------------------------------
# The log field: a spline of log V
# -----------------------------------------------------------------------------

"""
    LogVignetting(spline::AbstractVignetting)

A vignetting whose B-spline is log V: `V(u) = exp(s(u))`, `s` the
[`TensorVignetting`](@ref) or [`RadialVignetting`](@ref) `spline` read as the
logarithm (its control points are log V's). Positive by construction, and linear
in its control points in log, so a fit in log V is a linear least squares
(PanoramaStitching's self-calibration). Everything but the exponential is the
spline's own: its knots, degree, centre, support, [`pole_regular_basis`](@ref)
(`V′(0) = 0` exactly when `s′(0) = 0`) and [`log_spline`](@ref). Its
[`coefficient_jacobian`](@ref) is `∂V/∂P = V·∂s/∂P`, and
[`active_coefficients`](@ref) that row's nonzeros (`log_spline(v)`'s are
∂ log V/∂P). Callable, isbits: it runs in `pointwise!` on a device.
"""
struct LogVignetting{S <: AbstractVignetting} <: AbstractVignetting
    spline::S
end

"""
    log_spline(v::LogVignetting) -> AbstractVignetting

The spline of log V: `log(v(u)) == log_spline(v)(u)`.
"""
log_spline(v::LogVignetting) = v.spline

@inline (v::LogVignetting)(u::StaticVector{2}) = exp(v.spline(u))
@inline coefficient_jacobian(v::LogVignetting, u::StaticVector{2}) = v(u) .* coefficient_jacobian(v.spline, u)
@inline function active_coefficients(v::LogVignetting, u::StaticVector{2})
    idx, w = active_coefficients(v.spline, u)
    V = v(u)
    return idx, map(x -> V * x, w)
end
control_points(v::LogVignetting) = control_points(v.spline)
n_control_points(v::LogVignetting) = n_control_points(v.spline)
knot_vectors(v::LogVignetting) = knot_vectors(v.spline)
spline_degree(v::LogVignetting) = spline_degree(v.spline)
field_center(v::LogVignetting) = field_center(v.spline)
field_radius(v::LogVignetting{<:RadialVignetting}) = field_radius(v.spline)
radial_coordinate(v::LogVignetting{<:RadialVignetting}, u::StaticVector{2}) = radial_coordinate(v.spline, u)
pole_regular_basis(v::LogVignetting{<:RadialVignetting}) = pole_regular_basis(v.spline)
with_control_points(v::LogVignetting, P) = LogVignetting(with_control_points(v.spline, P))
convert_eltype(::Type{T}, v::LogVignetting) where {T <: Real} = LogVignetting(convert_eltype(T, v.spline))
support_trait(::Type{LogVignetting{S}}) where {S} = support_trait(S)
nodes(v::LogVignetting) = nodes(v.spline)
support(v::LogVignetting, i::CartesianIndex) = support(v.spline, i)
