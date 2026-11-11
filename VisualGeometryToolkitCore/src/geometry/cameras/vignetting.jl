# =============================================================================
# Vignetting — a camera's relative illumination V: sensor → ℝ, a B-spline in its coefficients
# =============================================================================
#
# Two fields, both LINEAR in their coefficients (`coefficient_jacobian(v, u) ⋅ vec(control_points(v)) == v(u)`) and both
# evaluated by the stack's one isbits Cox–de Boor basis (`bspline_basis`) on clamped uniform knots (`clamped_knots`):
#
#   TensorProductVignetting   V(u) = Σᵢ Σⱼ Nᵢ,ₚ(u₁) Mⱼ,ₚ(u₂) Pᵢⱼ    a tensor-product B-spline surface over a box of the sensor
#   RadialVignetting          V(u) = Σᵢ Nᵢ,ₚ(ρ(‖u − c‖)) Pᵢ        a profile in the radius about a centre, on a chart of the
#                                                                    distance d: linear ρ = d / r or projective ρ = d / (r + d)
#
# "Tensor product" names how the 2-D basis is built, Nᵢ(u₁)·Mⱼ(u₂) from two 1-D bases (de Boor); V itself is a scalar
# field on the sensor, not a tensor.
#
# Positions are sensor points in COLMAP pixels; outside the knots the basis is clamped (a constant continuation; the
# projective chart never leaves its knots). Both are
# isbits (kernel-lawful in `pointwise!`). The estimators that fit them are their owners' (a radiometric calibration, a
# panorama's self-calibration); the model, its basis and the regular subspace of a radial profile are here.
# =============================================================================

"""
    AbstractVignetting

A vignetting field `V` on the sensor, linear in its control points:
`coefficient_jacobian(v, u) ⋅ vec(control_points(v)) == v(u)`. Callable,
`v(u)` at a sensor point `u` (COLMAP pixels), and `v(u, f)` with the point's
ray `f` (not read: the form a panorama's render takes). Implementations:
[`TensorProductVignetting`](@ref) and [`RadialVignetting`](@ref).
"""
abstract type AbstractVignetting end

# -----------------------------------------------------------------------------
# The tensor-product surface
# -----------------------------------------------------------------------------

"""
    TensorProductVignetting(n, box::Rect2; degree = 3, T = Float64)

The tensor-product B-spline vignetting surface of `n = (nx, ny)` control
points over the sensor box `box` (COLMAP pixels): degree `degree` along both
axes (or a pair), clamped uniform knots ([`clamped_knots`](@ref)) over the
box's two ranges, control points `P ≡ 1` (the partition of unity: `V ≡ 1`).
`V(u) = Σᵢⱼ Nᵢ(u₁) Mⱼ(u₂) Pᵢⱼ`, a point outside the box clamped onto it.
"""
struct TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ} <: AbstractVignetting
    knots_x::SVector{KX, T}
    knots_y::SVector{KY, T}
    P::SMatrix{NX, NY, T, NXY}
end
function TensorProductVignetting(n::NTuple{2, Integer}, box::GeometryBasics.HyperRectangle{2}; degree = 3, T = Float64)
    p, q = degree isa Integer ? (Int(degree), Int(degree)) : (Int(degree[1]), Int(degree[2]))
    o = GeometryBasics.origin(box); w = GeometryBasics.widths(box)
    kx = clamped_knots(n[1], p, o[1], o[1] + w[1])
    ky = clamped_knots(n[2], q, o[2], o[2] + w[2])
    NX, NY = Int(n[1]), Int(n[2])
    return TensorProductVignetting{T, NX, NY, length(kx), length(ky), NX * NY, p, q}(SVector{length(kx), T}(kx), SVector{length(ky), T}(ky),
                                                                               ones(SMatrix{NX, NY, T, NX * NY}))
end

"""
    TensorProductVignetting(knots_x, knots_y, P; degree)

The surface on the given clamped knot vectors (COLMAP pixels; `degree`, or a
pair, the B-spline degree per axis) with the `nx × ny` control points `P`:
`length(knots_x) == nx + p + 1`, the knots nondecreasing and clamped (each end
repeated `p + 1` times). The knots are taken as they are — what a file's
reader needs to rebuild a surface exactly (`knot_vectors`, `spline_degree`,
`control_points` give them back).
"""
function TensorProductVignetting(knots_x::AbstractVector, knots_y::AbstractVector, P::AbstractMatrix; degree)
    p, q = degree isa Integer ? (Int(degree), Int(degree)) : (Int(degree[1]), Int(degree[2]))
    NX, NY = size(P)
    for (k, n, d, ax) in ((knots_x, NX, p, "x"), (knots_y, NY, q, "y"))
        length(k) == n + d + 1 || throw(ArgumentError("TensorProductVignetting: $(length(k)) knots along $(ax) for $(n) control points of degree $(d) (need $(n + d + 1))"))
        (issorted(k) && all(==(first(k)), k[1:(d + 1)]) && all(==(last(k)), k[(end - d):end])) ||
            throw(ArgumentError("TensorProductVignetting: the knots along $(ax) are not a clamped nondecreasing vector of degree $(d)"))
    end
    T = promote_type(eltype(knots_x), eltype(knots_y), eltype(P))
    KX, KY = length(knots_x), length(knots_y)
    return TensorProductVignetting{T, NX, NY, KX, KY, NX * NY, p, q}(SVector{KX, T}(knots_x), SVector{KY, T}(knots_y), SMatrix{NX, NY, T, NX * NY}(P))
end

# The two axis bases at a sensor point: every evaluation of the surface starts from them.
@inline _axis_bases(v::TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, u::StaticVector{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ} =
    (bspline_basis(v.knots_x, Val(DP), u[1]), bspline_basis(v.knots_y, Val(DQ), u[2]))
@inline _surface_value(v::TensorProductVignetting, Nx, My) = dot(Nx, v.P * My)
@inline _surface_jacobian(::TensorProductVignetting{T, NX, NY}, Nx, My) where {T, NX, NY} =
    SVector{NX * NY}(ntuple(k -> Nx[(k - 1) % NX + 1] * My[(k - 1) ÷ NX + 1], Val(NX * NY)))
@inline function _surface_active(::TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, Nx, My) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    ix = _first_active(Nx, DP); iy = _first_active(My, DQ)
    idx = ntuple(k -> (iy + (k - 1) ÷ (DP + 1) - 1) * NX + ix + (k - 1) % (DP + 1), Val((DP + 1) * (DQ + 1)))
    w = ntuple(k -> Nx[ix + (k - 1) % (DP + 1)] * My[iy + (k - 1) ÷ (DP + 1)], Val((DP + 1) * (DQ + 1)))
    return idx, w
end

@inline (v::TensorProductVignetting)(u::StaticVector{2}) = _surface_value(v, _axis_bases(v, u)...)

"""
    coefficient_jacobian(v::AbstractVignetting, u) -> SVector

`∂V(u)/∂vec(P)`: the basis functions at the sensor point `u`, ordered like
`vec(control_points(v))` (column-major; for a tensor-product surface
`Nᵢ(u₁) Mⱼ(u₂)` at `i + (j − 1)·nx`), so `V(u) = coefficient_jacobian(v, u) ⋅ vec(P)`.
V is linear in `P`: the Jacobian does not depend on `P`.
"""
@inline coefficient_jacobian(v::TensorProductVignetting, u::StaticVector{2}) = _surface_jacobian(v, _axis_bases(v, u)...)

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
of a tensor-product surface (linear indices of `vec(control_points(v))`), the entries
of [`coefficient_jacobian`](@ref)`(v, u)` at `idx`, every other entry zero.
The sparse form a scatter-add over many points reads. Kernel-lawful.
"""
@inline active_coefficients(v::TensorProductVignetting, u::StaticVector{2}) = _surface_active(v, _axis_bases(v, u)...)

# A field's value with its coefficient row (or that row's nonzeros) from ONE basis evaluation, by the same arithmetic as
# `v(u)`, `coefficient_jacobian` and `active_coefficients`, so bit for bit theirs: what a log field's chain rule reads.
@inline function _field_value_and_row(v::TensorProductVignetting, u::StaticVector{2})
    Nx, My = _axis_bases(v, u)
    return _surface_value(v, Nx, My), _surface_jacobian(v, Nx, My)
end
@inline function _field_value_and_active(v::TensorProductVignetting, u::StaticVector{2})
    Nx, My = _axis_bases(v, u)
    return _surface_value(v, Nx, My), _surface_active(v, Nx, My)
end

"""
    control_points(v::AbstractVignetting) -> SMatrix

The control points: `nx × ny` for a tensor-product surface, `N × 1` for a radial
profile, so `vec(control_points(v))` is the coefficient vector of either field
(the order of [`coefficient_jacobian`](@ref)).
"""
control_points(v::TensorProductVignetting) = v.P
"""
    n_control_points(v::AbstractVignetting) -> Int

The number of control points, `length(vec(control_points(v)))`.
"""
n_control_points(::TensorProductVignetting{T, NX, NY}) where {T, NX, NY} = NX * NY
"""
    knot_vectors(v::AbstractVignetting) -> Tuple

The clamped knot vectors, one per axis: `(knots_x, knots_y)` for a tensor
surface (camera pixels), `(knots,)` over `[0, 1]` for a radial profile.
"""
knot_vectors(v::TensorProductVignetting) = (v.knots_x, v.knots_y)
"""
    spline_degree(v::AbstractVignetting) -> Tuple

The B-spline degree per axis: `(p, q)` for a tensor-product surface, `(p,)` for a radial profile.
"""
spline_degree(::TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}) where {T, NX, NY, KX, KY, NXY, DP, DQ} = (DP, DQ)
"""
    center(v::AbstractVignetting) -> SVector{2}

The field's centre (COLMAP pixels), a method of Core's one `center` generic:
the middle of a tensor-product surface's box, a radial profile's centre of symmetry,
a log field's spline's.
"""
center(v::TensorProductVignetting) = SVector((first(v.knots_x) + last(v.knots_x)) / 2, (first(v.knots_y) + last(v.knots_y)) / 2)

"""
    with_control_points(v::AbstractVignetting, P) -> AbstractVignetting

The same field (knots, degree, centre) with the control points `P`: a matrix
shaped like `control_points(v)` or its `vec`. The element type is
`promote_type` of the field's and `P`'s, so the knots, centre and radius are
never narrowed (Int or Float32 control points on a Float64 field give a
Float64 field; Duals give a Dual field, its knots exact).
"""
function with_control_points(v::TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, P::AbstractMatrix{S}) where {T, NX, NY, KX, KY, NXY, DP, DQ, S}
    R = promote_type(T, S)
    return TensorProductVignetting{R, NX, NY, KX, KY, NXY, DP, DQ}(SVector{KX, R}(v.knots_x), SVector{KY, R}(v.knots_y), SMatrix{NX, NY, R, NXY}(P))
end
with_control_points(v::TensorProductVignetting{T, NX, NY}, P::AbstractVector) where {T, NX, NY} =
    with_control_points(v, reshape(collect(P), NX, NY))

# `T` any real type: a Dual for a device Jacobian, as well as a float.
convert_eltype(::Type{T}, v::TensorProductVignetting{S, NX, NY, KX, KY, NXY, DP, DQ}) where {T <: Real, S, NX, NY, KX, KY, NXY, DP, DQ} =
    TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}(SVector{KX, T}(v.knots_x), SVector{KY, T}(v.knots_y), SMatrix{NX, NY, T, NXY}(v.P))

# The surface's control points as a lattice of the support protocol (support.jl): control point (i, j) influences the knot span
# [kxᵢ, kxᵢ₊ₚ₊₁] × [kyⱼ, kyⱼ₊q₊₁], so `free_nodes` / `node_mask` over an imaged region give the points the region constrains and
# `BoundaryRule` the dependent ones.
support_trait(::Type{<:TensorProductVignetting}) = LocalSupport()
nodes(::TensorProductVignetting{T, NX, NY}) where {T, NX, NY} = CartesianIndices((NX, NY))
function support(v::TensorProductVignetting{T, NX, NY, KX, KY, NXY, DP, DQ}, i::CartesianIndex{2}) where {T, NX, NY, KX, KY, NXY, DP, DQ}
    # the knots' values under any Dual layer (base/duals.jl): a support is a region of the sensor, not a derivative
    x0, x1 = Float64(_primal(v.knots_x[i[1]])), Float64(_primal(v.knots_x[i[1] + DP + 1]))
    y0, y1 = Float64(_primal(v.knots_y[i[2]])), Float64(_primal(v.knots_y[i[2] + DQ + 1]))
    return Rect2{Float64}(x0, y0, x1 - x0, y1 - y0)
end

# -----------------------------------------------------------------------------
# The radial profile
# -----------------------------------------------------------------------------

"""The charts of a radial profile's normalized radius: `:linear` (`ρ = d / radius`) and `:projective` (`ρ = d / (radius + d)`, [`projective_chart`](@ref))."""
const RADIAL_CHARTS = (:linear, :projective)

"""
    projective_chart(r) -> Transformation

The projective chart of the distance `d` (a 1-vector) at the radius `r`: `ρ = d / (r + d)`, Core's 1-D `HomographyMap([1 0; 1 r])`
dehomogenized by `PerspectiveMap` (`ρ(r) = ½`, `ρ(∞) = 1`). Isbits: it runs in a kernel.
"""
@inline projective_chart(r::T) where {T} = CoordinateTransformations.PerspectiveMap() ∘ HomographyMap(SMatrix{2, 2, T, 4}(one(T), one(T), zero(T), r))

"""
    projective_knots(n, p) -> SVector{n + p + 1, Float64}

The knots of a radial profile on the projective chart of `n` control points of degree `p`: `m = n − p − 1` spans uniform in the
distance up to the radius — knots at `d = k·r/m`, which the chart maps to `k / (m + k)`, `k = 0 … m` (so ½ at the radius) — and one
tail span `[½, 1)` for every point beyond, clamped (`p + 1` knots at each end). Needs `n > p + 1`. With `n` one more than a linear
profile's count, the pixel spacing up to the radius is the linear profile's.
"""
function projective_knots(n::Integer, p::Integer)
    m = Int(n) - Int(p) - 1
    m >= 1 || throw(ArgumentError("projective_knots: need more than p + 1 = $(p + 1) control points, got $n"))
    χ = projective_chart(Float64(m))                                       # the chart at radius m: d = k ↦ k / (m + k)
    interior = [only(χ(SVector(Float64(k)))) for k in 1:m]                 # k = m is the rim, ρ = ½; then the tail span to 1
    return SVector{Int(n) + Int(p) + 1}(vcat(zeros(Int(p) + 1), interior, ones(Int(p) + 1)))
end

"""
    RadialVignetting(n, center, radius; degree = 3, T = Float64, chart = :linear)
    RadialVignetting(knots, center, radius, P; degree, chart = :linear)

The radial B-spline vignetting profile: `n` control points of degree `degree`
over a normalized radius `ρ ∈ [0, 1]` of the distance `d = ‖u − center‖`
(`center` and `radius` in COLMAP pixels, plain or `px`; a length is a
`DimensionError`, a radius that is not positive an `ArgumentError`), `P ≡ 1`
(`V ≡ 1`), `V(u) = Σᵢ Nᵢ(ρ(u)) Pᵢ`. The `chart` of `ρ` ([`RADIAL_CHARTS`](@ref)):

- `:linear`: `ρ = d / radius` on clamped uniform knots, `V` constant beyond `radius`;
- `:projective`: `ρ = d / (radius + d)` ([`projective_chart`](@ref)), `½` at
  `radius` and `→ 1` at infinity, so no sensor point is ever beyond the knots
  whatever the centre; on the knots [`projective_knots`](@ref)`(n, degree)`:
  uniform in the distance up to `radius` and one tail span beyond.

Radially symmetric by construction: a field about one centre.
[`pole_regular_basis`](@ref) is the subspace of the profiles that are smooth at
the centre (`V′(0) = 0` on either chart: `dρ/dd = 1 / radius` at the centre),
and `V(center) = P₁` exactly. The second form takes the clamped knot vector
itself (checked: `n + degree + 1` knots, nondecreasing, clamped, from 0 to 1)
and the control points: what a record is rebuilt from.
[`radial_chart`](@ref) is the chart, [`with_center`](@ref) moves the centre,
[`center_gradient`](@ref) is `∂V/∂center`.
"""
# The chart as a type parameter (isbits, a kernel specialises on it): `radial_chart` names it.
abstract type _RadialChart end
struct _LinearChart <: _RadialChart end
struct _ProjectiveChart <: _RadialChart end
_chart_type(chart::Symbol) = chart === :projective ? _ProjectiveChart : chart === :linear ? _LinearChart :
    throw(ArgumentError("RadialVignetting: chart = $(repr(chart)) is not one of $(join(repr.(RADIAL_CHARTS), ", "))"))
_chart_name(::Type{_LinearChart}) = :linear
_chart_name(::Type{_ProjectiveChart}) = :projective

struct RadialVignetting{T, N, K, DP, C <: _RadialChart} <: AbstractVignetting
    center::SVector{2, T}
    radius::T
    knots::SVector{K, T}
    P::SVector{N, T}
end
# the fields without a chart: the linear chart (the profile's form before the chart parameter)
RadialVignetting{T, N, K, DP}(center, radius, knots, P) where {T, N, K, DP} = RadialVignetting{T, N, K, DP, _LinearChart}(center, radius, knots, P)
function RadialVignetting(n::Integer, center, radius; degree::Integer = 3, T = Float64, chart::Symbol = :linear)
    C = _chart_type(chart)
    # pixel-sized: a plain number or a px quantity, a length refused (DimensionError; units.md)
    c1, c2, r = _strip_pixels(center[1]), _strip_pixels(center[2]), _strip_pixels(radius)
    r isa Real && isfinite(r) && r > 0 || throw(ArgumentError("RadialVignetting: the radius must be a positive finite number of pixels; got $(radius)"))
    knots = chart === :projective ? projective_knots(n, degree) : clamped_knots(n, degree, 0.0, 1.0)
    return RadialVignetting{T, Int(n), length(knots), Int(degree), C}(SVector{2, T}(c1, c2), T(r),
                                                                          SVector{length(knots), T}(knots), ones(SVector{Int(n), T}))
end
function RadialVignetting(knots::AbstractVector, center, radius, P::AbstractVector; degree::Integer, chart::Symbol = :linear)
    C = _chart_type(chart)
    N, p = length(P), Int(degree)
    length(knots) == N + p + 1 || throw(ArgumentError("RadialVignetting: $(length(knots)) knots for $N control points of degree $p (need $(N + p + 1))"))
    (issorted(knots) && all(==(first(knots)), knots[1:(p + 1)]) && all(==(last(knots)), knots[(end - p):end]) && first(knots) == 0 && last(knots) == 1) ||
        throw(ArgumentError("RadialVignetting: the knots are not a clamped nondecreasing vector of degree $p from 0 to 1"))
    r = float(_strip_pixels(radius))
    isfinite(r) && r > 0 || throw(ArgumentError("RadialVignetting: the radius must be a positive finite number of pixels; got $radius"))
    T = promote_type(typeof(r), eltype(knots), eltype(P)); K = length(knots)
    return RadialVignetting{T, N, K, p, C}(SVector{2, T}(_strip_pixels(center[1]), _strip_pixels(center[2])), T(r), SVector{K, T}(knots), SVector{N, T}(P))
end

"""
    radial_chart(v) -> Symbol

The chart of a radial profile's normalized radius: `:linear` or `:projective` ([`RADIAL_CHARTS`](@ref)); a log field's, its spline's.
"""
radial_chart(::RadialVignetting{T, N, K, DP, C}) where {T, N, K, DP, C <: _RadialChart} = _chart_name(C)

"""
    radial_coordinate(v::RadialVignetting, u) -> ρ

The normalized radius of the sensor point `u` (the profile's argument) on the profile's chart: `‖u − center‖ / radius` (linear) or
`‖u − center‖ / (radius + ‖u − center‖)` (projective).
"""
@inline function radial_coordinate(v::RadialVignetting{T, N, K, DP, C}, u::StaticVector{2}) where {T, N, K, DP, C <: _RadialChart}
    dx = u[1] - v.center[1]
    dy = u[2] - v.center[2]
    C === _ProjectiveChart && return only(projective_chart(v.radius)(SVector(sqrt(dx * dx + dy * dy))))   # a constant per type
    return sqrt(dx * dx + dy * dy) / v.radius
end

@inline coefficient_jacobian(v::RadialVignetting{T, N, K, DP}, u::StaticVector{2}) where {T, N, K, DP} =
    bspline_basis(v.knots, Val(DP), radial_coordinate(v, u))
@inline (v::RadialVignetting)(u::StaticVector{2}) = dot(coefficient_jacobian(v, u), v.P)
# A vignetting at a sensor point and its ray, `v(u, f)`, the form a panorama's render and sampling take (a field that reads the
# ray's direction is another callable): a field on the sensor reads `u` only.
@inline (v::AbstractVignetting)(u, f) = v(u)
@inline function _profile_active(::RadialVignetting{T, N, K, DP}, B) where {T, N, K, DP}
    i0 = _first_active(B, DP)
    return ntuple(k -> i0 + k - 1, Val(DP + 1)), ntuple(k -> B[i0 + k - 1], Val(DP + 1))
end
@inline active_coefficients(v::RadialVignetting, u::StaticVector{2}) = _profile_active(v, coefficient_jacobian(v, u))
@inline _field_value_and_row(v::RadialVignetting, u::StaticVector{2}) = (B = coefficient_jacobian(v, u); (dot(B, v.P), B))
@inline _field_value_and_active(v::RadialVignetting, u::StaticVector{2}) = (B = coefficient_jacobian(v, u); (dot(B, v.P), _profile_active(v, B)))

control_points(v::RadialVignetting{T, N}) where {T, N} = SMatrix{N, 1, T, N}(v.P)
n_control_points(::RadialVignetting{T, N}) where {T, N} = N
knot_vectors(v::RadialVignetting) = (v.knots,)
spline_degree(::RadialVignetting{T, N, K, DP}) where {T, N, K, DP} = (DP,)
center(v::RadialVignetting) = v.center
"""The radius (COLMAP pixels) at which a radial profile's normalized radius is 1 (linear chart) or ½ (projective chart)."""
field_radius(v::RadialVignetting) = v.radius

function with_control_points(v::RadialVignetting{T, N, K, DP, C}, P::AbstractVector{S}) where {T, N, K, DP, C <: _RadialChart, S}
    R = promote_type(T, S)
    return RadialVignetting{R, N, K, DP, C}(SVector{2, R}(v.center), R(v.radius), SVector{K, R}(v.knots), SVector{N, R}(P))
end
with_control_points(v::RadialVignetting, P::AbstractMatrix) = with_control_points(v, vec(collect(P)))

convert_eltype(::Type{T}, v::RadialVignetting{S, N, K, DP, C}) where {T <: Real, S, N, K, DP, C <: _RadialChart} =
    RadialVignetting{T, N, K, DP, C}(SVector{2, T}(v.center), T(v.radius), SVector{K, T}(v.knots), SVector{N, T}(v.P))

"""
    with_center(v::RadialVignetting, center) -> v′

The same radial profile (chart, knots, degree, radius, control points) about the centre `center` (COLMAP pixels): what a fit of the
centre steps. The element type is `promote_type` of the field's and the centre's (a Dual centre gives a Dual field, its knots exact).
"""
function with_center(v::RadialVignetting{T, N, K, DP, C}, center::StaticVector{2, S}) where {T, N, K, DP, C <: _RadialChart, S}
    R = promote_type(T, S)
    return RadialVignetting{R, N, K, DP, C}(SVector{2, R}(center), R(v.radius), SVector{K, R}(v.knots), SVector{N, R}(v.P))
end
with_center(v::RadialVignetting, center) = with_center(v, SVector{2}(center[1], center[2]))

struct _CenterTag end
"""
    center_gradient(v, u) -> SVector{2}

`∂V(u)/∂c` of a radial profile at its centre `c` (COLMAP pixels): the dipole `−V′(ρ) ∂ρ/∂d · (u − c)/d`, by a two-partial Dual
through [`with_center`](@ref) and the same basis evaluation as `v(u)` (isbits: it runs in a kernel); zero at the centre itself
(`V′(0) = 0` under the pole tie; the distance's derivative has no value there) and for a field without a centre parameter.
"""
@inline function center_gradient(v::RadialVignetting, u::StaticVector{2})
    T = eltype(control_points(v)); c = center(v)
    dx = T(u[1]) - c[1]; dy = T(u[2]) - c[2]
    (dx * dx + dy * dy) > 0 || return zero(SVector{2, T})
    d1 = ForwardDiff.Dual{_CenterTag}(c[1], one(T), zero(T)); d2 = ForwardDiff.Dual{_CenterTag}(c[2], zero(T), one(T))
    x = with_center(v, SVector(d1, d2))(SVector{2, T}(u[1], u[2]))
    return SVector{2, T}(ForwardDiff.partials(x, 1), ForwardDiff.partials(x, 2))
end
@inline center_gradient(v::AbstractVignetting, u::StaticVector{2}) = zero(SVector{2, eltype(control_points(v))})

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
[`TensorProductVignetting`](@ref) or [`RadialVignetting`](@ref) `spline` read as the
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
# Any other field: its value and row by its own verbs.
@inline _field_value_and_row(v::AbstractVignetting, u::StaticVector{2}) = (v(u), coefficient_jacobian(v, u))
@inline _field_value_and_active(v::AbstractVignetting, u::StaticVector{2}) = (v(u), active_coefficients(v, u))
# One basis evaluation each (`_field_value_and_row`, `_field_value_and_active`): bit for bit the two-evaluation form.
@inline function coefficient_jacobian(v::LogVignetting, u::StaticVector{2})
    s, J = _field_value_and_row(v.spline, u)
    return exp(s) .* J
end
@inline function active_coefficients(v::LogVignetting, u::StaticVector{2})
    s, (idx, w) = _field_value_and_active(v.spline, u)
    V = exp(s)
    return idx, map(x -> V * x, w)
end
control_points(v::LogVignetting) = control_points(v.spline)
n_control_points(v::LogVignetting) = n_control_points(v.spline)
knot_vectors(v::LogVignetting) = knot_vectors(v.spline)
spline_degree(v::LogVignetting) = spline_degree(v.spline)
center(v::LogVignetting) = center(v.spline)
field_radius(v::LogVignetting{<:RadialVignetting}) = field_radius(v.spline)
radial_coordinate(v::LogVignetting{<:RadialVignetting}, u::StaticVector{2}) = radial_coordinate(v.spline, u)
pole_regular_basis(v::LogVignetting{<:RadialVignetting}) = pole_regular_basis(v.spline)
radial_chart(v::LogVignetting{<:RadialVignetting}) = radial_chart(v.spline)
with_control_points(v::LogVignetting, P) = LogVignetting(with_control_points(v.spline, P))
convert_eltype(::Type{T}, v::LogVignetting) where {T <: Real} = LogVignetting(convert_eltype(T, v.spline))
support_trait(::Type{LogVignetting{S}}) where {S} = support_trait(S)
nodes(v::LogVignetting) = nodes(v.spline)
support(v::LogVignetting, i::CartesianIndex) = support(v.spline, i)

"""
    VIGNETTING_TAGS

The [`TagRegistry`](@ref) of the vignetting fields: `"tensor_product"`
([`TensorProductVignetting`](@ref)), `"radial"` ([`RadialVignetting`](@ref))
and `"log"` ([`LogVignetting`](@ref)), `snake_case` of the type's name with the
family's suffix `Vignetting` stripped — the `type` of VisualGeometryToolkitIO's
`vgio/vignetting/v1` body. RETIRED: `"tensor"`, the tensor-product surface's
tag before the type was named for its basis.
"""
const VIGNETTING_TAGS = TagRegistry(:vignetting, "Vignetting",
    "tensor_product" => TensorProductVignetting, "radial" => RadialVignetting, "log" => LogVignetting;
    retired = ("tensor" => "tensor_product",))

# -----------------------------------------------------------------------------
# The parameter protocol: a field's coefficients are its control points
# -----------------------------------------------------------------------------
#
# `parameter_vector(v)` is `vec(control_points(v))` — the order of
# `coefficient_jacobian` — named `c1, c2, …`, unbounded (a log field's control
# points are log gains; a plain field's gains are not bounded by the model),
# and `with_parameters(v, θ)` is `with_control_points(v, θ)`: knots, degree,
# centre and radius are the field's geometry, fixed. So a field is a
# `ParameterManifold` (an NLLSsolver variable through Core's adapter), and a
# linear restriction of it (`restrict(ch, Z)`) is a fit's control-point basis.
parameter_vector(v::AbstractVignetting) = (P = control_points(v); SVector{length(P), eltype(P)}(Tuple(P)))
parameter_names(v::AbstractVignetting) = ntuple(i -> Symbol(:c, i), Val(n_control_points(v)))
function with_parameters(v::AbstractVignetting, θ::AbstractVector)
    n = n_control_points(v)
    length(θ) == n || throw(ArgumentError("$(nameof(typeof(v))) takes $n parameters (its control points); got $(length(θ))"))
    return with_control_points(v, θ)
end
