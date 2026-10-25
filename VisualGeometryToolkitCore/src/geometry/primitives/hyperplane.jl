# =============================================================================
# Hyperplane - N-dimensional hyperplanes (lines in 2D, planes in 3D, etc.)
# =============================================================================
#
# Represents hyperplanes in M-dimensional Euclidean space using homogeneous
# coordinates: a₁x₁ + a₂x₂ + ... + aₘxₘ + d = 0
#
# Stored as normalized coefficients [a₁, ..., aₘ, d] where ||[a₁,...,aₘ]|| = 1
# This ensures:
#   - The normal vector is unit length
#   - The last coefficient d equals the signed distance from origin to hyperplane
#   - Point-to-hyperplane distance is simply dot(coeffs, [point..., 1])

using StaticArrays: StaticArrays, SVector, SMatrix, SOneTo, StaticVector
using StaticArrays.StaticArraysCore: StaticArray
using LinearAlgebra: norm, dot, cross, normalize
using GeometryBasics: GeometryBasics, Point, Point2, Point3, Vec2, Vec3, Rect2

# Import shared functions from parent to extend
import ..VisualGeometryToolkitCore: closest_point, distance, normal, offset, centroid

# =============================================================================
# Hyperplane Type
# =============================================================================

"""
    Hyperplane{M,N,T}

An M-dimensional hyperplane represented by normalized homogeneous coefficients.

The equation a₁x₁ + a₂x₂ + ... + aₘxₘ + d = 0 is stored as [a₁, ..., aₘ, d]
where the normal [a₁, ..., aₘ] is automatically normalized to unit length.

# Type Parameters
- `M`: Ambient space dimension (2 for lines, 3 for planes)
- `N`: Number of coefficients (always M+1)
- `T`: Coefficient type

# Properties
- `normal(h)`: Unit normal vector [a₁, ..., aₘ]
- `offset(h)`: Signed distance from origin (-d when normal is unit)

# Examples
```julia
# 2D line: x - y = 0 (diagonal through origin)
line = Hyperplane(SVector(1.0, -1.0, 0.0))
line = Line2D(1.0, -1.0, 0.0)

# 3D plane: z = 5 (horizontal plane at height 5)
plane = Hyperplane(SVector(0.0, 0.0, 1.0, -5.0))
plane = Plane3D(Point3(0,0,5), Vec3(0,0,1))

# Point-to-hyperplane signed distance
d = distance(line, Point2(1.0, 2.0))

# Closest point on hyperplane (orthogonal projection)
p_closest = closest_point(line, Point2(1.0, 2.0))
```
"""
struct Hyperplane{M, N, T}
    coeffs::SVector{N, T}

    # Inner constructor normalizes the normal part
    function Hyperplane{M, N, T}(v::AbstractVector) where {M, N, T}
        N == M + 1 || throw(ArgumentError("N must equal M+1 for Hyperplane{M,N,T}"))
        length(v) == N || throw(DimensionMismatch("Expected $N coefficients for Hyperplane{$M,$N}, got $(length(v))"))
        n_norm = norm(SVector{M, T}(v[1:M]))
        n_norm > 0 || throw(ArgumentError("Normal vector cannot be zero"))
        return new{M, N, T}(SVector{N, T}(v) / n_norm)
    end
end

# Convenience constructors - infer M from N
Hyperplane(v::StaticVector{N, T}) where {N, T} = Hyperplane{N - 1, N, T}(v)
Hyperplane{M}(v::StaticVector{N, T}) where {M, N, T} = Hyperplane{M, N, T}(v)
Hyperplane(h::Hyperplane) = h

# Vararg constructor
Hyperplane(a1::T, rest::Vararg{T, M}) where {M, T <: Real} = Hyperplane{M, M + 1, T}(SVector{M + 1, T}(a1, rest...))

# =============================================================================
# Basic Interface
# =============================================================================

"The normalized coefficients `(n..., d)` of a [`Hyperplane`](@ref): `n·x + d = 0`, `|n| = 1`."
coeffs(h::Hyperplane) = h.coeffs
Base.Tuple(h::Hyperplane) = Tuple(h.coeffs)
@inline Base.getindex(h::Hyperplane, i::Int) = h.coeffs[i]
Base.length(::Hyperplane{M, N}) where {M, N} = N
Base.size(::Hyperplane{M, N}) where {M, N} = (N,)
Base.iterate(h::Hyperplane, state = 1) = state > length(h) ? nothing : (h[state], state + 1)

# =============================================================================
# Type Aliases
# =============================================================================

"""2D line (hyperplane in 2D): ax + by + c = 0"""
const Line2D{T} = Hyperplane{2, 3, T}

"""3D plane (hyperplane in 3D): ax + by + cz + d = 0"""
const Plane3D{T} = Hyperplane{3, 4, T}

# Convenience aliases
# Explicit vararg constructors for type aliases
Line2D(a::T, b::T, c::T) where {T <: Real} = Line2D{T}(SVector{3, T}(a, b, c))
Line2D(a::Real, b::Real, c::Real) = Line2D(promote(a, b, c)...)
Plane3D(a::T, b::T, c::T, d::T) where {T <: Real} = Plane3D{T}(SVector{4, T}(a, b, c, d))
Plane3D(a::Real, b::Real, c::Real, d::Real) = Plane3D(promote(a, b, c, d)...)

# =============================================================================
# Accessors
# =============================================================================

"""
    normal(h::Hyperplane{M,N,T}) -> SVector{M,T}

Return the unit normal vector of the hyperplane.
"""
normal(h::Hyperplane{M, N, T}) where {M, N, T} = SVector{M, T}(h.coeffs[SOneTo(M)])

"""
    offset(h::Hyperplane) -> T

Return the signed distance from the origin to the hyperplane.
Positive if origin is on the positive side of the normal.
"""
offset(h::Hyperplane) = -h.coeffs[end]

# =============================================================================
# Geometric Operations
# =============================================================================

"""
    distance(hyperplane, point) -> T

Signed distance from `point` to `hyperplane`.
Positive if on the positive side of the normal.

# Example
```julia
line = Line2D(Point2(0.0, 0.0), Point2(1.0, 0.0))  # x-axis
distance(line, Point2(3.0, 5.0))  # 5.0
```
"""
function distance(h::Hyperplane{M, N, T}, p::StaticArray{Tuple{M}}) where {M, N, T}
    return dot(h.coeffs, SVector{N}(p..., one(eltype(p))))
end

distance(h::Hyperplane{M}, p::Point{M}) where {M} = distance(h, SVector{M}(p))

"""
    closest_point(hyperplane, p) -> Point{M}

Point on `hyperplane` closest to `p` (orthogonal projection).

# Example
```julia
line = Line2D(Point2(0.0, 0.0), Point2(1.0, 0.0))  # x-axis
closest_point(line, Point2(3.0, 5.0))  # Point2(3.0, 0.0)
```
"""
function closest_point(h::Hyperplane{M, N, T}, p::StaticArray{Tuple{M}}) where {M, N, T}
    d = distance(h, p)
    n = normal(h)
    return p - d * n
end

closest_point(h::Hyperplane{M}, p::Point{M}) where {M} = Point{M}(closest_point(h, SVector{M}(p))...)

"""
    side(h::Hyperplane, p) -> Int

Return which side of the hyperplane `p` is on:
- `+1` if on positive side (same direction as normal)
- `-1` if on negative side
- `0` if on the hyperplane (within tolerance)
"""
function side(h::Hyperplane{M, N}, p::Union{Point{M}, StaticArray{Tuple{M}}}; atol = 1.0e-10) where {M, N}
    d = distance(h, p)
    return abs(d) < atol ? 0 : (d > 0 ? 1 : -1)
end

"""
    on_hyperplane(h::Hyperplane, p; atol=1e-10) -> Bool

Check if point `p` lies on hyperplane `h` within tolerance.
"""
on_hyperplane(h::Hyperplane, p; atol = 1.0e-10) = abs(distance(h, p)) < atol

"""
    p ∈ h  or  in(p, h::Hyperplane)

Check if point `p` lies on hyperplane `h` (within tolerance 1e-10). The point
comes first here, unlike `on_hyperplane(h, p)`: `in` is Base's function and keeps
Base's `in(item, collection)` order.

For custom tolerance, use `on_hyperplane(h, p; atol=...)`.

# Examples
```julia
line = Line2D(1.0, 0.0, -1.0)  # x = 1
Point2(1.0, 5.0) ∈ line  # true
Point2(0.0, 0.0) ∈ line  # false
```
"""
Base.in(p::Union{Point{M}, StaticArray{Tuple{M}}}, h::Hyperplane{M}) where {M} = on_hyperplane(h, p)

# =============================================================================
# Line2D Specific Constructors and Operations
# =============================================================================

"""
    Line2D(ρ, θ) -> Line2D

Construct a 2D line in Hesse normal form (distance ρ from origin, angle θ).
The line equation is: x·cos(θ) + y·sin(θ) - ρ = 0
"""
function Line2D(ρ::Real, θ::Real)
    T = promote_type(typeof(ρ), typeof(θ))
    return Line2D{T}(SVector{3, T}(cos(θ), sin(θ), -ρ))
end

"""
    Line2D(p::Point2, q::Point2) -> Line2D

Construct a 2D line through two points.
"""
function Line2D(p::Point2, q::Point2)
    v = SVector{2}(q) - SVector{2}(p)
    # Normal perpendicular to direction (rotate 90°)
    n = SVector(-v[2], v[1])
    d = -dot(n, SVector{2}(p))
    T = promote_type(eltype(p), eltype(q))
    return Line2D{T}(SVector{3, T}(n..., d))
end

"""
    Line2D(p::Point2, n::Vec2) -> Line2D

Construct a 2D line from a point on the line and a normal vector.
"""
function Line2D(p::Point2{T1}, n::Vec2{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    nv = SVector{2, T}(n)
    d = -dot(nv, SVector{2, T}(p))
    return Line2D{T}(SVector{3, T}(nv..., d))
end

# Hesse normal form accessors for Line2D
"""Distance from origin in Hesse normal form."""
hesse_rho(l::Line2D) = offset(l)

"""Angle of normal in Hesse normal form."""
hesse_theta(l::Line2D) = atan(l[2], l[1])

"""Slope of line (dy/dx). Returns Inf for vertical lines."""
slope(l::Line2D) = -l[1] / l[2]

"""Y-intercept of line. Returns Inf for vertical lines."""
intercept(l::Line2D) = -l[3] / l[2]

# Property access for Line2D (for compatibility with vgtk0)
function Base.getproperty(l::Line2D, s::Symbol)
    if s === :ρ || s === :rho
        return hesse_rho(l)
    elseif s === :θ || s === :theta
        return hesse_theta(l)
    elseif s === :slope
        return slope(l)
    elseif s === :intercept
        return intercept(l)
    else
        return getfield(l, s)
    end
end

Base.propertynames(::Line2D) = (:coeffs, :ρ, :rho, :θ, :theta, :slope, :intercept)

# =============================================================================
# Plane3D Specific Constructors
# =============================================================================

"""
    Plane3D(point::Point3, normal::Vec3) -> Plane3D

Construct a 3D plane from a point on the plane and a normal vector.
"""
function Plane3D(point::Point3{T1}, normal::Vec3{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    n = normalize(SVector{3, T}(normal))
    d = -dot(n, SVector{3, T}(point))
    return Plane3D{T}(SVector{4, T}(n..., d))
end

"""
    Plane3D(p1::Point3, p2::Point3, p3::Point3) -> Plane3D

Construct a 3D plane from three non-collinear points.
The normal direction follows the right-hand rule for p1→p2→p3.
"""
function Plane3D(p1::Point3, p2::Point3, p3::Point3)
    v1 = SVector{3}(p2) - SVector{3}(p1)
    v2 = SVector{3}(p3) - SVector{3}(p1)
    n = cross(v1, v2)
    norm(n) > 0 || throw(ArgumentError("Points are collinear"))
    return Plane3D(p1, Vec3(n...))
end

"""
    Plane3D{T}() -> Plane3D{T}

Construct the canonical z=0 plane.
"""
Plane3D{T}() where {T} = Plane3D{T}(SVector{4, T}(0, 0, 1, 0))
Plane3D() = Plane3D{Float64}()

# =============================================================================
# Intersection Operations
# =============================================================================

"""
    intersect(l1::Line2D, l2::Line2D) -> Point2 or nothing

Find the intersection point of two 2D lines.
Returns `nothing` if lines are parallel.
"""
function Base.intersect(l1::Line2D{T1}, l2::Line2D{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    # Cross product of homogeneous line representations gives intersection point
    p = cross(SVector{3, T}(l1.coeffs), SVector{3, T}(l2.coeffs))
    abs(p[3]) < eps(T) && return nothing  # Parallel lines
    return Point2{T}(PerspectiveMap()(p))
end

"""
    clip_to_rect(l::Line2D, rect::Rect2) -> GeometryBasics.Line{2,T} or nothing

Clip the infinite 2D line `l` to the axis-aligned rectangle `rect`, returning
the line segment (as `GeometryBasics.Line`) of the portion inside. Returns
`nothing` if the line misses the rectangle.

Liang-Barsky clipping on the parametric line `p(t) = p₀ + t·(−b, a)`, where
`p₀ = (−ac, −bc)` is the closest point to the origin (uses `‖(a,b)‖=1`).
"""
function clip_to_rect(l::Line2D{T}, rect::Rect2) where {T}
    a, b, c = l[1], l[2], l[3]
    p0x = -a * c; p0y = -b * c
    dx = -b;     dy = a

    xmin = T(rect.origin[1]);            ymin = T(rect.origin[2])
    xmax = T(xmin + rect.widths[1]);     ymax = T(ymin + rect.widths[2])

    tmin = T(-Inf); tmax = T(Inf)

    # x constraint: xmin ≤ p0x + t*dx ≤ xmax
    if iszero(dx)
        (p0x < xmin || p0x > xmax) && return nothing
    else
        t1 = (xmin - p0x) / dx
        t2 = (xmax - p0x) / dx
        lo, hi = dx > 0 ? (t1, t2) : (t2, t1)
        tmin = max(tmin, lo); tmax = min(tmax, hi)
    end
    # y constraint: ymin ≤ p0y + t*dy ≤ ymax
    if iszero(dy)
        (p0y < ymin || p0y > ymax) && return nothing
    else
        t1 = (ymin - p0y) / dy
        t2 = (ymax - p0y) / dy
        lo, hi = dy > 0 ? (t1, t2) : (t2, t1)
        tmin = max(tmin, lo); tmax = min(tmax, hi)
    end
    tmin > tmax && return nothing

    p1 = Point2{T}(p0x + tmin * dx, p0y + tmin * dy)
    p2 = Point2{T}(p0x + tmax * dx, p0y + tmax * dy)
    return GeometryBasics.Line(p1, p2)
end

"""
    intersect(plane::Plane3D, line_origin::Point3, line_dir::Vec3) -> Point3 or nothing

Find the intersection of a 3D plane with a line defined by point and direction.
Returns `nothing` if line is parallel to plane.
"""
function Base.intersect(plane::Plane3D{T}, origin::Point3, dir::Vec3) where {T}
    n = normal(plane)
    denom = dot(n, SVector{3}(dir))
    abs(denom) < eps(T) && return nothing  # Line parallel to plane

    t = -(dot(n, SVector{3}(origin)) + plane[4]) / denom
    return Point3{T}((SVector{3}(origin) + t * SVector{3}(dir))...)
end

# =============================================================================
# Random Sampling (matching vgtk0 interface)
# =============================================================================

"""
    rand([rng], Line2D; max_rho=1.0) -> Line2D

Sample a random 2D line with Hesse distance up to `max_rho` from origin.
"""
function Random.rand(rng::Random.AbstractRNG, ::Type{Line2D{T}}; max_rho::Real = 1.0) where {T}
    ρ = rand(rng) * max_rho
    θ = rand(rng) * 2π
    return Line2D{T}(SVector{3, T}(cos(θ), sin(θ), -ρ))
end

Random.rand(rng::Random.AbstractRNG, ::Type{Line2D}; kwargs...) =
    rand(rng, Line2D{Float64}; kwargs...)

Random.rand(::Type{Line2D{T}}; kwargs...) where {T} =
    rand(Random.default_rng(), Line2D{T}; kwargs...)

Random.rand(::Type{Line2D}; kwargs...) =
    rand(Random.default_rng(), Line2D{Float64}; kwargs...)

"""
    rand([rng], Plane3D; bounds=1.0) -> Plane3D

Sample a random 3D plane with offset up to `bounds` from origin.
"""
function Random.rand(rng::Random.AbstractRNG, ::Type{Plane3D{T}}; bounds::Real = 1.0) where {T}
    # Random unit normal (uniform on sphere)
    n = randn(rng, SVector{3, T})
    n = n / norm(n)
    d = (2 * rand(rng) - 1) * bounds
    return Plane3D{T}(SVector{4, T}(n..., d))
end

Random.rand(rng::Random.AbstractRNG, ::Type{Plane3D}; kwargs...) =
    rand(rng, Plane3D{Float64}; kwargs...)

Random.rand(::Type{Plane3D{T}}; kwargs...) where {T} =
    rand(Random.default_rng(), Plane3D{T}; kwargs...)

Random.rand(::Type{Plane3D}; kwargs...) =
    rand(Random.default_rng(), Plane3D{Float64}; kwargs...)

# =============================================================================
# =============================================================================


# =============================================================================
# EuclideanMap Integration (rigid transforms)
# =============================================================================

# Import EuclideanMap from parent module
using ..VisualGeometryToolkitCore: EuclideanMap

"""
    (E::EuclideanMap{3})(plane::Plane3D) -> Plane3D

Transform a 3D plane through a Euclidean (rigid) transformation.

For Euclidean transform x ↦ R*x + t, plane with normal n and offset d transforms as:
- n' = R * n (rotate the normal)
- d' = d - n · t (adjust offset by translation along normal)

# Example
```julia
R = RotMatrix{3}(...)
t = SVector(1.0, 2.0, 3.0)
E = EuclideanMap(R, t)

plane = Plane3D(Point3(0,0,0), Vec3(0,0,1))  # z = 0
plane′ = E(plane)  # Transformed plane
```
"""
function (E::PlainEuclideanMap{3, T})(plane::Plane3D{S}) where {T, S}
    n = normal(plane)
    # Rotate normal
    n′ = E.R * n
    # Adjust offset: new_d = old_d - n · t
    # (translation moves the plane along/against its normal)
    d′ = offset(plane) - dot(n, SVector{3}(E.t))
    return Plane3D{promote_type(T, S)}(SVector{4, promote_type(T, S)}(n′..., -d′))
end

"""
    (E::EuclideanMap{2})(line::Line2D) -> Line2D

Transform a 2D line through a Euclidean (rigid) transformation.

# Example
```julia
R = RotMatrix{2}(π/4)
t = SVector(1.0, 2.0)
E = EuclideanMap(R, t)

line = Line2D(0.0, π/2)  # y = 0 (Hesse form)
line′ = E(line)  # Transformed line
```
"""
function (E::PlainEuclideanMap{2, T})(line::Line2D{S}) where {T, S}
    n = normal(line)
    # Rotate normal
    n′ = E.R * n
    # Adjust offset
    d′ = offset(line) - dot(n, SVector{2}(E.t))
    return Line2D{promote_type(T, S)}(SVector{3, promote_type(T, S)}(n′..., -d′))
end

# =============================================================================
# Batched ray–plane intersection (0.14; hoisted from RayTracedCalibration)
# =============================================================================
#
# Broadcasted structure-of-arrays intersection of ray bundles with one
# Plane3D. The plane constants are extracted ONCE into plain scalars
# (`_plane_constants`) before the broadcast; the ChainRulesCore extension
# marks that extraction `@non_differentiable`, so under reverse-mode AD the
# plane is a constant of the computation (the downstream calibration
# pipelines optimize rays and poses, never the target plane — and scalar
# getindex on the SVector coefficients would otherwise mint structured
# tangents that derail generic pullbacks).
#
# The two overloads keep their DISTINCT accumulation orders on purpose: the
# shared-origin form computes the scalar n·o fully parenthesized, the
# per-ray form accumulates left-to-right — both transcribed from the
# consumer bit-for-bit (Float32 association matters; downstream fixed-seed
# render tests pin the results).

# Plane constants as plain scalars: unit normal components and the
# homogeneous offset d of n·x + d = 0. Non-differentiable via the
# ChainRulesCore extension.
_plane_constants(plane::Plane3D) =
    (normal(plane)[1], normal(plane)[2], normal(plane)[3], plane.coeffs[4])

"""
    ray_plane_intersection(plane, origin, dir_x, dir_y, dir_z)
        -> (x, y, z, t)
    ray_plane_intersection(plane, o_x, o_y, o_z, dir_x, dir_y, dir_z)
        -> (x, y, z, t)

Broadcasted ray–plane intersection over structure-of-arrays ray bundles:
the world-space hit points and ray parameters of the rays `origin + t·dir`
against `plane` (convention `n·x + d = 0`). The first form shares one
`origin::StaticVector{3}` across the bundle; the second takes per-ray
origin component arrays (finite-aperture cameras, multi-bounce tracers).
Rays parallel to the plane divide by zero and produce `±Inf`/`NaN`
components — callers mask by validity, the kernel never throws.

The plane is a CONSTANT of the computation: its scalars are extracted once
before the broadcast and the extraction is `@non_differentiable` under the
ChainRulesCore extension; gradients flow through origins and directions
only.
"""
function ray_plane_intersection(
        plane::Plane3D, origin::StaticVector{3}, dir_x, dir_y, dir_z
    )
    n_x, n_y, n_z, d = _plane_constants(plane)
    c_x, c_y, c_z = origin

    t = .-(d .+ (n_x .* c_x .+ n_y .* c_y .+ n_z .* c_z)) ./
        (n_x .* dir_x .+ n_y .* dir_y .+ n_z .* dir_z)

    x = c_x .+ dir_x .* t
    y = c_y .+ dir_y .* t
    z = c_z .+ dir_z .* t

    return x, y, z, t
end

# Per-ray-origin overload — origins vary per sample; the n·o term is a
# per-sample array, accumulated LEFT-TO-RIGHT (consumer bit-parity).
function ray_plane_intersection(
        plane::Plane3D, o_x, o_y, o_z, dir_x, dir_y, dir_z
    )
    n_x, n_y, n_z, d = _plane_constants(plane)
    t = .-(d .+ n_x .* o_x .+ n_y .* o_y .+ n_z .* o_z) ./
        (n_x .* dir_x .+ n_y .* dir_y .+ n_z .* dir_z)
    return o_x .+ dir_x .* t, o_y .+ dir_y .* t, o_z .+ dir_z .* t, t
end
