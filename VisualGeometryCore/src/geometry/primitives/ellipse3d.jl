# =============================================================================
# Ellipse3D - Planar ellipse embedded in 3D space
# =============================================================================
#
# Parametric 8 DOF representation using center and axis vectors:
# - center: 3 DOF
# - axis_a: 3 DOF (semi-major axis vector, encodes direction + length)
# - axis_b: 2 DOF (semi-minor axis vector, perpendicular to axis_a)
#
# Total: 8 DOF stored as 9 values with constraint axis_a ⊥ axis_b
#
# This representation:
# - Transforms cleanly with EuclideanMap
# - Directly usable for plotting: p(θ) = center + cos(θ)*axis_a + sin(θ)*axis_b
# - Plane recoverable via: Plane3D(center, cross(axis_a, axis_b))

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross, normalize
using Rotations: RotMatrix

# Import interface functions to extend
import ..VisualGeometryCore: centroid, normal, boundary_point
import ..VisualGeometryCore: has_boundary, HasBoundary

# Import EuclideanMap
using ..VisualGeometryCore: EuclideanMap

# =============================================================================
# AbstractConic3D - shared interface for 3D conic primitives
# =============================================================================

"""
    AbstractConic3D{T}

Abstract supertype for planar conic sections embedded in 3D space.

Subtypes must have a `center::Point3{T}` field and implement:
- `_axis_a(c)::SVector{3,T}` — first axis vector (direction × length)
- `_axis_b(c)::SVector{3,T}` — second axis vector (perpendicular to first)

The boundary is parameterized as: `p(θ) = center + cos(θ)*axis_a + sin(θ)*axis_b`

See also: [`Ellipse3D`](@ref), [`Circle3D`](@ref)
"""
abstract type AbstractConic3D{T} end

# Axis vector accessor interface (concrete subtypes implement these)
function _axis_a end
function _axis_b end

# =============================================================================
# Ellipse3D struct
# =============================================================================

"""
    Ellipse3D{T}

A planar ellipse embedded in 3D space, represented by center and axis vectors.

# Fields
- `center::Point3{T}`: Center point of the ellipse (3 DOF)
- `axis_a::Vec3{T}`: Semi-major axis vector (3 DOF)
- `axis_b::Vec3{T}`: Semi-minor axis vector, perpendicular to axis_a (2 DOF)

# Properties
The ellipse boundary is parameterized as:
```
p(θ) = center + cos(θ) * axis_a + sin(θ) * axis_b
```

# Example
```julia
# Ellipse centered at origin, in xy-plane, semi-axes 3 and 2
e3d = Ellipse3D(
    Point3(0.0, 0.0, 0.0),
    Vec3(3.0, 0.0, 0.0),  # semi-major along x
    Vec3(0.0, 2.0, 0.0)   # semi-minor along y
)

# Transform with rigid motion
E = EuclideanMap(R, t)
e3d_transformed = E(e3d)

# Sample boundary point
θ = π/4
p = centroid(e3d) + cos(θ) * e3d.axis_a + sin(θ) * e3d.axis_b
```
"""
struct Ellipse3D{T} <: AbstractConic3D{T}
    center::Point3{T}
    axis_a::Vec3{T}
    axis_b::Vec3{T}

    function Ellipse3D{T}(center::Point3{T}, axis_a::Vec3{T}, axis_b::Vec3{T}) where {T}
        # Verify perpendicularity (allow small numerical error)
        @assert abs(dot(axis_a, axis_b)) < 1.0e-10 * norm(axis_a) * norm(axis_b) "axis_a and axis_b must be perpendicular"
        return new{T}(center, axis_a, axis_b)
    end
end

# Convenience constructor with type promotion
function Ellipse3D(center::Point3{T1}, axis_a::Vec3{T2}, axis_b::Vec3{T3}) where {T1, T2, T3}
    T = promote_type(T1, T2, T3)
    return Ellipse3D{T}(Point3{T}(center...), Vec3{T}(axis_a...), Vec3{T}(axis_b...))
end

# Axis vector accessors for Ellipse3D
_axis_a(e::Ellipse3D) = e.axis_a
_axis_b(e::Ellipse3D) = e.axis_b

# =============================================================================
# Lift 2D shapes to 3D via EuclideanMap frame
# =============================================================================

"""
    Ellipse3D(e::Ellipse, E::EuclideanMap{3}) -> Ellipse3D

Embed a 2D ellipse into 3D using a Euclidean frame.

The EuclideanMap `E` defines how the local xy-plane (containing the 2D ellipse)
maps to world coordinates:
- `E.R` rotates the xy-plane to its 3D orientation
- `E.t` translates the local origin to its 3D position

# Example
```julia
# Place ellipse in xy-plane at z=5
E = EuclideanMap(I, SVector(0, 0, 5))
e3d = Ellipse3D(ellipse_2d, E)

# Place ellipse in a tilted plane
R = RotX(π/4)  # tilt 45° around x-axis
E = EuclideanMap(R, SVector(0, 0, 0))
e3d = Ellipse3D(ellipse_2d, E)
```
"""
function Ellipse3D(e::Ellipse{T}, E::EuclideanMap{3}) where {T}
    cosθ, sinθ = cos(e.θ), sin(e.θ)
    c, a, b, RT = _embed_z0(E, e.center, (e.a * cosθ, e.a * sinθ), (-e.b * sinθ, e.b * cosθ), T)
    return Ellipse3D(c, a, b)
end

# THE lift of a planar shape into 3D: its centre `ctr` and two in-plane vectors
# `a`, `b` (2-tuples, in `T`) placed on the local z = 0 plane and carried by the
# rigid map `E` — the centre as a point (`R·x + t`), the vectors as directions
# (`R·v`). Returns them in the promoted type `RT`, with `RT`.
@inline function _embed_z0(E::EuclideanMap{3}, ctr, a, b, ::Type{T}) where {T}
    RT = promote_type(T, eltype(E.t))
    c = E.R * Point3{T}(ctr[1], ctr[2], zero(T)) + E.t
    return Point3{RT}(c), Vec3{RT}(E.R * Vec3{T}(a[1], a[2], zero(T))), Vec3{RT}(E.R * Vec3{T}(b[1], b[2], zero(T))), RT
end

"""
    Ellipse3D(c::Circle, E::EuclideanMap{3}) -> Ellipse3D

Embed a 2D circle into 3D using a Euclidean frame.
"""
function Ellipse3D(c::GeometryBasics.Circle{T}, E::EuclideanMap{3}) where {T}
    r = GeometryBasics.radius(c)
    ctr, a, b, _ = _embed_z0(E, GeometryBasics.origin(c), (r, zero(T)), (zero(T), r), T)
    return Ellipse3D(ctr, a, b)
end

# Convenience: default to identity frame (xy-plane at origin)
"""
    Ellipse3D(e::Ellipse) -> Ellipse3D
    Ellipse3D(c::Circle) -> Ellipse3D

Embed 2D shape into 3D in the xy-plane at z=0.
"""
Ellipse3D(e3::Ellipse3D) = e3
Ellipse3D(e::Ellipse{T}) where {T} = Ellipse3D(e, EuclideanMap(one(RotMatrix{3, T}), SVector{3, T}(0, 0, 0)))
Ellipse3D(c::GeometryBasics.Circle{T}) where {T} = Ellipse3D(c, EuclideanMap(one(RotMatrix{3, T}), SVector{3, T}(0, 0, 0)))

# =============================================================================
# Ellipse3D-specific Accessors
# =============================================================================

# =============================================================================
# Ellipse3D-specific EuclideanMap Transform
# =============================================================================

"""
    (E::EuclideanMap{3})(e::Ellipse3D) -> Ellipse3D

Transform a 3D ellipse through a rigid (Euclidean) transformation.

The center is translated and rotated, the axis vectors are rotated.
"""
function (E::EuclideanMap{3, T})(e::Ellipse3D{S}) where {T, S}
    RT = promote_type(T, S)
    new_center = Point3{RT}((E.R * SVector{3}(e.center) + E.t)...)
    new_axis_a = Vec3{RT}((E.R * SVector{3}(e.axis_a))...)
    new_axis_b = Vec3{RT}((E.R * SVector{3}(e.axis_b))...)
    return Ellipse3D{RT}(new_center, new_axis_a, new_axis_b)
end

# =============================================================================
# Shared AbstractConic3D methods
# =============================================================================
# These are defined here (loaded before circle3d.jl) and work for any subtype
# that implements _axis_a/_axis_b.

"""Return the centroid of a 3D conic."""
centroid(c::AbstractConic3D) = c.center

"""Return the unit normal of the conic's plane as a `Vec3`."""
function normal(c::AbstractConic3D)
    n = normalize(cross(_axis_a(c), _axis_b(c)))
    return Vec3(n...)
end

# HasBoundary trait for all AbstractConic3D
has_boundary(::AbstractConic3D) = HasBoundary()

"""
    boundary_point(c::AbstractConic3D, θ::Real) -> Point3

Return the point on the conic boundary at parameter θ.

The parameterization is: `p(θ) = center + cos(θ)*axis_a + sin(θ)*axis_b`
"""
function boundary_point(c::AbstractConic3D{T}, θ::Real) where {T}
    return Point3{T}((SVector{3}(c.center) + cos(θ) * _axis_a(c) + sin(θ) * _axis_b(c))...)
end

# boundary_points is provided by generic implementation in interface_functions.jl
#
# Specialization: returns n+1 points (closed loop) for 3D line plotting.

"""
    coordinates(c::AbstractConic3D, n::Int=64) -> Vector{Point3}

Return coordinates for plotting, compatible with GeometryBasics/Makie.

Returns n+1 points (closed loop) for use with `lines` or `poly`.
"""
function GeometryBasics.coordinates(c::AbstractConic3D{T}, n::Int = 64) where {T}
    pts = Vector{Point3{T}}(undef, n + 1)
    for (i, θ) in enumerate(range(0, 2π, length = n + 1))
        pts[i] = boundary_point(c, θ)
    end
    return pts
end

# =============================================================================
# Shared Axis-Aligned Bounding Box
# =============================================================================

"""
    extrema(c::AbstractConic3D) -> (Point3, Point3)

Return axis-aligned bounding box corners (min_pt, max_pt) for a 3D conic.

For a parametric conic `p(θ) = center + cos(θ)*axis_a + sin(θ)*axis_b`,
the extent along each axis i is: `center_i ± sqrt(axis_a_i² + axis_b_i²)`
"""
function Base.extrema(c::AbstractConic3D{T}) where {T}
    ctr = SVector{3}(c.center)
    a = _axis_a(c)
    b = _axis_b(c)
    extent = sqrt.(a .^ 2 .+ b .^ 2)
    min_pt = Point3{T}((ctr .- extent)...)
    max_pt = Point3{T}((ctr .+ extent)...)
    return (min_pt, max_pt)
end

"""
    Rect3(c::AbstractConic3D; snap_to_grid=false) -> Rect3

Construct axis-aligned bounding box for any 3D conic.

# Arguments
- `snap_to_grid::Bool=false`: If true, floor/ceil coordinates for voxel alignment

```julia
e3d = Ellipse3D(Point3(0.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0))
bbox = Rect3(e3d)
bbox_aligned = Rect3(e3d; snap_to_grid=true)
```
"""
function GeometryBasics.Rect3(c::AbstractConic3D{T}; snap_to_grid::Bool = false) where {T}
    min_pt, max_pt = extrema(c)
    snap_to_grid || return GeometryBasics.Rect3{T}(min_pt, max_pt - min_pt)
    lo, hi = _snap_outward(T, Point3{T}(min_pt), Point3{T}(max_pt))
    return GeometryBasics.Rect3{T}(lo, hi - lo)
end
