# =============================================================================
# Circle3D - Planar circle embedded in 3D space
# =============================================================================
#
# 7 DOF representation:
# - center: 3 DOF
# - radius: 1 DOF
# - normal: 2 DOF (unit vector, 3 components with ||n||=1 constraint)
# - rotation about normal: 1 DOF (but irrelevant for circle, so we store frame)
#
# We store orthonormal frame vectors (u, v) instead of just normal to enable
# consistent parameterization: p(θ) = center + radius * (cos(θ)*u + sin(θ)*v)
#
# Shared methods (centroid, plane, normal, boundary_point, coordinates,
# extrema, Rect3, has_boundary) are inherited from AbstractConic3D
# defined in ellipse3d.jl.

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross, normalize
using Rotations: RotMatrix

# Import interface functions to extend
import ..VisualGeometryCore: radius

# Import EuclideanMap
using ..VisualGeometryCore: EuclideanMap

"""
    Circle3D{T}

A planar circle embedded in 3D space.

# Fields
- `center::Point3{T}`: Center point of the circle
- `radius::T`: Radius of the circle
- `u::Vec3{T}`: First basis vector in circle's plane (unit length)
- `v::Vec3{T}`: Second basis vector in circle's plane (unit length, perpendicular to u)

# Properties
The circle boundary is parameterized as:
```
p(θ) = center + radius * (cos(θ) * u + sin(θ) * v)
```

The normal is `cross(u, v)`.

# Example
```julia
# Circle centered at origin, in xy-plane, radius 3
c3d = Circle3D(Point3(0.0, 0.0, 0.0), 3.0)

# Transform with rigid motion
E = EuclideanMap(R, t)
c3d_transformed = E(c3d)

# Sample boundary point
θ = π/4
p = centroid(c3d) + c3d.radius * (cos(θ) * c3d.u + sin(θ) * c3d.v)
```
"""
struct Circle3D{T} <: AbstractConic3D{T}
    center::Point3{T}
    radius::T
    u::Vec3{T}
    v::Vec3{T}

    function Circle3D{T}(center::Point3{T}, radius::T, u::Vec3{T}, v::Vec3{T}) where {T}
        @assert radius > 0 "radius must be positive"
        @assert abs(norm(u) - 1) < 1.0e-10 "u must be unit length"
        @assert abs(norm(v) - 1) < 1.0e-10 "v must be unit length"
        @assert abs(dot(u, v)) < 1.0e-10 "u and v must be perpendicular"
        return new{T}(center, radius, u, v)
    end
end

# Convenience constructor with type promotion
function Circle3D(center::Point3{T1}, radius::T2, u::Vec3{T3}, v::Vec3{T4}) where {T1, T2, T3, T4}
    T = promote_type(T1, T2, T3, T4)
    return Circle3D{T}(Point3{T}(center...), T(radius), Vec3{T}(u...), Vec3{T}(v...))
end

Circle3D(c::Circle3D) = c

# Axis vector accessors for Circle3D (implements AbstractConic3D interface)
_axis_a(c::Circle3D) = c.radius * c.u
_axis_b(c::Circle3D) = c.radius * c.v

# =============================================================================
# Convenience constructors
# =============================================================================

"""
    Circle3D(center::Point3, radius, normal::Vec3) -> Circle3D

Construct a circle from center, radius, and normal vector.

Automatically generates orthonormal basis vectors u, v in the circle's plane.
"""
function Circle3D(center::Point3{T1}, radius::T2, n::Vec3{T3}) where {T1, T2, T3}
    T = promote_type(T1, T2, T3)
    n_unit = normalize(Vec3{T}(n))

    # Generate orthonormal basis in plane perpendicular to n
    # Choose reference vector not parallel to n
    ref = abs(n_unit[1]) < 0.9 ? Vec3{T}(1, 0, 0) : Vec3{T}(0, 1, 0)
    u = normalize(cross(n_unit, ref))
    v = cross(n_unit, u)

    return Circle3D{T}(Point3{T}(center), T(radius), Vec3{T}(u), Vec3{T}(v))
end

"""
    Circle3D(center::Point3, radius) -> Circle3D

Construct a circle in the xy-plane (normal = +z).
"""
function Circle3D(center::Point3{T1}, radius::T2) where {T1, T2}
    T = promote_type(T1, T2)
    return Circle3D{T}(
        Point3{T}(center...),
        T(radius),
        Vec3{T}(1, 0, 0),
        Vec3{T}(0, 1, 0)
    )
end

# =============================================================================
# Lift 2D Circle to 3D via EuclideanMap frame
# =============================================================================

"""
    Circle3D(c::Circle, E::EuclideanMap{3}) -> Circle3D

Embed a 2D circle into 3D using a Euclidean frame.

The EuclideanMap `E` defines how the local xy-plane maps to world coordinates.
"""
function Circle3D(c::GeometryBasics.Circle{T}, E::EuclideanMap{3}) where {T}
    ctr, u, v, RT = _embed_z0(E, GeometryBasics.origin(c), (one(T), zero(T)), (zero(T), one(T)), T)
    return Circle3D{RT}(ctr, RT(GeometryBasics.radius(c)), u, v)
end

"""
    Circle3D(c::Circle) -> Circle3D

Embed 2D circle into 3D in the xy-plane at z=0.
"""
Circle3D(c::GeometryBasics.Circle{T}) where {T} =
    Circle3D(c, EuclideanMap(one(RotMatrix{3, T}), zero(Vec3{T})))

# =============================================================================
# Circle3D-specific Accessors
# =============================================================================

"""Return the radius of the 3D circle."""
radius(c::Circle3D) = c.radius

# =============================================================================
# Circle3D-specific EuclideanMap Transform
# =============================================================================

"""
    (E::EuclideanMap{3})(c::Circle3D) -> Circle3D

Transform a 3D circle through a rigid (Euclidean) transformation.
"""
function (E::EuclideanMap{3, T})(c::Circle3D{S}) where {T, S}
    RT = promote_type(T, S)
    new_center = Point3{RT}(E.R * c.center + E.t)
    new_u = Vec3{RT}(E.R * c.u)
    new_v = Vec3{RT}(E.R * c.v)
    return Circle3D{RT}(new_center, RT(c.radius), new_u, new_v)
end

# =============================================================================
# Conversion to Ellipse3D
# =============================================================================

"""
    Ellipse3D(c::Circle3D) -> Ellipse3D

Convert a Circle3D to an Ellipse3D (axis_a = radius*u, axis_b = radius*v).
"""
function Ellipse3D(c::Circle3D{T}) where {T}
    return Ellipse3D{T}(
        c.center,
        Vec3{T}((c.radius * SVector{3}(c.u))...),
        Vec3{T}((c.radius * SVector{3}(c.v))...)
    )
end
