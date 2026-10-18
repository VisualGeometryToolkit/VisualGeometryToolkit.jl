# =============================================================================
# Blob Types: FeatureFrame and Disc
# =============================================================================
#
# This file provides blob-like geometric primitives:
# - FeatureFrame: Abstract type for geometric feature frames
# - Disc: Isotropic blob (center, σ) - 3 DOF
#
# For anisotropic blobs, use:
# - EllipticFrame: 5 DOF geometric frame (center + shape matrix)
# - MvNormal: Statistical representation from Distributions.jl
#
# =============================================================================

# FeatureFrame is defined in base/types.jl (loaded before Primitives)

# Common interface
center(f::FeatureFrame) = f.center

# =============================================================================
# Disc (3 DOF) - Isotropic Feature Frame
# =============================================================================

"""
    Disc{T} <: FeatureFrame{T}

Isotropic feature frame with 3 degrees of freedom: translation (2) + scale (1).

Defines the transformation: `x = σ·u + center`

This maps the unit circle in canonical space to a circle of radius σ in image space.
Invariant to similarity transformations (up to rotation).

# Fields
- `center::Point2{T}`: Center position in image coordinates
- `σ::T`: Characteristic scale (not a boundary radius!)

# Example
```julia
disc = Disc(Point2(100.0, 200.0), 5.0)
T = AffineMap(disc)  # AffineMap: u → 5*u + [100, 200]
circ = region(disc, 3.0)  # Circle at 3σ
```

See also: [`OrientedDisc`](@ref), [`EllipticFrame`](@ref)
"""
struct Disc{T} <: FeatureFrame{T}
    center::Point2{T}
    σ::T

    function Disc{T}(center::Point2{T}, σ::T) where {T}
        σ > zero(σ) || throw(ArgumentError("σ must be positive, got $σ"))
        return new{T}(center, σ)
    end
end

# Convenience constructors
function Disc(center::Point2{S}, σ::T) where {S, T <: Number}
    _strip_coords((center[1], center[2], σ))   # one dimension, pixels or a length (checked, not kept)
    U = promote_type(S, T)
    return Disc{U}(Point2{U}(center), U(σ))
end
Disc(x::Real, y::Real, σ::Real) = Disc(Point2(x, y), σ)
Disc(center::AbstractVector, σ::Number) = Disc(Point2(center...), σ)
Disc(d::Disc) = d

# Properties
scale(d::Disc) = d.σ

"""
    Base.extrema(d::Disc) -> (Point2, Point2)

Compute axis-aligned bounding box of a Disc.
Returns `(min_point, max_point)` where the box spans `center ± σ`.
"""
function Base.extrema(d::Disc{T}) where {T}
    min_pt = Point2{T}(d.center[1] - d.σ, d.center[2] - d.σ)
    max_pt = Point2{T}(d.center[1] + d.σ, d.center[2] + d.σ)
    return (min_pt, max_pt)
end
