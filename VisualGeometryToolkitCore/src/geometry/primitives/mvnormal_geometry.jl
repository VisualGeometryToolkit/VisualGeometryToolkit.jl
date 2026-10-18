# =============================================================================
# MvNormal Geometry Integration
# =============================================================================
#
# Provides conversions between Distributions.MvNormal and geometric types.
# MvNormal is used for statistical operations (sampling, PDF).
# EllipticFrame is used for geometric operations (boundary, AffineMap).
#
# Usage:
#   d = MvNormal(SVector(100.0, 200.0), SMatrix{2,2}(25.0, 5.0, 5.0, 16.0))
#   rand!(d, buf)           # statistical: sample
#   pdf(d, x)               # statistical: density
#
#   frame = EllipticFrame(d)  # convert to geometry
#   ellipse = region(frame, 3.0)  # geometric: 3σ contour
#
# =============================================================================

# MvNormal and AbstractMvNormal are imported from parent via primitives.jl
using Statistics: mean, cov

# =============================================================================
# Convenience Constructors
# =============================================================================

"""
    MvNormal(center::Point2, Σ::SMatrix{2,2}) -> MvNormal

Construct a 2D MvNormal from a Point2 center and static covariance matrix.

# Example
```julia
d = MvNormal(Point2(100.0, 200.0), @SMatrix [25.0 5.0; 5.0 16.0])
```
"""
function MvNormal(center::Point2{T}, Σ::SMatrix{2, 2, T, 4}) where {T}
    return MvNormal(SVector{2, T}(center...), Σ)
end

"""
    MvNormal(center::Point2, σ::Real) -> MvNormal

Construct an isotropic 2D MvNormal from center and scalar σ.
Covariance is σ²I.

# Example
```julia
d = MvNormal(Point2(100.0, 200.0), 5.0)  # isotropic with σ=5
```
"""
function MvNormal(center::Point2{S}, σ::T) where {S, T <: Real}
    R = promote_type(S, T)
    Σ = SMatrix{2, 2, R}(σ^2, zero(R), zero(R), σ^2)
    return MvNormal(SVector{2, R}(center...), Σ)
end

"""
    MvNormal(d::Disc) -> MvNormal

Convert a Disc to an isotropic MvNormal.
"""
MvNormal(d::Disc{T}) where {T} = MvNormal(d.center, d.σ)

# =============================================================================
# Conversions to Geometric Types
# =============================================================================

"""
    EllipticFrame(d::AbstractMvNormal) -> EllipticFrame

Convert an MvNormal to an EllipticFrame for geometric operations.

The MvNormal covariance Σ is converted to shape matrix S = √Σ via eigendecomposition.

# Example
```julia
d = MvNormal(SVector(100.0, 200.0), SMatrix{2,2}(25.0, 5.0, 5.0, 16.0))
frame = EllipticFrame(d)
ellipse = region(frame, 3.0)  # 3σ contour
```
"""
function EllipticFrame(d::AbstractMvNormal)
    μ = mean(d)
    Σ = cov(d)
    T = eltype(μ)
    return EllipticFrame(Point2{T}(μ...), Σ, Val(:moment_matrix))
end

"""
    Disc(d::AbstractMvNormal) -> Disc

Convert an MvNormal to a Disc using the equivalent isotropic scale.
"""
function Disc(d::AbstractMvNormal)
    μ = mean(d)
    Σ = cov(d)
    σ = sqrt(sqrt(det(Σ)))  # characteristic_scale
    return Disc(Point2(μ...), σ)
end

# =============================================================================
# Centroid for Detection Interface
# =============================================================================

"""
    centroid(d::AbstractMvNormal) -> Point2

Return the centroid (mean) as a Point2.
"""
centroid(d::AbstractMvNormal) = Point2(mean(d)...)

# JSON serialization moved to VisualGeometryToolkitFeatures/serialization.jl
