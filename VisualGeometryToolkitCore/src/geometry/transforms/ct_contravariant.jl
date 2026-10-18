# ========================================================================
# CT Contravariant Transforms
# ========================================================================
#
# Makes CoordinateTransformations types (Translation, LinearMap, AffineMap)
# directly callable on contravariant geometric primitives:
#   - Circle (conic, contravariant)
#   - Ellipse (conic, contravariant)
#   - Line2D (hyperplane, contravariant)
#
# Also provides `+` operator shorthand for translation of primitives.
#
# For projective transforms or explicit matrix control, use HomographyMap.
#
# DEPENDENCY GRAPH:
#   matrices.jl          → homogeneous matrix types + hierarchy
#   conversions.jl       → to_homogeneous (CT → typed 3×3)
#   primitives/conic.jl  → Circle, Ellipse, HomCircleMat, HomEllipseMat
#   primitives/hyperplane.jl → Line2D, normal, offset
#
# ========================================================================

using .Primitives: Hyperplane, Line2D, normal, offset

# ========================================================================
# Translation — exact formulas (shift center / adjust offset)
# ========================================================================

"""
    (t::Translation{<:StaticVector{2}})(c::Circle) -> Circle

Translate a circle by shifting its center. Returns a Circle (translation preserves shape).
"""
@inline function (t::CoordinateTransformations.Translation{<:StaticVector{2}})(c::Circle)
    return Circle(Point2((centroid(c) .+ t.translation)...), GeometryBasics.radius(c))
end

"""
    (t::Translation{<:StaticVector{2}})(e::Ellipse) -> Ellipse

Translate an ellipse by shifting its center. Returns an Ellipse (translation preserves shape).
"""
@inline function (t::CoordinateTransformations.Translation{<:StaticVector{2}})(e::Ellipse)
    return Ellipse(Point2((e.center .+ t.translation)...), e.a, e.b, e.θ)
end

"""
    (t::Translation{<:StaticVector{2}})(l::Line2D) -> Line2D

Translate a 2D line. For line ax + by + c = 0, translating by (tx, ty) gives:
a*x + b*y + (c - a*tx - b*ty) = 0
"""
@inline function (t::CoordinateTransformations.Translation{<:StaticVector{2}})(l::Line2D)
    a, b, c = l.coeffs
    tx, ty = t.translation
    T_out = promote_type(eltype(l.coeffs), eltype(t.translation))
    return Line2D{T_out}(SVector{3, T_out}(a, b, c - a * tx - b * ty))
end

# ========================================================================
# + operator — translation shorthand (commutative)
# ========================================================================

Base.:+(e::Ellipse, v::StaticVector{2}) = Translation(v)(e)
Base.:+(v::StaticVector{2}, e::Ellipse) = Translation(v)(e)
Base.:+(l::Line2D, v::StaticVector{2}) = Translation(v)(l)
Base.:+(v::StaticVector{2}, l::Line2D) = Translation(v)(l)

# ========================================================================
# Shared helpers for conic and line transforms
# ========================================================================

# A CoordinateTransformations map as its HomographyMap: the conic and line laws
# are HomographyMap's (`_transform_conic_matrix`, the Hyperplane method), never
# re-spelled here.
@inline _ct_homography(ct) = HomographyMap(to_homogeneous(ct))

"""
    _ct_conic_transform(ct, Q::AbstractHomConicMat) -> HomEllipseMat

Apply contravariant conic transformation: Q' = H^{-T} Q H^{-1}
where H is the homogeneous matrix of the CT transform.  Used for
`LinearMap` / `AffineMap` callables — both affine, so an ellipse-input
is guaranteed to stay an ellipse.
"""
@inline function _ct_conic_transform(ct, Q::AbstractHomConicMat)
    Q′ = _transform_conic_matrix(_ct_homography(ct), SMatrix{3, 3}(Q))
    return HomEllipseMat{eltype(Q′)}(Tuple(Q′))
end

"""
    _ct_line_transform(ct, l::Line2D) -> Line2D

Apply contravariant line transformation: l' = H^{-T} l
where H is the homogeneous matrix of the CT transform.
"""
@inline function _ct_line_transform(ct, l::Line2D)
    return _ct_homography(ct)(l)
end

# ========================================================================
# LinearMap on Circle — trait dispatch
# ========================================================================

# Circle-preservation predicate — all that survives of the former
# TransformTrait family (audit #18): rotations and uniform scalings map
# circles to circles; everything else demotes to an ellipse.  Type-stable,
# so the branch below folds at compile time.
_preserves_circles(::Type{<:CoordinateTransformations.LinearMap{<:Rotations.RotMatrix}}) = true
_preserves_circles(::Type{<:CoordinateTransformations.LinearMap{<:UniformScaling}}) = true
_preserves_circles(::Type{<:CoordinateTransformations.LinearMap}) = false

"""
    (lm::LinearMap)(c::Circle) -> Circle or HomEllipseMat

Transform a circle through a linear map.
- Circle-preserving transforms (rotation, uniform scale) return Circle
- Everything else (anisotropic scale, general linear) returns HomEllipseMat
"""
@inline function (lm::CoordinateTransformations.LinearMap)(c::Circle)
    if _preserves_circles(typeof(lm))
        Q′ = _transform_conic_matrix(_ct_homography(lm), SMatrix{3, 3}(HomCircleMat(c)))
        return Circle(HomCircleMat{eltype(Q′)}(Tuple(Q′)))
    else
        return _ct_conic_transform(lm, HomCircleMat(c))
    end
end

# ========================================================================
# LinearMap on Ellipse — always returns HomEllipseMat
# ========================================================================

"""
    (lm::LinearMap)(e::Ellipse) -> Conic

Transform an ellipse through a linear map. Returns a conic matrix.
"""
@inline function (lm::CoordinateTransformations.LinearMap)(e::Ellipse)
    return _ct_conic_transform(lm, HomEllipseMat(e))
end

# ========================================================================
# AffineMap on Circle / Ellipse — trait dispatch, returns HomEllipseMat
# ========================================================================
# Docstrings live on the AffineMap constructors in coord_maps.jl. Julia's
# docstring system can't distinguish `AffineMap(::Circle)` (constructor)
# from `(am::AffineMap)(::Circle)` (call overload) — both reduce to
# `AffineMap :: Tuple{Circle}` for attachment purposes and collide. The
# constructor site is the canonical home; these call overloads are
# straightforward implementations of the conic-transform formula
# Q' = H^{-T}·Q·H^{-1} (see _transform_conic_matrix above).

@inline (am::CoordinateTransformations.AffineMap)(c::Circle) =
    _ct_conic_transform(am, HomCircleMat(c))

@inline (am::CoordinateTransformations.AffineMap)(e::Ellipse) =
    _ct_conic_transform(am, HomEllipseMat(e))

# ========================================================================
# LinearMap / AffineMap on Line2D
# ========================================================================

"""
    (lm::LinearMap)(l::Line2D) -> Line2D

Transform a 2D line through a linear map. Lines stay lines under linear transforms.
"""
@inline function (lm::CoordinateTransformations.LinearMap)(l::Line2D)
    return _ct_line_transform(lm, l)
end

"""
    (am::AffineMap)(l::Line2D) -> Line2D

Transform a 2D line through an affine map. Lines stay lines under affine transforms.
"""
@inline function (am::CoordinateTransformations.AffineMap)(l::Line2D)
    return _ct_line_transform(am, l)
end

# ========================================================================
# EuclideanMap{2} — rigid actions on primitives and feature frames
# ========================================================================
#
# A labeled rigid map — the pixel-origin frame maps
# `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` above all — must be
# applicable to the things that live in pixel coordinates, not only to bare
# vectors. Until 2026-09 only `Translation` had these methods, so BlobBoards
# rewrapped every frame map's `.t` in a `Translation` to shift a blob list.
# A rigid map preserves shape, so every action here is exact and closed:
# centre ↦ R c + t, orientation ↦ + angle(R), shape matrices ↦ conjugated by R.
# (Line2D already had its action in primitives/hyperplane.jl.)

@inline _rigid_angle(R) = atan(R[2, 1], R[1, 1])
@inline function _rigid_center(E::EuclideanMap{2}, c)
    p = E.R * SVector{2}(c[1], c[2]) + E.t
    return Point2(p[1], p[2])
end

"""
    (E::EuclideanMap{2})(c::Circle) -> Circle
    (E::EuclideanMap{2})(e::Ellipse) -> Ellipse
    (E::EuclideanMap{2})(f::FeatureFrame) -> same frame type

Rigid action on a 2D primitive or feature frame: the centre is mapped through
`E`, the shape is carried along exactly (a rotation rotates an orientation or
conjugates a shape matrix; a pure translation leaves it unchanged). Lets an
origin-conversion frame map be broadcast over a detection list directly:
`EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia}).(blobs)`.
"""
@inline (E::EuclideanMap{2})(c::Circle) = Circle(_rigid_center(E, centroid(c)), GeometryBasics.radius(c))
@inline (E::EuclideanMap{2})(e::Ellipse) = Ellipse(_rigid_center(E, e.center), e.a, e.b, e.θ + _rigid_angle(E.R))
# (Lines already have their rigid action: `(E::EuclideanMap{2})(::Line2D)` in
# primitives/hyperplane.jl.)
@inline (E::EuclideanMap{2})(f::Disc) = Disc(_rigid_center(E, f.center), f.σ)
@inline (E::EuclideanMap{2})(f::OrientedDisc) = OrientedDisc(_rigid_center(E, f.center), f.σ, f.θ + _rigid_angle(E.R))
@inline function (E::EuclideanMap{2})(f::EllipticFrame)
    R = SMatrix{2, 2}(E.R)
    return EllipticFrame(_rigid_center(E, f.center), R * f.S * R')
end
@inline (E::EuclideanMap{2})(f::AffineFrame) = AffineFrame(_rigid_center(E, f.center), SMatrix{2, 2}(E.R) * f.A)
