# ========================================================================
# CT Contravariant Transforms
# ========================================================================
#
# Makes CoordinateTransformations types (Translation, LinearMap, AffineMap),
# EuclideanMap{2} and SimilarityMap{2} directly callable on geometric
# primitives, each in its own class (the table below):
#   - Circle (conic, contravariant)
#   - Ellipse (conic, contravariant)
#   - Line2D (hyperplane, contravariant)
#   - the feature frames (Euclidean and similarity maps)
#
# Also provides `+` operator shorthand for translation of primitives.
#
# For projective transforms, use HomographyMap (homography_map.jl).
#
# DEPENDENCY GRAPH:
#   euclidean_map.jl, similarity_map.jl → the Euclidean and similarity maps
#   conversions.jl       → to_homogeneous (map → plain SMatrix)
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
# Primitive actions by map class
# ========================================================================
#
# The map's TYPE is its class, and a method exists only where the result's
# type changes; everything else goes through the coarser class:
#
#   | class      | maps                                         | Circle        | Ellipse       | Line2D | feature frames |
#   |------------|----------------------------------------------|---------------|---------------|--------|----------------|
#   | Euclidean  | EuclideanMap{2}, Translation                 | Circle        | Ellipse       | Line2D | same type      |
#   | similarity | SimilarityMap{2}, LinearMap{<:Rotation{2}},  | Circle        | Ellipse       | Line2D | same type      |
#   |            | LinearMap{<:UniformScaling}                  |               |               |        |                |
#   | affine     | AffineMap, any other LinearMap               | HomEllipseMat | HomEllipseMat | Line2D | —              |
#   | projective | HomographyMap (homography_map.jl)            | Conic         | Conic         | Line2D | MethodError    |
#
# This table is the one statement of the classes; the map types' docstrings
# point here. The conic law is HomographyMap's `_conic_congruence`, never re-spelled here;
# an affine map conjugates a conic and moves a line by its own analytic inverse,
# `to_homogeneous(inv(am))` (never a 3×3 inversion). A Euclidean map is the
# similarity of scale 1 and acts through it.

# Q′ = A⁻ᵀ·Q·A⁻¹ of an affine map, from its checked analytic inverse
# (`_inverse_homogeneous`: a singular linear part is an ArgumentError). An affine
# image of an ellipse is an ellipse, but its class is still decided by THE
# classifier, so a stored HomEllipseMat and `classify_conic` never disagree: a
# linear part so ill-conditioned that its image reads as a parabola (κ(A)² past
# the classifier's eps^(3/4) floor, e.g. Diagonal(1e7, 1)) is an ArgumentError.
@inline function _affine_conic(am, Q::AbstractHomConicMat)
    q = _conic_congruence(_inverse_homogeneous(am), Q)
    _is_real_ellipse(q) || _throw_ill_conditioned_affine(am)
    return HomEllipseMat{eltype(q)}(q, _Unchecked())
end
@noinline _throw_ill_conditioned_affine(am) = throw(
    ArgumentError(
        "the affine image of the ellipse is too ill-conditioned to be told from a parabola " *
            "(the map's linear part: $(to_homogeneous(am)[SOneTo(2), SOneTo(2)]))"
    )
)

# l′ = M⁻ᵀ·l of a map, from its checked analytic inverse: the line law of every affine-class map.
@inline function _line_action(m, l::Line2D{S}) where {S}
    l′ = transpose(_inverse_homogeneous(m)) * SVector{3}(l.coeffs)
    return Line2D{eltype(l′)}(l′)
end

# ========================================================================
# LinearMap / AffineMap on Circle / Ellipse — the affine class: HomEllipseMat
# ========================================================================
# Docstrings live on the AffineMap constructors in coord_maps.jl. Julia's
# docstring system can't distinguish `AffineMap(::Circle)` (constructor)
# from `(am::AffineMap)(::Circle)` (call overload) — both reduce to
# `AffineMap :: Tuple{Circle}` for attachment purposes and collide. The
# constructor site is the canonical home.

"""
    (lm::LinearMap)(c::Circle) -> HomEllipseMat
    (lm::LinearMap)(e::Ellipse) -> HomEllipseMat

Transform a circle or an ellipse through a general linear map: its
`HomEllipseMat` (an affine image of an ellipse is an ellipse). A rotation or a
uniform scaling is a similarity and keeps the primitive's type (a `Circle`, an
`Ellipse`): see [`SimilarityMap`](@ref).
"""
@inline (lm::CoordinateTransformations.LinearMap)(c::Circle) = _affine_conic(lm, HomCircleMat(c))
@inline (lm::CoordinateTransformations.LinearMap)(e::Ellipse) = _affine_conic(lm, HomEllipseMat(e))

@inline (am::CoordinateTransformations.AffineMap)(c::Circle) = _affine_conic(am, HomCircleMat(c))
@inline (am::CoordinateTransformations.AffineMap)(e::Ellipse) = _affine_conic(am, HomEllipseMat(e))

# ========================================================================
# The similarity subgroup of LinearMap: through its SimilarityMap
# ========================================================================

const _SimilarityLinearMap = Union{
    CoordinateTransformations.LinearMap{<:Rotation{2}},
    CoordinateTransformations.LinearMap{<:UniformScaling},
}
for P in (:Circle, :Ellipse, :Disc, :OrientedDisc, :EllipticFrame, :AffineFrame)
    @eval @inline (lm::_SimilarityLinearMap)(x::$P) = SimilarityMap(lm)(x)
end

# ========================================================================
# LinearMap / AffineMap on Line2D
# ========================================================================

"""
    (lm::LinearMap)(l::Line2D) -> Line2D

Transform a 2D line through a linear map. Lines stay lines under linear transforms.
"""
@inline (lm::CoordinateTransformations.LinearMap)(l::Line2D) = _line_action(lm, l)

"""
    (am::AffineMap)(l::Line2D) -> Line2D

Transform a 2D line through an affine map. Lines stay lines under affine transforms.
"""
@inline (am::CoordinateTransformations.AffineMap)(l::Line2D) = _line_action(am, l)

# ========================================================================
# SimilarityMap{2} — the similarity class on primitives and feature frames
# ========================================================================
#
# x ↦ s·R·x + t preserves angles and length ratios, so each action is exact
# and closed: centre ↦ s R c + t, lengths (a radius, the axes, σ) ↦ s·length,
# orientation ↦ + angle(R), a shape factor S (Σ = S²) ↦ s·R S Rᵀ, an affine
# frame's A ↦ s·R A, a line n·x + c = 0 ↦ (R n)·x′ + (s c − (R n)·t) = 0 (its
# normal's length kept). An EuclideanMap{2} is the similarity of scale 1 and
# acts through it, bit for bit (s·x is exact at s = 1): a labeled rigid map —
# the pixel-origin frame maps `EuclideanMap(PixelFrame{:colmap},
# PixelFrame{:julia})` above all — applies to everything that lives in pixel
# coordinates (`EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia}).(blobs)`).

@inline _rigid_angle(R) = atan(R[2, 1], R[1, 1])
@inline function _similar_center(S::SimilarityMap{2}, c)
    p = S(SVector{2}(c[1], c[2]))
    return Point2(p[1], p[2])
end

"""
    (S::SimilarityMap{2})(x) -> same type as x
    (E::EuclideanMap{2})(x) -> same type as x

The similarity (or rigid) action on a `Circle`, an `Ellipse`, a `Line2D` or a
feature frame (`Disc`, `OrientedDisc`, `EllipticFrame`, `AffineFrame`), exact
in its class: see the class table above. A rotation or a uniform scaling
`LinearMap` acts through its `SimilarityMap`.
"""
@inline (S::SimilarityMap{2})(c::Circle) = Circle(_similar_center(S, centroid(c)), S.s * GeometryBasics.radius(c))
@inline (S::SimilarityMap{2})(e::Ellipse) =
    Ellipse(_similar_center(S, e.center), S.s * e.a, S.s * e.b, e.θ + _rigid_angle(S.R))
"""
    (S::SimilarityMap{2})(w::WholePlane) -> WholePlane

The whole plane maps to itself under any similarity: the identity on the
region a lens without an image circle images (`valid_domain`), beside the
`Circle` and `Ellipse` actions, so a region is mapped without testing its type.
"""
@inline (S::SimilarityMap{2})(w::WholePlane) = w
@inline function (S::SimilarityMap{2, T, Sc})(l::Line2D{U}) where {T, Sc, U}
    V = promote_type(T, Sc, eltype(S.t), U)   # every part: a Dual scale or translation, a Float64 t over Float32 s, R
    n′ = S.R * normal(l)
    return Line2D{V}(SVector{3, V}(n′[1], n′[2], S.s * l.coeffs[3] - dot(n′, SVector{2}(S.t))))
end
@inline (S::SimilarityMap{2})(f::Disc) = Disc(_similar_center(S, f.center), S.s * f.σ)
@inline (S::SimilarityMap{2})(f::OrientedDisc) =
    OrientedDisc(_similar_center(S, f.center), S.s * f.σ, f.θ + _rigid_angle(S.R))
@inline function (S::SimilarityMap{2})(f::EllipticFrame)
    R = SMatrix{2, 2}(S.R)
    return EllipticFrame(_similar_center(S, f.center), S.s * (R * f.S * R'))
end
@inline (S::SimilarityMap{2})(f::AffineFrame) = AffineFrame(_similar_center(S, f.center), S.s * (SMatrix{2, 2}(S.R) * f.A))

for P in (:Circle, :Ellipse, :Line2D, :Disc, :OrientedDisc, :EllipticFrame, :AffineFrame)
    @eval @inline (E::EuclideanMap{2})(x::$P) =
        (has_units(E) && _refuse_unitful($(string("the action on a ", P))); SimilarityMap(E)(x))
end
