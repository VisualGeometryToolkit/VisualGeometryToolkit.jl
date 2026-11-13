# ========================================================================
# HomographyMap - Projective transformation for points, lines, and conics
# ========================================================================

# Note: This file is included in VisualGeometryToolkitCore main module which provides:
# - StaticArrays (SVector, SMatrix, Size, similar_type, StaticMatrix)
# - CoordinateTransformations (Transformation, AffineMap)
# - GeometryBasics (Point, Point2, Point3)
# - HomographyMat (matrices.jl), AbstractHomConicMat, HomCircleMat, HomEllipseMat, Conic (base/types.jl)
# - to_affine (from conversions.jl)

# ========================================================================
# HomographyMap - Projective transformation with cached inverse
# ========================================================================

"""
    HomographyMap{N,T,M<:StaticMatrix{N,N,T},Src<:Frame,Dst<:Frame} <: Transformation

Projective transformation with cached inverse for efficient
covariant (points) and contravariant (lines, conics) transformations.

A `HomographyMap` is ALWAYS projective in meaning, whatever its matrix: a
circle or an ellipse maps to a [`Conic`](@ref). The sub-projective classes are
their own maps, acting in their class (the class table of the Transforms
guide, "Maps Carry the Transform Class").

# Type Parameters
- `N`: Matrix dimension (3 for 2D projective, 4 for 3D projective)
- `T`: Element type
- `M`: Matrix type — a `HomographyMat` (a lifted homography matrix) or a plain
  `SMatrix` (every other lift and every mixed composition)
- `Src`/`Dst`: coordinate-[`Frame`](@ref) tags of the point FRAMES the map
  connects — `Frame{N-1}`, since the N×N matrix acts on (N-1)-dimensional
  frames (a 3×3 homography is 2D↔2D projective: `Frame{2}` on both sides).
  Same design as `EuclideanMap`: the map consumes `Src`-frame coordinates
  and produces `Dst`-frame coordinates (Drake monogram `H_AB` ≙
  `Src = B, Dst = A`), every frame-free constructor defaults both tags to
  `UnlabeledFrame{N-1}`, `inv` swaps the tags, and `∘` checks a labeled
  junction (identical wildcard truth table — see `Base.:∘` below).

`HomographyMat` stays frame-FREE algebra; frames are declared at the
Mat → Map lift:

```julia
H = HomographyMap{TargetPlaneFrame, PixelFrame{:colmap}}(H_mat)   # labeled lift
H = HomographyMap(H_mat)                                    # unlabeled lift
```

[`relabel`](@ref)/[`unlabel`](@ref) are the same explicit escape hatches as
for `EuclideanMap`.

# Transformation Rules (Hartley & Zisserman conventions)
- **Points** (covariant): p' = H·p
- **Lines** (contravariant): l' = H^{-T}·l
- **Conics** (contravariant): Q' = H^{-T}·Q·H^{-1}

# Dimension Convention
- N×N matrix operates on (N-1)-dimensional Euclidean space
- Point{N-1} → Euclidean input, lifted via to_affine
- Point{N} → Homogeneous input, used directly

# Examples
```julia
# Lift a homography matrix, or any structured map (its analytic inverse is stored)
h = HomographyMap(HomographyMat(camera))
h = HomographyMap(EuclideanMap(RotMatrix{2}(θ), t))
h = HomographyMap(AffineMap(A, b))

# Transform points (Euclidean → homogeneous output)
p′ = h(Point2(100, 200))  # → Point3 (homogeneous)

# Transform points (homogeneous → homogeneous output)
p′ = h(Point3(100, 200, 1))  # → Point3

# Transform conics: always a Conic (classify it, or apply the structured map itself)
h(circle)   # Conic
h(ellipse)  # Conic

# Transform lines
l′ = h(Line2D(1.0, -1.0, 0.0))  # → Line2D

# Composition
h_composed = h1 ∘ h2

# Inverse is O(1) - just swaps stored matrices
h_inv = inv(h)
```
"""
struct HomographyMap{
        N, T, M <: StaticMatrix{N, N, T},
        Src <: Frame, Dst <: Frame,
    } <: AbstractMatrixMap
    forward::M    # H (for points: p' = H·p)
    inverse::M    # H^{-1} (for lines/conics)
end

# Gradual-typing default: constructing through the historical 3-parameter
# surface (`HomographyMap{N, T, M}(fwd, inv)`) produces an UNLABELED map
# over the (N-1)-dimensional point frames. Every frame-free constructor
# below funnels through this method, so pre-0.7 code compiles unchanged.
@inline function (::Type{HomographyMap{N, T, M}})(
        fwd, bwd
    ) where {N, T, M <: StaticMatrix{N, N, T}}
    return HomographyMap{N, T, M, UnlabeledFrame{N - 1}, UnlabeledFrame{N - 1}}(fwd, bwd)
end

# Constructor computes inverse once, by the INPUT's `inv` (a structured
# matrix's analytic inverse: a rotation's transpose, a calibration matrix's
# closed form), then stores both as SMatrix — the map is projective, whatever
# the matrix's structure. A HomographyMat is stored as itself (below).
function HomographyMap(m::StaticMatrix{N, N, T}) where {N, T}
    mi = inv(m)
    S = promote_type(T, eltype(mi))
    fwd = SMatrix{N, N, S}(m)
    return HomographyMap{N, S, typeof(fwd)}(fwd, SMatrix{N, N, S}(mi))
end
HomographyMap(m::HomographyMat{T}) where {T} = HomographyMap{3, T, HomographyMat{T}}(m, inv(m))

HomographyMap(h::HomographyMap) = h

"""
    HomographyMap{N, T}(h::HomographyMap{N}) -> HomographyMap{N, T, SMatrix{N, N, T}}

`h` in element type `T`: the same map, its frame labels kept and its STORED
inverse converted (never recomputed), both as plain `SMatrix`es. Also reached by
`convert(HomographyMap{N, T}, h)`.
"""
function (::Type{HomographyMap{N, T}})(h::HomographyMap{N, T2, M2, Src, Dst}) where {N, T, T2, M2, Src, Dst}
    fwd = SMatrix{N, N, T}(h.forward)
    return HomographyMap{N, T, typeof(fwd), Src, Dst}(fwd, SMatrix{N, N, T}(h.inverse))
end
Base.convert(::Type{HomographyMap{N, T}}, h::HomographyMap{N}) where {N, T} = HomographyMap{N, T}(h)

"""
    HomographyMap{Src, Dst}(m::StaticMatrix{N, N}) -> HomographyMap

The labeled Mat → Map lift: wrap the frame-free homogeneous matrix `m` and
declare its frame endpoints in one step (the map consumes `Src`-frame
coordinates and produces `Dst`-frame coordinates; monogram `H_AB` ≙
`Src = B, Dst = A`). `Src`/`Dst` must be `Frame{N-1}` tags — a 3×3 matrix
lifts to a 2D↔2D projective map, so both frames are `Frame{2}`.

```julia
H = HomographyMap{TargetPlaneFrame, PixelFrame{:colmap}}(HomographyMat(camera))
```
"""
function (::Type{HomographyMap{Src, Dst}})(
        m::StaticMatrix{N, N}
    ) where {N, K, Src <: Frame{K}, Dst <: Frame{K}}
    K == N - 1 || throw(
        ArgumentError(
            "HomographyMap frame tags must be Frame{N-1} point frames: an " *
                "$(N)×$(N) matrix acts on $(N - 1)-dimensional frames, got Frame{$K}."
        )
    )
    return relabel(Src, Dst, HomographyMap(m))
end

# Frame labeling escape hatches — the HomographyMap methods of the
# EuclideanMap relabel/unlabel verbs (same semantics: deliberate, unchecked
# override; zero-cost — both stored matrices are reused bit-for-bit).
@inline function relabel(
        ::Type{Src}, ::Type{Dst}, h::HomographyMap{N, T, M}
    ) where {N, T, M <: StaticMatrix{N, N, T}, K, Src <: Frame{K}, Dst <: Frame{K}}
    K == N - 1 || throw(
        ArgumentError(
            "HomographyMap frame tags must be Frame{N-1} point frames: an " *
                "$(N)×$(N) map acts on $(N - 1)-dimensional frames, got Frame{$K}."
        )
    )
    return HomographyMap{N, T, M, Src, Dst}(h.forward, h.inverse)
end

@inline unlabel(h::HomographyMap{N}) where {N} =
    relabel(UnlabeledFrame{N - 1}, UnlabeledFrame{N - 1}, h)

# An uncertain homography is relabeled as its value: its covariance, in the
# chart of the map's own matrix (HomographyManifold8), does not move.
relabel(::Type{Src}, ::Type{Dst}, u::Uncertain{<:HomographyMap{N}}) where {N, K, Src <: Frame{K}, Dst <: Frame{K}} =
    Uncertain(relabel(Src, Dst, geom(u)), getfield(u, :attributes))
unlabel(u::Uncertain{<:HomographyMap{N}}) where {N} = relabel(UnlabeledFrame{N - 1}, UnlabeledFrame{N - 1}, u)
map_frames(::HomographyMap{N, T, M, Src, Dst}) where {N, T, M, Src, Dst} = (Src, Dst)

# AbstractMatrixMap interface: HomographyMap's stored matrix IS its homogeneous
# form — the one accessor to it (`inv(h)`'s is the stored inverse), as a plain SMatrix.
to_homogeneous(h::HomographyMap{N, T}) where {N, T} = SMatrix{N, N, T}(h.forward)

# HomographyMap stores its inverse, so it's always invertible

# ========================================================================
# Point Transformations (covariant): p' = H·p
# Compile-time dispatch based on dimension matching
# ========================================================================

# Homogeneous input: Point{N} matches matrix dimension N
# More specific method - selected when dimensions match
@inline function (h::HomographyMap{N, T, M})(p::Point{N}) where {N, T, M}
    p′ = h.forward * p
    return Point{N}(p′...)
end

# Euclidean input: Point{K} where K ≠ N (typically K = N-1)
# Less specific method - selected when dimensions don't match
@inline function (h::HomographyMap{N, T, M})(p::Point{K}) where {N, T, M, K}
    p′ = h.forward * to_affine(p)
    return Point{N}(p′...)
end

# Generic StaticVector support (accepts SVector, Vec, MVector — same dispatch logic)
@inline function (h::HomographyMap{N, T, M})(v::StaticVector{N}) where {N, T, M}
    return h.forward * v
end

@inline function (h::HomographyMap{N, T, M})(v::StaticVector{K}) where {N, T, M, K}
    return h.forward * to_affine(v)
end

# The derivative of the action above, which is LINEAR in homogeneous coordinates:
# the matrix itself for homogeneous input, its first K columns for a Euclidean
# point lifted by `to_affine`. The Euclidean transfer π(H·ũ) is the composite
# `PerspectiveMap() ∘ h`; CoordinateTransformations' chain rule gives its 2×2
# Jacobian `transform_deriv(PerspectiveMap(), H·ũ)·H[:, 1:K]` from these two
# pieces — the noise transport of a mapped point, with no separate name for it.
@inline CoordinateTransformations.transform_deriv(h::HomographyMap{N}, ::StaticVector{N}) where {N} = h.forward
@inline CoordinateTransformations.transform_deriv(h::HomographyMap{N}, ::StaticVector{K}) where {N, K} =
    h.forward[:, SOneTo(K)]

# ========================================================================
# Conic Transformations (contravariant): Q' = H^{-T}·Q·H^{-1}
# Maps operate on geometric primitives (Circle, Ellipse)
# For matrix operations, use the formula directly with h.inverse
# ========================================================================

# The conic congruence C′ = H⁻ᵀ·C·H⁻¹ from the inverse Hi = H⁻¹, as the six
# coefficients of the symmetric result: C′ᵢⱼ = Hi[:, i] ⋅ (C·Hi)[:, j], computed
# for i ≤ j only, so the result is exactly symmetric by storage.
@inline function _conic_congruence(Hi::StaticMatrix{3, 3}, Q::StaticMatrix{3, 3})
    T = promote_type(eltype(Hi), eltype(Q))
    H = SMatrix{3, 3, T, 9}(Hi)
    M = SMatrix{3, 3, T, 9}(Q) * H
    c(i, j) = H[1, i] * M[1, j] + H[2, i] * M[2, j] + H[3, i] * M[3, j]
    return (c(1, 1), c(1, 2), c(2, 2), c(1, 3), c(2, 3), c(3, 3))
end

# A conic of a circle or an ellipse under a projective map: any class
@inline function _projective_conic(h::HomographyMap{3}, Q::AbstractHomConicMat)
    q = _conic_congruence(h.inverse, Q)
    return Conic{eltype(q)}(q, _Unchecked())
end

"""
    (h::HomographyMap{3})(c::Circle) -> Conic

Transform a circle through a homography map: a `Conic`, whatever the map's
matrix (a `HomographyMap` is projective in meaning). An `EuclideanMap`, a
`SimilarityMap`, a translation, rotation or uniform scaling keeps a circle a
`Circle`, and an affine map gives a `HomEllipseMat`: apply them directly.

The returned conic could be:
- An ellipse (most common for camera projections of visible circles)
- A hyperbola (circle partially behind camera)
- A parabola (circle tangent to plane at infinity)
- A degenerate conic (point or lines)

Use `classify_conic()` to determine the conic type, then construct the
appropriate geometric primitive:

```julia
H = HomographyMap(HomographyMat(camera))
circle = Circle(Point2(100.0, 100.0), 50.0)
Q′ = H(circle)  # a Conic under a camera homography

trait = classify_conic(Q′)
if trait isa EllipseTrait
    ellipse = Ellipse(Q′)  # Construct geometric primitive
elseif trait isa HyperbolaTrait
    # Handle hyperbola case
end
```

# Example: Filter to valid ellipses
```julia
H = HomographyMap(HomographyMat(camera))
conic_matrices = H.(circles)
ellipses = [Ellipse(Q) for Q in conic_matrices if classify_conic(Q) isa EllipseTrait]
```
"""
@inline (h::HomographyMap{3})(c::Circle) = _projective_conic(h, HomCircleMat(c))

"""
    (h::HomographyMap{3})(e::Ellipse) -> Conic

Transform an ellipse through a homography map: a `Conic`, which could be any
conic type (an affine map applied directly gives a `HomEllipseMat`).
Use `classify_conic()` to determine the type and construct appropriately.

# Example
```julia
H = HomographyMap(HomographyMat(camera))
Q′ = H(ellipse)
if classify_conic(Q′) isa EllipseTrait
    ellipse′ = Ellipse(Q′)
end
```
"""
@inline (h::HomographyMap{3})(e::Ellipse) = _projective_conic(h, HomEllipseMat(e))

# ========================================================================
# Composition and Inverse
# ========================================================================

# Composition: (A ∘ B)(x) = A(B(x))
# Forward: A.forward * B.forward
# Inverse: B.inverse * A.inverse
# Two lifted homography matrices compose to a HomographyMat; any other pair
# (an SMatrix-backed lift on either side) to a plain SMatrix.
_compose_mat_type(::Type{<:HomographyMat{Ta}}, ::Type{<:HomographyMat{Tb}}) where {Ta, Tb} = HomographyMat{promote_type(Ta, Tb)}
function _compose_mat_type(::Type{Ma}, ::Type{Mb}) where {N, Ma <: StaticMatrix{N, N}, Mb <: StaticMatrix{N, N}}
    T = promote_type(eltype(Ma), eltype(Mb))
    return SMatrix{N, N, T, N * N}
end

@inline function _compose_homography(
        a::HomographyMap{N, Ta, Ma}, b::HomographyMap{N, Tb, Mb},
        ::Type{Src}, ::Type{Dst}
    ) where {N, Ta, Tb, Ma, Mb, Src <: Frame, Dst <: Frame}
    ResultMatType = _compose_mat_type(Ma, Mb)
    T_out = eltype(ResultMatType)
    fwd = SMatrix{N, N, T_out}(a.forward) * SMatrix{N, N, T_out}(b.forward)
    inv_mat = SMatrix{N, N, T_out}(b.inverse) * SMatrix{N, N, T_out}(a.inverse)
    return HomographyMap{N, T_out, ResultMatType, Src, Dst}(
        ResultMatType(Tuple(fwd)),
        ResultMatType(Tuple(inv_mat))
    )
end

# -----------------------------------------------------------------------------
# Frame-checked composition — SAME truth table as EuclideanMap
# -----------------------------------------------------------------------------
# The junction rule is `_junction_ok` (base/frames.jl) — the truth table at
# `Base.:∘` for `EuclideanMap`: a labeled junction must agree, `UnlabeledFrame`
# junction endpoints are unchecked wildcards, the result always takes
# `(B.Src, A.Dst)` as-is, and a labeled-labeled mismatch is an explicit
# `MethodError` throw — HomographyMap is also a CT `Transformation`, so an
# absent method would silently fall through to CT's lazy
# `ComposedTransformation` and defeat the check.
function Base.:∘(
        a::HomographyMap{N, Ta, Ma, SrcA, DstA},
        b::HomographyMap{N, Tb, Mb, SrcB, DstB}
    ) where {
        N, Ta, Tb, Ma <: StaticMatrix{N, N, Ta}, Mb <: StaticMatrix{N, N, Tb},
        SrcA <: Frame, DstA <: Frame, SrcB <: Frame, DstB <: Frame,
    }
    _check_junction(SrcA, DstB, ∘, (a, b))
    return _compose_homography(a, b, SrcB, DstA)
end

# ------------------------------------
# Cross-system composition bridges (Option A: demote to HomographyMap)
# ------------------------------------
#
# Composition rules:
#
#   HomographyMap ∘ HomographyMap → HomographyMap (HomographyMat storage when both are, else SMatrix)  [above]
#   EuclideanMap  ∘ EuclideanMap  → EuclideanMap (analytic R, t)        [euclidean_map.jl]
#   SimilarityMap ∘ SimilarityMap → SimilarityMap (analytic s, R, t)    [similarity_map.jl]
#
# Mixed shapes — lift the structured side to a HomographyMap (forward its
# homogeneous matrix, inverse the homogeneous matrix of its ANALYTIC inverse,
# never a 3×3 inversion), then compose as homographies: SMatrix storage.
#
#   EuclideanMap / SimilarityMap / CT leaf  ∘ HomographyMap → HomographyMap
#   HomographyMap ∘ EuclideanMap / SimilarityMap / CT leaf  → HomographyMap

# CT-foreign types mixed with HomographyMap.  Scoped to a Union of the
# concrete CT types (Translation/LinearMap/AffineMap) so the dispatch
# doesn't collide with HomographyMap ∘ HomographyMap, which is itself a
# CT.Transformation pair.
const _CTLeafTransformation = Union{
    CoordinateTransformations.Translation,
    CoordinateTransformations.LinearMap,
    CoordinateTransformations.AffineMap,
}

# THE affine class of planar maps, stated once: CoordinateTransformations'
# affines and the stack's own affine-class map types, each with its analytic
# inverse and an `AffineMap(m)`. A new affine-class map type is added to
# `_StructuredAffineMap` with its `AffineMap` method, and the homography lift,
# the `∘` bridges below and the warp fusion (warp.jl) all take it.
const _StructuredAffineMap = Union{EuclideanMap{2}, SimilarityMap{2}}
const _AffinePlanarMap = Union{_CTLeafTransformation, _StructuredAffineMap}

CoordinateTransformations.AffineMap(E::PlainEuclideanMap{2}) = AffineMap(SMatrix{2, 2}(E.R), SVector{2}(E.t))
CoordinateTransformations.AffineMap(S::SimilarityMap{2}) = AffineMap(S.s * SMatrix{2, 2}(S.R), SVector{2}(S.t))

"""
    HomographyMap(a) -> HomographyMap{3, T, SMatrix{3, 3, T, 9}}

Any structured planar map — a `Translation`, `LinearMap` or `AffineMap`, an
[`EuclideanMap`](@ref)`{2}`, a [`SimilarityMap`](@ref)`{2}` — lifted to a
homography: the forward matrix `to_homogeneous(a)`, the stored inverse
`to_homogeneous(inv(a))` (the map's own analytic inverse, never a 3×3
inversion). An `EuclideanMap`'s or `SimilarityMap`'s frame tags ride along.
The lift is projective in meaning: it maps a circle to a `Conic`.
"""
HomographyMap(a::_CTLeafTransformation) = _lift_homography(a, UnlabeledFrame{2}, UnlabeledFrame{2})
HomographyMap(a::EuclideanMap{2, T, R, V, Src, Dst}) where {T, R, V, Src, Dst} = _lift_homography(a, Src, Dst)
HomographyMap(a::SimilarityMap{2, T, S, R, V, Src, Dst}) where {T, S, R, V, Src, Dst} = _lift_homography(a, Src, Dst)
function _lift_homography(a, ::Type{Src}, ::Type{Dst}) where {Src, Dst}
    fwd = to_homogeneous(a)
    bwd = _inverse_homogeneous(a)   # checked: a singular affine is refused, never an Inf inverse stored
    T = promote_type(eltype(fwd), eltype(bwd))
    return HomographyMap{3, T, SMatrix{3, 3, T, 9}, Src, Dst}(SMatrix{3, 3, T, 9}(fwd), SMatrix{3, 3, T, 9}(bwd))
end

@inline Base.:∘(a::_AffinePlanarMap, b::HomographyMap{3}) = HomographyMap(a) ∘ b
@inline Base.:∘(a::HomographyMap{3}, b::_AffinePlanarMap) = a ∘ HomographyMap(b)

# Inverse is O(1) - just swap forward and inverse (preserves matrix type).
# Typed algebra: inv swaps the frame tags (H_AB⁻¹ = H_BA), as for EuclideanMap.
function Base.inv(h::HomographyMap{N, T, M, Src, Dst}) where {N, T, M, Src, Dst}
    return HomographyMap{N, T, M, Dst, Src}(h.inverse, h.forward)
end
_inverse_homogeneous(h::HomographyMap) = to_homogeneous(inv(h))   # its stored inverse

# ========================================================================
# Rect → Polygon (projective mapping of a rectangle)
# ========================================================================

using CoordinateTransformations: PerspectiveMap

const _PerspHomography = CoordinateTransformations.ComposedTransformation{PerspectiveMap, <:HomographyMap{3}}

# The 2-D homography map, `PerspectiveMap() ∘ h` (x ↦ π(H·x̃)): its 2 × 2 Jacobian at the point `x`, the
# dehomogenised map's — `transform_deriv(PerspectiveMap(), H·x̃) · H[:, 1:2]`, in the promoted number type of H and
# x. THE point Jacobian of a homography: `propagate` of an uncertain point through a framed homography
# (propagate_framed.jl) and the plane map of a posed camera (`plane_map`, cameras.jl) read it. (CT's chain rule
# over `transform_deriv(h, ·)`, the homogeneous 3 × 2 above, gives the same product; this method fixes its number
# type and its bits.)
@inline function CoordinateTransformations.transform_deriv(proj::_PerspHomography, x::StaticVector{2})
    H = proj.t2.forward
    T = promote_type(eltype(H), eltype(x))
    w = H * SVector{3, T}(x[1], x[2], one(T))
    dπ = transform_deriv(PerspectiveMap(), w)
    return dπ * SMatrix{3, 2, T, 6}(H[1], H[2], H[3], H[4], H[5], H[6])   # dπ · H[:, 1:2]
end

"""
    (proj)(rect::HyperRectangle{2}) -> Polygon

Project a 2D rectangle through a composed PerspectiveMap ∘ HomographyMap,
returning a 2D polygon (quadrilateral).
"""
(proj::_PerspHomography)(rect::GeometryBasics.HyperRectangle{2}) =
    GeometryBasics.Polygon([Point2f(proj(Point2(c))...) for c in rect_corners(rect)])

function (proj::_PerspHomography)(shape::Union{Circle, Ellipse})
    H = proj.t2  # the HomographyMap inside the composition
    return transfer_ellipse(H, shape)
end

"""
    transfer_ellipse(H::HomographyMap, shape::Union{Circle, Ellipse}) -> Ellipse | nothing

Push a circle/ellipse through the homography and classify: the transferred
conic when it IS an ellipse, `nothing` when the map carries it across the
line at infinity (hyperbola/parabola/degenerate). Downstream code spelled
`Q = H(region(...)); is_ellipse(Q) || continue; Ellipse(Q)` at six call
sites — this is that pattern, once.
"""
transfer_ellipse(H::HomographyMap{3}, shape::Union{Circle, Ellipse}) = _ellipse_or_nothing(H(shape))

"""
    transfer_ellipse(T::Transformation, shape::Union{Circle, Ellipse}) -> Ellipse | nothing

The ellipse-PRESERVING transfer of a circle/ellipse through any 2-D map of
the warping paradigm. A homography carries the conic exactly (the method
above); every other map is replaced by its local affinity at the shape's
centre — `c′ = T(c)`, `Σ′ = J·Σ·Jᵀ` with `J = transform_deriv(T, c)` — the
same linearisation a detection's shape gets when it is carried across an
undistortion. A `ComposedTransformation` transfers stage by stage, so the
exact stages stay exact and only the non-projective ones linearise: for a
lens camera, `distortion_map(model) ∘ HomographyMap(pinhole camera)` pushes
the conic through the plane homography exactly and applies the lens's
affinity at the transferred centre (see [`transfer_ellipse(::PosedCamera, _)`](@ref)).
`nothing` where the map is not defined or not finite at the centre (a lens
fold, the line at infinity).
"""
function transfer_ellipse(T::CoordinateTransformations.Transformation, shape::Union{Circle, Ellipse})
    c, Σ = _ellipse_center_shape(shape)
    c′ = T(c)
    (isfinite(c′[1]) && isfinite(c′[2])) || return nothing
    J = SMatrix{2, 2, Float64, 4}(transform_deriv(T, c))
    all(isfinite, J) || return nothing
    Σ′ = Symmetric(J * Σ * J')
    (Σ′[1, 1] > 0 && Σ′[2, 2] > 0 && Σ′[1, 1] * Σ′[2, 2] - Σ′[1, 2]^2 > 0) || return nothing
    return Ellipse(Point2(Float64(c′[1]), Float64(c′[2])), Σ′)
end
# A composition transfers stage by stage, the inner map first (`nothing` propagates).
function transfer_ellipse(T::CoordinateTransformations.ComposedTransformation, shape::Union{Circle, Ellipse})
    inner = transfer_ellipse(T.t2, shape)
    inner === nothing && return nothing
    return transfer_ellipse(T.t1, inner)
end
# The point-projective composite (PerspectiveMap ∘ HomographyMap) IS the
# homography's action on conics — never a stage-wise transfer.
transfer_ellipse(proj::_PerspHomography, shape::Union{Circle, Ellipse}) = transfer_ellipse(proj.t2, shape)

# (centre, shape matrix) of the shapes the transfer accepts.
_ellipse_center_shape(e::Ellipse) = (SVector{2, Float64}(e.center[1], e.center[2]), SMatrix{2, 2, Float64, 4}(shape_matrix(e)))
_ellipse_center_shape(c::Circle) = (
    SVector{2, Float64}(c.center[1], c.center[2]),
    SMatrix{2, 2, Float64, 4}(Float64(c.r)^2, 0.0, 0.0, Float64(c.r)^2),
)

"""
    (proj::_PerspHomography)(l::Line2D) -> Line2D

Lines transform contravariantly through the homography (π′ = H⁻ᵀπ) and need
no perspective division — the composed map delegates to the inner
`HomographyMap`. Closes the API-uniformity gap: the same composed projective
map handles points, conics, rects, polygons, and lines (coverage audit
2026-07-28).
"""
(proj::_PerspHomography)(l::Line2D) = proj.t2(l)

"""
    (proj::_PerspHomography)(p::Polygon) -> Polygon

Vertex-wise projective transform of a polygon — completes the region-mapping
family alongside the `HyperRectangle → Polygon` method above.
"""
function (proj::_PerspHomography)(p::GeometryBasics.Polygon)
    return GeometryBasics.Polygon(
        [
            Point2f(proj(Point2(Float64(q[1]), Float64(q[2])))...)
                for q in GeometryBasics.coordinates(p)
        ]
    )
end

# Convert HomographyMap to AffineMap (for affine or lower transforms)
function CoordinateTransformations.AffineMap(h::HomographyMap{3})
    H = SMatrix{3, 3}(h.forward)
    # Extract 2×2 linear part and 2×1 translation
    M = SMatrix{2, 2}(H[1, 1], H[2, 1], H[1, 2], H[2, 2])
    t = SVector{2}(H[1, 3], H[2, 3])
    return AffineMap(M, t)
end

# ========================================================================
# Hyperplane Transformations (contravariant): π' = H^{-T}·π
# Hyperplanes transform like lines (dual to points)
# ========================================================================

# Import Hyperplane types from Primitives
using .Primitives: Hyperplane, Line2D, Plane3D

"""
    (h::HomographyMap{N})(hp::Hyperplane{M}) -> Hyperplane{M}

Transform a hyperplane through a homography map.

Hyperplanes transform contravariantly (like lines): π' = H^{-T}·π

# Example
```julia
H = HomographyMap(AffineMap(A, b))
line = Line2D(1.0, -1.0, 0.0)  # x - y = 0
line′ = H(line)

plane = Plane3D(Point3(0,0,0), Vec3(0,0,1))  # z = 0
# For 4×4 homography map:
# plane′ = H(plane)
```
"""
@inline function (h::HomographyMap{N, T, M})(hp::Hyperplane{K, N, S}) where {N, T, M, K, S}
    # Contravariant transformation: π' = H^{-T}·π — THE line/plane law (a Line2D,
    # a Plane3D, and every CoordinateTransformations map through its homography)
    coeffs′ = transpose(h.inverse) * SVector{N}(hp.coeffs)
    return Hyperplane{K, N, promote_type(T, S)}(coeffs′)
end

