# ========================================================================
# HomographyMap - Projective transformation for points, lines, and conics
# ========================================================================

# Note: This file is included in VisualGeometryToolkitCore main module which provides:
# - StaticArrays (SVector, SMatrix, Size, similar_type, StaticMatrix)
# - CoordinateTransformations (Transformation, AffineMap)
# - GeometryBasics (Point, Point2, Point3)
# - AbstractHomMat, AbstractHomConicMat, HomCircleMat, HomEllipseMat (from matrices.jl)
# - to_affine (from conversions.jl)

# ========================================================================
# HomographyMap - Projective transformation with cached inverse
# ========================================================================

"""
    HomographyMap{N,T,M<:StaticMatrix{N,N,T},Src<:Frame,Dst<:Frame} <: Transformation

Projective transformation with cached inverse for efficient
covariant (points) and contravariant (lines, conics) transformations.

Sub-projective transforms (Translation, LinearMap, AffineMap) can act
directly on Circle, Ellipse, and Line2D via ct_contravariant.jl.
Use HomographyMap when you need a full projective (HomographyMat) wrapper.

# Type Parameters
- `N`: Matrix dimension (3 for 2D projective, 4 for 3D projective)
- `T`: Element type
- `M`: Matrix type (HomRotationMat, HomAffineMat, HomographyMat, etc.)
- `Src`/`Dst`: coordinate-[`Frame`](@ref) tags of the point FRAMES the map
  connects — `Frame{N-1}`, since the N×N matrix acts on (N-1)-dimensional
  frames (a 3×3 homography is 2D↔2D projective: `Frame{2}` on both sides).
  Same design as `EuclideanMap`: the map consumes `Src`-frame coordinates
  and produces `Dst`-frame coordinates (Drake monogram `H_AB` ≙
  `Src = B, Dst = A`), every frame-free constructor defaults both tags to
  `UnlabeledFrame{N-1}`, `inv` swaps the tags, and `∘` checks a labeled
  junction (identical wildcard truth table — see `Base.:∘` below).

`HomographyMat` and the rest of the `AbstractHomMat` hierarchy stay
frame-FREE algebra; frames are declared at the Mat → Map lift:

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
# Create from any homogeneous matrix type
h = HomographyMap(HomographyMat(camera))
h = HomographyMap(HomRotationMat(RotMatrix{2}(θ)))
h = HomographyMap(HomAffineMat(...))

# Transform points (Euclidean → homogeneous output)
p′ = h(Point2(100, 200))  # → Point3 (homogeneous)

# Transform points (homogeneous → homogeneous output)
p′ = h(Point3(100, 200, 1))  # → Point3

# Transform conics: the result type follows the matrix class
h(circle)   # Circle under a similarity, HomEllipseMat under an affinity, Conic otherwise
h(ellipse)  # HomEllipseMat under an affinity (similarities included), Conic otherwise

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

# Constructor computes inverse once
function HomographyMap(m::StaticMatrix{N, N, T}) where {N, T}
    M = typeof(m)
    m_inv = M(Tuple(inv(SMatrix{N, N}(m))))
    return HomographyMap{N, T, M}(m, m_inv)
end

# For AbstractHomMat types, preserve the matrix type
function HomographyMap(m::M) where {M <: AbstractHomMat}
    m_inv = M(Tuple(inv(SMatrix{3, 3}(m))))
    return HomographyMap{3, eltype(m), M}(m, m_inv)
end

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

# AbstractMatrixMap interface: HomographyMap's stored matrix IS its homogeneous
# form — the one accessor to it (`inv(h)`'s is the stored inverse).
to_homogeneous(h::HomographyMap) = h.forward

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

# Helper: apply conic transformation formula Q' = H^{-T}·Q·H^{-1} (a congruence: `propagate`)
@inline function _transform_conic_matrix(h::HomographyMap{3}, Q::SMatrix{3, 3})
    return propagate(transpose(SMatrix{3, 3}(h.inverse)), Q)
end

# Circle + similarity-preserving transform → Circle (exact)
@inline function _transform_circle(
        ::Type{<:AbstractSimilarityMat},
        h::HomographyMap{3, T, M}, c::Circle
    ) where {T, M}
    Q = SMatrix{3, 3}(HomCircleMat(c))
    Q′ = _transform_conic_matrix(h, Q)
    return Circle(HomCircleMat{eltype(Q′)}(Tuple(Q′)))
end

# Circle + affine (non-similarity) → HomEllipseMat (exact: affine of a circle is an ellipse).
@inline function _transform_circle(
        ::Type{<:AbstractAffineMat},
        h::HomographyMap{3, T, M}, c::Circle
    ) where {T, M}
    Q = SMatrix{3, 3}(HomCircleMat(c))
    Q′ = _transform_conic_matrix(h, Q)
    return HomEllipseMat{eltype(Q′)}(Tuple(Q′))
end

# Circle + general projective / raw StaticMatrix → Conic (could be any conic class).
@inline function _transform_circle(
        ::Type{<:StaticMatrix},
        h::HomographyMap{3, T, M}, c::Circle
    ) where {T, M}
    Q = SMatrix{3, 3}(HomCircleMat(c))
    Q′ = _transform_conic_matrix(h, Q)
    return Conic{eltype(Q′)}(Q′)
end

"""
    (h::HomographyMap{3})(c::Circle) -> Circle, HomEllipseMat or Conic

Transform a circle through a homography map.

Returns:
- a `Circle` for similarity-preserving transforms (rotation, translation, isotropic scale)
- `HomEllipseMat` for other affine transforms (an affine image of a circle is an ellipse)
- `Conic` for a general projective transform

A returned conic matrix could be:
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
@inline function (h::HomographyMap{3, T, M})(c::Circle) where {T, M}
    return _transform_circle(M, h, c)
end

"""
    (h::HomographyMap{3})(e::Ellipse) -> HomEllipseMat or Conic

Transform an ellipse through a homography map. Returns the conic matrix.

`HomEllipseMat` under an affine map; under a general projective transform a
`Conic`, which could be any conic type.
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
@inline function (h::HomographyMap{3, T, M})(e::Ellipse) where {T, M}
    return _transform_ellipse(M, h, e)
end

# Ellipse + affine → HomEllipseMat (exact: affine of an ellipse is an ellipse).
@inline function _transform_ellipse(
        ::Type{<:AbstractAffineMat},
        h::HomographyMap{3, T, M}, e::Ellipse
    ) where {T, M}
    Q = SMatrix{3, 3}(HomEllipseMat(e))
    Q′ = _transform_conic_matrix(h, Q)
    return HomEllipseMat{eltype(Q′)}(Tuple(Q′))
end

# Ellipse + general projective / raw StaticMatrix → Conic (could be any conic class).
@inline function _transform_ellipse(
        ::Type{<:StaticMatrix},
        h::HomographyMap{3, T, M}, e::Ellipse
    ) where {T, M}
    Q = SMatrix{3, 3}(HomEllipseMat(e))
    Q′ = _transform_conic_matrix(h, Q)
    return Conic{eltype(Q′)}(Q′)
end

# ========================================================================
# Composition and Inverse
# ========================================================================

# Composition: (A ∘ B)(x) = A(B(x))
# Forward: A.forward * B.forward
# Inverse: B.inverse * A.inverse
# Inner-matrix result type comes from the AbstractHomMat hierarchy via
# `Base.promote_type` (defined in matrices.jl) when BOTH inners are typed
# Hom mats.  A raw-SMatrix inner on either side demotes the result to a
# plain SMatrix — `promote_type(SMatrix, HomAffineMat)` would land on the
# abstract `StaticArray{Tuple{3,3},T,2}` and fail to construct.
function _compose_mat_type(::Type{Ma}, ::Type{Mb}) where {Ma <: AbstractHomMat, Mb <: AbstractHomMat}
    return promote_type(Ma, Mb)                       # parametric, e.g. HomEuclideanMat{Float64}
end
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
#   HomographyMap{Ma} ∘ HomographyMap{Mb} → HomographyMap{promote_type(Ma,Mb)}   [above]
#   EuclideanMap     ∘ EuclideanMap       → EuclideanMap (analytic R, t)        [in euclidean_map.jl]
#
# Mixed shapes — convert the non-HomographyMap side to its homogeneous matrix
# form, wrap as a HomographyMap, then delegate to the HomographyMap ∘
# HomographyMap method.  Inner-matrix categories tighten via the
# AbstractHomMat hierarchy in `Base.promote_rule`.
#
#   EuclideanMap   ∘ HomographyMap → HomographyMap{promote_type(HomEuclideanMat, Mb)}
#   HomographyMap  ∘ EuclideanMap  → HomographyMap{promote_type(Ma, HomEuclideanMat)}
#   CT.X           ∘ HomographyMap → HomographyMap{promote_type(typeof_M(X), Mb)}
#   HomographyMap  ∘ CT.X          → HomographyMap{promote_type(Ma, typeof_M(X))}

# EuclideanMap mixed with HomographyMap → HomographyMap (Option A: demote to
# the more general matrix-backed Map; inner matrix tightens via promote_rule).
# The EuclideanMap's frame tags ride along on the lift — never silently
# erased — so the junction stays frame-checked by the HomographyMap rules.
@inline Base.:∘(
    a::EuclideanMap{2, T, R, V, Src, Dst}, b::HomographyMap{3}
) where {T, R <: Rotation{2, T}, V <: StaticVector{2, T}, Src <: Frame{2}, Dst <: Frame{2}} =
    relabel(Src, Dst, HomographyMap(to_homogeneous(a))) ∘ b
@inline Base.:∘(
    a::HomographyMap{3}, b::EuclideanMap{2, T, R, V, Src, Dst}
) where {T, R <: Rotation{2, T}, V <: StaticVector{2, T}, Src <: Frame{2}, Dst <: Frame{2}} =
    a ∘ relabel(Src, Dst, HomographyMap(to_homogeneous(b)))

# CT-foreign types mixed with HomographyMap.  Scoped to a Union of the
# concrete CT types (Translation/LinearMap/AffineMap) so the dispatch
# doesn't collide with HomographyMap ∘ HomographyMap, which is itself a
# CT.Transformation pair.
const _CTLeafTransformation = Union{
    CoordinateTransformations.Translation,
    CoordinateTransformations.LinearMap,
    CoordinateTransformations.AffineMap,
}

# A CT leaf as a HomographyMap: its homogeneous matrix, and as the stored
# inverse the homogeneous matrix of ITS CT inverse (`inv(A)`, `−inv(A)·t` — the
# leaf's own analytic inverse), never a generic 3×3 inversion of the forward.
function HomographyMap(a::_CTLeafTransformation)
    fwd = AbstractHomMat(a)
    M = typeof(fwd)
    return HomographyMap{size(M, 1), eltype(M), M}(fwd, M(Tuple(to_homogeneous(inv(a)))))
end

@inline Base.:∘(a::_CTLeafTransformation, b::HomographyMap{3}) = HomographyMap(a) ∘ b
@inline Base.:∘(a::HomographyMap{3}, b::_CTLeafTransformation) = a ∘ HomographyMap(b)

# Inverse is O(1) - just swap forward and inverse (preserves matrix type).
# Typed algebra: inv swaps the frame tags (H_AB⁻¹ = H_BA), as for EuclideanMap.
function Base.inv(h::HomographyMap{N, T, M, Src, Dst}) where {N, T, M, Src, Dst}
    return HomographyMap{N, T, M, Dst, Src}(h.inverse, h.forward)
end

# ========================================================================
# Rect → Polygon (projective mapping of a rectangle)
# ========================================================================

using CoordinateTransformations: PerspectiveMap

const _PerspHomography = CoordinateTransformations.ComposedTransformation{PerspectiveMap, <:HomographyMap{3}}

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
function transfer_ellipse(H::HomographyMap{3}, shape::Union{Circle, Ellipse})
    Q = H(shape)
    return is_ellipse(Q) ? Ellipse(Q) : nothing
end

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
affinity at the transferred centre (see [`transfer_ellipse(::Camera, _)`](@ref)).
`nothing` where the map is not defined or not finite at the centre (a lens
fold, the line at infinity).
"""
function transfer_ellipse(T::CoordinateTransformations.Transformation, shape::Union{Circle, Ellipse})
    c, Σ = _ellipse_moments(shape)
    c′ = T(c)
    (isfinite(c′[1]) && isfinite(c′[2])) || return nothing
    J = SMatrix{2, 2, Float64, 4}(transform_deriv(T, c))
    all(isfinite, J) || return nothing
    Σ′ = Symmetric(J * Σ * J')
    (Σ′[1, 1] > 0 && Σ′[2, 2] > 0 && Σ′[1, 1] * Σ′[2, 2] - Σ′[1, 2]^2 > 0) || return nothing
    return Ellipse(Point2(Float64(c′[1]), Float64(c′[2])), Σ′)
end
function transfer_ellipse(T::CoordinateTransformations.ComposedTransformation, shape::Union{Circle, Ellipse})
    inner = transfer_ellipse(T.t2, shape)
    inner === nothing && return nothing
    return transfer_ellipse(T.t1, inner)
end
# The SAME chain when `∘` produced Base's composite rather than
# CoordinateTransformations': a lens that is not itself a `Transformation` makes
# `AffineMap ∘ lens ∘ AffineMap` (i.e. `distortion_map`) a `ComposedFunction`.
# Without this, composing the lens chain is a MethodError on exactly those lens
# types — the reason `transfer_ellipse(::Camera, _)` used to unroll the stages.
# `outer`/`inner` are Base's names for `t1`/`t2`.
function transfer_ellipse(T::ComposedFunction, shape::Union{Circle, Ellipse})
    inner = transfer_ellipse(T.inner, shape)
    inner === nothing && return nothing
    return transfer_ellipse(T.outer, inner)
end
# The point-projective composite (PerspectiveMap ∘ HomographyMap) IS the
# homography's action on conics — never a stage-wise transfer.
transfer_ellipse(proj::_PerspHomography, shape::Union{Circle, Ellipse}) = transfer_ellipse(proj.t2, shape)

# (centre, moment matrix) of the shapes the transfer accepts.
_ellipse_moments(e::Ellipse) = (SVector{2, Float64}(e.center[1], e.center[2]), SMatrix{2, 2, Float64, 4}(moment_matrix(e)))
_ellipse_moments(c::Circle) = (
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
H = HomographyMap(HomAffineMat(...))
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

