# ========================================================================
# Coordinate Conversions
# ========================================================================

"""
    to_homogeneous(map) -> SMatrix{N+1, N+1, T}
    to_homogeneous(A::StaticMatrix{N,N}, t::StaticVector{N}) -> SMatrix{N+1, N+1, T}

The homogeneous matrix `[A t; 0 1]` of an affine-family map of `N`-space, as
a plain `SMatrix` — for algebra and interop. Every map has one: a
`Translation`, a `LinearMap` (a rotation, a `UniformScaling` — 2-D — a
`Diagonal`, any matrix), an `AffineMap`, an [`EuclideanMap`](@ref), a
[`SimilarityMap`](@ref), a [`RotationMap`](@ref), a composition `a ∘ b` (the
product), and a [`HomographyMap`](@ref) (its forward matrix). Exact: the
map's entries copied, the last row the constant `0 … 0 1`.

The matrix carries no class: the map's TYPE does. Geometric operations take
the map (`S(circle)`, `a ∘ b`, `inv(a)`) and convert inside, once.

# Example
```julia
M = @SMatrix [2.0 0.0; 0.0 1.5]  # scaling
t = SVector(1.0, 2.0)            # translation
H = to_homogeneous(M, t)         # 3×3 homogeneous matrix
```
"""
@inline function to_homogeneous(A::StaticMatrix{N, N}, t::StaticVector{N}) where {N}
    T = promote_type(eltype(A), eltype(t))
    return SMatrix{N + 1, N + 1, T}(
        ntuple(Val((N + 1) * (N + 1))) do k
            i, j = mod1(k, N + 1), cld(k, N + 1)
            i <= N ? (j <= N ? T(A[i, j]) : T(t[i])) : (i == j ? one(T) : zero(T))
        end
    )
end

# The linear part of a CoordinateTransformations map as a static matrix (a
# UniformScaling is 2-D: it states no dimension).
@inline _static_linear(A::StaticMatrix) = A
@inline _static_linear(A::UniformScaling{T}) where {T} = SMatrix{2, 2, T}(A)
@inline _static_linear(A::Diagonal{T, <:StaticVector{N}}) where {N, T} = SMatrix{N, N, T}(A)
@inline _static_linear(A::AbstractMatrix{T}) where {T} = SMatrix{2, 2, T}(A)   # an unsized matrix: 2-D
@inline _static_translation(t::StaticVector) = t
@inline _static_translation(t::AbstractVector{T}) where {T} = SVector{2, T}(t)  # an unsized vector: 2-D
@inline _zero_translation(::StaticMatrix{N, N, T}) where {N, T} = zero(SVector{N, T})
@inline _identity_linear(::StaticVector{N, T}) where {N, T} = SMatrix{N, N, T}(I)

to_homogeneous(R::Rotations.Rotation{N, T}) where {N, T} = to_homogeneous(SMatrix{N, N, T}(R), zero(SVector{N, T}))
to_homogeneous(Tx::CoordinateTransformations.Translation) =
    (t = _static_translation(Tx.translation); to_homogeneous(_identity_linear(t), t))
to_homogeneous(L::CoordinateTransformations.LinearMap) =
    (A = _static_linear(L.linear); to_homogeneous(A, _zero_translation(A)))
to_homogeneous(Af::CoordinateTransformations.AffineMap) =
    to_homogeneous(_static_linear(Af.linear), _static_translation(Af.translation))
to_homogeneous(E::PlainEuclideanMap{N, T}) where {N, T} = to_homogeneous(SMatrix{N, N, T}(E.R), E.t)
# A physical map has no plain matrix: refused once here, for every lift and
# bridge that reads one (HomographyMap(E), E ∘ H, E ∘ a model matrix, a warp).
to_homogeneous(::UnitfulEuclideanMap) =
    _refuse_unitful("to_homogeneous (and so HomographyMap(E), ∘ with a homography or a model matrix)")

# A composition: the product of its factors' matrices, in either spelling
# (Base's `∘` of functions, CoordinateTransformations' `∘` of transformations)
to_homogeneous(C::ComposedFunction) = to_homogeneous(C.outer) * to_homogeneous(C.inner)
to_homogeneous(C::CoordinateTransformations.ComposedTransformation) = to_homogeneous(C.t1) * to_homogeneous(C.t2)

# ========================================================================
# The inverse of an affine-family map, as its homogeneous matrix
# ========================================================================
#
# THE inverse every affine-class action and lift reads (a conic's congruence,
# a line's cotransform, HomographyMap's stored inverse, a model matrix's
# correlation): analytic, static (a Matrix-backed map is made static first —
# never a heap inverse per primitive) and checked. A singular or non-finite
# linear part is an ArgumentError, so no NaN or Inf inverse is ever stored or
# handed on as a valid conic or line.
@noinline _throw_singular_affine(A) = throw(
    ArgumentError("the map's linear part is singular or non-finite, so it has no inverse: $(A)")
)
@inline function _checked_linear_inverse(A::StaticMatrix)
    Ai = inv(A)   # StaticArrays' closed form (adjugate / det) for N ≤ 3: Inf or NaN when singular
    (all(isfinite, A) && all(isfinite, Ai)) || _throw_singular_affine(A)
    return Ai
end
_inverse_homogeneous(Tx::CoordinateTransformations.Translation) =
    (t = _static_translation(Tx.translation); to_homogeneous(_identity_linear(t), -t))
_inverse_homogeneous(L::CoordinateTransformations.LinearMap) =
    (Ai = _checked_linear_inverse(_static_linear(L.linear)); to_homogeneous(Ai, _zero_translation(Ai)))
function _inverse_homogeneous(Af::CoordinateTransformations.AffineMap)
    Ai = _checked_linear_inverse(_static_linear(Af.linear))
    return to_homogeneous(Ai, -(Ai * _static_translation(Af.translation)))
end
# the rigid and similarity maps invert from their parts (never singular)
_inverse_homogeneous(m::Union{EuclideanMap, SimilarityMap}) = to_homogeneous(inv(m))

# ========================================================================
# from_homogeneous: 3×3 homogeneous → AffineMap (2×2 linear + 2D offset)
# ========================================================================

"""
    affine_part(M::StaticMatrix{3,3}) -> AffineMap

Extract the 2×2 linear part and 2D translation from a 3×3 homogeneous matrix.
The bottom row is not read; for a camera calibration matrix `K` it is the
pixel affine of the imaging chain, `affine_part(K) ∘ lens`.

Works with any 3×3 static matrix: a `CameraCalibrationMat`, a plain
`SMatrix{3,3}` (`to_homogeneous` of an affine map), etc.

# Example
```julia
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
am = affine_part(K)  # AffineMap([800 0; 0 800], [320, 240])
```
"""
function affine_part(M::StaticMatrix{3, 3, T}) where {T}
    A = SMatrix{2, 2, T}(M[1, 1], M[2, 1], M[1, 2], M[2, 2])
    t = SVector{2, T}(M[1, 3], M[2, 3])
    return AffineMap(A, t)
end

# Note: `to_affine` and `to_euclidean` are defined in base/coordinates.jl
# (loaded before Primitives) so primitive conic gradients can use them.
