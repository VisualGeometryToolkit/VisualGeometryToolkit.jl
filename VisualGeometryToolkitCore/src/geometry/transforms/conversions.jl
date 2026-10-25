# ========================================================================
# Coordinate Conversions
# ========================================================================

"""
    to_homogeneous(M::SMatrix{2,2,T}, t::SVector{2,T}) -> SMatrix{3,3,T}

Convert 2×2 linear transformation matrix and 2D translation vector to 3×3 homogeneous matrix.

# Arguments
- `M::SMatrix{2,2,T}`: 2×2 linear transformation matrix
- `t::SVector{2,T}`: 2D translation vector

# Returns
- `SMatrix{3,3,T}`: 3×3 homogeneous transformation matrix

# Example
```julia
M = @SMatrix [2.0 0.0; 0.0 1.5]  # scaling
t = SVector(1.0, 2.0)            # translation
H = to_homogeneous(M, t)         # 3×3 homogeneous matrix
```
"""
@inline to_homogeneous(M::StaticMatrix{2, 2, T}, t::StaticVector{2, T}) where {T} =
    @SMatrix [
    M[1, 1] M[1, 2] t[1]
    M[2, 1] M[2, 2] t[2]
    zero(T) zero(T) one(T)
]

# -------------------------------------------------
# Bridge: CoordinateTransformations / Rotations -> typed homogeneous
# -------------------------------------------------

# primitives
to_homogeneous(R::Rotations.Rotation{2, T}) where {T} =
    HomRotationMat{T}(Tuple(to_homogeneous(SMatrix{2, 2, T}(Matrix(R)), zero(Vec{2, T}))))

to_homogeneous(Tx::CoordinateTransformations.Translation{<:SVector{2, T}}) where {T} =
    HomTranslationMat{T}(
    Tuple(
        @SMatrix [
            1.0 0.0 Tx.translation[1]
            0.0 1.0 Tx.translation[2]
            0.0 0.0 1.0
        ]
    )
)

# LinearMap: Type-based dispatch for specific matrix types
# Uniform scaling (matches Julia's LinearAlgebra.UniformScaling)
to_homogeneous(L::CoordinateTransformations.LinearMap{<:UniformScaling{T}}) where {T} =
    HomUniformScalingMat{T}(Tuple(to_homogeneous(SMatrix{2, 2, T}(L.linear), zero(Vec{2, T}))))

# Diagonal scaling
function to_homogeneous(L::CoordinateTransformations.LinearMap{<:Diagonal{T}}) where {T}
    sx, sy = L.linear.diag[1], L.linear.diag[2]
    M = SMatrix{2, 2, T}(L.linear)
    if sx == sy
        return HomUniformScalingMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
    else
        return HomScalingMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
    end
end

# Rotation matrices (from Rotations.jl)
to_homogeneous(L::CoordinateTransformations.LinearMap{<:Rotations.RotMatrix{2, T}}) where {T} =
    HomRotationMat{T}(Tuple(to_homogeneous(SMatrix{2, 2, T}(L.linear), zero(Vec{2, T}))))

# General matrix - fallback with runtime checks
function to_homogeneous(L::CoordinateTransformations.LinearMap{Mat}) where {Mat <: AbstractMatrix}
    T = eltype(L.linear)
    M = SMatrix{2, 2, T}(L.linear)
    if iszero(M[1, 2]) && iszero(M[2, 1])
        sx, sy = M[1, 1], M[2, 2]
        if sx == sy
            return HomUniformScalingMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
        else
            return HomScalingMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
        end
    else
        return HomAffineMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
    end
end

# EuclideanMap: Rotation + Translation
to_homogeneous(E::PlainEuclideanMap{2, T}) where {T} = begin
    M = SMatrix{2, 2, T}(Matrix(E.R))
    t = SVector{2, T}(E.t)
    HomEuclideanMat{T}(Tuple(to_homogeneous(M, t)))
end

# AffineMap: ALWAYS Affine
to_homogeneous(Af::CoordinateTransformations.AffineMap{Mat, V}) where {Mat <: AbstractMatrix, V <: AbstractVector} = begin
    T = promote_type(eltype(Af.linear), eltype(Af.translation))
    M = SMatrix{2, 2, T}(Af.linear)
    t = SVector{2, T}(Af.translation)
    HomAffineMat{T}(Tuple(to_homogeneous(M, t)))
end

# Compositions: ALWAYS Affine (we do not reclassify composites)
to_homogeneous(C::ComposedFunction) = begin
    A = to_homogeneous(C.outer)
    B = to_homogeneous(C.inner)
    T = promote_type(eltype(A), eltype(B))
    H = SMatrix{3, 3, T}(Tuple(A)) * SMatrix{3, 3, T}(Tuple(B))
    HomAffineMat{T}(Tuple(H))
end

# Identity methods for already-homogeneous matrices
to_homogeneous(H::AbstractHomMat) = H

# -------------------------------------------------
# Constructors: foreign types → typed homogeneous
# -------------------------------------------------
#
# Each leaf constructor returns its own type (no type lies).  AffineMap and
# ComposedFunction stay HomAffineMat because they ARE general affines with
# no narrower structure to recover.  For "give me the tightest VGC type for
# this CT input," call `AbstractHomMat(x)` — it dispatches through
# `to_homogeneous`, which inspects the inner matrix and picks accordingly
# (Translation → HomTranslationMat, LinearMap{<:RotMatrix} → HomRotationMat,
# LinearMap{<:UniformScaling} → HomUniformScalingMat, etc.).

# Tight per-leaf constructors
HomTranslationMat(Tx::CoordinateTransformations.Translation{<:SVector{2}}) =
    to_homogeneous(Tx)

HomRotationMat(R::Rotations.Rotation{2}) = to_homogeneous(R)
HomRotationMat(L::CoordinateTransformations.LinearMap{<:Rotations.RotMatrix{2}}) =
    to_homogeneous(L)

HomUniformScalingMat(L::CoordinateTransformations.LinearMap{<:UniformScaling}) =
    to_homogeneous(L)

# Force anisotropic interpretation even when the diagonal is uniform —
# `to_homogeneous` may collapse to HomUniformScalingMat when sx == sy, but
# this constructor honors the caller's explicit choice.
function HomScalingMat(L::CoordinateTransformations.LinearMap{<:Diagonal{T}}) where {T}
    M = SMatrix{2, 2, T}(L.linear)
    return HomScalingMat{T}(Tuple(to_homogeneous(M, zero(Vec{2, T}))))
end

HomEuclideanMat(E::PlainEuclideanMap{2}) = to_homogeneous(E)

# General-affine inputs stay HomAffineMat (no narrower category exists).
HomAffineMat(Af::CoordinateTransformations.AffineMap) = to_homogeneous(Af)
HomAffineMat(C::ComposedFunction) = to_homogeneous(C)

# Smart generic constructor — pick the tightest VGC type via `to_homogeneous`.
# Return type is the appropriate AbstractHomMat subtype, determined by the
# argument's structure.  Use this at boundaries where you don't know (or
# don't want to commit to) the input category.
#
#   AbstractHomMat(Translation([1.0, 2.0]))                      → HomTranslationMat
#   AbstractHomMat(LinearMap(RotMatrix{2}(0.5)))                 → HomRotationMat
#   AbstractHomMat(LinearMap(UniformScaling(2.0)))               → HomUniformScalingMat
#   AbstractHomMat(LinearMap(Diagonal(SA[2.0, 3.0])))            → HomScalingMat
#   AbstractHomMat(AffineMap(...))                               → HomAffineMat
(::Type{AbstractHomMat})(x::CoordinateTransformations.Transformation) = to_homogeneous(x)

# ========================================================================
# from_homogeneous: 3×3 homogeneous → AffineMap (2×2 linear + 2D offset)
# ========================================================================

"""
    AffineMap(M::StaticMatrix{3,3}) -> AffineMap

Extract the 2×2 linear part and 2D translation from a 3×3 homogeneous matrix.

Works with any 3×3 static matrix: `HomAffineMat`, `CameraCalibrationMat`,
plain `SMatrix{3,3}`, etc.

# Example
```julia
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
am = AffineMap(K)  # AffineMap([800 0; 0 800], [320, 240])
```
"""
function CoordinateTransformations.AffineMap(M::StaticMatrix{3, 3, T}) where {T}
    A = SMatrix{2, 2, T}(M[1, 1], M[2, 1], M[1, 2], M[2, 2])
    t = SVector{2, T}(M[1, 3], M[2, 3])
    return AffineMap(A, t)
end

# Note: `to_affine` and `to_euclidean` are defined in base/coordinates.jl
# (loaded before Primitives) so primitive conic gradients can use them.
