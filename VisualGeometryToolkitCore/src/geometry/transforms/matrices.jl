# ========================================================================
# Transform Matrices: the one dense algebraic homography type
# ========================================================================
#
# Maps are the geometric objects (EuclideanMap, SimilarityMap, the
# CoordinateTransformations affines, HomographyMap): their TYPE states the
# transform class (Euclidean ⊂ similarity ⊂ affine ⊂ projective), and each
# acts on primitives exactly as its class allows (ct_contravariant.jl,
# homography_map.jl). Matrices are for algebra only: `HomographyMat` is a
# dense 3×3 (a solver's output, a DLT's), and `to_homogeneous(map)` is a
# plain `SMatrix` (conversions.jl). The 0.5 hierarchy of typed homogeneous
# matrices, whose promotion rules carried the class, is retired in 0.6 (the
# names are listed in `check_naming`).
#
# NOTE: Conic matrix types (HomCircleMat, HomEllipseMat) are defined in
# base/types.jl and loaded early to break circular dependencies.

@smatrix_wrapper HomographyMat 3 3

@doc """
    HomographyMat{T} <: StaticMatrix{3, 3, T}

A planar projective homography as a dense 3×3 matrix — algebra (a solver's,
a DLT's output), frame-free. The geometric object is its
[`HomographyMap`](@ref), which carries the inverse and the frame tags. The
product and the inverse of homographies are homographies.
""" HomographyMat

# Homographies of two eltypes promote to the homography of the promoted
# eltype (a Float32 and a Float64 one collect into a Vector{HomographyMat{Float64}}).
Base.promote_rule(::Type{HomographyMat{T}}, ::Type{HomographyMat{S}}) where {T, S} = HomographyMat{promote_type(T, S)}

Base.:*(a::HomographyMat, b::HomographyMat) =
    HomographyMat{promote_type(eltype(a), eltype(b))}(Tuple(SMatrix{3, 3}(a) * SMatrix{3, 3}(b)))
Base.inv(H::HomographyMat{T}) where {T} = HomographyMat(inv(SMatrix{3, 3, T, 9}(H)))

# =======================================================================
# AbstractMatrixMap — parent for callable Maps that have a matrix form
# =======================================================================
#
# Distinguishes "transforms whose action reduces to a homogeneous matrix"
# (HomographyMap, EuclideanMap, SimilarityMap) from non-matrix transforms
# (BoundedMap, lens models).  Every concrete subtype must implement
# `to_homogeneous(::M)` returning the homogeneous matrix form.

"Parent for `CT.Transformation`s that have a homogeneous matrix form."
abstract type AbstractMatrixMap <: CoordinateTransformations.Transformation end
