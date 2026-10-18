# ========================================================================
# Transform Matrix Types and Their Hierarchy
# ========================================================================
#
# This file defines:
# - Homogeneous matrix wrapper types (HomRotationMat, HomTranslationMat, HomAffineMat, etc.)
# - Abstract matrix hierarchy for compile-time transform composition
# - Trait-based multiplication and result type inference
#
# NOTE: Conic matrix types (HomCircleMat, HomEllipseMat) are defined in
# base/types.jl and loaded early to break circular dependencies.

# --- Abstract parent type for transforms ---
# (Conic abstract type and @smatrix_wrapper macro are in base/types.jl)

# =======================================================================
# Abstract type hierarchy — captures the natural subgroup chain of planar
# transforms.  Each leaf type has exactly one position in the chain:
#
#   AbstractHomMat                                    (any 3×3 in homog coords)
#   ├── HomographyMat                                 (general projective; only direct child)
#   └── AbstractAffineMat                             (last row [0,0,1])
#       ├── HomAffineMat                              (general 6-DOF affine)
#       ├── HomScalingMat                             (anisotropic — not a similarity)
#       ├── AbstractCalibrationMat                    (camera K family)
#       │   ├── CameraCalibrationMat                  (in cameras/intrinsics.jl)
#       └── AbstractSimilarityMat                     (uniform scale + rigid)
#           ├── HomSimilarityMat
#           ├── HomUniformScalingMat
#           └── AbstractEuclideanMat                  (rigid only)
#               ├── HomEuclideanMat
#               ├── HomRotationMat
#               └── HomTranslationMat
#
# Result-type promotion under composition (`a * b`) walks UP the chain to
# the lowest common ancestor — see Base.promote_rule definitions further
# down in this file.
# =======================================================================

"Abstract parent for all 3×3 homogeneous transform matrices."
abstract type AbstractHomMat{T} <: StaticArrays.StaticMatrix{3, 3, T} end

"Affine transform (last row [0,0,1])."
abstract type AbstractAffineMat{T} <: AbstractHomMat{T} end

"Similarity: rotation + translation + uniform scaling (preserves angles)."
abstract type AbstractSimilarityMat{T} <: AbstractAffineMat{T} end

"Euclidean (rigid): rotation + translation (preserves distances)."
abstract type AbstractEuclideanMat{T} <: AbstractSimilarityMat{T} end

"Camera calibration matrix family (K = diag(scaling) + translation)."
abstract type AbstractCalibrationMat{T} <: AbstractAffineMat{T} end

# Reparent the leaves at their tightest abstract level.
@smatrix_wrapper HomRotationMat        AbstractEuclideanMat
@smatrix_wrapper HomTranslationMat     AbstractEuclideanMat
@smatrix_wrapper HomEuclideanMat       AbstractEuclideanMat
@smatrix_wrapper HomUniformScalingMat  AbstractSimilarityMat
@smatrix_wrapper HomSimilarityMat      AbstractSimilarityMat
@smatrix_wrapper HomScalingMat         AbstractAffineMat       # anisotropic — not a similarity
@smatrix_wrapper HomAffineMat          AbstractAffineMat
@smatrix_wrapper HomographyMat         AbstractHomMat          # only direct child of AbstractHomMat (not affine)

# Docstrings for transform matrix types (must come after macro expansion)
@doc "2D rotation in homogeneous coordinates (3×3 matrix)." HomRotationMat
@doc "2D translation in homogeneous coordinates (3×3 matrix)." HomTranslationMat
@doc "Uniform (isotropic) 2D scaling in homogeneous coordinates. Matches Julia's `UniformScaling`." HomUniformScalingMat
@doc "Diagonal (non-uniform) 2D scaling in homogeneous coordinates." HomScalingMat
@doc "Euclidean transform: rotation + translation (rigid body motion)." HomEuclideanMat
@doc "Similarity transform: rotation + translation + uniform scaling." HomSimilarityMat
@doc "Affine transform: rotation + translation + scaling + shear." HomAffineMat
@doc "Planar projective homography (most general 2D transform)." HomographyMat

# NOTE: Conic matrices (HomEllipseMat, HomCircleMat) are defined in base/types.jl

# ------------------------------------
# Convenience Constructors
# ------------------------------------

"""
    HomTranslationMat(t::StaticVector{2})

Construct a 3×3 homogeneous translation matrix from a 2D translation vector.

# Arguments
- `t::StaticVector{2}`: 2D translation vector (unitless)

# Returns
- `HomTranslationMat{T}`: 3×3 homogeneous translation matrix

# Example
```julia
t = Vec2(1.0, 2.0)
H = HomTranslationMat(t)  # Creates [1 0 1; 0 1 2; 0 0 1]
```
"""
function HomTranslationMat(t::StaticVector{2, T}) where {T}
    return HomTranslationMat{T}(
        Tuple(
            @SMatrix [
                1.0 0.0 t[1]
                0.0 1.0 t[2]
                0.0 0.0 1.0
            ]
        )
    )
end

"""
    HomUniformScalingMat{N}(s::Real)

Construct a homogeneous uniform scaling matrix for N-dimensional transformations.
Matches Julia's `LinearAlgebra.UniformScaling` naming convention.

# Type Parameter
- `N`: Spatial dimension (2 for 2D → 3×3 matrix, 3 for 3D → 4×4 matrix)

# Arguments
- `s::Real`: Uniform scale factor

# Returns
- `HomUniformScalingMat{T}`: Homogeneous scaling matrix (size N+1 × N+1)

# Examples
```julia
H = HomUniformScalingMat{2}(2.5)  # 2D: [2.5 0 0; 0 2.5 0; 0 0 1]
H = HomUniformScalingMat{3}(2.5)  # 3D: 4×4 matrix (not yet implemented)
```
"""
function (::Type{HomUniformScalingMat{N}})(s::T) where {N, T <: Real}
    if N == 2
        # 2D: Create 3×3 homogeneous scaling matrix
        return HomUniformScalingMat{T}(
            Tuple(
                @SMatrix [
                    s 0.0 0.0
                    0.0 s 0.0
                    0.0 0.0 1.0
                ]
            )
        )
    elseif N == 3
        # 3D: 4×4 homogeneous scaling matrix (not yet supported by current type)
        error("3D homogeneous scaling (4×4 matrices) not yet supported. Current HomUniformScalingMat is 3×3 only.")
    else
        error("Unsupported dimension: $N. Only 2 and 3 are supported.")
    end
end

"""
    HomRotationMat{N}(R::SMatrix{M,M,T}) where {N,M,T}

Flexible constructor for HomRotationMat that handles both dimension cases:
- If N == M: Direct construction from already-homogeneous matrix (default case)
- If N == M + 1: Lift (M×M) rotation to (N×N) homogeneous form (convenience)

# Type Parameter
- `N`: Target homogeneous dimension (3 for 2D rotations)

# Arguments
- `R::SMatrix{M,M,T}`: Input rotation matrix (either M×M or (M-1)×(M-1))

# Returns
- `HomRotationMat{promote_type(T,Float64)}`: Homogeneous rotation matrix

# Examples
```julia
# Lift 2×2 rotation to 3×3 homogeneous
R_2x2 = @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)]
H = HomRotationMat{3}(R_2x2)  # Automatically lifts 2×2 → 3×3

# Direct construction from already-homogeneous 3×3
R_3x3 = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
H = HomRotationMat{3}(R_3x3)  # Direct construction
```
"""
function (::Type{HomRotationMat{N}})(R::SMatrix{M, M, T}) where {N, M, T}
    T_out = promote_type(T, Float64)

    if N == M
        # Already homogeneous dimension - direct construction (common case)
        return HomRotationMat{T_out}(Tuple(R))
    elseif N == M + 1
        # Lift to homogeneous dimension (convenience case)
        # Use to_homogeneous with zero translation
        H = to_homogeneous(R, zero(SVector{M, T_out}))
        return HomRotationMat{T_out}(Tuple(H))
    else
        error(
            "Dimension mismatch: cannot construct HomRotationMat{$N} from $(M)×$(M) matrix. " *
                "Expected either $N×$N (direct) or $(N - 1)×$(N - 1) (lift) matrix."
        )
    end
end

# NOTE (2026-07, #18): the TransformTrait/IsX classification family is
# GONE.  The abstract hierarchy (AbstractEuclideanMat ⊂ … ⊂ AbstractHomMat)
# is the source of truth for VGC's own matrix types, and the single real
# consumer (CT.LinearMap on Circle) needed only a circle-preservation
# predicate — see `_preserves_circles` in ct_contravariant.jl.

# InvertibleTrait for transform matrices (all homogeneous transforms are invertible)

# ------------------------------------
# Result Type Computation via Hierarchy-Driven `Base.promote_rule`
# ------------------------------------
#
# The abstract hierarchy (AbstractEuclideanMat ⊂ AbstractSimilarityMat ⊂
# AbstractAffineMat ⊂ AbstractHomMat) captures the natural subgroup chain
# of planar transforms.  Promotion under composition walks UP the chain
# to the lowest common ancestor and returns its canonical concrete type.
# Julia's most-specific dispatch finds the right level automatically.
#
# Five rules per parameter pair:
#   - leaves with same nominal type  → preserve (e.g. Rot × Rot → Rot)
#   - both Euclidean  → HomEuclideanMat
#   - both Similarity → HomSimilarityMat
#   - both Affine     → HomAffineMat
#   - any homogeneous → HomographyMat
#
# Self-rules for each leaf preserve identity under same-type composition.

for L in (
        :HomRotationMat, :HomTranslationMat, :HomUniformScalingMat,
        :HomScalingMat, :HomEuclideanMat, :HomSimilarityMat, :HomAffineMat,
        :HomographyMat,
    )
    @eval Base.promote_rule(::Type{<:$L{T1}}, ::Type{<:$L{T2}}) where {T1, T2} =
        $L{promote_type(T1, T2)}
end

# Cross-leaf rules — Julia picks the most specific abstract level both args satisfy.
Base.promote_rule(
    ::Type{<:AbstractEuclideanMat{T1}},
    ::Type{<:AbstractEuclideanMat{T2}}
) where {T1, T2} =
    HomEuclideanMat{promote_type(T1, T2)}

Base.promote_rule(
    ::Type{<:AbstractSimilarityMat{T1}},
    ::Type{<:AbstractSimilarityMat{T2}}
) where {T1, T2} =
    HomSimilarityMat{promote_type(T1, T2)}

Base.promote_rule(
    ::Type{<:AbstractAffineMat{T1}},
    ::Type{<:AbstractAffineMat{T2}}
) where {T1, T2} =
    HomAffineMat{promote_type(T1, T2)}

Base.promote_rule(
    ::Type{<:AbstractHomMat{T1}},
    ::Type{<:AbstractHomMat{T2}}
) where {T1, T2} =
    HomographyMat{promote_type(T1, T2)}

# NOTE: similar_type for transform matrices is generated by @smatrix_wrapper macro
# NOTE: Conic matrix similar_type is defined in base/types.jl

# Single multiplication function — dispatches on abstract type, uses
# `Base.promote_type` (the hierarchy promote_rule chain) for the result type.
function Base.:*(a::AbstractHomMat, b::AbstractHomMat)
    ResultType = promote_type(typeof(a), typeof(b))     # parametric, e.g. HomEuclideanMat{Float64}
    T = eltype(ResultType)
    C = SMatrix{3, 3, T}(a) * SMatrix{3, 3, T}(b)
    return ResultType(Tuple(C))
end

# Inverse - preserves matrix type (we don't reclassify)
Base.inv(A::AbstractHomMat) = similar_type(typeof(A), eltype(A))(Tuple(inv(SMatrix{3, 3, eltype(A)}(A))))

# =======================================================================
# AbstractMatrixMap — parent for callable Maps that have a matrix form
# =======================================================================
#
# Distinguishes "transforms whose action reduces to a homogeneous matrix"
# (HomographyMap, EuclideanMap) from non-matrix transforms (BoundedMap,
# lens models).  Every concrete subtype must implement
# `to_homogeneous(::M)` returning the homogeneous matrix form.
#
# NOTE (2026-07, #18): the AbstractProjectiveMap{N,T} /
# AbstractAffineSpaceMap{N,T} middle layers are GONE — each had exactly
# one subtype and zero dispatch consumers.

"Parent for `CT.Transformation`s that have a homogeneous matrix form."
abstract type AbstractMatrixMap <: CoordinateTransformations.Transformation end
