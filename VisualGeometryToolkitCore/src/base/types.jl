# =============================================================================
# Base Types for VisualGeometryToolkitCore
# =============================================================================
#
# This file defines foundational types loaded early to break circular dependencies:
# - Interface definitions (traits and function stubs)
# - Abstract detection types (FeatureFrame, AbstractDetection)
# - Conic matrix types (HomEllipseMat, HomCircleMat)
# - Attribute composition (Attributed) — fit RESULTS are FitResult (base/fit_result.jl)
#
# These are loaded before submodules so all modules can use them.

# ========================================================================
# Interface Definitions (traits and function stubs)
# ========================================================================
include("interface.jl")

# ========================================================================
# Abstract Detection Types
# ========================================================================
#
# These abstract types allow:
# - Primitives module to use AbstractDetection for Circle(detection, cutoff)
# - Features module to define DiscDetection <: AbstractDetection
# Without creating circular dependencies.

"""
    FeatureFrame{T}

Abstract type for geometric feature frames.

A feature frame defines a transformation from canonical space (unit circle/square)
to image space. The transformation encodes the feature's position, scale, and shape.

Concrete subtypes:
- `Disc{T}`: 3 DOF - isotropic (translation + uniform scale)
- `OrientedDisc{T}`: 4 DOF - isotropic with orientation
- `EllipticFrame{T}`: 5 DOF - anisotropic (symmetric shape factor)
- `AffineFrame{T}`: 6 DOF - full affine (general shape factor)

# Interface
All frames implement:
- `AffineMap(frame)` → `AffineMap` from canonical to image coordinates
- `region(frame, k=1.0)` → unitless `Circle` or `Ellipse` at the k·σ contour
- `center(frame)` → `Point2{T}` center position
- `Base.inv(frame)` → frame with inverted transformation
"""
abstract type FeatureFrame{T} end

# ONE `center` generic for every centred geometric object: `FeatureFrame`
# (geometry/primitives/blob_types.jl) and the radial models
# (geometry/radial_models.jl). Declared here because Primitives is included
# before the geometry layer that documents the family.
function center end


# ========================================================================
# Static matrix types: the conic matrices, the epipolar matrices
# ========================================================================
#
# Defined here, before the Primitives submodule (which builds conics from its
# Ellipse and Circle) and the transforms (which act on them).

using StaticArrays: StaticArrays, SMatrix, StaticMatrix, MVector, StaticVector

"Abstract parent for all 3×3 homogeneous conic matrices."
abstract type AbstractHomConicMat{T} <: StaticArrays.StaticMatrix{3, 3, T} end

# --- Generic static matrix wrapper macro ---
# Creates a parametric StaticMatrix wrapper type with efficient tuple storage
# Supports three forms:
#   @smatrix_wrapper Name Parent        - inherit from parametric abstract type (3×3)
#   @smatrix_wrapper Name m n           - inherit from StaticMatrix{m,n,T}
#   @smatrix_wrapper Name N             - N-parametric square wrapper {N,T,L}
macro smatrix_wrapper(Name, arg2, arg3 = nothing)
    name = esc(Name)
    if arg2 === :N && arg3 === :Frame
        # N-parametric square form holding a static matrix, with a trailing frame tag:
        # Name{N,T,M,C<:Frame{N}} <: StaticMatrix{N,N,T}. Used by the
        # statistical forms (CovMat/PrecMat, 0.9): the frame parameter names
        # WHICH frame the matrix's coordinates live in, defaulting
        # to `UnlabeledFrame{N}` through every historically-shaped constructor
        # (the standing gradual-typing rule — labels are opt-in, never viral).
        argerr_msg = string(Name, "{N,T} takes N*N args, got ")
        return quote
            # The form HOLDS a static matrix, and the matrix's type is its structure:
            # an `SMatrix` (full), an `SDiagonal` (diagonal) or a `ScaledIdentity`
            # (λ·I, isotropic) — the three a `GaussianBlur` holds. `parent(A)` is the
            # matrix, and structured arithmetic is the matrix's own (an SDiagonal sum
            # is an SDiagonal); the form itself inverts, relabels, converts and
            # promotes, and every other operation sees its full N × N entries and,
            # like any product J*Σ, escapes to an SMatrix (not a covariance).
            struct $name{N, T, M <: _FormStorage{N, T}, C <: Frame{N}} <: StaticArrays.StaticMatrix{N, N, T}
                matrix::M
                # Typed inner constructor (an argument of the storage union): an
                # untyped `Name{…}(m)` is ambiguous against StaticArrays' generic
                # `(::Type{<:StaticArray})(::AbstractArray)` converter.
                $name{N, T, M, C}(m::_FormStorage{N, T}) where {N, T, M, C <: Frame{N}} = new{N, T, M, C}(m)
            end
            Base.parent(A::$name) = getfield(A, :matrix)
            @inline Base.size(::Type{<:$name{N}}) where {N} = (N, N)
            # similar_type is what StaticArrays builds an eltype conversion as: the
            # full form, its frame kept.
            @inline StaticArrays.similar_type(::Type{$name{N, T, M, C}}, ::Type{S}) where {N, T, M, C, S} =
                $name{N, S, StaticArrays.SMatrix{N, N, S, N * N}, C}
            @inline StaticArrays.similar_type(::Type{<:$name{N}}, ::Type{S}) where {N, S} =
                $name{N, S, StaticArrays.SMatrix{N, N, S, N * N}, UnlabeledFrame{N}}
            @inline Base.eltype(::Type{<:$name{N, T}}) where {N, T} = T
            @inline Base.IndexStyle(::Type{<:$name}) = IndexLinear()
            @inline Base.getindex(A::$name, k::Int) = getfield(A, :matrix)[k]
            @inline Base.getindex(A::$name, i::Int, j::Int) = getfield(A, :matrix)[i, j]
            # every entry, column-major (what StaticArrays converts through)
            @inline Base.Tuple(A::$name{N, T}) where {N, T} = Tuple(StaticArrays.SMatrix{N, N, T}(getfield(A, :matrix)))
            # The unlabeled form of a static matrix: its structure is its type.
            @inline $name(m::StaticArrays.SMatrix{N, N, T}) where {N, T} = $name{N, T, typeof(m), UnlabeledFrame{N}}(m)
            @inline $name(m::LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}) where {N, T} = $name{N, T, typeof(m), UnlabeledFrame{N}}(m)
            @inline $name(m::ScaledIdentity{N, T}) where {N, T} = $name{N, T, typeof(m), UnlabeledFrame{N}}(m)
            # The full form of entries (column-major) or of any matrix of the size
            @inline $name{N, T}(data::NTuple{L}) where {N, T, L} = $name(StaticArrays.SMatrix{N, N, T}(data))
            @inline function $name{N, T}(args::Vararg{T, L}) where {N, T, L}
                L == N * N || throw(ArgumentError(string($argerr_msg, L)))
                return $name(StaticArrays.SMatrix{N, N, T}(args))
            end
            @inline $name{N, T}(m::AbstractMatrix) where {N, T} = $name(StaticArrays.SMatrix{N, N, T}(m))
            # Any 2-D StaticArray (the full AbstractMatrix ∩ StaticArray intersection):
            # more specific than StaticArrays' `(::Type{SA})(::StaticArray)` converter.
            @inline $name{N, T}(m::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) where {N, T} = $name(StaticArrays.SMatrix{N, N, T}(m))
            # a structured matrix in another eltype keeps its structure
            @inline $name{N, T}(m::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{N}}) where {N, T} = $name(_with_eltype(T, m))
            @inline $name{N, T}(m::ScaledIdentity{N}) where {N, T} = $name(_with_eltype(T, m))
            # Eltype conversion of a labeled form keeps its frame and its structure
            # (a label is never silently erased — the same rule as `convert`).
            @inline function $name{N, T}(A::$name{N, S, M, C}) where {N, T, S, M, C <: Frame{N}}
                m = _with_eltype(T, getfield(A, :matrix))
                return $name{N, T, typeof(m), C}(m)
            end
            # What StaticArrays builds through similar_type: the full form from its entries.
            @inline (::Type{$name{N, T, M, C}})(data::Tuple) where {N, T, M <: StaticArrays.SMatrix, C <: Frame{N}} =
                $name{N, T, M, C}(M(data))
        end
    elseif arg3 === nothing && arg2 === :N
        # N-parametric square form: Name{N,T,L} <: StaticMatrix{N,N,T}
        argerr_msg = string(Name, "{N,T} takes N*N args, got ")
        return quote
            struct $name{N, T, L} <: StaticArrays.StaticMatrix{N, N, T}
                data::NTuple{L, T}
                $name{N, T, L}(data::NTuple{L}) where {N, T, L} = new{N, T, L}(convert(NTuple{L, T}, data))
            end
            @inline Base.size(::Type{<:$name{N}}) where {N} = (N, N)
            @inline StaticArrays.similar_type(::Type{<:$name{N, T, L}}, ::Type{S}) where {N, T, L, S} =
                $name{N, S, L}
            @inline StaticArrays.similar_type(::Type{<:$name{N}}, ::Type{S}) where {N, S} =
                $name{N, S, N * N}
            @inline Base.eltype(::Type{<:$name{N, T}}) where {N, T} = T
            @inline Base.IndexStyle(::Type{<:$name}) = IndexLinear()
            @inline Base.getindex(A::$name, k::Int) = A.data[k]
            @inline Base.getindex(A::$name{N}, i::Int, j::Int) where {N} = A.data[(j - 1) * N + i]
            @inline Base.Tuple(A::$name) = A.data
            @inline $name{N, T}(data::NTuple{L, T}) where {N, T, L} = $name{N, T, L}(data)
            @inline function $name{N, T}(args::Vararg{T, L}) where {N, T, L}
                L == N * N || throw(ArgumentError(string($argerr_msg, L)))
                return $name{N, T, L}(args)
            end
            @inline $name(M::SMatrix{N, N, T, L}) where {N, T, L} = $name{N, T, L}(Tuple(M))
            @inline $name{N, T}(M::SMatrix{N, N, T, L}) where {N, T, L} = $name{N, T, L}(Tuple(M))
            @inline $name{N, T}(M::AbstractMatrix) where {N, T} = $name{N, T}(SMatrix{N, N, T}(M))
            @inline $name{N, T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) where {N, T} = $name{N, T}(SMatrix{N, N, T}(M))
        end
    elseif arg3 === nothing
        # Two-arg form: Name, Parent (abstract type) - assumes 3×3
        parent = esc(arg2)
        quote
            struct $name{T} <: $parent{T}
                data::NTuple{9, T}
                $name{T}(data::NTuple{9}) where {T} = new{T}(convert(NTuple{9, T}, data))
            end
            $name(data::NTuple{9, T}) where {T} = $name{T}(data)
            StaticArrays.size(::Type{$name{T}}) where {T} = (3, 3)
            # similar_type for parametric form (Name{T} → Name{S})
            StaticArrays.similar_type(::Type{$name{T}}, ::Type{S}) where {T, S} = $name{S}
            # similar_type for bare type (Name → Name{S}) - used by promote_type chain
            StaticArrays.similar_type(::Type{$name}, ::Type{S}) where {S} = $name{S}
            Base.eltype(::Type{$name{T}}) where {T} = T
            Base.IndexStyle(::Type{$name}) = IndexLinear()
            @inline Base.getindex(A::$name{T}, k::Int) where {T} = A.data[k]
            @inline Base.getindex(A::$name{T}, i::Int, j::Int) where {T} = A.data[(j - 1) * 3 + i]
            Base.Tuple(A::$name{T}) where {T} = A.data
            $name{T}(args::Vararg{T, 9}) where {T} = $name{T}(args)
            $name{T}(M::SMatrix{3, 3, T}) where {T} = $name{T}(Tuple(M))
            # Accept any AbstractMatrix of the right shape; runtime size/eltype check
            # routes through the SMatrix conversion above. Eltype-preserving form first,
            # explicit-T form for callers that want a conversion.
            $name{T}(M::AbstractMatrix) where {T} = $name{T}(SMatrix{3, 3, T}(M))
            $name(M::AbstractMatrix{T}) where {T} = $name{T}(M)
            # Disambiguate vs StaticArrays' generic `(::Type{SA})(sa::StaticArray)`
            # converter: any StaticMatrix routes through the SMatrix path.
            $name{T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) where {T} = $name{T}(SMatrix{3, 3, T}(M))
            $name(M::StaticArrays.StaticArray{<:Tuple, T, 2}) where {T} = $name{T}(M)
        end
    else
        # Three-arg form: Name, m, n (dimensions)
        size_m = arg2
        size_n = arg3
        size_total = size_m * size_n
        quote
            struct $name{T} <: StaticArrays.StaticMatrix{$size_m, $size_n, T}
                data::NTuple{$size_total, T}
                $name{T}(data::NTuple{$size_total}) where {T} = new{T}(convert(NTuple{$size_total, T}, data))
            end
            $name(data::NTuple{$size_total, T}) where {T} = $name{T}(data)
            StaticArrays.size(::Type{$name{T}}) where {T} = ($size_m, $size_n)
            # similar_type for parametric form (Name{T} → Name{S})
            StaticArrays.similar_type(::Type{$name{T}}, ::Type{S}) where {T, S} = $name{S}
            # similar_type for bare type (Name → Name{S}) - used by promote_type chain
            StaticArrays.similar_type(::Type{$name}, ::Type{S}) where {S} = $name{S}
            Base.eltype(::Type{$name{T}}) where {T} = T
            Base.IndexStyle(::Type{$name}) = IndexLinear()
            @inline Base.getindex(A::$name{T}, k::Int) where {T} = A.data[k]
            @inline Base.getindex(A::$name{T}, i::Int, j::Int) where {T} = A.data[(j - 1) * $size_m + i]
            Base.Tuple(A::$name{T}) where {T} = A.data
            $name{T}(args::Vararg{T, $size_total}) where {T} = $name{T}(args)
            $name{T}(M::SMatrix{$size_m, $size_n, T}) where {T} = $name{T}(Tuple(M))
            # Accept any AbstractMatrix of the right shape; runtime size/eltype check
            # routes through the SMatrix conversion above. Eltype-preserving form first,
            # explicit-T form for callers that want a conversion.
            $name{T}(M::AbstractMatrix) where {T} = $name{T}(SMatrix{$size_m, $size_n, T}(M))
            $name(M::AbstractMatrix{T}) where {T} = $name{T}(M)
            # Disambiguate vs StaticArrays' generic `(::Type{SA})(sa::StaticArray)`
            # converter: any StaticMatrix routes through the SMatrix path.
            $name{T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) where {T} = $name{T}(SMatrix{$size_m, $size_n, T}(M))
            $name(M::StaticArrays.StaticArray{<:Tuple, T, 2}) where {T} = $name{T}(M)
        end
    end
end

# ------------------------------------------------------------------------
# Conic matrices: one symmetric-coefficient storage, three classes
# ------------------------------------------------------------------------
#
# A conic matrix is symmetric, so it is stored as its upper triangle,
# column-major: (C₁₁, C₁₂, C₂₂, C₁₃, C₂₃, C₃₃) — the order of the
# point-on-conic constraint's `G_θ` (manifolds/ellipse.jl). Exactly symmetric
# by storage: `Q[i, j]` and `Q[j, i]` read the same coefficient. A matrix
# handed in is checked for symmetry (`isapprox(Q, Qᵀ; rtol = √eps)`) and its
# off-diagonal pairs averaged; an exactly symmetric matrix is stored bit for
# bit. The class (`K`) is the type: HomEllipseMat, HomCircleMat and Conic are
# the one struct `SymmetricConic{K}`, with one constructor set.
#
# Not a `_FormStorage` form (CovMat/PrecMat): a conic's 3×3 is indefinite (a
# real ellipse's has one eigenvalue of the other sign) and homogeneous (every
# nonzero multiple is the same curve), so it has none of a covariance form's
# inverse, frame or definiteness; it is six coefficients and a class.

# column-major entry k of the 3×3 → its coefficient
const _SYM3_INDEX = (1, 2, 4, 2, 3, 5, 4, 5, 6)

# The nine entries, column-major, of the symmetric 3×3 of six coefficients.
@inline _full3(q::NTuple{6}) = ntuple(k -> q[_SYM3_INDEX[k]], Val(9))

# the machine epsilon of a conic computation: Float64's for an AD number or an exact type
_conic_eps(::Type{T}) where {T <: AbstractFloat} = eps(T)
_conic_eps(::Type) = eps(Float64)
# the construction tolerance of a conic's symmetry and a circle's isotropy, from
# the precision the entries were given in
_conic_rtol(::Type{T}) where {T} = sqrt(_conic_eps(T))
# The type a conic's class is decided in: at least Float64. A Float32 conic at
# pixel coordinates holds |centre|²-sized terms whose cancellation leaves γ
# below Float32's rounding floor although its stored coefficients resolve it;
# Float32 arithmetic refused half of all 1–10 px blobs in a 4000×3000 image.
_conic_work_type(::Type{T}) where {T} = float(promote_type(T, Float64))

# The sentinel of an inner constructor that stores its parts as given: the
# caller guarantees the invariants the checked constructors test (a conic
# built symmetric and of its class, a calibration matrix's structure).
struct _Unchecked end

# The coefficients of a 3×3 `Q` in `T`, its off-diagonal pairs averaged in `T` (no check).
@inline _averaged_coefficients(::Type{T}, Q) where {T} = (
    T(Q[1, 1]), _mean_in(T, Q[1, 2], Q[2, 1]), T(Q[2, 2]),
    _mean_in(T, Q[1, 3], Q[3, 1]), _mean_in(T, Q[2, 3], Q[3, 2]), T(Q[3, 3]),
)
@inline _mean_in(::Type{T}, x, y) where {T} = T((T(x) + T(y)) / 2)   # x for x == y

# The coefficients of a symmetric 3×3 `Q` in `T`: its symmetry checked, its
# off-diagonal pairs averaged. The check is relative (every entry divided by the
# largest before any square, so neither 1e160·Q overflows nor 1e-170·Q
# underflows), computed in `float` of the entries, and its tolerance is the
# precision the entries were GIVEN in: Float32 data read into a Conic{Float64}
# carries Float32's asymmetry.
function _symmetric_coefficients(::Type{T}, Q, name) where {T}
    size(Q) == (3, 3) || throw(DimensionMismatch("$name takes a 3×3 matrix, got $(size(Q))"))
    S = float(eltype(Q))
    m = maximum(x -> abs(S(x)), Q)
    isfinite(m) || throw(ArgumentError("$name: the matrix has non-finite entries: $(Q)"))
    rtol = _conic_rtol(eltype(Q))
    if !iszero(m)
        r(i, j) = S(Q[i, j]) / m
        n2 = sum(x -> abs2(S(x) / m), Q)
        d2 = 2 * (abs2(r(1, 2) - r(2, 1)) + abs2(r(1, 3) - r(3, 1)) + abs2(r(2, 3) - r(3, 2)))
        d2 <= rtol^2 * n2 || throw(
            ArgumentError(
                "$name: a conic matrix is symmetric, got one with ‖Q − Qᵀ‖ / ‖Q‖ = $(sqrt(d2 / n2)) " *
                    "(tolerance √eps = $(rtol))"
            )
        )
    end
    return _averaged_coefficients(T, Q)
end

struct SymmetricConic{K, T} <: AbstractHomConicMat{T}
    coeffs::NTuple{6, T}
    SymmetricConic{K, T}(q::NTuple{6}, ::_Unchecked) where {K, T} = new{K, T}(convert(NTuple{6, T}, q))
end

"""
    HomEllipseMat{T} <: AbstractHomConicMat{T}

The homogeneous 3×3 matrix of a real, non-degenerate ellipse
`x̃ᵀ Q x̃ = 0`, `x̃ = (x, y, 1)`, stored as its six unique coefficients
`(Q₁₁, Q₁₂, Q₂₂, Q₁₃, Q₂₃, Q₃₃)` — exactly symmetric by storage; indexing
returns the full symmetric matrix, and arithmetic on it gives an `SMatrix`.
Built from an [`Ellipse`](@ref), a symmetric 3×3 matrix (or its nine entries,
column-major) or the six coefficients. A matrix is checked for symmetry
(`isapprox(Q, Qᵀ; rtol = √eps)`, its off-diagonal pairs averaged), and
`check = true` checks the class — [`classify_conic`](@ref) answering
`EllipseTrait`, the same test — with an `ArgumentError` naming what it is
instead; a general conic is a [`Conic`](@ref).
"""
const HomEllipseMat{T} = SymmetricConic{:ellipse, T}

"""
    HomCircleMat{T} <: AbstractHomConicMat{T}

The homogeneous 3×3 matrix of a real circle, stored and constructed as
[`HomEllipseMat`](@ref) is. Its class check adds isotropy: `Q₁₁ = Q₂₂` and
`Q₁₂ = 0` within `√eps · ‖A‖`.
"""
const HomCircleMat{T} = SymmetricConic{:circle, T}

"""
    Conic{T} <: AbstractHomConicMat{T}

A 2D conic section of any class (ellipse, hyperbola, parabola or
degenerate), stored and constructed as [`HomEllipseMat`](@ref) is, with no
class check. What a [`HomographyMap`](@ref) makes of a circle or an ellipse:
[`classify_conic`](@ref) names its class and `Ellipse(c)` (checked: an
`ArgumentError` for another class) reads the ellipse back; to map a shape and
get its ellipse or `nothing` in one non-throwing call, use
`transfer_ellipse(H, shape)` on the map and the shape instead.

```julia
Q = HomographyMap(M)(circle)   # a Conic
is_ellipse(Q) && Ellipse(Q)
```
"""
const Conic{T} = SymmetricConic{:general, T}

# -- the classes, read from the six coefficients ------------------------------

"""
    classify_conic(Q::Union{Conic, HomEllipseMat, HomCircleMat}) -> ConicTrait

Classify a conic matrix and return the appropriate Holy Trait.

Returns one of:
- `EllipseTrait()`: Valid ellipse (bounded, non-degenerate)
- `HyperbolaTrait()`: Hyperbola (indefinite quadratic form)
- `ParabolaTrait()`: Parabola (singular quadratic form)
- `DegenerateTrait()`: a point ellipse, an imaginary ellipse, or a conic with
  no quadratic part (a line). Intersecting line pairs classify as
  `HyperbolaTrait` and parallel pairs as `ParabolaTrait` (their quadratic forms).

Invariant to the conic's scale and sign (a homogeneous conic is defined only
up to a nonzero multiplier). A point ellipse is told from a real one by γ, the
constant term at the centre, against the rounding of the terms that produce it
(`|c| + 2|b||centre| + |λ|·|centre|²`) magnified by the quadratic block's
condition number κ — never against an absolute threshold, which called large or
distant ellipses degenerate. The bound was measured: on
20,000 elongated point ellipses (κ up to 1e6) the computed |γ| stays below
0.55·κ·eps of those terms, and the test allows 2·(1 + κ)·eps. The test runs in
at least Float64 on the coefficients divided by the largest, so a Float32 conic
is classified as its stored coefficients define it (a 3 px Float32 ellipse at
(1000, 800) is an ellipse; in Float32 arithmetic its γ sank below the noise
floor), and neither a huge nor a tiny multiple of a conic overflows or
underflows. An ellipse whose γ is within that noise — thin and very far along
its minor axis — reads degenerate, because the arithmetic cannot tell it from a
point. The checked `HomEllipseMat` constructor runs this same test, and
`Ellipse(Q)` reads the centre and γ in the same precision.
"""
classify_conic(Q::SymmetricConic) = _conic_trait(getfield(Q, :coeffs))

# The six coefficients in the work type, divided by the largest (relative only:
# a homogeneous conic has no absolute scale, and no square of a raw entry can
# overflow or underflow); `nothing` for a zero or non-finite conic.
@inline function _conic_scaled(q::NTuple{6, T}) where {T}
    W = _conic_work_type(T)
    m = maximum(x -> abs(W(x)), q)
    (iszero(m) || !isfinite(m)) && return nothing
    return map(x -> W(x) / m, q)
end

# THE conic classification, on the six coefficients, decided in at least
# Float64 (`_conic_work_type`) after scaling by the largest coefficient.
@inline function _conic_trait(q::NTuple{6})
    s = _conic_scaled(q)
    s === nothing && return DegenerateTrait()
    a, b, c, d, e, f = s
    W = typeof(a)
    ϵ = _conic_eps(W)
    A = SMatrix{2, 2, W}(a, b, b, c)
    bv = SVector{2, W}(d, e)
    cc = f

    # the quadratic block's eigenvalues ordered by MAGNITUDE: `extrema` orders
    # them by value, so a negated conic read its ratio upside down
    λ1, λ2 = eigen(Symmetric(A)).values
    small, large = abs(λ1) <= abs(λ2) ? (λ1, λ2) : (λ2, λ1)
    iszero(large) && return DegenerateTrait()          # no quadratic part: a line

    abs(small / large) < ϵ^(3 / 4) && return ParabolaTrait()
    small * large < zero(W) && return HyperbolaTrait()

    # definite: flip a negated conic so the block is positive definite
    if large < zero(W)
        A, bv, cc = -A, -bv, -cc
    end
    centre = -(A \ bv)
    γ = cc + dot(bv, centre)
    # γ < 0 is a real ellipse, unless it does not rise above the rounding of the
    # terms that produce it (a point ellipse is γ = 0 up to that rounding)
    κ = abs(large / small)
    terms = abs(cc) + 2 * norm(bv) * norm(centre) + abs(large) * dot(centre, centre)
    return γ < -2 * (1 + κ) * ϵ * terms ? EllipseTrait() : DegenerateTrait()
end

# The class predicates: THE test of classify_conic (an ellipse), and isotropy on
# top (a circle), relative to ‖A‖ and in the coefficients' own precision.
@inline _is_real_ellipse(q::NTuple{6}) = _conic_trait(q) isa EllipseTrait
@inline function _is_isotropic(q::NTuple{6, T}) where {T}
    s = _conic_scaled(q)
    s === nothing && return false
    a, b, c = s[1], s[2], s[3]
    tol = _conic_rtol(T) * sqrt(a * a + 2 * b * b + c * c)
    return abs(a - c) <= tol && abs(b) <= tol
end
@inline _is_of_class(::Type{<:SymmetricConic{:general}}, ::NTuple{6}) = true
@inline _is_of_class(::Type{<:SymmetricConic{:ellipse}}, q::NTuple{6}) = _is_real_ellipse(q)
@inline _is_of_class(::Type{<:SymmetricConic{:circle}}, q::NTuple{6}) = _is_real_ellipse(q) && _is_isotropic(q)

# A conic of class `J` is of class `K` by construction (no check needed).
_class_implies(J::Symbol, K::Symbol) = J === K || K === :general || (J === :circle && K === :ellipse)

_conic_name(::Type{<:SymmetricConic{:ellipse}}) = "HomEllipseMat"
_conic_name(::Type{<:SymmetricConic{:circle}}) = "HomCircleMat"
_conic_name(::Type{<:SymmetricConic{:general}}) = "Conic"

# The message is formatted here only, on the throw.
@noinline function _throw_conic_class(::Type{C}, q::NTuple{6}) where {C <: SymmetricConic}
    what = _describe_trait(_conic_trait(q))
    if what === nothing   # a real ellipse, so the class is a circle's
        a, b, c = q[1], q[2], q[3]
        what = "its quadratic block is not isotropic (A₁₁ − A₂₂ = $(a - c), A₁₂ = $(b)): an ellipse, not a circle"
    end
    throw(ArgumentError(string(_conic_name(C), ": ", what, " (a general conic is a `Conic`)")))
end
_describe_trait(::EllipseTrait) = nothing
_describe_trait(::HyperbolaTrait) = "its quadratic block is indefinite: a hyperbola or an intersecting line pair, not an ellipse"
_describe_trait(::ParabolaTrait) = "its quadratic block is singular: a parabola or a parallel line pair, not an ellipse"
_describe_trait(::DegenerateTrait) = "a point, imaginary or line conic (γ does not rise above its rounding), not a real ellipse"

# -- the constructors: one set, for every class ----------------------------------

function SymmetricConic{K, T}(q::NTuple{6}; check::Bool = true) where {K, T}
    qT = convert(NTuple{6, T}, q)
    check && !_is_of_class(SymmetricConic{K}, qT) && _throw_conic_class(SymmetricConic{K}, qT)
    return SymmetricConic{K, T}(qT, _Unchecked())
end
SymmetricConic{K}(q::NTuple{6, T}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(q; check)
SymmetricConic{K, T}(q::StaticVector{6}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(Tuple(q); check)
SymmetricConic{K}(q::StaticVector{6, T}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(Tuple(q); check)
# a symmetric 3×3: a matrix, or its nine entries column-major
SymmetricConic{K, T}(M::AbstractMatrix; check::Bool = true) where {K, T} =
    SymmetricConic{K, T}(_symmetric_coefficients(T, M, _conic_name(SymmetricConic{K})); check)
SymmetricConic{K}(M::AbstractMatrix{T}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(M; check)
# Disambiguate vs StaticArrays' generic `(::Type{SA})(sa::StaticArray)` converter.
SymmetricConic{K, T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}; check::Bool = true) where {K, T} =
    SymmetricConic{K, T}(_symmetric_coefficients(T, M, _conic_name(SymmetricConic{K})); check)
SymmetricConic{K}(M::StaticArrays.StaticArray{<:Tuple, T, 2}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(M; check)
SymmetricConic{K, T}(data::NTuple{9}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(SMatrix{3, 3}(data); check)
SymmetricConic{K}(data::NTuple{9, T}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(data; check)
SymmetricConic{K, T}(args::Vararg{Number, 9}; check::Bool = true) where {K, T} = SymmetricConic{K, T}(args; check)
# Another conic: symmetric by storage, its coefficients taken as they are; the
# class checked only where the source's class does not imply it.
SymmetricConic{K, T}(Q::SymmetricConic{J}; check::Bool = !_class_implies(J, K)) where {K, T, J} =
    SymmetricConic{K, T}(getfield(Q, :coeffs); check)
SymmetricConic{K}(Q::SymmetricConic{J, T}; check::Bool = !_class_implies(J, K)) where {K, J, T} =
    SymmetricConic{K, T}(Q; check)

# Immutable and of its class already: StaticArrays' generic `copy` and
# cross-eltype `convert` rebuild through the checked 9-entry constructor, which
# refused values built unchecked (`check = false`, `_Unchecked`, an ellipse
# whose class only the stored coefficients resolve). A same-class conversion
# converts the coefficients and checks nothing; another class still goes
# through the checked constructor.
Base.copy(Q::SymmetricConic) = Q
Base.convert(::Type{C}, Q::C) where {C <: SymmetricConic} = Q
Base.convert(::Type{SymmetricConic{K, T}}, Q::SymmetricConic{K}) where {K, T} =
    SymmetricConic{K, T}(getfield(Q, :coeffs), _Unchecked())
# FixedModels' padding of a solver's unused slots: the zero coefficients (no
# class holds them, but a padding slot is never read as a model)
_padding_value(::Type{SymmetricConic{K, T}}) where {K, T} = SymmetricConic{K, T}(ntuple(_ -> zero(T), Val(6)), _Unchecked())

Base.IndexStyle(::Type{<:SymmetricConic}) = IndexLinear()
@inline function Base.getindex(A::SymmetricConic, k::Int)
    @boundscheck 1 <= k <= 9 || throw(BoundsError(A, k))
    return @inbounds getfield(A, :coeffs)[_SYM3_INDEX[k]]
end
@inline Base.Tuple(A::SymmetricConic) = _full3(getfield(A, :coeffs))

# The ellipse of a conic when it is one, else `nothing`: the class tested once,
# never thrown (`Ellipse` reads it with no second check).
@inline function _ellipse_or_nothing(Q::SymmetricConic{K, T}) where {K, T}
    q = getfield(Q, :coeffs)
    return _is_real_ellipse(q) ? Ellipse(HomEllipseMat{T}(q, _Unchecked())) : nothing
end

# ========================================================================
# Fundamental / Essential Matrix Types
# ========================================================================
#
# FundMat and EssentialMat both encode the epipolar constraint as a 3×3
# matrix; they share an `AbstractFundMat` parent so that shared algebra
# (Sampson distance, etc.) can dispatch on the abstract type while
# solver- and codomain-specific paths still dispatch on the leaf type.
# Not geometric transforms (map points to lines, not points to points):
# dense algebraic StaticMatrix wrappers, like HomographyMat.

"Abstract parent for 3×3 epipolar matrices (FundMat, EssentialMat)."
abstract type AbstractFundMat{T} <: StaticArrays.StaticMatrix{3, 3, T} end

@smatrix_wrapper FundMat AbstractFundMat

@doc """
    FundMat{T} <: AbstractFundMat{T}

Typed wrapper for a 3×3 fundamental matrix encoding the epipolar constraint
`u₂ᵀ F u₁ = 0` between two views.

Not a geometric transform — maps points to epipolar lines, not points to points.
""" FundMat

# EssentialMat shares the algebraic shape but carries the extra E-manifold
# constraint σ₁ = σ₂, σ₃ = 0. We keep it as a distinct leaf type because:
#   - it unambiguously marks "model already contains the calibration", so
#     residuals and pose decomposition treat inputs as calibrated points;
#   - minimal-sample size (5) differs from F (7);
#   - LO/DLT refinements differ (E needs manifold projection).

@smatrix_wrapper EssentialMat AbstractFundMat

@doc """
    EssentialMat{T} <: AbstractFundMat{T}

Typed wrapper for a 3×3 essential matrix encoding the calibrated
epipolar constraint `x̂₂ᵀ E x̂₁ = 0`, where `x̂ᵢ = Kᵢ⁻¹ uᵢ`.

Distinguished from `FundMat` by the additional E-manifold constraint
(σ₁ = σ₂, σ₃ = 0) and by operating on calibrated (K⁻¹-mapped) points.
Minimal sample size: 5 (Nistér 2004).
""" EssentialMat

# ========================================================================
# FixedModels — Zero-allocation iterable for multi-solution solvers
# ========================================================================

"""
    FixedModels{N, M}

Stack-allocated container for up to `N` candidate models of type `M`.

Supports `for model in solutions` iteration and `length(solutions)`.
Used by multi-solution solvers (e.g., 7-point F-matrix) to return 1–3
results without heap allocation.

`isbitstype(FixedModels{N,M}) == true` when `isbitstype(M) == true`.
"""
struct FixedModels{N, M}
    count::Int
    data::NTuple{N, M}
end

@inline function Base.iterate(fm::FixedModels, i::Int = 1)
    i > fm.count && return nothing
    return (@inbounds fm.data[i], i + 1)
end
Base.length(fm::FixedModels) = fm.count
@inline function Base.getindex(fm::FixedModels, i::Integer)
    @boundscheck 1 <= i <= fm.count || throw(BoundsError(fm, i))
    return @inbounds fm.data[i]
end
Base.eltype(::Type{FixedModels{N, M}}) where {N, M} = M

"""
    map(f, fm::FixedModels{N, M}) -> FixedModels{N, M′}

`f` applied to each of `fm`'s `length(fm)` models, the count kept: the one way
to carry a solver's candidates through a per-model transform (a frame change
by composition, an eltype change). `M′` is `f`'s inferred return type on `M` and must
be concrete; the unused slots keep `fm`'s padding when `M′ == M` and are
its padding value otherwise (`zero(M′)`, or an unchecked zero-parameter value
for a structured type with no valid zero), as the builder pads them (so a model
type without either, a `Circle`, maps within its own type). `f` is never called
on a padding slot.
"""
function Base.map(f::F, fm::FixedModels{N, M}) where {F, N, M}
    M′ = Base.promote_op(f, M)
    isconcretetype(M′) || throw(ArgumentError("map(f, ::FixedModels{$N, $M}): f's return type $M′ is not concrete"))
    return FixedModels{N, M′}(fm.count, ntuple(i -> i <= fm.count ? convert(M′, f(@inbounds fm.data[i])) : _padding(M′, @inbounds fm.data[i]), Val(N)))
end
@inline _padding(::Type{M}, x::M) where {M} = x
@inline _padding(::Type{M′}, _) where {M′} = _padding_value(M′)

# The value of a FixedModels padding slot, never read as a model: `zero(M)`,
# or for a structured type with no valid zero (a calibration matrix's row 3 is
# `0 0 1`, no ellipse is zero) an unchecked zero-parameter value.
_padding_value(::Type{M}) where {M} = zero(M)

# =============================================================================
# Solver-output validity + FixedModels-from-builder constructor
# =============================================================================
#
# Decouples the three concerns shared by minimal-sample solvers (PoseLib +
# native): reading the raw output, wrapping/post-processing each candidate,
# and compacting valid candidates into a fixed-size container.
#
#   `is_valid_solution(::T)`    — type-dispatched validity predicate
#   `FixedModels{N,M}(make, n)` — build by calling `make(i)` for `i in 1:n`,
#                                  keep those satisfying `is_valid_solution`,
#                                  pad unused slots with `_padding_value(M)`.
#
# Always returns a `FixedModels{N,M}` (with `count=0` if no candidate is
# valid).  Callers iterate with `for x in fm` — empty case yields nothing
# naturally; no `Union{Nothing, FixedModels}` needed.

"""
    is_valid_solution(x) -> Bool

Predicate that decides whether a single solver output should be retained
when packing into `FixedModels`.  Defaults rule out non-finite results
(PoseLib emits NaN/Inf for numerically degenerate samples).

Add a method for any new model type that needs a stricter check.
"""
@inline is_valid_solution(m::StaticMatrix) = all(isfinite, m)
@inline is_valid_solution(v::StaticVector) = all(isfinite, v)
@inline is_valid_solution(x::Real) = isfinite(x)

# Builder constructor: `FixedModels{N,M}(make::Function, n::Int)`.
# Calls `make(i)` for `i in 1:n`, keeps those satisfying `is_valid_solution`,
# pads unused slots with `_padding_value(M)` (`zero(M)` unless the type has no
# valid zero).  Returns an empty FixedModels (count=0)
# if `n == 0` or every candidate failed validity.
@inline function FixedModels{NOUT, MType}(make::F, n::Integer) where {NOUT, MType, F <: Function}
    valid = MVector{NOUT, MType}(undef)
    n_valid = 0
    @inbounds for i in 1:n
        m = make(i)
        is_valid_solution(m) || continue
        n_valid += 1
        valid[n_valid] = m
    end
    z = _padding_value(MType)
    # Bind to a fresh local before the closure. `n_valid` is assigned inside the
    # loop above, so capturing it directly forces Julia to box it (`Core.Box`),
    # which both defeats inference in this hot minimal-solver path and makes it
    # un-provable as defined (JET flagged it, 2026-07-26). `nv` is assigned once
    # and never reassigned, so it is captured by value.
    nv = n_valid
    return FixedModels{NOUT, MType}(nv, ntuple(i -> i ≤ nv ? valid[i] : z, Val(NOUT)))
end

# The JSON codecs of these types and ScalarOrQuantity live in VisualGeometryToolkitIO (src/core_json.jl)

# ========================================================================
# Estimator Attributes
# ========================================================================
#
# Generic attributes for non-fit "value + attributes" pairs (scored
# correspondences, uncertain points). Estimator RESULTS use FitResult
# (base/fit_result.jl), which forwards into its metadata the same way.

# ========================================================================
# Attributed{V,A} - Generic estimator result container
# ========================================================================

"""
    Attributed{V,A}

Generic container composing a value with attributes. Used as the single
composition primitive for scored correspondences, detections, deployed losses,
uncertain values ([`Uncertain`](@ref) is the alias `Attributed{V, <:Union{CovMat,
PrecMat}}`) and any "value with metadata" pattern.

# Type Parameters
- `V`: The value (EllipticFrame, Homography, camera pose, correspondence, etc.)
- `A`: Attributes type (a score `Float64`, a covariance, a struct of attributes, …)

# Fields
- `value::V`: The wrapped value
- `attributes::A`: Associated attributes

# Property Access (via @generated getproperty)
- `r.value`, `r.attributes` → direct field access
- Named fields of `A` → forwarded from attributes; when `A` is a statistical
  form (`CovMat`/`PrecMat`, i.e. `r` is an [`Uncertain`](@ref)) its storage is
  not a record: `r.matrix` names the form itself and nothing else is forwarded
  from it
- Other symbols → forwarded to value

# Example
```julia
sc = Attributed(src => dst, 0.83)   # a scored correspondence
sc.value            # the Pair
sc.attributes       # 0.83
sc.first            # forwarded to value (the Pair)
```
"""
struct Attributed{V, A}
    value::V
    attributes::A
end

# Equality and hashing by value and attributes. The default (`===`, `objectid`)
# is identity for a value that holds heap data — an `Uncertain` thin-plate
# calibration with its solver matrices — so two equal ones compared unequal and
# did not deduplicate in a Set.
Base.:(==)(a::Attributed, b::Attributed) =
    getfield(a, :value) == getfield(b, :value) && getfield(a, :attributes) == getfield(b, :attributes)
Base.isequal(a::Attributed, b::Attributed) =
    isequal(getfield(a, :value), getfield(b, :value)) && isequal(getfield(a, :attributes), getfield(b, :attributes))
Base.hash(a::Attributed, h::UInt) = hash(getfield(a, :attributes), hash(getfield(a, :value), hash(:Attributed, h)))

# Property forwarding (`getproperty` / `propertynames`) is defined with the
# statistical forms below ("Uncertain{V,M}"): it treats a `CovMat`/`PrecMat`
# attribute apart, so it is stated once those types exist.

# --- StructArrays support: columnar decomposition of Attributed ---
StructArrays.staticschema(::Type{Attributed{V, A}}) where {V, A} =
    NamedTuple{(:value, :attributes), Tuple{V, A}}
StructArrays.component(x::Attributed, ::Val{:value}) = getfield(x, :value)
StructArrays.component(x::Attributed, ::Val{:attributes}) = getfield(x, :attributes)
StructArrays.createinstance(::Type{Attributed{V, A}}, v, a) where {V, A} = Attributed{V, A}(v, a)

# ========================================================================
# Correspondences — type aliases + StructArray constructor
# ========================================================================

"""
    Cspond{S,T} = Pair{S,T}

A correspondence between a source element of type `S` and a target of type `T`.
This is a type alias — correspondences are plain `Pair`s.

Correspondences are stored in columnar layout via `StructArray{Cspond{S,T}}`,
accessed as `cs.first` (sources) and `cs.second` (targets).
"""
const Cspond{S, T} = Pair{S, T}

"""
    csponds(sources, targets) -> StructArray{Cspond{S,T}}

Construct a columnar correspondence array from parallel source and target vectors.

# Examples
```julia
src = [SA[1.0, 2.0], SA[3.0, 4.0]]
dst = [SA[5.0, 6.0], SA[7.0, 8.0]]
cs = csponds(src, dst)
cs.first   # sources
cs.second  # targets
cs[1]      # first correspondence as Pair
```
"""
function csponds(sources::AbstractVector{S}, targets::AbstractVector{T}) where {S, T}
    length(sources) == length(targets) ||
        error("Source and target vectors must have the same length")
    return StructArrays.StructArray{Cspond{S, T}}((sources, targets))
end

"""
    ScoredCspond{S,T} = Attributed{Cspond{S,T}, Float64}

A correspondence with an associated match score (e.g., SNN ratio).
"""
const ScoredCspond{S, T} = Attributed{Cspond{S, T}, Float64}

ScoredCspond(source::S, target::T, score::Float64) where {S, T} =
    Attributed(Cspond{S, T}(source, target), score)

# Iterate over the wrapped Pair for destructuring: (source, target) = scored_cspond
Base.iterate(a::Attributed{<:Cspond}, state...) = iterate(getfield(a, :value), state...)

# Index/length/size forwarding for Attributed-wrapped vectors. Lets pixel coords
# be accessed uniformly across `Attributed{Point2, ...}` (RoMa csponds with
# precision attributes) and bare `Point2` (SP+LG csponds with no attributes),
# so callers like `validate_gt_metric` don't need an `_pixel_of`-style helper.
@inline Base.getindex(a::Attributed{<:AbstractVector}, i...) = getfield(a, :value)[i...]
@inline Base.length(a::Attributed{<:AbstractVector}) = length(getfield(a, :value))
@inline Base.size(a::Attributed{<:AbstractVector}, args...) = size(getfield(a, :value), args...)

# ========================================================================
# ScaledIdentity{N,T} - the static N×N matrix λ·I, stored as λ
# ========================================================================

"""
    ScaledIdentity{N}(λ)

The `N × N` matrix `λ·I`, stored as the one number `λ`: the static, sized
counterpart of `LinearAlgebra.UniformScaling` (which has no size), beside
StaticArrays' `SDiagonal` (one number per axis) and `SMatrix` (every entry).
Isotropic: a rotation leaves it unchanged. Isbits; reads as its full matrix.
It stays a `ScaledIdentity` under `inv`, a scalar factor and a sum with
another; a sum with an `SDiagonal` is an `SDiagonal`; every other operation
gives an `SMatrix`. The isotropic storage of [`CovMat`](@ref) /
[`PrecMat`](@ref) and the round [`GaussianBlur`](@ref).
"""
struct ScaledIdentity{N, T} <: StaticArrays.StaticMatrix{N, N, T}
    λ::T
    # typed (a Number): an untyped argument is ambiguous against StaticArrays' generic converters
    ScaledIdentity{N, T}(λ::Number) where {N, T} = new{N, T}(convert(T, λ))
end
ScaledIdentity{N}(λ::T) where {N, T <: Number} = ScaledIdentity{N, T}(λ)
@inline Base.getindex(A::ScaledIdentity{N, T}, i::Int, j::Int) where {N, T} = i == j ? A.λ : zero(T)
@inline Base.getindex(A::ScaledIdentity{N, T}, k::Int) where {N, T} = mod1(k, N) == (k - 1) ÷ N + 1 ? A.λ : zero(T)
@inline Base.Tuple(A::ScaledIdentity{N}) where {N} = ntuple(k -> A[k], Val(N * N))
@inline Base.IndexStyle(::Type{<:ScaledIdentity}) = IndexLinear()
# a map/broadcast result has every entry: an SMatrix
@inline StaticArrays.similar_type(::Type{<:ScaledIdentity{N}}, ::Type{S}) where {N, S} = SMatrix{N, N, S, N * N}
@inline Base.inv(A::ScaledIdentity{N}) where {N} = ScaledIdentity{N}(inv(A.λ))
@inline Base.:*(s::Number, A::ScaledIdentity{N}) where {N} = ScaledIdentity{N}(s * A.λ)
@inline Base.:*(A::ScaledIdentity{N}, s::Number) where {N} = ScaledIdentity{N}(A.λ * s)
@inline Base.:/(A::ScaledIdentity{N}, s::Number) where {N} = ScaledIdentity{N}(A.λ / s)
@inline Base.:+(A::ScaledIdentity{N}, B::ScaledIdentity{N}) where {N} = ScaledIdentity{N}(A.λ + B.λ)
@inline Base.:+(A::ScaledIdentity{N}, D::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{N}}) where {N} = LinearAlgebra.Diagonal(map(d -> A.λ + d, D.diag))
@inline Base.:+(D::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{N}}, A::ScaledIdentity{N}) where {N} = A + D
# its diagonal, as StaticArrays' SDiagonal
@inline LinearAlgebra.Diagonal(A::ScaledIdentity{N}) where {N} = LinearAlgebra.Diagonal(StaticArrays.SVector(ntuple(_ -> A.λ, Val(N))))

# The three storages of a statistical form (CovMat, PrecMat) and of a blur: full, diagonal, isotropic.
const _FormStorage{N, T} = Union{SMatrix{N, N, T}, LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}, ScaledIdentity{N, T}}

# ========================================================================
# PrecMat{N,T,M,F} - Per-observation information (precision) matrix
# ========================================================================

@smatrix_wrapper PrecMat N Frame

@doc """
    PrecMat{N,T,M,F} <: StaticMatrix{N,N,T}

N×N information (precision) matrix Λ = Σ⁻¹ for per-observation residual
uncertainty. Parametric in N so it sizes to the value dof for any geometric
value: 2 for Point2, 3 for Disc, 5 for EllipticFrame, M for Hyperplane{M},
etc. `M` is the static matrix it holds (`parent(Λ)`), and its type is the
storage: an `SMatrix` (full), an `SDiagonal` (diagonal) or a
[`ScaledIdentity`](@ref) (isotropic) — see [`DiagonalCovMat`](@ref).

Used as the storage form for sources that emit precision natively (e.g.,
RoMa v2 dense matcher, factor-graph SLAM backends, any EIV estimator that
propagates measurement noise to the constraint).

## Tensor character and transformation law

Precision is a **(0,2) tensor** — a bilinear form with two covariant
argument slots (it EATS a pair of displacement vectors: the Mahalanobis
quadratic `rᵀ Λ r`). Under a change of coordinates with Jacobian `J` it
pulls back:

    Λ ↦ J⁻ᵀ Λ J⁻¹

This is the OPPOSITE variance of [`CovMat`](@ref)'s pushforward `Σ ↦ JΣJᵀ`,
and this covariant/contravariant law pair is exactly the bug surface the
frame label exists to check: applying the covariance law to a precision
matrix (or vice versa) produces a plausible-looking, silently wrong matrix.
With labeled frames, [`propagate`](@ref) through a framed map dispatches to
the correct law per type and records the destination frame on the result.

## Frame parameter (0.9)

The trailing `F <: Frame{N}` names the frame the matrix's coordinates live
in (e.g. `ImageFrame{:left}` for a per-detection pixel precision). Every
historically-shaped constructor defaults it to
`UnlabeledFrame{N}` — the gradual-typing wildcard, so pre-0.9 code compiles
unchanged and unlabeled matrices propagate uncheckedly. Label with
[`relabel`](@ref); [`propagate`](@ref) through a framed map checks the frame
against the map's `Src` and stamps the result with `Dst`.

## Precision vs. covariance

This is distinct from [`Uncertain`](@ref)'s model-side covariance role,
which stores *parameter* uncertainty in **covariance** form because that
case is dominated by Jacobian propagation (`J Σ Jᵀ`). Per-observation
uncertainty on the residual side is dominated by Mahalanobis-style
weighting (`rᵀΛr`), so storing Λ directly avoids inversions on the scoring
hot path.

## Example

```julia
Λ = PrecMat{2,Float32}(1.0f0, 0.0f0, 0.0f0, 1.0f0)   # identity precision
r = SVector(0.3f0, -0.1f0)
maha_sq = dot(r, Λ * r)                              # Mahalanobis²
```

See also: [`CovMat`](@ref), [`Uncertain`](@ref), [`propagate`](@ref).
""" PrecMat

# ========================================================================
# CovMat{N,T,M,F} - Per-observation covariance matrix (counterpart to PrecMat)
# ========================================================================

@smatrix_wrapper CovMat N Frame

@doc """
    CovMat{N,T,M,F} <: StaticMatrix{N,N,T}

N×N covariance matrix Σ = Λ⁻¹ — the covariance counterpart to [`PrecMat`](@ref).

Use when the upstream source emits covariance natively (Gaussian process
regressors, MVN priors, Kalman-filter outputs).  The two forms are
interconvertible: `inv(Λ)` and `inv(Σ)` (frame-preserving — the
representation swaps, the frame does not).

## Tensor character and transformation law

Covariance is a **(2,0) tensor** — both slots contravariant (it PAIRS with
two covectors: `Σ = E[δx δxᵀ]` is built from displacement components). Under
a change of coordinates with Jacobian `J` it pushes forward:

    Σ ↦ J Σ Jᵀ

This is the OPPOSITE variance of [`PrecMat`](@ref)'s pullback
`Λ ↦ J⁻ᵀ Λ J⁻¹`, and this covariant/contravariant law pair is exactly the
bug surface the frame label exists to check: conjugating with the wrong
factor produces a plausible-looking, silently wrong matrix. With labeled
frames, [`propagate`](@ref) through a framed map dispatches to the correct
law per type and records the destination frame on the result.

## Frame parameter (0.9)

The trailing `F <: Frame{N}` names the frame the matrix's coordinates live
in; every historically-shaped constructor defaults it to
`UnlabeledFrame{N}` (the gradual-typing wildcard — see [`PrecMat`](@ref)).
Storage (the held matrix `M`, `parent(Σ)`: an `SMatrix`, `SDiagonal` or
[`ScaledIdentity`](@ref)) and the StaticArrays interface mirror [`PrecMat`](@ref).

## Example

```julia
Σ = CovMat{2,Float32}(4.0f0, 1.0f0, 1.0f0, 9.0f0)   # 2×2 covariance
e = Ellipse(Uncertain(Point2(100.0, 200.0), Σ); sigma=3)
```
""" CovMat

# --- Diagonal forms from standard deviations (keyword-only) ---
"""
    CovMat{N}(; σ)
    PrecMat{N}(; σ)

The N×N covariance `σ²·I` (precision `I/σ²`) of a standard deviation `σ`, an
isotropic form (a `ScaledIdentity`); an N-tuple / `SVector{N}` of them is the
diagonal `(σ₁², …, σ_N²)` (`(1/σ₁², …)`), a diagonal form (an `SDiagonal`).
Keyword-only so the number is unambiguously a standard deviation — there is no
positional scalar constructor. The element type is `float` of σ's; the frame is
`UnlabeledFrame{N}`, as for every unlabeled constructor (label with
[`relabel`](@ref)).

```julia
Uncertain(Point2(100.0, 200.0), CovMat{2}(σ = 0.5))          # Σ = 0.25·I, isotropic
Uncertain(Point2(100.0, 200.0), PrecMat{2}(σ = (0.5, 2.0)))  # Λ = diag(4, 0.25), diagonal
```
"""
@inline (::Type{CovMat{N}})(; σ) where {N} = _sigma_form(CovMat, Val(N), σ, x -> x * x)
@inline (::Type{PrecMat{N}})(; σ) where {N} = _sigma_form(PrecMat, Val(N), σ, x -> inv(x * x))
@inline _sigma_form(::Type{F}, ::Val{N}, σ::Real, f::G) where {F, N, G} = F(ScaledIdentity{N}(f(only(_checked_sigmas((float(σ),))))))
@inline _sigma_form(::Type{F}, ::Val{N}, σ, f::G) where {F, N, G} = F(LinearAlgebra.Diagonal(StaticArrays.SVector(map(f, _sigmas(Val(N), σ)))))

@inline _sigmas(::Val{N}, σ::Real) where {N} = _checked_sigmas(ntuple(_ -> float(σ), Val(N)))
@inline _sigmas(::Val{N}, σ::Tuple{Vararg{Real, N}}) where {N} = _checked_sigmas(promote(map(float, σ)...))
@inline _sigmas(::Val{N}, σ::StaticVector{N, <:Real}) where {N} = _checked_sigmas(Tuple(map(float, σ)))
@inline function _checked_sigmas(σ::Tuple{Vararg{AbstractFloat}})
    all(s -> s > 0 && isfinite(s), σ) || throw(ArgumentError("σ must be positive and finite, got $σ"))
    return σ
end

# ========================================================================
# Uncertain{V,M} - a value with its covariance or precision: an Attributed
# ========================================================================

"""
    Uncertain{V, M} = Attributed{V, M}   (M <: Union{CovMat, PrecMat})

A geometric value `V` with its 2nd-order uncertainty: an [`Attributed`](@ref)
whose attributes are one of the typed matrix wrappers —

- `CovMat{N,T,M}` — covariance Σ in parameter space
- `PrecMat{N,T,M}` — precision Λ = Σ⁻¹ in parameter/residual space

`Uncertain` is an alias, not a second struct: one value-with-attributes type
carries every value-plus-metadata pattern, and an uncertain value is the one
whose attributes are a statistical form. It is constructed ONLY from a
`CovMat` or a `PrecMat` — a bare matrix or a scalar σ does not say which form
is stored, so `Uncertain(v, m)` with any other `m` is a `MethodError`; wrap it
as what it is, `CovMat(Σ)` or `PrecMat(Λ)`. Everything `Attributed` does it does
(`u.value`, `u.attributes`, `centroid`, StructArrays decomposition, index and
length forwarding to a vector value), and everything written against
`Uncertain` — [`cov`](@ref), [`prec`](@ref), [`geom`](@ref),
[`propagate`](@ref), `Ellipse` — dispatches on exactly these attributes.
`u.matrix` reads the form (`getfield(u, :attributes)` is the raw field).

The matrix's nominal type tells consumers which form is stored; accessors
[`cov`](@ref) and [`prec`](@ref) return whichever form the caller asks for,
inverting on demand. Each consumer constructs `Uncertain` with the form it
expects to be queried most so the inverse is paid at most once at the
boundary, never in the inner loop.

## Two roles, one wrapper

- **Measurement-side**: `Uncertain{Point2, PrecMat{2,T,SMatrix{2,2,T,4}}}` — a feature
  detection (e.g. RoMa-v2 dense match) carrying its per-observation
  precision. Consumed by heteroscedastic Sampson, weighted DLT, IRLS.
- **Model-side**: `Uncertain{HomographyMat{T}, CovMat{8,T,SMatrix{8,8,T,64}}}` — a fitted
  model carrying its parameter covariance. Consumed by predictive scoring
  and downstream propagation.

# Type Parameters
- `V`: The wrapped geometric value (`Point2`, `HomographyMat`, `Disc`, …)
- `M`: The matrix wrapper (`CovMat{N,T,M}` or `PrecMat{N,T,M}`)

# Example
```julia
# Covariance-side (e.g. propagated parameter uncertainty)
p = Uncertain(Point2(100.0, 200.0), CovMat{2,Float64}(0.1, 0.0, 0.0, 0.2))
cov(p)        # CovMat — stored form, no inversion
prec(p)       # PrecMat — computed via inv(matrix)

# Precision-side (e.g. RoMa per-detection PrecMat)
Λ = PrecMat{2,Float32}(2.0f0, 0.1f0, 0.1f0, 3.0f0)
d = Uncertain(Point2(123.4, 56.7), Λ)
prec(d)       # PrecMat — stored form, no inversion
cov(d)        # CovMat — computed via inv(matrix)
d isa Attributed   # true: the same type as Attributed(Point2(123.4, 56.7), Λ)

# A bare matrix states nothing: wrap it as the form it is
p2 = Uncertain(Point2(100.0, 200.0), CovMat(@SMatrix [0.1 0.0; 0.0 0.2]))
# An isotropic standard deviation σ: the keyword constructor of the form
p3 = Uncertain(Point2(100.0, 200.0), CovMat{2}(σ = 0.5))   # Σ = 0.25·I
Uncertain(Point2(100.0, 200.0), 0.5)   # MethodError: neither a CovMat nor a PrecMat
```

See also: [`CovMat`](@ref), [`PrecMat`](@ref), [`cov`](@ref), [`prec`](@ref).
"""
const Uncertain{V, M <: Union{CovMat, PrecMat}} = Attributed{V, M}

# Attributed's property forwarding: `r.value`, `r.attributes`, the named fields
# of a record attribute, then the value's properties. A statistical form is a
# matrix, not a record: its storage is never forwarded, and `r.matrix` names it.
@generated function Base.getproperty(r::Attributed{V, A}, s::Symbol) where {V, A}
    exprs = Expr[]
    push!(exprs, :(s === :value      && return getfield(r, :value)))
    push!(exprs, :(s === :attributes && return getfield(r, :attributes)))
    if A <: Union{CovMat, PrecMat}
        push!(exprs, :(s === :matrix && return getfield(r, :attributes)))
    else
        for name in fieldnames(A)
            push!(exprs, :(s === $(QuoteNode(name)) && return getfield(getfield(r, :attributes), $(QuoteNode(name)))))
        end
    end
    push!(exprs, :(return getproperty(getfield(r, :value), s)))
    return Expr(:block, exprs...)
end

@generated function Base.propertynames(r::Attributed{V, A}) where {V, A}
    A <: Union{CovMat, PrecMat} && return :((:value, :attributes, :matrix))
    attr_names = fieldnames(A)
    return :((:value, :attributes, $(QuoteNode.(attr_names)...)))
end

# The one constructor: the form is a CovMat or a PrecMat, never a bare matrix or
# a scalar — which form is stored is part of the value, so the caller states it.
# Anything else is a MethodError (`Attributed(v, Σ::SMatrix)` is a plain
# Attributed, never an `Uncertain`).
Uncertain(v, m::Union{CovMat, PrecMat}) = Attributed(v, m)

# --- Cross-conversion: inv(::PrecMat) -> CovMat and vice versa ---
# Frame-preserving: inversion swaps the REPRESENTATION (covariance ↔
# precision, i.e. which transformation law applies), not the frame the
# coordinates live in.
@inline function Base.inv(Λ::PrecMat{N, T, M, C}) where {N, T, M, C}
    m = inv(parent(Λ))                       # the matrix's own inverse: an SDiagonal's is an SDiagonal, λ·I's is λ⁻¹·I
    return CovMat{N, eltype(m), typeof(m), C}(m)
end
@inline function Base.inv(Σ::CovMat{N, T, M, C}) where {N, T, M, C}
    m = inv(parent(Σ))
    return PrecMat{N, eltype(m), typeof(m), C}(m)
end

"""
    FullCovMat{N, T, C}, DiagonalCovMat{N, T, C}, IsotropicCovMat{N, T, C}
    FullPrecMat{N, T, C}, DiagonalPrecMat{N, T, C}, IsotropicPrecMat{N, T, C}

A covariance (precision) by the matrix it holds: an `SMatrix` (full), an
`SDiagonal` (`CovMat(SDiagonal(σ₁², σ₂²))`) or a [`ScaledIdentity`](@ref)
(`CovMat(ScaledIdentity{2}(σ²))`, `CovMat{2}(σ = …)`). The storages order
isotropic ⊂ diagonal ⊂ full. A form keeps its storage under `inv` and
`relabel`; `convert` and `promote` give the coarser of two storages, never a
finer one (a container of mixed storage is one concrete type); `propagate`
through a map gives the full form; a unit change gives the diagonal form of an
isotropic one (lengths and angles scale apart). Arithmetic on the structure is
the matrix's own, `parent(Σ)`; on the form itself it sees the full `N × N`
entries and gives an `SMatrix`.
"""
const DiagonalCovMat{N, T, C} = CovMat{N, T, LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}, C}
@doc (@doc DiagonalCovMat) const DiagonalPrecMat{N, T, C} = PrecMat{N, T, LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}, C}
@doc (@doc DiagonalCovMat) const IsotropicCovMat{N, T, C} = CovMat{N, T, ScaledIdentity{N, T}, C}
@doc (@doc DiagonalCovMat) const IsotropicPrecMat{N, T, C} = PrecMat{N, T, ScaledIdentity{N, T}, C}
@doc (@doc DiagonalCovMat) const FullCovMat{N, T, C} = CovMat{N, T, <:SMatrix{N, N, T}, C}
@doc (@doc DiagonalCovMat) const FullPrecMat{N, T, C} = PrecMat{N, T, <:SMatrix{N, N, T}, C}

# The storage order isotropic (1) ⊂ diagonal (2) ⊂ full (3), the storage of a rank in an eltype, and a matrix in a
# storage at least as coarse as its own (the one direction `convert` and `promote` take).
_storage_rank(::Type{<:ScaledIdentity}) = 1
_storage_rank(::Type{<:LinearAlgebra.Diagonal}) = 2
_storage_rank(::Type{<:SMatrix}) = 3
_storage_of(::Val{1}, ::Val{N}, ::Type{T}) where {N, T} = ScaledIdentity{N, T}
_storage_of(::Val{2}, ::Val{N}, ::Type{T}) where {N, T} = LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}
_storage_of(::Val{3}, ::Val{N}, ::Type{T}) where {N, T} = SMatrix{N, N, T, N * N}
_as_storage(::Type{<:SMatrix{N, N, T}}, m) where {N, T} = SMatrix{N, N, T}(m)
_as_storage(::Type{LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}}, m::LinearAlgebra.Diagonal) where {N, T} =
    LinearAlgebra.Diagonal(StaticArrays.SVector{N, T}(m.diag))
_as_storage(::Type{LinearAlgebra.Diagonal{T, StaticArrays.SVector{N, T}}}, m::ScaledIdentity{N}) where {N, T} =
    LinearAlgebra.Diagonal(StaticArrays.SVector(ntuple(_ -> convert(T, m.λ), Val(N))))
_as_storage(::Type{ScaledIdentity{N, T}}, m::ScaledIdentity{N}) where {N, T} = ScaledIdentity{N, T}(m.λ)

# A matrix of a storage in another eltype, its structure kept.
_with_eltype(::Type{T}, m::SMatrix{N, N}) where {T, N} = SMatrix{N, N, T}(m)
_with_eltype(::Type{T}, m::LinearAlgebra.Diagonal{<:Any, <:StaticArrays.SVector{N}}) where {T, N} = LinearAlgebra.Diagonal(StaticArrays.SVector{N, T}(m.diag))
_with_eltype(::Type{T}, m::ScaledIdentity{N}) where {T, N} = ScaledIdentity{N, T}(convert(T, m.λ))

for F in (:CovMat, :PrecMat)
    @eval begin
        # mixed storage promotes to the coarser one (literals, `vcat`, `promote`); frames as they are
        function Base.promote_rule(::Type{$F{N, T, M1, C}}, ::Type{$F{N, S, M2, C}}) where {N, T, S, M1, M2, C}
            R = promote_type(T, S)
            M = _storage_of(Val(max(_storage_rank(M1), _storage_rank(M2))), Val(N), R)
            return $F{N, R, M, C}
        end
    end
end

# --- Frame labeling escape hatches — relabel / unlabel for the statistical
# forms. One tag (the frame of the coordinates), not two: a covariance/
# precision matrix has a single frame, unlike a map's Src/Dst endpoints or a
# bilinear epipolar form's two point slots. Semantics are the house
# relabel/unlabel contract: deliberate, unchecked override; zero-cost.
@inline relabel(::Type{C}, Σ::CovMat{N, T, M}) where {N, T, M, C <: Frame{N}} =
    CovMat{N, T, M, C}(parent(Σ))
@inline relabel(::Type{C}, Λ::PrecMat{N, T, M}) where {N, T, M, C <: Frame{N}} =
    PrecMat{N, T, M, C}(parent(Λ))
@inline unlabel(Σ::CovMat{N}) where {N} = relabel(UnlabeledFrame{N}, Σ)
@inline unlabel(Λ::PrecMat{N}) where {N} = relabel(UnlabeledFrame{N}, Λ)

# Internal frame accessor (measurement side): the frame a statistical form's
# coordinates live in, `UnlabeledFrame` when nothing is recorded. Used by the
# checked propagate/Sampson layers; not exported.
@inline _frame(::CovMat{N, T, L, C}) where {N, T, L, C} = C
@inline _frame(::PrecMat{N, T, L, C}) where {N, T, L, C} = C
@inline _frame(u::Uncertain) = _frame(getfield(u, :attributes))

# --- Storage-boundary conversion (mirrors the EuclideanMap idiom) ---
# Containers typed against the historical 3-parameter surface
# (`Vector{Uncertain{Point2, PrecMat{2,T,SMatrix{2,2,T,4}}}}` and the like) must keep
# working: `setindex!`/`push!` route through `convert`, and the stored
# value's frame parameter would otherwise make it a different type.
#
# Frame policy (the same as EuclideanMap's): `convert` may LABEL
# (source frame is `UnlabeledFrame` and the target states a frame — the
# gradual-typing direction) or keep the frame as-is; it never silently
# RELABELS to a different frame and never silently ERASES one (that is the
# explicit `unlabel`). A labeled→labeled mismatch has no method.
# The labeled-mismatch case must be an explicit throw rather than a missing
# method: CovMat/PrecMat are StaticMatrix subtypes, so an absent method would
# NOT error — it would fall through to StaticArrays' generic
# `convert(::Type{SA}, ::StaticArray)` and rebuild under the new frame,
# silently relabeling (the composition method-(5) idiom, applied to convert).
# Storage: a form converts to its own storage or a coarser one (isotropic ⊂
# diagonal ⊂ full), never a finer one (that would read its values).
@inline function Base.convert(
        ::Type{CovMat{N, T, M, C}}, Σ::CovMat{N, S, M0, C0}
    ) where {N, T, S, M, M0, C <: Frame{N}, C0 <: Frame{N}}
    (C0 === C || C0 === UnlabeledFrame{N}) && _storage_rank(M) >= _storage_rank(M0) ||
        throw(MethodError(convert, (CovMat{N, T, M, C}, Σ)))
    return CovMat{N, T, M, C}(_as_storage(M, parent(Σ)))
end
@inline function Base.convert(
        ::Type{PrecMat{N, T, M, C}}, Λ::PrecMat{N, S, M0, C0}
    ) where {N, T, S, M, M0, C <: Frame{N}, C0 <: Frame{N}}
    (C0 === C || C0 === UnlabeledFrame{N}) && _storage_rank(M) >= _storage_rank(M0) ||
        throw(MethodError(convert, (PrecMat{N, T, M, C}, Λ)))
    return PrecMat{N, T, M, C}(_as_storage(M, parent(Λ)))
end
# Uncertain to a stated (possibly partially-parameterized) target: convert
# both components. The exact-type fast path stays with Base's
# `convert(::Type{T}, x::T)`, which is more specific.
@inline function Base.convert(::Type{Uncertain{V, M}}, u::Uncertain) where {V, M}
    return Uncertain{V, M}(
        convert(V, getfield(u, :value)), convert(M, getfield(u, :attributes))
    )
end

# ========================================================================
# Uncertainty accessors — geom / prec / cov
# ========================================================================
#
# Uniform interface across plain points and `Uncertain` wrappers so that
# distance/weight functions can be written once and dispatch on the leaf
# matrix type (CovMat vs PrecMat) for the correct fast path.
#
#   geom(x)  → the underlying geometric value (SVector or similar)
#   prec(x)  → precision matrix Λ = Σ⁻¹  (or `I` for plain points)
#   cov(x)   → covariance matrix Σ        (or `I` for plain points)
#
# For `Uncertain{V, M}` the accessor whose form matches `M` returns the
# stored matrix directly; the other inverts at the use site. Construct with
# the form your hot path queries most.

"""
    geom(x)

The geometric value of `x`: the wrapped value of an [`Uncertain`](@ref) (its
covariance dropped), `x` itself otherwise.
"""
@inline geom(p) = p
@inline geom(u::Uncertain) = geom(getfield(u, :value))

# `prec` is a VGC-introduced function — no clash with stdlib.
"""
    prec(u::Uncertain) -> PrecMat
    prec(x) -> LinearAlgebra.I

The precision (information) form of a measurement's uncertainty: returns
the stored [`PrecMat`](@ref) directly, or `inv` of a stored
[`CovMat`](@ref) (paying the inversion once at the boundary). The fallback
for plain values is the identity `LinearAlgebra.I` — an unweighted
measurement — so call sites that mix plain points and `Uncertain` inputs
need no special case for the no-weighting branch. Counterpart of `cov`
(the `Statistics.cov` extension below); see [`Uncertain`](@ref).
"""
@inline prec(::Any) = LinearAlgebra.I
@inline prec(u::Uncertain{V, <:PrecMat}) where {V} = getfield(u, :attributes)
@inline prec(u::Uncertain{V, <:CovMat}) where {V} = inv(getfield(u, :attributes))

# `cov` extends Statistics.cov only for Uncertain (true extension, not piracy).
# For plain points, callers use prec — there is no covariance to return.
# Import-then-extend binds the name in this module so the docstring and the
# `VisualGeometryToolkitCore.cov` doc reference resolve without a docs-build shim.
import Statistics: cov

"""
    cov(u::Uncertain) -> CovMat

The covariance form of a measurement's uncertainty: returns the stored
[`CovMat`](@ref) directly, or `inv` of a stored [`PrecMat`](@ref) (paying
the inversion once at the boundary). Extends `Statistics.cov` for
[`Uncertain`](@ref) only; counterpart of [`prec`](@ref).
"""
@inline cov(u::Uncertain{V, <:CovMat}) where {V} = getfield(u, :attributes)
@inline cov(u::Uncertain{V, <:PrecMat}) where {V} = inv(getfield(u, :attributes))

# ========================================================================
# BoundedRange - Parameter validation type
# ========================================================================

"""
    BoundedRange{T}

A validated range with min ≤ max constraint. Useful for detector parameters.

# Example
```julia
scale_range = BoundedRange(1.0, 100.0)
x in scale_range  # true if 1.0 ≤ x ≤ 100.0
clamp(x, scale_range)  # clamps to [1.0, 100.0]
```
"""
struct BoundedRange{T}
    min::T
    max::T

    function BoundedRange(min::T, max::T) where {T}
        min ≤ max || throw(ArgumentError("min ($min) must be ≤ max ($max)"))
        return new{T}(min, max)
    end
end

BoundedRange(min, max) = BoundedRange(promote(min, max)...)

Base.in(x, r::BoundedRange) = r.min ≤ x ≤ r.max
Base.clamp(x, r::BoundedRange) = clamp(x, r.min, r.max)
Base.length(r::BoundedRange) = r.max - r.min
Base.iterate(r::BoundedRange) = (r.min, Val(:done))
Base.iterate(r::BoundedRange, ::Val{:done}) = nothing

# For use with @kwdef structs
Base.convert(::Type{BoundedRange{T}}, x::BoundedRange{S}) where {T, S} =
    BoundedRange(T(x.min), T(x.max))

# ========================================================================
# AbstractDetection — an Attributed feature frame, certain or uncertain
# ========================================================================
#
# Detection is structurally identical to Attributed{V,A}: both are
# "value + metadata". Making AbstractDetection a UnionAll alias for
# Attributed unifies Detection with the single Attributed composition
# primitive. The frame may carry its uncertainty: a detection whose value is
# an `Uncertain` frame (`Attributed{<:FeatureFrame, <:Union{CovMat,PrecMat}}`)
# is a detection too, and every method on AbstractDetection reads its
# geometry through `feature_frame`, which unwraps that layer.

"""
    AbstractDetection

Type alias for `Attributed{<:Union{FeatureFrame, Uncertain{<:FeatureFrame}}}`
— any Attributed value whose wrapped value is a FeatureFrame, or a
FeatureFrame with its covariance or precision ([`Uncertain`](@ref)).

This replaces the former abstract type. `d isa AbstractDetection` is true
for any `Attributed{V,A}` where `V <: FeatureFrame` or
`V <: Uncertain{<:FeatureFrame}`. Methods on it read the geometry through
[`feature_frame`](@ref), never `d.value`, so both kinds behave alike.
"""
const AbstractDetection = Attributed{<:Union{FeatureFrame, Uncertain{<:FeatureFrame}}}

# The bare geometric frame of either kind: `geom` unwraps an `Uncertain`
# value and is the identity on a frame. One method, inferred per concrete
# detection type.
@inline feature_frame(d::AbstractDetection) = geom(getfield(d, :value))


# ScalarOrQuantity lives in VisualGeometryToolkitIO (src/core_json.jl)
