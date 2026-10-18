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
- `EllipticFrame{T}`: 5 DOF - anisotropic (symmetric shape matrix)
- `AffineFrame{T}`: 6 DOF - full affine (general shape matrix)

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
# Conic Matrix Types
# ========================================================================
#
# Homogeneous conic matrix types needed by Primitives before full
# transforms module loads. Extracted here to break circular dependency:
#   - matrices.jl defines transform types that need ConicTrait (from interface.jl)
#   - Primitives needs conic matrix types
#   - Primitives is loaded before the rest of transforms

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
        # N-parametric square form with a trailing frame tag:
        # Name{N,T,L,C<:Frame{N}} <: StaticMatrix{N,N,T}. Used by the
        # statistical forms (CovMat/PrecMat, 0.9): the frame parameter names
        # WHICH frame the matrix's coordinates live in, defaulting
        # to `UnlabeledFrame{N}` through every historically-shaped constructor
        # (the standing gradual-typing rule — labels are opt-in, never viral).
        argerr_msg = string(Name, "{N,T} takes N*N args, got ")
        return quote
            struct $name{N, T, L, C <: Frame{N}} <: StaticArrays.StaticMatrix{N, N, T}
                data::NTuple{L, T}
                # Typed inner constructor: the untyped default `Name{…}(data)`
                # is ambiguous against StaticArrays' generic converters.
                $name{N, T, L, C}(data::NTuple{L}) where {N, T, L, C <: Frame{N}} =
                    new{N, T, L, C}(convert(NTuple{L, T}, data))
            end
            @inline Base.size(::Type{<:$name{N}}) where {N} = (N, N)
            # Frame-preserving similar_type for ELTYPE conversion (promote
            # chains). Broadcast/matmul results still escape to SMatrix, as
            # for every other @smatrix_wrapper type — a product like J*Σ is
            # not a covariance and must not inherit the wrapper or the frame.
            @inline StaticArrays.similar_type(
                ::Type{$name{N, T, L, C}}, ::Type{S}
            ) where {N, T, L, C, S} = $name{N, S, L, C}
            @inline StaticArrays.similar_type(::Type{<:$name{N}}, ::Type{S}) where {N, S} =
                $name{N, S, N * N, UnlabeledFrame{N}}
            @inline Base.eltype(::Type{<:$name{N, T}}) where {N, T} = T
            @inline Base.IndexStyle(::Type{<:$name}) = IndexLinear()
            @inline Base.getindex(A::$name, k::Int) = A.data[k]
            @inline Base.getindex(A::$name{N}, i::Int, j::Int) where {N} = A.data[(j - 1) * N + i]
            @inline Base.Tuple(A::$name) = A.data
            # Gradual-typing default: the historical 3-parameter surface
            # (`Name{N,T,L}(data)`) records no frame. Pre-0.9 code compiles
            # and behaves unchanged.
            @inline (::Type{$name{N, T, L}})(data::NTuple{L, T}) where {N, T, L} =
                $name{N, T, L, UnlabeledFrame{N}}(data)
            @inline $name{N, T}(data::NTuple{L, T}) where {N, T, L} = $name{N, T, L}(data)
            @inline function $name{N, T}(args::Vararg{T, L}) where {N, T, L}
                L == N * N || throw(ArgumentError(string($argerr_msg, L)))
                return $name{N, T, L}(args)
            end
            @inline $name(M::SMatrix{N, N, T, L}) where {N, T, L} = $name{N, T, L}(Tuple(M))
            @inline $name{N, T}(M::SMatrix{N, N, T, L}) where {N, T, L} = $name{N, T, L}(Tuple(M))
            @inline $name{N, T}(M::AbstractMatrix) where {N, T} = $name{N, T}(SMatrix{N, N, T}(M))
            # Any 2-D StaticArray (the full AbstractMatrix ∩ StaticArray intersection):
            # more specific than StaticArrays' `(::Type{SA})(::StaticArray)` converter.
            @inline $name{N, T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) where {N, T} = $name{N, T}(SMatrix{N, N, T}(M))
            # Eltype conversion of a labeled wrapper keeps its frame (a label is
            # never silently erased — the same rule as `convert`).
            @inline $name{N, T}(M::$name{N, S, L, C}) where {N, T, S, L, C <: Frame{N}} =
                $name{N, T, L, C}(convert(NTuple{L, T}, Tuple(M)))
            # Labeled lift from an SMatrix (fully-parameterized surface).
            @inline (::Type{$name{N, T, L, C}})(M::SMatrix{N, N, T, L}) where {N, T, L, C <: Frame{N}} =
                $name{N, T, L, C}(Tuple(M))
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

# Define conic matrix types (internal representations)
@smatrix_wrapper HomEllipseMat AbstractHomConicMat
@smatrix_wrapper HomCircleMat AbstractHomConicMat

@doc "Homogeneous 3×3 matrix representation of an ellipse (conic section)." HomEllipseMat
@doc "Homogeneous 3×3 matrix representation of a circle (special conic)." HomCircleMat

# ========================================================================
# Conic — geometric primitive for indeterminate conic sections
# ========================================================================

"""
    Conic{T}

Geometric primitive representing a 2D conic section (ellipse, hyperbola,
parabola, or degenerate). Wraps a 3×3 homogeneous conic matrix.

Returned by `HomographyMap` when transforming geometric primitives
(Circle, Ellipse). Use `is_ellipse`, `classify_conic` to determine
the type, and `Ellipse(conic)`, `Circle(conic)` to convert.

# Examples
```julia
H = HomographyMap(M)
Q = H(circle)         # returns Conic
is_ellipse(Q)         # check type
ell = Ellipse(Q)      # convert to Ellipse
```
"""
struct Conic{T}
    Q::SMatrix{3, 3, T, 9}
end

Conic(Q::AbstractMatrix{T}) where {T} = Conic{T}(SMatrix{3, 3, T}(Q))

# Convert back to matrix types (for internal use)
HomEllipseMat(c::Conic{T}) where {T} = HomEllipseMat{T}(Tuple(c.Q))
HomCircleMat(c::Conic{T}) where {T} = HomCircleMat{T}(Tuple(c.Q))
SMatrix{3, 3}(c::Conic) = c.Q

# ========================================================================
# Fundamental / Essential Matrix Types
# ========================================================================
#
# FundMat and EssentialMat both encode the epipolar constraint as a 3×3
# matrix; they share an `AbstractFundMat` parent so that shared algebra
# (Sampson distance, etc.) can dispatch on the abstract type while
# solver- and codomain-specific paths still dispatch on the leaf type.
# Not geometric transforms (map points to lines, not points to points),
# so the supertype is StaticMatrix, not AbstractHomMat.

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
be concrete; the unused slots are `zero(M′)`, as the builder pads them. `f` is
never called on a padding slot.
"""
function Base.map(f::F, fm::FixedModels{N, M}) where {F, N, M}
    M′ = Base.promote_op(f, M)
    isconcretetype(M′) || throw(ArgumentError("map(f, ::FixedModels{$N, $M}): f's return type $M′ is not concrete"))
    z = zero(M′)
    return FixedModels{N, M′}(fm.count, ntuple(i -> i <= fm.count ? convert(M′, f(@inbounds fm.data[i])) : z, Val(N)))
end

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
#                                  pad unused slots with `zero(M)`.
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
# pads unused slots with `zero(M)`.  Returns an empty FixedModels (count=0)
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
    z = zero(MType)
    # Bind to a fresh local before the closure. `n_valid` is assigned inside the
    # loop above, so capturing it directly forces Julia to box it (`Core.Box`),
    # which both defeats inference in this hot minimal-solver path and makes it
    # un-provable as defined (JET flagged it, 2026-07-26). `nv` is assigned once
    # and never reassigned, so it is captured by value.
    nv = n_valid
    return FixedModels{NOUT, MType}(nv, ntuple(i -> i ≤ nv ? valid[i] : z, Val(NOUT)))
end

# NOTE: similar_type for conic matrices is generated by @smatrix_wrapper macro

# JSON helpers (_getfield, _hasfield, ScalarOrQuantity) moved to VisualGeometryToolkitFeatures

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
# PrecMat{N,T,L,F} - Per-observation information (precision) matrix
# ========================================================================

@smatrix_wrapper PrecMat N Frame

@doc """
    PrecMat{N,T,L,F} <: StaticMatrix{N,N,T}

N×N information (precision) matrix Λ = Σ⁻¹ for per-observation residual
uncertainty. Parametric in N so it sizes to the value dof for any geometric
value: 2 for Point2, 3 for Disc, 5 for EllipticFrame, M for Hyperplane{M},
etc. `L = N*N` is the flat storage length (required by StaticArrays).

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
# CovMat{N,T,L,F} - Per-observation covariance matrix (counterpart to PrecMat)
# ========================================================================

@smatrix_wrapper CovMat N Frame

@doc """
    CovMat{N,T,L,F} <: StaticMatrix{N,N,T}

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
Storage and StaticArrays interface mirror [`PrecMat`](@ref); SMatrix shape
(`L = N*N`) and getindex semantics are identical.

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

The N×N covariance `σ²·I` (precision `I/σ²`) of a standard deviation `σ`: a
positive real, or an N-tuple / `SVector{N}` of them for the diagonal
`(σ₁², …, σ_N²)` (`(1/σ₁², …)`). Keyword-only so the number is unambiguously a
standard deviation — there is no positional scalar constructor. The element
type is `float` of σ's; the frame is `UnlabeledFrame{N}`, as for every
unlabeled constructor (label with [`relabel`](@ref)).

```julia
Uncertain(Point2(100.0, 200.0), CovMat{2}(σ = 0.5))          # Σ = 0.25·I
Uncertain(Point2(100.0, 200.0), PrecMat{2}(σ = (0.5, 2.0)))  # Λ = diag(4, 0.25)
```
"""
@inline (::Type{CovMat{N}})(; σ) where {N} = _diagonal_form(CovMat, _sigmas(Val(N), σ), x -> x * x)
@inline (::Type{PrecMat{N}})(; σ) where {N} = _diagonal_form(PrecMat, _sigmas(Val(N), σ), x -> inv(x * x))

@inline _sigmas(::Val{N}, σ::Real) where {N} = _checked_sigmas(ntuple(_ -> float(σ), Val(N)))
@inline _sigmas(::Val{N}, σ::Tuple{Vararg{Real, N}}) where {N} = _checked_sigmas(promote(map(float, σ)...))
@inline _sigmas(::Val{N}, σ::StaticVector{N, <:Real}) where {N} = _checked_sigmas(Tuple(map(float, σ)))
@inline function _checked_sigmas(σ::Tuple{Vararg{AbstractFloat}})
    all(s -> s > 0 && isfinite(s), σ) || throw(ArgumentError("σ must be positive and finite, got $σ"))
    return σ
end
@inline function _diagonal_form(::Type{F}, σ::Tuple{T, Vararg{T}}, f::G) where {F, T, G}
    N = length(σ)
    d = map(f, σ)
    return F{N, T, N * N}(ntuple(k -> (k - 1) % N == (k - 1) ÷ N ? d[(k - 1) % N + 1] : zero(T), Val(N * N)))
end

# ========================================================================
# Uncertain{V,M} - a value with its covariance or precision: an Attributed
# ========================================================================

"""
    Uncertain{V, M} = Attributed{V, M}   (M <: Union{CovMat, PrecMat})

A geometric value `V` with its 2nd-order uncertainty: an [`Attributed`](@ref)
whose attributes are one of the typed matrix wrappers —

- `CovMat{N,T,L}` — covariance Σ in parameter space
- `PrecMat{N,T,L}` — precision Λ = Σ⁻¹ in parameter/residual space

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

- **Measurement-side**: `Uncertain{Point2, PrecMat{2,T,4}}` — a feature
  detection (e.g. RoMa-v2 dense match) carrying its per-observation
  precision. Consumed by heteroscedastic Sampson, weighted DLT, IRLS.
- **Model-side**: `Uncertain{HomographyMat{T}, CovMat{8,T,64}}` — a fitted
  model carrying its parameter covariance. Consumed by predictive scoring
  and downstream propagation.

# Type Parameters
- `V`: The wrapped geometric value (`Point2`, `HomographyMat`, `Disc`, …)
- `M`: The matrix wrapper (`CovMat{N,T,L}` or `PrecMat{N,T,L}`)

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
@inline Base.inv(Λ::PrecMat{N, T, L, C}) where {N, T, L, C} =
    CovMat{N, T, L, C}(Tuple(inv(SMatrix{N, N, T}(Λ))))
@inline Base.inv(Σ::CovMat{N, T, L, C}) where {N, T, L, C} =
    PrecMat{N, T, L, C}(Tuple(inv(SMatrix{N, N, T}(Σ))))

# --- Frame labeling escape hatches — relabel / unlabel for the statistical
# forms. One tag (the frame of the coordinates), not two: a covariance/
# precision matrix has a single frame, unlike a map's Src/Dst endpoints or a
# bilinear epipolar form's two point slots. Semantics are the house
# relabel/unlabel contract: deliberate, unchecked override; zero-cost.
@inline relabel(::Type{C}, Σ::CovMat{N, T, L}) where {N, T, L, C <: Frame{N}} =
    CovMat{N, T, L, C}(Tuple(Σ))
@inline relabel(::Type{C}, Λ::PrecMat{N, T, L}) where {N, T, L, C <: Frame{N}} =
    PrecMat{N, T, L, C}(Tuple(Λ))
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
# (`Vector{Uncertain{Point2, PrecMat{2,T,4}}}` and the like) must keep
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
@inline function Base.convert(
        ::Type{CovMat{N, T, L, C}}, Σ::CovMat{N, S, L, C0}
    ) where {N, T, S, L, C <: Frame{N}, C0 <: Frame{N}}
    (C0 === C || C0 === UnlabeledFrame{N}) ||
        throw(MethodError(convert, (CovMat{N, T, L, C}, Σ)))
    return CovMat{N, T, L, C}(NTuple{L, T}(Tuple(Σ)))
end
@inline function Base.convert(
        ::Type{PrecMat{N, T, L, C}}, Λ::PrecMat{N, S, L, C0}
    ) where {N, T, S, L, C <: Frame{N}, C0 <: Frame{N}}
    (C0 === C || C0 === UnlabeledFrame{N}) ||
        throw(MethodError(convert, (PrecMat{N, T, L, C}, Λ)))
    return PrecMat{N, T, L, C}(NTuple{L, T}(Tuple(Λ)))
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


# ScalarOrQuantity moved to VisualGeometryToolkitFeatures/serialization.jl
