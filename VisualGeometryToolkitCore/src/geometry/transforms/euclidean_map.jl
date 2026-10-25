# =============================================================================
# Frame tags — moved to base/frames.jl in 0.9
# =============================================================================
#
# The `Frame{N}` hierarchy now lives in src/base: the tags (UnlabeledFrame,
# WorldFrame, CameraFrame, ViewFrame, TargetFrame) in frames.jl, the 2-D image
# frames (PixelFrame, ImageFrame, TargetPlaneFrame, GnomonicFrame and the other
# retinal and distorted frames) in image_frames.jl — both loaded BEFORE
# base/types.jl so the statistical forms
# (`CovMat`/`PrecMat`) can constrain their trailing frame parameter. The
# frame-checked map algebra (Src/Dst parameters, composition truth table,
# relabel/unlabel) stays HERE.
#
# Drake monogram bridge (stated once, used throughout): the monogram symbol
# `X_AB` — "pose of frame B expressed in frame A", which maps B-coordinates
# to A-coordinates — corresponds to `EuclideanMap{..., Src = B, Dst = A}`.

# =============================================================================
# EuclideanMap - Rigid transformation (rotation + translation)
# =============================================================================
#
# Represents camera poses and other rigid body transformations in N dimensions.
# Uses Rotations.jl for the rotation component.

"""
    EuclideanMap{N,T,R<:Rotation{N,T},V<:StaticVector{N},Src<:Frame{N},Dst<:Frame{N}} <: Transformation

Rigid transformation `x ↦ R*x + t` in `N` dimensions, combining rotation and translation.

This represents a Euclidean (rigid body) transformation that preserves distances and angles.
It consists of a rotation `R` followed by a translation `t`.

The translation's element type is free. A PHYSICAL map — a pose, a mount, a
rig member — holds FlexUnits' base-form quantities, `Quantity{T,
Dimensions{FixRat32}}` in metres ([`UnitfulEuclideanMap`](@ref)); an
inner-loop map holds plain numbers — `Float64`, `Float32`, a ForwardDiff
`Dual` — beside a rotation of its own eltype `T`
([`PlainEuclideanMap`](@ref)). [`has_units`](@ref) tells them apart,
`ustrip(u, E)` gives the plain map in the unit `u`, and attaching a unit is
construction, `EuclideanMap(rotation(E), translation(E) .* u)`. A unitful map
meeting plain numbers — applied to a plain point, composed with a plain map —
is FlexUnits' `DimensionError`, never a promotion.

The trailing `Src`/`Dst` parameters are coordinate-[`Frame`](@ref) tags: the
map consumes `Src`-coordinates and produces `Dst`-coordinates (Drake
monogram: `X_AB` ≙ `Src = B, Dst = A`). All constructors that do not state
frames produce `Src = Dst = UnlabeledFrame{N}` — labels are opt-in and never
viral. `inv` swaps the frames; `∘` checks the junction frame between two
labeled maps (see the truth table at that method) and throws a `MethodError`
on a labeled mismatch. Use [`relabel`](@ref)/[`unlabel`](@ref) as explicit
escape hatches.

# Fields
- `R::Rotation{N,T}`: Rotation component (from Rotations.jl)
- `t::StaticVector{N}`: Translation vector — plain numbers, or lengths in metres

# Constructors
```julia
# From rotation and translation vector
R = RotMatrix{2}(π/4)  # 45° rotation
t = [2.0, 1.0]         # translation
euclidean = EuclideanMap(R, t)

# From MRP (Modified Rodrigues Parameters) for 3D
mrp = MRP(0.1, 0.2, 0.3)
euclidean_3d = EuclideanMap(mrp, [1.0, 2.0, 3.0])

# A physical pose: the translation a length (stored in metres)
pose = EuclideanMap(mrp, SVector(62.8, 0.0, 1200.0) .* mm)

# From tuple
euclidean = EuclideanMap((R, t))
```

# Usage
```julia
# Apply transformation to points
point = Point2f(1.0, 0.0)
transformed = euclidean(point)

# Compose transformations
combined = euclidean2 ∘ euclidean1

# Inverse transformation
inverse_euclidean = inv(euclidean)
```
"""
struct EuclideanMap{
        N, T, R <: Rotation{N, T}, V <: StaticVector{N},
        Src <: Frame{N}, Dst <: Frame{N},
    } <: AbstractMatrixMap
    R::R
    t::V  # plain numbers, or FlexUnits base-form lengths (metres)
end

"""
    PlainEuclideanMap{N}
    PlainEuclideanMap{N, T}

An [`EuclideanMap`](@ref) whose translation is plain numbers
(`V <: StaticVector{N, <:Real}`: `Float64`, `Float32`, a ForwardDiff `Dual`) —
the map the inner loop takes: minimal kernels, charts and manifolds, solvers.
A unitful map handed to a function typed on it is a `MethodError`. `T` is the
rotation's eltype.
"""
const PlainEuclideanMap{N, T} = EuclideanMap{N, T, R, V} where {R <: Rotation{N, T}, V <: StaticVector{N, <:Real}}

"""
    UnitfulEuclideanMap{N}
    UnitfulEuclideanMap{N, T}

An [`EuclideanMap`](@ref) whose translation is FlexUnits quantities — a
physical map, its translation in metres (base form). See
[`PlainEuclideanMap`](@ref), [`has_units`](@ref).
"""
const UnitfulEuclideanMap{N, T} = EuclideanMap{N, T, R, V} where {R <: Rotation{N, T}, V <: StaticVector{N, <:Quantity}}

"""
    rotation(E::EuclideanMap) -> Rotation

The rotation `R` of the rigid map `x ↦ R·x + t`.
"""
rotation(E::EuclideanMap) = E.R

"""
    translation(E::EuclideanMap) -> StaticVector

The translation `t` of the rigid map `x ↦ R·x + t`: lengths in metres
(FlexUnits base-form quantities) for a physical map, plain numbers for an
inner-loop one ([`has_units`](@ref)).
"""
translation(E::EuclideanMap) = E.t

# Gradual-typing default: constructing through the historical 4-parameter
# surface (`EuclideanMap{N, T, R, V}(R, t)`) produces an UNLABELED map.
# Every public constructor below funnels through this method, so pre-0.6
# code — which never states frames — compiles and behaves unchanged.
@inline function (::Type{EuclideanMap{N, T, R, V}})(
        Rv, tv
    ) where {N, T, R <: Rotation{N, T}, V <: StaticVector{N}}
    return EuclideanMap{N, T, R, V, UnlabeledFrame{N}, UnlabeledFrame{N}}(Rv, tv)
end

# Rotation-validity helper — used by all public EuclideanMap constructors, in
# every dimension. Accepts either a typed `Rotation{N}` or a raw
# `StaticMatrix{N,N}` and verifies `R Rᵀ ≈ I` and `det(R) ≈ 1` via
# `Rotations.isrotation`. Throws `ArgumentError` with the measured deviation on
# failure.
function _assert_rotation(
        R::Union{Rotation{N}, StaticMatrix{N, N}};
        atol::Real = sqrt(eps(Float64)) * 10
    ) where {N}
    Rotations.isrotation(R, atol) && return nothing
    # Compute diagnostics for the error message
    Rm = SMatrix{N, N, Float64}(R)
    dev_orth = norm(Rm * Rm' - SMatrix{N, N, Float64}(LinearAlgebra.I))
    dev_det = abs(det(Rm) - 1.0)
    throw(
        ArgumentError(
            "R is not a proper rotation matrix: |R Rᵀ - I| = $dev_orth, " *
                "|det R - 1| = $dev_det  (tolerance = $atol).  Supply `check=false` " *
                "to skip validation, or `project=true` to snap to the nearest valid " *
                "rotation before validation."
        )
    )
end


# The length every physical translation is: a quantity of another dimension is
# FlexUnits' `DimensionError`. (Untyped: FlexUnits' arithmetic may infer to any
# of its quantity types before `ubase` settles the base form.)
@inline function _check_length(q)
    dimension(q) == dimension(m) || throw(DimensionError(q, m))
    return q
end

"""
    EuclideanMap(R::Rotation{N}, t::AbstractVector{<:Quantity}; check = true)

A PHYSICAL map: every component of `t` a length, in any length unit (a
quantity of another dimension is a `DimensionError`), stored as FlexUnits'
base-form quantities — metres (`ubase`). The rotation is stored as a
`RotMatrix` of its own eltype.
"""
function EuclideanMap(
        R::Rotation{N, T}, t::AbstractVector{<:Quantity};
        check::Bool = true
    ) where {N, T}
    check && _assert_rotation(R)
    R_typed = RotMatrix{N, T}(R)
    t_m = SVector{N}(map(q -> _check_length(ubase(q)), Tuple(SVector{N}(t))))
    return EuclideanMap{N, T, typeof(R_typed), typeof(t_m)}(R_typed, t_m)
end

# A plain (non-static) translation: stored as an `SVector` of its own float
# type (B2: a Float32 vector stays Float32, an integer one becomes Float64);
# the rotation keeps its eltype.
function EuclideanMap(
        R::Rotation{N, T}, t::AbstractVector{<:Real};
        check::Bool = true
    ) where {N, T}
    check && _assert_rotation(R)
    R_typed = RotMatrix{N, T}(R)
    t_typed = SVector{N, float(eltype(t))}(t)
    return EuclideanMap{N, T, typeof(R_typed), typeof(t_typed)}(R_typed, t_typed)
end

"""
    EuclideanMap(R::Rotation{N,T}, t::StaticVector{N,<:Real}; check=true) -> EuclideanMap{N,T,...}

Element-type-preserving constructor that PRESERVES the input static-vector
subtype and eltype.  `Vec3`, `Point3`, `SVector`, `MVector`, `FieldVector`
are all bit-identical at the data level — accept whatever the caller hands
in. The translation's eltype is independent of the rotation's: a Float32 or
ForwardDiff `Dual` translation beside a Float64 rotation stays what it is
(an integer translation becomes its float type).

If a typed container (e.g. `MVector{4, Extrinsics3}` or
`FixedModels{4, Extrinsics3}`) needs a specific translation type, the
implicit `Base.convert` defined below performs the (zero-cost) bitcast at
the storage boundary.
"""
function EuclideanMap(
        R::Rotation{N, T}, t::StaticVector{N, <:Real};
        check::Bool = true
    ) where {N, T}
    check && _assert_rotation(R)
    R_typed = RotMatrix{N, T}(R)
    t_typed = _float_translation(t)
    return EuclideanMap{N, T, typeof(R_typed), typeof(t_typed)}(R_typed, t_typed)
end
_float_translation(t::StaticVector{N, <:Integer}) where {N} = SVector{N, float(eltype(t))}(t)
_float_translation(t::StaticVector) = t

# Storage-boundary conversion: bitcast translation/rotation to the alias
# parameter types. Triggered implicitly by container `setindex!` and
# tuple-typed constructors, so callers never need explicit `SVector{N,T}(t)`.
#
# Frame policy: `convert` may LABEL (source frame is `UnlabeledFrame` and the
# target states a frame — the gradual-typing direction) or keep frames as-is;
# it never silently RELABELS a labeled frame to a different label, and never
# silently ERASES one (that's the explicit `unlabel`). A labeled→labeled
# mismatch has no convert method — storing a pose into an extrinsics-typed
# slot is a `MethodError`.

# Frame-preserving convert to the historical 4-parameter surface.
@inline function Base.convert(
        ::Type{EuclideanMap{N, T, R, V}},
        e::EuclideanMap{N, T, <:Rotation{N, T}, <:StaticVector{N}, Src, Dst}
    ) where {N, T, R <: Rotation{N, T}, V <: StaticVector{N}, Src <: Frame{N}, Dst <: Frame{N}}
    return EuclideanMap{N, T, R, V, Src, Dst}(R(e.R), V(e.t))
end

# Fully-parameterized target: source frames must match, or be Unlabeled
# (junction of the diagonal `Union` constraint — labeling an unlabeled map
# into a labeled slot is allowed; anything else is a MethodError).
@inline function Base.convert(
        ::Type{EuclideanMap{N, T, R, V, Src, Dst}},
        e::EuclideanMap{N, T, <:Rotation{N, T}, <:StaticVector{N}, Src0, Dst0}
    ) where {
        N, T, R <: Rotation{N, T}, V <: StaticVector{N},
        Src <: Frame{N}, Dst <: Frame{N},
        Src0 <: Union{Src, UnlabeledFrame{N}}, Dst0 <: Union{Dst, UnlabeledFrame{N}},
    }
    return EuclideanMap{N, T, R, V, Src, Dst}(R(e.R), V(e.t))
end

"""
    EuclideanMap(R::SMatrix{3,3,T,9}, t::StaticVector{3};
                 check=true, project=false) -> EuclideanMap

Construct a rigid-body transform directly from a 3×3 `SMatrix`
(not already a `Rotation` type).  This is the *boundary constructor* — the
entry point at which raw 3×3 numerical data crosses into the typed
EuclideanMap.  All other constructors take a typed `Rotation{N,T}` whose
invariants are enforced by Rotations.jl.

# Validation policy

| `check` | `project` | semantics                               | typical caller        |
|---------|-----------|-----------------------------------------|-----------------------|
| `true`  | `false`   | strict — must already be a rotation     | math you trust        |
| `true`  | `true`    | snap-to-nearest then verify             | dataset loaders       |
| `false` | `false`   | trusted — skip everything               | internal algebra      |
| `false` | `true`    | snap, no verify                         | very dirty inputs     |

`project=true` snaps `R` to the nearest proper rotation via SVD
(`U·Vᵀ`, with a sign flip if the result has determinant −1).  Floating-point
drift from text-file roundtrip or naive `inv(T_B)·T_A` composition typically
sits at `1e-6`-ish, well within the snap-then-verify regime.

Narrowed to `SMatrix{3,3,T,9}` (rather than `StaticMatrix{3,3,T}`) to avoid
dispatch ambiguity with the `Rotation`-typed constructor: `RotMatrix{3,T}`
is a subtype of both `Rotation` and `StaticMatrix{3,3,T}`, but is not an
`SMatrix`, so narrowing here keeps the two paths disjoint.
"""
function EuclideanMap(
        R::SMatrix{3, 3, T, 9}, t::StaticVector{3};
        check::Bool = true,
        project::Bool = false
    ) where {T}
    R_used = project ? SMatrix{3, 3, T, 9}(Rotations.nearest_rotation(R)) : R
    check && _assert_rotation(R_used)
    return EuclideanMap(RotMatrix{3, T}(R_used), t; check = false)
end

EuclideanMap(R::Rotation{N, T}, t::NTuple{N, <:Real}) where {N, T} =
    EuclideanMap(R, SVector{N, Float64}(t))

"""
    skew(v::AbstractVector) -> SMatrix{3,3,Float64,9}

Skew-symmetric (cross-product) matrix `[v]_×` such that `[v]_× w = v × w`
for all `w ∈ R³`.  Used in essential-matrix construction `E = [t]_× R`
and in pose-perturbation Jacobians (see `constraint_with_jacobians`).

GTSAM spells this `skewSymmetric`; Sophus spells it `SO3::hat` (with `vee`
for the inverse).

Errors if `length(v) != 3`.

# Example
```julia
t = SVector{3,Float64}(1.0, 2.0, 3.0)
S = skew(t)        # 3×3 SMatrix
S * SVector(0.0, 1.0, 0.0) ≈ cross(t, SVector(0.0, 1.0, 0.0))   # true
```
"""
@inline function skew(v::AbstractVector)
    length(v) == 3 || throw(ArgumentError("skew expects length-3 vector, got $(length(v))"))
    T = eltype(v)
    z = zero(T)
    return SMatrix{3, 3, T, 9}(
        z, v[3], -v[2],
        -v[3], z, v[1],
        v[2], -v[1], z
    )
end

"""
    EuclideanMap(qw, qx, qy, qz, tx, ty, tz) -> EuclideanMap{3,Float64,…}
    EuclideanMap(quat::AbstractVector, t::AbstractVector) -> EuclideanMap{3,Float64,…}

COLMAP-style 3D rigid transform from a unit quaternion `(qw, qx, qy, qz)`
and a 3-vector translation `(tx, ty, tz)`.  Used by SfM-format readers
(ETH3D, IMC PhotoTourism, ScanNet++, custom COLMAP exports) where pose data
arrives as raw scalars on a per-line basis.

`Rotations.QuatRotation` auto-normalizes the quaternion, so callers don't
need to pre-normalize — the resulting rotation is always a valid SO(3)
element.

# Examples
```julia
# Per-scalar form (typical when parsing line-by-line)
T = EuclideanMap(0.998, 0.0, 0.064, 0.0, 1.5, 2.5, 3.0)

# Vector form (typical when receiving npz arrays)
T = EuclideanMap([qw, qx, qy, qz], [tx, ty, tz])
```
"""
EuclideanMap(
    qw::Real, qx::Real, qy::Real, qz::Real,
    tx::Real, ty::Real, tz::Real
) =
    EuclideanMap(QuatRotation(qw, qx, qy, qz), Vec{3, Float64}(tx, ty, tz))

function EuclideanMap(quat::AbstractVector, t::AbstractVector)
    length(quat) == 4 ||
        throw(ArgumentError("quat must be length 4 (qw,qx,qy,qz), got $(length(quat))"))
    length(t) == 3 ||
        throw(ArgumentError("t must be length 3 (tx,ty,tz), got $(length(t))"))
    return EuclideanMap(QuatRotation(quat[1], quat[2], quat[3], quat[4]), _translation3(t))
end
# The quaternion form's translation: plain numbers as Float64, lengths as quantities.
_translation3(t::AbstractVector{<:Quantity}) = SVector{3}(t[1], t[2], t[3])
_translation3(t::AbstractVector) = Vec{3, Float64}(t[1], t[2], t[3])

# Per-component translation (R, tx, ty, tz) of lengths: stored in metres
EuclideanMap(R::Rotation{3, T}, tx::Quantity, ty::Quantity, tz::Quantity) where {T} =
    EuclideanMap(R, SVector(ubase(tx), ubase(ty), ubase(tz)))

EuclideanMap(R::Rotation{3, T}, tx::Real, ty::Real, tz::Real) where {T} =
    EuclideanMap(R, Vec{3, Float64}(tx, ty, tz))

EuclideanMap(E::EuclideanMap) = E

EuclideanMap((R, t)::Tuple{<:Rotation, <:AbstractVector}) = EuclideanMap(R, t)

# Frame labeling escape hatches — relabel / unlabel

"""
    relabel(Src::Type{<:Frame{N}}, Dst::Type{<:Frame{N}}, m::EuclideanMap{N}) -> EuclideanMap

Return `m` with its frame tags replaced by `Src`/`Dst` (map consumes
`Src`-coordinates, produces `Dst`-coordinates). Zero-cost: the rotation and
translation are reused bit-for-bit; only the type changes.

This is a DELIBERATE escape hatch: it overrides whatever frames `m` carried,
with no check. Use it at trust boundaries (dataset loaders, solver outputs)
where the caller — not the type system — knows the frames. Attaching a
frame of the wrong dimension (e.g. a `Frame{2}` to an `N = 3` map) is a
type error (`MethodError`).

See also [`unlabel`](@ref), [`Frame`](@ref).
"""
@inline function relabel(
        ::Type{Src}, ::Type{Dst}, m::EuclideanMap{N, T, R, V}
    ) where {N, T, R <: Rotation{N, T}, V <: StaticVector{N}, Src <: Frame{N}, Dst <: Frame{N}}
    return EuclideanMap{N, T, R, V, Src, Dst}(m.R, m.t)
end

"""
    unlabel(m::EuclideanMap{N}) -> EuclideanMap{..., UnlabeledFrame{N}, UnlabeledFrame{N}}

Erase the frame tags of `m` (both endpoints become
[`UnlabeledFrame`](@ref)`{N}`). Zero-cost, deliberate escape hatch — the
result composes with anything under the wildcard rule. Inverse direction of
[`relabel`](@ref).
"""
@inline unlabel(m::EuclideanMap{N}) where {N} =
    relabel(UnlabeledFrame{N}, UnlabeledFrame{N}, m)

# An uncertain map is relabeled as its value: the covariance, in the
# `RigidExpManifold6` chart of the map's own coordinates, does not move.
relabel(::Type{Src}, ::Type{Dst}, u::Uncertain{<:EuclideanMap{N}}) where {N, Src <: Frame{N}, Dst <: Frame{N}} =
    Uncertain(relabel(Src, Dst, geom(u)), getfield(u, :attributes))
unlabel(u::Uncertain{<:EuclideanMap{N}}) where {N} = relabel(UnlabeledFrame{N}, UnlabeledFrame{N}, u)

# Operations
(E::EuclideanMap)(x) = E.R * x + E.t

# A physical map applied to a physical point of the same base-form quantity
# type: the generic `R * x + t` above, computed on the values. FlexUnits checks
# dimensions inside every `muladd` and `+` (nine checks for N = 3); here every
# component's dimension is checked once against the translation's, then the
# values take exactly the operations StaticArrays' `R * x` generates —
# `muladd(R[k,N], x[N], … muladd(R[k,2], x[2], R[k,1] * x[1]))`, FlexUnits'
# quantity `muladd` being the plain `muladd` of the values (`dstrip`) — and
# `+ t[k]`, so the result is the generic one bit for bit. A point of another
# dimension takes the generic path and its `DimensionError`.
@inline function (E::EuclideanMap{N, T, R, V})(
        x::SVector{N, Q}
    ) where {N, T, R <: Rotation{N, T}, Q <: Quantity{<:Real, <:FlexUnits.Dimensions}, V <: StaticVector{N, Q}}
    t = E.t
    d = dimension(t[1])
    all(q -> dimension(q) == d, Tuple(x)) || return E.R * x + t
    Rm = E.R
    xv = map(FlexUnits.dstrip, Tuple(x))
    return SVector{N}(ntuple(k -> Quantity(_rotated_row(Rm, xv, k) + FlexUnits.dstrip(t[k]), d), Val(N)))
end
@inline _rotated_row(Rm, xv::NTuple{N}, k) where {N} =
    foldl((acc, j) -> muladd(Rm[k, j], xv[j], acc), ntuple(j -> j + 1, Val(N - 1)); init = Rm[k, 1] * xv[1])

@inline function Base.inv(
        E::EuclideanMap{N, T, R, V, Src, Dst}
    ) where {N, T, R, V, Src, Dst}
    Rin = inv(E.R)
    tin = -(Rin * E.t)
    # The inverse of a valid rotation is algebraically valid — skip
    # revalidation; floating-point drift is bounded and unavoidable.
    # Typed algebra: inv swaps the frame tags (X_AB⁻¹ = X_BA).
    # The rotation keeps its representation when its inverse has it (a
    # QuatRotation inverts to its conjugate, a RotMatrix to its transpose);
    # the branch is on types, so it folds at compile time.
    Rin isa R && tin isa StaticVector{N} &&
        return relabel(Dst, Src, EuclideanMap{N, T, R, typeof(tin)}(Rin, tin))
    return relabel(Dst, Src, EuclideanMap(Rin, tin; check = false))
end

# -----------------------------------------------------------------------------
# Frame-checked composition
# -----------------------------------------------------------------------------
# `A ∘ B` applies B first: the JUNCTION is B's Dst meeting A's Src. The
# result always takes B's Src and A's Dst as-is. `UnlabeledFrame` is the
# gradual-typing wildcard: if either junction endpoint is unlabeled the
# junction is UNCHECKED (anti-virality — labels never force themselves onto
# unlabeled code).
#
# Truth table (junction = B.Dst vs A.Src; result frames = (B.Src, A.Dst)):
#
#   | B.Dst      | A.Src      | junction  | result             |
#   |------------|------------|-----------|--------------------|
#   | F (labeled)| F (same)   | checked ✓ | (B.Src, A.Dst)     |
#   | F (labeled)| G ≠ F      | REJECTED  | MethodError        |
#   | any        | Unlabeled  | unchecked | (B.Src, A.Dst)     |
#   | Unlabeled  | any        | unchecked | (B.Src, A.Dst)     |
#
# The mismatch is a deliberate explicit `MethodError` throw rather than a
# missing method: `EuclideanMap <: CoordinateTransformations.Transformation`,
# so an absent method would NOT error — it would silently fall through to
# CT's generic `∘(::Transformation, ::Transformation)` and build a lazy
# `ComposedTransformation`, defeating the frame check.

@inline function _compose_euclidean(
        A::EuclideanMap{N}, B::EuclideanMap{N}, ::Type{Src}, ::Type{Dst}
    ) where {N, Src <: Frame{N}, Dst <: Frame{N}}
    R_composed = A.R * B.R
    t_composed = A.R * B.t + A.t
    # The composition of two valid rotations is algebraically valid — skip
    # revalidation; cascading checks would otherwise reject legitimately drifted
    # but trustworthy chains.
    return relabel(Src, Dst, EuclideanMap(R_composed, t_composed; check = false))
end

# One method: the junction rule is `_junction_ok` (base/frames.jl), the table
# above its truth table, a mismatch its explicit `MethodError`.
function Base.:∘(
        A::EuclideanMap{N, TA, RA, VA, SrcA, DstA},
        B::EuclideanMap{N, TB, RB, VB, SrcB, DstB}
    ) where {
        N, TA, RA <: Rotation{N, TA}, VA <: StaticVector{N},
        TB, RB <: Rotation{N, TB}, VB <: StaticVector{N},
        SrcA <: Frame{N}, DstA <: Frame{N}, SrcB <: Frame{N}, DstB <: Frame{N},
    }
    _check_junction(SrcA, DstB, ∘, (A, B))
    return _compose_euclidean(A, B, SrcB, DstA)
end

# A rigid map after a translation is rigid: `(A ∘ Translation(d))(x) = R·(x + d) + t`,
# exactly `EuclideanMap(R, R·d + t)` — moving the source frame's origin (a plane
# pose's board origin to the normalization centre). The translation is an
# unlabeled CoordinateTransformations map, so the source side is the wildcard.
function Base.:∘(
        A::EuclideanMap{N, T, R, V, Src, Dst}, B::CoordinateTransformations.Translation{<:StaticVector{N}}
    ) where {N, T, R, V, Src <: Frame{N}, Dst <: Frame{N}}
    return relabel(UnlabeledFrame{N}, Dst, EuclideanMap(A.R, A.R * B.translation + A.t; check = false))
end

# Trait method for EuclideanMap

# EuclideanMap is always invertible (rotation + translation)

# =============================================================================
# Frame-pair aliases — CameraPose / CameraExtrinsics
# =============================================================================
#
# A bare EuclideanMap carries no record of WHICH direction it maps; a
# camera-to-world pose stored in an extrinsics slot is silently wrong (every
# accessor answers for the inverse camera). These frame-pair ALIASES make
# the direction a dispatch fact instead of a docstring convention. Named as
# frame-PAIR aliases (not `Pose3D`/`Extrinsics3D`): the 0.5-era `Pose3` — an
# unlabeled world-to-camera EuclideanMap alias under a pose name — made a
# one-letter-apart `Pose3D` with the opposite convention a worse footgun than
# the one being fixed. That misnomer is retired in 0.7: the concrete solver
# currency is now the labeled `Extrinsics3` alias below.

"""
    CameraPose

Frame-pair alias for a camera POSE (camera-to-world transform):
`CameraPose{T, R, V} = EuclideanMap{3, T, R, V, CameraFrame, WorldFrame}`
(monogram `X_WC`). Not a wrapper — a `CameraPose` IS an `EuclideanMap`, so
`.R`/`.t`, point application, and composition all apply directly.

`CameraPose(m::EuclideanMap{3})` labels `m` as a pose (an explicit
`relabel`, overriding whatever frames `m` carried).

The typed algebra subsumes the former wrapper machinery:
`inv(::CameraPose) -> CameraExtrinsics` (and back) via the frame swap, and
`Camera(model, ::CameraPose)` inverts internally into the extrinsics slot,
so handing a pose where extrinsics are expected is unrepresentable.

See also [`CameraExtrinsics`](@ref), [`lookat_pose`](@ref), [`Frame`](@ref).
"""
const CameraPose{T, R <: Rotation{3, T}, V <: StaticVector{3}} =
    EuclideanMap{3, T, R, V, CameraFrame, WorldFrame}

"""
    CameraExtrinsics

Frame-pair alias for camera EXTRINSICS (world-to-camera transform):
`CameraExtrinsics{T, R, V} = EuclideanMap{3, T, R, V, WorldFrame, CameraFrame}`
(monogram `X_CW`). The typed counterpart of [`CameraPose`](@ref);
`inv(::CameraExtrinsics) -> CameraPose` via the frame swap. Not a wrapper —
a `CameraExtrinsics` IS an `EuclideanMap`.

`CameraExtrinsics(m::EuclideanMap{3})` labels `m` as extrinsics (an explicit
`relabel`, overriding whatever frames `m` carried).
`Camera(model, ::CameraExtrinsics)` stores it directly in the extrinsics slot.
"""
const CameraExtrinsics{T, R <: Rotation{3, T}, V <: StaticVector{3}} =
    EuclideanMap{3, T, R, V, WorldFrame, CameraFrame}

# Labeling constructors on the aliases (explicit relabel; identity fast path
# keeps `===` when the map already carries the right frames).
(::Type{CameraPose})(m::EuclideanMap{3}) = relabel(CameraFrame, WorldFrame, m)
(::Type{CameraPose})(m::CameraPose) = m
(::Type{CameraExtrinsics})(m::EuclideanMap{3}) = relabel(WorldFrame, CameraFrame, m)
(::Type{CameraExtrinsics})(m::CameraExtrinsics) = m

# =============================================================================
# Units — a physical map and its plain form
# =============================================================================
#
# A physical map's translation is metres (FlexUnits base form); its plain form
# in a unit `u` is `ustrip(u, E)`, and a plain map takes a unit by
# construction, `EuclideanMap(rotation(E), translation(E) .* u)`. Both keep the
# rotation and the frames exactly. An `Uncertain` pose's 6×6 form lives in the
# `RigidExpManifold6` chart, rotation first, so its translation rows and
# columns (4–6) carry the length unit: they move with it, once, here.
#
# Every length crosses between the forms through one pair, `_strip_length` and
# `_attach_length` (FlexUnits' `ustrip` and `ubase`), so the conversion is
# stated once for maps, points and every carrier built on them.

# One length, plain in `u`; one plain number in `u`, as a length in metres.
@inline _strip_length(u::Units, q::Quantity) = ustrip(u, q)
@inline _attach_length(x::Real, u::Units) = _check_length(ubase(x * u))

has_units(::EuclideanMap{N, T, R, V}) where {N, T, R, V} = eltype(V) <: Quantity
has_units(u::Attributed) = has_units(getfield(u, :value))

"""
    ustrip(u::Units, E::EuclideanMap) -> EuclideanMap

The plain map of a physical `E`: its translation in the length unit `u`, the
rotation and the frames unchanged. An `ArgumentError` on a map that is
already plain ([`has_units`](@ref)); attach a unit by construction,
`EuclideanMap(rotation(E), translation(E) .* u)`.
"""
function FlexUnits.ustrip(u::Units, E::EuclideanMap{N, T, R, V, Src, Dst}) where {N, T, R, V, Src, Dst}
    has_units(E) || throw(
        ArgumentError(
            "ustrip: the map's translation is plain numbers already; a plain map takes a unit by " *
                "construction, `EuclideanMap(rotation(E), translation(E) .* u)`"
        )
    )
    t = map(q -> _strip_length(u, q), E.t)
    return EuclideanMap{N, T, R, typeof(t), Src, Dst}(E.R, t)
end

# A plain map read as being in the length unit `u`, stored in metres: the
# rotation and the frames kept exactly (construction would re-store the
# rotation as a `RotMatrix`).
function _attach_unit(E::EuclideanMap{N, T, R, V, Src, Dst}, u::Units) where {N, T, R, V, Src, Dst}
    has_units(E) && throw(ArgumentError("with_unit: the map's translation already has units"))
    t = map(x -> _attach_length(x, u), E.t)
    return EuclideanMap{N, T, R, typeof(t), Src, Dst}(E.R, t)
end

# The 6×6 pose form with its translation rows and columns (4–6) scaled by `f`:
# `op = *` multiplies a covariance's (Σ ↦ DΣD, D = diag(1, 1, 1, f, f, f)),
# `op = /` divides it; a precision moves the opposite way (Λ ↦ D⁻¹ΛD⁻¹). Each
# entry is scaled once by the product of its row and column factors (1, f or
# f·f), so a strip and its attach are inverse to within an ulp.
function _scale_translation_block(M::CovMat{6, T, 36, C}, f, op) where {T, C}
    return CovMat{6, T, 36, C}(ntuple(k -> _scaled_entry(M[k], k, f, op), Val(36)))
end
function _scale_translation_block(M::PrecMat{6, T, 36, C}, f, op) where {T, C}
    return PrecMat{6, T, 36, C}(ntuple(k -> _scaled_entry(M[k], k, f, _opposite(op)), Val(36)))
end
@inline function _scaled_entry(x, k, f, op)
    n = (mod1(k, 6) > 3) + ((k - 1) ÷ 6 + 1 > 3)
    return n == 0 ? x : op(x, n == 1 ? f : f * f)
end
_opposite(::typeof(*)) = /
_opposite(::typeof(/)) = *

"""
    ustrip(u::Units, X::Uncertain{<:EuclideanMap{3}}) -> Uncertain

The plain form of an uncertain physical pose: the map stripped in `u`
(`ustrip(u, E)`) and its 6×6 covariance (or precision) — rotation first, the
`RigidExpManifold6` chart — rescaled with it: the translation block in `u²`,
the rotation–translation block in `rad·u`. The inverse is
[`with_unit`](@ref)`(X, u)`.
"""
FlexUnits.ustrip(u::Units, X::Uncertain{<:EuclideanMap{3}}) = Uncertain(
    ustrip(u, geom(X)), _scale_translation_block(getfield(X, :attributes), _metre_factor(u), *)
)

with_unit(X::Uncertain{<:EuclideanMap{3}}, u::Units) = Uncertain(
    _attach_unit(geom(X), u), _scale_translation_block(getfield(X, :attributes), _metre_factor(u), /)
)

# =============================================================================
# Angular Distance Between Rotations and Translations
# =============================================================================

# The angle between two rotations is Rotations.jl's `rotation_angle` of their
# relative rotation, composed at the call: `rotation_angle(RotMatrix{3}(R₁ᵀR₂))`
# (through the quaternion, `2·atan(‖v‖, w)` — exact near 0, where the trace
# formula's `acos` loses half the digits).

"""
    translation_angle(t1, t2) -> Float64

Angular distance (radians) between two translation directions — plain
vectors, or lengths (compared in metres).

Since translation is only recoverable up to scale from epipolar geometry,
this compares directions, scale-free: `θ = atan(‖t₁ × t₂‖, t₁ · t₂)`, accurate
at every angle (the `acos` of the normalized dot product it replaced loses half
the digits near 0 and π). Also the angle between two bearings. `0` when either
vector is (numerically) zero.
"""
function translation_angle(t1::AbstractVector, t2::AbstractVector)
    # lengths are compared in metres (a length against a plain vector is a DimensionError)
    (has_units(t1) || has_units(t2)) && return translation_angle(_strip_metres(t1), _strip_metres(t2))
    (norm(t1) < 1.0e-15 || norm(t2) < 1.0e-15) && return 0.0
    a = SVector{3}(t1[1], t1[2], t1[3]); b = SVector{3}(t2[1], t2[2], t2[3])
    return atan(norm(cross(a, b)), dot(a, b))
end

# =============================================================================
# Extrinsics3 — Concrete labeled solver-currency alias (world-to-camera)
# =============================================================================

"""
    Extrinsics3

Concrete LABELED [`CameraExtrinsics`](@ref) alias:
`CameraExtrinsics{Float64, RotMatrix{3,Float64,9}, Vec{3,Float64}}`, i.e.
`EuclideanMap{3, Float64, …, Src = WorldFrame, Dst = CameraFrame}`
(monogram `X_CW`, applies as `x_cam = R·X_world + t`).

The model currency of the absolute-pose solvers: `p3p` returns
`FixedModels{4, Extrinsics3}` and `P3P <: MinimalSolver{Extrinsics3}`.
`isbitstype(Extrinsics3) == true` — stored inline in `FixedModels` and
`RansacWorkspace`, zero heap allocation on update.

Replaces the deprecated `Pose3` alias (removed in 0.7): same layout and
element types, but the world-to-camera direction is now a dispatch fact
instead of a docstring caveat — an `Extrinsics3` IS a `CameraExtrinsics`,
so `inv` yields a `CameraPose` and labeled composition is frame-checked.
Solver code that composes models against other labeled maps must have the
junction frames agree (or [`unlabel`](@ref) explicitly).
"""
const Extrinsics3 = CameraExtrinsics{
    Float64, RotMatrix{3, Float64, 9}, GeometryBasics.Vec{3, Float64},
}

# `zero` for any fully-concrete Float64/RotMatrix EuclideanMap type — the
# FixedModels/RansacWorkspace padding element (identity rotation, zero
# translation), frame labels preserved from the requested type.
function Base.zero(
        ::Type{EuclideanMap{3, Float64, RotMatrix{3, Float64, 9}, V, Src, Dst}}
    ) where {V <: StaticVector{3, Float64}, Src <: Frame{3}, Dst <: Frame{3}}
    return EuclideanMap{3, Float64, RotMatrix{3, Float64, 9}, V, Src, Dst}(
        RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64, 9})),
        zero(V)
    )
end

# EuclideanMap has named .R / .t fields, so the StaticMatrix/StaticVector
# defaults in base/types.jl don't apply — extend the predicate explicitly.
@inline is_valid_solution(p::EuclideanMap) = all(isfinite, p.R) && all(isfinite, p.t)

# =============================================================================
# constraint_with_jacobians — Reprojection constraint for rigid 3D models
# =============================================================================

"""
    constraint_with_jacobians(pose::EuclideanMap{3}, X::SVector{3}, ray::SVector{3})
        -> (g::SVector{2}, G_x::SMatrix{2,5}, G_θ::SMatrix{2,6})

Compute the reprojection constraint and its derivatives for a 3D rigid pose.

The constraint is the perspective-projected residual in calibrated coordinates:
  g = π(R·X + t) − π(ray)

where π is perspective division `[x,y,z] → [x/z, y/z]` and `ray` is the
backprojected observation (encodes camera calibration).

The model Jacobian `G_θ` uses the left-perturbation convention with a 6-DOF
parameterization `δ = (δω₁, δω₂, δω₃, δt₁, δt₂, δt₃)`:
  ∂g/∂δ = Jπ · [−[X_cam]×  I₃]

where `Jπ` is the 2×3 perspective division Jacobian and `[v]×` is the
skew-symmetric (cross-product) matrix of `v`.

Returns:
- `g::SVector{2,T}`      — reprojection residual (calibrated coordinates)
- `G_x::SMatrix{2,5,T}`  — ∂g/∂(X, ray₁, ray₂) measurement Jacobian
- `G_θ::SMatrix{2,6,T}`  — ∂g/∂(δω, δt) pose perturbation Jacobian

!!! warning "PINHOLE ONLY — this residual lives in the gnomonic chart"
    `π` divides by the third component, so this is the z = 1 chart
    ([`GnomonicFrame`](@ref)) and it exists only on the forward hemisphere. It
    is degenerate at 90° from the optical axis and cannot represent a ray with
    `Z ≤ 0` at all. That is correct for a PINHOLE camera and wrong for a
    fisheye or omnidirectional one, whose rays routinely go past 90°.

    It is legitimate where it is used — VisualGeometryToolkitEstimation's Pinhole PnP LM
    — and must not become the general path. The camera-general route is the
    ray-domain pixel residual (`constraint_with_jacobians(model, pose, X, u)`
    in cameras/camera_models.jl), which projects through the camera's own lens
    and therefore never assumes the chart exists. Calibrated two-view geometry
    takes the same view: it acts on unit bearings (unit rays,
    `StaticVector{3}`), never on this chart.
"""
@inline function constraint_with_jacobians(
        pose::EuclideanMap{3, Float64, <:Rotation{3, Float64}, <:StaticVector{3, Float64}},   # plain: the inner loop
        X::StaticVector{3, Float64},
        ray::StaticVector{3, Float64}
    )
    R = pose.R
    t = pose.t
    X_cam = R * X + t

    # Perspective division and its Jacobian
    π = PerspectiveMap()
    g = π(X_cam) - π(ray)

    # Jπ = ∂π/∂X_cam (2×3), the perspective map's own derivative
    Jπ = transform_deriv(π, X_cam)

    # G_θ = Jπ · [−[X_cam]×  I₃] (2×6): the action Jacobian of the left-perturbation
    # chart the constraint is differentiated in (one definition of that chart)
    G_θ = Jπ * action_jacobian(LeftRigidManifold6(pose), X)

    # G_x = ∂g/∂(X₁,X₂,X₃, ray₁/ray₃, ray₂/ray₃) (2×5)
    # ∂g/∂X = Jπ · R (2×3), ∂g/∂(ray_norm) = -I₂ (2×2)
    G_X = Jπ * SMatrix{3, 3, Float64}(R)
    G_x = hcat(G_X, -one(SMatrix{2, 2, Float64}))

    return (g, G_x, G_θ)
end

# A physical map's translation as plain metres, a plain one's as it is: what a
# public function reads where it only compares or measures a translation.
_translation_metres(E::EuclideanMap) = has_units(E) ? _strip_metres(translation(E)) : translation(E)

# -----------------------------------------------------------------------------
# Units of a point — a target's, a capture's: the same moves as a map's
# -----------------------------------------------------------------------------
#
# A physical point is a static vector of lengths (metres), or an `Uncertain`
# one whose d×d covariance is in m² (a precision in 1/m²); its plain form in a
# unit `u` scales the covariance by `ustrip(u, 1 m)²`, the inverse divides it.

_ustrip_point(u::Units, p::StaticVector{N, <:Quantity}) where {N} = map(q -> _strip_length(u, q), p)
_ustrip_point(u::Units, p::Uncertain) =
    Uncertain(_ustrip_point(u, geom(p)), _scale_point_form(getfield(p, :attributes), _metre_factor(u), *))
_ustrip_point(::Units, p) = throw(ArgumentError("ustrip: the point $(p) is plain numbers already; `with_unit` attaches a unit"))
_attach_point(u::Units, p::StaticVector{N, <:Real}) where {N} = map(x -> _attach_length(x, u), p)
_attach_point(u::Units, p::Uncertain) =
    Uncertain(_attach_point(u, geom(p)), _scale_point_form(getfield(p, :attributes), _metre_factor(u), /))
_attach_point(::Units, p) = throw(ArgumentError("with_unit: the point $(p) already has units"))

_scale_point_form(M::CovMat{N, T, L, C}, f, op) where {N, T, L, C} = CovMat{N, T, L, C}(map(x -> op(x, f * f), Tuple(M)))
_scale_point_form(M::PrecMat{N, T, L, C}, f, op) where {N, T, L, C} =
    PrecMat{N, T, L, C}(map(x -> _opposite(op)(x, f * f), Tuple(M)))

# A point's stored form: plain numbers as `Float64` (today's storage), lengths in
# metres. For the containers that hold either (targets, 3D target points).
_stored_point(::Val{N}, p::AbstractVector{<:Quantity}) where {N} = SVector{N}(map(q -> _check_length(ubase(q)), Tuple(SVector{N}(p))))
_stored_point(::Val{N}, p::AbstractVector) where {N} = SVector{N, Float64}(p)

# The containers hold one form: all plain or all lengths.
function _check_one_form(ps, what)
    isempty(ps) || allequal(has_units(geom(p)) for p in ps) ||
        throw(ArgumentError("$(what): the points are all plain numbers or all lengths, not a mix"))
    return ps
end
