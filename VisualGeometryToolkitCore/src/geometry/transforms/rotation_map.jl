# =============================================================================
# RotationMap — a rotation about the origin, frame-tagged
# =============================================================================
#
# The rotation-only sibling of EuclideanMap: `x ↦ R·x`, with the same frame
# algebra (Src/Dst tags, `inv` swaps them, `∘` checks the junction by
# `_check_junction`), and no translation at all: absent by construction, not a
# stored zero. It poses a camera that only turns about its centre (a panorama's
# frames, `OrientedCamera`), where a translation would be a quantity nobody
# measured.

"""
    RotationMap{N, T, R <: Rotation{N, T}, Src <: Frame{N}, Dst <: Frame{N}} <: AbstractMatrixMap

A rotation about the origin, `x ↦ R·x`, in `N` dimensions, carrying the frame
tags an [`EuclideanMap`](@ref) carries: it consumes `Src`-coordinates and
produces `Dst`-coordinates. Its one field is the rotation `R`, so it has no
translation at all, rather than a zero one: it poses a camera that turns
about its centre ([`OrientedCamera`](@ref)), and it holds no length, so it
is the same map plain and physical ([`has_units`](@ref) is `false`).

- `RotationMap(R::Rotation{N}; check = true)`, unlabeled. A `RotMatrix`, a
  `QuatRotation` or an `Angle2d` is stored as given, any other `Rotation` as a
  `RotMatrix` of its own eltype, by `EuclideanMap`'s rule.
  `RotationMap(R::StaticMatrix{N, N}; check = true)` takes a raw matrix as a
  `RotMatrix`. `check` verifies `R·Rᵀ = I`, `det R = 1`.
- [`relabel`](@ref)`(Src, Dst, M)` and [`unlabel`](@ref)`(M)` set and erase
  the tags.
- `M(x)` is `R·x`, for a vector or a point, plain or physical.
- `inv(M)` is the inverse rotation with the tags swapped.
- `A ∘ B` checks the junction (`B`'s `Dst` meets `A`'s `Src`, the unlabeled
  frame a wildcard). Two rotations compose to a `RotationMap`. A rotation and
  an `EuclideanMap`, in either order, compose to an `EuclideanMap`:
  `R_A·(R_B·x + t_B) = (R_A R_B)·x + R_A t_B` and `R_A R_B·x + t_A`.
- [`rotation`](@ref)`(M)` is `R`. `transform_deriv(M, x)` is `R`.
  `to_homogeneous(M)` is the `(N + 1) × (N + 1)` block `[R 0; 0 1]`.
  [`convert_eltype`](@ref)`(T, M)` rebuilds the rotation in `T` and keeps the
  tags.
- `==` compares the rotations as matrices and the tags. `isapprox` compares
  the rotations' matrices, and the tags must be equal.

It is isbits, so it is a device value as it is. A persisted rotation is a
Hamilton quaternion, as for poses (the stack's persisted-geometry convention);
no kind writes a `RotationMap` yet.

```julia
using Rotations: RotY
Rk = relabel(CameraFrame, WorldFrame, RotationMap(RotY(0.3)))   # a frame's orientation: camera → world
camera = PosedCamera(model, Rk)            # an OrientedCamera; stores the extrinsics inv(Rk)
pose(camera) == Rk
```
"""
struct RotationMap{N, T, R <: Rotation{N, T}, Src <: Frame{N}, Dst <: Frame{N}} <: AbstractMatrixMap
    R::R
end

function RotationMap(R::Rotation{N, T}; check::Bool = true) where {N, T}
    check && _assert_rotation(R)
    Rs = _stored_rotation(R)
    return RotationMap{N, T, typeof(Rs), UnlabeledFrame{N}, UnlabeledFrame{N}}(Rs)
end
RotationMap(R::StaticMatrix{N, N, T}; check::Bool = true) where {N, T} =
    RotationMap(RotMatrix{N, float(T)}(R); check)
RotationMap(M::RotationMap) = M

rotation(M::RotationMap) = M.R

@inline relabel(::Type{Src}, ::Type{Dst}, M::RotationMap{N, T, R}) where {N, T, R, Src <: Frame{N}, Dst <: Frame{N}} =
    RotationMap{N, T, R, Src, Dst}(M.R)
@inline unlabel(M::RotationMap{N}) where {N} = relabel(UnlabeledFrame{N}, UnlabeledFrame{N}, M)

# (Src, Dst) of a rotation map, read off its type.
_rotation_frames(::RotationMap{N, T, R, Src, Dst}) where {N, T, R, Src, Dst} = (Src, Dst)

(M::RotationMap)(x) = M.R * x

# The inverse keeps the rotation's representation where its inverse has it (a
# QuatRotation's conjugate, a RotMatrix's transpose), as EuclideanMap's does.
@inline function Base.inv(M::RotationMap{N, T, R, Src, Dst}) where {N, T, R, Src, Dst}
    Rin = inv(M.R)
    Rin isa R && return RotationMap{N, T, R, Dst, Src}(Rin)
    return relabel(Dst, Src, RotationMap(Rin; check = false))
end

function Base.:∘(A::RotationMap{N, TA, RA, SrcA, DstA}, B::RotationMap{N, TB, RB, SrcB, DstB}) where {N, TA, RA, SrcA, DstA, TB, RB, SrcB, DstB}
    _check_junction(SrcA, DstB, ∘, (A, B))
    return relabel(SrcB, DstA, RotationMap(A.R * B.R; check = false))
end
function Base.:∘(A::RotationMap{N, TA, RA, SrcA, DstA}, B::EuclideanMap{N, TB, RB, VB, SrcB, DstB}) where {N, TA, RA, SrcA, DstA, TB, RB, VB, SrcB, DstB}
    _check_junction(SrcA, DstB, ∘, (A, B))
    return relabel(SrcB, DstA, EuclideanMap(A.R * B.R, A.R * B.t; check = false))
end
function Base.:∘(A::EuclideanMap{N, TA, RA, VA, SrcA, DstA}, B::RotationMap{N, TB, RB, SrcB, DstB}) where {N, TA, RA, VA, SrcA, DstA, TB, RB, SrcB, DstB}
    _check_junction(SrcA, DstB, ∘, (A, B))
    return relabel(SrcB, DstA, EuclideanMap(A.R * B.R, A.t; check = false))
end

CoordinateTransformations.transform_deriv(M::RotationMap{N}, x) where {N} = SMatrix{N, N}(M.R)

function to_homogeneous(M::RotationMap{N, T}) where {N, T}
    R = SMatrix{N, N, T}(M.R)
    return SMatrix{N + 1, N + 1, T}(ntuple(Val((N + 1) * (N + 1))) do k
        i, j = mod1(k, N + 1), cld(k, N + 1)
        i <= N && j <= N ? R[i, j] : (i == j ? one(T) : zero(T))
    end)
end

# A rotation holds no length: its plain and physical forms are the one map, so
# stripping or attaching a unit returns it (and so does an oriented camera's).
has_units(::RotationMap) = false
FlexUnits.ustrip(::Units, M::RotationMap) = M
_attach_unit(M::RotationMap, ::Units) = M
with_unit(M::RotationMap, ::Units) = M

# Equal tags (which fixes N) and equal, or approximately equal, rotation matrices.
_rotation_matrix(M::RotationMap{N}) where {N} = SMatrix{N, N}(M.R)
Base.:(==)(A::RotationMap, B::RotationMap) =
    _rotation_frames(A) == _rotation_frames(B) && _rotation_matrix(A) == _rotation_matrix(B)
Base.isapprox(A::RotationMap, B::RotationMap; kwargs...) =
    _rotation_frames(A) == _rotation_frames(B) && isapprox(_rotation_matrix(A), _rotation_matrix(B); kwargs...)

function Base.show(io::IO, M::RotationMap{N, T, R, Src, Dst}) where {N, T, R, Src, Dst}
    print(io, "RotationMap(", M.R, ")")
    (Src <: UnlabeledFrame && Dst <: UnlabeledFrame) || print(io, ": ", nameof(Src), " → ", nameof(Dst))
    return nothing
end

# Where a pose puts its source frame's origin, the camera centre of a camera's
# pose: a rigid map's translation, the origin itself for a rotation.
_mapped_origin(P::EuclideanMap) = translation(P)
_mapped_origin(::RotationMap{N, T}) where {N, T} = zero(SVector{N, T})
