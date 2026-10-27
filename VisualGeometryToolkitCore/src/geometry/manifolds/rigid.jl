# =============================================================================
# The rigid-pose charts. See base/manifold.jl for the contract.
#
# ONE chart family, `RigidChart{P, C}`: a pose (R, t) perturbed on the LEFT,
#   R(v) = ΔR(ω)·R,   t(v) = t + τ          (C = FreeT)
#                     t(v) = ΔR(ω)·t + τ     (C = CarriedT),
# v = (ω₁..₃, τ₁..₃), ΔR the rotation parameterization P's increment:
#   MRPParam — RotMatrix(MRP(ω)) (Rotations' MRP: 4× the rotation vector at
#              the identity), ExpParam — exp_so3(ω) (the rotation vector).
# Every method is written once over four facts per kind: the rotation
# increment and its logarithm, its left Jacobian (the anchor value a multiple
# of I₃, `_rot_anchor_scale`), and the point the rotation carries. The three
# charts the stack names are aliases of the family:
#   RigidManifold6     = RigidChart{MRPParam, FreeT}   (P3P, the relpose-bundle
#                        pose block, BlobBoards' bundle; storage types free)
#   LeftRigidManifold6 = RigidChart{ExpParam, CarriedT} (the pose constraint's
#                        perturbation: PnP; float-typed storage)
#   RigidExpManifold6  = RigidChart{ExpParam, FreeT}    (hand-eye, robot-world)
# All are full rank (no gauge): the covariance round-trip pull(push(Σ)) = Σ is
# exact, which a metric pose covariance needs (EssentialManifold's normalized t
# cannot carry absolute scale). A pose has no single 3×3 model matrix, so
# `retract` returns a `EuclideanMap` and covariance transport vectorizes it as
# [vec(R); t] ∈ R¹² (T is 12×6).
#
# BIT-STABILITY: `recenter_pose` keeps the consumers' historical retraction
# CHARACTER-FOR-CHARACTER (`RotMatrix(MRP(δ))·R₀`, `t₀ + Vec3(δ₄:₆)` for the MRP
# chart) and is @noinline — the fmuladd contraction lesson of base/manifold.jl:
# one standalone compilation, context-independent arithmetic.
# =============================================================================

"""
    RotationParam

How a [`RigidChart`](@ref) parameterizes its rotation increment `ΔR(ω)`:
[`MRPParam`](@ref) or [`ExpParam`](@ref).
"""
abstract type RotationParam end
"`ΔR(ω) = RotMatrix(MRP(ω))`: Rotations' modified Rodrigues parameters, 4× the rotation vector at the identity."
struct MRPParam <: RotationParam end
"`ΔR(ω) = exp_so3(ω)`: the rotation vector (radians)."
struct ExpParam <: RotationParam end

"""
    TranslationCoupling

Whether a [`RigidChart`](@ref)'s rotation increment carries the translation:
[`FreeT`](@ref) (`t(v) = t + τ`) or [`CarriedT`](@ref) (`t(v) = ΔR·t + τ`, the
whole map perturbed on the left).
"""
abstract type TranslationCoupling end
"`t(v) = t + τ`: the translation is a free `R³` update."
struct FreeT <: TranslationCoupling end
"`t(v) = ΔR(ω)·t + τ`: the rotation carries the translation (`x ↦ ΔR·(R·x + t) + τ`)."
struct CarriedT <: TranslationCoupling end

"""
    RigidChart{P <: RotationParam, C <: TranslationCoupling, Rot, Tr} <: Manifold

The 6-dof rigid pose chart family, anchored at `(R::Rot, t::Tr)`: coordinates
`v = (ω₁..₃, τ₁..₃)` act as `R(v) = ΔR(ω)·R` and `t(v) = t + τ` ([`FreeT`](@ref))
or `ΔR(ω)·t + τ` ([`CarriedT`](@ref)), `ΔR` the increment of `P`
([`MRPParam`](@ref), [`ExpParam`](@ref)). Full rank (no gauge). `retract` returns
a `EuclideanMap`; the ambient vectorization is `[vec(R); t] ∈ R¹²`; `recenter`
re-projects the rotation onto SO(3) (the one policy) and keeps the anchor's
storage types, in the element type the contract's rule gives (the anchor's for a
step of its own float type); `local_coordinates` is closed form. The stack's named charts are
aliases: [`RigidManifold6`](@ref), [`LeftRigidManifold6`](@ref),
[`RigidExpManifold6`](@ref).
"""
struct RigidChart{P <: RotationParam, C <: TranslationCoupling, Rot, Tr} <: Manifold
    R::Rot
    t::Tr
end

"""
    RigidManifold6(R, t) <: Manifold

The MRP rigid pose chart, `RigidChart{MRPParam, FreeT}`: `R(v) = RotMatrix(MRP(ω))·R`,
`t(v) = t + τ`. Parametric in the anchor's storage types (a `RotMatrix` or an
`SMatrix`, any static 3-vector), which `recenter` keeps — the rigid-pose
retraction of P3P, the relpose-bundle pose block and BlobBoards' bundle. Its
action Jacobian at the anchor is `[−4·[R·x]×  I₃]` (the MRP's factor 4).
Distinct from [`EssentialManifold6`](@ref): the translation update is free, so
there is no radial-t gauge. See [`RigidChart`](@ref).
"""
const RigidManifold6{Rot, Tr} = RigidChart{MRPParam, FreeT, Rot, Tr}
RigidManifold6(R::Rot, t::Tr) where {Rot, Tr} = RigidManifold6{Rot, Tr}(R, t)

"""
    LeftRigidManifold6(R, t) <: Manifold
    LeftRigidManifold6(pose::EuclideanMap{3})

The LEFT-perturbation pose chart, `RigidChart{ExpParam, CarriedT}`: coordinates
`v = (δω₁..₃, δt₁..₃)` act on the whole map, `x ↦ exp([δω]×)·(R·x + t) + δt`,
so `retract(ch, v)` is `(exp_so3(δω)·R, exp_so3(δω)·t + δt)`.

It is the perturbation [`constraint_with_jacobians`](@ref) differentiates for a
pose (`G_θ = Jπ·[−[X_cam]×  I₃]`): `action_jacobian(ch, x) = [−[R·x + t]×  I₃]`,
so a PnP refit — and the PnP lift, which is the refit's rows at `v = 0` — that
solves in these coordinates re-anchors (`recenter`) with exactly the
perturbation its Jacobian assumed. `local_coordinates` is the closed-form
inverse (`δω = log_so3(R·R₀ᵀ)`, `δt = t − exp_so3(δω)·t₀`). Float-typed storage
(`SMatrix{3,3,T}`, `SVector{3,T}`). See [`RigidChart`](@ref).
"""
const LeftRigidManifold6{T} = RigidChart{ExpParam, CarriedT, SMatrix{3, 3, T, 9}, SVector{3, T}}

"""
    RigidExpManifold6(R, t) <: Manifold
    RigidExpManifold6(pose::EuclideanMap{3})

The DECOUPLED pose chart, `RigidChart{ExpParam, FreeT}`: `R(v) = exp_so3(δω)·R`,
`t(v) = t + δt`. Unlike [`LeftRigidManifold6`](@ref) the rotation does not carry
the translation; unlike [`RigidManifold6`](@ref) the rotation coordinate is the
rotation vector (radians), not the MRP. The chart of the hand-eye constraint's
`G_θ`: `action_jacobian(ch, x) = [−[R·x]×  I₃]`. Float-typed storage. Move a
covariance between it and the other rigid charts with
[`transform_covariance`](@ref). See [`RigidChart`](@ref).
"""
const RigidExpManifold6{T} = RigidChart{ExpParam, FreeT, SMatrix{3, 3, T, 9}, SVector{3, T}}

# The float-typed exp charts at (R, t), and at a pose.
for C in (:LeftRigidManifold6, :RigidExpManifold6)
    @eval function $C(R::AbstractMatrix, t::AbstractVector)
        T = float(promote_type(eltype(R), eltype(t)))
        return $C{T}(SMatrix{3, 3, T, 9}(R), SVector{3, T}(t))
    end
    @eval $C(pose::PlainEuclideanMap{3}) = $C(pose.R, pose.t)
end

# -----------------------------------------------------------------------------
# The per-kind facts
# -----------------------------------------------------------------------------

# ΔR(ω) and its inverse, the coordinates of a relative rotation.
@inline _rot_increment(::MRPParam, v) = RotMatrix(Rotations.MRP(v[1], v[2], v[3]))
@inline _rot_increment(::ExpParam, v) = exp_so3(SVector(v[1], v[2], v[3]))
@inline _rot_log(::MRPParam, ΔR::StaticMatrix{3, 3, T}) where {T} = SVector{3, T}(Rotations.params(Rotations.MRP(RotMatrix{3, T}(ΔR))))
@inline _rot_log(::ExpParam, ΔR::StaticMatrix) = log_so3(ΔR)

# The left Jacobian L(ω): ΔR(ω + dω) = exp([L(ω)·dω]×)·ΔR(ω). Rotations' MRP:
# 4·((1 − ‖ω‖²)·I + 2[ω]× + 2ωωᵀ)/(1 + ‖ω‖²)²; the rotation vector: SO(3)'s.
# At ω = 0 both are `_rot_anchor_scale`·I₃.
@inline function _rot_left_jacobian(::MRPParam, ω)
    p² = dot(ω, ω)
    return (4 / (1 + p²)^2) * ((1 - p²) * I + 2 * skew(ω) + 2 * ω * ω')
end
@inline _rot_left_jacobian(::ExpParam, ω) = left_jacobian(SO3, ω)
@inline _rot_anchor_scale(::MRPParam) = 4
@inline _rot_anchor_scale(::ExpParam) = 1

# t(v), the point the rotation increment carries, and τ from a target pose.
@inline _translation(::FreeT, ΔR, t, v) = t + Vec3(v[4], v[5], v[6])   # eltype-generic: ForwardDiff Duals flow through
@inline _translation(::CarriedT, ΔR, t, v) = ΔR * t + SVector(v[4], v[5], v[6])
@inline _carried_point(::FreeT, R, t, x) = R * x
@inline _carried_point(::CarriedT, R, t, x) = R * x + t
# the translation's log reads the rotation increment only when the rotation
# carries the translation: a FreeT chart never builds it (`ω` is the rotation's log)
@inline _translation_log(::FreeT, param, ω, t₀, t) = t - t₀
@inline _translation_log(::CarriedT, param, ω, t₀, t) = t - _rot_increment(param, ω) * t₀

_param(::RigidChart{P}) where {P} = P()
_coupling(::RigidChart{P, C}) where {P, C} = C()

# -----------------------------------------------------------------------------
# The chart, once
# -----------------------------------------------------------------------------

tangent_size(::RigidChart) = 6
zero_tangent(ch::RigidChart) = zero(SVector{6, float(eltype(ch.R))})

# a pose's ambient coordinates, [vec(R); t] — every rigid chart's ambient space.
# In the pose's float type: the retraction differential is ForwardDiff through
# it, and a Float32 chart's differential stays Float32.
function ambient_coordinates(p::PlainEuclideanMap{3})
    T = float(promote_type(eltype(p.R), eltype(p.t)))
    return vcat(SVector{9, T}(Tuple(SMatrix{3, 3}(p.R))), SVector{3, T}(p.t))
end

@noinline function recenter_pose(ch::RigidChart, v)
    ΔR = _rot_increment(_param(ch), v)
    return (ΔR * ch.R, _translation(_coupling(ch), ΔR, ch.t, v))
end

# Re-anchoring re-projects R onto SO(3) — the one policy (`_reproject_so3`,
# lie.jl) — and keeps the anchor's storage types in the contract's element type
# (base/manifold.jl): a step of the chart's own float type gives back the type
# a solver stepping a chart variable holds, a Dual step a Dual-valued chart.
# The chart's element type is its parts' promoted; a step that does not widen
# it keeps BOTH storage types exactly — the rotation's representation (a pose
# read from a file holds a `QuatRotation`) and the translation's own eltype (a
# Float64 rotation over a Float32 translation keeps the Float32): the step is
# converted into the storage, never promoted per part. A wider step (a Dual, a
# Float64 step of a Float32 chart) rebuilds both parts in the promoted type,
# each in its own representation.
_storage_with_eltype(::Type{<:RotMatrix{3}}, ::Type{T}) where {T} = RotMatrix{3, T, 9}
_storage_with_eltype(::Type{R}, ::Type{T}) where {R <: Rotation{3}, T} = Base.typename(R).wrapper{T}
function recenter(ch::RigidChart{P, C, Rot, Tr}, v) where {P, C, Rot, Tr}
    R, t = recenter_pose(ch, v)
    S = promote_type(eltype(Rot), eltype(Tr))
    T = promote_type(S, eltype(v))
    RotT, TrT = T === S ? (Rot, Tr) : (_storage_with_eltype(Rot, T), _storage_with_eltype(Tr, T))
    return RigidChart{P, C, RotT, TrT}(RotT(_reproject_so3(R)), TrT(t))
end

function retract(ch::RigidChart, v)
    R, t = recenter_pose(ch, v)
    return EuclideanMap(RotMatrix{3}(R), t; check = false)
end

model(ch::RigidChart) = EuclideanMap(RotMatrix{3}(ch.R), ch.t; check = false)

# ∂(R·x + t)/∂[vec(R); t] — the rigid pose's parameter derivative in the
# chart's ambient vectorization ([vec(R) column-major; t] ∈ R¹²): the 3×12
# [x₁I₃ x₂I₃ x₃I₃ I₃]. With `retract_jacobian` this is the generic
# `action_jacobian` of the rigid charts; the closed forms below override it.
function CoordinateTransformations.transform_deriv_params(::PlainEuclideanMap{3}, x::AbstractVector)
    I3 = SMatrix{3, 3, eltype(x), 9}(I)
    return hcat(x[1] * I3, x[2] * I3, x[3] * I3, I3)
end

# Closed form of the chain rule for the action R(v)·x + t(v): the carried point
# y moves by −[ΔR(ω)·y₀]×·L(ω)·dω and the translation by dτ. At the anchor
# L = s·I (s = 4 for the MRP, 1 for the rotation vector): `[−s·[y₀]×  I₃]`.
# Pinned against the generic default in the manifold tests. Eltype-generic.
function action_jacobian(ch::RigidChart, x::AbstractVector)
    y = _carried_point(_coupling(ch), ch.R, ch.t, x)
    return hcat(-_rot_anchor_scale(_param(ch)) * skew(y), SMatrix{3, 3, eltype(y), 9}(I))
end
function action_jacobian(ch::RigidChart, x::AbstractVector, v::StaticVector)
    ω = SVector(v[1], v[2], v[3])
    y = SMatrix{3, 3}(_rot_increment(_param(ch), ω)) * _carried_point(_coupling(ch), SMatrix{3, 3}(ch.R), ch.t, x)
    return hcat(-skew(y) * _rot_left_jacobian(_param(ch), ω), SMatrix{3, 3, eltype(y), 9}(I))
end

"""
    local_coordinates(ch::RigidChart, pose::EuclideanMap{3}) -> SVector{6}

The rigid chart's inverse retraction: the `v` with `retract(ch, v) == pose` —
`ω` the coordinates of `R·R₀ᵀ` (its MRP — the principal set, `|ω| < 1` for
rotations under π — or `log_so3`), `τ = t − t₀` (or `t − ΔR(ω)·t₀` when the
rotation carries the translation). Closed form and exact.
"""
function local_coordinates(ch::RigidChart, pose::PlainEuclideanMap{3})
    T = float(eltype(ch.R))
    R₀ = SMatrix{3, 3, T, 9}(ch.R)
    ω = _rot_log(_param(ch), SMatrix{3, 3, T, 9}(pose.R) * R₀')
    τ = _translation_log(_coupling(ch), _param(ch), ω, SVector{3, T}(ch.t), SVector{3, T}(pose.t))
    return vcat(ω, τ)
end
# a restricted pose manifold reads the same closed form on its free coordinates
local_coordinates(rc::SubsetManifold, pose::PlainEuclideanMap{3}) = local_coordinates(rc.manifold, pose)[rc.idx]
