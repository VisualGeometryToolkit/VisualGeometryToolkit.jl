# =============================================================================
# The essential-matrix manifolds. See base/manifold.jl for the contract.
#
# GAUGE CONVENTIONS (documented per manifold below). The 5-dof and 6-dof
# essential manifolds BOTH exist by design: the 5-dof so(3)⊕T_tS² manifold is the
# production lift manifold (minimal, gauge-fixed); the 6-dof MRP⊕R³-renormalize
# manifold is the eval GNC-family manifold (deliberately NOT minimal — one
# t-scale gauge direction, absorbed by GN damping) kept for solver-parity
# with the historical eval engines. The promotion keeps both as two manifolds
# of one model class; unification is a solver-parity decision that belongs
# to the engines, not the manifold layer.
# =============================================================================

using LinearAlgebra: normalize

# -----------------------------------------------------------------------------
# EssentialManifold{G} — one chart, two translation kinds. Anchor (R, t), |t| = 1;
# coordinates v = (ω₁, ω₂, ω₃, τ…):
#   R(v) = RotMatrix(MRP(ω)) · R,  t(v) = _t_increment(g, t, τ),  E(v) = skew(t(v)) · R(v).
# The kinds differ ONLY in the translation increment:
#   S2TangentT    — t(v) = normalize(t + τ₁·b1 + τ₂·b2), (b1, b2) an orthonormal
#                   basis of T_tS² frozen at the anchor (the per-iteration gauge
#                   fixing of the lifted engines). 5 dof, MINIMAL: rank(T) = 5.
#                   The PRODUCTION lift chart, `EssentialManifold5`.
#   RenormalizedT — t(v) = normalize(t + τ), τ ∈ R³. 6 dof, DELIBERATELY NOT
#                   MINIMAL: the radial direction τ ∥ t is a gauge nullity
#                   (rank(T) = 5) absorbed by the engines' GN damping — the eval
#                   GNC-family chart kept for solver parity, `EssentialManifold6`;
#                   a covariance transported into it (pinv) has the gauge
#                   projected out.
# The MRP convention makes the rotation block of T carry the factor-4 MRP
# differential — transported covariances are consistent as long as
# tangent-domain covariances come from the SAME chart (they do: the LO's GN
# Hessian lives there).
# -----------------------------------------------------------------------------

"""
    EssentialTranslation

The translation increment of an [`EssentialManifold`](@ref):
[`S2TangentT`](@ref) (2 dof, minimal) or [`RenormalizedT`](@ref) (3 dof, one
gauge direction).
"""
abstract type EssentialTranslation end

"`t(v) = normalize(t + τ₁·b1 + τ₂·b2)`: the tangent basis `(b1, b2)` of S² at the anchor's `t`, frozen."
struct S2TangentT{T} <: EssentialTranslation
    b1::Vec3{T}
    b2::Vec3{T}
end
"`t(v) = normalize(t + τ)`, `τ ∈ R³`: one gauge direction (`τ ∥ t`)."
struct RenormalizedT <: EssentialTranslation end

_t_dof(::Type{<:S2TangentT}) = 2
_t_dof(::Type{RenormalizedT}) = 3
@inline _t_increment(g::S2TangentT, t, v) = normalize(t + v[4] * g.b1 + v[5] * g.b2)
@inline _t_increment(::RenormalizedT, t, v) = normalize(t + Vec3(v[4], v[5], v[6]))   # eltype-generic (ForwardDiff)

"""
    EssentialManifold{G <: EssentialTranslation, T} <: Manifold

The essential-matrix chart anchored at `(R, t)`, `|t| = 1`: `R(v) =
RotMatrix(MRP(ω))·R`, `t(v)` the translation kind's increment, `E(v) =
skew(t(v))·R(v)`, stored in the float type `T`. The stack's two are aliases:
[`EssentialManifold5`](@ref) (minimal, production) and
[`EssentialManifold6`](@ref) (one gauge direction, eval). See
[`Manifold`](@ref).
"""
struct EssentialManifold{G <: EssentialTranslation, T} <: Manifold
    R::RotMatrix{3, T, 9}
    t::Vec3{T}
    g::G
end

"""
    EssentialManifold5(R, t) <: Manifold

The PRODUCTION 5-dof essential-matrix lift chart, `EssentialManifold{S2TangentT{Float64}}`:
`so(3) ⊕ T_tS²`, gauge-fixed. Anchored at `(R, t)` with `|t| = 1` and an
orthonormal tangent basis `(b1, b2)` of `S²` at `t` frozen at the anchor;
coordinates `v = (ω₁, ω₂, ω₃, τ₁, τ₂)` act as `R(v) = RotMatrix(MRP(ω))·R`,
`t(v) = normalize(t + τ₁·b1 + τ₂·b2)`, `E(v) = skew(t(v))·R(v)`. Minimal
(rank 5, no gauge nullity). See [`EssentialManifold6`](@ref) for the
deliberately non-minimal eval companion.
"""
const EssentialManifold5 = EssentialManifold{S2TangentT{Float64}}

"""
    EssentialManifold6(R, t) <: Manifold

The EVAL 6-dof essential-matrix chart, `EssentialManifold{RenormalizedT}`:
`MRP ⊕ unconstrained-t-then-normalize`. Coordinates `v = (ω₁..₃, τ₁..₃)` act as
`R(v) = RotMatrix(MRP(ω))·R`, `t(v) = normalize(t + τ)`, `E = skew(t)·R`.
DELIBERATELY NOT MINIMAL: the radial direction `τ ∥ t` is a gauge nullity
(rank 5) absorbed by the engines' GN damping — the historical eval solver chart
kept for solver-parity with the GNC/scheduled arms; a covariance transported
into it (pinv) has the gauge projected out. The production minimal chart is
[`EssentialManifold5`](@ref).
"""
const EssentialManifold6 = EssentialManifold{RenormalizedT}

"Tangent basis of S² at t — the production `_setup_tangent_basis` code, VERBATIM
(PoseLib relative.h port; branch structure and the b2 = normalize(cross(b1, t))
orientation preserved bit-for-bit for the consumer byte-identity gates)."
@inline function s2_tangent_basis(t)
    tx = abs(t[1]); ty = abs(t[2]); tz = abs(t[3])
    b1 = if tx < ty
        tx < tz ? cross(t, Vec3{Float64}(1, 0, 0)) : cross(t, Vec3{Float64}(0, 0, 1))
    else
        ty < tz ? cross(t, Vec3{Float64}(0, 1, 0)) : cross(t, Vec3{Float64}(0, 0, 1))
    end
    b1 = normalize(b1); b2 = normalize(cross(b1, t))
    return b1, b2
end

# The kind's translation data at an anchor translation `t`.
_translation_kind(::Type{<:S2TangentT}, t) = S2TangentT(s2_tangent_basis(t)...)
_translation_kind(::Type{RenormalizedT}, t) = RenormalizedT()

# In the anchor's float type T; the translation kind's data (S2TangentT's
# basis) is computed in T too, so a Dual anchor (`recenter`'s element-type rule)
# gives a Dual kind.
function EssentialManifold{G}(R::AbstractMatrix, t::AbstractVector) where {G <: EssentialTranslation}
    T = float(promote_type(eltype(R), eltype(t)))
    tv = Vec3{T}(t)
    g = _translation_kind(G, tv)
    return EssentialManifold{typeof(g), T}(RotMatrix{3, T}(R), tv, g)
end

"""
    EssentialManifold5(E::EssentialMat)
    EssentialManifold6(E::EssentialMat)

The chart anchored at the essential matrix `E`: the pose `(R, t)` of the FIRST
of `motion(E)`'s four candidates, with `t` normalized (`|t| = 1`), so
`model(ch) = skew(t)·R` is `E` up to scale and sign. The Sampson residuals are
invariant to both and to the candidate, which is why any one anchors a refit.
"""
EssentialManifold{G}(E::EssentialMat) where {G <: EssentialTranslation} = EssentialManifold{G}(_essential_anchor(E)...)

function _essential_anchor(E::EssentialMat)
    R, t = first(motion(SMatrix{3, 3, Float64}(E)))
    return RotMatrix(SMatrix{3, 3, Float64}(R)), Vec3{Float64}(normalize(SVector{3, Float64}(t)))
end

tangent_size(::EssentialManifold{G}) where {G} = 3 + _t_dof(G)

# The engines' joint-update retraction, EXTRACTED VERBATIM (update_model!'s
# `R = RotMatrix(MRP(Δθ[1:3]))·R; t = normalize(t + …)`) — eltype-generic
# update, storage stays typed. @noinline is LOAD-BEARING: see the bit-stability
# note in base/manifold.jl.
@noinline recenter_pose(ch::EssentialManifold, v) =
    (RotMatrix(Rotations.MRP(v[1], v[2], v[3])) * ch.R, _t_increment(ch.g, ch.t, v))

function retract(ch::EssentialManifold, v::StaticVector)
    R, t = recenter_pose(ch, v)
    return EssentialMat(Tuple(skew(t) * R))
end

# re-anchoring re-projects R onto SO(3) — the one policy (`_reproject_so3`, lie.jl);
# `recenter_pose` (the engines' verbatim update) does not
function recenter(ch::EssentialManifold{G}, v) where {G}
    R, t = recenter_pose(ch, v)
    return EssentialManifold{G}(_reproject_so3(R), t)
end
