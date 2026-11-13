# =============================================================================
# Checked, frame-labeled uncertainty propagation through framed maps (0.9)
# =============================================================================
#
# base/propagate.jl owns the raw-Jacobian layer: `propagate(J, Σ) = J Σ Jᵀ`,
# frame-free by design — a bare Jacobian carries no frames, so nothing is
# checked and nothing is labeled. This file adds the CHECKED layer on top:
# when uncertainty is pushed through a FRAMED map (EuclideanMap or
# HomographyMap with Src/Dst tags), the statistical form's frame must meet
# the map's `Src` (the standing wildcard truth table — see `Base.:∘` at
# EuclideanMap), and the result carries the map's `Dst` as-is.
#
# The two laws, dispatched by TYPE (this pair is the bug surface the frame
# labels exist to check — see the CovMat/PrecMat docstrings):
#
#   covariance  (2,0) tensor, pushforward:  Σ ↦ J Σ Jᵀ
#   precision   (0,2) bilinear form, pullback:  Λ ↦ J⁻ᵀ Λ J⁻¹
#
# Raw functions and Jacobians stay unlabeled — the framed overloads never
# force annotations onto code that has none.

# Frame junction: the statistical form's frame meeting the map's `Src`, by the
# one rule (`_check_junction`, base/frames.jl) — a mismatch is an explicit
# `MethodError`, since a missing method would be shadowed by the generic
# `propagate(J, Σ)` Jacobian fallback and silently apply the WRONG semantics.

# -----------------------------------------------------------------------------
# EuclideanMap — linear action, point-independent Jacobian J = R
# -----------------------------------------------------------------------------

"""
    propagate(m::EuclideanMap{N}, Σ::CovMat{N})  -> CovMat{N, ..., Dst}
    propagate(m::EuclideanMap{N}, Λ::PrecMat{N}) -> PrecMat{N, ..., Dst}

Checked, frame-labeled propagation of a statistical form through a framed
rigid map. The map's action on points is affine with constant Jacobian
`J = R`, so no evaluation point is needed. The form's frame must meet the
map's `Src` under the standing wildcard truth table (a labeled mismatch is
a `MethodError`; `UnlabeledFrame` on either side is unchecked), and the
result carries the map's `Dst` as-is.

The law dispatches on the REPRESENTATION — this is the point of the types:

- covariance, a (2,0) tensor, pushes forward: `Σ ↦ J Σ Jᵀ`
- precision, a (0,2) bilinear form, pulls back: `Λ ↦ J⁻ᵀ Λ J⁻¹`

(For a rigid map the two conjugations coincide numerically because `R` is
orthogonal; the code follows the honest law per type so the invariant
`propagate(m, inv(Σ)) == inv(propagate(m, Σ))` holds by construction, not
by coincidence.)

The raw-Jacobian layer `propagate(J, Σ)` is unchanged and unlabeled — use
it when no framed map is in hand.
"""
function propagate(
        m::EuclideanMap{N, T, R, V, Src, Dst}, Σ::CovMat{N, S, L, C}
    ) where {
        N, T, R <: Rotation{N, T}, V <: StaticVector{N, T},
        Src <: Frame{N}, Dst <: Frame{N}, S, L, C <: Frame{N},
    }
    _check_junction(C, Src, propagate, (m, Σ))
    return _pushforward(SMatrix{N, N, promote_type(T, S)}(m.R), Σ, Dst)
end

function propagate(
        m::EuclideanMap{N, T, R, V, Src, Dst}, Λ::PrecMat{N, S, L, C}
    ) where {
        N, T, R <: Rotation{N, T}, V <: StaticVector{N, T},
        Src <: Frame{N}, Dst <: Frame{N}, S, L, C <: Frame{N},
    }
    _check_junction(C, Src, propagate, (m, Λ))
    return _pullback(SMatrix{N, N, promote_type(T, S)}(inv(m.R)), Λ, Dst)
end

# The transport laws of a framed map with the constant Jacobian `J` (its inverse
# `Jinv`), the junction already checked: covariance pushes forward,
# `Σ ↦ J Σ Jᵀ`; precision pulls back, `Λ ↦ J⁻ᵀ Λ J⁻¹`. The result carries `Dst`.
# Shared by every framed map with a constant Jacobian (`EuclideanMap` here, the
# cameras' `FramedAffine`).
# The result is the FULL form whatever the input's storage: a map mixes axes (a diagonal form becomes full).
@inline _pushforward(J::SMatrix{N, N, P}, Σ::CovMat{N}, ::Type{Dst}) where {N, P, Dst} =
    CovMat{N, P, SMatrix{N, N, P, N * N}, Dst}(SMatrix{N, N, P, N * N}(J * SMatrix{N, N, P}(Σ) * J'))
@inline _pullback(Jinv::SMatrix{N, N, P}, Λ::PrecMat{N}, ::Type{Dst}) where {N, P, Dst} =
    PrecMat{N, P, SMatrix{N, N, P, N * N}, Dst}(SMatrix{N, N, P, N * N}(Jinv' * SMatrix{N, N, P}(Λ) * Jinv))

"""
    propagate(m::EuclideanMap{N}, u::Uncertain) -> Uncertain

Transport an observation — geometry AND uncertainty — through a framed rigid
map in one checked call: the value moves by the map's point action `m(x)`,
the statistical form by its own law (see the two-argument methods above,
which also perform the frame check). The result's uncertainty carries the
map's `Dst` frame.
"""
propagate(m::EuclideanMap{N}, u::Uncertain{V, <:Union{CovMat{N}, PrecMat{N}}}) where {N, V} =
    Uncertain(m(getfield(u, :value)), propagate(m, getfield(u, :attributes)))

# -----------------------------------------------------------------------------
# SimilarityMap — the same, with the constant Jacobian J = s·R
# -----------------------------------------------------------------------------

"""
    propagate(m::SimilarityMap{N}, Σ::CovMat{N})  -> CovMat{N, ..., Dst}
    propagate(m::SimilarityMap{N}, Λ::PrecMat{N}) -> PrecMat{N, ..., Dst}
    propagate(m::SimilarityMap{N}, u::Uncertain)  -> Uncertain

As for an [`EuclideanMap`](@ref) (the same junction check, the same two laws),
with the constant Jacobian `J = s·R`: `Σ ↦ s²·RΣRᵀ`, `Λ ↦ RΛRᵀ / s²`.
"""
function propagate(
        m::SimilarityMap{N, T, Sc, R, V, Src, Dst}, Σ::CovMat{N, S, L, C}
    ) where {N, T, Sc, R, V, Src <: Frame{N}, Dst <: Frame{N}, S, L, C <: Frame{N}}
    _check_junction(C, Src, propagate, (m, Σ))
    return _pushforward(SMatrix{N, N, promote_type(T, Sc, S)}(m.s * SMatrix{N, N}(m.R)), Σ, Dst)
end
function propagate(
        m::SimilarityMap{N, T, Sc, R, V, Src, Dst}, Λ::PrecMat{N, S, L, C}
    ) where {N, T, Sc, R, V, Src <: Frame{N}, Dst <: Frame{N}, S, L, C <: Frame{N}}
    _check_junction(C, Src, propagate, (m, Λ))
    return _pullback(SMatrix{N, N, promote_type(T, Sc, S)}(inv(m.s) * SMatrix{N, N}(inv(m.R))), Λ, Dst)
end
propagate(m::SimilarityMap{N}, u::Uncertain{V, <:Union{CovMat{N}, PrecMat{N}}}) where {N, V} =
    Uncertain(m(getfield(u, :value)), propagate(m, getfield(u, :attributes)))

# -----------------------------------------------------------------------------
# HomographyMap — projective action, point-DEPENDENT Jacobian
# -----------------------------------------------------------------------------

# The dehomogenized point action x ↦ π(H·x̃) at x and its Jacobian: the 2-D homography map
# `PerspectiveMap() ∘ h` and its `transform_deriv` (homography_map.jl), the package's one point
# Jacobian of a homography, at the point in the law's promoted number type.

"""
    propagate(h::HomographyMap{3}, u::Uncertain{<:StaticVector{2}}) -> Uncertain

Transport an uncertain 2D point through a framed collineation. The
projective point action `x ↦ π(H·x̃)` is nonlinear, so the Jacobian is
evaluated at the point — which is why the HomographyMap overloads take the
observation (`Uncertain`), not a bare statistical form. Frame checking and
labeling follow the framed-map contract (frame meets `Src` under the
wildcard truth table; result carries `Dst`), and the law dispatches on the
representation: covariance pushes forward `Σ ↦ JΣJᵀ`, precision pulls back
`Λ ↦ J⁻ᵀΛJ⁻¹` with the 2×2 point Jacobian `J`.
"""
function propagate(
        h::HomographyMap{3, T, M, Src, Dst},
        u::Uncertain{P, <:CovMat{2, S, <:Any, C}}
    ) where {T, M, Src <: Frame{2}, Dst <: Frame{2}, S, C <: Frame{2}, P <: StaticVector{2}}
    _check_junction(C, Src, propagate, (h, u))
    R = promote_type(T, S, eltype(P))
    proj, x = PerspectiveMap() ∘ h, SVector{2, R}(getfield(u, :value))
    x′, J = proj(x), transform_deriv(proj, x)
    Σ = getfield(u, :attributes)
    out = J * SMatrix{2, 2, R}(Σ) * J'                     # full, whatever Σ's storage
    return Uncertain(similar_type(P, R)(x′), CovMat{2, R, SMatrix{2, 2, R, 4}, Dst}(SMatrix{2, 2, R, 4}(out)))
end

function propagate(
        h::HomographyMap{3, T, M, Src, Dst},
        u::Uncertain{P, <:PrecMat{2, S, <:Any, C}}
    ) where {T, M, Src <: Frame{2}, Dst <: Frame{2}, S, C <: Frame{2}, P <: StaticVector{2}}
    _check_junction(C, Src, propagate, (h, u))
    R = promote_type(T, S, eltype(P))
    proj, x = PerspectiveMap() ∘ h, SVector{2, R}(getfield(u, :value))
    x′, J = proj(x), transform_deriv(proj, x)
    Jinv = inv(J)
    Λ = getfield(u, :attributes)
    out = Jinv' * SMatrix{2, 2, R}(Λ) * Jinv
    return Uncertain(similar_type(P, R)(x′), PrecMat{2, R, SMatrix{2, 2, R, 4}, Dst}(SMatrix{2, 2, R, 4}(out)))
end
