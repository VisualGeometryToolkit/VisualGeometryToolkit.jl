# =============================================================================
# The fundamental-matrix manifold. See base/manifold.jl for the contract.
# =============================================================================

using LinearAlgebra: svd

# -----------------------------------------------------------------------------
# FundMatManifold7 — the Bartoli–Sturm orthonormal representation (PAMI 2004):
# F ~ U·diag(1, σ, 0)·Vᵀ, U, V ∈ O(3), σ = σ₂/σ₁ ∈ (0, 1];
# F(v) = u₁(v)·v₁(v)ᵀ + σ(v)·u₂(v)·v₂(v)ᵀ with
# U(v) = U·R(MRP(v₁..₃)), V(v) = V·R(MRP(v₄..₆)), σ(v) = σ + v₇.
# Anchored at the seed's SVD; the F refit ITERATES IN THIS MANIFOLD (fixed
# anchor). Near-essential anchors (σ → 1) have a one-dimensional gauge
# (co-rotation of U, V about e₃; rank(T) = 6) — handled by LM damping,
# exactly as Bartoli–Sturm §5.1.4; a transport into the chart (pinv) projects it out.
# -----------------------------------------------------------------------------

struct FundMatManifold7 <: Manifold
    U::SMatrix{3, 3, Float64, 9}
    V::SMatrix{3, 3, Float64, 9}
    σ::Float64
end

"""
    anchor_svd(ch::FundMatManifold7) -> (U, σ, V)

The anchor of the Bartoli–Sturm chart: the orthonormal factors `U`, `V` and the
normalized second singular value `σ = s₂/s₁` it is centred on.
"""
anchor_svd(ch::FundMatManifold7) = (ch.U, ch.σ, ch.V)

"Manifold from a fundamental matrix anchor via its SVD (the `_orthonormal_manifold` convention)."
function FundMatManifold7(F₀::SMatrix{3, 3, Float64})
    S = svd(F₀)
    return FundMatManifold7(
        SMatrix{3, 3, Float64}(S.U), SMatrix{3, 3, Float64}(S.V),
        S.S[2] / max(S.S[1], eps())
    )
end

tangent_size(::FundMatManifold7) = 7

# The updated factors (U, V, σ) at local coordinates `v` (Bartoli–Sturm Eq. 2/3),
# and the ambient F they build. Eltype-generic (ForwardDiff Duals flow through).
@inline function _fundmat_factors(ch::FundMatManifold7, v)
    U = ch.U * RotMatrix(Rotations.MRP(v[1], v[2], v[3]))
    V = ch.V * RotMatrix(Rotations.MRP(v[4], v[5], v[6]))
    return U, V, ch.σ + v[7]
end
@inline _fundmat_from_factors(U, V, σ) = U[:, 1] * V[:, 1]' + σ * (U[:, 2] * V[:, 2]')

retract(ch::FundMatManifold7, v::StaticVector{7}) = FundMat(Tuple(_fundmat_from_factors(_fundmat_factors(ch, v)...)))

recenter(ch::FundMatManifold7, v) = FundMatManifold7(_fundmat_factors(ch, v)...)

