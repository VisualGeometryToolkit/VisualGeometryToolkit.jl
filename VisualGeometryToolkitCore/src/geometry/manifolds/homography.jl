# =============================================================================
# The homography manifold, and the sl(3) basis it is built on.
# See base/manifold.jl for the contract.
# =============================================================================

# -----------------------------------------------------------------------------
# HomographyManifold8 — the minimal sl(3) manifold about an anchor homography:
# H(v) = H₀·(I + Σ vᵢ·Gᵢ) with G₁…G₈ the traceless sl(3) basis. Tracelessness
# deletes the homogeneity direction H₀·εI (the gauge the dehomogenized
# residual cannot see) — minimal (rank(T) = 8), no ‖H‖ = 1 constraint, no
# fixed-entry singularity. First-order manifold (no dexp): ∂H/∂vᵢ = H₀·Gᵢ is
# EXACT and constant — the LM refit's Jacobian convention.
# -----------------------------------------------------------------------------

"""
    SL3_BASIS

The eight traceless generators `G₁…G₈` of `sl(3)` used by
[`HomographyManifold8`](@ref): the six off-diagonal elementary matrices plus
`diag(1, −1, 0)` and `diag(0, 1, −1)`.
"""
const SL3_BASIS = (
    SMatrix{3, 3, Float64, 9}(0, 0, 0, 1, 0, 0, 0, 0, 0),   # e₁₂
    SMatrix{3, 3, Float64, 9}(0, 0, 0, 0, 0, 0, 1, 0, 0),   # e₁₃
    SMatrix{3, 3, Float64, 9}(0, 1, 0, 0, 0, 0, 0, 0, 0),   # e₂₁
    SMatrix{3, 3, Float64, 9}(0, 0, 0, 0, 0, 0, 0, 1, 0),   # e₂₃
    SMatrix{3, 3, Float64, 9}(0, 0, 1, 0, 0, 0, 0, 0, 0),   # e₃₁
    SMatrix{3, 3, Float64, 9}(0, 0, 0, 0, 0, 1, 0, 0, 0),   # e₃₂
    SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, -1, 0, 0, 0, 0),  # diag(1,-1,0)
    SMatrix{3, 3, Float64, 9}(0, 0, 0, 0, 1, 0, 0, 0, -1),  # diag(0,1,-1)
)

"""
    HomographyManifold8(H0) <: Manifold

The minimal 8-dof `sl(3)` homography manifold about an anchor:
`H(v) = H₀·(I + Σ vᵢ·Gᵢ)` with `G₁…G₈` the traceless `sl(3)` basis
([`SL3_BASIS`](@ref)). Tracelessness deletes the homogeneity gauge
direction `H₀·εI`, so the manifold is minimal (rank 8) with no `‖H‖ = 1`
constraint and no fixed-entry singularity; `∂H/∂vᵢ = H₀·Gᵢ` is exact and
constant (first-order manifold, no dexp) — the LM refit's Jacobian
convention. The anchor is stored in its float type (`Float64` from a `Float64`
or integer `H0`). See [`Manifold`](@ref).
"""
struct HomographyManifold8{T} <: Manifold
    H0::SMatrix{3, 3, T, 9}
    HomographyManifold8{T}(H0) where {T} = new{T}(H0)
end
HomographyManifold8(H0::AbstractMatrix) = (T = float(eltype(H0)); HomographyManifold8{T}(SMatrix{3, 3, T, 9}(H0)))

tangent_size(::HomographyManifold8) = 8

# The ambient 3×3 at local coordinates `v`, H₀·(I + Σ vᵢGᵢ): eltype-generic, so
# ForwardDiff Duals flow through (`retract_jacobian`).
@inline function retract(ch::HomographyManifold8, v::StaticVector{8})
    M = SMatrix{3, 3, eltype(v), 9}(I) +
        v[1] * SL3_BASIS[1] + v[2] * SL3_BASIS[2] + v[3] * SL3_BASIS[3] +
        v[4] * SL3_BASIS[4] + v[5] * SL3_BASIS[5] + v[6] * SL3_BASIS[6] +
        v[7] * SL3_BASIS[7] + v[8] * SL3_BASIS[8]
    return HomographyMat(Tuple(ch.H0 * M))
end
# any other vector of local coordinates: read as the static 8-vector
retract(ch::HomographyManifold8, v::AbstractVector) = retract(ch, SVector{8}(v))

# Re-anchoring normalizes to unit Frobenius norm: the chart is first order, so a run
# of steps drifts the representative's scale (never its projective meaning — the
# coordinates are relative, H₀·(I + Σ vᵢGᵢ)); an ambient covariance is a covariance
# OF A REPRESENTATIVE, and this is the one it is of. @noinline per the
# bit-stability note in base/manifold.jl: a `recenter` kernel is compiled once,
# standalone, so its 3×3 products do not take a caller's fusion choices.
@noinline function recenter(ch::HomographyManifold8{S}, v) where {S}
    T = promote_type(S, eltype(v))
    H = SMatrix{3, 3, T, 9}(retract(ch, v))
    return HomographyManifold8{T}(H / norm(H))
end

