# =============================================================================
# Covariance Propagation
# =============================================================================
#
# propagate(J, Σ) = J * Σ * J'
#
# Propagates a covariance matrix Σ through a linear map with Jacobian J.
# This is the fundamental operation for:
#   - Measurement → constraint covariance:  Σ̃_g = propagate(∂g/∂x, Σ̃_x)
#   - Measurement → parameter covariance:   Σ̃_θ = propagate(∂θ/∂x, Σ̃_x)
#   - Projective conic transfer:            C*' = propagate(H, C*)
#
# Specializations avoid unnecessary matrix–matrix–matrix products when the
# covariance has simple structure (identity, scalar, diagonal).
# =============================================================================

"""
    propagate(J, Σ)

Propagate covariance `Σ` through the linear map with Jacobian `J`:

    propagate(J, Σ) = J * Σ * Jᵀ

Specializations:
- `Σ = I`  (`UniformScaling{Bool}`): returns `J * Jᵀ`
- `Σ = λI` (`UniformScaling`): returns `λ * (J * Jᵀ)`
- `Σ = diag(...)` (`Diagonal`): returns `J * Σ * Jᵀ` via column scaling
- General matrix: full sandwich product `J * Σ * Jᵀ`
- Static `J` with a dense `Σ` (`Matrix`, `Symmetric`, `Diagonal{Vector}`): `Σ`
  is converted once to its static size, so the result is static like `J`

One of three covariance verbs: `propagate` goes THROUGH A MAP (its Jacobian, or
a framed map carrying its frames), [`transform_covariance`](@ref) BETWEEN CHARTS
of one model, [`implicit_covariance`](@ref) of an ESTIMATE.

GTSAM equivalent: a function's `OptionalJacobian` `H`, with the caller forming
`H·Σ·Hᵀ`.
"""
@inline propagate(J, ::UniformScaling{Bool}) = J * J'
@inline propagate(J, Σ::UniformScaling) = Σ.λ * (J * J')
@inline propagate(J, Σ::Diagonal) = J * Σ * J'
@inline propagate(J, Σ) = J * Σ * J'
# A static J keeps the result static whatever Σ's storage: without these, a
# dense Σ turns `J * Σ * J'` into a heap Matrix.
@inline propagate(J::StaticMatrix, Σ::StaticMatrix) = J * Σ * J'
@inline propagate(J::StaticMatrix{m, n}, Σ::AbstractMatrix) where {m, n} = J * SMatrix{n, n}(Σ) * J'
@inline propagate(J::StaticMatrix{m, n}, Σ::Diagonal) where {m, n} = J * Diagonal(SVector{n}(Σ.diag)) * J'

# =============================================================================
# The Sampson degeneracy rule and the whitening Cholesky
# =============================================================================
#
# The pieces base/sampson.jl's two layers share: the one degeneracy rule, the
# relative rank verdict of a propagated covariance, and the frozen whitening
# factor. The layers themselves (sampson_residual_with_normal, whiten) live
# there, after the measurement-noise rule they read.
# =============================================================================

# THE ONE SAMPSON DEGENERACY RULE of the estimation stack. A correspondence whose
# measurement gradient vanishes (to rounding) has Sampson distance 0 when the
# constraint vanishes with it (it satisfies the constraint exactly) and Inf when
# it does not (no nearby measurement can). No floor, and no zero row that would
# score an unsatisfiable correspondence as a perfect inlier. The Sampson layer
# applies it to the residual it returns — per component, with the sign of gᵢ —
# and every Sampson distance and whitened residual is built on that residual,
# so they cannot disagree.
@inline _degenerate_sampson(vanishes::Bool) = ifelse(vanishes, 0.0, Inf)

# The value verdict of the rule at a rounding scale ρ (`nothing`: exact zeros);
# the gradient one is `_gradient_bound²` (base/sampson.jl).
@inline _vanishes(::Nothing, g) = all(iszero, g)
@inline _vanishes(ρ::NamedTuple, g::Real) = abs(g) <= ρ.value
@inline _vanishes(ρ::NamedTuple, g::StaticVector) = all(abs.(g) .<= ρ.value)

# The whitened residual of a degenerate measurement under the rule: 0, or ±Inf
# with the sign of each gᵢ.
@inline _degenerate_residual(g::Real, vanishes::Bool) = _signed_degenerate(g, vanishes)
@inline _degenerate_residual(g::StaticVector, vanishes::Bool) = map(gᵢ -> _signed_degenerate(gᵢ, vanishes), g)
@inline function _signed_degenerate(gᵢ, vanishes::Bool)
    z = oftype(gᵢ, _degenerate_sampson(vanishes))
    return ifelse(gᵢ < 0, -z, z)
end

# The Cholesky of a propagated constraint covariance and the ONE whitening
# degeneracy verdict: Σ_g must not lose rank to rounding RELATIVE to its diagonal
# (the homography Sampson distance's own test, so the two agree) — never an
# absolute threshold, which a model's homogeneous scale would decide. A barely-PSD
# Σ_g whose determinant rounds to ≤ 0 (measured: −7e-18 on a clean focal stack)
# is the same case; never `log` a non-positive number.
@inline function _whitening_cholesky(Σ_g::AbstractMatrix{T}) where {T}
    d = det(Σ_g)
    C = cholesky(Symmetric(Σ_g); check = false)
    return C, d, issuccess(C) && d > 8 * eps(T) * prod(Σ_g[i, i] for i in axes(Σ_g, 1))
end

"""
    whitening_factor(G_x::SMatrix{dg,nx}, Σ_x = I) -> SMatrix{dg,dg}

The inverse Cholesky factor `L⁻¹` of the propagated constraint covariance
`Σ_g = G_x·Σ_x·G_xᵀ = L·Lᵀ`: whitened rows are `L⁻¹·g` and `L⁻¹·G_θ`, the
factor frozen once (at a seed, or at the previous iteration's model of an
iteratively reweighted fit) where [`whiten`](@ref) re-derives it per call — the
Gauss–Newton currency of a constraint whose LM row holds its whitening fixed. A
scalar constraint's gradient (an `SVector`) gives the number `1/√(G_x·Σ_x·G_xᵀ)`.
A degenerate `Σ_g` (its relative rank verdict) is a ZERO factor — its rows drop
out of a solve; NaN propagates.
"""
@inline function whitening_factor(G_x::StaticMatrix{dg, nx, T}, Σ_x = I) where {dg, nx, T}
    C, d, ok = _whitening_cholesky(propagate(G_x, Σ_x))
    isnan(d) && return fill(T(NaN), SMatrix{dg, dg, T})
    return ok ? inv(SMatrix{dg, dg, T}(C.L)) : zero(SMatrix{dg, dg, T})
end
# A scalar constraint's gradient (an `SVector`): the factor is the number
# `1/√(G_x·Σ_x·G_xᵀ)`, zero when degenerate.
@inline whitening_factor(G_x::StaticVector{nx, T}, Σ_x = I) where {nx, T} =
    only(whitening_factor(SMatrix{1, nx, T}(Tuple(G_x)), Σ_x))
# A dense G_x (a block of a dense Jacobian, whose size is not a type
# parameter): the same rule, as a dense factor.
function whitening_factor(G_x::AbstractMatrix{T}, Σ_x = I) where {T <: AbstractFloat}
    dg = size(G_x, 1)
    C, d, ok = _whitening_cholesky(Matrix{T}(propagate(G_x, Σ_x)))
    isnan(d) && return fill(T(NaN), dg, dg)
    return ok ? Matrix(inv(C.L)) : zeros(T, dg, dg)
end

# ∂vec(A·M·B)/∂vec(M) = kron(Bᵀ, A) (column-major vec) — the derivative of a
# model conjugated by fixed maps, M ↦ A·M·B: what a constraint evaluated on the
# conjugated model pulls its model gradient back through (`Jᵀ·∂g/∂vec(M′)` for a
# scalar constraint's 9-vector, `G_θ·J` for a 2×9). The frame change of an
# epipolar model is A = H₂⁻ᵀ, B = H₁⁻¹; of a homography A = D₂, B = D₁⁻¹.
# Private until the composition API that states the conjugation lands.
@inline _conjugation_jacobian(A::StaticMatrix, B::StaticMatrix) = kron(transpose(B), A)
