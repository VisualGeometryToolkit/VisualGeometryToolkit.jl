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
# Whitening — Cholesky-whiten constraint + model Jacobian
# =============================================================================
#
# whiten(g, G_x, G_θ, Σ_x) → (r, J_w, ℓ)
#
# Given algebraic constraint g, measurement Jacobian G_x, and model Jacobian
# G_θ, compute the whitened residual and model Jacobian via Cholesky of the
# projected measurement covariance Σ_g = G_x Σ_x G_xᵀ.
#
# Two methods matching the codimension d_g:
#   - Vector (d_g ≥ 2): Cholesky forward substitution
#   - Scalar (d_g = 1):  division by √c
# =============================================================================

# THE ONE SAMPSON DEGENERACY RULE of the estimation stack. A correspondence whose
# measurement gradient vanishes (to rounding) has Sampson distance 0 when the
# constraint vanishes with it (it satisfies the constraint exactly) and Inf when
# it does not (no nearby measurement can). No floor, and no zero row that would
# score an unsatisfiable correspondence as a perfect inlier. `whiten` applies it
# to the residual it returns — per component, with the sign of gᵢ — and every
# Sampson distance is `whiten`'s residual (`sampson_distance`,
# `signed_sampson_distance`), so they cannot disagree.
@inline _degenerate_sampson(vanishes::Bool) = vanishes ? 0.0 : Inf

"""
    constraint_rounding_scale(M, u₁, u₂) -> NamedTuple{(:gradient, :value)} | nothing

The rounding scales of the constraint `g(M; u₁, u₂)` at a correspondence — the
one input the Sampson degeneracy rule needs beyond `(g, G_x)`: `gradient`, below
which the measurement gradient `‖G_x‖` is rounding (DEGENERATE), and `value`,
below which `|gᵢ|` is rounding (the constraint VANISHES; a scalar, or one per
component). Relative to the model's and the points' own magnitudes, so a
model's homogeneous scale never decides. [`whiten`](@ref) consumes it; the
default, `nothing`, makes only an exactly zero gradient degenerate and an
exactly zero `g` vanishing. The epipolar models return
[`epipolar_rounding`](@ref)'s scales, a homography its rows' term-by-term bound.
"""
constraint_rounding_scale(M, u₁, u₂) = nothing

# The two verdicts of the rule at a rounding scale ρ (`nothing`: exact zeros).
@inline _gradient_degenerate(::Nothing, G_x) = false
@inline _gradient_degenerate(ρ::NamedTuple, G_x) = sum(abs2, G_x) <= ρ.gradient^2
@inline _vanishes(::Nothing, g) = all(iszero, g)
@inline _vanishes(ρ::NamedTuple, g::Real) = abs(g) <= ρ.value
@inline _vanishes(ρ::NamedTuple, g::StaticVector) = all(abs.(g) .<= ρ.value)

# The whitened residual of a degenerate measurement under the rule: 0, or ±Inf
# with the sign of each gᵢ.
@inline _degenerate_residual(g::Real, vanishes::Bool) = _signed_degenerate(g, vanishes)
@inline _degenerate_residual(g::StaticVector, vanishes::Bool) = map(gᵢ -> _signed_degenerate(gᵢ, vanishes), g)
@inline function _signed_degenerate(gᵢ, vanishes::Bool)
    z = oftype(gᵢ, _degenerate_sampson(vanishes))
    return gᵢ < 0 ? -z : z
end

"""
    whiten(g::SVector{dg}, G_x::SMatrix{dg,nx}, G_θ::SMatrix{dg,nθ}, Σ_x = I, ρ = nothing)
        -> (r::SVector{dg}, J_w::SMatrix{dg,nθ}, ℓ::T)

Cholesky-whiten a vector constraint (d_g ≥ 2).

Returns `(L⁻¹g, L⁻¹G_θ, log|Σ_g|)` where `LLᵀ = Σ_g = propagate(G_x, Σ_x)`.
A degenerate `Σ_g` (Cholesky fails, or it loses rank to rounding relative to its
diagonal) follows the one Sampson degeneracy rule: the residual is `0` when `g`
vanishes and `±Inf` otherwise (no nearby measurement satisfies the
constraint), the Jacobian rows are zero and `ℓ` is `0`. "Vanishes" is at the
rounding scale `ρ` ([`constraint_rounding_scale`](@ref)), exactly zero without
one. A scorer reads it as an outlier; a fit that takes it as an inlier fails
loudly on the non-finite row.
"""
@inline function whiten(
        g::StaticVector{dg, T}, G_x::StaticMatrix{dg, nx, T},
        G_θ::StaticMatrix{dg, nθ, T}, Σ_x = I, ρ = nothing
    ) where {dg, nx, nθ, T}
    C, d, ok = _vector_whitening(G_x, Σ_x, ρ)
    isnan(d) && return (g .* T(NaN), G_θ .* T(NaN), T(NaN))   # NaN in, NaN out
    ok || return (_degenerate_residual(g, _vanishes(ρ, g)), zero(G_θ), zero(T))
    L = C.L
    return (L \ g, L \ G_θ, log(d))
end

# The residual alone, `first(whiten(g, G_x, G_θ, Σ_x, ρ))` without the model
# Jacobian and log-determinant a distance does not read (a `log` the compiler
# cannot drop, as it may throw): the Sampson distances' hot path. The verdicts
# are `whiten`'s own (`_vector_whitening`, `_scalar_whitening`).
@inline function _whiten_residual(g::StaticVector{dg, T}, G_x::StaticMatrix{dg, nx, T}, Σ_x = I, ρ = nothing) where {dg, nx, T}
    C, d, ok = _vector_whitening(G_x, Σ_x, ρ)
    isnan(d) && return g .* T(NaN)
    ok || return _degenerate_residual(g, _vanishes(ρ, g))
    return C.L \ g
end
@inline function _whiten_residual(g::T, G_x::StaticVector{nx, T}, Σ_x = I, ρ = nothing) where {nx, T <: Real}
    c, ok = _scalar_whitening(G_x, Σ_x, ρ)
    ok || return _degenerate_residual(g, _vanishes(ρ, g))
    return g / sqrt(c)
end

# The whitening of a vector constraint and its degeneracy verdict: Σ_g's
# Cholesky, its determinant and `ok` — Σ_g keeps its rank and the gradient is
# above the rounding scale ρ.
@inline function _vector_whitening(G_x, Σ_x, ρ)
    C, d, ok = _whitening_cholesky(propagate(G_x, Σ_x))   # static for any Σ_x storage
    return C, d, ok && !_gradient_degenerate(ρ, G_x)
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
factor frozen once (at a seed) where [`whiten`](@ref) re-derives it per call.
A degenerate `Σ_g` ([`whiten`](@ref)'s rule) is a ZERO factor — its rows drop
out of a solve; NaN propagates.
"""
@inline function whitening_factor(G_x::StaticMatrix{dg, nx, T}, Σ_x = I) where {dg, nx, T}
    C, d, ok = _whitening_cholesky(propagate(G_x, Σ_x))
    isnan(d) && return fill(T(NaN), SMatrix{dg, dg, T})
    return ok ? inv(SMatrix{dg, dg, T}(C.L)) : zero(SMatrix{dg, dg, T})
end
# A dense G_x (a block of a dense Jacobian, whose size is not a type
# parameter): the same rule, as a dense factor.
function whitening_factor(G_x::AbstractMatrix{T}, Σ_x = I) where {T <: AbstractFloat}
    dg = size(G_x, 1)
    C, d, ok = _whitening_cholesky(Matrix{T}(propagate(G_x, Σ_x)))
    isnan(d) && return fill(T(NaN), dg, dg)
    return ok ? Matrix(inv(C.L)) : zeros(T, dg, dg)
end

"""
    whiten(g::T, G_x::SVector{nx}, G_θ::SVector{nθ}, Σ_x = I, ρ = nothing)
        -> (r::T, J_w::SVector{nθ}, ℓ::T)

Scalar whitening for d_g = 1 constraints.

Returns `(g/√c, G_θ/√c, log(c))` where `c = G_xᵀ Σ_x G_x`. When the gradient is
degenerate — `c` is not positive, or `‖G_x‖` is below the rounding scale `ρ`
([`constraint_rounding_scale`](@ref)) — the one Sampson degeneracy rule applies,
as in the vector method: residual `0` when `g` vanishes (at `ρ`; exactly
without one), `±Inf` otherwise, zero Jacobian, `ℓ = 0`. Without `ρ` a sum of
squares has no cancellation to measure, so only an exactly vanishing gradient
is degenerate: a small `c` is a legitimately small (e.g. rescaled) model, whose
whitened residual is scale-free. NaN propagates. Generic in the element type
(`T <: Real`), so ForwardDiff duals flow through a scalar constraint's rows.
"""
@inline function whiten(
        g::T, G_x::StaticVector{nx, T},
        G_θ::StaticVector{nθ, T}, Σ_x = I, ρ = nothing
    ) where {nx, nθ, T <: Real}
    c, ok = _scalar_whitening(G_x, Σ_x, ρ)
    ok || return (_degenerate_residual(g, _vanishes(ρ, g)), zero(G_θ), zero(T))
    sqrt_c = sqrt(c)
    return (g / sqrt_c, G_θ / sqrt_c, log(c))
end

# c = G_xᵀΣ_xG_x and the scalar verdict: `ok` unless c is not positive (NaN
# propagates) or the gradient is below the rounding scale ρ.
@inline function _scalar_whitening(G_x, Σ_x, ρ)
    c = _propagate_scalar(G_x, Σ_x)
    return c, (c > zero(c) || isnan(c)) && !_gradient_degenerate(ρ, G_x)
end

# ∂vec(A·M·B)/∂vec(M) = kron(Bᵀ, A) (column-major vec) — the derivative of a
# model conjugated by fixed maps, M ↦ A·M·B: what a constraint evaluated on the
# conjugated model pulls its model gradient back through (`Jᵀ·∂g/∂vec(M′)` for a
# scalar constraint's 9-vector, `G_θ·J` for a 2×9). The frame change of an
# epipolar model is A = H₂⁻ᵀ, B = H₁⁻¹; of a homography A = D₂, B = D₁⁻¹.
# Private until the composition API that states the conjugation lands.
@inline _conjugation_jacobian(A::StaticMatrix, B::StaticMatrix) = kron(transpose(B), A)

@inline _propagate_scalar(v::StaticVector, ::UniformScaling{Bool}) = dot(v, v)
@inline _propagate_scalar(v::StaticVector, Σ::UniformScaling) = Σ.λ * dot(v, v)
@inline _propagate_scalar(v::StaticVector, Σ) = dot(v, Σ * v)

# =============================================================================
# WhiteningPolicy — the residual-currency STRATEGY of an estimator
# =============================================================================

"""
    WhiteningPolicy

How an algebraic constraint `g(θ, x) = 0` with derivatives `(g, G_x, G_θ)`
from [`constraint_with_jacobians`](@ref) becomes the residual/Jacobian rows a
least squares sees — the estimator's residual CURRENCY. Orthogonal to the
robust [`AbstractWeightPolicy`](@ref) (which weights rows by their size); the
two compose: whiten first (currency), weight second (robustness). One verb,
[`whitened_rows`](@ref), for every estimator in the family — the Sampson
error of F/E, the whitened transfer error of H, the point–radial-line
distance of the radial camera, the pixel currency of the division lift.

- [`NoWhitening`](@ref) — the algebraic rows as they are.
- [`SampsonWhitening`](@ref) — Cholesky-whiten by the propagated unit
  isotropic measurement noise `G_x·G_xᵀ`: first-order geometric error.
"""
abstract type WhiteningPolicy end

"Algebraic rows, unwhitened. See [`WhiteningPolicy`](@ref)."
struct NoWhitening <: WhiteningPolicy end
"First-order geometric (Sampson) whitening by G_x·G_xᵀ. See [`WhiteningPolicy`](@ref)."
struct SampsonWhitening <: WhiteningPolicy end
"""
    whitened_rows(policy::WhiteningPolicy, g, G_x, G_θ) -> (r, J)

The residual `r` and model-Jacobian rows `J` of one constraint under the
policy: `(g, G_θ)` for `NoWhitening`, `(L⁻¹g, L⁻¹G_θ)` from [`whiten`](@ref)
otherwise. Scalar (d_g = 1) and vector (d_g ≥ 2) constraints alike.
"""
@inline whitened_rows(::NoWhitening, g, G_x, G_θ) = (g, G_θ)
@inline function whitened_rows(::SampsonWhitening, g, G_x, G_θ)
    r, J, _ = whiten(g, G_x, G_θ)
    return r, J
end
