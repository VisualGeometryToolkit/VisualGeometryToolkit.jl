# =============================================================================
# The one model covariance — the constraint sandwich over an index set
# =============================================================================
#
# Every covariance RANSAC scores with is ONE function: VGC's
# `implicit_covariance` on the problem's own constraint rows,
# `constraint_with_jacobians(p, θ, i)` for `i` in an index set — the rows
# `residuals!` whitens and predictive scoring contracts, in the data frame
# (measurement maps included: the rows are already there). On a minimal
# sample the system is square and the sandwich IS the implicit-function
# covariance of the minimal solve (the hypothesis's `solver_covariance`); on an
# overdetermined set — the local optimization's inlier set, a seed's consensus
# set — it is the first-order covariance of the whitened-constraint estimator
# on that set (the refit's). Until 2026-09-22 each problem carried its own
# provider (VGC's minimal-only implicit covariance per model family, a
# hand-transported mapped-E special case, a closed-form 2-point line) and the
# refits emitted none.
# =============================================================================

# A problem is admitted by predictive scoring when its constraint rows exist:
# RSDR's `provides_solver_covariance(::Type{<:P}) = true`, declared beside each
# problem's `constraint_with_jacobians` (cspond.jl, p3p.jl, hand_eye.jl,
# radial_camera.jl, line.jl) — no central list to fall out of step with them.

# The covariance's coordinates → G_θ's: for the ambient-row models (H, F, E,
# the radial camera) G_θ is over the model's ambient coordinates (vec(M),
# vec(P)), and the covariance lives in the model's minimal chart
# (`minimal_manifold`: sl(3), Bartoli–Sturm, so(3) ⊕ T_tS², S⁵) — its
# directions are the chart's differential. The pose rows are already in chart
# coordinates (`nothing`: the identity), as are hand-eye's and the lines'.
_model_directions(::MinimalSolveProblem, θ) = retract_jacobian(minimal_manifold(θ))
_model_directions(::AbsolutePoseProblem, _θ) = nothing
_model_directions(_p, _θ) = nothing

# One index's block (Aᵢ, Bᵢ[, Σ̃ₓ]): the constraint's model rows in the
# covariance's coordinates, its measurement rows (VGC's `block_rows`, the form
# `implicit_covariance` reads), and the per-point noise shape Σ̃ₓ when the
# constraint carries one (its 4th value) — the problem's own noise model, the
# one `residuals!` whitens by. `noise_weighted = true` below makes the estimator
# the (BᵢΣ̃ₓBᵢᵀ)⁻¹-weighted one `residuals!` scores (VGC folds Σ̃ₓ into Bᵢ).
@inline _model_rows(::Nothing, Gθ) = block_rows(Gθ)
@inline _model_rows(T, Gθ) = block_rows(Gθ) * T
@inline function _constraint_block(T, cd::Tuple)
    A = _model_rows(T, cd[3])
    B = block_rows(cd[2])
    return _with_noise_shape(A, B, length(cd) >= 4 ? cd[4] : I)
end
# The isotropic shape (`I`, and a constraint without a 4th value) adds nothing;
# the rounding scale a 5th value carries is the scorers', not the covariance's.
@inline _with_noise_shape(A, B, ::LinearAlgebra.UniformScaling) = (A, B)
@inline _with_noise_shape(A, B, Σ̃ₓ) = (A, B, SMatrix{size(B, 2), size(B, 2)}(Σ̃ₓ))

# The covariance as a static D×D (D from the block's model rows — a compile-time
# size), in G_θ's coordinates.
@inline _static(Σ::AbstractMatrix, ::Val{D}) where {D} = SMatrix{D, D, Float64}(Σ)
@inline _in_model_coordinates(::Nothing, Σ) = Σ
@inline _in_model_coordinates(T, Σ) = propagate(T, Σ)

"""
    solver_covariance(p, θ, idx) -> SMatrix or nothing

UNIT-SCALE covariance of the model `θ` from the constraint rows
`constraint_with_jacobians(p, θ, i)`, `i ∈ idx`: VGC's `implicit_covariance`,
`(AᵀW₀A)⁻¹·AᵀW₀·BΣₓBᵀ·W₀A·(AᵀW₀A)⁻¹` with the estimator's own whitening
`W₀ = blockdiag((BᵢΣ̃ₓBᵢᵀ)⁻¹)`. On a minimal sample (square) it is the
implicit-function covariance of the minimal solve; on an overdetermined set
(the local optimization's inlier set, a seed's consensus set) the covariance
of the whitened-constraint estimator on it.

Coordinates are those of `G_θ` — `vec(M)` for H/F/E (the covariance of the
minimal chart `minimal_manifold(θ)`, propagated through its differential; rank ≤
its dof), the chart coordinates for the pose (LeftRigidManifold6), hand-eye
(RigidExpManifold6) and line. Unit noise in the data frame the rows are in,
so the MULTIPLICATIVE convention holds: physical `Σ = s²·Σ̃`, `s` profiled by
the scale marginal. `nothing` for fewer points than a minimal sample or a
singular system — a degenerate sample, skipped. Monte Carlo calibration:
test/ransac_mc_calibration_tests.jl.
"""
function solver_covariance(p::AbstractRansacProblem, θ, idx::AbstractVector{Int})
    # A problem with no constraint rows gets RSDR's refusal, the one message.
    provides_solver_covariance(typeof(p)) ||
        return invoke(solver_covariance, Tuple{AbstractConstraintProblem, Any, AbstractVector{Int}}, p, θ, idx)
    _holds_minimal_sample(p, length(idx)) || return nothing
    T = _model_directions(p, θ)
    block(i) = _constraint_block(T, constraint_with_jacobians(p, θ, i))
    Σ = implicit_covariance(block, idx; noise_weighted = true)
    Σ === nothing && return nothing
    return _in_model_coordinates(T, _static(Σ, Val(size(first(block(first(idx))), 2))))
end
