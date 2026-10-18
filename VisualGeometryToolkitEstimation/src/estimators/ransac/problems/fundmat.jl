# =============================================================================
# RANSAC Fundamental Matrix Problem — the Bartoli–Sturm chart rows
# =============================================================================
#
# `FundMatProblem` is the `MinimalSolveProblem{FundMat}` alias; its
# construction, minimal solve (VGC `Fundamental7pt` on the normalized view) and
# sample/model gates are the family's (cspond.jl). The sample gate is
# correspondence coincidence only — no collinearity filter: collinearity on
# C(7,3) = 35 triplets per image would reject ~75% of valid all-inlier samples
# (SuperANSAC's policy). F takes no measurement maps (refused at construction),
# so its data frame IS its model frame. This file holds the rows of its chart,
# shared by the chart-LM refit and the HQ/GNC lift (interface.jl `ChartLm`).
#
# The chart — Bartoli–Sturm orthonormal representation (PAMI 2004), VGC's
# FundMatManifold7: minimal 7-dof about the anchor's SVD, F ~ U·diag(1, σ, 0)·Vᵀ
# with U, V ∈ O(3), σ = σ₂/σ₁ ∈ (0, 1]; recovery F = u₁v₁ᵀ + σ·u₂v₂ᵀ (their
# Eq. 2), local update (their Eq. 3, MRP in place of Euler angles) U ← U·R(x),
# V ← V·R(y), σ ← σ + δ. Near-essential anchors (σ → 1) have a one-dimensional
# gauge (co-rotation of U, V about e₃; Jacobian rank 6), handled by the LM
# damping as Bartoli–Sturm §5.1.4 recommend. The residual is VGC's signed
# Sampson row (codimension 1).
# =============================================================================

chart_dof(::Type{<:FundMatProblem}) = 7

# The chart rows are the epipolar family's one method (epipolar_lm.jl): pixel
# columns, no per-view Jacobians.
_sampson_columns(::FundMatProblem) = (;)
