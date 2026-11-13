# =============================================================================
# RANSAC Homography Problem — gates, frames, and the sl(3) chart rows
# =============================================================================
#
# `HomographyProblem` is the `MinimalSolveProblem{HomographyMat}` alias; its
# construction, minimal solve and sample/model gates are the family's
# (cspond.jl). This file holds what is specific to the homography: the
# consensus gate, the model↔data frame law of a mapped problem, and the rows
# of its chart (the refit and the lift share them, interface.jl `ChartLm`).
# =============================================================================

# The model-class gates (`test_sample`, `test_model`, `test_consensus`) are
# GEOMETRY and live in VisualGeometryToolkitCore (`geometry/model_validity.jl`),
# shared verbatim with offline dataset generators; the family wires them to the
# problem's normalized view (cspond.jl). Only the consensus gate is H's own.

# Post-scoring consensus check (loop.jl, after sort-and-sweep populates
# `ws.mask`): re-run cheirality on the full inlier set to reject bow-tie
# homographies that a sample-only check misses. Delegates to VGC's
# `test_consensus(H, cs, mask) = test_cheirality(H, cs, mask)`, with `H` and the
# points in the SAME frame: `H` is a model-frame homography, so it meets the
# solve view (`cs_solve`, the observation-mapped points; `=== cs` unmapped).
test_consensus(p::MinimalSolveProblem{<:HomographyMat}, H::HomographyMat, mask::BitVector) =
    test_consensus(H, p.cs_solve, mask)

# The local optimizations (the chart below: the refit and the HQ/GNC lift) run
# in the DATA frame — on `p.cs`, whose whitened
# transfer residuals are the scoring currency the loss width is stated in —
# and hand back a MODEL-frame homography, the frame `solve` returns and scoring
# (`constraint_with_jacobians`) composes into the data frame. On a mapped
# problem the model changes frame by composition with the observation maps,
# once on entry (`_chart`: `inv(C₂) ∘ H ∘ C₁`) and once on exit (`_model_at`:
# `C₂ ∘ H_data ∘ inv(C₁)`); identity maps leave the matrix untouched.

# =============================================================================
# Sampson Error — Algebraic Constraint Formulation (Chum et al.)
# =============================================================================
#
# For correspondence (s, d) = ((x,y), (x',y')) and homography H, the algebraic
# constraint g = 0 encodes d̃ ∝ H·s̃:
#
#   g₁ = h₁ᵀs̃ − x'·h₃ᵀs̃
#   g₂ = h₂ᵀs̃ − y'·h₃ᵀs̃
#
# The observation Jacobian ∂ₓg (2×4) w.r.t. (x, y, x', y'):
#   ∂ₓg = [ H[1,1]−x'H[3,1]   H[1,2]−x'H[3,2]   −c    0  ]
#          [ H[2,1]−y'H[3,1]   H[2,2]−y'H[3,2]    0   −c  ]
# where c = h₃ᵀs̃.
#
# The constraint covariance (the observation-noise rule, VisualGeometryToolkitCore
# conventions/measurement-noise.md): Σ̃_{gᵢ} = G₁Σ₁G₁ᵀ + G₂Σ₂G₂ᵀ (2×2), with
# Σ = I for a plain point (so ∂ₓg·∂ₓgᵀ for two plain points), the point's own
# for an `Uncertain` one, 0 for an `exact` one. A board in metres is supplied
# `exact`: as a plain point its unit noise sits in METRES, Σ̃ ≈ (board→image
# scale)²·I and the residual comes out in metres — a 3 px threshold then admits
# 40 px mismatches. Exact, G₂ = −c·I gives Σ̃ = c²·I and r = the transfer error
# in target pixels.
#
# Sampson distance: d_S = √(gᵀ Σ̃_g⁻¹ g)
# Whitened residual: r_w = L⁻¹g where LLᵀ = Σ̃_g
#
# The model Jacobian ∂_θg (2×9) w.r.t. vec(H) (column-major):
#   ∂g₁/∂vec(H) = [x, 0, −x'x,  y, 0, −x'y,  1, 0, −x']
#   ∂g₂/∂vec(H) = [0, x, −y'x,  0, y, −y'y,  0, 1, −y']
#
# =============================================================================

# =============================================================================
# The chart — VGC's minimal sl(3) HomographyManifold8 (8 dof, codim 2)
#
# A homography is a 3×3 matrix modulo scale — locally SL(3). The chart about
# the anchor is H(x) = H₀·(I + Σ xᵢGᵢ) with G₁…G₈ the TRACELESS basis of
# sl(3): tracelessness deletes exactly the homogeneity direction H₀·εI (the
# gauge the dehomogenized residual cannot see), so the parameterization is
# minimal with no ‖H‖=1 constraint and no fixed-entry singularity, and the
# affine chart has EXACT constant Jacobians ∂H/∂xᵢ = H₀Gᵢ — no dexp.
#
# Residuals are the codimension-2 whitened algebraic constraint in the DATA
# frame — EXACTLY what scoring computes: rᵢ = √wᵢ · Lᵢ⁻¹gᵢ with LᵢLᵢᵀ =
# G_x·Σᵢ·G_xᵀ under the endpoints' observation noise (`I` for plain points;
# conventions/measurement-noise.md: an exact source gives the transfer error in
# target pixels), the factor re-evaluated at every iterate and its derivative
# carried — the Sampson-constraint problems' one chart-rows method
# (sampson_lm.jl, VGC's `weighted_whitened_residuals!` / `weighted_whitened_jacobian_rows!` with
# the homography's `measurement_gradient_jacobian`), one KernelAbstractions
# kernel each on a GPU backend, a plain loop on the CPU. (Until 0.4 the rows
# whitened the TRANSFER form with a factor frozen at the chart's anchor: a
# second function, equal to scoring's only at the anchor, whose fixed point is
# not the Sampson cost's minimum.) The chart lives in the data frame (the
# scoring currency); a mapped problem's model-frame H is composed in by
# `_chart` and out by `_model_at`. The same chart and rows drive the chart-LM
# refit and, at unit weights, the HQ/GNC lift (problems/manifold_providers.jl).
# =============================================================================

chart_dof(::Type{<:HomographyProblem}) = 8

# The chart at a model-frame H: VGC's HomographyManifold8 about its data-frame
# form, at the unit-Frobenius representative (the chart's own `recenter`, the
# one re-anchor policy — the representative the ambient covariance is of).
function _chart(p::HomographyProblem, H::HomographyMat)
    Hd = _has_observation_maps(p) ? inv(p.dst_frame_map) ∘ H ∘ p.src_frame_map : H
    return _reanchored(minimal_manifold(Hd))
end

# The model a chart stands for, composed back into the model frame — then the
# unmapped path's gauge and eltype: unit Frobenius norm, Float64.
function _model_at(p::HomographyProblem, ch::HomographyManifold8)
    Hd = model(ch)
    _has_observation_maps(p) || return Hd
    Hm = p.dst_frame_map ∘ Hd ∘ inv(p.src_frame_map)
    return HomographyMat{Float64}(Tuple(unit_normalize(SMatrix{3, 3, Float64, 9}(Hm))))
end

