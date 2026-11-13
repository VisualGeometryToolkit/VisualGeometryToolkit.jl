# =============================================================================
# sampson_lm.jl — the chart-LM rows of every Sampson-constraint problem, and
# the chart-LM workspace's resize hook.
#
# The chart LM (ManifoldLmWorkspace{N} + `lmsolve_chart!`, NLLSsolver on the
# rows) is VisualGeometryToolkitCore's (src/robust/lm_core.jl); every
# chart problem plugs its rows in through the
# `ChartLm` state (interface.jl) and VGC's `residual_rows!` /
# `residual_jacobian_rows!` protocol.
# =============================================================================

# RansacWorkspace resize hook: the LM buffers are allocated at the problem's
# full data size and excluded points enter with weight 0 — nothing to resize.
_set_n!(::ManifoldLmWorkspace, ::Int) = nothing

# =============================================================================
# The chart rows of a Sampson-constraint problem — ONE method
# =============================================================================
#
# Every chart problem whose residual is its constraint's whitened Sampson
# residual (F, E, H, the circle, the radial camera, the rotation) descends
# EXACTLY what it scores: VGC's batch rows over the problem's observation set
# (its `cs`, with its noise and keyword columns and the LM's √w as
# `sqrt_weights`), at the chart's model, along the chart's directions — the
# normalizer re-evaluated at every iterate and its derivative carried. A
# problem whose residual is something else (the ellipse's orthogonal distance,
# PnP's reprojection, the radial homography's transfer, the hand-eye and pose
# rows) defines its own rows on its `ChartLm`.
const SampsonChartProblem = Union{FundMatProblem, EssentialProblem, HomographyProblem, CircleProblem,
    RadialCameraProblem, RotationProblem}

# The per-problem facts: the constraint's model at chart coordinates `a`
# (`_constraint_model`, default the chart's `retract`) and its directions
# (`retract_jacobian`, in the coordinates of the constraint's `G_θ`).
function residual_rows!(r, lm::ChartLm{<:Any, <:SampsonChartProblem}, a)
    p = lm.problem
    return weighted_whitened_residuals!(r, _constraint_model(p, lm.ch, _chart_point(p, a)), _observation_set(lm))
end
function residual_jacobian_rows!(J, lm::ChartLm{<:Any, <:SampsonChartProblem}, a)
    p = lm.problem; av = _chart_point(p, a)
    return weighted_whitened_jacobian_rows!(J, _constraint_model(p, lm.ch, av), retract_jacobian(lm.ch, av), _observation_set(lm))
end
@inline _chart_point(p, a) = SVector{chart_dof(typeof(p))}(a)
_constraint_model(_p, ch, a) = retract(ch, a)
# The problem's observations with the LM's per-element √w (one per datum: the
# rows' √w repeats per constraint component).
_observation_set(lm::ChartLm) = withattrs(lm.problem.cs; sqrt_weights = _element_sqrtw(lm.sqrtw, Val(codimension(typeof(lm.problem)))))
@inline _element_sqrtw(sqrtw, ::Val{1}) = sqrtw
@inline _element_sqrtw(sqrtw, ::Val{C}) where {C} = view(sqrtw, 1:C:length(sqrtw))

# F (FundMatManifold7) and E (EssentialManifold5) are two of them: the
# problem's `cs` with its columns (E's per-view back-projection Jacobians
# `M₁`/`M₂`, read by VGC's `constraint_keywords`), at the chart's model, along
# its `retract_jacobian`.
#
# E's are TANGENT Sampson rows, the SAME implementation `mahalanobis_distances!` scores
# through. This is the fix for a regression introduced during the bearing port:
# these rows computed the IMAGE-PLANE Sampson on points that had become
# bearings, while `mahalanobis_distances!` computed the tangent form. Measured at one
# model, the two disagreed by a median ratio of 1.11 and a MAXIMUM of 446x —
# not a scale factor but a different function. The optimiser descended
# something unrelated to what was scored: cost flat-to-rising (0.006378 ->
# 0.006416) while reporting `:converged`, and a 3.4-degree seed landing at 26
# degrees. When two functions compute one quantity by different routes,
# changing one is not a partial fix, it is a new defect.
#
# WHY THE JACOBIAN IS THE THING TO VERIFY, not the residual: a per-view M₁/M₂
# swap moves the residual by 1.8% and the JACOBIAN by 341% (0.19% on the
# residual at equal focals, which hides inside anything anyone would call
# noise). A swap destroys the descent DIRECTION, not the cost — which is
# exactly the symptom above, and why the check that matters is a finite
# difference with two DIFFERENT focal lengths.
