# =============================================================================
# epipolar_lm.jl — the chart-LM workspace's resize hook.
#
# The chart LM (ManifoldLmWorkspace{N} + `lmsolve_chart!`, FastLevenbergMarquardt's
# in-place `lmsolve!`) is VisualGeometryCore's (src/robust/lm_core.jl); every
# chart problem plugs its rows in through the
# `ChartLm` state (interface.jl) and VGC's `residual_rows!` /
# `residual_jacobian_rows!` protocol.
# =============================================================================

# RansacWorkspace resize hook: the LM buffers are allocated at the problem's
# full data size and excluded points enter with weight 0 — nothing to resize.
_set_n!(::ManifoldLmWorkspace, ::Int) = nothing

# =============================================================================
# The epipolar chart rows — F (FundMatManifold7) and E (EssentialManifold5)
# =============================================================================
#
# ONE method for both: the √w-weighted signed Sampson rows of the chart's model
# at `a` (VGC `retract`) and their Jacobian along the chart directions (VGC
# `retract_directions`). The families differ only in the columns the Sampson
# form reads, `_sampson_columns(problem)`: none for F's pixels, the per-view
# back-projection Jacobians `M₁`/`M₂` for E's bearings (fundmat.jl, essential.jl).
#
# E's are TANGENT Sampson rows, the SAME implementation `residuals!` scores
# through. This is the fix for a regression introduced during the bearing port:
# these rows computed the IMAGE-PLANE Sampson on points that had become
# bearings, while `residuals!` computed the tangent form. Measured at one
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
#
# One body, stamped per family (not a `Union` signature: the chart and the
# problem must stay paired for inference — a union loses the pairing).
for LM in (:(ChartLm{<:FundMatManifold7, <:FundMatProblem}), :(ChartLm{<:EssentialManifold5, <:EssentialProblem}))
    @eval function residual_rows!(r::AbstractVector, lm::$LM, a)
        cs = lm.problem.cs
        return sampson_rows!(r, retract(lm.ch, a), sources(cs), targets(cs), lm.sqrtw; _sampson_columns(lm.problem)...)
    end
    @eval function residual_jacobian_rows!(J::AbstractMatrix, lm::$LM, a)
        cs = lm.problem.cs
        return sampson_jacobian_rows!(
            J, retract(lm.ch, a), VisualGeometryCore.retract_directions(lm.ch, a),
            sources(cs), targets(cs), lm.sqrtw; _sampson_columns(lm.problem)...
        )
    end
end
