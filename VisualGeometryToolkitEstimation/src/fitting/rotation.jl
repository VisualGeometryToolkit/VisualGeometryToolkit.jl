# =============================================================================
# Rotation Fitting — the pure rotation between two calibrated views by RANSAC
# =============================================================================
#
#   find_rotation — thin wrapper around ransac() on the `RotationProblem`
#                   (problems/rotation.jl): the 2-point rotation, the bearing
#                   transfer Sampson residual in pixels, the chart-LM refit.
# =============================================================================

"""
    find_rotation(camera1, camera2, pixel_cs; threshold = 1.0, max_trials, min_trials,
        local_optimization = Irls{TLSShape}(), scoring = nothing, config = nothing, rng)
    find_rotation(bearings::Csponds; kwargs...)

Robust estimation of the rotation `R` between two views of a camera that only
rotated — `f₂ ∥ R·f₁` for the unit bearings of a correspondence, the infinite
homography — by RANSAC on the [`RotationProblem`](@ref): 2-point samples
(VisualGeometryToolkitCore's `Rotation2pt`, `kabsch` on the pair), the two-sided
transfer Sampson residual on bearings whitened into PIXELS by the views'
back-projection Jacobians, and the local optimization's chart-LM refit on the
rotation chart. A sampler (it draws hypotheses and verifies them), so a `find_*`.

Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.rotation]`); a keyword overrides. The keywords are
[`find_homography`](@ref)'s; a scorer or local optimization from another package
(RANSACScoringDoneRight's `StudentTUniformScoring(RotationProblem, a)`,
LocalOptimizationDoneRight's lifts selected by tag) plugs in through `scoring` and
`local_optimization`.

- `camera1`, `camera2`, `pixel_cs`: PIXEL correspondences `u₁ => u₂` and the
  cameras of the two views, each view back-projected with its own camera (a
  rotating camera is `find_rotation(cam, cam, cs)`); the whole field of a fisheye
  takes part, past 90°. A `:score` column is the PROSAC prior.
- `bearings`: unit bearings with their `M₁`/`M₂` Jacobian columns (see
  [`RotationProblem`](@ref)).
- `threshold`: the MSAC inlier threshold of the default scoring, in PIXELS (the
  residual's unit, as [`find_essential`](@ref)'s). Values below
  VisualGeometryToolkitFeatures' 0.06 px refinement bound are refused.

Returns a `RansacEstimate` (its `estimate` a `RotMatrix{3, Float64}`), or
`nothing` for fewer than 2 correspondences; test `converged(result)` before
reading the model.
"""
function find_rotation(camera1, camera2, pixel_cs; kwargs...)
    _holds_minimal_sample(RotationProblem, length(pixel_cs)) || return nothing
    return _ransac_pipeline(RotationProblem(camera1, camera2, pixel_cs), :rotation; kwargs...)
end
function find_rotation(bearings::VgtkCore.Csponds; kwargs...)
    _holds_minimal_sample(RotationProblem, length(bearings)) || return nothing
    return _ransac_pipeline(RotationProblem(bearings), :rotation; kwargs...)
end
