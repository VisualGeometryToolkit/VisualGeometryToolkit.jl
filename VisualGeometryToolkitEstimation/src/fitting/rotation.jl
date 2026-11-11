# =============================================================================
# Rotation Fitting — the pure rotation between two calibrated views by RANSAC
# =============================================================================
#
#   find_rotation — thin wrapper around ransac() on the `RotationProblem`
#                   (problems/rotation.jl): the 2-point rotation, the bearing
#                   transfer Sampson residual in pixels, the chart-LM refit.
# =============================================================================

"""
    find_rotation(camera1, camera2, pixel_cs; scoring = MsacScoring(problem; threshold = 1.0), max_trials, min_trials,
        local_optimization_strategy = Irls{TLSShape}(), config = nothing, rng)
    find_rotation(bearings::Csponds; kwargs...)

Robust estimation of the rotation `R` between two views of a camera that only
rotated — `f₂ ∥ R·f₁` for the unit bearings of a correspondence, the infinite
homography — by RANSAC on the [`RotationProblem`](@ref): 2-point samples
(VisualGeometryToolkitCore's `Rotation2pt`, `kabsch` on the pair), the Sampson
distance of the joint measurement (both images' pixels, Σ = I) to the transfer
constraint on bearings, in PIXELS through the views' back-projection Jacobians, and the local optimization's chart-LM refit on the
rotation chart. A sampler (it draws hypotheses and verifies them), so a `find_*`.

Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.rotation]`); a keyword overrides. The keywords are
[`find_homography`](@ref)'s; a scorer or local optimization from another package
(RANSACScoringDoneRight's `StudentTUniformScoring(RotationProblem, a)`,
LocalOptimizationDoneRight's lifts selected by tag) plugs in through `scoring` and
`local_optimization_strategy`.

- `camera1`, `camera2`, `pixel_cs`: PIXEL correspondences `u₁ => u₂` and the
  cameras of the two views, each view back-projected with its own camera (a
  rotating camera is `find_rotation(cam, cam, cs)`); the whole field of a fisheye
  takes part, past 90°. A `:score` column is the PROSAC prior.
- `bearings`: unit bearings with their `M₁`/`M₂` Jacobian columns (see
  [`RotationProblem`](@ref)).
- `scoring`: the scorer, its width in PIXELS (the residual's unit, as
  [`find_essential`](@ref)'s); default `[pipelines.rotation] scoring`. Widths below
  VisualGeometryToolkitFeatures' 0.06 px refinement bound are refused.

Returns a `RansacEstimate` (its `estimate` a `RotMatrix{3, Float64}`), or
the `:no_model` result (`iterations` 0) for fewer than 2 correspondences; test `converged(result)` before
reading the model.
"""
function find_rotation(camera1, camera2, pixel_cs; kwargs...)
    return _guarded_ransac(() -> RotationProblem(camera1, camera2, pixel_cs), RotationProblem, length(pixel_cs), :rotation; kwargs...)
end
function find_rotation(bearings::VgtkCore.Csponds; kwargs...)
    return _guarded_ransac(() -> RotationProblem(bearings), RotationProblem, length(bearings), :rotation; kwargs...)
end

"""
    refine_rotation(R, camera1, camera2, pixel_cs; scoring = MsacScoring(problem; threshold = 1.0),
        optimization_strategy = Irls{TLSShape}(), workspace = nothing)
    refine_rotation(R, bearings::Csponds; kwargs...)
        -> RansacEstimate

The rotation `R` of a camera that only rotated, refined against its
correspondences: [`find_rotation`](@ref)'s step without the sampling, on the
same [`RotationProblem`](@ref) (the forms of `find_rotation`). `optimization_strategy`
is the rotation's RANSAC local-optimization step, run here alone from the given
model (no sampling). Every default is `find_rotation`'s (`[pipelines.rotation]`);
the keywords are those of every `refine_*`, documented on
[`refine_homography`](@ref). The `:no_model` result (`iterations` 0) for fewer than 2 correspondences.
"""
function refine_rotation(R, camera1, camera2, pixel_cs; kwargs...)
    return _guarded_refine(() -> RotationProblem(camera1, camera2, pixel_cs), RotationProblem, length(pixel_cs), :rotation, R; kwargs...)
end
function refine_rotation(R, bearings::VgtkCore.Csponds; kwargs...)
    return _guarded_refine(() -> RotationProblem(bearings), RotationProblem, length(bearings), :rotation, R; kwargs...)
end
