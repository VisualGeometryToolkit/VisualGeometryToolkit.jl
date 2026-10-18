# =============================================================================
# Absolute Pose Fitting
# =============================================================================
#
# Robust absolute pose estimation via P3P + RANSAC.
#
#   find_absolute_pose — thin wrapper around ransac() with composable scoring
#              and local optimization
#
# =============================================================================

# =============================================================================
# Public API
# =============================================================================

"""
    find_absolute_pose(camera_model, correspondences; threshold=8.0,
        max_trials=10000, min_trials=100,
        local_optimization=Irls{TLSShape}(),
        scoring=nothing, config=nothing)

Robust absolute pose estimation via P3P + RANSAC with MSAC scoring.
Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.absolute_pose]`); a keyword overrides.

Returns a `RansacEstimate` or `nothing` if fewer than 3 correspondences (the
P3P minimal sample).
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `correspondences`: `X => u` pairs (world point => raw pixel) — `Csponds`,
  `StructArray{<:Pair}` or a Pair vector; a `Csponds` `:score` column is the
  per-match prior (PROSAC; uniform otherwise).
- `threshold`: **in PIXELS** (the reprojection residual's unit), as [`find_homography`](@ref).
- `extent`: as [`find_homography`](@ref); the pose's `_check_extent` rejects
  poses failing cheirality or yielding a non-simple / non-convex image quad.
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace(AbsolutePoseProblem, n_data[, T])`.
- `local_optimization`: default `Irls{TLSShape}()` on the 6-dof pose chart, as
  [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).

# Examples
```julia
result = find_absolute_pose(camera_model, correspondences)
result = find_absolute_pose(camera_model, correspondences; threshold=10.0)
```
"""
function find_absolute_pose(
        camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:StaticVector{3}, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _holds_minimal_sample(AbsolutePoseProblem, length(pairs)) || return nothing
    problem = _with_extent(AbsolutePoseProblem(camera_model, pairs), extent)
    return _ransac_pipeline(problem, :absolute_pose; kwargs...)
end
