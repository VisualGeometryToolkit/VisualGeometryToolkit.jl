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

# Units
The world points are plain numbers or lengths, all one form. Lengths
(FlexUnits quantities, e.g. `X .* m`) are read in metres at entry — exactly,
the stored value — the search runs on plain metres, and the pose comes back
physical: a unitful `CameraExtrinsics`, its translation in metres, equal after
`ustrip(m, ·)` to the plain search's on `ustrip(m, X)` with the same `rng`.
Plain points give a plain pose in their own unit, as before. The residuals,
`rms` and `sigma` are pixels either way. A unitful `extent` goes with unitful
points (a plain one with plain points).

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
        pairs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2, <:Real}} = nothing,
        kwargs...
    )
    _holds_minimal_sample(AbsolutePoseProblem, length(pairs)) || return nothing
    problem = _with_extent(AbsolutePoseProblem(camera_model, pairs), extent)
    return _ransac_pipeline(problem, :absolute_pose; kwargs...)
end

# Physical world points: stripped in metres once, the plain search, the pose
# read back in metres once.
function find_absolute_pose(
        camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:_PhysicalPoint3, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2, <:Quantity}} = nothing,
        kwargs...
    )
    plain = _map_csponds(_plain_metres, identity, pairs)::AbstractVector{<:Pair{<:StaticVector{3, <:Real}}}
    result = find_absolute_pose(camera_model, plain; extent = extent === nothing ? nothing : _plain_metres(extent), kwargs...)
    return result === nothing ? nothing : _in_metres(result)
end

"""
    refine_absolute_pose(pose, camera_model, correspondences; threshold = 8.0,
        extent = nothing, local_optimization = Irls{TLSShape}(),
        scoring = nothing, workspace = nothing) -> RansacEstimate

The camera extrinsics `pose` (world → camera, `x_c = R·X + t`) refined against
`X => u` pairs (world point => raw pixel) of the camera `camera_model`:
[`find_absolute_pose`](@ref)'s step without the sampling, on the same
[`AbsolutePoseProblem`](@ref) — the raw-pixel reprojection error through the
full camera, on the `LeftRigidManifold6` chart; any lens, rays at and past 90°
included. `local_optimization` is the pose's RANSAC local-optimization step, run
here alone from the given model (no sampling): `pose` is scored, its consensus
set under the scorer is the one a consensus-anchored strategy reads its scale
from, and the strategy runs once from it. Plain least squares over every
correspondence is `local_optimization = Irls{GaussianShape}()` at a `threshold`
above every residual. Every default is `find_absolute_pose`'s
(`[pipelines.absolute_pose]`), `extent` included; the other keywords are those
of every `refine_*`, documented on [`refine_homography`](@ref). `nothing` for
fewer than 3 correspondences.

# Units
As [`find_absolute_pose`](@ref): world points (and the pose's translation) are
plain numbers or lengths, all one form; physical ones are read in metres and
the pose comes back physical, in metres. The residuals are pixels either way.
"""
function refine_absolute_pose(
        pose::PlainEuclideanMap{3}, camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2, <:Real}} = nothing,
        kwargs...
    )
    _holds_minimal_sample(AbsolutePoseProblem, length(pairs)) || return nothing
    problem = _with_extent(AbsolutePoseProblem(camera_model, pairs), extent)
    return _refine_pipeline(problem, :absolute_pose, unlabel(pose); kwargs...)
end

# Physical pose and world points: stripped in metres once, the plain
# refinement, the pose read back in metres once.
function refine_absolute_pose(
        pose::_PhysicalPose3, camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:_PhysicalPoint3, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2, <:Quantity}} = nothing,
        kwargs...
    )
    plain = _map_csponds(_plain_metres, identity, pairs)::AbstractVector{<:Pair{<:StaticVector{3, <:Real}}}
    result = refine_absolute_pose(_plain_metres(pose)::PlainEuclideanMap{3}, camera_model, plain;
        extent = extent === nothing ? nothing : _plain_metres(extent), kwargs...)
    return result === nothing ? nothing : _in_metres(result)
end
