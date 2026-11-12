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
    find_absolute_pose(camera_model, correspondences; scoring,
        max_trials, min_trials,
        local_optimization_strategy, config=nothing)

Robust absolute pose estimation via P3P + RANSAC with MSAC scoring.
Every default is `configs/default.toml`'s (`[ransac]`,
`[pipelines.absolute_pose]`); a keyword overrides.

Returns a `RansacEstimate`, the `:no_model` result (`iterations` 0) for fewer than 3 correspondences
(the P3P minimal sample).
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
- `scoring`: the scorer, its width **in PIXELS** (the reprojection residual's unit), as [`find_homography`](@ref); default `[pipelines.absolute_pose] scoring`.
- `extent`: as [`find_homography`](@ref); the pose's `_check_extent` rejects
  poses failing cheirality or yielding a non-simple / non-convex image quad.
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace([T,] AbsolutePoseProblem, n_data)`.
- `local_optimization_strategy`: default `[pipelines.absolute_pose] local_optimization_strategy` on the 6-dof pose chart, as
  [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).

# Examples
```julia
result = find_absolute_pose(camera_model, correspondences)
result = find_absolute_pose(camera_model, correspondences; scoring = MsacScoring(AbsolutePoseProblem; threshold = 10.0))
```
"""
function find_absolute_pose(
        camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{2}}};
        extent::Union{Nothing, HyperRectangle{2, <:Real}} = nothing,
        kwargs...
    )
    return _guarded_ransac(() -> _with_extent(AbsolutePoseProblem(camera_model, pairs), extent), AbsolutePoseProblem, length(pairs), :absolute_pose; kwargs...)
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
    return _attach_metres(find_absolute_pose(camera_model, plain; extent = extent === nothing ? nothing : _plain_metres(extent), kwargs...))
end

"""
    refine_absolute_pose(pose, camera_model, correspondences; optimization_strategy, scale | width, workspace = nothing)
    refine_absolute_pose(pose, bearing_correspondences; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The camera extrinsics `pose` (world → camera, `x_c = R·X + t`) refined against
`X => u` pairs (world point => raw pixel) of the camera `camera_model`, from
`pose` itself, on [`find_absolute_pose`](@ref)'s [`AbsolutePoseProblem`](@ref)
— the raw-pixel reprojection error through the full camera (codimension 2), on
the `LeftRigidManifold6` chart; any lens, rays at and past 90° included. Plain
least squares over every correspondence is `optimization_strategy = nothing`.
The keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.absolute_pose] optimization_strategy`). Fewer than 3 correspondences:
the guess back unrefined, `retcode` `:degenerate`.

# Bearings: a camera given by its rays (central or generalized)
The second form takes `X => f` pairs of world points and the camera-frame UNIT
bearings observing them (`StaticVector{3}` both), for a camera the stack holds
no `CameraModel` of — the caller back-projects; optional attribute columns
`origins` (each ray's origin in the camera frame: a non-central camera), `M`
(each bearing's back-projection Jacobian: residuals whitened into pixels,
else radians) and `weights` (fixed per-correspondence weights). Its problem is
a [`BearingPoseProblem`](@ref): the tangent-plane residual of bearings.md,
whose norm is the angle between the rays, on the same `LeftRigidManifold6`
chart; rays at and past 90° and behind the camera are data like any other.
Its default is every refine's (`[refine.bearing_pose]`, `Triggs{WelschShape}`
at the caller's `scale` or `width`, in the residual's unit: radians, or pixels
with an `M` column); least squares over every correspondence at the caller's
weights is `optimization_strategy = nothing`. It has no `find_*`: a
generalized camera has no minimal solver here. The guess's ray origins must
not coincide with their points (`ArgumentError`). The keywords are the shared
refine set; there is no `max_iter` (the cap is the strategy's).

# Frames and units
The refined pose comes back with the guess's frame labels, its translation an
`SVector{3, Float64}` (plain) or lengths. As [`find_absolute_pose`](@ref):
world points (and the pose's translation) — and ray origins — are plain
numbers or lengths, all one form; physical ones are read in metres and the
pose comes back physical, in metres. The residuals are pixels (radians for
bearings without `M`) either way.
"""
function refine_absolute_pose(
        pose::PlainEuclideanMap{3}, camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{2}}};
        kwargs...
    )
    return _with_frames(_guarded_refine(() -> AbsolutePoseProblem(camera_model, pairs), AbsolutePoseProblem, length(pairs),
                                        Val(:absolute_pose), unlabel(pose); kwargs...), pose)
end

# Physical pose and world points: stripped in metres once, the plain
# refinement, the pose read back in metres once.
function refine_absolute_pose(
        pose::_PhysicalPose3, camera_model::CameraModel,
        pairs::AbstractVector{<:Pair{<:_PhysicalPoint3, <:StaticVector{2}}};
        kwargs...
    )
    plain = _map_csponds(_plain_metres, identity, pairs)::AbstractVector{<:Pair{<:StaticVector{3, <:Real}}}
    return _attach_metres(refine_absolute_pose(_plain_metres(pose)::PlainEuclideanMap{3}, camera_model, plain; kwargs...))
end

function refine_absolute_pose(
        pose::PlainEuclideanMap{3},
        cs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}};
        kwargs...
    )
    θ₀ = unlabel(pose)
    function problem()
        p = BearingPoseProblem(cs)
        R0 = rotation(θ₀); t0 = translation(θ₀); o = p.context.origins
        for i in 1:data_size(p)
            iszero(norm(R0 * sources(p.cs)[i] + t0 - o[i])) &&
                throw(ArgumentError("refine_absolute_pose: point $i coincides with its ray's origin at the guess"))
        end
        return p
    end
    return _with_frames(_guarded_refine(problem, BearingPoseProblem, length(cs), Val(:bearing_pose), θ₀; kwargs...), pose)
end

# Physical pose, points and origins: in metres once, the plain refinement, the
# pose read back in metres once.
function refine_absolute_pose(
        pose::_PhysicalPose3, cs::AbstractVector{<:Pair{<:_PhysicalPoint3, <:StaticVector{3, <:Real}}}; kwargs...
    )
    plain = _map_csponds(_plain_metres, identity, cs)
    if plain isa VgtkCore.Csponds && VgtkCore.hasattr(plain, :origins)
        plain = withattrs(plain; origins = map(_plain_metres, plain.origins))
    end
    return _attach_metres(refine_absolute_pose(_plain_metres(pose)::PlainEuclideanMap{3}, plain; kwargs...))
end

# frames.md rule 1, frame in = frame out: the refined pose carries the guess's
# frame labels (the problem's workspace labels its models `Extrinsics3`'s own).
# A failed result (no estimate) stays failed.
_with_frames(r::FitResult, pose::EuclideanMap{3}) = _map_estimate(
    E -> relabel(VgtkCore.map_frames(pose)..., EuclideanMap(RotMatrix{3, Float64}(rotation(E)), SVector{3, Float64}(translation(E)))), r
)
