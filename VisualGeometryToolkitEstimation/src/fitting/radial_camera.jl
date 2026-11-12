"""
    find_radial_camera(correspondences; center,
        scoring, extent = nothing, workspace = nothing,
        local_optimization_strategy, max_trials, min_trials,
        config = nothing, rng = ...)

Robust estimation of one board's 1-D radial camera (VisualGeometryToolkitCore
`RadialCamera`: rows 1–2 of the plane homography, invariant to focal length
and to any radially symmetric distortion about the centre) from
`plane_point => raw_pixel` correspondences, via RANSAC on the linear 5-point
solve with MSAC scoring on the point–radial-line distance.
Returns a `RansacEstimate` whose model lives about the distortion centre
`center` (pixels, REQUIRED), the `:no_model` result (`iterations` 0) for fewer than 5 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`found_model(result)` before reading the model (see [`RansacEstimate`](@ref)).

The uncalibrated verification model for board correspondences: no distortion
parameter is estimated per board (that is the focal stack's job, across
boards), so the solve is closed form with no root to pick.
`scoring` (its width in pixels) and `local_optimization_strategy` default to
`configs/default.toml` `[pipelines.radial_camera]` — MSAC, and least squares
on the inliers, `Irls{TLSShape}()` — and the trial bounds to its `[ransac]`
table. The other keywords as [`find_homography`](@ref).

Units: the source points are plain numbers. A radial camera maps from the source
plane, so its entries carry that plane's unit, which a matrix cannot record:
physical (unitful) source points are an `ArgumentError` — strip them in the
unit the rest of your plain data uses — metres, as the stack's boundaries strip
(`ustrip.(m, X)`).
"""
function find_radial_camera(
        pairs::AbstractVector{
            <:Pair{<:_PlanePoint, <:_PlanePoint},
        };
        center,
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _refuse_physical_plane(pairs, "find_radial_camera")
    return _guarded_ransac(() -> _with_extent(RadialCameraProblem(pairs; center), extent), RadialCameraProblem{Float64}, length(pairs), :radial_camera; kwargs...)
end

"""
    refine_radial_camera(model, correspondences; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The 1-D radial camera `model` (VisualGeometryToolkitCore `RadialCamera`, about its
distortion centre) refined against `plane_point => raw_pixel` correspondences
from `model` itself (the pixel distance to the predicted radial line;
codimension 1). The problem's centre is the model's own (`center(model)`):
the keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.radial_camera] optimization_strategy`). Fewer than 5 correspondences: the guess back
unrefined, record `(ReturnCode.InitialFailure, NoRule)` (`retcode_symbol` `:degenerate`).
"""
function refine_radial_camera(
        model::VgtkCore.RadialCamera, pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        kwargs...
    )
    _refuse_physical_plane(pairs, "refine_radial_camera")
    return _guarded_refine(() -> RadialCameraProblem(pairs; center = center(model)), RadialCameraProblem{Float64}, length(pairs), Val(:radial_camera), model; kwargs...)
end
