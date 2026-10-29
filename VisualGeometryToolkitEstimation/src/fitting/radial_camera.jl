"""
    find_radial_camera(correspondences; center,
        threshold = 3.0, extent = nothing, workspace = nothing,
        local_optimization, max_trials = 10000, min_trials = 100,
        config = nothing, rng = ...)

Robust estimation of one board's 1-D radial camera (VisualGeometryToolkitCore
`RadialCamera`: rows 1–2 of the plane homography, invariant to focal length
and to any radially symmetric distortion about the centre) from
`plane_point => raw_pixel` correspondences, via RANSAC on the linear 5-point
solve with MSAC scoring on the point–radial-line distance.
Returns a `RansacEstimate` whose model lives about the distortion centre
`center` (pixels, REQUIRED), or `nothing` for fewer than 5 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

The uncalibrated verification model for board correspondences: no distortion
parameter is estimated per board (that is the focal stack's job, across
boards), so the solve is closed form with no root to pick.
`threshold` and `local_optimization` default to `configs/default.toml`
`[pipelines.radial_camera]` — least squares on the inliers, `Irls{TLSShape}()` —
and the trial bounds to its `[ransac]` table. `threshold` builds the
default scoring only; supplying it with `scoring` is an `ArgumentError`. The
other keywords as [`find_homography`](@ref).

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
    _holds_minimal_sample(RadialCameraProblem, length(pairs)) || return nothing
    problem = _with_extent(RadialCameraProblem(pairs; center), extent)
    return _ransac_pipeline(problem, :radial_camera; kwargs...)
end

"""
    refine_radial_camera(model, correspondences; threshold = 3.0, extent = nothing,
        local_optimization = Irls{TLSShape}(), scoring = nothing, workspace = nothing)
        -> RansacEstimate

The 1-D radial camera `model` (VisualGeometryToolkitCore `RadialCamera`, about its
distortion centre) refined against `plane_point => raw_pixel` correspondences:
[`find_radial_camera`](@ref)'s step without the sampling. The problem's centre
is the model's own (`center(model)`). `local_optimization` is the radial
camera's RANSAC local-optimization step, run here alone from the given model (no
sampling). Every default is `find_radial_camera`'s (`[pipelines.radial_camera]`),
`extent` included; the keywords are those of every `refine_*`, documented on
[`refine_homography`](@ref). `nothing` for fewer than 5 correspondences.
"""
function refine_radial_camera(
        model::VgtkCore.RadialCamera, pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _refuse_physical_plane(pairs, "refine_radial_camera")
    _holds_minimal_sample(RadialCameraProblem, length(pairs)) || return nothing
    problem = _with_extent(RadialCameraProblem(pairs; center = center(model)), extent)
    return _refine_pipeline(problem, :radial_camera, model; kwargs...)
end
