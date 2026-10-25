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
unit you want the model in (`ustrip.(mm, X)`).
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
