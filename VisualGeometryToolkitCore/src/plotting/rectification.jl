# =============================================================================
# plot_rectification — raw | rectified panels in ONE pixel frame
# =============================================================================
#
# Camera-level, so it lives here rather than in a downstream package: everything
# below is `CameraModel` + `undistort`/`undistorted_rect` + `imshow`. Downstream
# records (BlobBoards' `Calibration`, a rig, …) add their own one-line method
# that forwards to these.
# =============================================================================

# The principal point of a camera's pixel frame (every panel's cross and title)
# is Cameras' `_principal_point`, imported by the extension.

# Centre and 180° radius of a camera's pixel frame. `monotone_radius` returns
# `Inf` for a lens with no pole, so `isfinite(r_inf)` is the test for "this lens
# has a horizon to draw".
# (`horizon_radius` is the perspective-monotone horizon, not `image_circle`; see its definition.)
_horizon(camera::CameraModel) = (_principal_point(camera), horizon_radius(camera))

# The raw panel's per-camera overlay: the 180° horizon circle (when the lens has
# one, labelled with its radius for the legend) and the principal-point cross,
# both in `colour`. The circle goes to `poly!` as a 360-vertex polygon: the
# default circle tessellation is visibly faceted at a horizon's radius.
function _raw_overlay!(ax, camera::CameraModel, colour, label::AbstractString)
    c, r_inf = _horizon(camera)
    isfinite(r_inf) && poly!(
        ax, GeometryBasics.Polygon(boundary_points(Circle(c, r_inf), 360));
        strokecolor = colour, label = "$label: r∞ $(round(r_inf; digits = 0)) px"
    )
    Makie.scatter!(ax, [c[1]], [c[2]]; color = colour, marker = :cross, markersize = 18)
    return isfinite(r_inf)
end

# ONE rectified panel into a caller's grid slot, its cross and title in `colour`.
# Returns the axis so a caller can annotate it.
function _rectified_panel!(
        pos, camera::CameraModel, img::AbstractMatrix, max_px::Real, label::AbstractString, colour;
        max_fov::Real
    )
    H, W = size(img)
    f = calibration_matrix(camera)[1, 1]
    c = _principal_point(camera)
    # CAPPED BY FIELD OF VIEW, not by a fraction of the horizon: flattening the
    # last degree before the pole is unbounded (`tan θ`), and without this a
    # fisheye rectifies to a ~370 000 px plane that downscales to a thumbnail.
    ext = undistorted_rect(camera, (W, H); max_fov)
    σ = max(1.0, widths(ext)[1] / max_px)
    ci = undistort(camera, img, ext; scale = σ)
    fov_txt = isfinite(max_fov) ? ", fov ±$(round(Int, rad2deg(max_fov)))°" : ""
    ax, _ = imshow(
        pos, ci;
        title = "$label — f=$(round(f; digits = 1)), pp=($(round(c[1]; digits = 1)), $(round(c[2]; digits = 1)))$fov_txt  (↓×$(round(σ; digits = 2)))",
        titlecolor = colour
    )
    # The sensor footprint is the frame rect every other call in this chain
    # hands around (`undistorted_rect`, `undistort!`, `imshow(pos, img, rect)`),
    # so it is drawn as that rect rather than as a loose corner list.
    Makie.poly!(
        ax, Rect(0, 0, W, H);
        color = :transparent, strokecolor = :cyan, strokewidth = 1.5, linestyle = :dash
    )
    Makie.scatter!(ax, [c[1]], [c[2]]; color = colour, marker = :cross, markersize = 18)
    return ax
end

"""
    plot_rectification(cameras::AbstractVector, img; names = String[], max_px = 4200) -> Figure
    plot_rectification(camera::CameraModel, img; max_px = 4200, name = "") -> Figure

One raw panel, then one FULL-extent rectification PER CAMERA, all in the same
pixel frame. Each camera's 180° radius circle and principal-point cross are
drawn on the shared raw panel in the colour of its own rectified panel's title,
with the radius in the legend, so a camera whose pole falls INSIDE the frame is
visible before you look at its rectification — which is the usual way two
calibrations of the same session differ. Each rectified panel is a self-placing
`FramedImage` (the rect form of [`undistort`](@ref) returns the canvas WITH its
pixel-frame window), so the sensor footprint (dashed) and the cross land where
they belong with nothing carried by hand. `max_px` caps the rendered width; the
downsample folds into the fused warp.

This is the form for judging one fit against another: a linear calibration
against the bundle that refined it, or one robust cutoff against another. The
single-camera method is the same figure for one camera.
"""
function plot_rectification(
        cameras::AbstractVector, img::AbstractMatrix;
        names::AbstractVector{<:AbstractString} = String[], max_px::Real = 4200,
        max_fov::Real = DEFAULT_MAX_FOV
    )
    isempty(cameras) && throw(ArgumentError("plot_rectification: no cameras to compare"))
    labels = isempty(names) ? ["camera $i" for i in eachindex(cameras)] : collect(names)
    length(labels) == length(cameras) ||
        throw(ArgumentError("plot_rectification: $(length(names)) names for $(length(cameras)) cameras"))
    palette = Makie.wong_colors()
    fig = Makie.Figure(size = (760 * (length(cameras) + 1), 620))
    ax1, _ = imshow(fig[1, 1], img; title = "raw")
    any_horizon = false
    for (i, cam) in enumerate(cameras)
        colour = palette[mod1(i, length(palette))]
        any_horizon |= _raw_overlay!(ax1, cam, colour, labels[i])
        _rectified_panel!(fig[1, i + 1], cam, img, max_px, labels[i], colour; max_fov)
    end
    # a legend needs a labelled horizon: pinhole-like lenses have none to show
    any_horizon && Makie.axislegend(ax1; position = :lt, framevisible = true)
    return fig
end

plot_rectification(camera::CameraModel, img::AbstractMatrix; name::AbstractString = "", kwargs...) =
    plot_rectification([camera], img; names = [isempty(name) ? "rectified" : name], kwargs...)
