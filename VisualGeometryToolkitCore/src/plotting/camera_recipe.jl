# ==============================================================================
# Camera Visualization Recipe
# ==============================================================================
#
# Makie recipe for unified camera visualization supporting both SpecApi and
# non-SpecApi usage:
#
#   # Non-SpecApi (immediate)
#   cameraplot!(scene3d, cam, bounds)
#   cameraplot!(scene3d, cam, bounds; depth=500, show_axes=false)
#
#   # SpecApi (hybrid: create layout, then add to resulting LScene)
#   scene = LScene3D(); layout = S.GridLayout([(1,1) => scene])
#   fig = plot(layout); lscene = fig.figure.layout.content[1].content
#   cameraplot!(lscene, cam, bounds; frustum_color=:blue)
#
# ==============================================================================

using ..VisualGeometryToolkitCore.Cameras: default_frustum_depth

"""
    cameraplot(camera, sensor_bounds; kwargs...)
    cameraplot!(scene, camera, sensor_bounds; kwargs...)

Visualize a camera with center marker, coordinate axes, and view frustum.

Works with both SpecApi composition and direct plotting.

# Arguments
- `camera`: Camera object
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
## Depth Control
- `depth=automatic`: Frustum depth; defaults to fx from K.

## Component Visibility
- `show_center=true`: Show camera center marker
- `show_axes=true`: Show RGB coordinate axes (X=red, Y=green, Z=blue)
- `show_frustum=true`: Show view frustum wireframe
- `show_near_plane=false`: Show near clipping plane
- `show_up_indicator=true`: Show triangle indicating camera up direction

## Center Marker
- `center_color=:red`: Color of center marker
- `center_size=20.0`: Size of center marker

## Axes
- `axis_length=automatic`: Length of coordinate axes (defaults to `depth / 4`,
  following COLMAP's convention — a small gnomon at the camera center)
- `axis_linewidth=3.0`: Line width for axes

## Frustum
- `frustum_color=:orange`: Color of frustum wireframe
- `frustum_linewidth=2.0`: Line width for frustum edges
- `frustum_image=nothing`: Optional image to texture on far plane

## Other
- `near_depth=10.0`: Near plane depth (if show_near_plane=true)
- `indicator_size=40.0`: Size of up indicator triangle (in pixels)

# Examples

```julia
# Basic usage
fig = Figure()
scene3d = LScene(fig[1,1])
cam3d!(scene3d.scene)
cameraplot!(scene3d, camera, sensor_bounds)

# Customized
cameraplot!(scene3d, camera, bounds;
    show_axes=false,
    frustum_color=:blue,
    frustum_image=captured_image)

# Depth can also be given explicitly
cameraplot!(scene3d, logical_cam, bounds; depth=500.0)

# SpecApi composition (hybrid: create layout first, then add camera)
scene3d = LScene3D()
push!(scene3d.plots, S.Scatter([Point3(0,0,0)]; color=:black))  # placeholder
layout = S.GridLayout([(1,1) => scene3d])
fig = plot(layout)
lscene = fig.figure.layout.content[1].content  # get actual LScene
cameraplot!(lscene, camera, bounds; frustum_color=:cyan)

# Multiple cameras
for (cam, color) in zip(cameras, [:red, :green, :blue])
    cameraplot!(scene3d, cam, bounds; frustum_color=color, show_center=false)
end
```
"""
@recipe(CameraPlot, camera, sensor_bounds) do scene
    Attributes(
        # Depth control
        depth = automatic,

        # Component visibility
        show_center = true,
        show_axes = true,
        show_frustum = true,
        show_near_plane = false,
        show_up_indicator = true,

        # Center marker
        center_color = :red,
        center_size = 20.0,

        # Axes
        axis_length = automatic,
        axis_linewidth = 3.0,

        # Frustum
        frustum_color = :orange,
        frustum_linewidth = 2.0,
        frustum_image = nothing,
        frustum_image_alpha = 1.0,     # uniform alpha for the far-plane texture

        # Near plane
        near_depth = 10.0,

        # Up indicator
        indicator_size = 40.0,
    )
end

# Note: Uses effective_depth() and frustum_segments() from geometry.jl

# Main plot implementation
function Makie.plot!(p::CameraPlot)
    # Extract values from observables
    camera = _display_mm(p.camera[])   # a physical camera is drawn in mm
    sensor_bounds = p.sensor_bounds[]

    # Compute depth and axis length using shared helper
    depth = effective_depth(camera, p.depth[])
    # Default axis length follows COLMAP's convention: a small fraction of
    # the frustum depth so the RGB gnomon reads as a pose marker at the
    # camera center rather than competing visually with the frustum edges.
    axis_len = p.axis_length[] === automatic ? depth / 4 : p.axis_length[]

    # Get camera pose
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)

    # -------------------------------------------------------------------------
    # 1. Camera center marker
    # -------------------------------------------------------------------------
    if p.show_center[]
        scatter!(
            p, [cam_center];
            color = p.center_color,
            markersize = p.center_size
        )
    end

    # -------------------------------------------------------------------------
    # 2. Coordinate axes - use shared geometry
    # -------------------------------------------------------------------------
    if p.show_axes[]
        segments, colors, axis_ends = frame_segments(cam_pose; scale = axis_len)
        lines!(p, segments; color = colors, linewidth = p.axis_linewidth)

        # Endpoint markers
        scatter!(
            p, collect(axis_ends);
            color = [:red, :green, :blue],
            markersize = 8
        )
    end

    # -------------------------------------------------------------------------
    # 3. Frustum - use shared geometry
    # -------------------------------------------------------------------------
    if p.show_frustum[]
        segments, corners_world, _ = frustum_segments(camera, sensor_bounds, depth)

        # Optional: image texture on far plane (before wireframe)
        if !isnothing(p.frustum_image[])
            _draw_frustum_image!(
                p, corners_world, p.frustum_image[],
                p.frustum_image_alpha[]
            )
        end

        # Draw frustum wireframe using lines with shared segments
        lines!(
            p, segments;
            color = p.frustum_color,
            linewidth = p.frustum_linewidth
        )

        # Optional: near clipping plane
        if p.show_near_plane[]
            # Only draw the far plane rectangle part (skip center-to-corner edges)
            # Near plane is just the rectangle at near depth
            corners_2d = Point2{Float64}.(coordinates(sensor_bounds))
            near_cam = unproject.(Ref(camera.model), corners_2d, p.near_depth[])
            near_world = Point3f.(cam_pose.(near_cam))
            nan_pt = Point3f(NaN, NaN, NaN)
            near_rect = [
                near_world[1], near_world[2], nan_pt,
                near_world[2], near_world[3], nan_pt,
                near_world[3], near_world[4], nan_pt,
                near_world[4], near_world[1],
            ]
            lines!(
                p, near_rect;
                color = p.frustum_color,
                linewidth = p.frustum_linewidth
            )
        end

        # Up indicator triangle at top of frustum
        if p.show_up_indicator[]
            _draw_up_indicator!(
                p, camera, cam_pose, sensor_bounds, depth,
                p.indicator_size[], p.frustum_color[]
            )
        end
    end

    return p
end

# -----------------------------------------------------------------------------
# Convenience: derive sensor_bounds from the supplied frustum_image
# -----------------------------------------------------------------------------
"""
    cameraplot!(scene, camera; frustum_image, kwargs...)

Plot a camera using the supplied `frustum_image`'s size as the sensor
extent — eliminates the need to supply `sensor_bounds` separately when
the image fully determines it (COLMAP convention: `Rect2(0, 0, w, h)`).
All other keyword arguments match the three-argument form.
"""
function cameraplot!(scene, camera::AbstractCamera; frustum_image, kwargs...)
    h, w = size(frustum_image)
    bounds = Rect2(0.0, 0.0, Float64(w), Float64(h))
    return cameraplot!(scene, camera, bounds; frustum_image = frustum_image, kwargs...)
end

# -----------------------------------------------------------------------------
# Helper: draw image texture on frustum far plane
# -----------------------------------------------------------------------------
function _draw_frustum_image!(p, corners_world, image, alpha::Real = 1.0)
    # Create unit rectangle mesh with UV coordinates
    unit_rect_mesh = GeometryBasics.uv_normal_mesh(Rect(0, 0, 1, 1))
    rect_positions = GeometryBasics.coordinates(unit_rect_mesh)

    # Bilinear interpolation to map unit rect to frustum far plane
    transformed_positions = map(rect_positions) do pos
        u, v = pos[1], pos[2]
        p1, p2, p3, p4 = corners_world[1], corners_world[2], corners_world[3], corners_world[4]
        Point3f((1 - u) * (1 - v) * p1 + u * (1 - v) * p2 + u * v * p3 + (1 - u) * v * p4)
    end

    far_mesh = GeometryBasics.Mesh(
        transformed_positions,
        GeometryBasics.faces(unit_rect_mesh);
        uv = GeometryBasics.texturecoordinates(unit_rect_mesh)
    )

    # Makie textures expect (x, y) array order, but Julia images are
    # (row, col) = (y, x). The unit-rect mesh's UVs go from (0,0) at the
    # top-left corner to (1,1) at the bottom-right — same handedness as
    # the COLMAP image convention, so only a transpose is needed (no axis
    # flip). `rotr90` flips horizontally because it reverses one axis on
    # top of the transpose; use `permutedims` instead.
    texture = permutedims(parent(image))

    return mesh!(
        p, far_mesh;
        color = texture,
        shading = Makie.NoShading,
        interpolate = true,
        alpha = alpha,
        transparency = alpha < 1
    )
end

# -----------------------------------------------------------------------------
# Helper: draw up indicator triangle
# -----------------------------------------------------------------------------
function _draw_up_indicator!(p, camera, cam_pose, sensor_bounds, depth, indicator_size, color)
    sensor_center = Point2f(centroid(sensor_bounds))
    top_y = Float64(sensor_bounds.origin[2])  # Top edge (min y, since y points down in image)
    base_half_width = Float64(sensor_bounds.widths[1]) * 0.02  # 2% of sensor width

    # Triangle vertices in 2D image space (Float64 for unproject compatibility)
    tip_2d = Point2f(sensor_center[1], top_y - indicator_size)
    base_left_2d = Point2f(sensor_center[1] - base_half_width, top_y)
    base_right_2d = Point2f(sensor_center[1] + base_half_width, top_y)

    # Unproject to 3D and transform to world coordinates
    triangle_2d = [tip_2d, base_left_2d, base_right_2d]
    vertices_world = Point3f.(
        cam_pose.(
            unproject.(
                Ref(camera.model),
                Point2{Float64}.(triangle_2d), Ref(depth)
            )
        )
    )

    indicator_triangle = GeometryBasics.Triangle(vertices_world...)
    return mesh!(p, indicator_triangle; color = color)
end
