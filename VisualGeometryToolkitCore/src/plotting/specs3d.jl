# =============================================================================
# 3D PlotSpec Constructors
# =============================================================================
#
# Declarative PlotSpec constructors for 3D visualization:
#
#   scene3d = S.LScene(plots=[
#       VisualGeometryToolkitCore.FrameSpec(pose; scale=50),
#       Frustum(camera, bounds),
#   ])
#
# Each returns a SINGLE PlotSpec for clean composition.
#
# Also provides:
#   - LScene3D() - 3D scene with Camera3D controls
#   - camera!() - composite helper that adds frame + frustum + center
#
# =============================================================================

import Makie.SpecApi as MakieSpec
using ..VisualGeometryToolkitCore.Cameras: default_frustum_depth

# =============================================================================
# PlotSpec Constructors
# =============================================================================

"""
    FrameSpec(; origin=Point3f(0,0,0), rotation=I, scale=1.0, linewidth=3) -> PlotSpec
    FrameSpec(pose::EuclideanMap; scale=1.0, linewidth=3) -> PlotSpec

Create a coordinate frame PlotSpec (RGB XYZ axes as a single Lines spec).

Returns a single `S.Lines` PlotSpec using NaN-separated segments with per-vertex colors.

# Keyword Arguments
- `origin`: Origin point of the frame
- `rotation`: 3×3 rotation matrix to orient the frame
- `scale`: Length of each axis
- `linewidth`: Width of axis lines

# Example
```julia
import Makie.SpecApi as S

scene3d = S.LScene(plots=[
    VisualGeometryToolkitCore.FrameSpec(; scale=50.0),     # world frame
    VisualGeometryToolkitCore.FrameSpec(cam_pose; scale=30.0),  # camera frame
])
fig = plot(S.GridLayout([(1,1) => scene3d]))
```
"""
function FrameSpec(;
        origin::Point3 = Point3f(0, 0, 0),
        rotation::AbstractMatrix = SMatrix{3, 3, Float64}(I),
        scale::Real = 1.0,
        linewidth::Real = 3
    )
    segments, colors, _ = frame_segments(; origin = origin, rotation = rotation, scale = scale)
    return MakieSpec.Lines(segments; color = colors, linewidth = linewidth)
end

# Convenience: construct from EuclideanMap pose
function FrameSpec(pose::EuclideanMap; scale::Real = 1.0, linewidth::Real = 3)
    segments, colors, _ = frame_segments(pose; scale = scale)
    return MakieSpec.Lines(segments; color = colors, linewidth = linewidth)
end

# BoardMesh moved to BlobBoards

"""
    Frustum(camera, sensor_bounds; depth=automatic, color=:orange, linewidth=2) -> PlotSpec

Create a camera frustum wireframe PlotSpec.

Returns a single `S.Lines` PlotSpec showing the view frustum pyramid.

# Arguments
- `camera`: Camera object
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
- `depth`: Frustum depth (defaults to fx from K)
- `color`: Wireframe color
- `linewidth`: Line width

# Example
```julia
import Makie.SpecApi as S

# Compose camera visualization from primitives
cam_center = Point3f(pose(camera).t)
scene3d = S.LScene(plots=[
    VisualGeometryToolkitCore.FrameSpec(pose(camera); scale=50.0),  # camera axes
    Frustum(camera, bounds; color=:orange),  # frustum wireframe
    S.Scatter([cam_center]; color=:red),     # center marker
])
fig = plot(S.GridLayout([(1,1) => scene3d]))
```
"""
function Frustum(
        camera::Camera, sensor_bounds;
        depth = automatic,
        color = :orange,
        linewidth::Real = 2
    )
    eff_depth = effective_depth(camera, depth)
    segments, _, _ = frustum_segments(camera, sensor_bounds, eff_depth)
    return MakieSpec.Lines(segments; color = color, linewidth = linewidth)
end

# =============================================================================
# LScene3D - 3D Scene with Camera3D Controls
# =============================================================================

"""
    LScene3D(; show_axis=true, width=600, height=600, plots=PlotSpec[], scenekw=(;))

Create a 3D LScene with interactive Camera3D controls.

# Keyword Arguments
- `show_axis=true`: Show 3D axis
- `width=600`: Scene width
- `height=600`: Scene height
- `scenekw`: Additional scene keyword arguments

# Returns
- LScene configured for 3D visualization

# Examples
```julia
scene3d = LScene3D()

# Add visualizations
push!(scene3d.plots, VisualGeometryToolkitCore.FrameSpec(; scale=50.0))

# Or use camera! helper
camera!(scene3d, camera, bounds; frustum_color=:blue)

# Include in layout
import Makie.SpecApi as S
layout = S.GridLayout([(1,1) => scene3d])
fig = plot(layout)
```
"""
function LScene3D(;
        show_axis = true,
        width = 600,
        height = 600,
        plots = PlotSpec[],
        scenekw = (;)
    )

    lscene = MakieSpec.LScene(
        plots = plots,
        show_axis = show_axis,
        width = Makie.Fixed(width),
        height = Makie.Fixed(height),
        scenekw = scenekw
    )

    # Add callback to enable Camera3D controls
    function setup_camera3d(lscene_block)
        cam3d!(lscene_block.scene)
        return
    end

    push!(lscene.then_funcs, setup_camera3d)

    return lscene
end

# =============================================================================
# camera! - Composite Helper
# =============================================================================

"""
    camera!(lscene, camera, sensor_bounds; kwargs...)

Add camera visualization to an LScene using SpecApi (pushes specs to lscene.plots).

This is a composite helper that adds FrameSpec (axes), Frustum (wireframe), and
optionally a center marker. For individual components, use FrameSpec() and Frustum()
directly.

# Arguments
- `lscene`: LScene3D or MakieSpec.LScene to add plots to
- `camera`: Camera object
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
## Depth Control
- `depth=automatic`: Frustum depth; defaults to fx from K.

## Component Visibility
- `show_center=true`: Show camera center marker
- `show_axes=true`: Show RGB coordinate axes (X=red, Y=green, Z=blue)
- `show_frustum=true`: Show view frustum wireframe

## Styling
- `center_color=:red`: Color of center marker
- `center_size=20.0`: Size of center marker
- `axis_length=automatic`: Length of coordinate axes (defaults to depth)
- `axis_linewidth=3.0`: Line width for axes
- `frustum_color=:orange`: Color of frustum wireframe
- `frustum_linewidth=2.0`: Line width for frustum edges

# Examples
```julia
using VisualGeometryToolkitCore
import Makie.SpecApi as S

scene3d = LScene3D()
camera!(scene3d, camera, sensor_bounds; frustum_color=:blue)

# Add a frame
push!(scene3d.plots, VisualGeometryToolkitCore.FrameSpec(; scale=50.0))

# Compose and plot
layout = S.GridLayout([(1,1) => scene3d])
fig = plot(layout)
```
"""
function camera!(
        lscene, camera::Camera, sensor_bounds;
        depth = automatic,
        show_center = true,
        show_axes = true,
        show_frustum = true,
        center_color = :red,
        center_size = 20.0,
        axis_length = automatic,
        axis_linewidth = 3.0,
        frustum_color = :orange,
        frustum_linewidth = 2.0
    )
    camera = _display_mm(camera)   # a physical camera is drawn in mm

    # Compute depth and axis length using shared helper
    eff_depth = effective_depth(camera, depth)
    axis_len = axis_length === automatic ? eff_depth : axis_length

    # Get camera pose
    cam_pose = pose(camera)

    # 1. Camera center marker
    if show_center
        push!(
            lscene.plots, MakieSpec.Scatter(
                [Point3f(cam_pose.t)];
                color = center_color, markersize = center_size
            )
        )
    end

    # 2. Coordinate axes - use FrameSpec
    if show_axes
        push!(lscene.plots, FrameSpec(cam_pose; scale = axis_len, linewidth = axis_linewidth))
    end

    # 3. Frustum wireframe - use Frustum
    if show_frustum
        push!(
            lscene.plots, Frustum(
                camera, sensor_bounds;
                depth = eff_depth, color = frustum_color, linewidth = frustum_linewidth
            )
        )
    end

    return nothing
end
