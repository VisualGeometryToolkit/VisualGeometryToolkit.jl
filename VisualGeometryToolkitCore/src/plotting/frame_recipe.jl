# ==============================================================================
# Coordinate Frame Visualization Recipe
# ==============================================================================
#
# Makie recipe for visualizing coordinate frames (EuclideanMap) supporting both
# SpecApi and non-SpecApi usage:
#
#   # Non-SpecApi (immediate)
#   frameplot!(scene3d, pose)
#   frameplot!(scene3d, pose; scale=50, linewidth=3)
#
#   # SpecApi (hybrid: create layout, then add to resulting LScene)
#   scene = LScene3D()
#   layout = S.GridLayout([(1,1) => scene])
#   fig = plot(layout)
#   lscene = fig.figure.layout.content[1].content
#   frameplot!(lscene, pose; scale=50)
#
# ==============================================================================

using Rotations: Rotation, RotMatrix
using StaticArrays: SMatrix

"""
    frameplot(frame; kwargs...)
    frameplot!(scene, frame; kwargs...)

Visualize a coordinate frame with RGB axes (X=red, Y=green, Z=blue).

Works with both SpecApi composition and direct plotting.

# Arguments
- `frame`: EuclideanMap representing the frame's pose (rotation + translation).
  Use `EuclideanMap(I, zeros(3))` or `EuclideanMap(RotMatrix{3}(I), Point3f(0,0,0))`
  for the world frame at origin.

# Keyword Arguments
- `scale=1.0`: Length of each axis
- `linewidth=3.0`: Width of axis lines
- `markersize=8.0`: Size of endpoint markers
- `show_markers=true`: Whether to show markers at axis endpoints

# Examples

```julia
using VisualGeometryToolkitCore
using GLMakie

# Basic usage - world frame
fig = Figure()
scene3d = LScene(fig[1,1])
cam3d!(scene3d.scene)
frameplot!(scene3d, EuclideanMap(RotMatrix{3}(I), zeros(3)); scale=50.0)

# Camera frame
cam_pose = pose(camera)
frameplot!(scene3d, cam_pose; scale=30.0, linewidth=2.0)

# Multiple frames with different colors would need separate calls
# (axes are always RGB for XYZ)
```
"""
@recipe(FramePlot, frame) do scene
    Attributes(
        scale = 1.0,
        linewidth = 3.0,
        markersize = 8.0,
        show_markers = true,
    )
end

function Makie.plot!(p::FramePlot)
    # Extract frame (EuclideanMap)
    frame = p.frame[]

    # Use shared geometry computation
    segments, colors, axis_ends = frame_segments(frame; scale = p.scale[])

    # Draw axes as single Lines with NaN-separated segments
    lines!(p, segments; color = colors, linewidth = p.linewidth)

    # Add endpoint markers
    if p.show_markers[]
        scatter!(
            p, collect(axis_ends);
            color = [:red, :green, :blue],
            markersize = p.markersize
        )
    end

    return p
end

# =============================================================================
# 2D projected frame: frameplot!(ax, pose, camera_model; scale, ...)
# =============================================================================

using VisualGeometryToolkitCore.Cameras: CameraModel

"""
    frameplot!(ax, pose, camera_model; scale=30.0, tipwidth=8, tiplength=12)

Project a 3D coordinate frame into a 2D image axis via a camera model.

Draws RGB arrows (X=red, Y=green, Z=blue) at the pose origin, projected
through the full camera model (including lens distortion) onto the image plane.

# Arguments
- `ax`: Makie `Axis` (2D)
- `pose`: `EuclideanMap{3}` (e.g. [`Extrinsics3`](@ref)) — world-to-camera rigid transform
- `camera_model`: `CameraModel` — full camera model (intrinsics + lens)

# Keyword Arguments
- `scale=30.0`: Length of each axis in world units (mm)
- `tipwidth=8`: Arrow head width
- `tiplength=12`: Arrow head length

# Example
```julia
cam = CameraModel(K; lens=Lens(Perspective(), Radial((...))))
fig, ax, _ = imshow(img)
frameplot!(ax, pose, cam; scale=50.0)
```
"""
function frameplot!(
        ax::Makie.Axis, pose::EuclideanMap{3}, model::CameraModel;
        scale::Real = 30.0, tipwidth = 8, tiplength = 12
    )
    proj = x -> Point2f(project(model, pose(x)))

    origin_2d = proj(Point3f(0, 0, 0))
    for (axis_vec, color) in zip(
            eachcol(SMatrix{3, 3, Float32}(scale * I)),
            [:red, :green, :blue]
        )
        tip_2d = proj(Point3f(axis_vec...))
        d = tip_2d - origin_2d
        Makie.arrows2d!(
            ax, [origin_2d[1]], [origin_2d[2]], [d[1]], [d[2]];
            color, tipwidth, tiplength
        )
    end
    return ax
end
