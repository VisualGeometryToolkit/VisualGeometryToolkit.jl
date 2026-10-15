# Visualization

VisualGeometryCore provides Makie-based visualization through package extensions.

## Setup

The plotting functionality is optional and loads automatically when any Makie
backend (GLMakie, CairoMakie, WGLMakie) is imported:

```julia
using VisualGeometryCore
using GLMakie  # Activates the plotting extension
```

## Image Display Functions

VisualGeometryCore provides two approaches for displaying images:

### Axis-based Display (`imshow`)

For standard plotting workflows with Axis decorations (ticks, labels, etc.):

```julia
# Creates a new Figure with Axis
fig, ax, plt = imshow(img)

# Add overlays using standard Makie functions
scatter!(ax, [100, 200], [150, 250]; color=:red, markersize=10)

# Display at a specific position in an existing Figure
fig = Figure()
ax1, plt1 = imshow(fig[1, 1], img; title="Image 1")
ax2, plt2 = imshow(fig[1, 2], other_img; title="Image 2")

# Add to existing Axis
ax = Axis(fig[2, 1]; aspect=DataAspect(), yreversed=true)
imshow!(ax, img)
```

### LScene-based Display (`campixel_imshow`, `CampixelImshow`)

For pixel-perfect camera alignment with campixel! camera:

```julia
# Non-SpecApi: returns actual Makie objects
fig, lscene, plt = campixel_imshow(img; scale=0.5)
scatter!(lscene.scene, [100, 200], [150, 250]; color=:red)

# SpecApi: returns spec for composition
spec = CampixelImshow(img; scale=0.5)
```

## Spec API

VisualGeometryCore uses Makie's **SpecAPI** for declarative scene composition:

```julia
import Makie.SpecApi as S
```

### Creating Scenes

```julia
# Image display (SpecApi)
scene = CampixelImshow(img; scale=0.5)

# 3D scene
scene_3d = LScene3D(width=800, height=600)
```

## Plotting Geometric Primitives

### Circles and Ellipses

```julia
fig, ax, plt = imshow(img)          # overlays go on the Axis

# Add circles
circles = [Circle(Point2(100.0, 150.0), 25.0), ...]
poly!(ax, circles;
      strokecolor=:red,
      strokewidth=2.0,
      color=:transparent)

# Add ellipses
ellipses = [Ellipse(Point2(200.0, 100.0), 30.0, 20.0, π/4), ...]
poly!(ax, ellipses; strokecolor=:blue)
```

`poly!` forwards to `Makie.poly!` on an `Axis`/`LScene`. On a SpecApi scene
from `CampixelImshow`, overlays are plot specs pushed onto `scene.plots`:

```julia
import Makie.SpecApi as S
scene = CampixelImshow(img)
push!(scene.plots, S.Poly(circles; color=:transparent, strokecolor=:red, strokewidth=2.0))
```

### Detections

Detections (from VisualGeometryFeatures) render as circles via the
`Circle(detection, multiplier)` constructor:

```julia
# blobs :: Vector{<:AbstractDetection}, e.g. from VisualGeometryFeatures
circles = Circle.(blobs, Ref(3.0))  # 3σ radius
fig, ax, plt = imshow(img)
poly!(ax, circles; strokecolor=:yellow, strokewidth=2.0)
```

## Correspondence Visualization

[`matches`](@ref)/`matches!` are the correspondence overlays.
VisualGeometryFeatures extends them by dispatch for its feature types (the old
`cspond!`/`matchlines` names are gone from both packages).

## 3D Visualization

### Camera Frustums

```julia
# Create 3D scene (SpecApi)
scene_3d = LScene3D(width=800, height=600)

# Add camera visualization
camera = Camera(model, pose)
image_rect = Rect(sensor)

# camera! pushes axes, frustum, and center specs onto the SpecApi scene
camera!(scene_3d, camera, image_rect;
        center_color=:red,
        center_size=15.0,
        frustum_color=:orange,
        frustum_linewidth=2.0)

# Or push the frustum PlotSpec alone
push!(scene_3d.plots, Frustum(camera, image_rect; color=:blue))
```

On a realized `LScene` (not a spec), use the `cameraplot!` recipe, which also
textures the far plane with `frustum_image`:

```julia
cameraplot!(lscene, camera, image_rect; frustum_image=camera_view)
```

### 3D Points

```julia
# Add 3D points
X = [Point3(0.0, 0.0, 0.0), Point3(100.0, 0.0, 0.0), ...]

push!(scene_3d.plots, S.Scatter(X;
                                color=:black,
                                markersize=10.0))
```

## Layouts

Combine multiple scenes in a grid:

```julia
# Create individual scenes
scene1 = CampixelImshow(img1)
scene2 = CampixelImshow(img2)
scene_3d = LScene3D()

# Create grid layout
layout = S.GridLayout([
    (1, 1) => scene1,
    (1, 2) => scene2,
    (2, 1:2) => scene_3d  # Span two columns
])

# Plot and display
fig = plot(layout)
display(fig)
```

## Advanced Usage

### Custom Plotting

Add custom Makie plots to scenes:

```julia
scene = CampixelImshow(img)

# Add scatter points
points = [Point2(100.0, 150.0), Point2(200.0, 250.0)]
push!(scene.plots, S.Scatter(points;
                            color=:yellow,
                            markersize=20.0,
                            marker=:cross))

# Add lines
line_points = [Point2(0.0, 0.0), Point2(100.0, 100.0)]
push!(scene.plots, S.Lines(line_points;
                          color=:green,
                          linewidth=3.0))
```

### Interactive Displays

```julia
# Display with GLMakie screen
fig = plot(layout)
display(GLMakie.Screen(), fig)

# Save to file
using FileIO
save("output.png", fig)
```

## Function Summary

| Function | Returns | Use Case |
|----------|---------|----------|
| `imshow(img)` | `(Figure, Axis, Plot)` | Standard plotting with Axis |
| `imshow(pos, img)` | `(Axis, Plot)` | Add to existing Figure |
| `imshow!(ax, img)` | `Plot` | Add to existing Axis |
| `campixel_imshow(img)` | `(Figure, LScene, Plot)` | LScene with campixel camera |
| `campixel_imshow(pos, img)` | `(LScene, Plot)` | Add to existing Figure |
| `campixel_imshow!(ls, img)` | `Plot` | Add to existing LScene |
| `CampixelImshow(img)` | `LScene spec` | SpecApi composition |

## Package Extension Details

The plotting functionality is provided by two package extensions:
- `VisualGeometryCoreMakieExt` loads with Makie, i.e. with any backend
  (GLMakie, CairoMakie, WGLMakie), and implements every plotting function
  (source in `src/plotting/`); it has zero overhead when not using
  visualization
- `VisualGeometryCoreGLMakieExt` loads when GLMakie is imported and adds only
  the GLMakie window helpers `window!(fig, title)` and `closeall()`

The functions both extensions fill are declared as stubs in
`src/visualization_stubs.jl`, so their names resolve without Makie.

## See Also

- [Getting Started](getting-started.md) - Package setup
- [Cameras Guide](cameras.md) - 3D camera visualization
