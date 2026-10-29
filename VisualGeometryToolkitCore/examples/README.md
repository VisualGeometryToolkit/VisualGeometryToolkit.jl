# VisualGeometryToolkitCore examples

Minimal demos of the public entry points. Each file runs standalone from the
package environment:

```bash
julia --project=. examples/camera_demo.jl
julia --project=. examples/ellipse_transforms_demo.jl
julia --project=. examples/image_circle.jl
julia --project=. examples/imshow_demo.jl      # needs GLMakie in the environment
```

| File | Surface |
|---|---|
| `camera_demo.jl` | `CameraModel`, lens models, `PosedCamera` with a pose, `project` / `backproject`, normalized distortion |
| `image_circle.jl` | `valid_domain` (where the lens model holds) and `image_circle` from a datasheet through the sensor's pitch (where light falls), `Rect(extent) ∈ circle`, `fov`, and a grid lens's domain `imaged_region(camera, circle)` |
| `ellipse_transforms_demo.jl` | `Ellipse`, conic matrices, and covariant transforms under CoordinateTransformations maps |
| `imshow_demo.jl` | `imshow` (Axis-based) and `CampixelImshow` + `poly!` overlays through the Makie extension |

Research drivers and validation harnesses do not live here; the VLFeat
scale-space oracle moved with the detector to VisualGeometryToolkitFeatures.
