# Cameras API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Cameras, camera models, intrinsics, and sensors. The generic (grid) camera
models are on the [Generic Camera Models API](generic_models.md) page;
calibration (the radial
camera, intrinsics from homographies, view-angle limits) is on the
[Calibration API](calibration.md) page; lens models and the lens contract with
its batched surfaces are on the [Lenses API](lenses.md) page; the framed
imaging chain, ray maps, and stereo rigs are on the
[Imaging Chain and Stereo API](imaging.md) page. See the
[Cameras](../guide/cameras.md) guide.

## Cameras and Camera Models

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/cameras.jl", "cameras/camera_models.jl", "cameras/utilities.jl", "cameras/random.jl"]
```

## Intrinsics and Sensors

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/intrinsics.jl", "cameras/sensors.jl"]
```

## Boards

A planar target's [`Board`](@ref) is the sensor's analogue: the printed
board's size (lengths, or plain numbers for a unitless target) and its corner
in the design chart; `Rect(board)` is its
domain there, as `Rect(sensor)` is the image's in pixels.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/boards.jl"]
```

## Index

```@index
Pages = ["cameras.md"]
```
