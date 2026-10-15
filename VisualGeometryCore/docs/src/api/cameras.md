# Cameras API

```@meta
CurrentModule = VisualGeometryCore
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
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["cameras/cameras.jl", "cameras/camera_models.jl", "cameras/utilities.jl", "cameras/random.jl"]
```

## Intrinsics and Sensors

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["cameras/intrinsics.jl", "cameras/sensors.jl"]
```

## Index

```@index
Pages = ["cameras.md"]
```
