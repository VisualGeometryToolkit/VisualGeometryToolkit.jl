# Cameras API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Cameras, camera models, intrinsics, and sensors. The grid cameras (a
`SplineLens`, a `NoncentralSplineModel`) are on the [Spline Lens API](spline_lens.md) page;
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

The intrinsics and sensor types are on the
[Intrinsics and Sensors API](intrinsics.md) page.

## Boards and canvases

A planar target's [`Board`](@ref) is the sensor's analogue: the printed
board's size (lengths, or plain numbers for a unitless target) and its corner
in the design chart; `Rect(board)` is its
domain there, as `Rect(sensor)` is the image's in pixels.
The [`Canvas`](@ref) is the printed page the board lies on, with the image
printed on it ([`CanvasTexture`](@ref): a relative path, its SHA-256 and the
rectangle it covers) — canvas ⊃ board ⊃ pattern, all rectangles of the one
design chart the `Canvas` docstring states.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/boards.jl"]
```

## The overlap of a rotating camera's views

[`view_overlap`](@ref) measures, for every pair of frames of a camera turning
about its centre, the solid angle of the directions both image and its share
of each frame's footprint: the overlap a panorama's pair selection reads.

```@autodocs
Modules = [VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/view_overlap.jl"]
```

## Index

```@index
Pages = ["cameras.md"]
```
