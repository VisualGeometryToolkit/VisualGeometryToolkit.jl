# Generic Camera Models API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Generic camera models: a spline grid of rays over the image, central or
noncentral, with its window interface and undistortion. Split from the
[Cameras API](cameras.md) page. See the [Cameras](../guide/cameras.md) guide.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/generic_models.jl"]
```

## The central grid as a lens

The central grid camera as an ordinary `CameraModel(K, SplineLens(…))`: a
backward lens on the K-normalized image plane, `K` a gauge whose principal point
is the pixel imaging ẑ. The spline lattice's shared helpers (the grid size, the
quasi-interpolant) live beside it.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/spline_lens.jl"]
```

## The non-central grid on the lens's directions

`NoncentralSplineModel`: the central grid's directions (a `SplineLens`'s backward
spline) and a point spline over the same rectangle of the normalized plane, one
observation line per pixel.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/noncentral_spline_model.jl"]
```

## A forward map as a spline on a chart

A grid model's `project` is a small LM per ray. Its forward map can instead be
fitted as a [`SplineProjection`](@ref): the grid models' lattice as a
[`ChartSpline`](@ref) on the equidistant direction chart, ray → pixel in closed
form (VisualGeometryToolkitEstimation's `fit_camera(SplineProjection, source, region; tolerance)`).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/chart_spline.jl"]
```

## Index

```@index
Pages = ["generic_models.md"]
```
