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
