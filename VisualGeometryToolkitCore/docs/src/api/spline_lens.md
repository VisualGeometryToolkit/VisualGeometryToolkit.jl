# Spline Lens API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The grid cameras of Schöps et al. (CVPR 2020): a uniform cubic B-spline of rays
over the image, central (a [`SplineLens`](@ref) of a `CameraModel`) or
non-central ([`NoncentralSplineModel`](@ref)), and the lattice they share. Split
from the [Cameras API](cameras.md) page. See the [Lenses](../guide/lenses.md) guide.

## The central grid as a lens

The central grid camera as an ordinary `CameraModel(K, SplineLens(…))`: a
backward lens on the K-normalized image plane, `K` a gauge whose principal point
is the pixel imaging ẑ. Its optional forward spline maps the equidistant chart
about the lens axis to the image plane. The spline lattice's shared helpers (the
grid size, the quasi-interpolant, the window of a block of controls, the chart a
control moves in) live beside it.

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

## The lattice: a spline on a chart

The lattice of both, as a primitive: [`ChartSpline`](@ref), a uniform cubic
B-spline over a rectangle of a 2-D chart, its window and span weights, and the
equidistant direction chart a forward spline is laid on.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/chart_spline.jl"]
```

## Index

```@index
Pages = ["spline_lens.md"]
```
