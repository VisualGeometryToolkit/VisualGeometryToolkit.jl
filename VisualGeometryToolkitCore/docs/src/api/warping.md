# Warping API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Coordinate and region maps (canonical and log-polar), the `image_warp`
adapter and its KernelAbstractions resampler, and the `BoundedMap`
decorator. See the [Transforms](../guide/transforms.md) guide.

## Coordinate and Region Maps

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/coord_maps.jl", "transforms/region_maps.jl", "transforms/ct_contravariant.jl"]
```

## Image Warping

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/warp.jl", "transforms/bounded_map.jl"]
```

## Index

```@index
Pages = ["warping.md"]
```
