# Warping API

```@meta
CurrentModule = VisualGeometryCore
```

Coordinate and region maps (canonical and log-polar), the `image_warp`
adapter and its KernelAbstractions resampler, and the `BoundedMap`
decorator. See the [Transforms](../guide/transforms.md) guide.

## Coordinate and Region Maps

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["transforms/coord_maps.jl", "transforms/region_maps.jl", "transforms/ct_contravariant.jl"]
```

## Image Warping

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["transforms/warp.jl", "transforms/bounded_map.jl"]
```

## Index

```@index
Pages = ["warping.md"]
```
