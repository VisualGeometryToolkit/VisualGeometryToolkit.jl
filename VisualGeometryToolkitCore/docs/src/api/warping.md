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

Two bilinear lookups with two contracts: [`bilinear_sample`](@ref) reads a
texture in its unit-square (half-texel) chart, extends its edges, returns
several channels and differentiates in the position (a printed board's
albedo under ForwardDiff); [`bilinear_sample_clamped`](@ref) reads 1-based
array positions, rejects one off the image, and is exact at integers (a
pixel scan).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/warp.jl", "transforms/bounded_map.jl"]
```

## Index

```@index
Pages = ["warping.md"]
```
