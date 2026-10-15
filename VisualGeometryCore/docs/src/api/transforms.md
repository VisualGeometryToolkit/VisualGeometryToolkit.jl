# Transforms API

```@meta
CurrentModule = VisualGeometryCore
```

Matrix-backed transform types, framed maps, coordinate conversions,
uncertainty propagation, and the SO(3)/SE(3) helpers. Region maps and
image warping are on the [Warping API](warping.md) page. See the
[Transforms](../guide/transforms.md) and
[Coordinate Frames](../guide/frames.md) guides.

## Homogeneous Matrix Types

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["transforms/matrices.jl"]
```

## Framed Maps

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["transforms/euclidean_map.jl", "transforms/homography_map.jl"]
```

## Coordinate Conversions

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["base/coordinates.jl", "transforms/conversions.jl"]
```

## Uncertainty Propagation

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["base/propagate.jl", "transforms/propagate_framed.jl"]
```

## Measurement Noise

A plain point is unit isotropic noise, an [`Uncertain`](@ref) point carries its
covariance, an [`exact`](@ref) point has none (the Conventions page,
"Measurement noise").

```@autodocs
Modules = [VisualGeometryCore]
Private = false
Pages = ["base/measurement_noise.jl"]
```

## Lie-Group Helpers and Rotation Fitting

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["transforms/lie.jl", "transforms/rotation_fits.jl"]
```

## Index

```@index
Pages = ["transforms.md"]
```
