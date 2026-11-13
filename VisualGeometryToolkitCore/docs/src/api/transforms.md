# Transforms API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Framed maps — the geometric objects, whose type is their transform class —
the dense homography matrix, coordinate conversions, uncertainty
propagation, and the SO(3)/SE(3) helpers. Region maps and image warping are
on the [Warping API](warping.md) page. See the
[Transforms](../guide/transforms.md) and
[Coordinate Frames](../guide/frames.md) guides.

## Framed Maps

A map acts on a primitive in its class: an `EuclideanMap` or a `SimilarityMap`
keeps a circle a `Circle` (and a feature frame its type), an `AffineMap` gives
a `HomEllipseMat`, a `HomographyMap` a `Conic` (the actions are on the
[Warping API](warping.md) page).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/euclidean_map.jl", "transforms/similarity_map.jl", "transforms/rotation_map.jl", "transforms/homography_map.jl"]
```

## The Homography Matrix

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/matrices.jl"]
```

## Coordinate Conversions

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/coordinates.jl", "transforms/conversions.jl"]
```

## Uncertainty Propagation

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/propagate.jl", "transforms/propagate_framed.jl"]
```

## Sampson Distance and Whitening

The Sampson residual is geometric (a model and its observation, no noise);
[`whiten`](@ref) divides it by the observation's covariance along the normal.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/sampson.jl"]
```

## Observation Noise

A plain point is unit isotropic noise, an [`Uncertain`](@ref) point carries its
covariance, an [`exact`](@ref) point has none (the Conventions page,
"Observation noise").

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/observation_noise.jl"]
```

## Lie-Group Helpers and Rotation Fitting

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["transforms/lie.jl", "transforms/rotation_fits.jl"]
```

## Index

```@index
Pages = ["transforms.md"]
```
