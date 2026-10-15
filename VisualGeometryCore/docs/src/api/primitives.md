# Primitives API

```@meta
CurrentModule = VisualGeometryCore
```

Geometric primitives (circles, ellipses, discs, feature frames,
hyperplanes, 3D conics) and their traits and operations. See the
[Primitives](../guide/primitives.md) guide.

## Primitives and Feature Frames

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["primitives/primitives.jl", "primitives/feature_frames.jl", "primitives/blob_types.jl", "primitives/rectangles.jl"]
```

## Hyperplanes

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["primitives/hyperplane.jl"]
```

## Conics and 3D Primitives

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["primitives/conic.jl", "primitives/circle3d.jl", "primitives/ellipse3d.jl", "primitives/mvnormal_geometry.jl", "primitives/ellipse_area_native.jl"]
```

## Operations, Containment, and Traits

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["primitives/operators.jl", "primitives/containment.jl", "primitives/traits.jl", "primitives/bbox.jl"]
```

## Index

```@index
Pages = ["primitives.md"]
```
