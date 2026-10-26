# Primitives API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Geometric primitives (circles, ellipses, discs, feature frames,
hyperplanes, 3D conics) and their traits and operations. See the
[Primitives](../guide/primitives.md) guide.

## Primitives and Feature Frames

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["primitives/primitives.jl", "primitives/feature_frames.jl", "primitives/blob_types.jl", "primitives/rectangles.jl"]
```

## Hyperplanes

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["primitives/hyperplane.jl"]
```

## The Surface a Ray Meets

Two isbits primitives behind one protocol — [`intersection`](@ref)`(s, o, d)`,
[`normal`](@ref)`(s, x)`, [`chart`](@ref)`(s, x)` and [`bounds`](@ref)`(s)` —
generic in the element type (`Float32`, `Float64`, `ForwardDiff.Dual`),
allocation-free and usable in a KernelAbstractions kernel. A ray is an
origin and a direction, as [`ray_plane_intersection`](@ref) takes it. A
[`Quad`](@ref) is a bounded plane, its intersection
`ray_plane_intersection`'s arithmetic bit for bit; a [`HeightSurface`](@ref)
is a thin-plate height field over one, its intersection four Newton
iterations from the plane hit. A ray that misses (parallel, or outside the
rectangle) gives `NaN`. A posed target's primitive is
[`surface`](@ref)`(pt)` ([Imaging Chain and Stereo API](imaging.md)).

```julia
s = surface(PosedTarget(target, pose))          # a Quad, or a HeightSurface for a height field
t, x = VisualGeometryToolkitCore.intersection(s, o, d)
uv = VisualGeometryToolkitCore.chart(s, x)       # TargetFrame (x, y) of the hit
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["primitives/surfaces.jl"]
```

## Conics and 3D Primitives

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["primitives/conic.jl", "primitives/circle3d.jl", "primitives/ellipse3d.jl", "primitives/mvnormal_geometry.jl", "primitives/ellipse_area_native.jl"]
```

## Operations, Containment, and Traits

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["primitives/operators.jl", "primitives/containment.jl", "primitives/traits.jl", "primitives/bbox.jl"]
```

## Index

```@index
Pages = ["primitives.md"]
```
