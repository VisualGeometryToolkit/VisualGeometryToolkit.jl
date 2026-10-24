# Core API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The package docstring, units and quantities, utilities, synthetic-data
generation, the benchmark pool record, and the visualization stubs. The core composition and
uncertainty types are on the [Core Types API](types.md) page.

## The Package

```@docs
VisualGeometryToolkitCore
```

## Units and Quantities

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/units.jl", "base/size2.jl"]
```

## Utilities

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/utils.jl", "base/geometry_utils.jl", "base/device.jl"]
```

## TOML Configuration

The config-of-record machinery shared by the package family's config types
(public, not exported — import by name).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["base/config_toml.jl"]
```

## Synthetic Data

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras, VisualGeometryToolkitCore.Synthetic]
Private = false
Pages = ["synthetic/synthetic.jl", "synthetic/conics.jl", "synthetic/evaluation.jl", "synthetic/augmentation.jl"]
```

## Benchmark Pools

The pool record and the raw-pair interface (`using VisualGeometryToolkitCore.Pools`).
The pool is the stack's one declared exception to "frame in = frame out": its
data is stored in [`NormalizedImageFrame`](@ref), and every accessor that
crosses the boundary names it.

```@autodocs
Modules = [VisualGeometryToolkitCore.Pools]
Pages = ["pools/pools.jl", "pools/pool_pair.jl", "pools/raw_pair.jl"]
```

## Spatial Queries

Allocation-free queries on SpatialIndexing.jl's R-tree (`using
VisualGeometryToolkitCore.SpatialQueries`): the intersection query with early exit and
the branch-and-bound nearest neighbour, with a reusable state for batches,
and the in-place OMT bulk load. [`bbox`](@ref VisualGeometryToolkitCore.bbox) gives a shape's query rectangle.

```@autodocs
Modules = [VisualGeometryToolkitCore.SpatialQueries]
Pages = ["geometry/spatial_queries.jl"]
```

## Visualization Stubs

Stubs implemented by the Makie/GLMakie package extension (load a Makie
backend to activate them); see the
[Visualization](../guide/visualization.md) guide.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["visualization_stubs.jl"]
```

## Index

```@index
Pages = ["core.md"]
```
