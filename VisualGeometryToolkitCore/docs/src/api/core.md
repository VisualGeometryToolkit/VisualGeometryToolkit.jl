# Core API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The package docstring, units and quantities, utilities, the benchmark pool
record, and the visualization stubs. The core composition and uncertainty
types are on the [Core Types API](types.md) page, synthetic data and the Monte
Carlo calibration on the [Synthetic Data API](synthetic.md) page.

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

```@docs
VisualGeometryToolkitCore.StackUnits
```

## Utilities

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/utils.jl", "base/canonical.jl", "base/geometry_utils.jl", "base/device.jl", "base/duals.jl"]
```

## The artifact store

Published artifact trees — a paper project's DataArtifacts tiers, a package's
model weights — are installed into the **artifact store**, a directory of
`<git-tree-sha1>/` trees in Pkg's layout on storage that every job of a
machine shares, and never into a cluster's Julia depot (node-local and
evicted, or a home with a quota). [`artifact_store`](@ref) resolves it, the
first rung that names one:

| Rung | Set by |
|---|---|
| `VGTK_ARTIFACT_STORE` | the environment (a reproduction's jobs get it from their site) |
| the site profile `VGTK_SITE` names | its `[storage] artifact_store` (VisualGeometryToolkitEval's site profiles, registered with [`set_site_artifact_store!`](@ref)) |
| the preference `artifact_store` | `Preferences.set_preferences!(VisualGeometryToolkitCore, "artifact_store" => dir)` |
| the depot | Pkg's default; **refused on the cluster** (a job there sets `NODE_CACHE`) |

[`install_artifact`](@ref) installs by name from an Artifacts.toml into the
store, inside [`with_artifact_store`](@ref) (Pkg's `with_artifacts_directory`
under one lock), so Pkg's sha256 and git-tree-sha1 checks and its atomic move
into place hold; a Pidfile lock per tree makes the jobs that start together
download a tree once. A model published as a single file goes in with
[`store_file`](@ref) (verified by [`download_verified`](@ref)). A machine whose
depot already holds trees moves them with [`migrate_artifacts!`](@ref), and
[`store_name_clashes`](@ref) lists a name the store holds under two trees.

```julia
import VisualGeometryToolkitCore as VgtkCore
VgtkCore.artifact_store()                       # ArtifactStore(path, source)
tree = VgtkCore.install_artifact("tier_a", "Artifacts.toml")
VgtkCore.migrate_artifacts!("Artifacts.toml")   # the depot's copies into the store, hash-checked
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/artifacts.jl"]
```

Drawing into an array without Makie (masks, debug frames; public, not
exported):

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/raster.jl"]
```

## TOML Configuration

The config-of-record machinery shared by the package family's config types
(public, not exported — import by name).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["base/config_toml.jl"]
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
