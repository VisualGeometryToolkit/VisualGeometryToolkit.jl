# Core Types API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The core composition and uncertainty types (`Attributed`, `Uncertain`,
`CovMat`/`PrecMat`), the conic and epipolar matrix types, interface stubs,
and traits. See the [Design: Attributed{V,A}](../guide/design.md) and
[Stereo Rigs and Uncertainty](../guide/stereo.md) guides.

## Core Types

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/types.jl", "base/interface.jl", "base/interface_functions.jl", "base/traits.jl"]
```

## Uncertainty Accessors

`prec` is documented above with the core types; `cov` extends
`Statistics.cov` (unexported — reachable as `Statistics.cov`):

```@docs
VisualGeometryToolkitCore.cov
```

## Fit Results

Every estimator returns a [`FitResult`](@ref): a fixed core — the estimate,
why it stopped, how many iterations, the residual RMS over `n` — plus
solver-specific `metadata` selected by type parameter. Read it through the
accessors below (`estimate(r)`, `retcode(r)`, …, and the metadata's own such as
`weights(r)`), never by field name. A property spelling still forwards to the
metadata for older call sites; a name on neither is an error, never `nothing`.

```@docs
FitResult
BundleMetadata
estimate
retcode
converged
iterations
rms
metadata
weights
dof
```

## Index

```@index
Pages = ["types.md"]
```
