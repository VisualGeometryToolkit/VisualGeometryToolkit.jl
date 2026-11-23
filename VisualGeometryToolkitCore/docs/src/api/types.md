# Core Types API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The core composition and uncertainty types (`Attributed`, `Uncertain`,
`CovMat`/`PrecMat`), the conic and epipolar matrix types, interface stubs,
and traits; the estimation interface and its result, the `FitResult`, are on [The Estimation Interface](estimation.md). See the [Design: Attributed{V,A}](../guide/design.md) and
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

## Index

```@index
Pages = ["types.md"]
```
