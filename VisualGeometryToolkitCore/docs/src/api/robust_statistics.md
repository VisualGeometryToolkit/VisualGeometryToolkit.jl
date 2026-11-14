# Robust Statistics and Solvers API

The robust-statistics toolkit and the generic block-coordinate-descent loop: the
shared machinery under the contracts of the
[Robust Estimation API](estimators.md) page. The chart-rows contract and the
chart LM are on the [Chart LM API](lm_core.md) page.

```@meta
CurrentModule = VisualGeometryToolkitCore
```

## Robust-Statistics Toolkit

The consensus scale, the Student-t + uniform mixture and its posterior-weight
state, shared by the scorers, the local optimizers and the mixture fits.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/scale.jl", "robust/mixture.jl", "robust/posterior.jl", "robust/normaliser.jl"]
```

## Block Coordinate Descent

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["robust/block_coordinate_descent.jl"]
```

## Index

```@index
Pages = ["robust_statistics.md"]
```
