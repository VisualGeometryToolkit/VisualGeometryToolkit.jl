# Robust Statistics and Solvers API

The robust-statistics toolkit, the generic block-coordinate-descent loop, the
chart-rows contract and the chart LM (FastLevenbergMarquardt `lmsolve!`): the
shared machinery under the contracts of the
[Robust Estimation API](estimators.md) page.

```@meta
CurrentModule = VisualGeometryCore
```

## Robust-Statistics Toolkit

The consensus scale, the Student-t + uniform mixture and its posterior-weight
state, shared by the scorers, the local optimizers and the mixture fits.

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["robust/scale.jl", "robust/mixture.jl", "robust/posterior.jl", "robust/normaliser.jl"]
```

## Block Coordinate Descent

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["robust/block_coordinate_descent.jl"]
```

## Chart Rows and the LM Core

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["robust/rows.jl", "robust/lm_core.jl"]
```

## Index

```@index
Pages = ["robust_statistics.md"]
```
