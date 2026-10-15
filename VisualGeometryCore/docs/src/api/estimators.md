# Robust Estimation API

!!! note "Division of labor"
    The RANSAC loop, the problems and the core local optimizations live in
    [RobustVisualGeometry.jl](https://github.com/VisualGeometryToolkit/RobustVisualGeometry.jl).
    VisualGeometryCore provides the contracts and shared primitives
    documented here and on the pages beside it: the problem, scoring and
    local-optimization contracts with the baseline scorers and pool
    selection (this page); the robust-loss family and the width axes
    ([Robust Losses API](robust_losses.md)); the robust-statistics toolkit,
    the generic block-coordinate-descent loop, the chart-rows contract and
    the chart LM ([Robust Statistics and Solvers API](robust_statistics.md));
    the configuration tags, stack contracts and conformance helpers
    ([Configuration Tags and Stack Contracts API](stack_contracts.md)). The
    model manifolds and image-domain confidence regions are on the
    [Model Manifolds and Uncertainty API](manifolds.md) page.

```@meta
CurrentModule = VisualGeometryCore
```

## Estimation Interfaces

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["robust/estimation.jl"]
```

## Problem Protocol

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["robust/problem.jl"]
```

## Scoring Contract and Baseline Scorers

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["robust/scoring_contract.jl", "robust/baseline_scorers.jl"]
```

## Local-Optimization and External-Refit Contracts

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["robust/lo_contract.jl"]
```

## Pool Selection

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["robust/pool_selection.jl"]
```

## Index

```@index
Pages = ["estimators.md"]
```
