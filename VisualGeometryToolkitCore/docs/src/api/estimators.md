# Robust Estimation API

!!! note "Division of labor"
    The RANSAC loop, the problems and the core local optimizations live in
    [VisualGeometryToolkitEstimation.jl](@extref VisualGeometryToolkitEstimation :doc:`index`).
    VisualGeometryToolkitCore provides the contracts and shared primitives
    documented here and on the pages beside it: the problem, scoring and
    local-optimization contracts with the baseline scorers (this page); the
    optimization strategies and the local-optimization contract
    ([Optimization Strategies API](optimization_strategies.md)); the
    Student-t + uniform scorer ([Student-t + Uniform Scoring API](student_t_scoring.md));
    pool selection ([Pool Selection API](pool_selection.md)); the robust-loss family and the width axes
    ([Robust Losses API](robust_losses.md)); the robust-statistics toolkit,
    the generic block-coordinate-descent loop, the chart-rows contract and
    the chart LM ([Robust Statistics and Solvers API](robust_statistics.md));
    the configuration tags, stack contracts and conformance helpers
    ([Configuration Tags and Stack Contracts API](stack_contracts.md)). The
    model manifolds and image-domain confidence regions are on the
    [Model Manifolds and Uncertainty API](manifolds.md) page.

```@meta
CurrentModule = VisualGeometryToolkitCore
```

## Estimation Interfaces

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["robust/estimation.jl", "robust/weights.jl"]
```

## Problem Protocol

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/problem.jl"]
```

## Scoring Contract and Baseline Scorers

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/scoring_contract.jl", "robust/baseline_scorers.jl"]
```

## The Student-t + Uniform Marginal Scorer

The scale-free marginal score of "RANSAC Scoring Done Right" is on the
[Student-t + Uniform Scoring API](student_t_scoring.md) page.

## Optimization Strategies and the Local-Optimization Contract

The optimization strategies (`Irls`, `Triggs` and their registry) and the
local-optimization and external-refit contracts are on the
[Optimization Strategies API](optimization_strategies.md) page.

## Pool Selection

The pool-selection contract and its grid kernels are on the
[Pool Selection API](pool_selection.md) page.

## Index

```@index
Pages = ["estimators.md"]
```
