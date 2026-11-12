# Optimization Strategies API

The optimization strategies a fit, a refine and a local optimization take
(`Irls{K}`, `Triggs{K}`, their efficiency constants and the strategy
registry), and the local-optimization and external-refit contracts. The
problem and scoring contracts they work with are on the
[Robust Estimation API](estimators.md) page.

```@meta
CurrentModule = VisualGeometryToolkitCore
```

## Optimization Strategies

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["robust/strategies.jl"]
```

## Local-Optimization and External-Refit Contracts

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/lo_contract.jl"]
```

## Index

```@index
Pages = ["optimization_strategies.md"]
```
