# Pool Selection API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The pool-selection contract (`best_pool_model`, `select_on_grid!`) and the
grid kernels that score a pool's models across a width grid. The scoring and
local-optimization contracts it builds on are on the
[Robust Estimation API](estimators.md) page.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/pool_selection.jl", "robust/pool_grid_kernels.jl"]
```

## Index

```@index
Pages = ["pool_selection.md"]
```
