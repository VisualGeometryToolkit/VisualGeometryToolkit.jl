# Chart LM API

The chart-rows contract and the stack's one Levenberg–Marquardt (NLLSsolver, through one adapter): the algorithm
(`LevenbergMarquardt`, whose fields are its optimization strategy and damping), its stop settings (`LmStops`, with the
data's residual resolution), the chart, robust and multi-block solves (`lmsolve_chart!`, `LmProblem`, which owns its
damping), the one success rule, and the device LM (`StaticLevenbergMarquardt`, `StaticLmStops`).

```@meta
CurrentModule = VisualGeometryToolkitCore
```

## Chart Rows and the LM Core

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["robust/rows.jl", "robust/lm_core.jl"]
```

## The Refine: Chart and Block Contracts

What a problem implements to be refined by `LevenbergMarquardt` — one model on a minimal chart (`model_chart`,
`chart_rows`, `chart_model`, `chart_size`) or several blocks ([`AbstractBlockProblem`](@ref), `block_refit`) — the
hosts' refits (`chart_refit`, `triggs_refit`, `block_lm`), the parsed robust loss (`bundle_loss`) and the refine's
metadata ([`RefineMetadata`](@ref), its supertype [`AbstractFitMetadata`](@ref)).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["robust/refine.jl"]
```

## Index

```@index
Pages = ["lm_core.md"]
```
