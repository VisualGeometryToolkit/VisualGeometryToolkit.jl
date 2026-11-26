# Chart LM API

The chart-rows contract and the stack's one Levenberg–Marquardt (NLLSsolver, through one adapter): the algorithm
(`LevenbergMarquardt`, whose fields are its optimization strategy, its damping and its backend, the damping
defaulting to the backend's own, `default_damping`), its stop settings (`LmStops`, with the
data's residual resolution), the refine of a row, chart or block problem (`solve`/`init` on a `RefineProblem`; the
cache keeps its workspace across `reinit!`, and a `callback` sees each iteration's `LmIterationState`), the one
success rule, and the device LM (`StaticLevenbergMarquardt`, `StaticLmStops`).

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
`chart_rows`, `chart_model`, `chart_size`) or several blocks ([`AbstractBlockProblem`](@ref): `block_variables`,
`residual_blocks` or `block_structure`, `fixed_variables`, `residual_summary`, `block_result`) — the parsed robust loss (`bundle_loss`) and the refine's
metadata ([`RefineMetadata`](@ref), its supertype [`AbstractFitMetadata`](@ref)). A refine of raw data (a
correspondence vector, not a problem) by another solver backend, `solve(RefineProblem(cs, guess[, p]),
LevenbergMarquardt(backend; …))`, is forwarded to the backend contract generic `lm_backend_refine`, which the
backend's package implements (PoseLib.jl, for `PoseLibSolver()`).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["robust/refine.jl"]
```

## Index

```@index
Pages = ["lm_core.md"]
```
