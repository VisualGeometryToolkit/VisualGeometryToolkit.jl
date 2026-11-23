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

## Index

```@index
Pages = ["lm_core.md"]
```
