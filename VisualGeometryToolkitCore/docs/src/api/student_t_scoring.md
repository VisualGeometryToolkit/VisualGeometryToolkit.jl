# Student-t + Uniform Scoring API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The scale-free marginal score of "RANSAC Scoring Done Right" (moved from
RANSACScoringDoneRight, which keeps its research outlier models, grids and
loss-width families on these types): the scorer, its label priors and outlier
models, its workspace and the one partition sweep. The scoring contract it
implements is on the [Robust Estimation API](estimators.md) page.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["robust/partition_sweep.jl", "robust/student_t_uniform_scoring.jl"]
```

## Index

```@index
Pages = ["student_t_scoring.md"]
```
