# Synthetic Data API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Synthetic scenes, evaluation metrics and augmentation (the `Synthetic`
submodule), and the Monte Carlo calibration of the uncertainty propagation.

## Synthetic Data

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras, VisualGeometryToolkitCore.Synthetic]
Private = false
Pages = ["synthetic/synthetic.jl", "synthetic/conics.jl", "synthetic/evaluation.jl", "synthetic/augmentation.jl"]
```

## Monte Carlo calibration of the uncertainty propagation

Public, not exported (`Synthetic.run_calibration`, or import by name): the
scenes, the per-model trials and the coverage tests the stack's propagation
calibration items run.

```@autodocs
Modules = [VisualGeometryToolkitCore.Synthetic]
Private = false
Pages = ["synthetic/calibration.jl"]
```

## Index

```@index
Pages = ["synthetic.md"]
```
