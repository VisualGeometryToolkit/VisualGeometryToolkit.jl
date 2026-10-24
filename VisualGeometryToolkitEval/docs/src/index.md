# VisualGeometryToolkitEval

VisualGeometryToolkitEval is the evaluation package of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack. It holds what the stack's research projects share when they benchmark
robust estimators: statistics, a codified cross-validation (CV), an experiment
framework, figure renderers, scoring-pool input and output, raw dataset
readers, and an evaluator that composes a RANSAC scorer with a local optimizer
(LO).

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitEval")
```

The package requires Julia 1.12 or later. It depends on
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/),
[VisualGeometryToolkitIO](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/)
and
[VisualGeometryToolkitEstimation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/).

## The modules

Each module's names are imported from it, for example
`using VisualGeometryToolkitEval.Pools: load_pool`. The top module exports the
modules and the driver command line ([`cli_option`](@ref VisualGeometryToolkitEval.cli_option),
[`start_output_flush!`](@ref VisualGeometryToolkitEval.start_output_flush!)).

| Module | What it holds |
|---|---|
| [`Stats`](@ref VisualGeometryToolkitEval.Stats) | finite and lower medians, the validation reference scale, the mean average accuracy (mAA), bias-corrected and accelerated (BCa) bootstrap intervals, means of per-scene medians and the seed derivation |
| [`Tuning`](@ref VisualGeometryToolkitEval.Tuning) | the CV specification: grids, window rules, objectives, scene splits and unit samplings, budget draws, schemes, the `[tuning]` TOML parser and the tuning artifact |
| [`Evaluation`](@ref VisualGeometryToolkitEval.Evaluation) | the coarse-to-fine search of every CV, the bootstrap tuning harness and the composition evaluator |
| [`Experiments`](@ref VisualGeometryToolkitEval.Experiments) | the stage runner, run keys and run records, provenance, and the data layer of a paper project |
| [`Render`](@ref VisualGeometryToolkitEval.Render) | PGFPlots figures and tables in the house style |
| [`Pools`](@ref VisualGeometryToolkitEval.Pools) | the scoring pool, its residuals, and the perspective-n-point (PnP) pool file |
| [`Datasets`](@ref VisualGeometryToolkitEval.Datasets) | raw readers of PhotoTourism, HEB, RoMa and SuperPoint+LightGlue correspondences, and of the depth-validated stereo datasets ETH3D, ScanNet1500, KITTI and 7-Scenes |
| [`DataArtifacts`](@ref VisualGeometryToolkitEval.DataArtifacts) | a paper project's published input data as Julia artifacts |
| [`Reproduce`](@ref VisualGeometryToolkitEval.Reproduce) | the reproducible-figures contract of a paper |
| [`PaperProjects`](@ref VisualGeometryToolkitEval.PaperProjects) | the paper-project conventions: one check and one template |

## Where to read next

- [Getting started](@ref) runs the first examples.
- [The experimental protocol](@ref) is the one statement of how the stack's
  experiments tune, evaluate and report.
- [The `[tuning]` schema](@ref) lists every parameter of a cross-validation
  and the tuning artifact that persists it.
- [The composition evaluator](@ref) describes how a scorer and a local
  optimizer are evaluated together.
- [Reproducible figures](@ref) and [Paper projects](@ref) state the
  conventions every paper of the stack follows.
- [Defaults and file formats](@ref) lists where the defaults live and how
  artifacts are versioned.
- The API reference has one page per module.

```@docs
VisualGeometryToolkitEval
```

## Citing

VisualGeometryToolkitEval is part of VisualGeometryToolkit. Please cite the
toolkit. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button
exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Please also cite the papers behind the methods you use. They are listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).
