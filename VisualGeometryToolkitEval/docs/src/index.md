# VisualGeometryToolkitEval

VisualGeometryToolkitEval benchmarks robust estimators. It is the evaluation
package of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack and holds what the stack's research projects share: statistics of pose
errors, a codified cross-validation (CV) of hyperparameters, an experiment
framework, figure renderers, scoring-pool input and output, raw dataset
readers, and an evaluator that composes a RANSAC scorer with a local optimizer
(LO).

## What it provides

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

Its design principles:

- **One protocol.** Every experiment of the stack tunes, evaluates and reports
  the same way, stated once in [The experimental protocol](@ref).
- **TOML is the authority.** Every default that changes a reported number
  lives in one TOML file and is read through one function; an unknown key or
  tag is refused ([Defaults and file formats](@ref)).
- **Contracts, not paper code.** The evaluator is written against
  VisualGeometryToolkitCore's scoring and local-optimization contracts, so it
  runs any scorer with any local optimizer ([The composition evaluator](@ref)).
- **Every result is reproducible.** A persisted artifact carries its format
  stamp and the specification it ran under, and every figure of a paper comes
  from tracked code ([Reproducible figures](@ref)).

## Installation

```julia-repl
pkg> add VisualGeometryToolkitEval
```

The package requires Julia 1.12 or later. It depends on
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/),
[VisualGeometryToolkitIO](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/)
and
[VisualGeometryToolkitEstimation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/).

## A first example

The example summarizes the pose errors of seven image pairs, then tunes one
threshold by the package's coarse-to-fine cross-validation.

```julia
using VisualGeometryToolkitEval.Stats: finite_median, eligible_maa, bootstrap_ci
using VisualGeometryToolkitEval.Tuning: tuning_spec, grid_spec
using VisualGeometryToolkitEval.Evaluation: coarse_fine_cv
using Random: Xoshiro
using TOML

# Pose errors in degrees of seven image pairs; NaN marks a pair without a model.
errors = [0.4, 1.2, NaN, 0.8, 3.5, 0.6, 2.1]
finite_median(errors)                               # 1.0
eligible_maa(errors; max_error = 10.0)              # ≈ 0.73: mAA up to 10 degrees; NaN counts as 0
bootstrap_ci(finite_median, Xoshiro(1), errors)     # 95% BCa interval of the median

# A tuning specification, as a paper's TOML config states it. The keys left out
# take their defaults from the package's configs/tuning.toml.
config = TOML.parse("""
    [tuning.threshold]
    type = "coarse_fine"
    coarse = { type = "spaced", window = { type = "fixed", lo = 0.1, hi = 10.0 }, spacing = { type = "log_per_decade", per_decade = 4 } }
    cross_validation = { scene_split = { type = "registered", name = "phototourism" } }
    """)
spec = tuning_spec(config, "threshold")

# A stand-in for the validation errors at threshold x; they are smallest at x = 2.
validation_errors(x) = [abs(log(x / 2)) + 0.1 * k for k in 1:5]
result = coarse_fine_cv(validation_errors, grid_spec(spec.scheme.coarse), spec.scheme;
                        method_name = :threshold)
result[:x_star_fine]                                # ≈ 2.0, the tuned threshold
result[:boundary_hit]                               # false: the optimum is inside the window
```

## Documentation

**Start here.** [Getting started](@ref) runs the first examples: summarizing
pose errors and tuning a threshold.

**How to.** Configure a cross-validation with [The `[tuning]` schema](@ref),
which lists every parameter and the tuning artifact that persists it; make a
paper's figures reproducible ([Reproducible figures](@ref)); and set up a
paper project ([Paper projects](@ref)).

**Explanation.** [The experimental protocol](@ref) is the one statement of
how the stack's experiments tune, evaluate and report. [The composition
evaluator](@ref) describes how a scorer and a local optimizer are evaluated
together. [Defaults and file formats](@ref) says where the defaults live and
how artifacts are versioned.

**Reference.** The API pages, one per module, from
[The top module](api/toplevel.md) to [PaperProjects](api/paper_projects.md).

```@docs
VisualGeometryToolkitEval
```

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitEval builds on VisualGeometryToolkitCore (the scoring
and local-optimization contracts, the pool record), VisualGeometryToolkitIO
and VisualGeometryToolkitEstimation (the RANSAC loop). The research projects
that evaluate particular scorers or local optimizers depend on it, never the
other way round. The [portal](https://visualgeometrytoolkit.github.io) links
every package of the toolkit.

## Citing

VisualGeometryToolkitEval is part of VisualGeometryToolkit. Cite the toolkit.
Its
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

Cite also the papers behind the methods you use; the
[portal](https://visualgeometrytoolkit.github.io/#publications) lists them.
