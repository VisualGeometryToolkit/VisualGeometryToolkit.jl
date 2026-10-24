# VisualGeometryToolkitEval

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitEval)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitEval)

VisualGeometryToolkitEval is the evaluation package of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack. It holds what the stack's research projects share when they benchmark
robust estimators:

- summary statistics of pose errors, such as finite medians, the mean average
  accuracy (mAA) and bias-corrected and accelerated (BCa) bootstrap intervals;
- a codified cross-validation (CV) of hyperparameters, configured in TOML and
  persisted with its full specification;
- an evaluator that composes a RANSAC scorer with a local optimizer and
  measures the pose error of the result;
- readers for scoring pools and for raw two-view datasets;
- a framework for reproducible experiments and paper figures.

The evaluator is written against the contracts of
[VisualGeometryToolkitCore](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore),
so it runs any scorer with any local optimizer. A project that evaluates
particular scorers or optimizers depends on their packages itself.

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitEval")
```

The package requires Julia 1.12 or later.

## Quick start

Each part of the package is a submodule (`Stats`, `Tuning`, `Evaluation`,
`Pools` and others), and its names are imported from it. The example below
summarizes a set of pose errors, then tunes one threshold by the package's
coarse-to-fine cross-validation.

```julia
using VisualGeometryToolkitEval.Stats: finite_median, eligible_maa, bootstrap_ci
using VisualGeometryToolkitEval.Tuning: tuning_spec, grid_spec
using VisualGeometryToolkitEval.Evaluation: coarse_fine_cv
using Random: Xoshiro
using TOML

# Pose errors in degrees of seven image pairs; NaN marks a pair without a model.
errors = [0.4, 1.2, NaN, 0.8, 3.5, 0.6, 2.1]
finite_median(errors)                               # 1.0
eligible_maa(errors; T = 10.0)                      # mAA up to 10 degrees; NaN counts as 0
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

The [documentation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/stable/)
covers:

- the experimental protocol that the stack's papers follow;
- the `[tuning]` TOML schema;
- the composition evaluator;
- reproducible figures and the conventions of a paper project;
- the API reference, one page per submodule.

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

## License

MIT
