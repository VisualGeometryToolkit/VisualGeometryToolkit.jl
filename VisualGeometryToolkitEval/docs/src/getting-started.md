# Getting started

This page walks through the two things most users do first: summarize the pose
errors of an evaluation, and tune a hyperparameter by cross-validation. The
examples run without any data.

Each part of the package is a submodule, and its names are imported from it:

```@example start
using VisualGeometryToolkitEval.Stats: finite_median, eligible_maa, bootstrap_ci, stable_seed
using VisualGeometryToolkitEval.Tuning: tuning_spec, grid_spec, grid
using VisualGeometryToolkitEval.Evaluation: coarse_fine_cv
using Random: Xoshiro
using TOML
nothing # hide
```

## Summarizing pose errors

An evaluation produces one pose error per image pair, in degrees. A pair where
the estimator returned no model has the error `NaN`.

```@example start
errors = [0.4, 1.2, NaN, 0.8, 3.5, 0.6, 2.1]
finite_median(errors)
```

[`finite_median`](@ref) drops the non-finite errors before it takes the median.
The mean average accuracy (mAA) instead counts a failed pair as zero accuracy,
because the pair was eligible to run:

```@example start
eligible_maa(errors; T = 10.0)
```

[`bootstrap_ci`](@ref) gives a bias-corrected and accelerated (BCa) bootstrap
interval of any statistic. The random number generator (RNG) seeds the
resampling, so the interval is reproducible:

```@example start
bootstrap_ci(finite_median, Xoshiro(1), errors)
```

Every random sample of the stack is seeded by [`stable_seed`](@ref), which
derives a seed from a spec's seed and a name. Adding or removing another scene
therefore leaves a scene's sample unchanged:

```@example start
stable_seed(2026, "scene_a") == stable_seed(2026, "scene_a")
```

## Tuning a threshold

A tuning is specified in a `[tuning.<name>]` table of a TOML config. The table
names a scheme with `type` and states only what differs from the defaults in
the package's `configs/tuning.toml`. Every spec names its data through a
registered scene split; here, the PhotoTourism split of `configs/splits.toml`.

```@example start
config = TOML.parse("""
    [tuning.threshold]
    type = "coarse_fine"
    coarse = { type = "spaced", window = { type = "fixed", lo = 0.1, hi = 10.0 }, spacing = { type = "log_per_decade", per_decade = 4 } }
    cross_validation = { scene_split = { type = "registered", name = "phototourism" } }
    """)
spec = tuning_spec(config, "threshold")
spec.scheme
```

The coarse grid spans two decades at four points per decade:

```@example start
collect(grid(grid_spec(spec.scheme.coarse)))
```

[`coarse_fine_cv`](@ref) runs the coarse-to-fine search on an objective. The
objective returns the validation errors at a threshold `x`; the spec's
criterion (the median error, by default) reduces them to one number. Here a
stand-in objective is smallest at `x = 2`:

```@example start
validation_errors(x) = [abs(log(x / 2)) + 0.1 * k for k in 1:5]
result = coarse_fine_cv(validation_errors, grid_spec(spec.scheme.coarse), spec.scheme;
                        method_name = :threshold)
result[:x_star_fine]
```

The search first evaluates the coarse grid. When the best point lies near an
edge of the window, the grid extends on that side. A fine grid over the
neighbours of the coarse optimum then gives the tuned value. A result whose
optimum is still on the boundary after every extension is flagged, and the
readers of tuned values refuse it:

```@example start
result[:boundary_hit]
```

## Next steps

- [The experimental protocol](@ref) states how the stack's papers tune,
  evaluate and report.
- [The `[tuning]` schema](@ref) lists every parameter of a tuning spec.
- [The composition evaluator](@ref) evaluates a scorer together with a local
  optimizer on stored pools.
- [Paper projects](@ref) describes how a paper's figures are produced and
  checked.
