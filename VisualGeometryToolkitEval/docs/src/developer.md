# Developer notes

VisualGeometryToolkitEval is developed in the
[VisualGeometryToolkit.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
repository, a Julia workspace that holds the stack's packages, one directory
each. The repository's root `CLAUDE.md` describes the workspace, and the
conventions every stack package follows are in VisualGeometryToolkitCore's
`conventions/` folder.

## Running the tests

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'          # once per clone
julia --project=VisualGeometryToolkitEval -e 'using Pkg; Pkg.test()'
julia --project=VisualGeometryToolkitEval -e 'using Pkg; Pkg.test(; test_args = ["pool_io|tuning"])'
```

The suite uses ReTestItems. Every `test/*_tests.jl` file holds one test item
named after its file, and the items run in parallel on worker processes. The
`test_args` filter items by name; check the reported total, because a pattern
that matches nothing succeeds with zero tests. Fixtures that several items share
are `@testsetup` modules in `test/*_testsetup.jl`.

The default suite includes the quality items: Aqua, ExplicitImports, a JET
ratchet at zero reports, the calling conventions, a scan for undefined globals and the stack's
naming check. The `readme` item runs every `julia` block of the README.

## Building the documentation

```bash
julia --project=VisualGeometryToolkitEval/docs --startup-file=no VisualGeometryToolkitEval/docs/make.jl
```

The site is written to `VisualGeometryToolkitEval/docs/build/`.

## Dependency boundaries

The package depends on VisualGeometryToolkitCore, VisualGeometryToolkitIO and
VisualGeometryToolkitEstimation only. The evaluator is written against
VisualGeometryToolkitCore's contracts, so RANSACScoringDoneRight,
LocalOptimizationDoneRight and PoseLib never appear among its dependencies;
the projects that evaluate them depend on this package instead.

Four source files are free of dependencies by contract, so that an
environment that cannot load this package can include them by path:
`src/render/axis_style.jl`, `src/evaluation/cv_probe.jl`,
`src/experiments/stages.jl` and `src/reproduce/reproduce_index.jl`.

## Frames

A scoring pool ([`ScoringPool`](@ref)) is stored in `NormalizedImageFrame`
and says so ([`frame`](@ref)); a perspective-n-point (PnP) pool is stored in
`ImageFrame`. The pool and residual files hold plain data together with the
stored maps (`source_map`, `destination_map`), and a value leaves the pool frame
only through those maps. Nothing in this package converts a frame. The raw
readers of `Datasets` return pixels.
