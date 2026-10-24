# Developer Notes

This page is for people who work on the package itself. Users need none of it.

## The repository

VisualGeometryToolkitFeatures lives in the `VisualGeometryToolkitFeatures/`
directory of the
[VisualGeometryToolkit.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
repository, a Julia workspace of several packages with one Manifest at its root.
Instantiate it once per clone, from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The repository ships git hooks in `.githooks/` that re-resolve the root
Manifest after a pull, checkout or rebase that changes a `Project.toml`. They
are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## Tests

The suite uses ReTestItems; the items live in `test/*_tests.jl`. From the
repository root:

```bash
julia --project=VisualGeometryToolkitFeatures -e 'using Pkg; Pkg.test()'
julia --project=VisualGeometryToolkitFeatures -e 'using Pkg; Pkg.test(; test_args = ["JSON"])'
```

The second form runs the items whose names match the pattern. Check the total
it reports: a pattern that matches nothing runs nothing.

`test/readme_tests.jl` runs every `julia` block of the README except the
installation block, and checks each statement whose line ends in `# true`.
Keep the README's quick start self-contained: no files, no network.

## Documentation

The docs index is generated from the README by `docs/make.jl`, so edit the
README rather than `docs/src/index.md`. Build from the repository root:

```bash
julia --project=VisualGeometryToolkitFeatures/docs --startup-file=no VisualGeometryToolkitFeatures/docs/make.jl
```

The build runs the doctests and fails on an exported name without a docstring.

## Benchmarks and examples

`benchmark/mc` holds the Monte Carlo studies of the detector, which are not
part of the test suite: `keypoint_uncertainty_mc.jl` calibrates the keypoint
covariance, and `zeisl_comparison.jl` compares it with Zeisl et al. (BMVC 2009).
Both write their results to `benchmark/mc/results/`, which is not tracked.

`examples/` has its own environment, outside the workspace, because its CUDA
and cuDNN pins would otherwise hold back the CUDA of every package's tests.
Instantiate it with `julia --project=VisualGeometryToolkitFeatures/examples -e 'using Pkg; Pkg.instantiate()'`.

## Accuracy bounds that other code relies on

The refinement accuracy pinned by the test suite (two detections of the same
small, isolated blob differ by at most 0.06 px) is what
VisualGeometryToolkitCore's coincidence gate is built around. That gate,
`COINCIDENT_PX` = 0.1 px, supplied as `tol_px` in Core's
`src/geometry/model_validity.jl`, rejects minimal samples with coincident
points, and it is set to clear the 0.06 px bound with margin. A change to the
refinement that loosens the bound therefore changes which samples Core's
estimators reject. Coordinate it with Core instead of retuning the test here.

## Serialization

The JSON hooks (`JSON.lower`, `JSON.lift` and StructUtils' `structlike`) are
methods on functions of JSON.jl and StructUtils for types that
VisualGeometryToolkitCore owns. That is type piracy by Aqua's definition, and
the Aqua test lists those types under `treat_as_own`. The hooks would move to
VisualGeometryToolkitCore if Core took on a JSON dependency. The written format
is pinned by a test, since stored files depend on it.
