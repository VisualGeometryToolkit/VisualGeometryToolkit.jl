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
julia --project=VisualGeometryToolkitFeatures -e 'using Pkg; Pkg.test(; test_args = ["polarity"])'
```

The second form runs the items whose names match the pattern. Check the total
it reports: a pattern that matches nothing runs nothing.

`test/readme_tests.jl` checks that the README's install line names exactly the
packages its example loads, that the landing page (`docs/src/index.md`) shows
the same install line and the same example, and runs the example in a fresh
environment built from the install line. Keep the example self-contained: no
files, no network.

## Documentation

Build from the repository root:

```bash
julia --project=VisualGeometryToolkitFeatures/docs --startup-file=no VisualGeometryToolkitFeatures/docs/make.jl
```

The build runs the doctests and checks the links, fails on any warning, and
fails on a public docstring missing from the manual. `test/docs_tests.jl`
fails on a public name without a docstring, a docstring detached from its
definition, and a signature line no method accepts.

## Benchmarks and examples

`benchmark/mc` holds the Monte Carlo studies of the detector, which are not
part of the test suite: `keypoint_uncertainty_mc.jl` calibrates the keypoint
covariance, and `zeisl_comparison.jl` compares it with Zeisl et al. (BMVC 2009).
Both write their results to `benchmark/mc/results/`, which is not tracked.

`examples/` has its own environment, outside the workspace, so its plotting
and image-loading dependencies stay out of the one workspace Manifest. It holds
`CUDA_Runtime_jll` and `CUDNN_jll` (as the test environment does), whose CUDA 12
builds ONNX Runtime's CUDA provider takes (`src/onnx_runtime.jl`). The CUDA
build of ONNX Runtime is conda-forge's, the artifact `onnxruntime_cuda` of
`Artifacts.toml` (a tree hash only: the package installs it from the
conda-forge files `_ORT_CUDA_FILES` pins by sha256, since Pkg cannot unpack a
`.conda`). To move it to a new release, take the `onnxruntime-cpp` `cuda12x`
build and the `libprotobuf`, `libabseil` and `libre2-11` builds its metadata
requires, update their names and sha256s, and pin the tree hash the merged
install gives.
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

The package writes no file format. Detections, feature frames, `Size2` and
quantities are serialized by VisualGeometryToolkitIO (`json_value`,
`from_json`), which owns the stack's quantity JSON format: numbers in the
record's declared units. The `{"value", "unit"}` objects this package's
JSON.jl hooks once wrote are not read; a file holding them is migrated once
(`vgio_migrate`).
