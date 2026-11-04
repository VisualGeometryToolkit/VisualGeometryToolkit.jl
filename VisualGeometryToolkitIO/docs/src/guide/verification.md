# Testing and Verification

## What the suite checks

- The binary wire layer is round-tripped through pycolmap, COLMAP's reference
  implementation (`test/pycolmap_tests.jl`), and every value must come back
  identical. This check needs the pixi `py` environment (`pixi install -e py`).
  Without it, the check is recorded as skipped.
- Aqua, ExplicitImports and a JET inference ratchet with a baseline of zero
  reports run as part of the normal suite (`test/quality_tests.jl`).
- The README's example, which the home page of this manual shows verbatim,
  runs as written in a fresh environment built from the README's install line,
  and that line must name exactly the packages the example loads
  (`test/readme_tests.jl`). The docs build runs the examples in this manual as
  doctests.
- Every public name has a docstring, attached to its definition, whose
  signature lines a method accepts (`test/docs_tests.jl`).

## Running the tests

The suite uses ReTestItems, like VisualGeometryToolkitCore's. Every
`test/*_tests.jl` file holds `@testitem`s that run on worker processes.

```julia
using Pkg
Pkg.test("VisualGeometryToolkitIO")                                   # every item
Pkg.test("VisualGeometryToolkitIO"; test_args = ["migrate|cameras"])  # items whose name matches
```

Set `RETESTITEMS_NWORKERS=0` to run the items serially in one process, which
is easier to debug.

Before a release, it is worth running
`Pkg.test("VisualGeometryToolkitIO"; julia_args = ["--depwarn=error"])`. This
package sits under a stack that releases often (VisualGeometryToolkitCore,
then this package, then BlobBoards), so an upstream deprecation is cheapest to
fix in the release that introduces it.

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable. After a pull, checkout or rebase that changes a
`Project.toml`, they re-resolve the environment. They only resolve;
precompilation waits for the next `using`. The hooks are opt-in, once per
clone:

```bash
git config core.hooksPath .githooks
```
