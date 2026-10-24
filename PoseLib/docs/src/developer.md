# Developer notes

This page is for maintainers of PoseLib.jl. Paths are relative to the
package's directory, `PoseLib/`, in the
[VisualGeometryToolkit.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
repository.

## Layout

| Path | What it is |
|---|---|
| `src/PoseLib.jl` | the bindings: every `ccall` and its Julia types |
| `wrapper/julia_wrapper.cpp` | the flat C interface that every `ccall` targets |
| `yggdrasil/P/PoseLib/build_tarballs.jl` | the BinaryBuilder recipe that builds the library |
| `yggdrasil/P/PoseLib/bundled/` | copies of the wrapper and `LICENSE` that the recipe builds from |
| `yggdrasil/SUBMIT.md` | how the recipe is submitted to Yggdrasil as `PoseLib_jll` |
| `scripts/build_binaries.jl` | runs the recipe and writes `Artifacts.toml` |
| `Artifacts.toml` | the prebuilt tarballs, one per platform |

The recipe pins the upstream PoseLib commit that the package's results were
produced with.

## Tests

```sh
julia --project=PoseLib -e 'using Pkg; Pkg.test()'
```

The suite is plain `Test.@testset` in `test/runtests.jl` and runs as a whole.
Besides the solvers and refiners, it checks that the bundled copies match
`wrapper/julia_wrapper.cpp` and `LICENSE`, that `Artifacts.toml` matches the
recipe's version, and that every julia block of the README runs.

## Building the wrapper from source

This is needed only when changing `wrapper/julia_wrapper.cpp` ahead of a new
binary release. Build upstream PoseLib at the commit pinned in the recipe, with
Eigen 3.4.0, then compile the wrapper against it:

```sh
c++ -std=c++17 -O3 -fPIC -shared -I<poselib-install>/include -I<eigen>/include/eigen3 \
    wrapper/julia_wrapper.cpp -o lib/libposelib_jl.so -L<poselib-install>/lib -lPoseLib
```

Alternatively, run the recipe with BinaryBuilder, from an environment that
holds BinaryBuilder (not the package's own):

```sh
julia --project=<bb-env> scripts/build_binaries.jl --workdir=<dir> x86_64-linux-gnu-cxx11
```

This builds into `<dir>/products` and rewrites `Artifacts.toml`. Without a
platform triplet it builds every published Linux platform. `--all` builds the
recipe's full list, whose macOS entries need `BINARYBUILDER_AUTOMATIC_APPLE=true`.

## Loading your own build

Point the `PoseLib` artifact at your build with Julia's standard artifact
override: an entry in `<depot>/artifacts/Overrides.toml` naming a directory
that holds `lib/libposelib_jl.so`. The key is PoseLib.jl's UUID.

```toml
[a8c7e0b4-3d2f-4f1c-9a1e-7b2d9c8f4a3b]
PoseLib = "/abs/path/to/prefix"
```

Restart Julia for the override to take effect; deleting the entry undoes it.

## Changing the wrapper

When the wrapper changes:

1. Copy it into `yggdrasil/P/PoseLib/bundled/`. The test suite checks that the
   copies are identical, and the build script refuses to build otherwise.
2. Bump the recipe's `version`.
3. Rebuild every platform and publish the tarballs.
4. Only then may PoseLib.jl call the new symbols.

## Git hooks

The repository ships git hooks in `.githooks/` that keep the untracked root
`Manifest.toml` resolvable. After a pull, checkout or rebase that changes a
`Project.toml`, they re-resolve the environment. They are opt-in, once per
clone:

```sh
git config core.hooksPath .githooks
```
