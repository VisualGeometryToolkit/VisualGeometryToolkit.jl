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

The library is `PoseLib_jll`. Its recipe, Yggdrasil's `P/PoseLib/build_tarballs.jl`,
pins the upstream PoseLib commit that the package's results were produced
with and builds a copy of the wrapper bundled beside it.

## Tests

```sh
julia --project=PoseLib -e 'using Pkg; Pkg.test()'
```

The suite is plain `Test.@testset` in `test/runtests.jl` and runs as a whole.
Besides the solvers and refiners, it checks that every julia block of the
README runs.

## Building the wrapper from source

This is needed only when changing `wrapper/julia_wrapper.cpp` ahead of a new
`PoseLib_jll` release. Build upstream PoseLib at the commit pinned in the
recipe, with Eigen 3.4.0, then compile the wrapper against it:

```sh
c++ -std=c++17 -O3 -fPIC -shared -I<poselib-install>/include -I<eigen>/include/eigen3 \
    wrapper/julia_wrapper.cpp -o lib/libposelib_jl.so -L<poselib-install>/lib -lPoseLib
```

## Loading your own build

Point `PoseLib_jll`'s artifact at your build with Julia's standard artifact
override: an entry in `<depot>/artifacts/Overrides.toml` naming a directory
that holds `lib/libposelib_jl.so`. The key is `PoseLib_jll`'s UUID.

```toml
[a1d785cd-0a5b-55b9-92fb-09c8b868874a]
PoseLib = "/abs/path/to/prefix"
```

Restart Julia for the override to take effect; deleting the entry undoes it.

## Changing the wrapper

When the wrapper changes:

1. Open a Yggdrasil pull request that copies it into `P/PoseLib/bundled/` and
   bumps the recipe's `version`.
2. Once the new `PoseLib_jll` is registered, raise PoseLib.jl's `[compat]`
   floor for it.
3. Only then may PoseLib.jl call the new symbols.

## Git hooks

The repository ships git hooks in `.githooks/` that keep the untracked root
`Manifest.toml` resolvable. After a pull, checkout or rebase that changes a
`Project.toml`, they re-resolve the environment. They are opt-in, once per
clone:

```sh
git config core.hooksPath .githooks
```
