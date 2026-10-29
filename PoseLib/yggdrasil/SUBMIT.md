# Submitting `PoseLib_jll` to Yggdrasil

`P/PoseLib/` here is laid out exactly as it goes into
[JuliaPackaging/Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil):

```
P/PoseLib/build_tarballs.jl                   the recipe
P/PoseLib/bundled/wrapper/julia_wrapper.cpp   copy of wrapper/julia_wrapper.cpp
```

What the JLL ships: one shared library, `libposelib_jl`, which is upstream
PoseLib (built static, BSD-3-Clause), Eigen 3.4.0 (header-only, MPL-2.0) and
PoseLib.jl's flat C wrapper (MIT) linked together, plus the three license
texts under `share/licenses/PoseLib/`. Upstream PoseLib is pinned at
`a69263d` (v2.0.5 + 30 commits), the source PoseLib.jl's results were produced
with; the JLL version is `2.0.6` (see the comment in the recipe).

## 1. Optional: build locally

From a scratch environment holding BinaryBuilder (not the package's own):

```sh
cd P/PoseLib
julia --project=<scratch-env> build_tarballs.jl --verbose x86_64-linux-gnu-cxx11
# every platform the recipe lists:
julia --project=<scratch-env> build_tarballs.jl --verbose
# also generate a dev PoseLib_jll (in ~/.julia/dev) to test PoseLib.jl against:
julia --project=<scratch-env> build_tarballs.jl --deploy=local x86_64-linux-gnu-cxx11
```

Until `PoseLib_jll` is registered, PoseLib.jl resolves only against such a
local JLL: `Pkg.develop(path = joinpath(homedir(), ".julia", "dev", "PoseLib_jll"))`
in the PoseLib.jl environment (the Manifest is untracked, so this stays local).

Verified locally (2026-10-07, BinaryBuilder 0.6.6; GCC 8.1 where available): all 16
Linux, Windows and FreeBSD platforms build and clear the audit. The two macOS
platforms were not built locally (the macOS SDK licence must be accepted
first, `BINARYBUILDER_AUTOMATIC_APPLE=true`); Yggdrasil's CI builds them. With
the x86_64-linux-gnu-cxx11 tarball, PoseLib.jl's suite is green and the solver
and refiner outputs are bitwise identical to the former from-source build.

Since 2026-10-15 the recipe builds with GCC 14 (`preferred_gcc_version`) and
`julia_compat = "1.12"`: with GCC 8, PoseLib's refiners ran 64-69 % slower on
LocalOptimizationDoneRight's LO timing gate; GCC 14 meets it (comment in the
recipe). The build submitted in Yggdrasil#15061 predates this change.

## 2. Open the PR

```sh
gh repo fork JuliaPackaging/Yggdrasil --clone
cd Yggdrasil && git checkout -b poselib
cp -r <PoseLib.jl>/yggdrasil/P/PoseLib P/
git add P/PoseLib && git commit -m "[PoseLib] New recipe"
git push -u origin poselib && gh pr create
```

Yggdrasil CI builds every platform; after merge the registration bot opens the
`PoseLib_jll` PR on General.

## 3. Once PoseLib.jl is public

When <https://github.com/VisualGeometryToolkit/PoseLib.jl> is public and
tagged, follow up with a PR that replaces the `DirectorySource` with the
commented `GitSource` (the commit of that tag) and deletes `bundled/`; the
script reads the same paths either way. Any change to `julia_wrapper.cpp`
needs a new `PoseLib_jll` build (bump `version`) before PoseLib.jl may call new
symbols.

## 4. Then PoseLib.jl

Until `PoseLib_jll` is registered, PoseLib.jl ships the same recipe's Linux
tarballs as its own artifact (`Artifacts.toml`, written by `scripts/build_binaries.jl`).
Once the JLL is in General, a patch release replaces the artifact with a
dependency on `PoseLib_jll` (compat `2.0.6`) and deletes `Artifacts.toml`; the
`ccall` targets stay the same.
