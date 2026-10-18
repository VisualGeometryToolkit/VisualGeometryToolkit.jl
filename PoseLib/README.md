# PoseLib.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=PoseLib)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=PoseLib)

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++
library of minimal solvers and robust refiners for camera pose estimation by
**Viktor Larsson and contributors**. Every solver and refiner reached from
here is theirs; this package is a thin binding layer over it. If you use it in
published work, cite PoseLib (and the original publications of the methods
you use, listed in PoseLib's README):

```bibtex
@misc{PoseLib,
  title  = {{PoseLib - Minimal Solvers for Camera Pose Estimation}},
  author = {Viktor Larsson and contributors},
  URL    = {https://github.com/vlarsson/PoseLib},
  year   = {2020}
}
```

## Installation

```julia
using Pkg
Pkg.add("PoseLib")
```

**Linux only for now.** The C++ library ships prebuilt as this package's own
artifact (`Artifacts.toml`, one tarball per platform, fetched by `Pkg` at
install; no compiler is needed) for Linux: x86_64, aarch64, i686, armv6l and
armv7l (glibc and musl), powerpc64le and riscv64 (glibc). On macOS and Windows
the package installs but refuses to load, with an error pointing here; those
platforms arrive with the `PoseLib_jll` patch release below.
The tarballs are built with [BinaryBuilder](https://github.com/JuliaPackaging/BinaryBuilder.jl)
from the recipe in [`yggdrasil/P/PoseLib`](yggdrasil/P/PoseLib/build_tarballs.jl)
by [`scripts/build_binaries.jl`](scripts/build_binaries.jl), which also writes
`Artifacts.toml`. The same recipe is submitted to Yggdrasil as `PoseLib_jll`,
which builds it for every platform; a later patch release switches to that JLL
once it is registered, bringing macOS, Windows and FreeBSD, with no change to the
API. Source: <https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/PoseLib>.

Loading this package extends VisualGeometryToolkitCore with:

- `PoseLibSolver` backend methods for the shared minimal solvers —
  `homography_4pt`, `fundamental_7pt`, `p3p`, `essential_5pt`;
- `essential_dlt`, which has no native variant, so its no-backend default
  routes here;
- PoseLib's robust-loss LM refiners — `refine_relpose`, `refine_fundamental`,
  `refine_homography`, `refine_absolute_pose` — whose `loss` is a
  VisualGeometryToolkitCore loss shape (`TLSShape()`, the MSAC default; `HuberShape()`,
  `CauchyShape()`, `GaussianShape()`) or `LeZachTruncation()` (the same
  truncation under the Le & Zach TR-IRLS scheme), at a `loss_width` that is the
  width of that shape's own closed forms, and whose `max_iter` the caller
  supplies.


## Conventions

The stack's conventions — how estimators are named, how signatures order their
arguments, and the frame-in = frame-out rule — are one page, VisualGeometryToolkitCore's
[Conventions](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/VisualGeometryToolkitCore/conventions/README.md)
(`docs/src/guide/conventions.md`). This package is a BINDING LAYER, and that
fixes what it may and may not do with them.

- **Arguments are in the caller's frame, and results come back in it.** The seed
  model, its correspondences and the refined model are all in one frame, and the
  binding never asks which: `refine_fundamental` and `refine_homography` take
  `cs` in the frame their `F` or `H` is expressed in — pixels for a pixel-frame
  model. The camera-aware `refine_relpose` likewise takes `K1`/`K2` as the
  intrinsics of whatever frame its points are in, not a pixel `K` specifically
  (LocalOptimizationDoneRight's refit layer hands it the pool frame's `Kc = T·K`).
- **It performs no normalization of its own.** This package holds no estimator —
  only a call across the C ABI — so there is no conditioning here to own.
  Nothing takes or returns a `CspondNormalization`, and no argument is a
  pre-conditioned coordinate. Whatever PoseLib does inside is private to the C++
  and reaches no signature.
- **A loss width is in the unit of the data handed over.** `loss_width` is the
  loss's width on that refiner's own residual, in the caller's units: never
  converted, never pre-divided by a focal on the caller's behalf. PoseLib's
  `loss_scale` equals VisualGeometryToolkitCore's `loss_width` for every bound loss
  (truncated, Le–Zach, Huber, Cauchy: `ρ = min(r², w²)`, Huber's elbow at `w`,
  `w²·log1p(r²/w²)`), so it crosses unchanged. One width stays
  valid across the camera-aware `refine_relpose`'s two images even when their
  focals differ, because PoseLib calibrates internally through the cameras it is
  given.
- **Where the crossing is lossy, the binding says so.** Each `K` reaches PoseLib
  as a SIMPLE_PINHOLE camera — focal `(K₁₁ + K₂₂)/2`, principal point
  `(K₁₃, K₂₃)` — so skew and any `K₁₁ ≠ K₂₂` are dropped, and the model is exact
  only for a square-pixel, zero-skew `K`.

**Calibrated currency: unit bearings, everywhere.** The minimal solvers take
VGC's: `p3p`, `essential_5pt` and `essential_dlt` take 3-vector bearings, as the
bearing rule on that page requires, and a 2-vector endpoint is a `MethodError`
there exactly as in VGC's own minimal kernels. They do not check the norm — they are
inner-loop pieces, their algebra is scale-invariant per ray, and the guarantee
comes from the producer (`backproject`), not the consumer.

For relative-pose refinement the bearing entry point is
`refine_relpose(pose, cs)` on `f₁ => f₂` of unit bearings, carrying their 3×2
backprojection Jacobians as the attribute columns `M₁`/`M₂` — PoseLib's
tangent-Sampson refiner (`bundle.cc:228`), which the camera-aware pixel wrapper
itself reduces to. A ray at or beyond 90° is an ordinary bearing there. Its
`loss_width` is on the tangent-Sampson residual, and **`M` picks the unit and
the call cannot**: `M = ∂f/∂u` from `backproject_with_jacobian` makes it a
PIXEL width, an orthonormal tangent basis makes it radians.

For absolute pose, `refine_absolute_pose(pose, K, cs)` takes pixel points with
a PIXEL `loss_width`, refining the pose only.

**The z = 1 chart is refused.** Handing either calibrated entry point 2-vectors
raises an `ArgumentError` naming the form to use: the chart (`GnomonicFrame`)
has no point for a ray at the horizon, and calibrated geometry in this stack is
on unit bearings (unit rays). The bearing entry point also checks the norm,
because `StaticVector{3}` is a necessary condition and not a guarantee — a
chart point lifted to `(x/z, y/z, 1)` satisfies the signature, keeps the same
direction, and still refines to a plausible pose, so the 90° exclusion would
survive the type and merely stop being visible.

One gap is worth stating plainly: PoseLib has **no** absolute-pose refiner on
bearings — every `bundle_adjust` overload takes `Point2D`, the residual being a
reprojection into an image rather than an angle. The pixel form above is
therefore the only absolute-pose path here, and it crosses as SIMPLE_PINHOLE,
which cannot project a ray at or beyond 90° either. A fisheye caller has no
correct absolute-pose refinement in this package. That was equally true before;
it is now visible rather than hidden behind a chart that mis-measured it.

## Testing

Plain `Test.@testset` in `test/runtests.jl`; the whole suite runs at once.

## Building the wrapper from source

Only needed when changing `wrapper/julia_wrapper.cpp` (the flat C ABI every
`ccall` goes to) ahead of a new binary release. Build the library with the
recipe's own steps — upstream PoseLib at the commit pinned in
`yggdrasil/P/PoseLib/build_tarballs.jl`, Eigen 3.4.0, then

```sh
c++ -std=c++17 -O3 -fPIC -shared -I<poselib-install>/include -I<eigen>/include/eigen3 \
    wrapper/julia_wrapper.cpp -o lib/libposelib_jl.so -L<poselib-install>/lib -lPoseLib
```

— or run the recipe with BinaryBuilder, from an environment holding it (not the
package's own):

```sh
julia --project=<bb-env> scripts/build_binaries.jl --workdir=<dir> x86_64-linux-gnu-cxx11
```

which builds into `<dir>/products` and rewrites `Artifacts.toml` (omit the
triplet for every published Linux platform; `--all` builds the recipe's full
list, whose macOS entries need `BINARYBUILDER_AUTOMATIC_APPLE=true`). Then point
the `PoseLib` artifact at your build with the standard artifact override, an
entry in `<depot>/artifacts/Overrides.toml` naming a directory that holds
`lib/libposelib_jl.so`:

```toml
[a8c7e0b4-3d2f-4f1c-9a1e-7b2d9c8f4a3b]   # PoseLib.jl's UUID
PoseLib = "/abs/path/to/prefix"
```

Restart Julia; deleting the entry undoes it.

When the wrapper changes, keep `yggdrasil/P/PoseLib/bundled/` in step (the test
suite checks the copies are identical; the build script refuses to build
otherwise), bump the recipe's `version`, rebuild every platform and publish the
tarballs before PoseLib.jl uses the new symbols.

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes
`Project.toml`, they re-resolve the environment (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## License

PoseLib.jl is MIT-licensed (see `LICENSE`). The prebuilt binary it loads
contains PoseLib (BSD-3-Clause, © Viktor Larsson) and Eigen (MPL-2.0), whose
license texts ship inside that artifact under `share/licenses/PoseLib/`.
