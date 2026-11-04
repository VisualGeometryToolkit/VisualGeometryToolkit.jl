# PoseLib.jl

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++
library of minimal solvers and robust refiners for camera pose estimation by
Viktor Larsson and contributors. Every solver and refiner reached from this
package is theirs. PoseLib.jl is a thin binding layer and part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack: its inputs and outputs are the types of
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/).

## What it provides

- A PoseLib backend, `PoseLibSolver()`, for VisualGeometryToolkitCore's
  minimal solvers: the four-point homography, the seven-point fundamental
  matrix, the perspective-three-point (P3P) absolute pose and the five-point
  essential matrix, plus the eight-point essential matrix, `essential_dlt`.
  See [Minimal solvers](guide/solvers.md).
- PoseLib's robust Levenberg–Marquardt (LM) refiners:
  [`refine_homography`](@ref), [`refine_fundamental`](@ref),
  [`refine_relpose`](@ref) and [`refine_absolute_pose`](@ref). See
  [Refiners](guide/refiners.md).

Design principles, each explained on the [Refiners](guide/refiners.md) page:

- **Frame in, frame out.** The bindings take data in the caller's frame and
  return results in it; `loss_width` is in the unit of the data handed over.
- **Calibrated geometry is on unit bearings**, never on the z = 1 plane, so a
  ray at or beyond 90° is an ordinary input.
- **The loss is a VisualGeometryToolkitCore loss shape** (or
  [`LeZachTruncation`](@ref)`()`), mapped to PoseLib's own loss type only at the
  call into the library.

## Installation

```julia-repl
pkg> add PoseLib VisualGeometryToolkitCore
```

Julia 1.12 or later. The C++ library is downloaded prebuilt, as the binary
package `PoseLib_jll`, so no compiler is needed; see
[Platforms and binaries](guide/binaries.md).

## A first example

The example estimates a camera's pose from three points with PoseLib's P3P
solver, then refines it on all the points with PoseLib's absolute-pose
refiner.

```julia
using PoseLib, VisualGeometryToolkitCore

# A camera at the origin looks down the z axis at 25 points on a curved
# surface, six units away; K is its intrinsic matrix, in pixels.
K = [800.0 0.0 320.0; 0.0 800.0 240.0; 0.0 0.0 1.0]
X = [Point3(x, y, 6.0 + 0.5x * y) for x in -1.0:0.5:1.0 for y in -1.0:0.5:1.0]
pixel(x) = (y = K * x; Point2(y[1] / y[3], y[2] / y[3]))
u = pixel.(X)

# PoseLib's P3P solver on three of the points and their bearings (unit rays).
f = [Vec3(x / sqrt(sum(abs2, x))) for x in X]
poses = solve(P3P(PoseLibSolver()), Csponds(X, f), [1, 13, 25])

# Keep the solution that reprojects all 25 points best, then refine it.
reprojection(pose) = sum(sum(abs2, pixel(pose(x)) - v) for (x, v) in zip(X, u))
best = poses[argmin(reprojection.(poses))]
pose = refine_absolute_pose(unlabel(best), K, Csponds(X, u); max_iter = 50, loss_width = 2.0)
maximum(abs, pose.t) < 1e-8      # true: the camera is at the origin
```

## Documentation

**Start here**

- [Getting started](guide/getting-started.md): a P3P solve and each refiner,
  run as doctests.

**How to**

- [Minimal solvers](guide/solvers.md): selecting the PoseLib backend.
- [Platforms and binaries](guide/binaries.md): what is downloaded, for which
  platforms, and running against a library built from source.
- [Developer notes](developer.md): rebuilding the C wrapper and the binaries.

**Explanation**

- [Refiners](guide/refiners.md): the losses, the unit of `loss_width`,
  coordinate frames and bearings.
- VisualGeometryToolkitCore's
  [Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/):
  argument order, coordinate frames and bearings across the stack.

**Reference**

- [API reference](api.md).

## Role in the stack

VisualGeometryToolkitCore defines the minimal solvers and their interface,
`solve(solver, cs, idx)`; this package supplies the PoseLib backend. The
robust estimators of VisualGeometryToolkitEstimation and the benchmarks of
VisualGeometryToolkitEval select it to compare against PoseLib itself. The
[portal](https://visualgeometrytoolkit.github.io) links every package.

## Citing

Cite PoseLib, and the original publications of the methods you use (PoseLib's
[README](https://github.com/PoseLib/PoseLib#readme) lists them):

```bibtex
@misc{PoseLib,
  title  = {{PoseLib - Minimal Solvers for Camera Pose Estimation}},
  author = {Viktor Larsson and contributors},
  URL    = {https://github.com/vlarsson/PoseLib},
  year   = {2020}
}
```

PoseLib.jl is part of VisualGeometryToolkit. Cite the toolkit as well; its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  doi    = {10.5281/zenodo.23548372},
  year   = {2026}
}
```
