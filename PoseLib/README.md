# PoseLib.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=PoseLib)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=PoseLib)

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++
library of minimal solvers and robust refiners for camera pose estimation by
Viktor Larsson and contributors. Loading PoseLib.jl adds a PoseLib backend,
`PoseLibSolver()`, to the minimal solvers of
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/),
and exports PoseLib's robust Levenberg–Marquardt refiners for homographies,
fundamental matrices, relative poses and absolute poses.

## Installation

```julia-repl
pkg> add PoseLib VisualGeometryToolkitCore
```

Julia 1.12 or later. Every dependency is in the General registry. The C++
library is downloaded prebuilt, as the binary package `PoseLib_jll`, so no
compiler is needed.

## Example

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
pose = refine_absolute_pose(unlabel(best), K, Csponds(X, u); max_iter = 50, loss = TLSShape(), loss_width = 2.0)
maximum(abs, pose.t) < 1e-8      # true: the camera is at the origin
```

[Getting started](https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/getting-started/)
continues with the relative-pose and homography refiners.

## Documentation

- [Getting started](https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/getting-started/):
  a P3P solve and each refiner.
- [Minimal solvers](https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/solvers/)
  and [Refiners](https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/refiners/):
  selecting the PoseLib backend, the losses and the unit of `loss_width`.
- [Platforms and binaries](https://visualgeometrytoolkit.github.io/PoseLib/stable/guide/binaries/):
  what is downloaded, for which platforms, under which licenses.
- [API reference](https://visualgeometrytoolkit.github.io/PoseLib/stable/api/).

PoseLib.jl is part of the VisualGeometryToolkit packages. Its inputs and
outputs are VisualGeometryToolkitCore's types, and VisualGeometryToolkitEstimation's
RANSAC estimators can select its solver backend. The rules every package of the toolkit
follows are on VisualGeometryToolkitCore's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
page, and the [portal](https://visualgeometrytoolkit.github.io) links every
package.

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

PoseLib.jl is part of VisualGeometryToolkit. Cite the toolkit as well. Its
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

## License

PoseLib.jl is released under the MIT license (see [LICENSE](LICENSE)). The
prebuilt library it loads contains upstream
[PoseLib](https://github.com/PoseLib/PoseLib), released under the
BSD-3-Clause license, and [Eigen](https://eigen.tuxfamily.org), released under
the Mozilla Public License 2.0. Their license texts ship inside `PoseLib_jll`'s
artifact, under `share/licenses/PoseLib/`.
