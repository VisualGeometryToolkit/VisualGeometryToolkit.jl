# PoseLib.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/PoseLib/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=PoseLib)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=PoseLib)

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++
library of minimal solvers and robust refiners for camera pose estimation by
Viktor Larsson and contributors. Every solver and refiner reached from this
package is theirs. PoseLib.jl is a thin binding layer, part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack.

Loading the package does two things:

- It adds a PoseLib backend, `PoseLibSolver()`, to the minimal solvers of
  [VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/):
  the four-point homography, the seven-point fundamental matrix, the
  perspective-three-point (P3P) absolute pose and the five-point essential
  matrix. It also adds the eight-point essential matrix, `essential_dlt`.
- It exports PoseLib's robust Levenberg–Marquardt (LM) refiners:
  `refine_homography`, `refine_fundamental`, `refine_relpose` and
  `refine_absolute_pose`.

## Installation

PoseLib is in [CauRegistry](https://github.com/prittjam/CauRegistry.jl) until
its General registration merges; VisualGeometryToolkitCore is in General.

```julia
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl")   # once
Pkg.add(["PoseLib", "VisualGeometryToolkitCore"])
```

PoseLib.jl requires Julia 1.12 or later. The C++ library is downloaded
prebuilt, as the binary package `PoseLib_jll`, when the package is installed,
so no compiler is needed; see
[Platforms and binaries](https://visualgeometrytoolkit.github.io/PoseLib/dev/guide/binaries/).

## Quick start

Solve a P3P problem with PoseLib's solver. The camera sits at the origin and
looks down the z axis, so the bearing of each world point is the point scaled
to unit length.

```julia
using PoseLib, VisualGeometryToolkitCore

X = [Point3(1.0, 0.5, 5.0), Point3(-0.5, 1.0, 4.0), Point3(0.0, -0.5, 6.0)]
f = [Vec3(x / sqrt(sum(abs2, x))) for x in X]
poses = solve(P3P(PoseLibSolver()), Csponds(X, f), 1:3)
length(poses)          # the number of real solutions, at most 4
```

Refine a perturbed homography from point correspondences. The default loss is
a truncated quadratic, and `loss_width` is the residual in pixels beyond which
a correspondence counts as an outlier.

```julia
using PoseLib, VisualGeometryToolkitCore

H = HomographyMat([1.1 0.02 5.0; -0.03 0.95 -3.0; 1e-4 2e-4 1.0])
map_point(H, x) = (y = H * [x; 1.0]; Point2(y[1] / y[3], y[2] / y[3]))
x1 = [Point2(u, v) for u in 0.0:100.0:400.0 for v in 0.0:100.0:300.0]
x2 = [map_point(H, x) for x in x1]

H0 = HomographyMat([1.1 0.02 6.0; -0.03 0.96 -3.0; 1e-4 2e-4 1.0])
H1 = refine_homography(H0, Csponds(x1, x2); max_iter = 25, loss_width = 2.0)
H1 / H1[3, 3] ≈ H      # true
```

## Documentation

- [Stable](https://visualgeometrytoolkit.github.io/PoseLib/stable/) and
  [development](https://visualgeometrytoolkit.github.io/PoseLib/dev/) manuals:
  getting started, the minimal-solver backends, the refiners and their loss
  widths, platforms and binaries, and the API reference.
- The conventions the whole stack follows (argument order, coordinate frames,
  bearings) are in VisualGeometryToolkitCore's
  [Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/dev/guide/conventions/)
  page.

## Citing

If you use this package in published work, cite PoseLib, and also the
original publications of the methods you use (PoseLib's
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
