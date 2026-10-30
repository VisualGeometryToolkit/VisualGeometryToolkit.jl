# PoseLib.jl

Julia bindings to [PoseLib](https://github.com/PoseLib/PoseLib), the C++
library of minimal solvers and robust refiners for camera pose estimation by
Viktor Larsson and contributors. Every solver and refiner reached from this
package is theirs. PoseLib.jl is a thin binding layer and part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl)
stack: its inputs and outputs are the types of
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/).

Loading the package does two things.

- It adds a PoseLib backend, `PoseLibSolver()`, to VisualGeometryToolkitCore's
  minimal solvers: the four-point homography, the seven-point fundamental
  matrix, the perspective-three-point (P3P) absolute pose and the five-point
  essential matrix. It also adds the eight-point essential matrix,
  `essential_dlt`. See [Minimal solvers](guide/solvers.md).
- It exports PoseLib's robust Levenberg–Marquardt (LM) refiners:
  [`refine_homography`](@ref), [`refine_fundamental`](@ref),
  [`refine_relpose`](@ref) and [`refine_absolute_pose`](@ref). See
  [Refiners](guide/refiners.md).

## Installation

```julia
using Pkg
Pkg.add(["PoseLib", "VisualGeometryToolkitCore"])
```

PoseLib.jl requires Julia 1.12 or later. The C++ library is downloaded
prebuilt, as the binary package `PoseLib_jll`, when the package is installed,
so no compiler is needed; see [Platforms and binaries](guide/binaries.md).

## Contents

- [Getting started](guide/getting-started.md): a P3P solve and each refiner,
  run as doctests.
- [Minimal solvers](guide/solvers.md): selecting the PoseLib backend.
- [Refiners](guide/refiners.md): the losses, the unit of `loss_width`,
  coordinate frames and bearings.
- [Platforms and binaries](guide/binaries.md): what is downloaded, for which
  platforms, and under which licenses.
- [Developer notes](developer.md): rebuilding the C wrapper and the binaries.
- [API reference](api.md).

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
  year   = {2026}
}
```
