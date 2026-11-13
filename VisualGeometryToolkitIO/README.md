# VisualGeometryToolkitIO.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitIO)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitIO)

VisualGeometryToolkitIO saves and loads cameras, poses, rigs, trajectories and
target observations as self-describing JSON files, one schema per
[VisualGeometryToolkitCore](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore)
type, and a value read back from a file equals the value that was written.
Each file opens with one envelope that declares its units (length, pixel
frame, density, …) and its conventions (rotation encoding, pose direction,
1-based indices) and names the software that wrote it, and every earlier
schema version a published file holds still loads. It also reads and writes COLMAP models, reads COLMAP
databases and OpenCV and MATLAB calibration exports, and exports to the TUM,
EuRoC, BOP and TRC formats.

## Installation

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitIO
```

Julia 1.13 or later. Every dependency is in the General registry.
VisualGeometryToolkitCore defines the types that are saved and loaded.

## Example

The example saves a pose and a camera and loads them back. `load` and `save`
are public but not exported, because other packages (FileIO among them)
export functions with the same names, so they are called through a module
alias.

```julia
using VisualGeometryToolkitCore, VisualGeometryToolkitIO
import VisualGeometryToolkitIO as VgtkIO

dir = mktempdir()

# A pose from the camera rig's body frame to the camera's optical frame, its
# frames part of its type. The translation is a length, held in metres; the
# file states it in millimetres.
X = relabel(BodyFrame{:camera_rig}, CameraFrame,
    lookat(Vec3(0.0, 0.0, 500.0) .* mm, Vec3(0.0, 0.0, 0.0) .* mm, Vec3(0.0, -1.0, 0.0)))
path = joinpath(dir, "hand_eye.json")
VgtkIO.save(path, [X]; provenance = (method = "robot-world hand-eye",))
X2 = only(VgtkIO.load(Vector{EuclideanMap}, path))
X2.t == X.t                                        # true: the translation comes back exactly

# A camera store: named cameras with their image size.
model = CameraModel(2400.0px, [2120.0, 1416.0] .* px;
    lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004))
camera = BoundedCamera(model, Size2(4240, 2832))
store = joinpath(dir, "cameras.json")
VgtkIO.save(store, Dict(:left => camera))
VgtkIO.load(BoundedCamera, store; name = :left) == camera   # true
VgtkIO.validate(store)                             # "cameras"
```

The `provenance` keyword is written to the file but not returned by `load`.
The [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/getting-started/)
guide shows the files this writes.

## Documentation

- [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/getting-started/):
  save and load a pose and a camera, and read the files.
- [Kinds and the I/O Surface](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/kinds/)
  and [Third-Party Calibrations](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/third-party/):
  every kind and format, reading an earlier schema version, and converting
  OpenCV and MATLAB exports.
- [Schema Rules](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/schemas/)
  and [Conventions on Disk](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/guide/conventions/):
  why the files look the way they do, and how a file declares its conventions.
- [API Reference](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/api/native/).

VisualGeometryToolkitIO is built on VisualGeometryToolkitCore, whose
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
page states the rules every package of the stack follows. The
[portal](https://visualgeometrytoolkit.github.io) links all the packages.

## Citing

VisualGeometryToolkitIO is part of VisualGeometryToolkit. Cite the toolkit.
Its [CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button
exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  doi    = {10.5281/zenodo.23548372},
  year   = {2026}
}
```

Cite also the papers behind the methods you use; the
[portal](https://visualgeometrytoolkit.github.io/#publications) lists them.

## License

MIT
