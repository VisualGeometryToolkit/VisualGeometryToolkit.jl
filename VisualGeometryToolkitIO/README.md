# VisualGeometryToolkitIO

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitIO)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitIO)

File formats for the [VisualGeometryToolkit](https://github.com/VisualGeometryToolkit)
stack. It saves and loads cameras, poses, rigs, trajectories and target
measurements as self-describing JSON files, one schema per
[VisualGeometryToolkitCore](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore)
type. Each file declares its conventions (pixel frame, length unit, rotation
encoding, pose direction), so it can be read correctly without outside
knowledge. Rotations are stored as quaternions, and a value read back from a
file equals the value that was written.

The package also reads and writes COLMAP models, reads COLMAP databases and
OpenCV or MATLAB calibration exports, and exports trajectories and
measurements to the TUM, EuRoC, BOP and TRC formats.

## Installation

```julia
using Pkg
Pkg.add(["VisualGeometryToolkitCore", "VisualGeometryToolkitIO"])
```

VisualGeometryToolkitIO requires Julia 1.12 or later. The examples below also
use VisualGeometryToolkitCore, which defines the types that are saved and
loaded.

## Quick start

`load` and `save` are public but not exported, because other packages (FileIO
among them) export functions with the same names. Call them through a module
alias.

```julia
using VisualGeometryToolkitCore, VisualGeometryToolkitIO
import VisualGeometryToolkitIO as VgtkIO

dir = mktempdir()

# A pose whose frames are part of its type: the camera rig's body frame to the
# camera's optical frame, in millimetres.
X = relabel(BodyFrame{:camera_rig}, CameraFrame,
    lookat(Vec3(0.0, 0.0, 500.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0)))

path = joinpath(dir, "hand_eye.json")
VgtkIO.save(path, [X]; provenance = (method = "robot-world hand-eye",))
X2 = only(VgtkIO.load(Vector{EuclideanMap}, path))
@assert X2.R ≈ X.R && X2.t == X.t
@assert X2 isa EuclideanMap{3, Float64, <:Any, <:Any, BodyFrame{:camera_rig}, CameraFrame}

# A camera store: named cameras with their image size.
model = CameraModel(2400.0px, [2120.0, 1416.0] .* px;
    lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004))
camera = BoundedCamera(model, Size2(4240, 2832))
store = joinpath(dir, "cameras.json")
VgtkIO.save(store, Dict(:left => camera))
@assert VgtkIO.load(BoundedCamera, store; name = :left) == camera
@assert VgtkIO.validate(store) == "cameras"
```

The `provenance` keyword is written to the file but not returned by `load`.

## Documentation

- [Stable documentation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/stable/)
  and [development documentation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/dev/):
  the schema rules, every supported kind and format, the conventions a file
  declares, the third-party calibration migrations, and the API reference.
- The stack conventions shared by every package are on VisualGeometryToolkitCore's
  [Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
  page.

## Citing

VisualGeometryToolkitIO is part of VisualGeometryToolkit. Please cite the
toolkit. Its [CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button
exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Please also cite the papers behind the methods you use. They are listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).

## License

MIT
