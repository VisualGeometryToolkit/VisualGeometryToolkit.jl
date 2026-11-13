# VisualGeometryToolkitIO.jl

*File formats for the VisualGeometryToolkit stack.*

VisualGeometryToolkitIO saves and loads the geometry of a calibration or
reconstruction: cameras, poses, rigs, targets, trajectories and target
observations. Each `vgio/<kind>` file is the JSON serialization of one
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/)
type. `load` returns a value of that type, `save` takes one, and the round
trip is exact. There are no record structs of the package's own between the
file and the value.

## What it provides

- the native `vgio/<kind>` files, read and written with `load`, `save` and
  `validate`;
- every earlier schema version a published file holds, read on load and
  rewritten once by `vgio_migrate`;
- COLMAP model directories, read and written (`.bin` and `.txt`), and COLMAP
  `database.db` files, read lazily;
- OpenCV and MATLAB calibration exports, read and migrated to native files;
- write-only exporters to TUM, EuRoC, BOP, OpenSim TRC, and the flat
  correspondence arrays that external calibrators take.

Three principles shape the files:

- **A file declares its conventions.** The pixel frame, length unit, rotation
  encoding and pose direction are written in the file, derived from the
  value's types, so a file reads correctly without outside knowledge
  ([Conventions on Disk](guide/conventions.md)).
- **Lengths cross the file boundary decimal-exactly.** Memory holds metres, a
  file the unit that reads naturally, converted by a decimal shift, not a
  multiplication ([Conventions on Disk](guide/conventions.md#Lengths-cross-the-file-boundary-decimal-exactly)).
- **A published file keeps loading.** Only the current version is written,
  and every earlier version is read through one step per version
  ([Schema Rules](guide/schemas.md)).

## Installation

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitIO
```

Julia 1.13 or later; every dependency is in the General registry.
VisualGeometryToolkitCore defines the types that are saved and loaded.

## A first example

The example saves a pose and a camera and loads them back (the README's
example; the test suite runs both). `load` and `save` are public but not
exported, because other packages (FileIO among them) export functions with
the same names, so they are called through a module alias.

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

## Documentation

**Start here.** [Getting Started](guide/getting-started.md) saves and loads a
pose and a camera, and reads the files.

**How to.** Find the kind and the calls for a value
([Kinds and the I/O Surface](guide/kinds.md)), including reading an earlier
schema version; convert OpenCV and MATLAB exports
([Third-Party Calibrations](guide/third-party.md)); export to the community
formats ([Related Formats and Exporters](guide/related-formats.md)); run the
tests ([Testing and Verification](guide/verification.md)).

**Explanation.** [Schema Rules](guide/schemas.md) says why the files look the
way they do and what every vgio file guarantees;
[Conventions on Disk](guide/conventions.md),
[Quantities, Frames and Detections](guide/serialization.md) and
[Layers and Design](guide/design.md) explain the design.

**Reference.** [Native Kinds](api/native.md), [COLMAP](api/colmap.md),
[Third-Party Calibrations](api/third-party.md) and [Exporters](api/exporters.md)
document every public name.

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitIO is built on VisualGeometryToolkitCore, which defines
the types it reads and writes and the conventions every stack package follows
(its
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
page). VisualGeometryToolkitEval reads its datasets' COLMAP reconstructions
with it, and a package with a kind of its own (BlobBoards' boards) builds on its
building blocks and its schema migration. All toolkit documentation is
aggregated at the portal: <https://visualgeometrytoolkit.github.io>.

## Citing

VisualGeometryToolkitIO is part of VisualGeometryToolkit. Cite the toolkit;
its [CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
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

Cite also the papers behind the methods you use, listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).
