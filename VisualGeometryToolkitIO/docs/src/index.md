# VisualGeometryToolkitIO.jl

*File formats for the VisualGeometryToolkit stack.*

VisualGeometryToolkitIO saves and loads the geometry of a calibration or
reconstruction: cameras, poses, rigs, targets, trajectories and target
measurements. Each `vgio/<kind>` file is the JSON serialization of one
[VisualGeometryToolkitCore](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/)
type. `load` returns a value of that type, `save` takes one, and the round
trip is exact. There are no record structs of the package's own between the
file and the value.

Beside the native kinds, the package has adapters for foreign formats:

- COLMAP model directories, read and written (`.bin` and `.txt`);
- COLMAP `database.db` files, read lazily;
- OpenCV and MATLAB calibration exports, read and migrated to native files;
- write-only exporters to TUM, EuRoC, BOP, OpenSim TRC, and the flat
  correspondence arrays that external calibrators take.

Magnitudes cross the file boundary through a unit contract built on
FlexUnits.

## Installation

```julia
using Pkg
Pkg.add(["VisualGeometryToolkitCore", "VisualGeometryToolkitIO"])
```

VisualGeometryToolkitIO requires Julia 1.12 or later. It is part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack and is
built on VisualGeometryToolkitCore, which defines the types it reads and
writes.

## Where to go next

- [Getting Started](guide/getting-started.md): save and load a pose and a
  camera.
- [Schema Rules](guide/schemas.md): why the files look the way they do, and
  what every vgio file guarantees.
- [Kinds and the I/O Surface](guide/kinds.md): every kind, with the type it
  holds and the calls that read and write it.
- [Conventions on Disk](guide/conventions.md): how a file declares its pixel
  frame, length unit and pose direction.
- [Third-Party Calibrations](guide/third-party.md): converting OpenCV and
  MATLAB exports.
- [Related Formats and Exporters](guide/related-formats.md): how the kinds
  compare with ROS, Kalibr, TUM, EuRoC, BOP, TRC and COLMAP.
- [Layers and Design](guide/design.md): how the package is organized.
- [Testing and Verification](guide/verification.md): what the test suite
  checks and how to run it.
- API Reference: [Native Kinds](api/native.md), [COLMAP](api/colmap.md),
  [Third-Party Calibrations](api/third-party.md), [Exporters](api/exporters.md).

The conventions every stack package follows (estimator names, argument order,
coordinate frames) are on VisualGeometryToolkitCore's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
page.

## Citing

VisualGeometryToolkitIO is part of VisualGeometryToolkit. Please cite the
toolkit. Its [CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry:

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
