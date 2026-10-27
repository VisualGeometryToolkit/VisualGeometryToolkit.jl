# Schema Rules

## Why the files look the way they do

One 2026 dataset stored rotations in two different matrix layouts, and neither
was declared. The motion-capture poses were flat and column-major, the
detected poses nested and row-major. Reading the first as row-major corrupted
every frame-dependent computation downstream: the hand-eye self-residuals were
14.6° and 150 mm instead of 0.4° and 3.7 mm. The vgio schemas are designed so
that this class of bug cannot be written down:

- Rotations are stored as quaternions (Hamilton, w-x-y-z, w ≥ 0). No raw 3×3
  matrix appears on disk. A pose is read back as an `EuclideanMap` whose
  rotation is the stored `QuatRotation` itself, neither renormalized nor
  converted to a matrix. So file → value → file is byte-identical, and a value
  read from a file round-trips `==`. A rotation of another type (a
  `RotMatrix` computed in memory) is converted on write and reads back as the
  `QuatRotation` of that conversion.
- Every pose has an explicit direction. Either it names its `from` and `to`
  frames, or its frames follow from where it sits in the file (a sensor under
  its rig, a target under its gauge).
- Magnitudes cross the file boundary only through FlexUnits' dimension-checked
  `quantity(x, u)` on read and `ustrip(u, q)` on write. The package never
  calls the single-argument `ustrip`.

## What every vgio file guarantees

Every vgio file has a schema tag `vgio/<kind>/v<n>`. The current versions are
rig v4; cameras, transforms and trajectory v3; target, target_poses,
target_points2d and target_points3d v1; and v2 for the rest. A file can be
read exactly with nothing but itself. The rules that make this true:

**The envelope is about the file; attributes are about the value.** The
envelope holds what describes the file: the `schema` tag, the declared
conventions, the reserved `payload` key, and the free-form `provenance`
object. `provenance` is a `save` keyword and is never read back. None of the
envelope is part of the value. A fact about the value that must round-trip is
part of the VGC value instead: either a field of its type (a trajectory's
per-frame images and times, a measurement's sensors) or an `Attributed`
attribute (the `calibration` block of a camera or a target, or a covariance,
stored as `Uncertain`).

**Conventions are declared once.** Each convention a reader needs is stated in
the file: the pixel frame, the length unit, the rotation encoding, what a
transform does to a point (`x_to = R * x_from + t`), any matrix or grid
layout, and any index base. A reader refuses a file that leaves one out.
Nothing has a default.

**A lens is VGC's lens record.** Every key is a leaf tag or a combinator, and
every value is a coefficient array whose length is the arity, for example
`{"equidistant": [], "radial": [[k1, k2, k3, k4], []]}`. The file determines
the lens type exactly, and no parameter order lives outside the file.

**Each value is stored in one place.** Frame names are never repeated beside a
pose. Values that can be derived from stored ones (a session's lens
coefficients, centre and counts) are recomputed rather than stored. The
writer refuses data whose derived values disagree with their source.

**Measurements name their targets by content.** A measurement file
(`target_points2d`, `target_points3d`) names the rigs that define the targets
it indexes by their content identity, `content_id(rig)` or
`content_id(path)`. This is the SHA-256 of the rig's identity tree: built from
the in-memory rig under names of the identity's own, fixed once in
`content_id`'s docstring and independent of any record's key spellings, every
length in metres, the targets sorted by id, with no schema version, no
declaration and no provenance. The identity does not change when the file
format does: a rig, the file written from it in any decimal `length_unit`,
the rig loaded back from that file and a migration of the file to the next rig
version have one identity. A target's calibration kind, its parameters, its
covariance and its board are part of it.

The identity is of the persisted representation. A rotation enters as the
quaternion the writer stores, so a rig held with matrix rotations, its file
and the rig loaded from it share one identity; but a rotation converted
matrix → quaternion → matrix is not bit-exact, so two in-memory rigs built
from the same matrix by different routes can differ in the last ulp — mint an
identity from what you save. A decimal length unit (mm, cm, m, μm) is
decimal-exact; a non-decimal one (inch) is exact only for numbers born in it,
so an inch file of millimetre-born numbers reads back within an ulp, and its
identity differs from the value's. A test pins it for a rig of every target kind;
a change to the tree re-ids every rig and must migrate the measurement files
that name them.

**One validator per kind.** The validator for each kind (`validate.jl`) runs
on read and before every write, so a writer cannot produce a file that its
reader refuses. `provenance` is the only free-form object, and the package
never reads it. `VisualGeometryToolkitIO.validate(path)` runs the validator
on any file.

**A calibration block says where a calibration came from.** A stored camera
may carry an optional `calibration` block, and a target a
`calibration_record`, with the keys `tool`, `source` (a file or record name,
never a machine path), `date` (`YYYY-MM-DD`), `reprojection_rms` (in px),
`n_images`, `n_points` and `note`. Each key is omitted when it is unknown. In
memory the block is the value's attribute,
`Attributed(camera, (; calibration = (; tool = …, …)))` and
`Attributed(target, (; calibration_record = …))`, and it is read back exactly.
Unlike `provenance`, its keys are a closed set and are checked on write and
on read. (A target's own `calibration`, its `TargetCalibration`, is part of
its body.)

**Frame names follow one grammar.** The names are `world`, `camera_optical`,
`view:<id>`, `target:<id>` and `body:<id>` (a rigid body, such as a tracked
or mounted rig). They map to VGC's `WorldFrame`, `CameraFrame`,
`ViewFrame{id}`, `TargetFrame{id}` and `BodyFrame{id}`. A file's frames read
back as types, and a free-form name is refused.

**Uncertainty follows the type.** The rule is VGC's
(`conventions/persisted-geometry.md`). A pose or a point carries a
`covariance` if and only if it is `Uncertain`, and it reads back as
`Uncertain`. The covariance is an array of rows, in the value's own tangent
order and the file's units. A pose's covariance is 6×6 over `[δω; δt]`,
rotation first, in the chart `R ↦ exp(δω)·R, t ↦ t + δt`. A plain value means
that no covariance was stated; an exact value has an all-zero covariance.
`ros_pose_covariance` and `pose_covariance_from_ros` convert to and from the
ROS `PoseWithCovariance` layout, which puts translation first. Kinds whose
in-memory type cannot hold an `Uncertain` value (a rig sensor's `PosedCamera`
extrinsics, a correspondence row) are always plain.

## Storage

Every kind is JSON. Once a kind's bulk numeric arrays grow past about 10 MB,
the plan is for it to keep its JSON header (schema, conventions, frames,
sensor, provenance) and move the arrays into an Apache Arrow sidecar. The
sidecar is named by the reserved top-level key
`"payload": {"file", "sha256"}`, following glTF's JSON-plus-binary pattern. No
kind implements this yet, and a file that carries `payload` is refused until
one does.

## Retired versions

Earlier versions are retired, and nothing reads them: `load` refuses each by
name. Every stored record was migrated once, in 2026-10. The v1 files went
through a v1 migration. The kinds that carry lenses (cameras and rigs), which
moved to VGC's lens families at v3, went through a separate lens migration
that proved each lens computed the same function. Both tools are in the git
history. A v1 or v2 transforms or trajectory file never had a migration,
because its free-form frame names have no typed successor. Re-export such a
file from the program that produced it.

The direct-serialization redesign (2026-10) also retired these kinds, with no
reader and no migration. Each successor starts at v1 under its new name, and
the producer re-exports:

| Retired | Successor |
|---|---|
| `observations/v2` | `target_points2d/v1` |
| `point_observations/v1` | `target_points3d/v1` |
| `targets/v2`, `rig/v3` | `rig/v5` |
| `session/v3` | the calibration pipeline's own session kind |

A pipeline's checkpoint belongs to the pipeline, not to core geometry, which
is why `session` left this package. A reader refuses each retired kind and
names its successor.

`target/v1` and `rig/v4` (2026-10-26, the target calibrations) are migrated
rather than retired: a reader refuses them naming `vgio_migrate`, which
converts a file once (see [The Kinds](kinds.md#Migrating-an-earlier-version)).
