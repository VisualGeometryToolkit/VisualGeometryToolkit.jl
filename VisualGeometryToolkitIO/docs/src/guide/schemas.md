# Schema Rules

## Why the files look the way they do

One 2026 dataset stored rotations in two different matrix layouts, and neither
was declared. The motion-capture poses were flat and column-major, the
detected poses nested and row-major. Reading the first as row-major corrupted
every frame-dependent computation downstream: the hand-eye self-residuals were
14.6° and 150 mm instead of 0.4° and 3.7 mm. The vgio schemas are designed so
that this class of bug cannot be written down:

- Rotations are stored as quaternions (Hamilton, w-x-y-z, w ≥ 0; a stored
  w < 0 is refused, never flipped). No raw 3×3 matrix appears on disk. A pose
  is read back as an `EuclideanMap` whose rotation is the stored `QuatRotation`
  itself, neither renormalized nor converted to a matrix. So file → value →
  file is byte-identical, and a value read from a file round-trips `==`. A
  rotation of another type (a `RotMatrix` computed in memory) is converted on
  write and reads back as the `QuatRotation` of that conversion.
- Every pose has an explicit direction. Either it names its `from` and `to`
  frames, or its frames follow from where it sits in the file (a sensor under
  its rig, a target under its gauge).
- Every length crosses the file boundary through one decimal-exact pair,
  `read_length(x, u)` on read and `write_length(q, u)` on write, in the unit
  the file declares ([Conventions on Disk](conventions.md)).

## What every vgio file guarantees

Every vgio file has a schema tag `vgio/<kind>/v<major>.<minor>`; the current
version of each kind, and every earlier one it reads, are listed in
[Kinds and the I/O Surface](kinds.md#Reading-an-earlier-version). A file can be
read exactly with nothing but itself. The rules that make this true:

**The envelope is about the file; attributes are about the value.** The
envelope holds what describes the file: the `schema` tag, the `generator`, the
`units` and `conventions` objects ([The envelope](kinds.md#The-envelope)),
the reserved `payload` key, and the free-form `provenance` object.
`provenance` is a `save` keyword and is never read back. None of the envelope
is part of the value. A fact about the value that must round-trip is part of
the VGC value instead: either a field of its type (a trajectory's per-frame
images and times, an observation's sensors) or an `Attributed` attribute (the
`calibration_record` of a camera or a target, or a covariance, stored as
`Uncertain`).

**Conventions are declared once.** Each convention a reader needs is stated
once, in the envelope: the pixel frame and the units (`units`), the rotation
encoding, what a transform does to a point (`x_to = R * x_from + t`), the
matrix and grid layouts and the index base, 1 (`conventions`). No field states
a unit of its own: a field's unit is a power of a declared base, stated by its
schema. A reader refuses a file that leaves a declaration out. Nothing has a
default.

**A lens is VGC's lens record.** Every key is a leaf tag or a combinator, and
every value is a coefficient array whose length is the arity, for example
`{"equidistant": [], "radial": [[k1, k2, k3, k4], []]}`. The file determines
the lens type exactly, and no parameter order lives outside the file.

**Each value is stored in one place.** Frame names are never repeated beside a
pose. Values that can be derived from stored ones (a session's lens
coefficients, centre and counts) are recomputed rather than stored. The
writer refuses data whose derived values disagree with their source.

**Observations name their targets by content.** An observation file
(`observations`, `target_homographies`) names the rigs or single unposed
targets that define the targets it indexes by their content identity,
`content_id(rig)` or `content_id(path)`: the SHA-256 of the RFC 8785 (JCS)
serialization of the record's identity tree, in base units, with no envelope.
[Content Identity](content-identity.md) specifies it completely, so it can be
recomputed outside Julia.

**One validator per kind.** The validator for each kind (`validate.jl`) runs
on read and before every write, so a writer cannot produce a file that its
reader refuses. `provenance` is the only free-form object, and the package
never reads it. `VisualGeometryToolkitIO.validate(path)` runs the validator
on any file, and every kind's table is published as a JSON Schema
([JSON Schemas](json-schemas.md)).

**A calibration record says where a calibration came from.** A stored camera
and a target may carry an optional `calibration_record`, with the keys `tool`,
`source` (a file or record name, never a machine path), `date` (`YYYY-MM-DD`),
`reprojection_rms` (in px), `n_images`, `n_points`, `note` and `device`, the
hardware the calibration was made for: `{"body": {…}, "lens": {…}}`, each a
`DeviceIdentity` (`make`, `model`, `serial`; in memory `(; body, lens)`), the
one device-identity encoding (a capture settings file's `device` is the same
object), since a camera model itself holds no names. Each key is omitted when
it is unknown. In memory the record is the value's attribute,
`Attributed(camera, (; calibration_record = (; tool = …, …)))` and
`Attributed(target, (; calibration_record = …))`, and it is read back exactly.
Unlike `provenance`, its keys are a closed set and are checked on write and
on read. (A target's own `calibration`, its `TargetCalibration`, is part of
its body.)

**Frame names follow one grammar.** The names are `world`, `camera`,
`view:<id>`, `target:<id>` and `body:<id>` (a rigid body, such as a tracked
or mounted rig). They map to VGC's `WorldFrame`, `CameraFrame`,
`ViewFrame{id}`, `TargetFrame{id}` and `BodyFrame{id}`: a name mirrors its
type. A file's frames read back as types, and a free-form name is refused.

**Uncertainty follows the type.** The rule is VGC's
(`conventions/persisted-geometry.md`). A pose or a point carries a
`covariance` if and only if it is `Uncertain`, and it reads back as
`Uncertain`. The covariance is written in the storage its value holds — rows
(full), `{"diagonal": […]}` or `{"isotropic": λ}` — in the value's own
tangent order and the file's units. A pose's covariance is 6×6 over
`[δω; δt]`, rotation first, in the chart `R ↦ exp(δω)·R, t ↦ t + δt`. A plain
value means that no covariance was stated; an exact value has an all-zero
covariance. A detection whose producer has no covariance omits the key and
states why in its `covariance_status` attribute. `ros_pose_covariance` and
`pose_covariance_from_ros` convert to and from the ROS `PoseWithCovariance`
layout, which puts translation first. Kinds whose in-memory type cannot hold
an `Uncertain` value (a rig sensor's `PosedCamera` extrinsics, a
correspondence row) are always plain.

**Sizes are objects, indices are 1-based.** Every size (an image size, a
pixel pitch, a board, canvas or texture size, a lattice's `grid_size`) is
`{"width": w, "height": h}`; a rectangle (`area`) and the positional primitives
(an ellipse, a circle, a quaternion) are documented arrays. Every index in a
file counts from 1 (`"index_base": 1`); a foreign 0-based format converts in
its importer and exporter only.

## Storage

Every kind is JSON. Once a kind's bulk numeric arrays grow past about 10 MB,
the plan is for it to keep its JSON header (schema, conventions, frames,
sensor, provenance) and move the arrays into an Apache Arrow sidecar. The
sidecar is named by the reserved top-level key
`"payload": {"file", "sha256"}`, following glTF's JSON-plus-binary pattern. No
kind implements this yet, and a file that carries `payload` is refused until
one does. (A sensor characterisation's dark maps are artifacts of
its own, named and hashed inside its body.)

## Earlier versions

Every earlier version a published file holds is read, and only the current
one is written — but for one rule: a kind whose new major version only adds a
form may write its earlier version, that version's very bytes, for a record
the earlier version can hold (`vgio/sensor_characterisation/v1.0`, IO 0.10.1's
file, for a characterisation without a `Darks`), so a reader of that version
keeps reading it. The versions a writer emits are one table
(`_written_versions`); each has its published JSON Schema. A schema change must not force the published data (Zenodo,
Hugging Face; archives of several GB) to be uploaded again, so a published
file keeps loading: `load` runs the file's version steps in memory, one step
per version, and reads the current record they give. The versions each kind
reads are derived from those steps and tabled in
[Kinds and the I/O Surface](kinds.md#Reading-an-earlier-version). The old grid
records (`generic_camera/v2`–`v3`) are read, never rewritten: a central one
reads as the camera store of its one camera with a `SplineLens`, a
non-central one as the store of its `NoncentralSplineModel` camera, named
after the file (`fisheye_14mm.json` → `:fisheye_14mm`), and `vgio_migrate`
writes that store to a NEW file beside it (`<stem>.camera.json`,
`<stem>.noncentral.json`), never overwriting one.

A version no published file holds, and that no step reads, is refused by
name. A v1 transforms or trajectory file never had a migration, because its
free-form frame names have no typed successor; `rig/v1`–`v2`,
`generic_camera/v1`, the unreleased `generic_camera/v4` and `blur/v1` have
none either. Re-export such a file from the program that produced it.

These kinds were retired with no reader and no migration; the producer
re-exports:

| Retired | Successor |
|---|---|
| `observations/v2` | `observations/v3.0`, a new kind of the same name (the major starts at 3, so no `observations/v2` file is ever read under the new meaning) |
| `point_observations/v1` | `observations/v3.0` |
| `targets/v2`, `rig/v3` | `rig/v10.0` |
| `session/v3` | the calibration pipeline's own session kind |
| `correspondences/v2` | `export/correspondences/v2`, an export (`save_correspondences`) that no reader reads |

A pipeline's checkpoint belongs to the pipeline, not to core geometry, which
is why `session` left this package. A reader refuses each retired kind and
names its successor.
