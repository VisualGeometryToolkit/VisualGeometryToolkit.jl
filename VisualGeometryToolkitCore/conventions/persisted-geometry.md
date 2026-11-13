## Persisted geometry: uncertainty and storage

How geometry is written to and read from files (VisualGeometryToolkitIO's
`vgio/<kind>/v<major>.<minor>` JSON kinds and any other stack format). It continues the observation-noise rule above:
the same three cases, now on disk.

### Uncertainty mirrors the type

Every geometry-carrying record, whether a primitive (a 2D or 3D point) or a pose (a
rigid transform), stores uncertainty exactly as the in-memory type states it:

| In memory | On disk | Read back as |
|---|---|---|
| plain `T` | no `covariance` field | plain `T` |
| [`Uncertain`](@ref)`{T}` (`Attributed{T, CovMat}`) | `covariance` present | `Uncertain{T}` with that `CovMat` |
| [`exact`](@ref)`(p)` (zero covariance) | `covariance` all zeros | `Uncertain{T}` with a zero `CovMat` (= `exact(p)`) |

- A writer emits `covariance` if and only if the value is `Uncertain`. The key is
  `covariance` in every kind and every body, in one of the three storages below; no
  kind spells it otherwise (a flat vector, a `cov` with a separate size).
- **An absent covariance is stated by its producer.** A record whose producer has no
  finite covariance omits the key and states a per-record `covariance_status` (for
  example `"unrefined"`) with its reason. A `null` covariance is refused, and a reader
  never substitutes a default for an omitted one (the producer owns its observation
  policy).
- A reader returns the matching type, so a value round-trips exactly, type included.
- A precision-form `Uncertain` (`PrecMat`) is written as its covariance and read back
  in covariance form.
- **Plain means "no covariance stated", not "unit noise".** The consumer applies its
  own documented default. For an image point that default is the observation-noise
  rule's `Σ = I`; other consumers document theirs.
- `exact` means zero, and it is written as zeros.
- **An estimator states its uncertainty.** A file written from an estimate (a hand-eye
  calibration, a fitted pose) carries `Uncertain` values. A writer that has a
  covariance and drops it is a bug.

### One encoding per storage

The `covariance` states the storage the value's [`CovMat`](@ref) holds, so it
round-trips as that storage:

| In memory (`parent(Σ)`) | On disk |
|---|---|
| `SMatrix` (full) | a `d×d` array of rows (`d` arrays of `d` numbers), symmetric |
| `SDiagonal` (diagonal) | `{"diagonal": [d numbers]}` |
| [`ScaledIdentity`](@ref) (isotropic, `λ·I`) | `{"isotropic": λ}` |

An isotropic covariance over entries of different length dimensions (a pose's
angles and lengths) has no single number in the file's units, so it is written,
and read back, as its diagonal. The storage is told from the JSON's shape, never
from the numbers. Every form is in the primitive's own tangent order and units:

- **Points:** `(x, y)` or `(x, y, z)`, in the file's unit squared. That is px² in a
  file that declares a `pixel_frame`, and the declared `length_unit` squared
  otherwise.
- **Poses:** 6×6 over `[δω; δt]`, rotation first: rad² for the rotation block, the
  file's `length_unit` squared for the translation block, and their product for the
  cross block.
  - The chart is VisualGeometryToolkitCore's `RigidExpManifold6`: `R ↦ exp(δω)·R` and
    `t ↦ t + δt`. The rotation is perturbed on the LEFT, in the pose's destination
    (`to`) frame, and the translation additively in the same frame. This is the chart
    VisualGeometryToolkitEstimation reports estimator covariances in.
  - **Relation to GTSAM.** The order `[ω; v]` is GTSAM's Pose3 tangent order
    (`Pose3::Expmap`: `w = xi.head<3>(), v = xi.tail<3>()`).
  - GTSAM's chart is different. It is the RIGHT (body-frame) retraction
    `T·Exp(ξ)`, so a GTSAM marginal needs the adjoint, not a relabel.
  - **Relation to Sophus.** Sophus' SE3 tangent is `[υ; ω]`, translation first.
- **ROS `PoseWithCovariance`.** It is row-major 6×6 over `(x, y, z, rot X, rot Y,
  rot Z)`, with fixed-axis rotations in the parent frame and the translation in the
  parent frame. That is this chart with its blocks swapped.
  - The conversion is the permutation `P = [0 I₃; I₃ 0]`: `Σ_ros = P·Σ·Pᵀ`.
  - VisualGeometryToolkitIO's `ros_pose_covariance` and `pose_covariance_from_ros` implement
    it.

**Units: memory in base units, files in the declared unit.** In memory a
covariance is in base units, like the value it belongs to: m² for a 3D point,
and for a pose rad² in the rotation block, m² in the translation block and
m·rad in the cross block. On disk it is in the file's declared `length_unit`.
A writer converts each entry with VisualGeometryToolkitIO's `write_length`
rule, the same decimal shift as the value, raised to the entry's power of
length, and a reader converts back the same way. A file in mm therefore holds
mm² and mm·rad entries, and a covariance read from it is in m² and m·rad again.
Pixel covariances are px² and are never converted.

A reader refuses a `covariance` of the wrong size, a non-finite one, or an asymmetric
one. It does not test positive semi-definiteness (an `exact` zero is valid). Every
matrix a file holds (a covariance, a shape matrix, an affine shape factor) is written
in rows, the stack's `MATRIX_LAYOUT`.

**An uncertain feature frame's covariance is in the frame's parameter order, the
centre first.** A detection's [`FeatureFrame`](@ref) is written as its `center` and
its shape parameters ([`frame_dof`](@ref VisualGeometryToolkitCore.frame_dof) of them), and its covariance is over exactly
those parameters, `(x, y)` first: a `PointFrame`'s is the uncertain point's 2×2, so the
point frame is the 2-DOF case of the one frame codec, never a codec of its own. The
point observation of a detection is its first 2×2 block
([`center`](@ref)`(::Uncertain{<:FeatureFrame})`), read by that function only.

### One units object, one conventions object

Every file states its units ONCE, in one file-level `units` object, and its
conventions once, in one `conventions` object; neither has a default, and a file
without either, or without one of their keys, is refused:

```json
"units": {"length": "mm", "time": "s", "angle": "rad", "pixel": "colmap",
          "density": "dpi", "radiance": "relative", "length_roles": {…}},
"conventions": {"index_base": 1, "matrix_layout": "rows", "rotation_encoding": "quaternion_wxyz", …}
```

- **A field's unit is a power of a declared base, stated by the schema, never by the
  file.** A blur covariance is `length²` on a print and `px²` on a sensor; a focus
  surface is `length⁻¹`. No block carries its own `"unit"` string.
- **Time** values are in `units.time`; a `time_origin` is ISO 8601 with its UTC
  offset.
- **Angles** are radians, declared as the fixed token `"rad"`.
- **Sizes are objects.** Every `Size2` (an image size, a board, canvas or texture
  size, a pixel pitch, a lattice's `grid_size`, the identity tree's sizes) is
  `{"width": w, "height": h}`, never a positional pair. A rectangle (`area`) and the
  positional primitives (an ellipse, a circle) stay documented arrays, described field
  by field in the kind's JSON Schema.
- **Every index is 1-based**, Julia's, declared once as `"index_base": 1` in the
  conventions object: a point index, a lattice's `present` runs, a frame index. Only
  a foreign format's importer and exporter (Schöps, COLMAP, OpenCV, all 0-based)
  convert, at their own boundary; a stack reader and writer hold no ±1.

### Versions: a major and a minor

A schema tag is `vgio/<kind>/v<major>.<minor>`. A reader accepts any minor of its
major: a minor version adds optional keys its schema declares, and an unknown,
undeclared key stays refused. A major version changes a meaning. The versions a
reader accepts are derived from its migration steps, never kept as a list by hand.

### Storage: JSON now, a binary payload past a threshold

Every kind is a single JSON file today. A kind keeps that form while its bulk numeric
arrays stay small.

Once a kind's arrays exceed about **10 MB**, it follows glTF's JSON + bin pattern:

- The JSON header stays. It holds the schema tag, the declared conventions, the frames,
  the sensor and the `provenance`.
- The bulk arrays move to an Apache Arrow sidecar. The header references the sidecar
  under the reserved top-level key `"payload"`:
  `{"file": "<name>.arrow", "sha256": "<hex>"}`.

Every kind reserves `"payload"` now, so a sidecar can be added later without a schema
break. No kind implements it yet. Until one does, a reader refuses a file that carries
`"payload"`.

### Every stored kind is a VGC type

A persisted kind is the DIRECT serialization of a VisualGeometryToolkitCore type: the
reader returns the VGC value, the writer takes it, and the round trip is exact
by type (plain ↔ plain, `Uncertain` ↔ `Uncertain`, labels ↔ frame names). No
format-side record type stands between the file and the value; a foreign format
(COLMAP, an OpenCV or MATLAB export) keeps an adapter type that converts to and
from VGC types.

| Kind (VisualGeometryToolkitIO) | VGC value |
|---|---|
| `cameras` | `Dict{Symbol, …}` of unposed `BoundedCamera`s of a `CameraModel` (name → camera) |
| `transforms` | a `Vector` of labeled `EuclideanMap{3}`s (`Uncertain` allowed) |
| `rig` | [`Rig`](@ref): sensors, [`PosedTarget`](@ref)s, or both |
| `target` | an [`AbstractTarget`](@ref) ([`PointTarget`](@ref), [`PlanarTarget`](@ref)) |
| `trajectory` | [`Trajectory`](@ref) |
| `target_poses` | [`TargetPoses`](@ref) |
| `observations` | [`TargetObservations`](@ref) (pixels or 3D points, by its frame; each set names its entity, `target_point` or `landmark`). The 0.9 `target_detections`, `target_points2d` and `target_points3d` kinds are read through version steps into it |
| `detection` | a detection, `Attributed{Uncertain{<:FeatureFrame}}` (the one frame codec, every frame type) |
| `sensor_noise` | [`SensorNoise`](@ref), every field with its source (`measured`, `estimated`, `file`) |
| `capture_settings` | [`CaptureSettings`](@ref), `Attributed` with its identities' sources |

**File names state the kind, never the time.** A file's name is a
convention for people, not a format: every loader reads the file's `schema`
and never its name. VisualGeometryToolkitIO's docs ("Kinds and the I/O
Surface", *File names*) state it once.

The observation kind is COLMAP's decomposition: an observation ≈ COLMAP's
`images.txt` `points2D` with their `POINT3D_ID` — here the pair (target id,
point index) — and the target rig ≈ `points3D`, kept as per-target frames (each
target's points in its own frame, placed by its pose) with no stored tracks.

**Metadata: envelope for the file, attributes for the value.** The file's
envelope holds what is about the FILE — the `schema` tag, the declared
conventions (the `units` and `conventions` objects), the `generator` (the writing
package and its version), the reserved `payload`, and the free-form `provenance`,
which is written from a keyword and never read back — and none of it is part of
the value. A fact about the VALUE that must round-trip is part of the VGC value:
a field (a trajectory's per-frame images and times, an observation's target
identities) or an `Attributed` attribute — a covariance as `Uncertain`, a
camera's `calibration` block as
`Attributed(camera, (; calibration = (; tool, source, date, reprojection_rms, n_images, n_points, note, device)))`
(`device = (; body, lens)`, the [`DeviceIdentity`](@ref)s of the hardware it
was made for: values are math, identity is an attribute, so a camera model
holds no names),
a target's as `Attributed(target, (; calibration_record = (; …)))` (the same
closed set of keys; the target's own `calibration` is its
[`TargetCalibration`](@ref), part of the value, so the record is named apart).

**A target's calibration is part of its record.** A [`PlanarTarget`](@ref)'s
[`TargetCalibration`](@ref) is written in its body as one `calibration` record
(`vgio/target/v4`, `vgio/rig/v7`): its `type`, the kind's tag in
`TARGET_CALIBRATION_TAGS`; its `parameters` in `parameter_names` order, lengths
in the file's unit; `nodes` for a thin-plate kind; for a composition
(the affine print error `Aspect ∘ Skew`, `HeightField ∘ Aspect ∘ Skew`,
`HeightField ∘ ThinPlateSpline`, …), written by its
structure, `{"type": "composition", "compose": [outer, …, inner]}`, the members in
CoordinateTransformations' order, the outer map first;
and a `covariance` over every parameter, in that order, iff it is `Uncertain`.
A lens record lists its stages in sequence under the same key, `compose`, in
the same order. **A list whose order changes changes its key**: the retired
`calibration_chain` / `members` and lens `chain` listed the same members in
application order, and a reader refuses them by name in a current record
rather than read them in the new order (a file of the version that held them
is read through that version's migration). A target with a
`board` (the physical board it is printed on) writes it as
`{"size": {"width", "height"}, "origin": [u, v]}`, lengths in the file's unit.

**A referenced design is stored by its reference.** A [`PlanarTarget`](@ref)
whose design is content-addressed (a BlobBoards board uid) is written by its
`reference` and its calibration; its nominal points are the design's owner's
and are not restated. A reader returns it unresolved unless the caller supplies
the owner's resolver.

**A parametric design is stored by its layout.** A [`PlanarTarget`](@ref) built
from a layout (`PlanarTarget(layout; …)`, an [`AbstractTargetLayout`](@ref):
[`AprilGrid`](@ref), [`SchoepsPattern`](@ref)) records it ([`layout`](@ref)`(t)`)
and is written by it (`vgio/target/v4`, `vgio/rig/v7`): a `layout` record whose
`type` is the layout's tag in `TARGET_LAYOUT_TAGS`, its whole design (tag
family, ids) and its lengths in the file's unit; its points follow from the
layout and are not restated. Its identity is of its points, as a target given
by them: the layout is not hashed.

**An identity does not change when the file format does.** An observation names
the rig that defines its targets by `content_id`: the SHA-256 of a canonical
tree built from the in-memory rig, serialized by RFC 8785 (JCS, its number
formatting), under the current names, specified once in VisualGeometryToolkitIO's
docs, in base units (each length normalised to the Float64 its shortest
round-tripping decimal in METRES reads back to), its targets in a fixed
order, with no schema version and no envelope provenance. A rig, every file
written from it in any unit, and a migration of that file to the next record
version have one identity; a test pins it for a rig of every target kind.

**A published file keeps loading.** Every earlier schema version a published
file holds is READ, and only the current version is written: a reader that
meets an older version migrates it in memory, version by version, to the
current record (VisualGeometryToolkitIO's `migrate_schema` steps, run by
`migrate_record`; a schema's owner adds the steps of its own kinds), silently.
The reason is the published data: a schema change must never force an archive
on Zenodo or Hugging Face to be uploaded again. Rewriting a file once
(`vgio_migrate`) is optional, and runs the same steps. A version is refused
only where no published file holds it, or where it lacks what the current
record needs (transforms v1/v2's free-form frame names).

**Persisted rotations are quaternions.** A rotation is stored as a Hamilton
quaternion (w-x-y-z, w ≥ 0; a stored `w < 0` is refused on read, never flipped) and
READ AS ONE: the reader builds a
`EuclideanMap{3, Float64, QuatRotation{Float64}, …}` from the stored unit
quaternion without renormalizing it, so file → value → file is exact and a value
read from a file round-trips bit for bit. The writer accepts any `Rotation`: a
`QuatRotation` is written as it is (its sign made w ≥ 0 by exact negation, the
same rotation); any other — a `RotMatrix` an estimator computed — is converted
to a quaternion on write and reads back as a `QuatRotation`-valued map (equal to
the stored matrix to the conversion's rounding). Code that consumes a persisted
pose therefore takes any `Rotation`, never a `RotMatrix` specifically; one that
needs a matrix converts explicitly.

