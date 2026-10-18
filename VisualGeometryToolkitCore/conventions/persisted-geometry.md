## Persisted geometry: uncertainty and storage

How geometry is written to and read from files (VisualGeometryToolkitIo's `vgio/<kind>/v<n>`
JSON kinds and any other stack format). It continues the measurement-noise rule above:
the same three cases, now on disk.

### Uncertainty mirrors the type

Every geometry-carrying record, whether a primitive (a 2D or 3D point) or a pose (a
rigid transform), stores uncertainty exactly as the in-memory type states it:

| In memory | On disk | Read back as |
|---|---|---|
| plain `T` | no `covariance` field | plain `T` |
| [`Uncertain`](@ref)`{T}` (`Attributed{T, CovMat}`) | `covariance` present | `Uncertain{T}` with that `CovMat` |
| [`exact`](@ref)`(p)` (zero covariance) | `covariance` all zeros | `Uncertain{T}` with a zero `CovMat` (= `exact(p)`) |

- A writer emits `covariance` if and only if the value is `Uncertain`.
- A reader returns the matching type, so a value round-trips exactly, type included.
- A precision-form `Uncertain` (`PrecMat`) is written as its covariance and read back
  in covariance form.
- **Plain means "no covariance stated", not "unit noise".** The consumer applies its
  own documented default. For an image point that default is the measurement-noise
  rule's `Σ = I`; other consumers document theirs.
- `exact` means zero, and it is written as zeros.
- **An estimator states its uncertainty.** A file written from an estimate (a hand-eye
  calibration, a fitted pose) carries `Uncertain` values. A writer that has a
  covariance and drops it is a bug.

### One encoding

The `covariance` is a `d×d` array of rows (`d` arrays of `d` numbers), symmetric, in
the primitive's own tangent order and units:

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
  - VisualGeometryToolkitIo's `ros_pose_covariance` and `pose_covariance_from_ros` implement
    it.

A reader refuses a `covariance` of the wrong size, a non-finite one, or an asymmetric
one. It does not test positive semi-definiteness (an `exact` zero is valid).

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

| Kind (VisualGeometryToolkitIo) | VGC value |
|---|---|
| `cameras` | `Dict{Symbol, …}` of unposed `BoundedCamera`s of a `CameraModel` (name → camera) |
| `transforms` | a `Vector` of labeled `EuclideanMap{3}`s (`Uncertain` allowed) |
| `rig` | [`Rig`](@ref): sensors, [`Mounted`](@ref) targets, or both |
| `target` | an [`AbstractTarget`](@ref) ([`PointTarget`](@ref), [`PlanarTarget`](@ref)) |
| `trajectory` | [`Trajectory`](@ref) |
| `target_poses` | [`TargetPoses`](@ref) |
| `target_points2d`, `target_points3d` | [`TargetPoints2D`](@ref), [`TargetPoints3D`](@ref) |

The measurement kinds are COLMAP's decomposition: `target_points2d` ≈ COLMAP's
`images.txt` `points2D` with their `POINT3D_ID` — here the pair (target id,
point index) — and the target rig ≈ `points3D`, kept as per-target frames (each
target's points in its own frame, posed by its mount) with no stored tracks.

**Metadata: envelope for the file, attributes for the value.** The file's
envelope holds what is about the FILE — the `schema` tag, the declared
conventions (pixel frame, length unit, rotation encoding, transform convention,
point-index base), the reserved `payload`, and the free-form `provenance`,
which is written from a keyword and never read back — and none of it is part of
the value. A fact about the VALUE that must round-trip is part of the VGC value:
a field (a trajectory's per-frame images and times, a measurement's target
identities) or an `Attributed` attribute — a covariance as `Uncertain`, a
camera's or a target's `calibration` block as
`Attributed(value, (; calibration = (; tool, source, date, reprojection_rms, n_images, n_points, note)))`
(the stated keys only; a closed set).

**A referenced design is stored by its reference.** A [`PlanarTarget`](@ref)
whose design is content-addressed (a BlobBoards board uid) is written by its
`design` reference and its warp; its nominal points are the design's owner's
and are not restated. A reader returns it unresolved unless the caller supplies
the owner's resolver.

**Persisted rotations are quaternions.** A rotation is stored as a Hamilton
quaternion (w-x-y-z, w ≥ 0) and READ AS ONE: the reader builds a
`EuclideanMap{3, Float64, QuatRotation{Float64}, …}` from the stored unit
quaternion without renormalizing it, so file → value → file is exact and a value
read from a file round-trips bit for bit. The writer accepts any `Rotation`: a
`QuatRotation` is written as it is (its sign made w ≥ 0 by exact negation, the
same rotation); any other — a `RotMatrix` an estimator computed — is converted
to a quaternion on write and reads back as a `QuatRotation`-valued map (equal to
the stored matrix to the conversion's rounding). Code that consumes a persisted
pose therefore takes any `Rotation`, never a `RotMatrix` specifically; one that
needs a matrix converts explicitly.

