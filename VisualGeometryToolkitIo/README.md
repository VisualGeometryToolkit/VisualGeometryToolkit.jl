# VisualGeometryToolkitIo

Serialization for the VisualGeometry stack: every `vgio/<kind>` file is the
DIRECT serialization of a VisualGeometryToolkitCore type — `load` returns the VGC
value, `save` takes it, and the round trip is exact by type, with no IO-side
record structs — plus adapters for the foreign formats (COLMAP, OpenCV/MATLAB
calibration exports) and write-only exporters (TUM, EuRoC, BOP, TRC, the
correspondences shape external calibrators take), with a **unit contract**
boundary built on FlexUnits.

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitIo")
```

Requires Julia 1.12. Part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack; built on
[VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore).

## Why

One 2026 dataset stored rotations in two different undeclared matrix layouts
(flat column-major mocap, nested row-major detected poses); a row-major
misread of the first corrupted every frame-dependent computation downstream
(hand-eye self-residuals 14.6°/150 mm vs 0.4°/3.7 mm). This package makes
that bug class unrepresentable:

- rotations serialize as **quaternions** (Hamilton, w-x-y-z, w ≥ 0) — no raw
  3×3 on disk — and are READ as quaternions: a pose comes back as an
  `EuclideanMap` whose rotation is the stored `QuatRotation` itself (not
  renormalized, not converted to a matrix), so file → value → file is
  byte-identical and a value read from a file round-trips `==`. A rotation of
  another type (a `RotMatrix` computed in memory) is converted on write and
  reads back as the `QuatRotation` of that conversion;
- every pose has an explicit direction: named `from`/`to` frames, or frames
  given by where it sits (a sensor under its rig, a target under its gauge);
- magnitudes cross the file boundary only through FlexUnits' dimension-checked
  `quantity(x, u)` (read) and `ustrip(u, q)` (write); a single-argument
  `ustrip` is never used in this codebase.

## Schemas: sufficient and minimal

Every vgio file is `vgio/<kind>/v<n>` — rig v4; cameras, transforms and
trajectory v3; target, target_poses, target_points2d and target_points3d v1;
the rest v2 — and a file can be read EXACTLY with nothing but itself:

- **one rule for metadata: envelope for the FILE, attributes for the VALUE.**
  The envelope holds what is about the file — the `schema` tag, the declared
  conventions, the reserved `payload`, the free-form `provenance` (a `save`
  keyword, never read back) — and none of it is part of the value. A fact
  about the value that must round-trip is part of the VGC value: a field of
  its type (a trajectory's per-frame images and times, a measurement's
  sensors) or an `Attributed` attribute (a camera's or a target's
  `calibration` block; a covariance, as `Uncertain`);

- every convention a reader needs is **declared once** in the file — the pixel
  frame, the length unit, the rotation encoding, what a transform does to a
  point (`x_to = R * x_from + t`), a matrix or grid layout, an index base — and
  a reader refuses a file missing any of them; nothing defaults;
- a lens is **VGC's lens record** — every key a leaf tag or a combinator, every
  value a coefficient array whose length is the arity
  (`{"equidistant": [], "radial": [[k1, k2, k3, k4], []]}`) — so the file
  determines the lens type exactly and no parameter order lives outside it;
- every value has **one authority**: frame names are never restated beside a
  pose, and values derivable from stored ones (a session's lens coefficients,
  centre and counts) are recomputed, not stored — the writer refuses data
  whose derived values disagree with their authority;
- a measurement file (`target_points2d`, `target_points3d`) names the rigs
  defining the targets it indexes by their **content identity**,
  `content_id(rig | path)` — the SHA-256 of the canonical rig record without
  its provenance, the same for a rig and the file written from it;
- one **validator** per kind (`validate.jl`) runs on read AND before every
  write, so a writer cannot emit a file its reader refuses; `provenance` is the
  one free-form object, and VGIo never reads it. `VisualGeometryToolkitIo.validate(path)`
  runs it on any file;
- a stored camera or target may say where its calibration came from: an
  optional `calibration` block — `tool`, `source` (a file or record name, never
  a machine path), `date` (`YYYY-MM-DD`), `reprojection_rms` (px), `n_images`,
  `n_points`, `note`, each omitted when unknown — the value's attribute,
  `Attributed(camera, (; calibration = (; tool = …, …)))`, read back exactly.
  Unlike `provenance` its keys are closed and checked, on write and on read;
- every **frame name** goes through one grammar — `world`, `camera_optical`,
  `view:<id>`, `target:<id>`, `body:<id>` (a rigid body: a tracked or mounted
  rig) ↔ VGC's `WorldFrame`, `CameraFrame`, `ViewFrame{id}`, `TargetFrame{id}`,
  `BodyFrame{id}` — so a file's frames read back as types and a free-form name
  is refused;
- **uncertainty mirrors the type** (VGC `conventions/persisted-geometry.md`):
  a pose or a point carries a `covariance` (an array of rows, in its own
  tangent order and the file's units; a pose's is 6×6 over `[δω; δt]`,
  rotation first, in the chart `R ↦ exp(δω)·R, t ↦ t + δt`) if and only if it
  is `Uncertain`, and reads back `Uncertain`; plain means no covariance
  stated; `exact` is all zeros. `ros_pose_covariance` /
  `pose_covariance_from_ros` convert to and from ROS `PoseWithCovariance`
  (translation first). Kinds whose in-memory type cannot hold an `Uncertain`
  value (a rig sensor's `Camera` extrinsics, a correspondence row) are plain;
- **storage**: every kind is JSON. Once a kind's bulk numeric arrays exceed
  about 10 MB it keeps its JSON header (schema, conventions, frames, sensor,
  provenance) and moves the arrays into an Apache Arrow sidecar named by the
  reserved top-level key `"payload": {"file", "sha256"}` — glTF's JSON + bin
  pattern. No kind implements it yet; a file carrying `payload` is refused
  until one does.

Earlier versions are retired and read by nothing: `load` refuses them by
name. Every stored record was migrated once (2026-10): the v1 files by
a v1 migration, the lens-bearing kinds (cameras, rigs), which took VGC's lens
families at v3, by a standalone lens migration that proved each lens's
function equal; both tools are in the git history. A v1 or v2 transforms or
trajectory file never had a migration: its free-form frame names have no typed
successor — re-export it from its producer.

**Retired by the direct-serialization redesign (2026-10)**, with no reader and
no migration — each successor starts at v1 under its new name, and the
producer re-exports: `observations/v2` → `target_points2d/v1`,
`point_observations/v1` → `target_points3d/v1`, `targets/v2` and `rig/v3` →
`rig/v4`, `session/v3` → the calibration pipeline's own session kind (a
pipeline's checkpoint belongs to the pipeline, not to core geometry). A reader refuses each, naming its
successor.

## Conventions

The stack's conventions — how estimators are named, how signatures order their
arguments, and the frame-in = frame-out rule — are one page, VisualGeometryToolkitCore's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
(`docs/src/guide/conventions.md`). This section is what they come to once the
numbers are on disk. Nothing here normalizes: VGIo holds no estimator, so it
owns no conditioning.

### A file declares its conventions, and the labels are derived from the data's types

Three conventions travel with persisted numbers — the **pixel frame**, the
**length unit**, a pose's **direction** — and each is written from what the data
already is. None of them is a `save` argument:

- **Pixel frame.** Pixel data carries its frame in its TYPE: VGC's
  `TargetPoints2D{F}`, with `F` a VGC `PixelFrame{Origin}`.
  The producer that knows its pixels' convention attaches `F`; `save` writes
  `pixel_frame` from it and converts no number, so a label cannot come to
  disagree with the pixels under it. VGC owns the registered set of frames, so
  this package keeps no list of its own. The kinds that store no producer pixels
  — the camera store, rigs, the grid models — declare the canonical
  `colmap` frame, and a stored camera carries no frame of its own: its model's
  intrinsics are canonical by construction. A source frame is declared where the
  conversion happens, `CameraModel(K; image_origin = :opencv)`: a foreign file
  declaring `opencv` or `julia` is converted on read, and re-saved canonical.
- **Length unit.** Every writer writes millimetres, VGC's in-memory unit, and
  declares `length_unit` once; a reader honours any declared length unit and
  returns millimetres. Magnitudes cross the file boundary only through
  FlexUnits' dimension-checked `quantity`/`ustrip`, unit names through
  VisualGeometryToolkitCore's `parse_unit`/`unit_string`.
- **Pose direction.** Every pose's frames are its labels — a transforms
  record's `from`/`to` are the frame names of the stored `EuclideanMap{…, S, D}`
  (an unlabeled pose is refused) — and the file declares
  `transform_convention = "x_to = R * x_from + t"` with
  `rotation_encoding = "quaternion_wxyz_hamilton"`.

Readers hand the labels back as types — `load(TargetPoints2D, path)` is a
`TargetPoints2D{F}` with `F` the file's declared frame — so the frame travels
with the numbers instead of living in a comment. VGC's `convert_frame(G, x)` is
the only way to change it, and it moves every pixel by VGC's `EuclideanMap(F, G)`. The one
frame stated by hand is a foreign file's: `file_frame` on the OpenCV/MATLAB
readers and migrations (a statement about the file), and `to_frame` on the
OpenCV-shaped export (the target convention of a third-party consumer).

**Holding a file of unknown provenance**, read its `schema` first.
`VisualGeometryToolkitIo.validate(path)` checks any `vgio/<kind>/v2` file against its
kind's table — the same table the reader and the writer run — and such a file
answers every question about its own conventions. Anything else (an OpenCV
or MATLAB export) states less than it needs, and each missing statement is
an argument YOU supply, below. No reader infers one from the numbers: it refuses
rather than defaults.

### Third-party calibrations: `migrate_opencv`, `migrate_matlab`

Both convert one third-party export into the same two native files —
`<stem>_cameras.json` (a `vgio/cameras/v3` store, one camera named `<stem>`) and
`<stem>_poses.json` (a `vgio/transforms/v3` file, one pose per posed image
labeled `TargetFrame{target} → ViewFrame{image}`, in millimetres; an image the
export left `null` gets no record). The round trip is verified before anything is
written, the original is never touched, and an output that already exists is
refused.

What the export leaves implicit is an argument, and every one of them is a
statement about the FILE:

- **`file_frame`**, the pixel frame of `cx, cy`. `migrate_opencv` takes OpenCV's
  own — the top-left pixel's centre at (0, 0), `PixelFrame{:opencv}`;
  `migrate_matlab` takes MATLAB's documented (1, 1), `PixelFrame{:julia}`. A
  fitted principal point cannot tell the two apart (one capture's two fits differ
  by under a pixel), so override a default only for a file VERIFIED to be in
  another frame.
- **`file_unit`**, the unit of `tvec` — the unit whoever typed the object points
  gave the pattern in. REQUIRED unless the file declares it (`tvec_unit`, or this
  package's own `unit`); a declaration that disagrees with the caller is an
  error, never a silent rescale. On one capture the OpenCV export was in metres
  and the MATLAB export in millimetres.
- **`images`** (one unique, non-empty name per pose entry, in the export's order)
  and **`image_size`**: the positional join and the extent an export leaves to
  its reader. Both required.

MATLAB additionally: its `mean_error` is a MEAN reprojection error, not an rms,
and is recorded as such (`error_statistic = "mean"`). Its board frame has x and y
exchanged relative to OpenCV's for the same pattern, and the poses are carried in
MATLAB's own board frame unchanged — give `target` a name that keeps them apart
from an OpenCV calibration's. The toolbox's exports declare no `tvec_unit`, so
`file_unit` is required; this stack's captures were typed in millimetres. A
fisheye export is read as VGC's `Scaramuzza`: `stretch_matrix` is factored
`S = U·Q`, `a0·U` going into `K` and `Q` — a rotation about the optical axis, a
fraction of a degree here — into the camera frame, which every returned pose then
carries, so camera and poses stay one consistent model. A non-positive `a0` is
MATLAB's mirrored solution (the target behind the camera) and is refused rather
than read as a camera.

The readers under these are `load_opencv_calibration` / `load_matlab_calibration`,
a foreign-format adapter (`ExternalCalibration`) holding VGC values: the
camera a `CameraModel`, the poses `EuclideanMap`s labeled
`TargetFrame{target} → CameraFrame` (relabeled `→ ViewFrame{image}` by the
migration). `save_opencv_calibration` writes one of our cameras back out in the
OpenCV shape, and is the one place a pixel frame is a conversion TARGET
(`to_frame`) rather than a statement about a file.

### The command line

`bin/vgio_migrate.jl` converts a third-party calibration and is a DRY RUN
unless `--apply`: the export is converted to a temporary directory and
verified, and nothing is written.

    julia --project=<VisualGeometryToolkitIo> bin/vgio_migrate.jl opencv --help
    … opencv --images IMGDIR --image-size 6000x4000 --file-unit m calibration_opencv_full.json
    … matlab --images IMGLIST --image-size 6000x4000 --file-unit mm --target board_matlab \
        --apply calibration_matlab_fisheye.json

## The I/O surface: two verbs

Everything reads and writes through `load` and `save`, dispatched on the VGC
type. Both are **deliberately unexported** — FileIO and BlobBoards export their
own `load` — so qualify them, or alias the module:

```julia
import VisualGeometryToolkitIo as VGIo
```

`load` takes the type as an ASSERTION OF INTENT: `load(Trajectory, path)` on a
transforms file errors instead of guessing, and the return type is what you
asked for. `save(path, value; provenance)` needs no type argument — the value
already carries it.

| Kind | VGC value | `load` | `save` |
|---|---|---|---|
| `vgio/cameras/v3` (the camera store) | `Dict{Symbol, …}`: name → `BoundedCamera` of a `CameraModel` (unposed: COLMAP's "camera"), or one `Attributed` with its `(; calibration)` block | `load(Dict{Symbol, BoundedCamera}, path)`; one camera: `load(BoundedCamera, path; name)`, `load(CameraModel, path; name)` | `save(path, store)` |
| `vgio/transforms/v3` | `Vector` of labeled `EuclideanMap{3, …, S, D}` / `Uncertain` ones (mm) | `load(Vector{EuclideanMap}, path)` | `save(path, poses)` |
| `vgio/rig/v4` | `Rig{F, M}` in any storage: sensors (a tuple of `BoundedCamera`s), targets (a `Dict` of `Mounted` targets), or both | `load(Rig, path; resolve)` | `save(path, rig)` |
| `vgio/target/v1` | one `AbstractTarget` (`PointTarget`, `PlanarTarget{Rigid}`), or one `Attributed` with its `(; calibration)` block | `load(AbstractTarget, path; resolve)` | `save(path, target)` |
| `vgio/trajectory/v3` | `Trajectory{S, D}`: labeled poses per frame index, images, times, recovered flags | `load(Trajectory, path)` | `save(path, tr)`; exporters `save_tum_trajectory`, `save_euroc_trajectory` |
| `vgio/target_poses/v1` | `TargetPoses{F}`: per frame, `TargetFrame{id} → F` poses; `pose(tp, t, id)` | `load(TargetPoses, path)` | `save(path, tp)`; exporter `save_bop_scene` |
| `vgio/target_points2d/v1` | `TargetPoints2D{F <: PixelFrame}`: per frame, `(target, index, xy, weight)` | `load(TargetPoints2D, path)` | `save(path, obs)`; exporter `save_correspondences` |
| `vgio/target_points3d/v1` | `TargetPoints3D{F}`: per frame, `(target, index, xyz)` (mocap markers) | `load(TargetPoints3D, path)` | `save(path, obs)`; exporter `save_trc` |
| `vgio/generic_camera/v2` (Schöps grid models) | `CentralGenericModel`, `NoncentralGenericModel` | `load(CentralGenericModel, path)`, `load(NoncentralGenericModel, path)` | `save(path, m)` |
| `vgio/correspondences/v2` (flat 2D–3D pairs per frame and target) | an EXPORT of `TargetPoints2D` + target rigs; no reader | — | `save_correspondences([to_frame,] path, obs, targets...; resolve)` |
| COLMAP model dir (`.bin`/`.txt`) | `ColmapModel` (adapter; `camera_model`, `rigid_transform(image, contract)` convert) | `load(ColmapModel, dir; contract)` (`contract = UnknownScale()` for an SfM model with no metric scale) | `save(dir, m; format)` |
| OpenCV / MATLAB calibration exports | `ExternalCalibration` (adapter of VGC values) | `load_opencv_calibration([file_frame,] path; file_unit)`, `load_matlab_calibration([file_frame,] path; file_unit)`; `migrate_opencv` / `migrate_matlab` convert one to native files (`bin/vgio_migrate.jl opencv|matlab`, a dry run unless `--apply`) | `save_opencv_calibration([to_frame,] path, cal; to_unit, images)` |
| COLMAP `database.db` (lazy read handle) | — | `ColmapDatabase(path)`; accessors `keypoints`, `descriptors`, `matches`, `verified_matches`, … (public, qualify them) | — read-only |

**COLMAP, for orientation** (documentation only; there is no COLMAP exporter
for these kinds): `target_points2d` is COLMAP's `images.txt` `points2D` with
their `POINT3D_ID`, the `(target, index)` pair playing the `POINT3D_ID`; a rig
of targets (`rig/v4` targets) is COLMAP's `points3D`, kept as per-target frames
(each target's points in its own `TargetFrame`, placed by its mount) and with no
stored tracks — the measurements are the tracks.

**Targets.** A rig's target member is VGC's `Mounted(target, mount)`: the
target's geometry (an `AbstractTarget`) and its mount in the rig frame. A
`PointTarget` is written with its points (and their covariances iff
`Uncertain`); a `PlanarTarget` of explicit nominal points with them; a
`PlanarTarget` of a content-addressed design (a BlobBoards board uid) by its
`design` reference ALONE — its points are the design owner's — and it reads back
unresolved unless the caller supplies `resolve`, the owner's resolver
(`design::Symbol -> nominal points`). Warps: `{"type": "rigid"}` today; a warp
with parameters will carry them (and their covariance) in its record.

**Building blocks** for a package serializing a kind of its own (a
calibration pipeline's session): `pose_json` / `pose_from_json`, `camera_json` / `camera_from_json`,
`target_json` / `target_from_json`, `pose_declarations` — public, unexported.

## Kinds and related formats

The geometry kinds were designed against the community's formats — kept in
spirit, aligned to VGIo's rules (one JSON schema per kind, declared
conventions, typed frames, `Uncertain` ↔ covariance). Their exact syntax is
emitted only by the exporters. The references (shallow clones, 2026-10-09):

| kind | related formats | adopted | not adopted |
|---|---|---|---|
| `transforms/v3` | ROS `geometry_msgs/TransformStamped` (`common_msgs/geometry_msgs/msg/TransformStamped.msg:1-2,8-10`: `header.frame_id` → `child_frame_id`); Kalibr camchain `T_cn_cnm1`, `T_cam_imu` (`kalibr_common/ConfigReader.py:678-717`, "LastCamToHere", "ImuToCam") | both ends of a transform named, per record; what the transform does to a point stated | the direction in a variable name (`T_to_from`); ROS's x,y,z,w quaternion (`Quaternion.msg:3-6`) — ours is w,x,y,z, declared |
| `trajectory/v3` | TUM RGB-D `t tx ty tz qx qy qz qw` (evo `file_interface.py:121-174`); EuRoC/ASL `state_groundtruth_estimate0/data.csv` (`:224-243`, ns, w first, `p_RS_R`); KITTI odometry 3×4 rows, no timestamps (`:177-220`); evo stores w,x,y,z (`core/trajectory.py:64,71`) | an explicit time per pose (`time`, seconds since a declared `time_origin`); the body → world direction; absent, not NaN, when untracked | a positional join (KITTI) — frames carry `t` and the `image` name; the unit and the frames are declared, not implied |
| `target_poses/v1` | BOP `scene_gt.json` (`bop_toolkit_lib/inout.py:267-300`; `docs/bop_datasets_format.md:108-112`: per image `[{cam_R_m2c, cam_t_m2c, obj_id}]`), `scene_camera.json` (`:79-90`, `inout.py:221-260`) | the per-image object list; model → camera direction; an undetected object absent | an integer `obj_id` (targets keep their identity), K per image (a frame names its camera record, `sensor`) |
| `target_points3d/v1` | OpenSim TRC (`OpenSim/Common/TRCFileAdapter.cpp:236-376`, reader `:38-233`: `Frame# Time` + X/Y/Z per marker, empty = missing); C3D (`ezc3d/README.md:526-612`: POINT data, residual < 0 = occluded) | per-frame time; a unit; markers named | NaN or empty fields — an unseen marker is absent; the scalar residual — a covariance when known |
| pose covariance (every kind) | ROS `PoseWithCovariance` (`PoseWithCovariance.msg:5-9`: row-major 6×6, x y z, rot X Y Z); GTSAM Pose3 tangent `[ω; v]` (`gtsam/geometry/Pose3.cpp:100`, right retraction `T·Exp(ξ)`); Sophus SE3 `[υ; ω]` (`sophus/se3.hpp:840-850`) | GTSAM's ORDER (rotation first) | GTSAM's chart: ours perturbs the rotation on the left, in the destination frame (VGC `RigidExpManifold6`), the chart VisualGeometryToolkitEstimation reports in — the ROS layout is the same chart, blocks swapped (`ros_pose_covariance`) |
| `cameras/v3`, COLMAP | COLMAP `images.txt` (`doc/format.rst:143,153-163`: world → camera, Hamilton w,x,y,z), pixel centres at 0.5 (`doc/cameras.rst:52-57`) | Hamilton w,x,y,z; the COLMAP pixel frame as the canonical one | — |

Exporters (write-only; verified by reading their text back, `test/rig_schema_tests.jl`):

- `save_tum_trajectory(path, tr; stamp = :time)` — TUM, metres, x,y,z,w, the
  `from`-in-`to` pose; `# ` comment header (skipped by evo);
- `save_euroc_trajectory(path, tr; stamp = :time)` — EuRoC ground truth,
  ns timestamps, metres, w first (evo reads ≥ 8 columns);
- `save_bop_scene(to_frame, dir, tp, cameras; obj_ids, undistorted)` —
  `scene_gt.json` + `scene_camera.json` laid out as `bop_toolkit`'s
  `save_json` writes them, K moved into `to_frame`, the obj_id and image maps
  returned;
- `save_trc(path, obs; data_rate, stamp)` — OpenSim TRC, tab-separated, mm,
  missing markers as empty fields. No C3D writer: a maintained Python writer
  (ezc3d, conda-forge) exists, but a Python dependency in VGIo for a binary
  format is out of proportion — TRC carries the same marker trajectories.

## Layers

1. **Wire** (`ColmapCamera`, `ColmapImage`, `ColmapPoint3D`): unit-free,
   format-faithful, always Float64 on disk — the COLMAP adapter.
2. **Unit contract** (`UnitContract`, `MM_PX`): the
   checked boundary between file magnitudes and quantities.
3. **Assembly** (`camera_model`, `colmap_camera`, `rigid_transform`,
   `colmap_image`): conversion to/from VisualGeometryToolkitCore's
   `CameraModel`/rigid `EuclideanMap`s, quantity-first so VGC's construction-boundary unit
   enforcement applies to all file data. Every COLMAP 4.0 camera model reads
   as the VGC lens computing COLMAP's own function, and `colmap_model(lens)`
   names the model a lens writes as (`nothing` when none computes it). The
   OpenCV (`opencv_lens`, `opencv_coefficients`) and Kalibr (`kalibr_lens`)
   conventions are the same kind of exact adapter.
   `camera_model(Float32, c)` gives single-precision in-memory entities;
   the wire stays Float64.

Pixel data carries its frame in its type, and every persisted label is derived
from the data's types rather than supplied to a writer — see **Conventions** above.

The camera **store** is JSON, not COLMAP, on purpose: COLMAP's camera
vocabulary is closed and narrower than VGC's lens families (the double sphere
has no COLMAP model at all), so storing cameras as COLMAP would cap what a config can
refer to and fail at export time — long after calibration. COLMAP stays an
export for the expressible subset; `colmap_cameras(store)` performs it and
throws on a lens it cannot express rather than approximating one.

`append_jsonl(path, row)` is the one append-safe JSONL run-log primitive
(single `write` on an `O_APPEND` handle, plus a per-path lock).

## Verification

- The suite round-trips the binary wire layer through **pycolmap**, the
  reference implementation (`test/pycolmap_tests.jl`), and requires every value
  back identical. It needs the pixi `py` environment (`pixi install -e py`);
  without it the check is recorded as skipped.
- `Pkg.test("VisualGeometryToolkitIo"; julia_args = ["--depwarn=error"])` succeeds and
  is worth running before a release — this package sits under a stack that
  cascades often (VGC → VGIo → BlobBoards), so an upstream deprecation is
  cheapest to fix at the release that introduces it.

Aqua, ExplicitImports and the JET inference ratchet (baseline 0) run as part
of the normal suite (`test/quality_tests.jl`).

The suite is ReTestItems, like VisualGeometryToolkitCore's: every `test/*_tests.jl`
holds `@testitem`s run on worker processes. `Pkg.test(; test_args = ["migrate|cameras"])`
runs the items whose name matches, and `RETESTITEMS_NWORKERS=0` runs serially
in-process.

## Example

```julia
using VisualGeometryToolkitIo, VisualGeometryToolkitCore
import VisualGeometryToolkitIo as VGIo          # `load`/`save` are unexported

X = relabel(BodyFrame{:camera_rig}, CameraFrame, pose)        # a labeled pose: its frames ARE its labels
VGIo.save("hand_eye.json", [X]; provenance = (method = "robot-world hand-eye",))
X2 = only(VGIo.load(Vector{EuclideanMap}, "hand_eye.json"))   # == X's value, labels and all
```

`provenance` is written to the file's envelope and is NOT returned by `load` —
read the JSON directly when the provenance is what you are after.

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes
`Project.toml`, they re-resolve the environment (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## License

MIT
