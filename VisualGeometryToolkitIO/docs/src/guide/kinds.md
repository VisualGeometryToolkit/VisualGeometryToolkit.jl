# Kinds and the I/O Surface

## Two verbs

Every native kind is read with `load` and written with `save`, dispatched on
the VGC type. Neither is exported, because FileIO and BlobBoards export their
own `load`. Qualify them, or alias the module:

```julia
import VisualGeometryToolkitIO as VgtkIO
```

`load` takes the type as a statement of what you expect. `load(Trajectory,
path)` on a transforms file is an error rather than a guess, and the return
type is the one you asked for. `save(path, value; provenance)` needs no type
argument, because the value already carries it.

## The native kinds

| Kind | VGC value | Read | Write |
|---|---|---|---|
| `vgio/cameras/v3` (the camera store) | `Dict{Symbol, …}` of name → `BoundedCamera` of a `CameraModel` (unposed, like a COLMAP "camera"), or an `Attributed` one with its `(; calibration)` block | `load(Dict{Symbol, BoundedCamera}, path)`; one camera: `load(BoundedCamera, path; name)` or `load(CameraModel, path; name)` | `save(path, store)` |
| `vgio/transforms/v3` | `Vector` of labeled `EuclideanMap{3, …, S, D}`, or `Uncertain` ones (mm) | `load(Vector{EuclideanMap}, path)` | `save(path, poses)` |
| `vgio/rig/v5` | `Rig{F, M}` holding sensors (a tuple of `BoundedCamera`s), targets (a `Dict` of `PosedTarget`s), or both | `load(Rig, path; resolve)` | `save(path, rig)` |
| `vgio/target/v2` | one `AbstractTarget` (`PointTarget`, `PlanarTarget` of any `TargetCalibration`, `Uncertain` or not), or an `Attributed` one with its `(; calibration_record)` | `load(AbstractTarget, path; resolve)` | `save(path, target)` |
| `vgio/trajectory/v3` | `Trajectory{S, D}`: labeled poses per frame index, with images, times and recovered flags | `load(Trajectory, path)` | `save(path, tr)`; exporters `save_tum_trajectory`, `save_euroc_trajectory` |
| `vgio/target_poses/v1` | `TargetPoses{F}`: per frame, `TargetFrame{id} → F` poses; `pose(tp, t, id)` | `load(TargetPoses, path)` | `save(path, tp)`; exporter `save_bop_scene` |
| `vgio/target_points2d/v1` | `TargetPoints2D{F <: PixelFrame}`: per frame, `(target, index, xy, weight)` | `load(TargetPoints2D, path)` | `save(path, obs)`; exporter `save_correspondences` |
| `vgio/target_points3d/v1` | `TargetPoints3D{F}`: per frame, `(target, index, xyz)`, such as motion-capture markers | `load(TargetPoints3D, path)` | `save(path, obs)`; exporter `save_trc` |
| `vgio/generic_camera/v2` (Schöps grid models) | `CentralGenericModel`, `NoncentralGenericModel` | `load(CentralGenericModel, path)`, `load(NoncentralGenericModel, path)` | `save(path, m)` |
| `vgio/correspondences/v2` (flat 2D–3D pairs per frame and target) | an export of a `TargetPoints2D` and its target rigs; there is no reader | — | `save_correspondences([to_frame,] path, obs, targets...; resolve)` |

A camera bounded by its physical sensor (`BoundedCamera(camera, sensor::Sensor)`)
carries the sensor's pixel pitch in its camera body, `"pitch": [width, height]`,
a length per pixel in the file's `length_unit` — a rig sensor (parametric or
generic grid model), and a camera of the store, which then declares
`length_unit`. The readers restore the `Sensor` bound, and a rig's `content_id`
covers the pitch. A camera bounded by a bare pixel extent has no `pitch` key.

## The foreign formats

| Format | Value | Read | Write |
|---|---|---|---|
| COLMAP model directory (`.bin` or `.txt`) | `ColmapModel`, an adapter; `camera_model` and `rigid_transform(image, contract)` convert its records | `load(ColmapModel, dir; contract)`; supply `contract = UnknownScale()` for an SfM model with no metric scale | `save(dir, m; format)` |
| OpenCV or MATLAB calibration export | `ExternalCalibration`, an adapter holding VGC values | `load_opencv_calibration([file_frame,] path; file_unit)`, `load_matlab_calibration([file_frame,] path; file_unit)`; `migrate_opencv` and `migrate_matlab` convert one to native files | `save_opencv_calibration([to_frame,] path, cal; to_unit, images)` |
| Schöps et al.'s camera_calibration `intrinsics0.yaml` | `CentralGenericModel`, `NoncentralGenericModel` (VGC's own types; the file's corner pixel convention is `SCHOEPS_PIXEL_FRAME`, COLMAP's) | `load(CentralGenericModel, path)`, `load(NoncentralGenericModel, path; file_unit)`: the form is told from the content; the point grid's unit is the caller's | read-only (`save` writes `vgio/generic_camera/v2`) |
| Schöps et al.'s camera_calibration `dataset.bin` (`calib_data` v0) | `TargetPoints2D{SCHOEPS_PIXEL_FRAME}`: per image, `(target, feature id + 1, xy, 1.0)` | `load(TargetPoints2D, path; target)`: the target the ids index is the caller's | read-only |
| COLMAP `database.db` | a lazy read handle | `ColmapDatabase(path)`, with the accessors `keypoints`, `descriptors`, `matches`, `verified_matches` and others (public, so qualify them) | read-only |

See [Third-Party Calibrations](third-party.md) for the OpenCV and MATLAB
readers and migrations.

## COLMAP, for orientation

This section is documentation only; there is no COLMAP exporter for these
kinds. A `target_points2d` file corresponds to the `points2D` of COLMAP's
`images.txt` with their `POINT3D_ID`, where the `(target, index)` pair plays
the role of the `POINT3D_ID`. A rig of targets (the targets of a `rig/v5`)
corresponds to COLMAP's `points3D`. It is kept as per-target frames, with each
target's points in its own `TargetFrame` and placed by its pose. It stores no
tracks, because the measurements are the tracks.

## Targets

A rig's target member is VGC's `PosedTarget(target, pose)`: the target's geometry
(an `AbstractTarget`) and its pose in the rig frame. How each target is
written:

- A `PointTarget` is written with its points, and with their covariances if
  it is `Uncertain`.
- A `PlanarTarget` of explicit nominal points is written with those points.
- A `PlanarTarget` of a content-addressed design (a BlobBoards board uid) is
  written by its `reference` alone, since its points belong to the
  design's owner. It reads back unresolved unless the caller supplies
  `resolve`, the owner's resolver (`reference::Symbol -> nominal points`).

A `PlanarTarget`'s `calibration` (VGC's `TargetCalibration`, the map from the
design to the physical target) is one record whose `type` is the kind's tag
in VGC's `TARGET_CALIBRATION_TAGS`:

```json
"calibration": {"type": "rigid"}
"calibration": {"type": "affine", "parameters": [a11, a12, a21, a22, bx, by]}
"calibration": {"type": "thin_plate_spline", "nodes": [[u, v], …], "parameters": [fx1, …, fxK, fy1, …, fyK]}
"calibration": {"type": "height_field", "nodes": [[u, v], …], "parameters": [h1, …, hK]}
"calibration": {"type": "calibration_chain", "members": [{…}, {…}]}
```

`parameters` are the kind's `parameter_vector` in its `parameter_names` order:
lengths (an affine's `b`, the displacements, the heights) in the file's
`length_unit`, the affine's `A` plain. `nodes` are positions in the target's
plane, in the file's unit. A chain's members are records of their own, in
order. An `Uncertain` calibration adds `covariance` at the top of the record,
n × n over `parameter_vector` (a chain's covers every member's parameters, so
its members carry none), each entry in the file's unit raised to its length
dimension. A calibration with lengths in plain numbers is refused, as plain
points are.

A `PlanarTarget` may also state its `board`, the physical board it is
printed on (parallel to a camera's sensor; the board proper, inside the
sheet's margin and frame included, as BlobBoards' `board_size` — not the
sheet): its size in the `Size2` form and its lower-left corner in the design
chart, `"board": {"size": {"width": w, "height": h}, "origin": [u, v]}`, every
number a length in the file's unit. It is absent when the target has none.

It may also state its `canvas` (VGC's `Canvas`), the printed page the board
lies on, and the image printed on it: `"canvas": {"size": {…}, "origin": [u,
v], "texture": {"path": p, "sha256": h, "size": {…}, "origin": [u, v]}}`. The
board, the canvas and the texture are rectangles of one chart, the design
chart of the nominal points, each its lower-left corner and size in the file's
unit; the board's placement on the page is `board.origin − canvas.origin`,
derived, not stored. The texture's `path` is relative to the file that names
it and `sha256` is the image file's digest; the readers never open the image.
`texture` is optional, and `canvas` is absent when the target has none.

A target file may also carry a `calibration_record`, the record of how the
calibration was obtained (the keys of a camera's `calibration` block: `tool`,
`source`, `date`, `reprojection_rms`, `n_images`, `n_points`, `note`). In
memory it is the attribute `Attributed(target, (; calibration_record))`, so it
never shadows the target's `calibration`.

## Migrating an earlier version

`vgio/target/v1` and `vgio/rig/v4` files are not read: `load` refuses them,
naming `vgio_migrate`. `vgio_migrate(path)` converts one file, and
`bin/vgio_migrate.jl target|rig|file|tree|repoint` is its command (a dry run
unless `--apply`). The new file is written to a staging directory beside its
destination (the same filesystem), read back and compared with the old one
number for number, and only then renamed into place, one atomic `rename`; it
goes beside the original (`<stem>.migrated.json`), or over it with
`out = path` (`--in-place`), the original untouched until the rename. The
comparison is of values: every number comes back bit for bit, except that a
length written at full precision (16–17 digits; a computed value) may come
back up to 2 ulp away in the file, as the same metre value — the length
conversion's guarantee (see [Conventions](conventions.md#Lengths-cross-the-file-boundary-decimal-exactly)).

A rig's content identity changes with this migration, once: the v4 identity
hashed the record's format, and the identity is now version-free, so it will
not change again. `vgio_migrate` reports the rig's `old_ids` — the v4 identity
of the file as written, and the one v4 minted for the value in millimetres
when the file is in another unit — and its `new_id`. Re-pointing the
measurement files that name a migrated rig is a step of that one-time
migration: `vgio_migrate_tree` migrates a directory's files and re-points its
measurement files at its current rigs, refusing any identity that is neither
mapped nor a current rig's (the command's `repoint` does it for one file). It
persists the identity map in `vgio_migrate_ids.json` before it puts a rig in
place, so a run interrupted after its rigs were migrated re-points correctly
when rerun; a rerun of a migrated tree is idempotent, and every file it leaves
as it is gets a log line saying why.

A schema's owner extends the migration to its own files by adding a method of
`VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("<schema>")}, src, dst)`,
a declared extension point (`CONTRACT_GENERICS`).

## Building blocks for other kinds

A package that serializes a kind of its own (for example a calibration
pipeline's session) can reuse the bodies the native kinds embed:
`pose_json` / `pose_from_json`, `camera_json` / `camera_from_json`,
`target_json` / `target_from_json`, and `pose_declarations`. They are public
and not exported.

## Run logs

`append_jsonl(path, row)` appends one JSON line to a run log. It is safe to
call from several tasks and several processes: the line is written with a
single `write` on an append-mode handle, under a per-path lock.
