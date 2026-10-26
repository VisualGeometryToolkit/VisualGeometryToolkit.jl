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
| `vgio/rig/v4` | `Rig{F, M}` holding sensors (a tuple of `BoundedCamera`s), targets (a `Dict` of `PosedTarget`s), or both | `load(Rig, path; resolve)` | `save(path, rig)` |
| `vgio/target/v1` | one `AbstractTarget` (`PointTarget`, `PlanarTarget{Rigid}`), or an `Attributed` one with its `(; calibration)` block | `load(AbstractTarget, path; resolve)` | `save(path, target)` |
| `vgio/trajectory/v3` | `Trajectory{S, D}`: labeled poses per frame index, with images, times and recovered flags | `load(Trajectory, path)` | `save(path, tr)`; exporters `save_tum_trajectory`, `save_euroc_trajectory` |
| `vgio/target_poses/v1` | `TargetPoses{F}`: per frame, `TargetFrame{id} → F` poses; `pose(tp, t, id)` | `load(TargetPoses, path)` | `save(path, tp)`; exporter `save_bop_scene` |
| `vgio/target_points2d/v1` | `TargetPoints2D{F <: PixelFrame}`: per frame, `(target, index, xy, weight)` | `load(TargetPoints2D, path)` | `save(path, obs)`; exporter `save_correspondences` |
| `vgio/target_points3d/v1` | `TargetPoints3D{F}`: per frame, `(target, index, xyz)`, such as motion-capture markers | `load(TargetPoints3D, path)` | `save(path, obs)`; exporter `save_trc` |
| `vgio/generic_camera/v2` (Schöps grid models) | `CentralGenericModel`, `NoncentralGenericModel` | `load(CentralGenericModel, path)`, `load(NoncentralGenericModel, path)` | `save(path, m)` |
| `vgio/correspondences/v2` (flat 2D–3D pairs per frame and target) | an export of a `TargetPoints2D` and its target rigs; there is no reader | — | `save_correspondences([to_frame,] path, obs, targets...; resolve)` |

## The foreign formats

| Format | Value | Read | Write |
|---|---|---|---|
| COLMAP model directory (`.bin` or `.txt`) | `ColmapModel`, an adapter; `camera_model` and `rigid_transform(image, contract)` convert its records | `load(ColmapModel, dir; contract)`; supply `contract = UnknownScale()` for an SfM model with no metric scale | `save(dir, m; format)` |
| OpenCV or MATLAB calibration export | `ExternalCalibration`, an adapter holding VGC values | `load_opencv_calibration([file_frame,] path; file_unit)`, `load_matlab_calibration([file_frame,] path; file_unit)`; `migrate_opencv` and `migrate_matlab` convert one to native files | `save_opencv_calibration([to_frame,] path, cal; to_unit, images)` |
| COLMAP `database.db` | a lazy read handle | `ColmapDatabase(path)`, with the accessors `keypoints`, `descriptors`, `matches`, `verified_matches` and others (public, so qualify them) | read-only |

See [Third-Party Calibrations](third-party.md) for the OpenCV and MATLAB
readers and migrations.

## COLMAP, for orientation

This section is documentation only; there is no COLMAP exporter for these
kinds. A `target_points2d` file corresponds to the `points2D` of COLMAP's
`images.txt` with their `POINT3D_ID`, where the `(target, index)` pair plays
the role of the `POINT3D_ID`. A rig of targets (the targets of a `rig/v4`)
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
  written by its `design` reference alone, since its points belong to the
  design's owner. It reads back unresolved unless the caller supplies
  `resolve`, the owner's resolver (`design::Symbol -> nominal points`).

The only warp the `vgio/target/v1` record holds is `{"type": "rigid"}`: a
`PlanarTarget` whose `TargetCalibration` is anything but `Rigid` (or an
uncertain one) is refused at save. A calibration with parameters will carry
them, and their covariance, in its record.

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
