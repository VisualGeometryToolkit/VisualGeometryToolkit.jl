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
| `vgio/cameras/v5` (the camera store) | `Dict{Symbol, …}` of name → `BoundedCamera` of a `CameraModel` (unposed, like a COLMAP "camera"; a `SplineLens` one included, its lens the `{"spline": …}` record in the K-normalized plane), or an `Attributed` one with its `(; calibration)` block | `load(Dict{Symbol, BoundedCamera}, path)`; one camera: `load(BoundedCamera, path; name)` or `load(CameraModel, path; name)` (refused for a camera whose lens coverage is not its whole sensor) | `save(path, store)` |
| `vgio/transforms/v4` | `Vector` of labeled `EuclideanMap{3, …, S, D}`, or `Uncertain` ones (mm) | `load(Vector{EuclideanMap}, path)` | `save(path, poses)` |
| `vgio/rig/v9` | `Rig{F, M}` holding sensors (a tuple of `BoundedCamera`s, each with its lens coverage; a `SplineLens` camera or a `NoncentralSplineModel` too), targets (a `Dict` of `PosedTarget`s), or both | `load(Rig, path; resolve)` | `save(path, rig)` |
| `vgio/target/v5` | one `AbstractTarget` (`PointTarget`, `PlanarTarget` of any `TargetCalibration`, `Uncertain` or not, stated by its points, its reference or its layout), or an `Attributed` one with its `(; calibration_record)` | `load(AbstractTarget, path; resolve)`; a planar one: `load(PlanarTarget, path)` | `save(path, target)` |
| `vgio/trajectory/v4` | `Trajectory{S, D}`: labeled poses per frame index, with images, times and recovered flags | `load(Trajectory, path)` | `save(path, tr)`; exporters `save_tum_trajectory`, `save_euroc_trajectory` |
| `vgio/target_poses/v2` | `TargetPoses{F}`: per frame, `TargetFrame{id} → F` poses; `pose(tp, t, id)` | `load(TargetPoses, path)` | `save(path, tp)`; exporter `save_bop_scene` |
| `vgio/target_homographies/v1` | `TargetHomographies{F <: PixelFrame, H}`: per frame, each target's plane (mm) → pixel `HomographyMap` (plain or `Uncertain`), written as the parametric record `{"type": "homography_map", "parameters": [9]}` of its canonical entries (unit Frobenius norm, `h33 ≥ 0`; an 8×8 `covariance` in the `HomographyManifold8` chart), before the lens of the frame's `sensor`'s camera (a `RadialHomography`'s `H`); `targets` are the `content_id`s (a rig's or a single target's) its ids resolve in | `load(TargetHomographies, path)` | `save(path, th)` |
| `vgio/target_detections/v2` | `TargetDetections{F}`: per frame, per target, `csponds(index, detection)` — pixels in a `PixelFrame` or 3D points in a named 3D frame, plain or `Uncertain`, with an optional `weights` column; `targets` the `content_id`s of rigs or of single unposed targets | `load(TargetDetections, path)` (a `target_points2d/v1` or `target_points3d/v1` file too, converted — except one with a (target, index) twice in a frame or with plain and `Uncertain` points mixed, refused by name; it still loads as `load(TargetPoints2D/3D, path)`) | `save(path, td)` |
| `vgio/target_points2d/v2` | `TargetPoints2D{F <: PixelFrame}`: per frame, `(target, index, xy, weight)` | `load(TargetPoints2D, path)` | `save(path, obs)`; exporter `save_correspondences` |
| `vgio/target_points3d/v2` | `TargetPoints3D{F}`: per frame, `(target, index, xyz)`, such as motion-capture markers | `load(TargetPoints3D, path)` | `save(path, obs)`; exporter `save_trc` |
| `vgio/capture_settings/v1` | `CaptureSettings`: per frame, the exposure (time, f-number, ISO) and the nominal lens settings (focal length, focus distance); the body and lens `DeviceIdentity`, `Attributed` with each one's source (`"exif"` or `"config"`) and the EXIF lens claim a stated lens overrode | `load(CaptureSettings, path)` | `save(path, Attributed(cs, (; identity_sources = (body, lens)[, exif_lens])))` |
| `vgio/vignetting/v1` | an `AbstractVignetting`: `TensorProductVignetting` (its knots stated), `RadialVignetting` or `LogVignetting`, positions in COLMAP pixels; rebuilt bit for bit | `load(AbstractVignetting, path)` | `save(path, v)` |
| `vgio/environment_light/v1` | an `EnvironmentLight` of a `SphericalHarmonicMap`: centre and radius in `length_unit` (a plain light refused), coefficients per channel, the basis convention stated (`real_orthonormal_no_condon_shortley`) | `load(EnvironmentLight, path)` | `save(path, light)` |
| `vgio/blur/v1` | name → `GaussianBlur`, each one's `covariance` in the storage it holds (round, axis-aligned, general) and the `unit` it states: `"px^2"` on a sensor, `length_unit` squared (`"mm^2"`) on a print; the unit is recorded, the blur's frame tag is not | `load(Dict{Symbol, GaussianBlur}, path)` | `save(path, blurs)` |
| `vgio/calibration/v1` (`calibration.json`, the manifest) | the record files of one calibration result bound by SHA-256: each with its record schema, `role` (`"authority"` or `"derived"`) and `derived_from`; `kind` `"metric"`, `"projective"` or `"image"`, an optional `CameraModelSpec`, `metrics`, `provenance`; a load refuses a changed file by name and measurements naming targets the listed target and rig files do not hold | `load_manifest(dir)` | `save_manifest(dir, files; kind, model, metrics, provenance)` |
| `vgio/noncentral_spline/v1` (the non-central grid camera of Schöps et al.: its K, and its direction and point splines over the K-normalized plane, the held nodes `present` as runs per lattice row) | `BoundedCamera{<:NoncentralSplineModel}` | `load(BoundedCamera, path)`; an old central `vgio/generic_camera/v2`/`v3` file reads as the camera store of its one camera with a `SplineLens`, a non-central one through its step to `vgio/noncentral_spline/v1`; neither is rewritten (`vgio_migrate` writes their current records to new files beside them) | `save(path, camera)` |
| `vgio/correspondences/v2` (flat 2D–3D pairs per frame and target) | an export of a `TargetPoints2D` and its target rigs; there is no reader | — | `save_correspondences([to_frame,] path, obs, targets...; resolve)` |

A `target_detections` file holds, per frame, one record per detected target:
its point indices (1-based, `point_index_base`), its points, and, where the
value has them, one covariance per point (px², or the file's `length_unit`
squared) and its `weights`:

```json
{"schema": "vgio/target_detections/v2", "pixel_frame": "colmap", "point_index_base": 1,
 "targets": ["<content id of a rig, or of a single unposed target>"],
 "frames": [{"t": 17, "image": "DSC0017.png", "sensor": "cam1",
             "detections": [{"target": "b89", "index": [1, 4], "points": [[u, v], [u, v]],
                             "covariance": [[[…], […]], [[…], […]]], "weights": [1.0, 0.5]}]}]}
```

A 3D file states `"frame"` (a 3D frame name) and `"length_unit"` in place of
`"pixel_frame"`, its points are lengths in that unit, and its `targets` may be
absent; a 2D one requires each frame's `image` and `sensor` and at least one
`targets` identity, a 3D one each frame's `sensor`.

### File names

A file's name states its kind and what tells copies of it apart; the loaders
never read it (they read `schema`). The stem is the kind and the suffix the
distinguishing name: `target_detections_<sensor>.json`,
`target_poses_<sensor>.json`, `trajectory_<body>.json`, `targets_<rig>.json`,
`cameras.json`. A name holds no timestamp: the time is the file's
`time_origin` and `times`, the capture date is the session directory's, and a
rerun goes in its own run-key directory with its provenance. Published and
existing files keep their names.

A camera bounded by its physical sensor (`BoundedCamera(camera, sensor::Sensor)`)
carries the sensor's pixel pitch in its camera body, `"pitch": [width, height]`,
a length per pixel in the file's `length_unit` — a rig sensor (parametric or
SplineLens camera), and a camera of the store, which then declares
`length_unit`. The readers restore the `Sensor` bound, and a rig's `content_id`
covers the pitch. A camera bounded by a bare pixel extent has no `pitch` key.

## The foreign formats

| Format | Value | Read | Write |
|---|---|---|---|
| COLMAP model directory (`.bin` or `.txt`) | `ColmapModel`, an adapter; `camera_model` and `rigid_transform(image, contract)` convert its records | `load(ColmapModel, dir; contract)`; supply `contract = UnknownScale()` for an SfM model with no metric scale | `save(dir, m; format)` |
| OpenCV or MATLAB calibration export | `ExternalCalibration`, an adapter holding VGC values | `load_opencv_calibration([file_frame,] path; file_unit)`, `load_matlab_calibration([file_frame,] path; file_unit)`; `migrate_opencv` and `migrate_matlab` convert one to native files | `save_opencv_calibration([to_frame,] path, cal; to_unit, images)` |
| Kalibr camchain (`camchain.yaml`) | `BoundedCamera` | `load_kalibr_camchain([file_frame,] path; camera = "cam0")` | — |
| Schöps et al.'s camera_calibration `intrinsics0.yaml` | `BoundedCamera` of a `SplineLens` camera (central) or of a `NoncentralSplineModel` (the file's corner pixel convention is `SCHOEPS_PIXEL_FRAME`, COLMAP's; K the lattice's gauge) | `load(BoundedCamera, path; file_unit)`: the form is told from the content; a non-central point grid's unit is the caller's | read-only (`save` writes a `vgio/cameras` store or `vgio/noncentral_spline/v1`) |
| Schöps et al.'s pattern YAML (`create_calibration_pattern.py`) | `PlanarTarget(SchoepsPattern(…); canvas)`, the page its canvas | `load(PlanarTarget, path)` (or `load(AbstractTarget, path)`): the form is told from the content | read-only (`save` writes `vgio/target/v5`, the layout as a record) |
| Kalibr's target YAML of an `aprilgrid` | `PlanarTarget(AprilGrid(…))`, the gap `tagSize · tagSpacing` (Kalibr's `tagSpacing` is a ratio) | `load(PlanarTarget, path)` (or `load(AbstractTarget, path)`) | read-only (`save` writes `vgio/target/v5`) |
| Schöps et al.'s camera_calibration `dataset.bin` (`calib_data` v0) | `TargetPoints2D{SCHOEPS_PIXEL_FRAME}`: per image, `(target, feature id + 1, xy, 1.0)` | `load(TargetPoints2D, path; target)`: the target the ids index is the caller's | read-only |
| COLMAP `database.db` | a lazy read handle | `ColmapDatabase(path)`, with the accessors `keypoints`, `descriptors`, `raw_matches`, `verified_matches` and others (public, so qualify them) | read-only |

See [Third-Party Calibrations](third-party.md) for the OpenCV and MATLAB
readers and migrations.

## COLMAP, for orientation

This section is documentation only; there is no COLMAP exporter for these
kinds. A `target_points2d` file corresponds to the `points2D` of COLMAP's
`images.txt` with their `POINT3D_ID`, where the `(target, index)` pair plays
the role of the `POINT3D_ID`. A rig of targets (the targets of a `vgio/rig` file)
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
- A `PlanarTarget` built from a layout (`PlanarTarget(layout; …)`, which VGC
  records as `layout(t)`) is written by its `layout` (since `vgio/target/v4`,
  `vgio/rig/v7`); its points are VGC's from the layout and are not restated.
  The record's `type` is the layout's tag in VGC's `TARGET_LAYOUT_TAGS`, every
  key is written, and lengths are in the file's unit:

```json
"layout": {"type": "april_grid", "tag_rows": 6, "tag_cols": 6, "tag_size": 88.0, "tag_spacing": 26.4,
           "apriltag_family": "tag36h11", "first_id": 0}
"layout": {"type": "schoeps_pattern", "squares_x": 30, "squares_y": 30, "square_length": 33.0,
           "star_segments": 16, "apriltag_family": "tag36h11",
           "apriltags": [{"x": 13, "y": 13, "width": 4, "height": 4, "id": 9}]}
```

  `tag_spacing` is the gap between neighbouring tags, a length (Kalibr's
  `tagSpacing` is that gap over the tag size). A Schöps tag's `x`, `y`,
  `width` and `height` are squares, y down as the pattern is printed (the
  layout's own coordinates); the target's points are in VGC's design chart, y
  up. A body states exactly one of `reference`, `nominal` and `layout`. A
  layout target's `content_id` is of its points, as a nominal target's, so a
  layout file and a nominal file of the same design name the same target.
  A `vgio/target/v3` or `vgio/rig/v6` file (no layout) reads as it always did,
  through its version step, which moves no number.

A `PlanarTarget`'s `calibration` (VGC's `TargetCalibration`, the map from the
design to the physical target) is one record whose `type` is the kind's tag
in VGC's `TARGET_CALIBRATION_TAGS`:

```json
"calibration": {"type": "rigid"}
"calibration": {"type": "euclidean", "parameters": [angle, tx, ty]}
"calibration": {"type": "scale", "parameters": [log_scale]}
"calibration": {"type": "aspect", "parameters": [a1]}
"calibration": {"type": "skew", "parameters": [u12]}
"calibration": {"type": "thin_plate_spline", "nodes": [[u, v], …], "parameters": [fx1, …, fxK, fy1, …, fyK]}
"calibration": {"type": "height_field", "nodes": [[u, v], …], "parameters": [h1, …, hK]}
"calibration": {"type": "composition", "compose": [{outer}, …, {inner}]}
```

`parameters` are the kind's `parameter_vector` in its `parameter_names` order:
lengths (a euclidean's `tx, ty`, the displacements, the heights) in the file's
`length_unit`, the rest plain (a euclidean's angle in radians, the log-scale,
the aspect's log-stretch, the skew's shear). `nodes` are positions in the
target's plane, in the file's unit. The thin-plate kinds hold pure bending (no
affine term in the interpolant): a record whose values carry one is refused.

A composition is 2 to 6 kinds in VGC's canonical order
`HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, any
left out, each once — the affine print error `Aspect ∘ Skew` (with a ruler
`Scale ∘ Aspect ∘ Skew`, with a known pose `EuclideanMap ∘ Scale ∘ Aspect ∘
Skew`), `HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew`, … It is written by its
structure, flat: its members are records of their own, listed under `compose`
(the key a lens record lists its stages in sequence under) in
CoordinateTransformations' order, the outer map first.

`vgio/target/v2` and `vgio/rig/v5` wrote `{"type": "calibration_chain",
"members": […]}`, the members in application order, and an `"affine"` kind
(`[a11, a12, a21, a22, bx, by]`); the current reader refuses both by name.
`vgio_migrate` converts them: an affine is its Iwasawa factors
`EuclideanMap ∘ Scale ∘ Aspect ∘ Skew` (the same map, its covariance through
the factorization's Jacobian, and a rig's old content id reported for the
re-pointing). The record keeps all four factors, since a migration preserves
the calibrated geometry exactly; but `EuclideanMap` and `Scale` are the
target pose's gauges when the pose is estimated jointly, so VGC's
joint-pose manifold (`minimal_manifold(target)`) refuses such a calibration
until they are dropped or absorbed into the pose — or the caller states
what fixes them (`known = (:pose,)`, a known pose). It refuses a chain with no current form (two in-plane kinds out
of the canonical order) and a thin-plate spline or height field whose
interpolant carries an affine term: that part cannot move into the factors
(they compose after the spline) or a height plane into a rigid pose without
changing the map, so such a calibration is re-fitted.

An `Uncertain` calibration adds `covariance` at the top of the record, n × n
over `parameter_vector` (a composition's covers every member's parameters, in
that order, so its members carry none; a thin-plate kind's has rank 2K − 6 or
K − 3), each entry in the file's unit raised to its length dimension. A
calibration with lengths in plain numbers is refused, as plain points are.

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
`source`, `date`, `reprojection_rms`, `n_images`, `n_points`, `note`,
`device`). In
memory it is the attribute `Attributed(target, (; calibration_record))`, so it
never shadows the target's `calibration`.

## Reading an earlier version

Every earlier version a published file holds is read: `vgio/cameras/v1`–`v4`,
`vgio/target/v1`–`v4`, `vgio/rig/v4`–`v8`, `vgio/generic_camera/v2`–`v3` (read, never rewritten: `vgio_migrate` writes a central one's camera store to a new `<stem>.camera.json`, a non-central one's `vgio/noncentral_spline/v1` record to a new `<stem>.noncentral.json`), `vgio/transforms/v3`,
`vgio/trajectory/v3`, and v1 of `target_poses`, `target_points2d`, `target_points3d` and
`target_detections`.

**The covariance storage (2026-11)** moved every kind that carries a covariance up a
version: a covariance is written in the storage its value holds — rows (full),
`{"diagonal": […]}` or `{"isotropic": λ}` — and read back as that storage
(VisualGeometryToolkitCore's `CovMat` holds an `SMatrix`, `SDiagonal` or
`ScaledIdentity`). An isotropic covariance over a pose's angles and lengths is written
as its diagonal (no single number holds rad² and mm²). An earlier file's covariances
are all rows, so its version step only renames the schema. `load` runs the file's version
steps in memory, one per version
([`migrate_schema`](@ref VisualGeometryToolkitIO.migrate_schema)`(::Val{schema}, tree)`,
run by [`migrate_record`](@ref VisualGeometryToolkitIO.migrate_record)), and
reads the current record they give; the writers write
only the current version, so a file saved from what an old file loads as is
current. A load is silent (an `@debug` names the version), and `content_id` of
an old rig file is that of the value it loads as. The reason is the published
data: a schema change must not force the archives on Zenodo and Hugging Face
(several GB each) to be uploaded again, so a published file keeps loading.

A `vgio/cameras/v1` store loads as it is, and `vgio_migrate` rewrites it in
the current version beside the original:

```jldoctest old_version
julia> using VisualGeometryToolkitCore, VisualGeometryToolkitIO

julia> import VisualGeometryToolkitIO as VgtkIO

julia> path = joinpath(mktempdir(), "cameras.json");

julia> write(path, """
       {"schema": "vgio/cameras/v1", "cameras": [{
           "schema": "vgio/camera/v1", "name": "left", "pixel_chart": "colmap",
           "size": {"unit": "px", "width": 1280, "height": 960},
           "intrinsics": {"unit": "px", "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 480.0, "skew": 0},
           "lens": {"model": "KannalaBrandt", "params": [0.1, -0.02, 0.003, -0.0004]}}]}
       """);

julia> camera = VgtkIO.load(BoundedCamera, path; name = :left);

julia> Tuple(extent(camera))
(1280, 960)

julia> r = vgio_migrate(path);

julia> (r.from, r.to, basename(r.out))
("vgio/cameras/v1", "vgio/cameras/v5", "cameras.migrated.json")

julia> VgtkIO.load(BoundedCamera, r.out; name = :left) == camera
true
```

[`vgio_migrate`](@ref VisualGeometryToolkitIO.vgio_migrate)`(path)` rewrites one file with the same steps, and
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

A v4 rig's content identity was of its record's format; the identity is now
version-free, so a v4 rig is named by a new one (and so is a v5 rig whose
target held the retired affine kind, which migrates to its factors).
Measurement files written beside such a rig name it by its old identity, and
`save_correspondences` resolves them against the old rig file as they are.
`vgio_migrate` reports the rig's `old_ids` — the v4 identity of the file as
written, and the one v4 minted for the value in millimetres when the file is
in another unit — and its `new_id`, and `vgio_migrate_tree` migrates a directory's files and re-points its
measurement files at its current rigs, refusing any identity that is neither
mapped nor a current rig's (the command's `repoint` does it for one file). It
persists the identity map in `vgio_migrate_ids.json` before it puts a rig in
place, so a run interrupted after its rigs were migrated re-points correctly
when rerun; a rerun of a migrated tree is idempotent, and every file it leaves
as it is gets a log line saying why.

A schema's owner reads its own old files the same way: it adds a step
`VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("<schema>")}, tree)` for
each old version, calls `VisualGeometryToolkitIO.migrate_record(tree)` in its
reader, and adds the file form `migrate_schema(::Val{Symbol("<schema>")}, src,
dst)` that `vgio_migrate` runs — a declared extension point
(`CONTRACT_GENERICS`).

A package whose own kinds are files reads and writes them through the same two
verbs: it adds `VisualGeometryToolkitIO.load(::Type{<:T}, path)` and
`VisualGeometryToolkitIO.save(path, x::T)` methods for its types `T` (BlobBoards'
boards and patterns), both declared extension points (`CONTRACT_GENERICS`), and
writes its records through `write_record`/`read_record`.

## Building blocks for other kinds

Every VisualGeometryToolkitCore value a native kind embeds is written and read
on one serialization style, `VgioStyle(; length_unit = mm, pixel_frame =
PixelFrame{:colmap})`, a `JSON.JSONStyle` that carries what the enclosing file
declares. `JSON.json(x; style)` writes a value, `JSON.lower(style, x)` gives
its record and `JSON.lift(style, T, record)` reads one back from a record
read by property access — a parsed `JSON.Object` or a `NamedTuple` (a `Dict`
is not one); the methods are on `VgioStyle` only.

| Value | `T` in `lift` | Record |
|---|---|---|
| `EuclideanMap{3}`, plain or `Uncertain` | `EuclideanMap{3}` | `{"rotation": [w, x, y, z], "translation": [x, y, z], "covariance": …}` |
| `Trajectory` | `Trajectory` | the `vgio/trajectory` body without its schema and pose declarations: `{"from", "to", ["time_origin"], "frames": […]}` |
| `CameraModel` | `CameraModel` | `{"K": {"fx", "fy", "cx", "cy", "skew"}, "lens": {…}}`, `cx, cy` in `pixel_frame` |
| `CameraModel` with a `SplineLens` | `CameraModel` | the camera body, its `lens` `{"spline": {"area", "grid_size", "present"?, "controls", "forward"?}}` |
| `PointTarget`, `PlanarTarget` | `AbstractTarget` (keyword `resolve`, as `target_from_json`'s) | the target body, a planar target's calibration as the parametric record (below) |
| `AprilGrid`, `SchoepsPattern` | `AbstractTargetLayout` | `{"type": tag, every field}` |
| `Ellipse` | `Ellipse` | `[cx, cy, a, b, θ]` |
| `TensorProductVignetting`, `RadialVignetting`, `LogVignetting` | `AbstractVignetting` | the `vgio/vignetting` body, `{"type": …, …}`, positions in COLMAP pixels |
| `EnvironmentLight` of a `SphericalHarmonicMap` | `EnvironmentLight` | `{"center": [x, y, z], "radius": r, "map": {…}}`, lengths in `length_unit` (the `vgio/environment_light` body without `schema`/`length_unit`) |
| `GaussianBlur` | `GaussianBlur` | `{"unit": "px^2" \| "<length_unit>^2", "covariance": …}` in the storage it holds: px² on a sensor (plain), a print's in `length_unit`² (read back as m² quantities); the lift reads the stated unit |

A target's calibration is written and read as part of its target's body, not
by a `lower` of its own: VisualGeometryToolkitCore's `TargetCalibration` holds
every planar `EuclideanMap`, so a `lower` on it would write any 2D map left in
a record (a pixel-frame conversion) as a target calibration.

A parametric model is written as `{"type": tag, "parameters": [θ₁, …]}`: the
tag from its kind's `TagRegistry` (snake_case of the type's name), the
parameters VisualGeometryToolkitCore's `parameter_vector` in `parameter_names`
order, the lengths among them (`length_mask`) in the file's unit and decimal-exact,
and no `parameters` for a kind that has none. It is read back with
`with_parameters`; a record holding more or fewer parameters than its kind is
an `ArgumentError` naming the tag. A kind whose structure its type does not fix states that
structure beside its parameters (a thin-plate kind's `nodes`).

A package's own kind crosses the file boundary through the same pair the
native kinds do: [`write_record`](@ref VisualGeometryToolkitIO.write_record)`(path, record; style)` writes a record
(its VGC values lowered on `style`) in the vgio file shape, refusing a NaN or
±Infinity by its field, and [`read_record`](@ref VisualGeometryToolkitIO.read_record)`(path)` parses one back as
its kind's current record, through the owner's version steps
(`migrate_schema`, run by `migrate_record`) when the file is of an earlier
version; a step's refusal names the file and its schema. `read_record(io)`
reads the same from JSON text. The record states its own schema, `"<owner>/<kind>/v<n>"`; the owner
validates its fields. A run report that may hold a non-finite statistic is
written and read with `allownan = true`.

`pose_json` / `pose_from_json`, `camera_json` / `camera_from_json`,
`target_json` / `target_from_json` and `pose_declarations` are the same
records for a length unit or a pixel frame given directly. All of these are
public and not exported. A camera's body comes in three forms:
`camera_json(model::CameraModel, extent, frame)`, a non-central grid camera's
`camera_json(model::NoncentralSplineModel, extent, frame; length_unit)`, and a
bounded camera's `camera_json(camera::BoundedCamera, frame; length_unit)` — its
model's body with the sensor's pitch and the lens coverage, the body a camera
store, a rig sensor and the non-central record embed —, read back by
`camera_from_json(body, frame; length_unit)` (the model and its extent) and
`camera_from_json(BoundedCamera, body, frame; length_unit)`.

## Run logs

`append_jsonl(path, row)` appends one JSON line to a run log. It is safe to
call from several tasks and several processes: the line is written with a
single `write` on an append-mode handle, under a per-path lock.
