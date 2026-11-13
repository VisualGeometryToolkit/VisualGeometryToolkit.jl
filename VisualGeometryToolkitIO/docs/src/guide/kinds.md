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

## The envelope

Every vgio file opens with the same four keys, then holds its kind's body, then
the optional free-form `provenance`:

```json
{"schema": "vgio/transforms/v5.0",
 "generator": {"name": "VisualGeometryToolkitIO", "version": "0.10.0",
               "caller": {"name": "BlobBoards", "version": "0.7.0"}},
 "units": {"length": "mm", "time": "s", "angle": "rad", "pixel": "colmap", "density": "dpi",
           "radiance": "relative", "length_roles": {}},
 "conventions": {"rotation_encoding": "quaternion_wxyz_hamilton", "transform_convention": "x_to = R * x_from + t",
                 "matrix_layout": "rows", "grid_layout": "row_major", "index_base": 1},
 "transforms": […],
 "provenance": {"method": "robot-world hand-eye"}}
```

- **`schema`** is `vgio/<kind>/v<major>.<minor>` (below).
- **`generator`** names the package that wrote the file and its version, and
  the calling package when the writer runs inside
  [`with_caller`](@ref VisualGeometryToolkitIO.with_caller)`(f, Mod)`.
- **`units`** is the ONE units object. Every number of the body is in a unit it
  declares, raised to the power its field's schema states: a pose covariance's
  translation block `length²`, a sensor blur's `px²`, a print blur's `length²`.
  No field and no block states a unit of its own. A field whose lengths are in
  another unit than `length` names a ROLE, and its unit is
  `length_roles.<role>` (a board's pattern in `pt`); a role's value may be the
  reserved token `"chart"`, a CHART role whose numbers are plain numbers of a
  chart its record's owner defines (a height field's page-fraction nodes,
  `node`). `pixel` is the pixel frame
  of every pixel coordinate, `density` the unit of a print density, `radiance`
  that of a light's coefficients (`relative`: known up to scale); `time` (`s`)
  and `angle` (`rad`) are fixed tokens, as the rotation encoding is.
- **`conventions`** is the ONE conventions object: Hamilton w-x-y-z
  quaternions (w ≥ 0), what a transform does to a point, matrices as lists of
  their rows, grids row-major, and every index 1-based.

Every key is always written, whatever the kind holds (`length_roles` may be
`{}`), and a file missing one is refused: nothing defaults. In memory the
units are a [`VgioStyle`](@ref VisualGeometryToolkitIO.VgioStyle), and
`VgioStyle(record)` is the style a parsed record declares.

**Versions.** A version is `v<major>.<minor>` (a string written before IO
0.10, `v<n>`, is `v<n>.0`). The major moves when a record's meaning changes or
a key is renamed or removed; the minor moves when a kind gains an optional key.
A reader reads its own major at any minor: in a file of a higher minor it
tolerates the keys that minor added, and in a file of its own or a lower minor
it refuses an unknown key. IO 0.10 moved every kind up a major version (the
envelope is new for all of them).

## The native kinds

| Kind | VGC value | Read | Write |
|---|---|---|---|
| `vgio/cameras/v6.0` (the camera store) | `Dict{Symbol, …}` of name → `BoundedCamera`, central or non-central (unposed, like a COLMAP "camera"): a `CameraModel` of any lens family, a `SplineLens` one included, or a `NoncentralSplineModel`; or an `Attributed` one with its `(; calibration_record)` | `load(Dict{Symbol, BoundedCamera}, path)`; one camera: `load(BoundedCamera, path; name)` or `load(CameraModel, path; name)` (refused for a camera whose lens coverage is not its whole sensor) | `save(path, store)`; one camera: `save(path, camera; name)` |
| `vgio/transforms/v5.0` | `Vector` of labeled `EuclideanMap{3, …, S, D}`, or `Uncertain` ones | `load(Vector{EuclideanMap}, path)` | `save(path, poses)` |
| `vgio/rig/v10.0` | `Rig{F, M}` holding sensors (a tuple of `BoundedCamera`s, central or non-central, each with its lens coverage), targets (a `Dict` of `PosedTarget`s), or both | `load(Rig, path; resolve)` | `save(path, rig)` |
| `vgio/target/v6.0` | one `AbstractTarget` (`PointTarget`, `PlanarTarget` of any `TargetCalibration`, `Uncertain` or not, stated by its points, its reference or its layout), or an `Attributed` one with its `(; calibration_record)` | `load(AbstractTarget, path; resolve)`; a planar one: `load(PlanarTarget, path)` | `save(path, target)` |
| `vgio/trajectory/v5.0` | `Trajectory{S, D}`: labeled poses per frame index, with images, times and recovered flags | `load(Trajectory, path)` | `save(path, tr)`; exporters `save_tum_trajectory`, `save_euroc_trajectory` |
| `vgio/target_poses/v3.0` | `TargetPoses{F}`: per frame, `TargetFrame{id} → F` poses; `pose(tp, t, id)` | `load(TargetPoses, path)` | `save(path, tp)`; exporter `save_bop_scene` |
| `vgio/target_homographies/v2.0` | `TargetHomographies{F <: PixelFrame, H}`: per frame, each target's plane → pixel `HomographyMap` (plain or `Uncertain`), written as the parametric record `{"type": "homography_map", "parameters": [9]}` of its canonical entries (unit Frobenius norm, `h33 ≥ 0`; an 8×8 `covariance` in the `HomographyManifold8` chart), before the lens of the frame's `sensor`'s camera (a `RadialHomography`'s `H`); `targets` are the `content_id`s (a rig's or a single target's) its ids resolve in | `load(TargetHomographies, path)` | `save(path, th)` |
| `vgio/observations/v3.0` | `TargetObservations{F}`: per frame, per target, the observed points' indices and points — pixels in a `PixelFrame` or 3D points in a named 3D frame, plain or `Uncertain`, with an optional `weights` column; `targets` the `content_id`s of rigs or of single unposed targets | `load(TargetObservations, path)` (IO 0.9's `target_points2d`, `target_points3d` and `target_detections` files too, through their version steps) | `save(path, obs)`; exporters `save_correspondences`, `save_trc` |
| `vgio/detection/v1.0` | detections per image: `Dict` of image name → a vector of detections (`AbstractDetection`: `Attributed{Uncertain{<:FeatureFrame}}`, `PointFrame` to `AffineFrame`, plain or uncertain) | `load(Dict{String, <:AbstractVector{<:AbstractDetection}}, path)` | `save(path, detections)` |
| `vgio/capture_settings/v2.0` | `CaptureSettings`: per frame, the exposure (time, f-number, ISO) and the nominal lens settings (focal length, focus distance); the body and lens `DeviceIdentity`, `Attributed` with each one's source (`"exif"` or `"config"`), the EXIF lens claim a stated lens overrode, and optionally the `sensor` (the camera's name) | `load(CaptureSettings, path)` | `save(path, Attributed(cs, (; identity_sources = (body, lens)[, exif_lens, sensor])))` |
| `vgio/sensor_noise/v1.0` | `SensorNoise`: a body's noise at one ISO, keyed by `device.body`; each field (`electrons_per_dn`, `read_noise`, `black`, `dark`, `defects`) with its source and standard error, or absent; a master dark's maps and the defect mask are raster payload bodies beside the record (TIFFs of the raster profile), bound by their SHA-256 | `load(SensorNoise, path)` | `save(path, noise)` (writes the artifacts too) |
| `vgio/vignetting/v2.0` | an `AbstractVignetting`: `TensorProductVignetting` (its knots stated), `RadialVignetting` (its `chart` always stated) or `LogVignetting`, positions in any registered pixel frame (`units.pixel`); rebuilt bit for bit | `load(AbstractVignetting, path)` | `save(path, v)` |
| `vgio/radial_profile/v1.0` | a `BinnedRadialProfile`: a measured profile binned by its chart's coordinate (a flat field by field angle), its chart's centre and radius in `units.pixel`, its edges in `units.angle` | `load(BinnedRadialProfile, path)` | `save(path, p)` |
| `vgio/environment_light/v2.0` | an `EnvironmentLight` of a `SphericalHarmonicMap`: centre and radius in `units.length` (a plain light refused), coefficients per channel in `units.radiance`, the degree `max_degree` and the basis convention stated (`real_orthonormal_no_condon_shortley`) | `load(EnvironmentLight, path)` | `save(path, light)` |
| `vgio/blur/v2.0` | name → `GaussianBlur`, each one's `covariance` in the storage it holds (round, axis-aligned, general) and its PLANE: `pixel_frame` (or `"frame": "image:<id>"`) on a sensor, in px²; `"frame": "target_plane"` on a print, in `units.length`² | `load(Dict{Symbol, GaussianBlur}, path)` | `save(path, blurs)` |
| `vgio/panorama/v2.0` (a panorama's results: PanoramaStitching's) | the NamedTuple `(; camera, rotations, vignetting, gains, fit)` — no VGC type of its own: a `BoundedCamera` (the camera store's body), a `Trajectory{CameraFrame, WorldFrame}` of zero translations (the trajectory body: the camera only rotates; other frames and a translation that is not zero refused), an `AbstractVignetting` (the vignetting body; or `nothing`, a panorama with no Core vignetting), the frames' exposure gains (`K × C`, row k frame index k of the rotations' `1:K`, written per frame index `{"frame_index", "gain"}`; or `nothing`) and the bundle's termination record `fit` (`{"retcode", "stop_rule", "iterations", "rms", "n"}`, read back as a `FitResult{Nothing, Nothing}`; or `nothing`) | `load_panorama(path)` | `save_panorama(path, camera, rotations, vignetting, gains[, fit])` |
| `vgio/raster/v2.0` (a Float32 or UInt8 raster: a JSON header and its TIFF payload, bound by SHA-256; v1 through its step) | the NamedTuple `(; values, frame, nan)`: a `Matrix{Float32}` or `Matrix{UInt8}` (rows, columns; row 1 on top) read bit for bit, NaN included; its frame a chart rectangle `(chart = :page \| :texture, area)` (the design chart, `area` a `Rect2` in metres) or a `PixelFrame`; what a NaN means (`:unobserved`, or `:none`) | `load_raster(path)` (the payload's SHA-256 checked before it is decoded) | `save_raster(path, values; frame, nan)` |
| `vgio/calibration/v3.0` (`calibration.json`, the manifest) | the record files of one calibration result bound by SHA-256, of any owner (`<owner>/<kind>/v<major>[.<minor>]`): each with its `schema`, `role` (`"authority"` or `"derived"`) and `derived_from`; `result_kind` `"metric"`, `"projective"` or `"image"`, an optional `CameraModelSpec` `model`, closed `metrics`, `provenance` | `load_manifest(dir)`: every file hash-checked, VGIo's kinds validated, another owner's when it has a [`validate_record`](@ref VisualGeometryToolkitIO.validate_record) method | `save_manifest(dir, files; result_kind, model, metrics, provenance)` |
| `vgio/units/v3.0` (`vgio_units.json`, beside a COLMAP model) | the world unit of a COLMAP model directory (`UnitContract`, or `UnknownScale`) | `load(ColmapModel, dir)` reads it | `save(dir, m::ColmapModel)` writes it |
| `export/correspondences/v2` (flat 2D–3D pairs per frame and target; not a vgio kind) | an export of a 2D `TargetObservations` and its target rigs; there is no reader | — | `save_correspondences([to_frame,] path, obs, targets...; resolve)` |

### Observations

A `vgio/observations` file holds, per frame, one set per observed target: the
points' indices (1-based), its points in the one point vocabulary — `xy` a
pixel, `xyz` a 3D point in `units.length`, with a `covariance` where the value
is `Uncertain` (px², or `units.length`²) — and, where the set has them, its
`weights`:

```json
{"schema": "vgio/observations/v3.0", <envelope>, "entity": "target_point",
 "targets": ["<content id of a rig, or of a single unposed target>"],
 "frames": [{"frame_index": 17, "image": "DSC0017.png", "sensor": "cam1",
             "observations": [{"target": "b89", "index": [1, 4],
                               "points": [{"xy": [u, v], "covariance": [[…], […]]}, {"xy": [u, v]}],
                               "weights": [1.0, 0.5]}]}]}
```

`entity` says what an observation names: a `target_point` (the target's id
and the point's index) or a `landmark` (a scene point by its index; its sets
carry no `target`). VisualGeometryToolkitCore's `TargetObservations` holds
target points, so a `landmark` file is refused on load until a VGC container
holds landmarks. A 3D file states `"frame"` (a 3D frame name) and its points
are `xyz`; its `targets` may be absent. A 2D file requires each frame's
`image` and `sensor` and at least one `targets` identity, a 3D one each
frame's `sensor`. A frame with a `time` needs the file's `time_origin`, an
ISO 8601 date-time with its UTC offset.

### Rasters

A `vgio/raster/v2.0` file is a JSON header beside its values, a TIFF:

```json
{"schema": "vgio/raster/v2.0", <envelope: units.length "mm", units.pixel "colmap", …>,
 "size": {"width": 2480, "height": 3508},
 "element_type": "float32",
 "frame": {"type": "page", "area": [-105.0, -148.5, 210.0, 297.0]},
 "nan": "unobserved",
 "payload": {"file": "albedo.tiff", "sha256": "<hex>"}}
```

The header states everything the values mean: the raster's size (its columns
and rows), its element type (`"float32"`, or `"uint8"`: a mask, which holds no
NaN), its frame — a chart rectangle (`"page"` or `"texture"`: the design
chart, x right and y up, `area` its lower-left corner and widths in
`units.length`; row 1 is its top edge, column 1 its left edge) or an image's
pixels (`{"type": "pixel"}`: value `[r, c]` the pixel in row `r`, column `c`,
in `units.pixel`) —, what a NaN value means (`"unobserved"`; `"none"`: the
raster holds no NaN), and the SHA-256 of the TIFF's bytes, the reserved
`payload` key. A payload whose digest is not the header's is refused before it
is decoded. The payload body `{file, sha256}` and its TIFF are the stack's one
binding of a bulk array to a file: `vgio/sensor_noise` holds its dark maps and
defect mask the same way. A `vgio/raster/v1` header (IO 0.9.5) reads through
its version step.

The TIFF is of one profile, checked when it is written, on the file written,
and when it is read:

- a baseline TIFF: classic (not BigTIFF), little-endian, one image (one IFD);
- one sample per pixel, min-is-black; a float32 raster `BitsPerSample` 32 and
  `SampleFormat` 3 (IEEE floating point), so every Float32 value, NaN included,
  is stored bit for bit; a uint8 raster `BitsPerSample` 8, `SampleFormat` 1;
- `Compression` 1 (none) or 8 (Deflate), both lossless, and `Predictor` 1:
  no floating-point predictor;
- strips, not tiles; `Orientation` 1 when stated;
- no meaning in the tags: the writer sets only the structural tags above, and
  a reader reads only them, so a TIFF another program wrote (Python's
  `tifffile`, its own tags included) reads as a raster's payload.

### File names

A file's name states its kind and what tells copies of it apart; the loaders
never read it (they read `schema`). The stem is the kind and the suffix the
distinguishing name: `observations_<sensor>.json`,
`target_poses_<sensor>.json`, `trajectory_<body>.json`, `targets_<rig>.json`,
`cameras.json`. A name holds no timestamp: the time is the file's
`time_origin` and `time`s, the capture date is the session directory's, and a
rerun goes in its own run-key directory with its provenance. Published and
existing files keep their names.

### Cameras

A camera body is the same in the store, a rig sensor and another package's
record:

```json
{"name": "left", "image_size": {"width": 4240, "height": 2832},
 "K": {"fx": 2400.0, "fy": 2400.0, "cx": 2120.0, "cy": 1416.0, "skew": 0.0},
 "lens": {"equidistant": [], "radial": [[0.1, -0.02, 0.003, -0.0004], []]},
 "pitch": {"width": 0.00376, "height": 0.00376},
 "lens_coverage": {"circle": [2120.0, 1416.0, 1400.0]},
 "calibration_record": {"tool": "…", "date": "2026-01-19", "reprojection_rms": 0.159}}
```

`K` is in `units.pixel`, `lens` is VGC's lens record. A camera bounded by its
physical sensor (`BoundedCamera(camera, sensor::Sensor)`) carries the sensor's
pixel pitch, a length per pixel in `units.length`, and the readers restore the
`Sensor` bound; a camera bounded by a bare pixel extent has no `pitch`.
`lens_coverage` (a circle, an ellipse or a rectangle in pixels) says where light
reaches the sensor; absent, the whole sensor is lit. `calibration_record` says
where the calibration came from (below).

**The store holds any bounded camera.** A non-central grid camera (Schöps et
al.'s, a `NoncentralSplineModel`) is a body of `"type": "noncentral_spline"`
with its gauge `K` and its spline; a `SplineLens` camera is a camera body whose
`lens` is `{"spline": …}`, in the K-normalized plane. Both splines are the ONE
lattice record every spline uses:

```json
"spline": {"area": [x, y, w, h], "grid_size": {"width": gw, "height": gh},
           "present": [[row, [[col_lo, col_hi], …]], …],
           "controls": {"direction": [[x, y, z], …], "point": [[x, y, z], …]}}
```

`area` is the spline's rectangle, `grid_size` its lattice (columns, rows),
`present` (optional) the held nodes as runs per lattice row, 1-based (absent:
every node), and `controls` the held nodes' controls, row by row: `direction`
for every spline, `point` (the line points, in `units.length`) for a
non-central one. A `SplineLens` may add its fitted forward spline, `"forward":
{area, grid_size, controls: {"point": [[x, y], …]}}`. The retired
`vgio/noncentral_spline/v1` record is read as a store of its one camera, named
after its file.

## The foreign formats

| Format | Value | Read | Write |
|---|---|---|---|
| COLMAP model directory (`.bin` or `.txt`) | `ColmapModel`, an adapter; `camera_model` and `rigid_transform(image, contract)` convert its records | `load(ColmapModel, dir; contract)`; supply `contract = UnknownScale()` for an SfM model with no metric scale | `save(dir, m; format)` |
| OpenCV or MATLAB calibration export | `ExternalCalibration`, an adapter holding VGC values | `load_opencv_calibration([file_frame,] path; file_unit)`, `load_matlab_calibration([file_frame,] path; file_unit)`; `convert_opencv_calibration` and `convert_matlab_calibration` convert one to native files | `save_opencv_calibration([to_frame,] path, cal; to_unit, images)` |
| Kalibr camchain (`camchain.yaml`) | `BoundedCamera` | `load_kalibr_camchain([file_frame,] path; camera = "cam0")` | — |
| Schöps et al.'s camera_calibration `intrinsics0.yaml` | `BoundedCamera` of a `SplineLens` camera (central) or of a `NoncentralSplineModel` (the file's corner pixel convention is `SCHOEPS_PIXEL_FRAME`, COLMAP's; K the lattice's gauge) | `load(BoundedCamera, path; file_unit)`: the form is told from the content; a non-central point grid's unit is the caller's | read-only (`save` writes a `vgio/cameras` store) |
| Schöps et al.'s pattern YAML (`create_calibration_pattern.py`) | `PlanarTarget(SchoepsPattern(…); canvas)`, the page its canvas | `load(PlanarTarget, path)` (or `load(AbstractTarget, path)`): the form is told from the content | read-only (`save` writes `vgio/target/v6.0`, the layout as a record) |
| Kalibr's target YAML of an `aprilgrid` | `PlanarTarget(AprilGrid(…))`, the gap `tagSize · tagSpacing` (Kalibr's `tagSpacing` is a ratio) | `load(PlanarTarget, path)` (or `load(AbstractTarget, path)`) | read-only (`save` writes `vgio/target/v6.0`) |
| Schöps et al.'s camera_calibration `dataset.bin` (`calib_data` v0) | `TargetObservations{SCHOEPS_PIXEL_FRAME}`: per image, the points of `target`, each indexed by its feature id + 1 | `load(TargetObservations, path; target)`: the target the ids index is the caller's | read-only |
| COLMAP `database.db` | a lazy read handle | `ColmapDatabase(path)`, with the accessors `detections`, `descriptors`, `raw_matches`, `verified_matches` and others (public, so qualify them) | read-only |

See [Third-Party Calibrations](third-party.md) for the OpenCV and MATLAB
readers and conversions.

## COLMAP, for orientation

This section is documentation only; there is no COLMAP exporter for these
kinds. An observation of a target point corresponds to a `points2D` entry of
COLMAP's `images.txt` with its `POINT3D_ID`, where the `(target, index)` pair
plays the role of the `POINT3D_ID`. A rig of targets (the targets of a
`vgio/rig` file) corresponds to COLMAP's `points3D`. It is kept as per-target
frames, with each target's points in its own `TargetFrame` and placed by its
pose. It stores no tracks, because the observations are the tracks.

## Targets

A rig's target member is VGC's `PosedTarget(target, pose)`: the target's geometry
(an `AbstractTarget`) and its pose in the rig frame. How each target body is
written (its `type` is `point_target` or `planar_target`):

- A `PointTarget` is written with its points, `{"xyz": [x, y, z]}` (or `xy`
  for a plane's), each with its `covariance` if it is `Uncertain`.
- A `PlanarTarget` of explicit nominal points is written with those points.
- A `PlanarTarget` of a content-addressed design (a BlobBoards board uid) is
  written by its `reference` alone, since its points belong to the
  design's owner. It reads back unresolved unless the caller supplies
  `resolve`, the owner's resolver (`reference::Symbol -> nominal points`).
- A `PlanarTarget` built from a layout (`PlanarTarget(layout; …)`, which VGC
  records as `layout(t)`) is written by its `layout`; its points are VGC's from
  the layout and are not restated. The record's `type` is the layout's tag in
  VGC's `TARGET_LAYOUT_TAGS`, every key is written, and lengths are in
  `units.length`:

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
lengths (a euclidean's `tx, ty`, the displacements, the heights) in
`units.length`, the rest plain (a euclidean's angle in radians, the log-scale,
the aspect's log-stretch, the skew's shear). `nodes` are positions in the
target's plane, in `units.length`. The thin-plate kinds hold pure bending (no
affine term in the interpolant): a record whose values carry one is refused.

A composition is 2 to 6 kinds in VGC's canonical order
`HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, any
left out, each once — the affine print error `Aspect ∘ Skew` (with a ruler
`Scale ∘ Aspect ∘ Skew`, with a known pose `EuclideanMap ∘ Scale ∘ Aspect ∘
Skew`), `HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew`, … It is written by its
structure, flat: its members are records of their own, listed under `compose`
(the key a lens record lists its stages in sequence under) in
CoordinateTransformations' order, the outer map first.

An `Uncertain` calibration adds `covariance` at the top of the record, n × n
over `parameter_vector` (a composition's covers every member's parameters, in
that order, so its members carry none; a thin-plate kind's has rank 2K − 6 or
K − 3), each entry in `units.length` raised to its length dimension. A
calibration with lengths in plain numbers is refused, as plain points are.

A planar target may state its `placement`, `{"origin": [u, v], "side": s}`:
where the calibration's dimensionless chart lies in the design chart (the chart
point c is the design point `origin + side·c`), its origin and side lengths in
`units.length`. A placed target's calibration acts in that chart, so its
nodes, parameters and covariance are plain chart numbers.

A `PlanarTarget` may also state its `board`, the physical board it is
printed on (parallel to a camera's sensor; the board proper, inside the
sheet's margin and frame included, as BlobBoards' `board_size` — not the
sheet): its size and its lower-left corner in the design chart,
`"board": {"size": {"width": w, "height": h}, "origin": [u, v]}`, every
number a length in `units.length`. It is absent when the target has none.

It may also state its `canvas` (VGC's `Canvas`), the printed page the board
lies on, and the image printed on it: `"canvas": {"size": {…}, "origin": [u,
v], "texture": {"path": p, "sha256": h, "size": {…}, "origin": [u, v]}}`. The
board, the canvas and the texture are rectangles of one chart, the design
chart of the nominal points, each its lower-left corner and size in
`units.length`; the board's placement on the page is `board.origin −
canvas.origin`, derived, not stored. The texture's `path` is relative to the
file that names it and `sha256` is the image file's digest; the readers never
open the image. `texture` is optional, and `canvas` is absent when the target
has none.

A target file may also carry a `calibration_record`, the record of how the
calibration was obtained, with the keys of a camera's `calibration_record`
(`tool`, `source`, `date`, `reprojection_rms`, `n_images`, `n_points`,
`note`, `device`). In memory it is the attribute `Attributed(target, (;
calibration_record))`, so it never shadows the target's `calibration`.

## Reading an earlier version

Every earlier version a published file holds is read. The versions each kind
reads are derived from its version steps by
[`schema_versions`](@ref VisualGeometryToolkitIO.schema_versions), never kept
by hand; the table is generated from it:

```@eval
import VisualGeometryToolkitIO as VgtkIO
using Markdown: Markdown
rows = ["| `vgio/$(r.kind)` | `$(r.current)` | $(join(("`$(s)`" for s in r.reads if s != r.current), ", ")) |"
        for r in VgtkIO.schema_versions()]
Markdown.parse(join(["| Kind | Written | Also read |", "|---|---|---|", rows...], "\n"))
```

`load` runs the file's version steps in memory, one per version
([`migrate_schema`](@ref VisualGeometryToolkitIO.migrate_schema)`(::Val{schema}, tree)`,
run by [`migrate_record`](@ref VisualGeometryToolkitIO.migrate_record)), and
reads the current record they give; the writers write only the current
version, so a file saved from what an old file loads as is current. A load is
silent (an `@debug` names the version). The reason is the published data: a
schema change must not force the archives on Zenodo and Hugging Face (several
GB each) to be uploaded again, so a published file keeps loading.

**IO 0.10's step.** Each kind's IO 0.9 version has one step to its 0.10 major.
It moves the top-level declarations (`length_unit`, `pixel_frame`,
`rotation_encoding`, `transform_convention`, `point_index_base`, `time_unit`)
into the `units` and `conventions` objects, stating what a 0.9 record left
implicit as its version meant it; renames the frame join key `t` to
`frame_index`, the frame `camera_optical` to `camera`, a camera's
`calibration` block to `calibration_record`, the target body tags `points` /
`planar` to `point_target` / `planar_target` and `l_max` to `max_degree`;
writes every size as `{"width", "height"}`, the lattice `present` runs 1-based
and the one lattice record; and writes a stored quaternion with w < 0 as its
0.9 reader read it (the 0.10 reader refuses one). No other number moves. IO
0.9's `target_points2d`, `target_points3d` and `target_detections` become
`observations`, and a `noncentral_spline` record a store of its one camera.
A 0.9 `time_origin` with no UTC offset is refused by name: no offset is guessed.

**The covariance storage (2026-11)** moved every kind that carries a covariance
up a version: a covariance is written in the storage its value holds — rows
(full), `{"diagonal": […]}` or `{"isotropic": λ}` — and read back as that
storage (VisualGeometryToolkitCore's `CovMat` holds an `SMatrix`, `SDiagonal`
or `ScaledIdentity`). An isotropic covariance over a pose's angles and lengths
is written as its diagonal (no single number holds rad² and mm²).

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
("vgio/cameras/v1", "vgio/cameras/v6.0", "cameras.migrated.json")

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
An old grid file (`vgio/generic_camera/v2`/`v3`) is read, never rewritten:
`vgio_migrate` writes its camera store to a new file beside it,
`<stem>.camera.json` for a central one and `<stem>.noncentral.json` for a
non-central one.

**Content identity changes once.** IO 0.10's identity is the 0.10 record under
its current names, in metres ([Content Identity](content-identity.md)), so
every rig's and target's `content_id` changes once. Observation files written
beside an old rig name it by its old identity, and `save_correspondences` and
`load_manifest` resolve them against the old rig file as they are.
[`vgio_migrate_tree`](@ref VisualGeometryToolkitIO.vgio_migrate_tree)`(dir)`
migrates a directory's files, maps each rig's and target's old identities (its
version's, and IO 0.9's) to its new `content_id`, and re-points the
directory's observation and `target_homographies` files at them, refusing any
identity that is neither mapped nor a current rig's. A calibration manifest
of the tree is migrated last, each listed file it rewrote bound again (its
path, SHA-256 and schema). It persists the map in `vgio_migrate_ids.json`,
`{"identities": {old: new}, "definitions": {new: relative path}, "rewritten":
{old path: {"path", "sha256"}}}`, before it puts a file in place, so a run
interrupted after its rigs were migrated re-points correctly when rerun; a
rerun of a migrated tree is idempotent, and every file it leaves as it is gets
a log line saying why.
The command's `repoint --ids DIR/vgio_migrate_ids.json --apply FILE` re-points
an observation file kept outside the tree.

A schema's owner reads its own old files the same way: it adds a step
`VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("<schema>")}, tree)` for
each old version, calls `VisualGeometryToolkitIO.migrate_record(tree)` (or
[`read_record`](@ref VisualGeometryToolkitIO.read_record), which calls it) in
its reader, and adds the file form `migrate_schema(::Val{Symbol("<schema>")},
src, dst)` that `vgio_migrate` runs — a declared extension point
(`CONTRACT_GENERICS`). Its step from a record written before IO 0.10 calls
[`migrate_envelope`](@ref VisualGeometryToolkitIO.migrate_envelope)`(old,
schema)`, which moves the record's declarations into the 0.10 envelope, and,
for a record that declared its units in the retired UnitContract layout
(`{"world", "density", "lengths"}`), [`migrate_units`](@ref
VisualGeometryToolkitIO.migrate_units), which gives its `units` object.

A package whose own kinds are files reads and writes them through the same two
verbs: it adds `VisualGeometryToolkitIO.load(::Type{<:T}, path)` and
`VisualGeometryToolkitIO.save(path, x::T)` methods for its types `T` (BlobBoards'
boards and patterns), both declared extension points (`CONTRACT_GENERICS`), and
writes its records through `write_record`/`read_record`.

## Building blocks for other kinds

Every VisualGeometryToolkitCore value a native kind embeds is written and read
on one serialization style, a `JSON.JSONStyle` that carries the `units` the
enclosing file declares:

- `VgioStyle(; length_unit = mm, pixel_frame = PixelFrame{:colmap}, density = dpi, length_roles = (;))`,
  a writer's;
- `VgioStyle(contract::UnitContract; pixel_frame)`, the same from a
  `UnitContract`;
- `VgioStyle(record)`, the style a parsed record's `units` object declares (a
  reader's).

`JSON.json(x; style)` writes a value, `JSON.lower(style, x)` gives its record
and `JSON.lift(style, T, record)` reads one back from a record read by
property access — a parsed `JSON.Object` or a `NamedTuple` (a `Dict` is not
one); the methods are on `VgioStyle` only. What a plain number is (a pixel is
dimensionless) is the field's tag, `JSON.lift(style, T, x, (; quantity =
:length | :pixel | :density[, role]))`; [Quantities, Frames and
Detections](serialization.md) shows the codec of quantities, sizes and feature
frames.

| Value | `T` in `lift` | Record |
|---|---|---|
| a length, pixel count or density (`Quantity`) | `Quantity`, with the field's tags | a number in `units.length` (decimal-exact; or the role's unit), a pixel count, a density in `units.density` |
| `Size2` | `Size2` | `{"width": w, "height": h}`, plain or with the field's tags |
| a `FeatureFrame` (`PointFrame`, `Disc`, `OrientedDisc`, `EllipticFrame`, `AffineFrame`), plain or `Uncertain` | the frame type, or `Uncertain{F}` | `{"center": [x, y]}` and its shape (`sigma`, `angle`, `shape_matrix`, `affine_matrix`, matrices in rows), an uncertain one's `covariance` over its `frame_dof` parameters, centre first |
| `EuclideanMap{3}`, plain or `Uncertain` | `EuclideanMap{3}` | `{"rotation": [w, x, y, z], "translation": [x, y, z], "covariance": …}` |
| `Trajectory` | `Trajectory` | the `vgio/trajectory` body without its envelope: `{"from", "to", ["time_origin"], "frames": […]}` |
| `CameraModel` | `CameraModel` | `{"K": {"fx", "fy", "cx", "cy", "skew"}, "lens": {…}}`, `cx, cy` in `units.pixel` |
| `CameraModel` with a `SplineLens` | `CameraModel` | the camera body, its `lens` `{"spline": the lattice record, "forward"?}` |
| `PointTarget`, `PlanarTarget` | `AbstractTarget` (keyword `resolve`, as `target_from_json`'s) | the target body, a planar target's calibration as the parametric record (below) |
| `AprilGrid`, `SchoepsPattern` | `AbstractTargetLayout` | `{"type": tag, every field}` |
| `Ellipse` | `Ellipse` | `[cx, cy, a, b, θ]` |
| `TensorProductVignetting`, `RadialVignetting`, `LogVignetting` | `AbstractVignetting` | the `vgio/vignetting` body, `{"type": …, …}`, positions in `units.pixel` |
| `BinnedRadialProfile` | `BinnedRadialProfile` | the `vgio/radial_profile` body, its chart's centre in `units.pixel` |
| `EnvironmentLight` of a `SphericalHarmonicMap` | `EnvironmentLight` | `{"center": [x, y, z], "radius": r, "map": {…}}`, lengths in `units.length` (the `vgio/environment_light` body without its envelope) |
| `GaussianBlur` | `GaussianBlur` | `{"pixel_frame": "colmap" \| "frame": "image:<id>" \| "target_plane", "covariance": …}` in the storage it holds: px² on a sensor, `units.length`² on a print (read back as m² quantities); the plane is the blur's `CovMat` frame, read back with it |

A target's calibration is written and read as part of its target's body, not
by a `lower` of its own: VisualGeometryToolkitCore's `TargetCalibration` holds
every planar `EuclideanMap`, so a `lower` on it would write any 2D map left in
a record (a pixel-frame conversion) as a target calibration. A package that
stores a calibration on its own (RadiometricCamera's in-plane print error,
`Aspect ∘ Skew`) uses the named pair
[`target_calibration_json`](@ref VisualGeometryToolkitIO.target_calibration_json)`(style, c)` and
[`target_calibration_from_json`](@ref VisualGeometryToolkitIO.target_calibration_from_json)`(style, record)`,
the record a target body embeds; the reader validates the record first, as a
calibration in a file is.

A uniform cubic B-spline lattice — a `ChartSpline` over a rectangle — is one
record wherever it is stored, `{"area", "grid_size": {"width", "height"},
"present"?, "controls": {"<name>": […]}}` (the spline lens record's splines, a
non-central camera's, a package's own spline such as RadiometricCamera's
aperture radii):
[`lattice_json`](@ref VisualGeometryToolkitIO.lattice_json)`(spline; name, present, length_unit)` and
[`lattice_from_json`](@ref VisualGeometryToolkitIO.lattice_from_json)`(record, Val(K); name, length_unit)`,
the held nodes as runs per lattice row (1-based) and controls that are lengths
in a declared unit.

A height field stored on its own (RadiometricCamera's board warping) may hold
its nodes and its heights in units apart: the file's `units.length_roles`
declare `node` (`"chart"`: plain numbers of the owner's chart, page fractions;
or a length unit) and `height` (a length unit), and
`target_calibration_json` / `target_calibration_from_json` on a style with
those roles (`VgioStyle(; length_roles = (node = :chart, height = m))`) write
and read it. IO 0.9.5's `node_unit` / `height_unit` keys become these roles in
the 0.10 step.

A grey image file is a TIFF through the stack's one TIFF codec,
[`write_tiff`](@ref VisualGeometryToolkitIO.write_tiff) and [`read_tiff`](@ref VisualGeometryToolkitIO.read_tiff)
(libtiff: 8-bit or Float32, one sample per pixel, in strips, uncompressed, LZW
or Deflate); BlobBoards' board TIFFs and the `vgio/raster` payload are
written by it.

A parametric model is written as `{"type": tag, "parameters": [θ₁, …]}`: the
tag from its kind's `TagRegistry` (snake_case of the type's name), the
parameters VisualGeometryToolkitCore's `parameter_vector` in `parameter_names`
order, the lengths among them (`length_mask`) in `units.length` and decimal-exact,
and no `parameters` for a kind that has none. It is read back with
`with_parameters`; a record holding more or fewer parameters than its kind is
an `ArgumentError` naming the tag. A kind whose structure its type does not fix states that
structure beside its parameters (a thin-plate kind's `nodes`).

A package's own kind crosses the file boundary through the same pair the
native kinds do. [`write_record`](@ref VisualGeometryToolkitIO.write_record)`(path, record; style)`
writes a record (its VGC values lowered on `style`) in the vgio file shape:
the record states its own `schema`, `"<owner>/<kind>/v<major>.<minor>"`, and
nothing else of the envelope — `write_record` writes the `generator`, the
`units` object `style` declares and the `conventions` after it — and a NaN or
±Infinity is refused by its field.
[`read_record`](@ref VisualGeometryToolkitIO.read_record)`(path)` parses one
back as its kind's current record, through the owner's version steps
(`migrate_schema`, run by `migrate_record`) when the file is of an earlier
version; a step's refusal names the file and its schema. `read_record(io)`
reads the same from JSON text, and `VgioStyle(record)` is the style to lift
its values on. The owner validates its fields, and may register that check for
the manifest with [`validate_record`](@ref VisualGeometryToolkitIO.validate_record).
A run report that may hold a non-finite statistic is written and read with
`allownan = true`.

`pose_json` / `pose_from_json`, `camera_json` / `camera_from_json` and
`target_json` / `target_from_json` are the same bodies for a length unit or a
pixel frame given directly. All of these are public and not exported. A
camera's body comes in three forms: `camera_json(model::CameraModel, extent,
frame)`, a non-central grid camera's `camera_json(model::NoncentralSplineModel,
extent, frame; length_unit)`, and a bounded camera's
`camera_json(camera::BoundedCamera, frame; length_unit)` — its model's body
with the sensor's pitch and the lens coverage, the body a camera store and a
rig sensor embed —, read back by `camera_from_json(body, frame; length_unit)`
(the model and its extent) and `camera_from_json(BoundedCamera, body, frame;
length_unit)`. Both read an IO 0.9 camera body, or an old grid body, that
another package's record of an earlier version embeds
(RadiometricCamera's `rc/calibration_state/v1` camera) as the camera it
migrates to, so such a record keeps loading.

## Run logs

`append_jsonl(path, row)` appends one JSON line to a run log. It is safe to
call from several tasks and several processes: the line is written with a
single `write` on an append-mode handle, under a per-path lock.
