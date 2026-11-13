# VisualGeometryToolkitIO changelog

## 0.9.4 (unreleased)

Additive: one new kind and one more frame in the detection codec. No existing file, writer or reader changes.

- **`vgio/panorama/v1`: a panorama's results** (PanoramaStitching's: the owner's ruling of 2026-11-13, "everything an
  experiment produces or consumes is saved and loaded through IO"). `save_panorama(path, camera, rotations,
  vignetting; gains, length_unit = mm, provenance = (;))` and `load_panorama(path) -> (; camera, rotations,
  vignetting, gains)`, both public. The kind is composed ONLY of bodies other kinds already write, each through its own
  lower and lift: the camera store's bounded camera body (`camera`), the trajectory body (`rotations`: per frame index
  its pose camera → world, with its image and time; no IO body holds a rotation alone, so every translation is a zero
  length, and a translation that is not zero is refused on write and on read) and the vgio/vignetting body
  (`vignetting`). `gains` (required keyword: `nothing`, every frame at one exposure, or the `K × C` exposure gains,
  `gains[k, c]` of frame index `k` in colour plane `c`, every one positive) is written per frame index `t` as plain
  numbers, `{"t": k, "gain": [g_R, g_G, g_B]}`: no IO body holds a gain. Its validator (`validate`, the readers and the
  bytes before every write) composes the three bodies' tables: `_TRAJECTORY_BODY` is the trajectory file's table
  without its schema and declarations (the trajectory kind's own table is now built from it; the same checks). A
  panorama has no VisualGeometryToolkitCore type of its own, so its value is the NamedTuple of the four VGC values,
  through a pair of verbs as the calibration manifest's (`save_manifest`/`load_manifest`).
- **`PointFrame` in the detection codec** (`json_value` / `from_json`; VisualGeometryToolkitFeatures' Förstner
  detections, `Attributed{Uncertain{PointFrame}}`): the family's 2-DOF frame, `{"center": [x, y]}`, plain pixels, and an
  uncertain one's 2×2 covariance under `covariance` on the leaves an uncertain point (`Uncertain{Point2}`, a
  `vgio/target_points2d` point's `xy` and `covariance`) is written with: rows, `{"diagonal": […]}` or
  `{"isotropic": λ}`, the storage it holds; `null` when not finite (Förstner's unclassified window). One encoding of a
  point and its covariance. A frame object with a `center` alone reads as a `PointFrame` (it was refused before: a
  `Disc` without `σ`); a point frame's covariance under the other frames' `cov` is refused. A Float32 frame is written
  exactly and reads back as the same numbers in Float64; metadata read as parsed (a Symbol as a string, a `Point2` as
  an array), as for every frame.

## 0.9.3 (unreleased)

A patch: no API change.

- **Julia 1.13** (the owner's ruling of 2026-11-13: the whole stack requires 1.13): `[compat] julia = "1.13"` in the
  package and in every environment of its workspace; CI runs 1.13 only.
- **Every `[compat]` floor raised to the gated resolution** (the owner's ruling of 2026-11-13): a stack dependency at
  its registry latest or its in-wave version, a third-party one at the version the release gate resolved (a floor
  only, the semver upper bound kept); the monorepo's Downgrade workflow is deleted.
- **Internal: the private `_json`, a forward to `JSON.json`, is deleted** (the owner's no-shim order); its callers
  (the exporters' sorted JSON, the JSONL writer) call `JSON.json` directly. The same bytes.

## 0.9.2 (unreleased)

### Fixed

- **An old grid body embedded in a record reads.** `camera_from_json` (both forms) took any body with a `type` for
  the current non-central body, so the old grid bodies (`"type": "CentralGenericModel"` or
  `"NoncentralGenericModel"`, `direction_grid`, `point_grid`, `present`, `nominal_intrinsics`, `domain`) that
  RadiometricCamera's `rc/calibration_state/v1` and `rc/radiometric_camera/v1` store as their camera failed with
  `KeyError: key "spline" not found`. They now read as the camera they migrate to, through the one conversion an old
  grid file (`vgio/generic_camera/v2`/`v3`) and an old rig sensor take: a central one as a `CameraModel` with a
  `SplineLens`, a non-central one as a `NoncentralSplineModel` (its line points in the enclosing record's length
  unit), the domain the lens coverage; bit for bit the camera the whole-file migration of the same body gives. An
  old grid body is read in the colmap pixel frame, the frame of every old grid body, whatever frame the call names.

## 0.9.1 (2026-11-12)

Additive patch: public names for two private functions a downstream package (RadiometricCamera) called by their
private names; behaviour unchanged, the private names gone (no alias).

- **`with_provenance(body, provenance)`, public** (was `_with_provenance`): a record body with the writer's
  free-form provenance appended (the body itself when it is empty), as every `save` stores it; for a kind of another
  package written through `write_record`.
- **`metres(x, u::Units)`, public** (was `_metres`): a file number in the unit `u` as plain metres, `read_length`
  stripped (decimal-exact for a power-of-ten unit).

## 0.9.0 (2026-11-12)

On VisualGeometryToolkitCore 0.9 (a bounded camera's lens coverage).

- **`vgio/calibration/v2`: a manifest's model names the coefficients held, `"fixed"`** (VisualGeometryToolkitCore's
  `LensSpec`, the owner's ruling). A `vgio/calibration/v1` manifest is read through its version step: its model's
  `"free"` (the coefficients moved) becomes the complement `"fixed"` within the lens's `parameter_names`, the same
  LensSpec; it is never rewritten on disk, and `save_manifest` writes v2.
- **The exporters' SplineLens refusal names `approximate_camera`**: a SplineLens camera has no OpenCV/COLMAP form,
  and the message now points at `VisualGeometryToolkitEstimation.approximate_camera(KannalaBrandt{4}, camera, region)`
  (Estimation 0.9's `fit_camera` is linear only; its approximation from no guess is `approximate_camera`).

- **The 2026-09 hand-eye record reads** (owner ruling 2026-11-11): a `vgio/transforms/v2` file of the one layout
  bmvc2026's hand-eye writer wrote for CaliBlobs icra2026's motion-capture runs — `camera_rig → camera_optical`, then
  one `printed_board_<id>` or `scene_<c>` → `board_rig` per marker — reads through a version step
  (`migrate_schema(::Val{Symbol("vgio/transforms/v2")}, tree)`) as the frames the current writer names
  (`body:camera_rig`, `camera_optical`, `target:<id>`, `target:scene_<c>`, `body:board_rig`), every number as it
  stands. Saving writes `vgio/transforms/v4`. Any other v2 file is refused as before (its names are free strings).
- **CaliBlobs' 2026-09 track segment reads** (owner ruling 2026-11-11, Option A): a `vgio/trajectory/v2` file of the
  layout CaliBlobs icra2026's motion-capture runs wrote (`track:<s>` → `camera`, the frames' images and the segment's
  `component` in its provenance) reads through a version step as `target:scene_<c>` → `camera_optical` (the gauge
  named by its component, as the run's hand-eye record names it; v2 recorded no gauge board), each frame with its
  image, every pose as it stands. Saving writes `vgio/trajectory/v4`. Any other v2 trajectory is refused as before.

- **A radial vignetting body states a non-linear chart**: `vgio/vignetting/v1`'s `radial` body carries
  `"chart": "projective"` for a profile on the projective chart (VisualGeometryToolkitCore's
  `RadialVignetting(…; chart)`), so it reads back as itself. A linear profile's body has no `chart`, byte for byte the
  body IO 0.8 writes and reads (its validator refuses an unknown key under the same schema string), as the camera
  body's `pitch` is absent for a bare extent; a body stating `"chart": "linear"` reads too.

- **`load` and `save` are contract generics** (`CONTRACT_GENERICS = (migrate_schema, load, save)`): a package whose own
  kinds are files adds `load(::Type{<:T}, path)` and `save(path, x::T)` methods for its types (BlobBoards' boards and
  patterns), so every kind of the stack is read and written through the two verbs. The kinds guide states the
  extension point.

- **New: `Record`, `snap_integer`, `MATRIX_LAYOUT`** (public, not exported): what a lift reads (a parsed JSON object
  or a NamedTuple, read by property access; was `_Record`), a file number as a record writes it (an integral value as
  an `Int`; `Integer` and `Rational` methods beside the float one; was `_snap_integer`) and the matrix layout a record
  declares (`"rows"`; was `_MATRIX_LAYOUT`). BlobBoards' lifts and CaliBlobs' session record restated them. A `Bool`
  is not a number and is returned unchanged (`snap_integer(true) === true`, was `1`), and an integer past `Int`'s range
  (a large `UInt64`, a `BigInt`) is kept as it is (was an `InexactError`).

- **Internal: a stored grid's calibrated area is read as a `Rect`** (from its inclusive pixel bounds) and its size as a
  `Size2`, the one spelling each that Core's SplineLens constructors now take. Values unchanged.
- **Internal: an old grid body's missing K is detected by Core's `has_spline_gauge`**, not by matching the text of
  Core's refusal, and a K given to `load(…; intrinsics)` is written into the parsed record directly (the record was
  re-serialized to JSON and parsed again). The refusal's wording is IO's own; behaviour unchanged.
- **Internal: one lowering, one spline-lattice reader, a body by dispatch.** The lens record's `_plain` was `_lower`
  under another name (gone); both spline lens schemas read a record's grid size, present runs and area through one
  `_lattice_from_json`; a bounded camera's model body is `_model_body` by the model's type (was an `isa` branch over
  `_camera_body` and `_noncentral_body`). Records unchanged.
- **Breaking: the COLMAP database's raw matches are `raw_matches(cdb, i, j)`** (was `matches`), beside
  `verified_matches`. `matches` is VisualGeometryToolkitCore's plot of matched points; one public name had two meanings.
  Three private helpers that shared a name with Core's are renamed after what they do (`_CheckedSpec`, `_K_SPEC`,
  `_export_stamps`), and the frame pair of a pose is Core's `map_frames` (the copy here is deleted).
- **New: the public camera bodies of a bounded camera and of a non-central grid camera.**
  `camera_json(camera::BoundedCamera, frame; length_unit)` is the body a camera store's entry, a rig sensor and the
  non-central record embed — the model's body, the sensor's pitch and the lens coverage (which `camera_json` of a
  `CameraModel` does not carry) — and `camera_from_json(BoundedCamera, body, frame; length_unit)` reads it back,
  checked against its table. `camera_json(model::NoncentralSplineModel, extent, frame; length_unit)` is the non-central
  body, and `camera_from_json(body, frame; length_unit)` reads either kind's `(model, extent)`. The store, the rig and
  the non-central record write and read through these one implementation.

- A stored camera's `BoundedCamera` and a loaded target's geometry are read with Core's `value_of` (the private
  `_bounded` is gone); nothing a caller sees changes.

- **Breaking: a target's placement reads back as Core's `SimilarityMap{3}`** (scale the chart's side, the identity
  rotation, the translation `(origin, 0)`, plain metres), formerly a `ChartPlacement` with lengths; the record keeps its
  `"placement": {"origin", "side"}` keys, so every file reads as before. Writing a placed target without lengths
  (no unit form: its placement's numbers would be read as metres) is an `ArgumentError`.
- **Breaking: an old grid file whose lattice has no complete 4 × 4 window at the gauge's pixel, and no
  `nominal_intrinsics`, is refused** (Core's gauge: no scale to derive K from), where 0.8 loaded it; supply K with
  the new keyword `load(BoundedCamera, path; intrinsics = K)` (also `load(Dict{Symbol, BoundedCamera}, …)` and
  `load(CameraModel, …)`; an old rig's grid sensors take one per sensor id, `load(Rig, path; intrinsics =
  Dict(:id => K))`), used only when the record holds no K (a record that holds one refuses it); the file is
  never edited. With a stored nominal pinhole it loads as before.
- **Breaking: the OpenCV calibration JSON declares its pixel frame under `pixel_frame`**: `save_opencv_calibration`
  writes `"pixel_frame"`; `load_opencv_calibration` reads it, and also the legacy key of files written before 0.9
  (never both: an `ArgumentError`); a file on disk is never rewritten. Every caller of Core's `CameraModel(K; …)`
  calls it with `pixel_frame` (formerly `image_origin`).
- **Breaking: `vgio/cameras/v5`**: a stored camera may carry an optional `lens_coverage` — `{"circle": [cx, cy, r]}`,
  `{"ellipse": [cx, cy, a, b, θ]}` or `{"rect": [x, y, w, h]}` (a letterboxed or padded image), pixels in the store's
  pixel frame, moved into the canonical frame on read as K's principal point is — read back as its `lens_coverage`;
  absent, the whole sensor is covered. A v4 store reads through its migration (its schema line alone; one that
  already holds a `lens_coverage` was not written by v4 and is refused), with the whole sensor covered; `save` writes
  v5. `JSON.lower`/`lift` pairs for `Circle` and `Rect2` join the `Ellipse` one.
- **Breaking: the spline lens (VisualGeometryToolkitCore 0.9 folds the central grid camera into `CameraModel` as a
  `SplineLens`).** A SplineLens camera is stored as any camera, in `vgio/cameras/v5`: its camera body's `lens` is
  `{"spline": {"area", "grid_size", "present"?, "controls", "forward"?}}` — the backward spline over the K-normalized
  plane, the held nodes as runs per lattice row and their controls row by row, the forward spline exactly when it has
  been fitted (by Estimation's `fit_camera(SplineLens, …)`; a 0.8 `SplineProjection`'s spline is never stored as one:
  its chart is about another ray — refit it) — and K is the body's own (a gauge, skew 0). It reads back bit for bit,
  and moves between pixel frames by its K alone.
- **A stored spline camera's K is refused unless it is a skew-0 gauge** — a `{"spline": …}` lens in a camera store or a
  rig sensor, and a non-central record — by Core's one check of a spline camera's K, when the camera is built (the
  non-central record's own copy of the check is gone). A skewed spline lens record once loaded silently.
- **Breaking: `vgio/noncentral_spline/v1`** — the old grid kind `vgio/generic_camera` renamed after its `type` token
  (its v4 was never released) — holds the non-central grid camera alone, a
  `BoundedCamera{<:NoncentralSplineModel}`: `type` "noncentral_spline" (a token of the format, not a Julia type name), the extent, K and
  `{"spline": {"area", "grid_size", "present"?, "directions", "points"}}` (points in `length_unit`), with an optional
  `pitch` and `lens_coverage`; `save(path, camera)`, `load(BoundedCamera, path)`. A record naming the central type is
  refused, pointing at the read below.
- **Old grid files are read, never rewritten** (owner, 2026-11-09): a central `vgio/generic_camera/v2` or `v3` file
  reads as the camera store of its one camera, named after the file's basename (`fisheye_14mm.json` →
  `:fisheye_14mm`) — `load(BoundedCamera, path)` — K its `nominal_intrinsics` (skew 0) else the lattice's gauge, its
  `domain` the camera's lens coverage; a non-central one reads through its step to `vgio/noncentral_spline/v1`. `vgio_migrate` on a central
  one writes a NEW file, the `vgio/cameras/v5` record of its camera at `<stem>.camera.json` beside it, the input
  byte for byte untouched (an existing output, or `out == path`, is an `ArgumentError` naming the path);
  on a non-central one writes its `vgio/noncentral_spline/v1` record to a new `<stem>.noncentral.json` under the same
  rule; `vgio_migrate_tree` does the same for each, in place or not (a rerun finds the output current). Every pixel backprojects as the old models did (the old reader's values,
  `test/fixtures/spline_lens_golden_io.jl`).
- **Breaking: `vgio/rig/v9`**: a sensor's camera body carries its `lens_coverage` (a camera store's form, in the rig's
  pixel frame), which `save` used to refuse; a v8 sensor's old grid body becomes its SplineLens camera's body or the
  non-central body. A rig of a v5–v8 version holding a grid sensor reports the identity its version minted (its old
  identity tree) as its `old_ids`. Every rig sensor is a `BoundedCamera`, a grid camera's included; a grid sensor
  is no longer bound to the COLMAP pixel frame.
- Schöps' `intrinsics0.yaml` reads through `load(BoundedCamera, path; file_unit)`: a central model as a SplineLens
  camera, a non-central one as a `NoncentralSplineModel`, K the lattice's gauge.
- `load(CameraModel, path)` refuses a camera whose lens coverage is not its whole sensor, rather than drop it.
- The exporters to OpenCV and COLMAP (`opencv_parameters`, `colmap_camera`/`colmap_cameras`,
  `save_opencv_calibration`) refuse a SplineLens camera with one message: it has no OpenCV/COLMAP form.
- Removed: `load(CentralGenericModel, path)`, `load(NoncentralGenericModel, path)`, `save(path, m)` of a grid model
  (VisualGeometryToolkitCore 0.9 retired both types); the identity tree's `grid_model` form of a grid sensor (a
  SplineLens sensor is identified by its lens record, a non-central one by its `noncentral_spline` record).

## 0.8.0 (2026-11-09)

On VisualGeometryToolkitCore 0.8 (its `RadialHomography{T, N}`, `TargetHomographies` and
`PLANAR_MAP_TAGS` are this release's records' types); nothing of IO's own is removed.

- **New kind `vgio/target_homographies/v1`** (`save`, `load(TargetHomographies, path)`,
  `load(TargetHomographies{F}, path)`): per frame, each target's plane (mm) to
  pixel `HomographyMap`, against the camera the frame's `sensor` names, before
  its lens (VGC's `radial_homography(camera, H)` pairs them, for a projective
  camera and for a metric one with `fx == fy` and no skew). The pixel frame is
  the value's type; `targets` are the `content_id`s its ids resolve in. Each
  homography is ONE `JSON.lower` / `JSON.lift(::VgioStyle, …)` pair on the
  parametric record — `{"type": "homography_map", "parameters": [h11 … h33]}`,
  VGC's parameter protocol of a `HomographyMap` (nine entries row-major,
  canonical: unit Frobenius norm, `h33 ≥ 0`) — and an `Uncertain` one's 8×8
  `covariance` in the `HomographyManifold8` chart, in the storage it holds. Its
  frames are given by where it sits. The minimal-form rule (style.jl) names it.
- **`content_id` of a single unposed target** (a `vgio/target` path, an
  `AbstractTarget`, or an `Attributed` one): the SHA-256 of its `target` record,
  distinct from every rig's. A `PosedTarget` is an `ArgumentError`.
- **`target_detections` names single targets** (no new version): its `targets`
  may be the content id of a single unposed target (`vgio/target/v5`) as well
  as a rig's; the reader always took any content id, so a released reader reads
  such a file as written. `vgio_migrate_tree` resolves a measurement's ids
  against the tree's current single target files as well as its rigs (0.6.0
  logged such a file `FAILED`), and the manifest checks a `target_detections`
  file's targets like the other measurements'.
- **`vgio_migrate_tree` leaves every measurement file** (`target_points2d`, `target_points3d`,
  `target_detections`, any version) to its re-pointing step, which rewrites an earlier version as the
  current one. 0.6.0 migrated an earlier one by its file form in the first loop, so with `in_place = false`
  a file naming a stale rig id kept that id in its `<stem>.migrated.json` and the second loop found it
  "already re-pointed" (a test pins it now; checked against 0.6.0's order).
- **New kind `vgio/calibration/v1`, the manifest** (`save_manifest(dir, files;
  kind, model, metrics, provenance)`, `load_manifest(dir)`): one
  `calibration.json` binding a result's record files by SHA-256, each with its
  record schema, its `role` (`"authority"` or `"derived"`) and, for a derived
  file, `derived_from`. `kind` is `"metric"`, `"projective"` or `"image"`;
  `model` is an optional `CameraModelSpec`, written on `VgioStyle` as VGC's
  `toml_value` table and lifted by VGC's reader. A load refuses a changed file by
  name, and a `target_detections` / `target_homographies` / `target_points2d` file naming a targets
  identity or a rig target the listed target and rig files do not hold. A
  measurement names a single target by its `reference`, or by its content id
  when it has none; a rig's target by its id. Written by the kind's own writer
  and reader (`_write_store`, `_read_store`) in a `write_staged` directory, so a
  refused manifest leaves nothing at `calibration.json`.
- **Manifest refusals**: a `kind` of `"image"` listing a `trajectory` or
  `target_poses` record; `./a.json` and `a.json` as two entries of one file; a
  listed file that is a symlink resolving outside the manifest's directory.
  A malformed `.toml` file given to `vgio_migrate` (or read for its schema) is an
  `ArgumentError` naming the path, not a `TOML.ParserError`.
- **`vgio_migrate` rewrites a `.toml` record** whose schema has a
  `migrate_schema(::Val, tree)` step and no file form (a schema owner's config):
  parse, the steps, `TOML.print`, and a verified read-back (the same tree, types
  included, or an `ArgumentError`). Comments are dropped: when the file has any
  `#` (a comment line or a trailing comment; a `#` inside a string also counts,
  conservatively) the result's `comments_dropped` is `true` and a `@warn` says
  so; a file without one migrates silently.
- The Schöps YAML decimal reader binds its exponent capture once (a JET report on Julia 1.13).

## 0.7.0 (2026-11-07)

- **Breaking: `sha256_hex` is VisualGeometryToolkitCore's** (public there, with its new byte-vector and stream forms;
  a file's digest is `open(sha256_hex, path)`), so a package below IO (VisualGeometryToolkitFeatures) reads the
  same digest; IO no longer defines or exports it, and drops its SHA dependency. Migration:
  `VgtkIO.sha256_hex(x)` → `VgtkCore.sha256_hex(x)` (same value).
- **`same_value(a, b; length_ulps = 2)` (public)**, the typed-record comparison every migration verifies with
  (formerly the private `_same_value`): a length within `length_ulps` ulp of its metre value, every other leaf bit
  for bit; `length_ulps = 0` is exact everywhere (BlobBoards' board migration check). Two leaves it compared loosely
  now differ: two quantities of different numeric types (`3m` against `3.0m`), and a `Type` (compared by
  `isequal`, not field by field).
- **Named blurs as a body**: `JSON.lower(::VgioStyle, ::AbstractDict{Symbol, <:GaussianBlur})` and
  `JSON.lift(::VgioStyle, Dict{Symbol, GaussianBlur}, body)`, the `blurs` list of a `vgio/blur/v1` file (one block per
  blur, in name order; the lift refuses a repeated name), which `save`/`load` now call and a record of another kind
  nests (RadiometricCamera's per-channel in-focus blur). No file changes.
- The `vgio/vignetting/v1` body's `type` is read and written through VisualGeometryToolkitCore's `VIGNETTING_TAGS`
  (`tensor_product`, `radial`, `log`): the same strings, so no file changes; an unknown type is refused naming the
  registry's tags.

## 0.6.0 (2026-11-06)

- **`read_record(io::IO)`**: a record from JSON text (BlobBoards' `rebuild_board` takes a board's JSON as text),
  the same parse, version steps and refusals as `read_record(path)`. A schema owner's step that refuses a record
  (an `ArgumentError`, or a key its version always held missing) is an `ArgumentError` naming the file (or "the
  JSON text") and its schema, as the native kinds' readers name theirs.
- **One serializer, `VgioStyle` (public).** Every VisualGeometryToolkitCore value a record embeds is written by one
  `JSON.lower(::VgioStyle, x)` method and read by one `JSON.lift(::VgioStyle, T, record)` method (StructUtils' hooks,
  through JSON.jl's public names), on IO's own style, `VgioStyle <: JSON.JSONStyle`, which carries the file's
  `length_unit` and `pixel_frame`: a pose (`EuclideanMap{3}`, `Uncertain` too), a `CameraModel` (its K and lens), a
  generic grid model, a `PointTarget` or `PlanarTarget` body (`JSON.lift(style, AbstractTarget, record; resolve)`), an
  `AbstractTargetLayout` and an `Ellipse`. A target's calibration is part of its body and has no `lower` of its own (one
  on VGC's `TargetCalibration` union would claim every planar `EuclideanMap` left in a record). No method lands on
  those functions for another style, and there is no global `JSON.lower(x)`. Every writer serializes on its own file's
  style, `JSON.json(body; style = VgioStyle(…))` (a required argument of the one write path), so a value left in a body
  is lowered in the unit and pixel frame the file declares; a reader is parse → `migrate_record` → validate → `lift`.
  A parametric record with more or fewer parameters than its kind is an `ArgumentError` naming the tag, its kind's
  count and the record's.
- **The parametric record.** A target calibration kind's `{"type": tag, ["nodes": …], ["parameters": …]}` is written
  generically: the tag from `TARGET_CALIBRATION_TAGS`, the parameters VGC's `parameter_vector` (its lengths, by
  `length_mask`, through the decimal-exact shift), `parameters` omitted for a kind with none (`rigid`); it is read back
  by `with_parameters` on the kind's prototype, and a thin-plate kind (its nodes are structure, its values read back
  unprojected) by its constructor. A layout is the keyed record of its fields, read back by its keyword constructor.
  The per-kind writers and readers are gone.
- **One encoder per minimal form.** A K's five numbers (`fx, fy, cx, cy, skew`) and an ellipse's
  (`[cx, cy, a, b, θ]`) are written in one place each, for the camera body, a generic model's `nominal_intrinsics` and
  `domain`, and the rig identity tree, which now reads a generic model off the writer's own body; a board is a
  rectangle of the design chart, written as the canvas's are.
- **No file changes.** Every kind writes the same bytes as before (every test fixture, and each of the ~11,800 files
  and ~10,000 identity trees the test suite writes, written with the old and the new code and compared), no content
  identity moves, and no schema version moves. The lens record stays VisualGeometryToolkitCore's `lens_record`: it
  already is the generic structure-and-coefficients record (type tags and coefficient arrays, read by
  `lens_from_record`), and a `{"type", "parameters"}` lens would need that structure under another key and a new
  cameras and rig version, removing no code.
- `pose_json`/`pose_from_json`, `camera_json`/`camera_from_json` and `target_json`/`target_from_json` keep their
  signatures; each is the style's `lower`/`lift` for the unit or pixel frame it is given. The readers take a record read
  by property access, a parsed `JSON.Object` or a `NamedTuple` (`camera_from_json(camera_json(…))` round-trips in
  memory); a `Dict` (no property access) is a `TypeError` at the call.

- **The covariance storage (new versions).** A covariance is written in the storage its value's `CovMat` holds — rows
  (full, as before), `{"diagonal": […]}` or `{"isotropic": λ}` — and read back as that storage; an isotropic one over a
  pose's angles and lengths is written as its diagonal. Every kind that carries a covariance moved up a version:
  `target_detections` v2, `target_points2d` v2, `target_points3d` v2, `target_poses` v2, `trajectory` v4, `transforms`
  v4, `target` v5, `rig` v8. An earlier file reads through its version step (its covariances are rows; only the schema
  is renamed) and `vgio_migrate` rewrites it; a rig's identity does not change. Fixtures of each replaced version under
  `test/fixtures/old/`.
- **New kinds, a radiometric camera's values:** `vgio/vignetting/v1` (`TensorProductVignetting` with its knots stated,
  `RadialVignetting`, `LogVignetting`; COLMAP pixels; rebuilt bit for bit), `vgio/environment_light/v1` (an
  `EnvironmentLight` of a `SphericalHarmonicMap`: centre and radius in `length_unit`, a plain light refused; the basis
  convention stated) and `vgio/blur/v1` (name → `GaussianBlur`, each one's `covariance` — the blur is the density
  `N(0, Σ)` — in the storage it holds and in the unit its block states, `"px^2"` on a sensor or `length_unit` squared
  on a print by IO's length rule: a decimal variance reads back exactly, a variance computed in code within an ulp;
  the unit is recorded, the blur's frame tag is not). The three bodies are VGC values on `VgioStyle`, so another package's record can nest
  them: `JSON.lower(style, x)` and `JSON.lift(style, AbstractVignetting, record)`, `JSON.lift(style, EnvironmentLight,
  record)` (centre and radius in the style's length unit) and `JSON.lift(style, GaussianBlur, record)` (`{"unit",
  "covariance"}`: the lift reads the stated unit, px² as plain numbers and a length² as m² quantities); the kinds'
  save and load go through them.
- A target calibration's covariance and a point target's point covariances take the three storage forms too.
- Requires the VisualGeometryToolkitCore whose `CovMat{N, T, M, C}` holds a static matrix (`ScaledIdentity`), with
  `TensorProductVignetting(knots_x, knots_y, P; degree)` and `radius` public.

## 0.5.3 (2026-11-05)

- Requires VisualGeometryToolkitCore 0.5.7 (compat floor raised).

- **`vgio/target_detections/v1`**, VisualGeometryToolkitCore 0.5.7's
  `TargetDetections`: `save(path, td; length_unit = mm, provenance)` and
  `load(TargetDetections, path)`. Per frame (its stamps), per target, the
  point `index` array (1-based, `point_index_base`), the `points`, a
  `covariance` per point iff the points are `Uncertain` (px² as they are;
  3D in `length_unit²` through `write_length`/`read_length`), and the
  `weights` column where the set has one (the only column the kind holds; a
  set with another is refused). A pixel file declares `pixel_frame` and
  requires every frame's `image` and `sensor` and a `targets` identity; a 3D
  file declares `frame` and `length_unit`, requires every frame's `sensor`,
  and refuses plain points (`p .* m`). `payload` reserved and refused, as for
  every kind.
- **`load(TargetDetections, path)` reads a `vgio/target_points2d/v1` or
  `vgio/target_points3d/v1` file**, converted, except a file with a
  (target, index) twice in one frame or with plain and `Uncertain` points
  mixed, which those kinds allow: refused with an `ArgumentError` naming the
  path, frame, target and index, and still read by `load(TargetPoints2D/3D,
  path)`. `load(TargetPoints2D/3D, …)` and their writers are unchanged. `vgio_migrate_tree` and the re-pointing
  of a migrated rig's measurement files handle a `target_detections` file
  as the other measurement kinds.
- **File names** (docs, "Kinds and the I/O Surface"): the stem is the kind,
  the suffix what tells copies apart (`target_detections_<sensor>.json`,
  `target_poses_<sensor>.json`, `trajectory_<body>.json`,
  `targets_<rig>.json`, `cameras.json`), never a timestamp; loaders read
  `schema`, never the name, and published files keep theirs.
- Requires VisualGeometryToolkitCore 0.5.7 (the compat floor is raised with
  Core's version bump at the release: Core's Project.toml is still 0.5.6).

- **`vgio/target/v4` and `vgio/rig/v7`**: a planar target may state its design
  by its `layout` — VisualGeometryToolkitCore's `AprilGrid` or `SchoepsPattern`,
  the record's `type` the layout's tag in `TARGET_LAYOUT_TAGS` (`april_grid`,
  `schoeps_pattern`), every key written (the tag family, the first id, the star
  segments, each Schöps tag's id), lengths in the file's `length_unit` through
  `write_length`/`read_length` — instead of its `reference` or its `nominal`
  points; a body states exactly one of the three. A target built with
  `PlanarTarget(layout; …)` (VGC records it, `layout(t)`) is written by its
  layout and reads back the same target, its points VGC's from the layout.
- **`content_id` does not move**: a layout target enters the identity tree by
  its points, as a nominal one, so a layout file and a nominal file of the
  same design name one target, and every rig without a layout keeps its id.
- **`vgio/target/v3` and `vgio/rig/v6` keep loading**, through their version
  steps (a v3/v6 body is a v4/v7 one as it stands; only the schema changes),
  and `vgio_migrate` rewrites them; a rig's migration reports its id
  unchanged. The writers write v4 and v7 only.
- **The tools' own layout files load directly**, through the typed `load` of
  the target they define (`load(PlanarTarget, path)`, new, and
  `load(AbstractTarget, path)`; the form is told from the content): Schöps et
  al.'s pattern YAML (`create_calibration_pattern.py`) as
  `PlanarTarget(SchoepsPattern(…); canvas)`, its `page` the canvas (lower-left
  corner at `(−pattern_start_x, pattern_end_y − height)` in the design chart;
  the pattern rectangle checked against the layout; no board, which the file
  does not state), and Kalibr's `aprilgrid` target YAML as
  `PlanarTarget(AprilGrid(…))`, its `tagSpacing` ratio read as the gap
  `tagSize · tagSpacing` — the exact product of the two decimals, rounded once
  (0.088 · 0.3 is 26.4 mm, as a file's 26.4 reads). Either saves as a
  `layout` record. Both read YAML comments (a `#` at a line's start or after
  whitespace, outside quotes: Kalibr's stock files carry them), skip a UTF-8
  byte-order mark (the Schöps intrinsics YAML too), and take a list's items at
  column 0 under their key (PyYAML's default dump) as well as indented.
- `vgio_migrate`'s rig identities (`_old_rig_ids`) match rig v4, v5 and v6 by
  name and refuse any other version, so a future identity-changing step must
  state its own.
- Compat: VisualGeometryToolkitCore 0.5.7 (`AbstractTargetLayout`, `layout`,
  `TARGET_LAYOUT_TAGS`).

## 0.5.2 (2026-11-04)

- **`vgio/generic_camera/v3`**: a grid model's held nodes (VisualGeometryToolkitCore
  0.5.4's `present_nodes`) as `present`, runs per lattice row
  (`[[row, [[col_lo, col_hi], …]], …]`, 0-based, inclusive; any shape, holes
  included), and its grids over those nodes only, row-major; written only for a
  model that does not hold every node (a record without it holds every node).
  A model reads back with its mask and NaN in its absent slots, which no
  evaluation reads. The writer writes v3 only.
- **`vgio/generic_camera/v2` keeps loading**, through its version step (a v2
  record is a v3 one that holds every node: only the schema changes), and
  `vgio_migrate` rewrites it.
- **rig/v6 is unchanged**: a generic sensor's body carries `present` the same
  way, so every rig written before reads, writes and is identified
  (`content_id`) as before; a masked model's identity names its held nodes.
- **New kind `vgio/capture_settings/v1`** (`save`, `load(CaptureSettings, path)`):
  per frame index the exposure (time in seconds, f-number, ISO) and the
  nominal lens settings (focal length, focus distance where known) in the
  declared `length_unit`, and the body and lens identities, each with its
  source (`"exif"` or `"config"`; a config-stated lens may keep the EXIF claim
  it overrode). In memory `Attributed(cs, (; identity_sources[, exif_lens]))`.
- **A calibration block may state its `device`** (`{"body": {…}, "lens": {…}}`,
  each a `DeviceIdentity`): the hardware a camera's calibration was made for.
  Additive: every cameras/v4 file without it reads as before.
- Compat: VisualGeometryToolkitCore 0.5.4.

## 0.5.1 (2026-11-04)

Every earlier schema version a published file holds is read again. A reader
that meets one migrates it in memory, version by version, to its kind's
current record and reads that; the writers write only the current version.
The reason: every schema change had forced a re-upload of the published data
(Zenodo, Hugging Face; multi-GB archives), and a published file must keep
loading.

- **Read on load**: `vgio/cameras/v1`, `v2` and `v3`; `vgio/target/v1` and
  `v2`; `vgio/rig/v4` and `v5` (before: refused, naming `vgio_migrate`). A load
  is silent; an `@debug` names the version migrated. `content_id` of an old rig
  file is the identity of the value it loads as, which is its migrated file's;
  `validate` checks an old file as the record it migrates to.
- **One step per version, shared.** `migrate_schema(::Val{schema}, tree)`, the
  extension point's new in-memory form, takes a parsed record of `schema` to the
  next version's record; `migrate_record(tree)` (new, public) runs the steps
  until the record is current. The readers and `vgio_migrate`, which still
  rewrites a file (verified, staged, renamed into place), run the same steps. A
  schema's owner adds the steps of its own kinds and calls `migrate_record` in
  its reader; `migrate_schema(::Val{schema}, src, dst)`, the file form, stays
  the owner's migration for `vgio_migrate`.
- **cameras v1 and v2 restored** (VisualGeometryIo's v1 reader and v2 lens
  table): v1's per-camera records (each in its own `pixel_chart`) and positional
  lens parameters, v2's named ones, as VGC's lens families — BrownConrady as
  `opencv_lens` of OpenCV's shortest vector holding its non-zero coefficients,
  KannalaBrandt, DoubleSphere, Scaramuzza, the division and RationalRadial
  models, and the pinhole and EquidistantFisheye lenses, each the same map (the
  tests check each against VGC 0.x's formula). v1's top-level metadata becomes
  the provenance.
- **A version step refuses a record its version did not write** (a current
  record relabeled old: a v1/v4 target holding `calibration` or `reference`, a
  v2/v5 `composition`, a v3/v5 lens `compose`), as the readers refused a later
  key before.
- **Measurements naming an old rig**: `save_correspondences` resolves the
  identity a v4 rig file (or a v5 one whose affine calibration became its
  factors) was named by against that file.
- Still refused, by name: transforms and trajectory v1/v2 (free-form frame
  names, which have no typed successor), the retired kinds (`observations/v2`,
  `point_observations/v1`, `targets/v2`, `rig/v3`, `session/v3`), and `rig`
  v1/v2 and `generic_camera/v1`, which no published file holds.

## 0.5.0 (2026-11-04)

Breaking: the target and rig records bump with VisualGeometryToolkitCore
0.5's composition (CoordinateTransformations' `∘`) and its gauge-free target
calibration.

- **`vgio/target/v3` and `vgio/rig/v6`.** The calibration kinds are VGC 0.5's:
  new leaf records `euclidean` (a planar `EuclideanMap`, `[angle, tx, ty]`,
  the translation in the file's unit; the angle the one VGC's member stores,
  its `Angle2d`'s, written as it is and read back as an `Angle2d`, so value →
  file → value is exact — no search for an angle that rebuilds a matrix),
  `scale` (`[log_scale]`), `aspect` (`[a1]`) and
  `skew` (`[u12]`), the last three dimensionless; `thin_plate_spline` and
  `height_field` hold pure bending, and a record whose values carry an affine
  term is refused (VGC's constructor would move them). A composition — 2 to 6
  kinds in the canonical order `HeightField ∘ ThinPlateSpline ∘ EuclideanMap
  ∘ Scale ∘ Aspect ∘ Skew`, any left out — is written flat by its structure,
  `{"type": "composition", "compose": [outer, …, inner]}`, in CT's order (the
  outer map first), its covariance over the concatenated parameters in that
  order, and read back right-associated; v2/v5's
  `{"type": "calibration_chain", "members": […]}` listed them in application
  order. The key changed with the order, so the retired form in a current
  record is refused by name (an `ArgumentError` naming `vgio_migrate`), never
  read in the new order. The `"affine"` kind (CT's `AffineMap`) is retired:
  refused by name, naming its factors `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`
  and `vgio_migrate`. A calibration without lengths (the dimensionless
  factors) is written whatever its form; one with lengths must be physical.
- **A planar target's chart placement** (VGC's `ChartPlacement`): an optional
  `"placement": {"origin": [u, v], "side": s}` beside `board` and `canvas`,
  lengths in the file's unit (decimal-exact), refused on a points target. A
  placed target's calibration acts in the board's dimensionless chart, so its
  record — nodes, parameters, covariance — is plain chart numbers, written and
  read bit for bit in any unit. The placement enters the `content_id` tree
  (origin and side in metres). A body without one reads as before.
- **`vgio/cameras/v4`, and `vgio/rig/v6`'s sensors**: a lens's distortion
  stages in sequence (CT's `∘`) are listed in the lens record's `"compose"`
  outer first (the key a composed calibration lists its members under); v3/v5
  listed them under `"chain"` in application order, and a `"chain"` in a
  current record is refused naming `vgio_migrate`. `vgio_migrate` converts a
  v3 store (and a v4 or v5 rig's sensors with its targets). OpenCV's 14-coefficient lens is `Tilt ∘ Additive` and COLMAP's
  `RAD_TAN_THIN_PRISM_FISHEYE` `Additive ∘ Radial` in `opencv_lens` and the
  COLMAP assembly; the exported coefficients and COLMAP parameters are
  unchanged.
- **`vgio_migrate` converts `vgio/target/v2` and `vgio/rig/v5`** (reversing a
  chain's members and permuting its covariance). An affine calibration
  becomes its Iwasawa factors (A = Q·T by QR, det A > 0 required: ϑ =
  atan(Q₂₁, Q₁₁), log s = ½ log(t₁₁ t₂₂), a₁ = ½ log(t₁₁/t₂₂), u₁₂ = t₁₂/t₁₁),
  the same map, its covariance through the factorization's Jacobian; in a
  chain it is replaced by the four in its place. A chain with no current form
  (in-plane kinds out of the canonical order, a Rigid member) is refused, and
  so is a thin-plate spline or height field whose interpolant carries an
  affine term: no exact form exists (the factors compose after the spline,
  and a plane of heights is a shear, not a rigid motion a pose could take),
  so it is re-fitted rather than silently changed. The readers refuse v2/v5,
  naming the migration. A rig keeps its `content_id` unless its value
  changes: the identity tree lists a composition's members and a lens's
  stages in one fixed order, application order (the order v5 held), whatever
  order the record lists them in; a rig whose target held the affine kind
  changes it, and the migration reports v5's id as the old one so
  `vgio_migrate_tree` re-points its measurement files.
- Depends on CoordinateTransformations (the calibration's composition type).
- The calibration record reads the composition's structure from VGC (`members`,
  `TARGET_CALIBRATION_ORDER`, `planar_kind`, `is_pure_bending`) instead of
  restating it, and the v2 → v3 migration factors a retired affine with VGC's
  `iwasawa_parameters_with_jacobian`; records, covariances and messages are
  unchanged (a reflection's refusal is VGC's message, still naming `det A`; a
  chain with no affine member still has its covariance permuted exactly, an
  `Inf` or a `-0.0` entry included, not multiplied through `J Σ Jᵀ`).

## 0.4.0 (2026-11-02)

- The stack moves to the 0.4 series with VisualGeometryToolkitCore 0.4.0; no change to this package's API.

## 0.2.0 (2026-10-29)

- **A `:length` detection's covariance is written in the declared unit
  squared.** Example: a `Disc` detection's `cov` under `kind = :length`. Under
  `MM_PX` it is written in mm², and it is read back the same way. IO 0.1.0's
  `json_value` wrote it raw, in m², with no marker of the change. The same
  JSON read by 0.2.0 under a mm contract therefore comes back 10⁶ times
  smaller. Such a record from 0.1.0 has to be rewritten: multiply each `cov`
  entry by the number of length units per metre, squared. No writer in the
  stack persisted such records.
- **A detection's covariance size is the size of the value.** It is no
  longer tied to the frame type. For example, ASIFT's `EllipticFrame` carries
  a 3×3 covariance. A covariance that is not finite is written as `"cov":
  null` together with its size under `"cov_size"`. A `null` with no
  `cov_size` (IO 0.1.0's record) reads as 3×3, as it did in 0.1.0.
