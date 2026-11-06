# VisualGeometryToolkitIO changelog

## Unreleased

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
  convention stated) and `vgio/blur/v1` (name → `GaussianBlur` on a sensor, px², in the storage each holds).
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
