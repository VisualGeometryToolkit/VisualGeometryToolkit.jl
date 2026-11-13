# IO 0.9.5 duplication checklist (VisualGeometryToolkitIO, additive)

Every concept IO 0.9.5 adds or makes public, the stack searched for it BY CONCEPT before the code was written (Core, IO,
Features, Estimation, Eval, BlobBoards bb-j113 0850b81, PanoramaStitching ps-064, CaliBlobs, LODR, RSDR, and
RadiometricCamera read-only at /raid/data/jpr/rc-080), and what was done. A box is ticked when the final gate is green on it.

| # | concept | considered (file:line) | done | status |
|---|---|---|---|---|
| a | a target's calibration record, alone | IO rig_schema.jl `_target_calibration_json` / `_target_calibration_from_json` (0.9.4 :419, :527); RC records.jl:21-24 (its `inplane_print_error` is IO's record) | the existing pair made public under IO's `<thing>_json` / `<thing>_from_json` names; the public reader validates; the target lift keeps the validated path (`_target_calibration_from_record`). No copy | [x] |
| b | the lattice record (a B-spline lattice over a rectangle, held nodes, controls) | IO camera_schema.jl `_spline_json` / `_spline_from_json` (0.9.4 :125, :150); style.jl lattice pieces `_lattice_from_json`, `_present_runs`, `_present_from_runs`, `_controls_json`, `_controls_from_json` (:174-261); noncentral_spline_schema.jl:82-101; Core `ChartSpline` (chart_spline.jl:177); RC records.jl:52 `radius_grid` (RC's own grid of radii, to move onto this in RC 0.9) | the existing pair made public (`lattice_json` / `lattice_from_json`), moved beside the lattice pieces in style.jl, `length_unit` added for length controls; the lens and non-central records call it. No copy | [x] |
| c | a height field with its nodes' and heights' units apart | Core `HeightField` (target_calibration.jl:645: one form for nodes and heights), IO's `_CHART` form of a placed target (rig_schema.jl `_Chart`), RC warping.jl:79 `warping_field` and records.jl "warping"/"warping_aspect"; io10's units object (w10-io10 validate.jl:300-356, `length_roles`) | the existing parametric height-field record with two optional keys, `node_unit` / `height_unit`, read and written by the one calibration codec (`_reads_back` / `_check_writes_back` / `_calibration_from_record` take the second unit); no second height-field codec | [x] |
| d1 | a TIFF writer and reader | BlobBoards src/authoring/tiff.jl:35 `write_tiff` (libtiff) and src/io.jl:218 `_read_grey_tiff` (libtiff); BB test-only TiffImages (test/items_test.jl:185); nothing else in the stack (grep libtiff/TiffImages/tifffile over every repo) | HOISTED into IO (src/tiff.jl `write_tiff` / `read_tiff`), byte-identical for BB's board TIFFs (bb_tiff_identity.jl); BB's copy deleted by BB's next release (handoffs.md "io095 → bb") | [x] |
| d2 | a binary sidecar bound by its SHA-256 | IO validate.jl:418 the reserved `payload` key; calibration_manifest.jl `_check_manifest_files` (:206), `_is_inside_path` (:34), `_realpath_inside` (:197); rig_schema.jl canvas texture `{path, sha256}` (:286); Core `sha256_hex` (utils.jl:165); io10's sensor_noise artifacts (`{file, sha256}`, interfaces.md) | the reserved `payload` key, its first kind; the manifest's per-file check extracted as `_check_bound_file` and called by both; `_is_inside_path`, `write_staged` reused | [x] |
| d3 | a rectangle of the design chart in a length unit | style.jl `JSON.lower/lift(::VgioStyle, ::Rect2)` (:170); rig_schema.jl `_rect_json` (board/canvas `{size, origin}`); validate.jl `_lattice_problem`'s area check | the `[x, y, w, h]` area through the Rect2 lower/lift, each number through `_metres_json` / `metres`; the area check extracted as `_area_problem`, shared with the lattice | [x] |

Ticked on the final gate, 458f816b (all green; ~/logs/dgx-agent/io095.md). BlobBoards' copy (d1) leaves with BB's port
(handoffs.md "io095 → bb"); until then check_duplication does not pair them (below: 0 findings with bb-j113 in the stack).

## check_duplication (gate prep 12390, VGTK_STACK_PATHS = bb-j113, lodr-09, rsdr-09, cb-release, ps-064)

## Stopped with no code (searched, nothing to merge)

- H8 (same per-pair concept already one body), F9's first pass (superseded by dup-items-f), "one luma" (no Features
  defect: dup-items-f.md).

## Out of scope by the audit (not duplications)

Features ASIFT singular-covariance drop (F6 ruling), `is_stationary`'s gradient form, `pooled_noise_scale` (F8),
CountedLO's log hook (F9), the lens-circle hoist (F10).

## 0.9.2 / Features 0.8.0: the Bayer port (branch `bayer-port`)

The sensor image moved to Core (owner ruling 2026-11-12); the gate of the photosite-radiometry spec:

- [x] ONE stored RAW type: `RawMosaic`, defined and exported in Core only (Features 0.8.0 re-exports none of the moved
  names, nor Core's `feature_frame`, `Cspond`, `csponds`). No second image-with-frame type (`FramedImage` stays a warp's canvas).
- [x] ONE phase rule: `_photosite_color`, `_QUAD_SITES`, `_site_offset` and the one sample-lattice map `_sample_map`
  (Core `src/raw/raw_mosaic.jl`); `channelview(::RawMosaic)`, `bayer_plane_map` of a mosaic and of a `LinearRaw`,
  `green_cell_sites` and the cell kernel read them. The R/B channel maps ARE the bin2 plane maps (`===`, a test).
- [x] ONE linearisation, one public spelling: `linearview` (Core; the scalar `_linear_value` private, used by the cell
  kernel and the view); the green cells are the mean of the linearised view's green channels, bit for bit (a test).
- [x] ONE public path to the green-cell image: `LinearRaw(m; mode = :green_cells)` (the `green_cells(L)` accessor not
  kept); the levels are the mosaic's fields (`_bayer_levels` private).
- [x] No second frame conversion added: the channel maps compose the existing cell-grid map; `pixel_mask(region, map,
  w, h)` and the `SimilarityMap` region actions are new (Core had none: searched `ct_contravariant.jl`, `regions.jl`,
  `image_circle.jl`, `bbox.jl`).
- [x] No second demosaic: `load_raw(mosaic; mode = :demosaic)` runs LibRaw's one AHD on the mosaic's buffer
  (`libraw_open_bayer`), bit-identical to the same buffer's file-path demosaic (a test).
Downstream copies, outside this release (listed for their owners; not release items, so no box): RC `cvpr2027/frontends/scripts/convert_raw.py` (+
  `_night06/_night07` copies: rawpy decode, `(img - black)/(white - black)`, 2×2 binning, `"RGBG"` CFA string, raw-grid
  green mask), `detect_aprilgrid.py` and `green_copy.py` (rawpy demosaic as detection input), `cvpr2027/src/drivers.jl:375`
  (`cfa_string`'s body), `src/calibration_pipeline/io.jl:61-70` `_camera_at` (the size-ratio rescale standing in for
  `raw_frame_map`; spec §4.2); PS `examples/panorama_vignetting/run.jl:530,537`, `src/precompile.jl:14`,
  `test/registration_tests.jl:96` (hard-coded `x ↦ 2x` beside an available `frame_map`); Features
  `test/variance_stabilization_tests.jl:30` (a hand phase tiling in a test fixture).

## 0.9.2 / Features 0.8.0: the other branches

- [x] rc-hoists (041f3665): each new public name is the ONE body Core's or the package's own callers use, the private
  name gone with no alias. `normalized_point`, `bilinear_weights`, `julia_blocks` and `with_parameters(::MappedSurface)`
  in Core; `metres` and `with_provenance` in IO; `image_size` in Estimation. A review searched for a second
  normalized-point map, bilinear helper or image-size reader and found none.
- [x] features-image-circle (dfbe5fc6): `detect_image_circle` moved from RadiometricCamera (one copy, Features). Its
  green plane is `LinearRaw(m; mode = :green_cells).image[:, :, 1]` (its `_green_cell_plane` dropped). Its fit is
  Estimation's `find_ellipse`/`find_circle`, and the map is Core's `raw_frame_map`/`map_estimate`.
- [x] ONE shoelace sum: Core's `signed_area(ring)`, public (was `_shoelace_signed_area`). It is the body of
  `area(::Polygon)`, of the view-overlap clip's orientation, and of `detect_image_circle`'s hole size; Features'
  `_polygon_area` copy is deleted (12f7ce8e, bit for bit: the same loop and order).
- [x] Features 0.8.0 re-exports no Core name: no package reaches one through Features (a review checked every
  monorepo package, test, example and docs page).

## 0.9.3 / IO 0.9.2 / Features 0.8.1

- [x] ONE check for a package extension: Core's public `require_extension`, the up-front check and the fallback
  (the MethodError of a loaded extension at the caller's world age). Features' private `_require_extension` is
  deleted. BlobBoards' copy (`src/extensions.jl`) moves onto it on BlobBoards' own branch.

- [x] ONE set of explicit fused products: `_fma_dot`, `_fma_mul`, `_fma_tmul` in Core `base/fused_products.jl`. The
  spline lenses' 2-D LM, the MappedSurface root and a `Quad`'s plane coordinates compute with them.
- [x] ONE body for a `Quad`'s plane coordinates: `_plane_coordinates`, with the fused dot. The root's copy,
  `_frame_coordinates`, is deleted.
- [x] The chart map's fused traversal (`_map_value`, `_calibration_value`) is a second evaluation path, kept for the
  determinism and held to the map's own call and `transform_deriv`, member by member, by a test
  (`mapped_surface_traversal_tests.jl`).
- [x] Dead code: the private `_chart_affine` is deleted, with its test.
- [x] ONE conversion of an old grid camera body: `_current_grid_body`, shared by IO's embedded-body reader and the rig
  v8 → v9 step. ONE test of the old grid type: `_is_old_grid_type`, used by the migration, the rig intrinsics supply and
  the Schöps reader.

## 0.9.4 (Core alone)

- [x] ONE half-texel chart: the private `_texel_chart` (the size check and a position's 0-based texel-centre
  coordinates) and `_clamp_to_centres` (the clamp onto the centres' rectangle), read by `bilinear_weights` and the new
  public `is_inside_texel_centres`. RadiometricCamera's `texel_weights` restated the chart beside its `bilinear_weights`
  call (RC final audit item 9); it moves onto the predicate on RC's own branch.
- [x] ONE cell-grid frame map: `raw_frame_map(::Type{LinearRaw{mode}})` returns `_CELL_FRAME_MAP`, and the one
  composition `_cell_grid_map(m, mode) = raw_frame_map(m) ∘ raw_frame_map(LinearRaw{mode})` builds every `LinearRaw`'s
  plane maps and every mosaic channel map from it. RadiometricCamera's probe mosaic (`_CELL_PROBE`, and the cvpr2027
  `_FRAME_PROBE` alias of it), built only to read that map, goes on RC's own branch.
- [x] ONE record of that map: the private `_frame_map_record(S)`, which writes both `raw_provenance(LinearRaw{mode}).frame_map`
  and an image's `raw_provenance(L).frame_map`; an image's `mode` is the mode record's. With `raw_frame_map(LinearRaw{mode})`
  it lets RC delete `_CELL_PROBE` and `_FRAME_PROBE` entirely.

## 0.9.2 (Estimation alone): the held SplineLens camera, fit_forward_spline

- [x] ONE fixed-camera residual: a held SplineLens camera's rows (`fixed = (:controls,)`, and the poses-alone first stage
  of a free refine) are the parametric route's `_BoardViewResidual{1, (false, true, false)}`, the whole camera a
  constant through `project` (refine_spline_camera_and_poses.jl, `_spline_problem`'s `!free` branch). The held-window
  block `_SplinePoseBlock` and `_SplineReprojection`'s held-window field are deleted, not kept beside it.
- [x] ONE forward fit: `_forward_spline` (fit_camera.jl), run by `fit_camera(SplineLens, …; forward = true)`, by the
  new public `fit_forward_spline` and, through it, by the SplineLens refine's refit (`_refit_forward`); no second fitter.
  Downstream (for its owner; no box): PanoramaStitching's docs named `fit_camera(SplineLens, …)` for a held camera's
  forward, fixed on its own branch (ps-0.6.2) to name `fit_forward_spline`.

## 0.10.0 (Estimation; Features 0.8.2, Eval 0.7.1): the free-SplineLens image-space refine retired

Final gate green on a0016dbb (jobs 10706–10709, both Julia versions; check_duplication 0 findings).

- [x] ONE bundle solve: the held SplineLens route solves with `_solve_bundle` and reads its keywords with
  `_bundle_keywords`, the parametric and rotation-centre routes' own; its private NLLSsolver solve (`_spline_solve!`)
  and time-cap bookkeeping (`_remaining`, `_timed_out`) are deleted, not kept beside them.
- [x] No dead second route: the window blocks, patch charts, third-difference blocks, GCV search and window rounds of
  the retired image-space refine are deleted (refine_spline_camera_and_poses.jl), not left unreachable. The stack's one
  GCV (`gcv_curve`, `gcv_select`) stays, for its other callers.

## 0.9.5 / 0.8.3 (Core, Features): the SIFT landing (owner rulings 2026-11-13)

Final gate green on 94a303ff (jobs 10745–10748, both Julia versions, CPU and GPU; check_duplication 0 findings on Core and
Features). The concept-level audit of the whole diff: ~/logs/dgx-agent/sift-landing-functions.md.

- [x] ONE device LM: `static_lm` is the former private `_spline_lm` (renamed, no alias), at static N; N = 2 bit-identical.
- [x] ONE weight derivative: `cubic_bspline_weight_derivatives` is the former private `_cubic_bspline_weights_deriv` made
  public with the second derivative; no wrapper.
- [x] ONE B-spline interpolation prefilter home: `cubic_bspline_prefilter_taps` beside `_prefilter_matrix` in Core; Features'
  copy deleted.
- [x] ONE window sweep: Core's `_window_sums` generalised to second partials as `spline_window_sums` (the same bits for its
  callers); Features' `_spline_sums` deleted.
- [x] ONE positive-definiteness test: `is_positive_definite` at static N (2×2 unchanged); Features' `_min_eig_above` deleted.
- [x] ONE emission map: `_level_to_canvas` / `_slice_sigma`, shared by `_extract_at!` and the continuous refinement.
- [x] `_conv_taps`, `_lagrange_basis`: checked against the stack (Features `_separable_blur` linear_image.jl:210, the tap
  builders, Core `convolve(::GaussianBlur, ::GaussianBlur)` blur.jl:94; no Lagrange interpolation anywhere): no equivalent,
  kept private.
- Deferred by ruling (no box): merging `_continuous_covariance` with the tabulated `_ift_cov`, a follow-up Features minor.


## Wave 113 (Core 0.9.5, Estimation 0.10.1; IO, Features, Eval, PoseLib patches): owner rulings 2026-11-13

Final gate green on 7c550857 (jobs 11013–11020; Julia 1.13.3, CPU and GPU, every package strict, docs; check_duplication 0
findings on all six). The concept-level audit and the deleted shims: ~/logs/dgx-agent/wave113-functions.md.

- [x] ONE region reach: `farthest_radius` (Core); Estimation's `_exit_radii` deleted; both `division_bounded` methods read it.
- [x] ONE masked bending operator: `bending_rows(mask; spacing)` (RadiometricCamera's `_bending_operator` moves here, same bits).
- [x] ONE plane → pixel map: `plane_map`, every camera model, the homography form by dispatch; `transfer_ellipse` reads it (`_lens_ellipse` deleted).
- [x] ONE homography point Jacobian: `transform_deriv` of the 2-D homography map; `_homography_point_jacobian` folded.
- [x] ONE held-camera rule and ONE pixel-whitening test: `held_parameters`, `pixel_whitening_defect` (the private copies deleted).
- [x] No shims: `_window_sums`, `_strip_length`, SpatialQueries' node aliases and `_mbr`, `_line_of`, `_camera_poses_result`, `_opted_file`, IO's `_json` deleted.
- [x] The duplication item deterministic with and without the stack paths; the two test-fixture `[[similar]]` entries restored.
- Reported, not deleted (public, breaking): `LiftableProblem`, `ScalarOrQuantity`, `POOL_RNG`.

## IO 0.10.0 (wave10, the IO breaking wave; branch w10-io10)

Concept-level duplication gate of the spec ("Tests": one units codec, one covariance codec, one lattice codec, one
measurement kind) and the slices' concept checks. A box is ticked when its commit is in w10-io10 and the gate is green on it.

- [x] ONE units codec: the envelope's `units` object, written from `VgioStyle` (`_units_json`) and read by `VgioStyle(record)`;
  UnitContract's own layout, `json_value(contract)`/`from_json(UnitContract, …)`, the per-block blur `"unit"` and the top-level
  `length_unit`/`pixel_frame`/`time_unit` retired (old files through their steps; `migrate_units` for BB's boards).
- [x] ONE covariance codec: `covariance` in rows/{"diagonal"}/{"isotropic"} everywhere (`_cov_wire`/`_cov_from_wire`); `cov`,
  `cov_size`, `point_covariances`, the null-plus-size and the default-3×3 fallback deleted; an absent covariance is the
  producer's `covariance_status`.
- [x] ONE lattice codec: `_lattice_json`/`_lattice_from_json`/`_lattice_spec` (style.jl, validate.jl) for the spline lens, its
  forward spline and the non-central model; `_spline_json`, `_spline_from_json`, the non-central spline spec deleted.
- [x] ONE observation kind: `vgio/observations` (`TargetObservations`); target_points2d/3d and target_detections only as steps.
- [x] ONE frame codec (detections): every FeatureFrame a case of `JSON.lower/lift(::VgioStyle, ::FeatureFrame)`, PointFrame the
  2-DOF case (io10-kinds); `json_value`/`from_json`/`read_quantity`/`write_quantity` deleted, no alias.
- [x] ONE camera kind: the store holds any bounded camera; `noncentral_spline_schema.jl` deleted; one body spec `_bounded_body_spec`.
- [x] ONE old-grid conversion entry: `_current_camera_body` at `camera_from_json` only (review I2).
- [x] ONE version list: `schema_versions()` from the steps; `_MIGRATIONS_TO_CURRENT` and the docs' hand lists deleted.
- [x] ONE envelope step for owners: `migrate_envelope` (public), used by IO's steps, panorama's and CaliBlobs'.
- [x] ONE device-identity encoding: `device` {"body", "lens"} in capture_settings, calibration_record and sensor_noise.
- [x] ONE identity path: `_identity_tree` is the writer's record on `_ID_STYLE`; the `_id_*` re-shapers deleted; the frozen 0.9
  identity lives in migration only (`_v09_id`), for the re-id map.
- [x] check_duplication(VisualGeometryToolkitIO): 0 findings (slices' runs 12228, 12569; the integrated gate re-runs it).
