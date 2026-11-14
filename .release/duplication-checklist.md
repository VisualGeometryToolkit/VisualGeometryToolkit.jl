# IO 0.10.3 duplication checklist (patch-io0103), 2026-11-14

The leg's upstream duplication gate (cau-release `upstream_duplication.jl`, owner rule 2026-11-14) ran in IO's prep
16879 on the fresh clone of f5e77eca: 0 findings (IO's name-ownership upstream: Core). `check_duplication` of IO: 0;
`check_naming` of IO: 0.

| concept | the copies | the one copy | status |
|---|---|---|---|
| a length's identity in a comparison | a second tolerance beside the writer's last resort | `_persisted` (content_id's length leaf), read by `_same_metres` | [x] |
| the versions a file is current at | `_is_current`'s own "current major" beside the writer's table | `_written_versions`, read by `_is_current`, the writer and kinds.md's table | [x] |
| the persisted-form rule's text | IO's conventions page and Core's units.md each stating it | Core's `conventions/units.md` ("What is exact"); IO's page points at it | [x] |
| the v1.0 fixture with its rasters in a test | three copy loops | `SensorFixtures`' `copy_v1_fixture` | [x] |
| the persisted-length claims | the default item and the benchmark item | `PersistedLengthClaims`' `persisted_length_claims` | [x] |

---

# Core 0.10.3, IO 0.10.2, Features 0.9.3 duplication checklist (patch-core0103), 2026-11-14

The leg's upstream duplication gate (cau-release `upstream_duplication.jl`, owner rule 2026-11-14) ran in Core's prep
16565 on the fresh clone of 7b322318 for Core, IO and Features: 0 findings each (Features' name-ownership upstream:
Core and Estimation; IO's: Core). `check_duplication` of IO and Features in their preps (16569, 16572): 0.

| concept | the copies | the one copy | status |
|---|---|---|---|
| darks of several exposures | a second characterisation per exposure, or a band lookup in a consumer | Core's `Darks`, resolved inside `sensor_model` (`_frame_black(::Darks, t)`); `_pick` unchanged | [x] |
| the defect mask | RC/Features building the mask from a `DefectList` | Core's public `defect_mask`, LinearRaw's own `_defect_mask` | [x] |
| a linear view's code quantum | RC reading `LinearRawView.span` / `_quad_site` | Core's public `linear_quantum` | [x] (RC adopts it) |
| the versions a writer emits | a hand-kept list of published schemas beside the writer's choice | IO's `_written_versions` (with each version's validator), read by `_write_kind` and the JSON Schema generator | [x] |
| the optional channel black of a dark body | written and read twice in the codec | `_with_black` / `_optional_black_from` | [x] |
| the rim kernel launch | a bare `OnBackend(...)` + `synchronize` | Features' `_launch!` | [x] |

---

# Core 0.10.2 duplication checklist (patch-core0102), 2026-11-14

Core is the stack's root, so it has no upstream. `check_duplication(VisualGeometryToolkitCore)` ran in prep 16196 over the
whole monorepo tree of 19126b52: every in-repo package (Core, IO, Features, Estimation, Eval, PoseLib). VGTK_STACK_PATHS
covered the downstream wave repositories (w10-bb, w10-ps, w10-cb, w10-lodr, w10-rsdr). The result was 0 findings.

| concept | the copies | the one copy | status |
|---|---|---|---|
| the LM damping | a second LM, or a copied NLLSsolver iterator, for λ·I | `LmDamping`: NLLSsolver's own public iterators (`:levenbergmarquardt_dtd`, `:levenbergmarquardt`), chosen by dispatch in `_nlls_iterator` | [x] |
| the data's resolution in the success rule | a second notion beside `StaticLmStops`' | `residual_resolution`: the same name, the same meaning (the residual norm the data resolve, read only by the exact-fit test) on `LmStops` and `StaticLmStops` | [x] |
| the exact-fit threshold | `LmStops.ftol` read as an absolute cost | `_exact_fit_threshold` in `_lm_run!`, the one place the chart, robust and multi-block postconditions get it | [x] |

---

# Core 0.10.1 duplication checklist (patch-core0101), 2026-11-14

Core is the stack's root, so it has no upstream. `check_duplication(VisualGeometryToolkitCore)` ran in prep 15737 over the whole
monorepo tree of 78fa6599: every in-repo package (Core, IO, Features, Estimation, Eval, PoseLib). VGTK_STACK_PATHS covered the downstream
wave repositories (w10-bb, w10-ps, w10-cb, w10-lodr, w10-rsdr). The result was 0 findings.

| concept | the copies | the one copy | status |
|---|---|---|---|
| the photon-transfer fit (single frame, series) | a second fitting core for the series | ONE block kernel `_difference_blocks_kernel!` (one cell rule per input) and ONE line fit `_ptc_line` with `_ptc_curvature` | [x] |
| the photon-transfer result record | an ad-hoc NamedTuple | the stack's `FitResult` with `PhotonTransferMetadata` | [x] |
| holding LM variables | a second construction path | `_lm_problem`'s unfixed mask through `LmProblem(…; fixed)` | [x] |

---

# Estimation 0.11.1 duplication checklist (patch-est0111-fixture), 2026-11-13

| concept | the copies (searched by concept: Core, Estimation, Eval, LODR w10-lodr/rel10-lodr, BB, RSDR, CB, PS) | the one copy | status |
|---|---|---|---|
| the two-camera F scene (f 500/600, `RotY(0.15)`, `(1, 0.2, 0.1)`, `box_point_far`, the uniform outlier pairs) | Estimation test/ransac_fundmat_tests.jl `make_fundmat_data`/`make_exact_fundmat_data`; LODR accv2026/test/lodr/test_problem_arms.jl `make_fundmat_data` (a `MersenneTwister` stream) | `synthetic_scenes().EpipolarScenes.fundmat_scene(rng; …)` on `stereo_pixel_pairs`; Estimation's copy deleted (b26cfd06); LODR's to delete (handoff: `fundmat_scene(MersenneTwister(42); …)`, its draws unchanged) | [x] Estimation, [ ] LODR |
| other F scenes, considered and NOT this concept | `noiseless_ransac`'s Gaussian-cloud scene (one camera, `RotY(0.3)`), `scenes_testsetups`' `stereo_scene` (the find_fundmat pipeline's, one camera, 960 × 720), `contaminated_pixel_pairs` (`PIXEL_PAIRS_K`, a homography/F pool scene): other scenes, each one copy | — | n/a |
| the LO bench arm: `locally_optimize` of each pair's problem from each seed with that pair's scoring and workspace, timed by `call_set_arm` | Estimation test/lo_timing/bench.jl `lo_call` + `lo_arm`; LODR accv2026/test/lo_timing/bench.jl `lo_call` + `lo_arm` (the same text; only the scorers built beside them differ) | `test/lo_timing/call_set.jl`'s `lo_call_set_arm` (Estimation's file, which LODR's bench already includes for `call_set_arm`); Estimation's copy deleted; LODR's to delete (handoff) | [x] Estimation, [ ] LODR |
| `call_set_arm` itself | one copy, Estimation's call_set.jl (not Core's: Core's bench_record.jl holds `kernel_arm`, the record writer and the spread) | unchanged | [x] |

check_duplication / check_splat / check_naming (gate prep 15162, stack paths incl. w10-lodr, verbatim):
```
== check_duplication(VisualGeometryToolkitEstimation 0.11.1): 0 finding(s)
== check_splat(VisualGeometryToolkitEstimation 0.11.1): 0 finding(s)
== check_naming(/raid/data/jpr/w10-est-eval-run-b26cfd06/clone/VisualGeometryToolkitEstimation): 0 finding(s)
```

---

# Wave 10 duplication checklist: VisualGeometryToolkitCore 0.10.0 (branch w10-core10)

The owner's rule for the wave is ZERO duplication by concept. Each item is a concept that had more than one copy in the
stack, the ONE copy Core keeps, and the gate that keeps it one. A box is ticked when the copy is gone and the up-v13
integration gate a8f4aa02 (jobs 14786–14790, 48/48; `check_duplication` 0 on all six packages, wave stack paths) is
green on it.

| concept | the copies | the one copy (Core) | the gate | status |
|---|---|---|---|---|
| the LM adapter and its stop settings | NLLSsolver called from the bundles and refiners of Core, Estimation, Eval and RC (the LM call-site audit); `nlls_options`; `_robust_stationary` | `lmsolve_chart!` (chart, robust, multi-block `LmProblem`), `LmStops`/`StaticLmStops`, one success rule (robust/lm_core.jl) | `check_duplication` `:lm_primitive` | [x] |
| canonical JSON (RFC 8785) | IO's `canonical.jl` | `canonical_json` (base/canonical.jl) | `check_duplication` (the pair Core ⇔ IO; IO deleted its copy) | [x] |
| the detection → observation conversion | Features' detection_types.jl, RC's chrome_evaluation | `center(::Uncertain{<:FeatureFrame})` | handoffs to Features and RC | [x] |
| the FeatureFrame verbs | per-type `ustrip`, `*`, `/`, similarity actions, `extrema`, `Rect2`, `bbox`, `MvNormal`, `inv`, `Ellipse`, `Circle` | one method per verb on `FeatureFrame` or its DOF trait | the no-overhead tests, `feature_frame_interface_tests.jl` | [x] |
| region containment | the region-in-region `in` methods; Base's iterable `in` reached for two ellipses | one `⊆` (regions.jl, image_circle.jl) | `region_containment_tests.jl` | [x] |
| the GPU test tier | RC's test/gpu_decision.jl | `gpu_tier`/`gpu_problem`/`require_gpu` | DUPLICATION.toml `[[backend]]` | [x] |
| sensor noise: components, black subtraction, value variance | wave 10's interim sensor-noise names; hand-indexed channel blacks; hand-written variances | the sensor model; THE black subtraction (LinearRaw/linearview); ONE variance law, `_value_variance` (the kernel's per plane, the lazy view's per photosite) | `check_duplication` `:sensor_component`, `:black_subtraction`, `:value_variance_law` | [x] |
| launch then reduce, on the device | Estimation's `_LensRegressionLm`, RC's `_RadiometryLm` | `reduce_normal_equations(row!, …)`/`reduce_cost(row!, …)` (0.9.7 port) | the 0.9.7 tests | [x] |
| IoU below a bound | BlobBoards' `_iou_below` | `iou_below` | `check_duplication` with the wave stack paths | [x] |
| README-test helpers | BlobBoards' and CaliBlobs' restated helpers | `page_examples`, `install_lines` | `check_duplication` with the wave stack paths | [x] |
| the refine width rule and the unit-bearing check | PoseLib's and Estimation's copies (review E11, E12) | `refine_width`, `assert_unit_bearings` | `check_duplication` | [x] |
| the module walk of the quality gates | `check_naming`'s supertype table and QualityExt each walked the modules | Core's `_module_tree` | `check_splat` (Core `:enforce`) | [x] |
| the 2D/3D target point containers | `TargetPoints2D/3D` beside `TargetDetections` | `TargetObservations` | `check_naming` retired names | [x] |
| the stopping rule's record | per-package rule tables (LODR) | `stop_rule(criterion, method)` | the contract generic | [x] |

Form-only findings (not duplicates): none. The d366918b prep with the wave stack paths (bb, ps, cb, lodr, rsdr) reported
only the Core ⇔ IO canonical pair, which is now closed.

---

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

```
== check_duplication(VisualGeometryToolkitIO 0.9.5): 0 finding(s)
== check_duplication(VisualGeometryToolkitEval 0.7.2): 0 finding(s)
```

## Features 0.8.4 (patch-features084): no splat of a data-length collection, 2026-11-13

- [x] ONE transpose of a capture's identities into columns: StructArrays' `components(StructArray(ids))` (already a
  Features dependency), in `_merged`; no hand-written column loop (09a662e3, 3d8203a4).
- [x] ONE score-type rule over any number of descriptor sets: `_score_eltype(As)` on a collection (29888695).
- [x] The sweep of the stack's splats of a data-length collection, by owner, and the lint design for core10: wave10/handoffs.md
  ("patch084 → every owner", "patch084 → core10").
- [x] check_duplication: 0 findings on all six, Features 0.8.4 included (gate prep 14059).

## Core 0.9.7 / Estimation 0.10.3 (patch-core097-est0103): one launch-and-reduce, 2026-11-13

- [x] ONE launch of a device LM state's rows and reduction of their partial sums: Core's `reduce_normal_equations(row!,
  Val(N), args...; partials)` / `reduce_cost(row!, args...; partials)` (737813d1); Estimation's `_LensRegressionLm` calls
  them (a4978736), RadiometricCamera's `_RadiometryLm` too (RC 2d074aef). The pair RC linear_solve.jl:298/302 ⇔
  Estimation refine_camera.jl:178/182, found BEFORE (job 14318: 2 findings), gone against Estimation 0.10.3 with RC
  unchanged (job 14319: 0) and AFTER RC's change (job 14322: 0): docs-pass.md, verbatim.
- [x] check_duplication: 0 findings on all six (gate prep 14112).

## Eval 0.7.4 (patch-eval074): a registered install is committed code, 2026-11-13

- [x] ONE decision of a path's code tree: `code_trees` (code_identity.jl), which `refuse_uncommitted_code`, the runner
  and every driver record (`_recorded_code_trees`) already go through; the registered install is a branch of it, no
  second code path (7271049c, 4209e867).
- [x] ONE source of a registered install's tree hash: the active environment, through Pkg's API (`Pkg.dependencies()`:
  name, version, uuid, tree_hash); no hand-parsed Manifest for it. Searched by concept: Core's sync_manifest.jl:56 reads
  `Pkg.dependencies()` for direct dependencies' versions (another question: a Manifest's sync), and the Manifest
  `git-tree-sha1` of a loaded stack package stays `code_trees`' own (code_identity.jl `_loaded_entries`).
- [x] check_duplication 0, check_naming 0, check_splat 0 on Eval 0.7.4 (gate prep 14587).
## Core 0.9.8 (patch-core098): one device-capacity query, 2026-11-13

- [x] ONE count of a kernel's resident threads: `device_resident_threads` (base/device.jl), one method per backend of
  `_resident_threads` (the host's threads; the CUDA extension's compiled-kernel occupancy), no `isa` branch; a device
  query, so `check_duplication` leaves its methods per backend alone (`_TRANSFER_NAMES`), a compute function's still not
  (tested).
- [x] check_duplication: 0 findings on all six (gate prep 14548).

## Core 0.9.9 (patch-core099): one construction of the pointwise launch, 2026-11-13

- [x] ONE construction of `pointwise!`'s kernel, arguments and ndrange: `pointwise_kernel` (`_pointwise_launch`), which
  `pointwise!` launches; a caller asks `device_resident_threads` about that triple, never a copy of the construction.
- [x] check_duplication: 0 findings on all six (gate prep 14831).

## Core 0.9.6, Estimation 0.10.2, Eval 0.7.3 (patch-core096-eval073): the catch-up review's C1 and E1 and the owner's additions, 2026-11-13

The concept search: Core's LM core (robust/lm_core.jl) holds no reduction of per-work-item partial sums; the names
follow the protocol they serve (`residual_normal_equations!`, `residual_cost!`). The runner's reading of earlier
launches' stage records had one walk (`_mark_reuse!`); the upstream keys needed a second, folded into one reader.

- [x] ONE reduction of a device LM state's partial sums: Core's `reduce_normal_equations`/`reduce_cost` (lm_core.jl);
  Estimation's hand unpack in refine_camera.jl deleted (822a01cf); RadiometricCamera's `_RadiometryLm` is the next caller.
- [x] ONE `axis_gauged` of a bounded camera: the wrapped camera's own method (`_rebound(bc, axis_gauged(bc.camera, …))`,
  30debf6d), no second turn of the pose.
- [x] ONE reader of the earlier launches' stage records: `_stage_records` (runner.jl), read once per plan for the
  upstream keys and for reuse; the unused `_manifest_dir` deleted (ee512b8e).
- [x] ONE stage-key rule: the upstream keys folded in `plan_reproduction` alone (e36a1959).
- [x] ONE chart-LM stop setting: `refine_camera`'s regression reads `_CHART_LM_FTOL` ([chart_lm]), as every chart LM (bfd2282b).
- [x] ONE region rule for a bounded source: `_imaged_part` (region ∩ imaged_region), used by `fit_camera`,
  `refine_camera`, `approximate_camera` and so `optical_axis` (2b116fa9, 4c910232); `optical_axis(camera)` is its
  region form with the default, no second path.
- [x] check_duplication: 0 findings on all six packages, with the stack repositories in VGTK_STACK_PATHS (gate prep; verbatim
  in ~/logs/dgx-agent/patch-core096-eval073.md).

---

# IO 0.9.4 duplication checklist (monorepo)

The owner's rule (2026-11-13): no new codec where an existing body does the job, no shims or aliases. Each item names the
concept, the stack's one implementation reused, and the commit. The concept search, file:line and check_duplication output:
~/logs/dgx-agent/io094-ps064.md.

- [x] the panorama's camera → the camera store's bounded camera body (`_bounded_body` / `_bounded_from_body`, camera_schema.jl:327/336; table `_BOUNDED_CAMERA_BODY`, rig_schema.jl:792) (4e86078c)
- [x] the panorama's rotations → the trajectory body (`JSON.lower`/`JSON.lift` of `Trajectory`, captures_schema.jl:203/227); its table `_TRAJECTORY_BODY` is the trajectory file's own, the file's table now built from it (captures_schema.jl:173–183) (4e86078c)
- [x] the panorama's vignetting → the vgio/vignetting body (`_VignettingBody`, `JSON.lower`/`JSON.lift` of `AbstractVignetting`, radiometry_schema.jl) (4e86078c)
- [x] a matrix stored as rows (the gains; the vignetting's control points; a light's coefficients) → one reader `_rows_matrix` and one writer `_matrix_rows` (radiometry_schema.jl:99/101), replacing four inline copies (4e86078c, d593a27c)
- [x] the point frame's centre and covariance → the uncertain point's leaves (`_pixel_json`, captures_schema.jl:528; `_cov_json`/`_cov_matrix`, transform_schema.jl:68/69); a Symbol-keyed object normalised by `_lower` (rig_schema.jl:1076) (4e86078c, d593a27c)
- [x] the detection metadata read → one `_detection_metadata` (core_json.jl:454), shared by every frame (4e86078c)
- [x] check_duplication(VisualGeometryToolkitIO 0.9.4) and (VisualGeometryToolkitEval 0.7.2): 0 findings each, with the stack repositories in VGTK_STACK_PATHS (gate prep 11552)

---

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
