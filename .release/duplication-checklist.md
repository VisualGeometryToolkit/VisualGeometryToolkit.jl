# 0.9 duplication checklist (monorepo)

Every finding of the 0.9 concept-level duplication audit (~/logs/dgx-agent/dup-audit-monorepo.md items 1–15, the
hoist asks of the downstream audits) and the branch that closes it. The owner's ruling: ONE release, no kept copies.
A box is ticked only when its branch is in the final RC and the final gate is green on it.

Final RC: 9c74e264 (the version bump 1d816a22 and the release records; this file's ticks are a records-only commit on it). Final gate, fresh clone, all green: GPU six 9763, CPU six 9764, docs 9765/9766 (docs-pass.md section 3).

## Fixed on `dup-audit-monorepo` (2cae3a7c, landed 2f0e6155)

- [x] The 25 deleted copies of the audit's "Fixed" table: run-key hash, git reader, scratch flag, marker/record names,
  mount_dir join, tier-path join, failed-leg predicate, refine keyword parsing, least-squares refusal, strategy
  constant, TOML default strategy, scale source, recorded scale, Phase-2 LO, edge rotation, LossWidth → TOML, SplineLens
  forward drop, pose finiteness, duplicated import, `lm_retcode`, `_r13_callee`, Irls/Triggs method pairs, PoseLib
  `_strategy_shape`, Features pixel centre, IO file digest.

## Audit items 1–15

| # | concept | branch (head) | in chain as | status |
|---|---|---|---|---|
| 1 | strategy efficiency constant (Estimation, PoseLib, Core predicate) | dup-items-a (c60fdb9b) | 202223b3 | [x] |
| 2 | FitResult with its estimate mapped (`map_estimate`) | dup-items-a | 202223b3 | [x] |
| 3 | retired-keyword sentinel (`RetiredKeyword`, Core) | dup-items-a + monorepo-hoists H7 | 202223b3 | [x] |
| 4 | weighted Kabsch → Core `kabsch(Csponds(…; weights))` | dup-items-b (95ceb6a9) | b00d18db | [x] |
| 5 | IrlsWeights normalisation (one rule `ρ′/ρ′(0)`; defect) | dup-items-b | b00d18db | [x] |
| 6 | three fixed-scale robust refine loops → one `Triggs` LM; `Irls` refine refused; the hand-eye LM on the bundle's one solve | dup-item-6 (13e3e09a, 0773a8d1) | ec490c8a, 2b8a1283 | [x] |
| 7 | L2 chart LM on the pixel rows (`_triggs_state`; defect) | dup-items-b + dup-item-6 13e3e09a | ec490c8a | [x] |
| 8 | SO(3) chart covariance → Core `transform_covariance` | dup-items-a | 202223b3 | [x] |
| 9 | `_spline_record` → Core `worst_retcode` | dup-items-b | b00d18db | [x] |
| 10 | pair norms → Core `point_norms!` | dup-items-a | 202223b3 | [x] |
| 11 | twin fit metadata records (one `FitMetadata`; the three retired names refused by `check_naming`) | dup-items-11-14 (3a086ce7) | 7a9ebb99 | [x] |
| 12 | within-file twins (Eval reuse rule, K₀₂/fx₀ + domain cap, Hartley maps, scoring workspaces, ONNX arena growth, R13 TOML walk) | dup-items-a | 202223b3 | [x] |
| 13 | ci: sbatch option tail, docs.sbatch pieces, `VGTK_DOCS_LOCAL` reader | dup-items-a | 202223b3 | [x] |
| 14 | Eval partial overlaps M1–M7 (one copy each); the dependency-free runner's unknown-key refusal stays (it cannot load Core's `check_keys`: an owner ruling, recorded in Eval's CHANGELOG) | dup-items-11-14 (3a086ce7) | 7a9ebb99 | [x] |
| 15 | median seed (recorded), `_irls_record` vs refine Irls record (moot: Irls refines refused, item 6), PoseLib `_assert_unit_bearings` | dup-items-a + dup-item-6 | ec490c8a | [x] |

## Downstream audits' monorepo asks

| ask | concept | branch (head) | in chain as | status |
|---|---|---|---|---|
| BB 5 | prefix-sweep loop body (`prefix_sweep`) | dup-items-e (35be7a8c) + integrator fix 04796694 | 04796694 | [x] |
| BB 6 | IRLS termination record (`irls_termination`) | dup-items-e | 166a986f | [x] |
| BB 7 | renamed-key refusal (`check_keys(; renamed)`) | dup-items-e | 166a986f | [x] |
| BB 8 | artifact install (`install_artifact`, `huggingface_token` in Core) | dup-items-e | 166a986f | [x] |
| F1, F7, F8, F10 | `is_stationary(blocks, …)`, `rotation_center_pose`, `pooled_noise_scale(scales, dofs)`, `lens_coverage(camera, rim_margin)` | monorepo-hoists (f12bd8a0) | 47b72b04 | [x] |
| F9 | CountedLO per-call `call_records` | dup-items-f (591de3d7) | 52b9762a | [x] |
| H1–H10, H12–H14, H16, H17 | hoists (scoring constructors, TOML reader, device buffers, `fetch_data!`, radix buffers, `reproject_so3`, `RetiredKeyword`, uncertain `inv`, Student-t default, `rms(residuals, n)`, `bearing_csponds`, Features `propagate`, retired names, `reset_posterior!`) | monorepo-hoists | 47b72b04 | [x] |
| H11 | one launch policy: `OnBackend(backend; workgroupsize)(kernel)` (DUPLICATION.toml `[[backend]]`) | launch-policy (91f620ad) | 440168ef | [x] |
| H15 | χ² quantile (whitened units) | whitened-units (f9e35470) | 8300fbf5 | [x] |

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

