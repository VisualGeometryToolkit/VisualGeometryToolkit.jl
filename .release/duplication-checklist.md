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
