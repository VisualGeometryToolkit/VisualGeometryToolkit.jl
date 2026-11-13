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
