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
