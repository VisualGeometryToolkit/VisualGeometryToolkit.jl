docs-pass: c8fa6023
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io aa18513 (searched, not edited: no page names OnBackend, on_device, device_names, device_resident_threads, kernel_float or a saddle)

# Core 0.13.4: the per-backend choices of one kernel, each stated once; Synthetic.saddle_height — at c8fa6023

- Public API against 0.13.3, additions only: `kernel_float(backend)`, `dual_chunk(backend, slots, device_chunk)`,
  `OnBackend(backend; workgroupsize, host_workgroupsize)`, `device_resident_threads(backend)`,
  `on_device(T, S, device, x)`, `device_names(backend)`, `Synthetic.saddle_height(𝐛)`. Each has a docstring; the API
  pages pick them up (core.md autodocs base/device.jl, batched.md pointwise.jl, synthetic.md now lists surfaces.jl).
- CHANGELOG 0.13.4. The batched guide's `OnBackend(backend; workgroupsize)` stays correct (the new keyword defaults to it).
- Gates: strict docs build (VGTK_DOCS_LOCAL=1) exit 0; quality naming / duplication / calling conventions pass; full
  suite 418,934 passed, the remaining failures identical on origin/main 3d8a06da on this machine (photon_transfer and
  inv(::Uncertain) last-bit pins, stack_ci's git add under a global gitignore of Manifest.toml, a CUDA rand ambiguity
  when CUDA is loaded in the worker, a GPU OOM on the shared card).

docs-pass: d95ae7fc
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io c353718 (searched, not edited: no page builds a calibration matrix, names a principal point or calls center on a frame)
(the pass reviewed eb55880e; to d95ae7fc Core changed only by its version and one test line, `centroid(p)` for a removed `center(p)`)

# Core 0.13.2: CameraCalibrationMat from a focal length and the image; center(::FeatureFrame) removed — at eb55880e

- Public API against 0.13.1: added methods `CameraCalibrationMat(f, pp::AbstractVector{<:Real})`,
  `CameraCalibrationMat(f, r::Rect2{<:Real})`, `CameraCalibrationMat(f, sz::Size2{<:Real})` (one docstring with a
  jldoctest, `check` keyword; `f` a real, `(fx, fy)` or `px` quantities); removed the method `center(::FeatureFrame)`
  (`centroid(f)` is the frame's centre; `center` stays public for the radial models and `center(u)` of an uncertain
  frame). No name added or removed.
- Docstrings and pages: `centroid`'s method list (`Rect` returns a `Vec`, a frame a `Point2`), the `FeatureFrame`
  interface list (`centroid(frame)`), `conventions/feature-frames.md` (the centre is `centroid`), the cameras guide's
  intrinsics section, the lens guide's three cameras and `examples/image_circle.jl` (K from `Rect(s)` / `Size2`
  instead of a hand-written principal point), the CHANGELOG (0.13.2, and a planned-review line for the uncertain
  frame's `center` naming). README unchanged: its example builds the camera from a metric focal length and pitch.
- Gates: the `CameraCalibrationMat from a focal length` item 27/27; the `feature frames`, `docs`, `README`, `quality`,
  `naming`, `duplication` and `agent docs` items 2223/2224 locally, the one failure the untracked machine-local
  `.claude/settings.local.json` (absent from a clone). `Test.detect_ambiguities(VisualGeometryToolkitCore; recursive =
  true)` empty. The docstring's doctest checked in the REPL. Not run: the strict docs build (a kaimon session for
  `docs/` needed an approval nobody could give); the release gate's fresh clone runs the suite.
- Not changed, for a later release of their package: VisualGeometryToolkitIO's README, `docs/src/index.md` and
  `guide/getting-started.md` (`CameraModel(2400.0px, [2120.0, 1416.0] .* px; …)` beside `Size2(4240, 2832)`: the
  centre by hand; the new form needs Core 0.13.2 in IO's compat).

# Features 0.9.11: ASIFT's upsample count, nb_scales and nb_octaves — at 2b4d4867

- `asift_detect`'s `upsample` (a count of 2× upsamplings), `nb_scales` and `nb_octaves` keywords: their docstrings
  (`src/asift/api.jl`) and the CHANGELOG entry say what the code does; no public name added or removed. Features' docs
  strict green at 2b4d4867.

# Core 0.13.1 and Eval 0.11.0: the NFS-safe install lock, a site profile's home paths — merged at b50661a4

- Core: the install lock (a directory per tree, its owner's heartbeat, a stale holder taken over) described in the
  `base/artifacts.jl` header and its comments, `install_artifact`, `install_built_artifact` and `store_file`
  docstrings and `docs/src/api/artifacts.md`; the FileWatching dependency is gone; a profile file's `~/…` store. No
  public name added or removed.
- Eval: `Site` and `load_site` (a `[storage]` path may start at the home: `~/…`, `$HOME/…`), the site template, the
  `desktop` profile (`~/data/vgtk/…`), `paper-projects.md` ("Machines and storage") and `reproduce.md`. No public name
  added or removed.
- Builds: Core's and Eval's docs strict green at 5d3229b5 (b50661a4 changes one line of a private function). Simplify
  and code review on the diff, their findings applied (1b68e417, 5d3229b5, b50661a4).

# Eval 0.11.0 docs review (owner decisions after the 0.10.2 preparation) — branch eval-011-sites, merged at c118f57f

- Version 0.11.0 (breaking: a top-level `[render]` in a Product record is an error; the `dgx01` site profile is
  retired). Added: `Experiments.project_results_root`, the top-level `catalog_results_mount`, `is_product_input`,
  `product_input_path`, `product_input`; `missing_inputs` takes `results_root` and `site_profile`; `tier_data_root`
  takes `results_mount` and `results_root`. Every new name has its docstring (the suite's `docs` item).
- Pages: `paper-projects.md` (a produced entry naming another project's producer; the section "Another project's run
  of record": the product-input form, its roots, the results-mount tier), `reproduce.md` (the shipped sites, the
  examples on `vcai`); the `input_path`, `missing_inputs`, `tier_data_root` docstrings; the runner bootstrap's usage
  line. The `desktop` profile documents its storage.
- Builds: Eval's docs strict green at c118f57f. Simplify and code review on the diff, their findings applied
  (8a37cac2, 82d33c57).

# Core 0.13.1, Features 0.9.11, Eval 0.10.2 docs review — branch followup-store-site (merged to main at fd19dcdb)

- Public API against the registered trees (Core 0.13.0, Features 0.9.10, Eval 0.10.1). Core, added:
  `install_built_artifact`; changed: the reader `set_site_artifact_store!` registers returns `(; artifact_store,
  refuse_depot)` (a path string still accepted). Eval, added: `Reproduce.asset_stages`, `Reproduce.deploy_signoff`,
  `Experiments.produced_dir`, the `Site.refuse_depot` field; moved: `paper_figures_dir` is defined in Reproduce and
  exported by PaperProjects as before (one function). Features: no name added or removed.
- Docstrings: every new name has one, with its signature as in the code (the `docs` items of the three suites:
  `undocumented_public`, `signature_mismatches`, `detached_docstrings`); `artifact_store`, `set_site_artifact_store!`,
  `Site`, `data_root`, `results_root`, `resolve_roots` and `Product` say what the code now does.
- Pages: Core `docs/src/api/artifacts.md` (the depot rung's refusal, the built tree); Eval `paper-projects.md` (the
  `[storage]` example with `refuse_depot`, the roots table and the one ladder, a job's `VGTK_SITE`, the `[render]`
  refusal in a Product's record) and `reproduce.md` (the site template's keys, `asset_stages`); Features
  `guide/matching.md` and `developer.md` (the CUDA build in the artifact store). The site template documents
  `refuse_depot`. READMEs name none of the changed names.
- Builds: the three docs builds strict (no `warnonly`, `checkdocs = :public`, doctests, linkcheck) green at fd19dcdb.
- Simplify and code review (high) on the diff, their confirmed findings applied (commits d7f20206, b35529d1).

# Core 0.13.0, Eval 0.10.0, Features 0.9.10, IO 0.11.1, Estimation 0.12.2, PoseLib 0.3.3 docs review — branch core-store-cleanup

## 1. Public API since the 0.12.0 cascade (862ebba7)

- Core, removed: `SphericalSplineSystem`, `spherical_spline_kernel`, `spherical_spline_coefficients`,
  `spherical_spline_evaluate` (and their methods of `nodal_weights`, `bending_energy`, `bending_factor`,
  `bending_rows`), `dilog`. Added: `ArtifactStore`, `artifact_store`, `with_artifact_store`, `set_site_artifact_store!`,
  `store_file`, `download_verified`, `store_name_clashes`, `migrate_artifacts`. Changed: `install_artifact` (a `store`
  keyword; installs into the store).
- Eval, added: `site_root`, `active_site`, `site_profile`, `project_name`,
  the `Site` storage fields, the checks L5, L6, D5, `:storage` A2–A4, `:release` R1–R3. Features, IO, Estimation,
  PoseLib: no name added or removed.

- The review (simplify, code review; 1aa082b4) added Core's `is_published_url` (docstring, in the artifact store
  page's `@autodocs`) and made Eval's `MACHINE_PATH_DIRS`/`MACHINE_PATH_EXTS` and `data_policy_violations`'
  `machine_dirs`/`machine_exts` keywords private again (no caller); the docstrings of `with_artifact_store`,
  `migrate_artifacts`, `data_policy_violations` and `results_root`, the paper-projects page's roots table and D1 row,
  and both CHANGELOGs say what the reviewed code does.

- Eval at 659a6891 (owner decisions after the Core, IO, Estimation, PoseLib, Features submissions; Eval not yet
  registered): DataArtifacts' produced entries (`kind = "produced"`: header, `fetch_data!` and `catalog_violations`
  docstrings, the paper-projects page's catalog section with an example and its A1/A2 rows) and D1's wider scope
  (the data-policy header, `data_policy_violations`' docstring, the page's D1 row, the CHANGELOG). No public name
  added or removed. Simplify and code review on the diff, their findings applied.

- IO 0.11.2 at 9ab42cd1: `nested_roundtrip` and `edited_copy` added (public, docstrings with an example each, in the native
  API page's `@autodocs` through `kind_test_helpers.jl`), the CHANGELOG; IO's own tests use them. No other page names a
  test helper.

## 2. Docstrings and pages

- Core: the removed names' docstrings, the `docs/src/api/batched.md` paragraphs and ARCHITECTURE.md lines deleted (no
  `@ref` to them is left); a new page `docs/src/api/artifacts.md` (the store: its resolution table, the install, the
  single files, the migration); every new name has its docstring (`base/artifacts.jl`).
- Eval: `docs/src/paper-projects.md` (the rules, the enforcing test's table with L5, L6, D1, D5, `:storage`,
  `:release`, a section "Machines and storage"), `docs/src/reproduce.md` (the shipped `vcai` site, `[storage]`), the
  site template, the paper-project template's `data/README.md`, `README.md` and `scripts/fetch_data.jl`.
- CHANGELOGs of the six packages.

## 3a. Gates of the review fixes (1aa082b4; Slurm vcai, Julia 1.13.4, fresh clones from GitHub, `--depwarn=error`, logs under /cluster/projects/vgtk/logs/store/results/*-1aa0*)

- Strict docs builds: Core, Eval, Features, Estimation, IO, PoseLib: PASS.
- Suites: Core 418961 pass (core-1aa0, a B200 slice), Eval 3169 pass, Features 3939 pass (an RTX Pro 6000), Estimation
  8995 pass, IO 72521 pass, PoseLib 405 pass; registration_lint, agent_docs and affected (lint-1aa0): PASS.
- The leg's prep at f5d9671f (release_leg.sh, LEG_PHASES=prep): manifests up to date, floors equal, upstream
  duplication gate and check_naming clean for all six.

## 3. Gates (Slurm vcai, Julia 1.13.4, fresh clones from GitHub, `--depwarn=error`, a B200 slice or an RTX Pro 6000
per job, VGTK_ARTIFACT_STORE the project space's store; logs under /cluster/projects/vgtk/logs/store/results/)

- Strict docs builds (`checkdocs = :public`, doctests, link check): Core (core-3a56; its docs tree is 4a72c8a8's),
  Eval (eval-6e61), Features, Estimation, IO, PoseLib (e75ff855; their docs unchanged since): PASS.
- Suites: Core 418954 pass (core-4a72), Eval 3165 pass (eval-6e61), Features 3939 pass (feat-6e61, the LoMa weights
  installed into the store), Estimation (est-6e61), IO 72521 pass (io-6e61, on nereus), PoseLib 405 pass
  (poselib-6e61); registration_lint, agent_docs and affected (lint-6e61): PASS. From 6e612e03 to 4a72c8a8 only a Core
  test file changed.

docs-pass: 392c11c2
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io eac44cb (read, not edited: no card names a moved type, a moved kind or a version, and the install lines carry none; the IO card says "target measurements", which the naming rule spells "target observations"; no card or link names the Visual Radiometry Toolkit, the moved types' new owner)
(the pass reviewed a74e2d51; from there to the release commit named above the packages changed only by: `radius` declared public again in Core, a test of Eval, the version numbers, the dependents' compat, the CHANGELOG headings and the generator version of IO's test fixtures — no page, docstring or example)

# Core 0.12.0, IO 0.11.0, Estimation 0.12.1 docs review — branch vrtk/retire-radiometric

To be released: Core 0.12.0 and IO 0.11.0 (breaking: the radiometric types and kinds moved to
VisualRadiometryToolkitCore / VisualRadiometryToolkitIO, the panorama kind to PanoramaStitching), Estimation 0.12.1.
Baseline: the last review, 885fbb4f. The review covers the tree a74e2d51; after it only this record changes
(`.release/` is outside the gate). The versions in the Project.toml files are still the baseline's (the bump follows),
so the CHANGELOG sections are "Unreleased".

## 1. Public API since 885fbb4f

- Core, removed (each a refusal by name, `src/retired.jl`, `MOVED_NAMES`): the vignetting fields and their verbs, the
  binned radial profile, the radial charts, the environment light and its spherical-harmonic map and verbs; `radius`
  is no longer declared public (GeometryBasics' function). Added: `primal_value`, `strip_pixels`, `strip_point`,
  `attach_point`, `attach_length`, `parameter_offsets`, `MOVED_NAMES`, `intersection_at`, `stack_ci` and its ten parts.
  Changed: `cgls` and `normal_cg` on Krylov.jl (a preconditioned solve stops in the scaled norm), a `CSRMatrix` on CUDA
  multiplied by cuSPARSE.
- IO, removed: the `vgio/vignetting`, `vgio/environment_light`, `vgio/radial_profile` and `vgio/panorama` kinds
  (`save_panorama`, `load_panorama` refused by name, no longer public). Added: `KindSpec`, `write_kind`, `read_kind`,
  the nine `*_spec` combinators and `body_spec`, `migrate_file`, `migrate_body`, `fit_record_json`,
  `fit_record_from_json`, `pixel_frame`, `pixel_frame_offset`, `length_unit`, `matrix_rows`, `rows_matrix`,
  `renamed_key`, `MOVED_NAMES`.
  (`fit_record_json` and `fit_record_from_json` were removed in IO 0.12: a FitResult goes through the style codec,
  `JSON.lower`/`JSON.lift`.)
- Estimation: no name added or removed (`optical_axis` and the exact resample of `fit_camera(SplineLens, …)` on a
  CUDA device).

## 2. Docstrings and pages brought up to date

- Already true at aa1b3050 (the migration's own commits; read and kept): Core's `docs/src/api/batched.md` (the
  vignetting and lighting sections replaced by one paragraph naming VisualRadiometryToolkitCore; the solvers on
  Krylov.jl), `api/primitives.md` (`intersection_at`), `api/stack_contracts.md` (`retired.jl`, the workspace CI);
  IO's `guide/kinds.md` (the moved kinds' rows with their owners, "A kind of another package, with its table"),
  `api/native.md` (`kind_spec.jl`, `retired.jl`, `blur_schema.jl`), the four published JSON Schemas deleted.
- Changed in this pass (a74e2d51): IO's `docs/src/index.md` (the landing text said "a camera's radiometry": it is a
  camera's blur; a bullet for the hook of a kind another package owns and the moved kinds' owners), `guide/kinds.md`
  (the envelope's `radiance` unit names its reader), IO's CLAUDE.md ("Kinds of another package"), Core's CLAUDE.md
  (`src/retired.jl` and `src/quality/stack_ci.jl` in Key Files), Core's CHANGELOG (`radius` no longer public, the
  dependency table without PanoramaStitching's row, `nlls_termination`'s specialization). IO's and Estimation's
  "Unreleased" sections were complete against the diff.
- No `@ref` to a moved name is left in any docs page, docstring or convention (searched by every name of Core's and
  IO's `MOVED_NAMES`); the remaining mentions of a moved type are the refusals, their tests, the CHANGELOGs and the
  sentences that name the new owner. READMEs (the root's and the six packages') name no moved type and no version.
- Not changed: the root README (its downstream table has no row for the Visual Radiometry Toolkit; a row needs the
  repository's public address, the owner's to give).

## 3. Gates (Slurm, Julia 1.13.4, fresh clones at a74e2d51; logs under logs/migration/results/)

- Strict docs builds (`checkdocs = :public`, doctests, link check, no `warnonly`): Core (job 5986,
  dp-core-docs-a74e), IO (5987, dp-io-docs-a74e), Estimation (5988, dp-est-docs-a74e): PASS. Core's build prints
  Documenter's size warnings (not errors): `api/lenses.html` 193.0 KiB and `api/cameras.html` 191.2 KiB of the hard
  200 KiB (`api/stack_contracts.html`, with the workspace CI, 178.4 KiB), the search index 1.62 MiB over the 1.56 MiB warning threshold.
- The `docs`, `README`, `readme helpers`, `agent docs` and `stack_ci` agent-docs items (`"docs|readme|README|agent"`):
  Core 74/74 (5990), IO 25/25 (5991), Estimation 23/23 (5992): PASS.
- Not run: Julia 1.12 (the gate script runs 1.13.4 only); the Features, Eval and PoseLib builds (no docs or code of
  theirs changed; their compat lines change later); /simplify and /code-review as separate steps (the pass changed
  prose only).

# VisualGeometryToolkitCore 0.11.1 docs review — branch patch-core0111

Released: Core 0.11.1 (a patch: a 0.11.0 regression). Baseline: origin/main ad9c41e8 (Core 0.11.0, Estimation 0.12.0,
Features 0.9.8). The review covers the tree 885fbb4f. After it, only this record changes (`.release/` is outside the
gate). Code changes in VisualGeometryToolkitCore only; one new test in VisualGeometryToolkitEstimation.

## 1. Public API since Core 0.11.0

- No name added or removed, no signature changed. New internal constant `_NLLS_NONFINITE_BITS` (not public). Compat
  unchanged.

## 2. Docstrings and pages brought up to date

- `nlls_stop_code` (its order: bit 10 is no failure; its stop goes to the postcondition), `nlls_stop_reasons`
  (`:cost_disagreement` stops NLLSsolver one iteration later), `nlls_termination`'s table (rows for bit 10 alone and
  with a tolerance bit; the non-finite rows without 10), `lmsolve_chart!`'s success rule, `LmStops`' `stationarity_tol`
  and `residual_resolution` rows (read again by a host verdict), `StopRule`'s `NoRule` (and its per-value doc),
  `configs/default.toml`'s `stationarity_tol` comment, the 0.11.1 CHANGELOG entry.
- Docstring placement: `_NLLS_NONFINITE_BITS` and its comment sit ABOVE `nlls_stop_code`'s docstring, which stays
  attached to the function (the simplify review caught the first placement detaching it).
- `docs/src/api/termination.md` lists `NonFinite` and the record types only; nothing there states the bit decode, so
  it is unchanged. README unchanged (no example touches the decode).

## 3. Gates

- Strict docs build (`checkdocs = :public`, doctests, no `warnonly`) for Core and Estimation, the `docs`, `README`,
  `quality`, `naming` and `agent docs` items: in the release gate at the head carrying this record.
- No local Julia (kaimon down): every check ran on Slurm.

- Eval 0.10.1 at ab9070e0: the digest cache. `digest_cache_dir` added (public, docstring, in the top module's
  `@autodocs`); the docstrings of `file_sha256` (its cache across processes, the 64 MiB and 3 s thresholds, the
  opt-out), `tree_sha256`, `missing_inputs` and `verify_inputs`' record (`files`) say what the code does; the
  reproduce guide's input and digest-cache items, the site template's `[env]` example and the CHANGELOG. No other
  name added or removed. Simplify (reuse, simplification, efficiency, altitude) and code review (high) on the diff,
  their findings applied.
