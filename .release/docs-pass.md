docs-pass: 4a72c8a8
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 9cb964e (searched, not edited: no page names the spherical spline, the dilogarithm or the artifact store)

# Core 0.13.0, Eval 0.10.0, Features 0.9.10, IO 0.11.1, Estimation 0.12.2, PoseLib 0.3.3 docs review — branch core-store-cleanup

## 1. Public API since the 0.12.0 cascade (862ebba7)

- Core, removed: `SphericalSplineSystem`, `spherical_spline_kernel`, `spherical_spline_coefficients`,
  `spherical_spline_evaluate` (and their methods of `nodal_weights`, `bending_energy`, `bending_factor`,
  `bending_rows`), `dilog`. Added: `ArtifactStore`, `artifact_store`, `with_artifact_store`, `set_site_artifact_store!`,
  `store_file`, `download_verified`, `store_name_clashes`, `migrate_artifacts`. Changed: `install_artifact` (a `store`
  keyword; installs into the store).
- Eval, added: `site_root`, `active_site`, `site_profile`, `project_name`, `MACHINE_PATH_DIRS`, `MACHINE_PATH_EXTS`,
  the `Site` storage fields, the checks L5, L6, D5, `:storage` A2–A4, `:release` R1–R3. Features, IO, Estimation,
  PoseLib: no name added or removed.

## 2. Docstrings and pages

- Core: the removed names' docstrings, the `docs/src/api/batched.md` paragraphs and ARCHITECTURE.md lines deleted (no
  `@ref` to them is left); a new page `docs/src/api/artifacts.md` (the store: its resolution table, the install, the
  single files, the migration); every new name has its docstring (`base/artifacts.jl`).
- Eval: `docs/src/paper-projects.md` (the rules, the enforcing test's table with L5, L6, D1, D5, `:storage`,
  `:release`, a section "Machines and storage"), `docs/src/reproduce.md` (the shipped `vcai` site, `[storage]`), the
  site template, the paper-project template's `data/README.md`, `README.md` and `scripts/fetch_data.jl`.
- CHANGELOGs of the six packages.

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
