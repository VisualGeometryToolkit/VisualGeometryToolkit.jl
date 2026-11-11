# VisualGeometryToolkitEval changelog

## Unreleased

- **New: the paper projects' shared support** (BlobBoards' bmvc2026 and CaliBlobs' icra2026 restated each):
  `Experiments.assert_active_project(marker)` (DrWatson's `projectdir()` is the paper project, or an error naming
  both), `Experiments.save_hand_eye(file, X, Y)` / `load_hand_eye(file)` / `hand_eye_mount_file(file)` (a hand-eye
  calibration's two records: X `body:camera_rig → camera_optical` and the marker mount, a `Rig{BodyFrame{:board_rig}}`
  of pose-only targets), and `Render.station_cells(df; stations)` (a floorplan fan's cells per rig station and
  distance, the edge-on ±90° stations dropped, an empty station a zero). The bmvc2026 `_diag`/icra2026 `_fit_diag`
  field probe is not hoisted: it reflects over a fit's fields, which the stack's accessors rule forbids; the papers
  read the FitResult's accessors instead.

- **New: `engine_fixtures()`** (public, not exported; a Test extension, `using Test` loads it): the evaluation engine's
  shared test fixtures, the module `EngineFixtures` exporting `epair` (THE synthetic E pool pair, `EPAIR_GT_VARIANCE`),
  `starts_pool`, `subpair(pair, cols)` (a pool pair restricted to some of its stored models, built through the
  keyword constructor, not by field reflection), `engine_scene(pairs, residuals, r)` (a scene of the selection
  stream), `roster_tuning()` and `write_roster_sidecar(path, tuning, arms; code, mode, hits)` (the tuning record the
  roster readers are tested on). This package's engine test and LocalOptimizationDoneRight's accv2026
  engine-integration test, which copied them, build on it.

- **New: `Experiments.config_of_record(name; root = project_root())` and `keyed_cell_dir(cfg, experiment, kind)`**:
  a project's config of record `configs/<name>.toml` loaded as a run config (`load_run_config`), and the keyed
  directory of one cell, `<experiment_dir(cfg)>/<experiment>/<kind>-<run_key(cfg)>`, beside `experiment_dir`. Every
  paper project spells its keyed runs this way; LocalOptimizationDoneRight's accv2026 defined both.

- **New: `DataArtifacts.project_catalog(root = project_root())`**: a paper project's catalog, its
  `configs/data_artifacts.toml` loaded by `load_catalog` (bound to its Artifacts.toml or the catalog's env override).
  RANSACScoringDoneRight's neurips2026 and LocalOptimizationDoneRight's accv2026 each defined it as
  `artifact_catalog()`; `paper_project_violations` defaults its `catalog` to the same path.

- **New: `PaperProjects.paper_figures_dir(paper = "paper/main.tex"; root = project_root())`**: the directory a paper
  reads its generated assets from, `figures/` beside its main .tex under the project root — where a driver's deploy
  stage puts its renders. RANSACScoringDoneRight's neurips2026 and radiometric-camera's cvpr2027 each defined it as
  `paper_figdir()`; the template's driver reads it (`paper_figures_dir(PAPER)`, the paper named once in the
  project's `src/<Name>.jl`) and the template builder places each paper's `figures/` through it.

- **Internal: one dirtiness rule, Core's GPU names.** `results_branch` reads a checkout's HEAD and its dirtiness through
  the package's `_git_identity` (LibGit2, as a tuning artifact records it; was `git status --porcelain` in a
  subprocess), and the GPUs of a run through Core's `device_names` when `nvidia-smi` is absent.
- **New: results branches, a full reproduction under review** (`Reproduce`, src/reproduce/results_branches.jl).
  `results_branch(dir, assets; run_keys, tiers, machine, gpus, slurm_jobs, wall_times, …)` puts a reproduction's
  paper-facing assets (number macros, tables, figures) on a branch `results/<UTC yyyymmdd_HHMMSS>-<short commit>`
  of the paper's repository, created in a worktree of its own under a scratch root (`VGTK_RESULTS_WORKTREES`, else
  `<repository>-results` beside it), without touching the user's checkout; the branch carries `RUN.toml` (the code
  commit, each data tier's TIER.toml/MANIFEST sha256 or artifact git-tree-sha1, the run keys, the machine, GPUs,
  Slurm jobs, wall times) and `COMPARISON.md` (every macro against the base branch: its delta and IMPROVED,
  DEGRADED or UNCHANGED, CHANGED without a direction; every asset's verify verdict; changed PNGs side by side).
  `accept_results(dir, branch; replace_submitted)` deploys the assets into the base through `deploy_assets` as one
  merge commit and tags it `accepted/<stamp>`; `remove_results_worktree(dir, branch; delete_branch = false)` removes
  the worktree. A failure after the worktree is created removes the worktree and the branch; RUN.toml warns and
  records `missing` when no tiers, GPUs or wall times are given (GPUs detected by nvidia-smi or CUDA.jl); the
  acceptance tags before it moves the base and deletes the tag when the base cannot move, and warns when the base moved
  since the run. `add_number!` and `add_table_row!` take `better = :lower | :higher | :none` (or its String),
  recorded in the provenance comment (`; better=…`). A fast-forward of the user's checkout of the base is a
  `git merge` there and runs that checkout's `post-merge` hook; the commits `accept_results` makes skip hooks.
  The paper-project template's driver renders under `plots/example/` and deploys into the paper in a `deploy` stage
  (`deploy_assets`; `--replace-submitted` signs off a changed asset); with `--results-branch` it skips `deploy` and
  makes a results branch of its renders, recording its synthetic draws (`samples.tsv`) as the data tier `example_samples`.

- **New: the one writer of a paper's number macros** (`Reproduce`, src/reproduce/number_macros.jl, Base only).
  `PaperNumbers(; driver, config, run_key, commit)` collects a driver's numbers; `add_number!(nums, name, value;
  label, provenance, format...)` adds a number of the prose or a value a figure reads (`label = "fig:…"`);
  `add_table_row!(nums, row, column => value pairs)` adds a table row as one macro per cell and returns the row's
  tex; `write_numbers(path, nums)` writes the fragment, sorted by name, each macro after its provenance comment
  (`% provenance \<name>: driver=…; config=…; run=…; commit=…[; label=…]`). `tex_number` typesets a number
  (`digits` as Printf's `%.<d>f`, or `sigdigits` without an exponent; `percent`, `group_from`, `unit`, a range) and
  refuses a NaN or an infinity; `number_macro_name(prefix, parts...)` builds a letters-only name, digits spelled
  out. `gather_numbers` carries each macro's provenance comment into the numbers file, and `deploy_assets` deploys a
  numbers file that differs from the submitted one in its `% provenance` lines alone (a rerun at a new run key or
  commit) without sign-off; any other change, a changed value, a hand-written comment or whitespace, still needs it.
  The verify report gains the column `provenance_only`. The paper projects'
  own writers (radiometric-camera cvpr2027, BlobBoards bmvc2026, LocalOptimizationDoneRight accv2026,
  RANSACScoringDoneRight neurips2026, CaliBlobs icra2026) port to it.

- **Breaking: `load_sparse_matches(path)` returns a `Csponds` carrying the per-match confidence as its attribute column
  `confidence`** (`withattrs`; read `cs.confidence`), formerly the pair `(; csponds, confidence)`. The consolidated
  record's layout is unchanged.

- The dataset readers (KITTI, ScanNet-1500, the stereo datasets) build their cameras with Core's renamed keyword
  `CameraModel(K; pixel_frame = :opencv)` (formerly `image_origin`).

## 0.6.1 (2026-11-09)

- On VisualGeometryToolkitCore, VisualGeometryToolkitIO and VisualGeometryToolkitEstimation 0.8 (compat `"0.8"`).
- `reproduce_index`'s table-row splitter strips the outer pipes with `chopprefix`/`chopsuffix` (the same cells); JET
  no longer reports `lastindex(::IO)` on Julia 1.13.

## 0.6.0 (2026-11-07)

- **Breaking: `MsacSpec` is `LossWidthSpec(name, scoring, loss_width, lo = NoLocalOptimization();
  refinement_loss_width)`**, and `evaluate_sample_budget`'s keyword `msac_specs` is `loss_width_specs`. A column
  selects with any loss-width family's template (`width_axis(scoring) == LossWidth()`: Core's `MsacScoring(dg)` or
  `RansacScoring(dg)`, or a paper's family with `with_width` and `score!`, such as RANSACScoringDoneRight's MLESAC,
  MAGSAC++ and loss-cost families) built at `loss_width`; its codimension must be the problem's. Under
  `NoLocalOptimization` the column is selection only: its LO checkpoints record the selected model's `model_imc`
  (the statistic of every checkpoint), with no refit and no LO time. `MsacSpec(:msac, w, lo)` is
  `LossWidthSpec(:msac, MsacScoring(dg), w, lo)`. The warm-up now times each refiner type once (formerly the first
  spec's refiner, broadcast to every column).
- **`Tuning.layered_tuning_entry(cfg, name; under, over) -> Dict`** (exported): the `[tuning.<name>]` entry with a
  campaign's protocol layered around it, as `tuning_spec` reads it (which now calls it), before it is read.
  `under` is filtered by the scheme's TOML keys (`allowed_keys`), not its field names: a Greek field is kept by its
  ASCII spelling (LocalOptimizationDoneRight's `protocol_tuning` filtered by `fieldnames`). The result is always a
  fresh deep copy, layered or not: editing it never edits the config.
- **`Pools.pool_size(path) -> Int`** (exported): a light pool file's pair count from its header, under
  `load_pool`'s format check, without loading the pairs.
- **`Render.results_by_method(summary::DataFrame)`** (exported): the per-`(method_key, tuning_criterion)` bucketing
  of a summary already in hand; `load_results(path)` reads the file and calls it. Its buckets are `SubDataFrame`s
  (RANSACScoringDoneRight's `sensitivity_data` copy returned `DataFrame`s of the same rows).
- The tuning hash and an artifact tarball's digest use VisualGeometryToolkitCore's `sha256_hex` (no value changes).

## 0.5.0 (2026-11-06)

VisualGeometryToolkitCore 0.6, VisualGeometryToolkitIO 0.6 and
VisualGeometryToolkitEstimation 0.6.1 (compat `"0.6"`, `"0.6"`, `"0.6.1"`): the
breaking upstream releases; no change to this package's own code.

## 0.4.3 (2026-11-04)

VisualGeometryToolkitCore 0.5.3 (compat `"0.5.3"`).

- `Stats.stable_seed` is VisualGeometryToolkitCore's `stable_seed`, hoisted there
  so that packages outside the evaluation stack can seed by name without
  depending on Eval. `Stats` re-exports Core's function (no copy); the
  derivation and every seed it gives are unchanged.
- To be removed: the re-export goes in the next breaking release, with the
  VisualGeometryToolkitEstimation 0.7 wave. Import `stable_seed` from
  VisualGeometryToolkitCore (`using VisualGeometryToolkitCore: stable_seed`).
