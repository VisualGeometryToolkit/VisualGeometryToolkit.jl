# VisualGeometryToolkitEval changelog

## Unreleased

- **Breaking: the scale-multiple width axis is `EfficiencyConstant` (`:efficiency_constant`)** (VisualGeometryToolkitCore's
  rename): grids, arms and tuning keys read and write the new name. Sidecars persisted under the old axis name are
  migrated by the sidecar step (B6).
- **Breaking: tuning artifact format 7, the efficiency-constant names** (the stack's robust-estimation contract,
  2026-11-11). A tuning sidecar, checkpoint or bundle is written with the `efficiency_constant` names: the
  `efficiency_constant` axis (grid spec and display axis), `efficiency_constant_window`, and boundary hits named
  `efficiency_constant_lo`/`_hi`; its efficiency constant is the plain multiple (`width = c·ŝ`) for every loss.
  `TuningArtifact` gains `format` (the format the file was written in) and `recorded_hash` (the file's own hash).
- **A format-6 tuning sidecar is still read**, never rewritten: `load_tuning_artifact` carries it over in memory
  through its version step (`VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("vgtk_eval/tuning/v6")}, tree)`, run by
  `migrate_record`), which checks its recorded hash against the stored table and renames the old keys of the spec,
  the grid specs and the axis-named data (`boundary_hits`, `kinds`, a coarse-fine artifact's `axes`, `selected`,
  `boundary_hit`, `record.param_name`). `deployed_budget_settings` converts a format-6 Geman–McClure arm's
  efficiency constant (a half-weight depth) once, to the plain multiple its code deployed (`decision_width`), so the
  deployed LO is unchanged; every other loss's constant was already plain. Formats 1–5 are refused as before.
  The fixtures (test/data/tuning_v6/) were written by the format-6 code.

- **The local backend runs no stage as a Slurm job**: `run_local!` drops every `SLURM_*` variable of the environment
  the plan was made in, so a stage sees only the ones the backend sets (`SLURM_SUBMIT_DIR`, an array's
  `SLURM_ARRAY_TASK_ID`, the site's `SLURM_CPUS_PER_TASK`). Launched from inside a Slurm array task (the test suite's
  own job), the job's task id and submit directory reached every stage.

- **New: one command reproduces a paper project** (`Reproduce`, src/reproduce/runner.jl, src/reproduce/run_results_branch.jl).
  A project's stage graph of record (`load_stage_graph`: `[[stage]]` tables naming the project's own sbatch launcher, a
  command or the results branch, their `after` dependencies, the driver and config of record, an array over the config's
  cases or a list) runs on a site (`load_site`: `configs/sites/dgx01.toml`, a Slurm node, and `desktop.toml`, one GPU
  and no scheduler, are shipped; `_template.toml` documents every key, and `--site` takes a path). `reproduce_main`
  (`--graph`, `--site`, `--only` with or without `--no-deps`, `--dry-run`, `status`) checks every selected stage's
  `[inputs]` first and reports the missing ones as data (`missing_inputs`: path, tier, used by; exit 3, nothing run),
  submits the stages as sbatch arrays chained by `--dependency=afterok` (`submit_slurm!`; the site's options over the
  launcher's, a GPU limit) or runs them in order under bash with the Slurm variables the launchers read (`run_local!`),
  writes a run manifest per launch (commit, configs, roots, site, per stage its run key, command, job id and state) and
  reuses a stage of the same run key that completed, or depends on one still in flight. `reproduce_status` reads a
  launch back (task states, the first failed task's log and its first error line); `results_branch_of_run` makes the
  launch's renders a `results_branch` from its manifest. runner.jl needs only the stdlibs and src/cli.jl, so a
  project's launcher runs it by path from a bare `julia` before the project's environment is instantiated.

- **`Experiments.load_hand_eye` reads the 2026-09 one-file hand-eye record**: a record without its mount file (the
  `vgio/transforms/v2` file the 2026-09 motion-capture runs hold, which VisualGeometryToolkitIO now reads as the
  current transforms record) takes its marker mount from the file's `target:<id> → body:board_rig` poses.
  `save_hand_eye` still writes the pair.

- **New: `Render.write_station_fan_colorbar(colormap, io_or_path; span, ticks, tick_labels, label, logarithmic,
  reverse, colormap_name, orientation, length, comment)`**: the colorbar of a row of station fans, a pgfplots
  colorbar-only axis, vertical beside the row or horizontal beneath it, the colormap sampled at 15 points and defined
  with its reverse (`<colormap_name>`, `<colormap_name>rev`). Byte for byte the committed bars of BlobBoards' bmvc2026
  (`write_floorplan_tikz`'s colorbar loop) and CaliBlobs' icra2026 (`write_floorplan_colorbar` and its `_fp_cmap`
  colormap string); both restated it.
- **New: `Render.write_station_fan(colormap, io_or_path, cells; value, span, logarithmic, reverse, comment,
  ring_labels)`**: a floor plan's station fan as a bare TikZ picture, one annular wedge per `station_cells` cell
  coloured by `colormap` at the cell's statistic `value` placed in `span` (log10 when `logarithmic`, `1 - t` when
  `reverse`), a cell without a detection grey. Byte for byte what BlobBoards' bmvc2026 `write_floorplan_tikz` and
  CaliBlobs' icra2026 `write_floorplan_fan` (its verbatim copy) wrote; both restated it. The colormap is the paper's
  (`t -> get(cgrad(:RdYlGn), t)`), so the package takes no colour dependency; the tests take PlotUtils.
- **New: `write_numbers(io::IO, nums; order = :name)`**: the number-macro fragment written to an IO, the same bytes
  as the file form, which writes its section through it (one implementation). A figure that inlines its cell macro
  writes `sprint(io -> write_numbers(io, nums; order = :added))`; CaliBlobs' icra2026 `numbers_text` wrote a
  temporary file to read it back.
- **New: `Reproduce.number_provenance(run_dir; root = project_root())`**: the provenance of numbers read from one
  run, from its `_provenance.toml` (`Experiments.read_provenance`): `run_key`, `commit` (the recorded `gitcommit`) and
  `config` (the recorded `config_source`, made relative to `root`), each only where recorded, for `add_number!`'s
  `provenance` or splatted into `PaperNumbers`. CaliBlobs' icra2026 `run_provenance` and LocalOptimizationDoneRight's
  accv2026 `scripts/paper_numbers.jl` restated it.
- **New: `Reproduce.checkout_commit(root = project_root())`**: the checkout's commit as DrWatson describes it
  (`gitdescribe(root)`, `"unknown"` outside a repository), a space or `;` made `_`: the `PaperNumbers` `commit` of
  numbers a driver computes itself. BlobBoards' bmvc2026 `figure_numbers` and CaliBlobs' icra2026 `checkout_commit`
  each restated the line.
- **Fixed: a TeX length is not a re-typed number** (`macro_literal_violations`, and with it the reproduce index's
  `:macro_literal`): a number directly followed by a TeX unit (pt, cm, mm, in, em, ex, bp, pc, sp, mu, dd, cc, nd, nc;
  `\hspace{-0.3cm}`), or with a space where TeX reads a length (a whole braced argument, `\hspace{0.3 cm}`; after
  `\kern`, `\hskip`, `\vskip`, `\mkern`, `\mskip`), is no literal. In prose "0.3 cm" is a number with a unit and is
  still checked, as is "0.3cmx". RadiometricCamera's paper flagged `\hspace{-0.3cm}` against a macro of value 0.3.
- **New: the number-macro writer takes macros with parameters and shared files.** `add_macro!(nums, name, body;
  arity, label, provenance)` adds `\newcommand{\<name>}[<arity>]{<body>}` (a figure's helper, `#1` … `#<arity>` its
  arguments; not a value, so `tex_macro_values` skips it; `gather_numbers` gathers it as written).
  `write_numbers(path, nums; shared = true)` writes one driver's section of a file several drivers write: its section
  alone is replaced, the others kept, the sections ordered by driver (the same file whatever order the drivers ran
  in), a macro another section defines refused; `order = :added` keeps the order the macros were added (default
  `:name`, as before). CaliBlobs' icra2026 TeX writers port to it.

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

- **Internal: one git process for an acceptance's assets.** `accept_results` reads every asset from the branch with one
  `git cat-file --batch` (was one `git show` per asset), and reads the committer identity once per call with one
  `git config --get-regexp` (was two `git config` processes before each commit). 40 assets: about 290 ms to 75–120 ms.
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
