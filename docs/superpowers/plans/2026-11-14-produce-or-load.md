# produce_or_load: one keyed-step cache for every pipeline step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native; the coordinator's brief fixes execution).
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** every step of every pipeline (decode/prepare, detect, index, the start state, refine, evaluate, figures) caches its
intermediate with DrWatson's `produce_or_load`, keyed by ONLY the config fields and inputs that step reads, so a later step's
setting never recomputes an earlier step.

**Architecture:** VisualGeometryToolkitEval replaces its own caching (`keyed_cell_dir`, `keyed_data_dir`, `fresher`,
`run_complete`, `cached_result`, `legacy_hash_key`, the `Stage(name, desc, done, run)` done-closures) by ONE keyed step,
`Step` + `produce`, built on `DrWatson.produce_or_load` (`loadfile = false`, the save through a `DrWatson._wsave` method that
calls `VisualGeometryToolkitIO.save`). `run_key` stays the key function; `run_stages` stays the orchestrator, its stages now
declare their steps. RadiometricCamera's refinement splits its start (front end, start lens with its image circle, PnP poses,
masks, vignetting centre, focus start, the closed-form radiometric start, the GCV weights) into its own step, written as a start
record by `bin/refine.jl --stage=start` and the ONLY input of every refinement (`bin/refine.jl --start=<dir>`).

**Tech stack:** Julia 1.13, DrWatson 2.19.1, VisualGeometryToolkitIO records, ReTestItems suites on dgx01 Slurm.

**Spec:** the owner rule of 2026-11-14 and the coordinator's rulings (relayed in the brief; recorded in
`~/src/produce-or-load-handover.md`).

## Global Constraints

- REPLACE, don't add: no alias, no forwarding wrapper, no second mechanism beside the step. Every name that leaves is refused BY
  NAME with its replacement (`Experiments.<name>` → an `ArgumentError`/`error` naming `Step`/`produce`/`produced`).
- Everything through VGTK IO: records saved with `VisualGeometryToolkitIO.save` (through `DrWatson._wsave`), loaded with
  `VisualGeometryToolkitIO.load`; never JLD2/BSON/FileIO for records.
- Each step's key = only its own dependencies (its config section, its inputs, its upstream steps' keys).
- DrWatson's layout: a step's directory is `datadir("exp_pro", <experiment>, <step>, <key>)`; raw inputs `datadir("exp_raw", …)`;
  final assets `plotsdir()`. `results_root()` IS `datadir("exp_pro")` (its ENV binding `VGTK_RESULTS_ROOT` only relocates it, e.g.
  dgx01 `/raid/data/jpr/rc-thorlabs-run/results/<tree>`), `data_root()` IS `datadir("exp_raw")` (`VGTK_DATA_ROOT`).
- Provenance kept: `_provenance.toml` (DrWatson's tagged dict + `code_trees`) and `_config.toml` in every step directory, the
  provenance written LAST (the completion mark). DrWatson's `tag = true` is NOT equivalent (it tags dicts only and keeps no code
  trees) and is not used.
- Stack rules: no duplication (check_duplication with VGTK_STACK_PATHS), latest-only, TOML literals only, `fixed` keyword,
  everything through IO, GPU-first, no global run state.
- Monorepo commits carry no AI attribution; RC commits end with the two co-author lines of the brief.
- Never release, push to main or post.

## Design decisions (the implementer cannot make these alone)

1. **The unit is the step's keyed directory.** `produce_or_load`'s file is `<step dir>/<output>` (`filename = _ ->
   joinpath(key, output)`, `prefix = ""`, `suffix = ""`, under `path = <exp_pro>/<experiment>/<step>`). `f(dir)` either returns a
   record (saved by produce_or_load through `_wsave(file, ::Saved)` → `VgtkIO.save(file, record)`) or `nothing` after writing
   `<output>` itself (a child process, a directory of outputs — `Written()`; its `_wsave` only checks the file is there). A
   step that writes a directory of outputs (detections/<photo>.tsv, a decoded set) is ONE unit per step (not per photo): its
   `output` is the file it writes last (`summary.tsv`, `conversion.json`, `parameters.json`), so a half-written directory never
   reads as produced.
2. **Reuse** (`produced(step)`): `<output>` and `_provenance.toml` present, the recorded `code_trees` the current ones and the
   scratch flag this run's — the one reuse rule `run_complete` and `cached_result` applied, now in one place. `produce` passes
   `force = !produced(step)` to `produce_or_load`. DrWatson swallows a failed save (saving_files.jl:107-116): `produce` checks
   the file after the call and throws.
3. **Stages declare steps.** `Stage(name, desc; steps, run)`: `steps()` the stage's `Step`s, `run(step)` produces one (calls
   `produce`); `Stage(name, desc; check)` a stage with no product (an input check), run every time. `run_stages` prints
   `produced/total` per stage under `--plan`, and calls `run(step)` for every step of a selected stage — `produce` loads or
   produces (the produce_or_load spirit); `--force` is passed by the driver (`produce(…; force)`). The positional
   `Stage(name, desc, done, run)` is refused by name.
4. **Upstream** is a `Step` (or a key string): `Step(…; upstream = ("start" => start_step,))` folds `start_step.key` into
   `run_key`'s `keyed_with.upstream`.
5. **Keys of record are unchanged where the record is.** A step's key is `run_key(record; files, upstream, parts)` exactly as
   before, so a step whose record did not change keeps its key; only its directory moves (`<exp>/<stage>-<key>` →
   `<exp>/<stage>/<key>`): `migrate_keyed_dirs(root; dry_run = true)` (experiments/rekey.jl) moves every keyed directory whose
   `_provenance.toml` records that `run_key`, refusing a destination that holds another run. Steps whose record changes re-key
   and rerun: RC's `refine` (it now reads its start step) and the new `start` step.
6. **RC's start**: the start record is a directory `start/<key>/`: `start.json` (`rc/refinement_start/v1`: names, the image
   size, fpx, R0, d_c, f_m, the self-check bearing errors, the start's page flip / board y sign, the vignetting centre, the costs
   of the closed-form start, μ_lin, the vignetting and grid GCV selections, cstar_nc, the start options), the start lens
   (`grid.json` + `initialization_region.json`, or `lens.json`: the front end's lens with the run's image circle — what
   `start_from_init` and `load_start` read), `poses_init.json` (the start poses), `state.json` (the state after the closed-form
   start) and `masks/<k>.png` (each view's final mask). `refine` ALWAYS starts from a start record — `bin/refine.jl` without
   `--start` writes the start into `io.out/start/` and continues from what it wrote — so a loaded start and a computed one are
   one code path: bit-identical by construction, proven by the dgx01 gate (Task 9).
7. **RC's start key** is the run's options with the refinement-only options set to their defaults (`start_options(opts)`):
   `RadiometricCamera.REFINE_ONLY_KEYS = ("lm.max_iter", "lm.max_time", "lm.log", "io.checkpoint_each", "sweep.iterations",
   "sweep.lm_chunk", "resume.iteration_offset", "resume.sweep")`, declared in RC (cvpr2027's copy leaves). The start is COMPUTED
   from `start_options(opts)`, so a start that read a refine-only option would read its default: the exclusion cannot be wrong
   silently. `resume.state` keys the start (the front end reads its page flip). `bin/refine.jl --start` refuses a start whose
   recorded start options differ from `start_options(opts)`.
8. `io.save_init` is retired (refused by name: "bin/refine.jl --stage=start writes the start record"); cvpr2027's `[init.*]` /
   `init = "self"` runs become start steps.

## File structure

Monorepo (`~/src/vgtk-pol`, branch `produce-or-load`):
- Create `VisualGeometryToolkitEval/src/experiments/steps.jl` — `Step`, `step_dir`, `produced`, `produce`, `Saved`, `Written`,
  the `_wsave` methods, `load_output`.
- Modify `src/experiments/stages.jl` — `Stage(name, desc; steps, run)` / `Stage(name, desc; check)`, `run_stages`; `fresher`
  leaves (refused).
- Modify `src/experiments/run_records.jl` — `keyed_cell_dir`, `keyed_data_dir`, `experiment_dir` leave (refused);
  `_upstream_key` takes a `Step`.
- Modify `src/experiments/provenance.jl` — `run_complete`, `cached_result`, `legacy_hash_key` leave (refused); `_reuse_changes`
  is `produced`'s.
- Modify `src/experiments/rekey.jl` — `migrate_keyed_dirs`.
- Modify `src/Experiments.jl` exports, `src/experiments/data_policy.jl`, templates, docs, CHANGELOG, Eval version 0.8.0 → 0.9.0.
- Tests: `test/steps_tests.jl` (new), `test/stage_runner_tests.jl`, `test/experiments_data_tests.jl`, `test/run_records_tests.jl`,
  `test/rekey_tests.jl`.

RadiometricCamera (`~/src/rc-pol`, branch `rc-produce-or-load`):
- Create `src/calibration_pipeline/start.jl` — `REFINE_ONLY_KEYS`, `start_options`, `save_start`, `load_refinement_start`.
- Modify `src/calibration_pipeline/pipeline.jl` — `prepare_start(opts)` (the start, saved) and
  `prepare_refinement(opts, start)` (the state rebuilt from a start record; front ends take the stored start).
- Modify `bin/refine.jl` — `--stage=start`, `--start=<dir>`; without either: start into `io.out/start`, then refine from it.
- cvpr2027: `src/drivers.jl` (`keyed_run` → `Step`s; `start_step`, `refine_step`; `REFINE_ONLY_KEYS` from RC), every
  `scripts/*.jl` stage table, `src/chrome_evaluation.jl` / `src/frontends*.jl` run lookups, tests.

## Review Focus

1. A step interrupted mid-write (child crash after `parameters.json`, before the records): never reads as produced — the
   provenance is written last and removed before producing (test in Task 1).
2. A save that fails inside `produce_or_load` (DrWatson warns and returns): `produce` throws naming the file (Task 1).
3. A start record written by other start options (an edited `[pipeline]` key outside `REFINE_ONLY_KEYS`, a moved input) reused
   by `--start`: refused naming the differing keys (Task 6).
4. Old keyed directories on dgx01: `migrate_keyed_dirs` moves only directories whose recorded `run_key` is their suffix, and
   refuses a destination holding another run (Task 3).
5. A scratch run (`VGTK_SCRATCH=1`) never reuses or overwrites a run of record's step directory (scratch keys differ; Task 1).

---

### Task 1: Eval `Step` / `produce` / `produced` on DrWatson.produce_or_load

**Files:** Create `src/experiments/steps.jl`; modify `src/Experiments.jl`; Test `test/steps_tests.jl`.

**Interfaces — Produces:**
- `Step(experiment::AbstractString, name::AbstractString, record::AbstractDict; output::AbstractString, files = nothing,
  upstream = nothing, parts = nothing) -> Step` (fields `experiment, name, record, key, output`; `key = run_key(record; files,
  upstream, parts)`, an upstream value a `Step` or a key).
- `step_dir(s::Step) -> String` = `joinpath(results_root(), s.experiment, s.name, s.key)`.
- `produced(s::Step; gitpath = project_root()) -> Bool`.
- `produce(f, s::Step; driver, config_source, inputs = String[], producer = false, fields = Dict(), force = false,
  gitpath = project_root()) -> String` (the step directory; `f(dir)` returns a record or `nothing`).
- `load_output(T, s::Step; kwargs...)` = `VgtkIO.load(T, joinpath(step_dir(s), s.output); kwargs...)`.

- [ ] Step 1: tests — `produce` computes once and loads after (a counter in `f` stays 1 on the second call); the file is
  `<results_root>/<exp>/<step>/<key>/<output>`; a record returned is written by `VgtkIO.save` and read back by `load_output`
  (a `VgtkCore.EuclideanMap`/camera record round trip); `f` returning `nothing` without writing `<output>` throws; a changed
  `record` field is another key/directory, an unchanged one the same; an upstream `Step`'s key folds in (changing upstream's
  record changes the downstream key, changing the downstream record leaves upstream's key); a failing `f` leaves no
  `_provenance.toml`; `force = true` recomputes; a recorded `code_trees` that differs from the current ones recomputes; a
  scratch run's key differs from a run of record's.
- [ ] Step 2: run them on dgx01 (fail: `Step` undefined).
- [ ] Step 3: implement; `produce_or_load(…; loadfile = false, tag = false, verbose = false)`.
- [ ] Step 4: tests pass; commit "Eval: Step and produce — one keyed step on DrWatson's produce_or_load".

### Task 2: the old caching leaves; stages declare steps

**Files:** `src/experiments/stages.jl`, `run_records.jl`, `provenance.jl`, `Experiments.jl`, `data_policy.jl`; tests
`stage_runner_tests.jl`, `experiments_data_tests.jl`, `run_records_tests.jl`, `contracts_tests.jl`, `paper_projects_tests.jl`.

- [ ] Step 1: tests — `Stage(:a, "d"; steps = () -> [s1, s2], run = s -> produce(...))` runs each step once, `--plan` prints
  `1/2 produced`; `Stage(:c, "d"; check = () -> ...)` runs every time; `Stage(:a, "d", () -> true, () -> nothing)` throws
  naming `Stage(name, desc; steps, run)`; `fresher`, `run_complete`, `cached_result`, `legacy_hash_key`, `keyed_cell_dir`,
  `keyed_data_dir`, `experiment_dir` each throw naming its replacement; the contract rule R3 accepts `run_stages(`.
- [ ] Step 2-4: implement, migrate the Eval tests that used the old names to `Step`/`produce`, pass, commit.

### Task 3: `migrate_keyed_dirs`, docs, templates, version

- [ ] tests: a tree with `calibrate/refine-<k>/_provenance.toml(run_key = k)` moves to `calibrate/refine/<k>`; a dir whose
  record names another key is left and reported; an occupied destination is refused; `dry_run` moves nothing.
- [ ] templates/paper_project, docs/src/paper-projects.md, reproduce.md, CHANGELOG (0.9.0, breaking: the names that left and
  their replacements), Project.toml version 0.9.0. Commit.

### Task 4: RC `REFINE_ONLY_KEYS`, `start_options`, the start record

**Files:** Create `src/calibration_pipeline/start.jl`; tests `test/unit/refinement_start_tests.jl`.
- `start_options(opts::PipelineOptions) -> PipelineOptions` (the `REFINE_ONLY_KEYS` at `PipelineOptions()` defaults).
- `save_start(dir, s, meta, opts)`, `load_refinement_start(dir) -> NamedTuple` (fields of design decision 6).
- [ ] tests: `start_options` resets exactly `REFINE_ONLY_KEYS` (every other key kept, `lm.fixed` kept); a key outside
  PipelineOptions in `REFINE_ONLY_KEYS` throws at load; `save_start`/`load_refinement_start` round-trip `start.json` and masks
  bit-exactly on a golden-scene state.

### Task 5: RC `prepare_start` / `prepare_refinement(opts, start)`; `io.save_init` retired

- `prepare_start(opts; backend) -> (s, meta)` (the old body up to the GCV) and `prepare_refinement(opts, start; backend)`:
  the front end takes `stored = start` (no lens resample, no PnP / page-pose fit, no noncentral start, the recorded y sign /
  page flip), the image circle is the recorded lens's coverage, the masks the recorded ones (no Bernoulli, edge selection or cap),
  the photographs streamed at them, `load_state_into!` of `state.json`, `focus_prior!` and the grid prior at the recorded β.
- [ ] tests: `io.save_init` refused by name; `initialization_consumers_tests` updated to `prepare_start`.

### Task 6: `bin/refine.jl --stage=start | --start=<dir>`

- [ ] `--stage=start` writes `start_options(opts)`'s start into `io.out` and exits; `--start=<dir>` refuses a start whose
  `start.json` options ≠ `start_options(opts)` (names the keys), then refines; neither: `io.out/start/` then refine from it.

### Task 7: cvpr2027 drivers on steps

- `src/drivers.jl`: `keyed_run` → `driver_step(stage, record; experiment, output, upstream)`; `prepare_step`, `start_step`
  (record = `init_settings` of the case with RC's `REFINE_ONLY_KEYS`, the inputs, the prepare/init upstream), `refine_step`
  (record = the case's refine-only settings + views, upstream = its start step), `run_start`, `run_refine` through `produce`;
  `prepare_done`/`refine_done`/`figures_done`/`pending` leave.
- every `scripts/*.jl` stage table to `Stage(…; steps, run)` / `Stage(…; check)`: calibrate, radiometric_refinement,
  detector_bias, frontends, chrome_index_grid, chrome_evaluation, panorama_stitching, panorama_vignetting, texture_recovery,
  geometric_figures, vignetting_figures, sensor_characterisation, numbers, regenerate_paper.
- figures: the step writes its assets in its directory; the stage's `run` copies them into `plotsdir()` after `produce`.
- tests: `drivers_tests.jl` — a refine-only change (`lm.max_iter` 24 → 50) leaves the start step's key and changes the refine
  key; a start input change (`grid.size`, an input path) changes both; an evaluate-only change reuses the refine step (chrome
  evaluation: `[evaluate]` key change keeps the index/refine keys).

### Task 8: the cascade (plan; each repo adopts Eval 0.9 on its own branch after Eval 0.9 registers)

| repo | what moves | size |
|---|---|---|
| LocalOptimizationDoneRight accv2026 | src/cell_stages.jl (9 stages: run_complete, fresher, oracle_floor_done), 17 scripts' Stage tables, protocol_folds/e2e_timing/paper_numbers/superansac `experiment_dir` | large |
| RansacScoringDoneRight neurips2026 (+ untracked cvpr2027) | fig_profile, fig_sensitivity, paper_numbers, pool_selection Stage tables (run_complete, fresher) | small |
| BlobBoards bmvc2026 | src/cache.jl `cached` (cached_result → produce), detection_cv(_replay) stages, 11 single-stage `() -> false` scripts → `Stage(…; check)` | medium |
| CaliBlobs icra2026 | src/BlobBoardsIcra.jl `cached` (cached_result → produce) | small |
| LightRigCalibration thesis | calibrate, register, evaluate Stage tables (run_complete, fresher) | small |
| PanoramaStitching | none (no Eval dependency) | — |
| monorepo Core/IO/Features/Estimation/PoseLib | none | — |

### Task 9: gates on dgx01

- Eval suite (Slurm, nice 0, `--mem-per-cpu=8480M`), RC unit suite, cvpr2027 suite, check_naming / check_duplication.
- Bit-identity: `siemens_coarse_full` through `scripts/calibrate.jl` — run A (`lm.max_iter` M), run B (same tree, `lm.max_iter`
  M′: the start step loads, its directory untouched), run C (fresh tree, `lm.max_iter` M′): B's refine outputs equal C's bit for
  bit (`test/pipeline/compare_outputs.jl`); the golden gate against `golden/` (the start now passes through its record: a
  re-record if it moves, said in the commit).
- /simplify, /code-review high on both diffs.
