# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VisualGeometryToolkitEval: the evaluation base of the visual-geometry stack —
statistics (`Stats`), the codified cross-validation (`Tuning`, `Evaluation`),
the composition evaluator of scorer + local-optimization arms
(`Evaluation`, `src/evaluation/engine/`: `PairCtx`/`build_pair_ctx`, the one
refit layer `refit_imc!`, `LoRefitSpec`/`apply_lo_refits`, the sample-budget
engine, `live_runs`, rosters and their gate-enforced readers,
`stream_selections`), the stage and run-record framework (`Experiments`),
figure and table rendering (`Render`), scoring-pool IO (`Pools`), the raw
dataset readers (`Datasets`), the published-data catalog (`DataArtifacts`), the
reproducible-figures contract (`Reproduce`: a paper project's REPRODUCE.md
asset index checked against what its tex reads) and the paper-project
conventions (`PaperProjects`: `paper_project_violations`, the one check every
paper project's suite calls, and `new_paper_project`, the template whose files
live in `templates/paper_project/`; README "Paper project conventions"). It
depends on VisualGeometryToolkitCore, VisualGeometryToolkitIo and VisualGeometryToolkitEstimation only.
The evaluator is written against VisualGeometryToolkitCore's contracts (scoring, local
optimization, `RefitFamily`, `refine_external!`, `select_on_grid!`), never
against a paper scorer or strategy: RansacScoringDoneRight,
LocalOptimizationDoneRight and PoseLib must not appear in its deps, weakdeps,
extras or test target. The projects that evaluate those depend on it.
Each submodule's API is reached through it (`using
VisualGeometryToolkitEval.Pools: load_pool`); the top module exports the submodules
and the driver CLI (`cli_option`, `start_output_flush!`).

## Testing

ReTestItems: `Pkg.test(; test_args = ["pool_io|tuning"])` filters items by
name. Check the reported total.
The quality items (Aqua, ExplicitImports, JET ratchet at 0, calling
conventions, undefined globals, the stack's naming check) run in the default
suite.

## Defaults of record (TOML is the authority)

- `configs/default.toml`: every value that changes a reported statistic, a
  rendered panel or a raw read, read only through `_config_default(table, key)`
  (which throws on a missing key). Parsed at precompile: an edit needs a
  session restart.
- `configs/tuning.toml`: every cross-validation parameter — grid spacings,
  window rules, extension rules, argmin rule, objective, the cross-validation,
  the budget draws — parsed by ONE owner, `Tuning.tuning_spec`; unknown keys
  and tags refuse.
- `configs/splits.toml`: every scene list of the stack, named (never
  restated) by a spec's `cross_validation.scene_split`. Every random sample is
  seeded by `Stats.stable_seed` from a spec seed and a NAME.
- Persisted artifacts carry a format stamp (`ARTIFACT_FORMATS`,
  src/artifact_format.jl): one writer stamps each kind, one reader refuses any
  other format. An older file is converted or regenerated, never shimmed.

`src/render/axis_style.jl`, `src/evaluation/cv_probe.jl`,
`src/experiments/stages.jl` and `src/reproduce/reproduce_index.jl` are
dependency-free by contract: environments that
cannot load this package include them by path.

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "🤖 Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization ("frame in =
frame out"), pixel origins, bearings, Jacobians, units and the stack
contracts — live ONCE, in VisualGeometryToolkitCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

## Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

### The pool frame here

A scoring pool (`Pools.ScoringPool`, VisualGeometryToolkitCore `PoolPair` records)
is stored in `NormalizedImageFrame` and says so (`Pools.frame`); a PnP pool is
in `ImageFrame`. The pool and residual files hold plain data with the stored
maps (`source_map`, `target_map`); a value leaves the pool frame only through
those maps. Nothing in this package converts a frame. The raw readers
(`Datasets`) return pixels.
