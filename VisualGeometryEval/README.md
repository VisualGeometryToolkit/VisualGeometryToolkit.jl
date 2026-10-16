# VisualGeometryEval

The evaluation base of the VisualGeometry stack: the statistics, the codified
cross-validation, the experiment framework, the figure renderers, the
scoring-pool IO, the raw dataset readers, and the composition evaluator of
scorer + local-optimization arms that the stack's research projects share. It
depends on VisualGeometryCore, VisualGeometryIo and RobustVisualGeometry (the
core RANSAC loop and problems) only. The evaluator is written against
VisualGeometryCore's scoring and local-optimization contracts, so it runs any
scorer with any local optimizer; a project evaluating RansacScoringDoneRight's
scorers or LocalOptimizationDoneRight's strategies brings those packages itself.

| Module | What it holds |
|---|---|
| `Stats` | finite and lower medians, the validation reference scale, the eligibility-ruled mAA, BCa bootstrap CIs, mean-of-scene-median rows, `stable_seed` and the pool RNG |
| `Tuning` | the cross-validation specification: grid spacings and `grid`, window rules, grid extension, objectives, scene splits and unit samplings (`ExhaustiveSampling`, `StratifiedSampling`, `UniformSampling`), budget draws, the schemes (`CoarseFine`, `WindowedBootstrap`, `Factorial`), the scale-prior strategies (`PriorSpec`, `fit_prior`), the `[tuning]` TOML parser/writer and the tuning-artifact writer/reader |
| `Evaluation` | the single CV path of a scalar (`coarse_fine_cv`: an expanding coarse probe, then a fine free argmin), the bootstrap tuning harness (`WindowedGridMethod`, `compute_method_rows`), and the composition evaluator (see below) |
| `Experiments` | the stage runner (`Stage`, `run_stages`: `--plan`, `--stage`, `--force`), `run_driver`, `require_input`, run keys and run records (`_config.toml`, `_provenance.toml`), the codification contract check, and the one data layer of every paper project: the root ladder (`set_project_root!`, `data_root`, `results_root`; DrWatson's `datadir("exp_raw"/"exp_pro")`), the artifact view (`ArtifactView`, `install_artifact`, `artifact_view!`), the input write guard (`assert_not_input`), the provenance record (`write_provenance`, `read_provenance`: DrWatson's tagged dict, the retired layouts read too), `run_complete` with commit drift, the keyed cache (`cached_result`, `legacy_hash_key`) and the data policy checks (`data_policy_violations`) |
| `Render` | PGFPlots emission: the house axis style, the faceted-arm grammar, `PaperStyle`, panel and table emitters |
| `Pools` | the scoring pool (`ScoringPool`, `PoolResiduals`, bound by `pool_key`), its one writer and reader, `write_scene_pool`, the PnP pool file |
| `DataArtifacts` | a paper project's published input data as Julia artifacts: the catalog (`load_catalog`: kind, mount, Zenodo record, the config of record a file list derives from), `catalog_violations`, the author-side packer (`pack_artifacts`, `zenodo_url`) and the consumer fetch into the artifact view (`fetch_artifacts!`); `split_scenes`; an entry too large for one upload is split per scene (`split_by`, `artifact_parts`) |
| `Datasets` | raw readers returning pixel pairs: PhotoTourism (`load_pt_pairs`), HEB (`load_heb_pairs`), consolidated RoMa / SP+LG csponds (`load_roma_pairs`); and their producer from the depth-validated stereo datasets (`ETH3D`, `ScanNet1500`, `KITTI`, `SevenScenes`: `stereo_pairs`, `load_sparse_matches`, `load_validation_data`, `gt_config`, `consolidate_csponds`) |
| `Reproduce` | the reproducible-figures contract: `paper_asset_graph` (every file a paper reads), `read_reproduce_index` (a REPRODUCE.md asset index), `reproduce_index_violations` (where they disagree) |
| `PaperProjects` | the paper-project conventions: `paper_project_violations` (the one check a paper project's suite calls: layout, data policy, asset index, configs of record, provenance, data catalog) and `new_paper_project` (the template, on DrWatson's `initialize_project`); see "Paper project conventions" |

The evaluation protocol and the `[tuning]` schema are documented in
[docs/src/protocol.md](docs/src/protocol.md) and
[docs/src/tuning.md](docs/src/tuning.md); `julia --project=docs docs/make.jl`
builds the site with the API reference.

```julia
using VisualGeometryEval.Tuning: tuning_spec
using VisualGeometryEval.Evaluation: coarse_fine_cv
using VisualGeometryEval.Pools: load_pool

spec = tuning_spec(cfg, "default")          # a [tuning.default] table over configs/tuning.toml
pool = load_pool(path)                      # NormalizedImageFrame, format-checked
```

## The composition evaluator (`Evaluation`)

Evaluates a scorer (selection) composed with a local optimizer (refinement)
on stored pool pairs. Every arm is a pair of values implementing
VisualGeometryCore's contracts: a scoring (`score!`, `select_on_grid!`,
`with_width`) and a local optimization whose `RefitFamily` routes it either to
`NativeRefit` (`locally_optimize!` inside RobustVisualGeometry's loop) or to
`ExternalRefit` (a third-party model → model refine, `refine_external!`).
The evaluator holds no scorer or optimizer of its own.

| Piece | What it does |
|---|---|
| `PairCtx`, `build_pair_ctx(problem_type, pair; scoring)` | one pool pair's pool-frame problem, calibrations, GT pose and refit-scoring template, built once and shared by every refit |
| `refit_imc!(ctx, lo, seed, nat′; s2)`, `guarded_refit` | THE refit layer: one refit of a seed model, its pose error and wall time, with the seed fallback |
| `LoRefitSpec`, `apply_lo_refits(problem_type, specs, pairs; scoring)` | post-LO error matrices (pairs × grid columns) from a selection sweep |
| `stream_selections(refits_of, T, problem_type, loaders, grids, scoring, backend, batch)` | the scene stream: stage, `select_on_grid!` per grid, then one `apply_lo_refits` |
| `SampleBudgetMethodSpec(name, scoring)`, `MsacSpec(name, loss_width, lo)`, `evaluate_sample_budget` | the sample-budget engine: running-best selectors and LO checkpoints |
| `live_runs(scoring, lo, ctxs; problem_of, seed_of)`, `live_row` | end-to-end `ransac` on pool pairs, with the problem the caller builds per pair |
| `ComposedSpec`, `SoloSpec`, `Roster`, `deployed_*_settings`, `audit_operating_points` | a paper's arms and the gate-enforced readers of their tuned operating points |

Widths are VisualGeometryCore width values. A native refit is seeded under the
scoring template rebuilt at its own outlier half-width,
`with_width(scoring, OutlierHalfwidth(a))`, and an absolute width reaches the
strategy as `with_width(lo, fixed_width(lo, s2))`; an external refiner receives
`LossWidth(w)` (or `FixedS2(s2)` for a supplied scale).

```julia
using VisualGeometryEval.Evaluation: build_pair_ctx, refit_imc!, problem_type

P   = problem_type(:E)                                 # EssentialProblem
ctx = build_pair_ctx(P, pair; scoring = template)      # template: any scoring with an outlier half-width
imc, lo_ns = refit_imc!(ctx, lo, seed, nat)            # nat: lo's native width (consumed_axis(lo))
```

## Reproducible figures

Every figure, table and number macro in a camera-ready paper is generated by
tracked code from declared data, indexed in the paper project's `REPRODUCE.md`,
and enforced by a test. A hand-made asset is allowed only when it is declared
`static` with a provenance note.

The index is a markdown table with (at least) the columns `asset`, `kind` and
`generator`; `label`, `config`, `data` and `provenance` are read when present. Cells hold
their values in backticks; paths are relative to the project directory and an
asset may be a glob (`*` within a path segment, `**` across segments).

```markdown
| label | asset | kind | generator | config | data | provenance |
|---|---|---|---|---|---|---|
| `fig:teaser` | `papers/images/intro_overlay.png` | generated | `scripts/make_intro.jl` | `configs/figures/intro.toml` | images tier | |
| — | `papers/figures/stats.tex` | generated | `scripts/make_stats.jl` | `configs/figures/stats.toml` | promoted run | |
| `fig:arch` | `papers/images/architecture.pdf` | static | — | — | — | drawn in Inkscape by the authors from the network definition, 2026-05 |
```

A generator "writes" its asset when it names the file or declares it in a
comment line `reproduce: writes <path-or-glob>`, and "reads" its config of
record when it names the file or declares `reproduce: reads <path>`
(project-relative; any comment syntax). Every parameter of a generator — frame
picks, crops, seeds, panel lists, sizes, inputs — belongs in its config of
record, not in the script. The project's suite holds the contract:

```julia
using VisualGeometryEval.Reproduce: reproduce_index_violations

vs = reproduce_index_violations(project_dir; papers = ["papers/main.tex", "papers/supp.tex"],
                                require_config = true)
foreach(println, vs)
@test isempty(vs)
```

Violations: an asset the paper reads that no entry covers, a missing asset, a
reference whose path is not static, a stale entry, an unknown kind, a generated
entry without an (existing) generator, a generator that does not write its
asset, a generated entry without a config of record (with `require_config`), a
missing config or one its generator does not read, a static entry without
provenance, an entry filed under another float's
label, and (best effort) a number defined by a generated macro file re-typed as
a literal in the prose — exempt a coincidence with `% reproduce: literal-ok` on
its line. `src/reproduce/reproduce_index.jl` depends on Base only, so an
environment that cannot load this package includes it by path.

## Paper project conventions

Every paper of the stack is produced by a **paper project**: a
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/) project (not a
package) whose every figure, table and number comes from a tracked driver run
on a config of record, from declared data, with a provenance record beside
each result. The worked example is ACCV 2026's
[`accv2026`](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl/tree/main/accv2026)
in LocalOptimizationDoneRight, with its
[REPRODUCE.md](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl/blob/main/accv2026/REPRODUCE.md).
One check enforces the rules below and a template starts a project that
already satisfies them.

### Layout

| Path | Holds |
|---|---|
| `Project.toml` | a DrWatson project depending on DrWatson and VisualGeometryEval (`compat = "1"`); `Manifest.toml` is tracked: the environment of record |
| `configs/` | the configs of record (TOML), one per driver or cell |
| `scripts/` | the drivers, one file per paper asset (or cell), plus `scripts/tools/` for what they launch |
| `src/` | the shared prelude the drivers `include(srcdir(…))`: project code, never a data constant |
| `test/runtests.jl` | the suite; it calls `paper_project_violations` |
| `paper/` | the manuscript; generated assets under `paper/figures/` |
| `data/exp_raw`, `data/exp_pro` | the per-machine bindings of the input and results trees (links or directories, git-ignored); `data/sims` for smoke runs |
| `plots/` | local renderings that are not paper assets |
| `REPRODUCE.md` | the asset index (below) |
| `Artifacts.toml`, `configs/data_artifacts.toml` | the published input data and their catalog |

Paths come from DrWatson (`projectdir`, `srcdir`, `datadir`, `plotsdir`) and
the Experiments data layer, never from a literal machine path.

### Rules

- **Configs of record.** Every value a result depends on is in a TOML file
  under `configs/` that the driver reads; a driver holds no data constant — no
  literal seed, scene or frame list, data path or numeric table. A justified
  constant (a published test vector, say) is marked
  `# paper-project: allow C2 <reason>`.
- **Run keys from configs.** A result directory is named by `run_key(cfg)`
  (Experiments: the SHA-256 of the effective config, `[render]` excluded), so a
  changed config is a new directory, never an overwrite; never a wall-clock
  stamp or a hand-bumped `-v5`. (`savename` is DrWatson's flat-config
  equivalent; run configs are nested, hence the hash.)
- **Provenance per result.** Every result directory carries `_config.toml`
  and `_provenance.toml` written by `write_run_records` / `write_provenance`
  (DrWatson's `tag!` record: commit, and the patch of a dirty tree; the
  declared inputs relative to their roots). `cached_result` is the keyed cache
  (DrWatson's `produce_or_load` file names, tagged).
- **Stage runner.** A driver declares its inputs (`require_input`) and its
  stages (`Stage(name, desc, done, run)`) and ends with `run_stages`:
  `--plan` prints what is done, `--stage=NAME` runs one, `--force` re-runs.
- **Data policy.** Inputs under `data_root()` are immutable (writers refuse
  them, `assert_not_input`); results under `results_root()` are add-only. No
  machine path and no `artifact"…"` in code (`data_policy_violations`, D1–D4).
- **Published data as Julia artifacts.** Input data are published on Zenodo as
  lazy, hash-pinned artifacts (`Artifacts.toml`), catalogued in
  `configs/data_artifacts.toml` (DataArtifacts) and read through the artifact
  view `data_root()` falls back to — so a fresh machine runs the drivers
  unchanged.
- **REPRODUCE.md.** Every figure, table and macro file the paper reads is a
  row of the asset index — `label | asset | kind | generator | config | data |
  provenance` — `generated` by a driver that declares it
  (`# reproduce: writes <path>`; `# reproduce: reads <config>`) or `static`
  with a provenance note (Reproducible figures, above).

### The enforcing test

```julia
using DrWatson: projectdir
using VisualGeometryEval.PaperProjects: paper_project_violations

@testset "paper-project conventions" begin
    v = paper_project_violations(projectdir(); papers = ["paper/main.tex"])
    foreach(x -> @info(sprint(show, x)), v)
    @test isempty(v)
end
```

`paper_project_violations(dir; papers, index = "REPRODUCE.md", checks,
scripts = "scripts", configs = "configs", catalog =
"configs/data_artifacts.toml", data_policy = (;), reproduce = (;), allow)`
returns `PaperProjectViolation(check, rule, file, line, message)`s:

| check | rules |
|---|---|
| `:layout` | `L1` a required path is missing · `L2` Project.toml lacks DrWatson or VisualGeometryEval · `L3` git does not ignore `data/exp_raw`, `data/exp_pro` · `L4` the suite does not call this check |
| `:data_policy` | `D1` machine path · `D2` write into the input tree · `D3` unkeyed output · `D4` `artifact"…"` in code (`data_policy` forwards keywords) |
| `:reproduce` | every kind of `reproduce_index_violations` (`reproduce` forwards keywords) |
| `:configs` | `C1` a producing script reads no config of record · `C2` it holds a data constant · `C3` it names a config that does not exist |
| `:provenance` | `P1` no provenance through the shared writer (the script, what it includes, what it launches) · `P2` no run key · `S1` an index generator without `run_stages` |
| `:catalog` | `A1` `catalog_violations`, once the catalog or Artifacts.toml binds an artifact |
| `:contract` | opt-in: `contract_violations` of every stage file |

A *producing* script is a `.jl` under `scripts/` that runs stages, declares an
output, is an index generator or calls a result writer. The configs and
provenance scans are static and conservative — a necessary condition, not
proof. Opt out of one rule with `# paper-project: allow <RULE> <reason>` (on
the line or the line above for C2/C3, anywhere in the file for C1, P1, P2, S1;
no reason is `X1`). A project's deferred violations go in `allow`
(`DataPolicyAllow(rule, file, match)`), each with the fix in a comment beside
it; an entry that matches nothing is `STALE`, so the list only shrinks.

### The template

```julia
using VisualGeometryEval.PaperProjects: new_paper_project
new_paper_project("neurips2026"; papers = ["paper/main.tex"])   # git = false inside a package repository
```

`new_paper_project` runs DrWatson's `initialize_project` with
`PAPER_PROJECT_TEMPLATE` and writes the files above: `Project.toml`,
`README.md`, `CLAUDE.md` (a pointer to these conventions), `.gitignore`,
`REPRODUCE.md` with the index table, `configs/example.toml`,
`scripts/example_stage.jl` (stage runner, config of record, run key,
provenance; it writes a figure, its caption and a macro file), `src/<name>.jl`,
`test/runtests.jl`, the papers, `Artifacts.toml`, the catalog stub and
`data/README.md`. Then:

```
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/example_stage.jl      # the example's assets
julia --project=. test/runtests.jl              # zero violations
```

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryEval")
```

Requires Julia 1.12. Part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack; built on
[VisualGeometryCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryCore.jl)
[VisualGeometryIo.jl](https://github.com/VisualGeometryToolkit/VisualGeometryIo.jl)
and [RobustVisualGeometry.jl](https://github.com/VisualGeometryToolkit/RobustVisualGeometry.jl).

## Defaults of record

Every default lives in one TOML file and is read through one function:
`configs/default.toml` (statistics, rendering, raw-read thresholds),
`configs/tuning.toml` (every cross-validation parameter) and
`configs/splits.toml` (every scene split). Unknown keys are refused.

## Tests

`Pkg.test()` runs every `test/*_tests.jl` item in parallel (ReTestItems);
`Pkg.test(; test_args = ["pool_io|tuning"])` runs the matching items.

## License

MIT
