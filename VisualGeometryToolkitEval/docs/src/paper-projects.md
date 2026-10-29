# Paper projects

Every paper of the stack is produced by a *paper project*. A paper project is a
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/) project, not a
package. Each of its figures, tables and numbers comes from a tracked driver
that runs on a config of record and reads declared data, and each result has a
provenance record beside it.

The worked example is the project
[`accv2026`](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl/tree/main/accv2026)
of the ACCV 2026 paper (Asian Conference on Computer Vision), kept in
LocalOptimizationDoneRight, with its
[REPRODUCE.md](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl/blob/main/accv2026/REPRODUCE.md).
One check enforces the rules below, and a template starts a project that
already satisfies them.

## Layout

| Path | Holds |
|---|---|
| `Project.toml` | a DrWatson project depending on DrWatson and VisualGeometryToolkitEval; `Manifest.toml` is tracked, as the environment of record |
| `configs/` | the configs of record (TOML), one per driver or cell |
| `scripts/` | the drivers, one file per paper asset or cell, plus `scripts/tools/` for what they launch |
| `src/` | the shared prelude that the drivers include with `include(srcdir(…))`: project code, never a data constant |
| `test/runtests.jl` | the suite; it calls `paper_project_violations` |
| `paper/` | the manuscript, with generated assets under `paper/figures/` |
| `data/exp_raw`, `data/exp_pro` | the per-machine bindings of the input and results trees (links or directories, ignored by git); `data/sims` for smoke runs |
| `plots/` | local renderings that are not paper assets |
| `REPRODUCE.md` | the asset index (see [Reproducible figures](@ref)) |
| `Artifacts.toml`, `configs/data_artifacts.toml` | the published input data and their catalog |

Paths come from DrWatson (`projectdir`, `srcdir`, `datadir`, `plotsdir`) and
from the data layer of `Experiments`, never from a literal machine path.

## Rules

- **Configs of record.** Every value that a result depends on is in a TOML
  file under `configs/` that the driver reads. A driver holds no data
  constant: no literal seed, scene or frame list, data path or numeric table.
  A justified constant, such as a published test vector, is marked
  `# paper-project: allow C2 <reason>`.
- **Run keys from configs.** A result directory is named by `run_key(cfg)`,
  the SHA-256 hash of the effective config without its `[render]` table. A
  changed config therefore gets a new directory and never overwrites an old
  one. A wall-clock stamp or a hand-bumped version suffix is never used.
  DrWatson's `savename` is the equivalent for flat configs; run configs are
  nested, hence the hash.
- **Provenance per result.** Every result directory carries `_config.toml` and
  `_provenance.toml`, written by `write_run_records` and `write_provenance`.
  The provenance is DrWatson's `tag!` record: the commit, the patch of a dirty
  tree, and the declared inputs relative to their roots. `cached_result` is the
  keyed cache, with DrWatson's `produce_or_load` file names, tagged.
- **Stage runner.** A driver declares its inputs (`require_input`) and its
  stages (`Stage(name, desc, done, run)`), and it ends with `run_stages`.
  `--plan` prints what is done, `--stage=NAME` runs one stage and `--force`
  runs stages again.
- **Data policy.** Inputs under `data_root()` are immutable, and the writers
  refuse them (`assert_not_input`). Results under `results_root()` are
  add-only. Code holds no machine path and no `artifact"…"` string macro
  (`data_policy_violations`, rules D1 to D4).
- **Published data as Julia artifacts.** Input data are published on Zenodo
  as lazy, hash-pinned artifacts (`Artifacts.toml`). They are catalogued in
  `configs/data_artifacts.toml` (see [The data catalog](@ref)) and read
  through the artifact view that `data_root()` falls back to. A fresh machine
  therefore runs the drivers unchanged.
- **REPRODUCE.md.** Every figure, table and macro file that the paper reads is
  a row of the asset index. A row is either `generated` by a driver that
  declares it (`# reproduce: writes <path>` and `# reproduce: reads <config>`)
  or `static` with a provenance note.

## The enforcing test

```julia
using Test
using DrWatson: projectdir
using VisualGeometryToolkitEval.PaperProjects: paper_project_violations

@testset "paper-project conventions" begin
    v = paper_project_violations(projectdir(); papers = ["paper/main.tex"])
    foreach(x -> @info(sprint(show, x)), v)
    @test isempty(v)
end
```

[`paper_project_violations`](@ref) returns a list of
[`PaperProjectViolation`](@ref)s, each with a check, a rule, a file, a line and
a message:

| check | rules |
|---|---|
| `:layout` | `L1` a required path is missing · `L2` Project.toml lacks DrWatson or VisualGeometryToolkitEval · `L3` git does not ignore `data/exp_raw` and `data/exp_pro` · `L4` the suite does not call this check |
| `:data_policy` | `D1` a machine path · `D2` a write into the input tree · `D3` an unkeyed output · `D4` an `artifact"…"` in code (the `data_policy` keyword forwards keywords) |
| `:reproduce` | every kind of `reproduce_index_violations` (the `reproduce` keyword forwards keywords) |
| `:configs` | `C1` a producing script reads no config of record · `C2` it holds a data constant · `C3` it names a config that does not exist |
| `:provenance` | `P1` no provenance through the shared writer (in the script, what it includes, or what it launches) · `P2` no run key · `S1` an index generator without `run_stages` |
| `:catalog` | `A1` the findings of `catalog_violations`, once the catalog or Artifacts.toml binds an artifact |
| `:contract` | opt-in: the `contract_violations` of every stage file |

A *producing* script is a `.jl` file under `scripts/` that runs stages,
declares an output, is an index generator or calls a result writer. The scans
of configs and provenance are static and conservative: a necessary condition,
not a proof.

A file opts out of one rule with `# paper-project: allow <RULE> <reason>`. For
C2 and C3 the comment goes on the line or the line above; for C1, P1, P2 and
S1 it may go anywhere in the file. An opt-out without a reason is rule `X1`.
A project's deferred violations go in the `allow` keyword as
`DataPolicyAllow(rule, file, match)` values, each with its fix in a comment
beside it. An entry that matches nothing is reported as stale, so the list
only shrinks.

## The template

```julia
using VisualGeometryToolkitEval.PaperProjects: new_paper_project
new_paper_project("neurips2026"; papers = ["paper/main.tex"])   # git = false inside a package repository
```

[`new_paper_project`](@ref) runs DrWatson's `initialize_project` with
[`PAPER_PROJECT_TEMPLATE`](@ref) and writes the files of the layout above:
`Project.toml`, `README.md`, `CLAUDE.md` (a pointer to these conventions),
`.gitignore`, `REPRODUCE.md` with the index table, `configs/example.toml`,
`scripts/example_stage.jl`, `src/<name>.jl`, `test/runtests.jl`, the papers,
`Artifacts.toml`, the catalog stub and `data/README.md`. The example stage
shows the stage runner, a config of record, a run key and provenance; it
writes a figure, its caption and a macro file. Then:

```bash
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/example_stage.jl      # the example's assets
julia --project=. test/runtests.jl              # zero violations
```

## The data catalog

`configs/data_artifacts.toml` is read by [`load_catalog`](@ref) in the
`DataArtifacts` module:

```toml
[catalog]                                    # optional
artifacts_toml     = "../Artifacts.toml"     # the bindings, relative to the catalog
artifacts_toml_env = "ACCV2026_ARTIFACTS_TOML"   # a variable naming another copy

[records.<group>]      # one Zenodo record: title, placeholder, id, license
[artifacts.<name>]     # kind, mount, record, required, what, provenance, and the kind's keys
```

The variable named by `artifacts_toml_env` points to another copy of the
bindings, with other URLs and the same hashes; the fresh-machine test uses it
for a `file://` copy. A record has an `id` once it exists on Zenodo and a
`placeholder` until then.

The kinds of artifact and their keys:

| kind | keys | holds |
|---|---|---|
| `pool_sources` | `config`, `splits` | the raw files that the pool producer `config` reads for the scenes of `splits` |
| `pools` | `config`, `splits` | the pool tree that `config` writes |
| `stored` | `config` | the directory that the `[stored]` table of `config` names |
| `tree` | `dir`, optional `filter` | every file under `dir`; with `filter`, a list of globs over paths relative to the mount root, the files an include glob matches (every file when there is none) and no `!` glob matches |

A `filter` glob uses `*` for a run of characters other than `/`, `?` for one of
them, `**/` for any run of whole directories (none included) and a trailing
`/**` for everything below; every other character is literal. For example
`filter = ["**/*.ARW", "!archive/**"]` with `dir = "."` selects every raw file
outside `archive/`. [`pack_artifacts`](@ref) keeps the comment header of
Artifacts.toml (its leading comment lines), which Pkg's writer drops.

An entry too large to upload as one file can set `split_by = "scene"` (kinds
`pool_sources` and `pools`). It is then bound as one artifact per scene,
`<name>_<scene>`, plus `<name>_meta` for the files that no scene owns. The
entry stays one logical artifact: `required`, [`fetch_artifacts!`](@ref) and
[`pack_artifacts`](@ref) take its name and act on every part.

With `split_by`, the optional `max_part_bytes = N` caps one part at `N`
uncompressed bytes. A scene whose files exceed it is bound as sub-parts
`<name>_<scene>_part1` to `_partk`. A file larger than `N` is cut into the
fewest near-equal chunks `<file>.chunk<i>of<m>`, and the chunks are packed
greedily into parts. `fetch_artifacts!` reassembles each chunked file in the
artifact view, byte for byte. The author's file sizes derive the parts, and a
consumer reads them from the bindings.

Every artifact tree holds paths relative to its mount root: `data` is
`data_root()` and `results` is `results_root()`. The author's tree, a packed
artifact and a fetched view therefore share one layout. A project adds kinds
of its own with the `kinds` keyword (`kind => (cfg, entry, root) -> files`).
