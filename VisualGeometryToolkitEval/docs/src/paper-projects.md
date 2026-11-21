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
| `plots/` | the drivers' renders (ignored by git); a deploy stage copies the paper's into the paper |
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
  nested, hence the hash. What a run depends on beyond its config is folded
  in, never hashed privately: `run_key(cfg; files, upstream, parts)` (a file's
  content, an upstream run's key, a frame limit).
- **Every step a keyed product, through `produce_or_load`.** Each step of a
  pipeline (decode, detect, index, a start state, a refinement, an evaluation,
  a figure) is a `Product(experiment, name, record; output, files, upstream,
  parts)`, keyed by ONLY what it reads — its own config section, its inputs,
  the products it reads (`upstream = ("start" => start_product,)`: each one's key
  and its output's content hash, recorded when it was produced, so an upstream
  produced again with another output re-keys its consumers; build the products
  when their stage runs) — so a later step's setting never recomputes an earlier
  one. `produce(f, product)` produces the step's output once, through DrWatson's
  `produce_or_load`, and runs nothing while `produced(product)` holds (a file or a
  directory output alike); a record `f` returns is saved atomically (a
  temporary file renamed onto the output) — a `.jld2` output as JLD2's object
  `"record"` (any Julia value), any other through `VisualGeometryToolkitIO.save` —
  and read with `load_output`; a product that writes its own outputs returns
  `nothing`, its `output` the file (or directory) it writes last. Its directory is DrWatson's
  `datadir("exp_pro", experiment, name, key)` (`results_root()` is
  `datadir("exp_pro")`, `VGTK_RESULTS_ROOT` relocates it); a deeper layout is
  the experiment path (`joinpath(dataset, matcher, problem, experiment)`).
  A step of record another project reads is made an input of record with
  `promote_run(product_dir(product), tier)`.
- **Provenance per result.** Every result directory carries `_config.toml` and
  `_provenance.toml`, written by `write_run_records` and `write_provenance`.
  The provenance is DrWatson's `tag!` record: the commit, the patch of a dirty
  tree, and the declared inputs relative to their roots; a product's
  provenance is written last, so a half-written step never reads as produced.
- **Stage runner.** A driver declares its inputs (`require_input`) and its
  stages (`Stage(name, desc; items, product, run)`, each item's keyed product, or
  `Stage(name, desc; check)`), and it ends with `run_stages`. `--plan` prints
  each stage's produced products, `--stage=NAME` runs one stage and `--force`
  produces its products again.
- **Data policy.** Inputs under `data_root()` are immutable, and the writers
  refuse them (`assert_not_input`). Results under `results_root()` are
  add-only. No code, test, example, config, doc or tracked provenance holds a
  machine path, and no code an `artifact"…"` string macro or an artifact or
  download API of its own (`data_policy_violations`, rules D1 to D5).
- **Machine paths live in the site profiles.** Where a machine keeps every
  project's inputs and results, and its artifact store, is written once, in
  VisualGeometryToolkitEval's site profiles (`configs/sites/<site>.toml`,
  `[storage]`: see [Machines and storage](@ref)), selected by `VGTK_SITE`. A
  project names none.
- **Published data as Julia artifacts, in the artifact store.** Input data are
  published on Zenodo (or Hugging Face) as lazy, hash-pinned artifacts
  (`Artifacts.toml`). They are catalogued in `configs/data_artifacts.toml` (see
  [The data catalog](@ref)); every config input names its catalogued tier.
  `scripts/fetch_data.jl` (DataArtifacts' `fetch_data!`) installs them into
  the artifact store, never the Julia depot of a cluster, and links them into
  the artifact view that `data_root()` falls back to. A fresh machine therefore
  runs the drivers unchanged. Nothing under `data/` is tracked.
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
| `:layout` | `L1` a required path is missing · `L2` Project.toml lacks DrWatson or VisualGeometryToolkitEval · `L3` git does not ignore `data/exp_raw` and `data/exp_pro` · `L4` the suite does not call this check · `L5` `scripts/fetch_data.jl` is missing or does not call `fetch_data!` · `L6` git tracks data under `data/` (anything but `data/README.md`) |
| `:data_policy` | `D1` a machine path (code and launchers; tests, examples, configs, docs and provenance, comments included) · `D2` a write into the input tree · `D3` an unkeyed output · `D4` an `artifact"…"` in code · `D5` an artifact or download API outside DataArtifacts (the `data_policy` keyword forwards keywords) |
| `:reproduce` | every kind of `reproduce_index_violations` (the `reproduce` keyword forwards keywords) |
| `:configs` | `C1` a producing script reads no config of record · `C2` it holds a data constant · `C3` it names a config that does not exist |
| `:provenance` | `P1` no provenance through the shared writer (in the script, what it includes, or what it launches) · `P2` no run key · `S1` an index generator without `run_stages` |
| `:catalog` | `A1` the findings of `catalog_violations`, once the catalog or Artifacts.toml binds an artifact |
| `:storage` | `A2` a config's `[inputs.<id>]` names no tier the catalog lists · `A3` an artifact name bound to another tree in another project's Artifacts.toml (the `stack_artifacts` keyword) or in the artifact store's name record · `A4` a link of the artifact view `data/exp_raw` that resolves outside the artifact store |
| `:release` | opt-in, before a release: `R1` a record not yet published (no Zenodo `id`, no Hugging Face `revision`) · `R2` a Hugging Face record not stated `public = true` · `R3` a binding whose URL is a placeholder or a local file |
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

## Machines and storage

A machine is a **site profile** of VisualGeometryToolkitEval
(`configs/sites/<site>.toml`; [`load_site`](@ref)), selected by `VGTK_SITE`
(the runner sets it for every job). Its `[storage]` table is the only place a
machine path is written:

```toml
[storage]
data_root      = "/cluster/projects/vgtk/data/{project}"      # a project's inputs; {project} its name
results_root   = "/cluster/projects/vgtk/results/{project}"   # a project's results
artifact_store = "/cluster/projects/vgtk/artifacts"           # every artifact tree, shared
```

`{project}` is the project's name ([`project_name`](@ref): its stage graph's
`[project] name`, else its directory's name). The roots resolve, the first
that binds:

| Root | Order |
|---|---|
| `data_root()` | `VGTK_DATA_ROOT` → `data/exp_raw` → the site's `data_root` → the project's fallback → the artifact view |
| `results_root()` | `VGTK_RESULTS_ROOT` → `data/exp_pro` → the site's `results_root` |
| the artifact store | `VGTK_ARTIFACT_STORE` → the site's `artifact_store` → VisualGeometryToolkitCore's preference `artifact_store` → the depot (refused on the cluster) |

The **artifact store** (VisualGeometryToolkitCore's `artifact_store`) holds
the trees of every published tier and every model's weights, content-addressed
by their git-tree-sha1, so every project and package on a machine shares one
copy. VisualGeometryToolkitCore's `install_artifact` installs into it: Pkg's
`with_artifacts_directory`, its sha256 and tree-hash checks and its atomic move
into place, under a Pidfile lock per tree, so jobs that start together download
a tree once. A cluster's Julia depot (node-local and evicted, or a home with a
quota) is refused; the shipped `vcai` site names the cluster's store. A machine
whose depot already holds the trees moves them once with
`VisualGeometryToolkitCore.migrate_artifacts!(toml)` (each tree's hash
checked; `remove = true` frees the depot).

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
renders a figure, its caption and a macro file under `plots/example/`, and its
`deploy` stage copies them into the paper ([`deploy_assets`](@ref): a changed
submitted asset needs `--replace-submitted`). With `--results-branch` it skips
`deploy` and makes a [`results_branch`](@ref) of the renders instead, recording
its synthetic draws as the data tier. Then:

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

[records.<group>]      # one hosted record: title, placeholder, license, host (zenodo | huggingface) and its keys
[artifacts.<name>]     # kind, mount, record, required, what, provenance, and the kind's keys
```

The variable named by `artifacts_toml_env` points to another copy of the
bindings, with other URLs and the same hashes; the fresh-machine test uses it
for a `file://` copy. A record's `host` is where its tarballs are downloaded
from. On Zenodo (the default) it has an `id` once the record exists; on Hugging
Face it names its dataset `repo` and, once the tarballs are uploaded, the
`revision` — a commit, never a branch, so a binding stays immutable while the
dataset is edited. Until then the record carries its `placeholder`, an
UPPER_CASE token that `fetch_artifacts!` refuses to download. A typical use:
Hugging Face while the evaluation data is still changing, Zenodo (with its DOI)
at publication — the bindings' hashes are the same on both, only the URLs move:

```toml
[records.rc]
title = "…"
license = "CC-BY-4.0"
host = "huggingface"
repo = "radiometric-camera/radiometric-camera-data"
placeholder = "HF_REVISION_RC"
```

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

An entry published before the stack's terminology rules keeps its published
file name, which a host such as Zenodo cannot rename, while its binding takes a
current one: `published_name = "<old name>"` names the file the record holds
([`published_name`](@ref)). Its URL then ends in `<old name>.tar.gz` (a split
entry's parts replace `<name>` by it), [`pack_artifacts`](@ref) stages the
tarball under that name, and [`catalog_violations`](@ref) checks the URL's file
against it instead of the binding name:

```toml
[artifacts.eth3d_splg_correspondences]
published_name = "<the file name the record holds>"   # e.g. a name the record published in 2026-10
```

Such a published name is data, not a binding: VisualGeometryToolkitCore's
`check_naming` lists it among its foreign tokens and accepts it only in a
quoted TOML value, a string literal or a data fixture (an `Artifacts.toml`).

Every artifact tree holds paths relative to its mount root: `data` is
`data_root()` and `results` is `results_root()`. The author's tree, a packed
artifact and a fetched view therefore share one layout. A project adds kinds
of its own with the `kinds` keyword (`kind => (cfg, entry, root) -> files`).
