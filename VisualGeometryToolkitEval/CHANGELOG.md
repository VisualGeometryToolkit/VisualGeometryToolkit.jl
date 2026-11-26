# VisualGeometryToolkitEval changelog

## 0.12.0

**Breaking: on VisualGeometryToolkitCore 0.14 and VisualGeometryToolkitEstimation 0.13**, whose estimators are
`solve(problem, algorithm)` (`[compat]` VisualGeometryToolkitCore `"0.14"`, VisualGeometryToolkitIO `"0.12"`,
VisualGeometryToolkitEstimation `"0.13"`). The scale-prior fit is an algorithm of that interface. Numerics are
unchanged: the golden records of the prior fit (`ci/golden/`, group `eval_noise`) replay bit for bit.

### Conformance (no public estimator outside `solve`; one name per algorithm value)
- **`VisualGeometryToolkitEval.Pools` is `VisualGeometryToolkitEval.PoolFiles`** (the pool FILE IO: `ScoringPool`,
  `save_pool`, `load_pool`, `PoolResiduals`, `migrate_pool`, …): one `Pools` in the stack, VisualGeometryToolkitCore's
  (the pool record types). The same functions under the new module name.
- **`refit_model` is deleted**: a synonym of `solve` that dropped the result. Write
  `solve(RefineProblem(problem, seed), LocalOptimization(lo; scoring, score_guard)).u`, reading `found_model(r)`.
- **`JeffreysPriorSpec`, `IGRobustSpec` and `ALL_PRIOR_SPECS` are deleted**: second names of algorithm values, and a
  `const` holding one. Write `PriorSpec(Jeffreys(), Unfitted())` and `PriorSpec(InverseGamma(), RobustFit(α₀))`.
- **The sample-budget engine's score selection is VisualGeometryToolkitEstimation's `ScoreSelection`**:
  `ScoreSelector(name, scoring, width, pool; lo, refinement_width)` holds its streaming cache, sized by `pool`
  (`pair => R`, the largest pair), and reads the running best at each checkpoint (`reinit!(cache; p = (; budget))`),
  each model scored once (was a hand-rolled argmax on a workspace and scratch the caller supplied). The same
  selections.
- **The refit problem is a type, not a `Bool`**: `refit_imc!(ctx, lo, seed, nat′; refit_on = ReprojectionBundle())`,
  `LoRefitSpec(…; refit_on)` and `build_pair_ctx(…; refit_on)` (was `bundle = true`); the default
  `PoolFrameProblem()` refits on the pair's pool-frame problem. A bundle refit on a context built without the bundle
  problem is an `ArgumentError` (was an `ErrorException`).
- **Hyperparameter selection is an estimation (owner ruling G-j): the tuning schemes are the algorithms.**
  `TuningScheme <: AbstractEstimationAlgorithm`, on VisualGeometryToolkitEstimation's problem tag `Hyperparameter`:
  - the coarse-fine search: `solve(EstimationProblem(Hyperparameter, errorsfn, (; cells)), scheme::CoarseFine |
    JointCoarseFine)` (`FitKind`), `u` the selection per search key and `metadata.records` each key's record
    (what `save_coarse_fine_artifact` persists); `(MaxIters, IterationCap)` for a clipped window. `coarse_search`,
    `fine_search`, `coarse_fine_search`, `coarse_fine_cv`, `coarse_fine_cv_batched`, `probe_curve` and `fine_stage`
    are deleted (the two stages are internal);
  - the bootstrap harness's per-draw selection: `solve(EstimationProblem(Hyperparameter, ValidationErrors(label,
    scene_errors, scene_sigma2, native_grid), (; pairs, to_native)), scheme::WindowedBootstrap)` (`ClosedFormKind`);
    `bootstrap_trials(errors, scheme, protocol, n; to_native)` and `compute_method_rows(errors, scheme; protocol,
    to_native)` take the data and the scheme. `WindowedGridMethod`, `BootstrapMethod`, `tune` and `evaluate` are
    deleted;
  - the bootstrap of a selection: `solve(EstimationProblem(Hyperparameter, stats), SelectionBootstrap(criterion;
    n_resamples); rng)` (was `bootstrap_argmin`).
  The same numbers: each solve runs the old function's body.
- **An illegal `PriorSpec` is refused where it is built**: `Jeffreys` with a fitting, or `InverseGamma` unfitted, is an
  `ArgumentError` from the constructor (was an `ErrorException` at the solve), with the same message.

### Changed (breaking)
- **`fit_prior` is deleted: a `PriorSpec` is the algorithm.** `fit_prior(spec, σ²s)` →
  `solve(EstimationProblem(IGPrior, σ²s), spec)`, a `ClosedFormKind` whose result's `u` is the `IGPrior`;
  `fit_prior(family, fitting, σ²s)` → `solve(EstimationProblem(IGPrior, σ²s), PriorSpec(family, fitting))`. A result
  reads `r.u` (was the bare prior); too few data (none for `RobustFit`, fewer than two for `MLE`) give `u = nothing`
  with `(InitialFailure, NoRule)` (was a bare `nothing`). `n` is the number of σ̂² values the fit read (0 for
  Jeffreys, which reads none). The illegal combinations are the same `ErrorException`s, now at the solve. An `IGPrior`
  problem has no default algorithm.
- **`refit_model(problem, seed, lo, scoring; s2)` loses its `workspace` keyword**: it is
  `solve(RefineProblem(problem, seed), LocalOptimization(lo; scoring, score_guard))`, VisualGeometryToolkitEstimation's
  seeded local optimization (was `locally_optimize`). The refit layer (`refit_imc!`) keeps one `LocalOptimization`
  cache per pair, problem variant, scoring template, algorithm type and seed type, `init` once and
  `reinit!(cache, seed; alg)` per call with the call's scorer width (VisualGeometryToolkitCore 0.14's `reinit!` with a
  new algorithm): the workspace reuse of 0.11, bit-identical refits. `PairCtx.lo_ws` is a `Dict{Any, Any}` (its keys
  are tuples for the native caches, `Symbol`s for the external refiners' workspaces).
- `live_runs` runs `solve(EstimationProblem(f), Ransac(; scoring, local_optimization_strategy, config,
  on_improvement, score_guard); rng)` (was `ransac`); `trials` reads `r.stats.nsteps`. Same runs.

### Internal
- The duplication gate's findings: one file reader cached by mtime (`_cached_read`, code_identity.jl) serves the data
  catalog's entries and Experiments' site profiles and stage graphs (the catalog restated it); Reproduce's tex
  reference lookup is `_tex_reference_path` (its `_resolve` shared a name with VisualGeometryToolkitEstimation's).

## 0.11.0

**Breaking: a top-level `[render]` in a `Product` record is an error** (it keyed nothing: a product that read it was
reused after an edit of it; below), and **the `dgx01` site profile is retired** (a project that launched there names a
profile file of its own, `--site=path/to/mine.toml`). On VisualGeometryToolkitCore 0.13.1 (`[compat]` `"0.13.1"`: the
site reader returns the site's `refuse_depot` too).

### Removed
- **`configs/sites/dgx01.toml`**: the shipped profiles are `desktop` and `vcai`. The runner's tests run their Slurm
  stand-in on a profile of their own; the docs' examples and the runner bootstrap's usage line name `vcai`.

### Added
- **Another project's run of record as an input.** A config's `[inputs.<id>]` may name another project's keyed
  product, `project = "<name>"`, `product = "<experiment>/<name>"`, `key = "<run key>"`, optional `file`, `sha256`
  and `tier`, instead of a tier's `path`. `input_path` reads it under the first root that holds
  `<product>/<key>[/<file>]`: `VGTK_RESULTS_ROOT_<PROJECT>`, the active site profile's results root for that project
  (**`Experiments.project_results_root(project)`**), then the named tier's root; the product's record must name the
  run key, the `sha256` is verified, a scratch output refused. One check for the drivers and the runner's launch
  (**`product_input`**, **`product_input_path`**, **`is_product_input`**; `missing_inputs` takes `results_root` and
  `site_profile`); `A2` accepts it. Nothing is copied.
- **Results-mount tiers as inputs**: a tier the catalog lists with `mount = "results"` (a run of record published as a
  tier, linked into the results root by `fetch_artifacts!`) is read under `<results root>/<mount_dir>` by
  `data_root(; tier)`, `input_path` and the runner (`tier_data_root`'s results rung, **`catalog_results_mount`**).
- **A produced entry may name another project's producer**, `producer = "<project>:<stage>"`: `A1` checks the stage in
  that project's graph, the project found among `stack_artifacts`.
- **`Reproduce.asset_stages(product, cfg, assets; paper)`**: a driver's `:verify` and `:deploy` check stages over the
  renders of a keyed product (`verify_assets` and `deploy_assets` against `paper_figures_dir(paper)`, the deploy under
  the config's `[deploy] replace_submitted`, **`deploy_signoff(cfg)`**), each refusing a product not produced under
  its current key: the pair the paper projects copied. **`Experiments.produced_dir(product)`**: a product's
  directory once it is produced under its key, an error naming it otherwise. `paper_figures_dir` is now defined in
  Reproduce (its deploy stages read it); PaperProjects exports the same function, unchanged.
- **The `desktop` profile's `[storage]`**: every project's data under `~/data/vgtk/data/{project}`, its results under
  `~/data/vgtk/results/{project}` and the artifact store at `~/data/vgtk/artifacts`, one tree beside the workstation's
  existing `data/` directories (none moved). A project's own `[sites.desktop]` roots are still read first, and a data
  or results root is used once it exists.
- **A site profile's `[storage]` paths may start at the home**: `~/…`, `$HOME/…` or `${HOME}/…`, spelled out by
  `load_site` (VisualGeometryToolkitCore's own reader of a profile file takes `~/…`). Only there: a project's configs
  still name no home (D1).
- **`[storage] refuse_depot`** in a site profile (`Site.refuse_depot`: `true`, `false`, or `nothing` when unstated):
  whether a process there may fall back to the Julia depot as its artifact store. The shipped `vcai` sets it (and the
  template); VisualGeometryToolkitCore's `artifact_store` reads it through this package's site reader (`NODE_CACHE`,
  which it keyed on alone, decides only where no profile states it).
- Site profiles are read once per change of the file (cached by path and mtime, as the stage graph is), not on every
  root lookup.

### Changed
- **One ladder of a project's roots.** A launch's `resolve_roots` and a driver's `data_root()` / `results_root()`
  follow one precedence, owned by the runner: the command line's root (a launch), the variable, the project's own
  `data/exp_raw` / `data/exp_pro` (a driver), the stage graph's `[sites.<site>]` root, the site profile's root. A
  driver now reads the graph's `[sites.<site>] data_root` / `results_root` too (once it exists), as the launch did; a
  launch's roots are unchanged.
- **A job re-finds a site loaded by path**: the runner gives its jobs `VGTK_SITE` as the site's name where that name
  finds the same profile (a shipped one, or one in the project's sites directory), else as the profile's file (it gave
  the name, which found no profile, or another one, there). A driver reads the graph's `[sites.<site>]` tables (its
  tiers) by the profile's name either way.
- **Breaking: `Product` refuses a record holding a top-level `render`** (a section `RUN_KEY_EXCLUDED` drops from a run key):
  it keyed nothing, so a product that reads it was reused after an edit of it. A product that reads it keys it under
  its own name, the table moved (`r = copy(cfg); r["render_files"] = pop!(r, "render")`); one that does not leaves it
  out of its record (`delete!(copy(cfg), "render")`, the same key as before). Every other product's key is unchanged.
  **A driver must change** where it gives `Product` a whole config that holds `[render]`, or a record that holds the
  table both at the top and under another name.
- The data policy's scanned folders and the paper-project template's folders derive from one declared layout
  (`Experiments._PROJECT_LAYOUT`): the same folders as before (D1 reads `test`, `examples`, `configs`, `docs`,
  `provenance`, and the Markdown files of the root and `data/`; the code rules `src` and `scripts`), and
  `PAPER_PROJECT_TEMPLATE` the same folders (in another order). D5 also refuses VisualGeometryToolkitCore's
  `install_built_artifact` in a project's code.

## 0.10.1

### Added
- **The digest cache**: `file_sha256` keeps the hash of a file of 64 MiB or more across processes, so the many jobs of
  a launch hash a large input of record once rather than once each (an input's `sha256` check, `inspect_input`: the
  runner's `missing_inputs`, a driver's `verify_inputs` and `require_input`; a directory's `tree_sha256` through each
  of its files). An entry is a small file per real path holding the file's size, mtime, ctime and inode and its hash:
  an unchanged file costs a `stat` (a directory a `stat` walk: under 10 ms for the accv2026 pool trees, 22–109 files
  of up to 3.3 TB), a changed one is rehashed and its entry rewritten (to a temporary file, then renamed: concurrent
  processes are safe), so in a tree only the changed files are read again; a file changed in the last 3 s is not kept
  (a same-size rewrite within a coarse file system's time tick keeps its stat). A pin the hash does not match is
  refused as before. **`digest_cache_dir()`**: `VGTK_DIGEST_CACHE`, else `$XDG_CACHE_HOME/vgtk/input-digests`, else
  `~/.cache/vgtk/input-digests` (an absolute path), per user (created 0700; an entry another user owns is ignored)
  and never in the data tree; a site sets it in its profile's `[env]`, which
  the launch's input check uses too, so the launch warms the cache its jobs read. On by default;
  **`VGTK_DIGEST_CACHE=off`** hashes every input in full (a release gate's or a run of record's full verification).

### Changed
- `file_sha256`'s process cache is kept by the file's size, mtime, ctime and inode (it was size and mtime): a
  same-size rewrite that sets the mtime back is hashed anew.

## 0.10.0

**Breaking: the paper-project check enforces the storage design** (a project's suite that was clean may now report
the new rules until the project adopts it), on VisualGeometryToolkitCore 0.13.0 (`[compat]` `"0.13"`) and its artifact
store.

### Added
- **Site profiles carry the machine's storage** (`configs/sites/<site>.toml`, `[storage]`): `data_root` and
  `results_root` (`{project}` is the project's name) and `artifact_store`, absolute paths, the ONLY place a machine
  path is written. `Site` gains the fields `data_root`, `results_root`, `artifact_store`; **`site_root(site, key,
  project)`** reads one. A relative path, an unknown key or a `{project}` in the shared store is refused.
- **The `vcai` site** (`configs/sites/vcai.toml`): the vcai-eaic-mds Slurm cluster, every job sourcing
  `/cluster/projects/vgtk/env-node.sh`, the node athena excluded; data, results and the artifact store in the NFS
  project space (`/cluster/projects/vgtk/{data,results}/{project}`, `/cluster/projects/vgtk/artifacts`).
- **`active_site()`, `site_profile(spec)`, `project_name()`** (Experiments): the profile `VGTK_SITE` names (a shipped
  name, a path, or the project's own `sites_dir`; `nothing` for a name no profile has) and the project's name.
- **The artifact store's site rung**: this package registers its profile reader with Core's
  `set_site_artifact_store!` when it loads, so `VGTK_SITE` selects the store.
- **Paper-project rules**: `L5` `scripts/fetch_data.jl` is missing or does not call `fetch_data!`; `L6` git tracks
  data under `data/` (anything but `data/README.md`); `D1` also scans tests, examples, configs, docs and tracked
  provenance (every line of their code, config and text files, a site profile excepted) and every Markdown file of the
  project root and of `data/` (README.md, REPRODUCE.md, data/README.md, …), and the other text files of `src/` and
  `scripts/` (`.py` among them); it reads, in a git work tree, only the files git does not ignore; the machine roots
  gain `/cluster`, `/srv`, `/gpfs`, `/lustre`, and D1 also finds a home-relative path (`~/…`, `$HOME/…`, `"$HOME"/…`,
  `${HOME}/…`, `$(HOME)/…`, `homedir()`, `ENV["HOME"]`) and a host name (`dgx01`) other than a site selection's value
  (`--site=`, `--site `, `VGTK_SITE=`, `site = "…"`); `D5` an artifact or download API outside DataArtifacts;
  the check **`:storage`** (default): `A2` a config's `[inputs.<id>]` names no catalogued tier, `A3` an artifact name
  bound to another tree in another project's Artifacts.toml (`stack_artifacts`) or in the store's name record, `A4`
  a link of the artifact view outside the artifact store; the opt-in check **`:release`**: `R1` an unpublished
  record, `R2` a Hugging Face record not `public = true`, `R3` a placeholder or local download URL.
- **Produced catalog entries** (DataArtifacts, `kind = "produced"`): a derived input of record that is never
  published, with its `producer` (a stage of `configs/reproduce.toml`), the config inputs it `covers` (by their
  `[inputs.<id>] path`) of one `tier`, `what` and `provenance`. It has no record, mount or binding and is
  never packed, bound or fetched (`fetch_data!` names its producer); configs and their run keys are unchanged. `A2`
  accepts an input a produced entry covers; `A1` requires its producer to be a stage with a config of record whose inputs are published
  tiers or covered by another produced entry, with no cycle.
- **The template's `scripts/fetch_data.jl`** (DataArtifacts' `fetch_data!`), and its `data/README.md` and
  `README.md` naming the site profiles and the store.

### Changed
- **`data_root()`**: `VGTK_DATA_ROOT` → `data/exp_raw` → **the site profile's `data_root`, once it exists** → the
  project's fallback → the artifact view. **`results_root()`**: `VGTK_RESULTS_ROOT` → `data/exp_pro` → **the site profile's
  `results_root`** → `data/exp_pro`; created with `create_results` (as before, whichever rung named it), else a
  missing `data/exp_pro` is an error.
- **The runner**: a launch's roots fall back to the site profile's `[storage]` after the graph's `[sites.<site>]`
  (kept until the projects carry no machine paths), and its jobs get `VGTK_ARTIFACT_STORE` from the profile's
  `artifact_store` unless the caller sets it.
- **DataArtifacts**: `fetch_artifacts!` installs into the artifact store (Core's `install_artifact`); the packer
  stages through Core's `with_artifact_store`; `fetch_data!` prints the store.

## 0.9.3

A patch on VisualGeometryToolkitCore 0.12.0 and VisualGeometryToolkitIO 0.11.0 (`[compat]` `"0.12"`, `"0.11"`): the radiometric types
and kinds moved to the Visual Radiometry Toolkit. No code of this package changes (a test follows the stack's dependency
table, where PanoramaStitching now states its own row).

## 0.9.2

A patch: on VisualGeometryToolkitCore 0.11 and VisualGeometryToolkitEstimation 0.12 (`[compat]` `"0.11"`, `"0.12"`).
Eval reads no LM verdict (no `converged`, `retcode` or `Stalled` in its code); no behaviour or API changes.

## 0.9.1

On VisualGeometryToolkitCore 0.10.7, VisualGeometryToolkitIO 0.10.3 and VisualGeometryToolkitEstimation 0.11.5 (unchanged).

### Fixed

- **An integer's spelling no longer breaks or moves a key.** TOML parses a hex, octal or binary literal
  (`ransac_seed = 0x0b10b5eed`) as an unsigned integer, which VisualGeometryToolkitCore's `canonical_json` refuses, so
  `run_key` and the runner's stage key threw on such a config (`tuning_hash` reads typed spec fields and never did).
  The content key now lowers every integer to its `Int64` before encoding, so `0x0b10b5eed` and `2970312429` key
  alike (keying only: a config keeps the integers it parsed, as `3` and `3.0` key alike). Every key Core encoded
  exactly is unchanged. An integer past ±2^53 (not exactly a double) is an `ArgumentError` naming its
  key path (`arm.seeds[2]`), never an unnamed refusal; that includes `typemin(Int64)`, which Core's own bound let
  through.
- **A produced directory output is reused.** `produce` left reuse to DrWatson's `produce_or_load`, whose test is
  `isfile(output)`, so a product whose `output` is a directory (documented since 0.9.0) ran `f` again on every call
  into its standing output, though `produced` held. `produced(product)` is now the one reuse rule for a file and a
  directory output alike: a produced product runs nothing, and `produce_or_load` only produces.

## 0.9.0

On VisualGeometryToolkitCore 0.10.7, VisualGeometryToolkitIO 0.10.3 and VisualGeometryToolkitEstimation 0.11.5 (the
`[compat]` floors of the package and its test project).

### Breaking

- **One cache: the keyed product on DrWatson's `produce_or_load` replaces Eval's own caching.** `Product(experiment, name,
  record; output, files, upstream, parts, code)` and `produce(f, product)` (produce_or_load with `loadfile = false`; `f` may
  return a `Produced`: its record or `nothing`, the inputs it read, the save's keywords). A record `f` returns is saved
  through `DrWatson._wsave` ATOMICALLY (staged under its own name in a fresh temporary directory in the product's directory,
  then moved onto the output; a save that throws leaves no output and the product not produced) by the output's saver: a `.jld2` output holds
  the record as JLD2's one object `"record"` (any Julia value, a String- or Symbol-keyed `Dict` read back as it was), any
  other goes through `VisualGeometryToolkitIO.save`; `load_output(T, product)` reads it back the same way. An `output` may
  be a directory (a product that writes one: produced while it is there, removed whole when produced again). A product's
  directory is DrWatson's `datadir("exp_pro", experiment, name, key)` (`product_dir`); `_config.toml` and
  `_provenance.toml` (with `output_sha256`, the content hash of every file of the product's directory but its records, and
  `upstream_outputs`, its upstream products' as it read them) are written last; `produced(product)` is the reuse
  rule; a failed save is an error (produce_or_load only warns).
- **Removed, each refused by name with its replacement** (no alias; an `ArgumentError` naming it): `keyed_cell_dir` →
  `product_dir(Product(…))`; `keyed_data_dir` → a `Product` then `promote_run(product_dir(p), tier)`; `experiment_dir` → a
  `Product` whose experiment is a path; `fresher` and `run_complete` → `produced(product)`; `cached_result(f, prefix, cfg, dir; force, gitpath)
  -> (data, file)` → `p = Product(experiment, prefix, cfg; output = "$prefix.jld2")`, `produce(_ -> f(cfg), p; driver,
  config_source, force, gitpath)`, `data = load_output(Dict{String, Any}, p)`, `file = product_output(p)` (`dir` is the
  product's directory, `product_dir(p)`; its file holds the Dict as JLD2's object `"record"`, not as DrWatson's
  `wsave` datasets, and a forced run replaces it, keeping no `_#n` copy; `key`, `ext`, `load` and `save` have no
  counterpart: the key is `run_key`'s, the format the output's extension); `legacy_hash_key` → `run_key`; the 4-argument `Stage(name, desc, done, run)` →
  `Stage(name, desc; items, product, run)` (`run(item)` for every item: a produced product loads) or `Stage(name, desc; check)`.
  `--plan` prints each stage's produced products; `--force` unmarks each product just before its run.
- **The roots are DrWatson's layout**: `results_root()` is `datadir("exp_pro")` and `data_root()` is `datadir("exp_raw")`
  (`VGTK_RESULTS_ROOT` / `VGTK_DATA_ROOT` only relocate them).
- **Keys now include the code trees** (`run_key(…; code)`; a `Product` folds `current_code_trees`: every stack package's source
  tree — a path checkout's git trees, a registered version's Manifest tree hash — and the project's `src`), beside the record,
  the upstream products' keys and their outputs' content hashes as produced (a Merkle chain over the outputs, so an
  upstream produced again with other outputs — a `--force`, a nondeterministic step, an input it did not pin — re-keys
  every consumer, and with the same outputs does not; `produced` checks them again, so a consumer built before its upstream
  was produced again is not reused; an upstream given as a bare key folds the key alone) and the pinned inputs: a change of
  any of them is a new product. **Results of
  earlier versions are not found** by the new keys (nor in the retired `<kind>-<key>` layout); they stay where they are.

### Known issues

- A paper's external-arm environment that includes `experiments/stages.jl` by path (it cannot load this package) has no
  `Product` methods: its producing stages show `?` under `--plan` and fail under `--force`, so only `check` stages work
  there (no skip-if-done), and the 4-argument `Stage` is refused there too.
- The template's results branch records only the simulate product's key in `RUN.toml` (`run_keys`), not the fit's or the
  figure's, and a bare key does not locate a product (its directory needs the experiment and the name).
- Producing a product again removes its provenance and its `output` only: other files it wrote under the same key in an
  earlier production stay beside the new ones.
- Every `Product` construction recomputes `current_code_trees` (a stat walk of every path-loaded stack package), several
  times per item and run; supply `code` when building many.
- An experiment path with `\` is refused, so `joinpath` builds a refused path on Windows (write the segments with `/`).

## 0.8.0

The breaking wave 10, on VisualGeometryToolkitCore 0.10, VisualGeometryToolkitIO 0.10 and VisualGeometryToolkitEstimation 0.11.

### Breaking

- **`Stats.POOL_RNG` is retired, with no alias**: it was a second name for StableRNGs' `StableRNG`, which every pool
  stream uses directly. A pool's `pool_rng` stamp is unchanged. Core's `check_naming` refuses the old name.
- **The terminology ruling (2026-11-13).** `Datasets.consolidate_csponds` is `consolidate_correspondences`, and the
  readers' `min_csponds` keyword is `min_correspondences` (`load_heb_pairs`, `load_pt_pairs`, the pool configs' TOML key).
- **Pool files are format 6**: a pair's correspondences are stored under `correspondences`. `load_pool` refuses a
  format-5 file, naming the remedy: `Pools.migrate_pool` carries a format-4 or -5 file over (verified, the original
  untouched), and `scripts/migrate_pools.jl DIR --apply` a whole tree in place. The `pool_key` is unchanged, so a
  pool's residuals and sample-starts files stay bound to it.
- **Consolidated matches files are format 3** (`CONSOLIDATED_FORMAT`): the pairs' correspondences under
  `correspondences`. `load_roma_pairs` still reads formats 1 and 2.
- **The Seven-Scenes image size** (`configs/default.toml`) is a `{width, height}` table, the IO wave's one `Size2` layout.
- **The runner's stage keys change; earlier runs are not reused** (each stage runs once more). A stage's key no longer
  holds its CPU count (`--cpus-per-task`, `SLURM_CPUS_PER_TASK`): the results are the same bits at any thread count,
  so a stage run at 8 CPUs is reused by a launch at 2 (its device, `gres` and `CUDA_VISIBLE_DEVICES`, stays keyed).
  (The fold of the upstream stages' keys is 0.7.3's.)
- **Every content key is the stack's canonical JSON; runs re-key once.** `run_key`, `tuning_hash` and the runner's
  stage key are the SHA-256 (16 hex) of VisualGeometryToolkitCore's `canonical_json` of the tree, no longer of its
  sorted TOML (the second encoding, `_toml_hash`, is deleted). Key order still does not move a key, and now neither does
  a number's spelling: an integral float is the integer (`3` and `3.0` key alike). Every keyed run directory, cache
  file and stage reruns under its new key once. Tuning artifacts are format 8: a v7 (or v6) file reads, its stored
  sorted-TOML hash verified with that format's recipe and re-hashed; nothing new is written with the old recipe.
  A config (or tuning table) holding an integer beyond ±2^53, `Inf` or `NaN` has no canonical JSON and is now refused
  with an `ArgumentError` where it was keyed before.
  The runner bootstrap, in a bare `julia`, includes Core's Base-only `src/base/canonical.jl` by path before
  `src/cli.jl`: **a paper project regenerates its `scripts/reproduce/reproduce.sh`**
  (`Reproduce.write_reproduce_script(<repository root>; julia_project)`); an older one stops at once with an error that
  says so.
- **The dgx01 site profile sizes tasks small**: `cpus_per_task = 2` for a stage that declares none (was 28 for every
  stage), and `mem_per_cpu = "8480M"`, the node rule (was `"8G"`). A threaded stage declares its own `cpus_per_task`
  in its `[[stage]]`; the count keys no run.
- **`Render.station_cells` labels each attempt by the member of `stations` nearest its angle** (an angle beyond either
  end at that end), not by rank:
  a station without a capture is a hole, and a jittered angle stays in its station. A panel with a missing station or
  jittered angles changes.

### Added

- **A stage after one item of an array stage**: `after = ["<stage>:<item>"]` waits on that array task alone
  (`--dependency=afterok:<job>_<task>` on Slurm; a local site's order is unchanged). The key folds the upstream stage's
  key, as a whole-stage edge does. An item the launch does not run, or an item of a stage that is no array, is refused.
- **`Render.write_tsv(path, rows::AbstractVector{<:NamedTuple}; comments)`**: a table of named-tuple rows, its header
  their field names; an abstract element type (`NamedTuple[]`) is read from the rows, and an empty table writes the
  element type's fields as its header when it names them (else an empty header line; `read_tsv` reads either back as
  no rows). `verify_assets` writes its report through it.

- **`Tuning.WindowExtension(factor, max_rounds)`** and **`extend(::EfficiencyConstantWindow, sides; factor)`**: the
  expand-on-boundary rule of a search window, `GridExtension`'s twin (TOML `[types.window_extension]`, defaults
  factor 2, 8 rounds): the one owner of CaliBlobs icra2026's `WindowWidening`.
- **Runner arrays over any list or table of a stage's config, and a loop inside each task.** `array.from` is a dotted
  key of the config (a table's keys sorted, `"cases"` as before; a list of strings in its order: `from = "grid.sets"`),
  and `array.each = { from, env }` runs each task's command once per value at `from` of its item's config, with `env`
  set to the value (a grid of sets × variants, one task per set: RC experiments' `chrome_index_grid`). `stage_each` lists a
  task's values; the dry run prints them; the values key the stage. An unknown selected value names the config's; a key
  of no value (or `items = []`) is refused. A TERM, or a signal the launcher's `--signal` asks for, stops a task's loop:
  no later value starts.
- **Phases: a stage's CPU and GPU parts as separate jobs** (`[[stage]] phases = [{name, gpu, env, sbatch | command,
  cpus_per_task, mem_per_cpu, gres, time, per_item}, …]`): run one after another, each with its own environment
  (`VGTK_PHASE`), device and resources, task k after task k between arrays (`aftercorr`), so no GPU is held for CPU
  work (RC experiments' regenerate_paper: GPU drivers, then CPU numbers; chrome_index_grid: per set a CPU decode, then a GPU
  detect). The last phase is the stage's job; an earlier one writes its markers under `<stage markers>/<phase>/`, and
  `status` names a failed one (`failed (its <phase> phase, job <id>, failed …)`, with its log and error line). A
  phase's command and environment key the stage. **A GPU stage on a Slurm site compiles first**: the runner's
  `precompile` phase, a CPU-only job running `Pkg.precompile()` with the stage's `julia_flags` (new key: the flags its
  jobs run julia with); `precompile = false` (a Bool) leaves it out. The precompile phase keys nothing: it is
  environment state (the depot's compile cache), not a result. The manifest records `phases`, `phase_jobs` and the
  site's `precompile_time` (a measurement, in no key).
- **A site's `[slurm] precompile_time`**: the `--time` of a GPU stage's CPU-only precompile job, its measured time × 2
  (dgx01: `"0:30:00"`); a real limit lets the scheduler backfill the job.
- **A stage's `nice`** (`[[stage]] nice = <n>`): its Slurm `--nice`, so a research stage yields to a release gate's
  jobs. A label for the scheduler like the CPU count and the job-name prefix: it keys no run.

- **A run record states its sensor model's provenance**: `write_provenance(…; sensor_model)` (a `SensorModel`, or one
  per frame) records VisualGeometryToolkitCore's `component_sources` under `sensor_model` — the black's, noise's and
  defects' sources (`"measured"`, `"estimated"`, `"specified"`, `"file"`, or `"absent"`), the model's reason and its
  `specified_override`, one table per distinct model with its frame count — so a table can say "measured" or
  "estimated" per run (the sensor-model spec, §5).

### Fixed

- **A run of record of a REGISTERED package's code is no longer refused** (0.7.4's fix). A registered install (the
  package's source in the depot: no git checkout) is committed code, content-addressed and immutable; `code_trees`
  now identifies it by the tree hash the active environment records for it (`Pkg.dependencies()`: name, version, uuid,
  git-tree-sha1), and searches the active project's environment for the stack packages. Still refused: a git checkout
  with uncommitted changes, and a path that is neither a git checkout nor a registered install.
- **`evaluate_sample_budget`** took each pair's selectors by `Threads.threadid()` under a dynamic `@threads` loop, so
  two tasks could share one (a migrating task): each pair now takes its selectors from a pool. The results are the
  same bits at any thread count (a test at 2 and 8 threads).
- **A result of uncommitted code is never reused** (`code_tree_changes`, the one reuse rule of the runner,
  `run_complete` and `cached_result`): its tree's `+uncommitted` mark names HEAD, not the edits, so a scratch run under
  other edits carried the same value and was reused. Each such tree is a change, `"<name>: uncommitted when recorded"`;
  a scratch job of uncommitted code still in flight is not depended on.
- **The runner reads only its own project's earlier launches, in launch order**: a results root shared by several
  projects no longer lends one project's stage records to another, and `run_manifests` orders by the launch stamp
  that names each manifest, not by modification time (the results-branch stage rewrites its manifest, which made an
  older launch the "last completed" upstream run). Each record's state is read once from its markers.
- **The runner honours `VGTK_SCRATCH`** beyond the key: a scratch launch reads another scratch run's output, and runs
  uncommitted code with a warning (a run of record still refuses both); the dry run says so.
- **A local launch keeps `results_branch`** in its manifest (the stage writes it atomically, and the runner keeps it when
  it rewrites the manifest).
- **`write_numbers(…; shared = true)` holds a pid lock** (`<path>.lock`) over its read-merge-rename: concurrent drivers no
  longer drop each other's sections. `number_macros.jl` loads by path in a bare Julia with its own atomic write.
- **`promote_run`** no longer takes two runs without a `run_key` for the same run unless their records are identical.
- **`code_trees` runs git once per process** while nothing git reads has changed (a stat fingerprint of the code paths,
  the path-loaded packages, the Manifest and each repository's HEAD, refs, index and ignore files).
- Docstrings: a directory input's `sha256` is checked through `tree_sha256`.

## 0.7.4 (unreleased)

A patch: a bug fix in the code identity of a run, no API change.

- **Fixed: a run of record of a REGISTERED package's code is no longer refused.** A registered install (the package's
  source in the depot, `packages/<Name>/<slug>`: no git checkout) is committed code — content-addressed and immutable —
  but `code_trees` read it as "no git checkout", so `refuse_uncommitted_code` refused every run of record whose
  `gitpath` is a registered package (PanoramaStitching's example records its stages with
  `cached_result(…; gitpath = pkgdir(PanoramaStitching))`; only `VGTK_SCRATCH=1` got past it). `code_trees` now
  identifies such a directory by the tree hash the active environment records for it (`Pkg.dependencies()`: name,
  version, uuid and git-tree-sha1), and searches the active project's environment for the stack packages. Still
  refused: a git checkout with uncommitted changes, and a path that is neither a git checkout nor a registered install.

## 0.7.3 (unreleased)

A patch: a bug fix in the reproduce runner, no API change.

- **Fixed: a stage is no longer reused after a stage it runs after reruns.** A stage's run key hashed its own files,
  items, environment and command, never its upstream stages', and the reuse clearing skipped completed stages. So after
  a change to stage A's config, A reran but a stage B after it (its own files unchanged) was "reused (completed
  earlier)", its results computed from A's old output. Now a stage's key folds in the key of each stage it runs after:
  that stage's key in this launch, or, for one the launch does not run (`--no-deps`), the key of its last completed
  run, whose output the stage reads. So the fold is transitive, and launches of one stage at a time are covered in any
  order. A stage whose upstream this launch runs again is never reused, completed or in flight.
- **Existing run manifests** (`<results root>/_reproduce/*.toml`) are read as before; nothing is migrated. A stage
  with no `after` keeps its 0.7.2 key and is still reused. Every stage that runs after another gets a new key, so the
  first launch under 0.7.3 reruns it once (its results are rewritten, the same when its inputs were current). In a
  project whose environment loads VisualGeometryToolkitEval (the usual case), the upgrade also changes Eval's recorded
  code tree, and that first launch reruns every stage, "code changed" naming it, as after any Eval release.

## 0.7.2 (unreleased)

A patch: no API change.

- **Julia 1.13** (the owner's ruling of 2026-11-13: the whole stack requires 1.13): `[compat] julia = "1.13"` in the
  package and in every environment of its workspace; CI runs 1.13 only.
- **Every `[compat]` floor raised to the gated resolution** (the owner's ruling of 2026-11-13): a stack dependency at
  its registry latest or its in-wave version, a third-party one at the version the release gate resolved (a floor
  only, the semver upper bound kept); the monorepo's Downgrade workflow is deleted.
- **Internal: the private `_opted_file`, a forward to `haskey`, is deleted** (the owner's no-shim order); the paper-project
  rule checks call `haskey(marks, rule)` directly.

## 0.7.1 (2026-11-13)

A patch: on VisualGeometryToolkitEstimation 0.10 (`[compat]` VisualGeometryToolkitEstimation "0.10" in the package and
test environments). No code change: Eval never called the SplineLens refine that Estimation 0.10.0 retired.

## 0.7.0 (2026-11-12)

- **`install_artifact` is VisualGeometryToolkitCore's**, re-exported by `Experiments` (the same binding);
  `huggingface_token` and the internal `_published` are no longer reachable as `Experiments.*` (read Core's).
  Downloads is no longer a direct dependency.
- **`DataArtifacts.fetch_data!(cat; argv = ARGS, io = stdout)`**: a paper project's `scripts/fetch_data.jl` as one call
  (`--list`, `--all`, `--only=…`, the required artifacts by default; a split entry's parts listed and fetched, a chunked
  file reassembled), hoisted from LocalOptimizationDoneRight's accv2026 script; RANSACScoringDoneRight's neurips2026
  script was its copy.
- **Breaking: one copy each of Eval's partial overlaps** (the owner's ruling on the duplication audit's item 14,
  2026-11-12):
  - a codec-cached result's previous versions are numbered by DrWatson's own `recursively_clear_path`, as `safesave`
    numbers a JLD2 one: the newest previous version is `_#1`, older ones renumbered up (was: the first free number,
    so the oldest was `_#1`). `_backup_name` is deleted.
  - a driver reads the site's tier roots through the runner's `load_stage_graph` (was a TOML re-parse), so it honours the
    graph's `data_env`: a tier's variable is `<data_env>_<TIER>` for `data_root(; tier)`, `keyed_data_dir`,
    `promote_run` and `assert_not_input`, as for the runner (was always `VGTK_DATA_ROOT_<TIER>`). A malformed
    `configs/reproduce.toml` is now refused there too when `VGTK_SITE` is set.
  - a driver's view tier is the runner's (`catalog_view_mount`): a tier the project's `configs/data_artifacts.toml`
    lists mounts at its `mount_dir`, so a split entry's tier resolves where its parts are linked (was the view's root:
    the gate asked whether the view's Artifacts.toml binds the entry's own name, which a split entry never is). A view
    bound without a catalog, or a bound artifact the catalog does not list, is only looked up. The view is installed
    into only for a name its Artifacts.toml binds.
  - one HEAD-and-dirty reader, the runner's (`git rev-parse HEAD`, `git status --porcelain --untracked-files=no`):
    `results_branch` reads it; `Tuning._git_identity` (LibGit2 and DrWatson's `isdirty`) is deleted.
  - one record of the code a result came from, `code_trees`: `code_identity(code)` (a tuning sidecar's
    `code_identity`) keeps each module's `version` and the `julia` table and no longer records a `git` commit; the
    code's state is the run's `code_trees` (its run records and `_provenance.toml`).
  - `require_input` checks through `inspect_input`, the runner's and `verify_inputs`' one check: a directory's
    `sha256` is its `tree_sha256` (was a refusal from `file_sha256`), a `sha256` that is no SHA-256 is an
    `ArgumentError`, and a scratch run's output is refused unless the run is a scratch run.

- **The live and refit runs read a model when the search found one (`found_model`)**, not when it `converged`:
  VisualGeometryToolkitCore's `converged` is strict (a run that drew its trial budget without reaching the confidence
  did not converge but found a model), so every error and refit is computed as before.

- **Breaking: the LO budget's table is `[local_optimization_strategy]`** (`iteration_budget`; was `[local_optimization]`,
  the retired keyword spelling). `lo_iteration_budget()` reads it; the value is unchanged.
- **Breaking: the strategy names follow VisualGeometryToolkitCore/Estimation**: arms, specs and runners take
  `AbstractOptimizationStrategy` or `nothing` (no local optimization; was `NoLocalOptimization()`), and call
  `ransac`/`locally_optimize` with `local_optimization_strategy`.
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
  `boundary_hit`, `record.param_name`). A format-6 efficiency constant is the value its arm's `ctor` took then;
  it is read through the arm's own hook, `ComposedSpec(…; format6_constant)`, which the roster declares (the identity
  by default; `m -> decision_width(GemanMcclureShape(), m)` for a ctor that took a Geman–McClure half-weight depth
  directly), so the deployed LO is unchanged. Formats 1–5 are refused as before.
- **New: `Evaluation.tuned_width(art, arm)`**, the one reader of an arm's tuned width (through `operating_width`), in
  the current currency: a format-6 efficiency constant read through the arm's `format6_constant` hook, once.
  `deployed_budget_settings` and `deployed_basin_settings` read through it; a caller that needs the width itself
  (a renderer, a resumed run carrying it into a format-7 artifact) calls it instead of `art["tuned_width"]`.
  The fixtures (test/data/tuning_v6/) were written by the format-6 code.
- **New public names for helpers projects restated:** `Reproduce.provenance_token(s)` (a provenance field's
  spelling: spaces and `;` made `_`), `Reproduce.project_relative_path(path, root = project_root())` (a path as a
  provenance comment cites it) and `Render.pgf_colormap(colormap; reverse, samples = 15)` (a colormap as pgfplots'
  colormap body, the one `write_station_fan_colorbar` defines). They were the private `_provenance_token`,
  `_project_relative` and `_pgf_colormap`.

- **One tier resolver, read and write** (`tier_data_root`): the runner's `missing_inputs`, `data_root(; tier)`,
  `input_path`, `verify_inputs`, `tier_path`, `keyed_data_dir` and `promote_run` all resolve a tier through it, in one
  order: `VGTK_DATA_ROOT_<TIER>`, the site's `[sites.<site>.tiers]`, the artifact view (when the data root IS one and
  publishes the tier: its mount, the catalog's `mount_dir`; the runner reads it through `catalog_view_mount`), the
  data root. The view's mount applies only under the view, never to a `VGTK_DATA_ROOT`, an author's own
  `data/exp_raw` or a fallback, so flat layouts and producers' locations do not move.
- **A producer never installs an artifact and never writes into an artifact view**: `keyed_data_dir` and
  `promote_run` resolve a write root; a tier the view publishes is refused ("tier X is published read-only by the
  artifact view; bind VGTK_DATA_ROOT_X or a site tier to produce it"), and so is any tier whose root is the view or
  inside it — the view's own directory, a `VGTK_DATA_ROOT` or a tier root that is one ("the data root is an artifact
  view (read-only); bind VGTK_DATA_ROOT or VGTK_DATA_ROOT_<TIER> to a writable root"). `assert_not_input` guards the
  site table's tier roots as it guards `VGTK_DATA_ROOT_<TIER>`.
- **A scratch run never completes a run of record**: `run_complete` compares the record's `scratch` flag with this
  process's, as `cached_result` does; with the keys (`run_key`) and `promote_run`, a `VGTK_SCRATCH=1` result is never
  reused or promoted as a run of record, nor the reverse.
- `tree_sha256`'s `**/x` matches a top-level `x`; a path with a tab or a newline is refused. `_view_has` caches the
  parsed Artifacts.toml.

- **New: a directory input's `sha256`.** `VisualGeometryToolkitEval.tree_sha256(dir; exclude)` is the one directory
  digest: the SHA-256 of the sorted lines `<relative path>\t<file sha256>\n` of every regular file under it (links
  to files followed), the paths matching an `exclude` glob left out; each file hashed by `file_sha256` (streamed,
  cached by path, size, mtime). An `[inputs.<id>]` that is a directory takes it as `sha256`, with
  `sha256_exclude = [globs]`; `inspect_input` (the runner, `verify_inputs`, `input_path`) verifies it.
- **Changed: a producer writes through its tier's root.** `keyed_data_dir(cfg, tier, kind)` and `promote_run` are
  `<data_root(; tier)>/<tier>/<…>`, the tier's root resolved as a consumer's, so a producer and a consumer
  (`[inputs.<id>] tier, path = "<tier>/<kind>-<key>"`) agree wherever `VGTK_DATA_ROOT_<TIER>` or the site's tier
  table puts it. **New: `Experiments.tier_path(tier, parts...)`**, the one reader of a path in a tier that a config
  does not declare (a capture set: `tier_path(tier, set)`); a declared input is `input_path`'s.
- **New: a catalog entry's `mount_dir`** (DataArtifacts): the directory under its mount root a tier mounts at, its
  paths relative to it (`mount_dir(cat, name)`); `pack_artifacts`/`artifact_files` and `fetch_artifacts!` honour it,
  and `catalog_view(cat; dir, default)` is the project's `ArtifactView` mounting each tier there. A view tier's root
  (`data_root(; tier)`, so `input_path`) is where the view mounts it. A project's own tier-mount helper becomes this.
- **Fixed: paper-project rule C1 recognises `cli_option(ARGS, "config")`** (and `cli_option(args, "config", default)`)
  as reading a config of record.

- **Scratch isolation, tightened.** `write_provenance` refuses a directory that holds the record of the other kind
  (a scratch run's into a run of record's, or the reverse); `scratch_record` finds the run a path belongs to at any
  depth (`run/sub/x.csv`, the nearest record above it); the runner's stage key folds `VGTK_SCRATCH=1`. Data-policy
  D3 flags both unkeyed forms of the shared oracle-floor file, `oracle_floors_path()` and
  `oracle_floors_path(results_root())`, and never `oracle_floors_path(<keyed dir>)`. The runner's file hash is
  `file_sha256` itself.
- **Fixed: `data_root(; tier)` (and so `input_path`) installs its tier from an artifact view**, through the same
  resolver as `data_root(; tiers)`: on a machine with the artifact view a non-default tier was never fetched.

- **A scratch run is kept apart from the runs of record.** Its record says `scratch = true` (`write_provenance`,
  `cached_result`; a run of record says `false`); `run_key` folds `scratch`, so its keyed directories never are a run
  of record's, and `cached_result` names its file `<prefix>-scratch_<key>`. Its output (`scratch_record`) is refused as
  an input of record (`missing_inputs`, `verify_inputs`, `input_path`, except to another scratch run), as an upstream
  run of record (`run_key(…; upstream = <run dir>)`, which now also takes a run directory and folds its recorded
  `run_key`), as an input of a run of record's record (`write_provenance(…; inputs)`) and by `promote_run`.
- **One input check.** `VisualGeometryToolkitEval.inspect_input` (src/code_identity.jl) is the check of an
  `[inputs.<id>]` under its tier's root (there, its `sha256`, not a scratch output) that the runner's
  `missing_inputs` and a driver's `verify_inputs`/`input_path` both call; `data_root(; tier)` resolves through
  `tier_data_root`; the runner's file hashes are `file_sha256`.
- **New: the reserved input tier `"unpublished"`** (`Reproduce.UNPUBLISHED_TIER`): data of record kept on a machine,
  never fetched. The dry run lists its inputs apart, present or ABSENT, and a missing one is reported as never fetched.
- **New: `cached_result(…; ext, load, save)`**, any format cached through a codec (a vgio record), the previous version
  kept as safesave keeps it; **`write_run_records(…; fields)`**, a further payload of the record; **`ntasks` and
  `nodes` in a `[[stage]]`** for a distributed stage. A stage's `gres` and `mem_per_cpu` are validated.

- **Fixed: data-policy rule D3 flags only the bare `oracle_floors_path()`** (the unkeyed
  `<results root>/oracle_floors.jld2` every run rewrites), not `oracle_floors_path(keyed_dir)`, the documented keyed use.

- **New: `Experiments.input_path(cfg, id; root)`**, the one reader of a config's input: its `[inputs.<id>]` path
  under its tier's root (`data_root(; tier)`), checked to be there and, with a `sha256`, to have that content. A
  project's own `input_path`/`input_table` (`data_root` + `require_input`) becomes this.
- **New: per-stage resources in the stage graph.** A `[[stage]]` takes `cpus_per_task`, `mem_per_cpu` and `gres`
  over the site's (`gres` makes it a GPU stage); on a local site `cpus_per_task` is its `SLURM_CPUS_PER_TASK`.
- **New: the runner bootstrap is Eval's.** `templates/reproduce/reproduce.sh` (`Reproduce.REPRODUCE_SCRIPT_TEMPLATE`)
  is the one copy of a project's `scripts/reproduce/reproduce.sh`; `Reproduce.reproduce_script(; julia_project,
  graph, path)` fills it and `write_reproduce_script(root; …)` writes it (a project's suite compares its copy with
  `reproduce_script`). RadiometricCamera's experiments and RSDR's copies are this text.
- `keyed_cell_dir(cfg, "figures", name; base = results_root())` is the directory of a run over several datasets
  (figures, numbers): `<results root>/figures/<name>-<key>`.

- **New: per-tier input roots.** An input tier may live outside the data root on a machine (a capture on another
  disk). Its root resolves in one order, for the runner and the drivers alike (`VisualGeometryToolkitEval.tier_data_root`):
  the environment's `VGTK_DATA_ROOT_<TIER>` (`tier_variable`: the tier upper-cased, other characters `_`), then the
  stage graph's `[sites.<site>.tiers]` table (`<tier> = "<root>"`, a relative root relative to the workdir), then the
  one data root. Each `[inputs.<id>]` path is relative to its tier's root, so a config names no machine path (data
  policy D1). `missing_inputs(…; site)` and the runner resolve it so (a named tier root that is not there is
  reported with the variable or table that names it; the `sha256` check applies per input), the runner exports
  the site's tier roots and `VGTK_SITE` to its jobs, and `Experiments.data_root(; tier)` resolves the same order
  (the graph's table through `VGTK_SITE`); `verify_inputs` checks each input under its tier's root, and
  `assert_not_input` guards every tier root. No link directories and no sibling-checkout paths.

- **New: a results layout named by the config.** `[experiment] layout = ["mocap", "<camera>"]` (path segments) makes
  `experiment_dir` `<results root>/<layout…>` for a project whose results are not laid out by dataset, matcher and
  problem; `load_run_config` then needs no pool fields. `keyed_cell_dir(cfg, experiment, kind; base)` takes another
  base, refused unless it lies under the results root. CaliBlobs' icra2026 (`mocap/<camera>/…`,
  `absolute_pose_experiments/<set>/{selfcal,tune_radial_camera}`) keys its runs through it.

- **New: an input's content of record.** An `[inputs.<id>]` table may carry `sha256` (64 hex digits, of a file).
  When it does, `missing_inputs` (the runner, before anything runs; the dry run too), the new
  `Experiments.verify_inputs(cfg; root)` and `Experiments.require_input(path; …, sha256)` verify it, and another
  content is REFUSED (`ArgumentError`: the input id, its config and path, the expected and the actual hash; the
  runner exits 2). Without it the input is only checked to be there, as before. The hash is
  `VisualGeometryToolkitEval.file_sha256(path)`: streamed in blocks, and cached in the process by real path, size and
  modification time, so a large input is read once per process.
- **New: `run_key(cfg; files, upstream, parts)`** folds into the key what a run depends on beyond its config: the
  content hash of `files` (`name => path`), the run keys of the `upstream` runs it reads, and `parts` (a choice made
  outside the config, e.g. a smoke run's frame limit), as the table `keyed_with` (refused in a config). Without them
  the key is the config's, unchanged. `keyed_cell_dir` takes the same keywords. A project's private keys (file hashes,
  upstream keys, version tags, date stamps) become this.
- **New: `Experiments.keyed_data_dir(cfg, tier, kind; …)`**, the keyed directory of a run that produces data
  (training annotations, a derived pool) under the data root, `<data root>/<tier>/<kind>-<run_key>`, written with
  `producer = true`; and **`Experiments.promote_run(run_dir, tier; name)`**, a completed run of record (with its
  `_provenance.toml`) copied into the data root as an input of record of `tier`, where an `[inputs.<id>]` names it
  and a data artifact (an `ArtifactView` tier) publishes it; another run under the same name is refused.

- **New: a result is reused only while the code it was computed with is unchanged, by its git trees.** The run key
  stays the configuration's (DrWatson's savename / produce_or_load, the runner's stage key: the stage key no longer
  hashes the code). Beside DrWatson's `tag!` record (`gitcommit`, `gitpatch` when dirty), every result now keeps
  `code_trees` (`VisualGeometryToolkitEval.code_trees`, src/code_identity.jl): the git tree of each of the project's
  code paths (`git rev-parse HEAD:<path>`: the graph's `code` for the runner, the project's `src` for a driver's
  records) and of every stack package the project's environment loads (its `[deps]` and what they reach in its
  Manifest): a package loaded from a path by the trees of its `src`, `ext` and `Project.toml` in its own repository,
  a released one by its Manifest `git-tree-sha1`. A docs-, tests- or paper-only commit changes no tree.
  `VisualGeometryToolkitEval.code_tree_changes(recorded, current)` is the one reuse rule, and every consumer calls it:
  the runner's stage reuse (a stage whose code changed reruns, `rerun: code changed: <tree>: <old> → <new>`; one still
  in flight under other code stops the launch), `Experiments.run_complete` (no longer warns on another commit: the
  trees decide) and `Experiments.cached_result`. `Experiments.current_code_trees` is a driver's current record.
  RadiometricCamera's experiments, CaliBlobs' icra2026, BlobBoards' bmvc2026, RANSACScoringDoneRight and
  LocalOptimizationDoneRight get it through `write_provenance`/`write_run_records`, `run_complete` and the runner,
  with no change of their own; docs: "Code identity: when a stored result is reused" (Reproducible figures).
- **Behaviour change: a result recorded before the code trees existed runs once more** (a run directory's
  `_provenance.toml`, a `cached_result` file, a runner launch's stage record), and from then on carries them.
- **Changed: a run of record runs committed code.** Code with uncommitted changes (an edit, a staged, deleted or
  untracked file in the project's code paths or in a package its environment loads from a path), or code in no git
  checkout, is refused with one error (`VisualGeometryToolkitEval.refuse_uncommitted_code`): by the runner (exit 2;
  the dry run marks it `DIRTY`) and by a driver's records (`write_provenance`, `write_run_records`, `cached_result`,
  before computing). A scratch run (`VGTK_SCRATCH=1`, `Experiments.scratch_run`) records it with a warning instead,
  its trees marked `+uncommitted`, so its result is never reused for committed code. A dirty docs or paper tree is
  no code and is not refused. A launch that identifies none of the project's own code (no `src`, no `[project] code`)
  is refused; a driver's record without a `src` is warned.

- **The runner's reuse reads completion markers and the queue, never accounting** (`Reproduce`, src/reproduce/runner.jl).
  Every task of a stage writes a completion marker, `task-<t>.toml` under `<manifest without .toml>/<stage>/`
  (`stage_markers_dir`, `stage_markers`): its state (`completed`, `failed`), exit status, start and end, job id,
  commit, run key, host, `restart_count` (Slurm's `SLURM_RESTART_COUNT`) and the signal it received, written
  atomically. On a Slurm site the job writes it itself, through a wrapper the runner submits around the stage's
  launcher (the launcher's `#SBATCH` directives, then a copy of the launcher as submitted); the local backend writes
  it after every task. `stage_run_state` reads a recorded stage from its markers and the queue; the queue comes from
  `squeue` (`slurm_queue_states`), which needs no accounting storage, instead of `sacct`, which dgx01 does not keep
  (`slurm_job_states` remains, as information in `reproduce_status` only).
- **Changed: `plan_reproduction`'s `job_states` keyword is now `queue_states`** (`ids -> Dict(id => Dict(task =>
  state))`, what `slurm_queue_states` returns); a caller that supplies `job_states` gets an unsupported-keyword error.
  `reproduce_status` and `reproduce_main` take both (`job_states` there is the sacct extra).
- **New: a launch's job-name prefix** (`job_name_prefix` in a site's `[slurm]` table, `plan_reproduction(…;
  job_name_prefix)`, `--job-name-prefix`): every job is named `<prefix><name>`, the name the stage's `job_name`, else
  its launcher's own `#SBATCH --job-name`, else `<project>-<stage>`. The run keys are taken without it.
- **Behaviour change: a launch recorded before the completion markers existed reruns its completed Slurm stages
  once.** Its manifest has no `markers` for a stage, so a stage whose job has left the queue reads as `unknown`, never
  as completed; the next launch submits it again, and from then on its markers are read.
- **Fixed: a requeued task is in flight, not failed.** Slurm requeues a task under the same job id (preemption with
  `PreemptMode=REQUEUE`, `scontrol requeue`) after sending it TERM, which the wrapper records as a `failed` marker.
  `stage_run_state` read that marker over the queue, so a stage pending or running again read as failed, and the next
  launch submitted it a second time into the same results. A task the queue holds is now in flight whatever its
  marker says (a `COMPLETING` task reads its marker).
- **Fixed: a failing squeue is no empty queue.** `slurm_queue_states` read any non-zero exit as "the job left the
  queue", so a slurmctld timeout or a controller restart made every in-flight stage `unknown` and the launch submitted
  duplicates of all of them. Only squeue's "Invalid job id specified" is read as absent; any other failure is an
  error that stops the launch (and `status`) before anything is submitted (exit 4, squeue's message).
- **Fixed: `submit_slurm!` runs sbatch without the launching process's `SLURM_*` variables** (`SLURM_CONF` kept), as
  `run_local!` already did. sbatch's default `--export=ALL` gave a stage's job the parent job's variables: launched
  from inside an array task, a non-array stage inherited `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_JOB_ID`, so its
  marker was written for the wrong task under the wrong job id and the stage read as unknown on every launch.
- **Fixed: a job without a name is named `<project>-<stage>`**, not `marker_wrapper.sbatch`: a launcher stage with no
  `job_name` and no `#SBATCH --job-name` was submitted without `--job-name`, and Slurm named it after the submitted
  script, now the wrapper. The name is added at submission; the run key is unchanged.
- **Fixed: the wrapper forwards a launcher's `--signal` signals.** The copied `#SBATCH --signal=B:<sig>@<time>` now
  signals the wrapper, which had no trap for it and was killed by it, leaving no marker and the launcher unsignalled.
  The wrapper traps each such signal, forwards it to the launcher and records it in the marker (`signal`); the
  state follows the launcher's exit status.
- **Fixed: the launcher's copy runs from a `noexec` results root.** The wrapper executed the copy directly (mode
  755); it now runs it through the interpreter of its `#!` line (bash without one), and the copy is mode 644.

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
  stage puts its renders. RANSACScoringDoneRight's neurips2026 and radiometric-camera's experiments each defined it as
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
  own writers (radiometric-camera experiments, BlobBoards bmvc2026, LocalOptimizationDoneRight accv2026,
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
