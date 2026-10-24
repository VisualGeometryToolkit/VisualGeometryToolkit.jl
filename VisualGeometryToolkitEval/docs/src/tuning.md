# The `[tuning]` schema

This is the one document of the stack's cross-validation specification.
Every parameter that influences a tuned result is a field of a strategy type
and has its default in `configs/tuning.toml`. The two are joined by one
parser, `Tuning.tuning_spec`. A tuning run persists the spec it ran under, so
a deployed value can be traced back to the exact protocol that produced it.

Module: `VisualGeometryToolkitEval.Tuning` (`src/tuning/*.jl`). The protocol these
specs serve is [The experimental protocol](protocol.md).

## Vocabulary

Each quantity has one name, used in types, fields, TOML keys, grid axes and
artifact keys:

| name | meaning |
|---|---|
| `scale_multiple` | a width as a multiple of the estimated inlier scale ŝ (dimensionless); width = scale_multiple · ŝ |
| `loss_width` | a robust loss's width, supplied directly (no scale estimate): the width of a lifted loss, of a loss minimized by iteratively reweighted least squares (IRLS), or of a threshold score (a threshold is its truncated loss's width) |
| `outlier_halfwidth` | the Student-t marginal's uniform outlier half-width |
| `scale_multiple_window` | the standardized window `[lo, hi]` of scale multiples every search window is cut from |

## Where the parameters live: three levels

A config states `[tuning.<name>]` tables. Each names its scheme with `type`
and states only what differs from the defaults. A key it leaves out is
inherited through three levels:

1. `[tags.<family>.<tag>]`: the default parameters of a tagged strategy (for
   example, `log_count` has `n = 200`). The untagged strategies use
   `[types.<type>]` instead: the scale-multiple window, the grid extension,
   the cross-validation and the budget sampling.
2. `[schemes.<scheme>]`: the scheme's own defaults, where they differ from
   level 1.
3. The config's `[tuning.<name>]` table.

If a tagged table names a different `type` than the table it would inherit,
it starts again from its own tag's defaults: a strategy never inherits the
parameters of a different strategy. An unknown key or tag is an
`ArgumentError` that names the key path, for example
`[tuning.miscal.coarse.spacing] has unknown key(s) ["per_decad"]`.

`layer_tuning(lower, upper)` layers a paper's protocol-level table under a
cell's overrides with the same tag-aware rule; the parser does not layer, so a
paper layers its two tables before parsing. `resolve_tuning!(cfg)` replaces
every entry with its complete table. The Experiments `run_key` and
`write_run_records` both resolve before hashing or writing, so a run key
covers every tuning parameter, including defaults the config only inherited.

The writer `tuning_table(x)` emits the complete table. The round trips
TOML → type → TOML and type → persisted table (`read_tuning`) → type are both
lossless under `==`. A persisted table inherits nothing: a missing key is an
error.

## Tag families

Every tagged table carries `type = "<tag>"`. The registry is `TUNING_TAGS`.

### `spacing`: `GridSpacing`

`grid(lo, hi, spacing)` is the one grid constructor: a Base `range` (linear)
or `logrange` (log) of `npoints(lo, hi, spacing)` points. Both endpoints are
exactly `lo` and `hi`, and every grid has at least 2 points.

| tag | type | parameters | points |
|---|---|---|---|
| `log_per_decade` | `LogPerDecade` | `per_decade` | `round(log10(hi/lo)·per_decade) + 1` |
| `log_per_octave` | `LogPerOctave` | `per_octave` | `round(log2(hi/lo)·per_octave) + 1` |
| `log_count` | `LogCount` | `n` | `n` log-spaced points |
| `linear_count` | `LinearCount` | `n` | `n` linear points |

The `+1` is the fencepost: `d` decades at `p` per decade make `d·p`
intervals, so they have `d·p + 1` points.

### `window`: `WindowRule`

`window(rule; scale_multiple_window, native, s2, center)` returns `(lo, hi)`.
A rule refuses a context it needs and is not given.

| tag | type | parameters | window |
|---|---|---|---|
| `fixed` | `FixedWindow` | `lo`, `hi` (required) | `[lo, hi]` in native units |
| `standardized` | `StandardizedWindow` | none | the run's scale-multiple window |
| `coverage` | `CoverageWindow` | `q_lo`, `q_hi`, `log_margin`, `zeta_octaves` | `[native(lo, ŝ_min/ζ)·e^-m, native(hi, ŝ_max·ζ)·e^m]`, where `(ŝ_min, ŝ_max) = scale_bounds(s2, (q_lo, q_hi))`, `ζ = 2^zeta_octaves` and `m = log_margin` |
| `centered` | `CenteredWindow` | `factor` | `[center/factor, center·factor]` around a centre the axis supplies |

The coverage window depends on the data: on the validation pool's σ̂² and on
the codimension `d_g` inside `native`. The realized window is persisted in the
grid spec and the rule in the spec. `d_g` is not a tuning parameter: it is the
problem's codimension, has no default anywhere, and is persisted as the
artifact key `d_g`. `q_lo == q_hi` is the one-anchor case; at 0.5 it is the
reference scale, so `{ q_lo = 0.5, q_hi = 0.5, zeta_octaves = 0.0 }` is the
window-matched grid.

### `axis`: `TuningAxis`

| tag | type | fields |
|---|---|---|
| `spaced` | `SpacedAxis` | `window` (a rule), `spacing` |
| `listed` | `ListedAxis` | `values` (finite, strictly increasing) |

`grid_spec(axis; context...)` realizes an axis as a persisted `GridSpec`.

### `grid`: `GridSpec` (persisted, never authored)

| tag | type | fields |
|---|---|---|
| `spaced` | `SpacedGrid` | `lo`, `hi`, `spacing`, `n_lo`, `n_hi` |
| `listed` | `ListedGrid` | `values` |

`grid(spec)` rebuilds the realized grid: one range over the base window's
step continued `n_lo` steps below `lo` and `n_hi` above `hi` (a log step is a
ratio). `extend(spec, sides; npts) -> (spec′, n_lo)` is the one extension:
every point of `grid(spec)` reappears, to rounding, at index `+ n_lo`, so
computed values stay valid by index. `extension_offset(base, sup)` compares specs, never realized values:
it returns the offset when `sup` is `base` or an extension of it, and
`nothing` otherwise. `nearest_column(grid, x)` is the evaluation-only lookup
of a corrupted native (never a tuning's).

### `criterion`: the objective (`TuningCriterion`)

| tag | type | parameters | objective (minimized) |
|---|---|---|---|
| `median_error` | `MedianError` | none | median, failures as `Inf` |
| `finite_median` | `FiniteMedian` | none | median of the finite errors (`Inf` when none) |
| `scene_mean_lower_median` | `SceneMeanLowerMedian` | none | mean over scenes of each scene's lower median |
| `scene_mean_median` | `SceneMeanMedian` | none | mean over scenes of each scene's finite median (Stats `scene_mean_median`) |
| `mean_average_accuracy` | `MeanAverageAccuracy` | `max_error` (required) | minus the mean average accuracy (mAA) up to `max_error`, in the errors' own units (Stats `eligible_maa`) |

### `scene_split`: `SceneSplit`

| tag | type | parameters |
|---|---|---|
| `registered` | `RegisteredSceneSplit` | `name` (required); `validation_scenes`, `test_scenes` filled in from `configs/splits.toml` |

### `unit_sampling`: `UnitSampling`

| tag | type | parameters |
|---|---|---|
| `exhaustive` | `ExhaustiveSampling` | none |
| `stratified` | `StratifiedSampling` | `fraction`, `cap_per_scene` (`0` = no cap), `seed`, `rng` |
| `uniform` | `UniformSampling` | `fraction`, `cap` (`0` = no cap), `seed`, `rng` |

The retired tags `all` and `simple_random` (now `exhaustive` and `uniform`)
are refused as unknown.

### Untagged strategies (`[types.<key>]`)

| key | type | fields |
|---|---|---|
| `scale_multiple_window` | `ScaleMultipleWindow` | `lo`, `hi` |
| `grid_extension` | `GridExtension` | `npts` (per clipped side per round), `max_rounds` |
| `cross_validation` | `CrossValidation` | `scene_split` (no default), `validation`, `test` (both `exhaustive` by default) |
| `budget_sampling` | `BudgetSampling` | `sizes`, `n_draws`, `seed`, `rng` |

## Randomness

Each rule below is the definition. Every seed of a sample is derived by Stats
`stable_seed(seed, keys...)`: the first 8 bytes of the SHA-256 hash of the
seed and the keys as text. The algorithm is fixed, unlike `Base.hash`, which
changes across Julia versions, and the seed is keyed by a name rather than an
index:

- **Stratified sampling.** Scene `s` draws from `rng(stable_seed(seed, s))`.
- **Uniform sampling.** The pooled draw is `rng(stable_seed(seed,
  "simple_random"))` (the salt keeps its original spelling, so a sample did
  not move with the type's rename).
- **Budget draws.** Draw `r` at budget `n` is
  `rng(stable_seed(seed, "budget", n, r))` (`budget_rng`, `budget_units`); the
  interval over a budget's draws resamples from
  `rng(stable_seed(seed, "resampling", n))` (`resampling_rng`).
- **Scene-clustered bootstrap rows** (Stats `scene_mean_median_rows!`): point
  `j` resamples from `StableRNG(stable_seed(seed, "scene_mean_median", j))`
  (the mAA's: key `"scene_mean_maa"`).

`rng` names an algorithm in `RNG_ALGORITHMS`: `"StableRNG"` only
(StableRNGs.jl), whose streams are fixed across Julia versions; a permutation
is drawn with `shuffle`, which StableRNGs guarantees (`randperm` it does not).
The bias-corrected and accelerated (BCa) bootstrap itself (Bootstrap.jl) resamples from Julia's default stream
seeded by the row's rng, so a reported interval also depends on the Julia
version, recorded in every artifact's code identity.

## The cross-validation

Every scheme holds a `cross_validation::CrossValidation`: the data a tuning
reads and the data its evaluation reports on.

```julia
CrossValidation(scene_split::SceneSplit, validation::UnitSampling, test::UnitSampling)
```

- **The scene split.** `RegisteredSceneSplit(name, validation_scenes,
  test_scenes)`: a split of `configs/splits.toml`, the one file of record of
  every scene list in the stack. A config names it,
  `scene_split = { type = "registered", name = "eth3d" }`, and never lists
  scenes (a config that does must list exactly the registry's, else an
  `ArgumentError`). The parser fills the lists into the complete table, so
  `tuning_hash`, the run key and every artifact carry the exact lists used:
  editing a list moves every hash that names the split. The constructor
  refuses an empty side, a duplicate and any scene on both sides; the lists
  are kept sorted. There is no default split.
- **The unit samplings.** `ExhaustiveSampling()`; `StratifiedSampling(fraction,
  cap_per_scene, seed, rng)`, each scene sampled separately (the scene mix is
  preserved); `UniformSampling(fraction, cap, seed, rng)`, the units of all
  the side's scenes pooled. A sampler always permutes, keeps
  `max(1, round(fraction·n))` and applies the cap. Only then does it sort back
  into the data's order, so a cap never keeps a prefix of the data. Adding, removing or
  reordering other scenes leaves a stratified scene's sample unchanged.
- **The sampling unit** is a property of the data, never of the spec:
  `sampling_unit(::Type{D})` is `PairUnit()` (an image pair: the stereo
  datasets, the scoring pools) or `QueryUnit()` (a query image of an
  absolute-pose dataset), declared by each data type with no fallback. The
  tuning artifact records it.

`validation_units(cv, units_by_scene)` and `test_units(cv, units_by_scene)`
apply a side: `units_by_scene` maps scene names to their units in the data's
order and must hold every scene of that side; the result is
`scene => kept units`, scenes sorted. `sample_units(sampling, units_by_scene)`
is the sampler itself.

A driver states which samplings it supports in the signature of the method it
calls, never with a runtime check, so an unsupported spec is a `MethodError`:

```julia
two_view_units(cv::CrossValidation{<:SceneSplit,<:Union{ExhaustiveSampling,StratifiedSampling}}, pools) = …
query_units(cv::CrossValidation{<:SceneSplit,<:Union{ExhaustiveSampling,UniformSampling}}, queries) = …
```

The budget draws of a validation-budget run are the scheme's
`budget_sampling` (`BudgetSampling(sizes, n_draws, seed, rng)`): for each
budget `n`, `n_draws` uniform samples of `n` of the eligible validation units,
`budget_units(b, n_units) -> [(n, r, indices)]`.

```toml
[tuning.miscal.cross_validation]
scene_split = { type = "registered", name = "eth3d" }
validation = { type = "stratified", fraction = 0.25 }
test = { type = "exhaustive" }
```

## Schemes

### `joint_coarse_fine`: `JointCoarseFine` (Evaluation `coarse_fine_search`)

The coarse-to-fine search over the product of named axes. It runs in four
steps:

1. the coarse product grid, every cell scored;
2. while the joint argmin lies within `edge_margin` points of a face, that
   axis extends on that side (`extension.npts` points, at most
   `extension.max_rounds` per side) and only the new slab is scored;
3. the product of each axis's `fine` spacing over the coarse argmin's two
   neighbours on that axis; its free argmin is the selection;
4. `bootstrap_argmin` over the evaluated cells' `unit_statistics` gives the
   joint selection's sampling distribution, read per axis.

An axis still within the margin after its budget is a boundary hit, refused
by the one reader `deployed_setting`. An `expensive` axis is asked for one
value at a time: the objective is called once per round per distinct
expensive value, so a caller that keeps its expensive run (a detection run)
per value pays one run per value visited.

| field | meaning |
|---|---|
| `axes` | name → `CoarseFineAxis`: `coarse` (window rule + spacing), `extension` (`npts`, `max_rounds` per side), `fine` (spacing over the bracket), `expensive` |
| `edge_margin` | extend when the argmin lies within this many points of a face |
| `criterion` | the objective (the errors at a cell; the criterion aggregates them) |
| `cross_validation` | the scene split and the validation / test unit samplings |

A new search backend (a surrogate model, say) is a new scheme whose search
returns the same record (`selected`, `boundary_hit`, `evaluated`); the
artifact, the reader and the bootstrap serve it unchanged.

### `coarse_fine`: `CoarseFine` (Evaluation `coarse_fine_cv`)

The one-axis case of `joint_coarse_fine`, its axis spelled flat (the same
search; `coarse_fine_cv` and its batched form return its record in the scalar
keys).

| field | meaning |
|---|---|
| `coarse` | axis: the coarse probe's window rule and spacing |
| `extension` | `npts` per extension, `max_rounds` per side |
| `edge_margin` | extend when the argmin lies within this many points of an edge |
| `fine` | the fine spacing over the coarse argmin's bracket |
| `criterion` | the objective (`evalfn(x)` returns the errors; the criterion aggregates them) |
| `cross_validation` | the scene split and the validation / test unit samplings |

### `windowed_bootstrap`: `WindowedBootstrap` (Evaluation `WindowedGridMethod`)

| field | meaning |
|---|---|
| `scale_multiple_window` | the standardized window every search window is cut from |
| `outlier_halfwidth`, `loss_width` | the native grids (axes) |
| `scale_multiple` | the standardized display axis |
| `criterion` | the objective of each draw's windowed free argmin |
| `cross_validation` | the scene split and the validation / test unit samplings |
| `budget_sampling` | the validation budgets and draws (`compute_method_rows`) |

### `factorial`: `Factorial`

Every combination of named settings of a method's configuration keys: listed
axes, no reference scale, no boundary extension.

| field | meaning |
|---|---|
| `axes` | named axes (configuration key → `TuningAxis`, usually `listed`) |
| `criterion` | the objective |
| `cross_validation` | the scene split and the validation / test unit samplings |

## Examples

A scalar probe:

```toml
[tuning.polish]
type = "coarse_fine"
coarse = { window = { type = "fixed", lo = 0.5, hi = 16.0 }, spacing = { type = "log_per_decade", per_decade = 4 } }
extension = { npts = 4, max_rounds = 2 }
fine = { type = "log_count", n = 200 }
cross_validation = { scene_split = { type = "registered", name = "phototourism" } }
```

The bootstrap harness:

```toml
[tuning.lo_sensitivity]
type = "windowed_bootstrap"
scale_multiple_window = { lo = 1.0, hi = 3.3 }
scale_multiple = { spacing = { type = "linear_count", n = 17 } }
budget_sampling = { n_draws = 120 }
cross_validation = { scene_split = { type = "registered", name = "phototourism" } }
```

A joint detector tuning (one detection run per peak threshold):

```toml
[tuning.detection]
type = "joint_coarse_fine"
criterion = { type = "mean_average_accuracy", max_error = 10.0 }
axes.peak_threshold = { coarse = { type = "spaced", window = { type = "fixed", lo = 0.005, hi = 0.02 }, spacing = { type = "log_per_octave", per_octave = 2 } }, fine = { type = "log_count", n = 3 }, expensive = true }
axes.min_similarity = { coarse = { type = "spaced", window = { type = "fixed", lo = 0.3, hi = 0.8 }, spacing = { type = "linear_count", n = 6 } }, fine = { type = "linear_count", n = 5 } }
cross_validation = { scene_split = { type = "registered", name = "<a registered split>" } }
```

## The tuning artifact

`save_tuning_artifact(path, spec; grids, unit, code, data...)` is the one
writer of every artifact a tuning run persists (a tuning sidecar or its
tune-phase checkpoint, a probe CV artifact, a bundle of validation matrices);
`load_tuning_artifact(path) -> TuningArtifact` is the one reader.

| key | content |
|---|---|
| `tuning_format` | the `:tuning` artifact format (`src/artifact_format.jl`): the constant, not a number copied here |
| `tuning_name` | the `[tuning.<name>]` entry |
| `tuning_spec` | its complete scheme table, the cross-validation's resolved scene lists included |
| `tuning_hash` | `tuning_hash(spec)`: SHA-256 (16 hex) of the sorted TOML of the scheme table; names and spelling do not move it, any parameter does |
| `grid_specs` | axis → `GridSpec` table; the realized grids are rebuilt from these |
| `sampling_unit` | `"pair"` or `"query"`: the data's `sampling_unit` (`art.unit`) |
| `code_identity` | version and git commit (`-dirty` when uncommitted) of this package and of every module the caller names as the code its result came from; `julia`: its `version` and the `rng` algorithms |
| … | the run's own data (the arms, their tuned points as native values, the boundary hits, `d_g`, …) |

The reader refuses a file of another format first, with an `ArgumentError`
naming the file and the format change (regenerate it by re-running its
tuning). The same mechanism stamps pools, with `pool_format`. The reader then
refuses a stored spec whose hash does not match its table, and finally
rebuilds every grid: `art.grids[:axis]` is the realized grid.

## Provenance: from a deployed constant to the run

1. A deployed cross-validated constant carries
   `tuning = { config, name, hash, arm }`: the config of record holding the
   `[tuning.<name>]` spec, the spec's `tuning_hash`, and the arm it was tuned
   for. A transferred constant names the constant it transferred from.
2. Parsing the config's `[tuning.<name>]` (layered as the paper layers it)
   gives the spec; its hash must equal the constant's.
3. The run's artifact records the same spec and hash, its grid specs, its
   code identity and its run key, and the tuned point.
4. The run key (`run_key`: the SHA-256, 16 hex, of the resolved config's
   sorted TOML) names the directory that holds the artifact and its
   `_config.toml` / `_provenance.toml` records.
