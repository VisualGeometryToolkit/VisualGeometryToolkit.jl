# VisualGeometryToolkitEval changelog

## Unreleased

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
