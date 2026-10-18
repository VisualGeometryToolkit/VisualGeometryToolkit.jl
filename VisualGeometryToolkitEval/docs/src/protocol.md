# The experimental protocol

This page is the one statement of the evaluation protocol of the stack. A
paper's drivers and the package's own sources point here; their headers state
only what is specific to them (their CLI, schemas and stages). The mechanisms
live in this package's modules: `Tuning` (the cross-validation
specification), `Stats` (the statistics), `Evaluation` (the composition
evaluator and the single CV path), `Experiments` (run records), `Render`
(figures), `Pools` and `Datasets` (data). Every parameter of record is in
`configs/default.toml`, `configs/tuning.toml` or `configs/splits.toml`.

## 0. Cross-validation: tuning against reporting

Every tuning names its data in its `[tuning.<name>]` spec, as a
`cross_validation` (`Tuning.CrossValidation`; the schema is in
[The `[tuning]` schema](tuning.md)):

- **Scenes.** A `RegisteredSceneSplit`: a split NAMED by the spec and listed
  once, in `configs/splits.toml`, the one file of record of every scene list
  of the stack. Validation scenes are the tuning side (every hyperparameter is
  validated there), test scenes the reporting side; the two lists are
  disjoint (refused otherwise). The parser writes the resolved lists into the
  complete spec, so they are in the spec hash, the run key and every tuning
  artifact.
- **Units.** Per side, a `UnitSampling` over the data's sampling unit (a
  dataset declares it with `sampling_unit`: an image pair of a two-view
  dataset, a query of an absolute-pose one): `ExhaustiveSampling`,
  `StratifiedSampling` (a `fraction` of each scene, then at most
  `cap_per_scene`; the scene mix is preserved) or `UniformSampling` (a
  `fraction` of the side's pooled units, then at most `cap`). Every sampler
  permutes, keeps the fraction, caps, then sorts back into data order LAST, so
  a cap never keeps a prefix.
- **Seeds.** Every draw is seeded by `Stats.stable_seed` (SHA-256 of the
  spec's seed and the NAME it applies to: a scene name, a budget and draw
  index), never `Base.hash` (not stable across Julia versions) and never
  `seed + index` (so adding, removing or reordering scenes leaves every other
  scene's sample unchanged).

A driver accepts the samplings it supports by the SIGNATURE of the method it
calls (`cv::CrossValidation{<:SceneSplit, StratifiedSampling}`), so an
unsupported spec is a `MethodError`, never a silent reinterpretation. No
driver, paper config or package source states a scene list, a split, a pair
prefix or a sampling seed in code. Run directories come from the run records
(`Experiments.run_key`, `experiment_dir`, `results_root`).

## 1. The fairness axis: the scale multiple

Methods are compared at the SAME operating point, not at arbitrary native
values. The shared axis is the scale multiple (the number of inlier-noise
σ at which a method's inlier/outlier decision flips), anchored at the
reference scale `ŝ = √median σ̂²` of the validation data
(`Stats.reference_scale`). The scale-multiple → native maps are
VisualGeometryToolkitCore's (`to_native`, `native_map`, `decision_width`), one per
width axis:

| axis | native width at scale multiple `k` |
|---|---|
| `LossWidth` (threshold family) | `loss_width = k·ŝ` (the truncated loss: identity) |
| `OutlierHalfwidth` (Student-t marginal) | `outlier_halfwidth = ŝ·√(π/2)·e^{k²/2}` at codimension `d_g = 1` |
| a lifted loss | the half-weight width of its shape |

**One refit path, one scale port.** Every refiner runs through the one
refit layer (`Evaluation.refit_imc!`, VisualGeometryToolkitCore's `RefitFamily`:
`NativeRefit` or `ExternalRefit`). The deployed native reaches a refiner
solely as its width on the axis it consumes (`consumed_axis`,
`consumed_native`); a loss-width refiner riding an outlier-half-width selector
grid converts at the refit protocol's fixed `OPERATING_SCALE_MULTIPLE`
(`configs/default.toml [refit]`). No refiner estimates its own scale from a
plug-in window.

**The outlier-half-width axis carries a codimension.** The marginal map takes
the residual codimension `d_g = codimension(problem_type)`: 1 on the E and F
tracks, 2 on H. At `ŝ = 1` over the window [0.8, 6.0] the upper edge moves
from `8.2293e7` (`d_g = 1`) to `1.0156e4` (`d_g = 2`), 3.91 decades. A native
grid and the window searched inside it must be cut at the SAME `d_g`: the
`coverage` window rule cuts the grid through the very map the search window
uses, which is what makes its coverage guarantee hold. `d_g` has no default
anywhere, and every tuning artifact persists it as `"d_g"`.

## 2. Tuning: the single CV path

Every hyperparameter is cross-validated by ONE discipline, the coarse-fine
search (`Evaluation.coarse_fine_search`, a `[tuning.<name>]` spec of scheme
`joint_coarse_fine`, or `coarse_fine` for one scalar): a coarse product grid
that EXTENDS an axis whenever the joint argmin lies within `edge_margin`
points of a face, then a fine product over the coarse argmin's bracket on
every axis, then a FREE argmin of the validation objective. Windows are
located by expansion, never hand-pinned and never hand-widened. Several
searches share one sweep of a staged validation pool (the search keys of one
call; `coarse_fine_cv_batched` for scalars), and an expensive axis is
evaluated one value at a time. The artifact (`save_coarse_fine_artifact`)
stores each cell's per-unit sufficient statistics, from which the bootstrap
of the selection is exact; `deployed_setting` reads the selection. A paper tunes a score+LO arm as a sequence of such scalar
CVs (the selection width, then the refinement width), each of them this path.

The tuned point is persisted as a tuning artifact (`save_tuning_artifact`:
the spec, its hash, the grid specs, the sampling unit, the code identity), and
read only by the deployed readers (`deployed_arm_settings`,
`deployed_budget_settings`, `deployed_basin_settings`), which refuse a tuned
point on a boundary of its window (the artifact's `boundary_hits`;
`enforce_probe_interior` for a probe artifact): a window that clips the argmin
has not located it.

## 3. Miscalibration: the fixed-scale-multiple protocol

Tune each method ONCE at the true validation scale, then corrupt the DEPLOYED
plug-in scale, `σ̂ → ζσ̂`, holding the tuned scale multiple fixed
(`Tuning.corrupt_native`). Every native parameter is degree-1 homogeneous in
`σ̂`, so the corruption multiplies each method's tuned parameter by the SAME
factor ζ: the identical relative error in each method's own units. A tuned
scale multiple is anchored on the data's own partition and is not corrupted.
Test data is untouched. Evaluation reads the grid column nearest
`ζ·native*` (`nearest_column`; evaluation only, a tuning never takes a
nearest column).

## 4. The validation budget: the tuning-budget sweep

Draws come from the band-eligible validation units (a finite σ̂² inside the
`[ŝ_min², ŝ_max²]` band of `Stats.scale_bounds`). The draws are the spec's
`BudgetSampling` (`budget_units`: draw `r` of budget `n` is `n` units without
replacement from `rng(stable_seed(seed, "budget", n, r))`, sorted). Per draw:
the reference scale `ŝ(n)` of the drawn units, the free argmin inside the
window at `ŝ(n)`, and test evaluation at every chosen point
(`WindowedGridMethod`, `compute_method_rows`). The point is the mean over
draws of the per-draw test medians; the interval is BCa over draws, resampled
from `resampling_rng(budget, n)` (`rng(stable_seed(seed, "resampling", n))`,
its own stream, never a draw's).

## 5. The sample budget

`Evaluation.evaluate_sample_budget` walks each pair's shared hypothesis stream
with running-best selectors; selection checkpoints every `batch_size`
samples, LO checkpoints on the `lo_every` grid (`SampleBudgetConfig`). Every
arm's (selection, refinement) widths come from its tuning artifact through
`deployed_budget_settings`; there is no fallback and no self-tuning. Every LO
call runs at most the evaluation's equal model-update budget
(`lo_iteration_budget`, `configs/default.toml [local_optimization]`).

## 6. The metric: pose error, with the F upgrade

All pose errors are the coherent metric `max(ΔR, Δt)` in degrees over the
single best decomposition (VisualGeometryToolkitCore `imc_pose_error`). Pool models
are pool-frame F containers on both tracks, so a pool model is ALWAYS scored
through the intrinsics upgrade `model_imc(F, geom)` (K₂ᵀFK₁ → E →
cheirality-resolved pose), never the bare `imc_pose_error(θ, R, t)`, which
treats its argument as a metric essential matrix (that misuse once inflated an
ETH3D-E oracle floor from 0.19° to 0.73°).

## 7. The oracle floor: the selection lower bound

Per miscalibration cell, `floor = median_i min_{θ ∈ pool_i} imc(θ)` over the
test pairs (the budget engine's `OracleSelector`). Oracle selection consumes
no plug-in scale, so the floor does not depend on ζ and bounds SELECTION only:
a refit may cross it by leaving the discrete hypothesis pool. A
miscalibration panel carries its floor (`Render.emit_miscal_faceted`'s
`oracle`).

## 8. Statistics

Point estimates and intervals go through `Stats` only: `finite_median`
(argmin objectives use `finite_median(...; empty = Inf)` so an all-NaN column
never wins), `bootstrap_ci` (BCa), `scene_mean_median_rows!` (the mean of
per-scene medians with BCa over scenes, resampled from
`stable_seed(seed, "scene_mean_median", j)`). mAA follows ONE rule,
`eligible_maa`: a pair with ground truth and enough valid correspondences to
run is ELIGIBLE (a run that yields no model scores 0 in the denominator), and
a pair that cannot be processed is SKIPPED; `pair_outcomes` reports the counts
beside every mAA. mAA is the continuous AUC up to `[metrics]
maa_threshold_deg` (VisualGeometryToolkitCore `mean_average_accuracy`).

## 9. Figure conventions

House style, enforced by the `Render` emitters so figures are uniform across
papers:

- **Confidence intervals are shaded bands**, never error bars: scene-clustered
  BCa 95 % (§8).
- **Labels and markers come from the framework**, never from hand tables:
  VisualGeometryToolkitCore `lo_label(lo)` and `lo_marker(lo)`. A partition-anchored
  scale is marked `-\hat s`; a fixed cross-validated scale is unmarked; a
  self-scaling LO carries no suffix. A renderer never spells a method string.
- **One arm is one colour everywhere; the marker shape is the solver family.**
- **The oracle (GT-best) is a solid black line with no markers.**
- **Basin panels** (`Render.emit_basin_faceted`): the left panel is the
  success rate `P(e ≤ succ°)` on `[0, 1]`; the right panel is the conditional
  mean `E[e | e ≤ succ°]` on a linear axis with degree ticks, masked where the
  success rate is below `[render] min_success_for_cond_mean`; the no-LO
  reference arm is the perturbed seed.
- **Median-error panels** keep a log axis (median error is unbounded there).
