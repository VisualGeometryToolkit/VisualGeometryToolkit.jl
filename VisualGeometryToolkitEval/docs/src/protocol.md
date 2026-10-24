# The experimental protocol

This page is the one statement of the evaluation protocol of the stack. A
paper's drivers and the package's own sources point here, and their headers
state only what is specific to them: their command line, schemas and stages.

The mechanisms live in this package's modules: `Tuning` (the cross-validation
specification), `Stats` (the statistics), `Evaluation` (the composition
evaluator and the single cross-validation path), `Experiments` (run records),
`Render` (figures), and `Pools` and `Datasets` (data). Every parameter of
record is in `configs/default.toml`, `configs/tuning.toml` or
`configs/splits.toml`.

Abbreviations used on this page:

| Abbreviation | Meaning |
|---|---|
| CV | cross-validation |
| LO | local optimization, the refinement step after RANSAC selects a model |
| GT | ground truth |
| BCa | bias-corrected and accelerated (bootstrap interval) |
| mAA | mean average accuracy |
| AUC | area under the curve |
| E, F, H | the essential matrix, the fundamental matrix and the homography, and the tracks that estimate them |
| ETH3D | the ETH Zürich 3D reconstruction benchmark |

## 0. Cross-validation: tuning against reporting

Every tuning names its data in its `[tuning.<name>]` spec, as a
`cross_validation` value ([`CrossValidation`](@ref); the schema is in
[The `[tuning]` schema](@ref)). The spec has three parts.

- **Scenes.** A [`RegisteredSceneSplit`](@ref) is a split that the spec names
  and that is listed once, in `configs/splits.toml`, the one file of record of
  every scene list of the stack. Validation scenes are the tuning side, where
  every hyperparameter is validated. Test scenes are the reporting side. The
  two lists must be disjoint. The parser writes the resolved lists into the
  complete spec, so they enter the spec hash, the run key and every tuning
  artifact.
- **Units.** Each side has a [`UnitSampling`](@ref) over the data's sampling
  unit. A dataset declares its unit with [`sampling_unit`](@ref): an image pair
  for a two-view dataset, a query for an absolute-pose one. The samplings are
  [`ExhaustiveSampling`](@ref), [`StratifiedSampling`](@ref) (a `fraction` of
  each scene, then at most `cap_per_scene`, which preserves the mix of scenes)
  and [`UniformSampling`](@ref) (a `fraction` of the side's pooled units, then
  at most `cap`). Every sampler permutes, keeps the fraction, applies the cap,
  and only then sorts back into data order, so a cap never keeps a prefix of
  the data.
- **Seeds.** Every draw is seeded by [`stable_seed`](@ref), the SHA-256 hash of
  the spec's seed and of the name it applies to, such as a scene name or a
  budget and draw index. `Base.hash` is never used, because it is not stable
  across Julia versions. `seed + index` is never used either: with a named
  seed, adding, removing or reordering scenes leaves every other scene's sample
  unchanged.

A driver accepts the samplings it supports through the signature of the method
it calls, for example `cv::CrossValidation{<:SceneSplit, StratifiedSampling}`.
An unsupported spec is then a `MethodError` and is never silently
reinterpreted. No driver, paper config or package source states a scene list,
a split, a pair prefix or a sampling seed in code. Run directories come from
the run records ([`run_key`](@ref), [`experiment_dir`](@ref),
[`results_root`](@ref)).

## 1. The fairness axis: the scale multiple

Methods are compared at the same operating point, not at arbitrary native
values. The shared axis is the scale multiple: the number of inlier-noise
standard deviations σ at which a method's inlier/outlier decision flips. It is
anchored at the reference scale `ŝ = √median σ̂²` of the validation data
([`reference_scale`](@ref)). The maps from a scale multiple to a native width
are VisualGeometryToolkitCore's (`to_native`, `native_map`, `decision_width`),
one per width axis:

| axis | native width at scale multiple `k` |
|---|---|
| `LossWidth` (threshold family) | `loss_width = k·ŝ` (the truncated loss: identity) |
| `OutlierHalfwidth` (Student-t marginal) | `outlier_halfwidth = ŝ·√(π/2)·e^{k²/2}` at codimension `d_g = 1` |
| a lifted loss | the half-weight width of its shape |

**One refit path, one scale port.** Every refiner runs through the one refit
layer ([`refit_imc!`](@ref) and VisualGeometryToolkitCore's `RefitFamily`, with
the values `NativeRefit` and `ExternalRefit`). The deployed native width
reaches a refiner only as its width on the axis it consumes
([`consumed_axis`](@ref), [`consumed_native`](@ref)). A loss-width refiner
that rides on an outlier-half-width selector grid converts at the refit
protocol's fixed [`OPERATING_SCALE_MULTIPLE`](@ref) (`configs/default.toml`,
table `[refit]`). No refiner estimates its own scale from a plug-in window.

**The outlier-half-width axis carries a codimension.** The marginal map takes
the residual codimension `d_g = codimension(problem_type)`: 1 on the E and F
tracks and 2 on H. At `ŝ = 1` over the window [0.8, 6.0], the upper edge moves
from `8.2293e7` (`d_g = 1`) to `1.0156e4` (`d_g = 2`), a change of 3.91
decades. A native grid and the window searched inside it must therefore be cut
at the same `d_g`. The `coverage` window rule cuts the grid through the same
map that the search window uses, and this is what makes its coverage guarantee
hold. `d_g` has no default anywhere, and every tuning artifact persists it as
`"d_g"`.

## 2. Tuning: the single CV path

Every hyperparameter is cross-validated by one discipline, the coarse-to-fine
search ([`coarse_fine_search`](@ref), with a `[tuning.<name>]` spec of scheme
`joint_coarse_fine`, or `coarse_fine` for one scalar). It runs in three steps:

1. It evaluates a coarse product grid. Whenever the joint argmin lies within
   `edge_margin` points of a face, the grid extends that axis.
2. It evaluates a fine product grid over the bracket of the coarse argmin on
   every axis.
3. It takes the free argmin of the validation objective on the fine grid.

Windows are located by expansion; they are never pinned or widened by hand.
Several searches share one sweep of a staged validation pool (the search keys
of one call, or [`coarse_fine_cv_batched`](@ref) for scalars), and an
expensive axis is evaluated one value at a time. The artifact
([`save_coarse_fine_artifact`](@ref)) stores the per-unit sufficient
statistics of each cell, from which the bootstrap of the selection is exact;
[`deployed_setting`](@ref) reads the selection. A paper tunes a scorer and LO
arm as a sequence of such scalar CVs: first the selection width, then the
refinement width.

The tuned point is persisted as a tuning artifact
([`save_tuning_artifact`](@ref)): the spec, its hash, the grid specs, the
sampling unit and the code identity. Only the deployed readers read it
([`deployed_arm_settings`](@ref), [`deployed_budget_settings`](@ref),
[`deployed_basin_settings`](@ref)). They refuse a tuned point on a boundary of
its window (the artifact's `boundary_hits`, or
[`enforce_probe_interior`](@ref) for a probe artifact), because a window that
clips the argmin has not located it.

## 3. Miscalibration: the fixed-scale-multiple protocol

Each method is tuned once at the true validation scale. Then the deployed
plug-in scale is corrupted, `σ̂ → ζσ̂`, while the tuned scale multiple is held
fixed ([`corrupt_native`](@ref)). Every native parameter is homogeneous of
degree 1 in `σ̂`, so the corruption multiplies each method's tuned parameter by
the same factor ζ. Each method thus suffers the same relative error in its own
units. A tuned scale multiple is anchored on the data's own partition and is
not corrupted. Test data is untouched. Evaluation reads the grid column
nearest to `ζ·native*` ([`nearest_column`](@ref)). This lookup is for
evaluation only; a tuning never takes a nearest column.

## 4. The validation budget: the tuning-budget sweep

Draws come from the band-eligible validation units: those with a finite σ̂²
inside the band `[ŝ_min², ŝ_max²]` of [`scale_bounds`](@ref). The draws are the
spec's [`BudgetSampling`](@ref) ([`budget_units`](@ref)). Draw `r` of budget
`n` takes `n` units without replacement from
`rng(stable_seed(seed, "budget", n, r))` and sorts them.

For each draw, the protocol computes the reference scale `ŝ(n)` of the drawn
units, takes the free argmin inside the window at `ŝ(n)`, and evaluates the
chosen point on the test data ([`WindowedGridMethod`](@ref),
[`compute_method_rows`](@ref)). The reported point is the mean over draws of
the per-draw test medians. The interval is a BCa interval over draws,
resampled from `resampling_rng(budget, n)`, that is
`rng(stable_seed(seed, "resampling", n))`, a stream of its own and never a
draw's.

## 5. The sample budget

[`evaluate_sample_budget`](@ref) walks each pair's shared hypothesis stream
with running-best selectors. Selection checkpoints come every `batch_size`
samples, and LO checkpoints on the `lo_every` grid
([`SampleBudgetConfig`](@ref)). Every arm's selection and refinement widths
come from its tuning artifact through [`deployed_budget_settings`](@ref).
There is no fallback and no self-tuning. Every LO call runs within the
evaluation's equal budget of model updates
([`lo_iteration_budget`](@ref VisualGeometryToolkitEval.Evaluation.lo_iteration_budget),
`configs/default.toml`, table `[local_optimization]`).

## 6. The metric: pose error, with the F upgrade

All pose errors use the coherent metric `max(ΔR, Δt)` in degrees over the
single best decomposition (VisualGeometryToolkitCore's `imc_pose_error`, named
after the Image Matching Challenge, IMC). Pool models are pool-frame F
containers on both tracks. A pool model is therefore always scored through the
intrinsics upgrade [`model_imc`](@ref)`(F, geom)`: `K₂ᵀFK₁ → E →` the pose that
resolves cheirality. It is never scored with the bare `imc_pose_error(θ, R, t)`,
which treats its argument as a metric essential matrix. That misuse once
inflated the oracle floor of the ETH3D E track from 0.19° to 0.73°.

## 7. The oracle floor: the selection lower bound

For each miscalibration cell, the floor is
`floor = median_i min_{θ ∈ pool_i} imc(θ)` over the test pairs (the budget
engine's [`OracleSelector`](@ref VisualGeometryToolkitEval.Evaluation.OracleSelector)).
Oracle selection consumes no plug-in scale, so the floor does not depend on ζ.
It bounds selection only: a refit may go below it by leaving the discrete
hypothesis pool. A miscalibration panel carries its floor (the `oracle`
keyword of [`emit_miscal_faceted`](@ref)).

## 8. Statistics

Point estimates and intervals go through `Stats` only:

- [`finite_median`](@ref). Argmin objectives use
  `finite_median(...; empty = Inf)`, so that a column of only `NaN` never
  wins.
- [`bootstrap_ci`](@ref), the BCa interval.
- [`scene_mean_median_rows!`](@ref), the mean of per-scene medians, with a BCa
  interval over scenes resampled from `stable_seed(seed, "scene_mean_median", j)`.

The mAA follows one rule, [`eligible_maa`](@ref). A pair with ground truth and
enough valid correspondences to run is eligible, and a run on it that yields no
model scores 0 in the denominator. A pair that cannot be processed is skipped.
[`pair_outcomes`](@ref) reports the counts beside every mAA. The mAA is the
continuous AUC of the accuracy curve up to `maa_threshold_deg`
(`configs/default.toml`, table `[metrics]`; VisualGeometryToolkitCore's
`mean_average_accuracy`).

## 9. Figure conventions

The `Render` emitters enforce a house style, so that figures are uniform across
papers:

- **Confidence intervals are shaded bands**, never error bars: scene-clustered
  BCa 95% intervals (section 8).
- **Labels and markers come from the framework**, never from hand-written
  tables: VisualGeometryToolkitCore's `lo_label(lo)` and `lo_marker(lo)`. A
  partition-anchored scale is marked `-\hat s`, a fixed cross-validated scale
  is unmarked, and a self-scaling LO carries no suffix. A renderer never spells
  out a method's name.
- **One arm has one colour everywhere; the marker shape is the solver family.**
- **The oracle (the GT-best model) is a solid black line with no markers.**
- **Basin panels** ([`emit_basin_faceted`](@ref)): the left panel is the
  success rate `P(e ≤ succ°)` on `[0, 1]`. The right panel is the conditional
  mean `E[e | e ≤ succ°]` on a linear axis with degree ticks, masked where the
  success rate is below `min_success_for_cond_mean` (`configs/default.toml`,
  table `[render]`). The reference arm without LO is the perturbed seed.
- **Median-error panels** keep a log axis, since the median error is unbounded
  there.
