# The composition evaluator

The composition evaluator measures a RANSAC scorer (the selection step)
composed with a local optimizer (LO, the refinement step) on stored pool
pairs. It lives in the `Evaluation` module, in `src/evaluation/engine/`.

An evaluated method is called an *arm*. Each arm is a pair of values that
implement VisualGeometryToolkitCore's contracts:

- a scoring, which implements `score!`, `select_on_grid!` and `with_width`;
- a local optimization, whose `RefitFamily` routes it to one of two paths.
  `NativeRefit` runs `locally_optimize!` inside VisualGeometryToolkitEstimation's
  RANSAC loop. `ExternalRefit` hands the model to a third-party refiner that
  maps a model to a model (`refine_external!`).

The evaluator holds no scorer or optimizer of its own. A project that
evaluates particular scorers or optimizers depends on their packages and
supplies their values.

## The pieces

| Piece | What it does |
|---|---|
| [`PairCtx`](@ref), [`build_pair_ctx`](@ref) | builds, once per pool pair, the pool-frame problem, the calibrations, the ground-truth (GT) pose and the refit-scoring template that every refit shares |
| [`refit_imc!`](@ref), [`guarded_refit`](@ref) | the one refit layer: one refit of a seed model, with its pose error, its wall time and a fallback to the seed |
| [`LoRefitSpec`](@ref), [`apply_lo_refits`](@ref) | turns a selection sweep into post-LO error matrices (pairs by grid columns) |
| [`stream_selections`](@ref) | streams the scenes: it stages each scene, runs `select_on_grid!` per grid, then one `apply_lo_refits` |
| [`SampleBudgetMethodSpec`](@ref), [`MsacSpec`](@ref), [`evaluate_sample_budget`](@ref) | the sample-budget engine: running-best selectors and LO checkpoints along each pair's hypothesis stream |
| [`live_runs`](@ref), [`live_row`](@ref) | end-to-end `ransac` runs on pool pairs, with a problem the caller builds per pair |
| [`ComposedSpec`](@ref), [`SoloSpec`](@ref), [`Roster`](@ref), [`deployed_arm_settings`](@ref), [`audit_operating_points`](@ref) | a paper's arms, and the gated readers of their tuned operating points |

"MSAC" is the M-estimator sample consensus score, the truncated-quadratic
baseline scorer of VisualGeometryToolkitCore.

## Widths

Widths are VisualGeometryToolkitCore width values. Each path receives its width
in the form it consumes:

- A native refit is seeded under the scoring template rebuilt at its own
  outlier half-width, `with_width(scoring, OutlierHalfwidth(a))`.
- An absolute width reaches the strategy as
  `with_width(lo, fixed_width(lo, s2))`.
- An external refiner receives `LossWidth(w)`, or `FixedS2(s2)` when the
  caller supplies the scale.

## A refit, step by step

The snippet below shows the shape of one refit. It needs a pool pair, a
scoring template with an outlier half-width, a local optimization and a seed
model, which come from a scoring pool and from the packages that implement the
scorer and the optimizer.

```julia
using VisualGeometryToolkitEval.Evaluation: build_pair_ctx, refit_imc!, problem_type

P = problem_type(:E)                              # the essential-matrix problem type
ctx = build_pair_ctx(P, pair; scoring = template)
imc, lo_ns = refit_imc!(ctx, lo, seed, nat)       # nat: the width of lo on the axis it consumes
```

`imc` is the pose error of the refit model in degrees, `max(ΔR, Δt)` over the
best decomposition, as the Image Matching Challenge (IMC) defines it. `lo_ns` is
the optimizer's wall time in nanoseconds. [`consumed_axis`](@ref) and
[`consumed_native`](@ref) give the width that `lo` consumes.
