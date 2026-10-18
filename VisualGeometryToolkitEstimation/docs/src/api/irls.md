# IRLS & Posterior Weighting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## Generic Block-Coordinate-Descent Loop

The generic loop dispatches on `AbstractBlockCoordinateMethod`. Concrete
methods plug in by extending a small set of interface methods (all
declared `public`): `init_state!`, `update_model!`, `update_residuals!`,
`update_nuisance!`, `update_weights!`, `post_iteration!`, `stopping_criterion`
(a VGC `StoppingCriterion` value, composed with `|`; a new criterion implements
`stop_reason`), and `block_coordinate_result`. Every local-optimization strategy in this package plugs into
the `block_coordinate_descent!` loop through this interface, and a new one does the same.

The RANSAC LO refinement uses the loop via [`PosteriorIrlsMethod`](@ref)
(the iteration of [`Irls`](@ref), [`posterior_irls!`](@ref); downstream,
LocalOptimizationDoneRight's Student-t + uniform IRLS). Its model block is a bound, named update —
[`WeightedRefine`](@ref) (the problem's weighted `refine!`, the default) or
[`KeepModel`](@ref) (the model held: EM of the noise model on fixed residuals,
stopping on the mixture NLL) — and its state a [`PosteriorIrlsState`](@ref)
built from a seeded posterior and its residual buffers.

```@docs
AbstractBlockCoordinateMethod
block_coordinate_descent!
PosteriorIrlsMethod
PosteriorIrlsState
WeightedRefine
KeepModel
```

## Classifying a held model's residuals

Regression and classification apart: whatever fitted the model, its residuals
are FIXED here. The scorer's consensus set — any scorer with a uniform
outlier model (`outlier_halfwidth`), e.g. RansacScoringDoneRight's
`StudentTUniformScoring` — seeds the Student-t + uniform mixture, EM fits it
with the model held, and each residual's MAP label is VisualGeometryToolkitCore's
`is_map_inlier` of its posterior.

```@docs
fit_residual_mixture
ResidualMixtureMetadata
MixtureEmConfig
mixture_em_from_toml
```

## Student-t + Uniform Mixture

The inlier likelihood of the posterior-weight iteration and of direct
posterior weighting calls: the predictive Student-`t` density obtained by
marginalising the inlier scale `σ²` under the scorer's prior (`scale_prior`,
Jeffreys by default), against a uniform outlier density of half-width
`outlier_halfwidth(scoring)`.

A VisualGeometryToolkitCore `PosteriorWeightState` is seeded from a consensus set:

- **In the RANSAC loop**: [`consensus_posterior`](@ref) squares the seed
  model's residuals (already in the score buffers) and seeds `(ν₀, ŝ²₀)` from
  the scorer's `consensus_order` and `inlier_count` — no re-score, no sort.
- **Residuals only**: [`fit_residual_mixture`](@ref) scores the residuals once
  with the scorer and seeds from its consensus set. The outlier half-width is
  the scorer's; a scorer without one is refused
  ([`required_outlier_halfwidth`](@ref)).

[`consensus_s2!`](@ref) gives the consensus set's scale alone — the scale a
consensus-anchored loss runs at — for a refiner outside the loop.

[`PosteriorLikelihood`](@ref) updates the three nuisance parts — the scale, the
inlier prior and the Student-t dof — each under its policy: `Estimated()`, or
held at a value (VisualGeometryToolkitCore's `FixedS2(s2)`, `FixedPrior(π)`,
`FixedNu(ν)` and `ConsensusDof()`).

```@docs
PosteriorLikelihood
consensus_posterior
posterior_irls!
required_outlier_halfwidth
consensus_s2!
VisualGeometryToolkitCore.PosteriorWeightState
```

## Posterior Weighting

Lower-level posterior weighting utilities used by both the RANSAC inner loop
and stand-alone EM-style refinement. They are VisualGeometryToolkitCore's toolkit
(import them from there): spec arguments (`log_two_outlier_halfwidth`,
`model_dof`, `codimension`) are supplied explicitly, and the partition is a
consensus order and its size `k`, as a scorer's `consensus_order` and
`inlier_count` give them.

```@docs
VisualGeometryToolkitCore.posterior_weights!
VisualGeometryToolkitCore.posterior_inlier_probs!
VisualGeometryToolkitCore.soft_posterior_weights!
VisualGeometryToolkitCore.update_nu!
VisualGeometryToolkitCore.next_nu
VisualGeometryToolkitCore.median_seeded_posterior
```

## Scale and Mixture Estimation

The scale fit, the Student-t + uniform mixture and its ECM cycle, and the
degrees-of-freedom policies the posterior weighting above draws on. They are
VisualGeometryToolkitCore's (import them from there).

```@docs
VisualGeometryToolkitCore.fit_scale
VisualGeometryToolkitCore.update_scale!
VisualGeometryToolkitCore.partition_student_t
VisualGeometryToolkitCore.StudentTMixture
VisualGeometryToolkitCore.inlier_posterior
VisualGeometryToolkitCore.posterior_weight
VisualGeometryToolkitCore.mixture_nll
VisualGeometryToolkitCore.mixture_ecm!
VisualGeometryToolkitCore.update_inlier_prior!
VisualGeometryToolkitCore.sq_residuals
VisualGeometryToolkitCore.Estimated
VisualGeometryToolkitCore.FixedS2
VisualGeometryToolkitCore.FixedPrior
VisualGeometryToolkitCore.FixedNu
VisualGeometryToolkitCore.ConsensusDof
```

## Robust Losses

The loss SHAPES are VisualGeometryToolkitCore's (`robust/losses.jl`; import them
from there); they parameterize the loss-parametric LO strategies (the core
`Irls{K}`; downstream, LocalOptimizationDoneRight's `HQ{K}` and `Gnc{S}`), so
the loss is what carries the objective. A shape is
the ρ profile alone — it gains a width only when
`deploy(shape, ŝ²; scale_multiple)` composes the scale and the scale multiple
onto it, and it is that deployed loss every function below takes. See the
VisualGeometryToolkitCore documentation for the full API.

- `AbstractLossShape` — abstract supertype
- `GaussianShape` — no redescending (plain least squares)
- `HuberShape` — linear beyond the elbow
- `CauchyShape` — logarithmic penalty (heavy-tailed)
- `WelschShape` — exponential penalty; the paper's recipe deploys it as
  LocalOptimizationDoneRight's `HQ{WelschShape}` at its cross-validated scale
  multiples
- `GemanMcclureShape` — smooth redescender
- `TLSShape` / `SmoothTLSShape` — truncated least squares, hard and smoothed;
  `Irls{TLSShape}()` following the scorer's threshold is every `find_*`
  pipeline's default (least squares on the inliers)
- `BarronShape` — the α-parameterized family

`Irls{K}` refuses at construction a shape its mechanism cannot run:
`BarronShape` carries its α — the Barron family runs as
LocalOptimizationDoneRight's `BarronHQ` and `BarronAdaptive`.

Key functions: `weight(shape, s, sq_loss_width)` (equivalently
`lift_weight(loss, s)` on a deployed loss) and, for the half-quadratic lift,
`lift_rho` / `lift_weight` / `lift_zblock` / `lift_cost` — each on a
DEPLOYED loss, which carries the width, so no scale argument is supplied at
the call site. `decision_width(shape, decision_threshold)` converts the
residual at which a decision flips (for a lift arm, the half-weight radius
`r_½ = scale_multiple·ŝ`) into the shape's loss width, and takes the bare
shape because that is the conversion that produces a width in the first place.

!!! note "Retired API"
    The `AbstractLoss` family (`HuberLoss`, `CauchyLoss`, `TukeyLoss`,
    `WelschLoss`, `L2Loss`, `GemanMcClureLoss`, `FairLoss`) and the
    `AbstractScaleEstimator` family (`MadScale`, `WeightedScale`,
    `SpatialMadScale`) were deleted from VisualGeometryToolkitCore
    along with the file that held them. The loss shapes above replace the first;
    scale is estimated from the scorer's consensus set (`consensus_s2`)
    rather than by a standalone estimator object. (`FixedScale` is not
    retired: it is VisualGeometryToolkitCore's `ScaleInterface` label of a strategy
    deployed at an absolute width, `LossWidth`.)
