# Scoring API

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

The RANSAC loop scores through VisualGeometryToolkitCore's scoring contract. The
contract, its derived verbs and the baseline scorers are VisualGeometryToolkitCore's
and documented in its manual; this page lists them because the loop is built on
them. Paper scorers (the Student-t + uniform marginal `StudentTUniformScoring`)
are RANSACScoringDoneRight's and are documented there.

## The scoring contract

A scorer implements the required verbs; the optional ones default in
VisualGeometryToolkitCore:
[`AbstractScoring`](@extref VisualGeometryToolkitCore.AbstractScoring),
[`scoring_workspace`](@extref VisualGeometryToolkitCore.scoring_workspace),
[`score!`](@extref VisualGeometryToolkitCore.score!),
[`consensus_order`](@extref VisualGeometryToolkitCore.consensus_order),
[`is_predictive`](@extref VisualGeometryToolkitCore.is_predictive),
[`scored_model`](@extref VisualGeometryToolkitCore.scored_model),
[`scale_prior`](@extref VisualGeometryToolkitCore.scale_prior),
[`outlier_halfwidth`](@extref VisualGeometryToolkitCore.outlier_halfwidth),
[`decision_boundary`](@extref VisualGeometryToolkitCore.decision_boundary),
[`has_decision_boundary`](@extref VisualGeometryToolkitCore.has_decision_boundary) and
[`check_workspace`](@extref VisualGeometryToolkitCore.check_workspace).

## Derived verbs and buffers

Stated once in VisualGeometryToolkitCore for every scorer: the loop's one scoring
path ([`ScoreBuffers`](@extref VisualGeometryToolkitCore.ScoreBuffers),
[`score_model!`](@extref VisualGeometryToolkitCore.score_model!)), the consensus
set the local optimization starts from
([`ConsensusSet`](@extref VisualGeometryToolkitCore.ConsensusSet),
[`inlier_count`](@extref VisualGeometryToolkitCore.inlier_count),
[`consensus_set`](@extref VisualGeometryToolkitCore.consensus_set)), its scale
([`consensus_s2`](@extref VisualGeometryToolkitCore.consensus_s2)), and the
result's soft weights
([`inlier_probabilities!`](@extref VisualGeometryToolkitCore.inlier_probabilities!)).

## Baseline scorers

MSAC and RANSAC at a threshold,
[`MsacScoring`](@extref VisualGeometryToolkitCore.MsacScoring) and
[`RansacScoring`](@extref VisualGeometryToolkitCore.RansacScoring), with their
[`residual_quality`](@extref VisualGeometryToolkitCore.residual_quality).
MSAC (`[pipelines.<class>] scoring`) is the default scorer of every `find_*`
pipeline.

## SIMFIT++ loss-width estimation

A step-for-step port of SIMFIT++ ([Edstedt2025](@citet); the reference
implementation is [Parskatt/simfitpp](https://github.com/Parskatt/simfitpp)):
the noise scale of a robust fit, hence the loss width it runs at, estimated from
held-out residuals so that the fit's own bias does not shrink it. The geometry
estimator is a pair of callbacks, so the same loop runs on a full RANSAC refit
([`simfitpp_loss_width`](@ref)) or on a fixed hypothesis pool
([`simfitpp_select`](@ref), the paper's protocol: the MSAC-best pool model on
the train split). [`SimFitPP`](@ref) holds the reference's defaults (coverage
`alpha = 0.99`, loss-width bounds 0.25–8 in the residuals' unit, pixels in the
reference); the pair method of `simfitpp_select` therefore runs on the pool's
pixel residuals (VisualGeometryToolkitCore's `Pools.pixel_residuals`).

```@docs
SimFitPP
simfitpp_loss_width
simfitpp_select
unbiased_median_variance
with_residual_unit
```

## Scale priors

The prior a scorer's consensus scale is computed under (`scale_prior`,
Jeffreys by default):
[`IGPrior`](@extref VisualGeometryToolkitCore.IGPrior) and
[`JeffreysPrior`](@extref VisualGeometryToolkitCore.JeffreysPrior).

## Weight policies

[`AbstractWeightPolicy`](@extref VisualGeometryToolkitCore.AbstractWeightPolicy)
and [`NormalizedWeightPolicy`](@extref VisualGeometryToolkitCore.NormalizedWeightPolicy)
are VisualGeometryToolkitCore's.

## Local Optimization

A strategy implements VisualGeometryToolkitCore's local-optimization contract
(`lo_workspace`, `locally_optimize!`, returning `lo_estimate`). The core
strategies are [`NoLocalOptimization`](@ref), [`Irls`](@ref) and
[`Triggs`](@ref) — the robustified chart LM: one NLLSsolver solve under a
loss with Triggs' second-order correction, its width `scale_multiple·ŝ` at the
hypothesis's Student-t scale, for every chart problem; every other strategy is
LocalOptimizationDoneRight's. A width-carrying strategy's loss is
resolved by `loss_at` from its width mode (`lo_width`, set by `with_width`):
no width follows the scorer's `decision_boundary`, `ScaleMultiple(m)` anchors
it at the consensus scale, `LossWidth(w)` fixes it.

The contract is VisualGeometryToolkitCore's:
[`AbstractLocalOptimization`](@extref VisualGeometryToolkitCore.AbstractLocalOptimization),
[`lo_workspace`](@extref VisualGeometryToolkitCore.lo_workspace),
[`lo_width`](@extref VisualGeometryToolkitCore.lo_width),
[`with_width`](@extref VisualGeometryToolkitCore.with_width),
[`loss_at`](@extref VisualGeometryToolkitCore.loss_at),
[`check_composition`](@extref VisualGeometryToolkitCore.check_composition),
[`ScaleMultiple`](@extref VisualGeometryToolkitCore.ScaleMultiple) and
[`LossWidth`](@extref VisualGeometryToolkitCore.LossWidth). The methods of
`locally_optimize!` for this package's strategies are documented here.

```@docs
VisualGeometryToolkitCore.locally_optimize!
NoLocalOptimization
Irls
Triggs
default_local_optimization
```

### Placement and instrumentation

A strategy runs inside the loop, on every candidate that enters the
local-optimization gate, unless it is placed at the end: [`TerminalLO`](@ref)
runs its inner strategy once on the final incumbent of a race decided on raw
hypothesis scores. [`CountedLO`](@ref) counts a strategy's calls in either
placement.

```@docs
TerminalLO
lo_placement
InLoopPlacement
TerminalPlacement
CountedLO
```

### Loss resolution

The core strategies resolve the loss they deploy through these helpers. A
downstream strategy can use them to follow the same width rules.

```@docs
lo_loss
strategy_loss
deployed_shape
fixed_width
lo_width_value
plain_scale_multiple
locally_optimize_rescored!
```

### Lifted local optimization

The chart problems provide VisualGeometryToolkitCore's chart-rows contract, which
the half-quadratic lifts of LocalOptimizationDoneRight consume.

```@docs
ManifoldLiftProvider
LiftableProblem
```
