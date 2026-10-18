# Scoring API

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

The RANSAC loop scores through VisualGeometryToolkitCore's scoring contract; the
contract, its derived verbs and the baseline scorers are VisualGeometryToolkitCore's
and documented here because the loop is built on them. Paper scorers (the
Student-t + uniform marginal `StudentTUniformScoring`) are
RansacScoringDoneRight's and are documented there.

## The scoring contract

A scorer implements the required verbs; the optional ones default in
VisualGeometryToolkitCore.

```@docs
VisualGeometryToolkitCore.AbstractScoring
VisualGeometryToolkitCore.scoring_workspace
VisualGeometryToolkitCore.score!
VisualGeometryToolkitCore.consensus_order
VisualGeometryToolkitCore.is_predictive
VisualGeometryToolkitCore.scored_model
VisualGeometryToolkitCore.scale_prior
VisualGeometryToolkitCore.outlier_halfwidth
VisualGeometryToolkitCore.inlier_threshold
VisualGeometryToolkitCore.has_inlier_threshold
VisualGeometryToolkitCore.check_workspace
```

## Derived verbs and buffers

Stated once in VisualGeometryToolkitCore for every scorer: the loop's one scoring
path, the consensus set the local optimization starts from, its scale, and
the result's soft weights.

```@docs
VisualGeometryToolkitCore.ScoreBuffers
VisualGeometryToolkitCore.score_model!
VisualGeometryToolkitCore.ConsensusSet
VisualGeometryToolkitCore.inlier_count
VisualGeometryToolkitCore.consensus_set
VisualGeometryToolkitCore.consensus_s2
VisualGeometryToolkitCore.inlier_probabilities!
```

## Baseline scorers

MSAC and RANSAC at a threshold. `MsacScoring(problem; threshold)` is the
default scorer of every `find_*` pipeline.

```@docs
VisualGeometryToolkitCore.MsacScoring
VisualGeometryToolkitCore.RansacScoring
VisualGeometryToolkitCore.residual_quality
```

## Scale priors

The prior a scorer's consensus scale is computed under (`scale_prior`,
Jeffreys by default).

```@docs
VisualGeometryToolkitCore.IGPrior
VisualGeometryToolkitCore.JeffreysPrior
```

## Weight policies

```@docs
AbstractWeightPolicy
NormalizedWeightPolicy
```

## Local Optimization

A strategy implements VisualGeometryToolkitCore's local-optimization contract
(`lo_workspace`, `locally_optimize!`, returning `lo_estimate`). The core
strategies are [`NoLocalOptimization`](@ref) and [`Irls`](@ref); every other
strategy is LocalOptimizationDoneRight's. A width-carrying strategy's loss is
resolved by `loss_at` from its width mode (`lo_width`, set by `with_width`):
no width follows the scorer's `inlier_threshold`, `ScaleMultiple(m)` anchors
it at the consensus scale, `LossWidth(w)` fixes it.

```@docs
VisualGeometryToolkitCore.AbstractLocalOptimization
VisualGeometryToolkitCore.lo_workspace
VisualGeometryToolkitCore.locally_optimize!
VisualGeometryToolkitCore.lo_width
VisualGeometryToolkitCore.with_width
VisualGeometryToolkitCore.loss_at
VisualGeometryToolkitCore.check_composition
VisualGeometryToolkitCore.ScaleMultiple
VisualGeometryToolkitCore.LossWidth
NoLocalOptimization
Irls
default_local_optimization
```
