# Scoring API

```@meta
CurrentModule = RobustVisualGeometry
```

The RANSAC loop scores through VisualGeometryCore's scoring contract; the
contract, its derived verbs and the baseline scorers are VisualGeometryCore's
and documented here because the loop is built on them. Paper scorers (the
Student-t + uniform marginal `StudentTUniformScoring`) are
RansacScoringDoneRight's and are documented there.

## The scoring contract

A scorer implements the required verbs; the optional ones default in
VisualGeometryCore.

```@docs
VisualGeometryCore.AbstractScoring
VisualGeometryCore.scoring_workspace
VisualGeometryCore.score!
VisualGeometryCore.consensus_order
VisualGeometryCore.is_predictive
VisualGeometryCore.scored_model
VisualGeometryCore.scale_prior
VisualGeometryCore.outlier_halfwidth
VisualGeometryCore.inlier_threshold
VisualGeometryCore.has_inlier_threshold
VisualGeometryCore.check_workspace
```

## Derived verbs and buffers

Stated once in VisualGeometryCore for every scorer: the loop's one scoring
path, the consensus set the local optimization starts from, its scale, and
the result's soft weights.

```@docs
VisualGeometryCore.ScoreBuffers
VisualGeometryCore.score_model!
VisualGeometryCore.ConsensusSet
VisualGeometryCore.inlier_count
VisualGeometryCore.consensus_set
VisualGeometryCore.consensus_s2
VisualGeometryCore.inlier_probabilities!
```

## Baseline scorers

MSAC and RANSAC at a threshold. `MsacScoring(problem; threshold)` is the
default scorer of every `find_*` pipeline.

```@docs
VisualGeometryCore.MsacScoring
VisualGeometryCore.RansacScoring
VisualGeometryCore.residual_quality
```

## Scale priors

The prior a scorer's consensus scale is computed under (`scale_prior`,
Jeffreys by default).

```@docs
VisualGeometryCore.IGPrior
VisualGeometryCore.JeffreysPrior
```

## Weight policies

```@docs
AbstractWeightPolicy
NormalizedWeightPolicy
```

## Local Optimization

A strategy implements VisualGeometryCore's local-optimization contract
(`lo_workspace`, `locally_optimize!`, returning `lo_estimate`). The core
strategies are [`NoLocalOptimization`](@ref) and [`Irls`](@ref); every other
strategy is LocalOptimizationDoneRight's. A width-carrying strategy's loss is
resolved by `loss_at` from its width mode (`lo_width`, set by `with_width`):
no width follows the scorer's `inlier_threshold`, `ScaleMultiple(m)` anchors
it at the consensus scale, `LossWidth(w)` fixes it.

```@docs
VisualGeometryCore.AbstractLocalOptimization
VisualGeometryCore.lo_workspace
VisualGeometryCore.locally_optimize!
VisualGeometryCore.lo_width
VisualGeometryCore.with_width
VisualGeometryCore.loss_at
VisualGeometryCore.check_composition
VisualGeometryCore.ScaleMultiple
VisualGeometryCore.LossWidth
NoLocalOptimization
Irls
default_local_optimization
```
