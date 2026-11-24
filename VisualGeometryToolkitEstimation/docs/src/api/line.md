# Line Fitting

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problems

Two RANSAC problem types for 2D line fitting, distinguished by their noise
model:

```@docs
LineFittingProblem
InhomLineFittingProblem
```

`LineFittingProblem` uses Hesse normal form (`Line2D{T}`) with per-point
covariance shapes (standardized residual). `InhomLineFittingProblem` is the
inhomogeneous `y = a + b·x` form with errors only in `y`. Neither has a class of
record: `Ransac(; scoring = MsacScoring(f; threshold))` names its scorer.

## Robust Linear Least Squares

`solve(EstimationProblem(LinearModel, (A, y[, prior])), ReweightedLeastSquares(…); scale | width)`
is the robust coefficient vector `x` of `y ≈ A·x` (rows with gross outliers): a
reweighted least squares under a loss shape at one fixed width,
`efficiency_constant × scale` from the solve's `scale` (or its `width`), with
optional fixed prior row weights — the scalar-residual sibling of the rotation
from direction pairs.

```@docs
LinearModel
FitMetadata
```
