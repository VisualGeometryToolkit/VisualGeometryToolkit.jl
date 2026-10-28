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
inhomogeneous `y = a + b·x` form with errors only in `y`.

## Robust Linear Least Squares

`fit_linear` is the robust coefficient vector `x` of `y ≈ A·x` (rows with
gross outliers): a reweighted least squares under a loss shape at a multiple
of the residuals' MAD scale, with optional fixed prior row weights — the
scalar-residual sibling of `fit_rotation`.

```@docs
fit_linear
LinearFitMetadata
```
