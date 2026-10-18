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
