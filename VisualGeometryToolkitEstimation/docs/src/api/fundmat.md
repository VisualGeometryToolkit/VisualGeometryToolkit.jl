# Fundamental Matrix

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problem

`FundMatProblem` is a type alias for `MinimalSolveProblem{FundMat{T}, 7, T, ...}`
— a 2D-2D correspondence problem with the 7-point minimal solver and, for LO
refinement, Levenberg–Marquardt on the 7-dof Bartoli–Sturm chart about the
current `F` (signed Sampson residual).

```@docs
FundMatProblem
```

## Fitting Pipeline

```@docs
find_fundmat
```
