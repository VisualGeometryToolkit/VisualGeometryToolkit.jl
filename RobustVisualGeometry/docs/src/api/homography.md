# Homography

```@meta
CurrentModule = RobustVisualGeometry
```

## RANSAC Problem

`HomographyProblem` is a type alias for `MinimalSolveProblem{HomographyMat{T}, 4, T, ...}`
— a 2D-2D correspondence problem with the 4-point minimal solver and, for LO
refinement, Levenberg–Marquardt on the minimal 8-dof sl(3) chart about the
current `H` (Sampson-whitened transfer error).

```@docs
HomographyProblem
```

## Fitting Pipeline

```@docs
find_homography
```
