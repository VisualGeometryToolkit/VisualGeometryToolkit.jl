# Homography

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problem

`HomographyProblem` is a type alias for `MinimalSolveProblem{HomographyMat{T}, 4, T, ...}`
— a 2D-2D correspondence problem with the 4-point minimal solver and, for LO
refinement, Levenberg–Marquardt on the minimal 8-dof sl(3) chart about the
current `H` (Sampson-whitened transfer error).

```@docs
HomographyProblem
```

## Find and refine

The problem is the `f` of both:

```julia
f = HomographyProblem(cs)
r = solve(EstimationProblem(f, (; extent)), Ransac(); rng)     # the find; the extent is optional
h = solve(RefineProblem(f, r.u), LevenbergMarquardt(); scale = r)   # the refine, at the find's scale
```

[`Ransac`](@ref) documents the find's configuration (its class default is
`[homography_problem.ransac]`: MSAC at 3 px, `Irls{TLSShape}()`); VisualGeometryToolkitCore's
`LevenbergMarquardt` the refine's (`[homography_problem.levenberg_marquardt]`: `Triggs{WelschShape}()`). The
extent validates a fitted `H` over a world-plane rectangle (`BoundedProblem`): a
bow-tie or orientation-reversing `H` is refused. Physical (unitful) plane points
are refused at construction: a homography cannot record their unit.
