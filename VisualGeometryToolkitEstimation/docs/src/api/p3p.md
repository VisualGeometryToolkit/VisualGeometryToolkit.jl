# P3P

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problem

```@docs
AbsolutePoseProblem
```

## Fitting Pipeline

```@docs
find_absolute_pose
```

## Result Types

**Model type** — [`Extrinsics3`](@ref), VisualGeometryToolkitCore's world→camera rigid
pose (import it from VisualGeometryToolkitCore). It replaces the retired `Pose3`
alias. Do not confuse it with RVG's `Rigid3`, the *unlabeled* alias the
hand-eye problem uses — a different type.

```@docs
Extrinsics3
```

`model_type(::AbsolutePoseProblem) = Extrinsics3` and `solve` returns `FixedModels{4, Extrinsics3} | nothing`
(up to 4 P3P solutions).

## Local Optimization

`find_absolute_pose` deploys `Irls{TLSShape}()` (least squares on the
inliers, at the default MSAC scorer's 8 px threshold;
`[pipelines.absolute_pose]` in `configs/default.toml`). Any other strategy
composes, LocalOptimizationDoneRight's half-quadratic lifts on the 6-dof pose
chart among them. The per-iteration M-step for
`AbsolutePoseProblem` is a Levenberg–Marquardt refit on the 6-dof left-perturbation
chart about the current pose (VGC's `LeftRigidManifold6`: `R = exp(δω)·R₀`,
`t = exp(δω)·t₀ + δt`): it minimizes
the weighted calibrated reprojection error from
`constraint_with_jacobians(::Extrinsics3, X, ray)` in VisualGeometryToolkitCore (the
raw-pixel error for a distorting lens) with an analytic Jacobian (the chart's
closed-form `action_jacobian`), running VisualGeometryToolkitCore's chart LM
(FastLevenbergMarquardt's `lmsolve!`) on a `ManifoldLmWorkspace{6}`.

## Correspondence Infrastructure

`AbstractCspondProblem` is the abstract supertype for RANSAC problems based on
point correspondences (homographies, fundamental matrices, P3P).
