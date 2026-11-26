# P3P

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problem

```@docs
AbsolutePoseProblem
```

## Find and refine

`solve(EstimationProblem(AbsolutePoseProblem(camera, X .=> u), (; extent)), Ransac(); rng)`
is the find (the extent optional: a pose that fails cheirality or images the
board as a non-simple or non-convex quad is refused) and
`solve(RefineProblem(AbsolutePoseProblem(camera, X .=> u), pose), LevenbergMarquardt(); scale | width)`
the refine, from a guess that comes back under its own frame labels. World points
may be physical (lengths): stripped in metres at construction, the pose comes
back in metres; the extent of a physical problem is a physical rectangle.

## Result Types

**Model type** — [`CameraExtrinsics`](@extref VisualGeometryToolkitCore.CameraExtrinsics),
VisualGeometryToolkitCore's world→camera rigid pose (import it from
VisualGeometryToolkitCore). Do not confuse it with the *unlabeled* plain map the hand-eye problem uses — a
different type.

`model_type(::AbsolutePoseProblem) = `CameraExtrinsics`` and its `sample_models` returns
`FixedModels{4, `CameraExtrinsics`}` (up to 4 P3P solutions).

## Local Optimization

The class default deploys `Irls{TLSShape}()` (least squares on the
inliers, at the default MSAC scorer's 8 px threshold;
`[pipelines.absolute_pose]` in `configs/default.toml`). Any other strategy
composes, LocalOptimizationDoneRight's half-quadratic lifts on the 6-dof pose
chart among them. The per-iteration M-step for
`AbsolutePoseProblem` is a Levenberg–Marquardt refit on the 6-dof left-perturbation
chart about the current pose (VGC's `LeftRigidManifold6`: `R = exp(δω)·R₀`,
`t = exp(δω)·t₀ + δt`): it minimizes
the weighted calibrated reprojection error from
`constraint_with_jacobians(::`CameraExtrinsics`, X, ray)` in VisualGeometryToolkitCore (the
raw-pixel error for a distorting lens) with an analytic Jacobian (the chart's
closed-form `action_jacobian`), running VisualGeometryToolkitCore's chart LM
(the summary-free `LevenbergMarquardt` refine held in the problem's `RefitSlot`).

## Correspondence Infrastructure

`AbstractCorrespondenceProblem` is the abstract supertype for RANSAC problems based on
point correspondences (homographies, fundamental matrices, P3P).
