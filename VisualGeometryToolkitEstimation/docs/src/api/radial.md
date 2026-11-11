# Radial and Lens Calibration

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

These estimators calibrate a camera with lens distortion from correspondences
between planar calibration boards and their images.

## Radial homography

```@docs
RadialHomographyProblem
find_radial_homography
refine_radial_homography
```

## Radial camera

```@docs
RadialCameraProblem
find_radial_camera
refine_radial_camera
fit_radial_camera
```

## Division camera

`fit_division_camera` fits a camera with a division lens model to several
board views at once. `division_camera_step` is the model update it repeats
under its current weights.

```@docs
fit_division_camera
division_camera_step
DivisionCameraState
DivisionCameraError
RadialLift
division_coeffs(::DivisionCameraState)
board_sq_residuals
```

## Camera and poses

`refine_camera_and_poses` refines a camera — its intrinsics and lens, on
VisualGeometryToolkitCore's `ParameterManifold` — jointly with one pose per
view (`LeftRigidManifold6`) against the reprojection error of 3D → 2D
correspondences, from a guess of both: the bundle of a calibration with a known
target, for any lens the camera's family holds (a fisheye's rays past 90° are
ordinary data). `free = ()` holds the camera and refines the poses alone. A
SplineLens camera refines its backward controls instead (K frozen, the lens
axis held on ẑ), by implicit differentiation of its projection solve, with a
scale-free third-difference penalty over its lattice whose weight generalized
cross-validation chooses by default (`penalty = :gcv`; a number fixes it, `0`
turns it off); a forward spline is refitted after. With several rigid boards,
`refine_camera_and_poses(camera, poses, board_poses, views)` adds one pose per
board (board → world) to the same bundle, the correspondences keyed by
`(view, board)`; each connected group of views and boards keeps its gauge by a
held pose (its first board by default, `held_boards`/`held_views`), exactly.

```@docs
refine_camera_and_poses
CameraPosesRefineMetadata
```

## Penalty weights: normalization and generalized cross-validation

One convention for a penalty's weight across the stack: `λ = β · energy / N / m̄`
(VisualGeometryToolkitCore's `penalty_scale`), β relative to the data's energy
per parameter and to the penalty operator's mean positive eigenvalue `m̄`
(Core's `mean_positive_eigenvalue`), so β = 1 weighs the penalty like the data.
`gcv_select` chooses β by generalized cross-validation from a score of λ — a
coarse scan of log₁₀ β, then Brent's method — the score the closed-form curve
of one penalized least squares (`gcv_curve`: every λ from one generalized
eigendecomposition). A nonlinear fit evaluates it on its linearization, as the
SplineLens refine does (one solve at β = 1, the curve, one final solve).
Public, not exported.

```@docs
VisualGeometryToolkitEstimation.gcv_curve
VisualGeometryToolkitEstimation.gcv_select
VisualGeometryToolkitEstimation.gcv_score
VisualGeometryToolkitEstimation.check_penalty_weight
```

## Camera-model regression

`fit_camera(T, source, region; …)` regresses one camera model of type `T` onto a
source over a region: a lens family, a `SplineLens` (a `CameraModel` whose lens
is the backward spline of rays over the image) or a `NoncentralSplineModel` (its
direction and point splines). `pspline_grid(T, source, region, grid_size)` is the
spline targets' one penalized solve on a given lattice, `grid_size = (columns,
rows)` in Core's order; `fit_camera` repeats it at each spacing it tries.

```@docs
fit_camera
pspline_grid
CameraFitMetadata
```
