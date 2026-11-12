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
under its current weights. The robust loop is the stack's shared one: the
posterior of a `StudentTUniformScoring` (the `weighting` keyword) on the
posterior-weight iteration of a `MixtureEmConfig` (the `mixture_em` keyword),
over the stack as a `DivisionCameraProblem`.

```@docs
fit_division_camera
DivisionCameraMetadata
DivisionCameraProblem
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
ordinary data). `free = ()` holds the camera and refines the poses alone. Its
strategy is the refine contract's `optimization_strategy` — `Triggs{WelschShape}()`
by default, run from the start with no least-squares stage, at the caller's
`scale = σ` (a refine estimates no scale) or `width = w`; `nothing` is plain least
squares. A
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

Every bundle — this one, [`refine_robot_world_hand_eye`](@ref) and a downstream package's
(PanoramaStitching's rotation bundle) — reads its robust loss from its keywords through
one function:

```@docs
bundle_loss
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

Three verbs regress one camera model onto a source camera over a region of its sensor, by the stack's naming rule
(a fit has no guess, a refine needs one):

- `fit_camera(T, source, region; …)` is LINEAR: one least-squares solve. For a lens family `L` it solves the focal
  rescale and the family's coefficients from the samples' radial relation where the family has a linear form (a lift:
  `Division`, `Scaramuzza`; a fixed projection with an even radial factor: Kannala–Brandt, the polynomial and rational
  OpenCV models, Fisheye624), and refuses a family with none (`DoubleSphere`, `Unified`, `ExtendedUnified`, a `Tilt` or
  `FieldOfView` term) with an `ArgumentError` naming the two verbs below. Its targets `SplineLens` (a `CameraModel`
  whose lens is the backward spline of rays over the image) and `NoncentralSplineModel` (its direction and point
  splines) are penalized linear least squares. `pspline_grid(T, source, region, grid_size)` is the spline targets' one
  penalized solve on a given lattice, `grid_size = (columns, rows)` in Core's order; `fit_camera` repeats it at each
  spacing it tries.
- `refine_camera(guess, source, region; …)` is the chart LM from a guess to the exact pixel least-squares minimum of
  the guess's family (for a SplineLens source also its principal point, aspect and axis).
- `approximate_camera(L, source, region; …)` is the two: `refine_camera(estimate(fit_camera(L, source, region)), source, region)`,
  a family with no linear form starting from its anchor.

A SplineLens camera's `K` is a gauge, so its ẑ and principal point are wherever its frame put them.
`optical_axis(camera::BoundedCamera)` is the one `approximate_camera` call that finds the lens's optical axis: the closest
radially symmetric camera over the camera's own bound (a circular fisheye's image circle keeps its corners out), with its
principal point and its axis free. `fit_camera(SplineLens, …)` and the SplineLens
`refine_camera_and_poses` turn the camera onto that axis first (Core's `axis_gauged`).

```@docs
fit_camera
refine_camera
approximate_camera
optical_axis
pspline_grid
CameraFitMetadata
ViewAngleLimitError
```
