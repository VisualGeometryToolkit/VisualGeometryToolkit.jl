# Radial and Lens Calibration

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

These estimators calibrate a camera with lens distortion from correspondences
between planar calibration boards and their images.

## Radial homography

`solve(EstimationProblem(RadialHomographyProblem(cs)), Ransac(); rng)` is its find
and `solve(RefineProblem(RadialHomographyProblem(cs), H₀), LevenbergMarquardt(); scale | width)`
its refine.

```@docs
RadialHomographyProblem
```

## Radial camera

`solve(EstimationProblem(RadialCameraProblem(cs; center)), Ransac(); rng)` is the
find of one board's radial camera (distortion-free verification about a known
principal point), `solve(RefineProblem(RadialCameraProblem(cs; center), P₀),
LevenbergMarquardt(); scale | width)` its refine, and
`solve(EstimationProblem(RadialCamera, cs, (; center)), RadialCameraFit(; iterations))`
its whitened linear fit on validated inliers.

```@docs
RadialCameraProblem
RadialCameraFit
```

## Division camera

[`DivisionCameraFit`](@ref) fits a camera with a division lens model to several
board views at once, from each board's radial camera and its validated inliers:
`solve(EstimationProblem(CameraModel, (radial_cameras, views), (; image_size)), DivisionCameraFit())`.
The robust loop is the stack's shared one: the posterior of a
`StudentTUniformScoring` (the `weighting` field) on the posterior-weight
iteration of a `MixtureEmConfig` (the `mixture_em` field), over the stack as a
`DivisionCameraProblem`.

```@docs
DivisionCameraFit
DivisionCameraMetadata
DivisionCameraProblem
DivisionCameraState
DivisionCameraError
RadialLift
division_coeffs(::DivisionCameraState)
board_sq_residuals
```

## Camera and poses

[`CameraPosesProblem`](@ref) is the `f` of the refine of a camera — its intrinsics
and lens, on VisualGeometryToolkitCore's `ParameterManifold` — jointly with one pose
per view (`LeftRigidManifold6`) against the reprojection error of 3D → 2D
correspondences, from a guess of both:
`solve(RefineProblem(CameraPosesProblem(views; fixed), (; camera, poses)), LevenbergMarquardt(); scale)`.
It is the bundle of a calibration with a known target, for any lens the camera's
family holds (a fisheye's rays past 90° are ordinary data). Naming every parameter
in `fixed` holds the camera and refines the poses alone. Its strategy is the
refine's `optimization_strategy` — `Triggs{WelschShape}()` by default
(`[camera_poses_problem]`), run from the start with no least-squares stage, at the
caller's `scale = σ` (a refine estimates no scale) or `width = w`; `nothing` is plain
least squares. A SplineLens camera is refined only by back-projection (the
[`CameraRegression`](@ref) of a `SplineLens` target, [`PSpline`](@ref)), or held:
here it takes `fixed = (:controls,)`, the poses refined through its whole `project`
under the bundle's strategy and the camera returned as given (its forward spline
kept); the default `fixed = ()` and a `penalty` are refused for it. With several
rigid boards, the guess `(; camera, poses, board_poses)` adds one pose per board
(board → world) to the same bundle, the correspondences keyed by `(view, board)`;
each connected group of views and boards keeps its gauge by a held pose (its first
board by default, `held_boards`/`held_views`), exactly.

The guess `(; camera, views)`, `views` a [`RotationCenterViews`](@ref), refines
the views of a camera turned about one fixed point (a tripod head) under one
rigid motion model, one rotation per view and a shared centre and offset, the
camera held. Its residual is each dot's BEARING for every camera: the dot
back-projected once by the camera, the tangent angle to the predicted bearing
whitened into pixels by the back-projection Jacobian (the
`BearingPoseProblem`'s); the camera is never projected inside the solve, and a
dot it does not back-project, or whose Jacobian cannot whiten its residual
([`pixel_whitening_defect`](@ref)), makes no row and is counted in the metadata
(its `n_without_bearing` and `without_bearing` fields). Its `fixed` holds the
whole camera: `fixed = held_parameters(camera)`. The problem's `residual` setting
is `BearingResidual()` by default; `ReprojectionResidual()` adds a second stage
from the bearing estimate on the exact pixel reprojection error (a SplineLens
camera projected by inverting its backward spline), a candidate update that leaves
a prediction undefined rejected by the LM, the dots that do not project at the
bearing estimate counted (the `without_projection` field).

```@docs
CameraPosesProblem
CameraPosesRefineMetadata
RotationCenterViews
rotation_center_pose
held_parameters
RotationCenterRefineMetadata
AbstractViewResidual
BearingResidual
ReprojectionResidual
```

Every bundle — this one, the robot–world refine ([`RobotWorldHandEyeProblem`](@ref))
and a downstream package's (PanoramaStitching's rotation bundle) — reads its
robust loss from the refine's settings through one function, VisualGeometryToolkitCore's
[`bundle_loss`](@extref VisualGeometryToolkitCore.bundle_loss).

## Penalty weights: normalization and generalized cross-validation

One convention for a penalty's weight across the stack: `λ = β · energy / N / m̄`
(VisualGeometryToolkitCore's `penalty_scale`), β relative to the data's energy
per parameter and to the penalty operator's mean positive eigenvalue `m̄`
(Core's `mean_positive_eigenvalue`), so β = 1 weighs the penalty like the data.
`solve(EstimationProblem(Hyperparameter, score, (; scale)), GeneralizedCrossValidation())`
chooses β by generalized cross-validation from a score of λ — a coarse scan of
log₁₀ β, then Brent's method — the score the closed-form curve of one penalized
least squares (`gcv_curve`: every λ from one generalized eigendecomposition). A
nonlinear fit evaluates it on its linearization (one solve at β = 1, the curve,
one final solve). The curve, the score and the weight's check are public, not
exported.

```@docs
Hyperparameter
GeneralizedCrossValidation
VisualGeometryToolkitEstimation.gcv_curve
VisualGeometryToolkitEstimation.gcv_score
VisualGeometryToolkitEstimation.check_penalty_weight
```

## Camera-model regression

One problem, [`CameraRegressionProblem`](@ref), regresses one camera model onto a
source camera over a region of its sensor; the algorithm states the kind (a fit
has no guess, a refine needs one):

- `solve(EstimationProblem(CameraRegressionProblem(T, source, region)), CameraRegression())`
  is LINEAR: one least-squares solve. For a lens family `L` it solves the focal
  rescale and the family's coefficients from the samples' radial relation where the
  family has a linear form (a lift: `Division`, `Scaramuzza`; a fixed projection with
  an even radial factor: Kannala–Brandt, the polynomial and rational OpenCV models,
  Fisheye624), and refuses a family with none (`DoubleSphere`, `Unified`,
  `ExtendedUnified`, a `Tilt` or `FieldOfView` term) with an `ArgumentError` naming
  the refine below. Its targets `SplineLens` (a `CameraModel` whose lens is the
  backward spline of rays over the image) and `NoncentralSplineModel` (its direction
  and point splines) are penalized linear least squares; a SplineLens SOURCE is
  instead resampled exactly (Core's `extend_lattice` over the region,
  `subdivide_lattice` to the `lattice_spacing` cell, then the turn onto its optical
  axis), its extension and fold report recorded; `region ⊆ valid_domain` of the
  result always holds, a region on the source's lattice edge included (each edge
  exact where a float lands on it, else one step of K′'s map outward).
  [`PSpline`](@ref)`(; grid_size)` is the spline targets' one penalized solve on a
  given lattice, `grid_size = (columns, rows)` in Core's order; `CameraRegression`
  repeats it at each spacing it tries.
- `solve(RefineProblem(CameraRegressionProblem(source, region), guess), LevenbergMarquardt())`
  is the chart LM from a guess to the exact pixel least-squares minimum of the
  guess's family (for a SplineLens source also its principal point, aspect and axis).
- The linear fit then its refine is the two composed, a family with no linear form
  starting from its anchor. A bounded source is fitted over
  `region ∩ imaged_region(source)`: no sample where the camera images nothing.

A SplineLens camera's `K` is a gauge, so its ẑ and principal point are wherever its
frame put them. `solve(EstimationProblem(CameraRegressionProblem(camera[, region])), OpticalAxis())`
finds the lens's optical axis: the closest radially symmetric camera over the
camera's own bound (a circular fisheye's image circle keeps its corners out), or
over a given image region within it (a target's rectangle: the pixels whose rays
are measured), with its principal point and its axis free. It fits order by order
(1 → 2 → 3 → 4 radial terms, each refine from the last order's camera), and keeps
the highest order that holds by its own validity gate — converged, its regression
stationary at the estimate (on the axis), and not folding inside the region
(`r.metadata.family.fitted_terms`). The SplineLens `CameraRegression` and the
SplineLens camera-and-poses refine turn the camera onto that axis first (Core's
`axis_gauged`).

`solve(EstimationProblem(CameraRegressionProblem(camera)), ForwardSpline(; tolerance))`
gives a SplineLens camera its forward spline and keeps everything else — its
backward, K and frame — so poses estimated against it stay valid; with a forward,
`project` evaluates the spline and runs no projection solve. It runs the stack's one
forward fit (the fit `CameraRegression(; forward = true)` and the SplineLens
camera-and-poses refine's refit run).

```@docs
CameraRegressionProblem
AbstractImageMeasure
ImageMeasure
SolidAngleMeasure
UniformMeasure
CameraRegression
PSpline
ForwardSpline
OpticalAxis
CameraFitMetadata
ViewAngleLimitError
```
