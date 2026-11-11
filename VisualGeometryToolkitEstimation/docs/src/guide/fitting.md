# Fitting Algorithms

The `find_*` functions are end-to-end RANSAC pipelines: they construct the
problem, wire up scoring and local optimization, and return a
`RansacEstimate` (a `FitResult{M, RansacMetadata}`). The `fit_*` prefix is
kept for fits that are not a RANSAC search — iterative, reweighted or
repeated (`fit_division_camera` and its model-block update `division_camera_step`,
`fit_radial_camera`, `fit_camera`, `fit_robot_world_hand_eye`, `fit_line`,
`fit_rotation`, `fit_linear`, `fit_rotation_averaging`, `fit_rigid_alignment`) — and
`refine_*` for polishing a given model (one per `find_*`, and the bundles
`refine_robot_world_hand_eye` and `refine_camera_and_poses`). One-step
algebraic algorithms keep their algorithm's name and live in
VisualGeometryToolkitCore (`homography_dlt`, `kabsch`, `tls_line`, …).

Every `find_*` shares one keyword surface (documented on
[`find_homography`](@ref)) and the same core defaults, read from its class's
`[pipelines.<class>]` table in `configs/default.toml`: its `scoring` table
(read by [`scoring_from_toml`](@ref) — every class's is MSAC,
`{ type = "msac", threshold = … }`) and `Irls{TLSShape}()` local optimization
following the scorer's decision point. The default MSAC thresholds are
conventions, not tuned values:

| pipeline | MSAC `threshold` |
|:--|:--|
| `find_homography`, `find_fundmat`, `find_radial_homography`, `find_radial_camera` | 3 px |
| `find_essential` | 1 px |
| `find_absolute_pose` | 8 px |
| `find_hand_eye` | `0.01·L` (a fraction of the characteristic length) |

`scoring = …` replaces the scorer and `local_optimization = …` the strategy. A
bare `threshold` or `outlier_halfwidth` keyword is refused: a width is its
scorer's (a threshold scorer's `threshold`, the Student-t + uniform scorer's
`outlier_halfwidth`) and goes into that scorer.

## `find_*` and `refine_*`: the same local optimization, with or without the search

A `find_*` is sampling plus a refine: RANSAC draws minimal samples, scores each
hypothesis, and runs the class's local optimization from the hypotheses it
keeps. The matching `refine_*` is that local optimization alone, run once from
a model the caller gives — `refine_homography(H, pairs)`,
`refine_circle(circle, points)`, `refine_absolute_pose(pose, camera, pairs)`, …,
the model first, then the data as the `find_*` takes them. The given model is
scored exactly as a hypothesis is; its consensus set under the scorer is the
one a consensus-anchored strategy reads its scale from; the strategy runs from
it; the refined model is re-scored and returned as a `RansacEstimate`, or the
given model if the refined one fails the problem's consensus test.

One `refine_*` has no `find_*`: `refine_absolute_pose(pose, bearings)` on a
[`BearingPoseProblem`](@ref) — a camera given by its rays, possibly with
per-ray origins (a non-central camera, for which there is no minimal solver).
Its defaults (`[pipelines.bearing_pose]`) are least squares over every
correspondence.

Both read the same `[pipelines.<class>]` defaults and share the strategy
keywords — `scoring`, `local_optimization`, `workspace` — so any
strategy plugs into either: `Irls{K}`, `Triggs{K}`, `NoLocalOptimization()`,
LocalOptimizationDoneRight's half-quadratic lifts and GNC. A strategy a
problem cannot host (`Triggs` or a lift on the hand-eye problem, which has no
chart) is refused by both with the same error. The `find_*` sampling keywords
(`max_trials`, `min_trials`, `config`, `rng`) are not `refine_*` keywords, and
neither is a `max_iter`: an iteration cap is the strategy's own (`Irls`'s
`max_iter`; the chart LM's `[chart_lm]` cap). The bundles
(`refine_robot_world_hand_eye`, `refine_camera_and_poses`) are not RANSAC
problems and keep their own LM and keywords; `refine_camera_and_poses` takes its
robust strategy under `find_*`'s keyword, `local_optimization` (`Irls{Shape}` or
`Triggs{Shape}` at a `scale_multiple`).

## Robust fits: one strategy, one fixed scale

The reweighted fits — `fit_linear`, `fit_rotation`, `fit_rotation_averaging`,
`fit_rigid_alignment` and `fit_robot_world_hand_eye` — keep their own
objective (a least squares, a Kabsch, the staged hand-eye solve). Their only
robustness is `optimization_strategy = Irls{K}(…)`, an M-estimator weighting
at ONE loss width, set from the call's scale input and never re-estimated;
`optimization_strategy = nothing` is the plain least squares. The scale input
is exactly one of:

- `scale = σ` — the residuals' noise scale (`ScaleGiven`);
- `scale = result` — an upstream robust result's `noise_scale(result)`
  (`ScaleUpstream`), e.g. a `find_*`'s consensus scale handed down;
- `width = w` — the loss width itself (the result then records `ScaleNone`).

The width is `w`, or `c · σ` with `c` the strategy's efficiency constant, by
default `efficiency_constant(K, d)` at the fit's residual codimension `d`
(`fit_linear` 1, `fit_rotation` 2, `fit_rotation_averaging` and
`fit_rigid_alignment` 3, the hand-eye's whitened 6-vector 6). Neither, both,
a scale given to a least-squares fit, a refine's `Triggs`, or a shape with no
efficiency constant and no width is an `ArgumentError`. The result records
`scale`/`scale_source` (`noise_scale(result)` hands it on), its `loss_width`,
and per-datum `weights(result)`: the loss weights `ρ′` normalized to 1
(`IrlsWeights`; `inlier_mask(result, cut)` needs a cut), or lazy unit weights
(`BinaryWeights`) for least squares.

```julia
R     = fit_rotation(cs; scale = find_rotation(cs))               # Huber (the default) at 1.5011 × the consensus scale
align = fit_rigid_alignment(centres .=> truth; optimization_strategy = Irls{WelschShape}(), scale = 0.005)
```

## Homography

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: csponds, estimate, MsacScoring
using StaticArrays

src = [SA[100.0, 200.0], ...]
dst = [SA[112.0, 195.0], ...]

result = find_homography(csponds(src, dst); scoring = MsacScoring(HomographyProblem; threshold = 2.0))
H       = estimate(result)         # 3×3 homography matrix
inliers = result.inlier_mask
```

See [`find_homography`](@ref) on the [Homography](../api/homography.md) API page
for the full keyword surface.

## Essential Matrix (calibrated)

```julia
using VisualGeometryToolkitCore: CameraModel, csponds, estimate, px
using StaticArrays: SA

# PIXEL correspondences and the two cameras that took them; each view is
# back-projected to unit bearings by its own camera inside the pipeline.
cam1 = CameraModel(800.0px, SA[320.0px, 240.0px])
cam2 = CameraModel(1150.0px, SA[300.0px, 260.0px])
result = find_essential(cam1, cam2, csponds(src_pixels, dst_pixels))
E = estimate(result)
```

The scorer's width is in PIXELS (default MSAC at 1.0): the residual is whitened
by the back-projection Jacobians. A threshold scorer's threshold — or the
Student-t scorer's outlier half-width — below 0.06 px, VisualGeometryToolkitFeatures' sub-pixel refinement
bound, is refused as a likely calibrated leftover, by `find_essential`,
`ransac` and `locally_optimize` alike. See
[`find_essential`](@ref) on the [Essential Matrix](../api/essential.md) API
page for the full keyword surface.

## Fundamental Matrix

```julia
result = find_fundmat(csponds(src, dst); scoring = MsacScoring(FundMatProblem; threshold = 2.0))
F = estimate(result)
```

See [`find_fundmat`](@ref) on the [Fundamental Matrix](../api/fundmat.md) API
page for the full keyword surface.

## Absolute Pose (3D-2D)

```julia
using VisualGeometryToolkitCore: CameraModel, csponds, estimate, px, MsacScoring
using StaticArrays: SA

camera = CameraModel(f * px, SA[cx * px, cy * px])
result = find_absolute_pose(camera, csponds(world_points_3d, image_points_2d);
                  scoring = MsacScoring(AbsolutePoseProblem; threshold = 4.0))
pose = estimate(result)    # EuclideanMap{3,Float64} (rotation + translation)
```

See [`find_absolute_pose`](@ref) on the [P3P](../api/p3p.md) API page for the full
keyword surface.

## Hand-Eye Calibration (AX = XB)

Recover the rigid transform between a tracked rig and a rigidly-attached
calibration pattern from synchronized pose pairs `(A_i, B_i)`. Inputs are
two `EuclideanMap{3,Float64}` streams: `A_i = T_WR` (rig in world, from motion capture)
and `B_i = T_CP` (pattern in camera, from PnP).

```julia
using VisualGeometryToolkitCore: EuclideanMap, estimate

A = EuclideanMap{3,Float64}[ ... ]   # T_WR per frame (e.g. mocap)
B = EuclideanMap{3,Float64}[ ... ]   # T_CP per frame (e.g. PnP)

result = find_hand_eye(A .=> B)     # one A_i => B_i per frame, time order
X       = estimate(result)         # T_RP, the rigid pattern-in-rig transform
inliers = result.inlier_mask
```

Unlike the other `find_*` pipelines, its inputs are pose pairs rather than
correspondences. The nuisance camera-in-world transform `Y = T_WC` drops out
of the AX = XB formulation and is not returned.

See [`find_hand_eye`](@ref) on the [Hand-Eye Calibration](../api/hand_eye.md)
API page for the full keyword surface and
[Rig-to-Pattern Hand–Eye Calibration](hand-eye.md) for the formulation.

## Building the pipeline manually

To swap the scorer or the local optimization, hand them to the pipeline or
build the pieces directly. A scorer from RANSACScoringDoneRight and a strategy
from LocalOptimizationDoneRight compose like the core ones:

```julia
using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior
using LocalOptimizationDoneRight: HQ, WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

cs       = csponds(src, dst)
problem  = HomographyProblem(cs)
result   = find_homography(cs;
                  scoring = StudentTUniformScoring(problem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
                  local_optimization = HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY))

# or the loop itself
result   = ransac(problem, StudentTUniformScoring(problem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior());
                  local_optimization = HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY))
```

See the [RANSAC](ransac.md) page for the full configuration surface (samplers,
adaptive trial counts, width modes, etc.).

## Line Fitting

The line problems (`LineFittingProblem`, `InhomLineFittingProblem`) plug into
the same `ransac()` entry point; there is no dedicated `find_*` pipeline for
them. The unexported `fit_line(points::AbstractVector{<:Uncertain{<:Point2}})`
is the line problem's local-optimization refit, not a RANSAC pipeline: an
iterative covariance-weighted orthogonal fit that seeds its normal from an
unweighted scatter, then reweights each point by its variance along the normal
(σᵢ² = n⊤Σᵢn) and refits. It returns a `FitResult` whose estimate is an
`Uncertain{Line2D}` carrying the (φ, d) covariance. The one-step unweighted
total-least-squares line is VisualGeometryToolkitCore's `tls_line`.

```julia
using VisualGeometryToolkitCore: MsacScoring
problem  = LineFittingProblem(uncertain_points)
result   = ransac(problem, MsacScoring(problem; threshold = 1.0))
```
