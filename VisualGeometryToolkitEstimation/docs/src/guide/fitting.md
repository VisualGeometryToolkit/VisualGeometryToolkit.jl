# Fitting Algorithms

The `find_*` functions are end-to-end RANSAC pipelines: they construct the
problem, wire up scoring and local optimization, and return a
`RansacEstimate` (a `FitResult{M, RansacMetadata}`). The `fit_*` prefix is
kept for fits that are not a RANSAC search — iterative, reweighted or
repeated (`fit_division_camera` and its model-block update `division_camera_step`,
`fit_radial_camera`, `fit_lens`, `fit_robot_world_hand_eye`, `fit_line`) — and
`refine_*` for polishing a given model (`refine_robot_world_hand_eye`,
`refine_absolute_pose`). One-step
algebraic algorithms keep their algorithm's name and live in
VisualGeometryToolkitCore (`homography_dlt`, `kabsch`, `tls_line`, …).

Every `find_*` shares one keyword surface (documented on
[`find_homography`](@ref)) and the same core defaults, read from its class's
`[pipelines.<class>]` table in `configs/default.toml`: VisualGeometryToolkitCore's
`MsacScoring(problem; threshold)` and `Irls{TLSShape}()` local optimization
following the scorer's threshold. The default thresholds are conventions, not
tuned values:

| pipeline | `threshold` |
|:--|:--|
| `find_homography`, `find_fundmat`, `find_radial_homography`, `find_radial_camera` | 3 px |
| `find_essential` | 1 px |
| `find_absolute_pose` | 8 px |
| `find_hand_eye` | `0.01·L` (a fraction of the characteristic length) |

`scoring = …` replaces the scorer (then `threshold` is an `ArgumentError`) and
`local_optimization = …` the strategy; an `outlier_halfwidth` keyword is
refused — it is a marginal scorer's parameter and goes into that scorer.

## Homography

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: csponds, estimate
using StaticArrays

src = [SA[100.0, 200.0], ...]
dst = [SA[112.0, 195.0], ...]

result = find_homography(csponds(src, dst); threshold = 2.0)
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

`threshold` is in PIXELS (default 1.0): the residual is whitened by the
back-projection Jacobians. A threshold — or a supplied scorer's outlier
half-width — below 0.06 px, VisualGeometryToolkitFeatures' sub-pixel refinement
bound, is refused as a likely calibrated leftover, by `find_essential`,
`ransac` and `locally_optimize` alike. See
[`find_essential`](@ref) on the [Essential Matrix](../api/essential.md) API
page for the full keyword surface.

## Fundamental Matrix

```julia
result = find_fundmat(csponds(src, dst); threshold = 2.0)
F = estimate(result)
```

See [`find_fundmat`](@ref) on the [Fundamental Matrix](../api/fundmat.md) API
page for the full keyword surface.

## Absolute Pose (3D-2D)

```julia
using VisualGeometryToolkitCore: CameraModel, csponds, estimate, px
using StaticArrays: SA

camera = CameraModel(f * px, SA[cx * px, cy * px])
result = find_absolute_pose(camera, csponds(world_points_3d, image_points_2d);
                  threshold = 4.0)
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
using RANSACScoringDoneRight: StudentTUniformScoring
using LocalOptimizationDoneRight: HQ, WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

cs       = csponds(src, dst)
problem  = HomographyProblem(cs)
result   = find_homography(cs;
                  scoring = StudentTUniformScoring(problem, 20.0),
                  local_optimization = HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY))

# or the loop itself
result   = ransac(problem, StudentTUniformScoring(problem, 20.0);
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
