# Hand-Eye / Robot-World Calibration

```@meta
CurrentModule = RobustVisualGeometry
```

## RANSAC Problem

```@docs
HandEyeProblem
```

## Fitting Pipeline

```@docs
find_hand_eye
```

## Result Type

`find_hand_eye` returns a `FitResult{Rigid3, RansacMetadata}`
(`RansacEstimate{Rigid3}`, where `Rigid3 = EuclideanMap{3,Float64,...}`). The `.estimate` field is the recovered rigid
transform `X = T_RP` (pattern in rig). The nuisance camera-in-world
transform `Y = T_WC` is **not** returned — it drops out of the AX = XB
formulation. The remaining attributes (`.inlier_mask`, `.residuals`,
`.weights`, `.iterations` — for RANSAC the number of trials, hypotheses
drawn — `.converged`, …) match the standard `RansacEstimate` surface.

## Formulation

The pipeline implements the classical robot-world / hand-eye problem,
eliminated via the relative-motion ratio:

```
(A_j⁻¹ A_i) · X  =  X · (B_j⁻¹ B_i)        ⇔   A_ij · X = X · B_ij
```

`A_i = T_WR` (rig in world, e.g. from motion capture) and `B_i = T_CP`
(pattern in camera, e.g. from PnP). With `N` observations, `motion_pairs
= :consecutive` (default) builds `N - 1` relative motions from adjacent
frames; `:all` uses every `(i, j)` with `i < j`.

The minimal sample is **2 motion pairs** (Tsai–Lenz solvability condition:
two non-parallel rotation axes). The minimal solver is the closed-form
Tsai–Lenz algorithm (modified-Rodrigues LLS for rotation, then a 3×3
linear system for translation). The per-pair residual is a decoupled
6-vector

```
g_i(X) = [ L · Log(R_A R_X R_B^T R_X^T) ;  R_A t_X + t_A − R_X t_B − t_X ] ∈ ℝ⁶
```

where `L = char_length` (default: median rig-translation per motion)
balances rotation (radians) and translation (length) units.

## Local Optimization

Hand-eye uses `Irls{TLSShape}()` LO out of the box (least squares on the
inliers at the default MSAC scorer's threshold, `0.01·L`). The per-iteration M-step
for `HandEyeProblem` is a Levenberg–Marquardt refit on the 6-DOF decoupled
chart about the current `X` — `R_X = exp(δω)·R₀` plus an additive update
`t_X = t₀ + δt` (VisualGeometryCore's `RigidExpManifold6`, the chart of
`constraint_with_jacobians`' `Gθ` and of the solver covariance) — minimizing
`Σᵢ wᵢ ‖gᵢ‖²` over the stacked 6-vectors. It runs VisualGeometryCore's chart LM
(FastLevenbergMarquardt's `lmsolve!`) with a ForwardDiff Jacobian of the explicit residual: that
`Gθ` is only the near-optimum approximation of `∂g/∂δ`, so the refit does not
use it. Each RANSAC hypothesis is Tsai–Lenz fitted on its own two motions by
the same refit (`fit_hypothesis`), the estimator the solver covariance
describes.

## Identifiability

The problem requires **at least two relative motions with non-parallel
rotation axes** (Tsai & Lenz 1989; Chen 1991). With a smooth trajectory
of N ≫ 2 poses, this is satisfied generically. A degenerate dataset
(e.g. the rig only rotated about a single axis throughout the recording)
leaves the component of `X` about that axis unobservable.

`HandEyeProblem` admits only relative motions that rotate at least
`min_rel_angle` (`configs/default.toml` `[robot_world_hand_eye]` `min_rel_angle_deg` —
the same authority `fit_robot_world_hand_eye`'s init reads): its constructor is the one
gate, so every motion a sample draws has succeeded at it. `test_sample` then rejects a
minimal sample whose two axes lie within ~2.5° of each other. Both ill-condition
`X`, and near-identity motions put a tail on the hypotheses' scatter that their
first-order covariance does not predict. If you suspect
trajectory-level degeneracy across the full dataset, inspect the rotation
axes of the relative motions `A_ij` directly before relying on the
result.

## Limitations

* **No reprojection refinement here**: because `Y = T_WC` is eliminated,
  this pipeline cannot refine on pixel reprojection error (that requires `Y`
  to project pattern points into pixels). The robot-world form below keeps
  `Y`: `fit_robot_world_hand_eye` then `refine_robot_world_hand_eye` minimizes
  `Σ ‖uᵢⱼ − project(camera, Y⁻¹ ∘ Aᵢ ∘ Xₘ (Pⱼ))‖²` over `Y` and every `Xₘ`.
* **Single global characteristic length**: the default `L` is a scalar
  set from the data; the rotation-vs-translation trade-off is not
  per-frame heteroscedastic. See [Loss Variants](@ref handeye-variants) for
  the design alternatives (two-scale Mahalanobis, per-pair noise model,
  iter-profiled `L`) — none are currently implemented.

## Robot-World Calibration (Y kept)

`A_i X_m = Y B_i^m` with one shared `Y = T_WC` and per-marker `X_m = T_RPm`:
`fit_robot_world_hand_eye` is the staged linear solve from per-(frame, marker) poses
(its covariance is the first-order covariance of the staged estimator itself,
propagated through each stage's stationarity conditions by the implicit
function theorem, and `Inf` along unobservable directions); `refine_robot_world_hand_eye` refines it
against the markers' features — the reprojection error, camera and rig poses
fixed — on the product of `M + 1` `RigidExpManifold6` charts (any number of
markers; the LM is static in its `6(M + 1)` unknowns), and reports the
covariance of that estimator's own optimum, `ŝ²·implicit_covariance`, which
the Monte Carlo gate holds calibrated (Y, each `X_m`, the joint vector,
held-out reprojection, frame count, field of view, marker count and a poor
seed).

```@docs
fit_robot_world_hand_eye
refine_robot_world_hand_eye
RobotWorldHandEyeCalibration
RobotWorldHandEyeMetadata
```

## Design Document

The full formulation, including identifiability analysis, Tsai–Lenz
derivation, decoupled SE(3) residual + Jacobians, and a comparison of
loss-function variants, is the [Rig-to-Pattern Hand–Eye Calibration](../guide/hand-eye.md)
guide page; the robot-world form's objective is
[Robot–World Hand–Eye Calibration](../guide/robot-world-hand-eye.md).
