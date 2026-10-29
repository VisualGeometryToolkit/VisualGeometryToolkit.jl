# Essential Matrix

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## RANSAC Problem

`EssentialProblem` is a type alias for `MinimalSolveProblem{EssentialMat{T}, 5, T, ...}`
— a calibrated two-view problem with the 5-point minimal solver, on UNIT
BEARINGS (3-vectors), not normalized 2-vectors. Build it from pixels and the
two cameras, `EssentialProblem(camera1, camera2, pixel_cs)`, which
back-projects each view with its own camera and attaches the back-projection
Jacobians `M₁`/`M₂` that keep residuals and the threshold in PIXELS.

```@docs
EssentialProblem
```

## Fitting Pipeline

```@docs
find_essential
refine_essential
```

## Thresholds and operating points

`find_essential`'s default threshold, 1 px (`[pipelines.essential]` in
`configs/default.toml`), is a convention (OpenCV's `findEssentialMat`), not a
tuned value. The paper's cross-validated operating points — the per-class
scale multiples of the Welsch half-quadratic lift, with their provenance — are
LocalOptimizationDoneRight's tuning constants (`tuning_constant`,
`WELSCH_HQ_SCALE_MULTIPLE_ESSENTIAL`, …) and are supplied with its strategy:
`find_essential(cam1, cam2, cs; scoring = …, local_optimization =
HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_ESSENTIAL))`.

## The Pure Rotation

A camera that only rotates between two views induces the infinite homography,
a rotation `R` with `f₂ ∥ R·f₁` on unit bearings. `RotationProblem` is the
bearing problem of that model, built like the essential problem from the
cameras' back-projection with the `M₁`/`M₂` Jacobian columns (the whole field
of a fisheye takes part, past 90°): VisualGeometryToolkitCore's `Rotation2pt`
on two pairs, the residual the Sampson distance of the joint measurement (both
images' pixels, Σ = I) to the transfer constraint on bearings, in pixels (VisualGeometryToolkitCore's `constraint_with_jacobians(TransferForm(H),
f₁, f₂; M₁, M₂)` at `H = R`), and the chart-LM refit on VisualGeometryToolkitCore's
`SO3Manifold` through the homography's transfer rows. `find_rotation`
is its RANSAC; a scorer or a lift from another package plugs in as for
`find_essential`.

```@docs
RotationProblem
find_rotation
refine_rotation
```
