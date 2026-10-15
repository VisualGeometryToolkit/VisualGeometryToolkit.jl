# Essential Matrix

```@meta
CurrentModule = RobustVisualGeometry
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
