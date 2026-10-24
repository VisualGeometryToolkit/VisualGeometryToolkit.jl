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
```

## Radial camera

```@docs
RadialCameraProblem
find_radial_camera
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
division_coeffs
board_sq_residuals
```

## Lens fitting

```@docs
fit_lens
LensFitMetadata
```
