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
division_coeffs
board_sq_residuals
```

## Camera and poses

`refine_camera_and_poses` refines a camera — its intrinsics and lens, on
VisualGeometryToolkitCore's `ParameterManifold` — jointly with one pose per
view (`LeftRigidManifold6`) against the reprojection error of 3D → 2D
correspondences, from a guess of both: the bundle of a calibration with a known
target, for any lens the camera's family holds (a fisheye's rays past 90° are
ordinary data). `free = ()` holds the camera and refines the poses alone.

```@docs
refine_camera_and_poses
CameraPosesRefineMetadata
```

## Lens fitting

```@docs
fit_lens
LensFitMetadata
```
