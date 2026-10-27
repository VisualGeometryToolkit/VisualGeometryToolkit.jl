# Pose and Rotation from Directions

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

Estimators whose data are DIRECTIONS — unit bearings from a camera's
back-projection, star or gravity directions — rather than pixels. They work on
the sphere throughout (VisualGeometryToolkitCore `conventions/bearings.md`), so
rays at and past 90° from the optical axis, as a fisheye or omnidirectional
camera sees them, are ordinary data.

## Camera pose from bearings

`refine_absolute_pose` polishes a pose guess against world points and the unit
bearings observing them: the tangent-plane (logarithm-map) residual at each
bearing, whose norm is the angle between the observed and the predicted ray,
on VisualGeometryToolkitCore's `LeftRigidManifold6` chart with the chart LM. Ray
origins (an `origins` column) make it the generalized, non-central form; the
back-projection Jacobians (an `M` column) whiten its residuals into pixels.

```@docs
refine_absolute_pose
AbsolutePoseRefineMetadata
```
