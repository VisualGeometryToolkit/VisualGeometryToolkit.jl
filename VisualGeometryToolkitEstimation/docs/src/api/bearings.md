# Pose and Rotation from Directions

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

Estimators whose data are DIRECTIONS — unit bearings from a camera's
back-projection, star or gravity directions — rather than pixels. They work on
the sphere throughout (VisualGeometryToolkitCore `conventions/bearings.md`), so
rays at and past 90° from the optical axis, as a fisheye or omnidirectional
camera sees them, are ordinary data. (A camera pose from a guess is
[`refine_absolute_pose`](@ref), on the PnP problem's pixel rows — [P3P and
Absolute Pose](p3p.md).)

## Rotation from direction pairs

`fit_rotation` is the robust rotation `R` with `b ≈ R·a` from direction pairs
with gross outliers: VisualGeometryToolkitCore's `kabsch`, reweighted by a stack
loss shape at a width that follows the robust scale of the residual angles.

```@docs
fit_rotation
RotationFitMetadata
```

`fit_rotation_averaging` is the absolute rotations of a set of frames from the
relative rotations of a view graph, some of them wrong: VisualGeometryToolkitCore's
spectral seed, then a reweighted first-order Lie-algebra least squares over
every edge (Chatterjee & Govindu), each a sparse linear solve.

```@docs
fit_rotation_averaging
RotationAveragingMetadata
```
