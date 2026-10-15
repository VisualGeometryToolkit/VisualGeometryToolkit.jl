# Calibration API

```@meta
CurrentModule = VisualGeometryCore
```

Camera calibration from planar targets: the 1-D radial camera and its
minimal solver, intrinsics from homographies (`zhang_intrinsics`), and each
lens's honest view-angle domain. The iterative calibration built on them —
the radial camera's fit, the division-camera focal stack, lens-model
regression — lives in RobustVisualGeometry. Split from the
[Cameras API](cameras.md) page. See the [Cameras](../guide/cameras.md) guide.

## Radial Camera, Intrinsics

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["cameras/calibration.jl"]
```

## View-Angle Limits

[`view_angle_limit`](@ref) and [`radial_part`](@ref) are documented with the lenses.

## Index

```@index
Pages = ["calibration.md"]
```
