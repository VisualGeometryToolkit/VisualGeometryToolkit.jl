# Calibration API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Camera calibration from planar targets: the 1-D radial camera and its
minimal solver, intrinsics from homographies (`zhang_intrinsics`), and each
lens's honest view-angle domain. The iterative calibration built on them —
the radial camera's fit, the division-camera focal stack, lens-model
regression — lives in VisualGeometryToolkitEstimation. Split from the
[Cameras API](cameras.md) page. See the [Cameras](../guide/cameras.md) guide.

## Radial Camera, Intrinsics

`RadialCamera(::PosedCamera)` is exact only for a radially symmetric lens; which
lenses are is Core's camera trait [`RadialSymmetry`](@ref)
(`RadiallySymmetric()` / `RadiallyAsymmetric()`, public, not exported: reached as
`VisualGeometryToolkitCore.Cameras.RadialSymmetry`). It fails closed: a lens type
that has not declared itself symmetric is treated as asymmetric.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/calibration.jl"]
```

## View-Angle Limits

[`view_angle_limit`](@ref) and [`radial_part`](@ref) are documented with the lenses.

## Index

```@index
Pages = ["calibration.md"]
```
