# Imaging Chain and Stereo API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The framed imaging chain (frame-to-frame stages, ray maps, frame
transitions) and stereo rigs. See the
[The Framed Imaging Chain](../guide/imaging-chain.md),
[Ray Maps and Frame Transitions](../guide/sphere.md), and
[Stereo Rigs and Uncertainty](../guide/stereo.md) guides.

## The Framed Imaging Chain

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/imaging_chain.jl"]
```

## Targets

A calibration target is a MODEL, the analogue of a camera model:
[`AbstractTarget`](@ref) promises `points(target)` in its own `TargetFrame{id}`;
[`PointTarget`](@ref) holds explicit (nominal or estimated) points,
[`PlanarTarget`](@ref) a design — its nominal points, or the reference they
are resolved from — and a warp ([`Rigid`](@ref) today). A target whose
geometry an estimator refines implements the calibratable-target protocol
(`target_parameters`, `with_parameters`, `minimal_manifold`, `regularizer`).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/targets.jl"]
```

## Rigs

A [`Rig`](@ref) is members with fixed poses in one frame, each a (model,
mount) pair: a camera (its model and rig-relative extrinsics) or a
[`Mounted`](@ref) target. Sensors only, targets only, or both (a camera rig
carrying its marker constellation).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/rig.jl", "cameras/stereo.jl", "cameras/rectify.jl"]
```

## Captures

The per-frame values of a capture, keyed by frame index: a body's
[`Trajectory`](@ref), a detector's [`TargetPoses`](@ref), the measured target
points [`TargetPoints2D`](@ref) (pixels) and [`TargetPoints3D`](@ref) (3D
markers), and [`convert_frame`](@ref), the one way to change a pixel frame.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/captures.jl"]
```

## Index

```@index
Pages = ["imaging.md"]
```
