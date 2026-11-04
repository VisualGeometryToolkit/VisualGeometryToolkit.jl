# Rigs, Stereo and Captures API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Rigs of cameras and targets, stereo rectification, and the per-frame values
of a capture. Split from the [Imaging Chain and Targets API](imaging.md) page,
which holds the imaging chain and the targets. See the
[Stereo Rigs and Uncertainty](../guide/stereo.md) guide.

## Rigs

A [`Rig`](@ref) is members with fixed poses in one frame, each a (model,
pose) pair: a camera (its model and rig-relative extrinsics) or a
[`PosedTarget`](@ref). Sensors only, targets only, or both (a camera rig
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
The camera's settings per frame are a [`CaptureSettings`](@ref): each frame's
[`ExposureSettings`](@ref) and nominal [`LensSettings`](@ref), with the
[`DeviceIdentity`](@ref) of the body and the lens beside them (a camera model
holds no names) and [`is_constant`](@ref VisualGeometryToolkitCore.is_constant), whether every frame was exposed
alike.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/captures.jl"]
```

## Index

```@index
Pages = ["rigs.md"]
```
