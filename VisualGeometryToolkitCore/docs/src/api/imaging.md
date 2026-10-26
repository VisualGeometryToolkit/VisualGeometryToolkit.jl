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
are resolved from — and a [`TargetCalibration`](@ref) taking the design to
the physical target. A target whose geometry an estimator refines implements
the calibratable-target protocol (`parameter_vector`, `with_parameters`,
`minimal_manifold`, `regularizer`, `calibration_jacobian`).

### Target calibration

What a target calibration estimates about the physical target, as a camera
calibration estimates intrinsics and a lens. The kinds:

| kind | map | parameters | Jacobian | regularizer | gauges fixed by default |
|---|---|---|---|---|---|
| [`Rigid`](@ref) | `p ↦ p` | none | 2 × 0 | none | none |
| [`Affine`](@ref) | `p ↦ A p + b` | `a11, a12, a21, a22, bx, by` | 2 × 6 | none | `:translation_x, :translation_y, :rotation, :scale` |
| [`ThinPlateSpline`](@ref) | `p ↦ p + (f_x(p), f_y(p))` | `fx1…fxK, fy1…fyK` | 2 × 2K | bending energy, 2K rows | its affine part: `:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v` |
| [`HeightField`](@ref) | `p ↦ (p, h(p))` | `h1…hK` | 3 × K | bending energy, K rows | `:piston, :tilt_x, :tilt_y` |
| [`CalibrationChain`](@ref) | members in order | members', tag-prefixed | chain rule | stacked | members', tag-prefixed |

The thin-plate kinds are splines over `K` fixed nodes in the target's plane
(the numerics' [`ThinPlateSplineSystem`](@ref)), their parameters the nodal
values; a height is `tps_evaluate` of its spline, bit for bit. Lengths (an
`Affine`'s `b`, nodes, displacements, heights) are in the design's form,
plain or physical, and move with `ustrip`/`with_unit`. A calibration may be
`Uncertain(c, CovMat(Σ))`, `Σ` over its parameters, as a pose may be.

**Planarity is in the type.** [`is_planar`](@ref) is true for `Rigid`,
`Affine`, `ThinPlateSpline` and chains of them, false once a `HeightField` is
present; `points(t)` is then `SVector{2}`s on `z = 0` or `SVector{3}`s, and the
result type is inferred. A method that assumes the plane takes
`t::PlanarTarget{<:PlanarCalibration}` and refuses a height field by dispatch.

**Gauges.** Some directions of θ are the pose's: images cannot tell a target
from the same target shifted, turned in its plane or scaled (the pose moving
with it), nor a height field from the same one plus a plane (a tilted,
shifted pose). The default [`TargetManifold`](@ref) — `minimal_manifold(t)`,
which means "the pose is estimated jointly" — holds them fixed: an `Affine`
keeps only its stretch and shear, a `ThinPlateSpline` is a pure residual (its
affine part is the `Affine`'s; a standalone one carries none), a
`HeightField` loses its piston and tilts. The chart is linear on an
orthonormal basis of the complement of [`gauge_names`](@ref)' directions,
evaluated once at the anchor. With a known pose (motion capture) free the
gauges by name; a measured length frees the scale:

```julia
t = PlanarTarget(pts; calibration = CalibrationChain(Affine(A, b), ThinPlateSpline(nodes), HeightField(nodes)))
ch = minimal_manifold(t)                                   # the pose estimated jointly: every gauge fixed
ch = TargetManifold(t; free = :affine_scale)               # a ruler pins the metric scale
ch = TargetManifold(t; free = gauge_names(t))              # a known pose: nothing is a gauge
restrict(minimal_manifold(t), (:affine_a11, :affine_a12, :affine_a21, :affine_a22))   # the print's linear part only: stretch and shear
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/target_calibration.jl", "cameras/targets.jl"]
```

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

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/captures.jl"]
```

## Index

```@index
Pages = ["imaging.md"]
```
