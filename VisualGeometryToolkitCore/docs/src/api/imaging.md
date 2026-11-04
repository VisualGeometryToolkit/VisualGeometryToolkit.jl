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
`minimal_manifold`, `regularizer`, and CoordinateTransformations'
`transform_deriv_params`).

A standard layout is a constructor: Kalibr's AprilGrid is
`AprilGrid(; tag_rows, tag_cols, tag_size, tag_spacing)` (lengths, the spacing
the gap between two tags), `PlanarTarget(grid; board, canvas)` its tag
corners in Kalibr's point order and frame, and `point_index(grid, tag, corner)`
the point of a detected corner. Schöps et al.'s tagged star pattern is
`SchoepsPattern(; squares_x, squares_y, square_length, apriltags)`, and
`PlanarTarget(pattern; board, canvas)` its star features in Schöps' feature
order (feature id `k` is point `k + 1`).

### Target calibration

What a target calibration estimates about the physical target, as a camera
calibration estimates intrinsics and a lens: a CoordinateTransformations
`Transformation` of the design, its parameter Jacobian CT's
`transform_deriv_params` and its point Jacobian CT's `transform_deriv`. The
kinds:

| kind | map | parameters | Jacobian | regularizer | gauges fixed by default |
|---|---|---|---|---|---|
| [`Rigid`](@ref) | `p ↦ p` | none | 2 × 0 | none | none |
| [`Skew`](@ref) (`N = 2`) | `p ↦ U p`, `U` unit upper-triangular | `u12` (`U`'s strictly-upper entries, row by row) | 2 × 1 | none | none |
| [`Aspect`](@ref) (`N = 2`) | `p ↦ diag(e^{a}, e^{−a}) p` | `a1` | 2 × 1 | none | none |
| [`Scale`](@ref) | `p ↦ e^{k} p` | `log_scale` | 2 × 1 | none | none |
| a planar `EuclideanMap` | `p ↦ R(ϑ) p + t` | `angle, tx, ty` | 2 × 3 | none | none |
| [`ThinPlateSpline`](@ref) | `p ↦ p + (f_x(p), f_y(p))` | `fx1…fxK, fy1…fyK` | 2 × 2K | bending energy, 2K rows | its affine part: `:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v` |
| [`HeightField`](@ref) | `p ↦ (p, h(p))` | `h1…hK` | 3 × K | bending energy, K rows | `:piston, :tilt_x, :tilt_y` |
| a composition (CT's `∘`) | the members, the inner first | the members', tag-prefixed, the outer member's first | CT's chain rule, `hcat(p1, m1 * p2)` | stacked, the outer member's first | the members', tag-prefixed |

**The affine print error is its Iwasawa factors.** An affine map of the design
`p ↦ A p + b` factors as `A = s·R·D·U` (the KAN decomposition): a rotation, a
scale, a det-1 diagonal (the aspect) and a unit upper-triangular matrix (the
skew). The calibration composes them in that canonical order,
`EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, and holds only the ones its data
observes. With the target's pose estimated jointly, the rotation, the
translation and the scale are the pose's — a target and the same target turned,
shifted or scaled are the same images — so the print error is
`Aspect ∘ Skew`, two parameters and no gauges; a measured length (a ruler)
adds `Scale`; a known pose (a fixture, motion capture) the rigid member too.
The counts agree: an affine map of `R^N` has `N² + N` parameters, the pose
`N(N−1)/2 + N` and the scale one, which leaves `(N−1) + N(N−1)/2`. `Aspect`,
`Skew` and `Scale` are N-generic; a planar target takes `N = 2`.

A [`PlanarTarget`](@ref) takes one kind, the factors in the canonical order
(any of them left out), or a `HeightField` after those or after a
`ThinPlateSpline` (`HeightField ∘ Aspect ∘ Skew`,
`HeightField ∘ ThinPlateSpline`), and refuses any other composition with an
`ArgumentError`; it stores a composition right-associated, whatever
association `∘` built. A composition's parameters, names, bounds, Jacobian
columns, gauges and regularizer rows are in CoordinateTransformations' order,
the outer member's first. The 0.4 `AffineMap` kind (tag `"affine"`) is retired.

The thin-plate kinds are splines over `K` fixed nodes in the target's plane
(the numerics' [`ThinPlateSplineSystem`](@ref)), their parameters the nodal
values; a height is `tps_evaluate` of its spline, bit for bit. Lengths (a
rigid member's translation, nodes, displacements, heights) are in the design's
form, plain or physical, and move with `ustrip`/`with_unit`; `Scale`,
`Aspect` and `Skew` are dimensionless. A calibration may be
`Uncertain(c, CovMat(Σ))`, `Σ` over its parameters, as a pose may be.

**Planarity is in the type.** [`is_planar`](@ref) is true for `Rigid`, the
affine factors and `ThinPlateSpline`, false once a `HeightField` is present; `points(t)` is then `SVector{2}`s on `z = 0` or `SVector{3}`s, and the
result type is inferred. A method that assumes the plane takes
`t::PlanarTarget{<:PlanarCalibration}` and refuses a height field by dispatch.

**Gauges.** The thin-plate kinds have directions of θ that are the pose's: a
spline's affine part, and a height field plus a plane (a tilted, shifted
pose). The default [`TargetManifold`](@ref) — `minimal_manifold(t)`, which
means "the pose is estimated jointly" — holds them fixed: a `ThinPlateSpline`
is a pure residual, a `HeightField` loses its piston and tilts. The chart is
linear on an orthonormal basis of the complement of [`gauge_names`](@ref)'
directions. The affine factors have none: a calibration leaves the pose's
directions out of its composition instead.

```julia
t = PlanarTarget(pts; calibration = HeightField(nodes) ∘ Aspect(0.0) ∘ Skew(0.0))
ch = minimal_manifold(t)                                   # the pose estimated jointly: the height field's gauges fixed
PlanarTarget(pts; calibration = Scale(0.0) ∘ Aspect(0.0) ∘ Skew(0.0))   # a ruler: the scale is observable
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/target_calibration.jl", "cameras/targets.jl"]
```

### The surface a ray meets

[`surface`](@ref)`(pt)` gives a posed planar target's geometric primitive, in
the pose's destination frame (metres when physical), with the DESIGN chart as
its chart — an affine calibration (the factors, `Aspect ∘ Skew`, …) is the primitive's in-plane affine, so a
hit's chart coordinates are the nominal ones it shows — bounded by the
target's [`Board`](@ref) (`board = …` on [`PlanarTarget`](@ref), `Rect(board)` in
the design chart) or else by its nominal points: a [`Quad`](@ref) for a
`Rigid` or affine calibration, a [`HeightSurface`](@ref) for a
`HeightField` (alone or after the affine factors). An in-plane `ThinPlateSpline` is not
supported by the ray primitives yet. The choice is by type, so a
rigid target's ray caster never compiles the height field's Newton loop. See
[The Surface a Ray Meets](primitives.md#The-Surface-a-Ray-Meets), where
`surface` is documented with the primitives.

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
