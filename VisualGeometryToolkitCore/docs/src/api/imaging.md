# Imaging Chain and Targets API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The framed imaging chain (frame-to-frame stages, ray maps, frame
transitions) and calibration targets. See the
[The Framed Imaging Chain](../guide/imaging-chain.md) and
[Ray Maps and Frame Transitions](../guide/sphere.md) guides; rigs, stereo
and captures are on the [Rigs, Stereo and Captures API](rigs.md) page.

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

A standard layout is a constructor, an `AbstractTargetLayout`: Kalibr's AprilGrid is
`AprilGrid(; tag_rows, tag_cols, tag_size, tag_spacing, apriltag_family, first_id)` (lengths, the spacing
the gap between two tags), `PlanarTarget(grid; calibration, board, canvas, placement)` its tag
corners in Kalibr's point order and frame, and `point_index(grid, tag, corner)`
the point of a detected corner. Schöps et al.'s tagged star pattern is
`SchoepsPattern(; squares_x, squares_y, square_length, apriltags, star_segments, apriltag_family)`, and
`PlanarTarget(pattern; calibration, board, canvas, placement)` its star features in Schöps' feature
order (feature id `k` is point `k + 1`). The target records its layout,
`layout(t)`, so that VisualGeometryToolkitIO writes it as one (its tag in
`TARGET_LAYOUT_TAGS`).

### Target calibration

What a target calibration estimates about the physical target, as a camera
calibration estimates intrinsics and a lens: a CoordinateTransformations
`Transformation` of the design, its parameter Jacobian CT's
`transform_deriv_params` and its point Jacobian CT's `transform_deriv`. The
kinds:

| kind | map | parameters | Jacobian | regularizer | free directions |
|---|---|---|---|---|---|
| [`Rigid`](@ref) | `p ↦ p` | none | 2 × 0 | none | 0 |
| [`Skew`](@ref) (`N = 2`) | `p ↦ U p`, `U` unit upper-triangular | `u12` (`U`'s strictly-upper entries, row by row) | 2 × 1 | none | 1 |
| [`Aspect`](@ref) (`N = 2`) | `p ↦ diag(e^{a}, e^{−a}) p` | `a1` | 2 × 1 | none | 1 |
| [`Scale`](@ref) | `p ↦ e^{k} p` | `log_scale` | 2 × 1 | none | 1 |
| a planar `EuclideanMap` | `p ↦ R(ϑ) p + t` | `angle, tx, ty` | 2 × 3 | none | 3 |
| [`ThinPlateSpline`](@ref) | `p ↦ p + (f_x(p), f_y(p))`, no affine term | `fx1…fxK, fy1…fyK` | 2 × 2K | bending energy, 2K rows | 2K − 6 |
| [`HeightField`](@ref) | `p ↦ (p, h(p))`, no affine term | `h1…hK` | 3 × K | bending energy, K rows | K − 3 |
| a composition (CT's `∘`) | the members, the inner first | the members', tag-prefixed, the outer member's first | CT's chain rule, `hcat(p1, m1 * p2)` | stacked, the outer member's first | the members' sum |

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

**The thin-plate kinds are pure bending.** An in-plane spline's displacements
and a height field's heights have no affine term in their interpolant: the
spline through each nodal value vector is `Σ wᵢ U(·, pᵢ)` alone, its affine
coefficients `a` ([`tps_coefficients`](@ref)) zero — intrinsic to the spline,
whatever the node distribution. The constructor and `with_parameters` project
what they are given onto that subspace, `v ↦ v − M a(v)` with `M = [1 u v]`
at the nodes, keeping the bending (values whose affine term is within 1e-12
of them are kept bit for bit, so RadiometricCamera's board heights, the
bending part of its spline at its chart nodes, are kept unchanged). The
condition is defined on the nodes the spline is built on: the radial basis
`U(r) = r² log r` is not scale-invariant, and scaling the nodes moves a constant (a
piston) between the affine and the bending parts — build the thin-plate kinds
on a dimensionless chart of the board.
An affine displacement belongs to the affine factors and a plane of heights to
the pose; a spline built from either is the identity. So the canonical print
error with a jointly estimated pose is

```julia
HeightField(nodes, h) ∘ ThinPlateSpline(nodes, fx, fy) ∘ Aspect(a) ∘ Skew(u)
```

and no kind has a gauge: every direction a composition's parameters may move
in is observable, and the joint pose/calibration Jacobian has full column rank.
The stored parameters stay the nodal values (2K, K), with `2K − 6` (`K − 3`)
free directions; a covariance over them has that rank.

A [`PlanarTarget`](@ref) takes one kind, or kinds in the canonical order
`HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, any of
them left out, and refuses any other composition with an `ArgumentError`; it
stores a composition right-associated, whatever association `∘` built. A
composition's parameters, names, bounds, Jacobian columns and regularizer rows
are in CoordinateTransformations' order, the outer member's first. The 0.4
`AffineMap` kind (tag `"affine"`) is retired.

The thin-plate kinds are splines over `K` fixed nodes in the target's plane
(the numerics' [`ThinPlateSplineSystem`](@ref)), their parameters the nodal
values; a height is `tps_evaluate` of its spline, bit for bit. Lengths (a
rigid member's translation, nodes, displacements, heights) are in the design's
form, plain or physical, and move with `ustrip`/`with_unit`; `Scale`,
`Aspect` and `Skew` are dimensionless. A calibration may be
`Uncertain(c, CovMat(Σ))`, `Σ` over its parameters, as a pose may be.

**A calibration on the board's own chart.** "No affine term" is a property of
the nodes a thin-plate kind is built on (`r² log r` is not scale-invariant:
rescaling the nodes moves a piston into the affine part), so a field computed
on a dimensionless chart of the board — RadiometricCamera's page `[0, 1]²` —
stays on that chart, and the target places the chart with a placement, Core's
`SimilarityMap{3}` `S₃` of scale `side`, the identity rotation and the
translation `(origin, 0)` (the design point `p = origin + side·c`):
`PlanarTarget(…; calibration, placement)` applies its calibration as the
composition `S₃ ∘ lift ∘ C ∘ S₂⁻¹`, a height scaled by `side`. The calibration
is plain chart numbers whatever the points' form; the placement is fixed data,
plain numbers in the points' unit (metres for a physical target, rescaled by
`ustrip`/`with_unit`), and adds no parameters. A page whose chart runs down the design's y is placed at its
top-left corner on the reflected chart nodes `(u, −v)`, which keep the
bending pure.

```julia
nodes = [SVector(n[1], -n[2]) for n in grid_nodes(5; lo = 0.0, hi = 1.0)]   # the page's chart, rows down
t = PlanarTarget(:page; calibration = HeightField(nodes, heights ./ side),     # heights in chart units
                 placement = SimilarityMap(side, one(RotMatrix{3, Float64}), SVector(x0, ytop, 0.0)), board, canvas)
```

**Planarity is in the type.** [`is_planar`](@ref) is true for `Rigid`, the
affine factors and `ThinPlateSpline`, false once a `HeightField` is present; `points(t)` is then `SVector{2}`s on `z = 0` or `SVector{3}`s, and the
result type is inferred. A method that assumes the plane takes
`t::PlanarTarget{<:PlanarCalibration}` and refuses a height field by dispatch.

**The manifold.** [`TargetManifold`](@ref)`(t)` — `minimal_manifold(t)` — is
linear: the identity basis for the affine factors, the thin-plate kinds' free
directions (an orthonormal null-space basis of the affine-coefficient rows, from a pivoted
Householder QR: reproducible) for the splines. There is nothing to free or
fix: what the data observes is chosen by the composition's members.

```julia
t = PlanarTarget(pts; calibration = HeightField(nodes) ∘ ThinPlateSpline(nodes) ∘ Aspect(0.0) ∘ Skew(0.0))
ch = minimal_manifold(t)                                   # (K − 3) + (2K − 6) + 2 directions, all observable
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
`Rigid` or affine calibration, a [`MappedSurface`](@ref) for any other — a
`HeightField`, an in-plane `ThinPlateSpline`, either after the affine
factors — the surface of the composed chart map `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹`,
built by `MappedSurface(pose, calibration, placement, rect)` (a device caller
builds it from adapted parts). The choice is by type, so a rigid target's ray
caster never compiles the root.

**On the device.** `HeightField`, `ThinPlateSpline` and their
`ThinPlateSplineSystem` take `Adapt.adapt` (their matrices become device
arrays) and `convert_eltype` (a `Float32` form); [`tps_pure_bending`](@ref)
and [`tps_bending_coefficients`](@ref) give a field's pure-bending values and
spline coefficients from nodal values in a kernel, the same functions the
host constructors call. [`fit_height_field`](@ref) fits a pure-bending field
to scattered heights, the plane removed.

**No gauge freedom.** The thin-plate kinds' pure bending is the OBLIQUE
complement of the pose's directions (values orthogonal to the interpolant's
affine rows, not to the plane fields), so no deformation direction is a pose
change: the stacked Jacobian of the posed points in the deformation and the
pose has full rank (test/mapped_surface_tests.jl pins its condition number and
the smallest principal angle between the two spans for each composition). See
[The Surface a Ray Meets](primitives.md#The-Surface-a-Ray-Meets), where
`surface` is documented with the primitives.

## Index

```@index
Pages = ["imaging.md"]
```
