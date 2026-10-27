# Lens Families

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Every lens is one of two kinds, built from a handful of parts. The parts —
projections and terms — are *leaves*, structs of numbers and tuples of numbers;
everything a lens is asked (its coefficients, their names, its zero, its record,
its manifold, every Jacobian, every batch) is derived from their fields by one
walk, never tabulated per model. A coefficient is named by its field — a tuple
field `k` as `k1, k2, …` — and a leaf's record tag is its name in snake case
(`DoubleSphere` → `double_sphere`, `FieldOfView` → `field_of_view`; the Full
radial basis is `radial_full`).

## Two kinds of lens

- [`Lens`](@ref)`(projection, distortion)` — **forward**: the ray is projected
  onto the projection's frame, and the frame point is distorted there,
  `x_d = distortion(projection(X))`. Projection is closed form; back-projection
  inverts the distortion.
- [`LiftLens`](@ref)`(radial)` — **backward**: the image point lifts to its ray
  in closed form, `(x_d·N(ρ), D(ρ))`; projection is a root on the ray. `D`
  crossing zero is the 90° circle and `D < 0` the field behind it.

## Projections

| projection | radius | frame |
|---|---|---|
| [`Perspective`](@ref) | `r = tan θ` | `GnomonicFrame` |
| [`Equidistant`](@ref) | `r = θ` | `EquidistantFrame` |
| [`Stereographic`](@ref) | `r = 2 tan(θ/2)` | `StereographicFrame` |
| [`Equisolid`](@ref) | `r = 2 sin(θ/2)` | `EquisolidFrame` |
| [`Orthographic`](@ref) | `r = sin θ` | `OrthographicFrame` |
| [`DoubleSphere`](@ref)`(xi, alpha)` | [Usenko2018](@citet) | model-defined |
| [`Unified`](@ref)`(xi)` | [Mei2007](@citet) (Kalibr `omni`) | model-defined |
| [`ExtendedUnified`](@ref)`(alpha, beta)` | [Khomutenko2016](@citet) (Kalibr `eucm`) | model-defined |

`Lens(projection)` alone is the pure projection; [`Pinhole`](@ref) is
`Lens(Perspective())`.

## Terms and how they combine

| term | kind | function of the frame point `x`, `r² = ‖x‖²` | names |
|---|---|---|---|
| [`Radial`](@ref)`(k, d)` | map | `x·N/D`, `N = 1 + Σ kᵢ r²ⁱ`, `D = 1 + Σ dⱼ r²ʲ` | `k1…, d1…` |
| [`Tangential`](@ref)`(p1, p2)` | delta | `(2p₁uv + p₂(r² + 2u²), 2p₂uv + p₁(r² + 2v²))` | `p1, p2` |
| [`ThinPrism`](@ref)`(sx, sy)` | delta | `(Σ sxᵢ r²ⁱ, Σ syᵢ r²ⁱ)` | `sx1…, sy1…` |
| [`Tilt`](@ref)`(tx, ty)` | map | OpenCV's tilted-sensor projective map | `tx, ty` |
| [`FieldOfView`](@ref)`(omega)` | map | `x·atan(2r tan(ω/2))/(rω)` | `omega` |

The arity of a term is its tuple lengths — a type parameter — so a
four-coefficient Kannala–Brandt and a six-coefficient one are the same family.
`Radial{Full}` uses the powers `ρ², ρ³, …` instead of `ρ², ρ⁴, …` (Scaramuzza's
lift).

- [`Additive`](@ref)`(m, Δ₁, …)` — every term at the **same** point, summed:
  `m(x) + (Δ₁(x) + …)`, the deltas left to right. OpenCV's `projectPoints` and
  COLMAP's `Distortion()` are of this form.
- [`Chain`](@ref)`(D₁, D₂, …)` — in sequence, `D₂(D₁(x))`: Aria's Fisheye624
  (COLMAP `RAD_TAN_THIN_PRISM_FISHEYE`) applies its tangential and thin-prism
  terms to the radially distorted point; OpenCV's tilt follows its additive
  model.

## Familiar models

| model | lens |
|---|---|
| pinhole | [`Pinhole`](@ref)`()` = `Lens(Perspective())` |
| OpenCV 4/5/8/12 coefficients | `Lens(Perspective(), Additive(Radial((k1, k2, k3), (k4, k5, k6)), Tangential(p1, p2), ThinPrism((s1, s2), (s3, s4))))` — only the terms in use |
| OpenCV 14 (tilted) | `Lens(Perspective(), Chain(Additive(…), Tilt(τx, τy)))` |
| OpenCV fisheye, COLMAP `OPENCV_FISHEYE` | [`KannalaBrandt`](@ref)`(k1, k2, k3, k4)` = `Lens(Equidistant(), Radial((k1, k2, k3, k4)))` |
| COLMAP `SIMPLE_RADIAL`, `RADIAL` | `Lens(Perspective(), Radial((k,)))`, `Radial((k1, k2))` |
| COLMAP `THIN_PRISM_FISHEYE` | `Lens(Equidistant(), Additive(Radial((k1, …, k4)), Tangential(p1, p2), ThinPrism((sx1,), (sy1,))))` |
| COLMAP `FOV` | `Lens(Perspective(), FieldOfView(ω))` |
| division (Fitzgibbon), COLMAP `DIVISION` | [`Division`](@ref)`(k…)` = `LiftLens(Radial((), (k…)))` |
| BabelCalib rational | `LiftLens(Radial((a…), (b…)))` |
| Scaramuzza / MATLAB fisheye | [`Scaramuzza`](@ref)`(a2, a3, a4)` = `LiftLens(Radial{Full}((), (a2, a3, a4)))` |
| double sphere (Kalibr `ds`) | `Lens(DoubleSphere(ξ, α))` |
| Kalibr `omni-radtan` | `Lens(Unified(ξ), Additive(Radial((k1, k2)), Tangential(r1, r2)))` |

The adapters that read and write these conventions — and reproduce each
convention's function exactly — live in VisualGeometryToolkitIO.

## Coefficients

[`parameter_vector`](@ref)`(lens)` is the coefficient vector — projection first,
then the distortion's terms depth-first, left to right;
[`parameter_names`](@ref) names it, [`parameter_bounds`](@ref) bounds it
(unbounded, but the double sphere's `α` in `(0, 1)`), [`parameters`](@ref) pairs the names and values
in a `NamedTuple`, and [`with_parameters`](@ref) rebuilds the same structure
from a vector of any number type. The same three functions describe a
calibratable target, its [`TargetCalibration`](@ref) and a `LogicalIntrinsics`. [`LensManifold`](@ref) moves the coefficients
(a bounded one, the double sphere's `α`, through its logit), and
`restrict(LensManifold(lens), names)` frees a subset by name.

[`radial`](@ref)`(lens)` is the lens's radially symmetric factor — its folds and
poles bound what the lens images ([`view_angle_limit`](@ref),
`monotone_radius`) — and [`radial_part`](@ref) the lens with only that factor.

## The image circle

A local lens model — a grid of nodes rather than a formula — has observations
only where the lens lights the sensor, so its domain is
`Rect(extent(camera)) ∩ image_circle(camera)`. [`image_circle`](@ref) predicts
that region; nothing is segmented from an image. It is built either from the
fitted lens (the cone of [`view_angle_limit`](@ref) through the radial
projection and `K`) or from a lens datasheet's image-circle diameter through
the sensor's pitch, and both routes return the same primitive.

- The circle is a conic, an [`Ellipse`](@ref) in pixels about the principal
  point: an ellipse proper for an anisotropic `K` or pitch.
- For a lens that is not radially symmetric (a [`Chain`](@ref) with a tilt
  stage, or tangential terms) it is an inscribed bound: every point inside is
  imaged, and the true lit region may be slightly larger.
- A lens with no limit of its own (a perspective lens, an undistorted
  equidistant lens: `view_angle_limit` reports only its chart's or the
  sphere's edge, a cap) answers the [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane),
  the identity of `∩`, so `Rect ∩ image_circle(cam)` is the `Rect`; its circle,
  if any, comes from the datasheet. A global parametric lens is defined
  everywhere by its formula and needs no restriction at all.

```@example imagecircle
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: Pinhole, KannalaBrandt
s = sensor("Sony", "IMX250")                            # 2448 × 2048 @ 3.45 μm
k = -4 / (3π^2)                                         # r = θ(1 + kθ²) folds at 90°, r = (2/3)θ
f = ustrip(px, 6.0mm / (2 * s.pitch.width)) / (π / 3)   # a 180° lens drawing a 6 mm circle
cam = BoundedCamera(CameraModel(; focal = f, pp = (1224.0, 1024.0), lens = KannalaBrandt(k)), s)
q = image_circle(cam)                                   # from the lens
(centroid(q), q.a, q.b)
```

```@example imagecircle
qd = image_circle(cam, 6.0mm)                           # from the datasheet, through the pitch
(q.a - qd.a, Rect(extent(cam)) ∈ q, rad2deg(fov(cam)), rad2deg(fov(cam; dimension = :diagonal)))
```

```@example imagecircle
rcam = BoundedCamera(CameraModel(; focal = 2000.0, pp = (1224.0, 1024.0), lens = Pinhole()), s)
(image_circle(rcam), rad2deg(fov(rcam)), rad2deg(fov(rcam; dimension = :diagonal)))
```

```@example imagecircle
domain = Rect(extent(cam)) ∩ q                          # a grid lens's domain
nodes = [Point2(x, y) for x in range(0.0, 2448.0; length = 25), y in range(0.0, 2048.0; length = 21)]
lit = filter(n -> n ∈ domain, nodes)
(length(lit), length(nodes) - length(lit))              # lit, outside
```

The whole example, runnable, is `examples/image_circle.jl`.

## The lens record

[`lens_record`](@ref) writes a lens as a self-describing record, the form
VisualGeometryToolkitIO stores and configurations name: every key a leaf tag or a
combinator, every value a coefficient array whose length is the arity.

```julia
lens_record(KannalaBrandt(-0.043, 0.011))
# {"equidistant": [], "radial": [[-0.043, 0.011], []]}
lens_record(Division(-0.31, 0.024))
# {"lift": {"radial": [[], [-0.31, 0.024]]}}
lens_record(Lens(Perspective(), Additive(Radial((-0.28,)), Tangential(1.0e-3, -5.0e-4))))
# {"perspective": [], "add": [{"radial": [[-0.28], []]}, {"tangential": [0.001, -0.0005]}]}
```

[`lens_from_record`](@ref) inverts it exactly: the record determines the lens
type, and with shortest round-trip floats a `Float64` lens reads back bitwise.
