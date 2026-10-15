# Lens Families

```@meta
CurrentModule = VisualGeometryCore
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
| [`DoubleSphere`](@ref)`(xi, alpha)` | Usenko et al. 2018 | model-defined |
| [`Unified`](@ref)`(xi)` | Mei & Rives 2007 (Kalibr `omni`) | model-defined |
| [`ExtendedUnified`](@ref)`(alpha, beta)` | Khomutenko et al. 2016 (Kalibr `eucm`) | model-defined |

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
convention's function exactly — live in VisualGeometryIo.

## Coefficients

[`lens_parameters`](@ref)`(lens)` is the coefficient vector — projection first,
then the distortion's terms depth-first, left to right;
[`lens_parameter_names`](@ref) names it, [`parameters`](@ref) pairs the two
in a `NamedTuple`, and [`with_parameters`](@ref) rebuilds the same structure
from a vector of any number type. [`LensManifold`](@ref) moves the coefficients
(a bounded one, the double sphere's `α`, through its logit), and
`restrict(LensManifold(lens), names)` frees a subset by name.

[`radial`](@ref)`(lens)` is the lens's radially symmetric factor — its folds and
poles bound what the lens images ([`view_angle_limit`](@ref),
`monotone_radius`) — and [`radial_part`](@ref) the lens with only that factor.

## The lens record

[`lens_record`](@ref) writes a lens as a self-describing record, the form
VisualGeometryIo stores and configurations name: every key a leaf tag or a
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
