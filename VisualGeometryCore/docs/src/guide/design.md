# Design: Attributed{V,A} Composition

VisualGeometryCore uses a single composition primitive — `Attributed{V,A}`
— to wrap any value with metadata. This replaces deep inheritance
hierarchies with a flat, type-stable pattern used uniformly across the
toolkit: feature detection (VisualGeometryFeatures), robust estimation
(RobustVisualGeometry), and the geometric utilities in this package.

## The Pattern

```julia
struct Attributed{V,A}
    value::V       # The geometric/mathematical object
    attributes::A  # Metadata about how it was computed
end
```

Property access is forwarded via `@generated getproperty`:
- `r.value`, `r.attributes` → direct field access
- Named fields of `A` → forwarded from attributes
- Other symbols → forwarded to the value

This means an `Attributed` result can be treated as if it were the value
itself:

```julia
p = Attributed(Point2(320.5, 240.5), 1.35)   # a depth-attributed pixel
p.value        # the Point2
p.attributes   # the depth
```

## Uses in this package

- **Depth-attributed points.** `Attributed{Point2, T}` is the currency of
  multi-view transfer: [`transfer`](@ref)`(rig, p)` consumes a pixel with
  its depth and returns the target-view pixel with the transported depth
  (see [Stereo Rigs and Uncertainty](stereo.md)).
- **Detections.** [`AbstractDetection`](@ref) is the alias
  `Attributed{<:Union{FeatureFrame, Uncertain{<:FeatureFrame}}}` — a
  geometric frame (`Disc`, `EllipticFrame`, …), bare or with its covariance,
  with detector metadata; [`feature_frame`](@ref) reads the bare frame of
  either kind. The detectors live in
  VisualGeometryFeatures; the geometry and the alias live here so
  downstream packages agree on the type.
- **Fit results do NOT use `Attributed`.** They return a
  [`FitResult`](@ref): a fixed core (`estimate`, `retcode`, `iterations`,
  `rms`, `n`) plus solver-specific `metadata` selected by type parameter.
  It forwards properties into its metadata with the same `@generated`
  mechanism described below, so `r.weights` reads like `sc.first` does —
  `Attributed` with a fixed core, not a second paradigm.
- **Scored correspondences.** [`ScoredCspond`](@ref) is
  `Attributed{Pair, Float64}` — even scalar metadata uses the same
  pattern.

Downstream packages use `Attributed` for their own value-plus-attributes
pairs and `FitResult` for their estimator results (RANSAC estimates, IRLS
refinements, calibration fits), supplying their own concrete metadata
types; the aliases differ, the mechanism does not.

## `Uncertain{V,M}` — an `Attributed` whose attributes are a statistical form

Uncertainty is metadata with laws, but it is still metadata:
[`Uncertain`](@ref)`{V,M}` is the alias `Attributed{V, M}` with
`M <: Union{CovMat, PrecMat}` — a geometric value with a typed
[`CovMat`](@ref) or [`PrecMat`](@ref) as its attributes, not a second
wrapper. The matrix's nominal type records which form is *stored*; the
`cov`/`prec` accessors return whichever form is asked for, inverting at most
once at the boundary, and `u.matrix` reads the stored form. Since 0.9 the matrices carry a frame label checked by
[`propagate`](@ref) — see
[Stereo Rigs and Uncertainty](stereo.md#Frame-labeled-statistical-forms).

## Why This Design?

**Flat over deep.** No `AbstractResult → ...` hierarchy: two parametric
types cover every case — `Attributed{V, A}` for a value with attributes,
`FitResult{P, M}` for an estimate with a fixed diagnostic core — and
dispatch on `Attributed{ScoredCspond, Float64}` or
`FitResult{CameraModel, M}` is exactly as efficient as on a
dedicated struct.

**Zero-cost forwarding.** The `@generated getproperty` compiles to a chain
of `=== :field` comparisons — the same as hand-written field access.

**Uniform interface.** Every result has `.value` and `.attributes`; code
that filters results works the same whether the result came from a
detector, a solver, or a fitting loop.

**Composable.** Attributes can nest — a pipeline result can carry its
source detection (itself an `Attributed`) inside its attributes, and
property forwarding handles the chain transparently.
