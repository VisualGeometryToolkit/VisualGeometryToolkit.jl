# Feature Detection

!!! note "Module Moved"
    Feature detection (scale space, blob detection, affine adaptation,
    descriptor matching) has been extracted to
    [VisualGeometryToolkitFeatures.jl](@extref VisualGeometryToolkitFeatures :doc:`index`).
    Import `detect_features`, the detector types, scale-space machinery,
    and `match` from `VisualGeometryToolkitFeatures` instead of
    `VisualGeometryToolkitCore.Features`.

What remains in VisualGeometryToolkitCore is the *geometry* that detections are
made of, and the composition types that carry them:

- The feature-frame primitives — [`Disc`](@ref), `OrientedDisc`,
  [`EllipticFrame`](@ref), `AffineFrame`, and [`Paraboloid3D`](@ref) —
  documented in the [Primitives](primitives.md) guide.
- [`Attributed`](@ref)`{V,A}` and [`AbstractDetection`](@ref) — the
  value-plus-metadata composition pattern, documented in
  [Design: Attributed{V,A}](design.md).
- Correspondence types ([`Cspond`](@ref), [`Csponds`](@ref),
  [`ScoredCspond`](@ref)) and the endpoint-lift protocol — see the
  [Correspondence API](../api/matching.md).
