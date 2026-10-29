# Solvers API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Projective solver kernels (homography, fundamental, essential, P3P),
model-validity tests, the zero-allocation SVD workspace, and the
total-least-squares line. The slot-labeled epipolar forms have their own page,
the [Epipolar Forms API](epipolar_forms.md). The `solve` protocol and the
solver registry are documented with the [Correspondence API](matching.md); see the
[Epipolar Forms](../guide/forms.md) guide.

## Normalization

One scheme serves every solver. A normalization STRATEGY is a type named
`<Method>Normalization`, applied as a coordinate map with
`AffineMap(strategy, points[, mask, weights])`:

- [`HartleyNormalization`](@ref)`(; center)` — translate to the (weighted)
  centroid, or to a known `center`, and scale to mean distance √2. Estimation
  uses it: the centroid for H, F and E; the known distortion centre for the
  radial models, whose solvers are written about the origin.
- [`DiagonalNormalization`](@ref)`(size; center)` — the image centre and the
  half-diagonal (corners at unit radius, [`NormalizedImageFrame`](@ref)
  coordinates). For storage only: the benchmark pools, where one noise level
  and one threshold must mean the same across resolutions.

A correspondence set's normalization is one value,
[`CspondNormalization`](@ref): both sides' maps and their inverses, built
from two strategies or by model type (`CspondNormalization(M, cs; center)`),
with `normalize` and the side accessors, whose composition moves a model
(`D ∘ M ∘ inv(S)` into the normalized frame, `inv(D) ∘ M ∘ S` back). An
estimator normalizes once at its entry and returns results in its inputs'
frame; the DLT kernels take the enclosing normalization as `normalization`.

## Homography Solvers

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/homography_solvers.jl", "geometry/solvers/homography_5pt_radial.jl", "geometry/radial_models.jl"]
```

## Fundamental and Essential Matrix Solvers

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/fundmat.jl", "geometry/solvers/solvers.jl", "geometry/solvers/essential_5pt_native.jl", "geometry/solvers/p3p_native.jl"]
```

## Lines

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/tls_line.jl"]
```

## Circles

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/circle_3pt.jl"]
```

## Ellipses

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/direct_ellipse.jl"]
```

## Model Validity

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/model_validity.jl"]
```

## SVD Workspace

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/svd_workspace.jl"]
```

## Index

```@index
Pages = ["solvers.md"]
```
