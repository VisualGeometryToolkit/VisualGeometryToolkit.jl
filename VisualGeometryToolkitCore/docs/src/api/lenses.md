# Lenses API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The lens families — two lens kinds, the projections, the terms and their
combinators — the lens contract on rays with everything derived from it, the
coefficient chart and record, and the batch surfaces. See the
[Lens Families](../guide/lenses.md) and
[Back-projection and Undistortion](../guide/undistortion.md) guides.

## Projection and Lens Models

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/lenses.jl", "cameras/projections.jl"]
```

## The Lens Contract, Coefficients and Batches

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["cameras/lens_rays.jl", "cameras/lens_parameters.jl"]
```

## Index

```@index
Pages = ["lenses.md"]
```
