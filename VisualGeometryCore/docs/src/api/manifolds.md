# Model Manifolds and Uncertainty API

```@meta
CurrentModule = VisualGeometryCore
```

The minimal parameterizations the estimators refine on (the `Manifold`
contract and its instances), first-order covariance transport and solver
covariance, and the image-domain confidence regions that display it. Split
from the [Robust Estimation API](estimators.md) page. See the
[Robust Estimation](../guide/estimators.md) and
[Stereo Rigs and Uncertainty](../guide/stereo.md) guides.

## Model Manifolds

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["base/manifold.jl", "geometry/manifolds/essential.jl",
         "geometry/manifolds/fundmat.jl", "geometry/manifolds/homography.jl",
         "geometry/manifolds/rigid.jl", "geometry/manifolds/subset.jl",
         "geometry/manifolds/sphere.jl", "geometry/manifolds/product.jl",
         "geometry/manifolds/line.jl", "geometry/manifolds/covariance.jl"]
```

## Confidence Regions

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["geometry/confidence_regions.jl"]
```

## Index

```@index
Pages = ["manifolds.md"]
```
