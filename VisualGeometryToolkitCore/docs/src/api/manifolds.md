# Model Manifolds and Uncertainty API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The minimal parameterizations the estimators refine on (the `Manifold`
contract and its instances), first-order covariance transport and solver
covariance, and the image-domain confidence regions that display it. A
model's [`ambient_coordinates`](@ref) are what a manifold's default
[`retract_jacobian`](@ref) differentiates; a model type defined outside this
package adds its method. Split
from the [Robust Estimation API](estimators.md) page. See the
[Robust Estimation](../guide/estimators.md) and
[Stereo Rigs and Uncertainty](../guide/stereo.md) guides.

## Model Manifolds

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/manifold.jl", "geometry/manifolds/essential.jl",
         "geometry/manifolds/fundmat.jl", "geometry/manifolds/homography.jl",
         "geometry/manifolds/rigid.jl", "geometry/manifolds/subset.jl",
         "geometry/manifolds/sphere.jl", "geometry/manifolds/product.jl",
         "geometry/manifolds/line.jl", "geometry/manifolds/circle.jl",
         "geometry/manifolds/covariance.jl"]
```

## Confidence Regions

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/confidence_regions.jl"]
```

## Index

```@index
Pages = ["manifolds.md"]
```
