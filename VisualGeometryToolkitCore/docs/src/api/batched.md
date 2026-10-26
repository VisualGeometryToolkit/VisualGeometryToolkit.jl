# Batch API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Per-point maps over arrays, the B-spline kernels and the thin-plate spline. See the
[Batches](../guide/batched.md) guide; the lens surfaces (`project!`, …) are
documented with the [Lenses API](lenses.md).

## Pointwise maps

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["transforms/pointwise.jl"]
```

## B-spline Kernels

The iterative Cox–de Boor basis and the tensor-product surface/4D spline
kernels; the `ChainRulesCore` rrules attach via the package extension.

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["numerics/bspline.jl"]
```

## Thin-plate spline

`f(𝐩) = a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ)` with `U = ½ r² ln r²`, through
nodal values on scattered nodes (hoisted from RadiometricCamera, its arithmetic
unchanged). [`ThinPlateSpline`](@ref) is the host-side solver; the evaluator
[`tps_evaluate`](@ref) and its gradient [`tps_gradient`](@ref) take isbits
`(nodes, w, a)`, allocate nothing and run in a KernelAbstractions kernel on the
CPU backend.

```julia
tps = ThinPlateSpline(grid_nodes(5))            # 5 × 5 nodes over [-½, ½]²
w, a = tps_coefficients(tps, values)            # linear in the 25 nodal values
f = tps_evaluate(tps.nodes, w, a, SVector(0.1, -0.2))
∇f = tps_gradient(tps.nodes, w, a, SVector(0.1, -0.2))
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/thin_plate_spline.jl"]
```

## Index

```@index
Pages = ["batched.md"]
```
