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

The scalar Cox–de Boor basis [`bspline_basis`](@ref)`(knots, Val(p), x)` —
an `SVector` of every basis function at `x`, isbits and generic in the element
type, so it runs in a kernel and under ForwardDiff (hoisted from
RadiometricCamera, its arithmetic unchanged) — with
[`clamped_uniform_knots`](@ref); its broadcast over an array, and the
tensor-product surface/4D spline kernels built on it; the `ChainRulesCore`
rrules attach via the package extension.

```julia
knots = clamped_uniform_knots(7, 3, -0.5, 0.5)   # 7 cubic control points on [-½, ½]
N = bspline_basis(knots, Val(3), 0.12)           # SVector{7}, sum(N) == 1
```

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["numerics/bspline.jl"]
```

## Thin-plate spline

`f(𝐩) = a₁ + a_x 𝐩_x + a_y 𝐩_y + Σᵢ wᵢ U(𝐩, 𝐩ᵢ)` with `U = ½ r² ln r²`, through
nodal values on scattered nodes (hoisted from RadiometricCamera, its arithmetic
unchanged). [`ThinPlateSplineSystem`](@ref) is the host-side solver; the evaluator
[`tps_evaluate`](@ref) and its gradient [`tps_gradient`](@ref) take isbits
`(nodes, w, a)`, allocate nothing and run in a KernelAbstractions kernel on the
CPU backend.

```julia
tps = ThinPlateSplineSystem(grid_nodes(5))      # 5 × 5 nodes over [-½, ½]²
w, a = tps_coefficients(tps, values)            # linear in the 25 nodal values
f = tps_evaluate(tps.nodes, w, a, SVector(0.1, -0.2))
∇f = tps_gradient(tps.nodes, w, a, SVector(0.1, -0.2))
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/thin_plate_spline.jl"]
```

## Special functions

Device-safe scalar functions — pure arithmetic, generic over `Real`, so they
compile in a KernelAbstractions kernel at any float width and ForwardDiff
`Dual`s flow through: log Γ and the digamma/trigamma series of the Student-t
adaptation kernels, the normal CDF [`normcdf_as`](@ref) (Abramowitz & Stegun
7.1.26) and the dilogarithm [`dilog`](@ref).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/special.jl"]
```

## Quadrature

The probabilists' Gauss–Hermite rule [`gauss_hermite_prob`](@ref) (an
expectation over `N(0, 1)`, exact to degree `2n − 1`) and the Hammersley
points [`hammersley`](@ref) (a kernel's own low-discrepancy sample from its
index).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["numerics/quadrature.jl"]
```

## Index

```@index
Pages = ["batched.md"]
```
