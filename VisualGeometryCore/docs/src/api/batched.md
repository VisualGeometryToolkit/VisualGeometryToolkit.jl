# Batch API

```@meta
CurrentModule = VisualGeometryCore
```

Per-point maps over arrays, and the B-spline kernels. See the
[Batches](../guide/batched.md) guide; the lens surfaces (`project!`, …) are
documented with the [Lenses API](lenses.md).

## Pointwise maps

```@autodocs
Modules = [VisualGeometryCore]
Private = false
Pages = ["transforms/pointwise.jl"]
```

## B-spline Kernels

The iterative Cox–de Boor basis and the tensor-product surface/4D spline
kernels; the `ChainRulesCore` rrules attach via the package extension.

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["numerics/bspline.jl"]
```

## Index

```@index
Pages = ["batched.md"]
```
