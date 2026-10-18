# Epipolar Forms API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The slot-labeled epipolar forms: which form a model is in, and how forms
compose; see the [Epipolar Forms](../guide/forms.md) guide. The solvers that
estimate them are in the [Solvers API](solvers.md).

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/forms.jl"]
```

```@docs
Base.transpose(::EpipolarForm)
```

## Index

```@index
Pages = ["epipolar_forms.md"]
```
