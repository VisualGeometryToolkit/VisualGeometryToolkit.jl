# Correspondence API

```@meta
CurrentModule = VisualGeometryCore
```

Correspondence types and the solver layer built on them. Terminology: a
**correspondence** (`Cspond`) is a pair; a **correspondence set** (an n-tuple
of mutually corresponding elements, e.g. multiview tracks) lives in
VisualGeometryFeatures as `CspondSet`; the columnar **collection** of
pair-correspondences is `Csponds`.

Descriptor matching (`NNMatch`, `match`, strategies) lives in
VisualGeometryFeatures.

## Correspondences and Endpoint Lifts

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["base/cspond_set.jl"]
```

## Solver Layer: Frames, Verbs, Registry

```@autodocs
Modules = [VisualGeometryCore, VisualGeometryCore.Primitives, VisualGeometryCore.Cameras]
Private = false
Pages = ["geometry/solvers/solver_layer.jl"]
```

## Index

```@index
Pages = ["matching.md"]
```
