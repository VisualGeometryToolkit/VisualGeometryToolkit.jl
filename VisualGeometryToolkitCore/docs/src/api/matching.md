# Correspondence API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Correspondence types and the solver layer built on them. Terminology: a
**correspondence** (`Correspondence`) is a pair; a **correspondence set** (an n-tuple
of mutually corresponding elements, e.g. multiview tracks) lives in
VisualGeometryToolkitFeatures as `CorrespondenceSet`; the columnar **collection** of
pair-correspondences is `Correspondences`.

Descriptor matching (`NNMatch`, `match`, strategies) lives in
VisualGeometryToolkitFeatures.

## Correspondences and Endpoint Lifts

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["base/correspondence_set.jl"]
```

## Solver Layer: Frames, Verbs, Registry

```@autodocs
Modules = [VisualGeometryToolkitCore, VisualGeometryToolkitCore.Primitives, VisualGeometryToolkitCore.Cameras]
Private = false
Pages = ["geometry/solvers/solver_layer.jl"]
```

## Index

```@index
Pages = ["matching.md"]
```
