# Configuration Tags and Stack Contracts API

`TagRegistry` is the one TOML tag → type map; `CONTRACT_GENERICS` and the
dependency allow-list are what the quality gates (`test_quality_gates`) check
across the stack; the conformance helpers test an implementation of the
contracts of the [Robust Estimation API](estimators.md) page (their methods
load with `Test`).

```@meta
CurrentModule = VisualGeometryToolkitCore
```

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["quality/tag_registry.jl", "quality/stack_contracts.jl", "quality/contract_helpers.jl"]
```

## Quality gates

The checks every stack package's quality items run: the naming rules, the
rows Jacobian check, the gate bundle, the agent-docs policy and the one copy
of the conventions. `check_naming` and `test_quality_gates` load with `Test`
(the `Test` extension).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["quality/naming.jl", "quality/test_helpers.jl", "quality/agent_docs.jl", "quality/conventions.jl"]
```

```@docs
SCALE_INTERFACE_TAGS
```

## Index

```@index
Pages = ["stack_contracts.md"]
```
