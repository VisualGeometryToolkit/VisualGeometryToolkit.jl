# Configuration Tags and Stack Contracts API

`TagRegistry` is the one TOML tag → type map; `CONTRACT_GENERICS` and the
dependency allow-list are what the quality gates (`test_quality_gates`) check
across the stack; the conformance helpers test an implementation of the
contracts of the [Robust Estimation API](estimators.md) page (their methods
load with `Test`).

```@meta
CurrentModule = VisualGeometryCore
```

```@autodocs
Modules = [VisualGeometryCore]
Pages = ["quality/tag_registry.jl", "quality/stack_contracts.jl", "quality/contract_helpers.jl"]
```

## Index

```@index
Pages = ["stack_contracts.md"]
```
