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
Pages = ["quality/naming.jl", "quality/test_helpers.jl", "quality/agent_docs.jl", "quality/conventions.jl",
         "quality/duplication.jl", "quality/raw_decoding.jl", "quality/sync_manifest.jl", "quality/gpu_tier.jl",
         "quality/splat.jl", "retired.jl"]
```

```@docs
SCALE_INTERFACE_TAGS
```

## Workspace CI

The CI of a stack repository, one implementation for every workspace: the
packages a change affects, the registration rules, the agent-docs policy over
a checkout, a downstream project pointed at a workspace, the generated
`codecov.yml` and a sibling package's docs inventory. `stack_ci` is their
command line; a workspace's own parameters are its `WORKSPACE.toml`
(`workspace_settings`).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Pages = ["quality/stack_ci.jl"]
```

## Index

```@index
Pages = ["stack_contracts.md"]
```
