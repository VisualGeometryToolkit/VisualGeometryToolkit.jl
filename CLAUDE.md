# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code anywhere in this repository. Each package directory has its own
`CLAUDE.md` with what is specific to it; read both.

## The repository

VisualGeometryToolkit.jl is a Julia 1.12 **workspace** holding six separately
registered packages, one directory each (the directory name is the package
name, the Registrator `subdir`, the TagBot tag prefix `<Name>-v…` and the docs
`dirname`):

| directory | package | depends on (stack) |
|---|---|---|
| `VisualGeometryToolkitCore/` | types, frames, cameras, kernels, the robust-estimation contracts | — |
| `VisualGeometryToolkitIO/` | file formats and schemas | VisualGeometryToolkitCore |
| `VisualGeometryToolkitFeatures/` | feature detection and matching | VisualGeometryToolkitCore |
| `VisualGeometryToolkitEstimation/` | the RANSAC loop and robust estimators | VisualGeometryToolkitCore |
| `VisualGeometryToolkitEval/` | evaluation protocol, pools, paper-project template | VisualGeometryToolkitCore, VisualGeometryToolkitIO, VisualGeometryToolkitEstimation |
| `PoseLib/` | wrapper of the PoseLib C++ solvers (binaries: `PoseLib_jll`) | VisualGeometryToolkitCore |
| `ci/` | workspace tooling; a workspace member, never registered | — |

The stack dependency law is the allow-list
`VisualGeometryToolkitCore/configs/stack_dependencies.toml`; nothing else may name a
stack package.

The root `Project.toml` is not a package. Its `[sources]` resolve every stack
package to its directory, and its `[workspace]` lists the six packages and
`ci`; each package's own `[workspace]` adds its `test/` and `docs/`
environments. There is ONE Manifest, at the root, and it is never tracked.

## Commands

```bash
# Once per clone
git config core.hooksPath .githooks          # re-resolve the root Manifest when any Project.toml moves
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Test one package (inside the workspace: no sandbox, the root Manifest)
julia --project=VisualGeometryToolkitCore -e 'using Pkg; Pkg.test()'
julia --project=VisualGeometryToolkitEstimation -e 'using Pkg; Pkg.test(; test_args = ["ransac"])'
julia --project=VisualGeometryToolkitCore -e 'using Pkg; Pkg.test(; julia_args = ["--depwarn=error"])'

# Docs of one package. --startup-file=no is required (a startup.jl that
# activates another env overrides --project).
julia --project=VisualGeometryToolkitCore/docs --startup-file=no VisualGeometryToolkitCore/docs/make.jl

# Workspace checks (CI runs them on every PR)
julia --project=ci ci/registration_lint.jl --subsequent   # versions, compat, names, no tracked Manifest, allow-list
julia --project=ci ci/agent_docs.jl          # the agent-docs policy over every tracked agent file
julia --startup-file=no ci/affected.jl origin/main HEAD   # packages a change touches, with dependents
```

Test dependencies live in `<Package>/test/Project.toml` (no `[extras]`/`[targets]`).
GPU items record a skip without a device; run them locally before a release
with `CUDA_VISIBLE_DEVICES=0`. The LO timing gate (`LO_TIMING_GATE=1`) runs on
the release machine only. See `RELEASING.md`.

## Git commit conventions

- Never add "Co-Authored-By: Claude" or any other AI attribution to a commit message.
- Never add a "Generated with Claude Code" footer.
- A commit that changes several packages names each of them.

## Stack conventions

The canonical text lives once, in `VisualGeometryToolkitCore/conventions/` (it ships
with VisualGeometryToolkitCore; `VisualGeometryToolkitCore.conventions_dir()`). Change it
there and nowhere else; downstream repositories check their copies against it.

@VisualGeometryToolkitCore/conventions/overview.md
@VisualGeometryToolkitCore/conventions/naming.md
@VisualGeometryToolkitCore/conventions/signatures.md
@VisualGeometryToolkitCore/conventions/data-arguments.md
@VisualGeometryToolkitCore/conventions/frames.md
@VisualGeometryToolkitCore/conventions/pixel-origins.md
@VisualGeometryToolkitCore/conventions/bearings.md
@VisualGeometryToolkitCore/conventions/jacobians.md
@VisualGeometryToolkitCore/conventions/units.md
@VisualGeometryToolkitCore/conventions/stack-contracts.md
