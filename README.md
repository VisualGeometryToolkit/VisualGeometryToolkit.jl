<p align="center"><img src="assets/logo.svg" width="160" alt="VisualGeometryToolkit logo"></p>

# VisualGeometryToolkit.jl

by [James Pritts](https://prittjam.github.io/)

Composable Julia packages for robust multiple-view geometry: geometric types,
file formats, feature detection, robust estimation and RANSAC, calibration
(camera and lens models, hand–eye), fiducial markers (BlobBoards), evaluation
protocols, and a PoseLib wrapper, factored so each layer is usable on its own.

This repository is a Julia workspace. Each directory below is a separately
registered package with its own version, tests and documentation.

| Package | Role | Docs |
|---|---|---|
| [VisualGeometryToolkitCore](VisualGeometryToolkitCore) | Geometric types, frames, camera and lens models, minimal solvers, scale-space kernels, the robust-estimation contracts | [docs](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/) |
| [VisualGeometryToolkitIO](VisualGeometryToolkitIO) | Versioned file formats: cameras, rigs, observations, targets, correspondences | [docs](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitIO/) |
| [VisualGeometryToolkitFeatures](VisualGeometryToolkitFeatures) | Feature detection (ASIFT/DoG with keypoint covariance) and matching | [docs](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitFeatures/) |
| [VisualGeometryToolkitEstimation](VisualGeometryToolkitEstimation) | The RANSAC loop, problem implementations, IRLS, hand–eye calibration | [docs](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/) |
| [VisualGeometryToolkitEval](VisualGeometryToolkitEval) | Evaluation protocol, benchmark pools, tuning, reproducible paper projects | [docs](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEval/) |
| [PoseLib](PoseLib) | Julia wrapper of the PoseLib minimal solvers (prebuilt binaries from `PoseLib_jll`) | [docs](https://visualgeometrytoolkit.github.io/PoseLib/) |

### Built on the toolkit (separate repositories)

Public packages that build on these and live in their own repositories:

| Package | Role | Repository |
|---|---|---|
| RANSACScoringDoneRight | Scale-free marginal RANSAC scoring | [VisualGeometryToolkit/RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl) |
| LocalOptimizationDoneRight | Local optimization for RANSAC | [VisualGeometryToolkit/LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl) |
| BlobBoards | Fiducial blob-board targets: design, detection and tracking | [VisualGeometryToolkit/BlobBoards.jl](https://github.com/VisualGeometryToolkit/BlobBoards.jl) |
| PanoramaStitching | Rotation-only panoramas: feature matching on unit bearings, robust rotation averaging and a rotation bundle | [VisualGeometryToolkit/PanoramaStitching.jl](https://github.com/VisualGeometryToolkit/PanoramaStitching.jl) |

All documentation is aggregated at <https://visualgeometrytoolkit.github.io>.

## Package stack

```mermaid
graph TD
    VGC["VisualGeometryToolkitCore"]
    VGF["VisualGeometryToolkitFeatures"]
    VGIO["VisualGeometryToolkitIO"]
    RVG["VisualGeometryToolkitEstimation"]
    PL["PoseLib"]
    VGE["VisualGeometryToolkitEval"]
    subgraph separate["separate repositories"]
        RSDR["RANSACScoringDoneRight"]
        LODR["LocalOptimizationDoneRight"]
        BB["BlobBoards"]
        PS["PanoramaStitching"]
    end

    VGC --> VGF
    VGC --> VGIO
    VGC --> RVG
    VGC --> PL
    VGC --> RSDR
    VGIO --> VGE
    RVG --> VGE
    RVG --> LODR
    PL -.-> LODR
    VGF --> BB
    VGIO --> BB
    RSDR --> BB
    LODR --> BB
    VGF --> PS
    RVG --> PS
```

Each package builds on the ones above it (direct dependencies; edges implied through another package are omitted). Dashed: optional (package extension).

## Installation

VisualGeometryToolkitCore, VisualGeometryToolkitIO, VisualGeometryToolkitFeatures
and VisualGeometryToolkitEstimation are in the General registry:

```julia
using Pkg
Pkg.add("VisualGeometryToolkitEstimation")      # pulls VisualGeometryToolkitCore
```

VisualGeometryToolkitEval and PoseLib are in
[CauRegistry](https://github.com/prittjam/CauRegistry.jl) until their General
registrations merge. For them, add both registries once per depot (General
first: Pkg adds it on its own only to an empty depot):

```julia
Pkg.Registry.add("General")
Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl")
Pkg.add("VisualGeometryToolkitEval")
```

Manifests record each package by UUID, version and tree hash, never by
registry, so an environment resolves the same from either registry.

**The Schur complement of the LM driver.** The stack's Levenberg–Marquardt driver is NLLSsolver; release 5.3 contains our pull requests [#61](https://github.com/ojwoodford/NLLSsolver.jl/pull/61)–[#63](https://github.com/ojwoodford/NLLSsolver.jl/pull/63) (allocation-free copies, `OptimizeCache`, the trial-point reuse). One remains open, [#60](https://github.com/ojwoodford/NLLSsolver.jl/pull/60), the Schur-complement linear system for eliminable variables: without it, everything is correct but the bundles that eliminate variables (the relative-pose bundle, PanoramaStitching's rotation bundle) solve the full system and run slower. To use it now, pin our fork once in your environment:

```julia
using Pkg
Pkg.add(PackageSpec(url = "https://github.com/prittjam/NLLSsolver.jl", rev = "stack"))
```

`stack` is the latest upstream release plus #60. Verify with `using NLLSsolver; isdefined(NLLSsolver, :MultiVariateLSschur)` → `true`. When an upstream release contains #60, `Pkg.free("NLLSsolver")` returns the environment to the registry and this note is removed.

## Development

```bash
git clone https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl
cd VisualGeometryToolkit.jl
git config core.hooksPath .githooks
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=VisualGeometryToolkitCore -e 'using Pkg; Pkg.test()'
```

Inside the workspace every stack package resolves to its directory here, so a
change to VisualGeometryToolkitCore is seen by the packages that depend on it. See
[CLAUDE.md](CLAUDE.md) for the conventions and [RELEASING.md](RELEASING.md)
for registration.

## Citing

Cite the toolkit; [CITATION.cff](CITATION.cff) is the source of this entry
(GitHub's "Cite this repository" button exports it):

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Cite also the papers behind the methods you use, listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).

## License

MIT, James Pritts; PoseLib, MIT, Kiel University (each package directory
carries its own copy). See
[AUTHORS.md](AUTHORS.md).

Maintainer: [James Pritts](https://prittjam.github.io/).
