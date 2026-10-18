# VisualGeometryToolkitCore.jl

*Computational geometry, camera modeling, and projective solvers for
computer vision.*

VisualGeometryToolkitCore.jl is the foundation layer of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack:

- **Coordinate frames** — type-level frame tags, 3D and image-plane, on
  maps, forms, and covariance matrices; composition, inversion, and
  propagation are frame-checked by dispatch.
- **Camera models** — pinhole, Brown–Conrady, and the fisheye-projection
  family (Kannala–Brandt, equidistant, stereographic, equisolid,
  orthographic), one lens contract on rays (`project`/`backproject`, valid
  past 90°) from which every lens map is derived, and a framed,
  junction-checked imaging chain.
- **Two-view geometry** — stereo rigs, slot-labeled epipolar forms,
  homography/fundamental/essential/P3P solver kernels, minimal model
  charts with covariance transport.
- **Geometric primitives** — circles, ellipses, discs, feature frames,
  hyperplanes, and their homogeneous matrix representations.
- **Transforms** — matrix-backed homogeneous transforms, region maps,
  image warping, and a batched structure-of-arrays protocol with
  GPU-portable kernels.
- **Robust-estimation primitives** — the unified robust-loss family,
  block-coordinate-descent loop, and scoring interfaces shared by the downstream
  estimation engines.
- **Physical units** — `mm`, `μm`, `px`, `dpi` throughout, via
  FlexUnits with a pixel dimension.

## Design principles

**Labels are opt-in and never viral.** Coordinate-frame tags
(`Src`/`Dst` on maps, slot frames on forms, frame labels on covariance)
are singleton types with zero runtime cost. Unlabeled objects compose with
anything; labeled junctions are checked, and a labeled mismatch is a
`MethodError`, never a silent fall-through. One wildcard truth table
governs every junction in the package. See
[Coordinate Frames](guide/frames.md).

**Transformations have endpoints, forms have slots, rank lives on the
manifold.** Maps (`EuclideanMap`, `HomographyMap`) carry endpoint frames
and compose with `∘`; bilinear forms (`EpipolarForm`, `CovMat`/`PrecMat`)
carry slot or frame labels and transform by their own laws (the correlation
`EpipolarMap` composes with collineations; `propagate`) — the rank-2 epipolar correlation deliberately has no `inv`.
See [Epipolar Forms](guide/forms.md).

**Frames, not retinas, are the type currency of imaging.** Every central
camera shares one spherical retina; camera models differ in the 2D frame on
it that their distortion is written in. The imaging chain is a family of
framed, junction-checked stages over those frames, and a lens's action free
of any 2D frame is its sphere self-map `ray_map`. See
[The Framed Imaging Chain](guide/imaging-chain.md) and
[Ray Maps and Frame Transitions](guide/sphere.md).

**One lens contract, kernel-lawful numerics.** Every lens is two maps on
rays — `project` and `backproject`, valid past 90° wherever the lens images —
and every other lens map (perspective views, Jacobians, undistortion, warps)
is derived from them. The maps never throw (NaN where the lens does not
image), never allocate, and are precision- and Dual-generic, so the same code
runs scalar, batched, and on GPU via KernelAbstractions — backends come from
the array type, not from vendor packages. See
[Back-projection and Undistortion](guide/undistortion.md) and
[Batched Maps](guide/batched.md).

**Flat composition over deep hierarchies.** A single `Attributed{V,A}`
type carries every value-plus-metadata pattern; `Uncertain{V,M}` is its
alias for a value whose attributes are a covariance or precision. See
[Design: Attributed{V,A}](guide/design.md).

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitCore")
```

For visualization support:
```julia
Pkg.add("GLMakie")  # Automatically activates the plotting extension
```

## Module structure

- **`Cameras`** (submodule) — camera models, sensors, projection, stereo
  rigs, the imaging chain, undistortion.
- **`Primitives`** (submodule) — geometric primitives.
- **Transforms, frames, forms, solvers, robust primitives** — defined in
  the top-level module (not submodules, due to circular dependencies).

Commonly-used names are exported from the main module; the submodule paths
(`VisualGeometryToolkitCore.Cameras.Camera`) and the exported names (`Camera`)
are equivalent.

## Documentation

```@contents
Pages = [
    "guide/getting-started.md",
    "guide/design.md",
    "guide/frames.md",
    "guide/forms.md",
    "guide/stereo.md",
    "guide/cameras.md",
    "guide/imaging-chain.md",
    "guide/sphere.md",
    "guide/undistortion.md",
    "guide/batched.md",
    "guide/primitives.md",
    "guide/transforms.md",
    "guide/estimators.md",
]
Depth = 2
```

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitCore owns the contracts the robust-estimation packages compose
through (see [The composable RANSAC](@ref)). It is consumed by
[VisualGeometryToolkitEstimation.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitEstimation)
(the problems, the one RANSAC loop and the core local optimizations),
[RansacScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RansacScoringDoneRight.jl)
(the Student-t + uniform scorer and the grid selection families, implementing
the scoring and pool-selection contracts),
LocalOptimizationDoneRight (the paper's local optimizers, implementing the
local-optimization contract), VisualGeometryToolkitEval (datasets, pools, tuning and
the composition evaluator),
[VisualGeometryToolkitFeatures.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitFeatures)
(feature detection and matching), and
[BlobBoards.jl](https://github.com/VisualGeometryToolkit/BlobBoards.jl)
(fiducial blob-board targets).

All toolkit documentation is aggregated at the portal:
<https://visualgeometrytoolkit.github.io>.
