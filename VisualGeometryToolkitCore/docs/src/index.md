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
  FlexUnits (a pixel is a dimensionless count).

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

```julia-repl
pkg> add VisualGeometryToolkitCore
```

Julia 1.12 or later; every dependency is in the General registry. Any Makie
backend (`pkg> add GLMakie`) loads the plotting extension.

## A first example

A camera views a planar board. The example projects the board's points into
the image, estimates the board-to-image homography from those
correspondences, and maps a circle on the board to the ellipse it images as
(the README's example; the test suite runs both).

```julia
using VisualGeometryToolkitCore

# A Sony IMX174 sensor behind a lens with a 40° horizontal field of view,
# 2 m in front of the board and looking at its centre.
s = sensor("Sony", "IMX174")
f = focal_length(s, deg2rad(40.0); dimension = :horizontal)
model = CameraModel(f, s.pitch, centroid(Rect(s)) .* px)
camera = PosedCamera(model, lookat(Vec3(0.3, 0.2, -2.0) .* m, Vec3(0.0, 0.0, 0.0) .* m, Vec3(0.0, -1.0, 0.0)))

# A 5 × 5 grid of board points (metres, on z = 0) and their images (pixels).
board = [Point2(x, y) for x in -0.2:0.1:0.2 for y in -0.2:0.1:0.2]
pixels = [project(camera, Point3(p[1], p[2], 0.0) .* m) for p in board]

# The board-to-image homography, and what it does to the board.
H = HomographyMap(homography_dlt(csponds(board, pixels)))
to_euclidean(H(board[1])) ≈ pixels[1]                # true
image = transfer_ellipse(H, Circle(Point2(0.0, 0.0), 0.05))   # a 5 cm circle's image: an Ellipse (nothing past the horizon)
```

## Module structure

- **`Cameras`** (submodule) — camera models, sensors, projection, stereo
  rigs, the imaging chain, undistortion.
- **`Primitives`** (submodule) — geometric primitives.
- **Transforms, frames, forms, solvers, robust primitives** — defined in
  the top-level module (not submodules, due to circular dependencies).

Commonly-used names are exported from the main module; the submodule paths
(`VisualGeometryToolkitCore.Cameras.PosedCamera`) and the exported names (`PosedCamera`)
are equivalent.

## Documentation

**Start here.** [Getting Started](guide/getting-started.md) installs the
package and introduces units, frames and a first camera.

**How to.** Model a camera and its lens ([Cameras](guide/cameras.md),
[Lens Families](guide/lenses.md),
[Back-projection and Undistortion](guide/undistortion.md)); work with
[Geometric Primitives](guide/primitives.md) and
[Transforms](guide/transforms.md); run maps over [Batches](guide/batched.md)
on the CPU or a GPU; choose an estimator
([Robust Estimation](guide/estimators.md)); plot
([Visualization](guide/visualization.md)).

**Explanation.** [Conventions](guide/conventions.md) states the stack's
rules and the defect each one prevents;
[Coordinate Frames](guide/frames.md),
[Epipolar Forms](guide/forms.md),
[The Framed Imaging Chain](guide/imaging-chain.md),
[Ray Maps and Frame Transitions](guide/sphere.md),
[Stereo Rigs and Uncertainty](guide/stereo.md) and
[Design: Attributed{V,A}](guide/design.md) explain the design.

**Reference.** The API Reference pages, from [Frames](api/frames.md) to
[Synthetic Data](api/synthetic.md), document every public name.

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitCore owns the contracts the robust-estimation packages compose
through (see [The composable RANSAC](@ref)). It is consumed by
[VisualGeometryToolkitEstimation.jl](@extref VisualGeometryToolkitEstimation :doc:`index`)
(the problems, the one RANSAC loop and the core local optimizations),
[RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl)
(the Student-t + uniform scorer and the grid selection families, implementing
the scoring and pool-selection contracts),
LocalOptimizationDoneRight (the paper's local optimizers, implementing the
local-optimization contract), VisualGeometryToolkitEval (datasets, pools, tuning and
the composition evaluator),
[VisualGeometryToolkitFeatures.jl](@extref VisualGeometryToolkitFeatures :doc:`index`)
(feature detection and matching), and
[BlobBoards.jl](https://github.com/VisualGeometryToolkit/BlobBoards.jl)
(fiducial blob-board targets).

All toolkit documentation is aggregated at the portal:
<https://visualgeometrytoolkit.github.io>.

## Citing

VisualGeometryToolkitCore is part of VisualGeometryToolkit. Cite the toolkit; its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff) is the source of this entry (GitHub's "Cite this
repository" button exports it):

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
