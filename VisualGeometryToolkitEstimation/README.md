# VisualGeometryToolkitEstimation.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitEstimation)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitEstimation)

VisualGeometryToolkitEstimation provides robust estimation for geometric
vision: one generic RANSAC loop and the estimation problems of the
VisualGeometryToolkit packages, from lines, homographies, fundamental and
essential matrices and absolute pose to radial cameras and hand-eye
calibration. Every `find_*` pipeline runs MSAC scoring with a least-squares
refit on the inliers by default, and other scorers and local optimizers plug
in by keyword.

## Installation

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitEstimation
```

Julia 1.12 or later. Every dependency is in the General registry. The example
uses the geometry types of VisualGeometryToolkitCore, so the line adds both
packages.

## Example

The example builds 40 correspondences from a known homography, replaces 8 of
them with random points, and recovers the homography and the 32 inliers.

```julia
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation, Random

H_true = HomographyMat([1.02 -0.02 12.0; 0.01 0.98 -5.0; 1e-5 2e-5 1.0])

rng = Xoshiro(1)
src = [Point2(640rand(rng), 480rand(rng)) for _ in 1:40]    # source points, pixels
dst = [to_euclidean(HomographyMap(H_true)(x)) for x in src] # their images under H_true
for i in 1:8                                                # replace 8 with outliers
    dst[i] = Point2(640rand(rng), 480rand(rng))
end

result = find_homography(csponds(src, dst))   # MSAC at 3 px, the default
count(result.inlier_mask)                     # 32, the true correspondences

H = Matrix(estimate(result))
H ./ H[3, 3] ≈ Matrix(H_true)                 # true

result_2px = find_homography(csponds(src, dst); threshold = 2.0)   # a keyword overrides a default
```

The [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/getting-started/)
guide continues from here.

## Documentation

- [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/getting-started/):
  a first pipeline, composing a scorer and a local optimizer, and calling
  `ransac` directly.
- [Fitting Algorithms](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/fitting/):
  every `find_*` and `fit_*` pipeline with its default threshold.
- [Extending](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/extending/):
  writing a new RANSAC problem.
- [Architecture](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/architecture/):
  the type hierarchy, the contracts and the design decisions.
- [API Reference](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/api/ransac/).

The package builds on VisualGeometryToolkitCore, which holds the geometry
types, the minimal solvers and the contracts the loop is written against. The
paper scorers and local optimizers implement those contracts in
[RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl)
and
[LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl).
The [portal](https://visualgeometrytoolkit.github.io) links every package of
the toolkit.

## Citing

VisualGeometryToolkitEstimation is written by James Pritts, with contributions
from Philipp Böll ([@phib2b](https://github.com/phib2b)): opt-in joint scale
estimation for up-to-scale pose pairs in robot-world calibration. Cite the
toolkit. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  doi    = {10.5281/zenodo.23548372},
  year   = {2026}
}
```

Cite also the papers behind the methods you use; the
[portal](https://visualgeometrytoolkit.github.io/#publications) lists them.

## License

MIT
