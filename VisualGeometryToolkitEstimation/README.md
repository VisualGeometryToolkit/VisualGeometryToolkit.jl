# VisualGeometryToolkitEstimation.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitEstimation)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitEstimation)

Robust estimation for geometric vision. The package provides one generic
RANSAC loop and the estimation problems of the VisualGeometryToolkit stack:
lines, homographies, fundamental and essential matrices, absolute pose,
radial homographies and radial cameras, relative-pose bundle refinement, and
hand-eye and robot-world calibration.

Every `find_*` pipeline runs MSAC scoring with a least-squares refit on the
inliers by default. Other scorers and local optimizers can be supplied by
keyword. The paper methods live in
[RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl)
and
[LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl).

The package builds on
[VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore),
which holds the geometry types, the minimal solvers and the contracts the loop
is written against.

## Installation

The examples use names from both packages, so install both:

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitEstimation
```

Julia 1.12 or later is required.

## Quick start

Estimate a homography from 40 correspondences, 8 of which are outliers:

```julia
using VisualGeometryToolkitEstimation, Random
using VisualGeometryToolkitCore: HomographyMat, HomographyMap, Point2, csponds, estimate, to_euclidean

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

result = find_homography(csponds(src, dst); threshold = 2.0)   # a keyword overrides a default
```

## Documentation

The [documentation](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/)
covers the rest:

- [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/getting-started/):
  composing a scorer and a local optimizer, and calling `ransac` directly.
- [Architecture](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/architecture/):
  the type hierarchy, the contracts and the design decisions.
- [RANSAC](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/ransac/):
  local optimization, samplers, configuration and the result type.
- [Fitting Algorithms](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/fitting/):
  every `find_*` and `fit_*` pipeline with its default threshold.
- [Extending](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/guide/extending/):
  writing a new RANSAC problem.

All toolkit documentation is collected at <https://visualgeometrytoolkit.github.io>.

## Contributors

VisualGeometryToolkitEstimation is written by James Pritts, with contributions from
Philipp Böll ([@phib2b](https://github.com/phib2b)): opt-in joint scale
estimation for up-to-scale pose pairs in robot-world calibration.

## Citing

VisualGeometryToolkitEstimation is part of VisualGeometryToolkit. Cite the toolkit; its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry (GitHub's "Cite this repository" button exports it):

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Please also cite the papers behind the methods you use. They are listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).

## License

MIT
