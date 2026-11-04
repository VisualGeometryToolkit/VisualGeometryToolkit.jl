# VisualGeometryToolkitCore.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitCore)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitCore)

VisualGeometryToolkitCore provides the types and algorithms of multiple-view
geometry: coordinate frames, camera and lens models, geometric primitives,
projective transforms and minimal solvers, with physical units throughout.
It is the base of the VisualGeometryToolkit packages, and it defines the
contracts that their robust estimators implement.

## Installation

```julia-repl
pkg> add VisualGeometryToolkitCore
```

Julia 1.12 or later. Every dependency is in the General registry.

## Example

A camera views a planar board. The example projects the board's points into
the image, estimates the board-to-image homography from those
correspondences, and maps a circle on the board to the ellipse it images as.

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

The [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/getting-started/)
guide continues from here.

## Documentation

- [Getting Started](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/getting-started/):
  installation, units, frames and a first camera.
- [User Guide](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/cameras/):
  cameras and lenses, the imaging chain, primitives, transforms, batches and
  robust estimation.
- [Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/):
  the rules every package of the stack follows, and why.
- [API Reference](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/api/frames/).

The other packages of the toolkit build on this one: VisualGeometryToolkitIO
(file formats), VisualGeometryToolkitFeatures (feature detection and
matching), VisualGeometryToolkitEstimation (RANSAC and the robust estimators)
and VisualGeometryToolkitEval (evaluation). The
[portal](https://visualgeometrytoolkit.github.io) links them all.

## Citing

VisualGeometryToolkitCore is written by James Pritts, with contributions from
Hendrik Sauer (cross-platform build and CI). Cite the toolkit. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Cite also the papers behind the methods you use; the
[portal](https://visualgeometrytoolkit.github.io/#publications) lists them.

## License

MIT
