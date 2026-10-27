# VisualGeometryToolkitCore.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitCore)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitCore)

A Julia package for computational geometry, camera modeling, and the robust-estimation core of the VisualGeometry stack. Provides geometric primitives, projective transformations, camera models, and minimal solvers, with Makie visualization. Feature detection lives in VisualGeometryToolkitFeatures.

## Features

- **Geometric Primitives**: Ellipse, Circle, Disc, Hyperplane (Line2D, Plane3D), Ellipse3D
- **Transforms**: Euclidean, similarity, affine, and projective (homography) transforms
- **Image Warping**: `image_warp(math, output_rect)` — single adapter that wraps a math-space transform into a Julia-callable `BoundedMap`. VGC math is COLMAP throughout; image_warp bakes the fixed COLMAP↔Julia bridges
- **Camera System**: Pinhole/fisheye/orthographic projection models, P3P pose estimation
- **Matching**: Correspondence types for feature matching (`Cspond`, `Attributed`, `ScoredCspond`)
- **Robust-estimation contracts**: the problem protocol, the scoring contract (with the `RansacScoring`/`MsacScoring` baselines), the local-optimization and external-refit contracts, the chart-rows contract, pool selection and the width axes — the interfaces VisualGeometryToolkitEstimation's RANSAC loop composes through, implemented by RANSACScoringDoneRight (scorers) and LocalOptimizationDoneRight (local optimizers)
- **Units**: Physical (mm, μm) and logical (px, dpi) unit support via FlexUnits.jl (a pixel is dimensionless)
- **Visualization**: Makie extension (any backend) with SpecApi composition

## Quick Start

```julia
using VisualGeometryToolkitCore
using GLMakie

# Geometric primitives
ellipse = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, π/4)
circle = Circle(Point2(1.0, 2.0), 1.5)
line = Line2D(Point2(0.0, 0.0), Point2(1.0, 1.0))

# Interface functions work on all primitives
centroid(ellipse)      # Point2(0.0, 0.0)
centroid(circle)       # Point2(1.0, 2.0)
distance(line, Point2(0.0, 1.0))  # Signed distance

# Traits for dispatch
classify_conic(ellipse)  # EllipseTrait()
VisualGeometryToolkitCore.has_boundary(circle)   # HasBoundary() (public, not exported)
```

## Transforms

```julia
using StaticArrays

# A homography maps a circle to a conic — an ellipse when the circle is in view
H = HomographyMap(HomographyMat(camera))
Q = H(circle)                                   # a Conic
ellipse = classify_conic(Q) isa EllipseTrait ? Ellipse(Q) : nothing

# Euclidean transforms preserve shape
E = EuclideanMap(RotMatrix{3}(RotX(π/4)), SVector(0.0, 0.0, 5.0))
e3d = Ellipse3D(ellipse, E)  # Embed 2D ellipse in 3D
```

## Image Warping

```julia
# img :: Matrix{Float32}; image_warp! resamples bilinearly into a pre-allocated array

# Generic warp with any math-space transform
H = HomographyMap(HomographyMat(matrix))
w = image_warp(inv(H), Rect(0, 0, width, height))
warped = image_warp!(Matrix{Float32}(undef, height, width), w, img; fillvalue=1.0f0)  # white out of bounds

# Canonical (affine-normalized) patches from ellipses
math  = canonical_map(ellipse, 64; bound=1.0)
w     = image_warp(math, 64)
patch = image_warp!(Matrix{Float32}(undef, 64, 64), w, img)

# Log-polar patches for rotation/scale invariance
math = logpolar_map(circle, 256, 0.01)
w    = image_warp(math, 256)
lp_patch = image_warp!(Matrix{Float32}(undef, 256, 256), w, img)
```

## Camera System

```julia
# Setup camera from sensor specs
s = sensor("Sony", "IMX174")
f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
model = CameraModel(f, s.pitch, centroid(Rect(s)) .* px)   # Rect(s) is COLMAP convention

# Posed camera
extrinsics = lookat(SVector(0.0, 0.0, 500.0), SVector(0.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0))
camera = PosedCamera(model, extrinsics)

# Project 3D points
pixel = project(camera, Point3(1.0, 2.0, 10.0))   # world point in mm, the canonical length (as the pose is)

# P3P pose estimation from 3D-2D correspondences
extrinsics, csponds = sample_p3p(model, world_points, image_rect, Plane3D())
```

## Visualization (Makie Extension)

```julia
using GLMakie   # any Makie backend loads the plotting extension; GLMakie adds window!/closeall
using VisualGeometryToolkitCore

# Simple Axis-based image display
fig, ax, plt = imshow(img)
scatter!(ax, [100, 200], [150, 250]; color=:red)

# SpecApi composition with campixel camera
using VisualGeometryToolkitCore: CampixelImshow, poly!, LScene3D, camera!

fig, ax, plt = imshow(img)
poly!(ax, ellipse; strokecolor=:blue)   # overlays on the Axis (forwards to Makie.poly!)

scene = CampixelImshow(img)             # SpecApi scene; overlays are pushed as S.Poly specs
scene3d = LScene3D()
camera!(scene3d, camera, image_rect; depth=frustum_depth, frustum_color=:orange)
# Correspondence overlays: matches!/matches (VisualGeometryToolkitFeatures extends them for its feature types).

import Makie.SpecApi as S
layout = S.GridLayout([(1,1) => scene, (1,2) => scene3d])
fig = plot(layout)
```

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitCore")
```

Requires Julia 1.12. Documentation:
<https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/>.

## Architecture

The package uses Holy Traits for extensible dispatch:

| Trait | Values | Purpose |
|-------|--------|---------|
| `ConicTrait` | `CircleTrait`, `EllipseTrait`, `ParabolaTrait`, `HyperbolaTrait`, `DegenerateTrait` | Conic classification (`classify_conic`) |
| `HasBoundaryTrait` | `HasBoundary`, `NoBoundary` | Boundary-sampling fallback |
| `OverlapMethod` | `AnalyticOverlap`, `BoundarySampled` | `overlap_area` method selection |
| `BoundsTrait` | `HasBounds`, `NoBounds` | Image-extent dispatch (`BoundedCamera`, `BoundedMap`) |
| `FrameScaleTrait` | `IsotropicScale`, `ShapeMatrixScale` | Feature-frame scale |
| `AbstractSolverBackend` | `NativeSolver`, `PoseLibSolver` | Minimal-solver backend |
| `FrameDependence` | `GeometryDefined`, `ModelDefined` | Frame-tag commensurability |

The transform class (Euclidean ⊂ similarity ⊂ affine ⊂ projective) is carried by the `AbstractHomMat` type hierarchy, not a trait.

Interface functions (`centroid`, `project`, `distance`, `dilate`, etc.) are defined once and extended by primitives.

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes
`Project.toml`, they re-resolve the environment (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## Contributors

VisualGeometryToolkitCore is written by James Pritts, with contributions from
Hendrik Sauer (cross-platform build and CI).

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

## License

MIT
