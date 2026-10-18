# Getting Started

This guide covers installation and the concepts the rest of the
documentation assumes.

## Installation

Install VisualGeometryToolkitCore from the General registry (Julia 1.12+):

```julia
using Pkg
Pkg.add("VisualGeometryToolkitCore")
```

For visualization support, also install GLMakie:

```julia
Pkg.add("GLMakie")
```

## Basic Usage

```julia
using VisualGeometryToolkitCore
```

With visualization:
```julia
using VisualGeometryToolkitCore
using GLMakie  # Activates the plotting extension
```

## Quick Start

The README's first example, run by the docs build (a doctest):

```jldoctest quickstart
julia> using VisualGeometryToolkitCore

julia> ellipse = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, π/4);

julia> circle = Circle(Point2(1.0, 2.0), 1.5);

julia> line = Line2D(Point2(0.0, 0.0), Point2(1.0, 1.0));

julia> centroid(circle)
2-element GeometryBasics.Point{2, Float64} with indices SOneTo(2):
 1.0
 2.0

julia> distance(line, Point2(0.0, 1.0))   # signed distance
0.7071067811865475

julia> classify_conic(ellipse)
EllipseTrait()

julia> VisualGeometryToolkitCore.has_boundary(circle)   # public, not exported
VisualGeometryToolkitCore.HasBoundary()
```

## Key Concepts

### Physical Units

VisualGeometryToolkitCore uses
[FlexUnits.jl](https://github.com/Deduction42/FlexUnits.jl)
for physical quantities, with a pixel dimension added to SI:

```jldoctest units
julia> using VisualGeometryToolkitCore

julia> focal = 35.0mm; pixel_size = 3.45μm;   # lengths

julia> x_coord = 100.0px;                       # pixels (logical)

julia> ustrip(mm, 2.5cm)                        # strip in a named unit
25.0

julia> ustrip(px, 25.4mm * 300dpi)              # a length times a density is pixels
300.0
```

Values are stored in base units (metres, px): strip in a named unit,
`ustrip(mm, distance) == 50.0`. `ud"…"` and `uparse` parse unit strings, and
`dimension(q) == dimension(px)` checks a dimension at an API boundary.

### Coordinate Conventions

Different vision libraries place the pixel origin differently:

| frame | first-pixel center |
|-------|--------------------|
| `PixelFrame{:opencv}` | `(0, 0)` |
| `PixelFrame{:colmap}` | `(0.5, 0.5)` |
| `PixelFrame{:julia}`  | `(1, 1)` |

VGC math is COLMAP convention throughout. Origin conversion is a labeled
rigid map between pixel frames:

```julia
T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
pt_julia = T(pt_colmap)
```

See [Coordinate Frames](frames.md) for the frame-tag system this is part
of.

### Coordinate Frames

Maps carry `Src`/`Dst` frame tags at the type level; composition and
inversion are frame-checked, and unlabeled maps compose with anything (the
gradual-typing wildcard). A camera pose and its extrinsics are the same
`EuclideanMap` type under swapped labels — `inv(::CameraPose)` *is* a
`CameraExtrinsics`. This system is opt-in and pervades the package; the
[Coordinate Frames](frames.md) guide is the place to start.

### Module Structure

- **`VisualGeometryToolkitCore.Cameras`** — camera models, sensors, projection,
  stereo rigs, the imaging chain, undistortion.
- **`VisualGeometryToolkitCore.Primitives`** — geometric primitives.
- Transforms, frames, epipolar forms, solvers, and robust-estimation
  primitives are defined in the top-level module.

Commonly-used names are re-exported from the main module:

```julia
# These are equivalent:
camera = VisualGeometryToolkitCore.Cameras.Camera(model, pose)
camera = Camera(model, pose)  # Re-exported
```

Feature detection lives in VisualGeometryToolkitFeatures.jl; the estimation
engines live in VisualGeometryToolkitEstimation.jl (see
[Robust Estimation](estimators.md)).

### The `Attributed{V,A}` Pattern

Values that carry metadata use a single composition type:
`Attributed{V,A}` wraps a value `V` with attributes `A`, with property
access forwarded transparently. `Uncertain{V,M}` is the same type with a
covariance or precision matrix as the attributes.

```jldoctest attributed
julia> using VisualGeometryToolkitCore

julia> p = Attributed(Point2(320.5, 240.5), 1.35);   # a depth-attributed pixel, as consumed by transfer(rig, p)

julia> p.value                                        # the Point2
2-element GeometryBasics.Point{2, Float64} with indices SOneTo(2):
 320.5
 240.5

julia> p.attributes                                   # the depth
1.35

julia> u = Uncertain(Point2(100.0, 200.0), CovMat{2}(σ = 0.5));   # an uncertain point, Σ = σ²·I
```

See [Design: Attributed{V,A}](design.md) for the rationale.

## Next Steps

- [Design: Attributed{V,A}](design.md) — the unifying composition pattern
- [Coordinate Frames](frames.md) — the frame-tag system
- [Cameras](cameras.md) — camera models and projection
- [The Framed Imaging Chain](imaging-chain.md) — frames and stages
- [Geometric Primitives](primitives.md) — shapes and their transforms
- [Transforms](transforms.md) — coordinate transformations and warping
