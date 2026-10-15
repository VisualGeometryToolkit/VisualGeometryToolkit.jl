# =============================================================================
# Holy Traits for Geometric Primitives
# =============================================================================
#
# This file defines behavioral traits for compile-time dispatch on geometric types.
# Traits are loaded early so all submodules can implement them consistently.
#
# =============================================================================


# -----------------------------------------------------------------------------
# HasBoundary Trait
# -----------------------------------------------------------------------------

"""
    HasBoundaryTrait

Trait indicating whether a type has a parametric boundary that can be sampled.
"""
abstract type HasBoundaryTrait end
struct HasBoundary <: HasBoundaryTrait end
struct NoBoundary <: HasBoundaryTrait end

"""
    has_boundary(x) -> HasBoundaryTrait

Return whether `x` has a boundary that can be sampled.
"""
has_boundary(x) = NoBoundary()  # default

# -----------------------------------------------------------------------------
# OverlapMethod Trait
# -----------------------------------------------------------------------------

"""
    OverlapMethod

Trait describing how `overlap_area` computes the intersection area for a
pair of primitives.

- `AnalyticOverlap()`: an exact closed-form method exists for the pair
- `BoundarySampled()`: deterministic fallback — both boundaries are
  polygonized at 64 samples and the convex polygons are clipped exactly

Query with `overlap_method(a, b)`.
"""
abstract type OverlapMethod end

"Exact closed-form overlap computation."
struct AnalyticOverlap <: OverlapMethod end

"Deterministic boundary-polygonization fallback (64 samples per shape)."
struct BoundarySampled <: OverlapMethod end

# -----------------------------------------------------------------------------
# HasBounds Trait
# -----------------------------------------------------------------------------

"""
    BoundsTrait

Trait indicating whether a type carries pixel-space image bounds (width × height).
Cameras parsed from SfM/MVS exports (COLMAP, ETH3D, ScanNet) typically wrap their
intrinsics+lens+pose in a `BoundedCamera` so downstream geometry code can `extent(cam)`,
`is_in_image(cam, p)`, and bound frustum visualizations.
"""
abstract type BoundsTrait end
struct HasBounds <: BoundsTrait end
struct NoBounds <: BoundsTrait end

"""
    bounds_trait(x) -> BoundsTrait

Return whether `x` carries pixel-space image bounds.
"""
bounds_trait(x) = NoBounds()  # default

# -----------------------------------------------------------------------------
# Conic Trait (for 2D conic sections)
# -----------------------------------------------------------------------------

"""
    ConicTrait

Trait type for dispatch on conic sections.

Subtypes (proper conics):
- `CircleTrait`: Circle (equal semi-axes, eccentricity = 0)
- `EllipseTrait`: Ellipse (eccentricity < 1)
- `ParabolaTrait`: Parabola (eccentricity = 1, tangent to line at infinity)
- `HyperbolaTrait`: Hyperbola (eccentricity > 1, crosses line at infinity twice)

Special:
- `DegenerateTrait`: Degenerate conic (point, line, or two lines)
"""
abstract type ConicTrait end

"Trait indicating a circle (conic with equal semi-axes)."
struct CircleTrait <: ConicTrait end

"Trait indicating an ellipse (conic with unequal semi-axes)."
struct EllipseTrait <: ConicTrait end

"Trait indicating a parabola (tangent to line at infinity)."
struct ParabolaTrait <: ConicTrait end

"Trait indicating a hyperbola (crosses line at infinity twice)."
struct HyperbolaTrait <: ConicTrait end

"Trait indicating a degenerate conic (point, line, or two lines)."
struct DegenerateTrait <: ConicTrait end

"""
    classify_conic(Q) -> ConicTrait

Classify a conic matrix and return the appropriate trait.

Returns one of:
- `CircleTrait()`: Circle
- `EllipseTrait()`: Ellipse
- `ParabolaTrait()`: Parabola
- `HyperbolaTrait()`: Hyperbola
- `DegenerateTrait()`: Degenerate conic (point, line, or two lines)

Implementations are provided for `HomEllipseMat` and `HomCircleMat`.
"""
function classify_conic end


# -----------------------------------------------------------------------------
# FrameScale Trait (isotropic vs shape-matrix feature frames)
# -----------------------------------------------------------------------------

"""
    FrameScaleTrait

Trait distinguishing feature frame scale types for dispatch.

- `IsotropicScale()`: Frames with a single scale σ (Disc, OrientedDisc)
- `ShapeMatrixScale()`: Frames with a 2×2 shape matrix (EllipticFrame, AffineFrame)
"""
abstract type FrameScaleTrait end

"Trait indicating an isotropic frame with single scale σ."
struct IsotropicScale <: FrameScaleTrait end

"Trait indicating a frame with a 2×2 shape matrix."
struct ShapeMatrixScale <: FrameScaleTrait end

"""
    frame_scale(x) -> FrameScaleTrait

Return the frame scale trait for a feature frame.
"""
function frame_scale end

# -----------------------------------------------------------------------------
# SolverBackend Trait - Minimal solver implementation dispatch
# -----------------------------------------------------------------------------

"""
    AbstractSolverBackend

Holy trait for selecting which implementation of a minimal solver to use.

Concrete subtypes:
- `NativeSolver()`: Pure-Julia implementations (default, zero-allocation)
- `PoseLibSolver()`: PoseLib C++ library via ccall

Carried as the `backend` field of the `MinimalSolver` verb layer (the public
solve protocol) and dispatched on by the positional kernels:
```julia
solve(Homography4pt(), cs, idx)                # default (NativeSolver)
solve(Homography4pt(PoseLibSolver()), cs, idx) # PoseLib backend
p3p(PoseLibSolver(), rays, X)                  # kernel-level backend call
```
"""
abstract type AbstractSolverBackend end

"Pure-Julia solver implementation (default)."
struct NativeSolver <: AbstractSolverBackend end

"PoseLib C++ solver via ccall."
struct PoseLibSolver <: AbstractSolverBackend end
