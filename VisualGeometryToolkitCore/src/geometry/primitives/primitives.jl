"""
    Primitives

Geometric primitives submodule for VisualGeometryToolkitCore.

Provides geometric primitive types and operations:
- **Rectangles**: Rect constructors from intervals and ranges
- **Conics**: Ellipse and Circle types with conic matrix representations
- **Hyperplanes**: Lines in 2D, planes in 3D (generalized hyperplanes)
- **Traits**: Conic trait system for dispatch
- **Operations**: Geometric operations (dilate, intersect, containment)

# Main Types
- `Ellipse`, `Circle`, `Rect`
- `Hyperplane`, `Line2D`, `Plane3D`
- `ConicTrait`, `CircleTrait`, `EllipseTrait`

# Main Functions
- `dilate`, `intersects`, `gradient`
- `is_ellipse`, `classify_conic`
- `distance`, `closest_point`, `normal`, `offset`

# Example
```julia
using VisualGeometryToolkitCore.Primitives

ellipse = Ellipse(Point2(0.0, 0.0), 5.0, 3.0, π/4)
circle = Circle(Point2(1.0, 2.0), 2.0)
dilated = dilate(circle, 2.0)

# Hyperplanes
line = Line2D(Point2(0.0, 0.0), Point2(1.0, 1.0))
plane = Plane3D(Point3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0))
d = distance(line, Point2(1.0, 2.0))
```
"""
module Primitives

# Import general Julia packages directly (not through parent)
using StaticArrays: StaticArrays, SVector, SMatrix, @SMatrix, SOneTo, StaticVector, StaticMatrix
using StaticArrays.StaticArraysCore: StaticArray
using LinearAlgebra: LinearAlgebra, eigen, det, norm, dot, cross, Symmetric, normalize, svd, svdvals, Diagonal
using Statistics: Statistics
using IntervalSets: IntervalSets, ClosedInterval, leftendpoint, rightendpoint
import FlexUnits
using FlexUnits: Units, ustrip
using CoordinateTransformations: CoordinateTransformations, AffineMap, PerspectiveMap
using Rotations: Rotations
using Random: Random
using SpatialIndexing: SpatialIndexing

# Import from parent module (VisualGeometryToolkitCore-specific types only)
using ..VisualGeometryToolkitCore: GeometryBasics, Point, Point2, Point3
using ..VisualGeometryToolkitCore: Rect, Rect2, Vec2, Vec3
using ..VisualGeometryToolkitCore: to_affine, to_euclidean, _cross2d   # base/coordinates.jl
import ..VisualGeometryToolkitCore: Circle  # Import to extend
import ..VisualGeometryToolkitCore: center  # ONE `center` generic (geometry/radial_models.jl); FeatureFrame extends it

# Import transform types (defined in base/types.jl) - import to extend constructors
import ..VisualGeometryToolkitCore: HomEllipseMat, HomCircleMat, Conic

# Import canonical geometry constants (defined in base/interface.jl)

# NOTE: AffineMap constructor extensions for FeatureFrame live in
# primitives/feature_frames.jl (canonical → image space).

# Import detection types (needed for Circle construction)
using ..VisualGeometryToolkitCore: AbstractDetection, FeatureFrame, feature_frame
using ..VisualGeometryToolkitCore: Uncertain, PrecMat, CovMat, cov
using ..VisualGeometryToolkitCore: px, _strip_coords, _coord_unit   # the feature-coordinate compute boundary (base/feature_coords.jl)
import ..VisualGeometryToolkitCore: strip_or_bare   # extended for the feature frames (base/units.jl)
using ..VisualGeometryToolkitCore: propagate   # the congruence A^{-T}·Q·A^{-1} (base/propagate.jl)
using ..VisualGeometryToolkitCore: PlainEuclideanMap   # the primitives' rigid actions take plain maps; a physical one is a MethodError

# JSON serialization moved to VisualGeometryToolkitFeatures
using Accessors: @set

# Import interface traits and function stubs (to extend)
using ..VisualGeometryToolkitCore: HasBoundary, AnalyticOverlap, BoundarySampled
using ..VisualGeometryToolkitCore: CircleTrait, EllipseTrait, ParabolaTrait, HyperbolaTrait, DegenerateTrait
using ..VisualGeometryToolkitCore: IsotropicScale, ShapeMatrixScale
import ..VisualGeometryToolkitCore: has_boundary, classify_conic
import ..VisualGeometryToolkitCore: frame_scale
import ..VisualGeometryToolkitCore: centroid, normal, offset, distance, closest_point
import ..VisualGeometryToolkitCore: dilate, intersects, area, iou, coverage
import ..VisualGeometryToolkitCore: overlap_area, overlap_method, geom_rank
import ..VisualGeometryToolkitCore: moment_matrix, characteristic_scale, aspect_ratio
import ..VisualGeometryToolkitCore: boundary_point, boundary_points
import ..VisualGeometryToolkitCore: boundary, region, scale, shape_matrix

# Import MvNormal for geometry extensions
import Distributions: MvNormal
using Distributions: AbstractMvNormal

# =============================================================================
# INCLUDE ORDER - Dependency-aware loading within Primitives
# =============================================================================
#
# DEPENDENCY GRAPH (arrows show "depends on"):
#
#   conic.jl           (Ellipse, Circle, HomEllipseMat constructors)
#        ↓              No internal deps, but uses HomEllipseMat from parent
#   blob_types.jl      (FeatureFrame, Disc)
#        ↓              Uses: Ellipse (for boundary)
#   rectangles.jl      (Rect constructors from Ellipse, Circle, Disc)
#        ↓              Uses: Ellipse, Circle, Disc
#   feature_frames.jl  (OrientedDisc, EllipticFrame, AffineFrame)
#        ↓              Uses: Disc, Ellipse, MvNormal
#   containment.jl     (spatial containment queries)
#   operators.jl       (geometric operations: intersection, IOU, dilate)
#        ↓              Uses: Ellipse, Circle, Rect
#   hyperplane.jl      (Line2D, Plane3D)
#        ↓              Independent of conic types
#   ellipse3d.jl       (planar ellipse in 3D)
#   circle3d.jl        (planar circle in 3D)
#                       Uses: Ellipse, Circle, Hyperplane
#
# CRITICAL DEPENDENCIES:
# 1. conic.jl MUST be first (defines Ellipse used by everything else)
# 2. blob_types.jl MUST come after conic.jl (Disc boundary returns Ellipse)
# 3. rectangles.jl MUST come after blob_types.jl (Rect from Disc constructor)
# 4. feature_frames.jl MUST come after blob_types.jl (extends Disc to oriented frames)
# 5. operators.jl MUST come after conic.jl, rectangles.jl (IOU, intersection)
#
# =============================================================================

# Conic sections: Ellipse struct, Circle interface, conic traits, gradient
# NOTE: Extends HomEllipseMat/HomCircleMat imported from parent module
include("conic.jl")

# Blob types: FeatureFrame abstract type, Disc (3 DOF)
# NOTE: region(::Disc, k) returns Circle
include("blob_types.jl")

# Rectangle constructors: Rect from Ellipse, Circle, Disc
# NOTE: Depends on all conic and blob types being defined
include("rectangles.jl")

# Feature frames: OrientedDisc (4 DOF), EllipticFrame (5 DOF), AffineFrame (6 DOF)
# NOTE: Extends Disc with orientation and shape parameters
include("feature_frames.jl")

# MvNormal geometry extensions: geometric methods for Distributions.MvNormal
# NOTE: Provides characteristic_scale, Ellipse, Circle, Disc conversions for MvNormal
include("mvnormal_geometry.jl")

# Spatial queries: containment tests
include("containment.jl")

# Lazy region intersections: `a ∩ b` of a Rect, an Ellipse or another intersection
include("regions.jl")

# Geometric operators: overlap_area, IoU, dilate, coverage
# NOTE: Uses Ellipse, Circle for area computations
# Native GTE ellipse-ellipse intersection area (pure-Julia port; used by
# overlap_area in operators.jl)
include("ellipse_area_native.jl")

include("operators.jl")

# Hyperplanes: Line2D, Plane3D - generalized linear constraints
# NOTE: Independent of conic types, but uses Point2/Point3
include("hyperplane.jl")

# 3D primitives: planar shapes embedded in 3D space
# NOTE: Uses Ellipse, Circle, Hyperplane (Plane3D)
include("ellipse3d.jl")
include("circle3d.jl")

# ==============================================================================
# Trait Implementations
# ==============================================================================
# All Holy Trait implementations for primitives are collected in traits.jl
# NOTE: Must be included AFTER all primitive types are defined

include("traits.jl")

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================
#
# Note: Interface functions (centroid, normal, offset, closest_point, distance, dilate,
# intersects, plane, boundary_point, boundary_points,
# ranges) and traits (ConicTrait, etc.) are defined in
# base/interface.jl and exported from the main module.
#

# -----------------------------------------------------------------------------
# 1. CONIC SECTIONS - Ellipse and Circle types
# -----------------------------------------------------------------------------
export Ellipse, Circle
export homogeneous_w, maps_within
export best_iou_match
export is_ellipse, classify_conic, gradient
export RegionIntersection, WholePlane   # `a ∩ b` of two 2-D regions, lazily, and the identity of ∩ (regions.jl)

# -----------------------------------------------------------------------------
# 2. FEATURE FRAMES - Blob primitives with increasing DOF
# -----------------------------------------------------------------------------
export FeatureFrame            # Abstract base type
export Disc                    # 3 DOF: center + σ (isotropic)
export OrientedDisc            # 4 DOF: + orientation
export EllipticFrame           # 5 DOF: + aspect ratio (a, b, θ)
export Paraboloid3D            # Quadratic surface z = a₀ + b'x + x'Cx in ℝ³
export coeffs                  # Paraboloid3D{T,P} → SVector{P,T} free parameters
export is_valid_shape, DEFAULT_MAX_ASPECT_RATIO
export is_peak
export AffineFrame             # 6 DOF: + shear
# MvNormal from Distributions.jl - use EllipticFrame(d) for geometry

# -----------------------------------------------------------------------------
# 3. HYPERPLANES - Lines and planes
# -----------------------------------------------------------------------------
export Hyperplane, Line2D, Plane3D
export side
export slope, intercept, clip_to_rect
# Batched SoA ray-plane intersection (0.14; hoisted from RayTracedCalibration)
export ray_plane_intersection

# -----------------------------------------------------------------------------
# 4. 3D PRIMITIVES - Planar shapes in 3D space
# -----------------------------------------------------------------------------
export AbstractConic3D, Ellipse3D, Circle3D

end # module Primitives
