# =============================================================================
# Shared Function Stubs
# =============================================================================
#
# Functions defined here can be extended by any submodule without conflicts.
#
# =============================================================================
# Argument Order (conventions/signatures.md)
# =============================================================================
#
# The model or shape first, then the data it acts on or is measured against,
# then parameters:
#
#   project(camera, X)             sampson_distance(F, u₁, u₂)
#   distance(plane, p)             undistort(model, img, rect)
#   boundary_point(ellipse, θ)     focal_length_for_fov(sensor, θ)
#
# A given model leads in task verbs too (`localize(camera, detections)`); a
# required tool leads (`pairwise_matrix(metric, d1, d2)`), a defaulted policy
# stays last.
#
# Mutation and types follow Julia's style guide: a function that modifies a Julia
# object argument ends in `!` (RNG/sampler state, IO/file writes and keyword
# scratch buffers excepted); only a function argument, an IO or an RNG precede
# the mutated inputs; every type argument is positional and comes next, before
# every input that is not mutated (`zeros(Float32, n)`, `relabel(Src, Dst, m)`,
# `undistort!(dst, T, model, src)`):
#   undistort!(dst, model, src)    image_warp!(dst, bm, src)
#   rand!(rng, A)                  block_coordinate_descent!(state, likelihood, method, prob, θ₀)
#
# A method added to a function VGC does not own keeps its owner's order
# (`p ∈ ellipse`: Base's `in(item, collection)`). A symmetric relation takes
# either order (`intersects(a, b)`). RobustVisualGeometry's RANSAC family leads
# with the problem (`test_model(problem, model, idx)`).
#
# =============================================================================
# Correspondences (the correspondence standard, adopted 2026-09-22)
# =============================================================================
#
# Estimators take correspondences, and dictionaries of them for groups:
#
#   ungrouped  est(models…, cs::AbstractVector{<:Cspond}; config…) — any
#              sources/targets container, a `Csponds` preferred. Per-
#              correspondence priors, weights and covariances are ATTRIBUTE
#              COLUMNS (`withattrs(cs; weights = w)`, read as `cs.score`),
#              never keywords; per-correspondence outputs come back as columns.
#   grouped    est(models…, groups::AbstractDict{K, <:AbstractVector{<:Cspond}}; config…)
#              — the key is the identity of the per-group unknown; the block
#              order is `ids = block_order(groups)` (sorted keys), stored on the result.
#   nested     a measured quantity paired with its point evidence is the element
#              `Cspond{S, <:AbstractVector{<:Cspond}}` (rig pose => its features).
#
# Given per-group models lead, keyed by the same `K` (key sets checked, an
# `ArgumentError` otherwise: `block_order(groups, given…)`). Results are keyed by `K`; a joint covariance is
# ordered `[shared | per-group in ids order]`. Keywords are configuration only:
# thresholds, iteration counts, policies, and the DLT kernels' allocation-free
# `mask`/`frame`/`svd_ws`. A minimal solver's public protocol is
# `solve(solver, cs, idx)`; the positional kernels beneath it are kernels.
#
# =============================================================================

"""
    project(operator, geometry)

Project geometry via an operator.

Methods:
- `project(camera, point3d)`: Camera projection (3D → 2D)

See also: `closest_point(hyperplane, p)`, the orthogonal projection onto a hyperplane.
"""
function project end

"""
    distance(geometry, p)

Distance from the point `p` to `geometry`.

Methods:
- `distance(hyperplane, point)`: Signed distance from point to hyperplane
"""
function distance end

"""
    closest_point(geometry, p)

Return the point on `geometry` closest to `p`.

Methods:
- `closest_point(hyperplane, point)`: Orthogonal projection onto hyperplane

See also: `distance` (the norm of `p - closest_point(g, p)` for convex `g`).
"""
function closest_point end

"""
    centroid(primitive)

Return the centroid (geometric center) of a primitive.

Note: Use `centroid` for the geometric center/midpoint. This is distinct from
`GeometryBasics.origin` which returns the corner/origin of a primitive.

Methods:
- `centroid(::Ellipse)` → `Point2{T}`
- `centroid(::Circle)` → `Point2{T}`
- `centroid(::Disc)` → `Point2{T}`
- `centroid(::Ellipse3D)` → `Point3{T}`
- `centroid(::Rect)` → `Point{N,T}` (midpoint, not origin)
"""
function centroid end

"""
    normal(primitive)

Return the normal vector of a geometric primitive.

Methods:
- `normal(::Hyperplane)` → unit normal vector
- `normal(::Ellipse3D)` → unit normal of the ellipse's plane
"""
function normal end

"""
    offset(hyperplane)

Return the signed distance from the origin to the hyperplane.

For a hyperplane with coefficients [n₁, n₂, ..., nₘ, d]:
- `offset(h) = -d` (signed distance along normal)
"""
function offset end

"""
    dilate(primitive, factor)

Dilate (scale) a geometric primitive by a factor.

Methods:
- `dilate(::Circle, factor)` → Circle with scaled radius
- `dilate(::Ellipse, factor)` → Ellipse with scaled semi-axes
"""
function dilate end

"""
    intersects(a, b)

Test whether two geometric primitives intersect.

Methods:
- `intersects(::Rect, ::Rect)` → Bool
- `intersects(::Circle, ::Rect)` → Bool
"""
function intersects end

"""
    moment_matrix(g)

Return the moment matrix M = S·S' (or A·A') of a blob-like primitive.

This is the symmetric positive-definite matrix defining the elliptical shape
(second moment of the blob's spatial extent), NOT a statistical covariance.

The ellipse boundary is `{x : (x-c)ᵀ M⁻¹ (x-c) = 1}`.

Methods:
- `moment_matrix(::EllipticFrame)` → `SMatrix{2,2,T,4}`
- `moment_matrix(::AffineFrame)` → `SMatrix{2,2,T,4}`
- `moment_matrix(::Ellipse)` → `SMatrix{2,2,T,4}`
"""
function moment_matrix end

"""
    characteristic_scale(g)

Return the equivalent isotropic scale of a Gaussian: σ = det(Σ)^(1/4).

This is the geometric mean of the standard deviations along the principal axes.

Methods:
- `characteristic_scale(::EllipticFrame)` → `T`
- `characteristic_scale(::AffineFrame)` → `T`
"""
function characteristic_scale end

"""
    aspect_ratio(g)

Return the aspect ratio (a/b) of an ellipse-like shape.

For moment-matrix-based types (MvNormal, EllipticFrame): sqrt(λ_max / λ_min).
For geometric types (Ellipse): a/b (semi-major / semi-minor).

A value of 1.0 indicates isotropy (circular shape).

Methods:
- `aspect_ratio(::Ellipse)` → `T`
- `aspect_ratio(::Circle)` → `T` (always 1.0)
- `aspect_ratio(::EllipticFrame)` → `T`
- `aspect_ratio(::AffineFrame)` → `T`
"""
function aspect_ratio end

"""
    boundary_point(primitive, θ)

Return a point on the boundary of a primitive at parameter θ.

Requires `has_boundary(primitive) == HasBoundary()`.

Methods:
- `boundary_point(::Ellipse, θ)` → Point2
- `boundary_point(::Rect, θ)` → Point2 (constant arc length around the perimeter)
- `boundary_point(::Ellipse3D, θ)` → Point3
"""
function boundary_point end

"""
    boundary_points(primitive, n=64)

Return n points sampled uniformly around the boundary of a primitive.

Unlike `GeometryBasics.coordinates` (which returns mesh/face vertices for
rendering filled shapes), `boundary_points` samples the **outline** at
equispaced angles θ ∈ [0, 2π). Returns n distinct points (no closing
point duplication).

Requires `has_boundary(primitive) == HasBoundary()` and a corresponding
`boundary_point(primitive, θ)` method.

# Example
```julia
pts = boundary_points(ellipse, 32)  # 32 outline points
```
"""
function boundary_points(shape, n::Int = 64)
    return _boundary_points(has_boundary(shape), shape, n)
end

_boundary_points(::NoBoundary, shape, n) =
    error("$(typeof(shape)) does not have a boundary (has_boundary returns NoBoundary)")

function _boundary_points(::HasBoundary, shape, n::Int)
    return [boundary_point(shape, θ) for θ in range(0, 2π, length = n + 1)[1:(end - 1)]]
end

"""
    region(frame, k=1.0)

THE detection → primitive gate: the image region supported by a feature
frame at the `k`·σ contour, returned as a plain UNITLESS primitive (units
are stripped here, once, instead of ad-hoc `ustrip` calls in algorithms).

# Arguments
- `frame`: A `FeatureFrame` (Disc, OrientedDisc, EllipticFrame, AffineFrame)
- `k::Real=1.0`: Sigma multiplier (1.0 = 1σ contour)

# Returns
- `Circle` for `Disc` and `OrientedDisc`
- `Ellipse` for `EllipticFrame` and `AffineFrame`

Methods:
- `region(::Disc, k)` → `Circle`
- `region(::EllipticFrame, k)` → `Ellipse`
- `region(::AffineFrame, k)` → `Ellipse`
"""
function region end

"""
    boundary(center, Σ, n_sigma=1.0)

Construct an n-sigma confidence ellipse from a 2×2 covariance matrix.
For the detection → primitive gate on feature frames, use [`region`](@ref).
"""
function boundary end

"""
    scale(frame)

Return the characteristic scale of a feature frame.

For isotropic frames (Disc, OrientedDisc), this is the σ parameter.
For anisotropic frames, use `characteristic_scale` instead.

Methods:
- `scale(::Disc)` → `T`
- `scale(::OrientedDisc)` → `T`
"""
function scale end

"""
    shape_matrix(frame)

Return the shape matrix of an anisotropic feature frame.

For `EllipticFrame`, returns the symmetric shape matrix S where Σ = S².
For `AffineFrame`, returns the general shape matrix A where Σ = A·A'.

Methods:
- `shape_matrix(::EllipticFrame)` → `SMatrix{2,2}`
- `shape_matrix(::AffineFrame)` → `SMatrix{2,2}`
"""
function shape_matrix end

"""
    area(primitive)

Return the area of a 2D geometric primitive.

Accessor convention (single argument).

Methods:
- `area(::Ellipse)` → `T` (π * a * b)
- `area(::Circle)` → `T` (π * r²)
"""
function area end

"""
    overlap_area(a, b)

Compute the area of intersection between two geometric primitives.

Symmetric convention (order doesn't matter): arguments are put into
canonical order via [`geom_rank`](@ref) before dispatch.

Returns the area in squared units of the input coordinates.
Returns 0.0 if the shapes do not intersect.

Analytic methods (exact):
- `overlap_area(::Ellipse, ::Ellipse)` → `Float64` (GTE port)
- `overlap_area(::Circle, ::Circle)` → `Float64` (lens formula)
- `overlap_area(::Circle, ::Ellipse)` → `Float64` (via GTE)
- `overlap_area(::Ellipse, ::Rect)` → `Float64` (circle-polygon)
- `overlap_area(::Circle, ::Rect)` → `Float64` (circle-polygon)
- `overlap_area(::Circle/::Ellipse, ::Polygon)` → `Float64` (convex polygon)
- `overlap_area(::Rect, ::Rect)` → `Float64`

Pairs without an analytic method fall back to the
[`BoundarySampled`](@ref) overlap method (deterministic boundary
polygonization + convex clipping); see [`overlap_method`](@ref).
"""
function overlap_area end

"""
    overlap_method(a, b) -> OverlapMethod

Report how [`overlap_area`](@ref) computes the intersection area for this
pair of primitives: `AnalyticOverlap()` for pairs with an exact closed-form
method, `BoundarySampled()` for the deterministic polygonization fallback.
"""
function overlap_method end

"""
    geom_rank(::Type) -> Int

Canonical ordering rank of a geometric primitive type, used to symmetrize
binary predicates (`intersects`, `overlap_area`, `iou`): when a pair arrives
in non-canonical order (`geom_rank(T1) > geom_rank(T2)`), the arguments are
swapped and re-dispatched, so each unordered pair needs only one method.

Unranked types return `typemax(Int)`; a pair of unranked types is never
swapped and falls through to a `MethodError`.
"""
geom_rank(::Type) = typemax(Int)

"""
    iou(a, b)

Compute Intersection over Union between two geometric primitives.

Symmetric convention (order doesn't matter).

Derived from [`overlap_area`](@ref) and [`area`](@ref):
`iou = overlap / (area(a) + area(b) - overlap)`.
Returns a value in [0, 1] where 1 indicates identical shapes.
"""
function iou end

"""
    coverage(subject, reference)

Fraction of `subject`'s area covered by `reference`: the measured shape first,
the covering shape as its parameter.

Derived from [`overlap_area`](@ref):
`coverage = overlap_area(subject, reference) / area(subject)`.

Returns a value in [0, 1] where:
- 0 indicates no overlap
- 1 indicates `subject` is fully covered by `reference`

# Example
```julia
small = Ellipse(Point2(0.0, 0.0), 1.0, 0.5, 0.0)
large = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
coverage(small, large)  # 1.0 (small is inside large)
coverage(large, small)  # ~0.08 (only small portion of large is covered)
```
"""
function coverage end

# =============================================================================
# Stochastic Measurement Interface
# =============================================================================


# =============================================================================
# Residual precision
# =============================================================================
