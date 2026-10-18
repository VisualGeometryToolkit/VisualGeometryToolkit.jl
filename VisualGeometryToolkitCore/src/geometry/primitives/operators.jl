# =============================================================================
# Translation Operators for Circles and Ellipses
# =============================================================================

"""
    Base.+(c::Circle, offset) -> Circle

Translate a circle by adding a 2D offset vector to its center.

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
offset = SVector(1.0, -2.0)
translated = circle + offset  # Circle at (6.0, 1.0) with radius 2.5
```
"""
Base.:+(c::GeometryBasics.Circle, offset::AbstractVector) =
    GeometryBasics.Circle(c.center + offset, c.r)

"""
    Base.-(c::Circle, offset) -> Circle

Translate a circle by subtracting a 2D offset vector from its center.

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
offset = SVector(1.0, -2.0)
translated = circle - offset  # Circle at (4.0, 5.0) with radius 2.5
```
"""
Base.:-(c::GeometryBasics.Circle, offset::AbstractVector) =
    GeometryBasics.Circle(c.center - offset, c.r)

"""
    Base.+(e::Ellipse, offset) -> Ellipse

Translate an ellipse by adding a 2D offset vector to its center.

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
offset = SVector(2.0, -1.0)
translated = ellipse + offset  # Ellipse at (12.0, 19.0) with same axes and orientation
```
"""
Base.:+(e::Ellipse{T}, offset::AbstractVector) where {T} =
    Ellipse{T}(e.center + offset, e.a, e.b, e.θ)

"""
    Base.-(e::Ellipse, offset) -> Ellipse

Translate an ellipse by subtracting a 2D offset vector from its center.

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
offset = SVector(2.0, -1.0)
translated = ellipse - offset  # Ellipse at (8.0, 21.0) with same axes and orientation
```
"""
Base.:-(e::Ellipse{T}, offset::AbstractVector) where {T} =
    Ellipse{T}(e.center - offset, e.a, e.b, e.θ)

# =============================================================================
# Uniform Scaling Operators for Circles and Ellipses
# =============================================================================

"""
    Base.:*(s::Real, c::Circle) -> Circle
    Base.:*(c::Circle, s::Real) -> Circle

Uniformly scale a circle by multiplying both its center position and radius.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `c::Circle`: Circle to scale

# Returns
- `Circle`: Scaled circle with both position and size scaled

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
scaled = 2.0 * circle  # Circle at (10.0, 6.0) with radius 5.0
```

See also: [`dilate`](@ref) to scale only radius while keeping center fixed.
"""
Base.:*(s::Real, c::GeometryBasics.Circle) =
    GeometryBasics.Circle(s * c.center, s * c.r)

Base.:*(c::GeometryBasics.Circle, s::Real) = s * c

"""
    Base.:*(s::Real, e::Ellipse) -> Ellipse
    Base.:*(e::Ellipse, s::Real) -> Ellipse

Uniformly scale an ellipse by multiplying center position and both semi-axes.

The orientation angle remains unchanged.

# Arguments
- `s::Real`: Scale factor (must be positive)
- `e::Ellipse`: Ellipse to scale

# Returns
- `Ellipse`: Scaled ellipse with both position and size scaled

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
scaled = 2.0 * ellipse  # Ellipse at (20.0, 40.0) with a=10.0, b=6.0, θ=π/4
```

See also: [`dilate`](@ref) to scale only axes while keeping center fixed.
"""
Base.:*(s::Real, e::Ellipse{T}) where {T} =
    Ellipse{T}(s * e.center, s * e.a, s * e.b, e.θ)

Base.:*(e::Ellipse{T}, s::Real) where {T} = s * e

# =============================================================================
# Dilation Operators for Circles and Ellipses
# =============================================================================

"""
    dilate(c::Circle, factor::Real) -> Circle

Dilate a circle by scaling its radius while keeping the center fixed.

# Arguments
- `c::Circle`: Circle to dilate
- `factor::Real`: Dilation factor (must be positive)

# Returns
- `Circle`: Dilated circle with same center but scaled radius

# Example
```julia
circle = Circle(Point2(5.0, 3.0), 2.5)
dilated = dilate(circle, 2.0)  # Circle at (5.0, 3.0) with radius 5.0

# Useful for patch extraction with context
patch_region = dilate(detected_circle, 3.0)
```

See also: [`*`](@ref) to scale both position and radius.
"""
dilate(c::GeometryBasics.Circle, factor::Real) =
    GeometryBasics.Circle(c.center, factor * c.r)

"""
    dilate(e::Ellipse, factor::Real) -> Ellipse

Dilate an ellipse by scaling both semi-axes while keeping center and orientation fixed.

Preserves the ellipse's aspect ratio and orientation while scaling its size.

# Arguments
- `e::Ellipse`: Ellipse to dilate
- `factor::Real`: Dilation factor (must be positive)

# Returns
- `Ellipse`: Dilated ellipse with same center and orientation but scaled axes

# Example
```julia
ellipse = Ellipse(Point2(10.0, 20.0), 5.0, 3.0, π/4)
dilated = dilate(ellipse, 1.5)  # Ellipse at (10.0, 20.0) with a=7.5, b=4.5, θ=π/4

# Useful for patch extraction with context
patch_region = dilate(detected_ellipse, 1.5)
transform = patch_to_ellipse_transform(patch_region, 128)
```

See also: [`*`](@ref) to scale both position and axes.
"""
dilate(e::Ellipse{T}, factor::Real) where {T} =
    Ellipse{T}(e.center, factor * e.a, factor * e.b, e.θ)

# =============================================================================
# Canonical Argument Ordering (geom_rank) and Symmetrization
# =============================================================================
#
# Binary symmetric predicates (`intersects`, `overlap_area`, and everything
# derived from them) implement each unordered type pair once, in canonical
# order: ascending `geom_rank`. The generic fallbacks below swap and
# re-dispatch when a pair arrives in non-canonical order. Unranked types
# (rank `typemax(Int)`) are never swapped, so a missing method surfaces as a
# MethodError instead of infinite recursion.

geom_rank(::Type{<:GeometryBasics.Circle}) = 10
geom_rank(::Type{<:Ellipse}) = 20
geom_rank(::Type{<:Rect}) = 30
geom_rank(::Type{<:GeometryBasics.Polygon}) = 40

"""
    intersects(a, b)

Generic fallback for automatic symmetrization of `intersects`.
If called with types in non-canonical `geom_rank` order, swaps and retries.
"""
function intersects(a::T1, b::T2) where {T1, T2}
    if geom_rank(T1) > geom_rank(T2)
        return intersects(b, a)
    else
        # Canonical order but no method - let Julia's dispatch handle the error
        throw(MethodError(intersects, (a, b)))
    end
end

# =============================================================================
# Circle Intersection
# =============================================================================

"""
    intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) -> Bool

Check if two circles intersect (distance between centers < sum of radii).
"""
intersects(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle) =
    norm(c1.center - c2.center) < (c1.r + c2.r)

"""
    intersects(p::Disc, q::Disc, cutoff) -> Bool

Check if two Disc frames intersect at radius `cutoff*σ`.
Two discs intersect if distance between centers < cutoff*(σ1 + σ2). Unit-carrying
discs must share one dimension (pixels or a length) — see `_strip_coords`.

Allocation-free: SVector operations are stack-allocated.
"""
@inline function intersects(p::Disc, q::Disc, cutoff::Real)
    px, py, pσ, qx, qy, qσ = _strip_coords((p.center[1], p.center[2], p.σ, q.center[1], q.center[2], q.σ))
    diff = SVector(px - qx, py - qy)
    dist_sq = dot(diff, diff)
    combined_r = cutoff * (pσ + qσ)
    return dist_sq < combined_r * combined_r
end

# =============================================================================
# Ellipse-Rectangle Intersection
# =============================================================================

# The map taking image coordinates into the frame where `e` is the unit circle
# at the origin — the inverse of `AffineMap(e)`, evaluated in Float64 — as a
# closure `(px, py) -> (u, v)`. Shared by the ellipse–rect and ellipse–polygon
# overlap paths. Allocation-free.
@inline function _unit_circle_frame(e::Ellipse)
    cx, cy = Float64(e.center[1]), Float64(e.center[2])
    cosθ, sinθ = cos(Float64(e.θ)), sin(Float64(e.θ))
    inv_a, inv_b = 1.0 / Float64(e.a), 1.0 / Float64(e.b)
    return @inline function (px, py)
        dx, dy = px - cx, py - cy
        return (
            (dx * cosθ + dy * sinθ) * inv_a,
            (-dx * sinθ + dy * cosθ) * inv_b,
        )
    end
end

# Transform the corners of `r` (counter-clockwise) into the coordinate frame
# where `e` is the unit circle at the origin. Allocation-free.
@inline function _rect_corners_circle_space(e::Ellipse, r::Rect)
    _to_circle = _unit_circle_frame(e)
    return map(c -> _to_circle(Float64(c[1]), Float64(c[2])), rect_corners(r))
end

"""
    intersects(e::Ellipse, r::Rect) -> Bool

Check if an ellipse intersects with a rectangle.

Uses an affine transformation approach:
1. Quick rejection via bounding box overlap check
2. Transform to canonical space where ellipse becomes unit circle
3. Test if transformed rectangle intersects the unit circle

This gives accurate results by checking:
- If any rectangle corner is inside the unit circle
- If any rectangle edge intersects the unit circle

# Example
```julia
rect = Rect2(0.0, 0.0, 200.0, 200.0)
ellipse1 = Ellipse(Point2(100.0, 100.0), 30.0, 20.0, π/4)
ellipse2 = Ellipse(Point2(250.0, 100.0), 30.0, 20.0, 0.0)

intersects(ellipse1, rect)  # true - center inside
intersects(ellipse2, rect)  # true - partially overlaps
```
"""
function intersects(e::Ellipse, r::Rect)
    min_e, max_e = extrema(e)
    min_r = minimum(r)
    max_r = maximum(r)

    # Quick rejection: check if bounding boxes overlap
    if !(
            min_e[1] <= max_r[1] && max_e[1] >= min_r[1] &&
                min_e[2] <= max_r[2] && max_e[2] >= min_r[2]
        )
        return false
    end

    # OGC semantics: intersects returns true if geometries share any point,
    # including when one is fully contained in the other.

    # Check if ellipse center is inside rect (handles ellipse-inside-rect case)
    c = centroid(e)
    if min_r[1] <= c[1] <= max_r[1] && min_r[2] <= c[2] <= max_r[2]
        return true
    end

    # Transform rect corners to the frame where the ellipse is the unit circle
    corners = _rect_corners_circle_space(e, r)

    # Check if any corner is inside the unit circle (handles rect-inside-ellipse case)
    for (x, y) in corners
        if x^2 + y^2 <= 1.0
            return true
        end
    end

    # Check if any edge intersects the unit circle (handles partial overlap)
    for i in 1:4
        p1 = corners[i]
        p2 = corners[i % 4 + 1]  # Next corner (wraps around)

        if _line_segment_intersects_unit_circle(p1, p2)
            return true
        end
    end

    return false
end

"""
    _line_segment_intersects_unit_circle(p1, p2) -> Bool

Check if a line segment from p1 to p2 intersects the unit circle centered at origin.
Uses the geometric distance from origin to line segment.
"""
function _line_segment_intersects_unit_circle(p1::Tuple{T, T}, p2::Tuple{T, T}) where {T}
    x1, y1 = p1
    x2, y2 = p2

    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    # Compute parameter t for closest point on line segment to origin
    # Closest point: p1 + t*(p2-p1), where t is clamped to [0,1]
    t = -(x1 * dx + y1 * dy) / (dx^2 + dy^2)
    t = clamp(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Check if closest point is inside unit circle
    return closest_x^2 + closest_y^2 <= 1.0
end

# =============================================================================
# Overlap Area (Analytical Core)
# =============================================================================
#
# `overlap_area` is the root of the predicate algebra: `iou` and `coverage`
# are derived from it (see below), so every pair with an analytic method here
# gets exact IoU and coverage for free.

"""
    overlap_area(e1::Ellipse, e2::Ellipse) -> Float64

Compute the area of intersection between two ellipses.

Analytical computation via a native port of the Geometric Tools Engine (GTE)
algorithm (see `ellipse_area_native.jl`), which handles all cases including
tangent ellipses and containment.

Symmetric: `overlap_area(e1, e2) == overlap_area(e2, e1)`

# Returns
Area of the intersection region. Returns 0.0 if ellipses don't intersect.

# Example
```julia
e1 = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
e2 = Ellipse(Point2(1.0, 0.0), 3.0, 2.0, 0.0)  # Overlapping
overlap_area(e1, e2)  # Analytical area of overlap
```
"""
function overlap_area(e1::Ellipse, e2::Ellipse)
    return _native_ellipse_intersection_area(
        e1.center[1], e1.center[2], e1.a, e1.b, e1.θ,
        e2.center[1], e2.center[2], e2.a, e2.b, e2.θ
    )
end

"""
    overlap_area(c1::Circle, c2::Circle) -> Float64

Compute the area of intersection between two circles using closed-form formula.

Symmetric: `overlap_area(c1, c2) == overlap_area(c2, c1)`

# Returns
Area of the intersection region (lens-shaped). Returns 0.0 if circles don't
intersect, or the area of the smaller circle if one contains the other.

# Algorithm
Uses the formula for circular segment areas:
    A = r₁²·acos(d₁/r₁) - d₁·√(r₁²-d₁²) + r₂²·acos(d₂/r₂) - d₂·√(r₂²-d₂²)
where d is the center distance and d₁, d₂ are distances to the radical axis.

# Example
```julia
c1 = Circle(Point2(0.0, 0.0), 2.0)
c2 = Circle(Point2(1.5, 0.0), 2.0)  # Overlapping
overlap_area(c1, c2)  # ~8.29
```
"""
function overlap_area(c1::GeometryBasics.Circle, c2::GeometryBasics.Circle)
    x1, y1, r1, x2, y2, r2 = _strip_coords((c1.center[1], c1.center[2], c1.r, c2.center[1], c2.center[2], c2.r))
    d = hypot(x1 - x2, y1 - y2)

    # No intersection: circles are separated
    d >= r1 + r2 && return zero(float(typeof(d + r1 + r2)))

    # One circle contains the other
    d <= abs(r1 - r2) && return π * min(r1, r2)^2

    # Partial intersection: compute lens area using circular segments
    d1 = (d^2 + r1^2 - r2^2) / (2d)
    d2 = d - d1

    return r1^2 * acos(clamp(d1 / r1, -1, 1)) - d1 * sqrt(max(0, r1^2 - d1^2)) +
        r2^2 * acos(clamp(d2 / r2, -1, 1)) - d2 * sqrt(max(0, r2^2 - d2^2))
end

"""
    overlap_area(c::Circle, e::Ellipse) -> Float64

Mixed circle-ellipse overlap via the exact ellipse-ellipse method.
"""
overlap_area(c::GeometryBasics.Circle, e::Ellipse) = overlap_area(Ellipse(c), e)

"""
    overlap_area(Q1::HomEllipseMat, Q2::HomEllipseMat) -> Float64

Overlap directly on homogeneous conic matrices. Each conic is converted to
an `Ellipse` once per call (one eigendecomposition each); callers comparing
one conic against many should convert once with `Ellipse(Q)` and reuse it.

Together with `area(::HomEllipseMat)`, this makes the derived predicates
(`iou`, `coverage`) work on conic matrices as well.
"""
overlap_area(Q1::HomEllipseMat, Q2::HomEllipseMat) = overlap_area(Ellipse(Q1), Ellipse(Q2))

area(Q::HomEllipseMat) = area(Ellipse(Q))

# =============================================================================
# Derived Predicates: IoU and Coverage
# =============================================================================

"""
    iou(a, b) -> Float64

Compute Intersection over Union between two geometric primitives.

Derived from [`overlap_area`](@ref): `iou = overlap / (area(a) + area(b) - overlap)`,
so it is exact for every pair with an analytic overlap method.

Symmetric: `iou(a, b) == iou(b, a)`

# Returns
IoU value in [0, 1] where 1 indicates identical shapes.

# Examples
```julia
e1 = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
e2 = Ellipse(Point2(1.0, 0.0), 3.0, 2.0, 0.0)
iou(e1, e2)  # ~0.7 (significant overlap)

c1 = Circle(Point2(0.0, 0.0), 2.0)
c2 = Circle(Point2(1.5, 0.0), 2.0)
iou(c1, c2)  # ~0.55
```
"""
function iou(a, b)
    inter = overlap_area(a, b)
    u = area(a) + area(b) - inter
    return u > 0 ? clamp(inter / u, 0.0, 1.0) : 0.0
end

"""
    coverage(subject, reference) -> Float64

Fraction of `subject`'s area covered by `reference`.

Derived from [`overlap_area`](@ref):
`coverage = overlap_area(subject, reference) / area(subject)`, so it is exact
for every pair with an analytic overlap method.

# Returns
Value in [0, 1] where:
- 0 indicates no overlap
- 1 indicates `subject` is fully covered by `reference`

# Example
```julia
small = Ellipse(Point2(0.0, 0.0), 1.0, 0.5, 0.0)
large = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.0)
coverage(small, large)  # 1.0 (small is covered by large)
coverage(large, small)  # ~0.08 (only small portion of large is covered)
```
"""
function coverage(subject, reference)
    A = area(subject)
    A > 0 || return 0.0
    return clamp(overlap_area(subject, reference) / A, 0.0, 1.0)
end

# =============================================================================
# Closed-form circle-convex polygon intersection area
# =============================================================================
#
# Used by overlap_area(Ellipse/Circle, Rect/Polygon): transform the ellipse
# to a unit circle (rect becomes a parallelogram, polygon stays a polygon),
# then compute circle ∩ convex-polygon area exactly.
#
# Algorithm: for each polygon edge, accumulate signed area contributions —
# triangle area where the boundary follows the polygon edge (inside circle),
# circular sector area where it follows the circle arc (outside circle).

@inline _dot2d(A::NTuple{2}, B::NTuple{2}) = A[1] * B[1] + A[2] * B[2]

@inline function _signed_sector_area(A::NTuple{2}, B::NTuple{2}, r²::Float64)
    return 0.5 * r² * atan(_cross2d(A, B), _dot2d(A, B))
end

"""Signed area contribution of edge A→B to circle(origin, r) ∩ polygon."""
function _edge_circle_contribution(A::NTuple{2, Float64}, B::NTuple{2, Float64}, r²::Float64)
    dA² = A[1]^2 + A[2]^2
    dB² = B[1]^2 + B[2]^2
    A_in = dA² ≤ r²
    B_in = dB² ≤ r²

    # Both inside circle: triangle area
    if A_in && B_in
        return 0.5 * _cross2d(A, B)
    end

    # Line parameterization: P(t) = A + t*(B-A), solve |P(t)|² = r²
    dx, dy = B[1] - A[1], B[2] - A[2]
    a = dx^2 + dy^2
    b_coef = 2.0 * (A[1] * dx + A[2] * dy)
    c_coef = dA² - r²
    disc = b_coef^2 - 4 * a * c_coef

    # Both outside, no intersection: circular sector
    if !A_in && !B_in && disc ≤ 0
        return _signed_sector_area(A, B, r²)
    end

    sq = sqrt(max(disc, 0.0))
    inv2a = 0.5 / a
    t1 = (-b_coef - sq) * inv2a
    t2 = (-b_coef + sq) * inv2a
    P1 = (A[1] + t1 * dx, A[2] + t1 * dy)
    P2 = (A[1] + t2 * dx, A[2] + t2 * dy)

    if A_in  # A inside, B outside → exit at P2
        return 0.5 * _cross2d(A, P2) + _signed_sector_area(P2, B, r²)
    elseif B_in  # A outside, B inside → enter at P1
        return _signed_sector_area(A, P1, r²) + 0.5 * _cross2d(P1, B)
    else  # Both outside, line goes through circle
        if 0 < t1 < 1 && 0 < t2 < 1
            return _signed_sector_area(A, P1, r²) + 0.5 * _cross2d(P1, P2) +
                _signed_sector_area(P2, B, r²)
        else
            return _signed_sector_area(A, B, r²)
        end
    end
end

"""Exact intersection area of circle at origin with convex polygon (a tuple or a vector of vertices)."""
function _circle_polygon_intersection_area(vertices::Union{Tuple, AbstractVector}, r²::Float64)
    n = length(vertices)
    total = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        total += _edge_circle_contribution(vertices[i], vertices[j], r²)
    end
    return abs(total)
end

"""
    overlap_area(e::Ellipse, r::Rect) -> Float64

Exact ellipse-rectangle intersection area: transforms the ellipse to a unit
circle (rect becomes a parallelogram), computes the circle-polygon area, and
scales back by the |Jacobian| `a*b`.
"""
function overlap_area(e::Ellipse, r::Rect)
    # Fast path: ellipse fully inside the rect
    e in r && return Float64(area(e))
    verts = _rect_corners_circle_space(e, r)
    return _circle_polygon_intersection_area(verts, 1.0) * Float64(e.a) * Float64(e.b)
end

"""
    overlap_area(c::Circle, r::Rect) -> Float64

Exact circle-rectangle intersection area via circle-polygon intersection.
"""
function overlap_area(c::GeometryBasics.Circle, r::Rect)
    cx, cy = Float64.(Tuple(GeometryBasics.origin(c)))
    rr = Float64(GeometryBasics.radius(c))
    verts = map(c -> (Float64(c[1]) - cx, Float64(c[2]) - cy), rect_corners(r))
    return _circle_polygon_intersection_area(verts, rr * rr)
end

"""
    overlap_area(r1::Rect, r2::Rect) -> Float64

Exact axis-aligned rectangle intersection area.
"""
function overlap_area(r1::Rect, r2::Rect)
    min1, max1 = minimum(r1), maximum(r1)
    min2, max2 = minimum(r2), maximum(r2)
    w = min(Float64(max1[1]), Float64(max2[1])) - max(Float64(min1[1]), Float64(min2[1]))
    h = min(Float64(max1[2]), Float64(max2[2])) - max(Float64(min1[2]), Float64(min2[2]))
    return (w > 0 && h > 0) ? w * h : 0.0
end

"""
    overlap_area(e::Ellipse, poly::Polygon) -> Float64
    overlap_area(c::Circle, poly::Polygon) -> Float64

Exact intersection area with a CONVEX polygon (no holes), via the same
circle-polygon machinery as the `Rect` methods. Concave polygons are not
detected and give wrong answers.
"""
function overlap_area(e::Ellipse, poly::GeometryBasics.Polygon)
    to_circle = _unit_circle_frame(e)
    verts = [to_circle(Float64(p[1]), Float64(p[2])) for p in GeometryBasics.coordinates(poly)]
    return _circle_polygon_intersection_area(verts, 1.0) * Float64(e.a) * Float64(e.b)
end

function overlap_area(c::GeometryBasics.Circle, poly::GeometryBasics.Polygon)
    cx, cy = Float64.(Tuple(GeometryBasics.origin(c)))
    rr = Float64(GeometryBasics.radius(c))
    verts = [(Float64(p[1]) - cx, Float64(p[2]) - cy) for p in GeometryBasics.coordinates(poly)]
    return _circle_polygon_intersection_area(verts, rr * rr)
end

# =============================================================================
# Area for Rect and Polygon (needed by the derived predicates)
# =============================================================================

"""
    area(r::Rect) -> Float64

Area of an axis-aligned rectangle.
"""
area(r::Rect) = prod(Float64.(Tuple(GeometryBasics.widths(r))))

"""
    area(poly::Polygon) -> Float64

Area of a simple polygon (no holes) via the shoelace formula.
"""
function area(poly::GeometryBasics.Polygon)
    ring = GeometryBasics.coordinates(poly)
    return abs(_shoelace_signed_area(ring))
end

function _shoelace_signed_area(ring)
    n = length(ring)
    total = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        total += Float64(ring[i][1]) * Float64(ring[j][2]) -
            Float64(ring[j][1]) * Float64(ring[i][2])
    end
    return 0.5 * total
end

# =============================================================================
# Overlap Fallback: symmetrization + deterministic boundary sampling
# =============================================================================

"""
    overlap_area(a, b)

Generic fallback: symmetrizes via [`geom_rank`](@ref), then routes pairs
without an analytic method to the [`BoundarySampled`](@ref) computation.
"""
function overlap_area(a::T1, b::T2) where {T1, T2}
    if geom_rank(T1) > geom_rank(T2)
        return overlap_area(b, a)
    end
    return overlap_area(overlap_method(a, b), a, b)
end

"""
    overlap_method(a, b) -> OverlapMethod

How `overlap_area` computes this pair. Pairs with closed-form methods
report `AnalyticOverlap()`; everything else reports the default
`BoundarySampled()` fallback.
"""
overlap_method(a, b) = BoundarySampled()
overlap_method(::Ellipse, ::Ellipse) = AnalyticOverlap()
overlap_method(::GeometryBasics.Circle, ::GeometryBasics.Circle) = AnalyticOverlap()
overlap_method(::GeometryBasics.Circle, ::Ellipse) = AnalyticOverlap()
overlap_method(::Ellipse, ::GeometryBasics.Circle) = AnalyticOverlap()
overlap_method(::Union{Ellipse, GeometryBasics.Circle}, ::Union{Rect, GeometryBasics.Polygon}) = AnalyticOverlap()
overlap_method(::Union{Rect, GeometryBasics.Polygon}, ::Union{Ellipse, GeometryBasics.Circle}) = AnalyticOverlap()
overlap_method(::Rect, ::Rect) = AnalyticOverlap()

# Boundary samples per shape of the BoundarySampled fallback.
const _OVERLAP_BOUNDARY_SAMPLES = 64

function overlap_area(::BoundarySampled, a, b)
    (has_boundary(a) isa HasBoundary && has_boundary(b) isa HasBoundary) ||
        throw(MethodError(overlap_area, (a, b)))
    # `boundary_points` allocates both vectors fresh for this call, so the
    # clipper may ADOPT them — the copying entry point would duplicate two
    # n-point arrays per call, and this path runs per pair in IoU loops.
    n = _OVERLAP_BOUNDARY_SAMPLES
    return _convex_clip_area!(
        _as_point2f(boundary_points(a, n)), _as_point2f(boundary_points(b, n))
    )
end

# Already the clipper's element type (the common case — a Float64 primitive's
# boundary): adopt as-is. Anything else converts, which allocates once.
_as_point2f(v::Vector{Point2{Float64}}) = v
_as_point2f(v) = Point2{Float64}[Point2{Float64}(p[1], p[2]) for p in v]

# AnalyticOverlap never reaches the trait-routed path (analytic pairs have
# direct methods), but a missing method here would recurse; fail loudly.
overlap_area(::AnalyticOverlap, a, b) = throw(MethodError(overlap_area, (a, b)))

"""
    _convex_clip_area!(subject, clip) -> Float64

Exact area of the intersection of two CONVEX polygons, via Sutherland-Hodgman
clipping + shoelace. ADOPTS both vectors: it reverses them in place to
normalize winding and consumes `subject` as one of the two clipping buffers
(the other a scratch vector, the two swapped per clip edge). Both must
be `Vector{Point2{Float64}}` (see `_as_point2f`) and neither is usable
afterwards.
"""
function _convex_clip_area!(subject::Vector{Point2{Float64}}, clip::Vector{Point2{Float64}})
    _shoelace_signed_area(subject) < 0 && reverse!(subject)
    _shoelace_signed_area(clip) < 0 && reverse!(clip)

    poly = subject
    m = length(clip)
    # a convex polygon clipped by a convex one has at most the sum of their vertices
    clipped = sizehint!(Point2{Float64}[], length(subject) + m)
    for i in 1:m
        isempty(poly) && return 0.0
        A = clip[i]
        B = clip[i == m ? 1 : i + 1]
        edge = B - A
        empty!(clipped)
        n = length(poly)
        for j in 1:n
            P = poly[j]
            Q = poly[j == n ? 1 : j + 1]
            sP = edge[1] * (P[2] - A[2]) - edge[2] * (P[1] - A[1])
            sQ = edge[1] * (Q[2] - A[2]) - edge[2] * (Q[1] - A[1])
            if sP >= 0
                push!(clipped, P)
                sQ < 0 && push!(clipped, P + (sP / (sP - sQ)) * (Q - P))
            elseif sQ >= 0
                push!(clipped, P + (sP / (sP - sQ)) * (Q - P))
            end
        end
        poly, clipped = clipped, poly
    end
    return abs(_shoelace_signed_area(poly))
end

# =============================================================================
# Point-in-Polygon (ray-casting / crossing-number algorithm)
# =============================================================================

"""
    Base.in(point::Point2, polygon::Polygon) -> Bool

Test whether a 2D point lies inside a polygon using the ray-casting algorithm.

Casts a ray from `point` in the +x direction and counts edge crossings.
An odd count means the point is inside.

Works with any `GeometryBasics.Polygon` (convex or concave, no holes).

# Example
```julia
quad = Polygon([Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)])
Point2f(5, 5) in quad   # true
Point2f(15, 5) in quad   # false
```
"""
function Base.in(p::Point2, poly::GeometryBasics.Polygon)
    ring = GeometryBasics.coordinates(poly)
    return _point_in_ring(p, ring)
end

function _point_in_ring(p::Point2, ring)
    px, py = Float64(p[1]), Float64(p[2])
    n = length(ring)
    inside = false
    j = n
    for i in 1:n
        yi, yj = Float64(ring[i][2]), Float64(ring[j][2])
        xi, xj = Float64(ring[i][1]), Float64(ring[j][1])
        if ((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    best_iou_match(query, candidates) -> (idx, iou_value)

The candidate with the highest [`iou`](@ref) against `query`, as
`(index, iou)`; `(0, 0.0)` when `candidates` is empty or no pair overlaps.
Works for any pair `iou` accepts (ellipses, circles, mixed).
"""
function best_iou_match(query, candidates::AbstractVector)
    best_idx = 0
    best_iou = 0.0
    for (k, c) in enumerate(candidates)
        v = iou(query, c)
        if v > best_iou
            best_iou = v
            best_idx = k
        end
    end
    return (best_idx, best_iou)
end
