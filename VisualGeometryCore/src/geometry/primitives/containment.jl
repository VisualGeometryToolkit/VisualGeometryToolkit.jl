# =============================================================================
# Containment (Base.in extensions)
# =============================================================================

"""
    Base.in(e::Ellipse, r::Rect) -> Bool

Check if ellipse's axis-aligned bounding box is completely contained within rectangle.

Uses the analytical `extrema(e)` to compute the ellipse's bounding box efficiently,
then checks if that bounding box is inside the rectangle.

# Example
```julia
rect = Rect2(0.0, 0.0, 200.0, 200.0)
ellipse = Ellipse(Point2(100.0, 100.0), 30.0, 20.0, π/4)
ellipse in rect  # true if bounding box fits
```
"""
function Base.in(e::Ellipse, r::Rect)
    min_pt, max_pt = extrema(e)
    return min_pt in r && max_pt in r
end

"""
    Base.in(p::Point2, e::Ellipse) -> Bool

Check if point `p` is inside ellipse `e`. Enables `p ∈ e` syntax.

Uses the normalized ellipse equation: (x'/a)² + (y'/b)² ≤ 1
where (x', y') are coordinates in the ellipse's local frame.

# Example
```julia
ellipse = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, π/4)
Point2(0.0, 0.0) ∈ ellipse  # true (center)
Point2(5.0, 0.0) ∈ ellipse  # false (outside)
```
"""
function Base.in(p::Point2, e::Ellipse)
    # Transform point to ellipse-local coordinates (centered, axis-aligned)
    dx = p[1] - e.center[1]
    dy = p[2] - e.center[2]

    # Rotate by -θ to align with ellipse axes
    cosθ = cos(e.θ)
    sinθ = sin(e.θ)
    x_local = dx * cosθ + dy * sinθ
    y_local = -dx * sinθ + dy * cosθ

    # Check if inside: (x/a)² + (y/b)² ≤ 1
    return (x_local / e.a)^2 + (y_local / e.b)^2 <= 1.0
end

# =============================================================================
# Homography-mapped containment — the plane-visibility predicate
# =============================================================================

"""
    homogeneous_w(H::StaticMatrix{3, 3}, p::StaticVector{2}) -> Real

The homogeneous coordinate `H * [p; 1]` produces — the chirality reference
for [`to_euclidean`](@ref)'s guarded form and [`maps_within`](@ref).
Evaluate it once at a point known to be on the visible side (e.g. the
plane's own center) and supply it as `w_ref`.
"""
homogeneous_w(H::StaticMatrix{3, 3}, p::StaticVector{2}) =
    H[3, 1] * p[1] + H[3, 2] * p[2] + H[3, 3]

"""
    maps_within(H::StaticMatrix{3, 3}, rect::Rect, p::StaticVector{2}, w_ref::Real) -> Bool

Does `p` map through the homography `H` inside `rect` (closed bounds), on
the same side of the plane's horizon as `w_ref`? The one predicate behind
"does this image point back-project onto this board/plane region": points
beyond the horizon flip the homogeneous sign and would wrap into the
rectangle from behind the camera — `w_ref` (see [`homogeneous_w`](@ref))
keeps them out.
"""
function maps_within(
        H::StaticMatrix{3, 3}, rect::GeometryBasics.HyperRectangle{2},
        p::StaticVector{2}, w_ref::Real
    )
    q = to_euclidean(H * SVector(p[1], p[2], one(eltype(H))), w_ref)
    q === nothing && return false
    lo, hi = minimum(rect), maximum(rect)
    return (lo[1] <= q[1] <= hi[1]) & (lo[2] <= q[2] <= hi[2])
end
