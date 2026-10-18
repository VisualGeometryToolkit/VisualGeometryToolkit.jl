# ========================================================================
# Interval ↔ Range Conversions
# ========================================================================

"""
    range_interval(range; align_corners=false) -> ClosedInterval

Closed interval spanned by a range's samples. (An owned verb: the former
`ClosedInterval(::AbstractRange)` overload was type piracy of IntervalSets.)

- `align_corners=true`: endpoints at `[first(r), last(r)]`
- `align_corners=false`: range values are centers, extend by `±step/2`

```julia
range_interval(1:10; align_corners=true)   # [1, 10]
range_interval(1:10; align_corners=false)  # [0.5, 10.5]
```
"""
function range_interval(r::AbstractRange; align_corners::Bool = false)
    T = eltype(r)
    if align_corners
        return ClosedInterval(T(first(r)), T(last(r)))
    else
        T_out = T <: Integer ? Float64 : T
        s = T_out(step(r))
        return ClosedInterval(T_out(first(r)) - s / 2, T_out(last(r)) + s / 2)
    end
end

# ========================================================================
# Rect ↔ Interval Conversions
# ========================================================================

"""
    Rect((ix, iy)::Tuple{ClosedInterval, ClosedInterval}) -> Rect

Construct a Rect from x and y intervals.

```julia
ix, iy = ClosedInterval(0.5, 640.5), ClosedInterval(0.5, 480.5)
rect = Rect((ix, iy))  # origin=(0.5, 0.5), widths=(640, 480)
```
"""
function GeometryBasics.Rect(intervals::Tuple{ClosedInterval, ClosedInterval})
    ix, iy = intervals
    T = promote_type(eltype(ix), eltype(iy))
    origin = Point2{T}(T(leftendpoint(ix)), T(leftendpoint(iy)))
    widths = Vec2{T}(
        T(rightendpoint(ix) - leftendpoint(ix)),
        T(rightendpoint(iy) - leftendpoint(iy))
    )
    return GeometryBasics.Rect(origin, widths)
end

# ========================================================================
# Interval → Discrete Ranges (for sampling)
# ========================================================================

# ========================================================================
# Bounding Box Constructors for Primitives
# ========================================================================

"""
    Rect2(e::Ellipse; snap_to_grid=false) -> Rect2

Construct axis-aligned bounding box for an Ellipse.

# Arguments
- `snap_to_grid::Bool=false`: If true, floor/ceil coordinates for pixel alignment

```julia
e = Ellipse(Point2(100.0, 100.0), 50.0, 30.0, π/4)
bbox = Rect2(e)
bbox_aligned = Rect2(e; snap_to_grid=true)
```
"""
function GeometryBasics.Rect2(e::Ellipse{T}; snap_to_grid::Bool = false) where {T}
    min_pt, max_pt = extrema(e)
    return _make_rect2(T, min_pt, max_pt, snap_to_grid)
end

"""
    Rect2(d::Disc; snap_to_grid=false) -> Rect2

Construct axis-aligned bounding box for a Disc.

# Arguments
- `snap_to_grid::Bool=false`: If true, floor/ceil coordinates for pixel alignment

```julia
d = Disc(Point2(100.0, 200.0), 5.0)
bbox = Rect2(d)
bbox_aligned = Rect2(d; snap_to_grid=true)
```
"""
function GeometryBasics.Rect2(d::Disc{T}; snap_to_grid::Bool = false) where {T}
    min_pt, max_pt = extrema(d)
    return _make_rect2(T, min_pt, max_pt, snap_to_grid)
end

"""
    Rect2(c::GeometryBasics.Circle; snap_to_grid=false) -> Rect2

Construct axis-aligned bounding box for a Circle.

# Arguments
- `snap_to_grid::Bool=false`: If true, floor/ceil coordinates for pixel alignment

```julia
c = Circle(Point2(100.0, 100.0), 50.0)
bbox = Rect2(c)
bbox_aligned = Rect2(c; snap_to_grid=true)
```
"""
function GeometryBasics.Rect2(c::GeometryBasics.Circle{T}; snap_to_grid::Bool = false) where {T}
    return _make_rect2(T, _circle_extrema(c)..., snap_to_grid)
end

# A circle's axis-aligned extrema (origin ∓ r), the one spelling.
@inline function _circle_extrema(c::GeometryBasics.Circle{T}) where {T}
    o = GeometryBasics.origin(c); r = GeometryBasics.radius(c)
    return Point2{T}(o[1] - r, o[2] - r), Point2{T}(o[1] + r, o[2] + r)
end

# THE grid snap of a box, 2-D or 3-D: its low corner floored and its high
# corner ceiled (outward) — in each coordinate's own type, or as `R` (`Int` for
# an integer box).
@inline _snap_outward(lo, hi) = (map(floor, lo), map(ceil, hi))
@inline _snap_outward(::Type{R}, lo, hi) where {R} = (map(x -> floor(R, x), lo), map(x -> ceil(R, x), hi))

# Internal helper for snap_to_grid logic
function _make_rect2(::Type{T}, min_pt, max_pt, snap_to_grid::Bool) where {T}
    snap_to_grid || return GeometryBasics.Rect2{T}(min_pt, max_pt - min_pt)
    lo, hi = _snap_outward(T, Point2{T}(min_pt), Point2{T}(max_pt))
    return GeometryBasics.Rect2{T}(lo, hi - lo)
end

# ========================================================================
# Integer Bounding Boxes — Rect2{Int}(shape)
# ========================================================================
#
# Requesting Rect2{Int} implies pixel-grid alignment: floor for min, ceil for max.
# This is type-stable (output type determined by type parameter, not runtime kwarg).

# NOTE: A two-point Rect2{Int}(min_pt, max_pt) constructor is impossible because
# Point2 === Vec2 in GeometryBasics, so it would shadow Rect2{Int}(origin, widths).

@inline function _int_rect2(min_pt, max_pt)
    lo, hi = _snap_outward(Int, Point2(min_pt[1], min_pt[2]), Point2(max_pt[1], max_pt[2]))
    return GeometryBasics.Rect2{Int}(Point2{Int}(lo), Point2{Int}(hi) - Point2{Int}(lo))
end

"""
    Rect2{Int}(c::Circle) -> Rect2{Int}

Integer bounding box for a Circle.

```julia
c = Circle(Point2(100.3, 200.7), 5.2)
rect = Rect2{Int}(c)   # Rect2{Int}((95, 195), (11, 12))
(x0, y0), (w, h) = origin(rect), widths(rect)
patch = img[y0:(y0 + h), x0:(x0 + w)]
```
"""
(::Type{GeometryBasics.Rect2{Int}})(c::GeometryBasics.Circle) = _int_rect2(_circle_extrema(c)...)

"""
    Rect2{Int}(e::Ellipse) -> Rect2{Int}

Integer bounding box for an Ellipse.
"""
function (::Type{GeometryBasics.Rect2{Int}})(e::Ellipse)
    return _int_rect2(extrema(e)...)
end

"""
    Rect2{Int}(d::Disc) -> Rect2{Int}

Integer bounding box for a Disc.
"""
function (::Type{GeometryBasics.Rect2{Int}})(d::Disc)
    return _int_rect2(extrema(d)...)
end

# ========================================================================
# GeometryBasics.Rect ↔ SpatialIndexing.Rect Conversions
# ========================================================================

"""
    SpatialIndexing.Rect(r::GeometryBasics.Rect2) -> SpatialIndexing.Rect

Convert GeometryBasics Rect2 to SpatialIndexing Rect.
"""
function SpatialIndexing.Rect(r::GeometryBasics.Rect2{T}) where {T}
    lo = Tuple(r.origin)
    hi = Tuple(r.origin + r.widths)
    return SpatialIndexing.Rect(lo, hi)
end

"""
    SpatialIndexing.Rect(r::GeometryBasics.Rect3) -> SpatialIndexing.Rect

Convert GeometryBasics Rect3 to SpatialIndexing Rect.
"""
function SpatialIndexing.Rect(r::GeometryBasics.Rect3{T}) where {T}
    lo = Tuple(r.origin)
    hi = Tuple(r.origin + r.widths)
    return SpatialIndexing.Rect(lo, hi)
end

"""
    GeometryBasics.Rect2(r::SpatialIndexing.Rect) -> GeometryBasics.Rect2

Convert SpatialIndexing Rect to GeometryBasics Rect2.
"""
function GeometryBasics.Rect2(r::SpatialIndexing.Rect{T, 2}) where {T}
    origin = Point2{T}(r.low...)
    widths = Vec2{T}((r.high[1] - r.low[1], r.high[2] - r.low[2]))
    return GeometryBasics.Rect2{T}(origin, widths)
end

"""
    GeometryBasics.Rect3(r::SpatialIndexing.Rect) -> GeometryBasics.Rect3

Convert SpatialIndexing Rect to GeometryBasics Rect3.
"""
function GeometryBasics.Rect3(r::SpatialIndexing.Rect{T, 3}) where {T}
    origin = Point3{T}(r.low...)
    widths = Vec3{T}((r.high[1] - r.low[1], r.high[2] - r.low[2], r.high[3] - r.low[3]))
    return GeometryBasics.Rect3{T}(origin, widths)
end

"""
    rect_corners(rect::HyperRectangle{2}) -> NTuple{4, SVector{2}}
    rect_corners(o::SVector{2}, w::SVector{2})

The four corners of `rect` (origin `o`, widths `w`), counter-clockwise from its
origin: `o, o + (w₁, 0), o + w, o + (0, w₂)`, in the rectangle's element type.
The ONE corner order: every extent check ([`is_orientation_preserving`](@ref),
the pose / radial-homography extent gates in VisualGeometryToolkitEstimation), every
per-edge outline (`_rect_edge_samples`) and the ellipse/circle–rect
overlaps walk it.
"""
rect_corners(rect::GeometryBasics.HyperRectangle{2, T}) where {T} =
    rect_corners(SVector{2, T}(rect.origin), SVector{2, T}(rect.widths))
@inline rect_corners(o::SVector{2, T}, w::SVector{2, T}) where {T} =
    (o, SVector{2, T}(o[1] + w[1], o[2]), o + w, SVector{2, T}(o[1], o[2] + w[2]))

# The two per-edge grids of a rect walk: CLOSED samples both corners of an edge
# (so neighbouring edges share their corner — what per-edge extrema need), and
# HALF-OPEN samples the first corner and stops short of the next (so a polygon
# walk hits every corner exactly once).
struct Closed end
struct HalfOpen end

"""
    EdgeGrid{Closed}(n)
    EdgeGrid{HalfOpen}(n)

`n` evenly spaced samples along one rect edge ([`rect_edges`](@ref)): `Closed`
includes both of the edge's corners, `HalfOpen` its first corner and not the
next.
"""
struct EdgeGrid{E}
    n::Int
end
_edge_params(g::EdgeGrid{Closed}) = range(0, 1; length = g.n)
_edge_params(g::EdgeGrid{HalfOpen}) = range(0, 1; length = g.n + 1)[1:(end - 1)]

"""
    rect_edges(rect, grid::EdgeGrid) -> NTuple{4, Vector{SVector{2}}}

The samples along each edge of `rect`, counter-clockwise from its origin in
[`rect_corners`](@ref)' order — (y = y₀, x = x₁, y = y₁, x = x₀) for a rect
`[x₀, x₁] × [y₀, y₁]` — each edge from its corner toward the next on `grid`.
THE one per-edge rect walk: a camera's rect polygon and a radial model's board
outline walk it half-open, the undistorted/valid-rect frame walk closed (and
per edge, for per-edge extrema).
"""
function rect_edges(rect, grid::EdgeGrid)
    corners = rect_corners(rect)
    ts = _edge_params(grid)
    return ntuple(4) do i
        a, b = corners[i], corners[mod1(i + 1, 4)]
        [a .+ t .* (b .- a) for t in ts]
    end
end

"""
    _rect_edge_samples(f, rect, n) -> Vector{Point2{Float64}}

[`rect_edges`](@ref) on `EdgeGrid{HalfOpen}(n)`, every sample pushed through
`f`: a sample `f` maps to `nothing` (a point the map cannot image) is DROPPED;
the rest are kept as `Point2{Float64}`. Behind a camera's rect polygon and a
radial model's board outline.
"""
function _rect_edge_samples(f::F, rect, n::Int) where {F}
    pts = Point2{Float64}[]
    sizehint!(pts, 4n)
    for edge in rect_edges(rect, EdgeGrid{HalfOpen}(n)), p in edge
        d = f(p)
        d === nothing || push!(pts, Point2{Float64}(d[1], d[2]))
    end
    return pts
end

"""
    boundary_point(r::Rect, θ::Real) -> Point2

Return a point on the rectangle's boundary at parameter θ, walked
counter-clockwise from `origin(r)` at CONSTANT ARC LENGTH — so
`boundary_points(r, n)` gives `n` evenly spaced perimeter samples, exactly as
it does for an `Ellipse` or a `Circle`.

`has_boundary(::Rect)` already answers `HasBoundary()`, but the method it
promises was missing, so `boundary_points(rect, n)` threw — which is what
BlobBoards' Makie overlay calls.

CONSTANT ARC LENGTH IS NOT ALWAYS WHAT A RECT CALLER WANTS: `radial_outline`
and a camera's rect polygon sample PER EDGE (`rect_edges`), and the
undistorted-rect frame walk per edge too, so that every
corner is hit exactly — an
arc-length walk straddles the corners of a non-square rect (4x2 at n = 20 steps
by 0.6, landing at 3.6 then 4.2) — and `valid_rect`'s loop needs to know which
edge each sample came from to take per-edge inward extrema.
"""
function boundary_point(r::GeometryBasics.Rect{2, T}, θ::Real) where {T}
    o = GeometryBasics.origin(r)
    w = GeometryBasics.widths(r)
    s = mod(θ, 2π) * (w[1] + w[2]) / π                # arc length from the origin corner
    s <= w[1] && return Point2{T}(o[1] + s, o[2])         # bottom, left → right
    s -= w[1]
    s <= w[2] && return Point2{T}(o[1] + w[1], o[2] + s)  # right, bottom → top
    s -= w[2]
    s <= w[1] && return Point2{T}(o[1] + w[1] - s, o[2] + w[2])  # top, right → left
    s -= w[1]
    return Point2{T}(o[1], o[2] + w[2] - s)               # left, top → bottom
end
