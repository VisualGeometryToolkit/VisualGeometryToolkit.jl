# =============================================================================
# Region intersections — `a ∩ b` of two 2-D regions, held lazily
# =============================================================================
#
# A region here is any 2-D primitive that answers `p ∈ region` for a `Point2`
# and a bounding box `Rect2(region)`: a `Rect`, an `Ellipse`, or an
# intersection of them. `a ∩ b` holds both and answers membership in both; it
# never computes the boundary of the intersection.
#
# `Rect` is GeometryBasics' `HyperRectangle`: these `Base.intersect` methods
# extend a Base function over it, the deliberate glue the quality gates'
# piracy allowlist records (`GeometryBasics.HyperRectangle` in `treat_as_own`,
# test/quality_tests.jl) — the stack owns rect semantics. Rect ∩ Rect stays
# GeometryBasics' own method (an eager `Rect`), but Core meets rectangles by ONE
# convention, the CLOSED meet `_closed_meet` (below): rectangles that share an
# edge or a corner meet. GeometryBasics' strict `intersect` calls that empty;
# Core chooses closed because a region and its supports are closed sets — a
# node whose footprint touches the imaged region at its edge is free, and a
# region's bounding box must contain every rectangle `intersects` says it
# meets. `intersects`, a node's `support` clip and `Rect2(::RegionIntersection)`
# all read `_closed_meet`.

"""
    RegionIntersection{A, B}

The intersection `a ∩ b` of two 2-D regions, held lazily: a point is in it
when it is in both (`p ∈ s ⟺ p ∈ s.a && p ∈ s.b`), and its bounding box is
the intersection of the two bounding boxes, `Rect2(s) == Rect2(s.a) ∩ Rect2(s.b)`
(so [`bbox`](@ref VisualGeometryToolkitCore.bbox) works on it). Built by `Base.intersect` (`∩`) of a `Rect`, an
[`Ellipse`](@ref) or another intersection, in any order and nesting (a `Circle`
enters as its `Ellipse`):

```julia
domain = Rect(0, 0, 1920, 1200) ∩ Ellipse(Point2(960.0, 600.0), 700.0, 700.0, 0.0)
Point2(960.0, 100.0) ∈ domain    # true: inside both
Point2(10.0, 10.0) ∈ domain      # false: in the rect, outside the ellipse
Rect2(domain)                    # the rect clipped to the ellipse's box
```

The identity of `∩` is [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane):
`r ∩ WholePlane() === r` (what [`valid_domain`](@ref VisualGeometryToolkitCore.Cameras.valid_domain)
answers for a lens with no limit of its own).

Containment `s ⊆ region` (a `Rect`, an `Ellipse`, an intersection or the
`WholePlane`; a `Circle` as its `Ellipse`) never answers `true` for an `s` that
is not contained, to the rounding of its membership tests (and `Ellipse ⊆
Ellipse`'s relative slack of 8 eps). Into an intersection it is
`s ⊆ a && s ⊆ b`, as exact as those two. Into a `Rect` or an `Ellipse` it is
sufficient, not exact: `true` when the bounding box of `s` lies in the region
(or, for an `Ellipse`, when either part does), so neither part alone need fit.
That box is the meet of the parts' boxes, which can reach out of the region
where `s` does not; a true containment then answers `false`, and so does an
empty intersection (disjoint boxes).
"""
struct RegionIntersection{A, B}
    a::A
    b::B
end
# Equal parts, equal intersections (a part holding an array — a PolygonRegion — compares by value, not identity).
@fieldwise RegionIntersection

"""
    WholePlane()

The whole plane as a 2-D region: the identity of `∩`. Every point is in it
(`p ∈ WholePlane()`), every rectangle lies inside it (`r ∈ WholePlane()`), and
`a ∩ WholePlane() === a` for any region `a` (in either order). It has no
bounding box: `Rect2(WholePlane())` is an `ArgumentError`. It is what
[`valid_domain`](@ref VisualGeometryToolkitCore.Cameras.valid_domain) answers for a
lens with no limit of its own, so `Rect(extent) ∩ valid_domain(camera)` and
`Rect(extent) ∈ valid_domain(camera)` read the same for every lens, and what
[`imaged_region`](@ref VisualGeometryToolkitCore.Cameras.imaged_region) is given
when no image circle is known.
"""
struct WholePlane end

"""
    PolygonRegion(polygon::Polygon, center)
    PolygonRegion(vertices::AbstractVector{<:StaticVector{2}}, center)

A 2-D region bounded by a simple polygon that is STAR-SHAPED about `center`: its vertex ring `s.vertices`
(`Point2{Float64}`, counter-clockwise; a clockwise ring is reversed; `GeometryBasics.Polygon(s)` is it as a
GeometryBasics `Polygon`) in the region interface — `p ∈ s`, the bounding box `Rect2(s)`, `∩` (a
[`RegionIntersection`](@ref VisualGeometryToolkitCore.Primitives.RegionIntersection)), `⊆` and `∈` of a `Rect`,
`intersects(s, rect)`, `pixel_mask`, `dilate` about `center`. What
[`valid_domain`](@ref VisualGeometryToolkitCore.Cameras.valid_domain) answers for a lens that is not radially
symmetric: its boundary runs through each azimuth's own fold.

Membership is Core's one crossing-number test of a point in a polygon's ring (a `Polygon`'s `p ∈ polygon`), skipped
where it cannot change the answer: a point nearer `center` than every edge (`s.inner`) is in, one farther than every
vertex (`s.outer`) is out — so a pixel mask over a sensor tests the polygon only in the annulus between the two.

On a device: `Adapt.adapt(backend, s)` moves the vertex ring (the region is parametrized on its container, one
implementation for host and device), and `p ∈ s` evaluates inside a KernelAbstractions kernel as on the host; so does an
intersection holding it (an [`imaged_region`](@ref VisualGeometryToolkitCore.Cameras.imaged_region)). The other
operations (containment, `intersects`, `dilate`, `pixel_mask`) are host-side.

The constructor refuses (`ArgumentError`) a ring of fewer than 3 vertices, or one that is not star-shaped about `center`
(every edge must turn the same way about it, once round).
"""
struct PolygonRegion{V <: AbstractVector{Point2{Float64}}}
    vertices::V               # the ring, counter-clockwise about `center` (a device array once adapted)
    center::SVector{2, Float64}
    inner::Float64            # the distance from `center` to the nearest edge: the disc of that radius is inside
    outer::Float64            # the distance from `center` to the farthest vertex: the polygon lies in that disc
    lo::SVector{2, Float64}   # the bounding box: the vertices' extremes
    hi::SVector{2, Float64}
end
PolygonRegion(p::GeometryBasics.Polygon, center) = PolygonRegion(GeometryBasics.coordinates(p), center)
function PolygonRegion(vertices::AbstractVector{<:StaticVector{2}}, center)
    c = SVector{2, Float64}(center[1], center[2])
    ring = Point2{Float64}[Point2{Float64}(v[1], v[2]) for v in vertices]
    n = length(ring)
    n >= 3 || throw(ArgumentError("PolygonRegion: a polygon needs at least 3 vertices, got $n"))
    # star-shaped about c: every edge subtends a positive angle at c (counter-clockwise), the angles summing to one turn
    turn(i) = (a = ring[i] - c; b = ring[mod1(i + 1, n)] - c; atan(_cross2d(a, b), dot(a, b)))
    total = sum(turn, 1:n)
    total < 0 && (reverse!(ring); total = -total)
    (isapprox(total, 2π; rtol = 1.0e-9) && all(i -> turn(i) >= 0, 1:n)) || throw(
        ArgumentError("PolygonRegion: the polygon is not star-shaped about its centre $(Tuple(c))")
    )
    inner = minimum(i -> _segment_distance(c, ring[i], ring[mod1(i + 1, n)]), 1:n)
    outer = maximum(v -> norm(v - c), ring)
    lo = SVector(minimum(first, ring), minimum(last, ring))
    hi = SVector(maximum(first, ring), maximum(last, ring))
    return PolygonRegion{typeof(ring)}(ring, c, inner, outer, lo, hi)
end
GeometryBasics.Polygon(s::PolygonRegion) = GeometryBasics.Polygon(collect(s.vertices))
# The device form: the vertex ring moved, the discs and the box as they are (the same region, no re-validation).
Adapt.adapt_structure(to, s::PolygonRegion) = (v = Adapt.adapt(to, s.vertices); PolygonRegion{typeof(v)}(v, s.center, s.inner, s.outer, s.lo, s.hi))
# An intersection moves part by part, so an imaged region holding a polygon reaches a kernel whole.
Adapt.adapt_structure(to, s::RegionIntersection) = RegionIntersection(Adapt.adapt(to, s.a), Adapt.adapt(to, s.b))
# The distance from the point p to the closed segment ab.
function _segment_distance(p, a, b)
    d = SVector{2, Float64}(b - a)
    t = dot(d, d) > 0 ? clamp(dot(SVector{2, Float64}(p - a), d) / dot(d, d), 0.0, 1.0) : 0.0
    return norm(SVector{2, Float64}(p - a) - t * d)
end
_vertices(s::PolygonRegion) = s.vertices
_edges(s::PolygonRegion) = (v = _vertices(s); ((v[i], v[mod1(i + 1, length(v))]) for i in eachindex(v)))
Base.:(==)(a::PolygonRegion, b::PolygonRegion) = a.center == b.center && _vertices(a) == _vertices(b)
Base.hash(s::PolygonRegion, h::UInt) = hash(_vertices(s), hash(s.center, hash(:PolygonRegion, h)))
function Base.in(p::StaticVector{2}, s::PolygonRegion)
    d = hypot(p[1] - s.center[1], p[2] - s.center[2])
    d < s.inner && return true
    d > s.outer && return false
    return _point_in_ring(Point2{Float64}(p[1], p[2]), s.vertices)   # Core's one point-in-polygon test (operators.jl)
end
Base.in(r::Rect{2}, s::PolygonRegion) = r ⊆ s
dilate(s::PolygonRegion, factor::Real) = PolygonRegion([s.center + factor * (SVector{2, Float64}(v) - s.center) for v in _vertices(s)], s.center)
Base.show(io::IO, s::PolygonRegion) =
    print(io, "PolygonRegion(", length(_vertices(s)), " vertices about ", Tuple(s.center), ", radius ", s.inner, "–", s.outer, ")")

# Do the closed segments ab and cd share a point? The orientation test, with collinear overlaps counted.
function _segments_meet(a, b, c, d)
    o(p, q, r) = sign(_cross2d(q - p, r - p))
    on(p, q, r) = min(p[1], q[1]) <= r[1] <= max(p[1], q[1]) && min(p[2], q[2]) <= r[2] <= max(p[2], q[2])
    o1, o2, o3, o4 = o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)
    o1 != o2 && o3 != o4 && return true
    return (o1 == 0 && on(a, b, c)) || (o2 == 0 && on(a, b, d)) || (o3 == 0 && on(c, d, a)) || (o4 == 0 && on(c, d, b))
end
# Does a closed curve made of `segments` cross the polygon's boundary?
_boundary_meets(s::PolygonRegion, segments) = any(((a, b),) -> any(((c, d),) -> _segments_meet(a, b, c, d), _edges(s)), segments)
_rect_edges(r::Rect{2}) = (c = map(p -> Point2{Float64}(p[1], p[2]), rect_corners(GeometryBasics.Rect2{Float64}(r))); ((c[i], c[mod1(i + 1, 4)]) for i in 1:4))

# The regions `∩` combines. Rect ∩ Rect is GeometryBasics' eager method.
const _Region2 = Union{Ellipse, RegionIntersection, PolygonRegion}

Base.intersect(a::_Region2, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::Rect{2}, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::_Region2, b::Rect{2}) = RegionIntersection(a, b)
# A `Circle` (GeometryBasics' `HyperSphere{2}`) is the round `Ellipse`: it meets
# a region as `Ellipse(c)`, so its membership, bounding box and `intersects` are
# the ellipse's, one code path. Without these methods `Rect ∩ Circle` falls
# through to Base's iterable-set `intersect` (a `MethodError` on `iterate`).
Base.intersect(a::Union{Rect{2}, _Region2}, c::Circle) = a ∩ Ellipse(c)
Base.intersect(c::Circle, b::Union{Rect{2}, _Region2}) = Ellipse(c) ∩ b
Base.intersect(c₁::Circle, c₂::Circle) = Ellipse(c₁) ∩ Ellipse(c₂)
# the identity of ∩
Base.intersect(a::Union{Rect{2}, _Region2, Circle}, ::WholePlane) = a
Base.intersect(::WholePlane, b::Union{Rect{2}, _Region2, Circle}) = b
Base.intersect(w::WholePlane, ::WholePlane) = w

# A point is any `StaticVector{2}` (a `Point2` is one): an `SVector` asks the
# same question through the same method instead of falling through to Base's
# iterable `in`.
Base.in(p::StaticVector{2}, s::RegionIntersection) = p ∈ s.a && p ∈ s.b
Base.in(::StaticVector{2}, ::WholePlane) = true
# a rectangle in a region: in an intersection when in both parts (`⊆`, below: a Circle part as its Ellipse); in the
# whole plane always
Base.in(r::Rect{2}, s::RegionIntersection) = r ⊆ s
Base.in(::Rect{2}, ::WholePlane) = true
# THE region containment, `a ⊆ b` (`issubset`), every pair of the regions a fit and a support take — a `Rect`, an
# `Ellipse` (a `Circle` as its `Ellipse`), their `∩`, the `WholePlane` — so Base's iterable `issubset` is never reached.
# Sound, not always exact (RegionIntersection's docstring): an intersection lies in a region when its bounding box does
# (a ∩ b lies in its box), or, for an Ellipse region, when either part does; in an intersection when in both parts.
const _PlaneRegion = Union{Rect{2}, Ellipse, RegionIntersection, PolygonRegion, WholePlane}
Base.issubset(a::Rect{2}, b::Rect{2}) = a ∈ b                 # GeometryBasics' box-in-box
Base.issubset(a::Rect{2}, b::Ellipse) = a ∈ b                 # the corners (containment.jl): the ellipse is convex
Base.issubset(a::Ellipse, b::Rect{2}) = a ∈ b                 # the ellipse's exact bounding box (containment.jl)
# an ellipse in an ellipse: cameras/image_circle.jl (it reads the 1-D minimizer, numerics/minimize.jl, loaded later)
Base.issubset(a::Union{Rect{2}, Ellipse, RegionIntersection, PolygonRegion}, b::RegionIntersection) = a ⊆ b.a && a ⊆ b.b
# Into a star polygon, exact: a connected set lies in a simple polygon when one of its points does and its boundary
# meets none of the polygon's edges. A Rect, an Ellipse (its closed disc against each edge, in the frame where it is the
# unit circle) and a polygon are tested so; each first against the inscribed disc, which settles most cases at no cost.
function Base.issubset(a::Rect{2}, s::PolygonRegion)
    cs = rect_corners(GeometryBasics.Rect2{Float64}(a))
    all(p -> norm(SVector{2, Float64}(p) - s.center) < s.inner, cs) && return true
    return all(p -> Point2{Float64}(p) ∈ s, cs) && !_boundary_meets(s, _rect_edges(a))
end
function Base.issubset(e::Ellipse, s::PolygonRegion)
    e ⊆ Ellipse(Point2{Float64}(s.center), s.inner, s.inner, 0.0) && return true
    Point2{Float64}(e.center[1], e.center[2]) ∈ s || return false
    to_circle = _unit_circle_frame(e)
    return !any(((a, b),) -> _line_segment_intersects_unit_circle(to_circle(Float64(a[1]), Float64(a[2])), to_circle(Float64(b[1]), Float64(b[2]))), _edges(s))
end
function Base.issubset(a::PolygonRegion, s::PolygonRegion)
    a == s && return true
    return all(v -> v ∈ s, _vertices(a)) && !_boundary_meets(s, _edges(a))
end
# An intersection in a star polygon: sufficient, as into an Ellipse — its box, or either part.
function Base.issubset(a::RegionIntersection, s::PolygonRegion)
    lo, hi = _region_bounds(a)
    all(lo .<= hi) || return false
    all(isfinite, lo) && all(isfinite, hi) && GeometryBasics.Rect2{Float64}(lo, hi .- lo) ⊆ s && return true
    return a.a ⊆ s || a.b ⊆ s
end
# A star polygon in a convex region: its vertices (the polygon lies in their convex hull); in a Rect, its box.
Base.issubset(s::PolygonRegion, b::Ellipse) = all(v -> v ∈ b, _vertices(s))
Base.issubset(s::PolygonRegion, b::Rect{2}) = all(s.lo .>= minimum(b)) && all(s.hi .<= maximum(b))
# The box is compared at its parts' own bounds, never rebuilt as `origin + widths` (`Rect2(a)`), which rounds and can
# fall an ulp inside the region it bounds. Disjoint boxes (an empty intersection) make no claim.
function Base.issubset(a::RegionIntersection, b::Rect{2})
    lo, hi = _region_bounds(a)
    return all(lo .<= hi) && all(lo .>= minimum(b)) && all(hi .<= maximum(b))
end
function Base.issubset(a::RegionIntersection, b::Ellipse)
    lo, hi = _region_bounds(a)
    all(lo .<= hi) || return false
    corners = (Point2(lo[1], lo[2]), Point2(hi[1], lo[2]), Point2(lo[1], hi[2]), Point2(hi[1], hi[2]))
    all(isfinite, lo) && all(isfinite, hi) && all(p -> p ∈ b, corners) && return true   # the box, by its corners
    return a.a ⊆ b || a.b ⊆ b                                 # an Ellipse part may fit where its box does not
end
# a Circle is its Ellipse, as in `∩` (a part of an intersection built directly: VisualGeometryToolkitIO's lifted regions)
Base.issubset(c::Circle, b::Union{_PlaneRegion, Circle}) = Ellipse(c) ⊆ b
Base.issubset(a::_PlaneRegion, c::Circle) = a ⊆ Ellipse(c)

# The closed bounds `(lo, hi)` of a region's bounding box, each read where it is stored (a Rect's minimum and maximum,
# an Ellipse's analytic `extrema`); an intersection's are the meet of its parts' (`lo .> hi` on an axis: disjoint
# boxes), and a WholePlane part bounds nothing.
_region_bounds(r::Rect{2}) = (SVector{2, Float64}(minimum(r)), SVector{2, Float64}(maximum(r)))
_region_bounds(e::Ellipse) = ((l, h) = extrema(e); (SVector{2, Float64}(l), SVector{2, Float64}(h)))
_region_bounds(c::Circle) = _region_bounds(Ellipse(c))
_region_bounds(s::PolygonRegion) = (s.lo, s.hi)
_region_bounds(::WholePlane) = (SVector(-Inf, -Inf), SVector(Inf, Inf))
function _region_bounds(s::RegionIntersection)
    la, ha = _region_bounds(s.a)
    lb, hb = _region_bounds(s.b)
    return max.(la, lb), min.(ha, hb)
end
Base.issubset(::_PlaneRegion, ::WholePlane) = true
Base.issubset(::WholePlane, ::Union{Rect{2}, Ellipse, RegionIntersection, PolygonRegion}) = false


GeometryBasics.Rect2(::WholePlane; snap_to_grid::Bool = false) =
    throw(ArgumentError("the whole plane has no bounding box; intersect it with a Rect first"))
# A star polygon's bounding box: its vertices' extremes (snapped outward to the pixel grid when `snap_to_grid`).
GeometryBasics.Rect2(s::PolygonRegion; snap_to_grid::Bool = false) = _make_rect2(Float64, s.lo, s.hi, snap_to_grid)

"""
    Rect2(s::RegionIntersection; snap_to_grid = false) -> Rect2

The bounding box of an intersection: the intersection of its two parts'
boxes (each snapped outward to the pixel grid first when `snap_to_grid`),
closed: boxes that share only an edge meet in it. Disjoint boxes give
GeometryBasics' empty `Rect` (infinite origin, negative widths).
"""
GeometryBasics.Rect2(s::RegionIntersection; snap_to_grid::Bool = false) =
    something(_closed_meet(_region_box(s.a, snap_to_grid), _region_box(s.b, snap_to_grid)), GeometryBasics.Rect2{Float64}())

_region_box(r::Rect{2}, snap::Bool) =
    (b = GeometryBasics.Rect2{Float64}(r); snap ? _make_rect2(Float64, minimum(b), maximum(b), true) : b)
_region_box(x, snap::Bool) = GeometryBasics.Rect2(x; snap_to_grid = snap)

# -----------------------------------------------------------------------------
# Does a region meet a rectangle? (the support test of a local model's node)
# -----------------------------------------------------------------------------

"""
    intersects(region::Union{RegionIntersection, WholePlane}, r::Rect) -> Bool
    intersects(a::Rect, b::Rect) -> Bool

Do the closed region and the closed rectangle share a point? The whole plane
meets every rectangle. Two rectangles meet when their intersection is not
empty (touching edges count).

For an intersection the rectangle is first clipped by every `Rect` part of the
region (closed: a shared edge or corner counts); an empty clip is `false`. What remains
is tested against the curved parts:

- no curved part: `true` (exact);
- ONE curved part, an `Ellipse` — the imaged region `Rect(extent) ∩ circle`,
  and any nesting of it with more rectangles: `intersects(ellipse, clipped)`,
  exact, since the clipped rectangle lies inside every rectangle part; the
  same for ONE [`PolygonRegion`](@ref VisualGeometryToolkitCore.Primitives.PolygonRegion)
  (the imaged region of a camera whose lens is not radially symmetric);
- several curved parts (`Ellipse ∩ Ellipse`): each tested against the clipped
  rectangle separately — CONSERVATIVE: it may answer `true` when the parts meet
  the rectangle in disjoint places. A node judged free that way only costs a
  parameter; a node judged frozen is always outside the region.
"""
intersects(::WholePlane, ::Rect{2}) = true
intersects(r::Rect{2}, w::WholePlane) = intersects(w, r)
intersects(a::Rect{2}, b::Rect{2}) = _closed_meet(a, b) !== nothing
function intersects(s::RegionIntersection, r::Rect{2})
    c = _clip_by_rects(s, GeometryBasics.Rect2{Float64}(r))
    c === nothing && return false
    return all(p -> intersects(p, c), _curved_parts(s))
end
intersects(r::Rect{2}, s::RegionIntersection) = intersects(s, r)
# A star polygon meets a closed rectangle, exactly: a corner of the rectangle lies in it, a vertex of it in the
# rectangle, or their boundaries cross.
# Settled first by the bounding box (compared at its own bounds, never rebuilt as `origin + widths`) and by the discs
# the region holds: a rectangle whose nearest point to `center` lies past `outer` misses it, one that meets the inner
# disc meets it.
function intersects(s::PolygonRegion, r::Rect{2})
    b = GeometryBasics.Rect2{Float64}(r)
    lo, hi = SVector{2, Float64}(minimum(b)), SVector{2, Float64}(maximum(b))
    (all(lo .<= s.hi) && all(s.lo .<= hi)) || return false
    nearest = norm(clamp.(s.center, lo, hi) - s.center)
    nearest > s.outer && return false
    nearest < s.inner && return true
    any(p -> Point2{Float64}(p) ∈ s, rect_corners(b)) && return true
    any(v -> all(lo .<= v .<= hi), _vertices(s)) && return true
    return _boundary_meets(s, _rect_edges(b))
end
intersects(r::Rect{2}, s::PolygonRegion) = intersects(s, r)

# The rectangle clipped by every Rect part of a (nested) intersection.
# (`nothing` once empty), by the closed meet.
_clip_by_rects(s::RegionIntersection, r) = (c = _clip_by_rects(s.a, r); c === nothing ? nothing : _clip_by_rects(s.b, c))
_clip_by_rects(p::Rect{2}, r) = _closed_meet(p, r)
_clip_by_rects(_, r) = r
# The closed meet of two rectangles, `nothing` when they are disjoint — Core's one
# rect-meet convention (see the header): unlike GeometryBasics' `Rect ∩ Rect`, a
# shared edge or corner meets.
function _closed_meet(a::Rect{2}, b::Rect{2})
    lo = max.(Float64.(minimum(a)), Float64.(minimum(b)))
    hi = min.(Float64.(maximum(a)), Float64.(maximum(b)))
    return all(lo .<= hi) ? GeometryBasics.Rect2{Float64}(lo, hi .- lo) : nothing
end
# The parts that are not rectangles, flattened.
_curved_parts(s::RegionIntersection) = (_curved_parts(s.a)..., _curved_parts(s.b)...)
_curved_parts(::Rect{2}) = ()
_curved_parts(p) = (p,)
