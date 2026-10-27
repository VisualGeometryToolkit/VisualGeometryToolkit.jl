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
# GeometryBasics' own method (an eager `Rect`).

"""
    RegionIntersection{A, B}

The intersection `a ∩ b` of two 2-D regions, held lazily: a point is in it
when it is in both (`p ∈ s ⟺ p ∈ s.a && p ∈ s.b`), and its bounding box is
the intersection of the two bounding boxes, `Rect2(s) == Rect2(s.a) ∩ Rect2(s.b)`
(so [`bbox`](@ref VisualGeometryToolkitCore.bbox) works on it). Built by `Base.intersect` (`∩`) of a `Rect`, an
[`Ellipse`](@ref) or another intersection, in any order and nesting:

```julia
domain = Rect(0, 0, 1920, 1200) ∩ Ellipse(Point2(960.0, 600.0), 700.0, 700.0, 0.0)
Point2(960.0, 100.0) ∈ domain    # true: inside both
Point2(10.0, 10.0) ∈ domain      # false: in the rect, outside the ellipse
Rect2(domain)                    # the rect clipped to the ellipse's box
```

The identity of `∩` is [`WholePlane`](@ref VisualGeometryToolkitCore.Primitives.WholePlane):
`r ∩ WholePlane() === r` (what [`image_circle`](@ref VisualGeometryToolkitCore.Cameras.image_circle)
answers for a lens with no image circle of its own).
"""
struct RegionIntersection{A, B}
    a::A
    b::B
end

"""
    WholePlane()

The whole plane as a 2-D region: the identity of `∩`. Every point is in it
(`p ∈ WholePlane()`), every rectangle lies inside it (`r ∈ WholePlane()`), and
`a ∩ WholePlane() === a` for any region `a` (in either order). It has no
bounding box: `Rect2(WholePlane())` is an `ArgumentError`. It is what
[`image_circle`](@ref VisualGeometryToolkitCore.Cameras.image_circle) answers for a
lens with no limit of its own, so `Rect(extent) ∩ image_circle(camera)` and
`Rect(extent) ∈ image_circle(camera)` read the same for every lens.
"""
struct WholePlane end

# The regions `∩` combines. Rect ∩ Rect is GeometryBasics' eager method.
const _Region2 = Union{Ellipse, RegionIntersection}

Base.intersect(a::_Region2, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::Rect{2}, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::_Region2, b::Rect{2}) = RegionIntersection(a, b)
# the identity of ∩
Base.intersect(a::Union{Rect{2}, _Region2}, ::WholePlane) = a
Base.intersect(::WholePlane, b::Union{Rect{2}, _Region2}) = b
Base.intersect(w::WholePlane, ::WholePlane) = w

Base.in(p::Point2, s::RegionIntersection) = p ∈ s.a && p ∈ s.b
Base.in(::Point2, ::WholePlane) = true
# a rectangle in a region: in an intersection when in both parts; in the whole plane always
Base.in(r::Rect{2}, s::RegionIntersection) = r ∈ s.a && r ∈ s.b
Base.in(::Rect{2}, ::WholePlane) = true

GeometryBasics.Rect2(::WholePlane; snap_to_grid::Bool = false) =
    throw(ArgumentError("the whole plane has no bounding box; intersect it with a Rect first"))

"""
    Rect2(s::RegionIntersection; snap_to_grid = false) -> Rect2

The bounding box of an intersection: the intersection of its two parts'
boxes (each snapped outward to the pixel grid first when `snap_to_grid`). An
empty intersection of boxes is GeometryBasics' empty `Rect` (infinite origin,
negative widths).
"""
GeometryBasics.Rect2(s::RegionIntersection; snap_to_grid::Bool = false) =
    intersect(_region_box(s.a, snap_to_grid), _region_box(s.b, snap_to_grid))

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
  exact, since the clipped rectangle lies inside every rectangle part;
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

# The rectangle clipped by every Rect part of a (nested) intersection.
# (`nothing` once empty). Closed rectangles: touching edges meet, unlike
# GeometryBasics' `Rect ∩ Rect`, which calls a shared edge empty.
_clip_by_rects(s::RegionIntersection, r) = (c = _clip_by_rects(s.a, r); c === nothing ? nothing : _clip_by_rects(s.b, c))
_clip_by_rects(p::Rect{2}, r) = _closed_meet(p, r)
_clip_by_rects(_, r) = r
function _closed_meet(a::Rect{2}, b::Rect{2})
    lo = max.(Float64.(minimum(a)), Float64.(minimum(b)))
    hi = min.(Float64.(maximum(a)), Float64.(maximum(b)))
    return all(lo .<= hi) ? GeometryBasics.Rect2{Float64}(lo, hi .- lo) : nothing
end
# The parts that are not rectangles, flattened.
_curved_parts(s::RegionIntersection) = (_curved_parts(s.a)..., _curved_parts(s.b)...)
_curved_parts(::Rect{2}) = ()
_curved_parts(p) = (p,)
