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
(so [`bbox`](@ref) works on it). Built by `Base.intersect` (`∩`) of a `Rect`, an
[`Ellipse`](@ref) or another intersection, in any order and nesting:

```julia
domain = Rect(0, 0, 1920, 1200) ∩ Ellipse(Point2(960.0, 600.0), 700.0, 700.0, 0.0)
Point2(960.0, 100.0) ∈ domain    # true: inside both
Point2(10.0, 10.0) ∈ domain      # false: in the rect, outside the ellipse
Rect2(domain)                    # the rect clipped to the ellipse's box
```

A `Rect` intersected with `nothing` is the `Rect` itself: `nothing` is "no
constraint" (what [`image_circle`](@ref) answers for a lens with no image
circle of its own).
"""
struct RegionIntersection{A, B}
    a::A
    b::B
end

# The regions `∩` combines. Rect ∩ Rect is GeometryBasics' eager method.
const _Region2 = Union{Ellipse, RegionIntersection}

Base.intersect(a::_Region2, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::Rect{2}, b::_Region2) = RegionIntersection(a, b)
Base.intersect(a::_Region2, b::Rect{2}) = RegionIntersection(a, b)

"""
    intersect(r::Rect, ::Nothing) -> Rect

`nothing` is no constraint: a rectangle intersected with it is the rectangle.
So `Rect(extent(camera)) ∩ image_circle(camera)` is the extent itself for a lens
with no image circle (`image_circle` answers `nothing`).
"""
Base.intersect(r::Rect{2}, ::Nothing) = r

Base.in(p::Point2, s::RegionIntersection) = p ∈ s.a && p ∈ s.b

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
