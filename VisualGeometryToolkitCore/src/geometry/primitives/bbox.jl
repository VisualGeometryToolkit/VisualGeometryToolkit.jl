# =============================================================================
# Bounding-box helpers (shape → SpatialIndexing.Rect)
# =============================================================================
#
# Moved here from VisualGeometryToolkitFeatures/nms.jl: pure geometry with no
# detection dependency, used by BlobBoards (pattern placement, annotation)
# and anything doing R-tree queries over feature shapes.

import SpatialIndexing

"""
    bbox(d::Disc; scale=1.0, snap_to_grid=false) -> SpatialIndexing.Rect

Bounding box for a `Disc`, optionally scaled by `scale` around its center
and snapped outward to the pixel grid. Allocation-free.

# Example
```julia
d = Disc(100.5, 200.5, 5.0)
rect = bbox(d; scale=3.0, snap_to_grid=true)
```
"""
@inline function bbox(d::Disc; scale::Real = 1.0, snap_to_grid::Bool = false)
    cx, cy, σ = map(float, _strip_coords((d.center[1], d.center[2], d.σ)))
    r = σ * float(scale)
    lo = (cx - r, cy - r)
    hi = (cx + r, cy + r)
    snap_to_grid && ((lo, hi) = _snap_outward(lo, hi))
    return SpatialIndexing.Rect(lo, hi)
end

"""
    bbox(shape; scale=1.0, snap_to_grid=false) -> SpatialIndexing.Rect

Generic shape bounding box (`Circle`, `Ellipse`, feature frames, …) via the
shape's `Rect2` extent.
"""
function bbox(shape; scale::Real = 1.0, snap_to_grid::Bool = false)
    scaled = _scaled_boundary(shape, scale)
    return SpatialIndexing.Rect(Rect2(scaled; snap_to_grid = snap_to_grid))
end

# Internal: get scaled boundary shape
_scaled_boundary(shape, scale) = scale == 1.0 ? shape : _do_scale(shape, scale)
_do_scale(c::GeometryBasics.Circle, s) = dilate(c, s)
_do_scale(e::Ellipse, s) = dilate(e, s)
_do_scale(d::Disc, s) = region(d, s)
_do_scale(f::FeatureFrame, s) = region(f, s)
