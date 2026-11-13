# =============================================================================
# Bounding-box helpers (shape → SpatialIndexing.Rect)
# =============================================================================
#
# Moved here from VisualGeometryToolkitFeatures/nms.jl: pure geometry with no
# detection dependency, used by BlobBoards (pattern placement, annotation)
# and anything doing R-tree queries over feature shapes.

import SpatialIndexing

"""
    bbox(f::FeatureFrame; scale=1.0, snap_to_grid=false) -> SpatialIndexing.Rect

Bounding box of a feature frame at its `scale`-σ contour, optionally snapped
outward to the pixel grid, through the coordinate compute boundary (a frame in
quantities is stripped: pixels in px, lengths in mm). Every frame, on its DOF
trait: a `PointFrame`'s is the point (no extent at any `scale`), an isotropic
frame's `center ∓ scale·σ` (allocation-free), a shape-factor frame's its
`region(f, scale)` ellipse's exact box.

# Example
```julia
d = Disc(100.5, 200.5, 5.0)
rect = bbox(d; scale=3.0, snap_to_grid=true)
```
"""
@inline bbox(f::FeatureFrame; scale::Real = 1.0, snap_to_grid::Bool = false) =
    _frame_bbox(frame_scale(f), f, float(scale), snap_to_grid)
@inline function _frame_bbox(::PointShape, f, s, snap)
    cx, cy = map(float, _strip_coords((f.center[1], f.center[2])))
    return _bbox_rect((cx, cy), (zero(cx), zero(cy)), snap)
end
@inline function _frame_bbox(::IsotropicScale, d, s, snap)
    cx, cy, σ = map(float, _strip_coords((d.center[1], d.center[2], d.σ)))
    r = σ * s
    return _bbox_rect((cx, cy), (r, r), snap)
end
@inline _frame_bbox(::ShapeFactorScale, f, s, snap) = SpatialIndexing.Rect(Rect2(region(f, s); snap_to_grid = snap))   # region strips
@inline function _bbox_rect((cx, cy), (rx, ry), snap)
    lo, hi = (cx - rx, cy - ry), (cx + rx, cy + ry)
    snap && ((lo, hi) = _snap_outward(lo, hi))
    return SpatialIndexing.Rect(lo, hi)
end

"""
    bbox(shape; scale=1.0, snap_to_grid=false) -> SpatialIndexing.Rect

Generic shape bounding box (`Circle`, `Ellipse`, …) via the
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
