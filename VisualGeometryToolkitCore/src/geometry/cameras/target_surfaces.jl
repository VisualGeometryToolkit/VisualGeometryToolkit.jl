# =============================================================================
# surface(::PosedTarget): the primitive a ray meets on a posed planar target
# =============================================================================
#
# A planar calibration is a Quad, a height field (possibly after in-plane
# members) a HeightSurface (geometry/primitives/surfaces.jl). The choice is
# made on the calibration's TYPE, so code over a rigid target never compiles
# the Newton loop and nothing tests the heights at run time.
#
# Frames: the primitive is in the pose's destination frame F, its chart is
# the target's own TargetFrame (x, y) — plain numbers, metres for a physical
# target. Its rectangle is the bounding box of the calibrated design's
# in-plane coordinates.
# =============================================================================

# (documented at the generic, geometry/primitives/surfaces.jl: Documenter
# reads a binding's docstring from its owner module)
function surface(pt::PosedTarget{<:PlanarTarget})
    p = has_units(pt) ? ustrip(m, pt) : pt
    t = model(p)
    c = geom(calibration(t))
    return _surface(Val(is_planar(typeof(c))), c, _surface_quad(geom(pose(p)), points(t)))
end

function _surface_quad(X::EuclideanMap{3}, pts)
    isempty(pts) && throw(ArgumentError("surface: the target has no design points to bound its chart"))
    R, o = rotation(X), translation(X)
    lo = reduce((a, b) -> min.(a, b), (SVector(p[1], p[2]) for p in pts))
    hi = reduce((a, b) -> max.(a, b), (SVector(p[1], p[2]) for p in pts))
    return Quad(o, R[:, 1], R[:, 2], Rect2(lo[1], lo[2], hi[1] - lo[1], hi[2] - lo[2]))
end

_surface(::Val{true}, _, quad) = quad
_surface(::Val{false}, c, quad) = _height_surface(_height_field(c), quad)
_height_field(c::HeightField) = c
_height_field(c::CalibrationChain) = _height_field(last(c.members))
_height_surface(h::HeightField{K}, quad) where {K} = HeightSurface(quad, h.system.nodes, h.h, h.w, h.a)
