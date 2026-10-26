# =============================================================================
# surface(::PosedTarget): the primitive a ray meets on a posed planar target
# =============================================================================
#
# A planar calibration (Rigid, Affine, a chain of them) is a Quad, a height
# field after them a HeightSurface (geometry/primitives/surfaces.jl). The
# kind is chosen by the calibration's TYPE, so code over a rigid target never
# compiles the Newton loop and nothing tests the heights at run time.
#
# Frames: the primitive is in the pose's destination frame F; its chart is
# the DESIGN chart (the nominal points' coordinates), the in-plane
# calibration being the quad's affine; its rectangle is the board's, or the
# nominal points' bounding box — plain numbers, metres for a physical target.
# An in-plane ThinPlateSpline is not an affine: not supported by the ray
# primitives in this wave.
# =============================================================================

# (documented at the generic, geometry/primitives/surfaces.jl: Documenter
# reads a binding's docstring from its owner module)
function surface(pt::PosedTarget{<:PlanarTarget})
    t = model(pt)
    physical = has_units(pt)
    c = geom(calibration(t))
    c = physical ? _ustrip_calibration(m, c) : c
    X = geom(pose(pt))
    X = physical ? ustrip(m, X) : X
    A, b = _chart_affine(c)
    quad = Quad(translation(X), rotation(X), _design_rect(t, physical), A, b)
    return _surface(Val(is_planar(typeof(c))), c, quad)
end

# The rectangle in the design chart: the board, or the nominal points' box.
function _design_rect(t::PlanarTarget, physical::Bool)
    bd = board(t)
    bd === nothing || return Rect(bd)
    is_resolved(t) || throw(ArgumentError("surface: the target $(repr(reference(t))) is unresolved and has no board to bound its chart"))
    ps = physical ? [_ustrip_point(m, p) for p in t.nominal] : t.nominal
    isempty(ps) && throw(ArgumentError("surface: the target has no design points and no board to bound its chart"))
    lo = reduce((p, q) -> min.(p, q), ps)
    hi = reduce((p, q) -> max.(p, q), ps)
    # grown by a rounding margin: a ray at a design point on the box's edge
    # meets it, whatever the last bit of chart(s, x) = A⁻¹(m − b)
    δ = 16 * eps(Float64) * max(maximum(abs, lo), maximum(abs, hi), hi[1] - lo[1], hi[2] - lo[2])
    return Rect2(lo[1] - δ, lo[2] - δ, hi[1] - lo[1] + 2δ, hi[2] - lo[2] + 2δ)
end

# The in-plane calibration as the chart's affine, m = A c + b: members applied
# in order; a HeightField leaves the plane point as it is.
_chart_affine(::Rigid) = (SMatrix{2, 2, Float64}(1, 0, 0, 1), SVector(0.0, 0.0))
_chart_affine(c::Affine) = (c.A, c.b)
_chart_affine(::HeightField) = _chart_affine(Rigid())
function _chart_affine(c::CalibrationChain)
    A, b = _chart_affine(Rigid())
    for member in c.members
        Aₘ, bₘ = _chart_affine(member)
        A, b = Aₘ * A, Aₘ * b + bₘ
    end
    return A, b
end
_chart_affine(::ThinPlateSpline) = throw(
    ArgumentError(
        "surface: an in-plane ThinPlateSpline calibration is not supported by the ray primitives yet " *
            "(their chart is an affine of the design); the target's calibration has one"
    )
)
_chart_affine(c::TargetCalibration) = throw(ArgumentError("surface: no ray primitive for a $(nameof(typeof(c))) calibration"))

_surface(::Val{true}, _, quad) = quad
_surface(::Val{false}, c, quad) = _height_surface(_last_height_field(c), quad)
_last_height_field(c::HeightField) = c
_last_height_field(c::CalibrationChain) = _last_height_field(last(c.members))
_height_surface(h::HeightField{K}, quad) where {K} = HeightSurface(quad, h.system.nodes, h.h, h.w, h.a)
