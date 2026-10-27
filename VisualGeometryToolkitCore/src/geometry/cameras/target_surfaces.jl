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

# The rectangle in the design chart: the board, or the nominal points' box —
# grown by the rounding margin (`_grown_rect`), the one `PlanarTarget` checks
# its points against the board with.
function _design_rect(t::PlanarTarget, physical::Bool)
    bd = board(t)
    bd === nothing || return _grown_rect(Rect(bd))
    is_resolved(t) || throw(ArgumentError("surface: the target $(repr(reference(t))) is unresolved and has no board to bound its chart"))
    isempty(t.nominal) && throw(ArgumentError("surface: the target has no design points and no board to bound its chart"))
    plain(p) = physical ? _ustrip_point(m, p) : p     # no copy of the points
    lo = mapreduce(plain, (p, q) -> min.(p, q), t.nominal)
    hi = mapreduce(plain, (p, q) -> max.(p, q), t.nominal)
    return _grown_rect(lo, hi)
end

# The rectangle [lo, hi] grown by a rounding margin, 16 eps of its coordinates:
# a ray at a design point on its edge meets it, whatever the last bit of
# chart(s, x) = A⁻¹(m − b), and a point a unit round trip moved by an ulp is
# still on it.
function _grown_rect(lo::StaticVector{2}, hi::StaticVector{2})
    δ = 16 * eps(Float64) * max(maximum(abs, lo), maximum(abs, hi), hi[1] - lo[1], hi[2] - lo[2])
    return Rect2(lo[1] - δ, lo[2] - δ, hi[1] - lo[1] + 2δ, hi[2] - lo[2] + 2δ)
end
_grown_rect(r::Rect2) = _grown_rect(SVector{2}(r.origin), SVector{2}(r.origin + r.widths))

# The in-plane calibration as the chart's affine, m = A c + b, read off the
# calibration itself (`is_affine`): b its image of the design origin, A its
# point Jacobian there — the calibration's own composition, so a new affine
# kind needs no method here. A HeightField leaves the plane point as it is.
function _chart_affine(c::TargetCalibration)
    is_affine(c) || _refuse_chart(c)
    p, o = _in_plane(c), zero(SVector{2, Float64})
    return _point_jacobian(p, o), _apply(p, o)
end
_in_plane(c::TargetCalibration) = c
_in_plane(::HeightField) = Rigid()
_in_plane(c::CalibrationChain) = last(c.members) isa HeightField ? _chain_of(Base.front(c.members)) : c
_chain_of(members::Tuple{TargetCalibration}) = only(members)
_chain_of(members::Tuple) = CalibrationChain(members)
@noinline _refuse_chart(c::TargetCalibration) = throw(
    ArgumentError(
        _has_thin_plate(c) ?
            "surface: an in-plane ThinPlateSpline calibration is not supported by the ray primitives yet " *
            "(their chart is an affine of the design); the target's calibration has one" :
            "surface: no ray primitive for a $(nameof(typeof(c))) calibration"
    )
)
_has_thin_plate(c::TargetCalibration) = c isa ThinPlateSpline
_has_thin_plate(c::CalibrationChain) = any(_has_thin_plate, c.members)

_surface(::Val{true}, _, quad) = quad
_surface(::Val{false}, c, quad) = _height_surface(_last_height_field(c), quad)
_last_height_field(c::HeightField) = c
_last_height_field(c::CalibrationChain) = _last_height_field(last(c.members))
_height_surface(h::HeightField{K}, quad) where {K} = HeightSurface(quad, h.system.nodes, h.h, h.w, h.a)
