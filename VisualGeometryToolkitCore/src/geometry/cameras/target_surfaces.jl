# =============================================================================
# surface(::PosedTarget): the primitive a ray meets on a posed planar target
# =============================================================================
#
# A planar affine calibration (Rigid, the affine factors `EuclideanMap ∘ Scale
# ∘ Aspect ∘ Skew` or any of them) is a Quad, a height field (alone or after
# affine factors) a HeightSurface (geometry/primitives/surfaces.jl). The
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
    # (the bare calibration: an uncertain one's covariance is not the surface's, so it is not rescaled)
    c = geom(physical ? _calibration_unit_move(t, c -> _ustrip_calibration(m, geom(c))) : calibration(t))
    P = placement(t)
    X = geom(pose(pt))
    X = physical ? ustrip(m, X) : X
    P = P === nothing || !has_units(P) ? P : ustrip(m, P)
    A, b = _placed_affine(P, _chart_affine(c)...)
    quad = Quad(translation(X), rotation(X), _design_rect(t, physical), A, b)
    return _placed_surface(P, _surface(Val(is_planar(typeof(c))), c, quad))
end

# A placed target (`ChartPlacement` P, plain: p = o + s c): its calibration's
# in-plane affine y = J c + β in the chart is, on the design chart,
# p ↦ p + s (J c + β − c) = J p + (s β − (J − I) o) — the same J; b = s β −
# (J − I) o, exactly 0 with J = I when the chart affine is the identity (a
# height field alone), so its quad is the unplaced one bit for bit.
_placed_affine(::Nothing, J, β) = (J, β)
_placed_affine(P::ChartPlacement, J, β) = (J, P.side * β - (J - one(J)) * P.origin)
# Its height field over the quad's metric plane coordinates m = o + s c, where
# the height s·f(c) is exactly a thin-plate spline in m: with
# U(|c − cᵢ|) = U(|m − mᵢ|)/s² − |m − mᵢ|² ln s / s² and the spline's side
# conditions (Σwᵢ = 0, Σwᵢcᵢ = 0), the nodes o + s cᵢ, the heights s hᵢ, the
# weights wᵢ/s, and the affine part
#     a′ = (s a₁ − (a₂, a₃)·o − s ln s Σ wᵢ|cᵢ|²,  a₂,  a₃).
# (RadiometricCamera's `board_relief`, there with o = 0.) s = 1 keeps the
# weights; o = 0 and s = 1 keep the field.
_placed_surface(::Nothing, s) = s
_placed_surface(::ChartPlacement, q::Quad) = q
function _placed_surface(P::ChartPlacement, hs::HeightSurface{K}) where {K}
    s, o = P.side, P.origin
    c = hs.nodes
    bend = zero(eltype(hs.w))
    for i in 1:K
        bend += hs.w[i] * (c[i][1] * c[i][1] + c[i][2] * c[i][2])
    end
    a = hs.a
    a1 = s * a[1] - (a[2] * o[1] + a[3] * o[2]) - s * log(s) * bend
    nodes = SVector{K}(ntuple(i -> o + s * c[i], Val(K)))
    return HeightSurface(hs.quad, nodes, s * hs.heights, hs.w / s, SVector(a1, a[2], a[3]))
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
# a stored composition is right-associated, its height field (if any) outermost
_in_plane(c::_CalibrationComposition) = c.t1 isa HeightField ? c.t2 : c
@noinline _refuse_chart(c::TargetCalibration) = throw(
    ArgumentError(
        _has_thin_plate(c) ?
            "surface: an in-plane ThinPlateSpline calibration is not supported by the ray primitives yet " *
            "(their chart is an affine of the design); the target's calibration has one" :
            "surface: no ray primitive for a $(nameof(typeof(_not_affine(c)))) calibration"
    )
)
# the member a composition's refusal names (it is refused for its in-plane kind)
_not_affine(c::TargetCalibration) = c
_not_affine(c::_CalibrationComposition) = (ms = members(c); i = findfirst(l -> !is_affine(l), ms); i === nothing ? c : ms[i])
_has_thin_plate(c::TargetCalibration) = any(l -> l isa ThinPlateSpline, members(c))

_surface(::Val{true}, _, quad) = quad
_surface(::Val{false}, c, quad) = _height_surface(_last_height_field(c), quad)
_last_height_field(c::HeightField) = c
# (a stored composition that is not planar has its height field outermost)
_last_height_field(c::_CalibrationComposition) =
    (h = c.t1; h isa HeightField ? h : throw(ArgumentError("surface: a non-planar calibration has its HeightField outermost")))
_height_surface(h::HeightField{K}, quad) where {K} = HeightSurface(quad, h.system.nodes, h.h, h.w, h.a)
