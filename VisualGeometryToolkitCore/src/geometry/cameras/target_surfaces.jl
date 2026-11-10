# =============================================================================
# surface(::PosedTarget): the primitive a ray meets on a posed planar target
# =============================================================================
#
# A planar affine calibration (Rigid, the affine factors `EuclideanMap ∘ Scale
# ∘ Aspect ∘ Skew` or any of them) is a plane, the closed-form Quad; any other
# (a HeightField, an in-plane ThinPlateSpline, either composed with the affine
# factors) is the MappedSurface of the composed chart map
# `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹` (mapped_surface.jl). The kind is chosen by the
# calibration's TYPE, so code over a rigid target never compiles the root.
#
# Frames: the primitive is in the pose's destination frame F; its chart is
# the DESIGN chart (the nominal points' coordinates); its rectangle is the
# board's, or the nominal points' bounding box — plain numbers, metres for a
# physical target.
# =============================================================================

# (documented at the generic, geometry/primitives/surfaces.jl: Documenter
# reads a binding's docstring from its owner module)
function surface(pt::PosedTarget{<:PlanarTarget})
    t = model(pt)
    physical = has_units(pt)
    # (the bare calibration: an uncertain one's covariance is not the surface's, so it is not rescaled)
    c = geom(physical ? _calibration_unit_move(t, c -> _ustrip_calibration(m, geom(c))) : calibration(t))
    X = geom(pose(pt))
    X = physical ? ustrip(m, X) : X
    return _surface(Val(is_affine(typeof(c)) && is_planar(typeof(c))), X, c, placement(t), _design_rect(t, physical))
end
# An affine calibration is a plane: the closed-form Quad, its chart's affine the
# calibration's (placed: `A p + b` with `b = s β − (A − I) o`).
function _surface(::Val{true}, X, c, P, rect)
    A, b = _placed_affine(P, _chart_affine(c)...)
    return Quad(translation(X), rotation(X), rect, A, b)
end
# Any other: the composed chart map's surface.
_surface(::Val{false}, X, c, P, rect) = MappedSurface(X, c, P, rect)

"""
    MappedSurface(pose, calibration, placement, rect::Rect2) -> MappedSurface

The deformed surface of a posed planar target from its parts — THE builder:
[`surface`](@ref)`(pt)` calls it with the target's stored parts, and a device
caller with adapted ones (a [`HeightField`](@ref) or [`ThinPlateSpline`](@ref)
rebuilt from new nodal values by `with_parameters` on its adapted solver, in a
kernel). The chart map is the composition

    pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹        (design chart → the pose's frame)

of the pose (an `EuclideanMap{3}`, plain numbers), the placement `S₃` (a
`SimilarityMap{3}` or `nothing`, see [`placement`](@ref)) and its in-plane part's
inverse `S₂⁻¹`, the calibration `C` (any kind or composition, plain numbers)
and, for a planar calibration, the lift `(c, 0)`. `rect` is the design chart's
rectangle. Each member is a CoordinateTransformations map with its
`transform_deriv`; the composition's derivative is CT's chain rule.
"""
function MappedSurface(X::EuclideanMap{3}, c::TargetCalibration, P::Union{Nothing, SimilarityMap{3}}, rect::Rect2)
    # an unplaced target is placed by the identity (scale 1, no translation: the
    # same numbers exactly), so the surface's type does not depend on whether a
    # target is placed
    S = P === nothing ? _IDENTITY_PLACEMENT : _stored_placement(P)
    D = ComposedTransformation(_target_map(S, c), inv(_placement_plane(S)))
    return MappedSurface(ComposedTransformation(X, D), Quad(translation(X), rotation(X), rect))
end
const _IDENTITY_PLACEMENT = _Placement(1.0, RotMatrix{3, Float64}(SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)), SVector(0.0, 0.0, 0.0))

# THE target-frame map of a calibration chart point: S₃ ∘ lift ∘ C (C alone
# when it lifts the point itself, a height field outermost). Its points are
# `points(t)` (targets.jl), its posed composition the surface's chart map.
_target_map(::Nothing, c) = _lifted(Val(is_planar(typeof(c))), _PlainCalibration(c))
_target_map(P::SimilarityMap{3}, c) = ComposedTransformation(P, _lifted(Val(is_planar(typeof(c))), _PlainCalibration(c)))
_lifted(::Val{true}, c) = ComposedTransformation(_PLANE_LIFT, c)
_lifted(::Val{false}, c) = c

# The calibration as a member of the chart map: its own arithmetic on plain
# numbers (`_apply`, `_point_jacobian`: the kinds' and CT's chain rule over a
# composition's members), without the one-form check of `c(p)` — the map's
# parts are plain numbers by construction (the builder strips a physical
# target to metres), and the check's DimensionError cannot be compiled for a
# device.
struct _PlainCalibration{C} <: CoordinateTransformations.Transformation
    c::C
end
@inline (f::_PlainCalibration)(q::StaticVector{2}) = _apply(f.c, q)
@inline transform_deriv(f::_PlainCalibration, q::StaticVector{2}) = _point_jacobian(f.c, q)
Adapt.adapt_structure(to, f::_PlainCalibration) = _PlainCalibration(_adapt_members(to, f.c))
convert_eltype(::Type{T}, f::_PlainCalibration) where {T <: AbstractFloat} = _PlainCalibration(convert_eltype(T, f.c))
Base.:(==)(a::_PlainCalibration, b::_PlainCalibration) = a.c == b.c
Base.hash(f::_PlainCalibration, h::UInt) = hash(f.c, hash(:_PlainCalibration, h))
# the chart plane in its 3-D frame: (c, 0) — Bool entries, so the lift keeps
# the point's element type (Float32, Duals)
const _PLANE_LIFT = CoordinateTransformations.LinearMap(SMatrix{3, 2, Bool}(1, 0, 0, 0, 1, 0))

# A placed target's affine calibration y = J c + β in the chart is, on the
# design chart, p ↦ p + s (J c + β − c) = J p + (s β − (J − I) o): the same J,
# b = s β − (J − I) o — exactly 0 with J = I for the identity.
_placed_affine(::Nothing, J, β) = (J, β)
_placed_affine(P::SimilarityMap{3}, J, β) = (J, _placement_side(P) * β - (J - one(J)) * _placement_origin(P))

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

# A planar affine calibration as the chart's affine, m = A c + b, read off the
# calibration itself: b its image of the design origin, A its point Jacobian
# there — the calibration's own composition, so a new affine kind needs no
# method here.
function _chart_affine(c::TargetCalibration)
    o = zero(SVector{2, Float64})
    return _point_jacobian(c, o), _apply(c, o)
end

# ------------------------------------------------------------------------------
# The 1-D fast path of a height field over an affine chart
# ------------------------------------------------------------------------------
#
# When the calibration is a HeightField alone or over the planar affine factors
# (no in-plane ThinPlateSpline), the surface is {o + s·(y, h(y))} in the target
# frame and the chart is an affine of y, so the 2 × 2 root reduces to ONE
# unknown: Newton on the ray parameter t of g(t) = z(t) − s·h(y(t)), y(t) the
# chart point of the ray's plane coordinates, then the design point in closed
# form. It is the generic root's specialization for this map type, kept because
# it is faster at the same answer: measured 2026-11-10 on 20 000 rays of a
# 16-node field (CPU, Float64), 1.1–1.8 µs a ray against the 2 × 2 root's
# 17 µs (CT's chain rule re-evaluates the field's spline in every member's
# Jacobian: four sweeps a step against two), the hits within 8.9e-16 of each
# other and every hit and miss the same (alone, over Aspect ∘ Skew, placed). Plain
# numbers only: a Dual anywhere (the field, the pose, the ray) takes the generic
# root, whose implicit derivative is the Jacobian path's.
const _HeightOnlyMap{C} = ComposedTransformation{<:EuclideanMap{3}, <:ComposedTransformation{ComposedTransformation{_Placement, _PlainCalibration{C}}, <:SimilarityMap{2}}}
_height_over_affine(::Type) = false
_height_over_affine(::Type{<:HeightField}) = true
_height_over_affine(::Type{ComposedTransformation{A, B}}) where {A, B} = A <: HeightField && is_affine(B) && is_planar(B)
_field_and_affine(c::HeightField) = (c, SMatrix{2, 2, Float64}(1, 0, 0, 1), zero(SVector{2, Float64}))
_field_and_affine(c::ComposedTransformation) = (c.t1, _chart_affine(c.t2)...)

const _HEIGHT_NEWTON_ITERATIONS = 4
const _HEIGHT_NEWTON_MAX_ITER = 20

@inline function intersection(s::MappedSurface{<:_HeightOnlyMap{C}}, o::StaticVector{3}, d::StaticVector{3}) where {C}
    c = s.map.t2.t1.t2.c
    T = promote_type(eltype(s.frame.origin), eltype(parameter_vector(_field_and_affine(c)[1])), eltype(o), eltype(d))
    (_height_over_affine(C) && T <: AbstractFloat) || return _mapped_intersection(s, o, d)
    return _height_fast(s, s.map.t2.t1.t1, _field_and_affine(c)..., o, d)
end

@inline function _height_fast(s::MappedSurface, P::_Placement, H::HeightField, J, β, o::StaticVector{3}, d::StaticVector{3})
    q = s.frame
    n = normal(q)
    side, org = P.s, _placement_origin(P)
    nodes, w, a = H.system.nodes, H.w, H.a
    den = dot(d, n)                                # g′ without the height's slope
    s0 = dot(q.origin - o, n)                      # g(t) = t den − s0 − s h(y(t))
    cd = SVector(dot(d, q.e_u), dot(d, q.e_v))     # the plane velocity of the ray
    function terms(t, x)
        y = (_plane_coordinates(q, x) - org) / side
        return t * den - s0 - side * tps_evaluate(nodes, w, a, y), den - dot(tps_gradient(nodes, w, a, y), cd)
    end
    t, x = _plane_hit(q, o, d)                     # the seed
    for _ in 1:_HEIGHT_NEWTON_ITERATIONS
        g, gp = terms(t, x)
        t -= g / gp
        x = o + t * d
    end
    # the continuation: stop at a rounding-level update, which is not applied
    nd = norm(d)
    L = hypot(q.rect.widths[1], q.rect.widths[2])
    ε = eps(typeof(t))
    converged = false
    last = oftype(abs(t), Inf)
    for _ in (_HEIGHT_NEWTON_ITERATIONS + 1):_HEIGHT_NEWTON_MAX_ITER
        g, gp = terms(t, x)
        Δ = g / gp
        scale = L + abs(t) * nd
        if abs(Δ) * nd <= 64 * ε * scale
            converged = true
            break
        end
        last = abs(Δ) * nd / scale
        t -= Δ
        x = o + t * d
    end
    converged |= last <= sqrt(ε)
    # the design point: y = J c + β, p = org + side·c
    p = org + side * (J \ ((_plane_coordinates(q, x) - org) / side - β))
    return _masked(converged & (t >= 0) & _in_rect(q.rect, p), t, x)
end
