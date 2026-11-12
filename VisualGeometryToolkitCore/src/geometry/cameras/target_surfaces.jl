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
    # (the target checked and stored its calibration and placement)
    S = something(placement(t), _identity_placement(Float64))
    return _surface(Val(is_affine(typeof(c)) && is_planar(typeof(c))), X, c, S, Quad(translation(X), rotation(X), _design_rect(t, physical)))
end
# An affine calibration is a plane: the closed-form Quad, its chart's affine read
# off the design map S₂ ∘ C ∘ S₂⁻¹ (its Jacobian and its image of the origin).
function _surface(::Val{true}, X, c, S, frame::Quad)
    D = _design_map(S, c)
    o = zero(SVector{2, Float64})
    return Quad(frame.origin, rotation(X), frame.rect, transform_deriv(D, o), D(o))
end
# Any other: the composed chart map's surface.
_surface(::Val{false}, X, c, S, frame::Quad) = _mapped_surface(X, c, S, frame)

"""
    MappedSurface(pose, calibration, placement, rect::Rect2) -> MappedSurface

The deformed surface of a posed planar target from its parts — THE builder:
[`surface`](@ref)`(pt)` builds it from the target's stored parts. The chart
map is the composition

    pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹        (design chart → the pose's frame)

of the pose (an `EuclideanMap{3}`, plain numbers), the placement `S₃` (a
`SimilarityMap{3}` or `nothing`, see [`placement`](@ref)) and its in-plane part's
inverse `S₂⁻¹`, the calibration `C` (any kind or composition, plain numbers)
and, for a planar calibration, the lift `(c, 0)`. `rect` is the design chart's
rectangle. Each member is a CoordinateTransformations map with its
`transform_deriv`; the composition's derivative is CT's chain rule.

This form checks its parts (the calibration's composition, the placement, the
pose's frame), on the host. A device caller — a kernel that rebuilds a
`HeightField` or `ThinPlateSpline` from new nodal values by `with_parameters`
on its adapted solver — calls the unchecked inner builder
`VisualGeometryToolkitCore.Cameras._mapped_surface(pose, calibration,
placement, rect)` with parts it has checked once on the host; it keeps the
parts' element type (an unplaced target gets the identity placement in that
type).
"""
MappedSurface(X::EuclideanMap{3}, c::TargetCalibration, P::Union{Nothing, SimilarityMap{3}}, rect::Rect2) =
    _mapped_surface(X, _stored_calibration(c), something(_stored_placement(P), _identity_placement(Float64)), Quad(translation(X), rotation(X), rect))
# The unchecked builder: no check, no message to build, the parts' element type.
@inline _mapped_surface(X::EuclideanMap{3}, c, ::Nothing, frame) = _mapped_surface(X, c, _identity_placement(float(eltype(translation(X)))), frame)
@inline _mapped_surface(X::EuclideanMap{3}, c, S::SimilarityMap{3}, rect::Rect2) = _mapped_surface(X, c, S, _frame_quad(translation(X), rotation(X), rect))
@inline _mapped_surface(X::EuclideanMap{3}, c, S::SimilarityMap{3}, frame::Quad) =
    MappedSurface(ComposedTransformation(X, ComposedTransformation(_target_map(S, c), inv(_placement_plane(S)))), frame)

# THE target-frame map of a calibration chart point: S₃ ∘ lift ∘ C (C alone
# when it lifts the point itself, a height field outermost). Its points are
# `points(t)` (targets.jl), its posed composition the surface's chart map; the
# design map S₂ ∘ C ∘ S₂⁻¹ is a planar calibration's in the design chart.
_target_map(S::SimilarityMap{3}, c) = ComposedTransformation(S, _lifted(Val(is_planar(typeof(c))), _PlainCalibration(c)))
_lifted(::Val{true}, c) = ComposedTransformation(_PLANE_LIFT, c)
_lifted(::Val{false}, c) = c
_design_map(S::SimilarityMap{3}, c) =
    ComposedTransformation(_placement_plane(S), ComposedTransformation(_PlainCalibration(c), inv(_placement_plane(S))))

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
# ... and through the chart map's fused traversal (`_map_value_jacobian`, mapped_surface.jl): a composition's members with
# `_fma_mul`, a planar affine factor as its matrix times the point (plus a planar Euclidean map's translation), a thin-plate
# kind's own sums (plain arithmetic, never contracted)
@inline _map_value_jacobian(f::_PlainCalibration, q::StaticVector{2}) = _calibration_value_jacobian(f.c, q)
@inline _map_value(f::_PlainCalibration, q::StaticVector{2}) = _calibration_value(f.c, q)
@inline function _calibration_value_jacobian(c::_CalibrationComposition, q)
    y, J2 = _calibration_value_jacobian(c.t2, q)
    z, J1 = _calibration_value_jacobian(c.t1, y)
    return z, _fma_mul(J1, J2)
end
@inline _calibration_value(c::_CalibrationComposition, q) = _calibration_value(c.t1, _calibration_value(c.t2, q))
@inline _calibration_value_jacobian(c::_PlanarFactor, q) = (J = transform_deriv(c, q); (_with_offset(c, _fma_mul(J, q)), J))
@inline _calibration_value(c::_PlanarFactor, q) = _with_offset(c, _fma_mul(transform_deriv(c, q), q))
@inline _calibration_value_jacobian(c, q) = (_apply(c, q), _point_jacobian(c, q))
@inline _calibration_value(c, q) = _apply(c, q)
@inline _with_offset(E::EuclideanMap{2}, v) = v + translation(E)
@inline _with_offset(c, v) = v
# (the chart map's traversal descends into it: Adapt, convert_eltype, a Dual layer's values)
_map_members(f, c::_PlainCalibration) = _PlainCalibration(_map_members(f, c.c))

"""
    with_parameters(s::MappedSurface, θ::AbstractVector) -> MappedSurface

The surface `s` with its calibration rebuilt from the parameters `θ`: the chart
map's calibration member `C` (a [`HeightField`](@ref), a
[`ThinPlateSpline`](@ref), or a composition with the affine factors) replaced by
[`with_parameters`](@ref)`(C, θ)` — `θ` in `C`'s [`parameter_names`](@ref)
order, in the numbers the surface was built in (plain metres for a physical
target; a thin-plate kind projects them onto its pure-bending subspace) — and
every other member (the pose, the placement, the lift) and the frame kept as
they are. A surface on the device stays on the device (the kind's
`with_parameters` on its adapted solver), and `θ` may be of any element type
(`Float32`, ForwardDiff `Dual`s): a warped board's surface at new nodal heights
inside a kernel, with no rebuild from the target's parts (the checked
[`MappedSurface`](@ref)`(pose, calibration, placement, rect)`).
"""
@inline with_parameters(s::MappedSurface, θ::AbstractVector) = MappedSurface(_with_calibration_parameters(s.map, θ), s.frame)
# the calibration member found in the chart map (`_target_map`'s `_PlainCalibration`) and rebuilt; every other member kept
@inline _with_calibration_parameters(c::ComposedTransformation, θ) =
    ComposedTransformation(_with_calibration_parameters(c.t1, θ), _with_calibration_parameters(c.t2, θ))
@inline _with_calibration_parameters(c::_PlainCalibration, θ) = _PlainCalibration(with_parameters(c.c, θ))
@inline _with_calibration_parameters(c, θ) = c
Base.:(==)(a::_PlainCalibration, b::_PlainCalibration) = a.c == b.c
Base.hash(f::_PlainCalibration, h::UInt) = hash(f.c, hash(:_PlainCalibration, h))
# the chart plane in its 3-D frame: (c, 0) — Bool entries, so the lift keeps
# the point's element type (Float32, Duals)
const _PLANE_LIFT = CoordinateTransformations.LinearMap(SMatrix{3, 2, Bool}(1, 0, 0, 0, 1, 0))

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
# (no in-plane ThinPlateSpline), the surface is S₃(y, h(y)) in the target frame
# and the chart is an affine of y, so the 2 × 2 root reduces to ONE unknown:
# Newton on the ray parameter t of g(t) = z(t) − z_S(m(t)), the ray's height
# above the plane against the surface's at the ray's plane coordinates m — the
# surface's z through the chart map's own members (y = S₂⁻¹(m), the field's
# `_apply` and `_point_jacobian`, S₃ and their `transform_deriv`s) — then the
# design point S₂(A⁻¹(y)) in closed form. It is the generic root's
# specialization for this map type, kept because it is faster at the same
# answer: measured 2026-11-10 on 20 000 rays of a 16-node field (CPU, Float64),
# 1.1–1.8 µs a ray against the 2 × 2 root's 17 µs (CT's chain rule
# re-evaluates the field's spline in every member's Jacobian: four sweeps a
# step against two), the hits within 8.9e-16 of each other and every hit and
# miss the same (alone, over Aspect ∘ Skew, placed). Plain numbers only, by type:
# a Dual anywhere (the field, the pose, the ray) takes the generic root, whose
# implicit derivative is the Jacobian path's. Any float type, any placement's.
const _HeightOnlyMap{C} = ComposedTransformation{<:EuclideanMap{3},
    <:ComposedTransformation{<:ComposedTransformation{<:SimilarityMap{3}, _PlainCalibration{C}}, <:SimilarityMap{2}}}
_height_over_affine(::Type) = false
_height_over_affine(::Type{<:HeightField}) = true
_height_over_affine(::Type{ComposedTransformation{A, B}}) where {A, B} = A <: HeightField && is_affine(B) && is_planar(B)
_field_and_affine(c::HeightField) = (c, one(SMatrix{2, 2, eltype(c.h)}), zero(SVector{2, eltype(c.h)}))
_field_and_affine(c::ComposedTransformation) = (c.t1, reverse(_calibration_value_jacobian(c.t2, zero(SVector{2, Float64})))...)
# the numbers' type of a calibration type (its parameter vector's), at compile time
_numbers_type(::Type{C}) where {C} = eltype(Base.promote_op(parameter_vector, C))
_numbers_type(::Type{<:EuclideanMap{3, T}}) where {T} = T
_numbers_type(::Type{<:SimilarityMap{N, T, Sc}}) where {N, T, Sc} = promote_type(T, Sc)
_plain_fast(::Type{ComposedTransformation{X, ComposedTransformation{ComposedTransformation{S, _PlainCalibration{C}}, I₂}}}, ::Type{O}, ::Type{Dr}) where {X, S, C, I₂, O, Dr} =
    _height_over_affine(C) && promote_type(_numbers_type(X), _numbers_type(S), _numbers_type(C), eltype(O), eltype(Dr)) <: AbstractFloat
# the numbers of a hit of the chart map M with the ray (O, Dr), at compile time (a Dual: one implicit-function step for its partials)
_hit_numbers(::Type{ComposedTransformation{X, ComposedTransformation{ComposedTransformation{S, _PlainCalibration{C}}, I₂}}}, ::Type{O}, ::Type{Dr}) where {X, S, C, I₂, O, Dr} =
    promote_type(_numbers_type(X), _numbers_type(S), _numbers_type(C), _numbers_type(I₂), eltype(O), eltype(Dr))

const _HEIGHT_NEWTON_ITERATIONS = 4
const _HEIGHT_NEWTON_MAX_ITER = 20

@inline function intersection(s::MappedSurface{M}, o::StaticVector{3}, d::StaticVector{3}) where {C, M <: _HeightOnlyMap{C}}
    _plain_fast(M, typeof(o), typeof(d)) && return _height_fast(s, s.map.t2.t1.t1, _field_and_affine(s.map.t2.t1.t2.c)..., o, d)
    _height_over_affine(C) || return _mapped_intersection(s, o, d)
    # Duals in the field, the placement, the pose or the ray (heights, a board pose): the 1-D Newton's hit of the surface and
    # ray with D's outermost layer's values, then ONE implicit-function step for D's partials (`_implicit_partials`, the generic
    # root's) — never the generic 2 × 2 LM. Nested Duals (a Hessian): the generic root, seeded at the primal hit.
    D = _hit_numbers(M, typeof(o), typeof(d))
    D <: ForwardDiff.Dual{<:Any, <:AbstractFloat} && return _height_implicit(D, s, o, d)
    sP = _primal_surface(s)
    _, _, p, ok = _height_root(sP, sP.map.t2.t1.t1, _field_and_affine(sP.map.t2.t1.t2.c)..., map(_primal, o), map(_primal, d))
    return _mapped_intersection(s, o, d, p, ok)
end
# The implicit-function step at the 1-D Newton's hit `pV` of the values: ONE evaluation of the chart map with D's partials at
# `pV` (𝐗, its partials ∂𝐗/∂θ), the values' Jacobian J = ∂𝐗/∂p, the hit residual's partials in the ray's tangent basis
# (u₁, u₂), ∂p = −(UᵀJ)⁻¹ ∂r, and the point's by the chain rule, 𝐱 = 𝐗 + J (p − pV) — the generic root's `_implicit_partials`
# without a second Dual evaluation for the point.
@inline function _height_implicit(::Type{D}, s::MappedSurface, o::StaticVector{3}, d::StaticVector{3}) where {D}
    sV, oV, dV = map(x -> _layer_values(D, x), (s, o, d))
    _, _, pV, ok = _height_root(sV, sV.map.t2.t1.t1, _field_and_affine(sV.map.t2.t1.t2.c)..., oV, dV)
    pV = ifelse(ok, pV, zero(pV))
    b1, b2 = s2_tangent_basis(d)
    u1, u2 = oftype(d, b1), oftype(d, b2)
    X = _map_value(s.map, map(v -> _lift_dual(D, v), pV))  # the one Dual evaluation
    r = _across(u1, u2, X - o)
    _, J = _map_value_jacobian(sV.map, pV)                 # the values' Jacobian: no Dual arithmetic
    G = inv(_across(map(v -> _outer_value(D, v), u1), map(v -> _outer_value(D, v), u2), J))
    ∂(i) = -(G[i, 1] * ForwardDiff.partials(r[1]) + G[i, 2] * ForwardDiff.partials(r[2]))
    δp = SVector(_with_value(D, zero(pV[1]), ∂(1)), _with_value(D, zero(pV[2]), ∂(2)))
    x = X + _fma_mul(J, δp)
    t = _fma_dot(x - o, d) / _fma_dot(d, d)
    return _masked(ok, t, x)
end
# A surface with every Dual layer of its members and frame stripped (`_primal`).
@inline _primal_surface(s::MappedSurface) = MappedSurface(_map_members(m -> _member_numbers(_primal, m), s.map), _map_numbers(_primal, s.frame))

@inline function _height_fast(s::MappedSurface, S::SimilarityMap{3}, H::HeightField, J, β, o::StaticVector{3}, d::StaticVector{3})
    t, x, _, ok = _height_root(s, S, H, J, β, o, d)
    return _masked(ok, t, x)
end
# The 1-D Newton's hit: `(t, x, p, ok)` — the ray parameter, the point, its design point, whether it converged on the rectangle.
@inline function _height_root(s::MappedSurface, S::SimilarityMap{3}, H::HeightField, J, β, o::StaticVector{3}, d::StaticVector{3})
    q = s.frame
    n = normal(q)
    S₂ = _placement_plane(S)
    S₂⁻¹ = s.map.t2.t2                             # the chart map's own inverse placement
    field = _PlainCalibration(H)
    den = _fma_dot(d, n)                           # g′ without the surface's slope
    s0 = _fma_dot(q.origin - o, n)                 # the ray's height above the plane: t den − s0
    cd = SVector(_fma_dot(d, q.e_u), _fma_dot(d, q.e_v))   # the plane velocity of the ray
    # the surface's height z_S(m) at plane coordinates m and its gradient, through the members (the fused traversal)
    function terms(t, x)
        y, Ji = _map_value_jacobian(S₂⁻¹, _frame_coordinates(q, x))
        Y, Jf = _map_value_jacobian(field, y)
        Z, JS = _map_value_jacobian(S, Y)
        gz = _fma_tmul(Ji, _fma_tmul(Jf, SVector(JS[3, 1], JS[3, 2], JS[3, 3])))   # row 3 of JS Jf Ji: ∇ of the height
        return t * den - s0 - Z[3], den - _fma_dot(gz, cd)
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
    # the design point: y = J c + β of the affine factors, p = S₂(c)
    p = _map_value(S₂, J \ (_map_value(S₂⁻¹, _frame_coordinates(q, x)) - β))
    return t, x, p, converged & (t >= 0) & _in_rect(q.rect, p)
end
