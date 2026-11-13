# =============================================================================
# MappedSurface: the surface of a chart map, and the ray–board root
# =============================================================================
#
# A posed target's deformed surface is ONE composed map from the design chart
# to the pose's frame (target_surfaces.jl): `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹`, each
# member a CoordinateTransformations map with its `transform_deriv`, the
# composition's derivative the chain rule. A ray meets it where a 2 × 2 root
# in the chart vanishes; the root is solved behind one narrow function
# (`_board_root`), by the spline models' static LM, so its solver can be
# swapped. Here, beside that LM (spline_lens.jl), not with Quad in
# geometry/primitives/surfaces.jl.
# =============================================================================

"""
    MappedSurface(map, frame::Quad)

A surface given by its CHART MAP: the points `map(p)` (3-vectors in the
surface's frame) of the chart points `p` in the rectangle
[`bounds`](@ref)`(s) = bounds(frame)`. `map` is any isbits
CoordinateTransformations `Transformation` of a 2-vector to a 3-vector with a
`transform_deriv` (its point Jacobian, `3 × 2`) — for a posed target's
[`surface`](@ref), the composition `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹` of its pose,
its placement, its calibration and the placement's inverse on the design
chart, so the chart is the DESIGN chart. `frame` is the plane the map deforms
(the posed target's `z = 0` plane, its rectangle the chart's bounds): it
seeds the solves and orients the normal.

[`intersection`](@ref) solves for the chart point of the hit, `p`, such that
`map(p)` lies on the ray: the two components of `map(p) − o` across the ray
(`u₁`, `u₂`, an orthonormal pair perpendicular to `d`) vanish — a 2 × 2
least-squares root through `transform_deriv`, seeded at the chart coordinates
of the frame's plane hit and bounded to the chart's rectangle — and the hit
is `x = map(p)`, `t = (x − o)·d / (d·d)`. A ray misses (`NaN`) when the plane
hit does not exist, the solve does not converge within the rectangle, or
`t < 0`. [`chart`](@ref)`(s, x)` is the same root for the residual
`map(p) − x`, [`surface_point`](@ref)`(s, p) = map(p)`, and [`normal`](@ref)
the unit `∂map/∂p₁ × ∂map/∂p₂` at the point's chart point (geometric: on the
side of the frame's normal for a chart map of positive orientation).

Generic in the element type (Float32, Float64, ForwardDiff `Dual`s, nested
ones included) and allocation-free: under ForwardDiff the root is solved on
the values and differentiated by the implicit function theorem, one layer at
a time — never by differentiating the iteration.
"""
struct MappedSurface{M, T}
    map::M
    frame::Quad{T}
end
# The members of a chart map, each mapped by `f` and the composition rebuilt
# (`_map_members`, target_calibration.jl, through the plain calibration member
# too): ONE traversal for the device form, a float type and a Dual layer's values.
# The device form: every member adapted (a thin-plate kind's matrices become
# device arrays), the isbits frame as it is. (CT's composition has no Adapt rule
# of its own, and it is not this package's to give one.)
Adapt.adapt_structure(to, s::MappedSurface) = MappedSurface(_map_members(x -> Adapt.adapt(to, x), s.map), s.frame)
# The surface in another float type (a Float32 device form): every member of
# the chart map and the frame converted, nothing re-solved.
convert_eltype(::Type{T}, s::MappedSurface) where {T <: AbstractFloat} = MappedSurface(_map_members(x -> _member_eltype(T, x), s.map), Quad{T}(s.frame))
_member_eltype(::Type{T}, c::CoordinateTransformations.LinearMap{<:StaticMatrix{<:Any, <:Any, Bool}}) where {T} = c
_member_eltype(::Type{T}, S::SimilarityMap) where {T} = _member_numbers(v -> T(v), S)
_member_eltype(::Type{T}, c) where {T} = convert_eltype(T, c)
# A member with every number mapped by `f` (a Dual layer's values; a conversion):
# the solver data (a thin-plate kind's system, its bending factor) kept.
_member_numbers(f, c::CoordinateTransformations.LinearMap{<:StaticMatrix{<:Any, <:Any, Bool}}) = c
_member_numbers(f, S::SimilarityMap{N}) where {N} = _similarity(f(S.s), map(f, SMatrix{N, N}(S.R)), map(f, S.t))
# (the values or a conversion of an already checked rotation: rebuilt unchecked, as the similarity is;
# checked, the rebuild allocated 416 bytes per Dual-ray hit at d77d3556)
_member_numbers(f, E::EuclideanMap{N}) where {N} = EuclideanMap(RotMatrix{N}(map(f, SMatrix{N, N}(rotation(E)))), map(f, translation(E)); check = false)
_member_numbers(f, c::HeightField{K}) where {K} = (h = map(f, c.h); HeightField{K, eltype(h)}(c.system, h, map(f, c.w), map(f, c.a), c.bending, c.physical))
_member_numbers(f, c::ThinPlateSpline{K}) where {K} = (fx = map(f, c.fx);
    ThinPlateSpline{K, eltype(fx)}(c.system, fx, map(f, c.fy), map(f, c.wx), map(f, c.ax), map(f, c.wy), map(f, c.ay), c.bending, c.physical))
_member_numbers(f, c) = with_parameters(c, map(f, parameter_vector(c)))      # the affine factors
# a similarity of the given parts, unchecked (a device path: no message to build)
@inline _similarity(s, R::SMatrix{N, N}, t::StaticVector{N}) where {N} =
    SimilarityMap{N, eltype(R), typeof(s), RotMatrix{N, eltype(R), N * N}, SVector{N, eltype(t)}}(s, RotMatrix{N, eltype(R)}(R), SVector{N}(t))
bounds(s::MappedSurface) = bounds(s.frame)
surface_point(s::MappedSurface, p::StaticVector{2}) = _map_value(s.map, p)

# The chart map's point and Jacobian at a chart point, through its members, with the explicit fused products
# (`_fma_mul`, base/fused_products.jl) — CT's own `t1(t2(x))` and chain rule `transform_deriv(t1, t2(x)) * transform_deriv(t2, x)` are
# StaticArrays' products, which fuse as each compiled copy decides. The root (`_hit_residual`, `_chart_residual`, the height
# fast path) reads the map through these only, so it is one function of its arguments wherever it is inlined.
# `_map_value` is `_map_value_jacobian`'s point, operation for operation. A member of another kind takes its own call and
# `transform_deriv`.
@inline function _map_value_jacobian(m::CoordinateTransformations.ComposedTransformation, x)
    y, J2 = _map_value_jacobian(m.t2, x)
    z, J1 = _map_value_jacobian(m.t1, y)
    return z, _fma_mul(J1, J2)
end
@inline _map_value(m::CoordinateTransformations.ComposedTransformation, x) = _map_value(m.t1, _map_value(m.t2, x))
@inline _map_value_jacobian(E::EuclideanMap, x) = (R = SMatrix(rotation(E)); (_fma_mul(R, x) + translation(E), R))
@inline _map_value(E::EuclideanMap, x) = _fma_mul(SMatrix(rotation(E)), x) + translation(E)
@inline _map_value_jacobian(S::SimilarityMap, x) = (R = SMatrix(S.R); (S.s * _fma_mul(R, x) + S.t, S.s * R))
@inline _map_value(S::SimilarityMap, x) = S.s * _fma_mul(SMatrix(S.R), x) + S.t
@inline _map_value_jacobian(L::CoordinateTransformations.LinearMap{<:StaticMatrix}, x) = (_fma_mul(L.linear, x), L.linear)
@inline _map_value(L::CoordinateTransformations.LinearMap{<:StaticMatrix}, x) = _fma_mul(L.linear, x)
@inline _map_value_jacobian(m, x) = (m(x), transform_deriv(m, x))
@inline _map_value(m, x) = m(x)
# the components of v across the ray (u₁ᵀv, u₂ᵀv), and of each column of J
@inline _across(u1, u2, v::StaticVector{3}) = SVector(_fma_dot(u1, v), _fma_dot(u2, v))
@inline _across(u1, u2, J::StaticMatrix{3}) = _fma_tmul(hcat(u1, u2), J)

# The iteration budget of a board root, its box and its scale: the chart's
# rectangle (in primal Float64: a Dual pose's frame contributes its values), its
# unit per axis, the acceptance of a final residual below √eps of the chart's
# diagonal (rounding amplified at grazing incidence; the root itself is resolved
# to rounding, `_board_root`), and the diagonal itself, the residual's scale for
# the LM's precision-aware stops.
const _BOARD_MAXITER = 100
@inline function _board_box(q::Quad)
    r = q.rect
    o, wr = map(_primal, SVector{2}(r.origin)), map(_primal, SVector{2}(r.widths))
    lb = SVector{2, Float64}(o)
    ub = SVector{2, Float64}(o + wr)
    w = max.(SVector{2, Float64}(wr), floatmin(Float64))
    L = hypot(w[1], w[2])
    return lb, ub, _LM_AREA_UNITS ./ w, (sqrt(eps(_primal_float(q))) * L)^2, L
end
_primal_float(::Quad{T}) where {T} = primal_float(T)

# The residual and its Jacobian of a ray hit, `map(p) − o` across the ray
# (an orthonormal pair across d, Core's one tangent basis of S², in the ray's
# element type), and of a chart lookup, `map(p) − x`.
@inline function _hit_residual(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3})
    b1, b2 = s2_tangent_basis(d)
    u1, u2 = oftype(d, b1), oftype(d, b2)
    function fj(p)
        X, J = _map_value_jacobian(s.map, p)
        return _across(u1, u2, X - o), _across(u1, u2, J), X
    end
    return fj
end
@inline _chart_residual(s::MappedSurface, x::StaticVector{3}) = p -> ((X, J) = _map_value_jacobian(s.map, p); (X - x, J, X))
# The same residuals without their Jacobian (`_value_residual(residual)`): the implicit-function step's one evaluation on the
# Dual parts needs the residual's partials only — the chart map's Jacobian there would carry every partial for nothing.
@inline function _hit_value(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3})
    b1, b2 = s2_tangent_basis(d)
    u1, u2 = oftype(d, b1), oftype(d, b2)
    return p -> _across(u1, u2, _map_value(s.map, p) - o)
end
@inline _chart_value(s::MappedSurface, x::StaticVector{3}) = p -> _map_value(s.map, p) - x
@inline _value_residual(::typeof(_hit_residual)) = _hit_value
@inline _value_residual(::typeof(_chart_residual)) = _chart_value

@inline intersection(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3}) = _mapped_intersection(s, o, d)
# (the generic root; a height field over an affine chart has a 1-D fast path, target_surfaces.jl)
# `seed`: a chart point to start the root at instead of the plane hit's when `use_seed` (a fast path's hit of the primal
# surface, `intersection` of a height field with Duals).
@inline _mapped_intersection(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3}) =
    _mapped_intersection(s, o, d, zero(SVector{2, Float64}), false)
@inline function _mapped_intersection(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3}, seed::StaticVector{2}, use_seed::Bool)
    t0, x0 = _plane_hit(s.frame, o, d)
    p0 = map(_primal, _plane_coordinates(s.frame, x0))
    lb, ub, sc, tol2, L = _board_box(s.frame)
    seed_ok = all(isfinite, p0)
    x₀ = ifelse(use_seed, oftype(p0, seed), ifelse(seed_ok, p0, oftype(p0, (lb + ub) / 2)))
    p, ok, X = _board_root(_hit_residual, (s, o, d), x₀, lb, ub, sc, tol2, L)
    x = _root_point(X, s, p)
    t = _fma_dot(x - o, d) / _fma_dot(d, d)
    return _masked(seed_ok & ok & (t >= 0) & _in_rect(s.frame.rect, map(_primal, p)), t, x)
end

@inline function chart(s::MappedSurface, x::StaticVector{3})
    p0 = map(_primal, _plane_coordinates(s.frame, x))
    lb, ub, sc, tol2, L = _board_box(s.frame)
    p, ok, _ = _board_root(_chart_residual, (s, x), p0, lb, ub, sc, tol2, L)
    nan = oftype(p[1], NaN)
    return ifelse(ok, p, SVector(nan, nan))
end

@inline function normal(s::MappedSurface, x::StaticVector{3})
    _, J = _map_value_jacobian(s.map, chart(s, x))
    return LinearAlgebra.normalize(cross(J[:, 1], J[:, 2]))
end

# THE ray–board root behind one narrow interface, so the solver behind it can
# be swapped: `residual(parts...)` builds `fj(p) -> (r, ∂r/∂p, X)`, the residual,
# its Jacobian and the surface point `X = map(p)` at a chart point, from the
# problem's parts (the surface and the ray, or the surface and a point); `x₀` the
# seed, `[lb, ub]` the box, `s` the chart's unit per axis, `tol2` the acceptance
# of the final squared residual, `L` the residual's scale. Returns `(p, converged,
# X)`: the LM's own last evaluation's point for plain parts, `nothing` for Dual
# parts (their point carries the root's partials: `_root_point` maps the root).
# The numbers' type is read off `fj`'s inferred result, not from an evaluation.
#
# The ITERATION IS PLAIN. Plain parts are solved by Core's one static 2-D LM
# (`static_lm`, the SplineLens projection's, in their float type). Parts with
# ForwardDiff Duals are solved by the implicit function theorem one layer at a
# time: the parts with the outermost layer's values (the surface's members and
# frame, the ray) are solved first — recursively, so the LM only ever sees
# plain numbers — and then `∂p = −(JᵀJ)⁻¹Jᵀ ∂r` comes from ONE evaluation of `fj`
# on the Dual parts at that root (`J⁻¹` for a square J), the value parts the
# values' root exactly. Nested Duals (a Hessian) differentiate exactly.
@inline function _board_root(residual::R, parts::Tuple, x₀::StaticVector{2}, lb, ub, s, tol2, L) where {R}
    fj = residual(parts...)
    T = _root_numbers(Base.promote_op(fj, typeof(x₀)))
    T === nothing && (rJ = fj(x₀); T = promote_type(eltype(rJ[1]), eltype(rJ[2])))   # not inferred: one evaluation
    return _board_root(T, residual, parts, x₀, lb, ub, s, tol2, L)
end
@inline _root_numbers(::Type{<:Tuple{R, J, Vararg}}) where {R <: StaticArrays.StaticArray, J <: StaticArrays.StaticArray} = promote_type(eltype(R), eltype(J))
@inline _root_numbers(::Type) = nothing
@inline _root_point(X, s, p) = X
@inline _root_point(::Nothing, s, p) = _map_value(s.map, p)
@inline function _board_root(::Type{T}, residual::R, parts::Tuple, x₀, lb, ub, s, tol2, L) where {T <: AbstractFloat, R}
    fj = residual(parts...)
    fT(p) = (rJ = fj(p); (SVector{length(rJ[1]), T}(rJ[1]), SMatrix{size(rJ[2], 1), 2, T}(rJ[2]), rJ[3]))
    # resolved to rounding (cost floor 0): the stops are what T resolves of a residual of the chart's scale L, so a
    # nearly flat surface's plane-hit seed (a residual of its heights, below √eps·L for heights of 1e-8 L) is not
    # taken for its root, and the hit is a smooth function of the surface; `tol2` is only the acceptance
    p, ok, _, (_, _, X) = static_lm(fT, SVector{2, T}(x₀), SVector{2, T}(lb), SVector{2, T}(ub), SVector{2, T}(s), _BOARD_MAXITER, T(tol2), T(L), nothing, zero(T))
    return p, ok, X
end
@inline function _board_root(::Type{D}, residual::R, parts::Tuple, x₀, lb, ub, s, tol2, L) where {D <: ForwardDiff.Dual, R}
    V = ForwardDiff.valtype(D)
    vparts = map(x -> _layer_values(D, x), parts)
    pV, ok, _ = _board_root(V, residual, vparts, x₀, lb, ub, s, tol2, L)
    return _implicit_partials(D, residual, parts, vparts, pV), ok, nothing
end
# The implicit-function step at the values' root `pV` of `residual` on `vparts` (the parts with D's outermost layer's values):
# the root with D's partials, `∂p = −(JᵀJ)⁻¹Jᵀ ∂r`, from the values' Jacobian and ONE evaluation of the residual on the Dual parts.
@inline function _implicit_partials(::Type{D}, residual::R, parts::Tuple, vparts::Tuple, pV::StaticVector{2}) where {D <: ForwardDiff.Dual, R}
    Jv = residual(vparts...)(pV)[2]               # the values' Jacobian at their root: no Dual arithmetic
    rD = _value_residual(residual)(parts...)(map(v -> _lift_dual(D, v), pV))   # the residual's partials: one Dual evaluation
    G = _fma_mul(inv(_fma_tmul(Jv, Jv)), Jv')   # 2 × R: the least-squares inverse (J⁻¹ for a square J)
    ∂(i) = -sum(ntuple(k -> G[i, k] * ForwardDiff.partials(rD[k]), Val(length(rD))))
    return SVector(_with_value(D, pV[1], ∂(1)), _with_value(D, pV[2], ∂(2)))
end
# A part with D's outermost layer's values: a vector's numbers, a surface's members and frame.
@inline _layer_values(::Type{D}, x::StaticVector) where {D} = map(v -> _outer_value(D, v), x)
@inline _layer_values(::Type{D}, s::MappedSurface) where {D} =
    MappedSurface(_map_members(m -> _member_numbers(v -> _outer_value(D, v), m), s.map), _map_numbers(v -> _outer_value(D, v), s.frame))
