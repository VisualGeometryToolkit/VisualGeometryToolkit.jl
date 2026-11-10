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
# The device form: every member of the chart map adapted (a thin-plate kind's
# matrices become device arrays), the isbits frame as it is. (CT's composition
# has no Adapt rule of its own, and it is not this package's to give one.)
Adapt.adapt_structure(to, s::MappedSurface) = MappedSurface(_adapt_members(to, s.map), s.frame)
_adapt_members(to, c::ComposedTransformation) = ComposedTransformation(_adapt_members(to, c.t1), _adapt_members(to, c.t2))
_adapt_members(to, c) = Adapt.adapt(to, c)
# The surface in another float type (a Float32 device form): every member of
# the chart map and the frame converted, nothing re-solved.
convert_eltype(::Type{T}, s::MappedSurface) where {T <: AbstractFloat} = MappedSurface(_convert_members(T, s.map), Quad{T}(s.frame))
_convert_members(::Type{T}, c::ComposedTransformation) where {T} = ComposedTransformation(_convert_members(T, c.t1), _convert_members(T, c.t2))
_convert_members(::Type{T}, c::CoordinateTransformations.LinearMap{<:StaticMatrix{<:Any, <:Any, Bool}}) where {T} = c
_convert_members(::Type{T}, S::SimilarityMap{N}) where {T, N} = SimilarityMap(T(S.s), RotMatrix{N, T}(SMatrix{N, N}(S.R)), SVector{N, T}(S.t); check = false)
_convert_members(::Type{T}, c) where {T} = convert_eltype(T, c)
bounds(s::MappedSurface) = bounds(s.frame)
surface_point(s::MappedSurface, p::StaticVector{2}) = s.map(p)

# The iteration budget of a board root and its acceptance: a final residual
# below √eps of the chart's diagonal (rounding amplified at grazing incidence).
const _BOARD_MAXITER = 100
@inline function _board_box(q::Quad)
    r = q.rect
    o, wr = map(_primal, SVector{2}(r.origin)), map(_primal, SVector{2}(r.widths))   # a Dual pose's frame: its values
    lb = SVector{2, Float64}(o)
    ub = SVector{2, Float64}(o + wr)
    w = max.(SVector{2, Float64}(wr), floatmin(Float64))
    L = hypot(w[1], w[2])
    return lb, ub, _LM_AREA_UNITS ./ w, (sqrt(eps(_primal_float(q))) * L)^2
end
_primal_float(::Quad{T}) where {T} = primal_float(T)

@inline intersection(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3}) = _mapped_intersection(s, o, d)
# (the generic root; a height field over an affine chart has a 1-D fast path, target_surfaces.jl)
@inline function _mapped_intersection(s::MappedSurface, o::StaticVector{3}, d::StaticVector{3})
    t0, x0 = _plane_hit(s.frame, o, d)
    p0 = _plane_coordinates(s.frame, x0)
    b1, b2 = s2_tangent_basis(d)                 # an orthonormal pair across the ray (Core's one tangent basis of S²),
    u1, u2 = oftype(d, b1), oftype(d, b2)        # in the ray's element type
    function fj(p)
        X = s.map(p)
        J = transform_deriv(s.map, p)
        δ = X - o
        return SVector(dot(u1, δ), dot(u2, δ)), vcat(u1' * J, u2' * J)
    end
    lb, ub, sc, tol2 = _board_box(s.frame)
    seed_ok = all(isfinite, map(_primal, p0))
    p, ok = _board_root(fj, ifelse(seed_ok, map(_primal, p0), oftype(map(_primal, p0), (lb + ub) / 2)), lb, ub, sc, tol2)
    x = s.map(p)
    t = dot(x - o, d) / dot(d, d)
    return _masked(seed_ok & ok & (t >= 0) & _in_rect(s.frame.rect, map(_primal, p)), t, x)
end

@inline function chart(s::MappedSurface, x::StaticVector{3})
    p0 = _plane_coordinates(s.frame, x)
    fj(p) = (s.map(p) - x, transform_deriv(s.map, p))
    lb, ub, sc, tol2 = _board_box(s.frame)
    p, ok = _board_root(fj, map(_primal, p0), lb, ub, sc, tol2)
    nan = oftype(p[1], NaN)
    return ifelse(ok, p, SVector(nan, nan))
end

@inline function normal(s::MappedSurface, x::StaticVector{3})
    J = transform_deriv(s.map, chart(s, x))
    return LinearAlgebra.normalize(cross(J[:, 1], J[:, 2]))
end

# THE ray–board root behind one narrow interface, so the solver behind it can
# be swapped: `fj(p) -> (r, ∂r/∂p)` the residual and its Jacobian at a chart
# point, `x₀` the seed, `[lb, ub]` the box, `s` the chart's unit per axis and
# `tol2` the acceptance of the final squared residual. Returns `(p, converged)`.
# Generic in the element type: plain numbers are solved by Core's one static 2-D
# LM (`_spline_lm`, the SplineLens projection's, in their float type), a
# ForwardDiff `Dual` residual by the implicit function theorem one layer at a
# time — the root of the values (recursively), then `∂p = −(JᵀJ)⁻¹Jᵀ ∂r` from
# ONE evaluation of `fj` in the Duals at that root, the value parts the
# values' root exactly. No Dual enters the iteration.
@inline function _board_root(fj::F, x₀::StaticVector{2}, lb, ub, s, tol2) where {F}
    x = map(_primal, x₀)
    r, J = fj(x)
    return _board_root(promote_type(eltype(r), eltype(J)), fj, x, lb, ub, s, tol2)
end
@inline function _board_root(::Type{T}, fj::F, x₀, lb, ub, s, tol2) where {T <: AbstractFloat, F}
    fT(p) = (rJ = fj(p); (SVector{length(rJ[1]), T}(rJ[1]), SMatrix{size(rJ[2], 1), 2, T}(rJ[2])))
    return _spline_lm(fT, SVector{2, T}(x₀), SVector{2, T}(lb), SVector{2, T}(ub), SVector{2, T}(s), _BOARD_MAXITER, T(tol2))
end
@inline function _board_root(::Type{D}, fj::F, x₀, lb, ub, s, tol2) where {D <: ForwardDiff.Dual, F}
    V = ForwardDiff.valtype(D)
    lift(p) = map(v -> _lift_dual(D, v), p)
    fjV(p) = (rJ = fj(lift(p)); (map(x -> _outer_value(D, x), rJ[1]), map(x -> _outer_value(D, x), rJ[2])))
    pV, ok = _board_root(V, fjV, x₀, lb, ub, s, tol2)
    rD, JD = fj(lift(pV))
    Jv = map(x -> _outer_value(D, x), JD)
    G = inv(Jv' * Jv) * Jv'                       # 2 × R: the least-squares inverse (J⁻¹ for a square J)
    ∂(i) = -sum(ntuple(k -> G[i, k] * ForwardDiff.partials(rD[k]), Val(length(rD))))
    return SVector(_with_value(D, pV[1], ∂(1)), _with_value(D, pV[2], ∂(2))), ok
end
