# ==============================================================================
# Target calibration — what a target calibration estimates about a planar target
# ==============================================================================
#
# A `PlanarTarget` (targets.jl) is a DESIGN — its nominal plane points — and a
# CALIBRATION taking the design to the physical target, the way a camera model
# is its intrinsics and its lens. A calibration is a CoordinateTransformations
# `Transformation` of design points, `c(p)`, carrying its parameters, with the
# protocol every kind implements (the target analogue of the lens chart,
# lens_chart.jl):
#
#   parameter_vector(c)            θ, an SVector (empty for Rigid)
#   parameter_names(c)             θ's names, so a configuration frees or fixes them
#   parameter_bounds(c)            unbounded (lens_chart.jl's default)
#   with_parameters(c, θ)          the same kind rebuilt from θ
#   transform_deriv_params(c, x)   ∂c(x)/∂θ at a design point x (CoordinateTransformations')
#   transform_deriv(c, x)          ∂c(x)/∂x                    (CoordinateTransformations')
#   regularizer(c)                 the prior's residual rows r(θ), ‖r‖² the penalty
#
# The kinds are `Rigid`, CoordinateTransformations' `AffineMap` (2-D), the two
# thin-plate kinds, and ONE composition, a height field after an in-plane kind:
# `HeightField ∘ AffineMap` or `HeightField ∘ ThinPlateSpline`, CT's `∘`. A
# composition's parameters, names, bounds, Jacobian columns, gauges and
# regularizer rows are in CT's order — the OUTER map's first, as
# `transform_deriv_params(t1 ∘ t2, x) = hcat(p1, m1 * p2)` stacks them.
#
# LENGTHS. A calibration's lengths are in the form of the design points it
# moves — plain numbers, or lengths. An `AffineMap` carries them as FlexUnits
# does (its translation plain or lengths); the thin-plate kinds store them as
# plain METRES with a `physical` flag (their spline system, coefficients and
# bending factor are plain Float64 arrays shared bit for bit with
# RadiometricCamera, and their rebuilds run on ForwardDiff Duals), and
# `ustrip(u, c)` / `with_unit(c, u)` move between the forms as they do for
# points. Applied to a physical point a calibration computes in plain metres
# and attaches metres again; its parameters and Jacobians are plain numbers,
# the lengths among them in metres.

using CoordinateTransformations: ComposedTransformation

"""
    regularizer(c | target) -> SVector

The prior's residual rows `r(θ)` at the current parameters: `‖r‖²` is the
penalty a calibrating estimator adds (a thin-plate spline's bending energy, a
shrinkage toward the design). Empty for [`Rigid`](@ref) and an `AffineMap`.

The rows are a QUADRATIC BLOCK: only `‖r‖²` (equivalently `LᵀL`) is defined.
The thin-plate kinds' rows are `L v` with `L` an eigen-factor of the bending
form, and an eigenbasis is not unique (a symmetric node grid has repeated
eigenvalues; the basis and the row signs follow the LAPACK build). So never
compare rows one by one (across machines or against stored values), and never
weight them one by one (a robust per-row weight would depend on that basis):
weight the block as a whole.
"""
function regularizer end

"""
    Rigid() <: CoordinateTransformations.Transformation

The design, exactly: the identity, no parameters, no regularizer, no lengths
(so it sits with plain and physical designs alike).
"""
struct Rigid <: CoordinateTransformations.Transformation end
(::Rigid)(p::StaticVector) = p

parameter_vector(::Rigid) = SVector{0, Float64}()
parameter_names(::Rigid) = ()
with_parameters(c::Rigid, θ::AbstractVector) =
    (isempty(θ) || throw(ArgumentError("Rigid takes no parameters; got $(length(θ))")); c)
transform_deriv_params(::Rigid, x) = zero(SMatrix{2, 0, Float64})
transform_deriv(::Rigid, x) = SMatrix{2, 2, Float64}(LinearAlgebra.I)
regularizer(::Rigid) = SVector{0, Float64}()

# A point in plain numbers (a physical one in metres), and a plain result put
# back in the form of the point it came from.
@inline _plain_point(p::StaticVector{N, <:Quantity}) where {N} = SVector{N}(map(q -> _strip_length(m, q), Tuple(p)))
@inline _plain_point(p::StaticVector) = SVector(Tuple(p))
@inline _restore_point(x::StaticVector, ::StaticVector{N, <:Quantity}) where {N} = SVector(map(v -> _attach_length(v, m), Tuple(x)))
@inline _restore_point(x::StaticVector, ::StaticVector) = x

# Lengths read at a constructor: all plain or all lengths (stored as plain
# metres), and which.
function _lengths(xs::AbstractVector)
    isempty(xs) && return (Float64[], false)
    _check_one_form(xs)
    return has_units(first(xs)) ? ([_strip_length(m, q) for q in xs], true) : (collect(xs), false)
end
# Node positions: 2-vectors, plain or lengths, as plain `Float64` (metres).
function _node_points(nodes::AbstractVector)
    isempty(nodes) && throw(ArgumentError("a thin-plate spline needs nodes"))
    all(p -> length(p) == 2, nodes) || throw(ArgumentError("a thin-plate spline's nodes are 2-vectors in the target's plane"))
    flat, physical = _lengths([x for p in nodes for x in p])
    return [SVector{2, Float64}(flat[2i - 1], flat[2i]) for i in eachindex(nodes)], physical
end
@noinline _refuse_form(what, x, physical) = throw(
    DimensionError(x, physical ? "a length (the nodes are lengths)" : "a plain number (the nodes are plain)")
)
# One plain length moved between metres and a unit `u` (FlexUnits' conversion).
@inline _to_metres(x::Real, u::Units) = strip_metres(Quantity(x, u))
@inline _from_metres(x::Real, u::Units) = strip_or_bare(u, Quantity(x, m))
@noinline _refuse_strip(c) = throw(ArgumentError("ustrip: the $(nameof(typeof(c))) is plain numbers already; `with_unit` attaches a unit"))
@noinline _refuse_attach(c) = throw(ArgumentError("with_unit: the $(nameof(typeof(c))) already has units"))
@noinline _refuse_length(c, θ) = throw(ArgumentError("$(nameof(typeof(c))) takes $(length(parameter_vector(c))) parameters; got $(length(θ))"))

# ------------------------------------------------------------------------------
# AffineMap — an in-plane print error
# ------------------------------------------------------------------------------
#
# CoordinateTransformations' `AffineMap(A, b)`, `p ↦ A·p + b`: `A` a
# dimensionless static 2 × 2 matrix, `b` a static 2-vector in the design's
# length form (plain, or lengths). CT owns the type, its call and
# `transform_deriv`; the calibration protocol on it is this package's.
# Parameters `(:a11, :a12, :a21, :a22, :bx, :by)` (`A` row by row, then `b`,
# lengths in metres), unbounded; `transform_deriv_params` is 2 × 6; no
# regularizer. With the pose estimated jointly, four of its six directions are
# the pose's (gauges): the translation `b`, the in-plane rotation and the
# uniform scale — see `gauge_names`. With a known pose, free them by name.
const _AffineCalibration = AffineMap{<:StaticMatrix{2, 2, <:Real}, <:StaticVector{2}}

@inline _plain_length(q::Quantity) = _strip_length(m, q)
@inline _plain_length(x::Real) = x
parameter_vector(c::_AffineCalibration) = SVector(
    c.linear[1, 1], c.linear[1, 2], c.linear[2, 1], c.linear[2, 2], _plain_length(c.translation[1]), _plain_length(c.translation[2])
)
parameter_names(::_AffineCalibration) = (:a11, :a12, :a21, :a22, :bx, :by)
function with_parameters(c::_AffineCalibration, θ::AbstractVector)
    length(θ) == 6 || _refuse_length(c, θ)
    T = float(eltype(θ))
    b = SVector{2, T}(θ[5], θ[6])
    return AffineMap(SMatrix{2, 2, T}(θ[1], θ[3], θ[2], θ[4]), has_units(c) ? map(x -> _attach_length(x, m), b) : b)
end
function transform_deriv_params(::_AffineCalibration, x)
    q = _plain_point(x)
    o, z = one(eltype(q)), zero(eltype(q))
    # rows ∂(A q + b)₁ = (q₁, q₂, 0, 0, 1, 0), ∂(A q + b)₂ = (0, 0, q₁, q₂, 0, 1); column-major
    return SMatrix{2, 6}(q[1], z, q[2], z, z, q[1], z, q[2], o, z, z, o)
end
regularizer(::_AffineCalibration) = SVector{0, Float64}()

_has_lengths(::_AffineCalibration) = true
has_units(c::_AffineCalibration) = eltype(c.translation) <: Quantity
_length_sample(c::_AffineCalibration) = c.translation[1]
FlexUnits.ustrip(u::Units, c::_AffineCalibration) =
    has_units(c) ? AffineMap(c.linear, map(q -> _from_metres(_strip_length(m, q), u), c.translation)) : _refuse_strip(c)
with_unit(c::_AffineCalibration, u::Units) =
    has_units(c) ? _refuse_attach(c) : AffineMap(c.linear, map(x -> _attach_length(_to_metres(x, u), m), c.translation))

# ------------------------------------------------------------------------------
# The thin-plate spline kinds — an in-plane residual and an out-of-plane height
# ------------------------------------------------------------------------------
#
# Both are splines over K FIXED nodes in the target's plane (geometry, not
# parameters) whose parameters are the nodal values, built on
# numerics/thin_plate_spline.jl: the solver `ThinPlateSplineSystem` is formed
# once per node set, and the coefficients `(w, a)` once per parameter vector
# (in the constructor and `with_parameters`), so applying the calibration is
# `tps_evaluate` and nothing else. The Jacobian is the spline's
# `nodal_weights` (it is linear in its values).
#
# THE REGULARIZER: the bending energy `vᵀQv` of each spline's nodal values `v`,
# as residual rows `L v` with `‖L v‖² = vᵀQv`. `Q` is symmetric positive
# semidefinite with the affine values in its null space (3-dimensional), so it
# has no Cholesky factor; it is factored once per node set by its
# eigendecomposition `Q = V Λ Vᵀ`, `L = Λ₊^½ Vᵀ` (K × K) with `Λ₊` the
# eigenvalues clamped at zero (the null space's rounding residue, |λ| ~ 1e-16
# relative, may come out negative). No shift is added, so `‖L v‖²` is the
# bending energy to rounding, not a biased one; the rows along the null
# space are zero.

_bending_rows(L::Matrix{Float64}, v::SVector{K}) where {K} = SVector{K}(L * v)
_node_lengths(system, physical) = physical ? [map(x -> _attach_length(x, m), p) for p in system.nodes] : collect(system.nodes)
# 2 × 2K: (nᵀ, 0) over (0, nᵀ) — the two splines' nodal weights
_two_rows(a::SVector{K}, b::SVector{K}) where {K} = SMatrix{2, K}(ntuple(k -> isodd(k) ? a[(k + 1) ÷ 2] : b[k ÷ 2], Val(2K)))
_wrap(v::AbstractVector, ::Val{K}, o, ::Type{T}) where {K, T} = SVector{K, T}(ntuple(i -> v[o + i], Val(K)))

"""
    ThinPlateSpline(nodes, fx, fy) <: CoordinateTransformations.Transformation
    ThinPlateSpline(nodes)                       zero displacement

An in-plane residual, `p ↦ p + (f_x(p), f_y(p))`: two thin-plate splines
(numerics/thin_plate_spline.jl) over the same `K` fixed `nodes` in the
target's plane, through the nodal displacements `fx` and `fy`. Nodes and
displacements are in the design's length form (plain, or lengths).
Parameters `(:fx1, …, :fxK, :fy1, …, :fyK)`, unbounded;
`transform_deriv_params` is 2 × 2K (the splines' `nodal_weights`);
[`regularizer`](@ref) the two splines' bending-energy rows (`2K` rows; a
quadratic block, see its docstring).

It is a RESIDUAL: its affine part (each spline's `1`, `u`, `v` over the nodes)
is fixed by the default [`TargetManifold`](@ref) — `gauge_names` `(:fx_1,
:fx_u, :fx_v, :fy_1, :fy_u, :fy_v)` — because four of those six directions
are the pose's. An in-plane kind does not compose with another: free `:fx_u`,
… by name to let the spline carry a print stretch and shear. A
[`HeightField`](@ref) composes after it, `HeightField ∘ ThinPlateSpline`.
"""
struct ThinPlateSpline{K, T} <: CoordinateTransformations.Transformation
    system::ThinPlateSplineSystem{K}
    fx::SVector{K, T}
    fy::SVector{K, T}
    wx::SVector{K, T}
    ax::SVector{3, T}
    wy::SVector{K, T}
    ay::SVector{3, T}
    bending::Matrix{Float64}
    physical::Bool
end
function _thin_plate_spline(system::ThinPlateSplineSystem{K}, L, fx::SVector{K, T}, fy::SVector{K, T}, physical) where {K, T}
    wx, ax = tps_coefficients(system, Vector(fx))   # a Vector, as RadiometricCamera supplies its values:
    wy, ay = tps_coefficients(system, Vector(fy))   # the same BLAS product, the same bits
    return ThinPlateSpline{K, T}(system, fx, fy, wx, ax, wy, ay, L, physical)
end
function ThinPlateSpline(nodes::AbstractVector, fx::AbstractVector, fy::AbstractVector)
    K = length(nodes)
    length(fx) == length(fy) == K || throw(DimensionMismatch("ThinPlateSpline: $K nodes, $(length(fx)) and $(length(fy)) displacements"))
    ns, physical = _node_points(nodes)
    vx, px_ = _lengths(collect(fx))
    vy, py_ = _lengths(collect(fy))
    px_ == physical || _refuse_form("ThinPlateSpline", first(fx), physical)
    py_ == physical || _refuse_form("ThinPlateSpline", first(fy), physical)
    system = ThinPlateSplineSystem(ns)
    T = float(promote_type(eltype(vx), eltype(vy)))
    return _thin_plate_spline(system, bending_factor(system), SVector{K, T}(vx), SVector{K, T}(vy), physical)
end
function ThinPlateSpline(nodes::AbstractVector)
    ns, physical = _node_points(nodes)
    K = length(ns)
    system = ThinPlateSplineSystem(ns)
    return _thin_plate_spline(system, bending_factor(system), zero(SVector{K, Float64}), zero(SVector{K, Float64}), physical)
end
_apply(c::ThinPlateSpline, q) =
    q + SVector(tps_evaluate(c.system.nodes, c.wx, c.ax, q), tps_evaluate(c.system.nodes, c.wy, c.ay, q))
function _point_jacobian(c::ThinPlateSpline, q)
    gx = tps_gradient(c.system.nodes, c.wx, c.ax, q)
    gy = tps_gradient(c.system.nodes, c.wy, c.ay, q)
    return SMatrix{2, 2}(1 + gx[1], gy[1], gx[2], 1 + gy[2])
end

parameter_vector(c::ThinPlateSpline) = vcat(c.fx, c.fy)
# parameter_names(::ThinPlateSpline): generated from the type, beside the name tables below
function with_parameters(c::ThinPlateSpline{K}, θ::AbstractVector) where {K}
    length(θ) == 2K || _refuse_length(c, θ)
    T = float(eltype(θ))
    return _thin_plate_spline(c.system, c.bending, _wrap(θ, Val(K), 0, T), _wrap(θ, Val(K), K, T), c.physical)
end
function transform_deriv_params(c::ThinPlateSpline{K}, x) where {K}
    n = nodal_weights(c.system, _plain_point(x))
    z = zero(n)
    return hcat(_two_rows(n, z), _two_rows(z, n))
end
regularizer(c::ThinPlateSpline) = vcat(_bending_rows(c.bending, c.fx), _bending_rows(c.bending, c.fy))

_has_lengths(::ThinPlateSpline) = true
has_units(c::ThinPlateSpline) = c.physical
_length_sample(c::ThinPlateSpline) = first(first(_node_lengths(c.system, c.physical)))
length_mask(::Type{<:ThinPlateSpline{K}}) where {K} = SVector{2K}(ntuple(_ -> true, Val(2K)))
# A unit move to or from metres leaves the stored nodes and values as they are
# (lengths are stored as plain metres): the spline's solver, coefficients and
# bending factor are reused, only the form flips. Any other unit refits them.
# (A unit of another dimension is not metres: it takes the rebuild, which refuses it
# with FlexUnits' `DimensionError`.)
_in_metres(u::Units) = dimension(u) == dimension(m) && _metre_factor(u) == 1
function FlexUnits.ustrip(u::Units, c::ThinPlateSpline{K, T}) where {K, T}
    c.physical || _refuse_strip(c)
    _in_metres(u) && return ThinPlateSpline{K, T}(c.system, c.fx, c.fy, c.wx, c.ax, c.wy, c.ay, c.bending, false)
    return ThinPlateSpline([map(x -> _from_metres(x, u), p) for p in c.system.nodes], map(x -> _from_metres(x, u), c.fx), map(x -> _from_metres(x, u), c.fy))
end
function with_unit(c::ThinPlateSpline{K, T}, u::Units) where {K, T}
    c.physical && _refuse_attach(c)
    _in_metres(u) && return ThinPlateSpline{K, T}(c.system, c.fx, c.fy, c.wx, c.ax, c.wy, c.ay, c.bending, true)
    to(x) = _attach_length(x, u)
    return ThinPlateSpline([map(to, p) for p in c.system.nodes], map(to, c.fx), map(to, c.fy))
end
Base.:(==)(a::ThinPlateSpline, b::ThinPlateSpline) =
    a.system.nodes == b.system.nodes && a.fx == b.fx && a.fy == b.fy && a.physical == b.physical
Base.hash(c::ThinPlateSpline, h::UInt) = hash(c.physical, hash(c.fy, hash(c.fx, hash(c.system.nodes, hash(:ThinPlateSpline, h)))))
Base.show(io::IO, c::ThinPlateSpline{K}) where {K} =
    print(io, "ThinPlateSpline over ", K, " nodes", c.physical ? " (lengths in metres)" : "")

"""
    HeightField(nodes, heights) <: CoordinateTransformations.Transformation
    HeightField(nodes)                           flat

An out-of-plane deformation, `p ↦ (p, h(p))`: one thin-plate spline
(numerics/thin_plate_spline.jl) over `K` fixed `nodes` in the target's plane
through the nodal `heights`, both in the design's length form (plain, or
lengths). The height at `p` is `tps_evaluate` of the spline, nothing else.
Parameters `(:h1, …, :hK)`, unbounded; `transform_deriv_params` is 3 × K
(zero in the plane, the spline's `nodal_weights` in z); [`regularizer`](@ref)
the bending-energy rows (`K` rows).

A target with a height field is not planar ([`is_planar`](@ref)): its
[`points`](@ref) are 3-vectors. It composes after an in-plane kind,
`HeightField ∘ AffineMap` or `HeightField ∘ ThinPlateSpline`, its nodes then in
that kind's output (the calibrated plane, not the design). Its piston and two
tilts are the pose's (gauges): the default [`TargetManifold`](@ref) holds them
fixed — see [`gauge_names`](@ref).
"""
struct HeightField{K, T} <: CoordinateTransformations.Transformation
    system::ThinPlateSplineSystem{K}
    h::SVector{K, T}
    w::SVector{K, T}
    a::SVector{3, T}
    bending::Matrix{Float64}
    physical::Bool
end
function _height_field(system::ThinPlateSplineSystem{K}, L, h::SVector{K, T}, physical) where {K, T}
    w, a = tps_coefficients(system, Vector(h))   # a Vector, as RadiometricCamera supplies its heights: the same bits
    return HeightField{K, T}(system, h, w, a, L, physical)
end
function HeightField(nodes::AbstractVector, heights::AbstractVector)
    K = length(nodes)
    length(heights) == K || throw(DimensionMismatch("HeightField: $K nodes, $(length(heights)) heights"))
    ns, physical = _node_points(nodes)
    v, ph = _lengths(collect(heights))
    ph == physical || _refuse_form("HeightField", first(heights), physical)
    system = ThinPlateSplineSystem(ns)
    return _height_field(system, bending_factor(system), SVector{K, float(eltype(v))}(v), physical)
end
function HeightField(nodes::AbstractVector)
    ns, physical = _node_points(nodes)
    system = ThinPlateSplineSystem(ns)
    return _height_field(system, bending_factor(system), zero(SVector{length(ns), Float64}), physical)
end
_apply(c::HeightField, q) = SVector(q[1], q[2], tps_evaluate(c.system.nodes, c.w, c.a, q))
function _point_jacobian(c::HeightField, q)
    g = tps_gradient(c.system.nodes, c.w, c.a, q)
    o, z = one(eltype(g)), zero(eltype(g))
    return SMatrix{3, 2}(o, z, g[1], z, o, g[2])
end

parameter_vector(c::HeightField) = c.h
# parameter_names(::HeightField): generated from the type, beside the name tables below
function with_parameters(c::HeightField{K}, θ::AbstractVector) where {K}
    length(θ) == K || _refuse_length(c, θ)
    return _height_field(c.system, c.bending, _wrap(θ, Val(K), 0, float(eltype(θ))), c.physical)
end
function transform_deriv_params(c::HeightField{K}, x) where {K}
    n = nodal_weights(c.system, _plain_point(x))
    return SMatrix{3, K}(ntuple(k -> mod1(k, 3) == 3 ? n[(k + 2) ÷ 3] : 0.0, Val(3K)))
end
regularizer(c::HeightField) = _bending_rows(c.bending, c.h)

_has_lengths(::HeightField) = true
has_units(c::HeightField) = c.physical
_length_sample(c::HeightField) = first(first(_node_lengths(c.system, c.physical)))
length_mask(::Type{<:HeightField{K}}) where {K} = SVector{K}(ntuple(_ -> true, Val(K)))
function FlexUnits.ustrip(u::Units, c::HeightField{K, T}) where {K, T}
    c.physical || _refuse_strip(c)
    _in_metres(u) && return HeightField{K, T}(c.system, c.h, c.w, c.a, c.bending, false)
    return HeightField([map(x -> _from_metres(x, u), p) for p in c.system.nodes], map(x -> _from_metres(x, u), c.h))
end
function with_unit(c::HeightField{K, T}, u::Units) where {K, T}
    c.physical && _refuse_attach(c)
    _in_metres(u) && return HeightField{K, T}(c.system, c.h, c.w, c.a, c.bending, true)
    to(x) = _attach_length(x, u)
    return HeightField([map(to, p) for p in c.system.nodes], map(to, c.h))
end
Base.:(==)(a::HeightField, b::HeightField) = a.system.nodes == b.system.nodes && a.h == b.h && a.physical == b.physical
Base.hash(c::HeightField, h::UInt) = hash(c.physical, hash(c.h, hash(c.system.nodes, hash(:HeightField, h))))
Base.show(io::IO, c::HeightField{K}) where {K} =
    print(io, "HeightField over ", K, " nodes", c.physical ? " (lengths in metres)" : "")

# The thin-plate kinds applied to a design point: the kind computes on plain
# numbers (`_apply`), the point keeps its form. A kind with lengths and a point
# of the other form is FlexUnits' `DimensionError`.
function (c::Union{ThinPlateSpline, HeightField})(p::StaticVector)
    _check_one_form((c, p))
    return _restore_point(_apply(c, _plain_point(p)), p)
end
transform_deriv(c::Union{ThinPlateSpline, HeightField}, x) = _point_jacobian(c, _plain_point(x))

# ------------------------------------------------------------------------------
# The composition: a height field after an in-plane kind
# ------------------------------------------------------------------------------
#
# CoordinateTransformations' `∘`, its call and its chain-rule Jacobians
# (`transform_deriv`, `transform_deriv_params`: `hcat(p1, m1 * p2)`, the
# outer map's parameters first) are CT's. The rest of the protocol is here,
# stacked in the same order: the height field's, then the in-plane kind's.
const _InPlaneCalibration = Union{_AffineCalibration, ThinPlateSpline}
const _LiftedCalibration = ComposedTransformation{<:HeightField, <:_InPlaneCalibration}

"""
    TargetCalibration

What a target calibration estimates about a planar target: a
CoordinateTransformations `Transformation` `c(p)` of its design points to the
physical target — one of [`Rigid`](@ref), a 2-D `AffineMap` (static `A`, `b`),
[`ThinPlateSpline`](@ref), [`HeightField`](@ref), or a height field after an
in-plane kind, `HeightField ∘ AffineMap` or `HeightField ∘ ThinPlateSpline`.
The union of those, the one type every calibration-typed method takes.

The protocol: [`parameter_vector`](@ref), [`parameter_names`](@ref),
[`parameter_bounds`](@ref), [`with_parameters`](@ref)`(c, θ)`,
CoordinateTransformations' `transform_deriv_params(c, x)` (`∂c(x)/∂θ`) and
`transform_deriv(c, x)` (`∂c(x)/∂x`), [`regularizer`](@ref)`(c)`,
[`gauge_names`](@ref), [`length_mask`](@ref). A composition's are stacked in
CT's order, the height field's first; its names carry each member's tag,
`:height_field_h1`, …, `:affine_a11`, …. Jacobians are plain numbers; for a
physical calibration the point and the length parameters are in metres.

Two in-plane kinds never compose (an `AffineMap` after a `ThinPlateSpline`, a
`Rigid` member): a [`PlanarTarget`](@ref) refuses any other composition with an
`ArgumentError`.
"""
const TargetCalibration = Union{Rigid, _AffineCalibration, ThinPlateSpline, HeightField, _LiftedCalibration}

parameter_vector(c::_LiftedCalibration) = vcat(parameter_vector(c.t1), parameter_vector(c.t2))
function with_parameters(c::_LiftedCalibration, θ::AbstractVector)
    n1, n2 = length(parameter_vector(c.t1)), length(parameter_vector(c.t2))
    length(θ) == n1 + n2 || _refuse_length(c, θ)
    return with_parameters(c.t1, [θ[i] for i in 1:n1]) ∘ with_parameters(c.t2, [θ[n1 + i] for i in 1:n2])
end
regularizer(c::_LiftedCalibration) = vcat(regularizer(c.t1), regularizer(c.t2))
_has_lengths(::_LiftedCalibration) = true
has_units(c::_LiftedCalibration) = has_units(c.t2)
_length_sample(c::_LiftedCalibration) = _length_sample(c.t2)
length_mask(::Type{ComposedTransformation{A, B}}) where {A <: HeightField, B <: _InPlaneCalibration} =
    vcat(length_mask(A), length_mask(B))
FlexUnits.ustrip(u::Units, c::_LiftedCalibration) = has_units(c) ? ustrip(u, c.t1) ∘ ustrip(u, c.t2) : _refuse_strip(c)
with_unit(c::_LiftedCalibration, u::Units) = has_units(c) ? _refuse_attach(c) : with_unit(c.t1, u) ∘ with_unit(c.t2, u)
# CoordinateTransformations defines `==` on a composition, member by member,
# and no `hash`: the default hashes the thin-plate kinds' arrays by identity.
Base.hash(c::_LiftedCalibration, h::UInt) = hash(c.t2, hash(c.t1, hash(:ComposedTransformation, h)))

# --- lengths ---------------------------------------------------------------------
#
# `_has_lengths(c)`: whether a kind carries lengths at all (Rigid does not);
# `has_units(c)`: whether they are physical. `_length_sample(c)`: one of them in
# its own form, to name the mismatch in a `DimensionError`. `length_mask(c)`:
# which entries of θ are lengths (they scale with a unit change; the rest are
# dimensionless).
_has_lengths(::Rigid) = false
# the one-form rule's view of a calibration (euclidean_map.jl `_check_one_form`)
_unit_form(c::TargetCalibration) = _has_lengths(c) ? (has_units(c) ? :physical : :plain) : :none
# (a calibration with lengths: `Rigid` has none, so never a form to name)
_form_sample(c::Union{_AffineCalibration, ThinPlateSpline, HeightField, _LiftedCalibration}) = _length_sample(c)
has_units(::Rigid) = false
"""
    length_mask(::Type{C}) -> SVector{n, Bool}
    length_mask(c::TargetCalibration) -> SVector{n, Bool}

Which entries of [`parameter_vector`](@ref)`(c)` are lengths: `true` for an
entry that scales with a unit change (stored in metres for a physical
calibration), `false` for a dimensionless one — a function of the TYPE.
[`Rigid`](@ref): none; an `AffineMap`: `(false, false, false, false, true,
true)` (`A` dimensionless, `b` lengths); [`ThinPlateSpline`](@ref) and
[`HeightField`](@ref): every entry; a composition: its members', the height
field's first. A covariance over `θ` moves with a unit change by this mask; a
record of the calibration states its lengths by it.
"""
function length_mask end
length_mask(c::TargetCalibration) = length_mask(typeof(c))
length_mask(::Type{Rigid}) = SVector{0, Bool}()
length_mask(::Type{<:_AffineCalibration}) = SVector(false, false, false, false, true, true)
FlexUnits.ustrip(::Units, c::Rigid) = c
with_unit(c::Rigid, ::Units) = c

# A calibration's form moves with its unit: an uncertain one's covariance
# scales with it over its length entries (`_scale_length_form`, euclidean_map.jl).
_ustrip_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? ustrip(u, c) : c
_attach_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? with_unit(c, u) : c
function _ustrip_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(ustrip(u, g), _scale_length_form(getfield(c, :attributes), length_mask(g), _metre_factor(u), *))
end
function _attach_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(with_unit(g, u), _scale_length_form(getfield(c, :attributes), length_mask(g), _metre_factor(u), /))
end

# The internal pair a target's surface chart reads (target_surfaces.jl): a
# planar calibration applied to a plain point, and its point Jacobian there.
_apply(::Rigid, q) = q
_apply(c::_AffineCalibration, q) = c(q)
_point_jacobian(c::Union{Rigid, _AffineCalibration}, q) = transform_deriv(c, q)

# ------------------------------------------------------------------------------
# Names as a function of the type (what a composition's generated names read)
# ------------------------------------------------------------------------------

_static_parameter_names(::Type{Rigid}) = ()
_static_parameter_names(::Type{<:_AffineCalibration}) = (:a11, :a12, :a21, :a22, :bx, :by)
_static_parameter_names(::Type{<:ThinPlateSpline{K}}) where {K} =
    (ntuple(i -> Symbol(:fx, i), K)..., ntuple(i -> Symbol(:fy, i), K)...)
_static_parameter_names(::Type{<:HeightField{K}}) where {K} = ntuple(i -> Symbol(:h, i), K)
_static_gauge_names(::Type{Rigid}) = ()
_static_gauge_names(::Type{<:_AffineCalibration}) = (:translation_x, :translation_y, :rotation, :scale)
_static_gauge_names(::Type{<:ThinPlateSpline}) = (:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v)
_static_gauge_names(::Type{<:HeightField}) = (:piston, :tilt_x, :tilt_y)
# A literal tuple per type, so a NamedTuple over the names is inferred.
@generated parameter_names(c::Union{ThinPlateSpline, HeightField}) = QuoteNode(_static_parameter_names(c))
# A composition's names: each member's, prefixed by its tag, in CT's order.
_composed_names(f, ::Type{ComposedTransformation{A, B}}) where {A, B} =
    (_prefixed(calibration_tag(A), f(A))..., _prefixed(calibration_tag(B), f(B))...)
_prefixed(tag, names::Tuple) = map(n -> Symbol(tag, :_, n), names)
@generated parameter_names(c::_LiftedCalibration) = QuoteNode(_composed_names(_static_parameter_names, c))

"""
    TARGET_CALIBRATION_TAGS

The [`TagRegistry`](@ref) of the target calibration kinds — the one map from a
kind's tag to its type, read by a composition's parameter names
(`:affine_a11`) and by VisualGeometryToolkitIO's target record (`"type":
"affine"`): `"rigid"`, `"affine"` (CoordinateTransformations' `AffineMap`, the
family's suffix `Map` stripped), `"thin_plate_spline"`, `"height_field"`,
`snake_case` of the type's name. A composition has no tag: its record is
written by its structure.
"""
const TARGET_CALIBRATION_TAGS = TagRegistry(:target_calibration, "Map",
    "rigid" => Rigid, "affine" => AffineMap, "thin_plate_spline" => ThinPlateSpline, "height_field" => HeightField)

"""
    calibration_tag(::Type{C}) -> String

The tag of the calibration kind `C`: the prefix of its parameter and gauge
names in a composition (`:affine_a11`) and the `type` of its record — its
[`TARGET_CALIBRATION_TAGS`](@ref) tag. A composition has none (an
`ArgumentError`).
"""
function calibration_tag end
calibration_tag(::Type{T}) where {T} = tag_of(TARGET_CALIBRATION_TAGS, T)

# A Rigid calibration has no parameters, so nothing for a covariance to be
# over: `Uncertain(Rigid(), Σ)` is refused at construction (the target record
# writes no covariance for it either).
(::Type{Attributed})(::Rigid, ::Union{CovMat, PrecMat}) = throw(
    ArgumentError("a Rigid target calibration has no parameters, so no covariance: Uncertain(Rigid(), Σ) is refused — use Rigid()")
)

# ------------------------------------------------------------------------------
# Planarity — in the type
# ------------------------------------------------------------------------------

"""
    is_planar(::Type{C}) -> Bool
    is_planar(c::TargetCalibration) / is_planar(t::PlanarTarget) -> Bool

Whether a calibration keeps the design in its plane — a function of the TYPE,
so it folds at compile time: `true` for [`Rigid`](@ref), an `AffineMap` and
[`ThinPlateSpline`](@ref), `false` for a [`HeightField`](@ref) and a
composition (which ends in one). A planar target's [`points`](@ref) are
`SVector{2}`s (on `z = 0` of its frame), a non-planar one's `SVector{3}`s. An
[`Uncertain`](@ref) calibration is planar when its kind is.
"""
function is_planar end
is_planar(::Type{Rigid}) = true
is_planar(::Type{<:_AffineCalibration}) = true
is_planar(::Type{<:ThinPlateSpline}) = true
is_planar(::Type{<:HeightField}) = false
is_planar(::Type{<:_LiftedCalibration}) = false
is_planar(::Type{<:Attributed{C, <:Union{CovMat, PrecMat}}}) where {C <: TargetCalibration} = is_planar(C)
is_planar(c::TargetCalibration) = is_planar(typeof(c))
is_planar(u::Uncertain{<:TargetCalibration}) = is_planar(typeof(geom(u)))

"""
    is_affine(::Type{C}) -> Bool
    is_affine(c::TargetCalibration) -> Bool

Whether a calibration's IN-PLANE part — where it takes a design point in the
target's plane, a height field's lift aside — is an affine map of the design,
`p ↦ A p + b`: a function of the TYPE, `true` for [`Rigid`](@ref), an
`AffineMap`, [`HeightField`](@ref) (it leaves the plane point where it is)
and `HeightField ∘ AffineMap`, `false` for a [`ThinPlateSpline`](@ref) and
`HeightField ∘ ThinPlateSpline`. A planar target's [`surface`](@ref) reads its
chart's affine off such a calibration (its image of the design origin and its
point Jacobian there) and refuses any other.
"""
function is_affine end
is_affine(::Type{Rigid}) = true
is_affine(::Type{<:_AffineCalibration}) = true
is_affine(::Type{<:ThinPlateSpline}) = false
is_affine(::Type{<:HeightField}) = true
is_affine(::Type{ComposedTransformation{A, B}}) where {A <: HeightField, B <: _InPlaneCalibration} = is_affine(B)
is_affine(c::TargetCalibration) = is_affine(typeof(c))

"""
    PlanarCalibration

The planar calibration kinds — [`Rigid`](@ref), a 2-D `AffineMap` and
[`ThinPlateSpline`](@ref) — as one type, for a method that assumes the design
stays on `z = 0`: `f(t::PlanarTarget{<:PlanarCalibration})` refuses a target
with a [`HeightField`](@ref) by dispatch (a `MethodError`).
"""
const PlanarCalibration = Union{Rigid, _AffineCalibration, ThinPlateSpline}

# ------------------------------------------------------------------------------
# Gauges — the directions the pose absorbs
# ------------------------------------------------------------------------------

"""
    gauge_names(c | target) -> Tuple{Vararg{Symbol}}

A calibration's GAUGES: the directions of its parameter vector θ (length `n`)
that images of the target cannot observe because the target's pose absorbs
them, by name; [`gauge_directions`](@ref) gives them as columns in θ's
coordinates (plain numbers; lengths in metres). The default [`TargetManifold`](@ref) holds them fixed; a caller
frees one by name, `TargetManifold(t; free = :scale)`.

These defaults assume the pose is estimated jointly — that is what
`minimal_manifold(t)` means. With a known pose (motion capture, a fixture),
every kind's parameters are observable: free the gauges by name.

- An `AffineMap`: `:translation_x`, `:translation_y` (the axes `bx`, `by`),
  `:rotation` (`δA = J A₀`, `J` the quarter turn, `b` held: the direction
  `(-a21, -a22, a11, a12, 0, 0)`) and `:scale` (`δA = A₀`: `(a11, a12, a21,
  a22, 0, 0)`), the last two evaluated at the anchor `A₀`. A target and the
  same target shifted, turned in its plane or scaled are the same images once
  the pose moves with it (the scale with the pose's translation), so only the
  anisotropic stretch and the shear stay free; a measured length (a ruler)
  frees `:scale`.
- [`ThinPlateSpline`](@ref): its whole affine part, `:fx_1`, `:fx_u`, `:fx_v`,
  `:fy_1`, `:fy_u`, `:fy_v` — the columns `[1;0]`, `[u;0]`, `[v;0]`, `[0;1]`,
  `[0;u]`, `[0;v]` over the nodes, constants. Four are the pose's (a
  similarity); free `:fx_u`, … by name to let the spline carry a stretch and
  shear.
- [`HeightField`](@ref): `:piston`, `:tilt_x`, `:tilt_y` — heights that are an
  affine function of the node position, `hᵢ = α + β uᵢ + γ vᵢ`, the columns
  `1`, `u` and `v` over the nodes. The spline through affine heights is that
  plane, which the pose's translation and two rotations reproduce; fixing them
  leaves the heights orthogonal to every plane (`Σ hᵢ = Σ uᵢhᵢ = Σ vᵢhᵢ = 0`
  for the steps).
- A composition: its members', each name prefixed by its member's tag, the
  height field's first (`:height_field_piston`, …, `:affine_scale`).
- [`Rigid`](@ref): none.
"""
function gauge_names end
gauge_names(::Rigid) = ()
gauge_names(c::Union{_AffineCalibration, ThinPlateSpline, HeightField}) = _static_gauge_names(typeof(c))
@generated gauge_names(c::_LiftedCalibration) = QuoteNode(_composed_names(_static_gauge_names, c))
gauge_names(u::Uncertain{<:TargetCalibration}) = gauge_names(geom(u))

"""
    gauge_directions(c) -> Matrix{Float64}

The gauges of [`gauge_names`](@ref) as the columns of an `n × g` matrix in the
coordinates of `parameter_vector(c)`, evaluated at `c`.
"""
function gauge_directions end
gauge_directions(c::Rigid) = zeros(Float64, 0, 0)
function gauge_directions(c::_AffineCalibration)
    a11, a12, a21, a22 = Float64.(Tuple(parameter_vector(c))[1:4])
    s = hypot(a11, a12, a21, a22)
    s > 0 || throw(ArgumentError("gauge_directions: an AffineMap with A = 0 has no rotation or scale direction"))
    # rows: θ = (a11, a12, a21, a22, bx, by); columns: translation_x, translation_y,
    # rotation (δA = J A₀) and scale (δA = A₀), the last two of unit norm
    return [0.0 0.0 -a21/s a11/s
            0.0 0.0 -a22/s a12/s
            0.0 0.0  a11/s a21/s
            0.0 0.0  a12/s a22/s
            1.0 0.0  0.0   0.0
            0.0 1.0  0.0   0.0]
end
function gauge_directions(c::ThinPlateSpline{K}) where {K}
    o, z = ones(K), zeros(K)
    u = [Float64(p[1]) for p in c.system.nodes]
    v = [Float64(p[2]) for p in c.system.nodes]
    return [[o; z] [u; z] [v; z] [z; o] [z; u] [z; v]]
end
gauge_directions(c::HeightField{K}) where {K} =
    [ones(K) [Float64(p[1]) for p in c.system.nodes] [Float64(p[2]) for p in c.system.nodes]]
function gauge_directions(c::_LiftedCalibration)
    D1, D2 = gauge_directions(c.t1), gauge_directions(c.t2)
    D = zeros(Float64, size(D1, 1) + size(D2, 1), size(D1, 2) + size(D2, 2))
    D[1:size(D1, 1), 1:size(D1, 2)] = D1
    D[(size(D1, 1) + 1):end, (size(D1, 2) + 1):end] = D2
    return D
end

# --- an uncertain calibration -------------------------------------------------------
#
# `Uncertain(c, CovMat(Σ))`, Σ over `parameter_vector(c)`, as a pose is
# wrapped; the read-only protocol answers through it. `with_parameters` is the
# kind's: a new parameter vector is a new calibration, without the old one's
# covariance.
for f in (:parameter_vector, :parameter_names, :parameter_bounds, :regularizer)
    @eval $f(u::Uncertain{<:TargetCalibration}) = $f(geom(u))
end
transform_deriv_params(u::Uncertain{<:TargetCalibration}, x) = transform_deriv_params(geom(u), x)

"""
    with_parameters(u::Uncertain{<:TargetCalibration}, θ) -> TargetCalibration

The BARE kind rebuilt from `θ`: an estimate's covariance describes the
parameters it was estimated at and does not survive a change of them, so the
result is not `Uncertain`. (A `PlanarTarget`'s `with_parameters` does the same
to its calibration.)
"""
with_parameters(u::Uncertain{<:TargetCalibration}, θ::AbstractVector) = with_parameters(geom(u), θ)

# ------------------------------------------------------------------------------
# What a target holds — checked where a target is built (targets.jl)
# ------------------------------------------------------------------------------

# A calibration as a target stores it: an AffineMap's translation in the
# points' stored form (lengths in metres), its matrix dimensionless; a
# composition's members in one form. Anything but the kinds above and the one
# composition is refused.
_stored_calibration(c::Union{Rigid, ThinPlateSpline, HeightField}) = c
function _stored_calibration(c::_AffineCalibration)
    eltype(c.linear) <: Quantity && throw(ArgumentError("an AffineMap calibration's matrix A is dimensionless; the lengths are its translation b's"))
    return has_units(c) ? AffineMap(c.linear, _stored_point(Val(2), c.translation)) : c
end
function _stored_calibration(c::_LiftedCalibration)
    _check_one_form((c.t1, c.t2))
    return c.t1 ∘ _stored_calibration(c.t2)
end
_stored_calibration(c::Uncertain) = Uncertain(_stored_calibration(geom(c)), getfield(c, :attributes))
_stored_calibration(c::ComposedTransformation) = _refuse_composition(c)
_stored_calibration(c) = throw(
    ArgumentError(
        "a target calibration is Rigid, a 2-D AffineMap (static A and b), a ThinPlateSpline, a HeightField, " *
            "or HeightField ∘ AffineMap / HeightField ∘ ThinPlateSpline; got a $(nameof(typeof(geom(c))))"
    )
)
@noinline _refuse_composition(c) = throw(
    ArgumentError(
        "a target calibration composes only a HeightField after one in-plane kind — HeightField ∘ AffineMap or " *
            "HeightField ∘ ThinPlateSpline; in-plane kinds never compose with each other (a ThinPlateSpline frees its " *
            "own stretch and shear by name). Got $(_composition_spelling(c))"
    )
)
_composition_spelling(c::ComposedTransformation) = "$(_composition_spelling(c.t1)) ∘ $(_composition_spelling(c.t2))"
_composition_spelling(c) = string(nameof(typeof(c)))
