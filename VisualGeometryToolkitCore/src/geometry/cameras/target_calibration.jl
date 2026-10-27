# ==============================================================================
# Target calibration — what a target calibration estimates about a planar target
# ==============================================================================
#
# A `PlanarTarget` (targets.jl) is a DESIGN — its nominal plane points — and a
# CALIBRATION taking the design to the physical target, the way a camera model
# is its intrinsics and its lens. A calibration is a map of design points
# `c(p)` carrying its parameters, with the protocol every kind implements (the
# target analogue of the lens chart, lens_chart.jl):
#
#   parameter_vector(c)            θ, an SVector (empty for Rigid)
#   parameter_names(c)             θ's names, so a configuration frees or fixes them
#   parameter_bounds(c)            unbounded (lens_chart.jl's default)
#   with_parameters(c, θ)          the same kind rebuilt from θ
#   calibration_jacobian(c, x)     ∂c(x)/∂θ at a design point x
#   regularizer(c)                 the prior's residual rows r(θ), ‖r‖² the penalty
#
# LENGTHS. A calibration's lengths are in the form of the design points it
# moves — plain numbers, or lengths stored as plain METRES with `has_units(c)`
# true — and `ustrip(u, c)` / `with_unit(c, u)` move between the forms as they
# do for points. Applied to a physical point it strips the point in metres,
# computes in plain numbers, and attaches metres again; its parameters and
# Jacobians are plain numbers, the lengths among them in metres.

"""
    TargetCalibration

What a target calibration estimates about a planar target: a map `c(p)` of
its design points to the physical target, with the parameter protocol
[`parameter_vector`](@ref), [`parameter_names`](@ref),
[`parameter_bounds`](@ref), [`with_parameters`](@ref)`(c, θ)`,
[`calibration_jacobian`](@ref)`(c, x)` and [`regularizer`](@ref)`(c)`.
[`Rigid`](@ref) is the design taken exactly. A downstream kind subtypes this
and implements the protocol.
"""
abstract type TargetCalibration end

"""
    calibration_jacobian(c::TargetCalibration, x) -> SMatrix

`∂c(x)/∂θ` at the design point `x`, one column per entry of
[`parameter_vector`](@ref)`(c)`. Plain numbers; for a physical calibration
the point and the length parameters are in metres.
"""
function calibration_jacobian end

"""
    regularizer(c | target) -> SVector

The prior's residual rows `r(θ)` at the current parameters: `‖r‖²` is the
penalty a calibrating estimator adds (a thin-plate spline's bending energy, a
shrinkage toward the design). Empty for [`Rigid`](@ref).

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
    Rigid() <: TargetCalibration

The design, exactly: the identity, no parameters, no regularizer, no lengths
(so it sits with plain and physical designs alike).
"""
struct Rigid <: TargetCalibration end
(::Rigid)(p::StaticVector) = p

parameter_vector(::Rigid) = SVector{0, Float64}()
parameter_names(::Rigid) = ()
with_parameters(c::Rigid, θ::AbstractVector) =
    (isempty(θ) || throw(ArgumentError("Rigid takes no parameters; got $(length(θ))")); c)
calibration_jacobian(::Rigid, x) = zero(SMatrix{2, 0, Float64})
regularizer(::Rigid) = SVector{0, Float64}()

# --- lengths ---------------------------------------------------------------------
#
# `_has_lengths(c)`: whether a kind carries lengths at all (Rigid does not);
# `has_units(c)`: whether they are physical. `_length_sample(c)`: one of them in
# its own form, to name the mismatch in a `DimensionError`. `_length_mask(c)`:
# which entries of θ are lengths (they scale with a unit change; the rest are
# dimensionless).
# A kind another package defines states its lengths through the public
# protocol only (`has_units`, `ustrip`, `with_unit`); the chain's length-form
# check (one form across members) covers the kinds shipped here.
_has_lengths(::TargetCalibration) = false   # Rigid, and a downstream kind
has_units(::Rigid) = false
_length_mask(::Rigid) = SVector{0, Bool}()
FlexUnits.ustrip(::Units, c::Rigid) = c
with_unit(c::Rigid, ::Units) = c

# A calibration's form moves with its unit: an uncertain one's covariance
# scales with it over its length entries (`_scale_length_form`, euclidean_map.jl).
_ustrip_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? ustrip(u, c) : c
_attach_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? with_unit(c, u) : c
function _ustrip_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(ustrip(u, g), _scale_length_form(getfield(c, :attributes), _length_mask(g), _metre_factor(u), *))
end
function _attach_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(with_unit(g, u), _scale_length_form(getfield(c, :attributes), _length_mask(g), _metre_factor(u), /))
end

# A point in plain numbers (a physical one in metres), and a plain result put
# back in the form of the point it came from.
@inline _plain_point(p::StaticVector{N, <:Quantity}) where {N} = SVector{N}(map(q -> _strip_length(m, q), Tuple(p)))
@inline _plain_point(p::StaticVector) = SVector(Tuple(p))
@inline _restore_point(x::StaticVector, ::StaticVector{N, <:Quantity}) where {N} = SVector(map(v -> _attach_length(v, m), Tuple(x)))
@inline _restore_point(x::StaticVector, ::StaticVector) = x

# A target calibration applied to a design point: the kind computes on plain
# numbers (`_apply`), the point keeps its form. A calibration with lengths and
# a point of the other form is FlexUnits' `DimensionError`.
function (c::TargetCalibration)(p::StaticVector)
    _check_point_form(c, p)
    return _restore_point(_apply(c, _plain_point(p)), p)
end
@inline function _check_point_form(c, p)
    (_has_lengths(c) && has_units(c) != has_units(p)) || return nothing
    throw(DimensionError(p, _length_sample(c)))
end
_apply(::Rigid, q) = q
# ∂c(q)/∂q, the chain rule's factor; `Rigid` moves nothing.
_point_jacobian(::Rigid, q) = SMatrix{2, 2, Float64}(LinearAlgebra.I)

# Lengths read at a constructor: all plain or all lengths (stored as plain
# metres), and which.
function _lengths(xs::AbstractVector)
    isempty(xs) && return (Float64[], false)
    hs = map(has_units, xs)
    allequal(hs) || throw(DimensionError(xs[findfirst(hs)], xs[findfirst(!, hs)]))
    return first(hs) ? ([_strip_length(m, q) for q in xs], true) : (collect(xs), false)
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
@inline _to_metres(x::Real, u::Units) = ustrip(m, x * u)
@inline _from_metres(x::Real, u::Units) = ustrip(u, x * m)
@noinline _refuse_strip(c) = throw(ArgumentError("ustrip: the $(nameof(typeof(c))) is plain numbers already; `with_unit` attaches a unit"))
@noinline _refuse_attach(c) = throw(ArgumentError("with_unit: the $(nameof(typeof(c))) already has units"))
@noinline _refuse_length(c, θ) = throw(ArgumentError("$(nameof(typeof(c))) takes $(length(parameter_vector(c))) parameters; got $(length(θ))"))

# ------------------------------------------------------------------------------
# Affine — an in-plane print error
# ------------------------------------------------------------------------------

"""
    Affine(A, b) <: TargetCalibration

An in-plane affine print error, `p ↦ A·p + b`: `A` a dimensionless 2 × 2
matrix, `b` a translation in the design's length form (plain, or lengths).
Parameters `(:a11, :a12, :a21, :a22, :bx, :by)` (`A` row by row, then `b`),
unbounded; [`calibration_jacobian`](@ref) is 2 × 6; no regularizer.

With the pose estimated jointly, four of its six directions are the pose's
(gauges): the translation `b`, the in-plane rotation and the uniform scale.
The default [`TargetManifold`](@ref) holds them fixed and leaves the two that
images can see, the anisotropic stretch and the shear — see
[`gauge_names`](@ref). With a known pose, free them by name.
"""
struct Affine{T} <: TargetCalibration
    A::SMatrix{2, 2, T, 4}
    b::SVector{2, T}
    physical::Bool
end
function Affine(A::AbstractMatrix, b::AbstractVector)
    (size(A) == (2, 2) && length(b) == 2) ||
        throw(DimensionMismatch("Affine: A is 2 × 2 and b a 2-vector; got $(size(A)) and $(length(b))"))
    has_units(A) && throw(ArgumentError("Affine: A is dimensionless; the lengths are b's"))
    bv, physical = _lengths(collect(b))
    T = float(promote_type(eltype(A), eltype(bv)))
    return Affine{T}(SMatrix{2, 2, T}(A), SVector{2, T}(bv), physical)
end
_apply(c::Affine, q) = c.A * q + c.b
_point_jacobian(c::Affine, q) = c.A

parameter_vector(c::Affine) = SVector(c.A[1, 1], c.A[1, 2], c.A[2, 1], c.A[2, 2], c.b[1], c.b[2])
parameter_names(::Affine) = (:a11, :a12, :a21, :a22, :bx, :by)
function with_parameters(c::Affine, θ::AbstractVector)
    length(θ) == 6 || _refuse_length(c, θ)
    T = float(eltype(θ))
    return Affine{T}(SMatrix{2, 2, T}(θ[1], θ[3], θ[2], θ[4]), SVector{2, T}(θ[5], θ[6]), c.physical)
end
function calibration_jacobian(::Affine, x)
    q = _plain_point(x)
    o, z = one(eltype(q)), zero(eltype(q))
    # rows ∂(A q + b)₁ = (q₁, q₂, 0, 0, 1, 0), ∂(A q + b)₂ = (0, 0, q₁, q₂, 0, 1); column-major
    return SMatrix{2, 6}(q[1], z, q[2], z, z, q[1], z, q[2], o, z, z, o)
end
regularizer(::Affine) = SVector{0, Float64}()

_has_lengths(::Affine) = true
has_units(c::Affine) = c.physical
_length_sample(c::Affine) = c.physical ? _attach_length(c.b[1], m) : c.b[1]
_length_mask(::Affine) = SVector(false, false, false, false, true, true)
FlexUnits.ustrip(u::Units, c::Affine{T}) where {T} =
    c.physical ? Affine{T}(c.A, map(x -> _from_metres(x, u), c.b), false) : _refuse_strip(c)
with_unit(c::Affine{T}, u::Units) where {T} =
    c.physical ? _refuse_attach(c) : Affine{T}(c.A, map(x -> _to_metres(x, u), c.b), true)
Base.:(==)(a::Affine, b::Affine) = a.A == b.A && a.b == b.b && a.physical == b.physical
Base.hash(c::Affine, h::UInt) = hash(c.physical, hash(c.b, hash(c.A, hash(:Affine, h))))

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
    ThinPlateSpline(nodes, fx, fy) <: TargetCalibration
    ThinPlateSpline(nodes)                       zero displacement

An in-plane residual, `p ↦ p + (f_x(p), f_y(p))`: two thin-plate splines
(numerics/thin_plate_spline.jl) over the same `K` fixed `nodes` in the
target's plane, through the nodal displacements `fx` and `fy`. Nodes and
displacements are in the design's length form (plain, or lengths).
Parameters `(:fx1, …, :fxK, :fy1, …, :fyK)`, unbounded;
[`calibration_jacobian`](@ref) is 2 × 2K (the splines' `nodal_weights`);
[`regularizer`](@ref) the two splines' bending-energy rows (`2K` rows; a
quadratic block, see its docstring).

It is a RESIDUAL: its affine part (each spline's `1`, `u`, `v` over the nodes)
is fixed by the default [`TargetManifold`](@ref) — `gauge_names` `(:fx_1,
:fx_u, :fx_v, :fy_1, :fy_u, :fy_v)` — because four of those six directions
are the pose's and all six duplicate an `Affine`. A standalone
`ThinPlateSpline` therefore carries no affine part: chain it after an
[`Affine`](@ref) for a print stretch and shear, or free `:fx_u`, … by name.
"""
struct ThinPlateSpline{K, T} <: TargetCalibration
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
function calibration_jacobian(c::ThinPlateSpline{K}, x) where {K}
    n = nodal_weights(c.system, _plain_point(x))
    z = zero(n)
    return hcat(_two_rows(n, z), _two_rows(z, n))
end
regularizer(c::ThinPlateSpline) = vcat(_bending_rows(c.bending, c.fx), _bending_rows(c.bending, c.fy))

_has_lengths(::ThinPlateSpline) = true
has_units(c::ThinPlateSpline) = c.physical
_length_sample(c::ThinPlateSpline) = first(first(_node_lengths(c.system, c.physical)))
_length_mask(::ThinPlateSpline{K}) where {K} = SVector{2K}(ntuple(_ -> true, Val(2K)))
function FlexUnits.ustrip(u::Units, c::ThinPlateSpline)
    c.physical || _refuse_strip(c)
    return ThinPlateSpline([map(x -> _from_metres(x, u), p) for p in c.system.nodes], map(x -> _from_metres(x, u), c.fx), map(x -> _from_metres(x, u), c.fy))
end
function with_unit(c::ThinPlateSpline, u::Units)
    c.physical && _refuse_attach(c)
    to(x) = _attach_length(x, u)
    return ThinPlateSpline([map(to, p) for p in c.system.nodes], map(to, c.fx), map(to, c.fy))
end
Base.:(==)(a::ThinPlateSpline, b::ThinPlateSpline) =
    a.system.nodes == b.system.nodes && a.fx == b.fx && a.fy == b.fy && a.physical == b.physical
Base.hash(c::ThinPlateSpline, h::UInt) = hash(c.physical, hash(c.fy, hash(c.fx, hash(c.system.nodes, hash(:ThinPlateSpline, h)))))
Base.show(io::IO, c::ThinPlateSpline{K}) where {K} =
    print(io, "ThinPlateSpline over ", K, " nodes", c.physical ? " (lengths in metres)" : "")

"""
    HeightField(nodes, heights) <: TargetCalibration
    HeightField(nodes)                           flat

An out-of-plane deformation, `p ↦ (p, h(p))`: one thin-plate spline
(numerics/thin_plate_spline.jl) over `K` fixed `nodes` in the target's plane
through the nodal `heights`, both in the design's length form (plain, or
lengths). The height at `p` is `tps_evaluate` of the spline, nothing else.
Parameters `(:h1, …, :hK)`, unbounded; [`calibration_jacobian`](@ref) is
3 × K (zero in the plane, the spline's `nodal_weights` in z);
[`regularizer`](@ref) the bending-energy rows (`K` rows).

A target with a height field is not planar ([`is_planar`](@ref)): its
[`points`](@ref) are 3-vectors. In a [`CalibrationChain`](@ref) it is the last
member. Its piston and two tilts are the pose's (gauges): the default
[`TargetManifold`](@ref) holds them fixed — see [`gauge_names`](@ref).
"""
struct HeightField{K, T} <: TargetCalibration
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
function calibration_jacobian(c::HeightField{K}, x) where {K}
    n = nodal_weights(c.system, _plain_point(x))
    return SMatrix{3, K}(ntuple(k -> mod1(k, 3) == 3 ? n[(k + 2) ÷ 3] : 0.0, Val(3K)))
end
regularizer(c::HeightField) = _bending_rows(c.bending, c.h)

_has_lengths(::HeightField) = true
has_units(c::HeightField) = c.physical
_length_sample(c::HeightField) = first(first(_node_lengths(c.system, c.physical)))
_length_mask(::HeightField{K}) where {K} = SVector{K}(ntuple(_ -> true, Val(K)))
function FlexUnits.ustrip(u::Units, c::HeightField)
    c.physical || _refuse_strip(c)
    return HeightField([map(x -> _from_metres(x, u), p) for p in c.system.nodes], map(x -> _from_metres(x, u), c.h))
end
function with_unit(c::HeightField, u::Units)
    c.physical && _refuse_attach(c)
    to(x) = _attach_length(x, u)
    return HeightField([map(to, p) for p in c.system.nodes], map(to, c.h))
end
Base.:(==)(a::HeightField, b::HeightField) = a.system.nodes == b.system.nodes && a.h == b.h && a.physical == b.physical
Base.hash(c::HeightField, h::UInt) = hash(c.physical, hash(c.h, hash(c.system.nodes, hash(:HeightField, h))))
Base.show(io::IO, c::HeightField{K}) where {K} =
    print(io, "HeightField over ", K, " nodes", c.physical ? " (lengths in metres)" : "")

# ------------------------------------------------------------------------------
# Names as a function of the type (what a chain's generated names read)
# ------------------------------------------------------------------------------
#
# `nothing` for a type without a static table (a downstream kind): a chain
# holding one builds its names at run time.
_static_parameter_names(::Type) = nothing
_static_parameter_names(::Type{Rigid}) = ()
_static_parameter_names(::Type{<:Affine}) = (:a11, :a12, :a21, :a22, :bx, :by)
_static_parameter_names(::Type{<:ThinPlateSpline{K}}) where {K} =
    (ntuple(i -> Symbol(:fx, i), K)..., ntuple(i -> Symbol(:fy, i), K)...)
_static_parameter_names(::Type{<:HeightField{K}}) where {K} = ntuple(i -> Symbol(:h, i), K)
_static_gauge_names(::Type) = nothing
_static_gauge_names(::Type{Rigid}) = ()
_static_gauge_names(::Type{<:Affine}) = (:translation_x, :translation_y, :rotation, :scale)
_static_gauge_names(::Type{<:ThinPlateSpline}) = (:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v)
_static_gauge_names(::Type{<:HeightField}) = (:piston, :tilt_x, :tilt_y)
# a kind's tag, from the one registry (TARGET_CALIBRATION_TAGS, below the kinds)
_static_tag(::Type{T}) where {T} =
    any(K -> T <: K, values(TARGET_CALIBRATION_TAGS)) ? tag_of(TARGET_CALIBRATION_TAGS, T) : nothing
# A literal tuple per type, so a NamedTuple over the names is inferred.
@generated parameter_names(c::Union{ThinPlateSpline, HeightField}) = QuoteNode(_static_parameter_names(c))
function _static_chain_names(f, ::Type{C}) where {C <: Tuple}
    out = Symbol[]
    for M in fieldtypes(C)
        names, tag = f(M), _static_tag(M)
        (names === nothing || tag === nothing) && return nothing
        append!(out, (Symbol(tag, :_, n) for n in names))
    end
    return Tuple(out)
end

# ------------------------------------------------------------------------------
# CalibrationChain — calibrations applied in order
# ------------------------------------------------------------------------------

"""
    CalibrationChain(c₁, c₂, …) <: TargetCalibration

Calibrations applied in order, `cₙ(…c₁(p))` — an affine print error, then an
in-plane residual, then a height field, say: the target analogue of the
lens [`Chain`](@ref). They are two types because a type has one supertype:
the lens `Chain` is a distortion term (`AbstractDistortionTerm`), this a
`TargetCalibration`.

At least two members, none of them a chain, of distinct kinds, a
[`HeightField`](@ref) only last (the in-plane members act on plane points;
an `ArgumentError` otherwise), and the members with lengths all of one form
(`DimensionError`). Parameters and their names are the members', in order,
each name prefixed by its member's tag (`:affine_a11`, …, `:height_field_h1`);
[`calibration_jacobian`](@ref) composes the members' by the chain rule (3 rows
once a height field is present); [`regularizer`](@ref) stacks theirs.

A member acts on its predecessor's output, so its nodes (a thin-plate kind's)
are in that member's input coordinates: a `HeightField` after an `Affine` has
its nodes in the calibrated plane, not in the design.
"""
struct CalibrationChain{C <: Tuple} <: TargetCalibration
    members::C
    function CalibrationChain(members::C) where {C <: Tuple{Vararg{TargetCalibration}}}
        length(members) >= 2 || throw(ArgumentError("CalibrationChain: at least two members (one is the member itself)"))
        for (i, c) in enumerate(members)
            c isa CalibrationChain && throw(ArgumentError("CalibrationChain: member $i is a chain; list its members instead"))
            c isa HeightField && i < length(members) &&
                throw(ArgumentError("CalibrationChain: a HeightField is the last member (the others act on plane points); it is member $i of $(length(members))"))
        end
        tags = map(_calibration_tag, members)
        allunique(tags) || throw(ArgumentError("CalibrationChain: one member of each kind; got $(join(tags, ", "))"))
        withlengths = [c for c in members if _has_lengths(c)]
        isempty(withlengths) || allequal(map(has_units, withlengths)) ||
            throw(DimensionError(_length_sample(withlengths[findfirst(has_units, withlengths)]), _length_sample(withlengths[findfirst(!has_units, withlengths)])))
        return new{C}(members)
    end
end
CalibrationChain(members::TargetCalibration...) = CalibrationChain(members)

"""
    TARGET_CALIBRATION_TAGS

The [`TagRegistry`](@ref) of the target calibration kinds — the one map from a
kind's tag to its type, read by a chain's parameter names (`:affine_a11`) and
by VisualGeometryToolkitIO's target record (`"type": "affine"`): `"rigid"`,
`"affine"`, `"thin_plate_spline"`, `"height_field"`, `"calibration_chain"`,
`snake_case` of the type's name.
"""
const TARGET_CALIBRATION_TAGS = TagRegistry(:target_calibration, "",
    "rigid" => Rigid, "affine" => Affine, "thin_plate_spline" => ThinPlateSpline,
    "height_field" => HeightField, "calibration_chain" => CalibrationChain)
# a calibration's tag: the registry's reverse lookup for a shipped kind; for a
# kind another package defines (which the registry does not hold), the tag
# rule's spelling of its type name, so a chain holding one still constructs
_calibration_tag(c::TargetCalibration) = _calibration_tag(typeof(c))

# A Rigid calibration has no parameters, so nothing for a covariance to be
# over: `Uncertain(Rigid(), Σ)` is refused at construction (the target record
# writes no covariance for it either).
(::Type{Attributed})(::Rigid, ::Union{CovMat, PrecMat}) = throw(
    ArgumentError("a Rigid target calibration has no parameters, so no covariance: Uncertain(Rigid(), Σ) is refused — use Rigid()")
)
_calibration_tag(::Type{T}) where {T <: TargetCalibration} =
    any(K -> T <: K, values(TARGET_CALIBRATION_TAGS)) ? tag_of(TARGET_CALIBRATION_TAGS, T) : snake_case(nameof(T))

_apply(c::CalibrationChain, q) = _apply_members(c.members, q)
_apply_members(::Tuple{}, q) = q
_apply_members(cs::Tuple, q) = _apply_members(Base.tail(cs), _apply(first(cs), q))

parameter_vector(c::CalibrationChain) = vcat(map(parameter_vector, c.members)...)
# The members' names, each prefixed by its member's tag. For the kinds shipped
# here the names are a function of the TYPE, so a chain's are a literal tuple
# (generated from its member types) and `parameters(t)` infers its NamedTuple;
# a chain with a downstream member builds them at run time instead.
@generated function parameter_names(c::CalibrationChain{C}) where {C}
    names = _static_chain_names(_static_parameter_names, C)
    return names === nothing ? :(_prefixed_names(parameter_names, c.members)) : QuoteNode(names)
end
_prefixed_names(f, ::Tuple{}) = ()
_prefixed_names(f, cs::Tuple) =
    (_prefixed(_calibration_tag(first(cs)), f(first(cs)))..., _prefixed_names(f, Base.tail(cs))...)
_prefixed(tag, names::Tuple) = map(n -> Symbol(tag, :_, n), names)
function with_parameters(c::CalibrationChain, θ::AbstractVector)
    length(θ) == sum(x -> length(parameter_vector(x)), c.members) || _refuse_length(c, θ)
    return CalibrationChain(_with_members(c.members, θ, 0))
end
_with_members(::Tuple{}, θ, o) = ()
function _with_members(cs::Tuple, θ, o)
    x = first(cs)
    n = length(parameter_vector(x))
    return (with_parameters(x, [θ[o + i] for i in 1:n]), _with_members(Base.tail(cs), θ, o + n)...)
end
calibration_jacobian(c::CalibrationChain, x) = first(_chain_jacobians(c.members, _plain_point(x)))
# (∂out/∂θ of the members `cs` applied to q, ∂out/∂q): each member's own
# Jacobian carried through the point Jacobians of the members after it.
_chain_jacobians(::Tuple{}, q::StaticVector{N}) where {N} = (zero(SMatrix{N, 0, Float64}), SMatrix{N, N, Float64}(LinearAlgebra.I))
function _chain_jacobians(cs::Tuple, q)
    x = first(cs)
    J, D = _chain_jacobians(Base.tail(cs), _apply(x, q))
    return hcat(D * calibration_jacobian(x, q), J), D * _point_jacobian(x, q)
end
_point_jacobian(c::CalibrationChain, q) = last(_chain_jacobians(c.members, q))
regularizer(c::CalibrationChain) = vcat(map(regularizer, c.members)...)

_has_lengths(c::CalibrationChain) = any(_has_lengths, c.members)
has_units(c::CalibrationChain) = any(x -> _has_lengths(x) && has_units(x), c.members)
function _length_sample(c::CalibrationChain)
    for x in c.members
        _has_lengths(x) && return _length_sample(x)
    end
    throw(ArgumentError("a $(nameof(typeof(c))) without lengths has no length to name"))
end
_length_mask(c::CalibrationChain) = vcat(map(_length_mask, c.members)...)
function FlexUnits.ustrip(u::Units, c::CalibrationChain)
    has_units(c) || _refuse_strip(c)
    return CalibrationChain(map(x -> _ustrip_calibration(u, x), c.members))
end
function with_unit(c::CalibrationChain, u::Units)
    (_has_lengths(c) && has_units(c)) && _refuse_attach(c)
    return CalibrationChain(map(x -> _attach_calibration(u, x), c.members))
end
Base.:(==)(a::CalibrationChain, b::CalibrationChain) = a.members == b.members
Base.hash(c::CalibrationChain, h::UInt) = hash(c.members, hash(:CalibrationChain, h))

# ------------------------------------------------------------------------------
# Planarity — in the type
# ------------------------------------------------------------------------------

"""
    is_planar(::Type{C}) -> Bool
    is_planar(c::TargetCalibration) / is_planar(t::PlanarTarget) -> Bool

Whether a calibration keeps the design in its plane — a function of the TYPE,
so it folds at compile time: `true` for [`Rigid`](@ref), [`Affine`](@ref),
[`ThinPlateSpline`](@ref) and a [`CalibrationChain`](@ref) of them, `false`
for a [`HeightField`](@ref) and a chain ending in one. A planar target's
[`points`](@ref) are `SVector{2}`s (on `z = 0` of its frame), a non-planar
one's `SVector{3}`s. An [`Uncertain`](@ref) calibration is planar when its
kind is. A downstream kind declares its own method; there is no default.
"""
function is_planar end
is_planar(::Type{Rigid}) = true
is_planar(::Type{<:Affine}) = true
is_planar(::Type{<:ThinPlateSpline}) = true
is_planar(::Type{<:HeightField}) = false
is_planar(::Type{CalibrationChain{C}}) where {C} = _all_planar(C)
@inline _all_planar(::Type{Tuple{}}) = true
@inline _all_planar(::Type{C}) where {C <: Tuple} = is_planar(Base.tuple_type_head(C)) && _all_planar(Base.tuple_type_tail(C))
is_planar(::Type{<:Attributed{C, <:Union{CovMat, PrecMat}}}) where {C <: TargetCalibration} = is_planar(C)
is_planar(c::TargetCalibration) = is_planar(typeof(c))
is_planar(u::Uncertain{<:TargetCalibration}) = is_planar(typeof(geom(u)))

"""
    PlanarCalibration

The planar calibration kinds this package ships — [`Rigid`](@ref),
[`Affine`](@ref), [`ThinPlateSpline`](@ref) and a [`CalibrationChain`](@ref)
of them — as one type, for a method that assumes the design stays on `z = 0`:
`f(t::PlanarTarget{<:PlanarCalibration})` refuses a target with a
[`HeightField`](@ref) by dispatch (a `MethodError`). A method that should
also take a downstream planar kind checks [`is_planar`](@ref) instead.
"""
const PlanarCalibration = Union{
    Rigid, Affine, ThinPlateSpline,
    CalibrationChain{<:Tuple{Vararg{Union{Rigid, Affine, ThinPlateSpline}}}},
}

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

- [`Affine`](@ref): `:translation_x`, `:translation_y` (the axes `bx`, `by`),
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
  similarity), and all six duplicate an `Affine` it is chained after, so the
  spline is a pure residual; free `:fx_u`, … by name to let a standalone one
  carry a stretch.
- [`HeightField`](@ref): `:piston`, `:tilt_x`, `:tilt_y` — heights that are an
  affine function of the node position, `hᵢ = α + β uᵢ + γ vᵢ`, the columns
  `1`, `u` and `v` over the nodes. The spline through affine heights is that
  plane, which the pose's translation and two rotations reproduce; fixing them
  leaves the heights orthogonal to every plane (`Σ hᵢ = Σ uᵢhᵢ = Σ vᵢhᵢ = 0`
  for the steps).
- [`CalibrationChain`](@ref): its members', each name prefixed by its
  member's tag (`:affine_scale`, `:height_field_piston`, …).
- [`Rigid`](@ref): none. A downstream kind has none unless it adds methods
  to both functions.
"""
function gauge_names end
gauge_names(::TargetCalibration) = ()
gauge_names(c::Union{Affine, ThinPlateSpline, HeightField}) = _static_gauge_names(typeof(c))
@generated function gauge_names(c::CalibrationChain{C}) where {C}
    names = _static_chain_names(_static_gauge_names, C)
    return names === nothing ? :(_prefixed_names(gauge_names, c.members)) : QuoteNode(names)
end
gauge_names(u::Uncertain{<:TargetCalibration}) = gauge_names(geom(u))

"""
    gauge_directions(c) -> Matrix{Float64}

The gauges of [`gauge_names`](@ref) as the columns of an `n × g` matrix in the
coordinates of `parameter_vector(c)`, evaluated at `c`.
"""
function gauge_directions end
gauge_directions(c::TargetCalibration) = zeros(Float64, length(parameter_vector(c)), 0)
function gauge_directions(c::Affine)
    a11, a12, a21, a22 = Float64.(Tuple(parameter_vector(c))[1:4])
    s = hypot(a11, a12, a21, a22)
    s > 0 || throw(ArgumentError("gauge_directions: an Affine with A = 0 has no rotation or scale direction"))
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
function gauge_directions(c::CalibrationChain)
    ns = [length(parameter_vector(x)) for x in c.members]
    blocks = [gauge_directions(x) for x in c.members]
    D = zeros(Float64, sum(ns), sum(size.(blocks, 2)))
    r = 0; k = 0
    for (n, b) in zip(ns, blocks)
        D[(r + 1):(r + n), (k + 1):(k + size(b, 2))] = b
        r += n; k += size(b, 2)
    end
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
calibration_jacobian(u::Uncertain{<:TargetCalibration}, x) = calibration_jacobian(geom(u), x)

"""
    with_parameters(u::Uncertain{<:TargetCalibration}, θ) -> TargetCalibration

The BARE kind rebuilt from `θ`: an estimate's covariance describes the
parameters it was estimated at and does not survive a change of them, so the
result is not `Uncertain`. (A `PlanarTarget`'s `with_parameters` does the same
to its calibration.)
"""
with_parameters(u::Uncertain{<:TargetCalibration}, θ::AbstractVector) = with_parameters(geom(u), θ)
