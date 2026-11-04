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
# The kinds are `Rigid`; the four factors of an affine print error, Core's
# `EuclideanMap{2}` and `Scale`, `Aspect`, `Skew` (the Iwasawa factors,
# below); the two thin-plate kinds; and their compositions with CT's `∘`: the
# factors in their canonical order (`EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`,
# any of them left out), and a height field after one in-plane kind or chain
# (`HeightField ∘ Aspect ∘ Skew`, `HeightField ∘ ThinPlateSpline`). A
# composition's parameters, names, bounds, Jacobian columns, gauges and
# regularizer rows are in CT's order — the OUTER map's first, as
# `transform_deriv_params(t1 ∘ t2, x) = hcat(p1, m1 * p2)` stacks them.
#
# LENGTHS. A calibration's lengths are in the form of the design points it
# moves — plain numbers, or lengths. A rigid member carries them as an
# `EuclideanMap` does (its translation plain or lengths in metres); the
# thin-plate kinds store them as plain METRES with a `physical` flag (their
# spline system, coefficients and bending factor are plain Float64 arrays
# shared bit for bit with RadiometricCamera, and their rebuilds run on
# ForwardDiff Duals), and `ustrip(u, c)` / `with_unit(c, u)` move between the
# forms as they do for points. `Scale`, `Aspect` and `Skew` are dimensionless.
# Applied to a physical point a calibration computes in plain metres (or
# carries the lengths through a dimensionless map) and returns lengths; its
# parameters and Jacobians are plain numbers, the lengths among them in metres.

"""
    regularizer(c | target) -> SVector

The prior's residual rows `r(θ)` at the current parameters: `‖r‖²` is the
penalty a calibrating estimator adds (a thin-plate spline's bending energy, a
shrinkage toward the design). Empty for [`Rigid`](@ref) and the affine factors
([`Aspect`](@ref), [`Skew`](@ref), [`Scale`](@ref), an `EuclideanMap`).

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
@inline _plain_length(q::Quantity) = _strip_length(m, q)
@inline _plain_length(x::Real) = x
# θ's first K entries as a static vector of eltype T (θ any vector)
_wrap(v::AbstractVector, ::Val{K}, o, ::Type{T}) where {K, T} = SVector{K, T}(ntuple(i -> v[o + i], Val(K)))

# ------------------------------------------------------------------------------
# The affine print error — the Iwasawa factors
# ------------------------------------------------------------------------------
#
# An affine map of the design, p ↦ A·p + b with det A > 0, factors uniquely
# (the Iwasawa, or KAN, decomposition of A) as A = s·R·D·U: R a rotation,
# s > 0 a scale, D a det-1 diagonal (the ASPECT) and U a unit upper-triangular
# matrix (the SKEW). Each factor is a calibration kind, composed with CT's `∘`
# in the CANONICAL ORDER, the outer map first:
#
#   EuclideanMap ∘ Scale ∘ Aspect ∘ Skew       p ↦ R·(e^k·D·U·p) + t
#
# Any of the four may be left out, the rest keep that order. NO factor has a
# gauge. With the target's pose estimated jointly, the rotation and the
# translation are the pose's, and so is the scale (with the pose's
# translation): a target and the same target turned, shifted or scaled give
# the same images. The print error a camera calibration estimates is
# therefore `Aspect ∘ Skew`; a measured length (a ruler) makes the scale
# observable, `Scale ∘ Aspect ∘ Skew`; and a known pose (a fixture, motion
# capture) the whole affine, `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`. A
# composition holds only the directions its data observes, so there is
# nothing to hold fixed. The count: the affine maps of R^N have N² + N
# parameters, the pose N(N−1)/2 + N and the scale 1, which leaves
# (N−1) + N(N−1)/2 — Aspect's N−1 and Skew's N(N−1)/2.
#
# `Aspect`, `Skew` and `Scale` are N-generic; a planar target takes N = 2
# (one parameter each). A 3-D target would take N = 3 (2 + 3 parameters), but
# `PointTarget` has no calibration today (targets.jl).

"""
    Aspect(a::StaticVector{M}) <: CoordinateTransformations.Transformation      N = M + 1
    Aspect(a::Real)                                                             N = 2
    Aspect{N}()                                                                 the identity

The aspect of an affine print error, `x ↦ D·x` with `D = diag(e^{d₁}, …, e^{d_N})`
and `Σ dᵢ = 0`: a det-1 diagonal, which stretches one axis against the others
and changes no area. Its parameters are the `N − 1` free log-stretches
`aᵢ = dᵢ` (`i < N`), named `(:a1, …, :a(N-1))`, and `d_N = −Σ aᵢ`: for `N = 2`,
`D = diag(e^{a}, e^{−a})` — `a = 0.001` stretches x by 0.1 % and shrinks y by
as much. `det D = 1` exactly in exact arithmetic (`Σ dᵢ = 0` holds exactly
in floating point too; the product of the `e^{dᵢ}` is 1 to rounding).

Dimensionless (it carries lengths through, so it sits with plain and physical
designs alike): no length parameters, no gauges, no regularizer, unbounded.
`transform_deriv(c, x) = D`; `transform_deriv_params(c, x)` is `N × (N − 1)`,
column `j` is `e_j D_jj x_j − e_N D_NN x_N`. `inv(c)` negates `a`. A kind of
the Iwasawa factorization of an affine print error (see [`Skew`](@ref), the
factor it follows in `Aspect ∘ Skew`).
"""
struct Aspect{N, T <: Real} <: CoordinateTransformations.Transformation
    log_diagonal::SVector{N, T}   # (d₁, …, d_N), Σ dᵢ = 0: d_N = −Σ of the rest
    function Aspect{N, T}(a::StaticVector{M, <:Real}) where {N, T <: Real, M}
        N >= 2 || throw(ArgumentError("an Aspect is of dimension N ≥ 2; got N = $N"))
        M == N - 1 || throw(DimensionMismatch("Aspect{$N} takes $(N - 1) parameters; got $M"))
        b = SVector{M, T}(a)
        return new{N, T}(SVector{N, T}(b..., -sum(b)))
    end
end
Aspect(a::StaticVector{M, T}) where {M, T <: Real} = Aspect{M + 1, float(T)}(a)
Aspect(a::Real) = Aspect(SVector(a))
Aspect{N}() where {N} = Aspect{N, Float64}(zero(SVector{N - 1, Float64}))

(c::Aspect{N})(x::StaticVector{N}) where {N} = SVector{N}(ntuple(i -> exp(c.log_diagonal[i]) * x[i], Val(N)))
parameter_vector(c::Aspect{N}) where {N} = SVector{N - 1}(ntuple(i -> c.log_diagonal[i], Val(N - 1)))
function with_parameters(c::Aspect{N}, θ::AbstractVector) where {N}
    length(θ) == N - 1 || _refuse_length(c, θ)
    T = float(eltype(θ))
    return Aspect{N, T}(_wrap(θ, Val(N - 1), 0, T))
end
transform_deriv(c::Aspect{N}, x) where {N} =
    SMatrix{N, N}(ntuple(k -> (i = mod1(k, N); j = (k - 1) ÷ N + 1; i == j ? exp(c.log_diagonal[i]) : zero(eltype(c.log_diagonal))), Val(N * N)))
function transform_deriv_params(c::Aspect{N}, x) where {N}
    q = _plain_point(x)
    y = SVector{N}(ntuple(i -> exp(c.log_diagonal[i]) * q[i], Val(N)))
    z = zero(eltype(y))
    # ∂(D x)_i/∂a_j = δ_ij y_i − δ_iN y_N, column-major over (i, j)
    return SMatrix{N, N - 1}(ntuple(k -> (i = mod1(k, N); j = (k - 1) ÷ N + 1; (i == j ? y[i] : z) - (i == N ? y[N] : z)), Val(N * (N - 1))))
end
Base.inv(c::Aspect{N, T}) where {N, T} = Aspect{N, T}(-parameter_vector(c))
convert_eltype(::Type{T}, c::Aspect{N}) where {T <: AbstractFloat, N} = Aspect{N, T}(parameter_vector(c))
Base.show(io::IO, c::Aspect) = print(io, "Aspect(", Tuple(parameter_vector(c)), ")")

"""
    Skew(u::StaticVector{L}) <: CoordinateTransformations.Transformation      L = N(N − 1)/2
    Skew(u::Real)                                                             N = 2
    Skew{N}()                                                                 the identity

The skew of an affine print error, `x ↦ U·x` with `U` unit upper-triangular:
ones on the diagonal, zeros below it, and its `L = N(N − 1)/2` strictly-upper
entries the parameters, ROW BY ROW — `(U₁₂, U₁₃, …, U₁N, U₂₃, …, U₂N, …)`,
named `(:u12, :u13, …, :u23, …)`; a record persists them in that order. For
`N = 2`, `x ↦ (x₁ + u·x₂, x₂)`: a shear of x along y. `det U = 1` exactly.

Dimensionless: no length parameters, no gauges, no regularizer, unbounded.
`transform_deriv(c, x) = U`; `transform_deriv_params(c, x)` is `N × L`, the
column of `U_ij` being `x_j e_i`. `inv(c)` is the skew of `U⁻¹` (`−u` for
`N = 2`). The inner factor of the canonical print error `Aspect ∘ Skew`.
"""
struct Skew{N, T <: Real, L} <: CoordinateTransformations.Transformation
    entries::SVector{L, T}   # U's strictly-upper entries, row by row
    function Skew{N, T, L}(u::StaticVector{L, <:Real}) where {N, T <: Real, L}
        N >= 2 || throw(ArgumentError("a Skew is of dimension N ≥ 2; got N = $N"))
        L == N * (N - 1) ÷ 2 || throw(DimensionMismatch("Skew{$N} takes $(N * (N - 1) ÷ 2) parameters; got $L"))
        return new{N, T, L}(SVector{L, T}(u))
    end
end
function _skew_dimension(L::Integer)
    N = round(Int, (1 + sqrt(1 + 8L)) / 2)
    (L >= 1 && N * (N - 1) ÷ 2 == L) || throw(DimensionMismatch("a Skew takes N(N − 1)/2 parameters, N ≥ 2; got $L"))
    return N
end
Skew(u::StaticVector{L, T}) where {L, T <: Real} = Skew{_skew_dimension(L), float(T), L}(u)
Skew(u::Real) = Skew(SVector(u))
Skew{N}() where {N} = Skew{N, Float64, N * (N - 1) ÷ 2}(zero(SVector{N * (N - 1) ÷ 2, Float64}))

# The row-by-row position of U_ij (i < j) among the entries.
@inline _skew_index(N, i, j) = (i - 1) * N - (i - 1) * i ÷ 2 + (j - i)
# U as a static matrix.
function _unit_upper(c::Skew{N, T}) where {N, T}
    return SMatrix{N, N, T}(ntuple(k -> (i = mod1(k, N); j = (k - 1) ÷ N + 1; i == j ? one(T) : i < j ? c.entries[_skew_index(N, i, j)] : zero(T)), Val(N * N)))
end
# (i, j) of the k-th entry
@inline function _skew_pair(N, k)
    i = 1
    while k > N - i
        k -= N - i
        i += 1
    end
    return i, i + k
end

(c::Skew{N})(x::StaticVector{N}) where {N} = _unit_upper(c) * x
parameter_vector(c::Skew) = c.entries
function with_parameters(c::Skew{N, T, L}, θ::AbstractVector) where {N, T, L}
    length(θ) == L || _refuse_length(c, θ)
    S = float(eltype(θ))
    return Skew{N, S, L}(_wrap(θ, Val(L), 0, S))
end
transform_deriv(c::Skew, x) = _unit_upper(c)
function transform_deriv_params(c::Skew{N, T, L}, x) where {N, T, L}
    q = _plain_point(x)
    z = zero(eltype(q))
    # column k (the entry U_ij): x_j in row i
    return SMatrix{N, L}(ntuple(idx -> (r = mod1(idx, N); (i, j) = _skew_pair(N, (idx - 1) ÷ N + 1); r == i ? q[j] : z), Val(N * L)))
end
function Base.inv(c::Skew{N, T, L}) where {N, T, L}
    V = inv(_unit_upper(c))
    return Skew{N, T, L}(SVector{L, T}(ntuple(k -> ((i, j) = _skew_pair(N, k); V[i, j]), Val(L))))
end
convert_eltype(::Type{T}, c::Skew{N, S, L}) where {T <: AbstractFloat, N, S, L} = Skew{N, T, L}(c.entries)
Base.show(io::IO, c::Skew) = print(io, "Skew(", Tuple(c.entries), ")")

"""
    Scale(log_scale::Real = 0.0) <: CoordinateTransformations.Transformation

A uniform scale, `x ↦ e^{k}·x` with `k` the log-scale, any dimension: one
parameter, named `(:log_scale,)`; `transform_deriv(c, x) = e^{k} I`,
`transform_deriv_params(c, x) = e^{k} x` (one column). Dimensionless, no
gauges, no regularizer, unbounded.

The scale of an affine print error is the target's pose's when the pose is
estimated jointly (the pose's translation scales with it), so it belongs in a
calibration only when a LENGTH is measured — a ruler, a known pose:
`Scale ∘ Aspect ∘ Skew`, or `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`.
"""
struct Scale{T <: Real} <: CoordinateTransformations.Transformation
    log_scale::T
end
Scale(k::Integer) = Scale(float(k))
Scale() = Scale(0.0)

(c::Scale)(x::StaticVector) = exp(c.log_scale) * x
parameter_vector(c::Scale) = SVector(c.log_scale)
function with_parameters(c::Scale, θ::AbstractVector)
    length(θ) == 1 || _refuse_length(c, θ)
    return Scale(float(θ[1]))
end
transform_deriv(c::Scale, x::StaticVector{N}) where {N} = exp(c.log_scale) * SMatrix{N, N, typeof(float(c.log_scale))}(LinearAlgebra.I)
transform_deriv_params(c::Scale, x::StaticVector{N}) where {N} = SMatrix{N, 1}(exp(c.log_scale) * _plain_point(x))
Base.inv(c::Scale) = Scale(-c.log_scale)
convert_eltype(::Type{T}, c::Scale) where {T <: AbstractFloat} = Scale(T(c.log_scale))

# --- the rigid member: Core's EuclideanMap{2} ---------------------------------------
#
# The outer factor, `p ↦ R(ϑ)·p + t`, Core's own planar `EuclideanMap`: its
# parameters `(:angle, :tx, :ty)` — the angle ϑ (radians, counter-clockwise)
# and the translation, in metres for a physical map — the minimal coordinates
# of a planar pose, rotation first as in persisted-geometry.md's `[δω; δt]`
# (in the plane a change of the angle IS the left perturbation). Its
# `transform_deriv_params` is the 2 × 3 `[J R p  I₂]` (manifolds/rigid.jl), its
# `transform_deriv` R. It belongs in a calibration only with a KNOWN pose;
# with the pose estimated jointly it is the pose.
parameter_vector(E::EuclideanMap{2}) = (
    R = SMatrix{2, 2}(rotation(E)); t = translation(E);
    SVector(atan(R[2, 1], R[1, 1]), _plain_length(t[1]), _plain_length(t[2]))
)
parameter_names(::EuclideanMap{2}) = (:angle, :tx, :ty)
function with_parameters(E::EuclideanMap{2}, θ::AbstractVector)
    length(θ) == 3 || _refuse_length(E, θ)
    T = float(eltype(θ))
    t = SVector{2, T}(θ[2], θ[3])
    return EuclideanMap(RotMatrix{2, T}(θ[1]), has_units(E) ? map(x -> _attach_length(x, m), t) : t; check = false)
end
# a physical design point: the rotation column in metres (manifolds/rigid.jl's method takes plain points)
transform_deriv_params(E::EuclideanMap{2}, x::StaticVector{2, <:Quantity}) = transform_deriv_params(E, _plain_point(x))

# The four as one type: a planar affine factor (N = 2). The dimensionless
# three answer the read-only protocol in every dimension.
const _PlanarFactor = Union{EuclideanMap{2}, Scale, Aspect{2}, Skew{2}}
const _AnyFactor = Union{_PlanarFactor, Scale, Aspect, Skew}
regularizer(::_AnyFactor) = SVector{0, Float64}()
# their place in the canonical order, the outer map first
_factor_rank(::EuclideanMap) = 1
_factor_rank(::Scale) = 2
_factor_rank(::Aspect) = 3
_factor_rank(::Skew) = 4

# lengths: only the rigid member's translation
_has_lengths(::Union{Scale, Aspect, Skew}) = false
_has_lengths(::EuclideanMap{2}) = true
has_units(::Union{Scale, Aspect, Skew}) = false
_length_sample(E::EuclideanMap{2}) = translation(E)[1]
_length_sample(::Union{Scale, Aspect, Skew}) = nothing   # no length (a composition reads its next member's)
FlexUnits.ustrip(::Units, c::Union{Scale, Aspect, Skew}) = c
with_unit(c::Union{Scale, Aspect, Skew}, ::Units) = c

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
[`points`](@ref) are 3-vectors. It composes after an in-plane kind or chain,
`HeightField ∘ Aspect ∘ Skew` or `HeightField ∘ ThinPlateSpline`, its nodes
then in that kind's output (the calibrated plane, not the design). Its piston
and two tilts are the pose's (gauges): the default [`TargetManifold`](@ref)
holds them fixed — see [`gauge_names`](@ref).
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
# The compositions
# ------------------------------------------------------------------------------
#
# CoordinateTransformations' `∘`, its call and its chain-rule Jacobians
# (`transform_deriv`, `transform_deriv_params`: `hcat(p1, m1 * p2)`, the
# outer map's parameters first) are CT's. The rest of the protocol is here,
# over the composition's MEMBERS (its leaves, outer first, whatever the
# association `∘` built). A composition of up to five members has nesting
# depth at most four, which the type below spells out: a `ComposedTransformation`
# whose every member is a calibration kind — so every method on it is on a
# type this package owns a part of. Which compositions a TARGET accepts is a
# narrower rule, checked where it stores one (`_stored_calibration`, below).
const _CalibrationLeaf = Union{ThinPlateSpline, HeightField, _AnyFactor}   # (an N-dimensional factor composes too; a target takes N = 2)
const _CalibrationNode1 = Union{_CalibrationLeaf, ComposedTransformation{<:_CalibrationLeaf, <:_CalibrationLeaf}}
const _CalibrationNode2 = Union{_CalibrationNode1, ComposedTransformation{<:_CalibrationNode1, <:_CalibrationNode1}}
const _CalibrationNode3 = Union{_CalibrationNode2, ComposedTransformation{<:_CalibrationNode2, <:_CalibrationNode2}}
const _CalibrationComposition = ComposedTransformation{<:_CalibrationNode3, <:_CalibrationNode3}
# the planar chains: compositions of the affine factors alone
const _PlanarNode1 = Union{_PlanarFactor, ComposedTransformation{<:_PlanarFactor, <:_PlanarFactor}}
const _PlanarNode2 = Union{_PlanarNode1, ComposedTransformation{<:_PlanarNode1, <:_PlanarNode1}}
const _PlanarChain = Union{_PlanarNode2, ComposedTransformation{<:_PlanarNode2, <:_PlanarNode2}}

# The members, outer first; as values and as types.
_members(c::ComposedTransformation) = (_members(c.t1)..., _members(c.t2)...)
_members(c) = (c,)
_member_types(::Type{ComposedTransformation{A, B}}) where {A, B} = (_member_types(A)..., _member_types(B)...)
_member_types(::Type{T}) where {T} = (T,)
# The composition rebuilt member by member, its association kept.
_map_members(f, c::ComposedTransformation) = ComposedTransformation(_map_members(f, c.t1), _map_members(f, c.t2))
_map_members(f, c) = f(c)

"""
    TargetCalibration

What a target calibration estimates about a planar target: a
CoordinateTransformations `Transformation` `c(p)` of its design points to the
physical target — one of [`Rigid`](@ref), the affine factors (a planar
`EuclideanMap`, [`Scale`](@ref), [`Aspect`](@ref), [`Skew`](@ref)),
[`ThinPlateSpline`](@ref), [`HeightField`](@ref), or a composition of them
with CT's `∘`. The union of those, the one type every calibration-typed
method takes.

A [`PlanarTarget`](@ref) holds:
- one kind, or the affine factors composed in their canonical order,
  `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, any of them left out — the print
  error with a jointly estimated pose is `Aspect ∘ Skew`, with a measured
  length `Scale ∘ Aspect ∘ Skew`, with a known pose all four (see
  [`Aspect`](@ref));
- or a [`HeightField`](@ref) after one of those or after a
  [`ThinPlateSpline`](@ref): `HeightField ∘ Aspect ∘ Skew`,
  `HeightField ∘ ThinPlateSpline`.
Any other composition — a `Rigid` member, two in-plane kinds with a spline
among them, factors out of order or repeated, a member after a height field —
is refused with an `ArgumentError`. A target stores a composition
right-associated (`a ∘ (b ∘ c)`) whatever association `∘` built it in: the
members, their order and so the parameters are the same.

The protocol: [`parameter_vector`](@ref), [`parameter_names`](@ref),
[`parameter_bounds`](@ref), [`with_parameters`](@ref)`(c, θ)`,
CoordinateTransformations' `transform_deriv_params(c, x)` (`∂c(x)/∂θ`) and
`transform_deriv(c, x)` (`∂c(x)/∂x`), [`regularizer`](@ref)`(c)`,
[`gauge_names`](@ref), [`length_mask`](@ref). A composition's are stacked in
CT's order, the outer member's first; its names carry each member's tag,
`:height_field_h1`, …, `:aspect_a1`, `:skew_u12`. Jacobians are plain numbers;
for a physical calibration the point and the length parameters are in metres.
"""
const TargetCalibration = Union{Rigid, ThinPlateSpline, HeightField, _PlanarFactor, _CalibrationComposition}

parameter_vector(c::_CalibrationComposition) = vcat(parameter_vector(c.t1), parameter_vector(c.t2))
function with_parameters(c::_CalibrationComposition, θ::AbstractVector)
    n1, n2 = length(parameter_vector(c.t1)), length(parameter_vector(c.t2))
    length(θ) == n1 + n2 || _refuse_length(c, θ)
    return ComposedTransformation(with_parameters(c.t1, [θ[i] for i in 1:n1]), with_parameters(c.t2, [θ[n1 + i] for i in 1:n2]))
end
regularizer(c::_CalibrationComposition) = vcat(regularizer(c.t1), regularizer(c.t2))
_has_lengths(c::_CalibrationComposition) = any(_has_lengths, _members(c))
has_units(c::_CalibrationComposition) = any(l -> _has_lengths(l) && has_units(l), _members(c))
_length_sample(c::_CalibrationComposition) = (s = _length_sample(c.t1); s === nothing ? _length_sample(c.t2) : s)
length_mask(::Type{ComposedTransformation{A, B}}) where {A, B} = vcat(length_mask(A), length_mask(B))
FlexUnits.ustrip(u::Units, c::_CalibrationComposition) =
    has_units(c) ? _map_members(l -> _has_lengths(l) ? ustrip(u, l) : l, c) : _refuse_strip(c)
with_unit(c::_CalibrationComposition, u::Units) =
    has_units(c) ? _refuse_attach(c) : _map_members(l -> _has_lengths(l) ? with_unit(l, u) : l, c)
# CoordinateTransformations defines `==` on a composition, member by member,
# and no `hash`: the default hashes the thin-plate kinds' arrays by identity.
Base.hash(c::_CalibrationComposition, h::UInt) = hash(c.t2, hash(c.t1, hash(:ComposedTransformation, h)))

# --- lengths ---------------------------------------------------------------------
#
# `_has_lengths(c)`: whether a kind carries lengths at all (Rigid, Scale,
# Aspect and Skew do not); `has_units(c)`: whether they are physical.
# `_length_sample(c)`: one of them in its own form, to name the mismatch in a
# `DimensionError`. `length_mask(c)`: which entries of θ are lengths (they
# scale with a unit change; the rest are dimensionless).
_has_lengths(::Rigid) = false
# the one-form rule's view of a calibration (euclidean_map.jl `_check_one_form`)
_unit_form(c::TargetCalibration) = _has_lengths(c) ? (has_units(c) ? :physical : :plain) : :none
# (a calibration with lengths: a planar EuclideanMap's is its translation, euclidean_map.jl)
_form_sample(c::Union{ThinPlateSpline, HeightField, _CalibrationComposition}) = _length_sample(c)
has_units(::Rigid) = false
"""
    length_mask(::Type{C}) -> SVector{n, Bool}
    length_mask(c::TargetCalibration) -> SVector{n, Bool}

Which entries of [`parameter_vector`](@ref)`(c)` are lengths: `true` for an
entry that scales with a unit change (stored in metres for a physical
calibration), `false` for a dimensionless one — a function of the TYPE.
[`Rigid`](@ref): none; [`Scale`](@ref), [`Aspect`](@ref), [`Skew`](@ref):
none (every entry `false`); a planar `EuclideanMap`: `(false, true, true)`
(the angle, then the translation); [`ThinPlateSpline`](@ref) and
[`HeightField`](@ref): every entry; a composition: its members', the outer
member's first. A covariance over `θ` moves with a unit change by this mask; a
record of the calibration states its lengths by it.
"""
function length_mask end
length_mask(c::Union{TargetCalibration, _AnyFactor}) = length_mask(typeof(c))
length_mask(::Type{Rigid}) = SVector{0, Bool}()
length_mask(::Type{<:EuclideanMap{2}}) = SVector(false, true, true)
length_mask(::Type{<:Scale}) = SVector(false)
length_mask(::Type{<:Aspect{N}}) where {N} = SVector{N - 1}(ntuple(_ -> false, Val(N - 1)))
length_mask(::Type{<:Skew{N, T, L}}) where {N, T, L} = SVector{L}(ntuple(_ -> false, Val(L)))
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
# calibration applied to a plain point, and its point Jacobian there.
_apply(::Rigid, q) = q
_apply(c::_PlanarFactor, q) = c(q)
_apply(c::_CalibrationComposition, q) = _apply(c.t1, _apply(c.t2, q))
_point_jacobian(c::Union{Rigid, _PlanarFactor}, q) = transform_deriv(c, q)
_point_jacobian(c::_CalibrationComposition, q) = _point_jacobian(c.t1, _apply(c.t2, q)) * _point_jacobian(c.t2, q)

# ------------------------------------------------------------------------------
# Names as a function of the type (what a composition's generated names read)
# ------------------------------------------------------------------------------

_static_parameter_names(::Type{Rigid}) = ()
_static_parameter_names(::Type{<:EuclideanMap{2}}) = (:angle, :tx, :ty)
_static_parameter_names(::Type{<:Scale}) = (:log_scale,)
_static_parameter_names(::Type{<:Aspect{N}}) where {N} = ntuple(i -> Symbol(:a, i), N - 1)
_static_parameter_names(::Type{<:Skew{N}}) where {N} = ntuple(k -> ((i, j) = _skew_pair(N, k); Symbol(:u, i, j)), N * (N - 1) ÷ 2)
_static_parameter_names(::Type{<:ThinPlateSpline{K}}) where {K} =
    (ntuple(i -> Symbol(:fx, i), K)..., ntuple(i -> Symbol(:fy, i), K)...)
_static_parameter_names(::Type{<:HeightField{K}}) where {K} = ntuple(i -> Symbol(:h, i), K)
_static_gauge_names(::Type{Rigid}) = ()
_static_gauge_names(::Type{<:_AnyFactor}) = ()
_static_gauge_names(::Type{<:ThinPlateSpline}) = (:fx_1, :fx_u, :fx_v, :fy_1, :fy_u, :fy_v)
_static_gauge_names(::Type{<:HeightField}) = (:piston, :tilt_x, :tilt_y)
# A literal tuple per type, so a NamedTuple over the names is inferred.
@generated parameter_names(c::Union{Scale, Aspect, Skew, ThinPlateSpline, HeightField}) = QuoteNode(_static_parameter_names(c))
# A composition's names: each member's, prefixed by its tag, outer first (CT's order).
_composed_names(f, T::Type) = mapreduce(L -> _prefixed(calibration_tag(L), f(L)), (a, b) -> (a..., b...), _member_types(T))
_prefixed(tag, names::Tuple) = map(n -> Symbol(tag, :_, n), names)
@generated parameter_names(c::_CalibrationComposition) = QuoteNode(_composed_names(_static_parameter_names, c))

"""
    TARGET_CALIBRATION_TAGS

The [`TagRegistry`](@ref) of the target calibration kinds — the one map from a
kind's tag to its type, read by a composition's parameter names
(`:aspect_a1`) and by VisualGeometryToolkitIO's target record (`"type":
"aspect"`): `"rigid"`, `"euclidean"` (Core's `EuclideanMap`, the family's
suffix `Map` stripped), `"scale"`, `"aspect"`, `"skew"`,
`"thin_plate_spline"`, `"height_field"`, `snake_case` of the type's name. A
composition has no tag: its record is written by its structure.

RETIRED: `"affine"`, CoordinateTransformations' `AffineMap` as one kind with
six parameters (`A` row by row, then `b`) and four gauges. Its print error is
now the composition of the gauge-free factors, `Aspect ∘ Skew` (`Scale ∘` with
a measured length, `EuclideanMap ∘ Scale ∘` with a known pose); the registry
names its outer member, `"aspect"`, as its successor.
"""
const TARGET_CALIBRATION_TAGS = TagRegistry(:target_calibration, "Map",
    "rigid" => Rigid, "euclidean" => EuclideanMap, "scale" => Scale, "aspect" => Aspect, "skew" => Skew,
    "thin_plate_spline" => ThinPlateSpline, "height_field" => HeightField;
    retired = ("affine" => "aspect",))

"""
    calibration_tag(::Type{C}) -> String

The tag of the calibration kind `C`: the prefix of its parameter and gauge
names in a composition (`:aspect_a1`) and the `type` of its record — its
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
so it folds at compile time: `true` for [`Rigid`](@ref), the affine factors
and their compositions, and [`ThinPlateSpline`](@ref); `false` for a
[`HeightField`](@ref) and a composition with one. A planar target's
[`points`](@ref) are `SVector{2}`s (on `z = 0` of its frame), a non-planar
one's `SVector{3}`s. An [`Uncertain`](@ref) calibration is planar when its
kind is.
"""
function is_planar end
is_planar(::Type{Rigid}) = true
is_planar(::Type{<:_PlanarFactor}) = true
is_planar(::Type{<:ThinPlateSpline}) = true
is_planar(::Type{<:HeightField}) = false
is_planar(::Type{ComposedTransformation{A, B}}) where {A, B} = is_planar(A) && is_planar(B)
is_planar(::Type{<:Attributed{C, <:Union{CovMat, PrecMat}}}) where {C <: TargetCalibration} = is_planar(C)
is_planar(c::TargetCalibration) = is_planar(typeof(c))
is_planar(u::Uncertain{<:TargetCalibration}) = is_planar(typeof(geom(u)))

"""
    is_affine(::Type{C}) -> Bool
    is_affine(c::TargetCalibration) -> Bool

Whether a calibration's IN-PLANE part — where it takes a design point in the
target's plane, a height field's lift aside — is an affine map of the design,
`p ↦ A p + b`: a function of the TYPE, `true` for [`Rigid`](@ref), the affine
factors and their compositions, [`HeightField`](@ref) (it leaves the plane
point where it is) and a height field after affine factors; `false` for a
[`ThinPlateSpline`](@ref) and `HeightField ∘ ThinPlateSpline`. A planar
target's [`surface`](@ref) reads its chart's affine off such a calibration
(its image of the design origin and its point Jacobian there) and refuses any
other.
"""
function is_affine end
is_affine(::Type{Rigid}) = true
is_affine(::Type{<:_PlanarFactor}) = true
is_affine(::Type{<:ThinPlateSpline}) = false
is_affine(::Type{<:HeightField}) = true
is_affine(::Type{ComposedTransformation{A, B}}) where {A, B} = is_affine(A) && is_affine(B)
is_affine(c::TargetCalibration) = is_affine(typeof(c))

"""
    PlanarCalibration

The planar calibration kinds — [`Rigid`](@ref), [`ThinPlateSpline`](@ref), the
affine factors and their compositions — as one type, for a method that
assumes the design stays on `z = 0`: `f(t::PlanarTarget{<:PlanarCalibration})`
refuses a target with a [`HeightField`](@ref) by dispatch (a `MethodError`).
"""
const PlanarCalibration = Union{Rigid, ThinPlateSpline, _PlanarChain}

# ------------------------------------------------------------------------------
# Gauges — the directions the pose absorbs
# ------------------------------------------------------------------------------

"""
    gauge_names(c | target) -> Tuple{Vararg{Symbol}}

A calibration's GAUGES: the directions of its parameter vector θ (length `n`)
that images of the target cannot observe because the target's pose absorbs
them, by name; [`gauge_directions`](@ref) gives them as columns in θ's
coordinates (plain numbers; lengths in metres). The default [`TargetManifold`](@ref) holds them fixed; a caller
frees one by name, `TargetManifold(t; free = :piston)`.

These defaults assume the pose is estimated jointly — that is what
`minimal_manifold(t)` means. With a known pose (motion capture, a fixture),
every kind's parameters are observable: free the gauges by name.

- The affine factors ([`Aspect`](@ref), [`Skew`](@ref), [`Scale`](@ref), a
  planar `EuclideanMap`): none. An affine print error has gauges only as a
  whole; its factorization leaves them out instead — the pose's rotation,
  translation and scale are the members a composition omits
  (`Aspect ∘ Skew` with a jointly estimated pose).
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
  outer member's first (`:height_field_piston`, …).
- [`Rigid`](@ref): none.
"""
function gauge_names end
gauge_names(::Rigid) = ()
gauge_names(c::Union{_AnyFactor, ThinPlateSpline, HeightField}) = _static_gauge_names(typeof(c))
@generated gauge_names(c::_CalibrationComposition) = QuoteNode(_composed_names(_static_gauge_names, c))
gauge_names(u::Uncertain{<:TargetCalibration}) = gauge_names(geom(u))

"""
    gauge_directions(c) -> Matrix{Float64}

The gauges of [`gauge_names`](@ref) as the columns of an `n × g` matrix in the
coordinates of `parameter_vector(c)`, evaluated at `c` (`n × 0` for a kind
without gauges).
"""
function gauge_directions end
gauge_directions(c::Rigid) = zeros(Float64, 0, 0)
gauge_directions(c::_AnyFactor) = zeros(Float64, length(parameter_vector(c)), 0)
function gauge_directions(c::ThinPlateSpline{K}) where {K}
    o, z = ones(K), zeros(K)
    u = [Float64(p[1]) for p in c.system.nodes]
    v = [Float64(p[2]) for p in c.system.nodes]
    return [[o; z] [u; z] [v; z] [z; o] [z; u] [z; v]]
end
gauge_directions(c::HeightField{K}) where {K} =
    [ones(K) [Float64(p[1]) for p in c.system.nodes] [Float64(p[2]) for p in c.system.nodes]]
function gauge_directions(c::_CalibrationComposition)
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

# A calibration as a target stores it: a rigid member unlabeled, its
# translation in the points' stored form (lengths in metres); a composition's
# members in one form, in an accepted structure, right-associated. Anything
# else is refused.
_stored_calibration(c::Union{Rigid, ThinPlateSpline, HeightField, Scale, Aspect{2}, Skew{2}}) = c
_stored_calibration(c::EuclideanMap{2}) = unlabel(c)
function _stored_calibration(c::_CalibrationComposition)
    ms = _members(c)
    inplane = first(ms) isa HeightField ? Base.tail(ms) : ms
    accepted = !isempty(inplane) && (
        length(inplane) == 1 ? !(only(inplane) isa HeightField) :
            all(l -> l isa _PlanarFactor, inplane) && all(i -> _factor_rank(inplane[i]) < _factor_rank(inplane[i + 1]), 1:(length(inplane) - 1))
    )
    accepted || _refuse_composition(c)
    _check_one_form(ms)
    return foldr((a, b) -> ComposedTransformation(a, b), map(_stored_calibration, ms))
end
_stored_calibration(c::Uncertain) = Uncertain(_stored_calibration(geom(c)), getfield(c, :attributes))
_stored_calibration(c::ComposedTransformation) = _refuse_composition(c)
_stored_calibration(::AffineMap) = throw(
    ArgumentError(
        "an AffineMap is no longer a target calibration kind (its tag \"affine\" is retired): an affine print error is the " *
            "composition of its gauge-free Iwasawa factors, Aspect ∘ Skew with a jointly estimated pose (Scale ∘ Aspect ∘ Skew " *
            "with a measured length, EuclideanMap ∘ Scale ∘ Aspect ∘ Skew with a known pose)"
    )
)
_stored_calibration(c) = throw(
    ArgumentError(
        "a target calibration is Rigid, a planar EuclideanMap, a Scale, an Aspect{2} or a Skew{2} (or those composed in that " *
            "order), a ThinPlateSpline, a HeightField, or a HeightField after one of them; got a $(nameof(typeof(geom(c))))"
    )
)
@noinline _refuse_composition(c) = throw(
    ArgumentError(
        "a target calibration composes the affine factors in their canonical order, EuclideanMap ∘ Scale ∘ Aspect ∘ Skew " *
            "(any of them left out), and a HeightField only outermost, after those or after a ThinPlateSpline; a ThinPlateSpline " *
            "and Rigid compose with nothing else (a ThinPlateSpline frees its own stretch and shear by name). " *
            "Got $(_composition_spelling(c))"
    )
)
_composition_spelling(c::ComposedTransformation) = "$(_composition_spelling(c.t1)) ∘ $(_composition_spelling(c.t2))"
_composition_spelling(c) = string(nameof(typeof(c)))
