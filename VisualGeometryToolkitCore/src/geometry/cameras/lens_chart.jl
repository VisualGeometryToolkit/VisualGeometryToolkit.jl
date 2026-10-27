# =============================================================================
# The parameter protocol, the lens manifold, and the lens record
# =============================================================================
#
# One protocol for everything with a coefficient vector — a lens, a calibratable
# target and its calibration, a `LogicalIntrinsics`:
#
#   parameter_vector(x)     the coefficient vector (the manifold's coordinates)
#   parameter_names(x)      the coefficients' names, so a configuration can
#                           free or fix them by name
#   parameter_bounds(x)     the interval each coefficient may take (the ONE
#                           source of bounds; the lens manifold's logit chart
#                           reads it)
#   with_parameters(x, θ)   the same structure rebuilt from a vector
#   parameters(x)           names and values, as a NamedTuple
#
# Derived for every lens from its fields (lenses.jl), never tabulated.
#   lens_record / lens_from_record the lens as a self-describing record (JSON, TOML)

"""
    parameter_vector(x) -> SVector

The coefficients of a model `x` — a lens, a calibratable target or its calibration, or
a [`LogicalIntrinsics`](@ref) — in the order of [`parameter_names`](@ref).
Inverse of [`with_parameters`](@ref).

A lens's are the projection's, then the distortion's terms depth-first, left to
right, each term in its own documented order ([`Radial`](@ref): `k…, d…`;
[`Tangential`](@ref): `p1, p2`; [`ThinPrism`](@ref): `sx…, sy…`; …); empty for
a lens without coefficients. A `PlanarTarget`'s are its calibration's (empty for
[`Rigid`](@ref)). A `LogicalIntrinsics`'s are `f, aspect, skew, cx, cy` of
`K = [f s cx; 0 f·aspect cy; 0 0 1]`, absolute and in pixels.
"""
function parameter_vector end

"""
    parameter_names(x) -> NTuple{N, Symbol}

The names of [`parameter_vector`](@ref), in its order — what a configuration
refers to when it frees or fixes coefficients by name. A lens's are generated
from its leaves: `k1…, d1…` ([`Radial`](@ref)), `p1, p2`, `sx1…, sy1…`, `tx, ty`,
`omega`, `xi`, `alpha`, `beta`. A `LogicalIntrinsics`'s are
`(:f, :aspect, :skew, :cx, :cy)`.
"""
function parameter_names end

"""
    parameter_bounds(x) -> (lo::SVector, hi::SVector)

The interval each coefficient of [`parameter_vector`](@ref) may take, entry by
entry — the model's DOMAIN, where it is defined: `(-Inf, Inf)` unless the
model says otherwise. Any interval `(a, b)`, `a < b`, either end infinite: the
double sphere's `ξ > −1` and `α ∈ [0, 1]`, the unified model's `ξ > −1`, the
extended unified model's `α ∈ [0, 1]` and `β ≥ 0`, the field-of-view term's
`ω ∈ [0, π)`, a [`LogicalIntrinsics`](@ref)'s `f > 0` and `aspect > 0`; a
[`CameraModel`](@ref)'s are its intrinsics' then its lens's.

This is the only place a bound is stated: [`ParameterManifold`](@ref) (a
lens's [`LensManifold`](@ref)) reads it, so no step leaves the domain. An
optimizer's box — a prior range a solver searches, such as a focal length
between 0.1 and 50 — is not a model fact and is not stated here: it is the
solver's option, keyed by [`parameter_names`](@ref).
"""
function parameter_bounds end

# The unbounded default: any model with a parameter vector.
@inline function parameter_bounds(x)
    θ = parameter_vector(x)
    T = eltype(θ)
    return (zero(θ) .- T(Inf), zero(θ) .+ T(Inf))
end

@inline function parameter_vector(l::AbstractLensModel)
    c = _coefficients_of(l)
    T = _coeff_type(l)
    return SVector{length(c), T === Union{} ? Float64 : T}(c)
end

"""
    with_parameters(x, θ::AbstractVector) -> x

The model rebuilt from the coefficient vector `θ` ([`parameter_vector`](@ref)'
inverse): the same structure and arities, the number type promoted to `θ`'s.
"""
function with_parameters(l::AbstractLensModel, θ::AbstractVector)
    n = length(_coefficients_of(l))
    length(θ) == n || throw(ArgumentError("$(nameof(typeof(l))) takes $n parameters; got $(length(θ))"))
    return _with_parameters(l, θ)
end

parameter_names(l::AbstractLensModel) = _coefficient_names_of(l)

# The bounds, per leaf: an interval for each coefficient (Float64 constants,
# converted to the lens's number type in `parameter_bounds`) — the leaf's
# domain, where its map is defined and orientation-preserving at the axis:
# - DoubleSphere: ξ > −1 (the first sphere's profile ∝ 1 + ξ cos θ is
#   increasing at the axis), α ∈ [0, 1] (the second stage's weight);
# - Unified: ξ > −1, the same factor;
# - ExtendedUnified: α ∈ [0, 1], β ≥ 0 (d = √(β r² + z²) is real);
# - FieldOfView: ω ∈ [0, π) (tan(ω/2) finite; ω and −ω are one map).
# Every other coefficient is unbounded.
@inline _bound_pairs(leaf) = map(_ -> (-Inf, Inf), _coefficients(leaf))
@inline _bound_pairs(::DoubleSphere) = ((-1.0, Inf), (0.0, 1.0))
@inline _bound_pairs(::Unified) = ((-1.0, Inf),)
@inline _bound_pairs(::ExtendedUnified) = ((0.0, 1.0), (0.0, Inf))
@inline _bound_pairs(::FieldOfView) = ((0.0, Float64(π)),)

@inline function parameter_bounds(l::AbstractLensModel)
    pairs = _flatmap(_bound_pairs, _leaves(l))
    T = _bounds_eltype(l)
    return (SVector(map(p -> T(p[1]), pairs)), SVector(map(p -> T(p[2]), pairs)))
end
@inline _bounds_eltype(l) = (T = _coeff_type(l); T === Union{} ? Float64 : T)

"""
    parameters(x) -> NamedTuple

The model's coefficients by name: `parameters(KannalaBrandt(-0.04, 0.01)).k2`.
"""
parameters(l::AbstractLensModel) = NamedTuple{parameter_names(l)}(Tuple(parameter_vector(l)))

# --- LogicalIntrinsics: absolute, reportable coordinates --------------------------
#
# K = [f s cx; 0 f·aspect cy; 0 0 1], so a camera comparison reads the
# intrinsics' coordinates the way it reads a lens's.

parameter_names(::LogicalIntrinsics) = (:f, :aspect, :skew, :cx, :cy)
# f > 0 and aspect > 0: K maps the image the right way up (a positive focal per axis).
@inline function parameter_bounds(::LogicalIntrinsics{T}) where {T}
    F = float(T)
    return (SVector{5, F}(0, 0, -Inf, -Inf, -Inf), SVector{5, F}(Inf, Inf, Inf, Inf, Inf))
end
function parameter_vector(intr::LogicalIntrinsics{T}) where {T}
    K = intr.K
    fx = K[1, 1]
    return SVector{5, T}(fx, K[2, 2] / fx, K[1, 2], K[1, 3], K[2, 3])
end
function with_parameters(::LogicalIntrinsics, θ::AbstractVector)
    length(θ) == 5 || throw(
        ArgumentError("LogicalIntrinsics takes 5 parameters $((:f, :aspect, :skew, :cx, :cy)); got $(length(θ))")
    )
    f, aspect, skew, cx, cy = float.(θ)
    T = typeof(f)
    K = SMatrix{3, 3, T}(f, zero(T), zero(T), skew, f * aspect, zero(T), cx, cy, one(T))
    return LogicalIntrinsics(CameraCalibrationMat(K))
end

"""
    convert_eltype(::Type{T}, x) where {T <: AbstractFloat}

`x` rebuilt with every stored number in `T`: a lens (its coefficients), a
`LogicalIntrinsics` (`K`, with `K⁻¹` recomputed in `T`), a `CameraModel`
(both), a `CentralGenericModel` or `NoncentralGenericModel` (the grids;
the calibrated area stays as stored). Use it to move a camera to `Float32`
before a GPU launch, or back to `Float64`.

A lens's and a grid model's method is also what the batched and fused
kernels apply once per launch, so per-element math never promotes (Metal has
no Float64): a downstream lens type that adds a method here is converted by
them too.

Public, not exported; a downstream camera or lens type adds its own method.
"""
function convert_eltype end
# Widening keeps the radial factors' branch ends, cast (see `Radial`); narrowing
# recomputes them from the rounded coefficients.
convert_eltype(::Type{T}, l::AbstractLensModel) where {T <: AbstractFloat} =
    _widens(T, _coeff_type(l)) ? _with_own_parameters(l, map(T, parameter_vector(l))) : _with_parameters(l, map(T, parameter_vector(l)))
_widens(::Type{T}, ::Type{C}) where {T, C} = C === Union{} || promote_type(T, C) === T
convert_eltype(::Type{T}, intr::LogicalIntrinsics) where {T <: AbstractFloat} =
    LogicalIntrinsics(CameraCalibrationMat{T}(intr.K))
convert_eltype(::Type{T}, m::CameraModel) where {T <: AbstractFloat} =
    CameraModel(convert_eltype(T, m.intrinsics), convert_eltype(T, m.lens))

# --- CameraModel: the intrinsics' coordinates, then the lens's ---------------------
#
# A parametric camera's parameter vector is its intrinsics' followed by its
# lens's, each part answered by its own methods once; the names stay unique
# because the intrinsics' (`f, aspect, skew, cx, cy`) are no lens coefficient's.

parameter_names(m::CameraModel) = (parameter_names(m.intrinsics)..., parameter_names(m.lens)...)
@inline parameter_vector(m::CameraModel) = vcat(parameter_vector(m.intrinsics), parameter_vector(m.lens))
@inline function parameter_bounds(m::CameraModel)
    lo_i, hi_i = parameter_bounds(m.intrinsics)
    lo_l, hi_l = parameter_bounds(m.lens)
    return vcat(lo_i, lo_l), vcat(hi_i, hi_l)
end
function with_parameters(m::CameraModel, θ::AbstractVector)
    ni, nl = length(parameter_names(m.intrinsics)), length(parameter_names(m.lens))
    length(θ) == ni + nl ||
        throw(ArgumentError("CameraModel takes $(ni + nl) parameters $(parameter_names(m)); got $(length(θ))"))
    return _with_camera_parameters(m, θ, Val(ni), Val(nl))
end
@inline function _with_camera_parameters(m::CameraModel, θ, ::Val{ni}, ::Val{nl}) where {ni, nl}
    θi = SVector{ni}(ntuple(k -> θ[k], Val(ni)))
    θl = SVector{nl}(ntuple(k -> θ[ni + k], Val(nl)))
    return CameraModel(with_parameters(m.intrinsics, θi), with_parameters(m.lens, θl))
end
parameters(m::CameraModel) = NamedTuple{parameter_names(m)}(Tuple(parameter_vector(m)))
parameters(intr::LogicalIntrinsics) = NamedTuple{parameter_names(intr)}(Tuple(parameter_vector(intr)))

# --- the parameter manifold ----------------------------------------------------------

"""
    ParameterManifold(x) <: Manifold

The manifold of a model `x` with the parameter protocol — a lens, a
[`LogicalIntrinsics`](@ref), a parametric [`CameraModel`](@ref) (intrinsics
then lens) — anchored at `x`: [`parameter_vector`](@ref)'s coordinates, each
moved by a chart that keeps it inside its [`parameter_bounds`](@ref), the one
statement of the model's domain:

| bound | chart coordinate `v` of the value `t` (anchor `t₀`) |
|---|---|
| `(-Inf, Inf)` | `t = t₀ + v` |
| `(a, Inf)` | `t = a + (t₀ − a)·exp(v)` |
| `(-Inf, b)` | `t = b − (b − t₀)·exp(−v)` |
| `(a, b)` | `t = a + (b − a)·logistic(logit((t₀ − a)/(b − a)) + v)` |

so no step leaves the domain and every fitter (VisualGeometryToolkitEstimation's
`fit_lens`, a calibration bundle) keeps the model valid without knowing why. A
bound is open in the chart: an anchor on (or past) one is moved inside it by a
margin of 10⁻⁶ of the interval (of `max(1, |a|)` for a one-sided bound).
[`restrict`](@ref) it to the free coefficients by mask or by name
(`restrict(ch, (:k1, :k2))`; a camera's `restrict(ch, (:f, :cx, :cy, :k1))`
fixes its skew and aspect). `minimal_manifold(lens)` and
`minimal_manifold(camera)` are this manifold. `action_jacobian(ch, x)` is the
model's parameter Jacobian of its action on `x` through the chart: the lens's
`transform_deriv_params(lens, x)`, a camera's `∂project(camera, X)/∂θ`
(ForwardDiff).

[`LensManifold`](@ref) is its name for a lens.
"""
struct ParameterManifold{M} <: Manifold
    model::M
    ParameterManifold{M}(x::M) where {M} = new{M}(x)
end
function ParameterManifold(x)
    _check_bounds(x)
    return ParameterManifold{typeof(x)}(_interior(x))
end

"""
    LensManifold(lens) <: Manifold

The lens's [`ParameterManifold`](@ref): `parameter_vector`'s coordinates as
offsets from the anchor `lens`, each kept inside its
[`parameter_bounds`](@ref) by the chart that table gives (the double sphere's
`α ∈ [0, 1]` through its logit, its `ξ > −1` through a log). `minimal_manifold(lens)`
is this manifold; [`restrict`](@ref) it to a target's free coefficients by
mask or by name (`restrict(ch, (:k1, :k2))`).
"""
const LensManifold{L <: AbstractLensModel} = ParameterManifold{L}
LensManifold(l::AbstractLensModel) = ParameterManifold(l)

# A lens rebuilds through the generated walk (no length check: the length is
# the lens's own); any other model through its `with_parameters`.
@inline _rebuild(l::AbstractLensModel, θ) = _with_parameters(l, θ)
@inline _rebuild(x, θ) = with_parameters(x, θ)

# A bound is a pair (a, b) with a < b, either end infinite.
function _check_bounds(x)
    lo, hi = parameter_bounds(x)
    for (i, name) in enumerate(parameter_names(x))
        a, b = lo[i], hi[i]
        (isnan(a) || isnan(b) || !(a < b)) && throw(
            ArgumentError("ParameterManifold: `$name` of $(nameof(typeof(x))) is bounded to ($a, $b), not an interval")
        )
    end
    return nothing
end

# Which coordinates are bounded: read off `parameter_bounds`, the one source of
# bounds. A function of the model's type alone, so it constant-folds.
@inline function _bounded_mask(x)
    lo, hi = parameter_bounds(x)
    return map((a, b) -> isfinite(a) | isfinite(b), Tuple(lo), Tuple(hi))
end
_has_bound(x) = any(_bounded_mask(x))

# The chart of one coordinate (the table above): its value at chart offset d
# from the anchor t0, the derivative of that value in d, and the inverse.
@inline function _bounded_value(t0, d, a, b)
    fa, fb = isfinite(a), isfinite(b)
    fa & fb && return a + (b - a) * logistic(logit((t0 - a) / (b - a)) + d)
    fa && return a + (t0 - a) * exp(d)
    fb && return b - (b - t0) * exp(-d)
    return t0 + d
end
@inline function _bounded_slope(t0, d, a, b)
    fa, fb = isfinite(a), isfinite(b)
    fa & fb && return (s = logistic(logit((t0 - a) / (b - a)) + d); (b - a) * s * (1 - s))
    fa && return (t0 - a) * exp(d)
    fb && return (b - t0) * exp(-d)
    return one(promote_type(typeof(t0), typeof(d)))
end
# at the anchor, in closed form: the logistic's slope there is (t0 − a)(b − t0)/(b − a)
@inline function _bounded_slope_at_anchor(t0, a, b)
    fa, fb = isfinite(a), isfinite(b)
    fa & fb && return (t0 - a) * (b - t0) / (b - a)
    fa && return t0 - a
    fb && return b - t0
    return one(t0)
end
@inline function _bounded_offset(t, t0, a, b)
    fa, fb = isfinite(a), isfinite(b)
    t = _interior_value(t, a, b)
    fa & fb && return logit((t - a) / (b - a)) - logit((t0 - a) / (b - a))
    fa && return log((t - a) / (t0 - a))
    fb && return log((b - t0) / (b - t))
    return t - t0
end

# A boundary value — 0 is a natural default for α — sits at the chart's ±∞,
# where no step could move it: the manifold anchors inside the interval.
const _UNIT_MARGIN = 1.0e-6
@inline function _interior_value(t, a, b)
    fa, fb = isfinite(a), isfinite(b)
    fa & fb && return clamp(t, a + oftype(t, _UNIT_MARGIN) * (b - a), b - oftype(t, _UNIT_MARGIN) * (b - a))
    fa && return max(t, a + oftype(t, _UNIT_MARGIN) * max(one(t), abs(a)))
    fb && return min(t, b - oftype(t, _UNIT_MARGIN) * max(one(t), abs(b)))
    return t
end
function _interior(x)
    _has_bound(x) || return x
    θ = parameter_vector(x)
    lo, hi = parameter_bounds(x)
    return _rebuild(x, map(_interior_value, θ, lo, hi))
end

minimal_manifold(l::AbstractLensModel) = LensManifold(l)
minimal_manifold(m::CameraModel) = ParameterManifold(m)
ambient_coordinates(l::AbstractLensModel) = parameter_vector(l)
ambient_coordinates(m::CameraModel) = parameter_vector(m)
tangent_size(ch::ParameterManifold) = length(parameter_vector(ch.model))
model(ch::ParameterManifold) = ch.model
function retract(ch::ParameterManifold, v)
    θ0 = parameter_vector(ch.model)
    _has_bound(ch.model) || return _rebuild(ch.model, θ0 + v)
    lo, hi = parameter_bounds(ch.model)
    return _rebuild(ch.model, map(_bounded_value, θ0, v, lo, hi))
end
recenter(ch::ParameterManifold, v) = ParameterManifold(retract(ch, v))
function retract_jacobian(ch::ParameterManifold)
    n = tangent_size(ch)
    _has_bound(ch.model) || return SMatrix{n, n, Float64}(LinearAlgebra.I)
    θ = parameter_vector(ch.model)
    lo, hi = parameter_bounds(ch.model)
    d = map((t, a, b) -> Float64(_bounded_slope_at_anchor(t, a, b)), θ, lo, hi)
    return SMatrix{n, n, Float64}(LinearAlgebra.Diagonal(d))
end
# at coordinates v the slope is the chart's there
function retract_jacobian(ch::ParameterManifold, v::StaticVector)
    _has_bound(ch.model) || return retract_jacobian(ch)     # the chart is linear: constant differential
    n = tangent_size(ch)
    θ = parameter_vector(ch.model)
    lo, hi = parameter_bounds(ch.model)
    return SMatrix{n, n}(LinearAlgebra.Diagonal(map(_bounded_slope, θ, v, lo, hi)))
end
action_jacobian(ch::ParameterManifold, x) =
    _has_bound(ch.model) ? _parameter_jacobian(ch.model, x) * retract_jacobian(ch) : _parameter_jacobian(ch.model, x)
# The model's own parameter Jacobian of its action: a lens's closed-form route,
# a camera's ForwardDiff through `project`.
_parameter_jacobian(l::AbstractLensModel, x) = transform_deriv_params(l, x)
function _parameter_jacobian(m::CameraModel, X)
    Xf = _float_static(X)
    return _jacobian(q -> project(with_parameters(m, q), Xf), parameter_vector(m))
end
function local_coordinates(ch::ParameterManifold, x)
    _same_structure(ch.model, x) || throw(
        ArgumentError("local_coordinates: a $(_structure_name(x)) is not on the manifold of $(_structure_name(ch.model))")
    )
    θ, θ0 = parameter_vector(x), parameter_vector(ch.model)
    _has_bound(x) || return θ - θ0
    lo, hi = parameter_bounds(ch.model)
    return map(_bounded_offset, θ, θ0, lo, hi)
end
local_coordinates(rc::SubsetManifold{<:ParameterManifold}, x) = local_coordinates(rc.manifold, x)[rc.idx]

# Same structure: the same model type up to the coefficient number type.
_same_structure(a, b) = typeof(_structure(a)) === typeof(_structure(b))
_structure_name(x) = string(typeof(_structure(x)))
_structure(l::AbstractLensModel) = _precision(Float64, l)
_structure(x) = convert_eltype(Float64, x)

"""
    restrict(ch::ParameterManifold, names::Tuple{Vararg{Symbol}}) -> SubsetManifold

The manifold on the coefficients `names` ([`parameter_names`](@ref)'
vocabulary) — a target's complexity as a manifold, a camera's fixed skew.
"""
function restrict(ch::ParameterManifold, names::Tuple{Symbol, Vararg{Symbol}})
    all_names = parameter_names(ch.model)
    for n in names
        n in all_names || throw(
            ArgumentError("restrict: the model has no coefficient $(repr(n)); its coefficients are $(join(String.(all_names), ", "))")
        )
    end
    return restrict(ch, SVector{length(all_names), Bool}(ntuple(i -> all_names[i] in names, Val(length(all_names)))))
end

# --- the lens record -----------------------------------------------------------------
#
# A lens as a self-describing record (a Dict that JSON and TOML both carry), one
# canonical form per lens: every key a leaf tag (`_tag`) or a combinator, every
# value the leaf's fields — a list of numbers, or a list of lists for tuple fields.
#
#   lens   := { PROJ: [c…] }                                  a pure projection
#           | { PROJ: [c…], TERM: payload }                    projection + one term
#           | { PROJ: [c…], "add":   [ {TERM: payload}, … ] }
#           | { PROJ: [c…], "chain": [ {TERM | "add": …}, … ] }
#           | { "lift": { "radial" | "radial_full": [[k…], [d…]] } }
#   payload: radial / radial_full [[k…], [d…]]; tangential [p1, p2];
#            thin_prism [[sx…], [sy…]]; tilt [tx, ty]; field_of_view [omega]

"""
    lens_record(lens) -> Dict{String, Any}

The lens as a record: `{"equidistant": [], "radial": [[k₁, …], []]}`,
`{"perspective": [], "add": [{"radial": …}, {"tangential": [p₁, p₂]}]}`,
`{"lift": {"radial": [[], [k₁]]}}` — keys are the leaves' tags, arities are
array lengths, nothing is named or padded. [`lens_from_record`](@ref) inverts
it exactly: written with shortest round-trip floats, a Float64 lens reads back
bitwise identical.
"""
lens_record(l::Lens) = merge!(_record(l.projection), _record(l.distortion))
lens_record(l::LiftLens) = Dict{String, Any}("lift" => _record(l.radial))
toml_value(l::AbstractLensModel) = lens_record(l)     # a lens-valued config field writes as its record

_record(::Nothing) = Dict{String, Any}()
_record(a::Additive) = Dict{String, Any}("add" => Any[_record(t) for t in a.terms])
_record(c::Chain) = Dict{String, Any}("chain" => Any[_record(t) for t in c.stages])
_record(leaf) = Dict{String, Any}(
    _tag(typeof(leaf)) => any(f -> f isa Tuple, _fields(leaf)) ? Any[collect(Float64, f) for f in _fields(leaf)] :
        collect(Float64, _coefficients(leaf))
)

const _LEAVES = Dict{String, Any}(
    _tag(X) => X for X in (
        Perspective, Equidistant, Stereographic, Equisolid, Orthographic, DoubleSphere, Unified, ExtendedUnified,
        Radial{Even}, Radial{Full}, Tangential, ThinPrism, Tilt, FieldOfView,
    )
)
_is_projection_tag(k) = get(_LEAVES, String(k), Nothing) <: AbstractProjection

"""
    lens_from_record([T = Float64,] d::AbstractDict) -> AbstractLensModel

The lens of a [`lens_record`](@ref), its coefficients in `T`. Unknown keys,
missing or extra coefficients, and a second distortion are refused, naming
the offending key.
"""
lens_from_record(d::AbstractDict) = lens_from_record(Float64, d)
function lens_from_record(::Type{T}, d::AbstractDict) where {T <: AbstractFloat}
    keylist() = join(sort(collect(String.(keys(d)))), ", ")
    if haskey(d, "lift")
        length(d) == 1 || throw(ArgumentError("lens record: a lift has no other keys; got $(keylist())"))
        r = _from_record(T, d["lift"])
        r isa Radial || throw(ArgumentError("lens record: a lift is a radial or radial_full term"))
        return LiftLens(r)
    end
    projs = filter(_is_projection_tag, collect(keys(d)))
    length(projs) == 1 || throw(ArgumentError("lens record: needs exactly one projection; got $(keylist())"))
    p = _leaf(T, String(only(projs)), d[only(projs)])
    length(d) == 1 && return Lens(p)
    length(d) == 2 || throw(ArgumentError("lens record: one distortion per lens; got $(keylist())"))
    k = only(filter(!=(only(projs)), collect(keys(d))))
    return Lens(p, _from_record(T, Dict(String(k) => d[k])))
end

# A one-key term record.
function _from_record(::Type{T}, d) where {T}
    (d isa AbstractDict && length(d) == 1) || throw(ArgumentError("lens record: a term is a one-key record; got $(repr(d))"))
    k, v = String(only(keys(d))), only(values(d))
    if k == "add" || k == "chain"
        (v isa AbstractVector && !isempty(v)) || throw(ArgumentError("lens record: \"$k\" is a non-empty list of terms"))
        parts = map(x -> _from_record(T, x), v)
        return k == "add" ? Additive(parts...) : Chain(parts...)
    end
    (haskey(_LEAVES, k) && !_is_projection_tag(k)) || throw(
        ArgumentError("lens record: unknown key \"$k\"; known: add, chain, $(join(sort(filter(!_is_projection_tag, collect(keys(_LEAVES)))), ", "))")
    )
    return _leaf(T, k, v)
end

# A leaf from its payload: one entry per field, a number or a list of numbers.
function _leaf(::Type{T}, tag, v) where {T}
    X = _LEAVES[tag]
    n = _param_fields(X)
    entry_ok(e) = e isa Real || (e isa AbstractVector && all(x -> x isa Real, e))
    (v isa AbstractVector && length(v) == n && all(entry_ok, v)) || throw(
        ArgumentError("lens record: \"$tag\" takes $n entries, each a number or a list of numbers; got $(repr(v))")
    )
    args = map(e -> e isa Real ? T(e) : Tuple(map(T, e)), v)
    return try
        X(args...)
    catch e
        e isa MethodError ? throw(ArgumentError("lens record: \"$tag\" has the wrong shape: $(repr(v))")) : rethrow()
    end
end
