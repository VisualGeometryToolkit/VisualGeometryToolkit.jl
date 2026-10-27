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
entry: `(-Inf, Inf)` unless the model says otherwise. The double sphere's `α` is
the one bounded lens coefficient, in `(0, 1)`. This is the only place a bound is
stated: [`LensManifold`](@ref) moves a coefficient bounded to `(0, 1)` through
its logit, so no step leaves the interval.
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
# converted to the lens's number type in `parameter_bounds`). Only the double
# sphere's α departs from the unbounded default.
@inline _bound_pairs(leaf) = map(_ -> (-Inf, Inf), _coefficients(leaf))
@inline _bound_pairs(::DoubleSphere) = ((-Inf, Inf), (0.0, 1.0))

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
convert_eltype(::Type{T}, l::AbstractLensModel) where {T <: AbstractFloat} =
    _with_parameters(l, map(T, parameter_vector(l)))
convert_eltype(::Type{T}, intr::LogicalIntrinsics) where {T <: AbstractFloat} =
    LogicalIntrinsics(CameraCalibrationMat{T}(intr.K))
convert_eltype(::Type{T}, m::CameraModel) where {T <: AbstractFloat} =
    CameraModel(convert_eltype(T, m.intrinsics), convert_eltype(T, m.lens))

# --- the lens manifold -------------------------------------------------------------

"""
    LensManifold(lens) <: Manifold

The lens's manifold anchored at `lens`: `parameter_vector`' coordinates as
offsets from the anchor, except that a coefficient bounded to `(0, 1)` — the
double sphere's `α` — moves through its logit, so no step leaves the interval
and every fitter (VisualGeometryToolkitEstimation's `fit_lens`, a calibration bundle)
keeps the model valid without knowing why. `action_jacobian(ch, x)` is the
lens's own `transform_deriv_params(lens, x)` through that chart.
`minimal_manifold(lens)` is this manifold; [`restrict`](@ref) it to a target's
free coefficients by mask or by name (`restrict(ch, (:k1, :k2))`).
"""
struct LensManifold{L <: AbstractLensModel} <: Manifold
    lens::L
    LensManifold{L}(l::L) where {L <: AbstractLensModel} = new{L}(l)
end
function LensManifold(l::AbstractLensModel)
    _check_unit_bounds(l)
    return LensManifold{typeof(l)}(_interior(l))
end

# The chart's logit handles exactly the (0, 1) interval; any other finite bound
# would be silently ignored by `retract`, so it is refused here instead.
function _check_unit_bounds(l)
    lo, hi = parameter_bounds(l)
    for (i, name) in enumerate(parameter_names(l))
        a, b = lo[i], hi[i]
        (isfinite(a) || isfinite(b)) && !(a == 0 && b == 1) && throw(
            ArgumentError(
                "LensManifold: `$name` of $(nameof(typeof(l))) is bounded to ($a, $b); " *
                    "the chart handles only (0, 1) bounds"
            )
        )
    end
    return nothing
end

# Which coordinates are bounded to (0, 1): read off `parameter_bounds`, the one
# source of bounds. A function of the lens's type alone, so it constant-folds.
@inline function _logit_mask(l)
    lo, hi = parameter_bounds(l)
    return map((a, b) -> a == 0 && b == 1, Tuple(lo), Tuple(hi))
end
_has_logit(l) = any(_logit_mask(l))

# A boundary value — 0 is a natural default for α — sits at the logit's ±∞,
# where no step could move it: the manifold anchors δ inside the interval.
const _UNIT_MARGIN = 1.0e-6
_interior_unit(a) = clamp(a, oftype(a, _UNIT_MARGIN), one(a) - oftype(a, _UNIT_MARGIN))
function _interior(l::AbstractLensModel)
    _has_logit(l) || return l
    θ = parameter_vector(l)
    return _with_parameters(l, map((t, b) -> b ? _interior_unit(t) : t, θ, SVector(_logit_mask(l))))
end

minimal_manifold(l::AbstractLensModel) = LensManifold(l)
ambient_coordinates(l::AbstractLensModel) = parameter_vector(l)
tangent_size(ch::LensManifold) = length(parameter_vector(ch.lens))
model(ch::LensManifold) = ch.lens
function retract(ch::LensManifold, v)
    θ0 = parameter_vector(ch.lens)
    _has_logit(ch.lens) || return _with_parameters(ch.lens, θ0 + v)
    θ = map((t, d, b) -> b ? logistic(logit(t) + d) : t + d, θ0, v, SVector(_logit_mask(ch.lens)))
    return _with_parameters(ch.lens, θ)
end
recenter(ch::LensManifold, v) = LensManifold(retract(ch, v))
function retract_jacobian(ch::LensManifold)
    n = tangent_size(ch)
    _has_logit(ch.lens) || return SMatrix{n, n, Float64}(LinearAlgebra.I)
    θ = parameter_vector(ch.lens)
    d = map((t, b) -> b ? Float64(t * (1 - t)) : 1.0, θ, SVector(_logit_mask(ch.lens)))
    return SMatrix{n, n, Float64}(LinearAlgebra.Diagonal(d))
end
# at coordinates v the slope is the logistic's there: a(v)·(1 − a(v))
function retract_jacobian(ch::LensManifold, v::StaticVector)
    _has_logit(ch.lens) || return retract_jacobian(ch)     # the chart is linear: constant differential
    n = tangent_size(ch)
    θ = parameter_vector(ch.lens)
    d = map((t, dv, b) -> b ? (a = logistic(logit(t) + dv); a * (1 - a)) : one(promote_type(typeof(t), typeof(dv))), θ, v, SVector(_logit_mask(ch.lens)))
    return SMatrix{n, n}(LinearAlgebra.Diagonal(d))
end
action_jacobian(ch::LensManifold, x) =
    _has_logit(ch.lens) ? transform_deriv_params(ch.lens, x) * retract_jacobian(ch) : transform_deriv_params(ch.lens, x)
function local_coordinates(ch::LensManifold, l::AbstractLensModel)
    _same_structure(ch.lens, l) || throw(
        ArgumentError("local_coordinates: a $(_structure_name(l)) is not on the manifold of $(_structure_name(ch.lens))")
    )
    θ, θ0 = parameter_vector(l), parameter_vector(ch.lens)
    _has_logit(l) || return θ - θ0
    return map((t, t0, b) -> b ? logit(_interior_unit(t)) - logit(t0) : t - t0, θ, θ0, SVector(_logit_mask(l)))
end
local_coordinates(rc::SubsetManifold{<:LensManifold}, l::AbstractLensModel) = local_coordinates(rc.manifold, l)[rc.idx]

# Same structure: the same lens type up to the coefficient number type.
_same_structure(a::AbstractLensModel, b::AbstractLensModel) =
    typeof(_precision(Float64, a)) === typeof(_precision(Float64, b))
_structure_name(l) = string(typeof(_precision(Float64, l)))

"""
    restrict(ch::LensManifold, names::Tuple{Vararg{Symbol}}) -> SubsetManifold

The lens manifold on the coefficients `names` ([`parameter_names`](@ref)'
vocabulary) — a target's complexity as a manifold.
"""
function restrict(ch::LensManifold, names::Tuple{Symbol, Vararg{Symbol}})
    all_names = parameter_names(ch.lens)
    for n in names
        n in all_names || throw(
            ArgumentError("restrict: the lens has no coefficient $(repr(n)); its coefficients are $(join(String.(all_names), ", "))")
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
