# =============================================================================
# Lens parameters, the lens manifold, and the lens record
# =============================================================================
#
# Derived for every lens from its fields (lenses.jl), never tabulated:
#
#   lens_parameters(lens)          the coefficient vector (the manifold's coordinates)
#   with_parameters(lens, θ)       the same structure rebuilt from a vector
#   lens_parameter_names(lens)     the coefficients' names, so a configuration can
#                                  free or fix them by name
#   parameters(lens)               both, as a NamedTuple
#   lens_record / lens_from_record the lens as a self-describing record (JSON, TOML)

"""
    lens_parameters(lens) -> SVector

The lens's coefficients, in order: the projection's, then the distortion's
terms depth-first, left to right, each term in its own documented order
([`Radial`](@ref): `k…, d…`; [`Tangential`](@ref): `p1, p2`;
[`ThinPrism`](@ref): `sx…, sy…`; …). Empty for a lens without coefficients.
Inverse of [`with_parameters`](@ref).
"""
@inline function lens_parameters(l::AbstractLensModel)
    c = _coefficients_of(l)
    T = _coeff_type(l)
    return SVector{length(c), T === Union{} ? Float64 : T}(c)
end

"""
    with_parameters(lens, θ::AbstractVector) -> lens

The lens rebuilt from the coefficient vector `θ` ([`lens_parameters`](@ref)'
inverse): the same structure and arities, the number type promoted to `θ`'s.
"""
function with_parameters(l::AbstractLensModel, θ::AbstractVector)
    n = length(_coefficients_of(l))
    length(θ) == n || throw(ArgumentError("$(nameof(typeof(l))) takes $n parameters; got $(length(θ))"))
    return _with_parameters(l, θ)
end

"""
    lens_parameter_names(lens) -> NTuple{N, Symbol}

The names of [`lens_parameters`](@ref), in its order — what a configuration
refers to when it frees or fixes coefficients by name. Generated from the
leaves: `k1…, d1…` ([`Radial`](@ref)), `p1, p2`, `sx1…, sy1…`, `tx, ty`,
`omega`, `xi`, `alpha`, `beta`.
"""
lens_parameter_names(l::AbstractLensModel) = _coefficient_names_of(l)

"""
    parameters(lens) -> NamedTuple

The lens's coefficients by name: `parameters(KannalaBrandt(-0.04, 0.01)).k2`.
"""
parameters(l::AbstractLensModel) = NamedTuple{lens_parameter_names(l)}(Tuple(lens_parameters(l)))

# Precision hook (transforms/warp.jl's `_precision`): a lens rebuilt with
# T-typed coefficients — the batched and fused kernel surfaces convert ONCE per
# launch so per-element math never promotes (Metal has no Float64).
_precision(::Type{T}, l::AbstractLensModel) where {T <: AbstractFloat} =
    _with_parameters(l, map(T, lens_parameters(l)))

# --- the lens manifold -------------------------------------------------------------

"""
    LensManifold(lens) <: Manifold

The lens's manifold anchored at `lens`: `lens_parameters`' coordinates as
offsets from the anchor, except that a coefficient bounded to `(0, 1)` — the
double sphere's `α` — moves through its logit, so no step leaves the interval
and every fitter (RobustVisualGeometry's `fit_lens`, a calibration bundle)
keeps the model valid without knowing why. `action_jacobian(ch, x)` is the
lens's own `transform_deriv_params(lens, x)` through that chart.
`minimal_manifold(lens)` is this manifold; [`restrict`](@ref) it to a target's
free coefficients by mask or by name (`restrict(ch, (:k1, :k2))`).
"""
struct LensManifold{L <: AbstractLensModel} <: Manifold
    lens::L
    LensManifold{L}(l::L) where {L <: AbstractLensModel} = new{L}(l)
end
LensManifold(l::AbstractLensModel) = LensManifold{typeof(l)}(_interior(l))

# Which coordinates are bounded to (0, 1): the double sphere's α.
@inline _unit_bounded(leaf) = map(_ -> false, _coefficients(leaf))
@inline _unit_bounded(::DoubleSphere) = (false, true)
@inline _logit_mask(l) = _flatmap(_unit_bounded, _leaves(l))
_has_logit(l) = any(_logit_mask(l))

# A boundary value — 0 is a natural default for α — sits at the logit's ±∞,
# where no step could move it: the manifold anchors δ inside the interval.
const _UNIT_MARGIN = 1.0e-6
_interior_unit(a) = clamp(a, oftype(a, _UNIT_MARGIN), one(a) - oftype(a, _UNIT_MARGIN))
function _interior(l::AbstractLensModel)
    _has_logit(l) || return l
    θ = lens_parameters(l)
    return _with_parameters(l, map((t, b) -> b ? _interior_unit(t) : t, θ, SVector(_logit_mask(l))))
end

minimal_manifold(l::AbstractLensModel) = LensManifold(l)
_ambient_coordinates(l::AbstractLensModel) = lens_parameters(l)
tangent_size(ch::LensManifold) = length(lens_parameters(ch.lens))
model(ch::LensManifold) = ch.lens
function retract(ch::LensManifold, v)
    θ0 = lens_parameters(ch.lens)
    _has_logit(ch.lens) || return _with_parameters(ch.lens, θ0 + v)
    θ = map((t, d, b) -> b ? logistic(logit(t) + d) : t + d, θ0, v, SVector(_logit_mask(ch.lens)))
    return _with_parameters(ch.lens, θ)
end
recenter(ch::LensManifold, v) = LensManifold(retract(ch, v))
function retract_jacobian(ch::LensManifold)
    n = tangent_size(ch)
    _has_logit(ch.lens) || return SMatrix{n, n, Float64}(LinearAlgebra.I)
    θ = lens_parameters(ch.lens)
    d = map((t, b) -> b ? Float64(t * (1 - t)) : 1.0, θ, SVector(_logit_mask(ch.lens)))
    return SMatrix{n, n, Float64}(LinearAlgebra.Diagonal(d))
end
# at coordinates v the slope is the logistic's there: a(v)·(1 − a(v))
function retract_jacobian(ch::LensManifold, v::StaticVector)
    _has_logit(ch.lens) || return retract_jacobian(ch)     # the chart is linear: constant differential
    n = tangent_size(ch)
    θ = lens_parameters(ch.lens)
    d = map((t, dv, b) -> b ? (a = logistic(logit(t) + dv); a * (1 - a)) : one(promote_type(typeof(t), typeof(dv))), θ, v, SVector(_logit_mask(ch.lens)))
    return SMatrix{n, n}(LinearAlgebra.Diagonal(d))
end
action_jacobian(ch::LensManifold, x) =
    _has_logit(ch.lens) ? transform_deriv_params(ch.lens, x) * retract_jacobian(ch) : transform_deriv_params(ch.lens, x)
function local_coordinates(ch::LensManifold, l::AbstractLensModel)
    _same_structure(ch.lens, l) || throw(
        ArgumentError("local_coordinates: a $(_structure_name(l)) is not on the manifold of $(_structure_name(ch.lens))")
    )
    θ, θ0 = lens_parameters(l), lens_parameters(ch.lens)
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

The lens manifold on the coefficients `names` ([`lens_parameter_names`](@ref)'
vocabulary) — a target's complexity as a manifold.
"""
function restrict(ch::LensManifold, names::Tuple{Symbol, Vararg{Symbol}})
    all_names = lens_parameter_names(ch.lens)
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
    n = length(fieldnames(X))
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
