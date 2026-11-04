# =============================================================================
# LensSpec: a lens family by name and term counts
# =============================================================================
#
# A model is specified by family name and term counts, `LensSpec(:division;
# coefficients = 3)`, not by a `lens_record` of zeros. The NAME TABLE below maps
# each name to a structure built from the existing leaves (the grammar of
# `lens_record` / `lens_from_record`, lens_chart.jl); `_LEAVES` (lens_chart.jl)
# names the single-leaf projections. A lens the table does not name is stored as
# its record.

# A structure keyword: a term count of at least `min`, or a flag with a default.
struct _Count
    min::Int
end
struct _Flag
    default::Bool
end

# keywords: name => _Count | _Flag, in the order they are shown and written;
# build: the lens at the reference values, from the validated structure.
struct _LensFamily
    keywords::NamedTuple
    build::Function
end

_zeros(n) = ntuple(_ -> 0.0, n)

_projection_families() = Pair{Symbol, _LensFamily}[
    Symbol(tag) => _LensFamily((;), s -> Lens(P())) for (tag, P) in _LEAVES
        if P <: AbstractProjection && isconcretetype(P) && hasmethod(P, Tuple{})
]

# The name table. Parametric projections (`DoubleSphere`, `Unified`,
# `ExtendedUnified`) have no zero-argument constructor, so they take no entry yet.
const _LENS_FAMILIES = Dict{Symbol, _LensFamily}(
    :division => _LensFamily((; coefficients = _Count(1)), s -> Division(_zeros(s.coefficients)...)),
    :kannala_brandt => _LensFamily((; coefficients = _Count(1)), s -> KannalaBrandt(_zeros(s.coefficients)...)),
    :brown_conrady => _LensFamily(
        (; radial = _Count(1), tangential = _Flag(false)),
        s -> s.tangential ?
            Lens(Perspective(), Additive(Radial(_zeros(s.radial), ()), Tangential(0.0, 0.0))) :
            Lens(Perspective(), Radial(_zeros(s.radial), ())),
    ),
    :rational => _LensFamily(
        (; numerator = _Count(0), denominator = _Count(1)),
        s -> Lens(Perspective(), Radial(_zeros(s.numerator), _zeros(s.denominator))),
    ),
    _projection_families()...,
)

"""
    LensSpec(family::Symbol; free = nothing, structure...)
    LensSpec(record::AbstractDict)
    LensSpec(lens::AbstractLensModel)

A lens model specified by family name and term counts, not by a record of
zeros: the lens a calibrator starts from (see [`lens(::LensSpec)`](@ref)) and the
coefficients it moves ([`free_names`](@ref)).

The families and their structure keywords:

  - `:division`, `:kannala_brandt`: `coefficients` (an integer ≥ 1), the `d1…dn`
    of `Division`, the `k1…kn` of `KannalaBrandt`;
  - `:brown_conrady`: `radial` (≥ 1) and `tangential` (a `Bool`, default `false`),
    the `k1…kn`, `p1, p2` of OpenCV's perspective model;
  - `:rational`: `numerator` (≥ 0) and `denominator` (≥ 1), OpenCV's rational
    radial factor `k1…k(numerator)` over `d1…d(denominator)`;
  - `:perspective`, `:equidistant`, `:stereographic`, `:equisolid`,
    `:orthographic`: a lens with no distortion, no keywords.

`free` lists, by [`parameter_names`](@ref) of the lens, the coefficients a
calibrator moves; the default (`nothing`) is all of them.

The record form, `LensSpec(Dict("record" => lens_record(lens)))`, covers every
composition the table does not name, and is the only form with numbers: a lens
with non-zero starting or held coefficients. `LensSpec(lens)` is the inverse of
`lens(spec)`: the named form when the lens's structure has a name AND its
coefficients are the reference values (zeros), else the record form.

`toml_value(spec)` is `Dict("lens" => "division", "coefficients" => 3, …)` or
`Dict("lens" => Dict("record" => …))`, plus `"free"` when set;
`LensSpec(::AbstractDict)` reads it back (the keys of the table the spec sits
in). `==` compares the specification, not the lens.

Throws an `ArgumentError` for an unknown family (listing the known ones), a
keyword the family does not take or lacks, a count out of range, a `free` name
that is not a parameter name of the lens, and a record that is not a lens record.
"""
struct LensSpec
    family::Symbol                                          # :record for the record form
    structure::NamedTuple
    free::Union{Nothing, Tuple{Vararg{Symbol}}}
    record::Union{Nothing, Dict{String, Any}}
end

_known_families() = join(sort!(String.(collect(keys(_LENS_FAMILIES)))), ", ")

function _check_structure(family::Symbol, kw)
    fam = get(_LENS_FAMILIES, family, nothing)
    fam === nothing && throw(ArgumentError("LensSpec: unknown lens family :$family; known: $(_known_families()) (or a record)"))
    takes = join(keys(fam.keywords), ", ")
    for k in keys(kw)
        haskey(fam.keywords, k) || throw(
            ArgumentError(
                "LensSpec(:$family): unknown keyword $k; " *
                    (isempty(fam.keywords) ? "it takes none" : "it takes $takes")
            )
        )
    end
    vals = map(keys(fam.keywords), values(fam.keywords)) do name, spec
        if spec isa _Count
            haskey(kw, name) || throw(ArgumentError("LensSpec(:$family) needs the keyword $name (an integer ≥ $(spec.min))"))
            n = kw[name]
            (n isa Integer && !(n isa Bool) && n >= spec.min) ||
                throw(ArgumentError("LensSpec(:$family): $name must be an integer ≥ $(spec.min); got $(repr(n))"))
            Int(n)
        else
            b = get(kw, name, spec.default)
            b isa Bool || throw(ArgumentError("LensSpec(:$family): $name must be a Bool; got $(repr(b))"))
            b
        end
    end
    return NamedTuple{keys(fam.keywords)}(vals)
end

_reference_lens(family::Symbol, structure::NamedTuple) = _LENS_FAMILIES[family].build(structure)

function _free_tuple(free)
    free === nothing && return nothing
    (free isa Union{Tuple, AbstractVector} && all(x -> x isa Union{Symbol, AbstractString}, free)) ||
        throw(ArgumentError("LensSpec: free is a list of parameter names (symbols); got $(repr(free))"))
    return Tuple(Symbol(x) for x in free)
end

function _validated(spec::LensSpec)
    spec.free === nothing && return spec
    names = parameter_names(lens(spec))
    for n in spec.free
        n in names || throw(
            ArgumentError("LensSpec: free name :$n is not a parameter of the lens's chart; its parameter_names are $names")
        )
    end
    allunique(spec.free) || throw(ArgumentError("LensSpec: free names $(spec.free) repeat a name"))
    return spec
end

function LensSpec(family::Symbol; free = nothing, kwargs...)
    return _validated(LensSpec(family, _check_structure(family, NamedTuple(kwargs)), _free_tuple(free), nothing))
end

# The record is read and written back, so the stored form is canonical (and refused early when it is no lens).
_canonical_record(d::AbstractDict) = lens_record(lens_from_record(d))

_record_spec(record::AbstractDict, free) =
    _validated(LensSpec(:record, (;), _free_tuple(free), _canonical_record(record)))

function LensSpec(l::AbstractLensModel; free = nothing)
    for (family, fam) in _LENS_FAMILIES
        structure = _structure_of(family, l)
        structure === nothing && continue
        ref = fam.build(structure)
        typeof(ref) == typeof(l) && parameter_vector(ref) == parameter_vector(l) &&
            return _validated(LensSpec(family, structure, _free_tuple(free), nothing))
    end
    return _record_spec(lens_record(l), free)
end

# The structure keywords a lens would have under `family`, from its type and arities (nothing: not that shape).
_structure_of(::Symbol, ::AbstractLensModel) = nothing
_structure_of(family::Symbol, l::LiftLens{<:Radial{Even, 0, Q}}) where {Q} =
    family === :division && Q >= 1 ? (; coefficients = Q) : nothing
function _structure_of(family::Symbol, l::Lens)
    p, d = l.projection, l.distortion
    fam = get(_LENS_FAMILIES, family, nothing)
    fam === nothing && return nothing
    if d === nothing
        return isempty(fam.keywords) && Symbol(_tag(typeof(p))) === family ? (;) : nothing
    elseif d isa Radial{Even}
        n, m = length(d.k), length(d.d)
        p isa Equidistant && m == 0 && n >= 1 && return family === :kannala_brandt ? (; coefficients = n) : nothing
        p isa Perspective || return nothing
        m == 0 && n >= 1 && return family === :brown_conrady ? (; radial = n, tangential = false) : nothing
        m >= 1 && return family === :rational ? (; numerator = n, denominator = m) : nothing
    elseif d isa Additive && p isa Perspective && family === :brown_conrady && length(d.terms) == 2
        r, t = d.terms
        r isa Radial{Even} && isempty(r.d) && !isempty(r.k) && t isa Tangential && return (; radial = length(r.k), tangential = true)
    end
    return nothing
end

function LensSpec(d::AbstractDict)
    keys_ = Set(String(k) for k in keys(d))
    free = get(d, "free", nothing)
    if "lens" in keys_
        what = d["lens"]
        if what isa AbstractDict
            Set(String(k) for k in keys(what)) == Set(["record"]) ||
                throw(ArgumentError("LensSpec: a lens table has the one key \"record\"; got $(sort(collect(keys(what))))"))
            extra = setdiff(keys_, ("lens", "free"))
            isempty(extra) || throw(ArgumentError("LensSpec: a record lens has no structure keys; got $(sort(collect(extra)))"))
            return _record_spec(what["record"], free)
        end
        what isa AbstractString || throw(ArgumentError("LensSpec: \"lens\" is a family name or a { record = … } table; got $(repr(what))"))
        kw = (Symbol(k) => d[k] for k in keys(d) if String(k) ∉ ("lens", "free"))
        return LensSpec(Symbol(what); free, kw...)
    elseif "record" in keys_
        extra = setdiff(keys_, ("record", "free"))
        isempty(extra) || throw(ArgumentError("LensSpec: a record spec takes \"record\" and \"free\"; got also $(sort(collect(extra)))"))
        return _record_spec(d["record"], free)
    end
    throw(ArgumentError("LensSpec: a table needs \"lens\" (a family name, or { record = … }) or \"record\"; got keys $(sort(collect(keys_)))"))
end

"""
    lens(spec::LensSpec) -> AbstractLensModel

The lens of `spec` at its reference values: zero coefficients, and the
projection's own parameters; what a calibrator starts a regression or a solve
from. The record form gives the record's lens, its numbers kept.
"""
lens(spec::LensSpec) = spec.record === nothing ? _reference_lens(spec.family, spec.structure) : lens_from_record(Float64, spec.record)

"""
    free_names(spec::LensSpec) -> NTuple{N, Symbol}

The coefficients a calibrator moves, by [`parameter_names`](@ref) of
`lens(spec)`: `spec`'s `free` when it was given, else all of them.
"""
free_names(spec::LensSpec) = spec.free === nothing ? parameter_names(lens(spec)) : spec.free

function toml_value(spec::LensSpec)
    d = Dict{String, Any}()
    if spec.record === nothing
        d["lens"] = String(spec.family)
        for (k, v) in pairs(spec.structure)
            d[String(k)] = v
        end
    else
        d["lens"] = Dict{String, Any}("record" => deepcopy(spec.record))
    end
    spec.free === nothing || (d["free"] = Any[String(n) for n in spec.free])
    return d
end

Base.:(==)(a::LensSpec, b::LensSpec) = a.family == b.family && a.structure == b.structure && a.free == b.free && a.record == b.record
Base.hash(s::LensSpec, h::UInt) = hash((s.family, s.structure, s.free, s.record), hash(:LensSpec, h))

function Base.show(io::IO, spec::LensSpec)
    if spec.record === nothing
        print(io, "LensSpec(:", spec.family)
        args = String["$k = $v" for (k, v) in pairs(spec.structure)]
        spec.free === nothing || push!(args, "free = $(spec.free)")
        isempty(args) || print(io, "; ", join(args, ", "))
    else
        print(io, "LensSpec(record = {", join(sort(collect(keys(spec.record))), ", "), "}")
        spec.free === nothing || print(io, ", free = ", spec.free)
    end
    return print(io, ")")
end
