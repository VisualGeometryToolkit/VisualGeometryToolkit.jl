# =============================================================================
# LensSpec: a lens family by name and term counts
# =============================================================================
#
# A model is specified by family name and term counts, `LensSpec(:division;
# coefficients = 3)`, not by a `lens_record` of zeros. The NAME TABLE below maps
# each name to a structure built from the existing leaves (the grammar of
# `lens_record` / `lens_from_record`, lens_chart.jl). A lens the table does not
# name is stored as its record.

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

_nokw(P) = _LensFamily((;), s -> Lens(P()))

# The name table. The reference of a family is the member that EQUALS ITS BASE
# PROJECTION: the polynomial families at all-zero coefficients are perspective
# (or equidistant), and a parametric projection at the parameters that reduce it
# to perspective: DoubleSphere(0, 0), Unified(0), ExtendedUnified(0, 1) (α = 0).
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
    :double_sphere => _LensFamily((;), s -> Lens(DoubleSphere(0.0, 0.0))),
    :unified => _LensFamily((;), s -> Lens(Unified(0.0))),
    :extended_unified => _LensFamily((;), s -> Lens(ExtendedUnified(0.0, 1.0))),
    :perspective => _nokw(Perspective),
    :equidistant => _nokw(Equidistant),
    :stereographic => _nokw(Stereographic),
    :equisolid => _nokw(Equisolid),
    :orthographic => _nokw(Orthographic),
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
    `:orthographic`, `:double_sphere`, `:unified`, `:extended_unified`: a
    projection with no distortion, no keywords.

The reference values of a family are those of the member that equals its base
projection: all-zero coefficients for the polynomial families, and for the
parametric projections the parameters that reduce them to perspective
(`DoubleSphere(0, 0)`, `Unified(0)`, `ExtendedUnified(0, 1)`).

`free` lists, by [`parameter_names`](@ref) of the lens, the coefficients a
calibrator moves; the default (`nothing`) is all of them.

The record form, `LensSpec(Dict("record" => lens_record(lens)))`, covers every
composition the table does not name, and is the only form with numbers: a lens
with non-zero starting or held coefficients. `LensSpec(lens)` is the inverse of
`lens(spec)`: the named form only when the lens's structure has a name AND its
coefficients are the reference values, else the record form, so no value is
ever dropped.

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
    function LensSpec(family::Symbol, structure::NamedTuple, free, record)
        (family === :record) == (record !== nothing) ||
            throw(ArgumentError("LensSpec: the family is :record exactly when a record is given; got :$family and record = $(repr(record))"))
        return new(family, structure, free, record)
    end
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

# `ref`: the lens of `spec`, when the caller has already built it.
function _validated(spec::LensSpec, ref = nothing)
    spec.free === nothing && return spec
    names = parameter_names(ref === nothing ? lens(spec) : ref)
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
    named = _named_structure(l)
    if named !== nothing
        family, structure = named
        ref = _reference_lens(family, structure)
        typeof(ref) == typeof(l) && parameter_vector(ref) == parameter_vector(l) &&
            return _validated(LensSpec(family, structure, _free_tuple(free), nothing), ref)
    end
    return _record_spec(lens_record(l), free)
end

# The one (family, structure keywords) a lens's type and arities name (nothing: no family has that shape).
_named_structure(::AbstractLensModel) = nothing
_named_structure(::LiftLens{<:Radial{Even, 0, Q}}) where {Q} = Q >= 1 ? (:division, (; coefficients = Q)) : nothing
function _named_structure(l::Lens)
    p, d = l.projection, l.distortion
    if d === nothing
        family = Symbol(_tag(typeof(p)))
        fam = get(_LENS_FAMILIES, family, nothing)
        return fam !== nothing && isempty(fam.keywords) ? (family, (;)) : nothing
    elseif d isa Radial{Even}
        n, m = length(d.k), length(d.d)
        p isa Equidistant && m == 0 && n >= 1 && return (:kannala_brandt, (; coefficients = n))
        p isa Perspective || return nothing
        m == 0 && n >= 1 && return (:brown_conrady, (; radial = n, tangential = false))
        m >= 1 && return (:rational, (; numerator = n, denominator = m))
    elseif d isa Additive && p isa Perspective && length(d.terms) == 2
        r, t = d.terms
        r isa Radial{Even} && isempty(r.d) && !isempty(r.k) && t isa Tangential &&
            return (:brown_conrady, (; radial = length(r.k), tangential = true))
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

# =============================================================================
# CameraModelSpec: what a calibration is for
# =============================================================================

"""
    GenericGridSpec(; grid_size = nothing, calib_area = nothing)

The grid of a central generic camera (one ray per grid point, no lens family):
`grid_size`, the `(columns, rows)` of the grid, and `calib_area`, the
`(x0, y0, x1, y1)` pixel rectangle the grid covers; `nothing` leaves either to
the calibrator.

Throws an `ArgumentError` for a `grid_size` that is not two positive integers
or a `calib_area` that is not four integers.
"""
struct GenericGridSpec
    grid_size::Union{Nothing, Tuple{Int, Int}}
    calib_area::Union{Nothing, NTuple{4, Int}}
end

function _int_tuple(what::Symbol, x, n::Int)
    (x isa Union{Tuple, AbstractVector} && length(x) == n && all(v -> v isa Integer && !(v isa Bool), x)) ||
        throw(ArgumentError("GenericGridSpec: $what is a list of $n integers; got $(repr(x))"))
    return ntuple(i -> Int(x[i]), n)
end

function GenericGridSpec(; grid_size = nothing, calib_area = nothing)
    gs = grid_size === nothing ? nothing : _int_tuple(:grid_size, grid_size, 2)
    gs === nothing || all(>(0), gs) ||
        throw(ArgumentError("GenericGridSpec: grid_size must be positive; got $(repr(grid_size))"))
    ca = calib_area === nothing ? nothing : _int_tuple(:calib_area, calib_area, 4)
    return GenericGridSpec(gs, ca)
end

"""
    CameraModelSpec(lens::LensSpec; intrinsics::Type{<:AbstractIntrinsics} = LogicalIntrinsics)
    CameraModelSpec(:central_generic; grid_size = nothing, calib_area = nothing)
    CameraModelSpec(table::AbstractDict)

What a calibration is for: an intrinsics kind (the type parameter `I`: metric
[`LogicalIntrinsics`](@ref), or the projective `ProjectiveIntrinsics` of a
partial calibration) and a [`LensSpec`](@ref) or a [`GenericGridSpec`](@ref)
(the field `model`). The generic form is always metric.

`toml_value(spec)` is the table `Dict("intrinsics" => "metric" | "projective",
"lens" => …, …)` (a `LensSpec` table plus `intrinsics`) or
`Dict("type" => "central_generic", "grid_size" => […], "calib_area" => […])`
with the unset keys left out; `CameraModelSpec(table)` reads it back, and
`intrinsics` defaults to `"metric"` when absent (a `central_generic` table may
also name it, as `"metric"` only). `==` compares the specification.

Throws an `ArgumentError` for an `intrinsics` other than `metric` or
`projective` (and `toml_value` for an intrinsics kind with neither name), for
a `central_generic` table whose `intrinsics` is not `metric`, for an unknown
`type` or key, and for whatever [`LensSpec`](@ref) or
[`GenericGridSpec`](@ref) refuses.
"""
struct CameraModelSpec{I, S}
    model::S
    function CameraModelSpec{I, S}(model::S) where {I, S <: Union{LensSpec, GenericGridSpec}}
        I isa Type && I <: AbstractIntrinsics ||
            throw(ArgumentError("CameraModelSpec: the intrinsics kind is a subtype of AbstractIntrinsics; got $(repr(I))"))
        S === GenericGridSpec && I !== LogicalIntrinsics &&
            throw(ArgumentError("CameraModelSpec: a generic grid is metric (LogicalIntrinsics); got $I"))
        return new{I, S}(model)
    end
end

CameraModelSpec(lens::LensSpec; intrinsics::Type{<:AbstractIntrinsics} = LogicalIntrinsics) =
    CameraModelSpec{intrinsics, LensSpec}(lens)

function CameraModelSpec(type::Symbol; grid_size = nothing, calib_area = nothing)
    type === :central_generic ||
        throw(ArgumentError("CameraModelSpec: unknown type :$type; known: central_generic (or a LensSpec)"))
    return CameraModelSpec{LogicalIntrinsics, GenericGridSpec}(GenericGridSpec(; grid_size, calib_area))
end

"""
    intrinsics_kind(spec::CameraModelSpec) -> Type{<:AbstractIntrinsics}

The intrinsics type `spec` calibrates: `LogicalIntrinsics` for a metric
calibration.
"""
intrinsics_kind(::CameraModelSpec{I}) where {I} = I

# The intrinsics kinds a spec names, and the TOML name of each (methods, not a
# string => type table: these names are not the tag rule's): read by the table
# reader, written by toml_value.
const _INTRINSICS_KINDS = (LogicalIntrinsics, ProjectiveIntrinsics)
_intrinsics_name(::Type{<:LogicalIntrinsics}) = "metric"
_intrinsics_name(::Type{<:ProjectiveIntrinsics}) = "projective"
_intrinsics_name(I::Type) =
    throw(ArgumentError("CameraModelSpec: the intrinsics kind $I has no TOML name; known: $(_intrinsics_names())"))

_intrinsics_names() = join(map(_intrinsics_name, _INTRINSICS_KINDS), ", ")

function _intrinsics_type(name)
    for I in _INTRINSICS_KINDS
        name == _intrinsics_name(I) && return I
    end
    throw(ArgumentError("CameraModelSpec: unknown intrinsics $(repr(name)); known: $(_intrinsics_names())"))
end

# The fields of a generic grid a table names; the set ones, name => value, in the order they are shown and written.
const _GRID_KEYS = String.(fieldnames(GenericGridSpec))
_grid_args(g::GenericGridSpec) =
    Pair{Symbol, Any}[k => getfield(g, k) for k in fieldnames(GenericGridSpec) if getfield(g, k) !== nothing]
_grid_text(g::GenericGridSpec) = join(("$k = $v" for (k, v) in _grid_args(g)), ", ")

function CameraModelSpec(table::AbstractDict)
    d = Dict{String, Any}(String(k) => v for (k, v) in table)
    if haskey(d, "type")
        d["type"] == "central_generic" ||
            throw(ArgumentError("CameraModelSpec: unknown type $(repr(d["type"])); known: central_generic"))
        # a generic grid is metric: `intrinsics = "metric"` beside it is redundant but true
        name = pop!(d, "intrinsics", "metric")
        name == "metric" ||
            throw(ArgumentError("CameraModelSpec: a central_generic grid is metric; got intrinsics = $(repr(name))"))
        extra = setdiff(keys(d), ("type", _GRID_KEYS...))
        isempty(extra) || throw(ArgumentError("CameraModelSpec: a central_generic table takes type, $(join(_GRID_KEYS, ", ")); got also $(sort(collect(extra)))"))
        return CameraModelSpec(:central_generic; (Symbol(k) => d[k] for k in _GRID_KEYS if haskey(d, k))...)
    end
    name = pop!(d, "intrinsics", "metric")
    return CameraModelSpec(LensSpec(d); intrinsics = _intrinsics_type(name))
end

function toml_value(spec::CameraModelSpec{I, LensSpec}) where {I}
    d = toml_value(spec.model)
    d["intrinsics"] = _intrinsics_name(I)
    return d
end

function toml_value(spec::CameraModelSpec{LogicalIntrinsics, GenericGridSpec})
    d = Dict{String, Any}("type" => "central_generic")
    for (k, v) in _grid_args(spec.model)
        d[String(k)] = Any[v...]
    end
    return d
end

Base.:(==)(a::GenericGridSpec, b::GenericGridSpec) = a.grid_size == b.grid_size && a.calib_area == b.calib_area
Base.hash(g::GenericGridSpec, h::UInt) = hash((g.grid_size, g.calib_area), hash(:GenericGridSpec, h))
Base.:(==)(a::CameraModelSpec{I}, b::CameraModelSpec{J}) where {I, J} = I === J && a.model == b.model
Base.hash(s::CameraModelSpec{I}, h::UInt) where {I} = hash((I, s.model), hash(:CameraModelSpec, h))

Base.show(io::IO, g::GenericGridSpec) = print(io, "GenericGridSpec(", _grid_text(g), ")")

function Base.show(io::IO, s::CameraModelSpec{I, LensSpec}) where {I}
    print(io, "CameraModelSpec(", s.model)
    I === LogicalIntrinsics || print(io, "; intrinsics = ", I)
    return print(io, ")")
end

function Base.show(io::IO, s::CameraModelSpec{LogicalIntrinsics, GenericGridSpec})
    a = _grid_text(s.model)
    return print(io, "CameraModelSpec(:central_generic", isempty(a) ? "" : "; " * a, ")")
end
