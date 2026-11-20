# =============================================================================
# A kind of another package — its table, its two file boundaries, its JSON Schema
# =============================================================================
#
# A package that owns a file kind (VisualRadiometryToolkitIO's vignetting,
# environment light and radial profile; PanoramaStitching's panorama) states
# the kind's shape as a table, as every kind here does (validate.jl), and reads
# and writes its files through the same two boundaries: the table is checked on
# the very bytes a writer puts on disk and before a reader uses a field, an
# earlier version reads through the owner's version steps (`migrate_schema`),
# and the published JSON Schema is generated from the table. This file is that
# hook, and nothing else of the validator is public:
#
#   object_spec, token_spec, integer_spec, real_spec, array_spec, named_spec,
#   checked_spec, if_key_spec, tagged_spec      the table's combinators
#   body_spec(T)                                the table of a body this package writes (a camera, a trajectory, a fit)
#   KindSpec(schema, required, optional; …)     the kind: its schema string and its table under the envelope
#   write_kind(path, kind, body; style, …)      the write boundary     read_kind(path, kind)   the read boundary
#   json_schema(kind)                           its JSON Schema        schema_versions(kind)   the versions read
#   migrate_file(T, kind, src, dst; …)          the file form of a version step, through the owner's `load` and `save`
#   migrate_body(T, body)                       an IO 0.9 body this package writes as its 0.10 one (an owner's step)
#
# The body's values go through the style as ever (`JSON.lower(::VgioStyle, x)`,
# `JSON.lift(::VgioStyle, T, record)`; the owner's methods for its own types).
# The envelope is this package's: an owner states its body alone.

# --- the combinators -----------------------------------------------------------

_spec_pairs(ps) = Pair{Symbol, _Spec}[Symbol(first(p)) => last(p)::_Spec for p in ps]

"""
    object_spec(required::Pair{Symbol}...; optional = ()) -> spec

The table of an object with EXACTLY these keys, each `name => spec`: a missing
required key and an unknown key are refused alike, by name (an unknown key is
tolerated only in a file of a higher minor version than the reader's).
`optional` lists the keys that may be absent. The combinators of a
[`KindSpec`](@ref) table are this one and [`token_spec`](@ref),
[`integer_spec`](@ref), [`real_spec`](@ref), [`array_spec`](@ref),
[`named_spec`](@ref), [`checked_spec`](@ref), [`if_key_spec`](@ref),
[`tagged_spec`](@ref) and [`body_spec`](@ref).
"""
object_spec(required::Pair{Symbol}...; optional = ()) = _Obj(_spec_pairs(required), _spec_pairs(optional))

"""
    token_spec(value) -> spec

The table of one exact value (a declaration token: a `type` tag, a stated
convention). JSON Schema: `const`.
"""
token_spec(value) = _Tok(value)

"""
    integer_spec(; positive = false) -> spec

The table of an integer (never a Bool), `positive`: greater than zero.
"""
integer_spec(; positive::Bool = false) = positive ? _PosInt() : _Int()

"""
    real_spec() -> spec

The table of a finite real number (a NaN or ±Infinity is refused by its field).
"""
real_spec() = _Real()

"""
    array_spec(element; length = nothing) -> spec

The table of an array whose every entry is `element` (a spec), of exactly
`length` entries when one is given.
"""
array_spec(element; length::Union{Nothing, Integer} = nothing) = _Vec(element::_Spec, length === nothing ? -1 : Int(length))

"""
    named_spec(ok, what::AbstractString) -> spec

The table of a string the predicate `ok` accepts, `what` saying in words what
it is (the refusal reads "`x` is not `what`"). A predicate that is a set
membership, an equality or a pattern is stated in the JSON Schema too:
`in(values)` as `enum`, `==(value)` as `const`, `Base.Fix1(occursin, regex)` as
`pattern`; any other is the schema's `description` alone.
"""
named_spec(ok, what::AbstractString) = _Named(ok, String(what))

"""
    checked_spec(spec, check; rule = "", schema = []) -> spec

`spec`, then the cross-field check `check(x)` of the value it accepted, which
returns `nothing` or the refusal's message. `rule` states the check in words
(the JSON Schema's `description` of the object) and `schema` the JSON Schema
keywords that state it where JSON Schema can (one subschema, a `Dict`, or a
vector of them: all hold).
"""
checked_spec(spec, check; rule::AbstractString = "", schema = Dict{String, Any}[]) = _CheckedSpec(spec::_Spec, check; rule, schema)

"""
    if_key_spec(key::Symbol, then, otherwise) -> spec

The table of an object of one of two shapes, told apart by whether it holds
`key`: `then` (a spec) if it does, `otherwise` if not. JSON Schema:
`if`/`then`/`else`.
"""
if_key_spec(key::Symbol, then, otherwise) = _IfKey(key, then::_Spec, otherwise::_Spec)

"""
    tagged_spec(what, alternatives::Pair{<:AbstractString}...; def = "", unknown = nothing) -> spec
    tagged_spec(build, what; def, unknown = nothing) -> spec

The table of a body told by its `type`: `alternatives` are `tag => spec`, the
spec of the whole body of that tag (its `type` key included, a
[`token_spec`](@ref) of the tag). `what` names the body in the refusals ("a
vignetting"). A `type` none of them has is refused listing the tags, or with
`unknown(tag)`'s message when `unknown` is given (a tag registry's: a retired
tag with its successor).

A body that holds a body of its own kind (a log field's spline) is built by the
second form: `build(self)` returns the alternatives, `self` being the spec
under construction. Such a spec states `def`, its name among the JSON Schema's
`\$defs`, where it is defined once and referenced; with no `def` the
alternatives are stated in place (`oneOf`).
"""
function tagged_spec(what::AbstractString, alternatives::Pair{<:AbstractString}...; def::AbstractString = "", unknown = nothing)
    return _ByType(String(what), Pair{String, _Spec}[String(first(p)) => last(p)::_Spec for p in alternatives], String(def), unknown)
end
function tagged_spec(build, what::AbstractString; def::AbstractString, unknown = nothing)
    isempty(def) && throw(ArgumentError("tagged_spec: a body that holds itself states `def`, its name among the JSON Schema's definitions"))
    s = _ByType(String(what), Pair{String, _Spec}[], String(def), unknown)
    for p in build(s)
        push!(s.alts, String(first(p)) => last(p)::_Spec)
    end
    return s
end

"""
    body_spec(T) -> spec

The table of a body VisualGeometryToolkitIO writes, for a kind of another
package that embeds it:

- `body_spec(BoundedCamera)`: the bounded camera body, a camera store's entry
  without its `name` ([`camera_json`](@ref)`(camera::BoundedCamera, frame; length_unit)`,
  [`camera_from_json`](@ref)`(BoundedCamera, body, frame; length_unit)`);
- `body_spec(Trajectory)`: the trajectory body, a `vgio/trajectory` file
  without its envelope (`JSON.lower(style, trajectory)`,
  `JSON.lift(style, Trajectory, body)`);
- `body_spec(FitResult)`: a fit's termination record
  ([`fit_record_json`](@ref), [`fit_record_from_json`](@ref));
- `body_spec(PixelFrame)`: a pair of pixel coordinates `[x, y]`, in the file's
  `units.pixel`.
"""
body_spec(::Type{BoundedCamera}) = _BOUNDED_CAMERA_BODY
body_spec(::Type{Trajectory}) = _TRAJECTORY_BODY
body_spec(::Type{FitResult}) = _FIT_RECORD
body_spec(::Type{PixelFrame}) = _XY

# --- the kind --------------------------------------------------------------------

"""
    KindSpec(schema, required, optional = []; package, check = nothing, rule = "", schema_keywords = [])

A file kind of another package: its `schema` string,
`"<owner>/<kind>/v<major>.<minor>"` (the version its writer writes), and the
table of its files — the envelope every vgio file has (`schema`, `generator`,
`units`, `conventions`), then the body's `required` and `optional` keys (each a
vector of `name => spec`, built by [`object_spec`](@ref) and the other
combinators), then the free-form `provenance`. `package` names the owning
package (the JSON Schema says whose kind it is). `check(record)` is a
cross-field check of the whole record (`nothing`, or the refusal's message),
`rule` that check in words and `schema_keywords` the JSON Schema keywords that
state it (as [`checked_spec`](@ref)'s `rule` and `schema`).

The owner writes its files with [`write_kind`](@ref) and reads them with
[`read_kind`](@ref); [`json_schema`](@ref)`(kind)` is the kind's JSON Schema
and [`schema_versions`](@ref)`(kind)` the versions it reads. Its earlier
versions are its [`migrate_schema`](@ref) steps.

```jldoctest
julia> kind = VgtkIO.KindSpec("toy/gains/v1.0", [:gains => VgtkIO.array_spec(VgtkIO.real_spec())]; package = "Toy");

julia> path = joinpath(mktempdir(), "gains.json");

julia> VgtkIO.write_kind(path, kind, (gains = [1.0, 0.5],))

julia> VgtkIO.read_kind(path, kind).gains == [1.0, 0.5]
true
```
"""
struct KindSpec
    schema::String
    package::String
    spec::_Spec
    function KindSpec(schema::AbstractString, required::AbstractVector{<:Pair{Symbol}}, optional::AbstractVector{<:Pair{Symbol}} = Pair{Symbol, _Spec}[];
                      package::AbstractString, check = nothing, rule::AbstractString = "", schema_keywords = Dict{String, Any}[])
        v = _split_owned(schema)
        (v !== nothing && occursin(r"/v\d+\.\d+$", schema)) ||
            throw(ArgumentError("KindSpec: a schema is \"<owner>/<kind>/v<major>.<minor>\" (lowercase snake_case, the minor stated); got $(repr(schema))"))
        (v.owner == "vgio" && haskey(_SCHEMA_VERSION, v.kind)) &&
            throw(ArgumentError("KindSpec: vgio/$(v.kind) is VisualGeometryToolkitIO's own kind"))
        o = _Obj(
            _spec_table(Pair{Symbol, _Spec}[:schema => _OwnedSchemaTok(String(schema)), :generator => _GENERATOR, :units => _UNITS_OBJECT,
                                            :conventions => _CONVENTIONS_OBJECT], _spec_pairs(required)),
            _spec_table(_spec_pairs(optional), _PROVENANCE),
        )
        return new(String(schema), String(package), check === nothing ? o : _CheckedSpec(o, check; rule, schema = schema_keywords))
    end
end

"""
    write_kind(path, kind::KindSpec, body::NamedTuple; style = VgioStyle(), provenance = (;))

Write `body` — the body of a `kind` file, without its envelope; its
VisualGeometryToolkitCore values (and the owner's own) left for `style` to
lower — as a file of `kind`: the envelope (`kind`'s `schema`, the `generator`,
the `units` object `style` declares and the fixed `conventions`), the body's
keys in their order, then `provenance` when it is not empty. The very bytes
that land on disk are checked against `kind`'s table first, so a writer cannot
produce a file its reader refuses (an `ArgumentError` naming the file, the key
and the problem; nothing is written). [`read_kind`](@ref) reads the file.
"""
function write_kind(path::AbstractString, kind::KindSpec, body::NamedTuple; style::VgioStyle = VgioStyle(), provenance::NamedTuple = (;))
    for k in (:schema, :generator, :units, :conventions, :provenance)
        haskey(body, k) && throw(ArgumentError("write_kind: `$(k)` is the envelope's; the body does not state it"))
    end
    record = with_provenance(merge((schema = kind.schema,), _envelope_tail(style), body), provenance)
    text = _json_text(record, style)          # serialized once: these bytes are validated and written
    _validate(kind.spec, _parse_vgio_text(text, path), path, "")
    write(path, text)
    return nothing
end

"""
    read_kind(path, kind::KindSpec) -> JSON.Object

Read a file of `kind` as its current record, checked against `kind`'s table
before any field is used: parsed (a `JSON.Object`, read by property access — the
form `JSON.lift(::VgioStyle, T, record)` takes; `VgioStyle(record)` is the
style its `units` declare). A file of an earlier version of the kind is read
through the owner's version steps ([`migrate_schema`](@ref), run by
[`migrate_record`](@ref)) in memory; a file of the kind's major at a higher
minor is read with the keys that minor added tolerated. Any other schema is an
`ArgumentError` naming the file, its schema and the one expected, and so is a
record the table refuses (naming the key and the problem).
"""
function read_kind(path::AbstractString, kind::KindSpec)
    d = _read_vgio_json(path)
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    s = get(d, :schema, nothing)
    is_current = Base.Fix2(_is_major_of, kind.schema)
    (!is_current(s) && _is_earlier_owned(s, kind.schema) && _has_record_migration(s)) &&
        (d = _migrated_on_load(d, path, kind.schema, is_current, s))
    s2 = get(d, :schema, nothing)
    is_current(s2) || _refuse_owned_schema(path, kind, s)
    _refuse_payload(path, something(_split_owned(kind.schema)).kind, kind.spec, d)
    minor = something(_split_owned(s2)).minor
    with(_TOLERATE_UNKNOWN => minor > something(_split_owned(kind.schema)).minor) do
        _validate(kind.spec, d, path, "")
    end
    return d
end
# Whether `s` names an earlier major version of the kind `schema` names.
function _is_earlier_owned(s, schema::AbstractString)
    a, b = _split_owned(s), something(_split_owned(schema))
    return a !== nothing && a.owner == b.owner && a.kind == b.kind && a.major < b.major
end
function _refuse_owned_schema(path, kind::KindSpec, s)
    _is_earlier_owned(s, kind.schema) && throw(ArgumentError(
        "$(path) is a retired $(s) file, which no migration reads ($(kind.package) reads $(join(schema_versions(kind), ", "))): " *
            "re-export it from its producer"))
    a, b = _split_owned(s), something(_split_owned(kind.schema))
    (a !== nothing && a.owner == b.owner && a.kind == b.kind) && throw(ArgumentError(
        "$(path): a $(s) file is of a later major version than this $(kind.package) reads ($(kind.schema)): update $(kind.package)"))
    throw(ArgumentError("$(path): expected schema $(kind.schema), got $(repr(s))"))
end

"""
    json_schema(kind::KindSpec; id_base = "https://visualgeometrytoolkit.github.io/schemas/") -> JSON.Object

The JSON Schema (draft 2020-12) of another package's kind, generated from its
table as this package's own are: its `\$id` is `id_base` followed by the kind's
schema string and `.json`. `JSON.json(json_schema(kind); pretty = 4)` is the
file an owner publishes.
"""
json_schema(kind::KindSpec; id_base::AbstractString = _JSON_SCHEMA_ID_BASE) =
    _json_schema_document(kind.spec, kind.schema, id_base, kind.package)

"""
    schema_versions(kind::KindSpec) -> Vector{String}

The schema versions the owner of `kind` reads, derived from its version steps
(the [`migrate_schema`](@ref) methods of a record of an earlier version of the
kind), in version order, then the kind's current one.
"""
function schema_versions(kind::KindSpec)
    b = something(_split_owned(kind.schema))
    old = String[]
    for m in methods(migrate_schema)
        sig = m.sig
        (sig isa DataType && length(sig.parameters) == 3) || continue
        V = sig.parameters[2]
        (V isa DataType && V <: Val && V.parameters[1] isa Symbol) || continue
        s = String(V.parameters[1])
        a = _split_owned(s)
        (a !== nothing && a.owner == b.owner && a.kind == b.kind && _has_record_migration(s)) && push!(old, s)
    end
    order(s) = (v = something(_split_owned(s)); (v.major, v.minor))
    sort!(unique!(old); by = s -> (order(s), -length(s)))
    return push!(unique!(order, old), kind.schema)
end

"""
    migrate_file(T, kind::KindSpec, src, dst; what = "the value", length_unit = true) -> NamedTuple

The file form of a version step ([`migrate_schema`](@ref)`(::Val{schema}, src,
dst)`) of a kind whose files are one value of type `T`, read by the owner's
[`load`](@ref)`(T, path)` and written by its [`save`](@ref)`(path, value)`: the
file at `src` is loaded (through the version steps), saved to `dst` with the
provenance the migrated record holds (and, with `length_unit`, in the length
unit it declares: a `save` that takes no `length_unit` states `false`), read
back and compared with the value loaded ([`same_value`](@ref)), its unit and
provenance with the record's. A difference is an `ArgumentError` naming `src`
and `what`. Returns `(; to = kind.schema)`, the summary
[`vgio_migrate`](@ref) reports.

    VisualGeometryToolkitIO.migrate_schema(::Val{Symbol("toy/gains/v1")}, src::AbstractString, dst::AbstractString) =
        VisualGeometryToolkitIO.migrate_file(Gains, GAINS_KIND, src, dst; what = "the gains", length_unit = false)
"""
function migrate_file(::Type{T}, kind::KindSpec, src::AbstractString, dst::AbstractString; what::AbstractString = "the value",
                      length_unit::Bool = true) where {T}
    # (the owner's `load` and `save` methods may be newer than this package's world)
    Base.invokelatest(_migrate_file, T, src, dst, what; length = length_unit)
    return (; to = kind.schema)
end

"""
    migrate_body(T, body::AbstractDict) -> Dict{String, Any}

A body VisualGeometryToolkitIO 0.9 wrote, embedded in another package's
record, as the body IO 0.10 writes — the part of that package's version step
([`migrate_schema`](@ref)) that is this package's to state, beside
[`migrate_envelope`](@ref):

- `migrate_body(BoundedCamera, body)`: a bounded camera body (its `width` and
  `height` as `image_size`, its pitch an object, its lens and lattice records);
  a body already in the 0.10 form comes back as it is;
- `migrate_body(Trajectory, body)`: a trajectory body (each frame's `t` as
  `frame_index`, the frame name `camera_optical` as `camera`).

`body` is the parsed record with `String` keys, as a step holds it.
"""
migrate_body(::Type{BoundedCamera}, body::AbstractDict) = _camera_body_v010(body)
migrate_body(::Type{Trajectory}, body::AbstractDict) = _frame_names_v010(_frames_v010(body))
