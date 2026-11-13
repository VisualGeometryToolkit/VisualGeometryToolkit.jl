# =============================================================================
# The published JSON Schemas — generated from the validator's spec tables
# =============================================================================
#
# Every vgio kind's file shape is a spec table (validate.jl), and the table is
# data: `json_schema(kind)` walks it and writes the same shape as a JSON Schema
# (draft 2020-12), so a tool outside Julia (Python's `jsonschema`, an editor's
# completion) checks a file against the shape the Julia reader checks — one
# source, never a second hand-kept description. Each combinator maps to its
# JSON Schema form:
#
#   _Obj          an object: `properties`, `required`, `additionalProperties: false`
#   _Tok          `const`;   _NonEmpty a string of `minLength` 1
#   _Int/_PosInt/_Real   `integer` (`minimum` 1) / `number` (JSON has no NaN or ±Infinity)
#   _Vec          an array of its element (`minItems` = `maxItems` = n where n is fixed)
#   _Positional   an array of named numbers (`prefixItems`, each with its field's name)
#   _OrNull       `anyOf` [null, the spec]
#   _Named        a string with its description, and the `enum`, `const` or `pattern`
#                 its predicate is: `in(set)`, `==(x)`, `occursin(regex)`, or a
#                 registered set VisualGeometryToolkitCore holds (units, pixel frames)
#   _RealMap      an object of numbers;  _Free any object
#   _Either       `oneOf` its keys (exactly one present) and `if`/`then` per shape
#   _IfKey        `if` (the key present) / `then` / `else`
#   _CheckedSpec  its spec, its rule as `description`, and the JSON Schemas its
#                 table states for the check (`allOf`) where JSON Schema can say it
#   _SchemaTok    the kind's schema string at its major version, any minor (`pattern`)
#   _Cov          `oneOf` the three storages (rows, {"diagonal"}, {"isotropic"})
#   _RowRuns, _LengthRoles, _VignettingBody   their structure, their order rules in words
#   _RegionSpec   `$ref` to `$defs/region` (recursive: an intersection's parts) or `$defs/region_leaf`
#
# The Julia validator stays the authority: a cross-field check JSON Schema
# cannot state (a lattice's node count, a covariance's symmetry) is the
# `description` of its object, and a file the schema accepts may still be
# refused by the reader for it. Never the other way: a file the validator
# accepts, the schema accepts (test/json_schema_tests.jl holds both to it).
#
# The generated files are committed under docs/src/schemas/ at the path of
# their `$id` (`vgio/<kind>/v<major>.<minor>.json`); a test checks they are
# the generator's output (`_write_json_schemas(dir)` regenerates them).

const _JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
const _JSON_SCHEMA_ID_BASE = "https://visualgeometrytoolkit.github.io/schemas/"

# The relative path of a kind's schema (its `$id` below the base): `vgio/<kind>/v<major>.<minor>.json`.
_json_schema_path(kind::AbstractString) = _schema(kind) * ".json"

"""
    json_schema(kind::AbstractString) -> JSON.Object

The JSON Schema (draft 2020-12) of a `vgio/<kind>` file at the version this
package writes, generated from the kind's validator table — the shape
[`validate`](@ref) and every reader check. Its `\$id` is
`https://visualgeometrytoolkit.github.io/schemas/vgio/<kind>/v<major>.<minor>.json`.
A cross-field check JSON Schema cannot state is the `description` of the
object it constrains, so the Julia validator may refuse a file the schema
accepts, never the reverse. An unknown kind is an `ArgumentError` listing
the kinds. `JSON.json(json_schema("rig"); pretty = 4)` is the published file.

```jldoctest
julia> VgtkIO.json_schema("transforms")["title"]
"vgio/transforms/v5.0"
```
"""
function json_schema(kind::AbstractString)
    spec = (_schema_version(kind); _spec(kind))     # an unknown kind: the ArgumentError listing the kinds
    defs = JSON.Object{String, Any}()
    body = _json_schema(spec, defs)
    about = "A $(_schema(kind)) file, as VisualGeometryToolkitIO writes and reads it; generated " *
            "from its validator (VisualGeometryToolkitIO.json_schema). The validator is the authority: a check stated only " *
            "in a `description` is the reader's. A file of a higher minor version of this major may hold keys this " *
            "version does not declare: validate it against its own version's schema."
    out = JSON.Object{String, Any}("\$schema" => _JSON_SCHEMA_DIALECT, "\$id" => _JSON_SCHEMA_ID_BASE * _json_schema_path(kind),
                                   "title" => _schema(kind), "description" => about)
    for (k, v) in body
        out[k] = k == "description" ? about * " " * v : v
    end
    isempty(defs) || (out["\$defs"] = defs)
    return _ordered_json(out)
end

# A JSON Schema built here (`JSON.Object`s, in the order they are written) with
# the `Dict`s the spec tables state in sorted key order, so a schema's text is
# the same on every run.
_ordered_json(x::JSON.Object) = JSON.Object{String, Any}(String(k) => _ordered_json(v) for (k, v) in x)
_ordered_json(x::AbstractDict) = JSON.Object{String, Any}(String(k) => _ordered_json(x[k]) for k in sort!(collect(keys(x)); by = string))
_ordered_json(x::AbstractVector) = Any[_ordered_json(v) for v in x]
_ordered_json(x::Tuple) = Any[_ordered_json(v) for v in x]
_ordered_json(x) = x

_jobj(pairs::Pair...) = JSON.Object{String, Any}(pairs...)

function _json_schema(s::_Obj, defs)
    props = JSON.Object{String, Any}()
    for (k, sk) in Iterators.flatten((s.required, s.optional))
        props[String(k)] = _json_schema(sk, defs)
    end
    o = _jobj("type" => "object", "properties" => props)
    isempty(s.required) || (o["required"] = String[String(k) for (k, _) in s.required])
    o["additionalProperties"] = false
    return o
end
_json_schema(::_NonEmpty, defs) = _jobj("type" => "string", "minLength" => 1)
_json_schema(s::_Tok, defs) = _jobj("const" => s.value)
_json_schema(::_Int, defs) = _jobj("type" => "integer")
_json_schema(::_PosInt, defs) = _jobj("type" => "integer", "minimum" => 1)
_json_schema(::_Real, defs) = _jobj("type" => "number")
function _json_schema(s::_Vec, defs)
    o = _jobj("type" => "array", "items" => _json_schema(s.elem, defs))
    s.n >= 0 && (o["minItems"] = s.n; o["maxItems"] = s.n)
    return o
end
_json_schema(s::_Positional, defs) = _jobj(
    "type" => "array",
    "description" => "[" * join(first.(s.fields), ", ") * "]",
    "prefixItems" => [_jobj("title" => name, "description" => what, "type" => "number") for (name, what) in s.fields],
    "items" => false, "minItems" => length(s.fields),
)
_json_schema(s::_OrNull, defs) = _jobj("anyOf" => Any[_jobj("type" => "null"), _json_schema(s.inner, defs)])
function _json_schema(s::_Named, defs)
    o = _jobj("type" => "string", "description" => s.what)
    facet = _named_schema(s.ok)
    facet === nothing || merge!(o, facet)
    return o
end
_json_schema(::_RealMap, defs) = _jobj("type" => "object", "additionalProperties" => _jobj("type" => "number"))
_json_schema(::_Free, defs) = _jobj("type" => "object")
_json_schema(s::_Either, defs) = _jobj(
    "type" => "object",
    "oneOf" => [_jobj("required" => [String(k)]) for (k, _) in s.alts],
    "allOf" => [_jobj("if" => _jobj("required" => [String(k)]), "then" => _json_schema(sk, defs)) for (k, sk) in s.alts],
)
_json_schema(s::_IfKey, defs) = _jobj(
    "if" => _jobj("required" => [String(s.key)]), "then" => _json_schema(s.then, defs), "else" => _json_schema(s.otherwise, defs),
)
function _json_schema(s::_CheckedSpec, defs)
    o = _json_schema(s.inner, defs)
    isempty(s.rule) || (o["description"] = haskey(o, "description") ? "$(o["description"]) $(s.rule)" : s.rule)
    isempty(s.schema) || (o["allOf"] = vcat(Any[], get(o, "allOf", Any[]), s.schema))
    return o
end
function _json_schema(s::_SchemaTok, defs)
    M = first(_schema_version(s.kind))
    return _jobj("type" => "string", "description" => "$(_schema(s.kind)), or the same major version at any minor",
                 "pattern" => "^vgio/$(s.kind)/v$(M)(\\.[0-9]+)?\$")
end
function _json_schema(s::_Cov, defs)
    n = s.d > 0 ? s.d : nothing
    vec(elem) = (o = _jobj("type" => "array", "items" => elem); n === nothing || (o["minItems"] = n; o["maxItems"] = n); o)
    return _jobj(
        "description" => "a covariance in the storage it holds: $(n === nothing ? "n × n" : "$(n) × $(n)") rows (symmetric), " *
                         "{\"diagonal\": [its diagonal]} or {\"isotropic\": λ} (λ·I); in the value's tangent order and the file's units",
        "oneOf" => Any[
            vec(vec(_jobj("type" => "number"))),
            _jobj("type" => "object", "properties" => _jobj("diagonal" => vec(_jobj("type" => "number"))), "required" => ["diagonal"],
                  "additionalProperties" => false),
            _jobj("type" => "object", "properties" => _jobj("isotropic" => _jobj("type" => "number")), "required" => ["isotropic"],
                  "additionalProperties" => false),
        ],
    )
end
function _json_schema(::_RowRuns, defs)
    index = _jobj("type" => "integer", "minimum" => _INDEX_BASE)
    run = _jobj("type" => "array", "prefixItems" => [index, index], "items" => false, "minItems" => 2)
    return _jobj(
        "type" => "array",
        "description" => "the present nodes as runs per row, [[row, [[col_lo, col_hi], …]], …], 1-based: each row once, rows " *
                         "increasing; a row's runs nonempty, col_lo ≤ col_hi, increasing and apart (a gap of at least one node)",
        "items" => _jobj("type" => "array", "prefixItems" => [index, _jobj("type" => "array", "items" => run, "minItems" => 1)],
                         "items" => false, "minItems" => 2),
    )
end
_json_schema(::_LengthRoles, defs) = _jobj(
    "type" => "object",
    "description" => "length role → its length unit, for the fields whose lengths are not in `length`; \"chart\": a chart role, its numbers a chart's (plain)",
    "propertyNames" => _jobj("pattern" => _LENGTH_ROLE_RE.pattern, "not" => _jobj("enum" => collect(_RESERVED_ROLES))),
    "additionalProperties" => _jobj("anyOf" => Any[_json_schema(_LENGTH_UNIT, defs), _jobj("const" => _CHART_ROLE)]),
)
# A vignetting body names its kind by `type` (VGC's VIGNETTING_TAGS), and a log body holds a spline
# body: one definition, referenced (its own recursion included).
function _json_schema(::_VignettingBody, defs)
    if !haskey(defs, "vignetting")
        defs["vignetting"] = true                                  # a placeholder while the bodies refer to it
        defs["vignetting"] = _jobj("oneOf" => Any[_json_schema(_vignetting_body(VIGNETTING_TAGS[t]), defs) for t in keys(VIGNETTING_TAGS)])
    end
    return _jobj("\$ref" => "#/\$defs/vignetting")
end
# THE region body (core_json.jl): its leaves one definition (`region_leaf`, a camera's lens coverage), any region
# another that refers to them and, for an intersection's two parts, to itself.
function _json_schema(s::_RegionSpec, defs)
    haskey(defs, "region_leaf") ||
        (defs["region_leaf"] = _jobj("oneOf" => Any[_region_leaf_schema(k, defs) for k in keys(_REGION_LEAVES)]))
    s.composite || return _jobj("\$ref" => "#/\$defs/region_leaf")
    if !haskey(defs, "region")
        defs["region"] = true                                      # a placeholder while an intersection refers to it
        defs["region"] = _jobj(
            "description" => "a region in pixels of units.pixel: a leaf, the intersection of two regions (in order) or the whole plane",
            "oneOf" => Any[
                _jobj("\$ref" => "#/\$defs/region_leaf"),
                _region_object(:intersection, _jobj("type" => "array", "items" => _jobj("\$ref" => "#/\$defs/region"),
                                                    "minItems" => 2, "maxItems" => 2)),
                _region_object(:whole_plane, _json_schema(_Obj(), defs)),
            ],
        )
    end
    return _jobj("\$ref" => "#/\$defs/region")
end
# A region of one tag: `{"<tag>": body}`; a leaf's body its positional record with its extents positive.
_region_object(k::Symbol, body) =
    _jobj("type" => "object", "properties" => _jobj(String(k) => body), "required" => [String(k)], "additionalProperties" => false)
function _region_leaf_schema(k::Symbol, defs)
    leaf = _REGION_LEAVES[k]
    body = _json_schema(leaf.spec, defs)
    foreach(i -> body["prefixItems"][i]["exclusiveMinimum"] = 0, leaf.extents)
    k === :ellipse && (body["description"] *= " (a ≥ b: the major semi-axis first)")
    return _region_object(k, body)
end

_json_schema(::_Bool, defs) = _jobj("type" => "boolean")
# A detection file declares its records' shape in its header (`detection`: the frame type, the attributes, the
# covariance storage): the record schema of each frame type, selected by the header's `frame` (`if`/`then`), its
# `attributes` an object of the declared types' values (which names, the validator checks against the header).
function _json_schema(s::_DetectionSpec, defs)
    o = _json_schema(s.outer, defs)
    attribute = _jobj("anyOf" => Any[_jobj("type" => "number"), _jobj("type" => "boolean"), _jobj("type" => "string"), _json_schema(_XY, defs)])
    cases = map(collect(String, keys(_FRAME_TAGS))) do tag
        rec = _json_schema(_detection_record_spec(Dict{Symbol, Any}(:frame => tag, :attributes => Dict{String, Any}(), :covariance => true)), defs)
        rec["properties"]["attributes"] = _jobj("type" => "object", "additionalProperties" => attribute)
        _jobj("if" => _jobj("properties" => _jobj("detection" => _jobj("properties" => _jobj("frame" => _jobj("const" => tag))))),
              "then" => _jobj("properties" => _jobj("images" => _jobj("items" => _jobj("properties" => _jobj("detections" => _jobj("items" => rec)))))))
    end
    o["allOf"] = vcat(Any[], get(o, "allOf", Any[]), cases)
    rule = "Each detection is the record of the header's `frame` type: its `center`, its shape parameters, its `attributes` " *
           "(exactly the header's, each of its declared type) and, iff the header declares a `covariance`, a covariance " *
           "over the frame's parameters in the declared storage; a detection without one carries the attribute `covariance_status`."
    o["description"] = haskey(o, "description") ? "$(o["description"]) $(rule)" : rule
    return o
end

# The JSON Schema keywords a `_Named` predicate IS, or `nothing` (its description says it).
_named_schema(::Function) = nothing
_named_schema(f::Base.Fix2{typeof(in)}) = Dict("enum" => collect(f.x))
_named_schema(f::Base.Fix2{typeof(==)}) = Dict("const" => f.x)
_named_schema(f::Base.Fix1{typeof(occursin), Regex}) = Dict("pattern" => f.x.pattern)
# the registered sets VisualGeometryToolkitCore holds: its pixel frames, and its units of a dimension
_named_schema(::typeof(_is_pixel_frame_name)) = Dict("enum" => sort!([String(o) for o in keys(VgtkCore.INTRINSICS_COORDINATE_OFFSET)]))
_named_schema(::typeof(_is_length_name)) = Dict("enum" => _unit_names(_is_length_name))
_named_schema(::typeof(_is_density_name)) = Dict("enum" => _unit_names(_is_density_name))
_named_schema(::typeof(_is_sidecar_length_name)) = Dict("enum" => _unit_names(_is_sidecar_length_name))
_unit_names(ok) = (n = sort!([String(k) for k in keys(VgtkCore.StackUnits.UNITS) if ok(String(k))]);
    ok(_UNKNOWN_SCALE_NAME) ? push!(n, _UNKNOWN_SCALE_NAME) : n)

# Write every kind's schema under `dir` at its `$id` path (docs/src/schemas, the published files).
function _write_json_schemas(dir::AbstractString)
    for kind in sort!(collect(keys(_SCHEMA_VERSION)))
        path = joinpath(dir, _json_schema_path(kind))
        mkpath(dirname(path))
        _write_json(path, json_schema(kind))
    end
    return dir
end
