# =============================================================================
# TOML config skeleton — the generic half of a "one authority per value" loader
# =============================================================================
#
# Every package in the family with a configuration of record (RobustVisual-
# Geometry's `[ransac]`/`[optimization_strategy]`, BlobBoards' `[find_boards]`)
# needs the same mechanisms, and until 2026-09 each carried its own copy —
# ~180 lines forked and already drifting (one `_present_kwargs` took a
# `fallback` table, the other a key→field renaming; one `_toml_key` dispatched
# on the type, the other on a global table). The mechanisms, all here:
#
#   * `present_kwargs` — only the keys PRESENT in a table become keywords, so
#                         an absent key falls through to the struct's default.
#   * `check_keys`     — unknown keys are REFUSED, never ignored; retired keys
#                         (removed knobs that immutable sidecars still carry)
#                         load with a warning.
#   * `toml_table`     — struct → Dict derived from `fieldnames`, so a new
#                         field appears in the file with no edit anywhere. The
#                         per-type hooks (`toml_key`, `toml_skip_fields`,
#                         `toml_field`) are the ONLY places a package
#                         specialises, and they are generic functions the
#                         package extends for its own types.
#   * `allowed_keys`   — the reader's key list, derived from the SAME walk the
#                         writer uses, so a key cannot be writable without
#                         being readable.
#   * `toml_default`   — dotted-path lookup into a parsed defaults file that
#                         throws on a missing key: a keyword default that is
#                         not in the file would be a silent second source of
#                         truth.
#   * `parse_defaults` — parse a defaults file and record it as a precompilation
#                         dependency of the calling package.
#   * `write_toml`     — the one sorted writer behind every `write_config`.
#   * `migrate_config` — a package's versioned converter (`ConfigVersions`): a
#                         stored file of an earlier version read at the current
#                         one, so code sees only current table names.
#
# What stays in each package: its `configs/default.toml`, the
# `include_dependency` + parsed constant, its tag registries (TOML string →
# concrete type) and the hook methods for its own types.
# =============================================================================

using TOML: TOML

# The Greek field names of the strategy structs and the sensor components (`σ`, `σ_rel`) and their ASCII TOML keys: ONE
# table, both directions, so the loader and the writer cannot disagree.
const _KEY_ASCII = Dict{Symbol, String}(
    :μ => "mu", :η => "eta", :γ => "gamma", :ν => "nu", :α => "alpha",
    :α_min => "alpha_min", :σ => "sigma", :σ_rel => "sigma_rel",
)
const _KEY_FIELD = Dict{String, Symbol}(key => field for (field, key) in _KEY_ASCII)

"""
    toml_field_name(key::AbstractString) -> Symbol

The field name a TOML key spells: the inverse of [`toml_key`](@ref)'s Greek
spelling (`"mu"` → `:μ`), `Symbol(key)` for every other key. A loader whose
structs carry the Greek fields hands it to [`present_kwargs`](@ref) as
`field_of = toml_field_name`; it is not that function's default, since a struct
whose fields are themselves ASCII (`nu`, `gamma`, `alpha`) must not be renamed.
"""
toml_field_name(key::AbstractString) = get(_KEY_FIELD, key, Symbol(key))

"""
    present_kwargs(d::AbstractDict, spec::Pair{String}...; fallback = nothing, field_of = Symbol)
        -> Vector{Pair{Symbol, Any}}

Typed per-key overrides. For each `"key" => convert` in `spec`, emit
`field_of("key") => convert(d["key"])` when `d` has the key, else the same from
`fallback` when THAT has it, else nothing at all — so an absent key falls
through to the struct's own default rather than to a parallel literal here.
`field_of` maps a TOML spelling to a field name; the default `Symbol` keeps the
key as the field. A loader of structs with Greek fields (`:μ` read from an
ASCII `"mu"`) supplies `field_of = `[`toml_field_name`](@ref).
"""
function present_kwargs(d::AbstractDict, spec::Pair{String}...; fallback = nothing, field_of = Symbol)
    kw = Pair{Symbol, Any}[]
    for (k, f) in spec
        if haskey(d, k)
            push!(kw, field_of(k) => f(d[k]))
        elseif fallback !== nothing && haskey(fallback, k)
            push!(kw, field_of(k) => f(fallback[k]))
        end
    end
    return kw
end

"""
    check_keys(d::AbstractDict, allowed, section::AbstractString; retired = (), renamed = ())

Refuse unknown keys in table `d` with an `ArgumentError` naming the section,
the offending keys and the allowed set. A typo'd knob that silently does
nothing is the failure a config loader exists to prevent — a config that looks
like it set `max_trials` but spelled it `max_trial` would otherwise run
at the default and report success.

`retired` is a collection of exact `(section, key)` pairs: knobs removed
because nothing read them, which recorded run sidecars (immutable provenance)
legitimately still contain. Those load with a one-time warning; every other
unknown key throws. Retirement is section-scoped on purpose — the same key in
another table is a typo and must still fail.

`renamed` is a collection of `(section, key) => new_key` pairs: keys whose knob
lives on under a new name. Each is refused with an `ArgumentError` naming its
new key (there is no alias: the old spelling never loads), before the unknown
keys are checked; section-scoped like `retired`.
"""
function check_keys(d::AbstractDict, allowed, section::AbstractString; retired = (), renamed = ())
    for k in keys(d), ((sec, old), new) in renamed
        sec == section && old == k && throw(ArgumentError("[$section] `$k` was renamed `$new` (no alias)"))
    end
    unknown = setdiff(collect(String, keys(d)), allowed)
    ret = [k for k in unknown if (section, k) in retired]
    for k in ret
        @warn "[$section] key `$k` is retired (it was never read) — ignored" _id = Symbol(section, :_, k) maxlog = 1
    end
    unknown = setdiff(unknown, ret)
    isempty(unknown) || throw(
        ArgumentError(
            "[$section] has unknown key(s) $(sort(unknown)); allowed: $(sort(collect(String, allowed)))"
        )
    )
    return nothing
end

"""
    required_key(d, key, where) -> value

`d[key]`, or an `ArgumentError` naming `where` (the file and the table) and
the missing key, rather than a `KeyError` from deep inside a parsed record.
The one reader of a key a record or a config must hold.
"""
function required_key(d, key, where::AbstractString)
    haskey(d, key) || throw(ArgumentError("$(where): missing `$(key)` (a required key)"))
    return d[key]
end

"""
    is_finite_real(x) -> Bool

Whether `x` is a number a record may hold: a `Real`, not a `Bool`, finite, and
finite as a `Float64`.
"""
is_finite_real(x) = x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x))

# -----------------------------------------------------------------------------
# struct ⇄ table hooks — generic functions a package extends for its own types
# -----------------------------------------------------------------------------

"""
    toml_skip_fields(::Type) -> Tuple{Vararg{Symbol}}

Fields the walker must not emit for this type: data supplied at load (a
camera), a property derived from the inputs (a polarity), a scale interface
that is not a config key. Default: none. Declared per type so the reason lives
next to the type it applies to.
"""
toml_skip_fields(::Type) = ()

"""
    toml_key(::Type, f::Symbol) -> String
    toml_key(c, f::Symbol) -> String

The TOML spelling of field `f` of this type; default `String(f)`, except that
a Greek field name is spelled ASCII (`:μ` → `"mu"`, `:α_min` → `"alpha_min"`, the
table behind [`toml_field_name`](@ref)) so a config is typeable everywhere.
Dispatched on the TYPE so the reader's allowed-key list and the writer share
one mapping.
"""
toml_key(::Type, f::Symbol) = get(_KEY_ASCII, f, String(f))
toml_key(c, f::Symbol) = toml_key(typeof(c), f)

"""
    toml_value(x)

A leaf in TOML-representable form: numbers, booleans and strings are returned unchanged;
symbols become strings; vectors, tuples and dicts recurse. Anything else is
handed to the TOML writer as is (which refuses what it cannot represent).
"""
toml_value(x::Union{Real, AbstractString}) = x
toml_value(x::Symbol) = String(x)
toml_value(x::Union{AbstractVector, Tuple}) = Any[toml_value(v) for v in x]
toml_value(x::AbstractDict) = Dict{String, Any}(String(k) => toml_value(v) for (k, v) in x)
toml_value(x) = x

"""
    toml_field(c, ::Val{F})

The TOML value of field `F` of `c`; default `toml_value(getfield(c, F))`.
Override for the non-mechanical cases only — a nested config struct
(`toml_table(getfield(c, F))`), a tagged union (`type = ...` discriminator), a
sentinel (`typemax(Int)` written as `"unbounded"`).
"""
toml_field(c, ::Val{F}) where {F} = toml_value(getfield(c, F))

"""
    toml_table(c) -> Dict{String, Any}

`c` as a TOML table: `fieldnames(typeof(c))` minus [`toml_skip_fields`](@ref),
keyed by [`toml_key`](@ref), valued by [`toml_field`](@ref). Julia's TOML
stdlib serialises Dicts only, so the walk is ours — but it walks `fieldnames`,
which means a new field appears in the file with no edit and cannot be
forgotten.
"""
function toml_table(c)
    T = typeof(c)
    skip = toml_skip_fields(T)
    return Dict{String, Any}(
        toml_key(T, f) => toml_field(c, Val(f)) for f in fieldnames(T) if !(f in skip)
    )
end

"""
    allowed_keys(::Type{T}, extra...) -> Vector{String}

The keys a `[T]` table may carry: the same field walk [`toml_table`](@ref)
emits, plus `extra` (a `type` discriminator, say). Derived, never hand-listed,
so a field is readable and writable together — emitting a key the reader then
refuses is exactly the asymmetry this avoids.
"""
allowed_keys(::Type{T}, extra...) where {T} = String[
    [toml_key(T, f) for f in fieldnames(T) if !(f in toml_skip_fields(T))];
    String.(collect(extra))
]

"""
    read_table(::Type{T}, d::AbstractDict; skip = (), from_toml) -> Vector{Pair{Symbol, Any}}

The reader's half of [`toml_table`](@ref): the keyword overrides of a `[T]`
table `d`, one `f => from_toml(T, Val(f), d[toml_key(T, f)])` for every field
`f` of `T` whose key is PRESENT in `d`, in field order — the same
`fieldnames` / [`toml_skip_fields`](@ref) / [`toml_key`](@ref) walk the writer
emits, minus the fields in `skip` the caller reads itself. An absent key
yields nothing, so the field falls through to the struct's own default:
`T(; read_table(T, d)...)`. `from_toml(T, Val(f), v)` converts one value
(a package's per-field hook); the default is [`convert_field`](@ref) to the
field's type. Unknown keys are [`check_keys`](@ref)' job, with [`allowed_keys`](@ref).
"""
function read_table(::Type{T}, d::AbstractDict; skip = (), from_toml = _convert_from_toml) where {T}
    kw = Pair{Symbol, Any}[]
    for f in fieldnames(T)
        (f in skip || f in toml_skip_fields(T)) && continue
        k = toml_key(T, f)
        haskey(d, k) && push!(kw, f => from_toml(T, Val(f), d[k]))
    end
    return kw
end

"""
    construct_from_table(::Type{T}, d::AbstractDict; skip = (), section = string(nameof(T)), from_toml) -> T

THE construction of a `T` from its TOML table `d`, by keyword: `T(; f => from_toml(T, Val(f), v)…)` for every key of `d`
not in `skip` (a `type` discriminator, say), each key's field named by [`toml_field_name`](@ref) (the ASCII spelling of
a Greek field read back) — never by field order: `T` takes its fields as keywords (`@kwdef`, or a keyword constructor
of its own), and a field the table omits falls to that constructor's default. `from_toml` converts one value (the
default: [`convert_field`](@ref) to the field's type, the key named). Unknown keys are [`check_keys`](@ref)' job (with
[`allowed_keys`](@ref)), before this call; a missing required key and a value the constructor refuses are an
`ArgumentError` naming `section`.
"""
function construct_from_table(::Type{T}, d::AbstractDict; skip = (), section::AbstractString = string(nameof(T)),
        from_toml = _convert_from_toml) where {T}
    kws = Pair{Symbol, Any}[]
    for (k, v) in d
        k in skip && continue
        f = toml_field_name(String(k))
        push!(kws, f => from_toml(T, Val(f), v))
    end
    try
        return T(; kws...)
    catch e
        e isa UndefKeywordError && throw(ArgumentError("[$section] needs `$(toml_key(T, e.var))`"))
        e isa ArgumentError && throw(ArgumentError("[$section]: $(e.msg)"))
        rethrow()
    end
end

# read_table's default converter: the value in its field's type, the key named.
_convert_from_toml(::Type{T}, ::Val{F}, v) where {T, F} = convert_field(fieldtype(T, F), v, toml_key(T, F))

"""
    convert_field(::Type{T}, v)
    convert_field(::Type{T}, v, key)

A TOML value → the field type (`Float64(v)`, `Int(v)`, `Bool(v)`, `Symbol(v)`):
the one converter behind a typed [`present_kwargs`](@ref) spec.

The keyed form checks the value's shape and names `key` in every error (an
`ArgumentError`): a table where a value belongs is refused; a `Tuple` field
takes an array of exactly its length (any length for a `Vararg` tuple such as
`Tuple{Vararg{Float64}}`), converted element by element; a `Vector` field an
array, converted element-wise; any other array field (an `SVector`) an array,
converted by its constructor as the unkeyed form does; any other value for a
tuple or array field (a string would convert character by character) is
refused; a value that does not convert (`MethodError`, `InexactError`,
`ArgumentError`, `DimensionMismatch`) is reported as not a `T`.
[`read_table`](@ref)'s default converter is the keyed form.
"""
convert_field(::Type{T}, v) where {T} = T(v)
function convert_field(::Type{S}, v, key) where {S}
    v isa AbstractDict && throw(ArgumentError("`$key` is a value, not a table"))
    return _keyed_convert(S, v, key)
end
function convert_field(::Type{S}, v::AbstractVector, key) where {S <: Tuple}
    n = _tuple_length(S)
    n === nothing || length(v) == n || throw(ArgumentError("`$key` needs $n values, got $(length(v))"))
    return _keyed_convert(S, Tuple(convert_field(fieldtype(S, i), x, key) for (i, x) in enumerate(v)), key)
end
convert_field(::Type{Vector{E}}, v::AbstractVector, key) where {E} = E[convert_field(E, x, key) for x in v]
convert_field(::Type{S}, v::AbstractVector, key) where {S <: AbstractVector} = _keyed_convert(S, v, key)
function convert_field(::Type{S}, v, key) where {S <: Union{Tuple, AbstractVector}}
    n = S <: Tuple ? _tuple_length(S) : nothing
    throw(ArgumentError("`$key` = $(repr(v)) is not a $S (an array$(n === nothing ? "" : " of $n values"))"))
end

# The unkeyed conversion, its failure reported with the key.
function _keyed_convert(::Type{S}, v, key) where {S}
    try
        return convert_field(S, v)
    catch err
        err isa Union{MethodError, InexactError, ArgumentError, DimensionMismatch} || rethrow()
        throw(ArgumentError("`$key` = $(repr(v)) is not a $S"))
    end
end

# A tuple type's length; `nothing` for a `Vararg` tuple, which takes any length.
_tuple_length(::Type{S}) where {S <: Tuple} = Base.isvatuple(S) ? nothing : fieldcount(S)

"""
    parse_defaults(path::AbstractString; versions = nothing) -> Dict{String, Any}

Parse a package's defaults file (its `configs/default.toml`) and record it as a
precompilation dependency of the package being precompiled, so editing the file
recompiles it. Call it at a module's top level with an absolute path:
`const _TOML_DEFAULTS = parse_defaults(default_config_path(); versions = CONFIG_VERSIONS)`.
With the package's [`ConfigVersions`](@ref) the file is read through [`migrate_config`](@ref), so the code
reads the current table names only.
"""
function parse_defaults(path::AbstractString; versions = nothing)
    include_dependency(path)
    d = TOML.parsefile(path)
    return versions === nothing ? d : migrate_config(d, versions; source = path)
end

# -----------------------------------------------------------------------------
# Versioned configurations: one converter per package, its migrations its own
# -----------------------------------------------------------------------------

"""
    CONFIG_VERSION_KEY = "config_version"

The top-level key that states a configuration file's version (an integer). A file without it is version 1: the
files written before configurations were versioned.
"""
const CONFIG_VERSION_KEY = "config_version"

"""
    ConfigMigration(; tables, renames = [])

One version change of a package's configuration: `tables` are the top-level tables of the version it reads (every
one of them: a table of that version that is not listed is unknown, and refused); `renames` are `old => new`
dotted paths (`"old_table" => "family.new_table"`, `"table.key" => "other.table.key"`), applied in order, each table
moved whole (its keys and sub-tables kept) and each key with its value.
"""
struct ConfigMigration
    tables::Vector{String}
    renames::Vector{Pair{String, String}}
end
ConfigMigration(; tables, renames = Pair{String, String}[]) =
    ConfigMigration(collect(String, tables), Pair{String, String}[String(first(r)) => String(last(r)) for r in renames])

"""
    ConfigVersions(package, migrations::ConfigMigration...; tables)

A package's configuration converter: `tables` are the current version's top-level tables (every one: an unknown table
of a current file is refused, as of an earlier one), and `migrations[v]` reads version `v` and writes version `v + 1`,
so the current version is `length(migrations) + 1` ([`current_config_version`](@ref)). A package declares ONE, beside its
`configs/default.toml`, which states the current version under [`CONFIG_VERSION_KEY`](@ref); its defaults are read
through it (`parse_defaults(path; versions)`), and a caller migrates a stored file with
[`migrate_config`](@ref).
"""
struct ConfigVersions
    package::String
    tables::Vector{String}
    migrations::Vector{ConfigMigration}
end
ConfigVersions(package::AbstractString, migrations::ConfigMigration...; tables) =
    ConfigVersions(String(package), collect(String, tables), collect(ConfigMigration, migrations))

"""
    current_config_version(versions::ConfigVersions) -> Int

The version a package's code reads: one past its last migration.
"""
current_config_version(v::ConfigVersions) = length(v.migrations) + 1

"""
    migrate_config(d::AbstractDict, versions::ConfigVersions; source = "a configuration") -> Dict{String, Any}

The configuration `d` (a parsed TOML file of `versions.package`) at the current version, a new dictionary stamped
with it: each migration from the file's version on renames its tables (`d` is not modified). Refused with an
`ArgumentError` that names the key: a version newer than the package's, a version that is not a positive integer,
a table the file's version does not have (an unknown key, old or current), a renamed table whose new path is taken, and, in a
current file, a table under a name an earlier version used (it names the current one).
"""
function migrate_config(d::AbstractDict, v::ConfigVersions; source::AbstractString = "a configuration")
    cur = current_config_version(v)
    ver = get(d, CONFIG_VERSION_KEY, 1)
    ver isa Integer && ver >= 1 || throw(ArgumentError(
        "$source: `$CONFIG_VERSION_KEY` = $(repr(ver)) is not a version of $(v.package)'s configuration (a positive integer)"))
    ver <= cur || throw(ArgumentError(
        "$source: $(v.package) configuration version $ver is newer than this package's ($cur): update $(v.package)"))
    out = Dict{String, Any}(deepcopy(d))
    delete!(out, CONFIG_VERSION_KEY)
    for k in ver:(cur - 1)
        mig = v.migrations[k]
        for key in sort!(collect(keys(out)))
            key in mig.tables || throw(ArgumentError(
                "$source: [$key] is not a table of $(v.package)'s configuration version $k (its tables: " *
                    join(mig.tables, ", ") * ")"))
        end
        for (old, new) in mig.renames
            t = _config_pop!(out, old)
            t === nothing && continue
            _config_get(out, new) === nothing || throw(ArgumentError(
                "$source: [$old] moves to [$new] in $(v.package)'s configuration version $(k + 1), and [$new] is already there"))
            _config_put!(out, new, t)
        end
    end
    if ver == cur
        for (k, mig) in enumerate(v.migrations), (old, new) in mig.renames
            _config_get(out, old) === nothing || throw(ArgumentError(
                "$source: [$old] is $(v.package)'s configuration version $k's name; version $cur calls it [$new]"))
        end
    end
    for key in sort!(collect(keys(out)))
        key in v.tables || throw(ArgumentError(
            "$source: [$key] is not a table of $(v.package)'s configuration version $cur (its tables: " *
                join(v.tables, ", ") * ")"))
    end
    out[CONFIG_VERSION_KEY] = cur
    return out
end

"""
    CONFIG_VERSIONS

VisualGeometryToolkitCore's configuration converter ([`ConfigVersions`](@ref)): `configs/default.toml` is version 2.
Version 1 → 2 moves the LM's `[lm]` table to `[levenberg_marquardt.stops]` and its `device_max_iter` and
`device_acceptance` keys to `[static_levenberg_marquardt.stops]` as `max_iter` and `acceptance`.
"""
const CONFIG_VERSIONS = ConfigVersions("VisualGeometryToolkitCore",
    # version 1 → 2 (Core 0.14): an algorithm's stops live in `[<algorithm>.stops]` — the host LM's
    # `[levenberg_marquardt.stops]`, the device LM's `[static_levenberg_marquardt.stops]` (its keys lose `device_`)
    ConfigMigration(; tables = ["lm", "optimization_strategy", "photon_transfer"],
        renames = ["lm.device_max_iter" => "static_levenberg_marquardt.stops.max_iter",
                   "lm.device_acceptance" => "static_levenberg_marquardt.stops.acceptance",
                   "lm" => "levenberg_marquardt.stops"]);
    tables = ["levenberg_marquardt", "static_levenberg_marquardt", "optimization_strategy", "photon_transfer"])

# The table at a dotted path, or `nothing`: `toml_default`'s one dotted lookup, its KeyError for a missing table read
# as absence.
function _config_get(d::AbstractDict, path::AbstractString)
    segs = split(path, '.')
    return try
        toml_default(d, join(segs[1:(end - 1)], '.'), segs[end])
    catch err
        err isa KeyError || rethrow()
        nothing
    end
end
# Pop the value at a dotted path; a parent table the pop leaves empty goes with it (up the path), so a move out of
# `[schemes.<tag>]` leaves no `[schemes]` behind for the current version to refuse.
function _config_pop!(d::AbstractDict, path::AbstractString)
    segs = split(path, '.')
    parent = length(segs) == 1 ? d : _config_get(d, join(segs[1:(end - 1)], '.'))
    parent isa AbstractDict && haskey(parent, segs[end]) || return nothing
    x = pop!(parent, segs[end])
    for k in (length(segs) - 1):-1:1
        t = _config_get(d, join(segs[1:k], '.'))
        t isa AbstractDict && isempty(t) || break
        k == 1 ? delete!(d, segs[1]) : delete!(_config_get(d, join(segs[1:(k - 1)], '.')), segs[k])
    end
    return x
end
function _config_put!(d::AbstractDict, path::AbstractString, x)
    segs = split(path, '.')
    t = d
    for seg in segs[1:(end - 1)]
        t = get!(() -> Dict{String, Any}(), t, seg)
    end
    t[segs[end]] = x
    return d
end

"""
    toml_default(defaults::AbstractDict, table::AbstractString, key::AbstractString)

Look a default up in a parsed defaults file. `table` is a dotted path
(`"ransac"`, `"optimization_strategy.hq"`; `""` is the top level). Throws a
`KeyError` naming the missing table or key BY DESIGN: a keyword default that
is not in the file would be a silent second source of truth, which is what
the mechanism exists to prevent.
"""
function toml_default(defaults::AbstractDict, table::AbstractString, key::AbstractString)
    t = defaults
    for seg in split(table, '.'; keepempty = false)
        haskey(t, seg) || throw(KeyError("defaults have no table [$table]"))
        t = t[seg]
    end
    haskey(t, key) || throw(KeyError("defaults [$table] have no key `$key`"))
    return t[key]
end

"""
    write_toml(path, body::AbstractDict; header = "") -> path

Write `body` as sorted TOML, creating the directory, with `header` (one or
more `# ` comment lines, no trailing newline needed) first. The one writer
behind every package's `write_config`.
"""
function write_toml(path::AbstractString, body::AbstractDict; header::AbstractString = "")
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        isempty(header) || println(io, header)
        TOML.print(io, body; sorted = true)
    end
    return path
end

public construct_from_table
