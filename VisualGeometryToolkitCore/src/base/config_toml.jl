# =============================================================================
# TOML config skeleton — the generic half of a "one authority per value" loader
# =============================================================================
#
# Every package in the family with a configuration of record (RobustVisual-
# Geometry's `[ransac]`/`[local_optimization]`, BlobBoards' `[find_boards]`)
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
#
# What stays in each package: its `configs/default.toml`, the
# `include_dependency` + parsed constant, its tag registries (TOML string →
# concrete type) and the hook methods for its own types.
# =============================================================================

using TOML: TOML

# The Greek field names of the strategy structs and their ASCII TOML keys: ONE
# table, both directions, so the loader and the writer cannot disagree.
const _KEY_ASCII = Dict{Symbol, String}(
    :μ => "mu", :η => "eta", :γ => "gamma", :ν => "nu", :α => "alpha",
    :α_min => "alpha_min",
)
const _KEY_FIELD = Dict{String, Symbol}(key => field for (field, key) in _KEY_ASCII)

"""
    toml_field_name(key::AbstractString) -> Symbol

The field name a TOML key spells: the inverse of [`toml_key`](@ref)'s Greek
spelling (`"mu"` → `:μ`), `Symbol(key)` for every other key. The default
`field_of` of [`present_kwargs`](@ref).
"""
toml_field_name(key::AbstractString) = get(_KEY_FIELD, key, Symbol(key))

"""
    present_kwargs(d::AbstractDict, spec::Pair{String}...; fallback = nothing, field_of = toml_field_name)
        -> Vector{Pair{Symbol, Any}}

Typed per-key overrides. For each `"key" => convert` in `spec`, emit
`field_of("key") => convert(d["key"])` when `d` has the key, else the same from
`fallback` when THAT has it, else nothing at all — so an absent key falls
through to the struct's own default rather than to a parallel literal here.
`field_of` maps a TOML spelling to a field name (an ASCII `"mu"` for a Greek
`:μ` field); the default is [`toml_field_name`](@ref).
"""
function present_kwargs(d::AbstractDict, spec::Pair{String}...; fallback = nothing, field_of = toml_field_name)
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
    check_keys(d::AbstractDict, allowed, section::AbstractString; retired = ())

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
"""
function check_keys(d::AbstractDict, allowed, section::AbstractString; retired = ())
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
    convert_field(::Type{T}, v)

A TOML value → the field type (`Float64(v)`, `Int(v)`, `Bool(v)`, `Symbol(v)`):
the one converter behind a typed [`present_kwargs`](@ref) spec.
"""
convert_field(::Type{T}, v) where {T} = T(v)

"""
    parse_defaults(path::AbstractString) -> Dict{String, Any}

Parse a package's defaults file (its `configs/default.toml`) and record it as a
precompilation dependency of the package being precompiled, so editing the file
recompiles it. Call it at a module's top level with an absolute path:
`const _TOML_DEFAULTS = parse_defaults(default_config_path())`.
"""
function parse_defaults(path::AbstractString)
    include_dependency(path)
    return TOML.parsefile(path)
end

"""
    toml_default(defaults::AbstractDict, table::AbstractString, key::AbstractString)

Look a default up in a parsed defaults file. `table` is a dotted path
(`"ransac"`, `"local_optimization.hq"`; `""` is the top level). Throws a
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
