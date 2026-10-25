# =============================================================================
# src/tuning/toml.jl — the `[tuning]` parser and writer.
#
# configs/tuning.toml (this package) is the configuration of record of every
# tuning parameter's DEFAULT, parsed at precompile (`include_dependency`: an
# edit forces a recompile; restart a Revise session to see one). A
# config's `[tuning.<name>]` table names its scheme with `type` and states only
# what differs: every absent key inherits, through three levels —
#
#   1. `[tags.<family>.<tag>]` — the default parameters of a tagged strategy
#      (a spacing, a window rule, an objective, an axis, a unit sampling), or
#      `[types.<type>]` — those of an untagged one (the scale-multiple window,
#      the grid extension, the cross-validation, the budget sampling);
#   2. `[schemes.<scheme>]`    — the scheme's defaults, where they differ from 1;
#   3. the config's `[tuning.<name>]` table.
#
# A tagged table whose `type` differs from the inherited one starts again from
# its own tag's defaults. Unknown keys and unknown tags are an `ArgumentError`,
# never ignored. The writer emits the COMPLETE table (every key), so
# TOML → type → TOML → type is lossless; the reader of a PERSISTED table
# inherits nothing, so an incomplete persisted spec is refused.
#
# A registered scene split is named, never listed, by a config: its lists come
# from configs/splits.toml (sampling.jl `registered_split`) and are written
# into the complete table, so the hash and every artifact carry them. A
# config that lists scenes must list exactly the registry's.
#
# Field ⇄ key: a field's key is its name (VisualGeometryToolkitCore `toml_table` /
# `allowed_keys`), so a field cannot be added without its key.
# =============================================================================

"The path of the shipped tuning defaults of record, configs/tuning.toml."
const TUNING_DEFAULTS_PATH = normpath(joinpath(@__DIR__, "..", "..", "configs", "tuning.toml"))
"The shipped tuning defaults of record (configs/tuning.toml, parsed at precompile)."
const TUNING_DEFAULTS = VGC.parse_defaults(TUNING_DEFAULTS_PATH)

# -----------------------------------------------------------------------------
# The tag registries: TOML tag → strategy type, per family.
# -----------------------------------------------------------------------------

"""
    TUNING_TAGS

Every tagged strategy family of a `[tuning]` spec and its tag registry
(VisualGeometryToolkitCore's `TagRegistry`: TOML tag → type, the tag `snake_case` of
the type's name minus the family's suffix; retired tags name their successor).
A tagged table names its type with the key `type`.
"""
const TUNING_TAGS = Dict{String,VGC.TagRegistry}(
    "spacing"   => VGC.TagRegistry(:spacing, "", "log_per_decade" => LogPerDecade, "log_per_octave" => LogPerOctave,
                                   "log_count" => LogCount, "linear_count" => LinearCount),
    "window"    => VGC.TagRegistry(:window, "Window", "fixed" => FixedWindow, "standardized" => StandardizedWindow,
                                   "coverage" => CoverageWindow, "centered" => CenteredWindow),
    "axis"      => VGC.TagRegistry(:axis, "Axis", "spaced" => SpacedAxis, "listed" => ListedAxis),
    "grid"      => VGC.TagRegistry(:grid, "Grid", "spaced" => SpacedGrid, "listed" => ListedGrid),
    "criterion" => VGC.TagRegistry(:criterion, "", "median_error" => MedianError, "finite_median" => FiniteMedian,
                                   "scene_mean_lower_median" => SceneMeanLowerMedian,
                                   "scene_mean_median" => SceneMeanMedian,
                                   "mean_average_accuracy" => MeanAverageAccuracy),
    "scheme"    => VGC.TagRegistry(:scheme, "", "coarse_fine" => CoarseFine, "joint_coarse_fine" => JointCoarseFine,
                                   "windowed_bootstrap" => WindowedBootstrap, "factorial" => Factorial),
    "scene_split"   => VGC.TagRegistry(:scene_split, "SceneSplit", "registered" => RegisteredSceneSplit),
    "unit_sampling" => VGC.TagRegistry(:unit_sampling, "Sampling", "exhaustive" => ExhaustiveSampling,
                                       "stratified" => StratifiedSampling, "uniform" => UniformSampling;
                                       retired = ("all" => "exhaustive", "simple_random" => "uniform")),
)

# The abstract type of each family.
const _FAMILY_OF = Dict{Type,String}(
    GridSpacing => "spacing", WindowRule => "window", TuningAxis => "axis", GridSpec => "grid",
    TuningCriterion => "criterion", TuningScheme => "scheme",
    SceneSplit => "scene_split", UnitSampling => "unit_sampling",
)

_family(::Type{T}) where {T} = get(_FAMILY_OF, T, nothing)
function _tag(x)
    for (abs, fam) in _FAMILY_OF
        x isa abs || continue
        for (tag, T) in TUNING_TAGS[fam]
            typeof(x) === T && return tag
        end
    end
    throw(ArgumentError("no TOML tag for $(typeof(x))"))
end
_tagged(x) = any(abs -> x isa abs, keys(_FAMILY_OF))

function _tag_type(fam::String, tag, path)
    tag isa AbstractString || throw(ArgumentError("[$path]: `type` must name a $fam, got $(repr(tag))"))
    reg = TUNING_TAGS[fam]
    haskey(reg, tag) && return reg[tag]
    for (old, new) in VGC.retired_tags(reg)
        old == tag && throw(ArgumentError("[$path]: $fam type $(repr(tag)) was renamed $(repr(new))"))
    end
    throw(ArgumentError("[$path]: unknown $fam type $(repr(tag)); known: $(sort(collect(keys(reg))))"))
end

_tag_defaults(fam::String, tag::String) = get(get(get(TUNING_DEFAULTS, "tags", Dict()), fam, Dict()), tag, Dict{String,Any}())

# The untagged strategy types and their `[types.<key>]` default tables.
const _TYPE_KEYS = Dict{Type,String}(
    ScaleMultipleWindow => "scale_multiple_window", GridExtension => "grid_extension",
    CoarseFineAxis => "coarse_fine_axis", Incumbent => "incumbent", CrossValidation => "cross_validation", BudgetSampling => "budget_sampling",
)
_type_defaults(::Type{T}) where {T} =
    haskey(_TYPE_KEYS, T) ? get(get(TUNING_DEFAULTS, "types", Dict()), _TYPE_KEYS[T], Dict{String,Any}()) :
                            Dict{String,Any}()

# -----------------------------------------------------------------------------
# Writer: type → complete TOML table
# -----------------------------------------------------------------------------

# The spec types: every strategy, scheme and grid spec (the objectives
# included) is written as a table.
const _TABLE_TYPES = Union{GridSpacing, WindowRule, TuningAxis, GridSpec, TuningScheme,
                           SceneSplit, UnitSampling, ScaleMultipleWindow, GridExtension,
                           CoarseFineAxis, Incumbent, CrossValidation, BudgetSampling}

_table_value(x::Union{_TABLE_TYPES,TuningCriterion}) = tuning_table(x)
_table_value(x::AbstractVector) = Any[_table_value(v) for v in x]
_table_value(x::AbstractDict) = Dict{String,Any}(String(k) => _table_value(v) for (k, v) in x)
_table_value(x) = VGC.toml_value(x)

"""
    tuning_table(x) -> Dict{String,Any}

The COMPLETE TOML table of a tuning strategy, scheme, grid spec or
[`TuningSpec`](@ref) (whose table is its scheme's): every field under its own
name (VisualGeometryToolkitCore `toml_key`), a tagged type's tag under `type`.
Reading it back ([`tuning_spec`](@ref), [`read_tuning`](@ref)) gives an `==`
value.
"""
function tuning_table(x::Union{_TABLE_TYPES,TuningCriterion})
    T = typeof(x)
    t = Dict{String,Any}(VGC.toml_key(T, f) => _table_value(getfield(x, f)) for f in fieldnames(T)
                         if getfield(x, f) !== nothing)   # an absent optional table (an incumbent) is not written
    _tagged(x) && (t["type"] = _tag(x))
    return t
end
tuning_table(s::TuningSpec) = tuning_table(s.scheme)

"""
    tuning_toml(specs::TuningSpec...) -> Dict{String,Any}

The `[tuning]` table holding `specs` under their names (`write_toml`-ready).
"""
tuning_toml(specs::TuningSpec...) = Dict{String,Any}("tuning" => Dict{String,Any}(s.name => tuning_table(s) for s in specs))

"""
    tuning_hash(spec) -> String

The identity of a tuning protocol: the SHA-256 (hex, first 16) of its
complete scheme table printed as sorted TOML. Two specs with equal parameters
hash equal whatever their names or how their configs spelled them; any
parameter change moves it.
"""
tuning_hash(x::Union{TuningSpec,TuningScheme}) = _toml_hash(Dict{String,Any}("scheme" => tuning_table(x)))

"""
    _toml_hash(d::AbstractDict) -> String

The content hash of the evaluation stack: the SHA-256 (hex, first 16) of `d`
printed as sorted TOML. Independent of key order and fixed across Julia
versions (unlike `Base.hash`); a TOML round trip of `d` reproduces it. Behind
[`tuning_hash`](@ref) and Experiments `run_key`.
"""
function _toml_hash(d::AbstractDict)
    io = IOBuffer()
    TOML.print(io, d; sorted = true)
    return bytes2hex(sha256(take!(io)))[1:16]
end

# -----------------------------------------------------------------------------
# Reader: TOML table (+ inherited defaults) → type
# -----------------------------------------------------------------------------

# `d` is the config's table (or `nothing`), `dflt` the inherited one (or
# `nothing`); `inherit` adds the tag defaults beneath (off for a persisted
# table, which must be complete). Absent from both is an error naming the key
# path.
function _read(::Type{T}, d, dflt, path::String, inherit::Bool) where {T}
    fam = _family(T)
    fam === nothing || return _read_tagged(fam, d, dflt, path, inherit)
    base = inherit ? _type_defaults(T) : Dict{String,Any}()
    return _read_struct(T, d, dflt === nothing ? (isempty(base) ? nothing : base) : merge(base, dflt),
                        path, false, inherit)
end

function _read_tagged(fam::String, d, dflt, path, inherit::Bool)
    tag = d !== nothing && haskey(d, "type") ? d["type"] :
          dflt !== nothing && haskey(dflt, "type") ? dflt["type"] :
          throw(ArgumentError("[$path]: no `type` for this $fam"))
    C = _tag_type(fam, tag, path)
    # Inherit the default table only when it is of the same tag; the tag's own
    # defaults lie beneath both.
    base = inherit ? _tag_defaults(fam, String(tag)) : Dict{String,Any}()
    inherited = dflt !== nothing && get(dflt, "type", tag) == tag ? merge(base, dflt) : base
    inherit && (inherited = _registered(C, d, inherited, path))
    return _read_struct(C, d, inherited, path, true, inherit)
end

# A config names a registered scene split; its lists come from the registry
# (and a config that lists them must list exactly the registry's).
_registered(::Type, d, inherited, path) = inherited
function _registered(::Type{RegisteredSceneSplit}, d, inherited, path)
    name = d !== nothing && haskey(d, "name") ? d["name"] : get(inherited, "name", nothing)
    name isa AbstractString || throw(ArgumentError("[$path]: a registered scene split needs a `name`"))
    reg = try
        registered_split(name)
    catch e
        e isa ArgumentError && throw(ArgumentError("[$path]: $(e.msg)"))
        rethrow()
    end
    out = merge(inherited, Dict{String,Any}("name" => String(name)))
    for f in (:validation_scenes, :test_scenes)
        k = String(f)
        if d !== nothing && haskey(d, k)
            d[k] isa AbstractVector && sort(String.(d[k])) == getfield(reg, f) ||
                throw(ArgumentError("[$path]: `$k` differs from [splits.$name] of configs/splits.toml — " *
                                    "a config names a registered split, the registry lists its scenes"))
        end
        out[k] = getfield(reg, f)
    end
    return out
end

function _read_struct(::Type{C}, d, dflt, path, tagged::Bool, inherit::Bool) where {C}
    d === nothing && dflt === nothing && throw(ArgumentError("[$path]: missing table"))
    d === nothing || d isa AbstractDict || throw(ArgumentError("[$path]: expected a table, got $(repr(d))"))
    d === nothing || VGC.check_keys(d, VGC.allowed_keys(C, (tagged ? ("type",) : ())...), path)
    args = Any[]
    for f in fieldnames(C)
        key = VGC.toml_key(C, f)
        u = d === nothing ? nothing : get(d, key, nothing)
        v = dflt === nothing ? nothing : get(dflt, key, nothing)
        push!(args, _read_field(fieldtype(C, f), u, v, "$path.$key", inherit))
    end
    try
        return C(args...)
    catch e
        e isa ArgumentError && throw(ArgumentError("[$path]: $(e.msg)"))
        rethrow()
    end
end

_is_table(::Type{T}) where {T} = _family(T) !== nothing || haskey(_TYPE_KEYS, T)

function _read_field(::Type{T}, u, v, path, inherit::Bool) where {T}
    T === Union{Nothing,Incumbent} && return u === nothing && v === nothing ? nothing : _read(Incumbent, u, v, path, inherit)
    _is_table(T) && return _read(T, u, v, path, inherit)
    x = u === nothing ? v : u
    x === nothing && throw(ArgumentError("missing key `$path`"))
    return _leaf(T, x, path, inherit)
end

function _leaf(::Type{T}, x, path, inherit::Bool) where {T}
    try
        T === Float64 && x isa Real && !(x isa Bool) && return Float64(x)
        T === Int && x isa Integer && !(x isa Bool) && return Int(x)
        T === Bool && x isa Bool && return x
        T === String && x isa AbstractString && return String(x)
        T === Vector{Int} && x isa AbstractVector && all(y -> y isa Integer, x) && return collect(Int, x)
        T === Vector{Float64} && x isa AbstractVector && all(y -> y isa Real, x) && return collect(Float64, x)
        T === Vector{String} && x isa AbstractVector && all(y -> y isa AbstractString, x) && return collect(String, x)
        T === Dict{String,TuningAxis} && x isa AbstractDict &&
            return Dict{String,TuningAxis}(String(k) => _read(TuningAxis, a, nothing, "$path.$k", inherit) for (k, a) in x)
        T === Dict{String,Float64} && x isa AbstractDict && all(y -> y isa Real && !(y isa Bool), values(x)) &&
            return Dict{String,Float64}(String(k) => Float64(y) for (k, y) in x)
        T === Dict{String,CoarseFineAxis} && x isa AbstractDict &&
            return Dict{String,CoarseFineAxis}(String(k) => _read(CoarseFineAxis, a, nothing, "$path.$k", inherit) for (k, a) in x)
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("`$path`: cannot read $(repr(x)) as $T"))
    end
    throw(ArgumentError("`$path`: expected $T, got $(repr(x))"))
end

"""
    read_tuning(::Type{T}, table) -> T

A persisted COMPLETE table back to its type (no defaults inherited: a missing
key is an error). The reader of every spec and grid spec stored in an
artifact.
"""
read_tuning(::Type{T}, table::AbstractDict) where {T} = _read(T, table, nothing, string(nameof(T)), false)

"""
    tuning_spec(cfg, name) -> TuningSpec
    tuning_spec(path::AbstractString, name) -> TuningSpec

The `[tuning.<name>]` entry of a config (a parsed TOML `Dict` or a file path),
its absent keys inherited from the shipped defaults (configs/tuning.toml of
this package). The entry must name its scheme (`type`); unknown keys and
tags are an `ArgumentError` naming the key path.
"""
function tuning_spec(cfg::AbstractDict, name::AbstractString)
    haskey(cfg, "tuning") || throw(ArgumentError("config has no [tuning] table"))
    tt = cfg["tuning"]
    haskey(tt, name) ||
        throw(ArgumentError("config has no [tuning.$name] (has: $(sort(collect(String, keys(tt)))))"))
    entry = tt[name]
    entry isa AbstractDict || throw(ArgumentError("[tuning.$name] must be a table"))
    haskey(entry, "type") || throw(ArgumentError("[tuning.$name] names no scheme `type`"))
    tag = entry["type"]
    _tag_type("scheme", tag, "tuning.$name")
    dflt = get(get(TUNING_DEFAULTS, "schemes", Dict()), String(tag), nothing)
    dflt === nothing && throw(ArgumentError("configs/tuning.toml has no [schemes.$tag] defaults"))
    scheme = _read_tagged_scheme(entry, merge(Dict{String,Any}("type" => tag), dflt), "tuning.$name")
    return TuningSpec(String(name), scheme)
end
tuning_spec(path::AbstractString, name::AbstractString) = tuning_spec(TOML.parsefile(path), name)

_read_tagged_scheme(entry, dflt, path) = _read_struct(_tag_type("scheme", entry["type"], path), entry, dflt, path, true, true)

"""
    tuning_specs(cfg) -> Dict{String,TuningSpec}

Every `[tuning.<name>]` entry of a config, parsed ([`tuning_spec`](@ref)).
"""
tuning_specs(cfg::AbstractDict) =
    Dict{String,TuningSpec}(n => tuning_spec(cfg, n) for n in keys(get(cfg, "tuning", Dict())))
tuning_specs(path::AbstractString) = tuning_specs(TOML.parsefile(path))

"""
    resolve_tuning!(cfg) -> cfg

Replace every `[tuning.<name>]` entry of a parsed config by its COMPLETE table
(every parameter, defaults resolved — [`tuning_spec`](@ref) then
[`tuning_table`](@ref)), in place. Experiments' `run_key` and `write_run_records` key and record a
config resolved, so its run key and its recorded `_config.toml` cover every
tuning parameter, including the ones it inherits from this package's defaults.
"""
function resolve_tuning!(cfg::AbstractDict)
    haskey(cfg, "tuning") || return cfg
    specs = tuning_specs(cfg)
    cfg["tuning"] = Dict{String,Any}(n => tuning_table(s) for (n, s) in specs)
    return cfg
end

"""
    layer_tuning(lower, upper) -> Dict

`upper`'s `[tuning.<name>]` table layered over `lower`'s, the way the parser
layers a config over the defaults: a key of `upper` wins; a nested table
merges key by key — unless both name a `type` and the types differ, when
`upper`'s table replaces `lower`'s (a different strategy does not inherit
another's parameters). The one way a paper's protocol-level tuning table and a
cell's overrides are combined before [`tuning_spec`](@ref) /
[`resolve_tuning!`](@ref) read them.
"""
function layer_tuning(lower::AbstractDict, upper::AbstractDict)
    out = Dict{String,Any}(String(k) => v for (k, v) in lower)
    for (k, v) in upper
        k = String(k)
        l = get(out, k, nothing)
        out[k] = v isa AbstractDict && l isa AbstractDict &&
                 !(haskey(v, "type") && haskey(l, "type") && v["type"] != l["type"]) ?
            layer_tuning(l, v) : v
    end
    return out
end
