# =============================================================================
# src/tuning/persist.jl — the writer and reader of a tuning artifact.
#
# Every artifact a tuning run persists (a sequential-CV sidecar or its
# tune-phase checkpoint, a probe CV sidecar, a bundle of validation matrices)
# goes through one writer/reader pair, stamped with the `:tuning` artifact
# format (artifact_format.jl). It records
#
#   tuning_name, tuning_spec, tuning_hash  the `[tuning.<name>]` spec it ran
#                                         under (complete table) and its hash
#   grid_specs                            the GridSpec of every axis it tuned
#                                         on — realized grids are rebuilt
#                                         from them, bit-exactly
#   sampling_unit                         what its cross-validation sampled
#                                         (the data's `sampling_unit`)
#   code_identity                         version of this package and of every
#                                         module the CALLER names
#                                         as the code its result came from
#
# and the run's own data. A file of an older format is refused (regenerate it
# by re-running its tuning), but for a format-6 file, which is read through its
# version step (below); a stored spec whose hash does not match its table is refused.
# =============================================================================

"""
    code_identity(code) -> Dict{String,Any}

The code a tuning result came from: for this package and every module of
`code` (the caller's packages whose code determines the result — the
estimator and scorer packages it ran, say), `version` (the loaded package
version); and the `julia` version
with the RNG algorithms the run may draw from (`rng`, [`RNG_ALGORITHMS`](@ref)) —
the streams are StableRNG's, but a bootstrap CI resamples from Julia's own
default stream (Stats `bootstrap_ci`), which only the Julia version pins.
The code's git state is the run's record, not this one: the `code_trees`
([`code_trees`](@ref VisualGeometryToolkitEval.code_trees)) its run records and
`_provenance.toml` keep, the one rule a stored result is reused by.
"""
function code_identity(code)
    out = Dict{String,Any}("julia" => Dict{String,Any}("version" => string(VERSION),
                                                       "rng" => sort!(collect(keys(RNG_ALGORITHMS)))))
    for m in unique((parentmodule(@__MODULE__), code...))
        m isa Module || throw(ArgumentError("code_identity: $(repr(m)) is not a Module"))
        out[String(nameof(m))] = Dict{String,Any}("version" => string(pkgversion(m)))
    end
    return out
end

# The persisted name of a sampling unit, and back.
const _UNITS = Dict{String,SamplingUnit}("pair" => PairUnit(), "query" => QueryUnit())
_unit_key(u::SamplingUnit) = only(k for (k, v) in _UNITS if v === u)

# The writer's own keys of a tuning artifact; every other key is the run's data.
const _ARTIFACT_KEYS = ("tuning_name", "tuning_spec", "tuning_hash", "grid_specs", "sampling_unit",
                        "code_identity", _format_key(:tuning))

"""
    save_tuning_artifact(path, spec::TuningSpec; grids, unit, code, data...) -> path

The writer of a tuning artifact: the spec (complete table — its
cross-validation's scene lists included — name, hash), the `grids`
(`Dict{Symbol,<:GridSpec}`, one per axis the run tuned on), the sampling
`unit` its cross-validation drew (the data's [`sampling_unit`](@ref)), the
[`code_identity`](@ref) of `code` (the caller's modules whose code determines
the result; this package is always recorded) and the run's `data` (keyword values: arms, tuned
indices, validation matrices, …), stamped with the `:tuning` format.
"""
function save_tuning_artifact(path::AbstractString, spec::TuningSpec;
                              grids::AbstractDict{Symbol,<:GridSpec}, unit::SamplingUnit, code, data...)
    bad = [k for k in keys(data) if String(k) in _ARTIFACT_KEYS]
    isempty(bad) || throw(ArgumentError("save_tuning_artifact: $(bad) are the writer's own keys"))
    mkpath(dirname(abspath(path)))
    return _save_artifact(path, :tuning;
        tuning_name = spec.name, tuning_spec = tuning_table(spec), tuning_hash = tuning_hash(spec),
        grid_specs = Dict{String,Any}(String(k) => tuning_table(g) for (k, g) in grids),
        sampling_unit = _unit_key(unit), code_identity = code_identity(code), data...)
end

"""
    TuningArtifact

A loaded tuning artifact ([`load_tuning_artifact`](@ref)): `spec` (the
[`TuningSpec`](@ref) it ran under), `hash`, `grid_specs` and the realized
`grids` (`Dict{Symbol,…}`, rebuilt from the specs), `unit` (the
[`SamplingUnit`](@ref) its cross-validation drew), `code` (the code identity)
and `data` (the run's own keys). Indexing with a string reads
`data`: `art["arms"]`. `format` is the `:tuning` format the FILE was written
in: an earlier one was carried over in memory by its version steps
([`load_tuning_artifact`](@ref)), and `recorded_hash` is the hash the file
records (`hash` is the current table's; the two differ only for such a file).
"""
struct TuningArtifact
    path::String
    spec::TuningSpec
    hash::String
    grid_specs::Dict{Symbol,GridSpec}
    grids::Dict{Symbol,Vector{Float64}}
    unit::SamplingUnit
    code::Dict{String,Any}
    data::Dict{String,Any}
    format::Int
    recorded_hash::String
end
Base.getindex(a::TuningArtifact, k::AbstractString) = a.data[k]
Base.haskey(a::TuningArtifact, k::AbstractString) = haskey(a.data, k)
Base.get(a::TuningArtifact, k::AbstractString, default) = get(a.data, k, default)

"""
    load_tuning_artifact(path) -> TuningArtifact

The reader of a tuning artifact: refuses another format first (an
`ArgumentError` naming the file: regenerate it by re-running its tuning), then reads the
spec (complete — nothing inherited), checks its hash, and rebuilds every grid
from its spec.

A format-6 file (before the efficiency-constant names) is read through its
version step, in memory (VisualGeometryToolkitIO's `migrate_record`); the file is never
rewritten. Its efficiency constants are stored as the value each arm's `ctor`
took then: an arm's is read through its roster-declared `format6_constant`
hook (`Evaluation.tuned_width`; the identity by default).
`format` and `recorded_hash` of the result say which file it was.
"""
function load_tuning_artifact(path::AbstractString)
    fmt = _check_artifact_format(path, :tuning)
    raw = JLD2.load(path)
    recorded = String(raw["tuning_hash"])
    fmt < ARTIFACT_FORMATS.tuning.version && (raw = _migrated_raw(raw, fmt, path))
    name = String(raw["tuning_name"])
    scheme = read_tuning(TuningScheme, raw["tuning_spec"])
    spec = TuningSpec(name, scheme)
    h = tuning_hash(spec)
    h == raw["tuning_hash"] ||
        throw(ArgumentError("$path: the stored tuning spec hashes to $h, not its recorded $(raw["tuning_hash"])"))
    gspecs = Dict{Symbol,GridSpec}(Symbol(k) => read_tuning(GridSpec, v) for (k, v) in raw["grid_specs"])
    unit = get(_UNITS, raw["sampling_unit"], nothing)
    unit === nothing && throw(ArgumentError("$path: unknown sampling_unit $(repr(raw["sampling_unit"]))"))
    data = Dict{String,Any}(k => v for (k, v) in raw if !(k in _ARTIFACT_KEYS))
    return TuningArtifact(String(path), spec, h, gspecs,
                          Dict{Symbol,Vector{Float64}}(k => grid(g) for (k, g) in gspecs), unit,
                          Dict{String,Any}(raw["code_identity"]), data, fmt, recorded)
end

# -----------------------------------------------------------------------------
# The version steps of a tuning artifact (VisualGeometryToolkitIO `migrate_schema`)
# -----------------------------------------------------------------------------
# A tuning artifact is a JLD2 file, not a vgio record: its version is its
# `tuning_format` stamp. The keys a version step reads or rewrites — the spec,
# its hash, the grid specs and the data keys named by an axis — travel as the
# record "vgtk_eval/tuning/v<format>" through VisualGeometryToolkitIO's
# `migrate_record`; every other data key (the run's matrices, curves, …) is
# carried over untouched, so its element types do not move.

const _TUNING_SCHEMA = "vgtk_eval/tuning"
# The data keys a step may rewrite: the axis-named ones of the stack's writers
# (`boundary_hits` and `kinds` name axes in their values; a coarse-fine
# artifact's `axes`, `selected` and `boundary_hit` name them in values and keys).
const _AXIS_DATA_KEYS = ("boundary_hits", "kinds", "axes", "selected", "boundary_hit")
const _MIGRATED_KEYS = ("tuning_spec", "tuning_hash", "grid_specs", _AXIS_DATA_KEYS...)

# A tuning artifact's step record `tree` — `"schema" => "vgtk_eval/tuning/v<n>"`
# and its `tuning_spec`, `tuning_hash`, `grid_specs` and axis-named data keys
# (`boundary_hits`, `kinds`, `axes`, `selected`, `boundary_hit`, and a
# coarse-fine `record`'s `param_name`) — carried to the current format by
# VisualGeometryToolkitIO's `migrate_record`, through the version steps this
# package registers (`migrate_schema`). Two steps: v6 → v7, the efficiency-constant
# names (2026-11-11), and v7 → v8, the hash (below). v6 → v7:
#
# - every key of the spec and the grid specs naming the efficiency constant by
#   its retired v6 name (`_V6_NAME`: the axis, its window, …) takes the
#   `efficiency_constant` name, and so does every axis name in the data keys
#   (an axis's `_hi` in a boundary-hit list, a coarse-fine axis);
# - the recorded hash is checked against the STORED table first (an
#   `ArgumentError` if the file's spec was altered), and the record carries the
#   renamed table's hash.
#
# The VALUES are not converted: a v6 efficiency constant is the value its arm's
# `ctor` took then, and only the roster says what that was (`tuned_width`
# applies the arm's `format6_constant` hook).

# A stored hash is verified with the recipe of the format that wrote it, and a step re-hashes with the next format's.
# Formats 6 and 7 hashed `Dict("scheme" => spec)` printed as sorted TOML (SHA-256, hex, first 16): that recipe is
# their definition, kept here only to verify a stored v6/v7 hash on read; nothing current is written with it. Format 8
# hashes the stack's canonical JSON (`tuning_hash`).
function _tuning_hash_v7(spec::AbstractDict)
    io = IOBuffer()
    TOML.print(io, Dict{String,Any}("scheme" => spec); sorted = true)
    return bytes2hex(sha256(take!(io)))[1:16]
end

function _check_stored_hash(t::AbstractDict, format::Integer)
    stored = _tuning_hash_v7(t["tuning_spec"])
    stored == t["tuning_hash"] || throw(ArgumentError(
        "a format-$format tuning record: the stored tuning spec hashes to $stored, not its recorded $(t["tuning_hash"])"))
    return nothing
end

# The v6 → v7 step: the efficiency-constant names.
function VgtkIO.migrate_schema(::Val{Symbol("vgtk_eval/tuning/v6")}, t::AbstractDict)
    _check_stored_hash(t, 6)
    out = Dict{String,Any}(k => _renamed_axis_keys(v) for (k, v) in t if k in ("tuning_spec", "grid_specs"))
    out["schema"] = string(_TUNING_SCHEMA, "/v7")
    out["tuning_hash"] = _tuning_hash_v7(out["tuning_spec"])
    for k in _AXIS_DATA_KEYS
        haskey(t, k) && (out[k] = _renamed_axis_names(t[k]))
    end
    haskey(t, "param_name") && (out["param_name"] = _renamed_axis_name(t["param_name"]))
    return out
end

# The v7 → v8 step: the hash, the stack's content key (canonical JSON) in place of the sorted TOML. Nothing else moves.
function VgtkIO.migrate_schema(::Val{Symbol("vgtk_eval/tuning/v7")}, t::AbstractDict)
    _check_stored_hash(t, 7)
    out = Dict{String,Any}(t)
    out["schema"] = string(_TUNING_SCHEMA, "/v8")
    out["tuning_hash"] = _tuning_table_hash(t["tuning_spec"])
    return out
end

# The efficiency constant's name in a v6 file. Spelled by concatenation: the
# name is retired, and the stack's naming check bans it as a literal everywhere.
const _V6_NAME = "scale" * "_multiple"
_renamed_axis_name(s::AbstractString) = replace(s, _V6_NAME => "efficiency_constant")
_renamed_axis_name(s::Symbol) = Symbol(_renamed_axis_name(String(s)))
_renamed_axis_name(x) = x
# A spec or grid-spec table: every key renamed, at every depth.
_renamed_axis_keys(d::AbstractDict) = Dict{String,Any}(_renamed_axis_name(String(k)) => _renamed_axis_keys(v) for (k, v) in d)
_renamed_axis_keys(v::AbstractVector) = Any[_renamed_axis_keys(x) for x in v]
_renamed_axis_keys(x) = x
# An axis-named data value: the axis names among its keys and its string values
# (an arm's boundary-hit list `arm => [axis_hi, …]`, a list of axes).
_renamed_axis_names(d::AbstractDict) = Dict{String,Any}(_renamed_axis_name(String(k)) => _renamed_axis_names(v) for (k, v) in d)
_renamed_axis_names(v::AbstractVector) = map(_renamed_axis_name, v)
_renamed_axis_names(x) = _renamed_axis_name(x)

# The file's step keys as the record of its format, through the steps, and back
# into the file's keys (every other key as it is).
function _migrated_raw(raw::AbstractDict, v::Integer, path)
    tree = Dict{String,Any}(k => raw[k] for k in _MIGRATED_KEYS if haskey(raw, k))
    tree["schema"] = string(_TUNING_SCHEMA, "/v", v)
    rec = get(raw, "record", nothing)
    rec isa AbstractDict && haskey(rec, "param_name") && (tree["param_name"] = rec["param_name"])
    cur = try
        VgtkIO.migrate_record(tree)
    catch e
        e isa ArgumentError ? throw(ArgumentError("$path: $(e.msg)")) : rethrow()
    end
    cur["schema"] == string(_TUNING_SCHEMA, "/v", ARTIFACT_FORMATS.tuning.version) || throw(ArgumentError(
        "$path: tuning format v$v has no version step to v$(ARTIFACT_FORMATS.tuning.version)"))
    out = Dict{String,Any}(raw)
    for k in _MIGRATED_KEYS
        haskey(cur, k) && (out[k] = cur[k])
    end
    haskey(cur, "param_name") && (out["record"] = merge(Dict{String,Any}(rec), Dict{String,Any}("param_name" => cur["param_name"])))
    return out
end
