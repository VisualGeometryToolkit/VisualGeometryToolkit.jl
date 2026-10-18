# =============================================================================
# src/tuning/persist.jl — THE writer and reader of a tuning artifact.
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
#   code_identity                         version + git commit of this package
#                                         and of every module the CALLER names
#                                         as the code its result came from
#
# and the run's own data. A file of an older format is refused (regenerate it
# by re-running its tuning); a stored spec whose hash does not match its table is refused.
# =============================================================================

# A checkout's HEAD commit, "-dirty" when its tracked files differ — through
# LibGit2 and DrWatson's `isdirty`, not a `git` subprocess. "" outside a
# checkout of its own (a registry-installed package has no .git).
function _git_identity(dir)
    dir === nothing && return ""
    isdir(joinpath(dir, ".git")) || isfile(joinpath(dir, ".git")) || return ""
    sha = try string(LibGit2.head_oid(LibGit2.GitRepo(string(dir)))) catch; return "" end
    dirty = try isdirty(dir) catch; false end
    return dirty ? sha * "-dirty" : sha
end

"""
    code_identity(code) -> Dict{String,Any}

The code a tuning result came from: for this package and every module of
`code` (the caller's packages whose code determines the result — the
estimator and scorer packages it ran, say), `version` (the loaded package
version) and `git` (the checkout's HEAD commit, suffixed `-dirty` for
uncommitted changes; empty outside a git checkout); and the `julia` version
with the RNG algorithms the run may draw from (`rng`, [`RNG_ALGORITHMS`](@ref)) —
the streams are StableRNG's, but a bootstrap CI resamples from Julia's own
default stream (Stats `bootstrap_ci`), which only the Julia version pins.
"""
function code_identity(code)
    out = Dict{String,Any}("julia" => Dict{String,Any}("version" => string(VERSION),
                                                       "rng" => sort!(collect(keys(RNG_ALGORITHMS)))))
    for m in unique((parentmodule(@__MODULE__), code...))
        m isa Module || throw(ArgumentError("code_identity: $(repr(m)) is not a Module"))
        out[String(nameof(m))] = Dict{String,Any}("version" => string(pkgversion(m)),
                                                  "git" => _git_identity(pkgdir(m)))
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

THE writer of a tuning artifact: the spec (complete table — its
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
`data`: `art["arms"]`.
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
end
Base.getindex(a::TuningArtifact, k::AbstractString) = a.data[k]
Base.haskey(a::TuningArtifact, k::AbstractString) = haskey(a.data, k)
Base.get(a::TuningArtifact, k::AbstractString, default) = get(a.data, k, default)

"""
    load_tuning_artifact(path) -> TuningArtifact

THE reader of a tuning artifact: refuses another format first (an
`ArgumentError` naming the file: regenerate it by re-running its tuning), then reads the
spec (complete — nothing inherited), checks its hash, and rebuilds every grid
from its spec.
"""
function load_tuning_artifact(path::AbstractString)
    _check_artifact_format(path, :tuning)
    raw = JLD2.load(path)
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
                          Dict{String,Any}(raw["code_identity"]), data)
end
