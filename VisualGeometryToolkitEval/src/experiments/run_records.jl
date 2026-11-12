# =============================================================================
# run_records.jl — the configuration-of-record layer of a codified experiment
# (promoted 2026-09-08 from LocalOptimizationDoneRight scripts/run_records.jl; the
# paper file is now a shim). Conventions ("Configuration: TOML. Data: JSON."):
# the authored TOML is INTENT, the `_config.toml` sidecar is FACT, the
# `_provenance.toml` records what actually ran, and keyed run directories are
# identity.
#
#   set_project_root!(root)  the project the drivers run from (default DrWatson's
#                            projectdir()): data/results and the git record
#                            resolve against it (experiments/data.jl)
#   data_root()/results_root()  ENV (VGTK_DATA_ROOT / VGTK_RESULTS_ROOT) →
#                            <project>/data/exp_raw | data/exp_pro (DrWatson's
#                            datadir) → the project's fallback / artifact view
#   load_run_config(path)    TOML.parsefile + [experiment] table sanity
#   run_key(cfg)             the SHA-256 content hash (`_toml_hash`) of the
#                            EFFECTIVE config minus RUN_KEY_EXCLUDED, its [tuning.<name>] entries resolved
#                            to their COMPLETE tables (`resolve_tuning!`), so
#                            every tuning parameter keys the run, inherited
#                            defaults included. A changed config is a NEW run dir,
#                            never an overwrite. Sections that do not touch
#                            the run's DATA — `[render]`, the panel knobs of
#                            record — are excluded so a cosmetic edit never
#                            re-keys an experiment. CODE is not keyed: the
#                            record keeps it (DrWatson's tag!: gitcommit +
#                            dirty patch; and the code's git trees), and
#                            run_complete reruns a run whose trees changed.
#   run_complete(outdir; sidecar)  keyed skip-if-complete: sidecar AND
#                            records present, and the recorded code trees the
#                            current ones (experiments/provenance.jl)
#   experiment_dir(cfg)      <results root>/<dataset>/<matcher>/<problem>
#   write_run_records(outdir, cfg; ...)  `_config.toml` (effective config) +
#                            `_provenance.toml` (write_provenance: DrWatson's
#                            tagged dict — gitcommit, gitpatch — with driver,
#                            config source, run key, host, julia, timestamp and
#                            the declared inputs relative to their roots)
#
# Justified extension of DrWatson: run_key is a SHA-256 of the sorted TOML, not
# `savename`, because run configs are NESTED (savename drops every non-scalar
# value, so two configs differing only inside a table would share a name).
# =============================================================================
using TOML: TOML
using Dates: Dates
using JLD2: JLD2
using Printf: Printf
using FileWatching.Pidfile: mkpidlock
using ..Stats: maa_threshold

# The data/results binding (root ladder, artifact view, write guard) and the
# provenance record (writer/reader, commit drift, run_complete, the keyed cache).
include("data.jl")
include("provenance.jl")

"The pseudo-GT oracle-floor sidecar of a results tree (the floors' producer writes it): `<root>/oracle_floors.jld2`."
oracle_floors_path(root::AbstractString = results_root()) = joinpath(root, "oracle_floors.jld2")

"""
    oracle_floor_source(pool_dir, problem, scenes) -> Dict{String,Any}

The record of what an oracle floor was computed on: the pool tree, the problem
family and the (sorted) scenes. [`write_oracle_floor`](@ref) stores it as the
floor's `source`; [`oracle_floor_done`](@ref) compares it with the current one.
"""
oracle_floor_source(pool_dir, problem, scenes) = Dict{String,Any}(
    "pool_dir" => String(pool_dir), "problem" => String(problem),
    "scenes" => sort(String.(collect(scenes))))

"""
    oracle_floors_note() -> String

The `note` stored in the oracle-floor sidecar. The mAA threshold in it is the
configured one ([`VisualGeometryToolkitEval.Stats.maa_threshold`](@ref)), not a literal.
"""
oracle_floors_note() = "median over test pairs of pool-best model error (<TAG> = floor; " *
    "<TAG>_maa = mean pool-best mAA@$(Printf.@sprintf("%g", maa_threshold()))° ceiling); " *
    "metric = problem-family model_imc; selection-only bound (LO may cross); " *
    "sources[<TAG>] = the pools/scenes of record"

"""
    write_oracle_floor(path, tag, floor, maa; source, npairs, driver) -> path

Record the pool-best floor `floor` (median model error) and ceiling `maa` (mean
pool-best mAA) of cell `tag` in the sidecar `path`
([`oracle_floors_path`](@ref)), which holds `"floors"` (`tag => floor`,
`tag * "_maa" => maa`), `"sources"` (`tag => source` plus `npairs`, `driver`
and the time computed) and `"note"` ([`oracle_floors_note`](@ref)). The other
cells of the file are kept. The read-modify-write is serialized by a pid lock
(`path * ".lock"`), because the oracle stages of concurrent chains write one
file.
"""
function write_oracle_floor(path::AbstractString, tag::AbstractString, floor_v, maa;
                             source::AbstractDict, npairs::Integer, driver)
    mkpath(dirname(abspath(path)))
    mkpidlock(String(path) * ".lock"; wait = true, stale_age = 600) do
        f       = isfile(path) ? JLD2.load(path) : Dict{String,Any}()
        floors  = get(f, "floors",  Dict{String,Float64}())
        sources = get(f, "sources", Dict{String,Any}())
        floors[tag]          = floor_v
        floors["$(tag)_maa"] = maa
        sources[tag] = merge(Dict{String,Any}(source),
                             Dict{String,Any}("npairs" => npairs, "driver" => String(driver),
                                              "computed" => string(Dates.now())))
        JLD2.jldsave(path; floors, sources, note = oracle_floors_note())
    end
    return path
end

"""
    oracle_floor_done(path, tag; source) -> Bool

Whether cell `tag` has its floor in the sidecar `path` AND the floor was
computed on `source` ([`oracle_floor_source`](@ref)): every key of `source` must
equal the recorded one. Presence of the key alone is not enough: floors
computed on a retired pool tree must not read as done.
"""
function oracle_floor_done(path::AbstractString, tag::AbstractString; source::AbstractDict)
    isfile(path) || return false
    f = JLD2.load(path)
    floors = get(f, "floors", Dict{String,Float64}())
    (haskey(floors, tag) && haskey(floors, "$(tag)_maa")) || return false
    src = get(get(f, "sources", Dict{String,Any}()), tag, nothing)
    src === nothing && return false
    return all(get(src, k, nothing) == v for (k, v) in source)
end

"""
    RUN_CONFIG_KEYS :: Vector{Pair{String,Tuple}}

Allowed keys of the run-config sections whose vocabulary THIS package defines,
as dotted paths (`"render.main"` is the nested table). A section absent here
belongs to the PROJECT that ships the config and is not key-checked —
`[source]`, `[extract]`, `[pool]`, `[arm]`, `[ransac]` and the rest are read
ad hoc by its drivers, and `[source.recon_special]` is keyed by scene name, so
a fixed vocabulary would be wrong. `[tuning.<name>]` has its own, stricter
check (Tuning `_read_struct` → `VisualGeometryToolkitCore.check_keys`, driven by the
struct fields) and is therefore not restated here.

Nested tables are listed in their own right: a check on `[render]` alone sees
`main` as one known key and nothing inside it.
"""
const RUN_CONFIG_KEYS = [
    "experiment"     => ("dataset", "matcher", "problem", "pool_dir", "oracle_tag", "layout"),
    "protocol"       => ("cv_section",),
    "render"         => ("panel", "med_panel", "rows", "legend", "sweep", "deploy", "exclude",
                         "extra_arms", "xticks", "ymin", "ymax", "ytick_spacing",
                         "maamin", "maamax", "maa_spacing", "main",
                         # a budget cell's figure wiring (accv2026 render_budget.jl)
                         "row_label", "xlog", "seed_panel", "time",
                         # the polishing cell's ECDF figures (accv2026 polish.jl)
                         "panel_disp", "error_xmax", "disp_xmax", "panel_diff", "diff_xmax"),
    "render.main"    => ("panel", "panel_med", "rows", "legend_prefix", "legend_suffix"),
    "render.time"    => ("legend", "timing", "xmax", "xticks", "maamax"),
]

# The table at a dotted path, or `nothing` when the config does not carry it.
function _config_table(cfg::AbstractDict, path::AbstractString)
    d = cfg
    for part in split(path, '.')
        d isa AbstractDict && haskey(d, part) || return nothing
        d = d[part]
    end
    return d isa AbstractDict ? d : nothing
end

"""
    load_run_config(path) -> Dict

Parse a run config: it must carry `[experiment]` with `dataset`, `matcher`,
`problem` and `pool_dir` — or, for a project whose results are not laid out by
pool, a `layout` ([`experiment_dir`](@ref)) — and an unknown key in a section
this package owns ([`RUN_CONFIG_KEYS`](@ref)) is refused.
"""
function load_run_config(path::AbstractString)
    isfile(path) || error("run config not found: $path")
    cfg = TOML.parsefile(path)
    haskey(cfg, "experiment") || error("run config $path: no [experiment] table")
    if !haskey(cfg["experiment"], "layout")
        for k in ("dataset", "matcher", "problem", "pool_dir")
            haskey(cfg["experiment"], k) || error("run config $path: [experiment] lacks `$k` (or a `layout`)")
        end
    end
    # Unknown keys are REFUSED in the sections this package owns
    # ([`RUN_CONFIG_KEYS`](@ref)). Only `[tuning.<name>]` was checked before,
    # and `[render]` is the sharp case: it is excluded from `run_key`
    # ([`RUN_KEY_EXCLUDED`](@ref)) so that a panel edit does not orphan an
    # estimate — which also means a typo there does not change the key, so the
    # driver matched the PREVIOUS run's keyed directory, found it complete and
    # skipped, reusing the old panel. Silently, at the default.
    for (sec, allowed) in RUN_CONFIG_KEYS
        d = _config_table(cfg, sec)
        d === nothing || VgtkCore.check_keys(d, allowed, "$path $sec")
    end
    return cfg
end

"Top-level config sections that never touch a run's data and so never key it."
const RUN_KEY_EXCLUDED = ("render",)

"""
    run_key(cfg; files = nothing, upstream = nothing, parts = nothing) -> String

The run identity: the SHA-256 (hex, first 16) of the effective config printed
as sorted TOML (Tuning `_toml_hash`, the one content hash behind
`tuning_hash` too), with the [`RUN_KEY_EXCLUDED`](@ref) sections removed
(`[render]` holds panel knobs of record, not experiment inputs) and every
`[tuning.<name>]` entry resolved to its COMPLETE table (`resolve_tuning!`: a
default of this package's configs/tuning.toml that the config inherits keys
the run like a stated one). Key order does not move it, the Julia version does
not move it, and a TOML round trip of the effective config reproduces it, so
`run_key(TOML.parsefile("_config.toml"))` names the directory that holds it.

What a run depends on beyond its config is folded in by name, so a project
keys it here instead of in a private hash:

- `files`: `name => path`, each file's content hash (`file_sha256`: streamed,
  cached by path, size and mtime) — a calibration file, a board definition;
- `upstream`: `name => key`, the run keys of the runs it reads (a pose run's
  key in the key of the evaluation that reads its poses), or `name => run
  directory`, read for its recorded `run_key` (a scratch run's is refused);
- `parts`: `name => value` (a `Dict` or a `NamedTuple`, TOML values), a choice
  of the run that is not in its config (a frame limit of a smoke run).

They are keyed as the table `keyed_with` (`files`, `upstream`, `parts`, each
present when given; and `scratch` in a [`scratch_run`](@ref), so its outputs
never share a run of record's directory), which a config therefore may not
hold; without them the key is the config's alone, as before. A value of record belongs in the config:
`parts` is for a choice made on the command line of a driver.
"""
function run_key(cfg::AbstractDict; files = nothing, upstream = nothing, parts = nothing)
    d = Dict{String,Any}(k => v for (k, v) in resolve_tuning!(deepcopy(cfg)) if !(k in RUN_KEY_EXCLUDED))
    haskey(d, "keyed_with") && throw(ArgumentError("run_key: a config may not hold `keyed_with`, the table of what the key folds in"))
    folded = Dict{String,Any}()
    files === nothing || (folded["files"] = Dict{String,Any}(String(k) => file_sha256(String(p)) for (k, p) in pairs(files)))
    upstream === nothing || (folded["upstream"] = Dict{String,Any}(String(k) => _upstream_key(String(k), String(v)) for (k, v) in pairs(upstream)))
    scratch_run() && (folded["scratch"] = true)          # a scratch run never shares a run of record's key
    parts === nothing || (folded["parts"] = Dict{String,Any}(String(k) => v for (k, v) in pairs(parts)))
    isempty(folded) || (d["keyed_with"] = folded)
    return _toml_hash(d)
end

# An upstream run as a key: a key as given, or a run directory, read for its recorded `run_key`; a scratch run's
# directory is refused to a run of record.
function _upstream_key(name::AbstractString, v::AbstractString)
    isdir(v) || return v
    isfile(joinpath(v, PROVENANCE_FILE)) || throw(ArgumentError("run_key: upstream $name = $v is a directory with no $PROVENANCE_FILE"))
    !scratch_run() && scratch_record(v) &&
        throw(ArgumentError("run_key: upstream $name = $v is a scratch run's output (scratch = true), never an upstream run of record"))
    k = get(read_provenance(v), "run_key", "")
    isempty(k) && throw(ArgumentError("run_key: upstream $name = $v records no run_key"))
    return String(k)
end

"""
    experiment_dir(cfg) -> String

The results directory of the config's `[experiment]`:
`<results root>/<dataset>/<matcher>/<problem>`, or, when it names a `layout`
(an array of path segments, `layout = ["mocap", "cam0"]`, for a project whose
results are not laid out by dataset, matcher and problem), `<results
root>/<layout…>`. The layout is the config's, never the code's; a segment that
is empty, absolute, `.` or `..` is refused. A run's keyed directory lives under
it ([`keyed_cell_dir`](@ref)).
"""
function experiment_dir(cfg::AbstractDict)
    e = cfg["experiment"]
    haskey(e, "layout") || return joinpath(results_root(), e["dataset"], e["matcher"], e["problem"])
    seg = e["layout"]
    (seg isa AbstractVector && !isempty(seg) && all(s -> s isa AbstractString && !isempty(s) && !isabspath(s) &&
         !(s in (".", "..")) && !occursin('/', s) && !occursin('\\', s), seg)) ||
        throw(ArgumentError("[experiment] layout = $(repr(seg)): a non-empty array of path segments (no `/`, `.`, `..`)"))
    return joinpath(results_root(), String.(seg)...)
end

"""
    keyed_cell_dir(cfg, experiment, kind; base = experiment_dir(cfg), files, upstream, parts) -> String

The keyed directory of one cell of `kind` under the results of `experiment`:
`<base>/<experiment>/<kind>-<run_key(cfg; files, upstream, parts)>`, `base`
by default the config's [`experiment_dir`](@ref) (its `[experiment] layout`,
else dataset, matcher and problem). A `base` given must lie under the results
root (`ArgumentError` otherwise). A changed config (or a folded file, upstream
key or part, [`run_key`](@ref)) is a new directory, never an overwrite; the
config is keyed as given (fold the protocols of record into it first).
"""
function keyed_cell_dir(cfg::AbstractDict, experiment::AbstractString, kind::AbstractString;
                        base::Union{Nothing,AbstractString} = nothing, files = nothing, upstream = nothing, parts = nothing)
    b = base === nothing ? experiment_dir(cfg) : String(base)
    root = results_root()
    _is_under(b, root) ||
        throw(ArgumentError("keyed_cell_dir: base $b is not under the results root $root"))
    return joinpath(b, experiment, kind * "-" * run_key(cfg; files, upstream, parts))
end

"""
    keyed_data_dir(cfg, tier, kind; files, upstream, parts) -> String

The keyed directory of a run that PRODUCES data of the tier `tier` (training
annotations, a derived pool): `<the tier's root>/<tier>/<kind>-<run_key(cfg; …)>`,
the tier's root resolved by the consumers' resolver (`tier_data_root`:
`VGTK_DATA_ROOT_<TIER>`, the site's `[sites.<site>.tiers]`, the data root) as a
WRITE root: it never installs an artifact, and a tier an artifact view
publishes read-only is refused until its root is bound. A
consumer declares it as `[inputs.<id>] tier = "<tier>", path =
"<tier>/<kind>-<key>"`, the same path on every machine: producer and consumer
agree wherever the tier's root is. The one place a producer writes input data;
it writes its records with `producer = true` ([`write_run_records`](@ref),
[`assert_not_input`](@ref)). A changed config is a new directory, never a
resumed or overwritten one.
"""
keyed_data_dir(cfg::AbstractDict, tier::AbstractString, kind::AbstractString; files = nothing, upstream = nothing, parts = nothing) =
    joinpath(_tier_root(tier; write = true), tier, kind * "-" * run_key(cfg; files, upstream, parts))

"""
    tier_path(tier, parts...) -> String

A path in the input tier `tier`: `joinpath(data_root(; tier), parts...)`, the
tier's root resolved (and installed from an artifact view) as for an input. The
one reader of a dataset's own layout (a capture set, `tier_path(tier, set)`)
where a config does not name the path; a config's declared input is
[`input_path`](@ref)'s, which also checks its `sha256`.
"""
tier_path(tier::AbstractString, parts::AbstractString...) = joinpath(data_root(; tier), parts...)

"""
    promote_run(run_dir, tier; name = basename(run_dir)) -> String

A completed run of record (a directory with its `_provenance.toml`, under the
results root) made an INPUT of record: copied to
`<data_root(; tier)>/<tier>/<name>` (the tier's root resolved as a consumer's),
where an `[inputs.<id>]` table names it (`tier`, `path = "<tier>/<name>"`) and
a data artifact of that tier publishes it (an [`ArtifactView`](@ref) installs it
like any tier). A run read by another run belongs there, not in the results
root and not in a directory named by a date. Refuses a run without its record,
and a destination that holds a different run (another `run_key` in its
record), and a scratch run's output; the same run again is left as it is.
Returns the destination.
"""
function promote_run(run_dir::AbstractString, tier::AbstractString; name::AbstractString = basename(rstrip(run_dir, '/')))
    isfile(joinpath(run_dir, PROVENANCE_FILE)) ||
        throw(ArgumentError("promote_run: $run_dir holds no $PROVENANCE_FILE: only a recorded run is an input of record"))
    scratch_record(run_dir) &&
        throw(ArgumentError("promote_run: $run_dir is a scratch run's output (scratch = true): only a run of record is an input of record"))
    key = get(read_provenance(run_dir), "run_key", "")
    dst = joinpath(_tier_root(tier; write = true), tier, name)
    if isdir(dst)
        had = isfile(joinpath(dst, PROVENANCE_FILE)) ? get(read_provenance(dst), "run_key", "") : ""
        had == key && return dst
        throw(ArgumentError("promote_run: $dst holds another run (run_key \"$had\", this one \"$key\"): name it otherwise"))
    end
    assert_not_input(dst; producer = true)
    mkpath(dirname(dst))
    cp(run_dir, dst)
    return dst
end

"""
    verify_inputs(cfg; root = nothing) -> Dict{String,String}

The `[inputs.<id>]` of a config, checked as the runner checks them
(`missing_inputs`): each `path` (relative to its tier's root,
[`data_root`](@ref)`(; tier)`, or to `root` for every input when given) is there, and one with a
`sha256` has that content (streamed, cached: `file_sha256`), through the one
check the runner makes (`inspect_input`). A scratch run's output is refused as
an input, except to a scratch run ([`scratch_run`](@ref)). Returns `id =>`
absolute path; an `ArgumentError` names every missing input (id, tier, path),
every other content (id, path, expected and actual hash) and every scratch
output.
"""
function verify_inputs(cfg::AbstractDict; root::Union{Nothing,AbstractString} = nothing)
    out, bad = Dict{String,String}(), String[]
    for (id, t) in get(cfg, "inputs", Dict{String,Any}())
        (t isa AbstractDict && haskey(t, "path")) || continue
        r = root === nothing ? data_root(; tier = String(get(t, "tier", ""))) : root
        x = inspect_input(String(id), t, r; allow_scratch = scratch_run())       # the runner's check (src/code_identity.jl)
        if x.state === :missing
            push!(bad, "input $id ($(get(t, "tier", "?"))): $(x.problem)")
        elseif x.state === :refused
            push!(bad, "input $id ($(t["path"])): $(x.problem)")
        end
        out[String(id)] = joinpath(r, t["path"])
    end
    isempty(bad) || throw(ArgumentError("refused: the inputs are not the data of record\n  " * join(sort!(bad), "\n  ")))
    return out
end

"""
    input_path(cfg, id; root = nothing) -> String

The absolute path of the config's input `[inputs.<id>]`: its `path` under its
tier's root ([`data_root`](@ref)`(; tier)`: `VGTK_DATA_ROOT_<TIER>`, the stage
graph's `[sites.<site>.tiers]`, the data root), or under `root` when given;
checked to be there and, with a `sha256`, to have that content (streamed,
cached). The one reader a driver opens an input with; an `ArgumentError` for
an id the config does not declare, a missing input or another content.
"""
function input_path(cfg::AbstractDict, id::AbstractString; root::Union{Nothing,AbstractString} = nothing)
    ins = get(cfg, "inputs", Dict{String,Any}())
    t = get(ins, id, nothing)
    (t isa AbstractDict && haskey(t, "path")) ||
        throw(ArgumentError("input_path: the config declares no [inputs.$id] with a path (it declares: $(join(sort!(String.(collect(keys(ins)))), ", ")))"))
    return verify_inputs(Dict{String,Any}("inputs" => Dict{String,Any}(String(id) => t)); root)[String(id)]
end

"""
    config_of_record(name; root = project_root()) -> Dict

The config of record `name` of the project at `root` (`configs/<name>.toml`),
loaded as a run config ([`load_run_config`](@ref): its `[experiment]` checked,
unknown keys of the sections this package owns refused), not yet folded.
"""
config_of_record(name::AbstractString; root::AbstractString = project_root()) =
    load_run_config(joinpath(root, "configs", name * ".toml"))

"""
    write_run_records(outdir, cfg; driver, config_source, key, inputs = String[], producer = false, fields = Dict())

Write the run's `_config.toml` (the effective config, its tuning specs
resolved to their complete tables) and its provenance record
([`write_provenance`](@ref): `config_source`, `run_key`, the declared `inputs`
relative to their roots, and DrWatson's git record). `producer = true` is a
pool generator's declaration that `outdir` is a NEW tree under the data root.
`fields` is a further payload of the record (a replay's own counts, a detection
CV's folds): written beside `config_source` and `run_key`, which it may not
name.
"""
function write_run_records(outdir::AbstractString, cfg::AbstractDict;
                           driver::AbstractString, config_source::AbstractString,
                           key::AbstractString,
                           inputs::Vector{String} = String[], producer::Bool = false,
                           fields::AbstractDict = Dict{String,Any}())
    clash = intersect(String.(collect(keys(fields))), ("config_source", "run_key"))
    isempty(clash) || throw(ArgumentError("write_run_records: field(s) $(sort(clash)) are the records' own"))
    assert_not_input(outdir; producer)
    mkpath(outdir)
    open(joinpath(outdir, "_config.toml"), "w") do io
        println(io, "# EFFECTIVE configuration of this run (the FACT; the authored")
        println(io, "# file is intent and keeps evolving in git). Source: $(config_source)")
        TOML.print(io, resolve_tuning!(deepcopy(cfg)); sorted = true)
    end
    write_provenance(outdir, merge(Dict{String,Any}(String(k) => v for (k, v) in fields),
                                   Dict{String,Any}("config_source" => abspath(config_source), "run_key" => key));
                     driver, inputs, producer)
    return outdir
end
