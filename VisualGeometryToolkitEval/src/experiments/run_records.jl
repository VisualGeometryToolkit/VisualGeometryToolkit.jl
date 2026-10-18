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
#   data_root()/results_root()  ENV (CAU_DATA_ROOT / CAU_RESULTS_ROOT) →
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
#                            git record (+ dirty patch) covers it and
#                            run_complete WARNS on commit drift.
#   run_complete(outdir; sidecar)  keyed skip-if-complete: sidecar AND
#                            records present; warns when the recorded commit
#                            differs (experiments/provenance.jl)
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

# The data/results binding (root ladder, artifact view, write guard) and the
# provenance record (writer/reader, commit drift, run_complete, the keyed cache).
include("data.jl")
include("provenance.jl")

"The pseudo-GT oracle-floor sidecar of a results tree (the floors' producer writes it): `<root>/oracle_floors.jld2`."
oracle_floors_path(root::AbstractString = results_root()) = joinpath(root, "oracle_floors.jld2")

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
    "experiment"     => ("dataset", "matcher", "problem", "pool_dir", "oracle_tag"),
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
`problem` and `pool_dir`, and an unknown key in a section this package owns
([`RUN_CONFIG_KEYS`](@ref)) is refused.
"""
function load_run_config(path::AbstractString)
    isfile(path) || error("run config not found: $path")
    cfg = TOML.parsefile(path)
    haskey(cfg, "experiment") || error("run config $path: no [experiment] table")
    for k in ("dataset", "matcher", "problem", "pool_dir")
        haskey(cfg["experiment"], k) || error("run config $path: [experiment] lacks `$k`")
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
        d === nothing || VGC.check_keys(d, allowed, "$path $sec")
    end
    return cfg
end

"Top-level config sections that never touch a run's data and so never key it."
const RUN_KEY_EXCLUDED = ("render",)

"""
    run_key(cfg) -> String

The run identity: the SHA-256 (hex, first 16) of the effective config printed
as sorted TOML (Tuning `_toml_hash`, the one content hash behind
`tuning_hash` too), with the [`RUN_KEY_EXCLUDED`](@ref) sections removed
(`[render]` holds panel knobs of record, not experiment inputs) and every
`[tuning.<name>]` entry resolved to its COMPLETE table (`resolve_tuning!`: a
default of this package's configs/tuning.toml that the config inherits keys
the run like a stated one). Key order does not move it, the Julia version does
not move it, and a TOML round trip of the effective config reproduces it, so
`run_key(TOML.parsefile("_config.toml"))` names the directory that holds it.
"""
run_key(cfg::AbstractDict) =
    _toml_hash(Dict{String,Any}(k => v for (k, v) in resolve_tuning!(deepcopy(cfg)) if !(k in RUN_KEY_EXCLUDED)))

"""
    experiment_dir(cfg) -> String

The results directory of the config's `[experiment]`:
`<results root>/<dataset>/<matcher>/<problem>`. A run's keyed directory lives
under it, named by the project.
"""
experiment_dir(cfg::AbstractDict) =
    joinpath(results_root(), cfg["experiment"]["dataset"],
             cfg["experiment"]["matcher"], cfg["experiment"]["problem"])

"""
    write_run_records(outdir, cfg; driver, config_source, key, inputs = String[], producer = false)

Write the run's `_config.toml` (the effective config, its tuning specs
resolved to their complete tables) and its provenance record
([`write_provenance`](@ref): `config_source`, `run_key`, the declared `inputs`
relative to their roots, and DrWatson's git record). `producer = true` is a
pool generator's declaration that `outdir` is a NEW tree under the data root.
"""
function write_run_records(outdir::AbstractString, cfg::AbstractDict;
                           driver::AbstractString, config_source::AbstractString,
                           key::AbstractString,
                           inputs::Vector{String} = String[], producer::Bool = false)
    assert_not_input(outdir; producer)
    mkpath(outdir)
    open(joinpath(outdir, "_config.toml"), "w") do io
        println(io, "# EFFECTIVE configuration of this run (the FACT; the authored")
        println(io, "# file is intent and keeps evolving in git). Source: $(config_source)")
        TOML.print(io, resolve_tuning!(deepcopy(cfg)); sorted = true)
    end
    write_provenance(outdir, Dict{String,Any}("config_source" => abspath(config_source), "run_key" => key);
                     driver, inputs, producer)
    return outdir
end
