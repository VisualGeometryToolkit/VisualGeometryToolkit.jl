# =============================================================================
# reproduce/runner.jl — one command reproduces a paper project: its stage graph of record (a TOML of the project:
# which drivers on which configs of record, in which order), run on a site (a TOML of the machine: Slurm or local,
# partition, CPUs, GPUs), with every missing input reported as data before anything runs, a run manifest per launch,
# finished stages reused by their run keys, and the paper's results branch at the end.
#
#   reproduce_main(["--graph=<project graph>.toml", "--site=dgx01", "--dry-run"])     the job graph, nothing submitted
#   reproduce_main(["--graph=…", "--site=desktop", "--only=detector_bias"])          a subset (its prerequisites too)
#   reproduce_main(["status", "--graph=…", "--site=dgx01"])                          the last launch's jobs
#
# DEPENDENCY-FREE BY CONTRACT (Base and the stdlibs TOML, SHA and Dates; `cli_option` and `_write_atomic` come from the
# includer, src/cli.jl):
# the Reproduce module includes it, AND a project's launcher includes src/cli.jl and this file BY PATH from a bare `julia`,
# before the project's environment exists — the setup stage instantiates it, inside the scheduler. One copy for both.
# The results-branch stage runs in the project's environment and calls `results_branch_of_run`
# (reproduce/run_results_branch.jl), which is not dependency-free and lives in the module only.
# =============================================================================
using TOML: TOML
using Dates: Dates

"""
    SITES_DIR

The directory of the site profiles shipped with VisualGeometryToolkitEval (`configs/sites/<name>.toml`: the machines,
not the projects). `_template.toml` there documents every key; a project may keep its own site files and name them
by path, or list their directory as the graph's `[project] sites_dir`.
"""
const SITES_DIR = normpath(joinpath(@__DIR__, "..", "..", "configs", "sites"))

# ---------------------------------------------------------------------------
# Site profiles
# ---------------------------------------------------------------------------

const _SITE_KEYS = Dict(
    "site"  => ("name", "backend", "description", "setup"),
    "slurm" => ("partition", "gres", "cpus_per_task", "mem_per_cpu", "gpu_limit", "extra", "job_name_prefix"),
    "local" => ("cuda_visible_devices", "cpus_per_task"),
    "env"   => nothing)                                    # any variable

"""
    Site

A machine a reproduction runs on, read by [`load_site`](@ref) from a site profile: its `backend` (`:slurm`, sbatch
arrays chained by `--dependency=afterok`; `:local`, every stage in order in this process's machine), the Slurm
options every job gets (`partition`, a GPU stage's `gres`, `cpus_per_task`, `mem_per_cpu`, `gpu_limit` GPU array
tasks at a time, `extra` options, and `job_name_prefix`, put before every job's name: the launch's label for the
scheduler, never part of a run key), the local `cuda_visible_devices` and `cpus_per_task`, a shell `setup` run before
every stage, and the environment `env` every stage gets.
"""
struct Site
    name::String
    path::String
    backend::Symbol
    description::String
    setup::String
    partition::String
    gres::String
    cpus_per_task::Int
    mem_per_cpu::String
    gpu_limit::Int
    extra::Vector{String}
    job_name_prefix::String
    cuda_visible_devices::String
    env::Dict{String, String}
end

function _check_tables(d::AbstractDict, schema::AbstractDict, what::AbstractString)
    for (k, v) in d
        haskey(schema, k) || throw(ArgumentError("$what: unknown table [$k] (known: $(join(sort(collect(keys(schema))), ", ")))"))
        allowed = schema[k]
        (allowed === nothing || !(v isa AbstractDict)) && continue
        for kk in keys(v)
            kk in allowed || throw(ArgumentError("$what: unknown key $k.$kk (known: $(join(allowed, ", ")))"))
        end
    end
end

"""
    shipped_sites() -> Vector{String}

The names of the site profiles shipped in [`SITES_DIR`](@ref) (the template excluded).
"""
shipped_sites() = sort([first(splitext(f)) for f in readdir(SITES_DIR) if endswith(f, ".toml") && !startswith(f, "_")])

# A job-name prefix: letters, digits and `_ . -` (a scheduler's label, matched by a watcher's pattern), or none.
function _check_job_name_prefix(p, what)
    p isa AbstractString && occursin(r"\A[A-Za-z0-9_.-]*\z", p) ||     # \z: `$` would admit a final newline
        throw(ArgumentError("$what = $(repr(p)): letters, digits and _ . - only"))
    return String(p)
end

"""
    load_site(spec; dirs = String[]) -> Site

The site profile `spec`: a path to a TOML file (it ends in `.toml` or names an existing file), or a name looked up as
`<name>.toml` in `dirs` (a project's own site files first) and then in [`SITES_DIR`](@ref). An unknown table or key,
a backend other than `slurm` or `local`, or an unknown name (the error lists the known ones) is refused.
"""
function load_site(spec::AbstractString; dirs::AbstractVector{<:AbstractString} = String[])
    path = if endswith(spec, ".toml") || isfile(spec)
        isfile(spec) || throw(ArgumentError("site profile $spec: no such file"))
        abspath(spec)
    else
        found = [joinpath(d, spec * ".toml") for d in [dirs; SITES_DIR] if isfile(joinpath(d, spec * ".toml"))]
        isempty(found) && throw(ArgumentError("unknown site $(repr(spec)): the shipped sites are $(join(shipped_sites(), ", "))" *
            (isempty(dirs) ? "" : ", and none in $(join(dirs, ", "))") * "; or give the path of a site file (configs/sites/_template.toml)"))
        first(found)
    end
    d = TOML.parsefile(path)
    _check_tables(d, _SITE_KEYS, "site profile $path")
    s = get(d, "site", Dict{String, Any}())
    backend = get(s, "backend", "")
    backend in ("slurm", "local") || throw(ArgumentError("site profile $path: [site] backend = $(repr(backend)) is not \"slurm\" or \"local\""))
    sl, lo = get(d, "slurm", Dict{String, Any}()), get(d, "local", Dict{String, Any}())
    cpus = backend == "slurm" ? get(sl, "cpus_per_task", 0) : get(lo, "cpus_per_task", 0)
    prefix = _check_job_name_prefix(get(sl, "job_name_prefix", ""), "site profile $path: [slurm] job_name_prefix")
    return Site(get(s, "name", first(splitext(basename(path)))), path, Symbol(backend), get(s, "description", ""), get(s, "setup", ""),
                get(sl, "partition", ""), get(sl, "gres", ""), cpus, get(sl, "mem_per_cpu", ""), get(sl, "gpu_limit", 0),
                String.(get(sl, "extra", String[])), prefix, string(get(lo, "cuda_visible_devices", "")),
                Dict{String, String}(k => string(v) for (k, v) in get(d, "env", Dict{String, Any}())))
end

# ---------------------------------------------------------------------------
# The stage graph of record
# ---------------------------------------------------------------------------

const _GRAPH_KEYS = Dict(
    "project" => ("name", "workdir", "julia_project", "data_env", "results_env", "manifest_dir", "sites_dir", "code"),
    "sites" => nothing, "stage" => nothing,
    "results_branch" => ("project", "assets", "artifacts"))
const _STAGE_KEYS = ("name", "after", "what", "sbatch", "command", "results_branch", "driver", "config", "env", "job_name",
                     "gpu", "default", "time", "array", "inputs_from", "cpus_per_task", "mem_per_cpu", "gres", "ntasks", "nodes")
const _INPUTS_FROM_KEYS = ("config", "key", "base")
const _ARRAY_KEYS = ("items", "from", "select_env", "max_time", "time_margin_s")
const _SITE_ROOT_KEYS = ("data_root", "results_root", "env", "tiers")

"""
    UNPUBLISHED_TIER

The reserved input tier `"unpublished"`: data of record that is not (yet) published as a data artifact, so nothing
can fetch it — a capture kept on one machine. An `[inputs.<id>]` of this tier (its root through
`VGTK_DATA_ROOT_UNPUBLISHED` or `[sites.<site>.tiers]` like any tier) is checked to be there, and its `sha256` when
given; the dry run lists every such input apart, present or absent ([`print_plan`](@ref)), and a missing one is
reported as never fetched.
"""
const UNPUBLISHED_TIER = "unpublished"

"""
    StageSpec

One stage of a project's graph (`[[stage]]` of the graph TOML): its `name`, the stages it runs `after`, what it is
(`what`), and how it runs — exactly one of `sbatch` (the project's own launcher script: submitted as is on a Slurm
site, run by `bash` on a local one with the Slurm variables it reads set), `command` (an argv, wrapped by `sbatch
--wrap` on a Slurm site) or `results_branch = true` (the paper's results branch of the launch,
`results_branch_of_run`). `driver` and `config` name the driver and its config of record (the config may hold
`{item}`, one per array item): the run manifest hashes them, the input check reads the config's `[inputs]`, and the
dry run prints the driver's own command line. `array` makes the stage an array: over `items`, or over the config's
`[cases]` (`from = "cases"`, sorted by name); `inputs_from` (tables `{config, key, base}`) names further configs whose
`[inputs]` the stage reads, as the paths at a dotted `key` of `config` (a segment `*` is every key of that table;
each path relative to `base` under the workdir), for a stage that runs other drivers' configs; `select_env` names the variable that selects a subset (comma-separated)
and that the stage's launcher reads, and `max_time` (a dotted key of each case, seconds) plus `time_margin_s` sizes
its `--time`. `env` is exported to the stage, `job_name` and `time` override the launcher's, `gpu` marks a GPU stage
(`gres`, `CUDA_VISIBLE_DEVICES`, the site's `gpu_limit`), and `default = false` keeps it out of a run without `--only`.
`cpus_per_task`, `mem_per_cpu` and `gres` are the stage's own resources, over the site's (`0`/`""`: the site's;
validated: a count, `8G`/`4000M`, `gpu:4`), and `ntasks` and `nodes` its tasks and nodes (a distributed stage;
`0`: one task, Slurm's default): a
stage that needs 4 GPUs or 2 CPUs says so in the graph, not in its launcher; `gres` makes a stage a GPU stage. On a
local site `cpus_per_task` is the `SLURM_CPUS_PER_TASK` its launcher reads.
"""
struct StageSpec
    name::String
    after::Vector{String}
    what::String
    sbatch::String
    command::Vector{String}
    results_branch::Bool
    driver::String
    config::String
    env::Dict{String, String}
    job_name::String
    gpu::Bool
    default::Bool
    time::String
    items::Vector{String}
    items_from::String
    select_env::String
    max_time::String
    time_margin_s::Int
    is_array::Bool
    inputs_from::Vector{Dict{String, Any}}
    cpus_per_task::Int
    mem_per_cpu::String
    gres::String
    ntasks::Int
    nodes::Int
end

"""
    StageGraph

A project's stage graph of record, read by [`load_stage_graph`](@ref): the project's `name`, its `workdir` (where
every path of the graph is relative to and where the jobs are submitted from), its `julia_project`, the variables
that bind its data and results roots (`data_env`, `results_env`), the directory of the run manifests under the
results root (`manifest_dir`), each site's roots (`site_roots`), the project's own `sites_dir`, the `code` paths
whose content keys every stage, the `stages` in dependency order and the `[results_branch]` table.
"""
struct StageGraph
    path::String
    name::String
    workdir::String
    julia_project::String
    data_env::String
    results_env::String
    manifest_dir::String
    site_roots::Dict{String, Dict{String, Any}}
    sites_dir::String
    code::Vector{String}
    stages::Vector{StageSpec}
    results_branch::Dict{String, Any}
end

function _stage_spec(t::AbstractDict, where::AbstractString)
    for k in keys(t)
        k in _STAGE_KEYS || throw(ArgumentError("$where: unknown key $k (known: $(join(_STAGE_KEYS, ", ")))"))
    end
    name = get(t, "name", "")
    occursin(r"^[A-Za-z0-9_]+$", name) || throw(ArgumentError("$where: name = $(repr(name)) is not letters, digits and _"))
    how = count((haskey(t, "sbatch"), haskey(t, "command"), get(t, "results_branch", false) === true))
    how == 1 || throw(ArgumentError("stage $name: give exactly one of sbatch, command or results_branch = true"))
    a = get(t, "array", nothing)
    if a !== nothing
        for k in keys(a)
            k in _ARRAY_KEYS || throw(ArgumentError("stage $name: unknown key array.$k (known: $(join(_ARRAY_KEYS, ", ")))"))
        end
        haskey(a, "items") == (get(a, "from", "") == "cases") &&
            throw(ArgumentError("stage $name: an array takes items = [...] or from = \"cases\""))
        get(a, "from", "cases") == "cases" || throw(ArgumentError("stage $name: array.from = $(repr(a["from"])) is not \"cases\""))
        get(a, "from", "") == "cases" && !haskey(t, "config") &&
            throw(ArgumentError("stage $name: array.from = \"cases\" needs the stage's config"))
    end
    a = something(a, Dict{String, Any}())
    from = Dict{String, Any}.(get(t, "inputs_from", Dict{String, Any}[]))
    for f in from, k in keys(f)
        k in _INPUTS_FROM_KEYS || throw(ArgumentError("stage $name: unknown key inputs_from.$k (known: $(join(_INPUTS_FROM_KEYS, ", ")))"))
    end
    all(f -> haskey(f, "config") && haskey(f, "key"), from) || throw(ArgumentError("stage $name: inputs_from takes {config, key[, base]}"))
    return StageSpec(name, String.(get(t, "after", String[])), get(t, "what", ""), get(t, "sbatch", ""),
                     String.(get(t, "command", String[])), get(t, "results_branch", false), get(t, "driver", ""),
                     get(t, "config", ""), Dict{String, String}(k => string(v) for (k, v) in get(t, "env", Dict())),
                     get(t, "job_name", ""), get(t, "gpu", false), get(t, "default", true), get(t, "time", ""),
                     String.(get(a, "items", String[])), get(a, "from", ""), get(a, "select_env", ""),
                     get(a, "max_time", ""), get(a, "time_margin_s", 0), !isempty(a), from,
                     _cpus_count(name, get(t, "cpus_per_task", 0), "cpus_per_task"),
                     _resource(name, "mem_per_cpu", get(t, "mem_per_cpu", ""), r"^[0-9]+[KMGT]?$"),
                     _resource(name, "gres", get(t, "gres", ""), r"^[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)*(,[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)*)*$"),
                     _cpus_count(name, get(t, "ntasks", 0), "ntasks"), _cpus_count(name, get(t, "nodes", 0), "nodes"))
end

_cpus_count(name, v, key = "cpus_per_task") = (v isa Integer && v >= 0) ? Int(v) :
    throw(ArgumentError("stage $name: $key = $(repr(v)) is not a count"))
_resource(name, key, v, re) = (v isa AbstractString && (isempty(v) || occursin(re, v))) ? String(v) :
    throw(ArgumentError("stage $name: $key = $(repr(v)) is not a Slurm $(key == "gres" ? "gres (gpu:4, gpu:b200:2)" : "memory (8G, 4000M)")"))

# Kahn's order, ties in declaration order; a cycle or an unknown dependency is an error.
function _dependency_order(stages::Vector{StageSpec})
    names = [s.name for s in stages]
    allunique(names) || throw(ArgumentError("stage graph: a stage name is given twice: $(join(unique(filter(n -> count(==(n), names) > 1, names)), ", "))"))
    for s in stages, d in s.after
        d in names || throw(ArgumentError("stage $(s.name): after = $(repr(d)), no such stage"))
    end
    out, left = StageSpec[], copy(stages)
    while !isempty(left)
        i = findfirst(s -> all(d -> any(o -> o.name == d, out), s.after), left)
        i === nothing && throw(ArgumentError("stage graph: a dependency cycle among $(join((s.name for s in left), ", "))"))
        push!(out, popat!(left, i))
    end
    return out
end

"""
    load_stage_graph(path) -> StageGraph

A project's stage graph of record from its TOML (the project's `configs/reproduce.toml`): `[project]` (its `name`;
`workdir`, relative to the graph file's directory, default `"."`; `julia_project`, relative to `workdir`;
`data_env` and `results_env`, default `VGTK_DATA_ROOT` and `VGTK_RESULTS_ROOT`; `manifest_dir`, default
`"_reproduce"`; `sites_dir`, the project's own site files; `code`, the paths, relative to `workdir`, of the code
every stage runs, default `src` — their git trees, with those of every stack package the project's environment loads
([`code_trees`](@ref VisualGeometryToolkitEval.code_trees)), are the code a launch records: a finished stage is
reused only while they are unchanged, so a change elsewhere, the paper's text or the docs, reuses it, and a launch
with uncommitted changes in them is refused), `[sites.<site>]` (`data_root`, `results_root`, an `env`
table, and a `tiers` table, `<tier> = "<root>"`, the roots of the input tiers that are not under the data root on
that machine: where this project's data and results are on that machine), one `[[stage]]` per stage ([`StageSpec`](@ref)),
and `[results_branch]` (`project`, the paper project's directory relative to `workdir`; `assets`, tables
`{from, to, submitted_only}`: the renders matching the glob `from` go to the directory `to`, both relative to
`project`, only those with a submitted copy there when `submitted_only`; `artifacts`, an `Artifacts.toml` relative to
`project` whose entries' git-tree-sha1 are the data tiers). Unknown keys, unknown dependencies and cycles are refused.
"""
function load_stage_graph(path::AbstractString)
    path = abspath(path)
    d = TOML.parsefile(path)
    _check_tables(d, _GRAPH_KEYS, "stage graph $path")
    p = get(d, "project", Dict{String, Any}())
    haskey(p, "name") || throw(ArgumentError("stage graph $path: [project] name is required"))
    workdir = normpath(joinpath(dirname(path), get(p, "workdir", ".")))
    roots = Dict{String, Dict{String, Any}}()
    for (site, t) in get(d, "sites", Dict{String, Any}())
        for k in keys(t)
            k in _SITE_ROOT_KEYS || throw(ArgumentError("stage graph $path: unknown key sites.$site.$k (known: $(join(_SITE_ROOT_KEYS, ", ")))"))
        end
        roots[site] = t
    end
    stages = [_stage_spec(t, "stage graph $path, [[stage]] $i") for (i, t) in enumerate(get(d, "stage", Any[]))]
    isempty(stages) && throw(ArgumentError("stage graph $path: no [[stage]]"))
    rb = get(d, "results_branch", Dict{String, Any}())
    any(s -> s.results_branch, stages) && !haskey(rb, "assets") &&
        throw(ArgumentError("stage graph $path: a results_branch stage needs [results_branch] assets"))
    sites_dir = get(p, "sites_dir", "")
    return StageGraph(path, p["name"], workdir, get(p, "julia_project", "."), get(p, "data_env", "VGTK_DATA_ROOT"),
                      get(p, "results_env", "VGTK_RESULTS_ROOT"), get(p, "manifest_dir", "_reproduce"), roots,
                      isempty(sites_dir) ? "" : normpath(joinpath(workdir, sites_dir)), String.(get(p, "code", String[])),
                      _dependency_order(stages), rb)
end

function _stage(g::StageGraph, name)
    i = findfirst(s -> s.name == name, g.stages)
    i === nothing && throw(ArgumentError("unknown stage $(repr(name)); the graph has $(join((s.name for s in g.stages), ", "))"))
    return i
end

"""
    select_stages(g; only = String[], deps = true) -> Vector{String}

The stages a run of graph `g` covers, in dependency order: every `default` stage when `only` is empty, else the stages
`only` names; with `deps`, also every stage they run after, transitively (a finished one is reused by its run key, not
rerun). Without `deps` the named stages alone: their prerequisites are taken as done.
"""
function select_stages(g::StageGraph; only::AbstractVector{<:AbstractString} = String[], deps::Bool = true)
    want = Set{String}(isempty(only) ? [s.name for s in g.stages if s.default] : [g.stages[_stage(g, n)].name for n in only])
    if deps
        grow = true
        while grow
            n = length(want)
            for s in g.stages
                s.name in want && union!(want, s.after)
            end
            grow = length(want) > n
        end
    end
    return [s.name for s in g.stages if s.name in want]
end

# The value at a dotted key of a nested table.
_dotted(t, key) = foldl((x, k) -> x[k], split(key, '.'); init = t)

# Every string at a dotted key whose segment `*` is every key of that table (a list of strings: each of them); none where
# a key is absent.
function _dotted_all(t, key)
    vals = Any[t]
    for k in split(key, '.')
        vals = k == "*" ? Any[v for x in vals if x isa AbstractDict for v in values(x)] :
                          Any[x[k] for x in vals if x isa AbstractDict && haskey(x, k)]
    end
    return sort(unique(String[v for x in vals for v in (x isa AbstractVector ? x : [x]) if v isa AbstractString]))
end

_config_of(s::StageSpec, item) = replace(s.config, "{item}" => item)

"""
    stage_items(g, stage; env = ENV) -> Vector{String}

The items of an array stage, in task order: its `items`, or its config's `[cases]` sorted by name; the
comma-separated `env[select_env]`, when set, selects a subset (cases sorted, items in the order given; an unknown
case is an error). `[""]` for a stage that is not an array.
"""
function stage_items(g::StageGraph, s::StageSpec; env = ENV)
    s.is_array || return [""]
    sel = isempty(s.select_env) ? "" : get(env, s.select_env, "")
    picked = String.(split(sel, ','; keepempty = false))
    s.items_from == "cases" || return isempty(picked) ? s.items : picked
    cases = TOML.parsefile(joinpath(g.workdir, s.config))["cases"]
    isempty(picked) && return sort(collect(String, keys(cases)))
    bad = setdiff(picked, keys(cases))
    isempty(bad) || throw(ArgumentError("stage $(s.name): $(s.select_env) names unknown case(s) $(join(bad, ", "))"))
    return sort(picked)
end

# --time in minutes: the largest selected case's max_time plus the margin; 0 = the launcher's own.
function _array_minutes(g::StageGraph, s::StageSpec, items)
    (isempty(s.max_time) || s.items_from != "cases") && return 0
    cases = TOML.parsefile(joinpath(g.workdir, s.config))["cases"]
    m = maximum(round(Int, _dotted(cases[i], s.max_time)) for i in items)
    return cld(m + s.time_margin_s, 60)
end

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

"""
    missing_inputs(g, stages; data_root, env = ENV, site = "", unpublished = nothing) -> Vector{NamedTuple}

Every input of the selected `stages` that is not there, as data, before anything runs: one row
`(; path, tier, used_by)` per missing path (`used_by` every stage and config that reads it). A stage's inputs are its
configs of record (one per array item) and their `[inputs.<id>]` tables with a `path` (relative to `data_root`) and a
`tier`; a missing config is a row of tier `"config of record"`. With no data root every data input is missing (tier
as declared).

Each input's path is relative to ITS TIER's root ([`tier_data_root`](@ref VisualGeometryToolkitEval.tier_data_root)):
`env`'s `<data_env>_<TIER>`, else the graph's `[sites.<site>.tiers]` table, else `data_root`. A tier root that is
named but not there makes each of its inputs a row whose tier says so (`"mocap (root /x, from VGTK_DATA_ROOT_MOCAP,
is not there)"`). Each input is checked by `inspect_input` (src/code_identity.jl), the one check a driver's
`verify_inputs` makes too: a scratch run's output is refused like another content, unless this launch is a scratch one
(`VGTK_SCRATCH=1` in `env`, [`scratch_run`](@ref VisualGeometryToolkitEval.Experiments.scratch_run)). An input of the reserved tier
[`UNPUBLISHED_TIER`](@ref) is reported missing as never fetched; with `unpublished` (a vector), each such input is
also pushed there, `(; path, present, used_by)`, for the dry run to list.

An input with a `sha256` (64 hex digits, of a file) is verified: an input that is there with another content is
REFUSED, an `ArgumentError` naming each such input (its id, config, path, the expected and the actual hash), so
nothing runs on data other than the data of record. An input without one is not checked. The hash is streamed and
cached by path, size and modification time ([`file_sha256`](@ref VisualGeometryToolkitEval.file_sha256)): a file is
read once per process, and a launch that checks the same input from several configs reads it once.
"""
function missing_inputs(g::StageGraph, stages::AbstractVector{<:AbstractString}; data_root::AbstractString, env = ENV,
                        site::AbstractString = "", unpublished = nothing)
    rows = Dict{String, Tuple{String, Vector{String}}}()
    tiers = get(get(g.site_roots, site, Dict{String, Any}()), "tiers", nothing)
    view_mount = isempty(data_root) ? nothing :
                 catalog_view_mount(data_root, joinpath(g.workdir, g.julia_project, "configs", "data_artifacts.toml"))
    wrong = String[]
    add!(path, tier, user) = (r = get!(rows, path, (tier, String[])); user in r[2] || push!(r[2], user))
    function check!(cfg, user)
        p = joinpath(g.workdir, cfg)
        isfile(p) || (add!(cfg, "config of record", user); return)
        for (id, t) in get(TOML.parsefile(p), "inputs", Dict{String, Any}())
            (t isa AbstractDict && haskey(t, "path")) || continue
            tier = String(get(t, "tier", ""))
            root, from = tier_data_root(tier, data_root; env, tiers, workdir = g.workdir, prefix = g.data_env, view_mount)
            name = from == "env" ? tier_variable(tier; prefix = g.data_env) : from == "site" ? "[sites.$site.tiers]" : "the artifact view"
            r = inspect_input(id, t, root; root_source = from, root_name = name,        # the one check (src/code_identity.jl)
                              allow_scratch = scratch_run(env))
            if tier == UNPUBLISHED_TIER && unpublished !== nothing
                k = findfirst(u -> u.path == t["path"], unpublished)
                k === nothing ? push!(unpublished, (; path = String(t["path"]), present = r.state !== :missing && r.state !== :root_missing,
                                                     used_by = [user])) :
                                (user in unpublished[k].used_by || push!(unpublished[k].used_by, user))
            end
            if r.state === :root_missing
                add!(t["path"], r.problem, user)
            elseif r.state === :missing
                add!(t["path"], tier == UNPUBLISHED_TIER ? "$tier (never fetched: provide it on this machine)" : tier, user)
            elseif r.state === :refused
                push!(wrong, "input $id of $cfg ($(t["path"]), used by $user): $(r.problem)")
            end
        end
    end
    for n in stages
        s = g.stages[_stage(g, n)]
        if !isempty(s.config)
            for item in stage_items(g, s; env = env)
                cfg = _config_of(s, item)
                check!(cfg, "$(s.name) ($cfg)")
            end
        end
        for f in s.inputs_from
            src = joinpath(g.workdir, f["config"])
            isfile(src) || (add!(f["config"], "config of record", s.name); continue)
            for c in _dotted_all(TOML.parsefile(src), f["key"])
                cfg = normpath(joinpath(get(f, "base", "."), c))
                check!(cfg, "$(s.name) ($cfg, via $(f["config"]))")
            end
        end
    end
    isempty(wrong) || throw(ArgumentError("refused: an input is not the data of record\n  " * join(unique!(wrong), "\n  ")))
    return [(; path, tier = r[1], used_by = r[2]) for (path, r) in sort(collect(rows); by = first)]
end

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

"""
    PlannedStage

One stage of a launch: its spec, its `items`, the selected stages it runs `after`, its run `key` (every file and
option the stage's jobs read, and the keys of the stages it runs after, transitively: a stage rerun under a new key
reruns every stage after it; never the code: a stage of the same key that finished before is reused while the code
trees it recorded are the current ones, [`code_tree_changes`](@ref VisualGeometryToolkitEval.code_tree_changes)), the
environment its jobs get beyond the caller's (`env`), its `command` (on a Slurm site the sbatch argv without the
dependency; on a local site the argv run once per item), its `--time` in minutes (`minutes`, 0 = the launcher's),
`reuse` (`""`; `"completed"`, finished in an earlier launch; or the job id of an earlier launch's job still in
flight, which this launch depends on instead of submitting again) and, for that job, `reuse_markers`, the directory
its completion markers go to ([`stage_markers_dir`](@ref)); and `rerun`, why an earlier run of the same key is
not reused (`""` when there is none or it is: `"code changed: VisualGeometryToolkitCore: src 3fa29c01e2 → …"`).
"""
mutable struct PlannedStage
    spec::StageSpec
    items::Vector{String}
    after::Vector{String}
    key::String
    env::Dict{String, String}
    command::Vector{String}
    minutes::Int
    reuse::String
    reuse_markers::String
    rerun::String
end

"""
    ReproducePlan

A launch of a stage graph on a site: the graph, the site, the code `commit` (and `dirty`: tracked files anywhere
differ from it), the `data_root` and `results_root`, the `stamp` of the launch, the path of its run `manifest`, its
`stages` in dependency order, the environment the launch was planned in (`base_env`: the stages' environment is it
plus each stage's own), and the code it runs: `code_trees` (by name, the git trees of the graph's `code` paths and of
every stack package the project's environment loads: [`code_trees`](@ref VisualGeometryToolkitEval.code_trees)) and
`dirty_code`, those of them with uncommitted changes (a launch refuses to run them; a scratch launch runs them with
a warning), and `code_paths`, the project's own paths among them (a launch that identifies none is refused too).
"""
struct ReproducePlan
    graph::StageGraph
    site::Site
    commit::String
    dirty::Bool
    data_root::String
    results_root::String
    stamp::String
    manifest::String
    stages::Vector{PlannedStage}
    base_env::Dict{String, String}
    code_trees::Dict{String, String}
    dirty_code::Vector{String}
    code_paths::Vector{String}
end

# The commit and whether tracked files changed (anywhere: the code's own state is `code_trees`).
function _git_state(dir)
    commit = something(_git_line(dir, ["rev-parse", "HEAD"]), "")
    isempty(commit) && return ("(not a git checkout)", false)
    return (commit, !isempty(something(_git_line(dir, ["status", "--porcelain", "--untracked-files=no"]), "")))
end

"""
    resolve_roots(g, site; data_root = nothing, results_root = nothing, env = ENV) -> (data, results)

The data and results roots of a launch: the argument, else the graph's variables in `env` (`data_env`,
`results_env`), else the graph's `[sites.<site>]` roots (a relative one relative to the workdir); `""` when none
gives one.
"""
function resolve_roots(g::StageGraph, site::Site; data_root = nothing, results_root = nothing, env = ENV)
    t = get(g.site_roots, site.name, Dict{String, Any}())
    function pick(arg, var, key)
        v = String(something(arg, get(env, var, nothing), get(t, key, nothing), ""))
        return isempty(v) || isabspath(v) ? v : normpath(joinpath(g.workdir, v))
    end
    return (pick(data_root, g.data_env, "data_root"), pick(results_root, g.results_env, "results_root"))
end

_julia(site::Site, env) = get(site.env, "JULIA", get(env, "JULIA", "julia"))

# A stage's own resources over the site's.
_stage_gpu(s::StageSpec) = s.gpu || !isempty(s.gres)
_stage_cpus(site::Site, s::StageSpec) = s.cpus_per_task > 0 ? s.cpus_per_task : site.cpus_per_task

function _slurm_options(site::Site, s::StageSpec)
    o = String[]
    isempty(site.partition) || push!(o, "--partition=$(site.partition)")
    gres = isempty(s.gres) ? site.gres : s.gres
    _stage_gpu(s) && !isempty(gres) && push!(o, "--gres=$gres")
    _stage_cpus(site, s) > 0 && push!(o, "--cpus-per-task=$(_stage_cpus(site, s))")
    mem = isempty(s.mem_per_cpu) ? site.mem_per_cpu : s.mem_per_cpu
    isempty(mem) || push!(o, "--mem-per-cpu=$mem")
    s.ntasks > 0 && push!(o, "--ntasks=$(s.ntasks)")
    s.nodes > 0 && push!(o, "--nodes=$(s.nodes)")
    append!(o, site.extra)
    return o
end

_shell_quote(a) = occursin(r"^[A-Za-z0-9_./:=,%@+-]+$", a) ? a : "'" * replace(a, "'" => "'\\''") * "'"
_shell_line(argv) = join(_shell_quote.(argv), " ")

function _stage_argv(g::StageGraph, site::Site, s::StageSpec, manifest::AbstractString, env)
    s.results_branch || return isempty(s.sbatch) ? s.command : String[]
    return [_julia(site, env), "--startup-file=no", "--project=$(g.julia_project)", "-e",
            "using VisualGeometryToolkitEval.Reproduce: results_branch_of_run; results_branch_of_run(ARGS[1])", manifest]
end

# The name a stage's job has without a prefix: the stage's `job_name`, else its launcher's own `#SBATCH --job-name`
# (or `-J`), else `<project>-<stage>`.
function _base_job_name(g::StageGraph, s::StageSpec)
    isempty(s.job_name) || return s.job_name
    if !isempty(s.sbatch) && isfile(joinpath(g.workdir, s.sbatch))
        m = match(r"^#SBATCH\s+(?:--job-name=|-J\s*)(\S+)"m, read(joinpath(g.workdir, s.sbatch), String))
        m === nothing || return String(something(m.captures[1]))
    end
    return "$(g.name)-$(s.name)"
end

# The backend's command of a stage. `prefix` (a launch's job-name prefix) names every job `<prefix><base name>`; the
# run key is taken of the command without it, so a label for the scheduler never reruns a stage.
function _plan_command(g::StageGraph, site::Site, s::StageSpec, items, minutes, manifest, env; prefix::AbstractString = "")
    argv = _stage_argv(g, site, s, manifest, env)
    site.backend === :local && return isempty(s.sbatch) ? argv : ["bash", s.sbatch]
    c = ["sbatch", "--parsable"]
    append!(c, _slurm_options(site, s))
    jobname = isempty(prefix) ? (isempty(s.job_name) && isempty(s.sbatch) ? "$(g.name)-$(s.name)" : s.job_name) :
                                prefix * _base_job_name(g, s)
    isempty(jobname) || push!(c, "--job-name=$jobname")
    if s.is_array
        push!(c, "--array=0-$(length(items) - 1)" * (_stage_gpu(s) && site.gpu_limit > 0 ? "%$(site.gpu_limit)" : ""))
    end
    minutes > 0 ? push!(c, "--time=$minutes") : !isempty(s.time) && push!(c, "--time=$(s.time)")
    if isempty(s.sbatch)
        push!(c, "--output=logs/$(g.name)_$(s.name)_%j.log", "--wrap=$(_shell_line(argv))")
    else
        push!(c, s.sbatch)
    end
    return c
end

"""
    plan_reproduction(g, site; only = String[], deps = true, data_root = nothing, results_root = nothing,
                      job_name_prefix = nothing, env = ENV, stamp = <now, UTC>, queue_states = nothing) -> ReproducePlan

The launch of the selected stages ([`select_stages`](@ref)) of graph `g` on `site`, planned and nothing run: each
stage's items, dependencies, environment, command and run key, and whether an earlier launch's run of the same key is
reused ([`stage_run_state`](@ref): completed, by its completion markers; or in flight on a Slurm site, its job still
in the queue, `queue_states(ids)`, by default [`slurm_queue_states`](@ref) on `env`'s PATH; the earlier launches are
the manifests under `<results root>/<manifest_dir>`), and only while the code trees that launch recorded are the
current ones ([`code_tree_changes`](@ref VisualGeometryToolkitEval.code_tree_changes): a stage whose code changed runs
again, `rerun` naming the change; one still in flight under other code is an error). `job_name_prefix` (default: the
site's) is put before every job's name; the run keys are taken without it.
"""
function plan_reproduction(g::StageGraph, site::Site; only::AbstractVector{<:AbstractString} = String[], deps::Bool = true,
                           data_root = nothing, results_root = nothing, job_name_prefix = nothing, env = ENV,
                           stamp::AbstractString = Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS_sss"),
                           queue_states = nothing)
    prefix = _check_job_name_prefix(something(job_name_prefix, site.job_name_prefix), "job_name_prefix")
    names = select_stages(g; only, deps)
    data, results = resolve_roots(g, site; data_root, results_root, env)
    commit, dirty = _git_state(g.workdir)
    code = code_trees(g.workdir; code = isempty(g.code) ? nothing : g.code, julia_project = joinpath(g.workdir, g.julia_project))
    mdir = joinpath(isempty(results) ? "<results root>" : results, g.manifest_dir)
    manifest = joinpath(mdir, "$(stamp)-$(first(commit, 7))$(dirty ? "-dirty" : "").toml")
    n = 1
    while isfile(manifest)                                 # two launches in one millisecond: the second numbered
        manifest = joinpath(mdir, "$(stamp)-$(first(commit, 7))$(dirty ? "-dirty" : "")_$(n += 1).toml")
    end
    roots_env = Dict{String, String}()
    isempty(data) || (roots_env[g.data_env] = data)
    isempty(results) || (roots_env[g.results_env] = results)
    site_env = merge(Dict{String, String}(k => string(v) for (k, v) in get(get(g.site_roots, site.name, Dict()), "env", Dict())), site.env)
    stages = PlannedStage[]
    keys_of = Dict{String, String}()
    # Every stage the selected ones run after, transitively, is keyed too (selected or not, in dependency order): a
    # stage's key folds the keys of the stages it runs after, so a stage rerun under a new key reruns every stage after it.
    for name in select_stages(g; only = names)
        s = g.stages[_stage(g, name)]
        items = stage_items(g, s; env)
        minutes = _array_minutes(g, s, items)
        e = merge(roots_env, site_env, s.env)
        isempty(s.select_env) || (e[s.select_env] = join(items, ","))
        if site.backend === :local
            e["SLURM_SUBMIT_DIR"] = g.workdir
            _stage_cpus(site, s) > 0 && (e["SLURM_CPUS_PER_TASK"] = string(_stage_cpus(site, s)))
            e["CUDA_VISIBLE_DEVICES"] = _stage_gpu(s) ? site.cuda_visible_devices : ""
        end
        cmd = _plan_command(g, site, s, items, minutes, manifest, env)
        files = Dict{String, String}()
        for f in unique(filter(!isempty, [s.sbatch, s.driver, (_config_of(s, i) for i in items)...]))
            p = joinpath(g.workdir, f)
            files[f] = isfile(p) ? file_sha256(p) : "missing"
        end
        keyed = Dict{String, Any}("stage" => name, "files" => files, "items" => items,
                                  "env" => e, "command" => cmd, "site" => site.name)
        isempty(s.after) || (keyed["after"] = [keys_of[d.name] for d in g.stages if d.name in s.after])   # in dependency order
        scratch_run(env) && (keyed["scratch"] = true)     # a scratch launch never a run of record's key
        key = keys_of[name] = _toml_hash(keyed)
        name in names || continue
        isempty(prefix) || (cmd = _plan_command(g, site, s, items, minutes, manifest, env; prefix))
        push!(stages, PlannedStage(s, items, [d for d in s.after if d in names], key, e, cmd, minutes, "", "", ""))
    end
    base = Dict{String, String}(env)
    # The jobs resolve a tier's root as this launch did: the site's [sites.<site>.tiers] become the variables a
    # driver reads first (`tier_data_root`), unless the caller's environment already sets them; VGTK_SITE names the
    # site to a driver run by hand from inside a job.
    for (tier, _) in get(get(g.site_roots, site.name, Dict{String, Any}()), "tiers", Dict{String, Any}())
        var = tier_variable(tier; prefix = g.data_env)
        haskey(base, var) || (base[var] = first(tier_data_root(tier, data; env = base, workdir = g.workdir, prefix = g.data_env,
                                                              tiers = g.site_roots[site.name]["tiers"],
                                                              view_mount = isempty(data) ? nothing :
                                                                  catalog_view_mount(data, joinpath(g.workdir, g.julia_project, "configs", "data_artifacts.toml")))))
    end
    base["VGTK_SITE"] = site.name
    plan = ReproducePlan(g, site, commit, dirty, data, results, stamp, manifest, stages, base,
                         code.trees, code.dirty, code.project_paths)
    _mark_reuse!(plan, ids -> _queue_states(queue_states, ids, env))
    return plan
end

# ---------------------------------------------------------------------------
# Run manifests and reuse
# ---------------------------------------------------------------------------

_manifest_dir(plan::ReproducePlan) = joinpath(plan.results_root, plan.graph.manifest_dir)

"""
    run_manifests(dir) -> Vector{String}

The run manifests of earlier launches in `dir` (`<results root>/<manifest_dir>`), oldest first.
"""
run_manifests(dir::AbstractString) =
    isdir(dir) ? sort([joinpath(dir, f) for f in readdir(dir) if endswith(f, ".toml")]; by = f -> (mtime(f), f)) : String[]

# A program on `env`'s PATH; nothing when there is none.
function _which(prog::AbstractString, env)
    for d in split(get(env, "PATH", ""), ':'; keepempty = false)
        p = joinpath(d, prog)
        isfile(p) && Sys.isexecutable(p) && return p
    end
    return nothing
end

_job_states(job_states, ids, env) = job_states === nothing ? slurm_job_states(ids; env) : job_states(ids)
_queue_states(queue_states, ids, env) = queue_states === nothing ? slurm_queue_states(ids; env) : queue_states(ids)

const _IN_FLIGHT = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")

"""
    slurm_job_states(ids; env = ENV) -> Dict{String, Vector{String}}

The accounting state of every task of the Slurm jobs `ids` (`sacct`, found on `env`'s PATH), by job id:
`["COMPLETED", "FAILED", …]`. Empty when sacct is not there, fails (a cluster without accounting storage: "Slurm
accounting storage is disabled") or knows none of them. An extra of [`reproduce_status`](@ref) only, for a task
that left no completion marker; whether a stage finished is never read from it ([`stage_run_state`](@ref)).
"""
function slurm_job_states(ids::AbstractVector{<:AbstractString}; env = ENV)
    out = Dict{String, Vector{String}}()
    sacct = _which("sacct", env)
    (isempty(ids) || sacct === nothing) && return out
    txt = try readchomp(pipeline(Cmd(`$sacct -n -X -P -j $(join(ids, ",")) --format=JobID,State`; env = Dict{String, String}(env));
                                 stderr = devnull)) catch; "" end
    for l in split(txt, '\n'; keepempty = false)
        occursin('|', l) || continue
        id, st = split(l, '|'; limit = 2)
        push!(get!(out, String(first(split(id, '_'))), String[]), String(first(split(st))))
    end
    return out
end

"""
    slurm_queue_states(ids; env = ENV) -> Dict{String, Dict{Int, String}}

The tasks of the Slurm jobs `ids` that are still in the queue (`squeue`, found on `env`'s PATH; it needs no
accounting), by job id and task: `Dict(0 => "RUNNING", 1 => "PENDING")` (task 0 for a job that is not an array; task
-1 for a range of tasks squeue does not list one by one). Only the states of a job not yet finished (`PENDING`,
`RUNNING`, `COMPLETING`, …); a job that left the queue is absent (squeue's "Invalid job id specified"), as is every
job when squeue is not there, and so is a task pending on a dependency that can never be satisfied (an earlier stage
failed: it will never run). Any other failure of squeue (a slurmctld timeout, a controller restart) is an error: the
queue's state is then unknown, and reading its jobs as gone would submit a stage that is still running again.
"""
function slurm_queue_states(ids::AbstractVector{<:AbstractString}; env = ENV)
    out = Dict{String, Dict{Int, String}}()
    squeue = _which("squeue", env)
    squeue === nothing && return out
    fmt = "%i|%T|%r"
    for id in unique(ids)
        out_buf, err_buf = IOBuffer(), IOBuffer()
        rc = try
            run(pipeline(Cmd(`$squeue -h -r -j $id -o $fmt`; env = Dict{String, String}(env), ignorestatus = true);
                         stdout = out_buf, stderr = err_buf)).exitcode
        catch ex
            ex isa Base.IOError || rethrow()
            print(err_buf, sprint(showerror, ex)); -1
        end
        txt, err = String(take!(out_buf)), strip(String(take!(err_buf)))
        if rc != 0
            # A job that left the queue is what squeue calls an invalid id; any other failure (a slurmctld timeout,
            # a controller restart) says nothing about the job, and reading it as gone would resubmit a running stage.
            occursin("Invalid job id specified", err) && continue
            error("squeue -j $id failed (exit $rc): $(isempty(err) ? "no message" : err); the queue cannot be read, ",
                  "so whether the earlier launch's stages are still in flight is not known. Retry when squeue answers.")
        end
        for l in split(txt, '\n'; keepempty = false)
            occursin('|', l) || continue
            parts = strip.(split(l, '|'))
            jid, st = parts[1], parts[2]
            st in _IN_FLIGHT || continue
            (length(parts) >= 3 && occursin("DependencyNeverSatisfied", parts[3])) && continue
            base, task = occursin('_', jid) ? split(jid, '_'; limit = 2) : (jid, "0")
            base == id || continue
            t = something(tryparse(Int, task), -1)
            get!(out, id, Dict{Int, String}())[t] = st
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Completion markers
# ---------------------------------------------------------------------------

"""
    stage_markers_dir(manifest, stage) -> String

The directory of a stage's completion markers in the launch whose run manifest is `manifest`: `<manifest without
.toml>/<stage>/`. Each task of the stage writes `task-<t>.toml` there when it ends (`t` its array task, 0 for a stage
that is not an array): `state` (`"completed"` or `"failed"`), `exit_status`, `start` and `end` (UTC), `job_id`,
`commit`, `key` (the stage's run key), `stage`, `task`, `host`, `restart_count` (Slurm's `SLURM_RESTART_COUNT`,
0 for a first attempt), and `signal` when the job received one (TERM, then `state = "failed"`; or a signal the
launcher's `--signal` directive asks for, forwarded to it). On a
Slurm site the job writes it itself (a wrapper the runner submits around the stage's launcher, kept in the same
directory with the launcher's copy as submitted); the local backend writes it after each task. Every marker is
written atomically (a temporary file in the directory, renamed into place): a reader sees a whole marker or none.
"""
stage_markers_dir(manifest::AbstractString, stage::AbstractString) = joinpath(first(splitext(manifest)), stage)

_utc_now() = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z"

"""
    stage_markers(dir; job_id = nothing) -> Dict{Int, Dict{String, Any}}

The completion markers in a stage's marker directory ([`stage_markers_dir`](@ref)), by task; with `job_id`, only
those that job wrote (an earlier job of the same stage, resubmitted, is not this one).
"""
function stage_markers(dir::AbstractString; job_id = nothing)
    out = Dict{Int, Dict{String, Any}}()
    isdir(dir) || return out
    for f in readdir(dir)
        m = match(r"^task-(\d+)\.toml$", f)
        m === nothing && continue
        d = TOML.parsefile(joinpath(dir, f))
        (job_id === nothing || get(d, "job_id", "") == job_id) || continue
        out[parse(Int, something(m.captures[1]))] = d
    end
    return out
end

"""
    stage_run_state(record, queue = Dict()) -> (; state, tasks)

What the run of a stage that a run manifest records (`record`, one of its `[stages]`) came to, read from the
stage's completion markers ([`stage_markers`](@ref)) and the queue (`queue`: [`slurm_queue_states`](@ref)), never
from accounting. `tasks` maps each task to its state: its job's state while the queue holds the task
(`"PENDING"`, `"RUNNING"`, …), else its marker's (`"completed"`, `"failed"`), else `"unknown"`. The queue comes first
because Slurm requeues a task under the same job id (preemption with `PreemptMode=REQUEUE`, `scontrol requeue`): the
attempt it stopped left a `"failed"` marker, and the task is pending or running again. A task that is finishing
(`COMPLETING`) reads its marker when it has one. `state` is

- `"failed"` when a task's marker says so and the queue no longer holds it;
- `"unknown"` when a task has no marker and its job is no longer queued: it died without writing one (a node
  failure, a kill past the time limit's grace) or its markers are gone. Never taken as finished: a later launch
  runs the stage again;
- `"in flight"` when every task without a marker is still queued;
- `"completed"` when every task's marker says so.

A record of the local backend (no job id), which the runner itself writes after every task, is read as it stands:
`"completed"` or `"reused"` is completed, `"failed"` failed, anything else (a run stopped mid-way) unknown.
"""
function stage_run_state(r::AbstractDict, queue::AbstractDict = Dict{String, Dict{Int, String}}())
    n = length(get(r, "items", [""]))
    id = get(r, "job_id", "")
    rec = get(r, "state", "")
    if isempty(id)
        rec in ("completed", "reused") && return (; state = "completed", tasks = Dict(t => "completed" for t in 0:(n - 1)))
        tasks = Dict{Int, String}(parse(Int, k) => v for (k, v) in get(r, "tasks", Dict{String, Any}()))
        return (; state = rec == "failed" ? "failed" : "unknown", tasks)
    end
    marks = isempty(get(r, "markers", "")) ? Dict{Int, Dict{String, Any}}() : stage_markers(r["markers"]; job_id = id)
    queued = get(queue, id, Dict{Int, String}())
    tasks = Dict{Int, String}()
    for t in 0:(n - 1)
        q = get(queued, t, get(queued, -1, ""))
        # A task the queue holds again (requeued under the same job id: preemption, `scontrol requeue`) is in flight
        # whatever marker an earlier attempt left; only a task that is finishing (COMPLETING) reads its marker.
        tasks[t] = !isempty(q) && (q != "COMPLETING" || !haskey(marks, t)) ? q :
                   haskey(marks, t) ? marks[t]["state"] : "unknown"
    end
    v = values(tasks)
    state = any(==("failed"), v) ? "failed" : any(==("unknown"), v) ? "unknown" :
            all(==("completed"), v) ? "completed" : "in flight"
    return (; state, tasks)
end

# A stage of the same key in an earlier launch: completed (every task's marker) → reused; in flight → depended on,
# unless a stage it runs after is run again by this launch (the job waits on the old one); failed or unknown → run
# again. Reused or depended on only while the code trees it recorded are the current ones (`code_tree_changes`, the
# one rule): a completed stage whose code changed runs again, the change named; one still in flight under other code
# stops the launch (a second job would write the same results).
function _mark_reuse!(plan::ReproducePlan, queue_states)
    isempty(plan.results_root) && return plan
    last = Dict{String, Dict{String, Any}}()
    for m in run_manifests(_manifest_dir(plan))
        for (_, r) in get(TOML.parsefile(m), "stages", Dict{String, Any}())
            haskey(r, "key") && (last[r["key"]] = r)
        end
    end
    isempty(last) && return plan
    ids = unique(filter(!isempty, String[get(last[st.key], "job_id", "") for st in plan.stages if haskey(last, st.key)]))
    queue = isempty(ids) ? Dict{String, Dict{Int, String}}() : queue_states(ids)
    for st in plan.stages
        r = get(last, st.key, nothing)
        r === nothing && continue
        state = stage_run_state(r, queue).state
        if state in ("completed", "in flight")
            changes = code_tree_changes(get(r, "code_trees", nothing), plan.code_trees)
            if !isempty(changes)
                state == "in flight" && error("stage $(st.spec.name): job $(r["job_id"]) of an earlier launch is still in flight, ",
                                              "under other code ($(join(changes, "; "))): wait for it to end, or scancel it, then launch again")
                st.rerun = "code changed: " * join(changes, "; ")
                continue
            end
        end
        if state == "completed"
            st.reuse = "completed"
        elseif state == "in flight"
            st.reuse, st.reuse_markers = r["job_id"], get(r, "markers", "")
        end
    end
    rerun = Set(st.spec.name for st in plan.stages if isempty(st.reuse))     # in dependency order: a cleared stage clears those after it
    for st in plan.stages
        if st.reuse != "completed" && !isempty(st.reuse) && any(in(rerun), st.after)
            st.reuse, st.reuse_markers = "", ""
            push!(rerun, st.spec.name)
        end
    end
    return plan
end

function _manifest_dict(plan::ReproducePlan, records)
    g = plan.graph
    configs = Dict{String, String}()
    for st in plan.stages, item in st.items
        isempty(st.spec.config) && continue
        c = _config_of(st.spec, item)
        configs[c] = isfile(joinpath(g.workdir, c)) ? file_sha256(joinpath(g.workdir, c)) : "missing"
    end
    return Dict{String, Any}("project" => g.name, "graph" => relpath(g.path, g.workdir), "workdir" => g.workdir,
        "julia_project" => g.julia_project, "stamp" => plan.stamp, "commit" => plan.commit, "dirty" => plan.dirty,
        "site" => plan.site.name, "site_file" => plan.site.path, "backend" => String(plan.site.backend),
        "data_root" => plan.data_root, "results_root" => plan.results_root, "configs" => configs,
        "graph_sha256" => file_sha256(g.path), "code_trees" => plan.code_trees, "stages" => records)
end

# Atomically (a launch's reuse reads the manifests of launches that may still be writing theirs). The results-branch
# stage adds `results_branch` to the manifest (`results_branch_of_run`); a later rewrite keeps it.
function _write_manifest(plan::ReproducePlan, records)
    d = _manifest_dict(plan, records)
    if isfile(plan.manifest)
        branch = get(TOML.parsefile(plan.manifest), "results_branch", nothing)
        branch === nothing || (d["results_branch"] = branch)
    end
    _write_atomic(io -> TOML.print(io, d; sorted = true), plan.manifest)
    return plan.manifest
end

function _stage_record(plan::ReproducePlan, st::PlannedStage; state, job_id = "", log = "", tasks = Dict{String, String}(), markers = "")
    return Dict{String, Any}("key" => st.key, "items" => st.items, "after" => st.after, "command" => _shell_line(st.command),
                             "code_trees" => plan.code_trees,
                             "state" => state, "job_id" => job_id, "log" => log, "tasks" => tasks, "markers" => markers,
                             "config" => st.spec.config, "driver" => st.spec.driver, "sbatch" => st.spec.sbatch)
end

# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------

function _dependency(plan::ReproducePlan, st::PlannedStage; ids = nothing)
    deps = String[]
    for d in st.after
        p = plan.stages[findfirst(x -> x.spec.name == d, plan.stages)]
        p.reuse == "completed" && continue
        push!(deps, ids === nothing ? (isempty(p.reuse) ? "\$$d" : p.reuse) : ids[d])
    end
    return deps
end

"""
    print_plan(io, plan; missing = nothing, unpublished = nothing)

The job graph of a launch, as the dry run prints it: the commit, the roots, the site, then per stage its items,
dependencies, the driver's own command line (when the stage names a driver and config), the command the backend runs
(on a Slurm site the sbatch line, a dependency on a stage of this launch written `\$<stage>`), and whether an earlier
run is reused; then the missing inputs, and the inputs of the tier [`UNPUBLISHED_TIER`](@ref), present or ABSENT
(`unpublished`, from [`missing_inputs`](@ref)).
"""
function print_plan(io::IO, plan::ReproducePlan; missing = nothing, unpublished = nothing)
    g, site = plan.graph, plan.site
    println(io, "reproduce $(g.name) on $(site.name) ($(site.backend)): $(length(plan.stages)) stage(s)")
    println(io, "  commit   $(plan.commit)$(plan.dirty ? " (dirty: uncommitted changes to tracked files)" : "")")
    println(io, "  graph    $(relpath(g.path, g.workdir))  (workdir $(g.workdir))")
    println(io, "  site     $(site.path)")
    println(io, "  data     $(isempty(plan.data_root) ? "<unset: $(g.data_env)>" : plan.data_root)  ($(g.data_env))")
    println(io, "  results  $(isempty(plan.results_root) ? "<unset: $(g.results_env)>" : plan.results_root)  ($(g.results_env))")
    println(io, "  manifest $(plan.manifest)")
    println(io, "  markers  $(first(splitext(plan.manifest)))/<stage>/task-<t>.toml  (each task's completion marker: status and reuse read them)")
    println(io, "  code     $(length(plan.code_trees)) tree(s): $(join(sort!(collect(keys(plan.code_trees))), ", "))",
            isempty(plan.dirty_code) ? "" : "  DIRTY: $(join(plan.dirty_code, ", ")) " *
                (scratch_run(plan.base_env) ? "(a scratch launch: run with a warning)" : "(a launch is refused)"),
            isempty(plan.code_paths) ? "  NO PROJECT CODE: no `src`, no [project] code (a launch is refused)" : "")
    isempty(site.setup) || println(io, "  setup    $(site.setup)")
    for (i, st) in enumerate(plan.stages)
        s = st.spec
        arr = s.is_array ? "  array of $(length(st.items)): $(join(st.items, " "))" : ""
        println(io, "\n[$i] $(s.name)$(arr)")
        isempty(s.what) || println(io, "    what     $(s.what)")
        println(io, "    after    $(isempty(st.after) ? "—" : join(st.after, ", "))")
        println(io, "    key      $(st.key)", st.reuse == "completed" ? "  REUSED (completed in an earlier launch)" :
                                            !isempty(st.reuse) ? "  IN FLIGHT (job $(st.reuse) of an earlier launch)" :
                                            isempty(st.rerun) ? "" : "  RERUN ($(st.rerun))")
        if !isempty(s.driver)
            cfg = isempty(s.config) ? "" : " --config=$(replace(s.config, "{item}" => "<item>"))"
            println(io, "    driver   julia --project=$(g.julia_project) $(s.driver)$cfg")
        end
        envs = sort([k => v for (k, v) in st.env if !(k in (g.data_env, g.results_env))]; by = first)
        envline = join(("$(k)=$(_shell_quote(v))" for (k, v) in envs), " ")
        if site.backend === :slurm
            st.reuse == "completed" && continue
            deps = _dependency(plan, st)                       # `$<stage>` placeholders: shown as they are, never quoted
            line = _shell_line(st.command)
            isempty(deps) || (line = replace(line, "sbatch --parsable" => "sbatch --parsable --dependency=afterok:" * join(deps, ":"); count = 1))
            println(io, "    submit   ", isempty(envline) ? "" : envline * " ", line)
        else
            task = s.is_array ? "SLURM_ARRAY_TASK_ID={0..$(length(st.items) - 1)} " : ""
            println(io, "    run      ", task, isempty(envline) ? "" : envline * " ", _shell_line(st.command))
        end
    end
    if missing !== nothing
        println(io)
        print_missing(io, missing; data_root = plan.data_root, data_env = g.data_env)
    end
    if unpublished !== nothing && !isempty(unpublished)
        println(io, "unpublished inputs: $(length(unpublished)) (tier $UNPUBLISHED_TIER: never fetched, kept on the machine)")
        for u in sort(unpublished; by = u -> u.path)
            println(io, u.present ? "present" : "ABSENT ", '\t', u.path, '\t', join(u.used_by, "; "))
        end
    end
end

"""
    print_missing(io, rows; data_root, data_env)

The missing-input report ([`missing_inputs`](@ref)): one tab-separated line `path  tier  used by` per missing path.
"""
function print_missing(io::IO, rows; data_root::AbstractString, data_env::AbstractString = "VGTK_DATA_ROOT")
    if isempty(rows)
        println(io, "inputs   every input present under $(data_root)")
        return
    end
    println(io, "missing inputs: $(length(rows)) under $(isempty(data_root) ? "<unset: $data_env>" : data_root)")
    println(io, "path\ttier\tused by")
    for r in rows
        println(io, r.path, '\t', r.tier, '\t', join(r.used_by, "; "))
    end
end

# ---------------------------------------------------------------------------
# The backends
# ---------------------------------------------------------------------------

# The argv a stage runs: its program looked up on the stage's own PATH (a site's env may set it), after the site's setup.
function _with_setup(site::Site, argv, env)
    isempty(site.setup) || return ["bash", "-c", site.setup * "\nexec \"\$@\"", "bash", argv...]
    occursin('/', argv[1]) && return argv
    p = _which(argv[1], env)
    return p === nothing ? argv : [p; argv[2:end]]
end

# The `#SBATCH` directives of a launcher: the comment lines sbatch reads, those before its first command.
function _sbatch_directives(text::AbstractString)
    out = String[]
    for (i, l) in enumerate(eachline(IOBuffer(text)))
        t = strip(l)
        (i == 1 && startswith(t, "#!")) && continue
        isempty(t) && continue
        startswith(t, "#") || break
        startswith(t, "#SBATCH") && push!(out, String(t))
    end
    return out
end

const _MARKER_WRAPPER = raw"""#!/bin/bash
@@HEADER@@
# Written by VisualGeometryToolkitEval's runner (src/reproduce/runner.jl) for stage @@STAGE@@ of the launch
# @@MANIFEST@@: it runs the stage and then writes this task's completion marker, task-<array task>.toml in
# @@DIR@@, atomically (a temporary file in that directory renamed into place): its state (completed or failed), exit
# status, start and end, job id, commit and run key. A TERM (scancel, the time limit) is forwarded to the stage and
# recorded as failed; a signal the launcher's own --signal directive asks for is forwarded to the stage and recorded,
# its state then the stage's exit status; a job killed outright (a node failure, a KILL) writes none, which status
# reports as unknown. The launcher's copy is run through its interpreter, so a noexec results root cannot stop it.
_vgtk_dir=@@DIR@@
_vgtk_task=${SLURM_ARRAY_TASK_ID:-0}
_vgtk_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
_vgtk_marker() {
  local tmp
  mkdir -p "$_vgtk_dir" && tmp=$(mktemp "$_vgtk_dir/.task-$_vgtk_task.XXXXXX") && chmod 644 "$tmp" || return 1
  {
    printf 'stage = "%s"\n' @@STAGE@@
    printf 'task = %d\n' "$_vgtk_task"
    printf 'job_id = "%s"\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
    printf 'slurm_job_id = "%s"\n' "${SLURM_JOB_ID:-}"
    printf 'commit = "%s"\n' @@COMMIT@@
    printf 'key = "%s"\n' @@KEY@@
    printf 'restart_count = %d\n' "${SLURM_RESTART_COUNT:-0}"
    printf 'host = "%s"\n' "$(hostname)"
    printf 'start = "%s"\n' "$_vgtk_start"
    printf 'end = "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'exit_status = %d\n' "$2"
    printf 'state = "%s"\n' "$1"
    [ -z "$3" ] || printf 'signal = "%s"\n' "$3"
  } > "$tmp" && mv -f "$tmp" "$_vgtk_dir/task-$_vgtk_task.toml" || { rm -f "$tmp"; return 1; }
}
_vgtk_signal=
_vgtk_forwarded=
@@BODY@@ &
_vgtk_child=$!
trap '_vgtk_signal=TERM; kill -TERM "$_vgtk_child" 2>/dev/null' TERM
@@TRAPS@@
wait "$_vgtk_child"; _vgtk_rc=$?
while kill -0 "$_vgtk_child" 2>/dev/null; do wait "$_vgtk_child"; _vgtk_rc=$?; done
# a trap interrupts wait (status 128 + the signal's number): when the stage exited before the loop looked again, its
# own status is still to be collected (127: none left to collect, the status above is the stage's)
wait "$_vgtk_child" 2>/dev/null; _vgtk_r=$?; [ "$_vgtk_r" -eq 127 ] || _vgtk_rc=$_vgtk_r
if [ "$_vgtk_rc" -eq 0 ] && [ -z "$_vgtk_signal" ]; then
  _vgtk_marker completed 0 "$_vgtk_forwarded"
else
  _vgtk_marker failed "$_vgtk_rc" "${_vgtk_signal:-$_vgtk_forwarded}"
fi
exit "$_vgtk_rc"
"""

# The signals a launcher's `#SBATCH --signal=[{R|B}:]<sig>[@<time>]` directives ask Slurm to send (names without
# `SIG`, or numbers), TERM aside: the wrapper, which now receives them in the launcher's place, forwards each.
function _directive_signals(header)
    out = String[]
    for d in header
        m = match(r"^#SBATCH\s+--signal(?:=|\s+)(?:[RB]+:)?(?:SIG)?([A-Za-z0-9]+)"i, d)
        m === nothing && continue
        sig = uppercase(String(something(m.captures[1])))
        sig in ("TERM", "15") || sig in out || push!(out, sig)
    end
    return out
end

# The command line that runs a launcher's copy `snap` through its interpreter (its `#!` line: the program and at
# most one argument, as the kernel reads it; bash without one), never by executing the file itself: a results root
# mounted noexec refuses that.
function _interpreted(snap::AbstractString, text::AbstractString)
    first_line = isempty(text) ? "" : first(eachline(IOBuffer(text)))
    startswith(first_line, "#!") || return "/bin/bash " * _shell_quote(snap)
    parts = split(strip(first_line[3:end]), r"\s+"; limit = 2, keepempty = false)
    isempty(parts) && return "/bin/bash " * _shell_quote(snap)
    return join(_shell_quote.([String.(parts); snap]), " ")
end

# The script a Slurm stage is submitted as, written into its marker directory `dir`: the launcher's #SBATCH
# directives, then the stage (a copy of the launcher, taken now as sbatch would take it and run through its
# interpreter; or the command of a `command` stage, `wrap`) and its completion marker. Returns its path.
function _marker_wrapper(plan::ReproducePlan, st::PlannedStage, dir::AbstractString; wrap::AbstractString = "")
    g, s = plan.graph, st.spec
    mkpath(dir)
    header, body = if isempty(s.sbatch)
        String[], wrap
    else
        src = joinpath(g.workdir, s.sbatch)
        snap = joinpath(dir, basename(s.sbatch))
        cp(src, snap; force = true)
        chmod(snap, 0o644)                                 # read, never executed (`_interpreted`)
        text = read(src, String)
        _sbatch_directives(text), _interpreted(snap, text)
    end
    traps = join(("trap '_vgtk_forwarded=$sig; kill -$sig \"\$_vgtk_child\" 2>/dev/null' $sig" for sig in _directive_signals(header)), "\n")
    text = _MARKER_WRAPPER
    for (k, v) in ("@@HEADER@@" => join(header, "\n"), "@@MANIFEST@@" => plan.manifest, "@@STAGE@@" => _shell_quote(s.name),
                   "@@DIR@@" => _shell_quote(dir), "@@COMMIT@@" => _shell_quote(plan.commit), "@@KEY@@" => _shell_quote(st.key),
                   "@@BODY@@" => body, "@@TRAPS@@" => traps)
        text = replace(text, k => v)
    end
    path = joinpath(dir, "marker_wrapper.sbatch")
    write(path, text)
    return path
end

# The sbatch argv of a stage through its marker wrapper: the wrapper in place of the launcher, or of `--wrap=<line>`.
# A job given no name is named as a prefix would name it, without the prefix (`_base_job_name`): Slurm would otherwise
# name it after the script submitted, the wrapper, `marker_wrapper.sbatch` for every such stage. The run key, taken of
# the planned command, is unchanged.
function _through_wrapper(plan::ReproducePlan, st::PlannedStage, c::Vector{String}, dir::AbstractString)
    c = copy(c)
    if isempty(st.spec.sbatch)
        i = findfirst(a -> startswith(a, "--wrap="), c)
        c[i] = _marker_wrapper(plan, st, dir; wrap = c[i][(length("--wrap=") + 1):end])
    else
        @assert c[end] == st.spec.sbatch
        c[end] = _marker_wrapper(plan, st, dir)
    end
    any(a -> a == "-J" || startswith(a, "--job-name"), c) ||
        insert!(c, length(c), "--job-name=$(_base_job_name(plan.graph, st.spec))")
    return c
end

# A run of record runs committed code: its results name it by its trees, which uncommitted changes are not in (the
# one refusal, the drivers' records too). A scratch launch (`VGTK_SCRATCH=1`) runs it with a warning, as a driver's
# scratch records do: its trees carry `+uncommitted`, so its stages are never reused for committed code.
function _refuse_dirty_code(plan::ReproducePlan)
    identity = (; dirty = plan.dirty_code, project_paths = plan.code_paths)
    if scratch_run(plan.base_env) && !isempty(plan.dirty_code)
        @warn "a scratch launch runs uncommitted code (VGTK_SCRATCH=1): its stages are never reused for committed code" uncommitted = join(plan.dirty_code, ", ")
        identity = (; dirty = String[], project_paths = plan.code_paths)    # the project's code must still be identified
    end
    return refuse_uncommitted_code(identity)
end

# The line a backend prints before running a stage an earlier launch ran under other code.
_print_rerun(io, st::PlannedStage) = isempty(st.rerun) || println(io, rpad(st.spec.name, 32), " rerun: ", st.rerun)

"""
    submit_slurm!(plan; io = stdout) -> Dict{String, Any}

Submits the plan's stages in order with sbatch, each after the jobs of the stages it depends on
(`--dependency=afterok`; a GPU array also after the GPU array submitted before it, `afterany`, when the site sets
`gpu_limit`); a reused stage is not submitted (a completed one is no dependency, one in flight is depended on). Each
stage is submitted through a wrapper that runs it and writes each task's completion marker
([`stage_markers_dir`](@ref)): the stage's own launcher, copied as submitted, with its `#SBATCH` directives (a
signal its `--signal` directive asks for is forwarded to it), named `<project>-<stage>` when neither the stage nor its
launcher names it. sbatch runs without the `SLURM_*` variables of the process that launches it (`SLURM_CONF` kept):
launched from inside a Slurm job, that job's array task and job id would otherwise reach every stage's job.
Returns the manifest records; the manifest is written after every submission, so a failed sbatch leaves the record of
what was submitted.
"""
function submit_slurm!(plan::ReproducePlan; io::IO = stdout)
    _refuse_dirty_code(plan)
    g, site = plan.graph, plan.site
    # The launching process's own job (a launcher or a test suite run under sbatch) is not the stage's: its SLURM_*
    # variables, exported to the job by sbatch's default --export=ALL, would give a non-array stage the parent's array
    # task and job id (its marker then written for the wrong task, under the wrong job). SLURM_CONF, where the site
    # sets it, tells sbatch which cluster to submit to.
    base_env = Dict(k => v for (k, v) in plan.base_env if !startswith(k, "SLURM_") || k == "SLURM_CONF")
    mkpath(joinpath(g.workdir, "logs")); mkpath(plan.results_root)
    ids, records = Dict{String, String}(), Dict{String, Any}()
    last_gpu = ""
    for st in plan.stages
        s = st.spec
        if st.reuse == "completed"
            records[s.name] = _stage_record(plan, st; state = "reused")
            println(io, rpad(s.name, 32), " reused (completed earlier)")
            continue
        elseif !isempty(st.reuse)
            ids[s.name] = st.reuse
            records[s.name] = _stage_record(plan, st; state = "in flight", job_id = st.reuse, markers = st.reuse_markers)
            println(io, rpad(s.name, 32), " ", st.reuse, "  (in flight, an earlier launch)")
            continue
        end
        _print_rerun(io, st)
        deps = _dependency(plan, st; ids)
        c = copy(st.command)
        dep = isempty(deps) ? "" : "afterok:" * join(deps, ":")
        if _stage_gpu(s) && s.is_array && site.gpu_limit > 0 && !isempty(last_gpu)
            dep = (isempty(dep) ? "" : dep * ",") * "afterany:" * last_gpu
        end
        isempty(dep) || insert!(c, 3, "--dependency=$dep")
        markers = stage_markers_dir(plan.manifest, s.name)
        c = _through_wrapper(plan, st, c, markers)
        out = IOBuffer(); err = IOBuffer()
        e = merge(base_env, st.env)
        ok = try
            success(pipeline(Cmd(Cmd(_with_setup(site, c, e)); env = e, dir = g.workdir); stdout = out, stderr = err))
        catch ex
            ex isa Base.IOError || rethrow()
            print(err, sprint(showerror, ex)); false
        end
        txt = strip(String(take!(out)))
        if !ok || isempty(txt)
            records[s.name] = _stage_record(plan, st; state = "submit failed", markers)
            _write_manifest(plan, records)
            error("sbatch of stage $(s.name) failed: $(strip(String(take!(err))))\n  $(_shell_line(c))")
        end
        id = String(first(split(txt, ';')))
        ids[s.name] = id
        _stage_gpu(s) && s.is_array && (last_gpu = id)
        records[s.name] = _stage_record(plan, st; state = "submitted", job_id = id, markers)
        _write_manifest(plan, records)
        println(io, rpad(s.name, 32), " ", id, s.is_array ? "  (array 0-$(length(st.items) - 1): $(join(st.items, " ")))" : "")
    end
    _write_manifest(plan, records)
    return records
end

"""
    run_local!(plan; io = stdout) -> Dict{String, Any}

Runs the plan's stages in order on this machine, each item of an array in turn (`SLURM_ARRAY_TASK_ID` set, so a
project's sbatch launcher runs unchanged under `bash`), its output in `logs/<project>_<stage>[_<task>].log` under the
workdir. A completed stage of an earlier launch is skipped, as is every task it completed; the first failing task
stops the run (a later stage depends on it). The manifest and the task's completion marker
([`stage_markers_dir`](@ref)) are written after every task. Returns the records.

A local run is no Slurm job, whatever the launching process is: every `SLURM_*` variable of the environment the
plan was made in is dropped, and a stage sees only the ones this backend sets (`SLURM_SUBMIT_DIR`, an array's
`SLURM_ARRAY_TASK_ID`, the site's `SLURM_CPUS_PER_TASK`) — launched from inside a Slurm job (a test suite's array
task), the job's own task id and submit directory would otherwise reach every stage.
"""
function run_local!(plan::ReproducePlan; io::IO = stdout)
    _refuse_dirty_code(plan)
    g, site = plan.graph, plan.site
    base_env = Dict(k => v for (k, v) in plan.base_env if !startswith(k, "SLURM_"))
    logs = mkpath(joinpath(g.workdir, "logs")); mkpath(plan.results_root)
    records = Dict{String, Any}()
    for st in plan.stages
        s = st.spec
        if st.reuse == "completed"
            records[s.name] = _stage_record(plan, st; state = "reused")
            println(io, rpad(s.name, 32), " reused (completed earlier)")
            continue
        end
        _print_rerun(io, st)
        tasks = Dict{String, String}()
        markers = stage_markers_dir(plan.manifest, s.name)
        for (i, item) in enumerate(st.items)
            t = i - 1
            log = joinpath(logs, s.is_array ? "$(g.name)_$(s.name)_$t.log" : "$(g.name)_$(s.name).log")
            e = merge(base_env, st.env)
            s.is_array && (e["SLURM_ARRAY_TASK_ID"] = string(t))
            t0, start = time(), _utc_now()
            rc = open(log, "w") do f
                try
                    run(pipeline(Cmd(Cmd(_with_setup(site, st.command, e)); env = e, dir = g.workdir, ignorestatus = true);
                                 stdout = f, stderr = f)).exitcode
                catch ex
                    ex isa Base.IOError || rethrow()
                    println(f, "ERROR: ", sprint(showerror, ex)); 127
                end
            end
            ok = rc == 0
            tasks[string(t)] = ok ? "completed" : "failed"
            marker = Dict{String, Any}("stage" => s.name, "task" => t, "job_id" => "", "commit" => plan.commit, "key" => st.key,
                                       "host" => gethostname(), "start" => start, "end" => _utc_now(), "exit_status" => rc,
                                       "restart_count" => 0, "state" => tasks[string(t)])
            _write_atomic(io -> TOML.print(io, marker; sorted = true), joinpath(markers, "task-$t.toml"))
            println(io, rpad(s.name, 32), " ", s.is_array ? "task $t ($item) " : "", ok ? "ok" : "FAILED",
                    "  $(round(time() - t0; digits = 1)) s  $(relpath(log, g.workdir))")
            records[s.name] = _stage_record(plan, st; state = ok ? (i == length(st.items) ? "completed" : "running") : "failed", log, tasks, markers)
            _write_manifest(plan, records)
            if !ok
                err = _first_error(log)
                error("stage $(s.name)$(s.is_array ? " task $t ($item)" : "") failed; log $log" * (isempty(err) ? "" : "\n  $err"))
            end
        end
    end
    _write_manifest(plan, records)
    return records
end

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

# The first line of a log that reads as an error; "" when there is none (or no log).
function _first_error(log::AbstractString)
    isfile(log) || return ""
    for l in eachline(log)
        occursin(r"ERROR|Error|error:|FAILED|Killed|CANCELLED|DUE TO TIME LIMIT", l) && return String(strip(l))
    end
    return ""
end

# The log of a task of an sbatch stage: the launcher's `#SBATCH --output=` pattern with %j/%A/%a/%x filled.
function _slurm_log(g::StageGraph, s::StageSpec, id::AbstractString, task::Integer)
    pattern = if isempty(s.sbatch)
        "logs/$(g.name)_$(s.name)_%j.log"
    else
        m = match(r"^#SBATCH\s+--output=(\S+)"m, read(joinpath(g.workdir, s.sbatch), String))
        m === nothing ? "slurm-%j.out" : m.captures[1]
    end
    p = replace(pattern, "%A" => id, "%a" => string(task), "%x" => "*" * _base_job_name(g, s),   # a site's job-name prefix comes first
                "%j" => s.is_array ? "*" : id)
    if occursin('*', p)
        dir = joinpath(g.workdir, dirname(p))
        re = Regex("^" * replace(basename(p), "." => "\\.", "*" => ".*") * "\$")
        hits = isdir(dir) ? filter(f -> occursin(re, f), readdir(dir)) : String[]
        return isempty(hits) ? joinpath(g.workdir, p) : joinpath(dir, first(hits))
    end
    return joinpath(g.workdir, p)
end

"""
    reproduce_status(manifest; queue_states = nothing, job_states = nothing, env = ENV) -> Vector{NamedTuple}

The state of every stage of the launch a run manifest records: `(; stage, job_id, state, log, error)`. The state is
[`stage_run_state`](@ref)'s, from the stages' completion markers and the queue (`queue_states(ids)`, by default
[`slurm_queue_states`](@ref)), with its tasks counted by state (`"failed (completed 2, failed 1)"`); a task with no
marker that is no longer queued is `unknown`, and where Slurm keeps accounting (`job_states(ids)`, by default
[`slurm_job_states`](@ref): empty when sacct fails) its accounting state is added for that task, as information
only. The log is the first failed or unknown task's (else the first's), with its first error line.
"""
function reproduce_status(manifest::AbstractString; queue_states = nothing, job_states = nothing, env = ENV)
    m = TOML.parsefile(manifest)
    g = load_stage_graph(joinpath(m["workdir"], m["graph"]))
    recs = m["stages"]
    ids = unique(String[r["job_id"] for (_, r) in recs if !isempty(get(r, "job_id", ""))])
    queue = isempty(ids) ? Dict{String, Dict{Int, String}}() : _queue_states(queue_states, ids, env)
    rows = NamedTuple[]
    acct = nothing                                          # sacct, asked once and only for a task without a marker
    for s in g.stages
        haskey(recs, s.name) || continue
        r = recs[s.name]
        id = get(r, "job_id", "")
        run_state = stage_run_state(r, queue)
        counts = Dict{String, Int}()
        for v in values(run_state.tasks)
            counts[v] = get(counts, v, 0) + 1
        end
        state = get(r, "state", "") == "reused" ? "reused (completed in an earlier launch)" : run_state.state * (isempty(counts) ? "" : " (" * join(("$(k) $(v)" for (k, v) in sort(collect(counts))), ", ") * ")")
        if !isempty(id) && run_state.state == "unknown"
            acct === nothing && (acct = _job_states(job_states, ids, env))
            a = get(acct, id, String[])
            isempty(a) || (state *= " [sacct: " * join(sort(unique(a)), ", ") * "]")
        end
        bad = sort([t for (t, v) in run_state.tasks if v in ("failed", "unknown")])
        if isempty(id)
            log = get(r, "log", "")                         # a local run stops at its first failing task: its log
        else
            log = _slurm_log(g, s, id, isempty(bad) ? 0 : first(bad))
        end
        push!(rows, (; stage = s.name, job_id = id, state, log, error = isempty(log) ? "" : _first_error(log)))
    end
    return rows
end

# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

function _cli_value(args, key)
    v = cli_option(args, key, nothing)
    v !== nothing && return v
    i = findfirst(==("--$key"), args)
    (i === nothing || i == length(args)) && return nothing
    return args[i + 1]
end

"""
    REPRODUCE_USAGE

The runner's usage text, printed by [`reproduce_main`](@ref) when `--graph` or `--site` is missing.
"""
const REPRODUCE_USAGE = """
usage: reproduce [run | status] --graph=<stage graph .toml> --site=<name | path to a site .toml>
                 [--only=<stage>,<stage>…] [--no-deps] [--dry-run] [--data-root=<dir>] [--results-root=<dir>]
                 [--allow-missing] [--manifest=<run manifest>] [--job-name-prefix=<prefix>]
  run (default)  every default stage of the graph (with --only the named stages and, without --no-deps, what they
                 run after); finished stages are reused by their run keys and completion markers while their
                 code trees are unchanged (a changed one is named and its stage reruns); uncommitted changes to the
                 code refuse the launch
  --job-name-prefix  put before every Slurm job's name (default: the site's [slurm] job_name_prefix); no run key
                 changes
  --dry-run      the job graph (stages, items, dependencies, commands) and the missing inputs; nothing runs
  status         the last launch's stages (or --manifest's): state (from the completion markers and the queue),
                 log, first error line
Exit codes: 0 done (or submitted), 2 usage, a config error or uncommitted code, 3 missing inputs (nothing ran),
4 a stage failed (or squeue failed, or a stage is in flight under other code)."""

"""
    reproduce_main(args = ARGS; io = stdout, env = ENV, queue_states = nothing, job_states = nothing) -> Int

The command line of the runner ([`REPRODUCE_USAGE`](@ref)); returns the exit code. A launch checks the inputs first
and runs nothing when one is missing ([`missing_inputs`](@ref), exit 3; `--allow-missing` runs anyway), writes the run
manifest and then submits (Slurm) or runs (local) the stages. `queue_states` and `job_states` stand in for squeue
and sacct ([`slurm_queue_states`](@ref), [`slurm_job_states`](@ref)).
"""
function reproduce_main(args::AbstractVector{<:AbstractString} = ARGS; io::IO = stdout, env = ENV,
                        queue_states = nothing, job_states = nothing)
    # the subcommand: a bare word anywhere (a launcher puts its own options first), never the value of a spaced option
    cmd = any(i -> args[i] == "status" && (i == 1 || !(args[i - 1] in ("--only", "--graph", "--site", "--manifest",
              "--data-root", "--results-root", "--job-name-prefix"))), eachindex(args)) ? "status" : "run"
    try
        gpath, sname = _cli_value(args, "graph"), _cli_value(args, "site")
        (gpath === nothing || sname === nothing) && (println(io, REPRODUCE_USAGE); return 2)
        g = load_stage_graph(gpath)
        site = load_site(sname; dirs = isempty(g.sites_dir) ? String[] : [g.sites_dir])
        if cmd == "status"
            m = _cli_value(args, "manifest")
            if m === nothing
                _, results = resolve_roots(g, site; results_root = _cli_value(args, "results-root"), env)
                ms = run_manifests(joinpath(results, g.manifest_dir))
                isempty(ms) && (println(io, "no launch recorded under $(joinpath(results, g.manifest_dir))"); return 2)
                m = last(ms)
            end
            println(io, "launch $m")
            for r in reproduce_status(m; queue_states, job_states, env)
                println(io, rpad(r.stage, 32), " ", rpad(r.job_id, 10), " ", r.state)
                isempty(r.log) || println(io, " "^33, "log   ", r.log)
                isempty(r.error) || println(io, " "^33, "error ", r.error)
            end
            return 0
        end
        only = something(_cli_value(args, "only"), "")
        plan = plan_reproduction(g, site; only = String.(split(only, ','; keepempty = false)), deps = !("--no-deps" in args),
                                 data_root = _cli_value(args, "data-root"), results_root = _cli_value(args, "results-root"),
                                 job_name_prefix = _cli_value(args, "job-name-prefix"), env, queue_states)
        unpub = NamedTuple[]
        miss = missing_inputs(g, [st.spec.name for st in plan.stages]; data_root = plan.data_root, env, site = site.name,
                              unpublished = unpub)
        if "--dry-run" in args
            print_plan(io, plan; missing = miss, unpublished = unpub)
            println(io, "\ndry run: nothing submitted or run")
            return 0
        end
        isempty(plan.results_root) && (println(io, "no results root: set $(g.results_env), --results-root or [sites.$(site.name)] results_root"); return 2)
        isempty(plan.data_root) && (println(io, "no data root: set $(g.data_env), --data-root or [sites.$(site.name)] data_root"); return 2)
        if !isempty(miss)
            print_missing(io, miss; data_root = plan.data_root, data_env = g.data_env)
            "--allow-missing" in args || (println(io, "nothing ran (--allow-missing runs anyway)"); return 3)
        end
        println(io, "reproduce $(g.name) on $(site.name): commit $(plan.commit)$(plan.dirty ? " (dirty)" : ""), manifest $(plan.manifest)")
        site.backend === :slurm ? submit_slurm!(plan; io) : run_local!(plan; io)
        println(io, "manifest $(plan.manifest)")
        return 0
    catch e
        e isa ArgumentError && (println(io, "reproduce: ", e.msg); return 2)
        e isa ErrorException && (println(io, "reproduce: ", e.msg); return 4)
        rethrow()
    end
end
