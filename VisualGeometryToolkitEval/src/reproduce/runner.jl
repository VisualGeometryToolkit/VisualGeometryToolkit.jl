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
# DEPENDENCY-FREE BY CONTRACT (Base and the stdlibs TOML, SHA and Dates; `cli_option` comes from the includer, src/cli.jl):
# the Reproduce module includes it, AND a project's launcher includes src/cli.jl and this file BY PATH from a bare `julia`,
# before the project's environment exists — the setup stage instantiates it, inside the scheduler. One copy for both.
# The results-branch stage runs in the project's environment and calls `results_branch_of_run`
# (reproduce/run_results_branch.jl), which is not dependency-free and lives in the module only.
# =============================================================================
using TOML: TOML
using SHA: sha256
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
    "slurm" => ("partition", "gres", "cpus_per_task", "mem_per_cpu", "gpu_limit", "extra"),
    "local" => ("cuda_visible_devices", "cpus_per_task"),
    "env"   => nothing)                                    # any variable

"""
    Site

A machine a reproduction runs on, read by [`load_site`](@ref) from a site profile: its `backend` (`:slurm`, sbatch
arrays chained by `--dependency=afterok`; `:local`, every stage in order in this process's machine), the Slurm
options every job gets (`partition`, a GPU stage's `gres`, `cpus_per_task`, `mem_per_cpu`, `gpu_limit` GPU array
tasks at a time, `extra` options), the local `cuda_visible_devices` and `cpus_per_task`, a shell `setup` run before
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
    return Site(get(s, "name", first(splitext(basename(path)))), path, Symbol(backend), get(s, "description", ""), get(s, "setup", ""),
                get(sl, "partition", ""), get(sl, "gres", ""), cpus, get(sl, "mem_per_cpu", ""), get(sl, "gpu_limit", 0),
                String.(get(sl, "extra", String[])), string(get(lo, "cuda_visible_devices", "")),
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
                     "gpu", "default", "time", "array", "inputs_from")
const _INPUTS_FROM_KEYS = ("config", "key", "base")
const _ARRAY_KEYS = ("items", "from", "select_env", "max_time", "time_margin_s")
const _SITE_ROOT_KEYS = ("data_root", "results_root", "env")

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
                     get(a, "max_time", ""), get(a, "time_margin_s", 0), !isempty(a), from)
end

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
every stage runs — their committed trees and uncommitted changes key every stage, default the whole repository, so a
change elsewhere, the paper's text, reuses the finished stages), `[sites.<site>]` (`data_root`, `results_root` and an `env`
table: where this project's data and results are on that machine), one `[[stage]]` per stage ([`StageSpec`](@ref)),
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
    missing_inputs(g, stages; data_root, env = ENV) -> Vector{NamedTuple}

Every input of the selected `stages` that is not there, as data, before anything runs: one row
`(; path, tier, used_by)` per missing path (`used_by` every stage and config that reads it). A stage's inputs are its
configs of record (one per array item) and their `[inputs.<id>]` tables with a `path` (relative to `data_root`) and a
`tier`; a missing config is a row of tier `"config of record"`. With no data root every data input is missing (tier
as declared).
"""
function missing_inputs(g::StageGraph, stages::AbstractVector{<:AbstractString}; data_root::AbstractString, env = ENV)
    rows = Dict{String, Tuple{String, Vector{String}}}()
    add!(path, tier, user) = (r = get!(rows, path, (tier, String[])); user in r[2] || push!(r[2], user))
    function check!(cfg, user)
        p = joinpath(g.workdir, cfg)
        isfile(p) || (add!(cfg, "config of record", user); return)
        for (_, t) in get(TOML.parsefile(p), "inputs", Dict{String, Any}())
            (t isa AbstractDict && haskey(t, "path")) || continue
            (!isempty(data_root) && ispath(joinpath(data_root, t["path"]))) && continue
            add!(t["path"], get(t, "tier", ""), user)
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
    return [(; path, tier = r[1], used_by = r[2]) for (path, r) in sort(collect(rows); by = first)]
end

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

"""
    PlannedStage

One stage of a launch: its spec, its `items`, the selected stages it runs `after`, its run `key` (the commit and
the code (the graph's `code` paths), every file and option the stage's jobs read: a stage of the same key that finished before is reused), the
environment its jobs get beyond the caller's (`env`), its `command` (on a Slurm site the sbatch argv without the
dependency; on a local site the argv run once per item), its `--time` in minutes (`minutes`, 0 = the launcher's),
and `reuse` (`""`; `"completed"`, finished in an earlier launch; or the job id of an earlier launch's job still in
flight, which this launch depends on instead of submitting again).
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
end

"""
    ReproducePlan

A launch of a stage graph on a site: the graph, the site, the code `commit` (and `dirty`), the `data_root` and
`results_root`, the `stamp` of the launch, the path of its run `manifest`, its `stages` in dependency order, and the
environment the launch was planned in (`base_env`: the stages' environment is it plus each stage's own).
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
end

_sha_file(path) = bytes2hex(open(sha256, path))
_sha_text(s) = bytes2hex(sha256(s))

# The commit, whether tracked files changed, and the code's state: the committed trees of the `code` paths (the
# whole tree when none) and their uncommitted changes.
function _git_state(dir, code)
    run_git(args...) = try readchomp(pipeline(`git -C $dir $args`; stderr = devnull)) catch; "" end
    commit = run_git("rev-parse", "HEAD")
    isempty(commit) && return ("(not a git checkout)", false, "")
    dirty = !isempty(run_git("status", "--porcelain", "--untracked-files=no"))
    trees = isempty(code) ? [run_git("rev-parse", "HEAD^{tree}")] : [run_git("rev-parse", "HEAD:./" * c) for c in code]
    diff = isempty(code) ? run_git("diff", "HEAD") : run_git("diff", "HEAD", "--", code...)
    return (commit, dirty, _sha_text(join(trees, "\n") * "\n" * diff))
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

function _slurm_options(site::Site, s::StageSpec)
    o = String[]
    isempty(site.partition) || push!(o, "--partition=$(site.partition)")
    s.gpu && !isempty(site.gres) && push!(o, "--gres=$(site.gres)")
    site.cpus_per_task > 0 && push!(o, "--cpus-per-task=$(site.cpus_per_task)")
    isempty(site.mem_per_cpu) || push!(o, "--mem-per-cpu=$(site.mem_per_cpu)")
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

function _plan_command(g::StageGraph, site::Site, s::StageSpec, items, minutes, manifest, env)
    argv = _stage_argv(g, site, s, manifest, env)
    site.backend === :local && return isempty(s.sbatch) ? argv : ["bash", s.sbatch]
    c = ["sbatch", "--parsable"]
    append!(c, _slurm_options(site, s))
    jobname = isempty(s.job_name) && isempty(s.sbatch) ? "$(g.name)-$(s.name)" : s.job_name
    isempty(jobname) || push!(c, "--job-name=$jobname")
    if s.is_array
        push!(c, "--array=0-$(length(items) - 1)" * (s.gpu && site.gpu_limit > 0 ? "%$(site.gpu_limit)" : ""))
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
                      env = ENV, stamp = <now, UTC>, job_states = nothing) -> ReproducePlan

The launch of the selected stages ([`select_stages`](@ref)) of graph `g` on `site`, planned and nothing run: each
stage's items, dependencies, environment, command and run key, and whether an earlier launch's run of the same key is
reused (completed, or in flight on a Slurm site: `job_states(ids)` gives each job's task states, by default
[`slurm_job_states`](@ref) on `env`'s PATH; the earlier launches are the manifests under
`<results root>/<manifest_dir>`).
"""
function plan_reproduction(g::StageGraph, site::Site; only::AbstractVector{<:AbstractString} = String[], deps::Bool = true,
                           data_root = nothing, results_root = nothing, env = ENV,
                           stamp::AbstractString = Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS_sss"),
                           job_states = nothing)
    names = select_stages(g; only, deps)
    data, results = resolve_roots(g, site; data_root, results_root, env)
    commit, dirty, code = _git_state(g.workdir, g.code)
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
    for name in names
        s = g.stages[_stage(g, name)]
        items = stage_items(g, s; env)
        minutes = _array_minutes(g, s, items)
        e = merge(roots_env, site_env, s.env)
        isempty(s.select_env) || (e[s.select_env] = join(items, ","))
        if site.backend === :local
            e["SLURM_SUBMIT_DIR"] = g.workdir
            site.cpus_per_task > 0 && (e["SLURM_CPUS_PER_TASK"] = string(site.cpus_per_task))
            e["CUDA_VISIBLE_DEVICES"] = s.gpu ? site.cuda_visible_devices : ""
        end
        cmd = _plan_command(g, site, s, items, minutes, manifest, env)
        files = Dict{String, String}()
        for f in unique(filter(!isempty, [s.sbatch, s.driver, (_config_of(s, i) for i in items)...]))
            p = joinpath(g.workdir, f)
            files[f] = isfile(p) ? _sha_file(p) : "missing"
        end
        keyed = Dict{String, Any}("stage" => name, "code" => code, "files" => files, "items" => items,
                                  "env" => e, "command" => cmd, "site" => site.name)
        key = first(_sha_text(sprint(io -> TOML.print(io, keyed; sorted = true))), 16)
        push!(stages, PlannedStage(s, items, [d for d in s.after if d in names], key, e, cmd, minutes, ""))
    end
    plan = ReproducePlan(g, site, commit, dirty, data, results, stamp, manifest, stages, Dict{String, String}(env))
    _mark_reuse!(plan, ids -> _job_states(job_states, ids, env))
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

const _IN_FLIGHT = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")

"""
    slurm_job_states(ids; env = ENV) -> Dict{String, Vector{String}}

The state of every task of the Slurm jobs `ids` (`sacct`, found on `env`'s PATH), by job id: `["COMPLETED", "FAILED",
…]`. Empty when sacct is not there or knows none of them.
"""
function slurm_job_states(ids::AbstractVector{<:AbstractString}; env = ENV)
    out = Dict{String, Vector{String}}()
    sacct = _which("sacct", env)
    (isempty(ids) || sacct === nothing) && return out
    txt = try readchomp(pipeline(Cmd(`$sacct -n -X -P -j $(join(ids, ",")) --format=JobID,State`; env = Dict{String, String}(env));
                                 stderr = devnull)) catch; "" end
    for l in split(txt, '\n'; keepempty = false)
        id, st = split(l, '|'; limit = 2)
        push!(get!(out, String(first(split(id, '_'))), String[]), String(first(split(st))))
    end
    return out
end

# A stage of the same key in an earlier launch: completed (every task) → reused; in flight → depended on.
function _mark_reuse!(plan::ReproducePlan, job_states)
    isempty(plan.results_root) && return plan
    last = Dict{String, Dict{String, Any}}()
    for m in run_manifests(_manifest_dir(plan))
        for (_, r) in get(TOML.parsefile(m), "stages", Dict{String, Any}())
            haskey(r, "key") && (last[r["key"]] = r)
        end
    end
    isempty(last) && return plan
    ids = String[]
    for st in plan.stages
        haskey(last, st.key) || continue
        id = get(last[st.key], "job_id", "")
        isempty(id) || push!(ids, id)
    end
    states = isempty(ids) ? Dict{String, Vector{String}}() : job_states(ids)
    for st in plan.stages
        r = get(last, st.key, nothing)
        r === nothing && continue
        if get(r, "state", "") in ("completed", "reused")
            st.reuse = "completed"
        elseif !isempty(get(r, "job_id", ""))
            s = get(states, r["job_id"], String[])
            isempty(s) && continue
            all(==("COMPLETED"), s) ? (st.reuse = "completed") :
                (all(x -> x == "COMPLETED" || x in _IN_FLIGHT, s) && (st.reuse = r["job_id"]))
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
        configs[c] = isfile(joinpath(g.workdir, c)) ? _sha_file(joinpath(g.workdir, c)) : "missing"
    end
    return Dict{String, Any}("project" => g.name, "graph" => relpath(g.path, g.workdir), "workdir" => g.workdir,
        "julia_project" => g.julia_project, "stamp" => plan.stamp, "commit" => plan.commit, "dirty" => plan.dirty,
        "site" => plan.site.name, "site_file" => plan.site.path, "backend" => String(plan.site.backend),
        "data_root" => plan.data_root, "results_root" => plan.results_root, "configs" => configs,
        "graph_sha256" => _sha_file(g.path), "stages" => records)
end

function _write_manifest(plan::ReproducePlan, records)
    mkpath(dirname(plan.manifest))
    open(io -> TOML.print(io, _manifest_dict(plan, records); sorted = true), plan.manifest, "w")
    return plan.manifest
end

function _stage_record(st::PlannedStage; state, job_id = "", log = "", tasks = Dict{String, String}())
    return Dict{String, Any}("key" => st.key, "items" => st.items, "after" => st.after, "command" => _shell_line(st.command),
                             "state" => state, "job_id" => job_id, "log" => log, "tasks" => tasks,
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
    print_plan(io, plan; missing = nothing)

The job graph of a launch, as the dry run prints it: the commit, the roots, the site, then per stage its items,
dependencies, the driver's own command line (when the stage names a driver and config), the command the backend runs
(on a Slurm site the sbatch line, a dependency on a stage of this launch written `\$<stage>`), and whether an earlier
run is reused; then the missing inputs.
"""
function print_plan(io::IO, plan::ReproducePlan; missing = nothing)
    g, site = plan.graph, plan.site
    println(io, "reproduce $(g.name) on $(site.name) ($(site.backend)): $(length(plan.stages)) stage(s)")
    println(io, "  commit   $(plan.commit)$(plan.dirty ? " (dirty: uncommitted changes to tracked files)" : "")")
    println(io, "  graph    $(relpath(g.path, g.workdir))  (workdir $(g.workdir))")
    println(io, "  site     $(site.path)")
    println(io, "  data     $(isempty(plan.data_root) ? "<unset: $(g.data_env)>" : plan.data_root)  ($(g.data_env))")
    println(io, "  results  $(isempty(plan.results_root) ? "<unset: $(g.results_env)>" : plan.results_root)  ($(g.results_env))")
    println(io, "  manifest $(plan.manifest)")
    isempty(site.setup) || println(io, "  setup    $(site.setup)")
    for (i, st) in enumerate(plan.stages)
        s = st.spec
        arr = s.is_array ? "  array of $(length(st.items)): $(join(st.items, " "))" : ""
        println(io, "\n[$i] $(s.name)$(arr)")
        isempty(s.what) || println(io, "    what     $(s.what)")
        println(io, "    after    $(isempty(st.after) ? "—" : join(st.after, ", "))")
        println(io, "    key      $(st.key)", st.reuse == "completed" ? "  REUSED (completed in an earlier launch)" :
                                            isempty(st.reuse) ? "" : "  IN FLIGHT (job $(st.reuse) of an earlier launch)")
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

"""
    submit_slurm!(plan; io = stdout) -> Dict{String, Any}

Submits the plan's stages in order with sbatch, each after the jobs of the stages it depends on
(`--dependency=afterok`; a GPU array also after the GPU array submitted before it, `afterany`, when the site sets
`gpu_limit`); a reused stage is not submitted (a completed one is no dependency, one in flight is depended on).
Returns the manifest records; the manifest is written after every submission, so a failed sbatch leaves the record of
what was submitted.
"""
function submit_slurm!(plan::ReproducePlan; io::IO = stdout)
    g, site = plan.graph, plan.site
    mkpath(joinpath(g.workdir, "logs")); mkpath(plan.results_root)
    ids, records = Dict{String, String}(), Dict{String, Any}()
    last_gpu = ""
    for st in plan.stages
        s = st.spec
        if st.reuse == "completed"
            records[s.name] = _stage_record(st; state = "reused")
            println(io, rpad(s.name, 32), " reused (completed earlier)")
            continue
        elseif !isempty(st.reuse)
            ids[s.name] = st.reuse
            records[s.name] = _stage_record(st; state = "in flight", job_id = st.reuse)
            println(io, rpad(s.name, 32), " ", st.reuse, "  (in flight, an earlier launch)")
            continue
        end
        deps = _dependency(plan, st; ids)
        c = copy(st.command)
        dep = isempty(deps) ? "" : "afterok:" * join(deps, ":")
        if s.gpu && s.is_array && site.gpu_limit > 0 && !isempty(last_gpu)
            dep = (isempty(dep) ? "" : dep * ",") * "afterany:" * last_gpu
        end
        isempty(dep) || insert!(c, 3, "--dependency=$dep")
        out = IOBuffer(); err = IOBuffer()
        e = merge(plan.base_env, st.env)
        ok = try
            success(pipeline(Cmd(Cmd(_with_setup(site, c, e)); env = e, dir = g.workdir); stdout = out, stderr = err))
        catch ex
            ex isa Base.IOError || rethrow()
            print(err, sprint(showerror, ex)); false
        end
        txt = strip(String(take!(out)))
        if !ok || isempty(txt)
            records[s.name] = _stage_record(st; state = "submit failed")
            _write_manifest(plan, records)
            error("sbatch of stage $(s.name) failed: $(strip(String(take!(err))))\n  $(_shell_line(c))")
        end
        id = String(first(split(txt, ';')))
        ids[s.name] = id
        s.gpu && s.is_array && (last_gpu = id)
        records[s.name] = _stage_record(st; state = "submitted", job_id = id)
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
stops the run (a later stage depends on it). The manifest is written after every task. Returns the records.
"""
function run_local!(plan::ReproducePlan; io::IO = stdout)
    g, site = plan.graph, plan.site
    logs = mkpath(joinpath(g.workdir, "logs")); mkpath(plan.results_root)
    records = Dict{String, Any}()
    for st in plan.stages
        s = st.spec
        if st.reuse == "completed"
            records[s.name] = _stage_record(st; state = "reused")
            println(io, rpad(s.name, 32), " reused (completed earlier)")
            continue
        end
        tasks = Dict{String, String}()
        for (i, item) in enumerate(st.items)
            t = i - 1
            log = joinpath(logs, s.is_array ? "$(g.name)_$(s.name)_$t.log" : "$(g.name)_$(s.name).log")
            e = merge(plan.base_env, st.env)
            s.is_array && (e["SLURM_ARRAY_TASK_ID"] = string(t))
            t0 = time()
            ok = open(log, "w") do f
                try
                    success(pipeline(Cmd(Cmd(_with_setup(site, st.command, e)); env = e, dir = g.workdir, ignorestatus = true);
                                     stdout = f, stderr = f))
                catch ex
                    ex isa Base.IOError || rethrow()
                    println(f, "ERROR: ", sprint(showerror, ex)); false
                end
            end
            tasks[string(t)] = ok ? "completed" : "failed"
            println(io, rpad(s.name, 32), " ", s.is_array ? "task $t ($item) " : "", ok ? "ok" : "FAILED",
                    "  $(round(time() - t0; digits = 1)) s  $(relpath(log, g.workdir))")
            records[s.name] = _stage_record(st; state = ok ? (i == length(st.items) ? "completed" : "running") : "failed", log, tasks)
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
    p = replace(pattern, "%A" => id, "%a" => string(task), "%x" => isempty(s.job_name) ? "*" : s.job_name,
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
    reproduce_status(manifest; job_states = nothing, env = ENV) -> Vector{NamedTuple}

The state of every stage of the launch a run manifest records: `(; stage, job_id, state, log, error)`, the state
from Slurm (each task's, summarised) or the local record, the log of the first task that did not complete (else the
first) and that log's first error line.
"""
function reproduce_status(manifest::AbstractString; job_states = nothing, env = ENV)
    m = TOML.parsefile(manifest)
    g = load_stage_graph(joinpath(m["workdir"], m["graph"]))
    recs = m["stages"]
    ids = String[r["job_id"] for (_, r) in recs if !isempty(get(r, "job_id", ""))]
    states = _job_states(job_states, ids, env)
    rows = NamedTuple[]
    for s in g.stages
        haskey(recs, s.name) || continue
        r = recs[s.name]
        id = get(r, "job_id", "")
        if !isempty(id) && haskey(states, id)
            st = states[id]
            counts = Dict(x => count(==(x), st) for x in unique(st))
            state = join(("$(k) $(v)" for (k, v) in sort(collect(counts))), ", ")
            bad = findfirst(!=("COMPLETED"), st)
            log = _slurm_log(g, s, id, something(bad, 1) - 1)
        else
            state = r["state"]                              # a local run stops at its first failing task: its log
            log = get(r, "log", "")
            isempty(log) && !isempty(id) && (log = _slurm_log(g, s, id, 0))
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
                 [--allow-missing] [--manifest=<run manifest>]
  run (default)  every default stage of the graph (with --only the named stages and, without --no-deps, what they
                 run after); finished stages are reused by their run keys
  --dry-run      the job graph (stages, items, dependencies, commands) and the missing inputs; nothing runs
  status         the last launch's stages (or --manifest's): state, log, first error line
Exit codes: 0 done (or submitted), 2 usage or a config error, 3 missing inputs (nothing ran), 4 a stage failed."""

"""
    reproduce_main(args = ARGS; io = stdout, env = ENV, job_states = nothing) -> Int

The command line of the runner ([`REPRODUCE_USAGE`](@ref)); returns the exit code. A launch checks the inputs first
and runs nothing when one is missing ([`missing_inputs`](@ref), exit 3; `--allow-missing` runs anyway), writes the run
manifest and then submits (Slurm) or runs (local) the stages.
"""
function reproduce_main(args::AbstractVector{<:AbstractString} = ARGS; io::IO = stdout, env = ENV,
                        job_states = nothing)
    # the subcommand: a bare word anywhere (a launcher puts its own options first), never the value of a spaced option
    cmd = any(i -> args[i] == "status" && (i == 1 || !(args[i - 1] in ("--only", "--graph", "--site", "--manifest",
              "--data-root", "--results-root"))), eachindex(args)) ? "status" : "run"
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
            for r in reproduce_status(m; job_states, env)
                println(io, rpad(r.stage, 32), " ", rpad(r.job_id, 10), " ", r.state)
                isempty(r.log) || println(io, " "^33, "log   ", r.log)
                isempty(r.error) || println(io, " "^33, "error ", r.error)
            end
            return 0
        end
        only = something(_cli_value(args, "only"), "")
        plan = plan_reproduction(g, site; only = String.(split(only, ','; keepempty = false)), deps = !("--no-deps" in args),
                                 data_root = _cli_value(args, "data-root"), results_root = _cli_value(args, "results-root"),
                                 env, job_states)
        miss = missing_inputs(g, [st.spec.name for st in plan.stages]; data_root = plan.data_root, env)
        if "--dry-run" in args
            print_plan(io, plan; missing = miss)
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
