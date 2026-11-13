# =============================================================================
# provenance.jl — the provenance record of a derived artifact directory, the
# skip-if-complete commit-drift check, the legacy cache key and the keyed
# result cache (hoisted 2026-10-07: VisualGeometryToolkitEval `write_run_records`,
# bmvc2026 src/provenance.jl `write_provenance`/`read_provenance`, bmvc2026
# src/cache.jl `cached`, bmvc2026 sweep.jl's drift warning).
#
# The SCHEMA IS DRWATSON'S TAGGED DICT. `_provenance.toml` holds one flat table:
# the caller's fields plus `driver`, `timestamp`, `hostname`, `julia` and what
# DrWatson's `tag!` adds under its own names — `gitcommit` (`gitdescribe`,
# "-dirty" on an uncommitted tree) and `gitpatch` (the `git diff HEAD` that
# restores it) — and, beside a patch, `uncommitted_files`, the paths it
# touches; and `code_trees`, the git trees of the code the run loaded
# (`code_trees` of src/code_identity.jl: the project's src, every stack
# package). A stored result is reused only while those trees are the current
# ones (`code_tree_changes`, the one rule): the run key names the configuration
# only, the trees the code. Declared inputs are recorded RELATIVE to the roots they live under,
# with the roots beside them, so a record is machine-portable.
# Justified extension: the file is TOML, written with TOML.print, because
# DrWatson's `wsave` has no TOML writer and the record must stay readable
# without Julia.
#
# Readers accept the two retired layouts: VisualGeometryToolkitEval ≤ 0.1 (flat, the
# patch in a sibling `_dirty.patch`) and bmvc2026's (`[provenance]` git table +
# `[config]` payload table); [`read_provenance`](@ref) returns every layout as
# the one flat dict.
# =============================================================================
using DrWatson: DrWatson, tag!, safesave, gitdescribe, wload
using LibGit2: LibGit2

# DrWatson's `tag!(d; storepatch = true)` — except inside a git WORKTREE, where
# DrWatson's `gitpatch` builds `--git-dir` from `LibGit2.path(repo)`, which
# there is the worktree's admin directory, so `git diff` fails and the patch of
# a dirty tree is silently dropped (DrWatson saving_tools.jl:136-143). There
# the patch is read with `git -C <worktree> diff HEAD`, the same command
# DrWatson runs, under the same `gitpatch` key.
function _tag!(d::AbstractDict; gitpath::AbstractString)
    worktree = try
        occursin("/.git/worktrees/", LibGit2.path(LibGit2.GitRepoExt(gitpath)))
    catch
        false
    end
    d = tag!(d; gitpath, storepatch = !worktree, warn = false)
    if worktree && endswith(string(get(d, "gitcommit", "")), "-dirty") && !haskey(d, "gitpatch")
        patch = try read(`git -C $gitpath diff --submodule=diff HEAD`, String) catch; "" end
        isempty(patch) || (d["gitpatch"] = patch)
    end
    return d
end

# Keys the writer owns; a caller field of the same name is refused.
const _PROVENANCE_RESERVED = ("driver", "timestamp", "hostname", "julia", "gitcommit", "gitpatch", "uncommitted_files", "inputs",
                              "code_trees", "scratch", "sensor_model")

# The sensor model's provenance in a run record (the sensor-model spec §5: the tables say per run where the black, the
# noise and the defects came from): Core's ONE view of it, `component_sources`, rendered for TOML — a source by its tag
# ("measured", "estimated", "specified", "file"), an absent slot "absent", the reason by its name. A run over several
# frames records each distinct model's sources once, with the number of frames that had it.
_source_tag(s) = s === nothing ? "absent" : string(s)
function _sensor_record(m::SensorModel)
    c = component_sources(m)
    return Dict{String,Any}("black" => _source_tag(c.black), "noise" => _source_tag(c.noise), "defects" => _source_tag(c.defects),
                            "reason" => string(c.reason), "specified_override" => c.specified_override)
end
function _sensor_records(models::AbstractVector{<:SensorModel})
    records = _sensor_record.(models)
    return [merge(r, Dict{String,Any}("frames" => count(==(r), records))) for r in unique(records)]
end

"""
    current_code_trees(gitpath = project_root()) -> Dict{String,String}

The code trees a result computed now records (`code_trees` of the top module):
the project's `src` (of `gitpath`) and every stack package its environment
(`gitpath`'s, or the Manifest above it) loads. [`write_provenance`](@ref) and
[`cached_result`](@ref) store them; [`run_complete`](@ref) and
[`cached_result`](@ref) reuse a stored result only while they are unchanged
(`code_tree_changes`).
"""
current_code_trees(gitpath::AbstractString = project_root()) = _code_identity(gitpath).trees

_code_identity(gitpath::AbstractString) = isempty(gitpath) ?
    (; trees = Dict{String,String}(), dirty = String[], project_paths = String[]) :
    code_trees(gitpath; julia_project = gitpath)

# The code trees a result being recorded now carries: refused for a run of record when the code is not committed
# (the runner's refusal, `refuse_uncommitted_code`), warned for a scratch run; a project whose own code is not
# identified (no `src`) is warned either way: only the packages it loads name its code.
function _recorded_code_trees(gitpath::AbstractString, what::AbstractString)
    ident = _code_identity(gitpath)
    if !isempty(ident.dirty)
        scratch_run() || refuse_uncommitted_code(ident; what = "a run of record")
        @warn "a scratch run records $what of uncommitted code (VGTK_SCRATCH=1): it is never reused (the mark names HEAD, not the edits)" uncommitted = join(ident.dirty, ", ") maxlog = 1
    end
    isempty(gitpath) || !isempty(ident.project_paths) ||
        @warn "$what records no tree of the project's own code (no src under $gitpath): only the packages it loads identify its code" maxlog = 1
    return ident.trees
end

# The paths a `git diff` touches, in order: what a dirty tree changed, readable
# without applying the patch (a "-dirty" commit alone does not say whether the
# edit was a Manifest or the driver).
_patch_files(patch::AbstractString) =
    unique!([String(something(m.captures[1])) for m in eachmatch(r"^diff --git a/(\S+) b/"m, patch)])

# TOML cannot hold `nothing`/symbols: render anything exotic as its string, so
# a value can never silently fail to be recorded (deliberately lenient).
_toml_render(v::Union{Real,AbstractString,Bool}) = v
_toml_render(v::AbstractVector) = [_toml_render(x) for x in v]
_toml_render(v::AbstractDict) = Dict{String,Any}(String(k) => _toml_render(x) for (k, x) in v)
_toml_render(v) = string(v)

# The declared inputs, each relative to the most specific root it lives under.
function _relative_inputs(inputs)
    roots = Pair{String,String}[]
    r = try results_root() catch; nothing end
    r === nothing || push!(roots, "results" => r)
    d = _bound_data_root()
    d === nothing || push!(roots, "data" => d)
    out = Dict{String,Any}()
    for p in String.(inputs)
        hit = findfirst(((_, root),) -> _is_under(p, root), roots)
        kind, rel = hit === nothing ? ("absolute", abspath(p)) :
                    (roots[hit].first, relpath(_resolved_path(p), _resolved_path(roots[hit].second)))
        push!(get!(out, kind, String[]), rel)
    end
    for (kind, root) in roots
        haskey(out, kind) && (out[kind * "_root"] = root)
    end
    return out
end

"""
    write_provenance(dir, fields = Dict(); driver, inputs = nothing, sensor_model = nothing,
                     gitpath = project_root(), producer = false) -> String

Write `<dir>/_provenance.toml`: `fields` (the caller's record — run key,
config source, network hash, frame counts, …) plus `driver`, `timestamp`,
`hostname`, `julia`, and DrWatson's `tag!` record of `gitpath` (`gitcommit`,
`gitpatch` on a dirty tree, with `uncommitted_files` — the paths that patch
touches) and `code_trees`, the git trees of the code the run loaded
([`current_code_trees`](@ref)). `inputs` (paths) are recorded relative to the
results or data root they live under, the roots beside them
(`inputs = {data = […], data_root = …}`). Refuses to write into the data root
unless `producer` ([`assert_not_input`](@ref)), records `scratch` (whether this
is a [`scratch_run`](@ref); a run of record refuses `inputs` that are a scratch
run's output, and a directory already holding a record with the other flag is
refused), and refuses a run of record of uncommitted code (`refuse_uncommitted_code`, the runner's refusal: an edited,
staged, deleted or untracked file in the project's `src` or a package it loads
from a path, or code in no git checkout); a [`scratch_run`](@ref) records it
with a warning, its trees marked `+uncommitted`. `sensor_model` (a run's
`SensorModel`, or one per frame) records where its black, noise and defects came
from — VisualGeometryToolkitCore's `component_sources`, one table per distinct
model with its frame count, under `sensor_model` — so a table can state
"measured" or "estimated" per run. Returns the record's path.
"""
function write_provenance(dir::AbstractString, fields::AbstractDict = Dict{String,Any}();
                          driver::AbstractString, inputs = nothing,
                          sensor_model::Union{Nothing, SensorModel, AbstractVector{<:SensorModel}} = nothing,
                          gitpath::AbstractString = project_root(), producer::Bool = false)
    assert_not_input(dir; producer)
    clash = intersect(String.(collect(keys(fields))), _PROVENANCE_RESERVED)
    isempty(clash) || throw(ArgumentError("write_provenance: field(s) $(sort(clash)) are the writer's own"))
    prov = Dict{String,Any}(String(k) => _toml_render(v) for (k, v) in fields)
    prov["driver"] = String(driver)
    prov["timestamp"] = string(Dates.now())
    prov["hostname"] = gethostname()
    prov["julia"] = string(VERSION)
    sensor_model === nothing || (prov["sensor_model"] = _sensor_records(sensor_model isa SensorModel ? [sensor_model] : sensor_model))
    if inputs !== nothing
        scratch_in = filter(scratch_record, String.(inputs))
        scratch_run() || isempty(scratch_in) ||
            throw(ArgumentError("a run of record reads no scratch output, and $(join(scratch_in, ", ")) " *
                                "$(length(scratch_in) == 1 ? "is one" : "are") (its record says scratch = true): run it as a run of record first"))
        prov["inputs"] = _relative_inputs(inputs)
    end
    prov["scratch"] = scratch_run()
    old = joinpath(dir, PROVENANCE_FILE)
    if isfile(old)
        was = get(TOML.parsefile(old), "scratch", false) === true
        was == prov["scratch"] || throw(ArgumentError("write_provenance: $dir holds the record of a $(was ? "scratch run" : "run of record"), " *
            "and this is a $(was ? "run of record" : "scratch run"): the two never share a directory (key it, keyed_cell_dir)"))
    end
    prov = _tag!(prov; gitpath)
    haskey(prov, "gitpatch") && (prov["uncommitted_files"] = _patch_files(prov["gitpatch"]))
    prov["code_trees"] = _recorded_code_trees(gitpath, "the record of $dir")
    mkpath(dir)
    out = joinpath(dir, PROVENANCE_FILE)
    open(out, "w") do io
        println(io, "# What actually ran (DrWatson's tagged dict: gitcommit, and gitpatch restores a dirty tree; code_trees, the code's git trees).")
        TOML.print(io, prov; sorted = true)
    end
    return out
end

"""
    read_provenance(dir) -> Dict{String,Any}

The provenance record of `dir` as the one flat dict of
[`write_provenance`](@ref), whatever layout wrote it: the current one; the
VisualGeometryToolkitEval ≤ 0.1 one (its `_dirty.patch` read back as `gitpatch`);
bmvc2026's `[provenance]` + `[config]` tables (merged; `created` read as
`timestamp`). Errors when the record is absent.
"""
function read_provenance(dir::AbstractString)
    p = joinpath(dir, PROVENANCE_FILE)
    isfile(p) || error("no $PROVENANCE_FILE in $dir — the artifact predates the provenance policy")
    raw = TOML.parsefile(p)
    d = if haskey(raw, "provenance") && raw["provenance"] isa AbstractDict &&
           haskey(raw, "config") && raw["config"] isa AbstractDict && length(raw) == 2
        meta, cfg = raw["provenance"], raw["config"]
        clash = intersect(keys(meta), keys(cfg))
        isempty(clash) || error("$p: the bmvc2026 [provenance] and [config] tables share key(s) $(sort!(collect(clash)))")
        merge(Dict{String,Any}(cfg), Dict{String,Any}(meta))
    else
        Dict{String,Any}(raw)
    end
    haskey(d, "created") && !haskey(d, "timestamp") && (d["timestamp"] = pop!(d, "created"))
    patch = joinpath(dir, "_dirty.patch")
    !haskey(d, "gitpatch") && isfile(patch) && (d["gitpatch"] = read(patch, String))
    return d
end

"""
    provenance_inputs(prov) -> Vector{String}

The declared inputs of a provenance record as absolute paths: relative entries
joined to the roots recorded beside them; a retired record's absolute list as is.
"""
function provenance_inputs(prov::AbstractDict)
    ins = get(prov, "inputs", String[])
    ins isa AbstractVector && return String.(ins)
    out = String[]
    for kind in ("data", "results")
        append!(out, (joinpath(ins[kind * "_root"], p) for p in get(ins, kind, String[])))
    end
    append!(out, String.(get(ins, "absolute", String[])))
    return out
end

# A gitdescribe string's commit, as a hex prefix: "<tag>-<n>-g<sha>", a bare
# sha, or an exact annotated tag (resolved in the repository). "-dirty" dropped.
function _described_commit(s::AbstractString, gitpath::AbstractString)
    s = String(chopsuffix(s, "-dirty"))
    m = match(r"-g([0-9a-f]{4,40})$", s)
    m === nothing || return String(something(m.captures[1]))
    occursin(r"^[0-9a-f]{7,40}$", s) && return s
    return try
        repo = LibGit2.GitRepoExt(gitpath)
        string(LibGit2.GitHash(LibGit2.GitCommit(repo, s)))
    catch
        s
    end
end

"""
    commit_drift(prov; gitpath = project_root()) -> Union{Nothing, Tuple{String,String}}

`(recorded, current)` when the commit a provenance record names is not the
current one of `gitpath`, `nothing` when they agree or either is unknown.
Compared as COMMITS, through DrWatson's `gitdescribe(gitpath; dirty_suffix = "")`:
a describe string with an annotated tag (`v0.7.0-12-g89f65bd`), a bare sha and
an exact tag all name their commit, and a dirty suffix is not a different commit.
"""
function commit_drift(prov::AbstractDict; gitpath::AbstractString = project_root())
    rec = get(prov, "gitcommit", "")
    isempty(rec) && return nothing
    cur = try gitdescribe(gitpath; dirty_suffix = "", warn = false) catch; nothing end
    cur === nothing && return nothing
    a, b = _described_commit(rec, gitpath), _described_commit(cur, gitpath)
    n = min(length(a), length(b))
    return a[1:n] == b[1:n] ? nothing : (String(rec), String(cur))
end

"""
    warn_commit_drift(prov, what; gitpath = project_root()) -> Bool

Skip-if-complete's WARNING (code is not keyed; the record restores it): warn
when `prov` was recorded under another commit ([`commit_drift`](@ref)) —
re-run with force if that change was result-affecting. Returns whether it warned.
"""
function warn_commit_drift(prov::AbstractDict, what; gitpath::AbstractString = project_root())
    d = commit_drift(prov; gitpath)
    d === nothing && return false
    @warn "skipping a complete run recorded under a DIFFERENT commit — re-run with force if the change was result-affecting" what recorded = d[1] current = d[2] maxlog = 4
    return true
end

"""
    run_complete(outdir; sidecar, gitpath = project_root()) -> Bool

Keyed skip-if-complete: a run counts as done only when its result `sidecar`
(a file name under `outdir`) AND its `_provenance.toml` exist, and the code
trees that record keeps are the current ones (`code_tree_changes` of the
recorded `code_trees` and [`current_code_trees`](@ref)`(gitpath)`). When a
tree changed, the run is not complete: it runs again, and an `@info` names
each tree that changed. A record from before the trees were kept has none and
runs once more. Another commit with the same trees (docs, tests, the paper) is
the same code: complete, without a warning.
"""
function run_complete(outdir::AbstractString; sidecar::AbstractString, gitpath::AbstractString = project_root())
    isfile(joinpath(outdir, sidecar)) || return false
    isfile(joinpath(outdir, PROVENANCE_FILE)) || return false
    prov = try read_provenance(outdir) catch; return false end
    changes = _reuse_changes(prov, current_code_trees(gitpath), scratch_run())
    isempty(changes) && return true
    @info "rerunning $outdir: its code changed" changes = join(changes, "; ")
    return false
end

# Why a stored result (its record `rec`: a run's provenance, a cached file's data) is not reused now: the code trees
# that changed since it was computed (`code_tree_changes` against `trees`), and a scratch flag that differs from
# `scratch` (a scratch run's result never stands for a run of record, nor the reverse). Empty: reuse it. The one reuse
# rule of `run_complete` and `cached_result`.
function _reuse_changes(rec, trees, scratch::Bool)
    changes = code_tree_changes(get(rec, "code_trees", nothing), trees)
    get(rec, "scratch", false) === scratch || push!(changes, "scratch: $(get(rec, "scratch", false)) → $scratch")
    return changes
end

"""
    legacy_hash_key(x) -> String

LEGACY, version-UNSTABLE key: `string(hash(x))`. It names the existing caches of
bmvc2026 and other paper projects (`<prefix>_<hash(cfg)>.jld2`, bmvc2026's detection
`run_key`), which must not move before their next full regeneration — that one
re-keys them to [`run_key`](@ref) (SHA-256 of the canonical JSON). Never use it
for a new key.
"""
legacy_hash_key(x) = string(hash(x))

"""
    cached_result(f, prefix, cfg, dir; key = legacy_hash_key, force = false,
                  gitpath = project_root(), ext = ".jld2", load = wload, save = nothing) -> (data, file)

The keyed result cache: `<dir>/<prefix>_<key(cfg)>.jld2` is loaded (DrWatson
`wload`) when present and its `code_trees` are the current ones
([`current_code_trees`](@ref), `code_tree_changes`; a file without them, from
before they were kept, is computed once more), else `f(cfg)` (a String-keyed
Dict) is computed and saved tagged like DrWatson's `tagsave` (gitcommit +
gitpatch ride along, and `code_trees`; uncommitted code is refused before
computing, as by [`write_provenance`](@ref)) and written
with DrWatson's `safesave`, so a forced re-run keeps the previous version as
`_#n`. The file name is
exactly the one DrWatson's `produce_or_load(…; filename = hash, prefix)` builds,
so its existing caches are found. Justified extension: `produce_or_load` itself
swallows save errors (DrWatson saving_files.jl:107-116 — a failed write returns
the data as if cached) and forces `safe = false`; here a save error propagates.

Another format (a vgio record) is cached through a codec: `ext` names the file's
extension, `load(file) -> Dict` reads it back and `save(file, data)` writes the
String-keyed Dict (code trees and DrWatson's tag included); the previous version
is kept as safesave keeps it (DrWatson's own `recursively_clear_path`: the newest
previous version is `<name>_#1<ext>`, older ones renumbered up). A [`scratch_run`](@ref)
caches as `<prefix>-scratch_<key><ext>`, never in a run of record's file, and
its data say `scratch = true`.
"""
function cached_result(f, prefix::AbstractString, cfg, dir::AbstractString;
                       key = legacy_hash_key, force::Bool = false,
                       gitpath::AbstractString = project_root(),
                       ext::AbstractString = ".jld2", load = wload, save = nothing)
    assert_not_input(dir)
    scratch = scratch_run()
    # a scratch run's cache is never a run of record's file
    file = joinpath(dir, string(prefix, scratch ? "-scratch" : "", "_", key(cfg), ext))
    if !force && isfile(file)
        trees = current_code_trees(gitpath)
        stored = load(file)
        changes = _reuse_changes(stored, trees, scratch)
        isempty(changes) && return stored, file
        @info "recomputing $file: its code changed" changes = join(changes, "; ")
    end
    trees = _recorded_code_trees(gitpath, file)                    # before computing: a refusal costs nothing
    data = f(cfg)
    mkpath(dir)
    data = _tag!(data; gitpath)
    data["code_trees"] = trees
    data["scratch"] = scratch
    if save === nothing
        safesave(file, copy(data))
    else
        DrWatson.recursively_clear_path(file)                         # safesave's own versioning (newest `_#1`), for any codec
        save(file, copy(data))
    end
    return data, file
end
