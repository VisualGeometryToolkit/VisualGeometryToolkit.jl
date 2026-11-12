# =============================================================================
# src/code_identity.jl — the code a result was computed with, as git tree hashes, and the one rule that decides whether
# a stored result still stands for the current code.
#
#   code_trees(workdir; code, julia_project) -> (; trees, dirty)   the trees of the project's code paths and of every
#                                                                 stack package the project's environment loads
#   code_tree_changes(recorded, current) -> Vector{String}        what changed between two such records ("" = none):
#                                                                 a stored result is reused only when it is empty
#   file_sha256(path) -> String                                   a file's content hash, streamed and cached by
#                                                                 (path, size, mtime): the inputs' `sha256` check
#                                                                 and the files a run key folds in
#
# A result's run key names its CONFIGURATION only (DrWatson's savename / produce_or_load, the runner's stage key); the
# code it ran is recorded beside it (DrWatson's `tag!`: gitcommit, and gitpatch when dirty) together with these trees.
# A tree hash moves only when the files under it change, so a commit that touches the docs, the tests or the paper
# leaves every tree, and with it every stored result, as it was.
#
# DEPENDENCY-FREE BY CONTRACT (Base and the stdlibs TOML and SHA): src/cli.jl includes it, so the package has it (the
# Experiments records, the runner) and so does a project's launcher, which includes src/cli.jl and
# reproduce/runner.jl by path from a bare `julia` before the project's environment exists.
# =============================================================================
using TOML: TOML
using SHA: sha256

# (realpath, size, mtime) → sha256: a file is hashed once per process while it is unchanged.
const _FILE_SHA256 = Dict{Tuple{String, Int64, Float64}, String}()
const _FILE_SHA256_LOCK = ReentrantLock()

"""
    file_sha256(path) -> String

The SHA-256 (64 hex digits) of the file at `path`, read in blocks (a file of any size costs its read, never its size
in memory) and cached in this process by its real path, size and modification time: a second call on an unchanged
file costs a `stat`. A file rewritten in place within the file system's mtime resolution with the same size keeps its
cached hash; a check of record that must exclude that runs in a fresh process. An `ArgumentError` for a directory or a
missing path.
"""
function file_sha256(path::AbstractString)
    isfile(path) || throw(ArgumentError("file_sha256: $path is $(isdir(path) ? "a directory" : "not there"), not a file"))
    st = stat(path)
    k = (realpath(path), Int64(st.size), Float64(st.mtime))
    hit = lock(() -> get(_FILE_SHA256, k, nothing), _FILE_SHA256_LOCK)
    hit === nothing || return hit
    h = bytes2hex(open(sha256, path))
    lock(() -> (_FILE_SHA256[k] = h), _FILE_SHA256_LOCK)
    return h
end

# The registered stack packages, identified by their Manifest `git-tree-sha1` when the environment loads a released
# version (a package loaded from a path, `[sources]` or `Pkg.develop`, is identified by its own repository's trees,
# whatever its name). The published packages of VisualGeometryToolkitCore's configs/stack_dependencies.toml, whose
# paper projects are never a dependency; a test holds the two lists together.
const _STACK_PACKAGES = ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures",
                         "VisualGeometryToolkitEstimation", "VisualGeometryToolkitEval", "PoseLib", "RANSACScoringDoneRight",
                         "LocalOptimizationDoneRight", "BlobBoards", "PanoramaStitching")

# What of a package loaded from a path is its code: what `using` reads. Its docs, tests and paper are not.
const _PACKAGE_CODE = ("src", "ext", "Project.toml")

# The mark of a tree whose files differ from HEAD: its value is then no commit's tree, so no clean record matches it.
const _DIRTY_MARK = "+uncommitted"

_git_line(dir, args) = try
    readchomp(pipeline(Cmd(`git -C $dir $args`); stderr = devnull))
catch
    nothing
end

# The tree (or blob) of `path`, relative to `dir`, at HEAD; nothing when HEAD does not hold it.
_head_tree(dir, path) = _git_line(dir, ["rev-parse", "--verify", "--quiet", path == "." ? "HEAD:./" : "HEAD:./" * path])

# Whether `paths` (relative to `dir`) differ from HEAD: a modified, staged, deleted or untracked (not ignored) file.
_uncommitted(dir, paths) = !isempty(something(_git_line(dir, ["status", "--porcelain", "--untracked-files=normal", "--", paths...]), "?"))

# The Manifest of a project directory: its own, else the nearest one above it (a workspace's, at its root).
function _find_manifest(project::AbstractString)
    d = abspath(project)
    while true
        for f in ("JuliaManifest-v$(VERSION.major).$(VERSION.minor).toml", "Manifest-v$(VERSION.major).$(VERSION.minor).toml",
                  "JuliaManifest.toml", "Manifest.toml")
            isfile(joinpath(d, f)) && return joinpath(d, f)
        end
        p = dirname(d)
        (p == d || isdir(joinpath(d, ".git")) || isfile(joinpath(d, ".git"))) && return nothing
        d = p
    end
end

# The Manifest entries the project's own [deps] reach, by name.
function _loaded_entries(project::AbstractString, manifest::AbstractString)
    pfile = first(f for f in (joinpath(project, "JuliaProject.toml"), joinpath(project, "Project.toml"), "") if isempty(f) || isfile(f))
    isempty(pfile) && return Dict{String, Any}()
    m = TOML.parsefile(manifest)
    deps = get(m, "deps", m)                               # format 2.0 (`[[deps.Name]]`), else the old one
    out = Dict{String, Any}()
    todo = String.(collect(keys(get(TOML.parsefile(pfile), "deps", Dict{String, Any}()))))
    while !isempty(todo)
        n = pop!(todo)
        (haskey(out, n) || !(get(deps, n, nothing) isa AbstractVector)) && continue
        e = first(deps[n])
        out[n] = e
        d = get(e, "deps", String[])
        append!(todo, String.(d isa AbstractDict ? collect(keys(d)) : d))
    end
    return out
end

"""
    code_trees(workdir; code = nothing, julia_project = workdir) -> (; trees, dirty, project_paths)

The code identity of a run in the git checkout `workdir`: `trees`, by name, the git tree hash of each of the
project's `code` paths (relative to `workdir`: `git rev-parse HEAD:<path>`; by default `src` when there is one) and of
every stack package `julia_project`'s environment loads (its `[deps]` and what they reach in its Manifest, found in
it or above it, a workspace's): a package loaded from a path (`[sources]`, `Pkg.develop`) by the trees of its
`src`, `ext` and `Project.toml` in its own repository (`"src:<tree> ext:<tree> Project.toml:<blob>"`), a released
stack package by its Manifest `git-tree-sha1`. `dirty` names each of them whose files differ from HEAD (an edit, a
staged or deleted file, an untracked one), or that is in no git checkout: a run of record refuses to run with it
([`refuse_uncommitted_code`](@ref)). Such a tree's value carries `+uncommitted` (or is `"uncommitted"`), so a result
recorded with it is never reused for committed code. `project_paths` lists the project's own paths recorded (none:
nothing identifies the project's code).

A tree moves only when a file under it changes: a commit that touches only docs, tests or the paper changes no tree.
Compare two records with [`code_tree_changes`](@ref).
"""
function code_trees(workdir::AbstractString; code = nothing, julia_project::AbstractString = workdir)
    trees, dirty = Dict{String, String}(), String[]
    paths = code === nothing ? (isdir(joinpath(workdir, "src")) ? ["src"] : String[]) : String.(code)
    for c in paths
        t = _head_tree(workdir, c)
        if t === nothing
            ispath(joinpath(workdir, c)) || continue       # neither committed nor there: nothing to identify
            trees[c] = "uncommitted"
            push!(dirty, c)
        elseif _uncommitted(workdir, [c])
            trees[c] = t * _DIRTY_MARK                     # never equal to a clean record: never reused for one
            push!(dirty, c)
        else
            trees[c] = t
        end
    end
    project_paths = sort!(collect(keys(trees)))
    manifest = _find_manifest(julia_project)
    if manifest !== nothing
        for (name, e) in _loaded_entries(abspath(julia_project), manifest)
            if haskey(e, "path")
                dir = normpath(joinpath(dirname(manifest), e["path"]))
                subs = [s for s in _PACKAGE_CODE if ispath(joinpath(dir, s)) || _head_tree(dir, s) !== nothing]
                parts = [(s, _head_tree(dir, s)) for s in subs]
                if _git_line(dir, ["rev-parse", "--show-toplevel"]) === nothing
                    trees[name] = "not in git: $dir"
                    push!(dirty, name)
                    continue
                end
                marked = [(s, t === nothing ? "uncommitted" : _uncommitted(dir, [s]) ? t * _DIRTY_MARK : t) for (s, t) in parts]
                trees[name] = join(("$s:$t" for (s, t) in marked), " ")
                any(p -> p[2] == "uncommitted" || endswith(p[2], _DIRTY_MARK), marked) && push!(dirty, name)
            elseif name in _STACK_PACKAGES && haskey(e, "git-tree-sha1")
                trees[name] = e["git-tree-sha1"]
            end
        end
    end
    return (; trees, dirty = sort!(dirty), project_paths)
end

"""
    refuse_uncommitted_code(identity; what = "a run of record") -> nothing

The refusal of a run of record whose code is not committed, the same for the runner and a driver's records:
`identity` is [`code_trees`](@ref)'s result. An `ArgumentError` when any of its trees has uncommitted changes, or
is in no git checkout (its results could not name the code they ran), or when it identifies none of the project's
own code (no `src`, no `code` path).
"""
function refuse_uncommitted_code(identity; what::AbstractString = "a run of record")
    d = identity.dirty
    isempty(d) || throw(ArgumentError("$what runs committed code, and $(join(d, ", ")) " *
                                      "$(length(d) == 1 ? "has" : "have") uncommitted changes (or no git checkout): " *
                                      "commit them (or stash them), then run again"))
    isempty(identity.project_paths) && throw(ArgumentError("$what records the code it runs, and none of the project's own " *
        "code is identified (no `src`, no `code` path): name its code paths (a stage graph's [project] code)"))
    return nothing
end

# A package loaded from a path: "src:<tree> ext:<tree> Project.toml:<blob>", by part.
const _PARTS = r"^(src|ext|Project\.toml):"
_tree_parts(v) = Dict(String(p[1]) => String(p[2]) for p in (split(x, ':'; limit = 2) for x in split(v)))

_short_tree(t) = occursin(' ', t) || !occursin(r"^[0-9a-f]{40}$", t) ? t : first(t, 10)

"""
    code_tree_changes(recorded, current) -> Vector{String}

THE reuse rule of a stored result, for every project: the result stands for the current code only when this is
empty. `recorded` is the code trees stored with the result (`nothing` when it has none: a result from before code
identity was kept, which is computed once more and then records them), `current` the code's now
([`code_trees`](@ref)'s `trees`). Each entry names what changed: `"VisualGeometryToolkitCore: src 3fa29c01e2 →
b71d0e44a9"`, `"src: 18c2… → 9e0a…"`, `"<name>: new"` (loaded now, not then), `"<name>: no longer loaded"`.
"""
function code_tree_changes(recorded, current::AbstractDict)
    recorded === nothing && return ["no code trees recorded (a result from before code identity was kept)"]
    out = String[]
    for k in sort!(unique!([String.(collect(keys(recorded))); String.(collect(keys(current)))]))
        a, b = get(recorded, k, nothing), get(current, k, nothing)
        a == b && continue
        if a === nothing
            push!(out, "$k: new")
        elseif b === nothing
            push!(out, "$k: no longer loaded")
        elseif occursin(_PARTS, string(a)) && occursin(_PARTS, string(b))
            pa, pb = _tree_parts(string(a)), _tree_parts(string(b))
            for s in sort!(unique!([collect(keys(pa)); collect(keys(pb))]))
                x, y = get(pa, s, "none"), get(pb, s, "none")
                x == y || push!(out, "$k: $s $(_short_tree(x)) → $(_short_tree(y))")
            end
        else
            push!(out, "$k: $(_short_tree(string(a))) → $(_short_tree(string(b)))")
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Inputs: the root of each tier
# -----------------------------------------------------------------------------

"""
    tier_variable(tier; prefix = "VGTK_DATA_ROOT") -> String

The environment variable that gives the root of the input tier `tier`: `<prefix>_<TIER>`, the tier upper-cased with
every character other than a letter or a digit made `_` (`blobcalib_202609` → `VGTK_DATA_ROOT_BLOBCALIB_202609`).
"""
tier_variable(tier::AbstractString; prefix::AbstractString = "VGTK_DATA_ROOT") =
    prefix * "_" * uppercase(replace(tier, r"[^A-Za-z0-9]" => "_"))

"""
    tier_data_root(tier, data_root; env = ENV, tiers = nothing, workdir = "", prefix = "VGTK_DATA_ROOT") -> (root, source)

THE root of an input tier, in one order for the runner and the drivers: the environment's
[`tier_variable`](@ref)`(tier)` (`source = "env"`), else the machine's table `tiers` (a stage graph's
`[sites.<site>.tiers]`: `tier = "<root>"`, a relative root relative to `workdir`; `source = "site"`), else the one
`data_root` (`source = "data root"`). An `[inputs.<id>]` path is relative to its tier's root, so a config names no
machine path whichever disk a tier lives on.
"""
function tier_data_root(tier::AbstractString, data_root::AbstractString; env = ENV, tiers = nothing,
                        workdir::AbstractString = "", prefix::AbstractString = "VGTK_DATA_ROOT")
    v = String(get(env, tier_variable(tier; prefix), ""))
    isempty(v) || return (v, "env")
    if tiers !== nothing && !isempty(tier) && haskey(tiers, tier)
        t = String(tiers[tier])
        return (isabspath(t) || isempty(workdir) ? t : normpath(joinpath(workdir, t)), "site")
    end
    return (String(data_root), "data root")
end

"""
    scratch_record(path) -> Bool

Whether `path` (a run directory, or a file in one) is the output of a scratch run: the `_provenance.toml` beside
it says `scratch = true` (`Experiments.scratch_run`). A scratch output is never an input of record, an upstream run
of record, or a promoted run.
"""
function scratch_record(path::AbstractString)
    f = joinpath(isdir(path) ? path : dirname(path), "_provenance.toml")
    return isfile(f) && get(TOML.parsefile(f), "scratch", false) === true
end

"""
    inspect_input(id, table, root; root_source = "data root", root_name = "", allow_scratch = false)
        -> (; state, path, problem)

THE check of one `[inputs.<id>]` table under its tier's `root` ([`tier_data_root`](@ref)), for the runner's
`missing_inputs` and a driver's `verify_inputs` alike. `state` is `:present`; `:root_missing` (a tier root other
than the data root, named by `root_name`, that is not there); `:missing` (no `path` under `root`); or `:refused`
(another content than its `sha256`, or a scratch run's output, [`scratch_record`](@ref), unless `allow_scratch`).
`problem` says what, without the input's id. A `sha256` that is no 64 hex digits, or one on a directory, is an
`ArgumentError` (a config error).
"""
function inspect_input(id::AbstractString, t::AbstractDict, root::AbstractString; root_source::AbstractString = "data root",
                       root_name::AbstractString = "", allow_scratch::Bool = false)
    rel, tier = String(t["path"]), String(get(t, "tier", ""))
    root_source != "data root" && !isdir(root) &&
        return (; state = :root_missing, path = "", problem = "$tier (root $root, from $root_name, is not there)")
    full = isempty(root) ? "" : joinpath(root, rel)
    (isempty(full) || !ispath(full)) &&
        return (; state = :missing, path = full, problem = "$rel is not under $(isempty(root) ? "<no root>" : root)")
    if haskey(t, "sha256")
        want = lowercase(string(t["sha256"]))
        occursin(r"^[0-9a-f]{64}$", want) || throw(ArgumentError("input $id: sha256 = \"$(t["sha256"])\" is no SHA-256 (64 hex digits)"))
        isfile(full) || throw(ArgumentError("input $id ($rel): a sha256 names a file's content, and it is a directory"))
        got = file_sha256(full)
        got == want || return (; state = :refused, path = full, problem = "sha256 expected $want, actual $got")
    end
    !allow_scratch && scratch_record(full) &&
        return (; state = :refused, path = full, problem = "a scratch run's output (its record says scratch = true), never an input of record")
    return (; state = :present, path = full, problem = "")
end
