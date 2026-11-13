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
# DEPENDENCY-FREE BY CONTRACT (Base and the stdlibs TOML and SHA; and `canonical_json`, VisualGeometryToolkitCore's
# Base-only src/base/canonical.jl, which a bare `julia` includes by path before src/cli.jl): src/cli.jl includes it, so
# the package has it (the Experiments records, the runner) and so does a project's launcher, which includes src/cli.jl
# and reproduce/runner.jl by path from a bare `julia` before the project's environment is instantiated.
# =============================================================================
using TOML: TOML
using SHA: sha256
import Pkg   # a registered install's tree hash, from the active environment (`Pkg.dependencies`)

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

"""
    scratch_run(env = ENV) -> Bool

Whether this process runs scratch work rather than runs of record: `VGTK_SCRATCH=1` in the environment (`env`: the runner gives its launch's environment). A run of
record (the default) refuses to record a result of uncommitted code (`refuse_uncommitted_code`, the runner's
refusal); a scratch run records it with a warning, its trees marked `+uncommitted`, so it is never reused (the
mark names HEAD, not the edits).

A scratch run is kept apart from the runs of record: its record says `scratch = true` ([`write_provenance`](@ref VisualGeometryToolkitEval.Experiments.write_provenance),
[`cached_result`](@ref VisualGeometryToolkitEval.Experiments.cached_result)); its keys fold `scratch` ([`run_key`](@ref VisualGeometryToolkitEval.Experiments.run_key)), so it never shares a keyed directory or a
cache file with a run of record; and its output is refused as an input of record ([`verify_inputs`](@ref VisualGeometryToolkitEval.Experiments.verify_inputs),
[`input_path`](@ref VisualGeometryToolkitEval.Experiments.input_path), `missing_inputs`), an upstream run of record ([`run_key`](@ref VisualGeometryToolkitEval.Experiments.run_key)'s `upstream`), an input of a
run of record ([`write_provenance`](@ref VisualGeometryToolkitEval.Experiments.write_provenance)'s `inputs`) and a promoted run ([`promote_run`](@ref VisualGeometryToolkitEval.Experiments.promote_run)).
"""
scratch_run(env = ENV) = get(env, "VGTK_SCRATCH", "") == "1"

# `canonical_json`: VisualGeometryToolkitCore's one canonical encoding (its src/base/canonical.jl, Base only), bound by
# the includer: the package's top module imports Core's before including src/cli.jl; a bare `julia` (a project's runner
# bootstrap, scripts/reproduce/reproduce.sh) includes Core's file by path first. A bootstrap generated before 0.8.0 does
# not: it is told to regenerate itself, never left to fail on an undefined name.
isdefined(@__MODULE__, :canonical_json) || error(
    "src/code_identity.jl needs VisualGeometryToolkitCore's canonical_json, the run keys' encoding: a bare julia includes " *
    "Core's src/base/canonical.jl before src/cli.jl. A project's scripts/reproduce/reproduce.sh from before " *
    "VisualGeometryToolkitEval 0.8.0 does not: regenerate it with Reproduce.write_reproduce_script(<repository root>; " *
    "julia_project = \"<project>\")")

"""
    _content_key(tree) -> String

The content key of the evaluation stack: the SHA-256 (hex, first 16) of `tree`'s canonical JSON (VisualGeometryToolkitCore's
`canonical_json`, the stack's one canonical encoding; `tree` a lowered tree: `nothing`, `Bool`, `Int64`, `Float64`,
strings, vectors and string-keyed dicts, as a TOML config parses). Key order does not move it, nor a number's spelling:
an integral float is the integer (`3` and `3.0` key alike, as do `1.0` and `1e0`). Fixed across Julia versions (unlike
`Base.hash`). Behind Experiments `run_key`, Tuning `tuning_hash` and the runner's stage key.
"""
_content_key(tree) = bytes2hex(sha256(canonical_json(tree)))[1:16]

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

# (workdir, code paths, project) → (the state git reads, `code_trees`' answer then): `code_trees` once per process
# while that state holds (a driver asks before every cell of a sweep).
const _CODE_TREES = Dict{Tuple{String, Vector{String}, String}, Any}()
const _CODE_TREES_LOCK = ReentrantLock()

# A registered install: a package's source in a depot (no git checkout), content-addressed and immutable, so
# committed code. The active environment's record of it (Pkg's, never the Manifest parsed by hand): its name, uuid,
# version and tree hash. `nothing` for a git checkout or a directory the active environment does not install.
function _registered_install(dir::AbstractString)
    (isdir(dir) && _git_dirs(dir) === nothing) || return nothing
    rd = realpath(dir)
    for (uuid, info) in Pkg.dependencies()
        (info.tree_hash === nothing || info.is_tracking_path || info.source === nothing || !isdir(info.source)) && continue
        realpath(info.source) == rd && return (; info.name, uuid, info.version, tree = string(info.tree_hash))
    end
    return nothing
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
recorded with it is never reused (the mark names HEAD, not the edits). `project_paths` lists the project's own paths recorded (none:
nothing identifies the project's code).

A `workdir` that is a REGISTERED install (a package's source in a depot, no git checkout: a registered package's
example run with `gitpath = pkgdir(Pkg)`) is committed code, content-addressed and immutable: it is identified, by
its name, by the tree hash the active environment records for it (`"<git-tree-sha1> <name> <version> <uuid>"`, from
`Pkg.dependencies()`), and the environment searched for stack packages is the active project's. Only a git checkout
with uncommitted changes, or a path that is neither, is refused.

A tree moves only when a file under it changes: a commit that touches only docs, tests or the paper changes no tree.
Compare two records with [`code_tree_changes`](@ref).

The answer is kept in this process (git runs several times per path), and given again while nothing git reads has
changed: every file under the code paths and under each path-loaded package's `src`, `ext` and `Project.toml` (by
path, size, modification and change time and inode: an edit, a new, deleted or replaced file), the project's
`Project.toml` and Manifest, and per repository its `HEAD`, the ref `HEAD` names, `packed-refs`, the index and the
ignore files (a commit, a checkout, a staged or unstaged change). A file rewritten within the file system's time
resolution with the same size and inode keeps the kept answer, as [`file_sha256`](@ref)'s cache does; so does a
change to the index alone made while `code_trees` runs git (whose own status run rewrites the index).
"""
function code_trees(workdir::AbstractString; code = nothing, julia_project::AbstractString = workdir)
    reg = _registered_install(workdir)
    if reg !== nothing                                     # committed code in a depot: its tree, never "no git checkout"
        julia_project == workdir && Base.active_project() !== nothing && (julia_project = dirname(Base.active_project()))
    end
    paths = reg !== nothing ? String[] : code === nothing ? (isdir(joinpath(workdir, "src")) ? ["src"] : String[]) : String.(code)
    k = (abspath(workdir), paths, abspath(julia_project))
    seen = _code_state(workdir, paths, julia_project)
    hit = lock(() -> get(_CODE_TREES, k, nothing), _CODE_TREES_LOCK)
    r = hit !== nothing && hit[1] == seen ? hit[2] : nothing
    tries = 0
    while r === nothing
        # Kept only when the state held while git ran, the index aside: `git status` itself rewrites the index
        # (refreshing its stat data, every time while an entry is as new as the index), so the answer is kept under
        # the state after it. A file edited meanwhile is a change: computed again.
        r = _code_trees(workdir, paths, julia_project, reg)
        after = _code_state(workdir, paths, julia_project)
        if _but_index(after) == _but_index(seen)
            lock(() -> (_CODE_TREES[k] = (after, r)), _CODE_TREES_LOCK)
        elseif (tries += 1) < 3
            seen, r = after, nothing
        end
    end
    return (; trees = copy(r.trees), dirty = copy(r.dirty), project_paths = copy(r.project_paths))
end

# A path's stat as the state of `code_trees` sees it: what an edit, a replacement or a removal moves.
function _stat_state(p::AbstractString)
    st = stat(p)
    return (String(p), Int64(st.size), Float64(st.mtime), Float64(st.ctime), UInt64(st.inode))
end

# The git directory and common directory of the repository holding `dir` (a worktree's `.git` is a file naming its
# own git directory, whose `commondir` names the shared one), and the repository's top; nothing outside one.
function _git_dirs(dir::AbstractString)
    d = abspath(dir)
    while true
        g = joinpath(d, ".git")
        if isdir(g)
            return (; top = d, git = g, common = g)
        elseif isfile(g)
            m = match(r"^gitdir:\s*(.+?)\s*$"m, read(g, String))
            m === nothing && return nothing
            git = normpath(joinpath(d, String(something(m.captures[1]))))
            c = joinpath(git, "commondir")
            return (; top = d, git, common = isfile(c) ? normpath(joinpath(git, strip(read(c, String)))) : git)
        end
        p = dirname(d)
        p == d && return nothing
        d = p
    end
end

# The state of one repository that git status and rev-parse read beyond the files: HEAD, the ref it names,
# packed-refs, the index, and the ignore files from `under` up to the top.
function _repo_state!(out, dir::AbstractString, under)
    r = _git_dirs(dir)
    r === nothing && return push!(out, ("no git", String(abspath(dir))))
    head = joinpath(r.git, "HEAD")
    push!(out, _stat_state(head), (:index, _stat_state(joinpath(r.git, "index"))...), _stat_state(joinpath(r.common, "packed-refs")),
          _stat_state(joinpath(r.common, "info", "exclude")))
    m = isfile(head) ? match(r"^ref:\s*(\S+)", read(head, String)) : nothing
    m === nothing || push!(out, _stat_state(joinpath(r.common, String(something(m.captures[1])))))
    for u in under
        d = abspath(u)
        while startswith(d, r.top)
            push!(out, _stat_state(joinpath(d, ".gitignore")))
            d == r.top && break
            d = dirname(d)
        end
    end
    return out
end

# The state without the repositories' index files, which git's own status run may rewrite.
_but_index(state) = filter(x -> !(first(x) === :index), state)

# Every file under `p` (a file or a directory, `.git` left out), by its stat.
function _tree_state!(out, p::AbstractString)
    isdir(p) || return push!(out, _stat_state(p))
    for (root, dirs, files) in walkdir(p)
        filter!(!=(".git"), dirs)
        for f in sort(files)
            push!(out, _stat_state(joinpath(root, f)))
        end
    end
    return out
end

# What `code_trees` reads that can change while a process runs, for the memo's key.
function _code_state(workdir, paths, julia_project)
    out = Any[]
    for c in paths
        _tree_state!(out, joinpath(workdir, c))
    end
    _repo_state!(out, workdir, [joinpath(workdir, c) for c in paths])
    manifest = _find_manifest(julia_project)
    for f in ("JuliaProject.toml", "Project.toml")
        push!(out, _stat_state(joinpath(julia_project, f)))
    end
    manifest === nothing && return out
    push!(out, _stat_state(manifest))
    for (_, e) in sort!(collect(_loaded_entries(abspath(julia_project), manifest)); by = first)
        haskey(e, "path") || continue
        dir = normpath(joinpath(dirname(manifest), e["path"]))
        for s in _PACKAGE_CODE
            _tree_state!(out, joinpath(dir, s))
        end
        _repo_state!(out, dir, [joinpath(dir, s) for s in _PACKAGE_CODE])
    end
    return out
end

function _code_trees(workdir, paths, julia_project, reg)
    trees, dirty = Dict{String, String}(), String[]
    reg === nothing || (trees[reg.name] = "$(reg.tree) $(reg.name) $(reg.version) $(reg.uuid)")
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
    project_paths = sort!(collect(keys(trees)))           # a registered install's: its name
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
            elseif name in _STACK_PACKAGES && haskey(e, "git-tree-sha1") && !haskey(trees, name)   # (a registered workdir: named above)
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

# Whether a recorded tree names no committed code: marked, uncommitted, or in no git checkout (`code_trees`).
_is_uncommitted_tree(v::AbstractString) = occursin(_DIRTY_MARK, v) || occursin(r"(^|:)uncommitted( |$)", v) || startswith(v, "not in git: ")

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

A result recorded with uncommitted code (a scratch run's: a tree marked `+uncommitted`, `"uncommitted"`, or in no
git checkout) is never reused, even for the same marked trees: the mark names HEAD, not the edits, so two runs under
different uncommitted edits carry the same value. Each such tree is a change, `"<name>: uncommitted when recorded"`.
"""
function code_tree_changes(recorded, current::AbstractDict)
    recorded === nothing && return ["no code trees recorded (a result from before code identity was kept)"]
    out = String[]
    for k in sort!(unique!([String.(collect(keys(recorded))); String.(collect(keys(current)))]))
        a, b = get(recorded, k, nothing), get(current, k, nothing)
        a !== nothing && _is_uncommitted_tree(string(a)) && (push!(out, "$k: uncommitted when recorded"); continue)
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

# A glob over a relative path: `*` within a segment, `**` across segments, `?` one character.
function _path_glob(g::AbstractString)
    out = IOBuffer(); i = firstindex(g)
    while i <= lastindex(g)
        c = g[i]
        if c == '*' && i < lastindex(g) && g[nextind(g, i)] == '*'
            j = nextind(g, nextind(g, i))
            if j <= lastindex(g) && g[j] == '/'                      # `**/`: any directories, none included
                print(out, "(?:.*/)?"); i = nextind(g, j)
            else
                print(out, ".*"); i = j
            end
            continue
        end
        print(out, c == '*' ? "[^/]*" : c == '?' ? "[^/]" : occursin(c, ".+()[]{}^\$|\\") ? "\\" * c : string(c))
        i = nextind(g, i)
    end
    return Regex("^" * String(take!(out)) * "\$")
end

"""
    tree_sha256(dir; exclude = String[]) -> String

THE content digest of a directory, for an input that is one: the SHA-256 of the lines `<relative path>\t<file
sha256>\n`, one per regular file under `dir` (links followed to their files, not into linked directories), sorted
by relative path (`/`-separated); a path holding a tab or a newline is refused. A relative path matching a glob
of `exclude` (`*` within a segment, `**` across, `**/` any directories including none, `?`) is left out, so a declared scratch file (a log, an index the reader rebuilds) does not move it. Empty
directories do not count. Each file's hash is [`file_sha256`](@ref)'s: streamed, and cached by path, size and
mtime, so a second digest of an unchanged tree costs a `stat` per file.
"""
function tree_sha256(dir::AbstractString; exclude = String[])
    isdir(dir) || throw(ArgumentError("tree_sha256: $dir is not a directory"))
    pats = [_path_glob(g) for g in exclude]
    rows = Tuple{String, String}[]
    for (root, _, files) in walkdir(dir), f in files
        p = joinpath(root, f)
        isfile(p) || continue
        rel = join(splitpath(relpath(p, dir)), "/")
        (occursin('\t', rel) || occursin('\n', rel)) &&
            throw(ArgumentError("tree_sha256: $(repr(rel)) under $dir holds a tab or a newline, which the digest's lines cannot carry"))
        any(r -> occursin(r, rel), pats) && continue
        push!(rows, (rel, file_sha256(p)))
    end
    sort!(rows; by = first)
    return bytes2hex(sha256(join(("$r\t$h\n" for (r, h) in rows))))
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

# The file that marks a directory as an artifact view (the runner's and Experiments' `is_artifact_view`), and a run's
# record (Experiments' `write_provenance`; `scratch_record` reads it).
const ARTIFACT_VIEW_MARKER = ".artifact_view"
const PROVENANCE_FILE = "_provenance.toml"

"""
    tier_data_root(tier, data_root; env = ENV, tiers = nothing, workdir = "", prefix = "VGTK_DATA_ROOT",
                   view_mount = nothing, write = false) -> (root, source)

THE root of an input tier: the one resolver of every tier lookup, read and write — the runner's `missing_inputs`,
and `Experiments.data_root(; tier)`, `input_path`, `verify_inputs`, `keyed_data_dir`, `promote_run` and `tier_path`.
In order:

1. the environment's [`tier_variable`](@ref)`(tier)` (`source = "env"`);
2. the machine's table `tiers` (a stage graph's `[sites.<site>.tiers]`: `tier = "<root>"`, a relative root relative
   to `workdir`; `source = "site"`);
3. when `data_root` is an artifact view that publishes the tier — `view_mount(tier)` returns where the view mounts it
   (`""` its top), `nothing` for a tier it does not publish — `<data_root>/<mount>` (`source = "view"`). The mount
   applies only there: never to a data root set by `VGTK_DATA_ROOT`, an author's own `data/exp_raw` or a fallback;
4. the one `data_root` (`source = "data root"`).

An `[inputs.<id>]` path is relative to its tier's root, so a config names no machine path whichever disk a tier
lives on. With `write = true` (a producer's root) a view tier is refused: an artifact view is read-only, and a
producer never installs one; it binds the tier's root (the environment or the site table) to produce it.
"""
function tier_data_root(tier::AbstractString, data_root::AbstractString; env = ENV, tiers = nothing,
                        workdir::AbstractString = "", prefix::AbstractString = "VGTK_DATA_ROOT",
                        view_mount = nothing, write::Bool = false)
    var = tier_variable(tier; prefix)
    v = String(get(env, var, ""))
    isempty(v) || return (v, "env")
    if tiers !== nothing && !isempty(tier) && haskey(tiers, tier)
        t = String(tiers[tier])
        return (isabspath(t) || isempty(workdir) ? t : normpath(joinpath(workdir, t)), "site")
    end
    m = view_mount === nothing || isempty(tier) ? nothing : view_mount(tier)
    if m !== nothing
        write && throw(ArgumentError("tier $tier is published read-only by the artifact view; bind $var or a site tier " *
                                     "([sites.<site>.tiers]) to produce it"))
        return (isempty(m) ? String(data_root) : joinpath(data_root, String(m)), "view")
    end
    return (String(data_root), "data root")
end

"""
    catalog_view_mount(data_root, catalog) -> Union{Nothing, Function}

The `view_mount` of [`tier_data_root`](@ref) for a data root read without the package (the runner): when
`data_root` is an artifact view (its marker file), each tier the project's data catalog `catalog`
(`configs/data_artifacts.toml`) lists mounts at its entry's `mount_dir` (`""` without one); `nothing` when the root
is no view or there is no catalog. A project builds its view from the same catalog (`DataArtifacts.catalog_view`),
so the runner and the drivers resolve a view tier to the same directory.
"""
function catalog_view_mount(data_root::AbstractString, catalog::AbstractString)
    (isfile(joinpath(data_root, ARTIFACT_VIEW_MARKER)) && isfile(catalog)) || return nothing
    entries = get(TOML.parsefile(catalog), "artifacts", Dict{String, Any}())
    return t -> haskey(entries, t) ? _entry_mount_dir(entries[t]) : nothing
end

# A data-catalog entry's `mount_dir` (`""` without one): the one reader, the runner's and DataArtifacts'.
_entry_mount_dir(a::AbstractDict) = String(get(a, "mount_dir", ""))

"""
    scratch_record(path) -> Bool

Whether `path` (a run directory, or anything at any depth inside one: `run/sub/x.csv`) is the output of a scratch
run: the nearest `_provenance.toml` at or above it, the record of the run it belongs to, says `scratch = true`
(`Experiments.scratch_run`). A scratch output is never an input of record, an upstream run of record, or a promoted
run. A path under no record is no scratch output.
"""
function scratch_record(path::AbstractString)
    d = abspath(isdir(path) ? path : dirname(path))
    while true
        f = joinpath(d, PROVENANCE_FILE)
        isfile(f) && return get(TOML.parsefile(f), "scratch", false) === true
        p = dirname(d)
        p == d && return false
        d = p
    end
end

"""
    inspect_input(id, table, root; root_source = "data root", root_name = "", allow_scratch = false)
        -> (; state, path, problem)

THE check of one `[inputs.<id>]` table under its tier's `root` ([`tier_data_root`](@ref)), for the runner's
`missing_inputs` and a driver's `verify_inputs` alike. `state` is `:present`; `:root_missing` (a tier root other
than the data root, named by `root_name`, that is not there); `:missing` (no `path` under `root`); or `:refused`
(another content than its `sha256` — a file's [`file_sha256`](@ref), a directory's [`tree_sha256`](@ref) with
the table's `sha256_exclude` globs — or a scratch run's output, [`scratch_record`](@ref), unless `allow_scratch`).
`problem` says what, without the input's id. A `sha256` that is no 64 hex digits, or a `sha256_exclude` on a file,
is an `ArgumentError` (a config error).
"""
function inspect_input(id::AbstractString, t::AbstractDict, root::AbstractString; root_source::AbstractString = "data root",
                       root_name::AbstractString = "", allow_scratch::Bool = false)
    rel, tier = String(t["path"]), String(get(t, "tier", ""))
    root_source in ("env", "site") && !isdir(root) &&
        return (; state = :root_missing, path = "", problem = "$tier (root $root, from $root_name, is not there)")
    full = isempty(root) ? "" : joinpath(root, rel)
    (isempty(full) || !ispath(full)) &&
        return (; state = :missing, path = full, problem = "$rel is not under $(isempty(root) ? "<no root>" : root)")
    if haskey(t, "sha256")
        want = lowercase(string(t["sha256"]))
        occursin(r"^[0-9a-f]{64}$", want) || throw(ArgumentError("input $id: sha256 = \"$(t["sha256"])\" is no SHA-256 (64 hex digits)"))
        exclude = String.(get(t, "sha256_exclude", String[]))
        !isempty(exclude) && isfile(full) && throw(ArgumentError("input $id ($rel): sha256_exclude is for a directory, and it is a file"))
        got = isdir(full) ? tree_sha256(full; exclude) : file_sha256(full)
        got == want || return (; state = :refused, path = full, problem = "sha256 expected $want, actual $got")
    end
    !allow_scratch && scratch_record(full) &&
        return (; state = :refused, path = full, problem = "a scratch run's output (its record says scratch = true), never an input of record")
    return (; state = :present, path = full, problem = "")
end
