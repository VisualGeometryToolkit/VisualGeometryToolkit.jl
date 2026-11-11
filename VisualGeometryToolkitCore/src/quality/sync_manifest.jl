# =============================================================================
# The git hooks that keep a stack repository's untracked Manifest legal
# =============================================================================
#
#   * `sync_manifest`     — re-resolve the active project after git moved a
#                           Project.toml, and report the direct dependencies
#                           that moved.
#   * `install_githooks`  — point a repository's `core.hooksPath` at the hooks
#                           that call it (`githooks/` beside `src/`).
#
# Why: a repository's Manifest is untracked — regenerable from Project.toml plus
# the registries — but not self-regenerating: a pull that raises a [compat]
# lower bound leaves it illegal, and every targeted Pkg operation then dies with
# "empty intersection between X@old and project compatibility new". Nothing in
# git or Pkg runs the regenerate; `githooks/sync-manifest` (installed as
# post-merge, post-checkout and post-rewrite) does, through `sync_manifest`.
# One copy for every stack repository: the monorepo's `.githooks` is a link to
# `VisualGeometryToolkitCore/githooks`, and a downstream repository runs
# `install_githooks(repo)` once per clone.
#
# DEPENDENCY-FREE BY CONTRACT (Base only; Pkg, a standard library, is loaded
# when `sync_manifest` runs): the hook INCLUDES THIS FILE BY PATH rather than
# loading VisualGeometryToolkitCore, because the hook runs exactly when the
# Manifest may be illegal, and loading the package from an illegal Manifest can
# fail (a dependency it lists is not installed yet) or precompile the old
# resolution. VisualGeometryToolkitCore includes the same file.
# =============================================================================

const _PKG_ID = Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg")

"""
    sync_manifest(mode = "update"; io = stdout, repo = nothing) -> NamedTuple{(:created, :moved, :seconds)}

Re-resolve the active project after git moved a Project.toml: `mode =
"update"` runs a full `Pkg.update()` (an untracked Manifest: a targeted update
fails on an illegal Manifest), `mode = "instantiate"` a `Pkg.instantiate()` (a
tracked Manifest: download only). Prints to `io` what moved and returns
`created` (no Manifest held a direct dependency before), `moved` (each direct
dependency whose version changed, `(name, before, after)`, `before ===
nothing` for a new one) and `seconds`. With `repo` (a git working tree whose
`core.hooksPath` [`install_githooks`](@ref) set to an installed
VisualGeometryToolkitCore's hooks), the hooks path is moved to the newly
resolved VisualGeometryToolkitCore's, so a Core update never leaves the
repository on a copy that `Pkg.gc` may delete.

The git hooks run it (`githooks/sync-manifest`) by including this file by path,
without loading the package; it is VisualGeometryToolkitCore's for every other
caller.
"""
function sync_manifest(mode::AbstractString = "update"; io::IO = stdout, repo = nothing)
    mode in ("update", "instantiate") ||
        throw(ArgumentError("sync_manifest: mode is \"update\" or \"instantiate\", got $(repr(mode))"))
    Pkg = Base.require(_PKG_ID)
    direct = Set(keys(Base.invokelatest(Pkg.project).dependencies))
    snapshot() = try
        Dict{String, Any}(i.name => i.version for (_, i) in Base.invokelatest(Pkg.dependencies) if i.name in direct)
    catch
        Dict{String, Any}()
    end
    before = snapshot()
    seconds = @elapsed (mode == "instantiate" ? Base.invokelatest(Pkg.instantiate; io = devnull) :
                                                Base.invokelatest(Pkg.update; io = devnull))
    after = snapshot()
    moved = sort!([(n, get(before, n, nothing), v) for (n, v) in after if get(before, n, nothing) != v]; by = first)
    secs = round(seconds; digits = 1)
    if isempty(before)
        println(io, "Manifest created: $(length(after)) direct dependencies resolved ($secs s)")
    elseif isempty(moved)
        println(io, "Manifest legal; no direct dependency moved ($secs s)")
    else
        println(io, "$(length(moved)) direct dependenc$(length(moved) == 1 ? "y" : "ies") moved ($secs s):")
        for (n, a, b) in moved
            println(io, "  $n  $(a === nothing ? "new" : a) → $b")
        end
    end
    repo === nothing || _follow_resolved_githooks(repo; io)
    return (; created = isempty(before), moved, seconds)
end

"""
    install_githooks(repo = pwd(); hooks = githooks_dir()) -> String

Point the git working tree `repo`'s `core.hooksPath` at `hooks` (default: this
VisualGeometryToolkitCore's `githooks/`, whose `sync-manifest` re-resolves the
repository's root Manifest through [`sync_manifest`](@ref) whenever a
checkout, merge or rebase changes a Project.toml). Run once per clone; the
hooks keep the path on the resolved VisualGeometryToolkitCore afterwards.
Returns the hooks path set. The git config is the repository's, so it holds in
every worktree of it.
"""
function install_githooks(repo::AbstractString = pwd(); hooks::AbstractString = githooks_dir())
    isfile(joinpath(hooks, "sync-manifest")) ||
        throw(ArgumentError("install_githooks: $hooks holds no sync-manifest hook"))
    run(`git -C $repo config core.hooksPath $hooks`)
    return hooks
end

"""
    githooks_dir() -> String

The `githooks/` folder of this VisualGeometryToolkitCore installation: the one copy
of the stack's git hooks (`sync-manifest`, and `post-checkout`, `post-merge`,
`post-rewrite` linking to it), which [`install_githooks`](@ref) installs.
"""
githooks_dir() = normpath(joinpath(@__DIR__, "..", "..", "githooks"))   # src/quality → the package root

# After a re-resolve: a repository whose hooks are an installed
# VisualGeometryToolkitCore's follows the one the active project now resolves.
# A repository-relative hooks path (the monorepo's `.githooks`) is left alone.
function _follow_resolved_githooks(repo; io::IO = stdout)
    current = try
        readchomp(`git -C $repo config --get core.hooksPath`)
    catch
        return nothing
    end
    (isabspath(current) && basename(normpath(current)) == "githooks") || return nothing
    entry = Base.find_package("VisualGeometryToolkitCore")
    entry === nothing && return nothing
    hooks = normpath(joinpath(dirname(entry), "..", "githooks"))
    (hooks != normpath(current) && isfile(joinpath(hooks, "sync-manifest"))) || return nothing
    install_githooks(repo; hooks)
    println(io, "git hooks now VisualGeometryToolkitCore's at $hooks")
    return hooks
end
