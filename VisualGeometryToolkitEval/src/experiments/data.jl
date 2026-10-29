# =============================================================================
# data.jl — the data/results binding of a codified experiment project: the root
# ladder, the artifact view and the input-tree write guard. One implementation
# for every paper project of the stack (hoisted 2026-10-07 from
# VisualGeometryToolkitEval run_records.jl `_root` and the paper projects' own
# src/data.jl).
#
# DrWatson owns the layout: the project is DrWatson's `projectdir()` (or the
# root a project binds with `set_project_root!`), and its data live where
# DrWatson's `datadir` puts them — `data/exp_raw` (inputs, read-only) and
# `data/exp_pro` (results). Per machine those two are bound by
#
#   data_root():    ENV VGTK_DATA_ROOT    → <project>/data/exp_raw → the project's
#                   fallback rung (e.g. a package preference) → its artifact view
#   results_root(): ENV VGTK_RESULTS_ROOT → <project>/data/exp_pro
#
# The artifact view is a DIRECTORY of per-file links into installed Julia
# artifacts (Pkg.Artifacts, sha256- and tree-hash-pinned), marked by
# `.artifact_view`: drivers read it through `data_root()` like any data tree
# and cannot tell whose machine they are on. Its home is the project's
# `data/exp_raw` (`ArtifactView`'s default `dir`), so DrWatson code reading
# `datadir("exp_raw")` reads the view.
# =============================================================================
using DrWatson: projectdir
using Pkg.Artifacts: ensure_artifact_installed, download_artifact
using Downloads: Downloads
using Artifacts: artifact_exists, artifact_path
using Base: SHA1

"""
    ArtifactView(toml; dir, mount = name -> "", default = String[])

A project's published input data as Julia artifacts: `toml` is the
Artifacts.toml binding them (lazy, Zenodo-hosted tarballs), `dir` the view
directory their trees are linked into (default `<project>/data/exp_raw`, so the
view IS the project's DrWatson data directory), `mount(name)` the
subdirectory of `dir` artifact `name` mounts at (trees overlap directories, so
the view is per-file links), and `default` the artifacts [`data_root`](@ref)
installs when asked for none in particular.
"""
struct ArtifactView
    toml::String
    dir::String
    mount::Function
    default::Vector{String}
end
ArtifactView(toml::AbstractString; dir::AbstractString = "", mount::Function = _ -> "", default = String[]) =
    ArtifactView(String(toml), String(dir), mount, String.(default))

# The view directory; "" is the bound project's data/exp_raw, resolved at use.
_view_dir(av::ArtifactView) = isempty(av.dir) ? joinpath(project_root(), "data", "exp_raw") : av.dir

# ---- the project binding ------------------------------------------------------
# One binding per process: a driver runs one project. The binding is what
# `set_project_root!` registers; everything else here reads it.
Base.@kwdef mutable struct _Binding
    root::String = ""
    resolve_links::Bool = false
    data_fallback::Union{Nothing,Function} = nothing
    artifacts::Union{Nothing,ArtifactView} = nothing
    create_results::Bool = false
end
const _BINDING = _Binding()

"""
    set_project_root!(root = projectdir(); resolve_links = false, data_fallback = nothing,
                      artifacts = nothing, create_results = false)

Bind the project the drivers run from (its git record, its `data/exp_raw` and
`data/exp_pro`). The keywords are the per-project rungs of the ladder, kept so
that every existing path resolves byte-identically:

- `resolve_links`: return `realpath` of the `data/exp_raw` link rather than the
  link path itself (bmvc2026: its run keys hash paths under it);
- `data_fallback`: a `() -> path | nothing` rung after the link (bmvc2026's
  `BlobBoards.data_dir()` preference);
- `artifacts`: the [`ArtifactView`](@ref) of last resort;
- `create_results`: create `data/exp_pro` when nothing binds a results root
  (accv2026 errors instead).

`set_project_root!("")` unbinds (the root then follows DrWatson's `projectdir()`).
"""
function set_project_root!(root::AbstractString = projectdir(); resolve_links::Bool = false,
                           data_fallback::Union{Nothing,Function} = nothing,
                           artifacts::Union{Nothing,ArtifactView} = nothing,
                           create_results::Bool = false)
    _BINDING.root = String(root)
    _BINDING.resolve_links = resolve_links
    _BINDING.data_fallback = data_fallback
    _BINDING.artifacts = artifacts
    _BINDING.create_results = create_results
    return nothing
end

"The project the drivers run from: [`set_project_root!`](@ref)'s root, else DrWatson's `projectdir()`."
project_root() = isempty(_BINDING.root) ? projectdir() : _BINDING.root

const ARTIFACT_VIEW_MARKER = ".artifact_view"

"Whether `dir` is an artifact view (a directory carrying the `.artifact_view` marker)."
is_artifact_view(dir::AbstractString) = isfile(joinpath(dir, ARTIFACT_VIEW_MARKER))

# The rungs that need no download: ENV, the project's data/exp_raw (a link, a
# directory or an artifact view), the project's fallback. `nothing` when none binds.
function _bound_data_root()
    haskey(ENV, "VGTK_DATA_ROOT") && return String(ENV["VGTK_DATA_ROOT"])
    link = joinpath(project_root(), "data", "exp_raw")
    if _BINDING.resolve_links
        ispath(link) && return realpath(link)
    else
        (islink(link) || isdir(link)) && return link
    end
    fb = _BINDING.data_fallback
    if fb !== nothing
        r = try fb() catch; nothing end
        r === nothing || return String(r)
    end
    return nothing
end

"""
    data_root(; tiers = nothing) -> String

The project's input tree (read-only): `ENV["VGTK_DATA_ROOT"]` → the project's
`data/exp_raw` (DrWatson `datadir("exp_raw")`; a link, a directory, or the
artifact view) → the project's fallback rung → its [`ArtifactView`](@ref),
installing `tiers` (default: the view's `default`). Configs state input paths
RELATIVE to it; nothing under it is written except by a declared producer
([`assert_not_input`](@ref)).
"""
function data_root(; tiers::Union{Nothing,AbstractVector} = nothing)
    r = _bound_data_root()
    av = _BINDING.artifacts
    if r !== nothing
        # a view already in place still installs what a caller newly asks for
        (av !== nothing && tiers !== nothing && r == _view_dir(av)) && artifact_view!(av, tiers)
        return r
    end
    av === nothing && error("no data root: set VGTK_DATA_ROOT or create the data/exp_raw link under $(project_root())")
    return artifact_view!(av, tiers === nothing ? av.default : tiers)
end

"""
    results_root() -> String

The project's results tree: `ENV["VGTK_RESULTS_ROOT"]` → the project's
`data/exp_pro` (DrWatson `datadir("exp_pro")`), created when the project binds
`create_results = true`, an error otherwise.
"""
function results_root()
    root = haskey(ENV, "VGTK_RESULTS_ROOT") ? String(ENV["VGTK_RESULTS_ROOT"]) :
           joinpath(project_root(), "data", "exp_pro")
    if _BINDING.create_results
        mkpath(root)
    elseif !haskey(ENV, "VGTK_RESULTS_ROOT")
        islink(root) || isdir(root) || error(
            "no results root: set VGTK_RESULTS_ROOT or create the data/exp_pro symlink under $(project_root())")
    end
    return root
end

"""
    is_primary_copy() -> Bool

Whether the data root is a writable primary copy (bound by ENV, the project's
`data/exp_raw` or its fallback) rather than an artifact view, whose trees are
content-addressed and read-only. Same resolution as [`data_root`](@ref).
"""
is_primary_copy() = (r = _bound_data_root(); r !== nothing && !is_artifact_view(r))

"""
    assert_primary_copy(what)

Producers that write INTO the data tree (calibration generators, converters,
packagers) call this first: an artifact view is read-only.
"""
assert_primary_copy(what::AbstractString) = is_primary_copy() || error(
    "$what writes into the data tree, which requires the primary copy; data_root() is the " *
    "read-only artifact view here. Set VGTK_DATA_ROOT (or bind data/exp_raw) to a writable data root.")

# ---- the artifact view ------------------------------------------------------------

# A record not yet published or uploaded binds its placeholder, an UPPER_CASE token, where the Zenodo id
# or the Hugging Face revision goes (DataArtifacts).
_published(url) = !occursin(r"/[A-Z][A-Z0-9]*(_[A-Z0-9]+)+/", url)

"""
    install_artifact(name, toml) -> String

Install artifact `name` of `toml` (download, verify sha256 and git-tree-sha1,
unpack into the depot) unless present, and return its tree. A Hugging Face
tarball is downloaded with the user's Hugging Face token when there is one
([`huggingface_token`](@ref)), so a private dataset installs for whoever has
access to it. A placeholder URL (an UPPER_CASE token in place of the Zenodo id or
the Hugging Face revision: the record is not published or uploaded) and a failed
download (an unpublished Zenodo draft, a private dataset without a token) are
refused with the way out.
"""
function install_artifact(name::AbstractString, toml::AbstractString)
    meta = TOML.parsefile(toml)
    haskey(meta, name) || error("unknown artifact '$name' (see $toml)")
    hash = SHA1(meta[name]["git-tree-sha1"])
    artifact_exists(hash) && return artifact_path(hash)
    url = meta[name]["download"][1]["url"]
    _published(url) || error(
        "artifact '$name' is not published yet ($toml carries the placeholder URL $url) and no data " *
        "root is bound: set VGTK_DATA_ROOT or link data/exp_raw at the data tree.")
    hf = startswith(url, "https://huggingface.co/")
    token = hf ? huggingface_token() : nothing
    try
        if token === nothing
            ensure_artifact_installed(name, toml)
        else
            mktempdir() do dir
                tarball = Downloads.download(url, joinpath(dir, "$name.tar.gz"); headers = ["Authorization" => "Bearer $token"])
                r = download_artifact(hash, "file://" * tarball, meta[name]["download"][1]["sha256"])
                r === true || throw(r)            # Pkg returns the verification or unpack error
            end
        end
    catch err
        why = hf && token === nothing ?
            "A private Hugging Face dataset needs your token: log in with `hf auth login` (or set HF_TOKEN) and retry." :
            "If its record is still an UNPUBLISHED Zenodo draft its file URLs are not live yet."
        error("downloading artifact '$name' ($toml) failed. $why Otherwise bind a data root (VGTK_DATA_ROOT or " *
              "data/exp_raw). Underlying error: $(sprint(showerror, err))")
    end
    return artifact_path(hash)
end

"""
    huggingface_token() -> Union{String, Nothing}

The user's Hugging Face token: `ENV["HF_TOKEN"]`, else the file `hf auth login`
writes (`\$HF_HOME/token`, `HF_HOME` defaulting to `~/.cache/huggingface`), else
`nothing`. [`install_artifact`](@ref) sends it to huggingface.co only.
"""
function huggingface_token()
    t = strip(get(ENV, "HF_TOKEN", ""))
    isempty(t) || return String(t)
    f = joinpath(get(ENV, "HF_HOME", joinpath(homedir(), ".cache", "huggingface")), "token")
    isfile(f) || return nothing
    t = strip(read(f, String))
    return isempty(t) ? nothing : String(t)
end

"""
    link_tree!(tree, dst) -> Int

Link every file of `tree` into `dst` (per-file symlinks: artifact trees share
directories, so directory links would not compose); an existing path — a link
made earlier or an author's own file — is left alone. Returns the number of
new links.
"""
function link_tree!(tree::AbstractString, dst::AbstractString)
    n = 0
    for (r, _, fs) in walkdir(tree), f in fs
        target = joinpath(dst, relpath(joinpath(r, f), tree))
        (ispath(target) || islink(target)) && continue
        mkpath(dirname(target))
        symlink(joinpath(r, f), target)
        n += 1
    end
    return n
end

"""
    artifact_view!(view, names) -> String

Install the artifacts `names` of `view` ([`install_artifact`](@ref)) and link
each tree into the view directory at its mount ([`link_tree!`](@ref)), marking
the directory `.artifact_view`. Idempotent; returns the view directory.
"""
function artifact_view!(view::ArtifactView, names::AbstractVector)
    for name in names
        tree = install_artifact(String(name), view.toml)
        link_tree!(tree, joinpath(_view_dir(view), view.mount(String(name))))
    end
    mkpath(_view_dir(view))
    marker = joinpath(_view_dir(view), ARTIFACT_VIEW_MARKER)
    isfile(marker) || write(marker,
        "An ARTIFACT VIEW: per-file links into the installed artifact trees of $(view.toml), plus whatever " *
        "declared producers write. Read-only inputs; remove the directory to rebuild it.\n")
    return _view_dir(view)
end

# ---- the write guard ---------------------------------------------------------------

# A path with every EXISTING ancestor resolved (the leaf may not exist yet).
function _resolved_path(p::AbstractString)
    p = abspath(p)
    rest = String[]
    while !ispath(p) && dirname(p) != p
        pushfirst!(rest, basename(p))
        p = dirname(p)
    end
    return joinpath(ispath(p) ? realpath(p) : p, rest...)
end

_is_under(p, root) = (c = _resolved_path(p); r = _resolved_path(root); c == r || startswith(c, rstrip(r, '/') * "/"))

"""
    assert_not_input(path; producer = false)

The runtime guard of every writer here: refuse a write at or under
[`data_root`](@ref) — the read-only input tree — unless the caller declares
itself a `producer` (a pool generator writing a NEW tree under the data root).
A results root nested inside the data root (one author binding has
`data_root()/results/…`) is not input. Nothing is guarded when no data root is
bound.
"""
function assert_not_input(path::AbstractString; producer::Bool = false)
    producer && return nothing
    droot = _bound_data_root()
    droot === nothing && return nothing
    _is_under(path, droot) || return nothing
    rroot = try results_root() catch; nothing end
    rroot !== nothing && _is_under(path, rroot) && return nothing
    error("refusing to write $path: it is under the data root $droot, the read-only input tree " *
          "(results go under results_root(); a producer of a new input tree declares producer = true)")
end
