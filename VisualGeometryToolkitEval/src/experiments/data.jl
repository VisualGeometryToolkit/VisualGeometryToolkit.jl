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
#   data_root():    ENV VGTK_DATA_ROOT    → <project>/data/exp_raw → the stage
#                   graph's [sites.<site>] data_root → the site profile's
#                   data_root (VGTK_SITE) → the project's fallback rung (e.g. a
#                   package preference) → its artifact view
#   results_root(): ENV VGTK_RESULTS_ROOT → <project>/data/exp_pro → the stage
#                   graph's [sites.<site>] results_root → the site profile's
#                   results_root (VGTK_SITE)
#
# The order up to the site profile is THE ladder of a project's roots, owned by
# the runner (Reproduce `_root_ladder`) and shared with a launch's roots
# (`resolve_roots`, which has a command line's roots first and no project link).
#
# The site profiles (VisualGeometryToolkitEval's configs/sites/, `[storage]`) are
# the only place a machine's paths are written; a project names none.
#
# The artifact view is a DIRECTORY of per-file links into installed Julia
# artifacts (Pkg.Artifacts, sha256- and tree-hash-pinned), marked by
# `.artifact_view`: drivers read it through `data_root()` like any data tree
# and cannot tell whose machine they are on. Its home is the project's
# `data/exp_raw` (`ArtifactView`'s default `dir`), so DrWatson code reading
# `datadir("exp_raw")` reads the view.
# =============================================================================
using DrWatson: projectdir
# The one artifact install (download, verify, unpack, a Hugging Face token) is
# VisualGeometryToolkitCore's (base/artifacts.jl); Experiments re-exports it.
using VisualGeometryToolkitCore: install_artifact

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

"Whether `dir` is an artifact view (a directory carrying the `.artifact_view` marker)."
is_artifact_view(dir::AbstractString) = isfile(joinpath(dir, ARTIFACT_VIEW_MARKER))

# ---- the site profile ------------------------------------------------------------

"""
    active_site() -> Union{Nothing, Site}

The site profile `ENV["VGTK_SITE"]` names (a shipped site's name, a profile
file's path, or a name in the bound project's own sites directory, the stage
graph's `[project] sites_dir`), read by the runner's `load_site`; `nothing`
when `VGTK_SITE` is unset or empty, or names no profile (a site a project's
stage graph alone knows, `[sites.<site>]`).
"""
active_site() = _active_site()[2]

"""
    site_profile(spec) -> Union{Nothing, Site}

The site profile `spec` names, as [`active_site`](@ref) reads `VGTK_SITE`: the
runner's `load_site` over the bound project's own sites directory and the
shipped profiles; `nothing` for a name no profile has (a malformed profile is
an error). VisualGeometryToolkitCore's `artifact_store` reads its
`[storage] artifact_store` through it (registered when this package loads).
"""
site_profile(spec::AbstractString) = _site_profile(spec, _project_graph())

function _site_profile(spec::AbstractString, g)
    dirs = g === nothing || isempty(g.sites_dir) ? String[] : [g.sites_dir]
    path = _reproduce()._site_path(String(spec); dirs)    # the runner's lookup: `nothing` for an unknown name
    path === nothing && return nothing
    return _cached_read(_reproduce().load_site, _SITE_CACHE, path)   # once per change of the file
end
const _SITE_CACHE = Dict{String, Tuple{Float64, Any}}()

# `read(path)` once per change of the file (its mtime), kept in `cache`; the caches are shared by a driver's threads.
const _CACHE_LOCK = ReentrantLock()
function _cached_read(read::Function, cache::AbstractDict, path::AbstractString)
    t = mtime(path)
    @lock _CACHE_LOCK begin
        c = get(cache, path, nothing)
        (c === nothing || c[1] != t) && (c = cache[path] = (t, read(path)))
        return c[2]
    end
end

# The site VGTK_SITE names: its name (the profile's; VGTK_SITE as given for a site the stage graph alone knows,
# `[sites.<site>]`), its profile or `nothing`, and the bound project's graph `graph()`, read only with VGTK_SITE set;
# ("", nothing, nothing) without it. A job of the runner's gets the profile's name, or its file for a profile loaded by
# path.
function _active_site(graph::Function = _project_graph)
    spec = String(strip(get(ENV, "VGTK_SITE", "")))
    isempty(spec) && return ("", nothing, nothing)
    g = graph()
    s = _site_profile(spec, g)
    return (s === nothing ? spec : s.name, s, g)
end

# The runner (Reproduce), included after this module: resolved at the call.
_reproduce() = parentmodule(@__MODULE__).Reproduce

# The bound project's stage graph (configs/reproduce.toml), or `nothing`; parsed once per change of the file (every
# root lookup on a site reads it).
const _GRAPH_CACHE = Dict{String, Tuple{Float64, Any}}()
function _project_graph()
    root = project_root()
    path = isempty(root) ? "" : joinpath(root, "configs", "reproduce.toml")
    isfile(path) || return nothing
    return _cached_read(_reproduce().load_stage_graph, _GRAPH_CACHE, path)
end

"""
    project_name() -> String

The bound project's name, the `{project}` of a site profile's roots: its stage
graph's `[project] name` (`configs/reproduce.toml`), else the name of its
directory.
"""
project_name() = _project_name(_project_graph())
_project_name(g) = g === nothing ? basename(rstrip(project_root(), '/')) : g.name

# A driver's root `key` ("data_root" or "results_root") through THE ladder of a project's roots, the runner's
# (`Reproduce._root_ladder`, shared with a launch's `resolve_roots`): `var` in ENV → `bound` (the project's own
# data/exp_raw or data/exp_pro) → the stage graph's `[sites.<site>]` root → the site profile's, each of the last two
# once `ready` (it exists); `nothing` when none binds. The graph and the profile are read only past the first two, and
# only with VGTK_SITE set.
function _driver_root(key::AbstractString, var::AbstractString, bound; ready::Function)
    function sites()
        name, s, g = _active_site()
        isempty(name) && return _reproduce()._NO_SITES
        table = g === nothing ? nothing : get(g.site_roots, name, nothing)
        return (; table, site = s, project = _project_name(g), workdir = g === nothing ? "" : g.workdir)
    end
    return _reproduce()._root_ladder(key; env = ENV, var, bound, sites, ready)
end

# The project's own binding of its data/exp_raw (a link, a directory or an artifact view), or `nothing`.
function _project_data_binding()
    link = joinpath(project_root(), "data", "exp_raw")
    _BINDING.resolve_links && return ispath(link) ? realpath(link) : nothing
    return islink(link) || isdir(link) ? link : nothing
end

# The rungs that need no download: the ladder (ENV, the project's data/exp_raw, the site's roots once they exist; a
# site's root not made yet gives the artifact view, as before), then the project's fallback. `nothing` when none binds.
function _bound_data_root()
    r = _driver_root("data_root", "VGTK_DATA_ROOT", _project_data_binding(); ready = isdir)
    r === nothing || return r
    fb = _BINDING.data_fallback
    if fb !== nothing
        r = try fb() catch; nothing end
        r === nothing || return String(r)
    end
    return nothing
end

"""
    data_root(; tiers = nothing, tier = nothing) -> String

The project's input tree (read-only): `ENV["VGTK_DATA_ROOT"]` → the project's
`data/exp_raw` (DrWatson `datadir("exp_raw")`; a link, a directory, or the
artifact view) → the stage graph's `[sites.<site>] data_root` once it exists →
the site profile's `data_root` once it exists ([`active_site`](@ref), its
`{project}` the [`project_name`](@ref)) → the project's fallback rung → its [`ArtifactView`](@ref),
installing `tiers` (default: the view's `default`). Configs state input paths
RELATIVE to it; nothing under it is written except by a declared producer
([`assert_not_input`](@ref)).

With `tier`, the root of that input tier, which may live elsewhere on a
machine (`tier_data_root`, the runner's order): `ENV["<data_env>_<TIER>"]`
(the graph's `data_env`, default `VGTK_DATA_ROOT`) → the project's stage graph
(`configs/reproduce.toml`, read by the runner's `load_stage_graph`)
`[sites.<site>.tiers]`, `<site>` from `ENV["VGTK_SITE"]` (the runner sets both
for its jobs) → the one data root above, through the same resolver and the
runner's view gate (`catalog_view_mount`): a tier the project's catalog
(`configs/data_artifacts.toml`) lists is the view's, its root where the view
mounts it (`<data root>/<mount_dir>`, a split entry's parts included), installed
when the view's Artifacts.toml binds it; a tier the catalog does not list is
only looked up. An `[inputs.<id>]` path is relative to its tier's root, so
[`input_path`](@ref) fetches what it reads.
"""
function data_root(; tiers::Union{Nothing,AbstractVector} = nothing, tier::Union{Nothing,AbstractString} = nothing)
    tier === nothing || return _tier_root(tier; write = false)
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
`data/exp_pro` (DrWatson `datadir("exp_pro")`; a link or a directory) → the
stage graph's `[sites.<site>] results_root`, then the site profile's
`results_root` ([`active_site`](@ref)), each once it exists or when the
project binds `create_results = true`; without any of them `data/exp_pro`.
The order is the runner's launch's too (Reproduce `resolve_roots`), one ladder. The root is created when the project binds
`create_results = true`; otherwise a missing `data/exp_pro` is an error.
"""
function results_root()
    local_root = joinpath(project_root(), "data", "exp_pro")
    # a site's root is used once it exists (as the data root's), or when the project creates its results root
    r = _driver_root("results_root", "VGTK_RESULTS_ROOT", islink(local_root) || isdir(local_root) ? local_root : nothing;
                     ready = p -> !isempty(p) && (_BINDING.create_results || isdir(p)))
    root, bound = r === nothing ? (local_root, false) : (r, true)
    if _BINDING.create_results
        mkpath(root)
    elseif !bound
        error(
            "no results root: set VGTK_RESULTS_ROOT, VGTK_SITE to a site profile with [storage] results_root, " *
            "or create the data/exp_pro symlink under $(project_root())")
    end
    return root
end

"""
    project_results_root(project) -> Union{Nothing, String}

Where ANOTHER project's results live on this machine's site: the active site profile's `results_root` for that
project ([`active_site`](@ref), its `{project}` the project's name: the runner's
`site_root(site, "results_root", project)`); `nothing` without one. A config's product input
(`[inputs.<id>] project = …`, [`input_path`](@ref)) is read under it, after `VGTK_RESULTS_ROOT_<PROJECT>`.
"""
function project_results_root(project::AbstractString)
    s = active_site()
    s === nothing && return nothing
    r = _reproduce().site_root(s, "results_root", String(project))
    return isempty(r) ? nothing : r
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

Install the artifacts `names` of `view` (VisualGeometryToolkitCore's `install_artifact`) and link
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
    return foldl(joinpath, rest; init = ispath(p) ? realpath(p) : p)
end

# Whether path `p` is `root` or lies under it, as text (neither is resolved; DataArtifacts' produced entries' `covers`).
_path_within(p, root) = p == root || startswith(p, rstrip(root, '/') * "/")
_is_under(p, root) = _path_within(_resolved_path(p), _resolved_path(root))

# Whether the artifact view's Artifacts.toml publishes `name` (a tier that is no artifact is only looked up). The
# parsed file is cached by path and mtime.
const _VIEW_TOML = Dict{String, Tuple{Float64, Set{String}}}()
function _view_has(av::ArtifactView, name::AbstractString)
    isfile(av.toml) || return false
    t = mtime(av.toml)
    c = get(_VIEW_TOML, av.toml, nothing)
    if c === nothing || c[1] != t
        c = (t, Set{String}(keys(TOML.parsefile(av.toml))))
        _VIEW_TOML[av.toml] = c
    end
    return name in c[2]
end

# Whether `path` is an artifact view or lies inside one (its marker at or above the nearest existing ancestor).
function _inside_view(path::AbstractString)
    d = _resolved_path(path)
    while true
        isdir(d) && is_artifact_view(d) && return true
        p = dirname(d)
        p == d && return false
        d = p
    end
end

# A tier's root through THE resolver (`tier_data_root`, src/code_identity.jl): the environment, the site table, the
# artifact view when the data root is one (installing the tier to read it; refused to a producer), the data root.
# `write = true` is a producer's root: it never installs an artifact.
function _tier_root(tier::AbstractString; write::Bool)
    site_tiers, workdir, prefix = _site_tiers()
    catalog = joinpath(project_root(), "configs", "data_artifacts.toml")
    results_mount = catalog_results_mount(catalog)
    if results_mount !== nothing && results_mount(tier) !== nothing   # a run of record published as a tier: no data ladder
        r, _ = tier_data_root(tier, ""; env = ENV, tiers = site_tiers, workdir, prefix, results_mount,
                              results_root, write)          # the results root read only when the env and site rungs miss
        return r
    end
    bound = _bound_data_root()
    av = _BINDING.artifacts
    from_view = av !== nothing && (bound === nothing || is_artifact_view(bound))
    # a reader's view is in place before its tiers are looked up (the gate reads its marker); a producer's never made
    main = bound !== nothing ? bound : av === nothing ? "" : write ? _view_dir(av) : artifact_view!(av, String[])
    # the runner's view gate (src/code_identity.jl): a tier the project's catalog lists mounts at its `mount_dir`
    view_mount = from_view ? catalog_view_mount(main, catalog) : nothing
    r, source = tier_data_root(tier, main; env = ENV, tiers = site_tiers, workdir, prefix, view_mount, write)
    if write
        isempty(r) && error("no data root to produce tier $tier into: set VGTK_DATA_ROOT, $(tier_variable(tier; prefix)) or a site tier")
        # a producer never writes into an artifact view, whatever named its root (the view's own directory, a
        # VGTK_DATA_ROOT or a tier root that is one, or a directory inside one)
        if (from_view && source == "data root") || _inside_view(r)
            throw(ArgumentError("the data root is an artifact view (read-only); bind VGTK_DATA_ROOT or " *
                                "$(tier_variable(tier; prefix)) to a writable root to produce tier $tier"))
        end
        return r
    end
    source in ("env", "site") && return r
    # installed (an entry the view's Artifacts.toml binds; a split entry's parts are fetched by `fetch_artifacts!`), then its mount
    source == "view" && (av !== nothing && _view_has(av, tier) && data_root(; tiers = String[tier]); return r)
    return data_root()
end

# The project's stage graph's `[sites.<site>.tiers]` for the site `VGTK_SITE` names (by its name), the graph's workdir (a relative
# root's base) and its `data_env` (the prefix of a tier's variable), read by the runner's own reader
# (`Reproduce.load_stage_graph`, included after this module: resolved at the call); `(nothing, "", "VGTK_DATA_ROOT")`
# without one.
function _site_tiers()
    site, _, g = _active_site()                         # by the site's name, VGTK_SITE a name or a profile's file
    (isempty(site) || g === nothing) && return (nothing, "", "VGTK_DATA_ROOT")
    return (get(get(g.site_roots, site, Dict{String,Any}()), "tiers", nothing), g.workdir, g.data_env)
end

"""
    assert_not_input(path; producer = false)

The runtime guard of every writer here: refuse a write at or under
[`data_root`](@ref) — the read-only input tree — unless the caller declares
itself a `producer` (a pool generator writing a NEW tree under the data root).
A results root nested inside the data root (one author binding has
`data_root()/results/…`) is not input. Nothing is guarded when no data root is
bound. A tier root set elsewhere (`VGTK_DATA_ROOT_<TIER>`, [`data_root`](@ref)'s
`tier`) is input too.
"""
function assert_not_input(path::AbstractString; producer::Bool = false)
    producer && return nothing
    site_tiers, workdir, prefix = _site_tiers()        # the resolver's table: the site's tier roots are input too
    for (k, v) in ENV
        (startswith(k, prefix * "_") && !isempty(v) && _is_under(path, v)) &&
            error("refusing to write $path: it is under the root of an input tier ($k = $v), read-only input")
    end
    if site_tiers !== nothing
        for (t, _) in site_tiers
            r = first(tier_data_root(String(t), ""; env = Dict{String,String}(), tiers = site_tiers, workdir))
            isempty(r) || !_is_under(path, r) ||
                error("refusing to write $path: it is under the root of the input tier $t ([sites.<site>.tiers]), read-only input")
        end
    end
    droot = _bound_data_root()
    droot === nothing && return nothing
    _is_under(path, droot) || return nothing
    rroot = try results_root() catch; nothing end
    rroot !== nothing && _is_under(path, rroot) && return nothing
    error("refusing to write $path: it is under the data root $droot, the read-only input tree " *
          "(results go under results_root(); a producer of a new input tree declares producer = true)")
end
