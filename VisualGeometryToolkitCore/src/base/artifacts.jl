# =============================================================================
# artifacts.jl — the ARTIFACT STORE and the one install of a published Julia
# artifact: by name from an Artifacts.toml, lazily, verified against its pinned
# sha256 and git-tree-sha1, a private Hugging Face dataset with the user's
# token. Hoisted 2026-11-12 from VisualGeometryToolkitEval's experiments/data.jl
# (its `install_artifact`, which Experiments re-exports) so a package without
# an Eval dependency (BlobBoards' descriptor networks and example data) installs
# through the same code.
#
# The store (0.13.0) is where the trees live: a directory of `<git-tree-sha1>/`
# trees in Pkg's layout, on large shared storage, never the Julia depot of a
# cluster (a home quota, or a node-local cache the cluster evicts). Resolution:
#
#   VGTK_ARTIFACT_STORE (env) → the site profile VGTK_SITE names
#   (VisualGeometryToolkitEval's site profiles, `[storage] artifact_store`) →
#   the preference `artifact_store` of this package (Preferences.jl,
#   LocalPreferences.toml) → the depot, refused where the site profile says so
#   (`[storage] refuse_depot = true`) or, with no profile to read, where
#   NODE_CACHE is set (a cluster job's node-local depot)
#
# An install runs inside Pkg's `with_artifacts_directory(store)` (under one
# process-wide lock: Pkg's override is a global), so Pkg's own download keeps
# its sha256 check, its tree-hash check and its atomic move into place; a
# Pidfile lock per tree serialises the processes that install the same tree
# into a shared store (one download, the others wait and read it).
# =============================================================================
using Pkg.Artifacts: ensure_artifact_installed, download_artifact, create_artifact, remove_artifact
using Pkg.GitTools: tree_hash
using Downloads: Downloads
using Artifacts: Artifacts, artifact_exists, artifact_path, with_artifacts_directory
using FileWatching.Pidfile: mkpidlock
using Preferences: load_preference
using Base: SHA1

"""
    is_published_url(url) -> Bool

Whether `url` is a live download URL: a record not yet published or uploaded
binds its placeholder, an UPPER_CASE token (`ZENODO_ID`), as a path segment
where the Zenodo id or the Hugging Face revision goes (VisualGeometryToolkitEval's
DataArtifacts). [`install_artifact`](@ref) refuses such a URL, and a paper
project's release check (VisualGeometryToolkitEval's R3) flags it.
"""
is_published_url(url::AbstractString) = !occursin(r"/[A-Z][A-Z0-9]*(_[A-Z0-9]+)+/", url)

# ---- the store ---------------------------------------------------------------------

"""
    ArtifactStore(path, source)

Where artifact trees are installed and read: `path`, the directory holding the
`<git-tree-sha1>/` trees (Pkg's layout), and `source`, the rung that named it —
`:env` (`VGTK_ARTIFACT_STORE`), `:site` (the site profile `VGTK_SITE` names),
`:preference` (this package's preference `artifact_store`) or `:depot` (the
first depot's `artifacts/`, Pkg's default). [`artifact_store`](@ref) resolves it.
"""
struct ArtifactStore
    path::String
    source::Symbol
end

# The site rung's resolver: `site -> Union{Nothing, NamedTuple, String}`, the `[storage]` of the site profile `site`
# names. VisualGeometryToolkitEval (the owner of the site profiles) sets it when it loads.
const _SITE_STORE = Ref{Union{Nothing, Function}}(nothing)

"""
    set_site_artifact_store!(f) -> f

Register the site rung of [`artifact_store`](@ref): `f(site)` returns the
storage of the site profile `site` names (the value of `VGTK_SITE`) as
`(; artifact_store, refuse_depot)` — its `[storage] artifact_store` (`""` when
it names none) and whether it refuses the depot (`[storage] refuse_depot`) —
or `nothing` when no profile has that name; a path string is read as the
artifact store alone. VisualGeometryToolkitEval, which owns the site profiles,
registers its reader when it loads; without it a `VGTK_SITE` that names a
profile FILE (a path ending in `.toml`) is still read here (its `[storage]`),
and a site NAME gives no rung: the runner gives its jobs
`VGTK_ARTIFACT_STORE` from the profile, so a package without
VisualGeometryToolkitEval (BlobBoards) installs into the site's store there too.
"""
set_site_artifact_store!(f::Function) = (_SITE_STORE[] = f; f)

# The storage of a site: its artifact store ("" for none) and whether it refuses the depot.
const _SiteStorage = @NamedTuple{artifact_store::String, refuse_depot::Bool}

# The `[storage]` of a site profile file, read without its owner (VisualGeometryToolkitEval).
function _site_file_storage(path::AbstractString)
    isfile(path) || error("VGTK_SITE = $path: no such site profile file")
    st = get(TOML.parsefile(path), "storage", Dict{String, Any}())
    s = get(st, "artifact_store", "")
    s isa AbstractString || error("site profile $path: [storage] artifact_store = $(repr(s)) is not a path")
    isempty(s) || isabspath(s) || error("site profile $path: [storage] artifact_store = $(repr(s)) is not an absolute path")
    r = get(st, "refuse_depot", false)
    r isa Bool || error("site profile $path: [storage] refuse_depot = $(repr(r)) is not true or false")
    return _SiteStorage((String(s), r))
end

# The storage of the site `VGTK_SITE` names, or `nothing` (no site, or a name and its reader not loaded).
function _site_storage(site::AbstractString)
    isempty(site) && return nothing
    f = _SITE_STORE[]
    f === nothing && return endswith(site, ".toml") ? _site_file_storage(site) : nothing
    v = f(site)
    v === nothing && return nothing
    v isa AbstractString && return _SiteStorage((String(v), false))       # a reader of the artifact store alone
    return _SiteStorage((String(v.artifact_store), Bool(v.refuse_depot)))
end

# Why the depot is refused here, or `nothing`: the site profile says so (`[storage] refuse_depot`), or, the one
# heuristic for a process that has no profile to read (a site named by name without its reader, or no VGTK_SITE),
# NODE_CACHE is set — the vcai cluster's job environment puts the depot (JULIA_DEPOT_PATH) in that node-local, evicted
# cache. Either way the depot is no place for data: node-local and evicted, or a home with a quota.
function _depot_refusal(site::AbstractString, st)
    st !== nothing && st.refuse_depot && return "the site profile $site sets [storage] refuse_depot"
    isempty(get(ENV, "NODE_CACHE", "")) || return "NODE_CACHE is set: the depot is a cluster node's evicted cache"
    return nothing
end

"""
    artifact_store() -> ArtifactStore

The artifact store of this process, the first rung that names one:

1. `ENV["VGTK_ARTIFACT_STORE"]`;
2. the site profile `ENV["VGTK_SITE"]` names, its `[storage] artifact_store`
   (VisualGeometryToolkitEval's site profiles; [`set_site_artifact_store!`](@ref));
3. this package's preference `artifact_store` (`Preferences.set_preferences!(
   VisualGeometryToolkitCore, "artifact_store" => dir)`, LocalPreferences.toml);
4. the depot (`<first depot>/artifacts`, Pkg's own) — refused where the
   site profile sets `[storage] refuse_depot = true` (VisualGeometryToolkitEval's
   `vcai`: its depot is node-local and evicted) and, with no profile to read
   (no `VGTK_SITE`, or a site name and VisualGeometryToolkitEval not loaded),
   where `NODE_CACHE` is set (that cluster's job environment): one of the
   rungs above must name the store there.

Every rung but the depot names a directory on storage that outlives a job and
is shared by its machines; its trees are content-addressed by their
git-tree-sha1, so every project and package shares one copy of a tree.
"""
function artifact_store()
    e = strip(get(ENV, "VGTK_ARTIFACT_STORE", ""))
    isempty(e) || return ArtifactStore(abspath(e), :env)
    site = String(strip(get(ENV, "VGTK_SITE", "")))
    st = _site_storage(site)
    st === nothing || isempty(st.artifact_store) || return ArtifactStore(abspath(st.artifact_store), :site)
    p = load_preference(@__MODULE__, "artifact_store", "")
    isempty(p) || return ArtifactStore(abspath(p), :preference)
    why = _depot_refusal(site, st)
    why === nothing || error(
        "no artifact store, and the depot is refused ($why): set VGTK_ARTIFACT_STORE, or VGTK_SITE to a site " *
        "profile with [storage] artifact_store (a profile file by its path, or a name once VisualGeometryToolkitEval " *
        "is loaded), or the preference `artifact_store` of VisualGeometryToolkitCore; a depot's trees move with " *
        "`migrate_artifacts`")
    return ArtifactStore(joinpath(first(DEPOT_PATH), "artifacts"), :depot)
end

# Pkg's directory override is a process-wide global that its `with_artifacts_directory` resets to `nothing` on
# exit: every reader and writer of a store holds this lock, and a nested call into the same store does not reset it.
# While a block runs, the override is also what any OTHER task's Pkg artifact lookup sees (a lazy `artifact"…"`
# of another package reads the store then): Pkg offers no task-local form, so the blocks are kept to the install.
const _STORE_LOCK = ReentrantLock()
# The store whose block the current task runs in, or `nothing`: a scoped value, so the tasks a block spawns (the
# packer's) inherit it and enter the same store without the lock their parent holds, and no other task does.
const _IN_STORE = Base.ScopedValues.ScopedValue{Union{Nothing, String}}(nothing)

"""
    with_artifact_store(f, store = artifact_store())

Run `f()` with Pkg's artifact functions (`artifact_exists`, `artifact_path`,
`ensure_artifact_installed`, `create_artifact`, …) reading and writing `store`
(an [`ArtifactStore`](@ref) or a directory): Pkg's `with_artifacts_directory`,
under one process-wide lock (Pkg's override is a global), nesting into the
same store allowed: the block's task and the tasks it spawns enter it without
the lock, and another store from inside a block is an error (it would wait on
the lock the block holds). The depot store runs `f` without an override. The one
wrapper of Pkg's override in the stack: DataArtifacts' packer stages through it.
"""
function with_artifact_store(f::Function, store::ArtifactStore = artifact_store())
    # inside this store's block (the block's own task, or one it spawned, as the packer's): no second lock
    inside = _IN_STORE[]
    inside == store.path && return f()
    # a task of another store's block would wait for ever on the lock its parent holds
    inside === nothing ||
        error("with_artifact_store: inside the store $inside's block; not entering $(store.path) (resolve it before the block)")
    lock(_STORE_LOCK) do
        Artifacts.ARTIFACTS_DIR_OVERRIDE[] === nothing ||
            error("with_artifact_store: Pkg's artifact directory is overridden ($(Artifacts.ARTIFACTS_DIR_OVERRIDE[])) outside the store's lock; not entering $(store.path)")
        store.source === :depot && return Base.ScopedValues.with(f, _IN_STORE => store.path)
        mkpath(store.path)
        return Base.ScopedValues.with(() -> with_artifacts_directory(f, store.path), _IN_STORE => store.path)
    end
end
with_artifact_store(f::Function, dir::AbstractString) = with_artifact_store(f, ArtifactStore(abspath(dir), :env))

# The Pidfile lock of one install into directory `dir` (a store's, or a depot file's), beside Pkg's own `<hash>.pid`
# (which its download takes: a second lock on that file from this process would wait on itself). Stale after a
# minute without its holder.
_install_lock(f, dir::AbstractString, key::AbstractString) =
    (mkpath(dir); mkpidlock(f, joinpath(dir, "$key.install.pid"); stale_age = 60))

# The store remembers which name installed which tree (`names/<name>/<tree hash>`, an empty file each, so
# concurrent installs never write one file): a name with two trees is one name for two different data
# (`store_name_clashes`). The depot keeps no such record (Pkg's gc deletes what it does not know there).
function _record_name(store::ArtifactStore, name::AbstractString, hash::SHA1)
    store.source === :depot && return nothing
    f = joinpath(store.path, "names", name, bytes2hex(hash.bytes))
    isfile(f) && return nothing          # recorded: no write to the shared store on a read
    mkpath(dirname(f))
    touch(f)
    return nothing
end

"""
    store_name_clashes(store = artifact_store()) -> Dict{String, Vector{String}}

The artifact names the store installed under more than one tree: name => its
git-tree-sha1s. One name means one tree stack-wide (a tier re-cut under its old
name is two data under one name); empty when every name has one tree.
"""
function store_name_clashes(store::ArtifactStore = artifact_store())
    root = joinpath(store.path, "names")
    out = Dict{String, Vector{String}}()
    isdir(root) || return out
    for name in readdir(root)
        hs = sort!(readdir(joinpath(root, name)))
        length(hs) > 1 && (out[name] = hs)
    end
    return out
end

# ---- the install ---------------------------------------------------------------------

"""
    install_artifact(name, toml; store = artifact_store()) -> String

Install artifact `name` of `toml` into the artifact store ([`artifact_store`](@ref):
download, verify sha256 and git-tree-sha1, unpack, move into place atomically)
unless present there, and return its tree. Processes installing the same tree
into a shared store take turns (a Pidfile lock per tree: one downloads, the
others then read it). A Hugging Face tarball is downloaded with the user's
Hugging Face token when there is one ([`huggingface_token`](@ref)), so a
private dataset installs for whoever has access to it. A placeholder URL (an
UPPER_CASE token in place of the Zenodo id or the Hugging Face revision: the
record is not published or uploaded) and a failed download (an unpublished
Zenodo draft, a private dataset without a token) are refused with the way out.
A tree in the depot but not in the store is not read: move it with
[`migrate_artifacts`](@ref).
"""
function install_artifact(name::AbstractString, toml::AbstractString; store::ArtifactStore = artifact_store())
    meta = TOML.parsefile(toml)
    haskey(meta, name) || error("unknown artifact '$name' (see $toml)")
    hash = SHA1(meta[name]["git-tree-sha1"])
    return with_artifact_store(store) do
        if !artifact_exists(hash)
            url = meta[name]["download"][1]["url"]
            is_published_url(url) || error(
                "artifact '$name' is not published yet ($toml carries the placeholder URL $url) and no data " *
                "root is bound: set VGTK_DATA_ROOT or link data/exp_raw at the data tree.")
            _install_lock(store.path, bytes2hex(hash.bytes)) do
                artifact_exists(hash) || _download!(name, toml, meta, hash, url)
            end
        end
        _record_name(store, name, hash)
        artifact_path(hash)
    end
end

function _download!(name, toml, meta, hash, url)
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
    return nothing
end

"""
    install_built_artifact(build!, name, hash; store = artifact_store()) -> String

A tree Pkg cannot download and unpack itself (ONNX Runtime's CUDA build,
merged from conda packages: VisualGeometryToolkitFeatures) installed into the
artifact store under its pinned git-tree-sha1 `hash` (a `Base.SHA1`) unless
present there, and its tree returned: `build!(dir)` fills a new directory
(Pkg's `create_artifact`, inside [`with_artifact_store`](@ref)), whose tree
hash must be `hash` (another tree is removed, and is an error). Processes
building the same tree into a shared store take turns under the tree's
Pidfile lock, as [`install_artifact`](@ref)'s do (one builds, the others then
read it), and the store records the tree under `name`
([`store_name_clashes`](@ref)).
"""
function install_built_artifact(build!::Function, name::AbstractString, hash::SHA1; store::ArtifactStore = artifact_store())
    return with_artifact_store(store) do
        if !artifact_exists(hash)
            _install_lock(store.path, bytes2hex(hash.bytes)) do
                artifact_exists(hash) && return nothing
                got = create_artifact(build!)
                got == hash && return nothing
                remove_artifact(got)
                error("artifact '$name': the built tree is $(bytes2hex(got.bytes)), not the pinned $(bytes2hex(hash.bytes))")
            end
        end
        _record_name(store, name, hash)
        artifact_path(hash)
    end
end

"""
    store_file(url, sha256, file; depot_dir, store = artifact_store()) -> String

A single file that is not an artifact tree (a model's weights a release
publishes as a file, LoMa's ONNX graphs) in the artifact store: downloaded once
to `<store>/files/<sha256>/<file>`, checked against its `sha256` (a mismatch
is an error and leaves no file) and moved into place atomically, under the
same per-file Pidfile lock as a tree. With the depot store its home is
`depot_dir` (the caller's scratch space), `<depot_dir>/<file>`.
"""
function store_file(url::AbstractString, digest::AbstractString, file::AbstractString;
                    depot_dir::AbstractString, store::ArtifactStore = artifact_store())
    dir = store.source === :depot ? String(depot_dir) : joinpath(store.path, "files", digest)
    path = joinpath(dir, file)
    isfile(path) && return path
    mkpath(dir)
    _install_lock(store.source === :depot ? dir : store.path, "file-" * digest) do
        isfile(path) && return path
        @info "Downloading $file into the artifact store ($(store.source)): $dir"
        download_verified(url, digest, path)
    end
    return path
end

"""
    download_verified(url, sha256, path) -> String

`url` downloaded to `path`, checked against its `sha256` (hex): a mismatch is
an error and leaves no file at `path`. The bytes land in a temporary file in
`path`'s directory and are moved into place, so no partial file is ever at
`path`. The stack's one verified download of a single file ([`store_file`](@ref)'s;
VisualGeometryToolkitFeatures' ONNX Runtime packages).
"""
function download_verified(url::AbstractString, digest::AbstractString, path::AbstractString)
    tmp = tempname(dirname(path))   # a String: Downloads.download's return is typed loosely
    try
        Downloads.download(url, tmp)
        got = open(sha256_hex, tmp)
        got == digest || error("$url: sha256 $got, expected $digest")
        mv(tmp, path; force = true)   # the same directory: atomic
    finally
        rm(tmp; force = true)
    end
    return String(path)
end

# ---- migration: depot → store ----------------------------------------------------------

"""
    migrate_artifacts(toml; store = artifact_store(), names = keys of toml, remove = false)
        -> Vector{Pair{String, Symbol}}

Move the artifacts of `toml` that the depot holds into the store, so a machine
that installed them before the store existed does not download them again:
each tree is copied into a temporary directory in the store, its git-tree-sha1
recomputed and checked, and moved into place atomically under the tree's
install lock (the depots of `DEPOT_PATH` tried in order, the first copy that
hashes to it migrated); with `remove = true` the depots' copies are deleted
once the store holds the tree. Returns `name => status` per name: `:migrated`,
`:present` (the store had it), `:absent` (no depot holds it either) or
`:mismatch` (no depot copy hashes to its git-tree-sha1: left alone, never
copied). The store must not be the depot.
"""
function migrate_artifacts(toml::AbstractString; store::ArtifactStore = artifact_store(),
                            names = nothing, remove::Bool = false)
    store.source === :depot && throw(ArgumentError("migrate_artifacts: the store is the depot; set VGTK_ARTIFACT_STORE or VGTK_SITE"))
    meta = TOML.parsefile(toml)
    out = Pair{String, Symbol}[]
    for name in something(names, keys(meta))
        haskey(meta, name) || error("unknown artifact '$name' (see $toml)")
        hex = meta[name]["git-tree-sha1"]
        dst = joinpath(store.path, hex)
        src = [joinpath(d, "artifacts", hex) for d in DEPOT_PATH if isdir(joinpath(d, "artifacts", hex))]
        status = if isdir(dst)
            :present
        elseif isempty(src)
            :absent
        else
            _install_lock(store.path, hex) do
                isdir(dst) && return :present
                for s in src           # the first depot copy that hashes to its git-tree-sha1
                    tmp = mktempdir(store.path)
                    try
                        cp(s, joinpath(tmp, "tree"); follow_symlinks = false)
                        bytes2hex(tree_hash(joinpath(tmp, "tree"))) == hex || continue
                        mv(joinpath(tmp, "tree"), dst)
                        return :migrated
                    finally
                        rm(tmp; recursive = true, force = true)
                    end
                end
                :mismatch
            end
        end
        status in (:migrated, :present) && _record_name(store, name, SHA1(hex))
        # never the store's own tree (a store that is a depot's artifacts directory)
        keep = isdir(dst) ? realpath(dst) : ""
        remove && status in (:migrated, :present) &&
            foreach(p -> realpath(p) == keep || rm(p; recursive = true, force = true), src)
        push!(out, String(name) => status)
    end
    return out
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
