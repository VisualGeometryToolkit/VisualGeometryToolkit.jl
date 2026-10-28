# =============================================================================
# DataArtifacts — a paper project's published input data as Julia artifacts:
# the CATALOG (what each artifact holds, where it mounts, which Zenodo record
# carries it, from which config of record its file list is derived), the
# author-side PACKER (Pkg.Artifacts builds and binds the lazy tarballs) and
# the consumer-side FETCH (install, then link into the roots the drivers read).
# One implementation for every paper project (accv2026 first; RSDR's
# neurips2026 next), over the Experiments data layer: the artifact view,
# `install_artifact`, `link_tree!`, `data_root()`/`results_root()`.
#
# The catalog is a TOML file in the project's configs/ (accv2026:
# configs/data_artifacts.toml):
#
#   [catalog]                         optional
#   artifacts_toml     = "../Artifacts.toml"   the bindings, relative to the catalog
#   artifacts_toml_env = "ACCV2026_ARTIFACTS_TOML"  a variable naming another
#                                     copy (other URLs, same hashes: the
#                                     fresh-machine test's file:// copy)
#   [records.<group>]   title, placeholder, id?, license?   one Zenodo record
#                       each; `id` once the record exists (the bindings then
#                       name it), `placeholder` until then
#   [artifacts.<name>]  kind, mount, record, required, what, provenance,
#                       + the kind's keys:
#     pool_sources  config, splits   the raw files the pool producer `config`
#                                    reads for the scenes of `splits`
#     pools         config, splits   the pool tree `config` writes
#     stored        config           the directory `config`'s [stored] dir names
#     tree          dir              every file under `dir`
#   split_by = "scene"   (pool_sources, pools; optional) bind the entry as one
#                        artifact PER SCENE, `<name>_<scene>`, plus `<name>_meta`
#                        for the files no scene owns (a pool tree's run records,
#                        a correspondence manifest) — for an entry too large to
#                        upload as one file. The entry stays one logical
#                        artifact: `required`, `fetch_artifacts!` and
#                        `pack_artifacts` take its name and act on every part.
#   max_part_bytes = N   (with split_by; optional) the size cap of one part, in
#                        uncompressed bytes. A scene whose files exceed it is
#                        bound as sub-parts `<name>_<scene>_part1..k`: its files
#                        in order, a file larger than N cut into the fewest
#                        near-equal chunks `<file>.chunk<i>of<m>` (each at most
#                        N), packed greedily into parts of at most N bytes.
#                        `fetch_artifacts!` reassembles each chunked file in the
#                        artifact view, byte for byte. The rule, not a list:
#                        the author's file sizes derive the parts; a consumer
#                        reads them from the bindings.
#
# Every artifact tree holds paths RELATIVE TO ITS MOUNT ROOT — `data` is
# data_root(), `results` is results_root() — so the author's tree, a packed
# artifact and a fetched view have one layout. A project adds kinds of its own
# with the `kinds` keyword (`kind => (cfg, entry, root) -> files`).
# =============================================================================
"""
    DataArtifacts

A paper project's published input data as lazy Julia artifacts. The catalog
([`load_catalog`](@ref), [`DataCatalog`](@ref)) states what each artifact holds,
where it mounts, which Zenodo record carries it and which config of record its
file list derives from. The author packs and binds the artifacts
([`pack_artifacts`](@ref)); a consumer fetches them into the artifact view that
the drivers read ([`fetch_artifacts!`](@ref)). [`catalog_violations`](@ref)
checks a catalog against its bindings.
"""
module DataArtifacts

using TOML: TOML
using SHA: sha256
using Base: SHA1
using Pkg.Artifacts: bind_artifact!, create_artifact, archive_artifact, remove_artifact
using Artifacts: with_artifacts_directory
using ..Experiments: data_root, results_root, project_root, load_run_config, install_artifact,
    link_tree!, is_artifact_view, ARTIFACT_VIEW_MARKER
using ..Tuning: registered_split
using ..Datasets: PtRawTrees, pt_scene_files

export DataCatalog, load_catalog, required_artifacts, split_scenes, artifact_files, artifact_parts,
    fetch_artifacts!, pack_artifacts, zenodo_url, catalog_violations

const KINDS = ("pool_sources", "pools", "stored", "tree")
const MOUNTS = ("data", "results")
const ENTRY_KEYS = ("kind", "mount", "record", "required", "what", "provenance")
const SPLIT_KINDS = ("pool_sources", "pools")
const META_PART = "meta"
const SUBPART = "_part"
const CHUNK_RE = r"^(.+)\.chunk([0-9]+)of([0-9]+)$"

# One file of a packed artifact tree: `dst` (its path in the tree, relative to
# the mount root) holds bytes `offset .+ (0:len-1)` of the author's file `src`.
# A whole file has dst == src, offset 0, len its size.
struct Piece
    dst::String
    src::String
    offset::Int
    len::Int
end

"""
    DataCatalog

A loaded data-artifact catalog ([`load_catalog`](@ref)): `path` (the catalog
TOML), `artifacts_toml` (the Artifacts.toml its artifacts are bound in, after
the `artifacts_toml_env` override), `configs` (the directory its `config`
names resolve in: the catalog's own), `records` and `artifacts` (the tables).
"""
struct DataCatalog
    path::String
    artifacts_toml::String
    configs::String
    records::Dict{String,Any}
    artifacts::Dict{String,Any}
end

"""
    load_catalog(path) -> DataCatalog

Read the catalog at `path`. Its bindings are `[catalog] artifacts_toml`
(relative to the catalog; default `../Artifacts.toml`, the project root's),
unless the variable `[catalog] artifacts_toml_env` names another copy.
"""
function load_catalog(path::AbstractString)
    path = abspath(path)
    raw = TOML.parsefile(path)
    meta = get(raw, "catalog", Dict{String,Any}())
    toml = normpath(joinpath(dirname(path), get(meta, "artifacts_toml", joinpath("..", "Artifacts.toml"))))
    env = get(meta, "artifacts_toml_env", "")
    isempty(env) || (toml = get(ENV, env, toml))
    return DataCatalog(path, toml, dirname(path), get(raw, "records", Dict{String,Any}()),
                       get(raw, "artifacts", Dict{String,Any}()))
end

"The names of the artifacts fetched by default (`required = true`), sorted."
required_artifacts(cat::DataCatalog) = sort([n for (n, a) in cat.artifacts if get(a, "required", false)])

"""
    split_scenes(side) -> Vector{String}

The scenes of one side of a registered scene split, `side` =
`"<split>:validation"` or `"<split>:test"` (configs/splits.toml,
`Tuning.registered_split`): how a catalog entry's (and a pool producer
config's) `splits` names its scenes.
"""
function split_scenes(side::AbstractString)
    parts = split(side, ':')
    (length(parts) == 2 && parts[2] in ("validation", "test")) || throw(ArgumentError(
        "split side $(repr(side)): expected <split>:validation or <split>:test (a split registered in configs/splits.toml)"))
    reg = registered_split(String(parts[1]))
    return parts[2] == "validation" ? reg.validation_scenes : reg.test_scenes
end

_scenes(splits) = sort(unique(reduce(vcat, split_scenes.(splits); init = String[])))

# The raw files the pool producer `cfg` reads for `scene`, relative to the data
# root: the readers' own path rules (Datasets), applied to the config's trees.
function _pool_source_files(cfg, scene)
    src = cfg["source"]
    route = get(src, "route", cfg["experiment"]["dataset"])
    if route == "phototourism"
        trees = PtRawTrees(; (Symbol(chop(k; tail = 4)) => src[k] for k in keys(src) if endswith(k, "_dir"))...,
                           recon_special = Dict{String,String}(get(src, "recon_special", Dict{String,String}())))
        return pt_scene_files(scene, trees)
    elseif route == "heb"
        return [joinpath(src["raw_dir"], "$(scene)_homographies.h5")]
    elseif route == "roma"
        return [joinpath(src["input_jld2_dir"], "$(scene).jld2")]
    end
    error("data artifacts: [source] route = $route has no file rule")
end

_files_under(root, d) = [joinpath(d, relpath(joinpath(r, f), joinpath(root, d)))
                         for (r, _, fs) in walkdir(joinpath(root, d)) for f in fs]

# (files no scene owns, [scene => its files]) of a pool_sources / pools entry.
# A pool tree ships its run records (_config.toml, _provenance.toml), never a
# source-tree diff (_dirty.patch: the policy of record — no artifact ships one).
function _scene_files(kind, cfg, scenes, root)
    if kind == "pool_sources"
        shared = String[]
        # the extraction manifest beside consolidated correspondences (provenance)
        if get(cfg["source"], "route", "") == "roma"
            m = joinpath(cfg["source"]["input_jld2_dir"], "manifest.tsv")
            isfile(joinpath(root, m)) && push!(shared, m)
        end
        return shared, [s => _pool_source_files(cfg, s) for s in scenes]
    end
    pd = cfg["experiment"]["pool_dir"]
    shared = [joinpath(pd, f) for f in ("_config.toml", "_provenance.toml") if isfile(joinpath(root, pd, f))]
    return shared, [s => [joinpath(pd, "$s.jld2"), joinpath(pd, "$s.residuals.jld2")] for s in scenes]
end

function _kind_files(kind, cfg, a, root)
    if kind in SPLIT_KINDS
        shared, per = _scene_files(kind, cfg, _scenes(a["splits"]), root)
        return reduce(vcat, last.(per); init = shared)
    elseif kind == "stored"
        return _files_under(root, String(cfg["stored"]["dir"]))
    elseif kind == "tree"
        return _files_under(root, String(a["dir"]))
    end
    error("unknown artifact kind $kind")
end

_entry_config(cat, a) = haskey(a, "config") ? load_run_config(joinpath(cat.configs, a["config"] * ".toml")) : nothing
_is_split(a) = !isempty(get(a, "split_by", ""))

_cap(a) = get(a, "max_part_bytes", 0)

# The parts of `files` (relative to `root`) under the cap: whole files in order,
# a file larger than the cap cut into m = cld(size, cap) near-equal chunks, then
# packed greedily into groups of at most `cap` bytes. One group when everything fits.
function _plan(files, root, cap)
    pieces = Piece[]
    for f in files
        n = filesize(joinpath(root, f))
        if cap <= 0 || n <= cap
            push!(pieces, Piece(f, f, 0, n))
        else
            m = cld(n, cap)
            cuts = [fld(i * n, m) for i in 0:m]
            append!(pieces, Piece("$f.chunk$(i)of$(m)", f, cuts[i], cuts[i+1] - cuts[i]) for i in 1:m)
        end
    end
    groups = Vector{Piece}[]
    for p in pieces
        if isempty(groups) || (cap > 0 && sum(q -> q.len, last(groups)) + p.len > cap)
            push!(groups, Piece[])
        end
        push!(last(groups), p)
    end
    return isempty(groups) ? [Piece[]] : groups
end

_whole(root, files) = [Piece(f, f, 0, filesize(joinpath(root, f))) for f in files]

# The author's root, or `nothing` where the mount does not resolve.
_author_root(a) = try _mount_root(a["mount"]) catch; nothing end

# [(bound name, part, sub)] of a split entry: part = "meta" or a scene; sub = 0
# for a whole scene, k for sub-part k. Under `max_part_bytes` the sub-parts of a
# scene derive from the author's file sizes when the files are present, else
# from the bindings (a consumer); a scene bound neither way is listed whole
# (and `catalog_violations` flags it unbound).
function _parts_table(cat::DataCatalog, name::AbstractString)
    a = cat.artifacts[name]
    a["split_by"] == "scene" || error("artifact $name: split_by must be \"scene\" (got $(repr(a["split_by"])))")
    a["kind"] in SPLIT_KINDS || error("artifact $name: split_by needs kind pool_sources | pools (got $(a["kind"]))")
    scenes = _scenes(a["splits"])
    META_PART in scenes && error("artifact $name: a scene named $META_PART collides with the meta part")
    cap = _cap(a)
    (cap isa Integer && cap >= 0) || error("artifact $name: max_part_bytes must be a positive integer (got $(repr(cap)))")
    cfg = _entry_config(cat, a)
    cfg === nothing && error("artifact $name: a split entry (kind $(a["kind"])) names its producer run config (config = …)")
    meta = a["kind"] == "pools" || get(cfg["source"], "route", "") == "roma"
    out = meta ? [("$(name)_$(META_PART)", META_PART, 0)] : Tuple{String,String,Int}[]
    if cap == 0
        append!(out, ("$(name)_$s", s, 0) for s in scenes)
        return out
    end
    root = _author_root(a)
    per = root === nothing ? nothing : last(_scene_files(a["kind"], cfg, scenes, root))
    bound = isfile(cat.artifacts_toml) ? keys(TOML.parsefile(cat.artifacts_toml)) : String[]
    for s in scenes
        fs = per === nothing ? String[] : only(f for (t, f) in per if t == s)
        k = if !isempty(fs) && all(f -> isfile(joinpath(root, f)), fs)
            length(_plan(fs, root, cap))                       # the author: the rule
        else
            j = 0                                              # a consumer: the bindings
            while "$(name)_$(s)$(SUBPART)$(j + 1)" in bound; j += 1; end
            j
        end
        k <= 1 ? push!(out, ("$(name)_$s", s, 0)) : append!(out, ("$(name)_$(s)$(SUBPART)$i", s, i) for i in 1:k)
    end
    return out
end

"""
    artifact_parts(cat, name) -> Vector{String}

The Artifacts.toml names catalog entry `name` is bound as: `[name]`, or for an
entry with `split_by = "scene"` one artifact per scene of its `splits`,
`<name>_<scene>`, preceded by `<name>_meta` (the files no scene owns) when the
kind has such files (a pool tree's run records; a `roma` route's manifest).
With `max_part_bytes`, a scene over the cap is `<name>_<scene>_part1..k`
instead (the rule in the catalog header; the author's sizes derive k, a
consumer reads it from the bindings).
"""
function artifact_parts(cat::DataCatalog, name::AbstractString)
    haskey(cat.artifacts, name) || error("unknown data artifact $name (catalog: $(cat.path))")
    _is_split(cat.artifacts[name]) || return [String(name)]
    return first.(_parts_table(cat, name))
end

# The catalog entry a bound artifact belongs to, and which part it is:
# (entry, nothing, 0) for an unsplit entry, (entry, "meta" | scene, sub) for a part.
function _entry_of(cat::DataCatalog, bound::AbstractString)
    haskey(cat.artifacts, bound) && return (String(bound), nothing, 0)
    for (n, a) in cat.artifacts
        (_is_split(a) && startswith(bound, n * "_")) || continue
        for (b, part, sub) in _parts_table(cat, n)
            b == bound && return (n, part, sub)
        end
    end
    error("unknown data artifact $bound (catalog: $(cat.path))")
end

_mount_root(mount) = mount == "data" ? data_root() : mount == "results" ? results_root() :
    error("artifact mount must be data | results (got $mount)")

"""
    artifact_files(cat, name; kinds = Dict()) -> (; mount, root, files, pieces)

The files of artifact `name` — a catalog entry (all its parts), or one part
of a split entry ([`artifact_parts`](@ref)): paths RELATIVE to its mount root `root`
(data_root() for `mount = "data"`, results_root() for `"results"`), derived
from the config of record the entry names (or, for `tree`, its directory).
`kinds` adds project kinds, `kind => (cfg, entry, root) -> files` (`cfg` is
`nothing` when the entry names no config). Every file must exist under
`root` — the author's tree the artifact is packed from. `pieces` maps each of
`files` (its path in the packed tree) to the bytes of the author's file it
holds: the whole file, or one chunk of a sub-part (`max_part_bytes`).
"""
function artifact_files(cat::DataCatalog, name::AbstractString; kinds::AbstractDict = Dict{String,Function}())
    entry, part, sub = _entry_of(cat, name)
    a = cat.artifacts[entry]
    mount = a["mount"]
    root = _mount_root(mount)
    cfg = _entry_config(cat, a)
    kind = a["kind"]
    src = if part !== nothing
        shared, per = _scene_files(kind, cfg, _scenes(a["splits"]), root)
        part == META_PART ? shared : only(fs for (s, fs) in per if s == part)
    else
        haskey(kinds, kind) ? kinds[kind](cfg, a, root) : _kind_files(kind, cfg, a, root)
    end
    src = sort(unique(String.(src)))
    absent = [f for f in src if !isfile(joinpath(root, f))]
    isempty(absent) || error("artifact $name: missing under $root: $(join(absent, ", "))")
    pieces = sub == 0 ? (part === nothing || part == META_PART || _cap(a) == 0 ? _whole(root, src) :
                         only(_plan(src, root, _cap(a)))) :
             _plan(src, root, _cap(a))[sub]
    pieces = sort(pieces; by = p -> p.dst)
    return (; mount, root, files = [p.dst for p in pieces], pieces)
end

# ---- the consumer side ------------------------------------------------------------

# The data view, or `nothing` when a data root is bound some other way (it wins:
# VGTK_DATA_ROOT, or an author's data/exp_raw link or directory, is never written).
function _data_view!()
    haskey(ENV, "VGTK_DATA_ROOT") && return nothing
    view = joinpath(project_root(), "data", "exp_raw")
    (islink(view) || (isdir(view) && !is_artifact_view(view))) && return nothing
    mkpath(view)
    marker = joinpath(view, ARTIFACT_VIEW_MARKER)
    isfile(marker) || write(marker,
        "An ARTIFACT VIEW: per-file links into installed artifact trees plus whatever declared " *
        "producers write. Read-only inputs; remove the directory to rebuild it.\n")
    return view
end

# The results root; the project's data/exp_pro is created when nothing binds one.
function _results_dir!()
    haskey(ENV, "VGTK_RESULTS_ROOT") && return String(ENV["VGTK_RESULTS_ROOT"])
    link = joinpath(project_root(), "data", "exp_pro")
    (islink(link) || isdir(link)) || mkpath(link)
    return results_root()
end

"""
    fetch_artifacts!(cat, names = required_artifacts(cat)) -> Vector{NamedTuple}

Install `names` ([`Experiments.install_artifact`](@ref VisualGeometryToolkitEval.Experiments.install_artifact): download, verify,
unpack into the depot; a placeholder URL is refused with the way out) and link
each tree into its mount root: `data` artifacts into the project's
`data/exp_raw` as an artifact view — skipped when a data root is already bound
(VGTK_DATA_ROOT, or an author's own data/exp_raw) — and `results` artifacts
into results_root() (data/exp_pro is created when nothing binds one).
Idempotent. One `(; name, tree, mount, dst, linked)` per artifact; `dst` is
`nothing` where nothing was linked. A split entry
([`artifact_parts`](@ref)) installs and links every part, one row each (the
parts' trees share directories; the view links per file); a part's own name
fetches that part alone. A chunked file (`max_part_bytes`) is reassembled in
the view once all its chunks are linked: the chunks concatenated in order
into the file (a regular file; the chunk links stay), each chunk already
verified by its artifact's tree hash.
"""
function fetch_artifacts!(cat::DataCatalog, names::AbstractVector = required_artifacts(cat))
    out = NamedTuple[]
    for name in _bound_names(cat, names)
        tree = install_artifact(name, cat.artifacts_toml)
        mount = cat.artifacts[first(_entry_of(cat, name))]["mount"]
        dst = mount == "data" ? _data_view!() : mount == "results" ? _results_dir!() :
              error("artifact $name: mount must be data | results (got $mount)")
        linked = dst === nothing ? 0 : link_tree!(tree, dst)
        dst === nothing || _reassemble!(tree, dst)
        push!(out, (; name, tree, mount, dst, linked))
    end
    return out
end

# Concatenate `<file>.chunk1..mofm` into `<file>` under `dst` for every chunk
# `tree` carries, once all m are present and `<file>` is not. Returns the files written.
function _reassemble!(tree, dst)
    written = String[]
    for (r, _, fs) in walkdir(tree), f in fs
        m = match(CHUNK_RE, f)
        m === nothing && continue
        stem, count = m[1], m[3]
        (stem === nothing || count === nothing) &&
            error("data artifacts: chunk file $(repr(f)) matched $(CHUNK_RE) without its stem and chunk count")
        rel = relpath(joinpath(r, stem), tree)
        target, n = joinpath(dst, rel), parse(Int, count)
        chunks = [joinpath(dst, "$rel.chunk$(i)of$(n)") for i in 1:n]
        (ispath(target) || !all(isfile, chunks)) && continue
        tmp = target * ".reassembling"
        open(tmp, "w") do io
            for c in chunks
                open(cin -> write(io, cin), c)
            end
        end
        mv(tmp, target; force = true)
        push!(written, target)
    end
    return written
end

# Catalog names (or part names) -> the Artifacts.toml names they cover, in order.
_bound_names(cat, names) = unique(reduce(vcat,
    (haskey(cat.artifacts, n) ? artifact_parts(cat, n) : (_entry_of(cat, n); [n]) for n in String.(names));
    init = String[]))

# ---- the author side ----------------------------------------------------------------

"""
    zenodo_url(cat, name; record_ids = Dict()) -> String

The download URL of artifact `name`:
`https://zenodo.org/records/<id>/files/<name>.tar.gz?download=1`, with the id
`record_ids` gives the entry's record group, else the group's `id`, else its
`placeholder` (which [`fetch_artifacts!`](@ref) refuses to download). A part
of a split entry is hosted in its entry's record under its own name.
"""
function zenodo_url(cat::DataCatalog, name::AbstractString; record_ids::AbstractDict = Dict{String,String}())
    g = cat.artifacts[first(_entry_of(cat, name))]["record"]
    r = cat.records[g]
    id = get(record_ids, g, get(r, "id", r["placeholder"]))
    return "https://zenodo.org/records/$id/files/$name.tar.gz?download=1"
end

_listing(root, pieces) = join(("$(p.dst)\t$(p.src)\t$(p.offset)\t$(p.len)\t$(filesize(joinpath(root, p.src)))\t$(mtime(joinpath(root, p.src)))"
                               for p in pieces), '\n') * '\n'

function _write_piece(root, p::Piece, dst)
    mkpath(dirname(dst))
    src = joinpath(root, p.src)
    if p.offset == 0 && p.len == filesize(src)
        cp(src, dst)
    else
        open(src) do io
            seek(io, p.offset)
            left = p.len
            open(dst, "w") do out
                buf = Vector{UInt8}(undef, 1 << 24)
                while left > 0
                    k = readbytes!(io, buf, min(left, length(buf)))
                    k > 0 || error("short read of $src at $(p.offset + p.len - left)")
                    write(out, view(buf, 1:k))
                    left -= k
                end
            end
        end
    end
end

# Pack `name` into <stage>/tarballs, reusing a staged tarball whose `.files`
# listing (relpath, size, mtime of every source file) is current.
function _pack(cat, name, stage, kinds)
    (; root, pieces) = artifact_files(cat, name; kinds)
    mkpath(joinpath(stage, "tarballs"))
    tarball = joinpath(stage, "tarballs", "$name.tar.gz")
    listing = _listing(root, pieces)
    sidecar, treefile = tarball * ".files", tarball * ".tree"
    if isfile(tarball) && isfile(sidecar) && isfile(treefile) && read(sidecar, String) == listing
        hash = SHA1(String(strip(read(treefile, String))))
    else
        rm.((tarball, sidecar, treefile); force = true)
        hash = create_artifact() do dir
            for p in pieces
                _write_piece(root, p, joinpath(dir, p.dst))
            end
        end
        archive_artifact(hash, tarball)
        remove_artifact(hash)          # the tarball is the product; the staged tree is not kept
        write(treefile, bytes2hex(hash.bytes) * '\n')
        write(sidecar, listing)
    end
    sha = bytes2hex(open(sha256, tarball))
    return (; name = String(name), tarball, hash, sha, bytes = filesize(tarball),
            raw = sum(p -> p.len, pieces; init = 0))
end

"""
    pack_artifacts(cat, names = sort(collect(keys(cat.artifacts))); stage,
                   record_ids = Dict(), local_toml = "", kinds = Dict()) -> Vector{NamedTuple}

AUTHOR-SIDE: pack each artifact from the author's data/results roots
([`artifact_files`](@ref)) with Pkg.Artifacts — `create_artifact` in a scratch
artifacts directory under `stage` (never the depot), `archive_artifact` to
`<stage>/tarballs/<name>.tar.gz` — concurrently, then `bind_artifact!` each
into the catalog's Artifacts.toml (`lazy = true`, the tree hash, the tarball's
sha256, [`zenodo_url`](@ref)). A staged tarball whose file listing is current
is reused. `local_toml` also writes a copy of the bindings with `file://` URLs
to the staged tarballs (the fresh-machine test's; never committed).
"""
function pack_artifacts(cat::DataCatalog, names::AbstractVector = sort(collect(keys(cat.artifacts)));
                        stage::AbstractString, record_ids::AbstractDict = Dict{String,String}(),
                        local_toml::AbstractString = "", kinds::AbstractDict = Dict{String,Function}())
    isempty(stage) && throw(ArgumentError("pack_artifacts: `stage` (a scratch directory with room for the tarballs) is required"))
    # one scratch artifacts directory for every concurrent pack (the override is
    # process-global: set once, around all of them)
    built = with_artifacts_directory(joinpath(stage, "artifacts")) do
        fetch.([Threads.@spawn _pack(cat, n, stage, kinds) for n in _bound_names(cat, names)])
    end
    for b in built
        bind_artifact!(cat.artifacts_toml, b.name, b.hash;
                       download_info = [(zenodo_url(cat, b.name; record_ids), b.sha)], lazy = true, force = true)
    end
    if !isempty(local_toml)
        meta = TOML.parsefile(cat.artifacts_toml)
        for (n, m) in meta
            t = joinpath(stage, "tarballs", "$n.tar.gz")
            isfile(t) && (m["download"][1]["url"] = "file://" * abspath(t))
        end
        open(io -> TOML.print(io, meta; sorted = true), local_toml, "w")
    end
    return built
end

"""
    catalog_violations(cat; kinds = String[]) -> Vector{String}

Every inconsistency between the catalog and its Artifacts.toml, empty when
none (a project's test, in one line): the two name the same artifacts; each
entry has the common keys, a known kind (or one of the project's `kinds`), a
known mount, a defined record and an existing config; each record has a
placeholder and, once it exists, a numeric `id`; each binding is lazy,
hash-pinned and downloads from its record on Zenodo — the record's `id` when it
has one, else its placeholder; never a local path.
"""
function catalog_violations(cat::DataCatalog; kinds = String[])
    v = String[]
    isfile(cat.artifacts_toml) || return ["no Artifacts.toml at $(cat.artifacts_toml)"]
    bound = TOML.parsefile(cat.artifacts_toml)
    parts = Dict{String,Vector{String}}()
    for n in keys(cat.artifacts)
        parts[n] = try artifact_parts(cat, n) catch e; push!(v, sprint(showerror, e)); [n] end
    end
    named = reduce(vcat, values(parts); init = String[])
    for n in setdiff(named, keys(bound)); push!(v, "$n: catalogued but not bound"); end
    for n in setdiff(keys(bound), named); push!(v, "$n: bound but not catalogued"); end
    isempty(required_artifacts(cat)) && push!(v, "no artifact is required")
    for (g, r) in cat.records
        haskey(r, "placeholder") || push!(v, "record $g: no placeholder")
        haskey(r, "id") && !occursin(r"^[0-9]+$", string(r["id"])) && push!(v, "record $g: id $(repr(r["id"])) is not a Zenodo record number")
    end
    for (n, a) in sort(collect(cat.artifacts); by = first)
        for k in ENTRY_KEYS
            haskey(a, k) || push!(v, "$n: no `$k`")
        end
        kind = get(a, "kind", "")
        kind in KINDS || kind in kinds || push!(v, "$n: unknown kind $(repr(kind))")
        kind in ("pool_sources", "pools", "stored") && !haskey(a, "config") && push!(v, "$n: kind $kind needs `config`")
        kind in ("pool_sources", "pools") && !haskey(a, "splits") && push!(v, "$n: kind $kind needs `splits`")
        kind == "tree" && !haskey(a, "dir") && push!(v, "$n: kind tree needs `dir`")
        haskey(a, "max_part_bytes") && !_is_split(a) && push!(v, "$n: max_part_bytes needs split_by")
        get(a, "mount", "") in MOUNTS || push!(v, "$n: mount must be data | results")
        haskey(a, "config") && !isfile(joinpath(cat.configs, a["config"] * ".toml")) &&
            push!(v, "$n: no config $(a["config"]).toml")
        g = get(a, "record", "")
        haskey(cat.records, g) || (push!(v, "$n: undefined record $(repr(g))"); continue)
        for p in parts[n]
            haskey(bound, p) || continue
            b = bound[p]
            get(b, "lazy", false) === true || push!(v, "$p: binding not lazy")
            occursin(r"^[0-9a-f]{40}$", get(b, "git-tree-sha1", "")) || push!(v, "$p: no git-tree-sha1")
            dl = get(b, "download", [])
            if length(dl) != 1
                push!(v, "$p: expected one download")
            else
                occursin(r"^[0-9a-f]{64}$", get(dl[1], "sha256", "")) || push!(v, "$p: no sha256")
                want = zenodo_url(cat, p)
                get(dl[1], "url", "") == want || push!(v, "$p: download URL $(repr(get(dl[1], "url", ""))) is not its Zenodo record's ($want)")
            end
        end
    end
    return v
end

end # module DataArtifacts
