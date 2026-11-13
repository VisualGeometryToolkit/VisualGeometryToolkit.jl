# =============================================================================
# vgio/calibration/v3.0 — the manifest of a calibration result's record files
# =============================================================================
#
# {"schema": "vgio/calibration/v3.0", <envelope>,
#  "result_kind": "metric" | "projective" | "image",
#  "model": {CameraModelSpec table},                                  (optional; VGC's closed table)
#  "files": [{"path": "camera.json", "sha256": "<hex>", "schema": "vgio/cameras/v6.0",
#             "role": "authority", "derived_from": []},
#            {"path": "target_poses.json", "sha256": "<hex>", "schema": "vgio/target_poses/v3.0",
#             "role": "derived", "derived_from": ["trajectory_1.json", "targets_1.json"]},
#            {"path": "camera_model.json", "sha256": "<hex>", "schema": "rc/radiometric_camera/v2",
#             "role": "authority", "derived_from": []}, …],
#  "metrics": {"reprojection_rms": 0.21, …},                          (optional; name → number, closed)
#  "provenance": {…}}                                                 (optional; the one free-form object)
#
# One `calibration.json` binds a result's record files by the SHA-256 of their
# bytes and says which are AUTHORITIES (the observation or the estimate) and
# which are DERIVED (computed from other listed files: `target_poses` from the
# trajectory and the targets), so a loader can check the set is consistent —
# that a derived file names files the manifest lists, that an observation
# file's targets resolve in the listed target and rig files — without any one
# record knowing of the others. A file's `schema` is the schema of the record
# it holds, of ANY owner (`<owner>/<kind>/v<major>[.<minor>]`: VGIo's `vgio/…`,
# RadiometricCamera's `rc/…`, CaliBlobs' `caliblobs/…`): every file is
# hash-checked, and a file is validated when its owner has a reader for it
# (VGIo's kinds by their validators; another owner's by its `validate_record`
# method). `result_kind` says what the result is: "metric" (a placement in
# space), "projective" (a partial calibration), "image" (an image-based result
# with no placement in space). `path`s are relative to the manifest's directory.

const _MANIFEST_FILE = "calibration.json"
const _MANIFEST_KINDS = ("metric", "projective", "image")
const _MANIFEST_ROLES = ("authority", "derived")

# A relative path that stays inside the manifest's directory.
_is_inside_path(p) = !isempty(p) && !isabspath(p) && !('\\' in p) && !(".." in splitpath(p))

_manifest_file_check(f) = begin
    f[:role] == "authority" && !isempty(f[:derived_from]) &&
        return "$(f[:path]): an authority is derived from nothing, but lists `derived_from` $(collect(f[:derived_from]))"
    f[:role] == "derived" && isempty(f[:derived_from]) &&
        return "$(f[:path]): a derived file says which listed files it is derived from; `derived_from` is empty"
    f[:path] in f[:derived_from] && return "$(f[:path]): a file is not derived from itself"
    nothing
end

# The cross-entry checks: unique paths, every `derived_from` a listed path, no cycle.
function _manifest_files_check(files)
    isempty(files) && return "a manifest lists at least one file"
    # `./a.json` and `a.json` are one file
    msg = _unique_check([normpath(f[:path]) for f in files], "file paths")
    msg === nothing || return msg
    from = Dict(normpath(f[:path]) => normpath.(String.(f[:derived_from])) for f in files)
    for f in files, d in f[:derived_from]
        haskey(from, normpath(d)) || return "$(f[:path]): `derived_from` names $(repr(d)), which the manifest does not list"
    end
    state = Dict{String, Symbol}()
    visit(p) = begin
        st = get(state, p, :new)
        st === :done && return nothing
        st === :open && return p
        state[p] = :open
        for d in from[p]
            c = visit(d)
            c === nothing || return c
        end
        state[p] = :done
        nothing
    end
    for p in keys(from)
        c = visit(p)
        c === nothing || return "$(c): a file is derived from itself through `derived_from`"
    end
    return nothing
end

# The clear contradictions between a result's `result_kind` and the records it lists:
# an image result has no placement in space, so it lists no trajectory or
# target_poses record.
function _manifest_kind_check(x)
    x[:result_kind] == "image" || return nothing
    for f in x[:files]
        k = _split_schema(f[:schema])
        (k !== nothing && _current_kind(k.kind) in ("trajectory", "target_poses")) &&
            return "`result_kind` is \"image\" (a result with no placement in space), but it lists $(f[:path]), a $(repr(f[:schema])) record"
    end
    return nothing
end

# A model's intrinsics must be the kind's (a spline lens camera is metric); a kind its records do not contradict.
function _manifest_model_check(x)
    msg = _manifest_kind_check(x)
    msg === nothing || return msg
    haskey(x, :model) || return nothing
    m = x[:model]
    i = haskey(m, :type) ? "metric" : get(m, :intrinsics, "metric")
    return (x[:result_kind] in ("metric", "projective") && i != x[:result_kind]) ?
        "`model` calibrates $(repr(i)) intrinsics, but the result's `result_kind` is $(repr(x[:result_kind]))" : nothing
end

const _CALIBRATION_SPEC = _kind_spec(
    "calibration",
    Pair{Symbol, _Spec}[
        :result_kind => _Named(k -> k in _MANIFEST_KINDS, "a calibration result kind ($(join(_MANIFEST_KINDS, ", ")))"),
        :files => _CheckedSpec(
            _Vec(
                _CheckedSpec(
                    _Obj(
                        :path => _Named(_is_inside_path, "a relative path inside the manifest's directory"),
                        :sha256 => _HEX64,
                        :schema => _Named(s -> _split_owned(s) !== nothing, "a record schema (`<owner>/<kind>/v<major>[.<minor>]`)"),
                        :role => _Named(r -> r in _MANIFEST_ROLES, "a role ($(join(_MANIFEST_ROLES, ", ")))"),
                        :derived_from => _Vec(_NonEmpty()),
                    ),
                    _manifest_file_check,
                )
            ),
            _manifest_files_check,
        ),
    ],
    Pair{Symbol, _Spec}[:model => _CheckedSpec(_Free(), _manifest_model_problem), :metrics => _RealMap()];
    check = _manifest_model_check,
)
# The model is VGC's closed `CameraModelSpec` table: VGC's reader is its check.
function _manifest_model_problem(x)
    try
        VgtkCore.CameraModelSpec(_lower(x))
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return nothing
end

"""
    validate_record(::Val{owner}, path, schema) -> Any

The check of a record of another owner's kind (`owner` a `Symbol`, the first
part of its `schema`, `<owner>/<kind>/v<major>.<minor>`): a declared extension
point (`CONTRACT_GENERICS`). An owner that adds a method for its `owner` has a
reader [`load_manifest`](@ref) runs on every listed file of its kinds (after the
hash check every listed file gets); the method throws an `ArgumentError` naming
what is wrong. VGIo's own kinds are checked by [`validate`](@ref).
"""
function validate_record end

# A file entry as the writer takes it: a path (an authority), or `(; path[, role, derived_from])`.
function _manifest_entry(dir, e)
    path, role, from = e isa AbstractString ? (e, "authority", String[]) :
        e isa NamedTuple && haskey(e, :path) && issubset(keys(e), (:path, :role, :derived_from)) ?
        (e.path, String(get(e, :role, "authority")), String[String(p) for p in get(e, :derived_from, String[])]) :
        throw(ArgumentError("save_manifest: a file is a path, or `(; path[, role, derived_from])`; got $(repr(e))"))
    full = joinpath(dir, path)
    isfile(full) || throw(ArgumentError("save_manifest: $(full) is not a file"))
    s = _peek_schema(full)
    _split_owned(s) === nothing && throw(ArgumentError("$(full): not a record of any owner (schema $(repr(s))); a manifest binds records `<owner>/<kind>/v<major>.<minor>`"))
    return (path = String(path), sha256 = open(sha256_hex, full), schema = s, role = role, derived_from = from)
end

"""
    save_manifest(dir, files; result_kind, model = nothing, metrics = nothing, provenance = (;)) -> String

Write `dir/calibration.json`, a `vgio/calibration/v3.0` manifest binding a
calibration result's record files, and return its path. `files` lists the
record files beside it — VGIo's kinds and any other owner's (`rc/…`,
`caliblobs/…`) — each a path relative to `dir` (an authority) or `(; path, role
= "authority", derived_from = String[])`, where `role` is `"authority"` or
`"derived"` and a derived file names, in `derived_from`, the listed files it is
computed from (`target_poses` from the trajectory and the targets). Each entry
records the file's SHA-256 and its record `schema`.

`result_kind` is `"metric"`, `"projective"` or `"image"` (an image-based result
with no placement in space). `model` is an optional VGC `CameraModelSpec`
(written through its `toml_value`; its intrinsics must be the `result_kind`'s);
`metrics` an optional `NamedTuple` or `Dict` of the result's numbers (name →
finite number, a closed object); `provenance` a free-form `NamedTuple` no reader
interprets.

The manifest is checked as [`load_manifest`](@ref) checks it, before anything
is written: an `ArgumentError` for a missing file or one with no record schema,
a derived file without `derived_from` (or an authority with one), a
`derived_from` naming a file that is not listed or forming a cycle, a file its
owner's reader refuses, and an observation file (`observations`,
`target_homographies`) naming a target the listed target and rig files do not hold.
"""
function save_manifest(dir::AbstractString, files::AbstractVector; result_kind::AbstractString, model = nothing, metrics = nothing, provenance::NamedTuple = (;))
    path = joinpath(dir, _MANIFEST_FILE)
    entries = [_manifest_entry(dir, e) for e in files]
    body = (result_kind = String(result_kind), files = entries)
    model === nothing || (body = merge(body, (model = model::VgtkCore.CameraModelSpec,)))   # lowered on the style
    metrics === nothing || (body = merge(body, (metrics = metrics,)))
    # The kind's writer and reader, staged: the very bytes `_write_kind` writes are
    # read back by `_read_store` and checked against the directory before the one
    # rename puts them at `path`; a refusal leaves nothing written.
    write_staged(path => _MANIFEST_FILE; dry_run = false) do staged
        _write_kind(staged, "calibration", body, provenance; style = VgioStyle())
        tree = _read_store(staged, "calibration")
        _manifest_model(tree, path)
        _check_manifest_files(dir, tree, path)
    end
    return path
end

# A CameraModelSpec on the wire: VGC's own table (`toml_value`, the config form),
# and back by VGC's own reader.
JSON.lower(::VgioStyle, spec::VgtkCore.CameraModelSpec) = VgtkCore.toml_value(spec)
JSON.lift(::VgioStyle, ::Type{VgtkCore.CameraModelSpec}, x::JSON.Object) = VgtkCore.CameraModelSpec(_lower(x))   # a parsed table

# The model a manifest states, lifted; its refusal names the manifest.
function _manifest_model(d, path)
    haskey(d, :model) || return nothing
    return try
        JSON.lift(VgioStyle(), VgtkCore.CameraModelSpec, d.model::JSON.Object)
    catch e
        e isa ArgumentError ? throw(ArgumentError("$(path): model: $(e.msg)")) : rethrow()
    end
end

# Whether `full`, with symlinks resolved, lies inside `dir` (also resolved).
function _realpath_inside(full, dir)
    root = realpath(dir)
    p = realpath(full)
    return p == root || startswith(p, joinpath(root, ""))
end

# The files a manifest lists, against the directory: each exists, holds the recorded
# schema, and hashes to the recorded SHA-256 (a mismatch names the file); then the
# targets the measurement files name resolve in the listed target and rig files.
function _check_manifest_files(dir, d, mpath)
    for f in d.files::Vector{Any}
        full = joinpath(dir, f.path::String)
        isfile(full) || throw(ArgumentError("$(mpath): lists $(f.path), which is not a file in $(dir)"))
        _realpath_inside(full, dir) || throw(
            ArgumentError("$(full): resolves outside $(dir) (a symlink leaves the manifest's directory); a manifest binds files inside its directory")
        )
        got = open(sha256_hex, full)
        got == f.sha256 || throw(
            ArgumentError("$(full): its SHA-256 is $(got), but $(mpath) records $(f.sha256) — the file changed since the manifest was written")
        )
        s = _peek_schema(full)
        s == f.schema || throw(ArgumentError("$(full): holds a $(repr(s)) record, but $(mpath) lists it as $(repr(f.schema))"))
        _validate_listed(full, s)
    end
    _check_manifest_targets(dir, d, mpath)
    return nothing
end

# A listed file through its owner's reader: VGIo's kinds by their validators,
# another owner's by its `validate_record` method when it has one (a file of an
# owner with none is bound by its hash alone).
function _validate_listed(full, s)
    v = _split_owned(s)
    v.owner == "vgio" && return (validate(full); nothing)
    hasmethod(validate_record, Tuple{Val{Symbol(v.owner)}, String, String}) && validate_record(Val(Symbol(v.owner)), String(full), String(s))
    return nothing
end

# R18: every target/rig file the manifest lists is loaded, by content id. An
# `observations` / `target_homographies` file (an IO 0.9 one through its version
# steps) naming a targets identity none of them has, or a target name not inside
# the identities it names, is refused naming that file. A rig holds target ids;
# a single target is observed under its design `reference` (a planar body's,
# BlobBoards' board uid) or, with no reference, under its content id.
# The observation kinds whose `targets` a manifest resolves, and the per-frame list naming a target.
const _MANIFEST_OBSERVATIONS = Dict("observations" => :observations, "target_homographies" => :homographies)
function _check_manifest_targets(dir, d, mpath)
    rigs = Dict{String, Set{String}}()          # a rig's content id → its target ids
    singles = Dict{String, String}()            # a single target's content id → the name it is measured under
    for f in d.files::Vector{Any}
        full = joinpath(dir, f.path::String)
        k = _split_schema(f.schema)
        k === nothing && continue
        if _current_kind(k.kind) == "rig"
            rigs[content_id(full)] = Set(String(t.id) for t in get(_read_store(full, "rig"), :targets, Any[]))
        elseif _current_kind(k.kind) == "target"
            id = content_id(full)
            singles[id] = String(get(_read_store(full, "target").target, :reference, id))
        end
    end
    for f in d.files::Vector{Any}
        full = joinpath(dir, f.path::String)
        k = _split_schema(f.schema)
        k === nothing && continue
        kind = _current_kind(k.kind)
        haskey(_MANIFEST_OBSERVATIONS, kind) || continue
        m = _read_store(full, kind)
        for id in get(m, :targets, Any[])::Vector{Any}
            (haskey(rigs, id) || haskey(singles, id)) || throw(
                ArgumentError("$(full): names the targets identity $(id), which none of the target or rig files $(mpath) lists has")
            )
        end
        haskey(m, :targets) || continue
        allowed = Set{String}(n for id in m.targets for n in (haskey(rigs, id) ? rigs[id] : (singles[id],)))
        for fr in m.frames::Vector{Any}, o in getproperty(fr, _MANIFEST_OBSERVATIONS[kind])::Vector{Any}
            haskey(o, :target) || continue          # a landmark observation names no target
            (o.target in allowed) || throw(
                ArgumentError("$(full): frame $(fr.frame_index) names the target $(repr(o.target)), which is not in the targets it references ($(sort!(collect(allowed))))")
            )
        end
    end
    return nothing
end

"""
    load_manifest(dir) -> NamedTuple

Read `dir/calibration.json`, a `vgio/calibration/v3.0` manifest (an earlier one
read through its version steps), and check the set it binds: every listed file
exists, holds the record schema the manifest lists and hashes to the recorded
SHA-256 — an `ArgumentError` naming the file otherwise — whatever its owner; a
file of VGIo's kinds is validated, and one of another owner's when that owner
has a reader ([`validate_record`](@ref)). The targets resolve: every target and
rig file listed is loaded, and an `observations` or `target_homographies` file
naming a targets identity (`content_id`) none of them has, or a target name not
inside the targets it references, is refused naming that file. A rig's targets
are named by their ids; a single target by its design `reference` (a planar
body's) or, when it has none, by its `content_id`. A listed file of an earlier
schema version is read through its version steps.

Returns `(; dir, result_kind, model, files, metrics, provenance)`: `result_kind`
the manifest's `"metric"`, `"projective"` or `"image"`; `model` the VGC
`CameraModelSpec` or `nothing`; `files` a vector of `(; path, sha256, schema,
role, derived_from)`, `path`s relative to `dir`; `metrics` and `provenance` the
parsed objects or `nothing`. Read a file with `load(T, joinpath(dir, path))`.
"""
function load_manifest(dir::AbstractString)
    path = joinpath(dir, _MANIFEST_FILE)
    isfile(path) || throw(ArgumentError("$(dir) holds no $(_MANIFEST_FILE)"))
    d = _read_store(path, "calibration")
    _check_manifest_files(dir, d, path)
    files = [
        (
                path = String(f.path::String), sha256 = String(f.sha256::String), schema = String(f.schema::String),
                role = String(f.role::String), derived_from = String[String(p) for p in f.derived_from::Vector{Any}],
            ) for f in d.files::Vector{Any}
    ]
    return (
        dir = String(dir), result_kind = String(d.result_kind::String), model = _manifest_model(d, path), files = files,
        metrics = get(d, :metrics, nothing), provenance = get(d, :provenance, nothing),
    )
end
