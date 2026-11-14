# =============================================================================
# Migrations — third-party calibrations into the native kinds (convert_opencv_calibration,
# convert_matlab_calibration), earlier schema versions into the current ones (vgio_migrate,
# migrate_schema; a migrated rig's observation files re-pointed) — and the
# bin/vgio_migrate.jl command
# =============================================================================
#
# Every conversion READS the foreign file, WRITES the native kinds to a
# staging directory through the writers' validation, READS them back and
# COMPARES them with what it read — and only then, unless it is a dry run,
# renames them beside the original (or to `out`): `write_staged`. The original is
# never touched.

# The one refusal of every guarded conversion: what it wrote does not read back
# as the file it read.
function _verify(ok::Bool, src, what)
    ok || throw(ArgumentError("$(src): the migration does not read back as the file read ($(what)); nothing was written"))
    return nothing
end

# A pose read back from the quaternion encoding: its translation to the length
# conversion's guarantee — exact for a file-born value, within 2 ulp of the
# metre value for a computed one (a fisheye's frame twist), which a file in a
# unit other than metres may have no number for (lengths.jl) — its rotation to
# the encoding's precision (bit for bit when a quaternion reads back to it
# exactly — `_qvec` — and within a few ulp otherwise).
_same_length(a, b) = _same_metres(Float64(ustrip(m, a)), Float64(ustrip(m, b)))
# Two metre values (or m², a covariance entry's) within the length conversion's
# guarantee: 2 ulp, the sign of a zero kept (`ulps = 0`: bit for bit).
_same_metres(x::Float64, y::Float64, ulps::Integer = 2) = x === y || (signbit(x) == signbit(y) && abs(x - y) <= ulps * eps(abs(x)))
_same_pose(a::EuclideanMap, b::EuclideanMap) =
    all(map(_same_length, a.t, b.t)) && maximum(abs, Matrix(a.R) - Matrix(b.R)) <= 8 * eps()

# =============================================================================
# convert_opencv_calibration — a third-party OpenCV calibration into the native kinds
# =============================================================================

# The tvec unit of a third-party export: its own declaration (`tvec_unit`, or
# this package's `unit`) when it has one — a stated `file_unit` must then agree
# — and otherwise the caller's, which is then required.
function _tvec_unit(d, file_unit, path)
    declared = get(d, :tvec_unit, get(d, :unit, nothing))
    if declared === nothing
        file_unit === nothing && throw(
            ArgumentError("$(path) declares no `tvec_unit`; state `file_unit`, the unit of its tvec (e.g. m or mm)")
        )
        return file_unit
    end
    declared isa AbstractString || throw(ArgumentError("$(path): its tvec unit must be a name, got $(repr(declared))"))
    u = parse_unit(declared)
    (file_unit === nothing || string(file_unit) == string(u)) || throw(
        ArgumentError("$(path) declares its tvec in $(declared); the caller states $(string(file_unit))")
    )
    return u
end

"""
    convert_opencv_calibration([file_frame = PixelFrame{:opencv},] path; images, image_size,
                   file_unit = nothing, target = "calibration_target",
                   out = dirname(path), dry_run = false) -> NamedTuple

Convert a third-party OpenCV calibration export (`calibration_opencv_*.json`:
`fx, fy, cx, cy`, named distortion coefficients, `rms`, one `{rvec, tvec}` per
image) into native files (a `vgio/cameras` store, a `vgio/transforms` file), where everything the export leaves implicit is
declared. The original is never touched. Written to `out`:

- `<stem>_cameras.json` — a `vgio/cameras` store holding one camera named
  `<stem>`: its `image_size` from `image_size`, `K` in the canonical pixel
  frame, and the lens record of `opencv_lens` for the export's distortion
  vector (the terms its length evaluates). The export's reprojection
  statistic is its `provenance` (`reprojection_error`, `error_statistic =
  "rms"`, `source = "opencv"`, `migrated_from`).
- `<stem>_poses.json` — a `vgio/transforms` file, one record per image the
  solver posed, NAMED by that image: `from = "target:<target>"`, `to =
  "view:<image>"` — the target in the camera frame of that view, `x_view = R
  x_target + t`, in the export's tvec unit (declared as its `units.length`). An image whose pose the export has as `null`
  has no record.

`images` is REQUIRED — one unique, non-empty name per entry of the export's
`poses`, in the export's order — and replaces the positional join the export
leaves to its reader. `image_size` (width, height) is required because a camera
is not sufficient without its extent and an OpenCV export does not record it.

`file_unit` is the unit of `tvec`: taken from the file's `tvec_unit` (or this
package's own `unit`) when it declares one — and then a stated `file_unit`
must agree — and otherwise REQUIRED; there is no default.

`file_frame` is the pixel frame of `cx, cy`, and the ONLY place a pixel
frame is stated by hand: by the user's rule OpenCV's own, the top-left pixel's
centre at (0, 0) (`PixelFrame{:opencv}`). Override it only for a calibration
VERIFIED to be in another frame. A type, it leads, as a type does in a
non-mutating call. Everything downstream carries the frame in
its type: the camera read is canonical by construction, and the files
declare what their types say.

The conversion is verified before anything is written to `out`: the files,
written to a staging directory beside `out` (a dry run's: a temporary one) and
renamed into place only once verified, must read back equal to the OpenCV read —
the camera `==`, every pose's translation bit for bit (a file-born value; a
computed one within the length conversion's 2 ulp) and its rotation to the
quaternion encoding's precision (`_qvec`). With
`dry_run` nothing is written to `out`. Returns `(; path, cameras, poses,
n_posed, n_unposed)`, the output paths `nothing` in a dry run.
"""
convert_opencv_calibration(path::AbstractString; kwargs...) = convert_opencv_calibration(PixelFrame{:opencv}, path; kwargs...)

function convert_opencv_calibration(
        file_frame::Type{<:PixelFrame}, path::AbstractString; images, image_size,
        file_unit::Union{Nothing, Units} = nothing,
        target::AbstractString = "calibration_target", out::AbstractString = dirname(abspath(path)),
        dry_run::Bool = false,
    )
    d = _read_json(path)::AbstractDict
    unit = _tvec_unit(d, file_unit, path)
    cal = load_opencv_calibration(file_frame, path; file_unit = unit, target)
    return _migrate_external(cal, "opencv", path; images, image_size, target, out, dry_run, unit)
end

# The native files of a third-party calibration already read into the
# stack's conventions (canonical camera, poses in metres), verified against
# that read before anything is written; shared by the OpenCV and MATLAB
# migrations. The poses file keeps the export's own length unit (`unit`).
function _migrate_external(cal::ExternalCalibration, source::AbstractString, path; images, image_size, target, out, dry_run, unit::Units)
    names = String[String(n) for n in images]
    (length(names) == length(cal.poses) && all(!isempty, names) && allunique(names)) || throw(
        ArgumentError(
            "$(path) has $(length(cal.poses)) poses; `images` must name each, uniquely — got $(length(names)) names" *
                "$(allunique(names) ? "" : ", not unique")$(any(isempty, names) ? ", some empty" : "")"
        )
    )
    (cal.images === nothing || cal.images == names) ||
        throw(ArgumentError("$(path) names its own images, and they are not the `images` given"))
    w, h = Int(image_size[1]), Int(image_size[2])
    stem = splitext(basename(path))[1]
    posed = [(i, names[i], cal.poses[i]) for i in eachindex(cal.poses) if cal.poses[i] !== nothing]
    records = [relabel(TargetFrame{Symbol(target)}, ViewFrame{Symbol(n)}, p) for (_, n, p) in posed]
    prov = (source = String(source), migrated_from = String(abspath(path)))
    cams_out, poses_out = joinpath(out, "$(stem)_cameras.json"), joinpath(out, "$(stem)_poses.json")
    if !dry_run
        for p in (cams_out, poses_out)
            ispath(p) && throw(ArgumentError("$(p) exists; refusing to overwrite it"))
        end
    end
    counts = (n_posed = length(posed), n_unposed = length(names) - length(posed))
    # staged beside `out` and renamed into place once verified (a dry run: in a
    # temporary directory, removed when this returns)
    write_staged(cams_out => basename(cams_out), poses_out => basename(poses_out); dry_run) do cams_tmp, poses_tmp
        save(
            cams_tmp, Dict(Symbol(stem) => BoundedCamera(cal.camera, Size2(w, h)));
            provenance = merge(prov, (reprojection_error = cal.reprojection_error, error_statistic = String(cal.error_statistic))),
        )
        save(poses_tmp, records; length_unit = unit, provenance = prov)
        store = load(Dict{Symbol, BoundedCamera}, cams_tmp)
        back = load(Vector{EuclideanMap}, poses_tmp)
        _verify(
            collect(keys(store)) == [Symbol(stem)] && model(store[Symbol(stem)]) == cal.camera && extent(store[Symbol(stem)]) == Size2(w, h) &&
                length(back) == length(posed) &&
                all(((X, Y),) -> VgtkCore.map_frames(X) == VgtkCore.map_frames(Y) && _same_pose(X, Y), zip(back, records)),
            path, "the $(source) calibration"
        )
    end
    return dry_run ? (; path = String(path), cameras = nothing, poses = nothing, counts...) :
        (; path = String(path), cameras = cams_out, poses = poses_out, counts...)
end

# =============================================================================
# convert_matlab_calibration — a third-party MATLAB calibration into the native kinds
# =============================================================================

"""
    convert_matlab_calibration([file_frame = PixelFrame{:julia},] path; images, image_size, file_unit = nothing,
                   target = "calibration_target", out = dirname(path), dry_run = false) -> NamedTuple

Convert a third-party MATLAB calibration export (`calibration_matlab_*.json`:
the radial/tangential camera `fx, fy, cx, cy, r1…r3, t1, t2`, or the fisheye
`mapping_coeffs`, `distortion_center`, `stretch_matrix`; `mean_error`; one
`{rvec, tvec}` per image) into the same native files as
[`convert_opencv_calibration`](@ref), which it mirrors: `<stem>_cameras.json` (the camera
as [`load_matlab_calibration`](@ref) reads it — MATLAB's polynomial lens, or the
fisheye as `Scaramuzza` with its stretch in `K`; `provenance` records the
`reprojection_error` with `error_statistic = "mean"`, MATLAB's statistic, not
an rms) and `<stem>_poses.json` (`target:<target>` → `view:<image>` per posed
image, in the export's tvec unit — the fisheye's frame twist applied, as the reader
applies it). The original is never touched, and the round trip is verified
before anything is written.

`file_frame` defaults to MATLAB's documented pixel convention, the top-left
pixel's centre at (1, 1) (`PixelFrame{:julia}`); override it only for a
calibration VERIFIED to be in another frame. `file_unit` is REQUIRED unless
the file declares its tvec unit (the toolbox's exports do not; its tvec is in
the unit the pattern was given in — millimetres for this stack's captures).
MATLAB's board frame has x and y exchanged relative to OpenCV's for the same
pattern, and the poses are carried in MATLAB's own board frame, unchanged: give
`target` a name that keeps them apart from an OpenCV calibration's.
"""
convert_matlab_calibration(path::AbstractString; kwargs...) = convert_matlab_calibration(PixelFrame{:julia}, path; kwargs...)

function convert_matlab_calibration(
        file_frame::Type{<:PixelFrame}, path::AbstractString; images, image_size,
        file_unit::Union{Nothing, Units} = nothing,
        target::AbstractString = "calibration_target", out::AbstractString = dirname(abspath(path)),
        dry_run::Bool = false,
    )
    d = _read_json(path)::AbstractDict
    unit = _tvec_unit(d, file_unit, path)
    cal = load_matlab_calibration(file_frame, path; file_unit = unit, target)
    return _migrate_external(cal, "matlab", path; images, image_size, target, out, dry_run, unit)
end

# =============================================================================
# vgio_migrate — a file of an earlier schema version into the current one
# =============================================================================
#
# EVERY EARLIER VERSION A PUBLISHED FILE HOLDS IS READ (2026-11): a schema
# change must not force the published data (Zenodo, Hugging Face; multi-GB
# archives) to be uploaded again, so a published file keeps loading. Each
# version has ONE step, `migrate_schema(::Val{schema}, tree)`: a parsed record
# of that version, in memory, as the record of the next one. The readers run
# the steps on load (`_read_store`, through `migrate_record`); the writers
# write only the current version (or an earlier one a writer still emits, `_written_versions`). `vgio_migrate` rewrites a file with the same
# steps, by the guarded pattern of the third-party conversions above: the file
# read (through the steps), its value WRITTEN by the current writer (its
# validation) to a staging directory beside `out` (`write_staged`), READ back
# and COMPARED, number for number, with the value — only then renamed to
# `out`. Nothing is written when any check fails, and the original is
# untouched unless `out` is the original's own path (an in-place migration,
# replaced only after the verification).
#
# `migrate_schema` is a declared extension point (`CONTRACT_GENERICS`): the
# owner of a schema adds the step of each of its old versions, and the file
# form `migrate_schema(::Val{schema}, src, dst)` that `vgio_migrate` runs —
# VisualGeometryToolkitIO for cameras, targets and rigs, BlobBoards for its
# boards — and its reader calls `migrate_record`.

"""
    migrate_schema(::Val{schema}, tree::AbstractDict) -> AbstractDict
    migrate_schema(::Val{schema}, src, dst) -> NamedTuple

The migration of a record of the (old) `schema` — a `Symbol`,
`Val(Symbol("vgio/rig/v5"))` — in two forms.

The STEP, `migrate_schema(::Val{schema}, tree)`: the parsed record `tree` (a
`Dict{String, Any}` of that schema) as the record of the next version, its
`"schema"` that version's. One step per version, and the one implementation
of it: [`migrate_record`](@ref) runs the steps until the record is current,
the readers run that on load, and the file form runs it too. A step refuses
(`ArgumentError`) a record it cannot carry over exactly.

The FILE form, `migrate_schema(::Val{schema}, src, dst)`: the file at `src`
in its kind's current schema, written to `dst` (a path in the staging
directory [`vgio_migrate`](@ref) owns). A method reads `src` (through the
steps), writes the current file to `dst` through its writer, reads `dst` back
with the current reader and compares every number with `src`'s value (a length
as its value, within the length conversion's 2-ulp guarantee), throwing an
`ArgumentError` on any difference; it returns a summary `NamedTuple` with at
least `to`, the schema written (and whatever the owner reports: a rig's or a
target's `old_ids` and `new_id`).

A declared extension point (VisualGeometryToolkitIO's `CONTRACT_GENERICS`): the
owner of a schema adds both forms for each of its old versions.
VisualGeometryToolkitIO's own are listed by [`schema_versions`](@ref): every
kind's earlier versions up to its IO 0.9 one, each with one step, and IO 0.9's
version of each kind with its step to the 0.10 major (migrate_v010.jl); a
renamed or folded kind's into its successor (`generic_camera` →
`noncentral_spline` → `cameras`; `target_points2d`, `target_points3d`,
`target_detections` → `observations`). An old grid file is read, never
rewritten: its file form writes its camera store to a new file.
"""
function migrate_schema end

"""
    schema_versions() -> Vector{NamedTuple}
    schema_versions(kind) -> Vector{String}

The schema versions VisualGeometryToolkitIO reads, DERIVED from its version
steps (the `migrate_schema` methods of a record: every schema with a step, and
each kind's current one), never kept by hand: per current kind, its earlier
schemas (a former kind's among them: `generic_camera` and `noncentral_spline`
for `cameras`; `target_points2d`, `target_points3d`, `target_detections` for
`observations`) in version order, then its current one. The docs table of the
versions read is generated from it.
"""
function schema_versions(kind::AbstractString)
    old = String[]
    for m in methods(migrate_schema)
        sig = m.sig                       # every step method is concrete: `Tuple{typeof(migrate_schema), Val{s}, <tree or paths>}`
        (sig isa DataType && length(sig.parameters) == 3) || continue
        V = sig.parameters[2]
        (V isa DataType && V <: Val && V.parameters[1] isa Symbol) || continue
        s = String(V.parameters[1])
        v = _split_schema(s)
        (v !== nothing && _current_kind(v.kind) == kind && _has_record_migration(s)) && push!(old, s)
    end
    # the old grid kind and the non-central kind are read through the store reader (`_one_camera_store`)
    kind == "cameras" && append!(old, _OLD_GRID_SCHEMAS, ("vgio/noncentral_spline/v1",))
    order(s) = (v = _split_schema(s); (v.kind == kind, v.kind, v.major, v.minor))
    return push!(sort!(unique!(old); by = order), _schema(kind))
end
schema_versions() = [(kind = k, current = _schema(k), reads = schema_versions(k)) for k in sort!(collect(keys(_SCHEMA_VERSION)))]
# Whether a schema has a file migration (`vgio_migrate`), a step (the readers).
_has_migration(s::AbstractString) = hasmethod(migrate_schema, Tuple{Val{Symbol(s)}, String, String})
_has_record_migration(s::AbstractString) = hasmethod(migrate_schema, Tuple{Val{Symbol(s)}, Dict{String, Any}})

"""
    migrate_record(tree) -> Dict{String, Any}

A parsed record (`tree`, an `AbstractDict` as JSON parsed it) of an earlier
schema version as its kind's current record, in memory: each version's step,
[`migrate_schema`](@ref)`(::Val{schema}, tree)`, in turn, until no step is
left for the record's schema. A current record comes back as it is (its keys
`String`s). VisualGeometryToolkitIO's readers call it on every file of an
earlier version; the reader of a schema owner's kind calls it before it reads
a file, so an old file loads as the current version. An `ArgumentError` for a
step that does not move the version, and whatever a step refuses.

A `vgio/cameras/v1` record, carried to the current version by the v1 … v5
steps:

```jldoctest
julia> v1 = Dict("schema" => "vgio/cameras/v1", "cameras" => [Dict(
           "schema" => "vgio/camera/v1", "name" => "left", "pixel_chart" => "colmap",
           "size" => Dict("unit" => "px", "width" => 1280, "height" => 960),
           "intrinsics" => Dict("unit" => "px", "fx" => 900.0, "fy" => 900.0,
                                "cx" => 640.0, "cy" => 480.0, "skew" => 0),
           "lens" => Dict("model" => "KannalaBrandt", "params" => [0.1, -0.02, 0.003, -0.0004]))]);

julia> VgtkIO.migrate_record(v1)["schema"]
"vgio/cameras/v6.0"
```
"""
function migrate_record(tree::AbstractDict)
    t = _lower(tree)
    while (s = get(t, "schema", nothing)) isa AbstractString && _has_record_migration(s)
        t = _lower(migrate_schema(Val(Symbol(s)), t))
        _moves_forward(s, get(t, "schema", nothing)) ||
            throw(ArgumentError("the migration step of $(s) returned a $(repr(get(t, "schema", nothing))) record, not a later version of its kind"))
    end
    return t
end
# Whether a step's schema `to` is a later version of the kind `from` names
# (`<owner>/<kind>/v<major>[.<minor>]`, an owner's kinds included): what keeps
# the steps from looping. A step into a renamed kind (`_RENAMED_KIND`) moves
# forward from any version of its former name.
function _moves_forward(from::AbstractString, to)
    a, b = _split_owned(from), _split_owned(to)
    (a === nothing || b === nothing || a.owner != b.owner) && return false
    a.kind == b.kind && return _version(b) > _version(a)
    return a.owner == "vgio" && get(_RENAMED_KIND, a.kind, nothing) == b.kind
end

"""
    vgio_migrate(path; out = <stem>.migrated.json beside path (an old grid file: <stem>.camera.json or <stem>.noncentral.json), dry_run = false) -> NamedTuple

An old grid file (`vgio/generic_camera/v2` or `v3`) is never rewritten: its
camera store — the `vgio/cameras` record of its one camera, named after the
file — is written to a NEW file, by default `<stem>.camera.json` beside a
central one, `<stem>.noncentral.json` beside a non-central one (an existing one
refused, `out == path` refused).

Rewrite the file at `path`, of the earlier schema version it declares, in its
kind's current one, by the file form of [`migrate_schema`](@ref) for that
schema (every earlier version [`schema_versions`](@ref) lists that a published
file holds, and any a schema owner adds),
which runs the same version steps a reader runs on load: an old file need not
be rewritten to be read, and this is how it is rewritten once. The migration
is written through the current writer to a staging directory BESIDE `out`
(the same filesystem), read back with the current reader and compared with the
original, number for number; only then is it renamed to `out`, one atomic
`rename`. Every number comes back bit for bit except a length (or a
covariance entry of lengths) the file holds at full precision, 16 or 17
digits: no metre value writes back to every such number, so the migrated file
may hold one that differs by at most 2 ulp — reading back as the same metre
value, within the length conversion's 2-ulp guarantee ([`read_length`](@ref)).
A number of at most 15 significant digits is rewritten as its own text. The original is untouched unless `out == path` (an in-place
migration, replaced by that rename only once verified); an existing `out` is
otherwise refused. With `dry_run` the migration is verified and nothing is
written. Returns `(; path, out, from, to, …)`, `out` `nothing` in a dry run,
with whatever the schema's migration reports — a rig's or a target's
`old_ids` (the identities observation files may name it by: the ones its own
version minted, and IO 0.9's identity of its value) and its `new_id`, its
identity from IO 0.10 on.

A `.toml` record whose schema has a step but no file form (a schema owner's
TOML config) is rewritten by the generic file form: parse, [`migrate_record`](@ref),
`TOML.print` to the staging directory, re-parse and require the very tree back
(types included: an `Int` stays an `Int`, a `1.0` a `Float64`, a date a `Date`),
an `ArgumentError` otherwise. The rewrite drops the file's comments: the
result's `comments_dropped` is `true`, and a `@warn` says so, when the file has
any `#` (a comment line or a trailing comment; a `#` inside a string also
counts, conservatively). The result is `(; path, out, from, to, comments_dropped)`.
"""
vgio_migrate(path::AbstractString; out::AbstractString = _default_out(path), dry_run::Bool = false) =
    _vgio_migrate(path; out, dry_run)

# `vgio_migrate`, with `before_rename(r)` called on the verified result before
# anything is put at `out` (a tree persists its identity map there).
function _vgio_migrate(path::AbstractString; out::AbstractString, dry_run::Bool, before_rename = r -> nothing)
    tree = _is_toml(path) ? _parse_toml_file(path) : nothing   # a TOML file is parsed once, here
    s = tree === nothing ? _peek_schema(path) : get(tree, "schema", nothing)
    s isa AbstractString || throw(ArgumentError("$(path): not a vgio file (no schema)"))
    in_place = abspath(out) == abspath(path)
    (in_place && s in _OLD_GRID_SCHEMAS) && throw(ArgumentError("$(path): $(_OLD_GRID_REFUSAL)"))
    (dry_run || in_place || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    if !_has_migration(s) && tree !== nothing && _has_record_migration(s)
        return _migrate_toml(path, tree, out, dry_run, s, before_rename)
    end
    _has_migration(s) ||
        throw(ArgumentError("$(path): no migration for a $(s) file (no method of VisualGeometryToolkitIO.migrate_schema for it)"))
    return write_staged(out => basename(path); dry_run) do staged
        r = merge((; path = String(path), out = dry_run ? nothing : String(out), from = String(s)), migrate_schema(Val(Symbol(s)), String(path), staged))
        before_rename(r)
        r
    end
end

_is_toml(path) = endswith(lowercase(path), ".toml")
# An old grid file (vgio/generic_camera v2/v3) is read, never rewritten in place.
const _OLD_GRID_REFUSAL = "an old grid file is never rewritten in place: vgio_migrate(path) writes its current record to a new " *
    "<stem>.camera.json (a central one) or <stem>.noncentral.json (a non-central one); load(BoundedCamera, path) reads it as it is"

# Where `vgio_migrate` writes a file by default: an old grid file's current record to a
# NEW file beside it (the input is never rewritten) — a central one's camera store,
# `<stem>.camera.json`, a non-central one's record, `<stem>.noncentral.json`; any other
# file beside itself as `<stem>.migrated<ext>`.
function _default_out(path)
    _try_schema(path) in _OLD_GRID_SCHEMAS || return _migrated_path(path)   # the schema peeked; a grid file parsed only then
    return string(first(splitext(path)), _is_central_grid(_read_vgio_json(path)) ? ".camera.json" : ".noncentral.json")
end

# The generic file form for a TOML record whose schema has a step and no file
# form: from the parsed `tree`, run the steps, print, read the print back and
# require the very tree (types included), so a value TOML.print cannot reproduce
# is a refusal.
function _migrate_toml(path, tree, out, dry_run, s, before_rename)
    # any `#`, trailing comments included (a `#` inside a string is a harmless false positive)
    comments_dropped = occursin('#', read(path, String))
    comments_dropped &&
        @warn "vgio_migrate rewrites $(path) through TOML.print, which drops its comments; the original is kept unless out == path"
    migrated = migrate_record(tree)
    return write_staged(out => basename(out); dry_run) do staged
        try
            open(io -> TOML.print(io, migrated), staged, "w")
        catch err
            err isa ErrorException || rethrow()   # TOML.print refuses a value type it has no form for
            throw(ArgumentError("$(path): the migrated record cannot be written as TOML ($(err.msg)); nothing written"))
        end
        back = TOML.parsefile(staged)
        _same_tree(back, migrated) ||
            throw(ArgumentError("$(path): the migrated record does not read back from TOML.print as itself (a value TOML.print cannot reproduce exactly); nothing written"))
        r = (; path = String(path), out = dry_run ? nothing : String(out), from = String(s),
            to = String(migrated["schema"]), comments_dropped)
        before_rename(r)
        r
    end
end

# Equal AND of the same type throughout (`1 == 1.0` is not enough), element by
# element: not `same_value`, whose container check is `typeof` too, and a TOML
# round trip turns an array's `Vector{Any}` into a `Vector{String}`.
_same_tree(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(_same_tree(a[k], b[k]) for k in keys(a))
_same_tree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(_same_tree(x, y) for (x, y) in zip(a, b))
_same_tree(a, b) = typeof(a) === typeof(b) && isequal(a, b)

const _STAGE_PREFIX = ".vgio_migrate_"

"""
    write_staged(f, out => name, …; dry_run) -> f's result

Write files through a staging directory, the guarded pattern of every
migration: `f(staged...)` is called with one path per `out => name` (each
named `name`, in a fresh directory BESIDE the destinations, so on their
filesystem) and builds and verifies the files there; then, unless a
`dry_run`, each is put at its `out` by one `rename`, in the order given —
atomic on POSIX, so an original at `out` is untouched until its rename, and
nothing is moved across filesystems. The destinations share one directory,
created if missing (an `ArgumentError` otherwise, before anything is written). A dry run stages in a temporary directory and puts
nothing anywhere. The staging directory goes when this returns, whatever
happened. On a failure — `f` throws, or a rename does — the destinations are
left as they were: a destination that did not exist and was already renamed
into place is removed again (one that existed was replaced by its rename and
stays so), and a directory this call created for them is removed.
"""
function write_staged(f, outs::Pair{<:AbstractString, <:AbstractString}...; dry_run::Bool)
    dir = dirname(abspath(first(first(outs))))
    for (out, _) in outs
        dirname(abspath(out)) == dir || throw(
            ArgumentError("write_staged: the destinations share one directory (the staging is beside them, each put by one rename); $(out) is not in $(dir)")
        )
    end
    created = dry_run ? nothing : _first_missing_dir(dir)   # the outermost directory mkpath creates, if any
    dry_run || mkpath(dir)
    fresh = [!ispath(abspath(out)) for (out, _) in outs]
    renamed = 0
    stage = dry_run ? mktempdir() : mktempdir(dir; prefix = _STAGE_PREFIX)
    try
        staged = [joinpath(stage, name) for (_, name) in outs]
        r = f(staged...)
        if !dry_run
            for (p, (out, _)) in zip(staged, outs)
                Base.Filesystem.rename(p, abspath(out))
                renamed += 1
            end
        end
        return r
    catch
        for i in 1:renamed
            fresh[i] && rm(abspath(first(outs[i])); force = true)
        end
        created === nothing || rm(created; recursive = true, force = true)
        rethrow()
    finally
        rm(stage; recursive = true, force = true)
    end
end
# The outermost ancestor of `dir` (itself included) that does not exist, or `nothing`.
function _first_missing_dir(dir::AbstractString)
    ispath(dir) && return nothing
    parent = dirname(dir)
    return (parent == dir || ispath(parent)) ? dir : _first_missing_dir(parent)
end
_migrated_path(path) = ((stem, ext) = splitext(path); string(stem, ".migrated", ext))

# The file's own tree as a Dict{String, Any}, every number as parsed.
_raw_tree(path) = _lower(_read_vgio_json(path))

# A file's free-form `provenance`, carried over unchanged: the writer's
# keyword takes a NamedTuple.
_namedtuple(x::AbstractDict) = NamedTuple(Symbol(k) => _namedtuple(v) for (k, v) in x)
_namedtuple(x::AbstractVector) = Any[_namedtuple(v) for v in x]
_namedtuple(x) = x
_provenance_of(d::AbstractDict) = haskey(d, "provenance") ? _namedtuple(d["provenance"]) : (;)
# The writer's keywords that re-save an old file's value as that file: the
# length unit its current record declares (`units.length`; `length = false`
# for a writer that takes none) and its provenance.
_resave_kwargs(current::AbstractDict; length::Bool = true) = merge(
    length ? (; length_unit = parse_unit(current["units"]["length"]::String)) : (;), (; provenance = _provenance_of(current)),
)

"""
    same_value(a, b; length_ulps = 2) -> Bool

Whether two TYPED values are the same record: the same type at every level,
compared field by field, an array element by element, a dict key by key, an
`Uncertain` value (VisualGeometryToolkitCore's) with its covariance and a correspondence set through
its endpoints and attribute columns. A LENGTH — a quantity of dimension length
or length², a covariance entry whose two parameters' length powers add up to 1
or 2, a non-central model's points — is compared as its metre value within
`length_ulps` ulp (the sign of a zero kept); every other number (a rotation, a
focal length, a pixel, a time, a dimensionless quantity) and every string,
symbol or type must be identical (`===`, `isequal`).

The default, 2 ulp, is the decimal-exact length conversion's guarantee
([`read_length`](@ref)/[`write_length`](@ref)): a full-precision file number
has no metre value that writes back to its own text, so a migration that reads
an old file and the file it wrote compares within it. `length_ulps = 0` is
exact everywhere, for a schema whose lengths read back bit for bit (a product
codec's).
"""
function same_value end

# The migration's comparison is of TYPED records: the old file read by the
# migration's reader and the new file read by the current reader are compared
# value for value. A length is compared as its metre value within the length
# conversion's guarantee (`_same_metres`); every other number is written from
# the identical Float64 by every writer and must come back bit for bit.
same_value(a::Quantity, b::Quantity; length_ulps::Integer = 2) =
    typeof(a) === typeof(b) && dimension(a) == dimension(b) &&
    (dimension(a) in (dimension(m), dimension(m * m)) ? _same_metres(Float64(ustrip(a)), Float64(ustrip(b)), length_ulps) : ustrip(a) === ustrip(b))
same_value(a::AbstractFloat, b::AbstractFloat; length_ulps::Integer = 2) = a === b   # (NaN === NaN: an absent node's slot)
same_value(a::Union{Number, AbstractString, Symbol, Type, Nothing, Missing}, b; length_ulps::Integer = 2) = typeof(a) === typeof(b) && isequal(a, b)
same_value(a::AbstractDict, b::AbstractDict; length_ulps::Integer = 2) =
    typeof(a) === typeof(b) && Set(keys(a)) == Set(keys(b)) && all(k -> same_value(a[k], b[k]; length_ulps), keys(a))
same_value(a::AbstractArray, b; length_ulps::Integer = 2) =
    typeof(a) === typeof(b) && size(a) == size(b) && all(i -> same_value(a[i], b[i]; length_ulps), eachindex(a))
same_value(a::Union{Tuple, NamedTuple}, b; length_ulps::Integer = 2) =
    typeof(a) === typeof(b) && all(i -> same_value(a[i], b[i]; length_ulps), 1:length(a))
function same_value(a::Uncertain, b::Uncertain; length_ulps::Integer = 2)
    typeof(a) === typeof(b) && same_value(geom(a), geom(b); length_ulps) || return false
    Σa, Σb = cov(a), cov(b)
    p = _cov_length_powers(geom(a))
    return size(Σa) == size(Σb) && all(
        CartesianIndices(Σa)) do I
            n = p === nothing ? 0 : p[I[1]] + p[I[2]]
            n == 0 ? Σa[I] === Σb[I] : _same_metres(Float64(Σa[I]), Float64(Σb[I]), length_ulps)
        end
end
# A correspondence set: its endpoints and its attribute columns, through its
# accessors (the AbstractArray method above would compare its pairs only).
same_value(a::Correspondences, b; length_ulps::Integer = 2) = typeof(a) === typeof(b) &&
    same_value(sources(a), sources(b); length_ulps) && same_value(targets(a), targets(b); length_ulps) && same_value(a.attrs, b.attrs; length_ulps)
# A non-central model's point controls are plain metres (its directions are not lengths).
same_value(a::NoncentralSplineModel, b::NoncentralSplineModel; length_ulps::Integer = 2) =
    typeof(a) === typeof(b) && all(f -> f === :points ?
        a.points.area === b.points.area && size(a.points.controls) == size(b.points.controls) &&
        all(map((p, q) -> all(map((x, y) -> isnan(x) && isnan(y) || _same_metres(x, y, length_ulps), Float64.(p), Float64.(q))),
            a.points.controls, b.points.controls)) :
        same_value(getfield(a, f), getfield(b, f); length_ulps), fieldnames(typeof(a)))
# Any other value: field by field.
function same_value(a, b; length_ulps::Integer = 2)
    typeof(a) === typeof(b) || return false
    isempty(fieldnames(typeof(a))) && return isequal(a, b)
    return all(f -> same_value(getfield(a, f), getfield(b, f); length_ulps), fieldnames(typeof(a)))
end
# The length power of each parameter a covariance is over: a pose's (rotation
# then translation), a point's (every coordinate a length), a target
# calibration's (`length_mask`); `nothing` (every entry compared bit for bit)
# for anything else.
_cov_length_powers(::EuclideanMap) = _POSE_COV_DIMS
_cov_length_powers(x::StaticVector) = ntuple(_ -> 1, length(x))
_cov_length_powers(x) = applicable(length_mask, x) ? Tuple(Int.(length_mask(x))) : nothing

# What the file holds beside its value: the length unit it declares and its
# provenance (which no value carries), compared as they are. (Its pixel frame is
# its value's, which the value comparison covers.)
_same_file_extras(new::AbstractDict, old::AbstractDict) =
    isequal(new["units"]["length"], old["units"]["length"]) && isequal(get(new, "provenance", nothing), get(old, "provenance", nothing))

# The old record's keys renamed to the new, and back. A target body's `warp`
# (v1: rigid only, `{"type": "rigid"}` — already the v2 calibration record) is
# its `calibration`, its `design` its `reference`; the target file's
# `calibration` block its `calibration_record`.
const _TARGET_BODY_RENAMES = ("warp" => "calibration", "design" => "reference")
_renamed(d::AbstractDict, renames) = Dict{String, Any}(get(Dict(renames), k, k) => v for (k, v) in d)
# A version step's refusal of a record holding what only a later version holds
# (a current record relabeled old): that version did not write it.
_not_written_by(record, why) = throw(ArgumentError("$(record) $(why), so this record was not written by them"))
function _body_v1_to_v2(b)
    # a key of the later records was never a v1/v4 body's (its renames would collide)
    for (old, new) in _TARGET_BODY_RENAMES
        haskey(b, new) && _not_written_by("a vgio/target/v1 or vgio/rig/v4 target", "holds no $(repr(new)) (v1 named it $(repr(old)))")
    end
    return _renamed(b, _TARGET_BODY_RENAMES)
end
_body_v2_to_v1(b) = _renamed(b, map(reverse, _TARGET_BODY_RENAMES))

# A rig tree with each target body through `f` (a shallow copy).
function _map_rig_targets(f, tree::AbstractDict)
    t = copy(tree)
    haskey(t, "targets") && (t["targets"] = Any[merge(Dict{String, Any}(x), Dict{String, Any}("target" => f(x["target"]))) for x in tree["targets"]])
    return t
end
# the v5 record of a v4 one: its target bodies' keys renamed (v4 listed a
# lens's stages in application order, as v5 did)
_rig_v4_to_v5(old) = (t = _map_rig_targets(_body_v1_to_v2, old); t["schema"] = "vgio/rig/v5"; t)
# a current record back in v4's form: v6 → v5 (the compositions back in
# application order, under v5's keys) then v5's keys renamed back
_rig_v6_to_v4(new, old) = (t = _map_rig_targets(_body_v2_to_v1, _rig_v6_to_v5(new)); t["schema"] = old["schema"]; t)
# A v1 target's / v4 rig target's `warp` was RIGID ONLY: v4's writer refused
# any other calibration and its reader knew no other type. A body claiming one
# (an "affine", a spline, a chain) was never a v1/v4 record — v4's identities
# (`_rig_v4_id_mm`: the record re-emitted in millimetres, rigid warps carrying
# no lengths) are defined only for rigid warps — so it is refused by name,
# never migrated as if v4 had written it.
function _check_v1_warp(body::AbstractDict)
    w = get(body, "warp", nothing)
    (w === nothing || w == Dict{String, Any}("type" => "rigid")) && return nothing
    throw(
        ArgumentError(
            "a vgio/target/v1 or vgio/rig/v4 record whose \"warp\" is $(repr(get(w, "type", w))): " *
                "those records held a rigid warp only ({\"type\": \"rigid\"}), so this file was not written by them"
        )
    )
end
# the v2 target record of a v1 one (v1 held rigid targets only: no composition)
function _target_v1_to_v2(old)
    haskey(old, "calibration_record") && _not_written_by("a vgio/target/v1 record", "holds no \"calibration_record\" (v1 named it \"calibration\")")
    _check_v1_warp(old["target"])
    body = _renamed(old, ("calibration" => "calibration_record",))
    body["schema"] = "vgio/target/v2"
    body["target"] = _body_v1_to_v2(old["target"])
    return body
end

# The guarded file migration of a VGIo record: the file at `src` read as a `T`
# (through the version steps, validated as the current record), written by the
# current writer to `dst` in the unit and with the provenance of the record the
# steps give, read back as a `T` and compared with the value (`same_value`: a
# length within the conversion's guarantee, every other number bit for bit),
# its unit and provenance with that record's. Returns the value.
function _migrate_file(::Type{T}, src::AbstractString, dst::AbstractString, what; current = migrate_record(_raw_tree(src)), length::Bool = true) where {T}
    value = load(T, src)
    save(dst, value; _resave_kwargs(current; length)...)
    _verify(same_value(value, load(T, dst)) && _same_file_extras(_raw_tree(dst), current), src, what)
    return value
end

# vgio/target/v2 → v3 and vgio/rig/v5 → v6 (CoordinateTransformations'
# composition and the gauge-free print error, 2026-11):
# - a composed calibration's members are listed in CT's order, the outer map
#   first, under `compose` (`{"type": "composition", …}`), and its covariance
#   is over that order — v2/v5's `{"type": "calibration_chain", "members": […]}`
#   listed them in application order;
# - the AFFINE kind is retired: `{"type": "affine", "parameters": [a11, a12,
#   a21, a22, bx, by]}` (p ↦ A·p + b, A row by row) is its Iwasawa factors
#   `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew` (A = s·R·D·U), exactly the same map
#   (VGC's `iwasawa_parameters`), its covariance carried through the factorization's
#   Jacobian; a chain's affine member becomes those four in its place. All
#   four are KEPT (a migration preserves the geometry exactly): EuclideanMap
#   and Scale are the jointly estimated pose's gauges, so VGC's joint-pose
#   manifold refuses such a calibration unless a known pose is stated
#   (`minimal_manifold(target; known = (:pose,))`) — drop or absorb them into
#   the pose before a joint refinement;
# - a thin-plate kind holds pure bending: a v2/v5 spline whose interpolant
#   carries an affine term is REFUSED (by the current reader, naming why), not
#   moved — no exact form exists: an affine displacement inside a
#   thin_plate_spline cannot move into the factors, which compose AFTER the
#   spline (`T ∘ A` needs the bending re-expressed at the moved points, which
#   a spline on the same nodes cannot hold), and a plane of heights is a shear
#   of the target, not a rigid motion its pose could take. Dropping either
#   would change the calibrated geometry silently;
# - the result must be a composition in VGC's canonical order
#   (`HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`):
#   any other v2/v5 chain (two in-plane kinds out of that order, a Rigid
#   member) has no v3 form and is refused by the current reader, naming the
#   rule.
# The v2 counts are frozen on purpose, not derived from `length_mask`, so the v2
# migrator does not follow a later change of a kind's parameter layout.
_v2_parameter_count(r) = (t = r["type"]; t == "rigid" ? 0 : t == "affine" ? 6 :
    t == "thin_plate_spline" ? 2 * length(r["nodes"]) : t == "height_field" ? length(r["nodes"]) :
    throw(ArgumentError("a calibration record of unknown type $(repr(t))")))

# A v2 member as its v3 members (CT's order) and the Jacobian of their
# parameters in its own: an affine as [euclidean, scale, aspect, skew] over
# (ϑ, bx, by, log s, a₁, u₁₂) ← (a11, a12, a21, a22, bx, by) — A's Iwasawa
# factors A = s·R·D·U (VGC's `iwasawa_parameters_with_jacobian`, its Jacobian
# in A's entries row by row, as v2 stored them), its b carried over unchanged
# in the file's unit; any other kind as it is, its Jacobian the identity.
function _member_v2_to_v3(r::AbstractDict)
    n = _v2_parameter_count(r)
    r["type"] == "affine" || return (records = Any[r], J = [Float64(i == j) for i in 1:n, j in 1:n], n)
    p = r["parameters"]
    (ϑ, k, a, u), Ja = iwasawa_parameters_with_jacobian(SMatrix{2, 2}(Float64(p[1]), Float64(p[3]), Float64(p[2]), Float64(p[4])))
    rec(ty, θ) = Dict{String, Any}("type" => ty, "parameters" => Any[θ...])
    records = Any[rec("euclidean", (ϑ, p[5], p[6])), rec("scale", (k,)), rec("aspect", (a,)), rec("skew", (u,))]
    J = zeros(6, 6)
    J[1, 1:4] = Ja[1, :]
    J[2, 5] = J[3, 6] = 1.0
    J[4:6, 1:4] = Ja[2:4, :]
    return (; records, J, n)
end
# A v2 calibration record (a kind, a chain, an affine) as its v3 record.
function _calibration_v2_to_v3(c::AbstractDict)
    ty = get(c, "type", nothing)
    ty == _COMPOSITION_TYPE && _not_written_by("a target v2 / rig v5 calibration", "is never a \"$(_COMPOSITION_TYPE)\" (v2/v5 held a \"$(_RETIRED_COMPOSITION_TYPE)\")")
    ty in (_RETIRED_COMPOSITION_TYPE, "affine") || return c
    parts = [_member_v2_to_v3(r) for r in (ty == "affine" ? Any[c] : c["members"])]
    out = Dict{String, Any}(k => v for (k, v) in c if !(k in ("type", "members", "parameters", "covariance")))
    out["type"] = _COMPOSITION_TYPE
    out[_COMPOSE_KEY] = reduce(vcat, [p.records for p in reverse(parts)])
    haskey(c, "covariance") && (out["covariance"] = _covariance_v2_to_v3(c["covariance"], parts))
    return out
end
# The covariance over the v2 members (application order) as the one over the
# v3 members (CT's order): Σ′ = J Σ Jᵀ with J the members' block Jacobians
# (an identity block for a kind carried over as it is), their row blocks
# reversed. With no affine member J is a pure reorder: the entries are
# permuted, exactly (an Inf stays Inf, a -0.0 keeps its sign).
function _covariance_v2_to_v3(rows, parts)
    Σ = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows)]
    perm = _block_reverse_perm([p.n for p in parts])
    all(p -> p.J == I, parts) && return Any[Any[Σ[perm[i], perm[j]] for j in eachindex(perm)] for i in eachindex(perm)]
    J = _block_diagonal([p.J for p in parts])[perm, :]
    S = J * Σ * J'
    return Any[Any[i <= j ? S[i, j] : S[j, i] for j in axes(S, 2)] for i in axes(S, 1)]   # exactly symmetric
end
# The block-diagonal matrix of the square blocks `Js`, in order (a fill: no splat of a data-length list).
function _block_diagonal(Js::AbstractVector)
    n = sum(size(J, 1) for J in Js; init = 0)
    B = zeros(Float64, n, n)
    o = 0
    for J in Js
        k = size(J, 1)
        B[(o + 1):(o + k), (o + 1):(o + k)] = J
        o += k
    end
    return B
end
# The permutation that lists consecutive blocks of `sizes` entries in reverse
# block order, each block's entries kept in order.
function _block_reverse_perm(sizes)
    offsets = cumsum([0; sizes[1:(end - 1)]])
    return reduce(vcat, reverse([collect((o + 1):(o + n)) for (o, n) in zip(offsets, sizes)]); init = Int[])
end

# The list's key and order changed together, so a reader can tell them apart:
# v2/v5's `{"type": "calibration_chain", "members": [inner, outer]}` (application
# order) is v3/v6's `{"type": "composition", "compose": [outer, inner]}` (CT's
# order), the covariance permuted with the members; v3/v5's lens record
# `"chain": [inner, …, outer]` is v4/v6's `"compose": [outer, …, inner]`.
# `_composition_v3_to_v2` puts a current composition record (type key `key`:
# "type" in a file, "kind" in the identity tree) back under v5's names and in
# application order — the identity tree's fixed form, which keeps every id v5
# could name; any other record as it is.
_composition_v3_to_v2(c::AbstractDict, key = "type") =
    get(c, key, nothing) == _COMPOSITION_TYPE ? _recomposed(c, key, _COMPOSE_KEY, _RETIRED_COMPOSITION_TYPE, "members") : c
function _recomposed(c::AbstractDict, key, from, type, to)
    out = Dict{String, Any}(k => v for (k, v) in c if k != from)
    out[key] = type
    ms = c[from]
    out[to] = reverse(ms)
    if haskey(c, "covariance")
        p = _block_reverse_perm([_record_parameter_count(r[key], get(r, "nodes", ())) for r in ms])
        Σ = c["covariance"]
        out["covariance"] = Any[Any[Σ[p[i]][p[j]] for j in eachindex(p)] for i in eachindex(p)]
    end
    return out
end
# A lens record in the other form: v3/v5's "chain" (application order) as
# v4/v6's "compose" (CT's order, the outer first), and back; any other lens
# record as it is.
function _lens_v3_to_v4(l::AbstractDict)
    _holds_key(l, _COMPOSE_KEY) && _not_written_by("a cameras v3 / rig v5 lens record", "holds no \"$(_COMPOSE_KEY)\" (v3/v5 listed its stages under \"chain\")")
    return _relisted(l, "chain", _COMPOSE_KEY)
end
_lens_v4_to_v3(l::AbstractDict) = _relisted(l, _COMPOSE_KEY, "chain")
_lens_v3_to_v4(l) = l
_lens_v4_to_v3(l) = l
# At every depth: a lens record nests records (an `add`'s terms, a `lift`'s
# lens, a composition's stages), and a composition inside any of them is
# relisted as well.
function _relisted(l::AbstractDict, from, to)
    out = Dict{String, Any}(String(k) => _relisted(v, from, to) for (k, v) in l)
    haskey(out, from) && (out[to] = reverse(pop!(out, from)))
    return out
end
_relisted(v::AbstractVector, from, to) = map(x -> _relisted(x, from, to), v)
_relisted(x, from, to) = x
# a camera body (a store's camera, a rig's sensor camera) with its lens in the other form
_camera_lens(f, c::AbstractDict) = haskey(c, "lens") ? merge(Dict{String, Any}(c), Dict{String, Any}("lens" => f(c["lens"]))) : c
_camera_v3_to_v4(c::AbstractDict) = _camera_lens(_lens_v3_to_v4, c)
_camera_v4_to_v3(c::AbstractDict) = _camera_lens(_lens_v4_to_v3, c)
function _cameras_v3_to_v4(old)
    t = copy(old)
    t["schema"] = "vgio/cameras/v4"
    t["cameras"] = Any[_camera_v3_to_v4(c) for c in old["cameras"]]
    return t
end

# vgio/cameras/v1 → v2 → v3 (VisualGeometryIo's v1 reader, `v1.jl`, and its v2
# lens table, `camera_schema.jl`, restored from that package's history):
# - v1 held one `vgio/camera/v1` record per camera — `size` and `intrinsics`
#   each declaring `"unit": "px"`, its own `pixel_chart`, and the lens as
#   `{"model": <VGC 0.x type name>, "params": [...]}`, a flat array in the
#   type's field order; any other top-level key was its writer's metadata;
# - v2 declared the pixel frame once, put a camera's size as `width` and
#   `height`, and named every lens coefficient (`{"model": <tag>, "params":
#   {"k1": …}}`), the metadata under `provenance`;
# - v3 holds a lens as VGC's lens record (the lens families).
# The v1 model names are VGC 0.x type names the stack has retired (the
# retired-name list of `check_naming`). A v1 file holds them as DATA, so they
# are spelt here in pieces, as that list spells them: tokens a published file
# holds, never a name this code binds or calls.
const _V1_BROWN_CONRADY = "Brown" * "Conrady"
const _V1_RATIONAL_RADIAL = "Rational" * "Radial"
const _V1_EQUIDISTANT = "Equidistant" * "Fisheye"
# A v1 lens model as its v2 tag and parameter names, in v1's order (VGC 0.x's
# field order and parameter names). A v1 Brown–Conrady lens written before
# the thin-prism terms holds 8, the same camera with s1 = … = s4 = 0, as v1
# read it; a rational-radial one holds its degrees `[P, Q]` first.
const _V1_LENS_MODELS = Dict(
    "Pinhole" => ("pinhole", String[]),
    _V1_EQUIDISTANT => ("equidistant_fisheye", String[]),
    _V1_BROWN_CONRADY => ("brown_conrady", ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]),
    "KannalaBrandt" => ("kannala_brandt", ["k1", "k2", "k3", "k4"]),
    "DoubleSphere" => ("double_sphere", ["xi", "alpha"]),
    "Scaramuzza" => ("scaramuzza", ["a2", "a3", "a4"]),
)
_v2_params(names, values) = Dict{String, Any}(n => Float64(v) for (n, v) in zip(names, values))
function _lens_v1_to_v2(l::AbstractDict)
    Set(keys(l)) == Set(("model", "params")) ||
        throw(ArgumentError("a vgio/camera/v1 lens is {\"model\", \"params\"}; got the keys $(sort!(collect(keys(l))))"))
    model, p = l["model"], Any[x for x in l["params"]]
    if model == _V1_RATIONAL_RADIAL
        length(p) >= 2 || throw(ArgumentError("a v1 $(model) lens holds its degrees [P, Q] first; got $(length(p)) params"))
        P, Q = Int(p[1]), Int(p[2])
        length(p) == 2 + P + Q ||
            throw(ArgumentError("a v1 $(model) lens of degrees [$(P), $(Q)] holds $(2 + P + Q) params; got $(length(p))"))
        names = P == 0 ? _numbered("k", Q) : vcat(_numbered("num", P), _numbered("den", Q))
        return Dict{String, Any}("model" => P == 0 ? "division" : "rational_radial", "params" => _v2_params(names, p[3:end]))
    end
    haskey(_V1_LENS_MODELS, model) || throw(
        ArgumentError("unknown vgio/camera/v1 lens model $(repr(model)); v1 held $(join(sort!([collect(keys(_V1_LENS_MODELS)); _V1_RATIONAL_RADIAL]), ", "))")
    )
    tag, names = _V1_LENS_MODELS[model]
    model == _V1_BROWN_CONRADY && length(p) == 8 && append!(p, zeros(4))
    length(p) == length(names) || throw(ArgumentError("a v1 $(model) lens takes $(length(names)) params; got $(length(p))"))
    return Dict{String, Any}("model" => tag, "params" => _v2_params(names, p))
end

# A v2 lens as the lens VGC 0.x built from it, in VGC's lens families: its
# record. A v2 lens named every coefficient, and the names must be exactly the
# model's (v2's own check); the degrees of the division and rational models
# are how many it names.
_count_named(have, prefix) = count(n -> occursin(Regex("^$(prefix)\\d+\$"), n), have)
_numbered(prefix, n) = ["$(prefix)$(i)" for i in 1:n]
const _V2_FIXED_LENS_NAMES = Dict(tag => names for (tag, names) in values(_V1_LENS_MODELS))
_v2_lens_names(tag, have) = tag == "division" ? _numbered("k", _count_named(have, "k")) :
    tag == "rational_radial" ? vcat(_numbered("num", _count_named(have, "num")), _numbered("den", _count_named(have, "den"))) :
    _V2_FIXED_LENS_NAMES[tag]
const _V2_LENS_TAGS = vcat(sort!(collect(keys(_V2_FIXED_LENS_NAMES))), ["division", "rational_radial"])
function _lens_v2_to_v3(l::AbstractDict)
    (Set(keys(l)) == Set(("model", "params")) && l["params"] isa AbstractDict) ||
        throw(ArgumentError("a vgio/cameras/v2 lens is {\"model\", \"params\": {…}}; got the keys $(sort!(collect(keys(l))))"))
    tag, p = l["model"], l["params"]
    tag in _V2_LENS_TAGS || throw(ArgumentError("unknown vgio/cameras/v2 lens model $(repr(tag)); v2 held $(join(_V2_LENS_TAGS, ", "))"))
    have = String.(collect(keys(p)))
    names = _v2_lens_names(tag, have)
    Set(have) == Set(names) ||
        throw(ArgumentError("a v2 $(tag) lens takes the parameters ($(join(names, ", "))); the record has ($(join(sort!(have), ", ")))"))
    v = Float64[Float64(p[n]) for n in names]
    nnum = count(startswith("num"), names)
    lens = tag == "pinhole" ? Pinhole() :
        tag == "equidistant_fisheye" ? Lens(Equidistant()) :
        tag == "brown_conrady" ? _brown_conrady_lens(v) :
        tag == "kannala_brandt" ? KannalaBrandt(v...) :       # θ_d = θ(1 + k₁θ² + … + k₄θ⁸), VGC 0.x's and VGC's alike
        tag == "double_sphere" ? Lens(DoubleSphere(v[1], v[2])) :
        tag == "scaramuzza" ? Scaramuzza(v...) :
        tag == "division" ? Division(v...) :
        LiftLens(Radial(Tuple(v[1:nnum]), Tuple(v[(nnum + 1):end])))     # x_u = x_d·P(ρ²)/Q(ρ²)
    return _lens_v4_to_v3(lens_record(lens))      # VGC's record, in v3's form (a composition's stages under "chain")
end
# VGC 0.x's Brown–Conrady lens — OpenCV's model with its fields k1…k6, p1, p2,
# s1…s4 — as `opencv_lens` of OpenCV's vector [k1, k2, p1, p2, k3, k4, k5, k6,
# s1…s4] cut to the shortest length (4, 5, 8 or 12) that holds every non-zero
# coefficient: the same map, in the arity VGC's families evaluate (the record
# BlobBoards' own cameras v3 of these files held).
function _brown_conrady_lens(v)
    c = [v[1], v[2], v[7], v[8], v[3], v[4], v[5], v[6], v[9], v[10], v[11], v[12]]
    n = something(findlast(!iszero, c), 0)
    return opencv_lens(c[1:first(l for l in (4, 5, 8, 12) if l >= n)])
end

# A v1 camera record as its v2 body, its principal point in `frame`. v1's
# reader built each camera in its record's own chart (`CameraModel`'s
# `pixel_frame`): a store whose cameras share one chart declares it, every
# number as it is; one whose cameras do not has each principal point moved into
# the canonical frame by the pixel frames' map, as that constructor did.
const _V1_CAMERA_KEYS = ("schema", "name", "size", "pixel_chart", "intrinsics", "lens")
# A v1 camera record's shape, checked before a field is used (the numbers are
# checked by the current record's validator, after the steps).
function _check_v1_camera(c)
    (c isa AbstractDict && get(c, "schema", nothing) == "vgio/camera/v1") ||
        throw(ArgumentError("a vgio/cameras/v1 camera is a vgio/camera/v1 record; got $(c isa AbstractDict ? "schema $(repr(get(c, "schema", nothing)))" : repr(c))"))
    Set(keys(c)) == Set(_V1_CAMERA_KEYS) ||
        throw(ArgumentError("a vgio/camera/v1 record holds exactly $(join(_V1_CAMERA_KEYS, ", ")); got $(join(sort!(collect(keys(c))), ", "))"))
    sz, k = c["size"], c["intrinsics"]
    (sz isa AbstractDict && get(sz, "unit", nothing) == "px" && all(in(keys(sz)), ("width", "height")) &&
        k isa AbstractDict && get(k, "unit", nothing) == "px" && all(in(keys(k)), ("fx", "fy", "cx", "cy", "skew")) &&
        c["pixel_chart"] isa AbstractString && c["lens"] isa AbstractDict) ||
        throw(ArgumentError("camera $(repr(c["name"])): a vgio/camera/v1 record's size (width, height) and intrinsics (fx, fy, cx, cy, skew) are in px"))
    return nothing
end
function _camera_v1_to_v2(c::AbstractDict, frame::Type{<:PixelFrame})
    sz, k = c["size"], c["intrinsics"]
    from = pixel_frame_from_name(c["pixel_chart"])
    cx, cy = from === frame ? (k["cx"], k["cy"]) : EuclideanMap(from, frame)(SVector{2, Float64}(Float64(k["cx"]), Float64(k["cy"])))
    return Dict{String, Any}(
        "name" => c["name"], "width" => sz["width"], "height" => sz["height"],
        "K" => Dict{String, Any}("fx" => k["fx"], "fy" => k["fy"], "cx" => cx, "cy" => cy, "skew" => k["skew"]),
        "lens" => _lens_v1_to_v2(c["lens"]),
    )
end
function _cameras_v1_to_v2(old)
    get(old, "cameras", nothing) isa AbstractVector || throw(ArgumentError("a vgio/cameras/v1 store holds its `cameras`, an array"))
    foreach(_check_v1_camera, old["cameras"])
    charts = unique(String[c["pixel_chart"] for c in old["cameras"]])
    frame = length(charts) == 1 ? pixel_frame_from_name(only(charts)) : CANONICAL_PIXEL_FRAME
    t = Dict{String, Any}(
        "schema" => "vgio/cameras/v2", "pixel_frame" => pixel_frame_name(frame),
        "cameras" => Any[_camera_v1_to_v2(c, frame) for c in old["cameras"]],
    )
    # v1's other top-level keys were its writer's metadata: the provenance
    meta = Dict{String, Any}(k => v for (k, v) in old if !(k in ("schema", "cameras")))
    isempty(meta) || (t["provenance"] = meta)
    return t
end
function _cameras_v2_to_v3(old)
    t = copy(old)
    t["schema"] = "vgio/cameras/v3"
    t["cameras"] = Any[_camera_lens(_lens_v2_to_v3, c) for c in old["cameras"]]
    return t
end

_map_rig_sensors(f, tree::AbstractDict) = haskey(tree, "sensors") ?
    merge(Dict{String, Any}(tree), Dict{String, Any}("sensors" => Any[merge(Dict{String, Any}(x), Dict{String, Any}("camera" => f(x["camera"]))) for x in tree["sensors"]])) : tree

_body_calibration(f, b::AbstractDict) = haskey(b, "calibration") ? merge(Dict{String, Any}(b), Dict{String, Any}("calibration" => f(b["calibration"]))) : b
_body_v2_to_v3(b::AbstractDict) = _body_calibration(_calibration_v2_to_v3, b)
_body_v3_to_v2(b::AbstractDict) = _body_calibration(_composition_v3_to_v2, b)
function _target_v2_to_v3(old)
    t = copy(old)
    t["schema"] = "vgio/target/v3"
    t["target"] = _body_v2_to_v3(old["target"])
    return t
end
# the v5 → v6 change and back: every sensor lens's and target calibration's
# composition in the other form
_rig_v5_to_v6(old) = (t = _map_rig_sensors(_camera_v3_to_v4, _map_rig_targets(_body_v2_to_v3, old)); t["schema"] = "vgio/rig/v6"; t)
_rig_v6_to_v5(new) = _map_rig_sensors(_camera_v4_to_v3, _map_rig_targets(_body_v3_to_v2, new))

# vgio/target/v3 → v4 and vgio/rig/v6 → v7 (the target layouts, 2026-11): a v4
# planar body may state its design by a `layout`; a v3 body (a reference or
# nominal points) is a v4 body as it stands, so the step moves the version and
# no number, and a rig's identity does not move.
function _layout_free_body(b::AbstractDict, record)
    haskey(b, "layout") && _not_written_by(record, "holds no \"layout\" (a target layout is target v4 / rig v7)")
    return b
end
function _target_v3_to_v4(old)
    _layout_free_body(old["target"], "a vgio/target/v3 record")
    t = copy(old)
    t["schema"] = "vgio/target/v4"
    return t
end
_rig_v6_to_v7(old) = (t = _map_rig_targets(b -> _layout_free_body(b, "a vgio/rig/v6 record"), old); t["schema"] = "vgio/rig/v7"; t)

# -----------------------------------------------------------------------------
# The steps, one per version, and the file migrations that run them
# -----------------------------------------------------------------------------

migrate_schema(::Val{Symbol("vgio/cameras/v1")}, t::AbstractDict) = _cameras_v1_to_v2(t)
migrate_schema(::Val{Symbol("vgio/cameras/v2")}, t::AbstractDict) = _cameras_v2_to_v3(t)
migrate_schema(::Val{Symbol("vgio/cameras/v3")}, t::AbstractDict) = _cameras_v3_to_v4(t)
# vgio/cameras/v4 → v5: v5 adds the optional per-camera `lens_coverage` (a camera's lens coverage); a v4 store has none, so its
# cameras read with the whole sensor lit. No number moves.
function _cameras_v4_to_v5(old)
    any(c -> haskey(c, "lens_coverage"), old["cameras"]) &&
        _not_written_by("a vgio/cameras/v4 store", "holds no \"lens_coverage\" (a camera's lens coverage is cameras v5)")
    return _with_schema(old, "vgio/cameras/v5")
end
migrate_schema(::Val{Symbol("vgio/cameras/v4")}, t::AbstractDict) = _cameras_v4_to_v5(t)
migrate_schema(::Val{Symbol("vgio/target/v1")}, t::AbstractDict) = _target_v1_to_v2(t)
migrate_schema(::Val{Symbol("vgio/target/v2")}, t::AbstractDict) = _target_v2_to_v3(t)
migrate_schema(::Val{Symbol("vgio/target/v3")}, t::AbstractDict) = _target_v3_to_v4(t)
migrate_schema(::Val{Symbol("vgio/rig/v6")}, t::AbstractDict) = _rig_v6_to_v7(t)
migrate_schema(::Val{Symbol("vgio/rig/v4")}, t::AbstractDict) = (foreach(x -> _check_v1_warp(x["target"]), get(t, "targets", ())); _rig_v4_to_v5(t))
migrate_schema(::Val{Symbol("vgio/rig/v5")}, t::AbstractDict) = _rig_v5_to_v6(t)
migrate_schema(::Val{Symbol("vgio/generic_camera/v2")}, t::AbstractDict) = _generic_camera_v2_to_v3(t)
migrate_schema(::Val{Symbol("vgio/rig/v8")}, t::AbstractDict) = _rig_v8_to_v9(t)
migrate_schema(::Val{Symbol("vgio/transforms/v2")}, t::AbstractDict) = _hand_eye_transforms_v2_to_v3(t)
migrate_schema(::Val{Symbol("vgio/trajectory/v2")}, t::AbstractDict) = _track_segment_trajectory_v2_to_v3(t)
migrate_schema(::Val{Symbol("vgio/calibration/v1")}, t::AbstractDict) = _calibration_v1_to_v2(t)

"""
    _calibration_v1_to_v2(tree) -> Dict{String, Any}

The version step `vgio/calibration/v1` → v2 (the owner's ruling, 2026-11-11: the stack's one
parameter-holding keyword is `fixed`). A v1 manifest's `model` lens table named the coefficients a
calibrator MOVES (`"free"`); v2 names those it HOLDS (`"fixed"`), the complement within the lens's
`parameter_names`, in their order — the same LensSpec. A model without `"free"` (every coefficient
moving) is unchanged. Read on load only, never rewritten on disk; saving writes v2.
"""
function _calibration_v1_to_v2(old::AbstractDict)
    t = Dict{String, Any}(String(k) => v for (k, v) in old)
    if haskey(t, "model")
        m = Dict{String, Any}(String(k) => v for (k, v) in t["model"])
        if haskey(m, "free")
            free = Symbol.(m["free"])
            body = Dict{String, Any}(k => v for (k, v) in m if k ∉ ("free", "intrinsics"))
            names = VgtkCore.parameter_names(VgtkCore.lens(VgtkCore.LensSpec(body)))
            for n in free
                n in names || throw(ArgumentError("a vgio/calibration/v1 model's free name :$n is no coefficient of its lens ($names)"))
            end
            delete!(m, "free")
            fixed = [String(n) for n in names if !(n in free)]
            isempty(fixed) || (m["fixed"] = fixed)
        end
        t["model"] = m
    end
    t["schema"] = "vgio/calibration/v2"
    return t
end

"""
    _track_segment_trajectory_v2_to_v3(tree) -> Dict{String, Any}

The version step `vgio/trajectory/v2` → v3 for CaliBlobs' track segment of 2026-09 (owner ruling
2026-11-11, Option A), the one v2 trajectory layout read: the `track_segment_<s>.json` CaliBlobs
icra2026's motion-capture runs wrote — `from = "track:<s>"`, `to = "camera"`, each frame `{t, pose}`,
the frames' images and the segment's `component` in its provenance. Any other v2 trajectory is refused
(its names are free strings), as before; saving writes only the current version.

THE GAUGE. The current writer names a segment's gauge by its poses' destination frame
(CaliBlobs icra2026/src/trajectory.jl:1172, `gauge = _dst_frame(first(sg.poses))`), which is the
component's gauge board frame `target:<uid>` (trajectory.jl:304, `G = VgtkCore.reference_frame(comp.targets)`).
The v2 file does not record that uid, so the gauge is named by its COMPONENT, as the same run's hand-eye
record names the component's frame: `target:scene_<c>` (trajectory.jl:491, the hand-eye fit's marker
`"scene_\$(s.component)"`; its mount file, trajectory.jl:1123; the v2 hand-eye step reads `scene_<c>`
the same). `camera` is `camera_optical`; each frame takes its image from the provenance; every pose
as it stands. (Line numbers at CaliBlobs 72a6dba.)
"""
function _track_segment_trajectory_v2_to_v3(old::AbstractDict)
    free(why) = throw(ArgumentError("a vgio/trajectory/v2 file's frame names are free strings, which the frame grammar " *
        "does not type; the one v2 layout read is CaliBlobs' 2026-09 track segment (`track:<s>` → `camera`, its " *
        "`component` and the frames' images in its provenance): $(why). Re-export it from its producer"))
    from, to = String(old["from"]), String(old["to"])
    (occursin(r"^track:\d+$", from) && to == "camera") || free("its frames are $(repr(from)) → $(repr(to))")
    prov = get(old, "provenance", nothing)
    (prov isa AbstractDict && get(prov, "component", nothing) isa Integer) ||
        free("its provenance states no integer `component`, which names its gauge")
    images = Dict{Int, String}(Int(i["t"]) => String(i["image"]) for i in get(prov, "images", Any[]))
    frames = Any[]
    for f in old["frames"]
        n = Dict{String, Any}(f)
        haskey(images, Int(f["t"])) && (n["image"] = images[Int(f["t"])])
        push!(frames, n)
    end
    t = Dict{String, Any}(old)
    t["schema"] = "vgio/trajectory/v3"
    t["from"], t["to"] = "target:scene_$(prov["component"])", "camera_optical"
    t["frames"] = frames
    return t
end

# vgio/transforms/v2 → v3: THE HAND-EYE RECORD OF 2026-09 (owner ruling 2026-11-11).
# A v2 file names its frames by free strings, which v3 types through the frame
# grammar, so a v2 file is in general not readable. ONE v2 layout is: the
# hand-eye record bmvc2026's writer wrote for CaliBlobs icra2026's motion-capture
# runs (2026-09; the published runs hold it) — the camera mount `camera_rig →
# camera_optical`, then the marker mount, one `<marker> → board_rig` per
# marker, a printed board `printed_board_<id>` or a reconstructed scene
# `scene_<c>`. Its names map one to one onto the frames the current hand-eye
# writer names (`body:camera_rig`, `camera_optical`, `target:<id>`,
# `target:scene_<c>`, `body:board_rig`); every number is carried as it stands.
# Any other v2 file is refused as before. Saving writes only the current version.
const _V2_HAND_EYE_CAMERA = ("camera_rig" => "camera_optical") => ("body:camera_rig" => "camera_optical")
const _V2_HAND_EYE_MOUNT = "board_rig" => "body:board_rig"
function _v2_hand_eye_marker(name::AbstractString)
    m = match(r"^printed_board_([A-Za-z0-9_]+)$", name)
    m === nothing || return "target:" * m[1]
    occursin(r"^scene_\d+$", name) && return "target:" * name
    return nothing
end
function _hand_eye_transforms_v2_to_v3(old::AbstractDict)
    free(why) = throw(ArgumentError("a vgio/transforms/v2 file's frame names are free strings, which the frame grammar " *
        "does not type; the one v2 layout read is the 2026-09 hand-eye record (`camera_rig → camera_optical`, then " *
        "`printed_board_<id>` or `scene_<c>` → `board_rig`): $(why). Re-export it from its producer"))
    ts = old["transforms"]
    ts isa AbstractVector && !isempty(ts) || free("it holds no transforms")
    out = Any[]
    for (k, r) in enumerate(ts)
        from, to = String(r["from"]), String(r["to"])
        name = if k == 1
            (from => to) == first(_V2_HAND_EYE_CAMERA) || free("its first transform is $(repr(from)) → $(repr(to)), not the camera mount")
            last(_V2_HAND_EYE_CAMERA)
        else
            marker = to == first(_V2_HAND_EYE_MOUNT) ? _v2_hand_eye_marker(from) : nothing
            marker === nothing && free("its transform $(repr(from)) → $(repr(to)) is not a marker of the mount")
            marker => last(_V2_HAND_EYE_MOUNT)
        end
        n = Dict{String, Any}(r)
        n["from"], n["to"] = first(name), last(name)
        push!(out, n)
    end
    length(out) >= 2 || free("it holds the camera mount and no marker")
    t = Dict{String, Any}(old)
    t["schema"] = "vgio/transforms/v3"
    t["transforms"] = out
    return t
end

# The covariance storage (2026-11): a covariance is written in the storage its
# value holds (VisualGeometryToolkitCore's CovMat: an SMatrix, SDiagonal or
# ScaledIdentity) — rows, `{"diagonal": […]}` or `{"isotropic": λ}` — so the
# kinds that carry one move up a version, and an earlier file, whose every
# covariance is rows, is a file of the next version as it stands. No number
# moves. (target_detections v1 → v2, target_points2d / target_points3d v1 → v2,
# target_poses v1 → v2, trajectory v3 → v4, transforms v3 → v4, target v4 → v5,
# rig v7 → v8.)
const _COVARIANCE_STORAGE_VERSIONS = (
    "vgio/target_detections/v1" => "vgio/target_detections/v2",
    "vgio/target_points2d/v1" => "vgio/target_points2d/v2",
    "vgio/target_points3d/v1" => "vgio/target_points3d/v2",
    "vgio/target_poses/v1" => "vgio/target_poses/v2",
    "vgio/trajectory/v3" => "vgio/trajectory/v4",
    "vgio/transforms/v3" => "vgio/transforms/v4",
    "vgio/target/v4" => "vgio/target/v5",
    "vgio/rig/v7" => "vgio/rig/v8",
)
_with_schema(t::AbstractDict, s::AbstractString) = (n = Dict{String, Any}(t); n["schema"] = s; n)
for (from, to) in _COVARIANCE_STORAGE_VERSIONS
    @eval migrate_schema(::Val{Symbol($from)}, t::AbstractDict) = _with_schema(t, $to)
end

# vgio/generic_camera/v2 → v3 (the present nodes, 2026-11): a v2 model held
# every node of its lattice, row-major; v3 lists the held nodes as runs per
# lattice row (`present`) and its grids over them, and a record without
# `present` holds every node, so a v2 record is a v3 one as it stands. No number
# moves.
function _generic_camera_v2_to_v3(old)
    haskey(old, "present") && throw(ArgumentError("a vgio/generic_camera/v2 record holds every node; it has no `present`"))
    t = copy(old)
    t["schema"] = "vgio/generic_camera/v3"
    return t
end

# THE OLD GRID BODIES (generic_camera v2/v3, a rig/v6–v8 sensor's), read once into
# the cameras of VisualGeometryToolkitCore 0.9 (the spline lens): a pixel lattice
# over the calibrated area [min, max + 1) of `grid_width × grid_height` direction
# controls (the held nodes `present`, row by row), with an optional nominal pinhole
# and image domain:
#
#   {"type": <the old central or non-central type>, "width", "height",
#    "calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y",
#    "grid_width", "grid_height", "grid_layout": "row_major", "present"?,
#    "direction_grid", "point_grid"? (non-central, in the file's length unit),
#    <the nominal pinhole>?: [fx, fy, cx, cy, skew], "domain"?: {"ellipse": [cx, cy, a, b, θ]}, "pitch"? (a rig's)}
#
# The camera is Core's `CameraModel(SplineLens, controls, size)` of the pixel
# lattice: K the nominal pinhole (skew 0), else the lattice's gauge; the backward area K⁻¹ of the calibrated area, so every pixel
# backprojects as the old model did. The domain becomes the camera's lens
# coverage (the camera store's form); the controls and the held nodes are the
# record's own numbers, the points copied as the file holds them (no unit
# round trip). An old body is read, never written (owner, 2026-11-09).
# (The old type names and the nominal pinhole's key are retired as names; as the
# tokens these records store they are check_naming's FOREIGN_FORMAT_TOKENS.)
const _OLD_NOMINAL_KEY = "nominal_intrinsics"   # (the type names: style.jl)
const _OLD_GRID_KEYS = ("type", "width", "height", "calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y",
    "grid_width", "grid_height", "grid_layout", "direction_grid")
# 0-based runs as the 1-based runs they read as; `nothing` for a value that is no list of runs (the check refuses it).
_shifted_runs(r) = try
    _runs_v010(r)
catch e
    e isa Union{MethodError, BoundsError, TypeError} || rethrow()
    nothing
end
function _old_grid_camera(b::AbstractDict)
    for k in _OLD_GRID_KEYS
        haskey(b, k) || _not_written_by("an old grid body (generic_camera v2/v3, a rig/v6–v8 sensor)", "has no $(repr(k)), which every one held")
    end
    _is_old_grid_type(b["type"]) ||
        _not_written_by("an old grid body", "is of type $(repr(b["type"])), not $(_OLD_CENTRAL_TYPE) or $(_OLD_NONCENTRAL_TYPE)")
    # its fields' types, as its version's spec held them (a value of another type is not a record it wrote)
    what = "an old grid body"
    for k in ("width", "height", "grid_width", "grid_height")
        _validate(_PosInt(), b[k], what, k)
    end
    for k in ("calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y")
        _validate(_Int(), b[k], what, k)
    end
    for k in ("direction_grid", "point_grid")
        haskey(b, k) && _validate(_Vec(_XYZ), b[k], what, k)
    end
    haskey(b, _OLD_NOMINAL_KEY) && _validate(_Vec(_Real(), 5), b[_OLD_NOMINAL_KEY], what, _OLD_NOMINAL_KEY)
    haskey(b, "domain") && _validate(_Obj(:ellipse => _Vec(_Real(), 5)), b["domain"], what, "domain")
    b["grid_layout"] == _GRID_LAYOUT || _not_written_by("an old grid body", "has the grid_layout $(repr(b["grid_layout"])), not $(repr(_GRID_LAYOUT))")
    gw, gh = Int(b["grid_width"]::Integer), Int(b["grid_height"]::Integer)
    # its runs are 0-based (before IO 0.10): checked as the 1-based runs they read as
    runs = haskey(b, "present") ? something(_shifted_runs(b["present"]), b["present"]) : nothing
    runs === nothing || _validate(_RowRuns(), runs, "an old grid body (its 0-based runs + 1)", "present")
    msg = _lattice_count_problem(gw, gh, runs, [k => b[k] for k in ("direction_grid", "point_grid") if haskey(b, k)])
    msg === nothing || throw(ArgumentError("an old grid body: $(msg)"))
    (b["type"] == _OLD_NONCENTRAL_TYPE) == haskey(b, "point_grid") ||
        throw(ArgumentError("an old grid body holds a point_grid exactly when it is non-central"))
    P = haskey(b, "present") ? _present_from_runs(b["present"], gw, gh; base = 0) : trues(gh, gw)   # (0-based: before IO 0.10)
    x0, y0 = Int(b["calibration_min_x"]), Int(b["calibration_min_y"])
    bounds = VgtkCore.Rect(x0, y0, Int(b["calibration_max_x"]) - x0 + 1, Int(b["calibration_max_y"]) - y0 + 1)   # the inclusive pixel bounds
    nominal = haskey(b, _OLD_NOMINAL_KEY) ? LogicalIntrinsics(_k_from_values(b[_OLD_NOMINAL_KEY])) : nothing
    C = _controls_from_json(b["direction_grid"], P, Val(3))
    sz, present = Size2(Int(b["width"]), Int(b["height"])), all(P) ? nothing : P
    # no nominal pinhole: K is the lattice's gauge, which a lattice without a complete window at its pixel lacks
    # (Core's `has_spline_gauge`) — name the reader's keyword that supplies it
    nominal === nothing && !VgtkCore.has_spline_gauge(C, sz; calib_area = bounds, present) && throw(
        ArgumentError(
            "an old grid body stores no nominal pinhole, and its lattice has no complete 4 × 4 window at the pixel its gauge " *
                "reads, so K cannot be derived from it; supply K: an old grid file that stores none takes it from the reader: " *
                "load(BoundedCamera, path; intrinsics = K), or load(Rig, path; intrinsics = Dict(:sensor_id => K))"
        )
    )
    return CameraModel(SplineLens, C, sz; calib_area = bounds, intrinsics = nominal, present)
end
# An old body's domain as the camera's lens coverage, and its pitch (a rig's), as the new bodies hold them.
_old_grid_extras(b::AbstractDict) = merge(
    haskey(b, "domain") ? Dict{String, Any}("lens_coverage" => Dict{String, Any}("ellipse" => b["domain"]["ellipse"])) : Dict{String, Any}(),
    haskey(b, "pitch") ? Dict{String, Any}("pitch" => _size_v010(b["pitch"])) : Dict{String, Any}(),
)
# A central body as the camera body of its SplineLens camera (in the canonical pixel frame, every old grid's), in
# the 0.10 body's shape (the one conversion builds the current body; `_camera_body_v010` leaves it as it is).
function _central_grid_body(b::AbstractDict)
    cam = _old_grid_camera(b)
    body = Dict{String, Any}(
        "image_size" => _size_v010(Any[b["width"], b["height"]]), "K" => _lower(_k_record(cam.intrinsics.K, CANONICAL_PIXEL_FRAME)),
        "lens" => _lower(_lens_json(cam.lens)),
    )
    return merge(body, _old_grid_extras(b))
end
# A non-central body as the non-central record's body, its points the old body's own numbers.
function _noncentral_grid_body(b::AbstractDict)
    cam = _old_grid_camera(b)
    d = cam.lens.backward
    m = NoncentralSplineModel(cam.intrinsics, d, ChartSpline(d.area, zero(d.controls)); present = cam.lens.present)
    body = _lower(_model_body(VgioStyle(), m, Size2(Int(b["width"]), Int(b["height"])); points = b["point_grid"]))
    return merge(body, _old_grid_extras(b))
end

_with_given_intrinsics(d, ::Nothing, path) = d
function _with_given_intrinsics(d, K::CameraCalibrationMat, path)
    (d isa AbstractDict && get(d, :schema, nothing) in _OLD_GRID_SCHEMAS) || _holds_k(path, "the record")
    return _give_nominal!(d, K, path, "the record")
end
function _with_given_intrinsics(d, Ks::AbstractDict{Symbol, <:CameraCalibrationMat}, path)
    sensors = Dict{String, Any}(String(s["id"]) => s for s in get(d, "sensors", Any[]) if s isa AbstractDict && haskey(s, "id"))
    for (id, K) in Ks
        s = get(sensors, String(id), nothing)
        s === nothing && throw(ArgumentError("$(path): no sensor $(repr(String(id))) to supply K to; sensors: $(sort!(collect(keys(sensors))))"))
        c = s["camera"]
        (c isa AbstractDict && _is_old_grid_type(get(c, "type", nothing))) || _holds_k(path, "sensor $(repr(String(id)))")
        _give_nominal!(c, K, path, "sensor $(repr(String(id)))")
    end
    return d
end
# An old grid body `b` without a nominal pinhole takes K as one.
function _give_nominal!(b::AbstractDict, K::CameraCalibrationMat, path, what)
    haskey(b, _OLD_NOMINAL_KEY) && _holds_k(path, what)
    b[_OLD_NOMINAL_KEY] = Any[Float64(K[1, 1]), Float64(K[2, 2]), Float64(K[1, 3]), Float64(K[2, 3]), Float64(K[1, 2])]
    return b
end
_holds_k(path, what) = throw(
    ArgumentError("$(path): $(what) holds its K; `intrinsics` supplies K only to an old grid body that stores none")
)

# An old grid body (central or non-central): a vgio/generic_camera/v2 or v3 record's, a rig/v6–v8 sensor's, or one
# another record embeds; and the central one. `_is_old_grid_type` is the one test of the type, for a parsed body (string
# keys) as for a lowered one (symbol keys).
_is_old_grid_type(t) = t in (_OLD_CENTRAL_TYPE, _OLD_NONCENTRAL_TYPE)
_is_old_grid_body(b) = b isa AbstractDict && _is_old_grid_type(get(b, :type, nothing))
_is_old_central_body(b) = b isa AbstractDict && get(b, :type, nothing) == _OLD_CENTRAL_TYPE
# A parsed file that is a central vgio/generic_camera/v2 or v3 record.
_is_central_grid(d) = d isa AbstractDict && get(d, :schema, nothing) in _OLD_GRID_SCHEMAS && _is_old_central_body(d)
# An old grid body (lowered) as the body it migrates to: a central one's camera body, a non-central one's
# non-central body — the one conversion every reader of an old body takes.
_current_grid_body(b::AbstractDict) = b["type"] == _OLD_CENTRAL_TYPE ? _central_grid_body(b) : _noncentral_grid_body(b)
# A camera body as the camera readers read it, with the style to read it in: an old grid body EMBEDDED in a
# record that is no old grid file (RadiometricCamera's rc/calibration_state/v1 camera) becomes the body it
# migrates to (`_current_grid_body`, the conversion an old grid file and an old rig sensor take too), parsed
# as a reader meets a body. It is read in the colmap pixel frame (CANONICAL_PIXEL_FRAME), whatever frame the
# enclosing record declares — the frame the old grid code reads every old grid body's numbers in — and a
# non-central one's line points in the enclosing record's length unit. Any other body is read as it stands.
function _current_camera_body(s::VgioStyle, b)
    if _is_old_grid_body(b)
        body = _parse_vgio_text(JSON.json(_current_grid_body(_lower(b))), "an old grid body")
        return _with_pixel_frame(s, CANONICAL_PIXEL_FRAME), body
    end
    (b isa AbstractDict && haskey(b, :width)) || return s, b      # a current body
    return s, _parse_vgio_text(JSON.json(_camera_body_v010(_lower(b))), "an IO 0.9 camera body")
end
_with_pixel_frame(s::VgioStyle{U, F, C}, G::Type{<:PixelFrame}) where {U, F, C} = VgioStyle{U, G, C}(s.length_unit, s.contract, s.charts)
# …as the vgio/cameras record of its one camera, named after the file (`fisheye_14mm.json` → `fisheye_14mm`;
# `d` parsed, so the reader meets what a store file holds).
function _central_grid_store(d, path)
    t = _lower(d)
    t["schema"] == "vgio/generic_camera/v2" && (t = _generic_camera_v2_to_v3(t))
    store = try
        merge(migrate_envelope(_file_head(t), "vgio/cameras/v6.0"),
              Dict{String, Any}("cameras" => Any[merge(Dict{String, Any}("name" => first(splitext(basename(path)))), _central_grid_body(t))]))
    catch e
        e isa Union{ArgumentError, KeyError} || rethrow()
        throw(ArgumentError("$(path): a $(t["schema"]) central grid camera, read as a camera with a spline lens: $(sprint(showerror, e))"))
    end
    return _parse_vgio_text(JSON.json(store), path)
end

# vgio/rig/v8 → v9 (the spline lens, 2026-11): a sensor's old grid body becomes
# the camera body of its SplineLens camera (central) or the non-central body; v9
# also holds a sensor's lens coverage, which a v8 file could not (its writer
# refused one), so a v8 sensor holds none but an old domain.
function _rig_v8_to_v9(old)
    t = _map_rig_sensors(old) do c
        haskey(c, "lens_coverage") && _not_written_by("a vgio/rig/v8 sensor", "holds no \"lens_coverage\" (a sensor's lens coverage is rig v9)")
        haskey(c, "type") || return c
        return _current_grid_body(c)
    end
    t["schema"] = "vgio/rig/v9"
    return t
end

_migrate_cameras(src, dst) = (_migrate_file(Dict{Symbol, BoundedCamera}, src, dst, "the camera store"); (; to = _schema("cameras")))
migrate_schema(::Val{Symbol("vgio/cameras/v1")}, src::AbstractString, dst::AbstractString) = _migrate_cameras(src, dst)
migrate_schema(::Val{Symbol("vgio/cameras/v2")}, src::AbstractString, dst::AbstractString) = _migrate_cameras(src, dst)
migrate_schema(::Val{Symbol("vgio/cameras/v3")}, src::AbstractString, dst::AbstractString) = _migrate_cameras(src, dst)
migrate_schema(::Val{Symbol("vgio/cameras/v4")}, src::AbstractString, dst::AbstractString) = _migrate_cameras(src, dst)
# An old grid file (vgio/generic_camera v2/v3) is never rewritten: its current
# record — a central one's camera store, the store its reader gives; a non-central
# one's noncentral_spline record, through its steps — is written to the new file
# `dst` (`vgio_migrate`'s default `<stem>.camera.json` / `<stem>.noncentral.json`).
function _migrate_old_grid(src, dst)
    d = _read_vgio_json(src)
    _is_central_grid(d) || return (_migrate_file(Dict{Symbol, BoundedCamera}, src, dst, "the camera store of the old non-central grid file"; current = _lower(_one_camera_store(d, src))); (; to = _schema("cameras")))
    _migrate_file(Dict{Symbol, BoundedCamera}, src, dst, "the camera store of the old central grid file"; current = _lower(_central_grid_store(d, src)))
    return (; to = _schema("cameras"))
end
migrate_schema(::Val{Symbol("vgio/generic_camera/v2")}, src::AbstractString, dst::AbstractString) = _migrate_old_grid(src, dst)
migrate_schema(::Val{Symbol("vgio/generic_camera/v3")}, src::AbstractString, dst::AbstractString) = _migrate_old_grid(src, dst)
# The file forms of every earlier version a published file holds, each the
# guarded rewrite of its value by the current writer (`_migrate_file`).
# A rig or a target: its identity is re-minted (IO 0.10, once), and the
# identities its old version was named by are reported — `old_ids`, the one
# its own version minted first — so the observation files naming it are
# re-pointed (`vgio_migrate_tree`).
function _migrate_target(s, src, dst)
    value = _migrate_file(AbstractTarget, src, dst, "the target record")
    new_id = content_id(dst)
    old_ids = _old_target_ids(s, _raw_tree(src), value)
    return (; to = _schema("target"), old_id = first(old_ids), old_ids, new_id)
end
function _migrate_rig(s, src, dst)
    value = _migrate_file(Rig, src, dst, "the rig record")
    new_id = content_id(dst)
    old_ids = _old_rig_ids(s, _raw_tree(src), value)
    return (; to = _schema("rig"), old_id = first(old_ids), old_ids, new_id)
end
const _OLD_RIG_VERSIONS = Tuple("vgio/rig/v$(n)" for n in 4:9)
const _OLD_TARGET_VERSIONS = Tuple("vgio/target/v$(n)" for n in 1:5)
for s in _OLD_RIG_VERSIONS
    @eval migrate_schema(::Val{Symbol($s)}, src::AbstractString, dst::AbstractString) = _migrate_rig($s, src, dst)
end
for s in _OLD_TARGET_VERSIONS
    @eval migrate_schema(::Val{Symbol($s)}, src::AbstractString, dst::AbstractString) = _migrate_target($s, src, dst)
end
# Every other kind's, through its value: `length` whether its writer takes the
# file's length unit (a kind with no length — the homographies, the vignetting
# field — writes none).
for (s, T, kind, what, length) in (
        ("vgio/target_detections/v1", TargetObservations, "observations", "the target observations", true),
        ("vgio/target_points2d/v1", TargetObservations, "observations", "the target observations (2D)", true),
        ("vgio/target_points3d/v1", TargetObservations, "observations", "the target observations (3D)", true),
        ("vgio/target_detections/v2", TargetObservations, "observations", "the target observations", true),
        ("vgio/target_points2d/v2", TargetObservations, "observations", "the target observations (2D)", true),
        ("vgio/target_points3d/v2", TargetObservations, "observations", "the target observations (3D)", true),
        ("vgio/target_homographies/v1", TargetHomographies, "target_homographies", "the target homographies", false),
        ("vgio/target_poses/v1", TargetPoses, "target_poses", "the target poses", true),
        ("vgio/target_poses/v2", TargetPoses, "target_poses", "the target poses", true),
        ("vgio/trajectory/v2", Trajectory, "trajectory", "the trajectory", true),
        ("vgio/trajectory/v3", Trajectory, "trajectory", "the trajectory", true),
        ("vgio/trajectory/v4", Trajectory, "trajectory", "the trajectory", true),
        ("vgio/transforms/v2", Vector{EuclideanMap}, "transforms", "the transforms", true),
        ("vgio/transforms/v3", Vector{EuclideanMap}, "transforms", "the transforms", true),
        ("vgio/transforms/v4", Vector{EuclideanMap}, "transforms", "the transforms", true),
        ("vgio/cameras/v5", Dict{Symbol, BoundedCamera}, "cameras", "the camera store", true),
        ("vgio/capture_settings/v1", CaptureSettings, "capture_settings", "the capture settings", true),
        ("vgio/vignetting/v1", AbstractVignetting, "vignetting", "the vignetting field", false),
        ("vgio/environment_light/v1", EnvironmentLight, "environment_light", "the environment light", true),
    )
    @eval migrate_schema(::Val{Symbol($s)}, src::AbstractString, dst::AbstractString) =
        (_migrate_file($T, src, dst, $what; length = $length); (; to = _schema($kind)))
end
# A non-central camera's own kind (before IO 0.10) as the camera store of its
# one camera, named after the file (`_one_camera_store`, as its reader reads it).
migrate_schema(::Val{Symbol("vgio/noncentral_spline/v1")}, src::AbstractString, dst::AbstractString) = (
    _migrate_file(Dict{Symbol, BoundedCamera}, src, dst, "the camera store of the non-central record";
                  current = _lower(_one_camera_store(_read_vgio_json(src), src))); (; to = _schema("cameras"))
)
# The manifest: its record through the step, written by its kind's writer and
# read back as that record (its files are bound by their bytes, and a listed
# file is checked where the manifest lies: `load_manifest`, not here).
function _migrate_manifest(src, dst)
    t = migrate_record(_raw_tree(src))
    _write_manifest(dst, t)
    back = _raw_tree(dst)
    _verify(_same_tree(merge(back, Dict("generator" => t["generator"])), t), src, "the calibration manifest")
    return (; to = _schema("calibration"))
end
# A current manifest record `t` (a parsed tree), written by its kind's writer.
function _write_manifest(dst, t::AbstractDict)
    body = (result_kind = t["result_kind"], files = [(path = f["path"], sha256 = f["sha256"], schema = f["schema"], role = f["role"],
                                                       derived_from = f["derived_from"]) for f in t["files"]])
    haskey(t, "model") && (body = merge(body, (model = t["model"],)))
    haskey(t, "metrics") && (body = merge(body, (metrics = t["metrics"],)))
    _write_kind(dst, "calibration", body, _provenance_of(t); style = VgioStyle())
    return nothing
end
migrate_schema(::Val{Symbol("vgio/calibration/v1")}, src::AbstractString, dst::AbstractString) = _migrate_manifest(src, dst)
migrate_schema(::Val{Symbol("vgio/calibration/v2")}, src::AbstractString, dst::AbstractString) = _migrate_manifest(src, dst)
# A COLMAP model's units sidecar: its contract, written by the sidecar's writer
# (the file keeps its name, `vgio_units.json`, which the reader looks for).
function _migrate_units_sidecar(src, dst)
    basename(src) == UNITS_SIDECAR || throw(ArgumentError("$(src): a units sidecar is a model directory's $(UNITS_SIDECAR)"))
    c = load_units_sidecar(dirname(src))
    c === nothing && throw(ArgumentError("$(src): no units sidecar to migrate"))
    save_units_sidecar(dirname(dst), c)
    _verify(load_units_sidecar(dirname(dst)) == c, src, "the units sidecar")
    return (; to = _schema("units"))
end
migrate_schema(::Val{Symbol("vgio/units/v2")}, src::AbstractString, dst::AbstractString) = _migrate_units_sidecar(src, dst)

# The identities an old rig file was named by, the one its own version minted
# first, then IO 0.9's identity of its value (what 0.9's `content_id(path)`
# minted for a file of any version: a producer that named the rig by its path
# or its loaded value wrote that one). A v4 identity hashed the record
# (`_rig_v4_id`: the file as written, and the value in millimetres); a v5–v8
# one is 0.9's identity of the value, unless a target held the retired affine
# kind or a sensor an old grid body (`_v5_id`). Every version is matched by
# name: a future step that changes an identity must add its arm, so any other
# version is refused.
function _old_rig_ids(s, old, value)
    if s == "vgio/rig/v4"
        return unique!([_rig_v4_id(old), _rig_v4_id_mm(value, old), _v09_id(value)])
    elseif s in ("vgio/rig/v5", "vgio/rig/v6", "vgio/rig/v7", "vgio/rig/v8")
        return unique!([_v5_id(old, value), _v09_id(value)])
    elseif s == "vgio/rig/v9"
        return [_v09_id(value)]
    else
        throw(ArgumentError("the identities a $(s) rig was named by are not known: a rig version's step that migrates it states them here"))
    end
end
# …and an old target file's: IO 0.9's identity of its value, and for a target
# v2 file whose calibration held the retired affine kind the one v2 minted
# (its calibration as v2's tree held it, `_v5_identity_calibration`).
function _old_target_ids(s, old, value)
    s in _OLD_TARGET_VERSIONS ||
        throw(ArgumentError("the identities a $(s) target was named by are not known: a target version's step that migrates it states them here"))
    v09 = _v09_id(value)
    c = get(old["target"], "calibration", nothing)
    (s == "vgio/target/v2" && c isa AbstractDict && _has_affine(c)) || return [v09]
    tree = _identity_tree_v09(value isa Attributed ? value.value : value)
    tree["calibration"] = _v5_identity_calibration(c, parse_unit(old["length_unit"]::String))
    return unique!([_hash_v09(tree), v09])
end

# A targets argument that is an old rig or target file: each identity its
# version named it by mapped to its current one; none for a rig or a current file.
_old_id_map(::Rig) = Pair{String, String}[]
function _old_id_map(p::AbstractString)
    old = _raw_tree(p)
    s = get(old, "schema", nothing)
    if s in _OLD_RIG_VERSIONS
        value = load(Rig, p)
        new_id = content_id(value)
        return [o => new_id for o in _old_rig_ids(s, old, value) if o != new_id]
    elseif s in _OLD_TARGET_VERSIONS
        value = load(AbstractTarget, p)
        new_id = content_id(value)
        return [o => new_id for o in _old_target_ids(s, old, value) if o != new_id]
    end
    return Pair{String, String}[]
end

# v5's content identity of a rig: 0.9's tree of its value, but for each target
# whose v5 calibration held an affine kind that calibration as v5's tree held
# it — its record with `kind` for `type`, every length leaf 0.9's persisted
# metre value of the file's number, its covariance in metres (v5 built its tree
# from the value read back, which is exactly that).
#
# The same for every v5–v8 rig holding a grid sensor (an old grid body, which the
# v9 step turns into a spline camera): v5–v8's tree held its camera in the old
# grid form (`_v8_grid_identity`). Otherwise 0.9's identity of the value.
_has_affine(r::AbstractDict) = get(r, "type", nothing) == "affine" || any(_has_affine, get(r, "members", ()))
function _v5_id(old::AbstractDict, value)
    olds = Dict(x["id"] => x["target"] for x in get(old, "targets", ()))
    affine = [id for (id, t) in olds if _has_affine(get(t, "calibration", Dict{String, Any}()))]
    grids = Dict(x["id"] => x["camera"] for x in get(old, "sensors", ()) if haskey(x["camera"], "type"))
    tree = _identity_tree_v09(value)
    (isempty(affine) && isempty(grids)) && return _hash_v09(tree)
    u = parse_unit(old["length_unit"]::String)
    for x in get(tree, "targets", ())
        x["id"] in affine && (x["target"]["calibration"] = _v5_identity_calibration(olds[x["id"]]["calibration"], u))
    end
    for x in get(tree, "sensors", ())
        haskey(grids, x["id"]) || continue
        pitch = get(x["camera"], "pitch", nothing)
        x["camera"] = _v8_grid_identity(grids[x["id"]], u)
        pitch === nothing || (x["camera"]["pitch"] = pitch)
    end
    return _hash_v09(tree)
end
# A grid sensor's camera as the identity tree of rig v5–v8 held it, from its old
# body (v8's `content_id` docstring: frozen here, as v8 minted it): the extent,
# calibrated area, grid, central or non-central, the held nodes' directions as
# stored and points as their persisted metre values, and the held-node runs, the
# nominal pinhole and the domain iff the body has them. (Its pitch is 0.9's
# tree's, unchanged.)
function _v8_grid_identity(b::AbstractDict, u::Units)
    d = Dict{String, Any}(
        "extent" => Any[Int(b["width"]), Int(b["height"])],
        "calibrated_area" => Any[Int(b[k]) for k in ("calibration_min_x", "calibration_min_y", "calibration_max_x", "calibration_max_y")],
        "grid" => Any[b["grid_width"], b["grid_height"]], "grid_model" => b["type"] == _OLD_CENTRAL_TYPE ? "central" : "noncentral",
        "directions" => Any[Float64[Float64(c) for c in v] for v in b["direction_grid"]],
    )
    haskey(b, "point_grid") && (d["points"] = Any[Float64[_persisted_v09(_read_power(c, u, 1)) for c in v] for v in b["point_grid"]])
    haskey(b, "present") && (d["present"] = _lower(b["present"]))
    haskey(b, _OLD_NOMINAL_KEY) && (d[_OLD_NOMINAL_KEY] = Any[Float64(x) for x in b[_OLD_NOMINAL_KEY]])
    haskey(b, "domain") && (d["domain"] = Dict{String, Any}("ellipse" => Any[Float64(x) for x in b["domain"]["ellipse"]]))
    return d
end
# which of a v2 calibration record's parameters are lengths (its frozen layout)
_v2_length_mask(r) = (t = r["type"]; t == "affine" ? Bool[0, 0, 0, 0, 1, 1] : t == "calibration_chain" ?
    reduce(vcat, [_v2_length_mask(x) for x in r["members"]]; init = Bool[]) : fill(true, _v2_parameter_count(r)))
function _v5_identity_calibration(r::AbstractDict, u::Units)
    P(x) = _persisted_v09(ustrip(m, read_length(x::Real, u)))
    out = Dict{String, Any}("kind" => r["type"])
    haskey(r, "members") && (out["members"] = Any[_v5_identity_calibration(x, u) for x in r["members"]])
    haskey(r, "nodes") && (out["nodes"] = Any[Any[P(p[1]), P(p[2])] for p in r["nodes"]])
    if haskey(r, "parameters")
        mask = _v2_length_mask(r)
        out["parameters"] = Any[mask[i] ? P(v) : Float64(v) for (i, v) in enumerate(r["parameters"])]
    end
    if haskey(r, "covariance")
        dims = _v2_length_mask(r)
        out["covariance"] = _cov_wire((x, n) -> _persisted_v09(x, n), _length_cov_from_json(r["covariance"], length(dims), u, dims), dims)
    end
    return out
end

# The content identities a vgio/rig/v4 rig had. v4 hashed the record without
# its provenance (by 0.9's canonical JSON): `content_id(path)` the record as the
# file holds it (its parsed tree, hashed here) and `content_id(rig)` the record
# of the value in MILLIMETRES, which differs for a rig whose file is in another
# unit and is what a producer that named the rig by its value put in its
# observation files. Both are mapped.
_rig_v4_id(old::AbstractDict) = _hash_v09(Dict{String, Any}(k => v for (k, v) in old if k != "provenance"))

# v4's in-memory value held millimetre numbers, converted from the file's unit
# by FlexUnits at load — `ustrip(mm, quantity(x, unit))` per length, a
# covariance scaled by mm-per-unit to its power — and `content_id(rig)` hashed
# them as written. That arithmetic is reproduced here, leaf for leaf, from the
# file's own numbers, for every unit (mm included: v4 converted an mm file
# through the same arithmetic): the decimal-exact pair (`read_length`/
# `write_length`) gives another millimetre number for about a quarter of
# 6-digit metre values, and so another identity. v4's value also differed
# from the file where v4's reader changed a number: a quaternion was
# canonicalized to w ≥ 0 (as the 0.10 step stores it), and an old grid model's
# direction controls were renormalized unless unit to 1e-12
# (VisualGeometryToolkitCore 0.1's `_unit_or_normalize`, which the current
# reader no longer applies: `_v4_direction`). The rest of the record (order,
# rotations, cameras) is the value's millimetre record in v4's form (IO 0.9's
# record, `_rig_record_v09`, taken back through the v6 → v4 renames), which
# v4's writer agrees with.
_v4_mm(x::Real, u::Units) = Float64(ustrip(mm, quantity(Float64(x), u)))
_v4_mm_points(v, u) = Any[Any[_v4_mm(c, u) for c in p] for p in v]
function _rig_v4_id_mm(value, old::AbstractDict)
    rec = _rig_record_v09(value, (x, n) -> x, false)
    merge!(rec, Dict{String, Any}("length_unit" => "mm", "rotation_encoding" => _ROTATION_ENCODING, "transform_convention" => _TRANSFORM_CONVENTION))
    haskey(rec, "sensors") && (rec["pixel_frame"] = pixel_frame_name(CANONICAL_PIXEL_FRAME))
    rec = _rig_v6_to_v4(rec, old)
    u = parse_unit(old["length_unit"]::String)
    s = _v4_mm(1.0, u)                                                    # mm per unit, as v4 scaled a covariance
    for (key, fix!) in (("sensors", _v4_mm_sensor!), ("targets", _v4_mm_member!))
        haskey(rec, key) || continue
        olds = Dict(x["id"] => x for x in old[key])
        for x in rec[key]
            fix!(x, olds[x["id"]], u, s)
        end
    end
    return _rig_v4_id(rec)
end
function _v4_mm_pose!(p, o, u, s)
    p["translation"] = Any[_v4_mm(c, u) for c in o["translation"]]
    if haskey(o, "covariance")
        d = (1.0, 1.0, 1.0, s, s, s)
        p["covariance"] = Any[Any[Float64(o["covariance"][i][j]) * (d[i] * d[j]) for j in 1:6] for i in 1:6]
    end
    return nothing
end
# A grid sensor's camera is v4's grid body, which the current writer no longer
# writes (the spline camera's body replaced it): v4's writer emitted the body as
# the file holds it, but for the numbers its reader changed — the points in
# millimetres, the directions renormalized.
function _v4_mm_sensor!(x, o, u, s)
    _v4_mm_pose!(x["extrinsics"], o["extrinsics"], u, s)
    haskey(o["camera"], "type") || return nothing
    c = x["camera"] = Dict{String, Any}(o["camera"])
    haskey(c, "point_grid") && (c["point_grid"] = _v4_mm_points(c["point_grid"], u))
    c["direction_grid"] = Any[collect(_v4_direction(d)) for d in c["direction_grid"]]
    return nothing
end
# A direction control as VisualGeometryToolkitCore 0.1 held it: kept when unit
# to 1e-12, else normalized (its `_unit_or_normalize`).
_v4_direction(d) = (v = Vec3{Float64}(Float64.(d)...); abs(norm(v) - 1) < 1.0e-12 ? v : normalize(v))
function _v4_mm_member!(x, o, u, s)
    _v4_mm_pose!(x["pose"], o["pose"], u, s)
    t, ot = x["target"], o["target"]
    for k in ("points", "nominal")
        haskey(ot, k) && (t[k] = _v4_mm_points(ot[k], u))
    end
    haskey(ot, "point_covariances") &&
        (t["point_covariances"] = Any[Any[Any[Float64(c) * s^2 for c in r] for r in Σ] for Σ in ot["point_covariances"]])
    return nothing
end

# =============================================================================
# The files naming target definitions, re-pointed — the one re-id of IO 0.10
# (`vgio_migrate_tree`, the command's `repoint`)
# =============================================================================
#
# An observation file (`vgio/observations`, and IO 0.9's `target_points2d`,
# `target_points3d`, `target_detections`) and a `target_homographies` file name
# the rigs and single targets they resolve in by their `content_id`. IO 0.10
# re-minted every identity ONCE (JCS over the 0.10 identity tree, lengths in
# metres; before it, a rig v4 file's identity hashed its record, and a few
# v5–v8 ones their retired shapes), so the files naming a definition follow it
# when a tree is rewritten: each definition's old identities (`_old_rig_ids`,
# `_old_target_ids`; IO 0.9's `_v09_id` among them) map to its new one. Nothing
# else re-points.

# The value type of a kind that names target definitions by identity, any
# version that reads (`nothing`: a kind that names none).
function _target_naming_type(s)
    v = _split_schema(s)
    v === nothing && return nothing
    k = _current_kind(v.kind)
    (k == "observations" && (v.kind != "observations" || v.major >= 3)) && return TargetObservations   # (observations/v2: retired)
    k == "target_homographies" && return TargetHomographies
    return nothing
end
# What a `targets` identity may name in a tree: a current rig or single target file.
_is_target_definition(s) = (v = _split_schema(s); v !== nothing && v.kind in ("rig", "target") && _is_current_major(s, v.kind))

# Every observed (frame, target, point index) of a value; a homography names a target, no point.
_observed_points(obs::TargetObservations) = ((f, id, i) for (f, sets) in obs for (id, cs) in sets for i in sources(cs))
_observed_points(th::TargetHomographies) = ((f, id, nothing) for (f, hs) in th for id in keys(hs))
# The value with other `targets` identities, everything else as it is.
_with_targets(x::TargetObservations{F}, targets) where {F} = TargetObservations{F}(x.frames; stamps = x.stamps, targets)
_with_targets(x::TargetHomographies{F}, targets) where {F} =
    TargetHomographies{F}(x.frames; images = x.images, sensors = x.sensors, times = x.times, time_origin = x.time_origin, targets)
# A homography file states no length unit of its own (`_resave_kwargs`).
_writes_length(::TargetObservations) = true
_writes_length(::TargetHomographies) = false

# Re-point a file naming target definitions (`_target_naming_type`, any
# version that reads) at the definitions its identities were migrated to: every
# identity in its `targets` that `ids` (old → new, the definitions' migrations'
# `old_ids` and `new_id`) maps is replaced; nothing else changes, and the file
# is written as its kind's current version (an IO 0.9 target_points2d file as
# `vgio/observations`). Every identity the file names must be mapped or be the
# `content_id` of one of `rigs` (rigs, current rig or target paths; `hashes`
# their identities, when the caller has them): any other is an `ArgumentError`
# naming the file and the identity, and nothing is written — a stale identity
# is never kept silently. The guarded pattern: written by the current writer
# beside `out`, read back, compared with the original value for value (only the
# mapped identities differ), every observation's (target, index) checked to
# resolve in the definitions its identities name — an unresolved (referenced)
# target's points need `resolve`, its owner's resolver; without it, such a
# target is checked by id alone and counted in `unchecked`. Only then is it
# renamed to `out` (`out == path` replaces it in place). `old` is the file's
# parsed tree, when the caller has it. Returns `(; path, out, to, repointed, unchecked)`.
function _repoint_targets(
        path::AbstractString, ids::AbstractDict, rigs; resolve = nothing,
        out::AbstractString = _migrated_path(path), dry_run::Bool = false,
        hashes = _target_hashes(_target_list(rigs)), old::AbstractDict = _raw_tree(path), before_rename = r -> nothing,
    )
    s = get(old, "schema", nothing)
    T = _target_naming_type(s)
    T === nothing && throw(ArgumentError(
        "$(path): a $(repr(s)) file names no target definitions to re-point (observations, target_homographies, and IO 0.9's target_points2d, target_points3d, target_detections)"))
    in_place = abspath(out) == abspath(path)
    (dry_run || in_place || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    obs = load(T, path)
    map_ = Dict{String, String}(String(k) => String(v) for (k, v) in ids)
    rig_list = _target_list(rigs)
    for t in obs.targets
        new = get(map_, t, t)
        new in hashes || throw(
            ArgumentError(
                "$(path): it names targets identity $(t)" * (haskey(map_, t) ? ", re-pointed to $(new)," : "") *
                    " which is neither re-pointed nor the content_id of any rig or target given; nothing was written"
            )
        )
    end
    targets = String[get(map_, t, t) for t in obs.targets]
    repointed = count(t -> haskey(map_, t), obs.targets)
    named = [r for (r, h) in zip(rig_list, hashes) if h in targets]   # the definitions the observations resolve in
    # (a file that names no definitions — 3D observations of an unknown constellation — resolves in none)
    unchecked = isempty(targets) ? 0 : _check_resolves_by_id(obs, _target_rigs(named; resolve), path)
    current = migrate_record(old)
    return write_staged(out => basename(path); dry_run) do staged
        save(staged, _with_targets(obs, targets); _resave_kwargs(current; length = _writes_length(obs))...)
        re = load(T, staged)
        _verify(re.targets == targets, path, "its targets")
        _verify(all(f -> f === :targets || same_value(getfield(re, f), getfield(obs, f)), fieldnames(typeof(obs))) &&
            _same_file_extras(_raw_tree(staged), current), path, "the observations")
        r = (; path = String(path), out = dry_run ? nothing : String(out), to = String(current["schema"]), repointed, unchecked)
        before_rename(r)
        r
    end
end

# Every observation's (target, index) resolves in the definitions: by its
# points where the target is resolved, by its id alone where it is a reference
# without its points (counted); a homography by its target's id.
function _check_resolves_by_id(obs, rigs, path)
    npts = Dict{Symbol, Union{Nothing, Int}}()
    for rig in rigs, id in ids(rig)
        t = rig[id].target
        npts[id] = is_resolved(t) ? length(points(t)) : nothing
    end
    unchecked = 0
    for (f, target, index) in _observed_points(obs)
        haskey(npts, target) || throw(ArgumentError("$(path): frame $(f) observes target $(target), which none of the definitions given holds"))
        index === nothing && continue
        n = npts[target]
        if n === nothing
            unchecked += 1
        else
            1 <= index <= n || throw(ArgumentError("$(path): frame $(f) observes point $(index) of target $(target), which has $(n)"))
        end
    end
    return unchecked
end

# =============================================================================
# A directory tree: every migratable file, then the observation files re-pointed
# =============================================================================

"""
    vgio_migrate_tree(dir; in_place = false, dry_run = false, io = stdout) -> Vector{NamedTuple}

Migrate every `.json` file under `dir` of an earlier schema version whose
schema has a [`migrate_schema`](@ref) file form (every vgio kind's earlier
versions that a published file holds, and any schema whose owner added one),
then re-point the files naming target definitions — `vgio/observations`
(IO 0.9's `target_points2d`, `target_points3d`, `target_detections` among them)
and `vgio/target_homographies` — at the current rigs and single targets of the
tree, rewriting them as their kind's current version; each by the guarded
pattern of [`vgio_migrate`](@ref) (a full-precision length may be rewritten
within 2 ulp, as the same metre value), with one line per file to `io` — a
file left as it is says why (`current`; `SKIPPED`: no schema, or a schema
nothing here migrates, another owner's record whose owner is not loaded among
them). Written beside each original as `<stem>.migrated.json`, or over it with
`in_place`; an old grid file is never rewritten, in place or not: its current
record goes to a new `<stem>.camera.json` (central) or `<stem>.noncentral.json`
(non-central), and a rerun finds it current.

THE RE-ID. IO 0.10 re-minted every rig's and target's `content_id` once (JCS
over the 0.10 identity tree), and earlier versions' identities differed too (a
rig v4 file's hashed its record). Each migrated rig or target file's old
identities — the ones its own version minted, and IO 0.9's identity of its
value — map to its new one. The map is PERSISTED in `dir/vgio_migrate_ids.json`
(`{"identities": {old: new, …}, "definitions": {new: path relative to dir, …},
"rewritten": {old path: {"path": its current file, "sha256": the old bytes'},
…}}`, paths relative to `dir`) before a file is put in place, and read by a
later run, so a run that
fails after its definitions were migrated in place re-points correctly when
rerun, and an observation file kept outside the tree can be re-pointed later
from the map (the command's `repoint`); a rerun of a migrated tree is idempotent
(every file current, logged so).

A calibration manifest of the tree (`vgio/calibration`) is migrated last: each
listed file the run rewrote is bound again — its `path`, `sha256` and `schema`
its current file's, every `derived_from` naming it renamed with it — and the
manifest is written as the current version and checked as
[`load_manifest`](@ref) checks it before it is put in place; a current
manifest none of whose files moved is left as it is.

A file the migration refuses — its migration
does not read back, its `<stem>.migrated.json` exists and is not current, an
observation file names an identity that is neither mapped nor a current
definition's of the tree, or an observation does not resolve in its
definitions — is logged `FAILED` with the reason, nothing is written for it,
and the run goes on with the other files; it then ends in ONE `ArgumentError`
listing every refusal.
"""
function vgio_migrate_tree(dir::AbstractString; in_place::Bool = false, dry_run::Bool = false, io::IO = stdout)
    idsfile = joinpath(dir, _IDS_SIDECAR)
    files = _tree_json(dir; io)
    schemas = Dict(f => _try_schema(f) for f in files)
    results = NamedTuple[]
    m = _TreeMap(idsfile, _read_ids(idsfile)..., _read_rewritten(idsfile), dry_run)
    ids = m.ids
    rigs = String[]                                                    # the current target definitions of the tree (rigs, single targets)
    failed = Pair{String, String}[]                                    # file => why it was refused
    for f in files
        s = schemas[f]
        if s === nothing
            println(io, "SKIPPED   ", f, ": no readable schema")
            continue
        elseif _target_naming_type(s) !== nothing || _is_manifest(s)
            continue                                                   # re-pointed below; a manifest last
        elseif _is_current(s)
            _is_target_definition(s) && push!(rigs, f)
            println(io, "current   ", f, ": ", s)
            continue
        elseif !_has_migration(s)
            println(io, "SKIPPED   ", f, ": ", s, " has no migration (a retired version, or a schema no package loaded migrates)")
            continue
        end
        out = in_place && !(s in _OLD_GRID_SCHEMAS) ? f : _default_out(f)   # an old grid file is never rewritten
        so = isfile(out) ? _try_schema(out) : nothing
        if (!in_place || s in _OLD_GRID_SCHEMAS) && so isa String && _is_current(so)   # an old grid's output is never in place
            println(io, "current   ", f, ": already migrated to ", out)
            continue
        end
        # verified, its identities mapped (and persisted before it is put in place)
        r = _refused(failed, f, io) do
            _vgio_migrate(f; out, dry_run, before_rename = r -> _map_ids!(m, r, dir, f, out))
        end
        r === nothing && continue
        (!dry_run && _is_target_definition(r.to)) && push!(rigs, out)
        push!(results, r)
        println(io, dry_run ? "verified  " : "migrated  ", f, ": ", s, " → ", r.to, out == f ? "" : " in $(out)",
            haskey(r, :old_ids) ? "  (content id $(join(r.old_ids, ", ")) → $(r.new_id))" : "")
    end
    sort!(unique!(rigs))
    hashes = String[content_id(r) for r in rigs]
    current = Set(hashes)
    for f in files
        s = schemas[f]
        (s !== nothing && _target_naming_type(s) !== nothing) || continue
        old = _raw_tree(f)
        ts = String[t for t in something(get(old, "targets", nothing), Any[])]
        if _is_current(s) && all(in(current), ts)
            println(io, "current   ", f, ": ", isempty(ts) ? "names no targets" : "names current definitions")
            continue
        end
        stale = [t for t in ts if !(t in current) && !(haskey(ids, t) && (dry_run || ids[t] in current))]
        if !isempty(stale)
            msg = "names targets identit$(length(stale) == 1 ? "y" : "ies") $(join(stale, ", ")), neither re-pointed nor a current definition's"
            println(io, "FAILED    ", f, ": ", msg)
            push!(failed, f => msg)
            continue
        end
        out = in_place ? f : _migrated_path(f)
        if !in_place && isfile(out) && (so = _try_schema(out); so isa String && _is_current(so))
            println(io, "current   ", f, ": already re-pointed to ", out)
            continue
        end
        if dry_run
            println(io, "would re-point  ", f, isempty(ts) ? " (names no targets: rewritten as $(_schema(_current_kind(_split_schema(s).kind))))" : " (once the definitions are written)")
            continue
        end
        r = _refused(() -> _repoint_targets(f, ids, rigs; out, hashes, old, before_rename = r -> _map_ids!(m, r, dir, f, out)), failed, f, io)
        r === nothing && continue
        push!(results, r)
        println(io, "re-pointed  ", f, ": ", s, " → ", r.to, out == f ? "" : " in $(out)", ", ", r.repointed, " targets identit$(r.repointed == 1 ? "y" : "ies")",
            r.unchecked > 0 ? " ($(r.unchecked) observations of referenced targets checked by id)" : "")
    end
    # the manifests last: each binds files this run (or an interrupted one) rewrote
    for f in files
        s = schemas[f]
        (s !== nothing && _is_manifest(s)) || continue
        out = in_place ? f : _migrated_path(f)
        r = _refused(() -> _migrate_tree_manifest(f, s, dir, m; out, dry_run, io), failed, f, io)
        r === nothing || push!(results, r)
    end
    isempty(failed) || throw(
        ArgumentError(
            "vgio_migrate_tree($(dir)): $(length(failed)) file(s) refused, nothing written for them:\n" *
                join(("  $(f): $(msg)" for (f, msg) in failed), "\n")
        )
    )
    return results
end

# One file's migration or re-pointing in a tree, `work()`, its refusal (an
# `ArgumentError`: a read-back that differs, an existing destination, an
# observation that does not resolve) logged FAILED and collected in `failed`,
# the tree going on: `nothing` then.
function _refused(work, failed::AbstractVector, f::AbstractString, io::IO)
    try
        return work()
    catch e
        e isa ArgumentError || rethrow()
        println(io, "FAILED    ", f, ": ", e.msg)
        push!(failed, f => e.msg)
        return nothing
    end
end

# The tree's persisted map (`vgio_migrate_ids.json`): a migrated definition's
# old → new identities, its new identity → its path, and every file the run
# rewrote → where its current file is and the SHA-256 of the bytes it held
# before (what a manifest that binds it recorded) — persisted, unless a dry
# run, before the file is put in place.
struct _TreeMap
    file::String
    ids::Dict{String, String}
    defs::Dict{String, String}
    rewritten::Dict{String, Dict{String, String}}
    dry_run::Bool
end
function _map_ids!(m::_TreeMap, r, dir::AbstractString, f::AbstractString, out::AbstractString)
    if haskey(r, :old_ids)
        for o in r.old_ids
            m.ids[o] = r.new_id
        end
        m.defs[r.new_id] = relpath(out, dir)
    end
    m.rewritten[relpath(f, dir)] = Dict{String, String}("path" => relpath(out, dir), "sha256" => open(sha256_hex, f))
    m.dry_run || _write_ids(m.file, m.ids, m.defs, m.rewritten)
    return nothing
end

_is_manifest(s) = (v = _split_schema(s); v !== nothing && v.kind == "calibration")
# One listed file of a manifest in `mdir`, bound again if the tree rewrote it
# (marked `_rebound`; a renamed path recorded in `renamed`), else as it stands.
function _rebound_entry(e::AbstractDict, m::_TreeMap, dir::AbstractString, mdir::AbstractString, renamed::AbstractDict, dry_run::Bool)
    w = get(m.rewritten, relpath(normpath(joinpath(mdir, e["path"])), dir), nothing)
    (w === nothing || w["sha256"] != e["sha256"]) && return Dict{String, Any}(e)
    new = joinpath(dir, w["path"])
    np = relpath(new, mdir)
    np == e["path"] || (renamed[e["path"]] = np)
    n = merge(Dict{String, Any}(e), Dict{String, Any}("path" => np, "_rebound" => true))
    dry_run || merge!(n, Dict{String, Any}("sha256" => open(sha256_hex, new), "schema" => _peek_schema(new)))
    return n
end
# A manifest of the tree (`vgio/calibration`, any version that reads), after
# the tree's files: each listed file this migration rewrote (the map's
# `rewritten`, its recorded SHA-256 the manifest's) bound again — its `path` (a
# file written beside, an old grid's new camera store), `sha256` and `schema`
# its current file's, every `derived_from` naming it renamed with it — and the
# manifest written as the current version (in place, or beside as
# `<stem>.migrated.json`) by the guarded pattern: written by its writer, read
# back, and checked as `load_manifest` checks it (every listed file's hash and
# schema, its owner's reader, the observations' targets) before it is put in
# place. A current manifest none of whose files moved is left as it is.
function _migrate_tree_manifest(f::AbstractString, s, dir::AbstractString, m::_TreeMap; out, dry_run::Bool, io::IO)
    if out != f && isfile(out) && (so = _try_schema(out); so isa String && _is_current(so))
        println(io, "current   ", f, ": already migrated to ", out)
        return nothing
    end
    t = migrate_record(_raw_tree(f))
    mdir = dirname(f)
    renamed = Dict{String, String}()
    t["files"] = Any[_rebound_entry(e, m, dir, mdir, renamed, dry_run) for e in t["files"]]
    moved = count(e -> haskey(e, "_rebound"), t["files"])
    foreach(e -> delete!(e, "_rebound"), t["files"])
    for e in t["files"]
        e["derived_from"] = Any[get(renamed, p, p) for p in e["derived_from"]]
    end
    if _is_current(s) && moved == 0
        println(io, "current   ", f, ": ", s, " (binds no rewritten file)")
        return nothing
    end
    if dry_run
        println(io, "would rebind  ", f, ": ", s, " → ", _schema("calibration"), ", ", moved, " rewritten file(s)")
        return nothing
    end
    (out == f || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    r = write_staged(out => basename(f); dry_run = false) do staged
        _write_manifest(staged, t)
        _check_manifest_files(mdir, _read_store(staged, "calibration"), staged)
        (; path = String(f), out = String(out), from = String(s), to = _schema("calibration"), rebound = moved)
    end
    println(io, "rebound   ", f, ": ", s, " → ", r.to, out == f ? "" : " in $(out)", ", ", moved, " rewritten file(s)")
    return r
end

# The `.json` files of a tree, sorted, without the identity map and without
# what is under a staging directory (`.vgio_migrate_*`, `write_staged`'s): one left
# by an interrupted migration is logged, never walked.
function _tree_json(dir; io::IO = stdout)
    out = String[]
    for (r, dirs, fs) in walkdir(dir)
        staged = any(c -> startswith(c, _STAGE_PREFIX), splitpath(relpath(r, dir)))
        staged && continue                                          # inside a staging directory
        for d in dirs
            startswith(d, _STAGE_PREFIX) &&
                println(io, "SKIPPED   ", joinpath(r, d), ": a staging directory left by an interrupted migration (remove it)")
        end
        append!(out, joinpath(r, f) for f in fs if endswith(f, ".json") && !(r == dir && f == _IDS_SIDECAR))
    end
    return sort!(out)
end

# The persisted map of a tree's migration (the sidecar's format, interfaces.md
# "io10-migrate"): `{"identities": {old: new, …}, "definitions": {new: path
# relative to the tree, …}, "rewritten": {path: {"path": its current file's,
# "sha256": its bytes' before}, …}}`, its keys sorted; a sidecar written by IO
# 0.9's migration holds `identities` only.
const _IDS_SIDECAR = "vgio_migrate_ids.json"
_str_dict(x) = Dict{String, String}(String(k) => String(v) for (k, v) in x)
function _read_ids(p)
    isfile(p) || return Dict{String, String}(), Dict{String, String}()
    d = _read_json(p)
    return _str_dict(d["identities"]), _str_dict(get(d, "definitions", Dict{String, String}()))
end
_read_rewritten(p) = isfile(p) ?
    Dict{String, Dict{String, String}}(String(k) => _str_dict(v) for (k, v) in get(_read_json(p), "rewritten", Dict{String, Any}())) :
    Dict{String, Dict{String, String}}()
function _write_ids(p, ids::AbstractDict, defs::AbstractDict, rewritten::AbstractDict)
    t = tempname(dirname(abspath(p)))
    _write_json(t, Dict("identities" => Dict(sort!(collect(ids); by = first)), "definitions" => Dict(sort!(collect(defs); by = first)),
                        "rewritten" => Dict(sort!(collect(rewritten); by = first))))
    Base.Filesystem.rename(t, p)
    return nothing
end
# Whether a schema is its kind's current version (any minor).
_is_current(s::AbstractString) = (v = _split_schema(s); v !== nothing && haskey(_SCHEMA_VERSION, v.kind) && _is_current_major(s, v.kind))
_try_schema(f) = try
    s = _peek_schema(f)
    s isa String ? s : nothing
catch
    nothing
end

# `--images` for the command line: a directory's image files, sorted by name
# (the join a positional export assumed, now made explicit), or a text file
# naming one image per line.
const _IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
function _image_names(src::AbstractString)
    isdir(src) && return sort!([f for f in readdir(src) if lowercase(splitext(f)[2]) in _IMAGE_EXTENSIONS])
    isfile(src) || throw(ArgumentError("--images $(src): no such directory or file"))
    return [String(strip(l)) for l in eachline(src) if !isempty(strip(l))]
end

const _CLI_EXTERNAL_USAGE = """
usage: julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl opencv|matlab [options] FILE

Convert a third-party OpenCV or MATLAB calibration (calibration_opencv_*.json,
calibration_matlab_*.json) into native files (<stem>_cameras.json,
<stem>_poses.json). A DRY RUN unless --apply: converted to a temporary
directory and verified, nothing written. The original is never touched.

  --images DIR|LIST      the image of every pose, in the export's order: a
                         directory's image files sorted by name, or a text
                         file with one name per line (REQUIRED)
  --image-size WxH       the images' width and height in pixels (REQUIRED)
  --file-unit U          the tvec unit (m, mm, …) when the file declares none
  --file-frame F         the cx, cy convention; opencv (OpenCV) or julia (MATLAB)
                         unless VERIFIED otherwise
  --target NAME          the pattern's name (default calibration_target)
  --out DIR              where to write (default: beside FILE)
  --apply                write the files
"""

# The command line's options: each of `valued` takes the next argument (every
# occurrence kept, in order), each of `switches` none, and any other `--…` is
# unknown. Returns `(opts, positional)` — an option given maps to its values
# (a switch's: one "" per occurrence) — or the exit code, the usage printed:
# 0 for `-h`/`--help`, 2 for a usage error.
function _cli_options(args, valued, switches, usage::AbstractString, io::IO)
    opts, positional = Dict{String, Vector{String}}(), String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in switches
            push!(get!(opts, a, String[]), "")
        elseif a in valued
            i < length(args) || (print(io, usage); return 2)
            push!(get!(opts, a, String[]), String(args[i += 1]))
        elseif a in ("-h", "--help")
            print(io, usage)
            return 0
        elseif startswith(a, "--")
            println(io, "unknown option $(a)")
            print(io, usage)
            return 2
        else
            push!(positional, String(a))
        end
        i += 1
    end
    return opts, positional
end

function _migrate_external_cli(kind::AbstractString, args::AbstractVector{<:AbstractString}; io::IO = stdout)
    parsed = _cli_options(args, ("--images", "--image-size", "--file-unit", "--file-frame", "--target", "--out"), ("--apply",), _CLI_EXTERNAL_USAGE, io)
    parsed isa Int && return parsed
    all_opts, files = parsed
    opts = Dict(k => last(v) for (k, v) in all_opts)           # a repeated option: its last value
    apply = haskey(opts, "--apply")
    (length(files) == 1 && haskey(opts, "--images") && haskey(opts, "--image-size")) || (print(io, _CLI_EXTERNAL_USAGE); return 2)
    file = only(files)
    m = match(r"^(\d+)x(\d+)$", opts["--image-size"])
    m === nothing && (println(io, "--image-size is WIDTHxHEIGHT, got $(opts["--image-size"])"); return 2)
    try
        migrate = kind == "opencv" ? convert_opencv_calibration : convert_matlab_calibration
        r = migrate(
            pixel_frame_from_name(get(opts, "--file-frame", kind == "opencv" ? "opencv" : "julia")), file;
            images = _image_names(opts["--images"]),
            image_size = (parse(Int, m[1]::AbstractString), parse(Int, m[2]::AbstractString)),
            file_unit = haskey(opts, "--file-unit") ? parse_unit(opts["--file-unit"]) : nothing,
            target = get(opts, "--target", "calibration_target"),
            out = get(opts, "--out", dirname(abspath(file))), dry_run = !apply,
        )
        println(
            io, apply ? "migrated  " : "verified  ", file, ": ", r.n_posed, " posed, ", r.n_unposed, " unposed",
            apply ? "  → $(r.cameras), $(r.poses)" : " (dry run; --apply to write)"
        )
        return 0
    catch e
        e isa ArgumentError || rethrow()
        println(io, "FAILED    ", file, ": ", e.msg)
        return 1
    end
end


const _CLI_VERSION_USAGE = """
usage: julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl target|rig|file [options] FILE
       julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl tree [options] DIR
       julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl repoint [options] FILE

Migrate a file of an earlier schema version into its kind's current one
(target: vgio/target v1–v5 → v6.0; rig: vgio/rig v4–v9 → v10.0; file: any
schema with a migration — cameras, capture settings, the observation kinds,
the manifest, the radiometric kinds, a units sidecar among them), every such
file under a directory (tree: then re-pointing the observation and homography
files at the re-identified rigs and targets, the identity map persisted in
DIR/vgio_migrate_ids.json), or re-point one observation or homography file
(repoint). A DRY RUN unless --apply: migrated to a temporary directory, read
back and compared number for number, nothing written. Written beside the
original as <stem>.migrated.json, or over it with --in-place (only once
verified).

  --out FILE             where to write (target|rig|file|repoint)
  --in-place             replace the original once verified
  --id OLD=NEW           repoint: a targets identity and its replacement (repeatable)
  --ids MAP              repoint: a tree's vgio_migrate_ids.json, its identities
                         (and, without --rig, its definitions as the rigs)
  --rig PATH             repoint: a current rig or target the new identities name (repeatable)
  --apply                write
"""

function _migrate_version_cli(kind::AbstractString, args::AbstractVector{<:AbstractString}; io::IO = stdout)
    parsed = _cli_options(args, ("--out", "--id", "--ids", "--rig"), ("--apply", "--in-place"), _CLI_VERSION_USAGE, io)
    parsed isa Int && return parsed
    opts, files = parsed
    apply, in_place = haskey(opts, "--apply"), haskey(opts, "--in-place")
    out = haskey(opts, "--out") ? last(opts["--out"]) : nothing
    rigs = get(opts, "--rig", String[])
    ids = Dict{String, String}()
    for m in get(opts, "--ids", String[])
        isfile(m) || (println(io, "--ids $(m): no such file"); return 2)
        mids, defs = _read_ids(m)
        merge!(ids, mids)
        haskey(opts, "--rig") || append!(rigs, joinpath(dirname(abspath(m)), r) for r in values(defs))
    end
    for v in get(opts, "--id", String[])
        kv = split(v, '='; limit = 2)
        length(kv) == 2 || (println(io, "--id is OLD=NEW, got $(v)"); return 2)
        ids[String(kv[1])] = String(kv[2])
    end
    (length(files) == 1 && !(in_place && out !== nothing)) || (print(io, _CLI_VERSION_USAGE); return 2)
    file = only(files)
    dest = in_place ? file : something(out, _migrated_path(file))
    try
        if kind == "tree"
            vgio_migrate_tree(file; in_place, dry_run = !apply, io)
            apply || println(io, "(dry run; --apply to write)")
            return 0
        elseif kind == "repoint"
            (isempty(ids) || isempty(rigs)) && (print(io, _CLI_VERSION_USAGE); return 2)
            r = _repoint_targets(file, ids, rigs; out = dest, dry_run = !apply)
            println(io, apply ? "re-pointed  " : "verified  ", file, ": ", r.repointed, " targets identities",
                apply ? "  → $(r.out)" : " (dry run; --apply to write)")
            return 0
        end
        s = _peek_schema(file)
        # the command's kind (`target`, `rig`) migrates the earlier versions of that kind only (`file`: any)
        v = _split_schema(s)
        (kind == "file" || (v !== nothing && _current_kind(v.kind) == kind)) || throw(
            ArgumentError("$(file) is a $(repr(s)) file, not a $(kind) one (`file` migrates any)")
        )
        r = vgio_migrate(file; out = dest, dry_run = !apply)
        println(
            io, apply ? "migrated  " : "verified  ", file, ": ", r.from, " → ", r.to,
            haskey(r, :old_id) ? "  (content id $(r.old_id) → $(r.new_id))" : "",
            apply ? "  → $(r.out)" : " (dry run; --apply to write)"
        )
        return 0
    catch e
        e isa ArgumentError || rethrow()
        println(io, "FAILED    ", file, ": ", e.msg)
        return 1
    end
end

"""
    _migrate_cli(args; io = stdout) -> Int

The `bin/vgio_migrate.jl` command: `opencv|matlab [options] FILE` (a
third-party calibration), `target|rig|file [options] FILE`, `tree [options]
DIR` and `repoint [options] FILE` (earlier schema versions), each a dry run
unless `--apply`. Returns the process exit code: 0 on success, 1 when the
conversion failed, 2 on a usage error.
"""
function _migrate_cli(args::AbstractVector{<:AbstractString}; io::IO = stdout)
    (!isempty(args) && first(args) in ("opencv", "matlab")) && return _migrate_external_cli(first(args), args[2:end]; io)
    (!isempty(args) && first(args) in ("target", "rig", "file", "tree", "repoint")) && return _migrate_version_cli(first(args), args[2:end]; io)
    (!isempty(args) && first(args) in ("-h", "--help")) && (print(io, _CLI_EXTERNAL_USAGE, "\n", _CLI_VERSION_USAGE); return 0)
    print(io, _CLI_EXTERNAL_USAGE, "\n", _CLI_VERSION_USAGE)
    return 2
end
