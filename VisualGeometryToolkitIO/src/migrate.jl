# =============================================================================
# Migrations — third-party calibrations into the native kinds (migrate_opencv,
# migrate_matlab), earlier schema versions into the current ones (vgio_migrate,
# migrate_schema; a migrated rig's measurement files re-pointed) — and the
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
_same_length(a, b) = (x = ustrip(m, a); y = ustrip(m, b); abs(x - y) <= 2 * eps(abs(x)))
_same_pose(a::EuclideanMap, b::EuclideanMap) =
    all(map(_same_length, a.t, b.t)) && maximum(abs, Matrix(a.R) - Matrix(b.R)) <= 8 * eps()

# =============================================================================
# migrate_opencv — a third-party OpenCV calibration into the native kinds
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
    migrate_opencv([file_frame = PixelFrame{:opencv},] path; images, image_size,
                   file_unit = nothing, target = "calibration_target",
                   out = dirname(path), dry_run = false) -> NamedTuple

Convert a third-party OpenCV calibration export (`calibration_opencv_*.json`:
`fx, fy, cx, cy`, named distortion coefficients, `rms`, one `{rvec, tvec}` per
image) into native files (cameras v3, transforms v2), where everything the export leaves implicit is
declared. The original is never touched. Written to `out`:

- `<stem>_cameras.json` — a `vgio/cameras/v3` store holding one camera named
  `<stem>`: `width`/`height` from `image_size`, `K` in the canonical pixel
  frame, and the lens record of `opencv_lens` for the export's distortion
  vector (the terms its length evaluates). The export's reprojection
  statistic is its `provenance` (`reprojection_error`, `error_statistic =
  "rms"`, `source = "opencv"`, `migrated_from`).
- `<stem>_poses.json` — a `vgio/transforms/v2` file, one record per image the
  solver posed, NAMED by that image: `from = "target:<target>"`, `to =
  "view:<image>"` — the target in the camera frame of that view, `x_view = R
  x_target + t`, in the export's tvec unit (declared as its `length_unit`). An image whose pose the export has as `null`
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

The conversion is verified before anything is written to `out`: the v2 files,
written to a staging directory beside `out` (a dry run's: a temporary one) and
renamed into place only once verified, must read back equal to the OpenCV read —
the camera `==`, every pose's translation bit for bit (a file-born value; a
computed one within the length conversion's 2 ulp) and its rotation to the
quaternion encoding's precision (`_qvec`). With
`dry_run` nothing is written to `out`. Returns `(; path, cameras, poses,
n_posed, n_unposed)`, the output paths `nothing` in a dry run.
"""
migrate_opencv(path::AbstractString; kwargs...) = migrate_opencv(PixelFrame{:opencv}, path; kwargs...)

function migrate_opencv(
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
                all(((X, Y),) -> _pose_frames(X) == _pose_frames(Y) && _same_pose(X, Y), zip(back, records)),
            path, "the $(source) calibration"
        )
    end
    return dry_run ? (; path = String(path), cameras = nothing, poses = nothing, counts...) :
        (; path = String(path), cameras = cams_out, poses = poses_out, counts...)
end

# =============================================================================
# migrate_matlab — a third-party MATLAB calibration into the native kinds
# =============================================================================

"""
    migrate_matlab([file_frame = PixelFrame{:julia},] path; images, image_size, file_unit = nothing,
                   target = "calibration_target", out = dirname(path), dry_run = false) -> NamedTuple

Convert a third-party MATLAB calibration export (`calibration_matlab_*.json`:
the radial/tangential camera `fx, fy, cx, cy, r1…r3, t1, t2`, or the fisheye
`mapping_coeffs`, `distortion_center`, `stretch_matrix`; `mean_error`; one
`{rvec, tvec}` per image) into the same native files as
[`migrate_opencv`](@ref), which it mirrors: `<stem>_cameras.json` (the camera
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
migrate_matlab(path::AbstractString; kwargs...) = migrate_matlab(PixelFrame{:julia}, path; kwargs...)

function migrate_matlab(
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
# No reader reads an earlier version: a stored file is MIGRATED once, by the
# same guarded pattern as the third-party conversions above. `vgio_migrate`
# finds the file's schema and calls the method of `migrate_schema` for it,
# which reads the old record, builds the current value, WRITES it through the
# current writer (its validation) to a staging directory beside `out`
# (`write_staged`), READS it back with the current reader and COMPARES every
# number with the old file — only then does `vgio_migrate` rename the result to
# `out`. Nothing is written when any
# check fails, and the original is untouched unless `out` is the original's
# own path (an in-place migration, replaced only after the verification).
#
# `migrate_schema` is a declared extension point (`CONTRACT_GENERICS`): the
# owner of a schema adds the method for its old version — VisualGeometryToolkitIO
# for vgio/target/v1 and vgio/rig/v4, BlobBoards for its boards — and
# `vgio_migrate` (and the command) then migrates those files too, without this
# package knowing their schema.

"""
    migrate_schema(::Val{schema}, src, dst) -> NamedTuple

The migration of a file of the (old) `schema` — a `Symbol`, `Val(Symbol("vgio/rig/v4"))` —
at `src` into its kind's current schema, written to `dst` (a path in the
staging directory [`vgio_migrate`](@ref) owns). A method reads `src`, writes
the current file to `dst` through its writer, reads `dst` back with the
current reader and compares every number with `src`, throwing an
`ArgumentError` on any difference; it returns a summary `NamedTuple` with at
least `to`, the schema written (and whatever the owner reports: a rig's
`old_id` and `new_id`).

A declared extension point (VisualGeometryToolkitIO's `CONTRACT_GENERICS`): the
owner of a schema adds the method for each of its old versions.
VisualGeometryToolkitIO's own: `vgio/target/v1` → `vgio/target/v2` and
`vgio/rig/v4` → `vgio/rig/v5`.
"""
function migrate_schema end
# Whether a schema has a migration: a method of `migrate_schema` for it.
_has_migration(s::AbstractString) = hasmethod(migrate_schema, Tuple{Val{Symbol(s)}, String, String})

"""
    vgio_migrate(path; out = <stem>.migrated.json beside path, dry_run = false) -> NamedTuple

Migrate the file at `path` from the earlier schema version it declares to its
kind's current one, by the method of [`migrate_schema`](@ref) for that schema
(`vgio/target/v1`, `vgio/rig/v4`, and any a schema owner adds). The migration
is written through the current writer to a staging directory BESIDE `out`
(the same filesystem), read back with the current reader and compared with the
original, number for number; only then is it renamed to `out`, one atomic
`rename`. The original is untouched unless `out == path` (an in-place
migration, replaced by that rename only once verified); an existing `out` is
otherwise refused. With `dry_run` the migration is verified and nothing is
written. Returns `(; path, out, from, to, …)`, `out` `nothing` in a dry run,
with whatever the schema's migration reports — a rig's `old_ids` (the v4
identities measurement files may name it by: the one hashed from the file as
written, and the one v4 minted for the value in millimetres, when the file is
in another unit) and its `new_id`.
"""
vgio_migrate(path::AbstractString; out::AbstractString = _migrated_path(path), dry_run::Bool = false) =
    _vgio_migrate(path; out, dry_run)

# `vgio_migrate`, with `before_rename(r)` called on the verified result before
# anything is put at `out` (a tree persists its identity map there).
function _vgio_migrate(path::AbstractString; out::AbstractString, dry_run::Bool, before_rename = r -> nothing)
    s = _peek_schema(path)
    s isa AbstractString || throw(ArgumentError("$(path): not a vgio file (no schema)"))
    in_place = abspath(out) == abspath(path)
    (dry_run || in_place || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    _has_migration(s) ||
        throw(ArgumentError("$(path): no migration for a $(s) file (no method of VisualGeometryToolkitIO.migrate_schema for it)"))
    return write_staged(out => basename(path); dry_run) do staged
        r = merge((; path = String(path), out = dry_run ? nothing : String(out), from = String(s)), migrate_schema(Val(Symbol(s)), String(path), staged))
        before_rename(r)
        r
    end
end

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
created if missing. A dry run stages in a temporary directory and puts
nothing anywhere. The staging directory goes when this returns, whatever
happened. On a failure — `f` throws, or a rename does — the destinations are
left as they were: a destination that did not exist and was already renamed
into place is removed again (one that existed was replaced by its rename and
stays so), and a directory this call created for them is removed.
"""
function write_staged(f, outs::Pair{<:AbstractString, <:AbstractString}...; dry_run::Bool)
    dir = dirname(abspath(first(first(outs))))
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
# The writer's keywords that re-save an old file's value as that file: its own
# length unit (when it declares one) and its provenance.
_resave_kwargs(old::AbstractDict) = merge(
    haskey(old, "length_unit") ? (; length_unit = parse_unit(old["length_unit"]::String)) : (;), (; provenance = _provenance_of(old)),
)

# Two parsed trees, number for number: the same keys, the same strings, and
# every number the same Float64 (an integer `0` and `0.0` agree; the sign of a
# zero must).
_same_tree(a::AbstractDict, b::AbstractDict) = Set(keys(a)) == Set(keys(b)) && all(k -> _same_tree(a[k], b[k]), keys(a))
_same_tree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(i -> _same_tree(a[i], b[i]), eachindex(a))
_same_tree(a::Real, b::Real) = !(a isa Bool) && !(b isa Bool) ? (Float64(a) === Float64(b)) : a === b
_same_tree(a, b) = isequal(a, b)

# The old record's keys renamed to the new, and back. A target body's `warp`
# (v1: rigid only, `{"type": "rigid"}` — already the v2 calibration record) is
# its `calibration`, its `design` its `reference`; the target file's
# `calibration` block its `calibration_record`.
const _TARGET_BODY_RENAMES = ("warp" => "calibration", "design" => "reference")
_renamed(d::AbstractDict, renames) = Dict{String, Any}(get(Dict(renames), k, k) => v for (k, v) in d)
_body_v1_to_v2(b) = _renamed(b, _TARGET_BODY_RENAMES)
_body_v2_to_v1(b) = _renamed(b, map(reverse, _TARGET_BODY_RENAMES))

# A rig tree with each target body through `f` (a shallow copy).
function _map_rig_targets(f, tree::AbstractDict)
    t = copy(tree)
    haskey(t, "targets") && (t["targets"] = Any[merge(Dict{String, Any}(x), Dict{String, Any}("target" => f(x["target"]))) for x in tree["targets"]])
    return t
end
_rig_v4_to_v5(old) = (t = _map_rig_targets(_body_v1_to_v2, old); t["schema"] = _schema("rig"); t)
_rig_v5_to_v4(new, old) = (t = _map_rig_targets(_body_v2_to_v1, new); t["schema"] = old["schema"]; t)
function _target_v1_to_v2(old)
    body = _renamed(old, ("calibration" => "calibration_record",))
    body["schema"] = _schema("target")
    body["target"] = _body_v1_to_v2(old["target"])
    return body
end
function _target_v2_to_v1(new, old)
    back = _renamed(new, ("calibration_record" => "calibration",))
    back["schema"] = old["schema"]
    back["target"] = _body_v2_to_v1(new["target"])
    return back
end

# The guarded migration of a VGIo record: the old tree at `src` renamed into the
# current record (`forward(old)`), read by the current reader as a `T` (validated
# there), written by the current writer to `dst` in the old file's unit and with
# its provenance, read back, and every number compared with the old file once
# renamed back (`backward(new, old)`). Returns the old tree and the value.
function _migrate_record(::Type{T}, spec, src::AbstractString, dst::AbstractString, forward, backward, what) where {T}
    old = _raw_tree(src)
    staged = dst * ".staged.json"
    _write_store(staged, spec, forward(old))
    value = load(T, staged)
    rm(staged)
    save(dst, value; _resave_kwargs(old)...)
    load(T, dst)
    _verify(_same_tree(backward(_raw_tree(dst), old), old), src, what)
    return old, value
end

# vgio/target/v1 → vgio/target/v2
function migrate_schema(::Val{Symbol("vgio/target/v1")}, src::AbstractString, dst::AbstractString)
    _migrate_record(AbstractTarget, _TARGET_SPEC, src, dst, _target_v1_to_v2, _target_v2_to_v1, "the target record")
    return (; to = _schema("target"))
end

# vgio/rig/v4 → vgio/rig/v5. The rig's content identity changes with the
# migration (content_id: the fixed canonical tree replaced the v4 record's
# hash); both are reported, so the measurement files naming the rig are
# re-pointed (`vgio_migrate_tree`).
function migrate_schema(::Val{Symbol("vgio/rig/v4")}, src::AbstractString, dst::AbstractString)
    old, value = _migrate_record(Rig, _RIG_SPEC, src, dst, _rig_v4_to_v5, _rig_v5_to_v4, "the rig record")
    old_ids = unique!([_rig_v4_id(old), _rig_v4_id_mm(value, old, dst)])
    return (; to = _schema("rig"), old_id = first(old_ids), old_ids, new_id = content_id(dst))
end

# The content identities a vgio/rig/v4 rig had. v4 hashed the record without
# its provenance: `content_id(path)` the record its loaded rig writes in the
# FILE's unit — which, for a file this migration accepts (it re-writes to the
# file's own numbers), is the record as written, hashed here — and
# `content_id(rig)` the record of the value in MILLIMETRES, which differs for a
# rig whose file is in another unit and is what a producer that named the rig
# by its value put in its measurement files. Both are mapped.
_rig_v4_id(old::AbstractDict) =
    sha256_hex(canonical_json(Dict{String, Any}(k => v for (k, v) in old if k != "provenance")))
function _rig_v4_id_mm(value, old::AbstractDict, dst::AbstractString)
    old["length_unit"] == "mm" && return _rig_v4_id(old)
    p = dst * ".mm.json"
    save(p, value; length_unit = mm)
    mmrec = _raw_tree(p)
    rm(p)
    return _rig_v4_id(_rig_v5_to_v4(mmrec, old))
end

# =============================================================================
# A migrated rig's measurement files, re-pointed — a step of the v4 → v5 rig
# migration (`vgio_migrate_tree`, the command's `repoint`)
# =============================================================================
#
# The rig identity v4 minted hashed the record's format, so the rig/v4 → v5
# migration re-ids every rig, ONCE (the v5 identity is version-free), and the
# measurement files naming a rig follow it. Nothing else re-points.

const _MEASUREMENT_KINDS = Dict("vgio/target_points2d/v1" => TargetPoints2D, "vgio/target_points3d/v1" => TargetPoints3D)

# Re-point a measurement file (`vgio/target_points2d/v1`, `vgio/target_points3d/v1`)
# at the rigs its targets were migrated to: every identity in its `targets`
# that `ids` (old → new, a rig migration's `old_ids` and `new_id`) maps is
# replaced; nothing else changes, and the file stays at its version. Every
# identity the file names must be mapped or be the `content_id` of one of
# `rigs` (rigs or `vgio/rig/v5` paths; `hashes` their identities, when the
# caller has them): any other is an `ArgumentError` naming the file and the
# identity, and nothing is written — a stale identity is never kept silently.
# The guarded pattern: written by the current writer beside `out`, read back,
# compared with the original number for number (only the mapped identities
# differ), every measurement's (target, index) checked to resolve in the rigs
# its identities name — an unresolved (referenced) target's points need
# `resolve`, its owner's resolver; without it, such a target is checked by id
# alone and counted in `unchecked`. Only then is it renamed to `out`
# (`out == path` replaces it in place). `old` is the file's parsed tree, when
# the caller has it. Returns `(; path, out, repointed, unchecked)`.
function _repoint_targets(
        path::AbstractString, ids::AbstractDict, rigs; resolve = nothing,
        out::AbstractString = _migrated_path(path), dry_run::Bool = false,
        hashes = _target_hashes(_target_list(rigs)), old::AbstractDict = _raw_tree(path),
    )
    s = get(old, "schema", nothing)
    T = get(_MEASUREMENT_KINDS, s, nothing)
    T === nothing && throw(ArgumentError("$(path): a $(repr(s)) file names no targets to re-point (target_points2d/v1, target_points3d/v1)"))
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
                    " which is neither re-pointed nor the content_id of any rig given; nothing was written"
            )
        )
    end
    targets = String[get(map_, t, t) for t in obs.targets]
    repointed = count(t -> haskey(map_, t), obs.targets)
    named = [r for (r, h) in zip(rig_list, hashes) if h in targets]   # the rigs the measurements resolve in
    unchecked = _check_resolves_by_id(obs, _target_rigs(named; resolve), path)
    return write_staged(out => basename(path); dry_run) do staged
        save(staged, typeof(obs)(obs.frames; images = obs.images, sensors = obs.sensors, times = obs.times, time_origin = obs.time_origin, targets); _resave_kwargs(old)...)
        new = _raw_tree(staged)
        _verify(load(T, staged).targets == targets, path, "its targets")
        back = copy(new)
        haskey(old, "targets") && (back["targets"] = old["targets"])
        _verify(_same_tree(back, old), path, "the measurements")
        (; path = String(path), out = dry_run ? nothing : String(out), repointed, unchecked)
    end
end

# Every measurement's (target, index) resolves in the rigs: by its points
# where the target is resolved, by its id alone where it is a reference
# without its points (counted).
function _check_resolves_by_id(obs, rigs, path)
    npts = Dict{Symbol, Union{Nothing, Int}}()
    for rig in rigs, id in ids(rig)
        t = rig[id].target
        npts[id] = is_resolved(t) ? length(points(t)) : nothing
    end
    unchecked = 0
    for (f, os) in obs, o in os
        haskey(npts, o.target) || throw(ArgumentError("$(path): frame $(f) measures target $(o.target), which none of the rigs given holds"))
        n = npts[o.target]
        if n === nothing
            unchecked += 1
        else
            1 <= o.index <= n || throw(ArgumentError("$(path): frame $(f) measures point $(o.index) of target $(o.target), which has $(n)"))
        end
    end
    return unchecked
end

# =============================================================================
# A directory tree: every migratable file, then the measurements re-pointed
# =============================================================================

"""
    vgio_migrate_tree(dir; in_place = false, dry_run = false, io = stdout) -> Vector{NamedTuple}

Migrate every `.json` file under `dir` whose schema has a migration (a
`vgio/target/v1`, a `vgio/rig/v4`, and any schema whose owner added a
[`migrate_schema`](@ref) method), then re-point the measurement files under
`dir` (`vgio/target_points2d/v1`, `vgio/target_points3d/v1`) at the current
rigs of the tree, each by the guarded pattern, with one line per file to `io`
— a file left as it is says why (`current`, or no `targets`). Written beside
each original as `<stem>.migrated.json`, or over it with `in_place`.

The re-pointing is a one-time consequence of the rig/v4 → v5 migration: the v4
identity hashed the record's format, the v5 one does not, so each migrated
rig's old identities map to its new one. The map is PERSISTED in
`dir/vgio_migrate_ids.json` before a rig is put in place, and read by a later
run, so a run that fails after its rigs were migrated in place re-points
correctly when rerun; a rerun of a migrated tree is idempotent (every file
current, logged so). A measurement file naming an identity that is neither
mapped nor a current rig's of the tree is refused (logged `FAILED`, nothing
written for it), and the run ends in an `ArgumentError` listing every refusal.
"""
function vgio_migrate_tree(dir::AbstractString; in_place::Bool = false, dry_run::Bool = false, io::IO = stdout)
    idsfile = joinpath(dir, _IDS_SIDECAR)
    files = _tree_json(dir; io)
    schemas = Dict(f => _try_schema(f) for f in files)
    results = NamedTuple[]
    ids = _read_ids(idsfile)
    rigs = String[]                                                    # the current rigs of the tree
    failed = String[]
    for f in files
        s = schemas[f]
        if !(s isa String && _has_migration(s))
            if s === nothing
                println(io, "SKIPPED   ", f, ": no readable schema")
            elseif _is_current(s)
                s == _schema("rig") && push!(rigs, f)
                haskey(_MEASUREMENT_KINDS, s) || println(io, "current   ", f, ": ", s)   # (a measurement file: below)
            else
                println(io, "SKIPPED   ", f, ": ", s, " has no migration (a retired version, or a schema no package migrates)")
            end
            continue
        end
        out = in_place ? f : _migrated_path(f)
        so = isfile(out) ? _try_schema(out) : nothing
        if !in_place && so isa String && _is_current(so)
            println(io, "current   ", f, ": already migrated to ", out)
            continue
        end
        # verified, its identities mapped (and persisted before it is put in place)
        r = _vgio_migrate(f; out, dry_run, before_rename = r -> _map_ids!(ids, r, idsfile, dry_run))
        (!dry_run && r.to == _schema("rig")) && push!(rigs, out)
        push!(results, r)
        println(io, dry_run ? "verified  " : "migrated  ", f, ": ", s, " → ", r.to,
            haskey(r, :old_ids) ? "  (content id $(join(r.old_ids, ", ")) → $(r.new_id))" : "")
    end
    sort!(unique!(rigs))
    hashes = String[content_id(r) for r in rigs]
    current = Set(hashes)
    for f in files
        s = schemas[f]
        haskey(_MEASUREMENT_KINDS, something(s, "")) || continue
        old = _raw_tree(f)
        ts = String[t for t in something(get(old, "targets", nothing), Any[])]
        if isempty(ts)
            println(io, "current   ", f, ": names no targets (nothing to re-point)")
            continue
        end
        if all(in(current), ts)
            println(io, "current   ", f, ": names current rigs")
            continue
        end
        stale = [t for t in ts if !(t in current) && !(haskey(ids, t) && (dry_run || ids[t] in current))]
        if !isempty(stale)
            println(io, "FAILED    ", f, ": names targets identit$(length(stale) == 1 ? "y" : "ies") $(join(stale, ", ")), neither re-pointed nor a current rig's")
            push!(failed, f)
            continue
        end
        out = in_place ? f : _migrated_path(f)
        if !in_place && isfile(out)
            println(io, "current   ", f, ": already re-pointed to ", out)
            continue
        end
        if dry_run
            println(io, "would re-point  ", f, " (once the rigs are written)")
            continue
        end
        r = _repoint_targets(f, ids, rigs; out, hashes, old)
        push!(results, r)
        println(io, "re-pointed  ", f, ": ", r.repointed, " targets identit$(r.repointed == 1 ? "y" : "ies")",
            r.unchecked > 0 ? " ($(r.unchecked) measurements of referenced targets checked by id)" : "")
    end
    isempty(failed) || throw(ArgumentError("vgio_migrate_tree($(dir)): $(length(failed)) measurement file(s) name identities no rig of the tree has: $(join(failed, ", "))"))
    return results
end

# A migrated rig's old → new identities, into the tree's map — persisted,
# unless a dry run, before the rig is put in place.
function _map_ids!(ids::AbstractDict, r, idsfile, dry_run::Bool)
    haskey(r, :old_ids) || return nothing
    for o in r.old_ids
        ids[o] = r.new_id
    end
    dry_run || _write_ids(idsfile, ids)
    return nothing
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

# The persisted old → new identity map of a tree's migration.
const _IDS_SIDECAR = "vgio_migrate_ids.json"
_read_ids(p) = isfile(p) ? Dict{String, String}(String(k) => String(v) for (k, v) in _read_json(p)["identities"]) : Dict{String, String}()
function _write_ids(p, ids::AbstractDict)
    t = tempname(dirname(abspath(p)))
    _write_json(t, Dict("identities" => Dict(sort!(collect(ids)))))
    Base.Filesystem.rename(t, p)
    return nothing
end
# Whether a schema is its kind's current version.
function _is_current(s::AbstractString)
    m_ = match(r"^vgio/([a-z0-9_]+)/v(\d+)$", s)
    m_ === nothing && return false
    return s == _schema(String(m_[1]::AbstractString))
end
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
        migrate = kind == "opencv" ? migrate_opencv : migrate_matlab
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

Migrate a file of an earlier schema version into the current one (target:
vgio/target/v1 → v2; rig: vgio/rig/v4 → v5; file: any schema with a
migration), every such file under a directory (tree, then re-pointing the
measurement files that name a migrated rig), or re-point one measurement file
(repoint). A DRY RUN unless --apply: migrated to a temporary directory, read
back and compared number for number, nothing written. Written beside the
original as <stem>.migrated.json, or over it with --in-place (only once
verified).

  --out FILE             where to write (target|rig|file|repoint)
  --in-place             replace the original once verified
  --id OLD=NEW           repoint: a targets identity and its replacement (repeatable)
  --rig PATH             repoint: a migrated rig the new identities name (repeatable)
  --apply                write
"""

function _migrate_version_cli(kind::AbstractString, args::AbstractVector{<:AbstractString}; io::IO = stdout)
    parsed = _cli_options(args, ("--out", "--id", "--rig"), ("--apply", "--in-place"), _CLI_VERSION_USAGE, io)
    parsed isa Int && return parsed
    opts, files = parsed
    apply, in_place = haskey(opts, "--apply"), haskey(opts, "--in-place")
    out = haskey(opts, "--out") ? last(opts["--out"]) : nothing
    rigs = get(opts, "--rig", String[])
    ids = Dict{String, String}()
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
        # the command's kind migrates every old schema `_MIGRATIONS_TO_CURRENT` names it for
        (kind == "file" || get(_MIGRATIONS_TO_CURRENT, s, nothing) == kind) || throw(
            ArgumentError("$(file) is a $(repr(s)) file, not a $(join(sort!([o for (o, k) in _MIGRATIONS_TO_CURRENT if k == kind]), " or ")) one")
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
