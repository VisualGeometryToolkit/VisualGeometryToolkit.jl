# =============================================================================
# Migrations — third-party calibrations into the native kinds (migrate_opencv,
# migrate_matlab), earlier schema versions into the current ones (vgio_migrate,
# migrate_schema, repoint_targets) — and the bin/vgio_migrate.jl command
# =============================================================================
#
# Every conversion READS the foreign file, WRITES the native kinds to a
# temporary directory through the writers' validation, READS them back and
# COMPARES them with what it read — and only then, unless it is a dry run,
# writes them beside the original (or to `out`). The original is never touched.

_verify(ok::Bool, src, what) = ok || throw(
    ArgumentError("$(src): the conversion does not read back as the file read ($(what)); nothing was written")
)

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
written to a temporary directory, must read back equal to the OpenCV read —
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

    # the temporary copy goes when this returns, dry run or not — never left for
    # the process's exit in a long session
    return mktempdir() do tmp
        cams_tmp, poses_tmp = joinpath(tmp, "$(stem)_cameras.json"), joinpath(tmp, "$(stem)_poses.json")
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
        dry_run && return (; path = String(path), cameras = nothing, poses = nothing, n_posed = length(posed), n_unposed = length(names) - length(posed))
        mkpath(out)
        cams_out, poses_out = joinpath(out, "$(stem)_cameras.json"), joinpath(out, "$(stem)_poses.json")
        for p in (cams_out, poses_out)
            ispath(p) && throw(ArgumentError("$(p) exists; refusing to overwrite it"))
        end
        mv(cams_tmp, cams_out)
        mv(poses_tmp, poses_out)
        return (; path = String(path), cameras = cams_out, poses = poses_out, n_posed = length(posed), n_unposed = length(names) - length(posed))
    end
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
# current writer (its validation) to a temporary directory, READS it back with
# the current reader and COMPARES every number with the old file — only then
# does `vgio_migrate` move the result to `out`. Nothing is written when any
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
at `src` into its kind's current schema, written to `dst` (a path in a
temporary directory [`vgio_migrate`](@ref) owns). A method reads `src`, writes
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
migrate_schema(::Val{S}, src::AbstractString, dst::AbstractString) where {S} = throw(
    ArgumentError("$(src): no migration for a $(S) file (no method of VisualGeometryToolkitIO.migrate_schema for it)")
)

"""
    vgio_migrate(path; out = <stem>.migrated.json beside path, dry_run = false) -> NamedTuple

Migrate the file at `path` from the earlier schema version it declares to its
kind's current one, by the method of [`migrate_schema`](@ref) for that schema
(`vgio/target/v1`, `vgio/rig/v4`, and any a schema owner adds). The migration
is written to a temporary directory through the current writer, read back with
the current reader and compared with the original, number for number; only
then is it moved to `out`. The original is untouched unless `out == path` (an
in-place migration, replaced only once verified); an existing `out` is
otherwise refused. With `dry_run` the migration is verified and nothing is
written. Returns `(; path, out, from, to, …)`, `out` `nothing` in a dry run,
with whatever the schema's migration reports (a rig's `old_id` and `new_id`,
the content identities measurement files name it by before and after).
"""
function vgio_migrate(path::AbstractString; out::AbstractString = _migrated_path(path), dry_run::Bool = false)
    s = _peek_schema(path)
    s isa AbstractString || throw(ArgumentError("$(path): not a vgio file (no schema)"))
    in_place = abspath(out) == abspath(path)
    (dry_run || in_place || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    return mktempdir() do tmp
        staged = joinpath(tmp, basename(path))
        r = migrate_schema(Val(Symbol(s)), String(path), staged)
        dry_run && return merge((; path = String(path), out = nothing, from = String(s)), r)
        mkpath(dirname(abspath(out)))
        mv(staged, out; force = in_place)
        return merge((; path = String(path), out = String(out), from = String(s)), r)
    end
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

function _migrate_check(ok::Bool, src, what)
    ok || throw(ArgumentError("$(src): the migration does not read back as the file read ($(what)); nothing was written"))
    return nothing
end

# vgio/target/v1 → vgio/target/v2
function migrate_schema(::Val{Symbol("vgio/target/v1")}, src::AbstractString, dst::AbstractString)
    old = _raw_tree(src)
    body = _renamed(old, ("calibration" => "calibration_record",))
    body["schema"] = _schema("target")
    body["target"] = _body_v1_to_v2(old["target"])
    # the current value, built by the current reader from the renamed record
    # (validated there), then written by the current writer in the file's unit
    staged = dst * ".staged.json"
    _write_store(staged, _TARGET_SPEC, body)
    value = load(AbstractTarget, staged)
    rm(staged)
    save(dst, value; length_unit = parse_unit(old["length_unit"]::String), provenance = _provenance_of(old))
    # read back, and every number compared with the old file's
    load(AbstractTarget, dst)
    new = _raw_tree(dst)
    back = _renamed(new, ("calibration_record" => "calibration",))
    back["schema"] = old["schema"]
    back["target"] = _body_v2_to_v1(new["target"])
    _migrate_check(_same_tree(back, old), src, "the target record")
    return (; to = _schema("target"))
end

# vgio/rig/v4 → vgio/rig/v5. The rig's content identity changes with the
# migration (content_id: the fixed canonical tree replaced the v4 record's
# hash); both are reported, so measurement files naming the rig can be
# re-pointed (`repoint_targets`).
function migrate_schema(::Val{Symbol("vgio/rig/v4")}, src::AbstractString, dst::AbstractString)
    old = _raw_tree(src)
    body = copy(old)
    body["schema"] = _schema("rig")
    haskey(old, "targets") &&
        (body["targets"] = Any[merge(Dict{String, Any}(t), Dict{String, Any}("target" => _body_v1_to_v2(t["target"]))) for t in old["targets"]])
    staged = dst * ".staged.json"
    _write_store(staged, _RIG_SPEC, body)
    value = load(Rig, staged)
    rm(staged)
    save(dst, value; length_unit = parse_unit(old["length_unit"]::String), provenance = _provenance_of(old))
    load(Rig, dst)
    new = _raw_tree(dst)
    back = copy(new)
    back["schema"] = old["schema"]
    haskey(new, "targets") &&
        (back["targets"] = Any[merge(Dict{String, Any}(t), Dict{String, Any}("target" => _body_v2_to_v1(t["target"]))) for t in new["targets"]])
    _migrate_check(_same_tree(back, old), src, "the rig record")
    return (; to = _schema("rig"), old_id = _rig_v4_id(old), new_id = content_id(dst))
end

# The content identity a vgio/rig/v4 file had: the SHA-256 of the canonical
# JSON of its record without its provenance (the v4 definition, exact for a
# file whose numbers have at most 15 significant digits — every file the
# stack's writer wrote from file-born values).
_rig_v4_id(old::AbstractDict) =
    sha256_hex(canonical_json(Dict{String, Any}(k => v for (k, v) in old if k != "provenance")))

# =============================================================================
# repoint_targets — a measurement file's `targets` re-pointed at migrated rigs
# =============================================================================

const _MEASUREMENT_KINDS = Dict("vgio/target_points2d/v1" => TargetPoints2D, "vgio/target_points3d/v1" => TargetPoints3D)

"""
    repoint_targets(path, ids, rigs; resolve = nothing, out = <stem>.migrated.json beside path,
                    dry_run = false) -> NamedTuple

Re-point a measurement file (`vgio/target_points2d/v1`, `vgio/target_points3d/v1`)
at the rigs its targets were migrated to: every identity in its `targets` that
`ids` (old → new, as [`vgio_migrate`](@ref) reports a rig's `old_id` and
`new_id`) maps is replaced; nothing else changes, and the file stays at its
version. The guarded pattern: written by the current writer to a temporary
directory, read back, compared with the original number for number (only the
mapped identities differ), every new identity checked to be the
[`content_id`](@ref) of one of `rigs` (rigs or `vgio/rig/v5` paths), and
every measurement's (target, index) checked to resolve in them — an
unresolved (referenced) target's points need `resolve`, its owner's resolver;
without it, such a target is checked by id alone and counted in `unchecked`.
Only then is it moved to `out` (`out == path` replaces it in place). Returns
`(; path, out, repointed, unchecked)`.
"""
function repoint_targets(
        path::AbstractString, ids::AbstractDict, rigs; resolve = nothing,
        out::AbstractString = _migrated_path(path), dry_run::Bool = false,
    )
    s = _peek_schema(path)
    T = get(_MEASUREMENT_KINDS, s, nothing)
    T === nothing && throw(ArgumentError("$(path): a $(repr(s)) file names no targets to re-point (target_points2d/v1, target_points3d/v1)"))
    in_place = abspath(out) == abspath(path)
    (dry_run || in_place || !ispath(out)) || throw(ArgumentError("$(out) exists; refusing to overwrite it"))
    old = _raw_tree(path)
    obs = load(T, path)
    map_ = Dict{String, String}(String(k) => String(v) for (k, v) in ids)
    targets = String[get(map_, t, t) for t in obs.targets]
    repointed = count(t -> haskey(map_, t), obs.targets)
    loaded = _target_rigs(rigs; resolve)
    hashes = Set(_target_hashes(rigs))
    for t in obs.targets
        haskey(map_, t) && (map_[t] in hashes || throw(ArgumentError("$(path): $(t) is re-pointed to $(map_[t]), which is none of the rigs given")))
    end
    unchecked = _check_resolves_by_id(obs, loaded, path)
    return mktempdir() do tmp
        staged = joinpath(tmp, basename(path))
        _save_repointed(staged, obs, targets, old)
        new = _raw_tree(staged)
        load(T, staged).targets == targets || _migrate_check(false, path, "its targets")
        back = copy(new)
        haskey(old, "targets") && (back["targets"] = old["targets"])
        _migrate_check(_same_tree(back, old), path, "the measurements")
        dry_run && return (; path = String(path), out = nothing, repointed, unchecked)
        mkpath(dirname(abspath(out)))
        mv(staged, out; force = in_place)
        return (; path = String(path), out = String(out), repointed, unchecked)
    end
end

_save_repointed(p, obs::TargetPoints2D{F}, targets, old) where {F} = save(
    p, TargetPoints2D{F}(obs.frames; images = obs.images, sensors = obs.sensors, times = obs.times, time_origin = obs.time_origin, targets);
    provenance = _provenance_of(old),
)
_save_repointed(p, obs::TargetPoints3D{F}, targets, old) where {F} = save(
    p, TargetPoints3D{F}(obs.frames; images = obs.images, sensors = obs.sensors, times = obs.times, time_origin = obs.time_origin, targets);
    length_unit = parse_unit(old["length_unit"]::String), provenance = _provenance_of(old),
)

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
[`migrate_schema`](@ref) method), then re-point every measurement file under
`dir` that names a migrated rig ([`repoint_targets`](@ref)), each by the
guarded pattern; one line per file to `io`. Written beside each original as
`<stem>.migrated.json`, or over it with `in_place`. A failure stops the run
before anything else is written and is rethrown (files already migrated stay
migrated, and a rerun skips them: they are current).
"""
function vgio_migrate_tree(dir::AbstractString; in_place::Bool = false, dry_run::Bool = false, io::IO = stdout)
    files = sort!([joinpath(r, f) for (r, _, fs) in walkdir(dir) for f in fs if endswith(f, ".json")])
    schemas = Dict(f => _try_schema(f) for f in files)
    results = NamedTuple[]
    ids = Dict{String, String}()
    rigs = String[]
    for f in files
        s = schemas[f]
        (s isa String && _has_migration(s)) || continue
        r = vgio_migrate(f; out = in_place ? f : _migrated_path(f), dry_run)
        push!(results, r)
        if haskey(r, :old_id)
            ids[r.old_id] = r.new_id
            push!(rigs, dry_run ? f : r.out)
        end
        println(io, dry_run ? "verified  " : "migrated  ", f, ": ", s, " → ", r.to, haskey(r, :old_id) ? "  (content id $(r.old_id) → $(r.new_id))" : "")
    end
    isempty(ids) && return results
    for f in files
        haskey(_MEASUREMENT_KINDS, something(schemas[f], "")) || continue
        any(t -> haskey(ids, t), something(get(_raw_tree(f), "targets", nothing), Any[])) || continue
        dry_run && (println(io, "would re-point  ", f, " (after the rigs are written)"); continue)
        r = repoint_targets(f, ids, rigs; out = in_place ? f : _migrated_path(f))
        push!(results, r)
        println(io, "re-pointed  ", f, ": ", r.repointed, " targets identit$(r.repointed == 1 ? "y" : "ies")",
            r.unchecked > 0 ? " ($(r.unchecked) measurements of referenced targets checked by id)" : "")
    end
    return results
end
_try_schema(f) = try
    s = _peek_schema(f)
    s isa String ? s : nothing
catch
    nothing
end
# Whether a schema has a migration: a method of `migrate_schema` other than the refusing fallback.
_has_migration(s::AbstractString) = which(migrate_schema, Tuple{Val{Symbol(s)}, String, String}) !==
    which(migrate_schema, Tuple{Val{:__none__}, String, String})

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

function _migrate_external_cli(kind::AbstractString, args::AbstractVector{<:AbstractString}; io::IO = stdout)
    opts = Dict{String, String}()
    apply, files = false, String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--apply"
            apply = true
        elseif a in ("--images", "--image-size", "--file-unit", "--file-frame", "--target", "--out")
            i < length(args) || (print(io, _CLI_EXTERNAL_USAGE); return 2)
            opts[a] = args[i += 1]
        elseif a in ("-h", "--help")
            print(io, _CLI_EXTERNAL_USAGE)
            return 0
        elseif startswith(a, "--")
            println(io, "unknown option $(a)")
            print(io, _CLI_EXTERNAL_USAGE)
            return 2
        else
            push!(files, String(a))
        end
        i += 1
    end
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
    apply, in_place, out, files = false, false, nothing, String[]
    ids, rigs = Dict{String, String}(), String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--apply"
            apply = true
        elseif a == "--in-place"
            in_place = true
        elseif a in ("--out", "--id", "--rig")
            i < length(args) || (print(io, _CLI_VERSION_USAGE); return 2)
            v = args[i += 1]
            if a == "--out"
                out = String(v)
            elseif a == "--rig"
                push!(rigs, String(v))
            else
                kv = split(v, '='; limit = 2)
                length(kv) == 2 || (println(io, "--id is OLD=NEW, got $(v)"); return 2)
                ids[String(kv[1])] = String(kv[2])
            end
        elseif a in ("-h", "--help")
            print(io, _CLI_VERSION_USAGE)
            return 0
        elseif startswith(a, "--")
            println(io, "unknown option $(a)")
            print(io, _CLI_VERSION_USAGE)
            return 2
        else
            push!(files, String(a))
        end
        i += 1
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
            r = repoint_targets(file, ids, rigs; out = dest, dry_run = !apply)
            println(io, apply ? "re-pointed  " : "verified  ", file, ": ", r.repointed, " targets identities",
                apply ? "  → $(r.out)" : " (dry run; --apply to write)")
            return 0
        end
        s = _peek_schema(file)
        want = Dict("target" => "vgio/target/v1", "rig" => "vgio/rig/v4")
        (kind == "file" || s == want[kind]) || throw(ArgumentError("$(file) is a $(repr(s)) file, not a $(want[kind]) one"))
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
