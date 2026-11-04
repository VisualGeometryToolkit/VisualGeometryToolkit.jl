# =============================================================================
# The v2 validator — one table per file kind, run by the reader AND the writer
# =============================================================================
#
# Every vgio v2 file is checked against its kind's SPEC before a reader uses
# a field of it and before a writer puts it on disk (`_write_v2` validates the
# exact bytes it is about to write), so a writer cannot emit a file its reader
# refuses and a reader never meets a shape it did not declare. A spec is data:
#
#   _Obj(required, optional)   an object with EXACTLY these keys — a missing
#                              required key and an unknown key are refused alike
#   _NonEmpty                  a non-empty string
#   _Tok(v)                    one exact value (a declaration token)
#   _Int / _PosInt / _Real     an integer / a positive integer / a finite real
#   _Vec(elem, n)              an array of `elem`, length `n` (-1: any)
#   _OrNull(s)                 `null` or `s`
#   _Named(ok, what)           a string `ok` accepts (a unit, a frame name, …)
#   _RealMap                   an object of name → finite real (lens parameters,
#                              whose NAMES the owning reader checks against VGC)
#   _Free                      any object — `provenance` alone, never read
#   _Either(key => s, …)       one of several object shapes, told apart by a
#                              key only that shape carries
#   _Checked(s, f)             `s`, then a cross-field check `f(x)` returning
#                              an error message or `nothing`
#
# The same tables drive the tests: every required key of every kind is deleted
# in turn from a valid file, and each deletion must be refused by name.

abstract type _Spec end

struct _NonEmpty <: _Spec end
struct _Tok{T} <: _Spec
    value::T
end
struct _Int <: _Spec end
struct _PosInt <: _Spec end
struct _Real <: _Spec end
struct _Vec{S <: _Spec} <: _Spec
    elem::S
    n::Int
end
_Vec(elem::_Spec) = _Vec(elem, -1)
struct _OrNull{S <: _Spec} <: _Spec
    inner::S
end
struct _Named <: _Spec
    ok::Function
    what::String
end
struct _RealMap <: _Spec end
struct _Free <: _Spec end
struct _Obj <: _Spec
    required::Vector{Pair{Symbol, _Spec}}
    optional::Vector{Pair{Symbol, _Spec}}
end
_Obj(required::Pair{Symbol}...) = _Obj(collect(Pair{Symbol, _Spec}, required), Pair{Symbol, _Spec}[])
struct _Either <: _Spec
    alts::Vector{Pair{Symbol, _Spec}}
end
struct _Checked{S <: _Spec} <: _Spec
    inner::S
    check::Function
end

_refuse(file, at, msg) = throw(ArgumentError("$(file): $(isempty(at) ? "top level" : at): $(msg)"))
_at(at, k) = isempty(at) ? String(k) : "$(at).$(k)"

# finite as the Float64 it is read as (JSON.jl parses 1e400 as a BigFloat)
_isreal(x) = x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x))

function _validate(s::_Obj, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object")
    for (k, sk) in s.required
        haskey(x, k) || _refuse(file, _at(at, k), "missing (a required key)")
        _validate(sk, x[k], file, _at(at, k))
    end
    for (k, sk) in s.optional
        haskey(x, k) && _validate(sk, x[k], file, _at(at, k))
    end
    for k in keys(x)
        (any(p -> p.first === Symbol(k), s.required) || any(p -> p.first === Symbol(k), s.optional)) ||
            _refuse(file, _at(at, k), "unknown key (only `provenance` is free-form)")
    end
    return nothing
end
_validate(::_NonEmpty, x, file, at) =
    ((x isa AbstractString && !isempty(x)) || _refuse(file, at, "expected a non-empty string"); nothing)
_validate(s::_Tok, x, file, at) =
    (isequal(x, s.value) || _refuse(file, at, "must be $(repr(s.value)), got $(repr(x))"); nothing)
_validate(::_Int, x, file, at) = ((x isa Integer && !(x isa Bool)) || _refuse(file, at, "expected an integer"); nothing)
_validate(::_PosInt, x, file, at) =
    ((x isa Integer && !(x isa Bool) && x > 0) || _refuse(file, at, "expected a positive integer"); nothing)
_validate(::_Real, x, file, at) = (_isreal(x) || _refuse(file, at, "expected a finite number"); nothing)
function _validate(s::_Vec, x, file, at)
    x isa AbstractVector || _refuse(file, at, "expected an array")
    (s.n < 0 || length(x) == s.n) || _refuse(file, at, "expected $(s.n) entries, got $(length(x))")
    for (i, v) in enumerate(x)
        _validate(s.elem, v, file, "$(at)[$(i)]")
    end
    return nothing
end
_validate(s::_OrNull, x, file, at) = (x === nothing || _validate(s.inner, x, file, at); nothing)
function _validate(s::_Named, x, file, at)
    x isa AbstractString || _refuse(file, at, "expected $(s.what) (a string)")
    s.ok(x) || _refuse(file, at, "$(repr(x)) is not $(s.what)")
    return nothing
end
function _validate(::_RealMap, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object of named numbers")
    for (k, v) in x
        _isreal(v) || _refuse(file, _at(at, k), "expected a finite number")
    end
    return nothing
end
_validate(::_Free, x, file, at) = (x isa AbstractDict || _refuse(file, at, "expected an object"); nothing)
function _validate(s::_Either, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object")
    hits = [sk for (k, sk) in s.alts if haskey(x, k)]
    length(hits) == 1 || _refuse(
        file, at, "expected exactly one of the keys $(join(("`$(k)`" for (k, _) in s.alts), ", ")) to say which shape this is"
    )
    return _validate(only(hits), x, file, at)
end
function _validate(s::_Checked, x, file, at)
    _validate(s.inner, x, file, at)
    msg = s.check(x)
    msg === nothing || _refuse(file, at, msg)
    return nothing
end

# -----------------------------------------------------------------------------
# Shared pieces
# -----------------------------------------------------------------------------

_is_length_name(s) = try
    dimension(parse_unit(s)) == dimension(m)
catch
    false
end
_is_density_name(s) = try
    dimension(parse_unit(s)) == dimension(dpi)
catch
    false
end
_is_pixel_frame_name(s) = try
    pixel_frame_from_name(s)
    true
catch
    false
end

const _ROTATION_ENCODING = "quaternion_wxyz_hamilton"
const _TRANSFORM_CONVENTION = "x_to = R * x_from + t"
const _MATRIX_LAYOUT = "rows"
const _GRID_LAYOUT = "row_major"
const _POINT_INDEX_BASE = 1

# The current schema of each kind. A kind's version moves when its records
# change shape: the lens-bearing kinds (cameras, rigs with sensors) took v3 at
# the lens families (2026-10), whose lens record replaced v2's named-parameter
# one; the rig took v4 when it held targets too (the target model, 2026-10);
# every other older kind is v2.
#
# target v2 and rig v5 (the target calibrations, 2026-10-26): a planar target's
# `"warp"` became its `"calibration"` record (every TargetCalibration kind,
# with its parameters and covariance), its `"design"` reference `"reference"`,
# and the target file's `"calibration"` block `"calibration_record"`. Their
# predecessors are MIGRATED, once, by `vgio_migrate` (migrate.jl); no reader
# reads them.
#
# cameras v4 and rig v6 (the lens record, the same change): a lens's distortion
# stages in sequence (CT's `∘`) are listed in the record's "compose" in CT's
# order, the outer map first; v3/v5 listed them under "chain" in application
# order (refused by name in a current record). Migrated by `vgio_migrate`.
#
# target v3 and rig v6 (CoordinateTransformations' composition, 2026-11): a
# target calibration is CT's `∘` of a height field after one in-plane kind,
# its record `{"type": "composition", "compose": [outer, inner]}`, the members
# in CT's order and its covariance over that order; v2/v5's
# `{"type": "calibration_chain", "members": [inner, outer]}` listed them in
# application order (refused by name in a current record). Migrated by `vgio_migrate`, which refuses a v2/v5 chain with no v3
# form (two in-plane kinds).
#
# transforms and trajectory are v3 since their frames are typed (2026-10): the
# `from`/`to` names go through the frame grammar (`frame_from_name`), so a
# free-form name is refused, and a pose may carry its covariance. The kinds
# born after that are v1 (target, target_poses, target_points2d, target_points3d).
#
# generic_camera v3 (the present nodes, 2026-11): a grid model holds the nodes
# its imaged region reaches, listed as `present` (runs per lattice row), and
# its grids hold those nodes only; v2 held every node. A v2 file reads through
# its version step as every node present. A rig sensor's generic body carries
# `present` only when its model does not hold every node, so every rig/v6 file
# written before reads as it did, and rig stays v6.
const _SCHEMA_VERSION = Dict(
    "cameras" => 4, "rig" => 6, "transforms" => 3, "trajectory" => 3, "generic_camera" => 3,
    "target" => 3, "target_poses" => 1, "target_points2d" => 1, "target_points3d" => 1,
    "capture_settings" => 1,
)
_schema_version(kind) = get(_SCHEMA_VERSION, kind, 2)
_schema(kind) = "vgio/$(kind)/v$(_schema_version(kind))"
_schema_tok(kind) = _Tok(_schema(kind))
const _PIXEL_FRAME = _Named(_is_pixel_frame_name, "a registered pixel frame (colmap, opencv, julia)")
const _LENGTH_UNIT = _Named(_is_length_name, "a registered length unit (mm, cm, m, um, inch)")
const _DENSITY_UNIT = _Named(_is_density_name, "a registered density unit (dpi)")
const _POSE_DECLARATIONS = Pair{Symbol, _Spec}[
    :length_unit => _LENGTH_UNIT,
    :rotation_encoding => _Tok(_ROTATION_ENCODING),
    :transform_convention => _Tok(_TRANSFORM_CONVENTION),
]
const _PROVENANCE = Pair{Symbol, _Spec}[:provenance => _Free()]

const _POSE = _Obj(:rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3))

# A covariance on the wire (VisualGeometryToolkitCore conventions/persisted-geometry.md):
# a d×d array of rows, finite and exactly symmetric, in the value's own tangent
# order and the file's units. Present iff the value is `Uncertain`; all zeros
# is `exact`. Positive semi-definiteness is not tested (zero is valid).
_cov_spec(d::Int) = _Checked(
    _Vec(_Vec(_Real(), d), d),
    c -> all(c[i][j] == c[j][i] for i in 1:d, j in 1:d) ? nothing : "covariance must be symmetric",
)
const _COV2 = _cov_spec(2)
const _COV3 = _cov_spec(3)
const _COV6 = _cov_spec(6)
# A pose that may be uncertain: its 6×6 covariance over [δω; δt] — rotation
# first, rad and the file's length unit — in the chart `R ↦ exp(δω)·R,
# t ↦ t + δt` (VGC `RigidExpManifold6`; GTSAM's Pose3 order, not its chart).
const _UPOSE = _Obj(
    Pair{Symbol, _Spec}[:rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3)],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _XY = _Vec(_Real(), 2)
const _XYZ = _Vec(_Real(), 3)

# -----------------------------------------------------------------------------
# The two file boundaries every v2 kind goes through
# -----------------------------------------------------------------------------

# Kinds retired by the direct-serialization redesign (2026-10): no reader, no
# migration — each successor started at v1 under its new name (or moved out of
# VGIo), and the producer re-exports.
const _RETIRED_KINDS = Dict(
    "vgio/observations/v2" => "vgio/target_points2d/v1", "vgio/point_observations/v1" => "vgio/target_points3d/v1",
    "vgio/targets/v2" => "vgio/rig/v6", "vgio/rig/v3" => "vgio/rig/v6",
    "vgio/session/v3" => "the calibration pipeline's own session kind (a pipeline's checkpoint is the pipeline's, not VGIo's)",
)

# The earlier versions read through their version steps (migrate.jl), which
# `vgio_migrate` rewrites as the current one, and the command's kind argument
# for each.
const _MIGRATIONS_TO_CURRENT = Dict(
    "vgio/target/v1" => "target", "vgio/rig/v4" => "rig", "vgio/target/v2" => "target", "vgio/rig/v5" => "rig",
    "vgio/cameras/v1" => "file", "vgio/cameras/v2" => "file", "vgio/cameras/v3" => "file",
    "vgio/generic_camera/v2" => "file",
)

function _refuse_schema(path, kind::AbstractString, s)
    if s isa AbstractString && haskey(_RETIRED_KINDS, s)
        throw(
            ArgumentError(
                "$(path) is a retired $(s) file; its successor is $(_RETIRED_KINDS[s]). There is no reader for it: " *
                    "re-export it from its producer"
            )
        )
    end
    if s isa AbstractString && kind in ("transforms", "trajectory") && s in ("vgio/$(kind)/v1", "vgio/$(kind)/v2", "vgio/transform/v1")
        throw(
            ArgumentError(
                "$(path) is a retired $(s) file: its frame names are free strings, which $(_schema(kind)) " *
                    "types through the frame grammar (\"world\", \"camera_optical\", \"view:<id>\", \"target:<id>\", " *
                    "\"body:<id>\"). There is no reader for it: re-export it from its producer"
            )
        )
    end
    # an earlier version of the kind with no version step (`_read_store` reads
    # every one that has one): no published file holds it, so nothing reads it
    if _is_earlier_version(s, kind)
        throw(
            ArgumentError(
                "$(path) is a retired $(s) file, which no migration reads (VisualGeometryToolkitIO reads " *
                    "$(_schema(kind)) and the earlier versions a published file holds): re-export it from its producer"
            )
        )
    end
    throw(ArgumentError("$(path): expected schema $(_schema(kind)), got $(repr(s))"))
end

# THE RESERVED `payload` KEY. Every kind keeps its JSON header and may, once its
# bulk numeric arrays exceed ~10 MB, move them to an Apache Arrow sidecar named
# by `"payload": {"file": "<name>.arrow", "sha256": "<hex>"}` — glTF's JSON +
# bin pattern (VisualGeometryToolkitCore conventions/persisted-geometry.md). No kind
# implements it yet; the key is reserved so adding it breaks no schema, and a
# file carrying it is refused until its kind does.
const _PAYLOAD_KEY = :payload
_refuse_payload(path, d) = haskey(d, _PAYLOAD_KEY) && throw(
    ArgumentError(
        "$(path): carries a `payload` sidecar reference — reserved for the Arrow sidecar of bulk arrays, " *
            "which no kind implements yet"
    )
)

# Read a file of `kind`, refusing any other schema (a retired one by name), and
# validate it against `spec` before any field is used. A file of an earlier
# version of the kind is read through its version steps (`migrate_record`,
# migrate.jl), in memory: the record they give is validated as the current
# record and read as one.
function _read_store(path::AbstractString, kind::AbstractString, spec::_Spec)
    d = _read_vgio_json(path)                 # a non-finite number is refused by its field
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    s = get(d, :schema, nothing)
    s == _schema(kind) || (_reads_through_migration(s, kind) && (d = _migrated_on_load(d, path, kind, s)))
    get(d, :schema, nothing) == _schema(kind) || _refuse_schema(path, kind, s)
    _refuse_payload(path, d)
    _validate(spec, d, path, "")
    return d
end

# A vgio schema string's kind and version, or `nothing`.
function _split_schema(s)
    m = s isa AbstractString ? match(r"^vgio/([a-z0-9_]+)/v(\d+)$", s) : nothing
    return m === nothing ? nothing : (kind = String(m[1]::AbstractString), version = parse(Int, m[2]::AbstractString))
end
# Whether `s` names an earlier version of `kind`, and one its version steps read.
_is_earlier_version(s, kind::AbstractString) = (v = _split_schema(s); v !== nothing && v.kind == kind && v.version < _schema_version(kind))
_reads_through_migration(s, kind::AbstractString) = _is_earlier_version(s, kind) && _has_record_migration(s)

# An old file's record through its version steps, parsed again from its JSON
# text, so the reader meets what it meets in a current file. Silent: an old
# published file is a legitimate input; `@debug` names the version.
function _migrated_on_load(d, path, kind, s)
    refuse(why) = throw(ArgumentError("$(path): a $(s) file, read through its migration to $(_schema(kind)): $(why)"))
    t = try
        migrate_record(d)
    catch e
        # a step's refusal; or a record its version never wrote — a key every
        # such record held missing, a value of the wrong type — named by its error
        e isa ArgumentError && refuse(e.msg)
        e isa KeyError && refuse("the record has no $(repr(e.key)), which every $(s) record held")
        e isa Union{MethodError, InexactError, BoundsError, TypeError} && refuse("a value its version never wrote ($(sprint(showerror, e)))")
        rethrow()
    end
    reached = get(t, "schema", nothing)
    reached == _schema(kind) || refuse("its version steps end at $(repr(reached))")
    @debug "$(path): a $(s) file, read through its migration to $(reached)"
    return _parse_vgio_text(JSON.json(t; allownan = true), path)   # (compact: parsed, never written)
end

# Write `body` as a v2 file after validating the very bytes that land on disk
# against `spec`: a writer cannot produce a file its own reader refuses.
function _write_store(path::AbstractString, spec::_Spec, body)
    text = _json_text(body)                   # serialized once: these bytes are validated and written
    _validate(spec, _parse_vgio_text(text, path), path, "")   # (a NaN/Inf refused by its field)
    write(path, text)
    return nothing
end

# `provenance` as the writer stores it: the caller's free-form object, omitted
# when empty — never read back by VGIo.
_with_provenance(body::NamedTuple, provenance::NamedTuple) =
    isempty(provenance) ? body : merge(body, (provenance = provenance,))

# The spec of every v2 kind, by the name its schema carries (the tables live
# beside their readers and writers; looked up at call time).
function _spec(kind::AbstractString)
    kind == "units" && return _UNITS_SPEC
    kind == "transforms" && return _TRANSFORMS_SPEC
    kind == "cameras" && return _CAMERAS_SPEC
    kind == "generic_camera" && return _GENERIC_CAMERA_SPEC
    kind == "rig" && return _RIG_SPEC
    kind == "target" && return _TARGET_SPEC
    kind == "trajectory" && return _TRAJECTORY_SPEC
    kind == "target_poses" && return _TARGET_POSES_SPEC
    kind == "target_points2d" && return _TARGET_POINTS2D_SPEC
    kind == "target_points3d" && return _TARGET_POINTS3D_SPEC
    kind == "capture_settings" && return _CAPTURE_SETTINGS_SPEC
    kind == "correspondences" && return _CORRESPONDENCES_SPEC
    return nothing
end

"""
    validate(path) -> String

Check a vgio v2 file — or a COLMAP model directory's `vgio_units.json`, when
`path` is the directory — against its kind's table, the check every v2 reader
runs before using a field and every writer runs before writing, and return
the kind (`"cameras"`, `"rig"`, …). An `ArgumentError` names the file, the
offending key and the problem. A file of an earlier version is checked as
the current record its version steps give (as `load` reads it); a version no
step reads is refused by name, a retired kind with its successor. Semantic
checks that
need more than the one file — that measurements resolve in their targets —
are the exporters'.
"""
function validate(path::AbstractString)
    file = isdir(path) ? joinpath(path, UNITS_SIDECAR) : path
    s = _peek_schema(file)
    v = _split_schema(s)
    if v !== nothing
        spec = _spec(v.kind)
        if spec !== nothing && (s == _schema(v.kind) || _reads_through_migration(s, v.kind))
            _read_store(file, v.kind, spec)
            return v.kind
        end
    end
    v !== nothing && _spec(v.kind) !== nothing && _refuse_schema(file, v.kind, s)
    throw(ArgumentError("$(file): not a vgio file (schema $(repr(s)))"))
end
