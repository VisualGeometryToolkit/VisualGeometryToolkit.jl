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
#   _CheckedSpec(s, f)         `s`, then a cross-field check `f(x)` returning
#                              an error message or `nothing`
#   _IfKey(key, a, b)          `a` if the object holds `key`, else `b`
#   _SchemaTok(kind) / _Cov(d) / _RowRuns / _LengthRoles   a schema string of the kind's
#                              major; a covariance; lattice runs; length roles
#
# The same tables drive the tests: every required key of every kind is deleted
# in turn from a valid file, and each deletion must be refused by name.

abstract type _Spec end

# Whether an unknown key is tolerated rather than refused: in a file of a higher
# minor version than the reader's own, whose added keys are optional by the
# version rule (the read boundary, `_checked_store`, scopes it).
const _TOLERATE_UNKNOWN = ScopedValue(false)

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
struct _CheckedSpec{S <: _Spec} <: _Spec
    inner::S
    check::Function
end
# One of two object shapes, told apart by whether the object holds `key`
# (`then` if it does, `otherwise` if not): JSON Schema's if/then/else.
struct _IfKey{A <: _Spec, B <: _Spec} <: _Spec
    key::Symbol
    then::A
    otherwise::B
end

_refuse(file, at, msg) = throw(ArgumentError("$(file): $(isempty(at) ? "top level" : at): $(msg)"))
_at(at, k) = isempty(at) ? String(k) : "$(at).$(k)"

# finite as the Float64 it is read as (JSON.jl parses 1e400 as a BigFloat)

# A key of a parsed object (a `JSON.Object`, read by Symbol) or of a plain
# `Dict{String}` (a record built in memory, `_plain`'s), by its Symbol.
_haskey(x, k::Symbol) = haskey(x, k) || (x isa AbstractDict{String} && haskey(x, String(k)))
_getkey(x, k::Symbol) = x isa AbstractDict{String} ? x[String(k)] : x[k]

function _validate(s::_Obj, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object")
    for (k, sk) in s.required
        _haskey(x, k) || _refuse(file, _at(at, k), "missing (a required key)")
        _validate(sk, _getkey(x, k), file, _at(at, k))
    end
    for (k, sk) in s.optional
        _haskey(x, k) && _validate(sk, _getkey(x, k), file, _at(at, k))
    end
    _TOLERATE_UNKNOWN[] && return nothing
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
_validate(::_Real, x, file, at) = (VgtkCore.is_finite_real(x) || _refuse(file, at, "expected a finite number"); nothing)
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
        VgtkCore.is_finite_real(v) || _refuse(file, _at(at, k), "expected a finite number")
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
function _validate(s::_IfKey, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object")
    return _validate(_haskey(x, s.key) ? s.then : s.otherwise, x, file, at)
end
function _validate(s::_CheckedSpec, x, file, at)
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


# The runs of a spline lattice's present nodes (style.jl), `[[row, [[col_lo, col_hi], …]], …]`:
# each row once, rows increasing, its runs nonempty, increasing and apart (a
# gap of at least one node), every index a positive integer (1-based, the
# file's `index_base`). The lattice's
# bounds and the node count are the record's check (`_lattice_problem`).
struct _RowRuns <: _Spec end
function _validate(::_RowRuns, x, file, at)
    x isa AbstractVector || _refuse(file, at, "expected an array of [row, [[col_lo, col_hi], …]] entries")
    isint(v) = v isa Integer && !(v isa Bool) && v >= _INDEX_BASE
    last_row = _INDEX_BASE - 1
    for (i, e) in enumerate(x)
        a = "$(at)[$(i)]"
        (e isa AbstractVector && length(e) == 2 && isint(e[1]) && e[2] isa AbstractVector && !isempty(e[2])) ||
            _refuse(file, a, "expected [row, [[col_lo, col_hi], …]], a row index and its nonempty runs")
        e[1] > last_row || _refuse(file, a, "rows are listed once, increasing; row $(e[1]) follows row $(last_row)")
        last_row = e[1]
        last_hi = _INDEX_BASE - 2
        for (j, r) in enumerate(e[2])
            (r isa AbstractVector && length(r) == 2 && isint(r[1]) && isint(r[2]) && r[1] <= r[2]) ||
                _refuse(file, "$(a)[2][$(j)]", "expected a run [col_lo, col_hi] of positive (1-based) integers, col_lo ≤ col_hi")
            r[1] > last_hi + 1 ||
                _refuse(file, "$(a)[2][$(j)]", "a row's runs are increasing and apart; [$(r[1]), $(r[2])] follows a run ending at $(last_hi)")
            last_hi = r[2]
        end
    end
    return nothing
end

# THE LATTICE RECORD, one for every spline (a SplineLens's backward spline and
# its forward one, a non-central model's, RadiometricCamera's aperture and
# focus lattices): its `area` `[x, y, w, h]`, its `grid_size` `{width, height}`
# (columns, rows), the held nodes `present` (runs per row, 1-based; absent:
# every node) and its `controls`, an object of named control arrays over the
# held nodes, row by row (`direction`, `point`). `_lattice_spec(controls,
# optional)` is the spec of a lattice whose controls are `name => dimension`.
const _SIZE_INT = _Obj(:width => _PosInt(), :height => _PosInt())
_lattice_spec(controls::Vector{<:Pair{Symbol, Int}}, optional::Vector{<:Pair{Symbol, Int}} = Pair{Symbol, Int}[]) = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :area => _Vec(_Real(), 4), :grid_size => _SIZE_INT,
            :controls => _Obj(Pair{Symbol, _Spec}[k => _Vec(_Vec(_Real(), d)) for (k, d) in controls],
                              Pair{Symbol, _Spec}[k => _Vec(_Vec(_Real(), d)) for (k, d) in optional]),
        ],
        Pair{Symbol, _Spec}[:present => _RowRuns()],
    ),
    _lattice_problem,
)
# Its own check: the `area` a finite rectangle of positive widths, then the
# lattice (`_lattice_count_problem`) with each of its control arrays; `s` a
# parsed record (keys by Symbol or String); the message, or `nothing`.
function _lattice_problem(s)
    a = _getkey(s, :area)
    (a[3] > 0 && a[4] > 0) || return "the spline's area [x, y, w, h] has positive widths; got w = $(a[3]), h = $(a[4])"
    g = _getkey(s, :grid_size)
    runs = _haskey(s, :present) ? _getkey(s, :present) : nothing
    c = _getkey(s, :controls)
    return _lattice_count_problem(_getkey(g, :width), _getkey(g, :height), runs, [String(k) => v for (k, v) in c])
end
# A lattice of `gw × gh` nodes (at least 4 × 4), its runs of present nodes `runs`
# (`nothing`: every node; already through `_RowRuns`) inside it, and each control
# array (`name => array`) holding one entry per held node — the records' check and
# the old grid body's (migrate.jl).
function _lattice_count_problem(gw, gh, runs, arrays)
    (gw >= 4 && gh >= 4) || return "a cubic lattice has at least 4 × 4 nodes; grid_size = {width: $(gw), height: $(gh)}"
    if runs !== nothing
        isempty(runs) && return "`present` holds no node"
        for e in runs
            e[1] <= gh || return "`present` names row $(e[1]) of a lattice of $(gh) rows"
            last(e[2])[2] <= gw || return "`present` names column $(last(e[2])[2]) of a lattice of $(gw) columns"
        end
    end
    n = runs === nothing ? gw * gh : _runs_count(runs)
    what = runs === nothing ? "grid_size's $(gw) × $(gh) = $(n)" : "the $(n) nodes `present` holds"
    for (k, v) in arrays
        length(v) == n || return "the spline's $(k) has $(length(v)) entries, expected $(what)"
    end
    return nothing
end

# THE VERSION OF A KIND is `v<major>.<minor>` (`vgio/<kind>/v<major>.<minor>`; a
# string written before IO 0.10, `v<n>`, is `v<n>.0`). The MAJOR version moves
# when the meaning of a record changes or a key is renamed or removed: an older
# reader cannot read it. The MINOR version moves when a kind gains an optional
# key: a reader reads its own major at ANY minor, and in a file of a HIGHER minor
# than its own it tolerates the keys that minor added (they are optional by the
# rule, so the value it reads is the file's value less what it does not know); in
# a file of its own or a lower minor, an unknown key is refused as ever
# (`_validate(::_Obj)`, the scoped `_TOLERATE_UNKNOWN`).
#
# The history of each kind is the list of its version steps (`migrate_schema`
# methods, migrate.jl), never a hand-kept list: `schema_versions()` derives the
# table the docs print. IO 0.10 moved EVERY kind up a major version: the one
# file-level `units` and `conventions` objects and the `generator` (the envelope,
# below), every Size2 a `{width, height}` object, every index 1-based, one
# covariance key, ASCII snake_case keys and `frame_index` as the frame join key.
# Each kind's previous version has ONE step to it; the earlier steps are as
# they were (each step's own comment says what it moved).
const _SCHEMA_VERSION = Dict{String, Tuple{Int, Int}}(
    "cameras" => (6, 0), "rig" => (10, 0), "transforms" => (5, 0), "trajectory" => (5, 0),
    "target" => (6, 0), "target_poses" => (3, 0), "target_homographies" => (2, 0),
    "observations" => (3, 0),   # (v3: `vgio/observations/v2` was a retired kind of another meaning, refused by name)
    "capture_settings" => (2, 0), "calibration" => (3, 0),
    "vignetting" => (2, 0), "environment_light" => (2, 0), "blur" => (2, 0),
    "units" => (3, 0),
    "panorama" => (2, 0),   # (v1: IO 0.9.4's, read through its step)
    "detection" => (1, 0), "sensor_noise" => (1, 0),   # IO 0.10: detections per image; a body's sensor noise
)
_schema_version(kind::AbstractString) = (v = get(_SCHEMA_VERSION, kind, nothing);
    v === nothing ? throw(ArgumentError("no vgio kind $(repr(kind)) (the kinds: $(join(sort!(collect(keys(_SCHEMA_VERSION))), ", ")))")) : v)
_version_string((M, n)::Tuple{Int, Int}) = "v$(M).$(n)"
_schema(kind) = "vgio/$(kind)/$(_version_string(_schema_version(kind)))"
# A record's `schema` field accepts its kind's current major at any minor.
struct _SchemaTok <: _Spec
    kind::String
end
_schema_tok(kind) = _SchemaTok(kind)
function _validate(s::_SchemaTok, x, file, at)
    v = _split_schema(x)
    (v !== nothing && v.kind == s.kind && v.major == first(_schema_version(s.kind))) ||
        _refuse(file, at, "must be $(repr(_schema(s.kind))) (any minor version), got $(repr(x))")
    return nothing
end

# THE ENVELOPE. Every vgio file states, before its body, what is about the file:
#
#   "schema":      "vgio/<kind>/v<major>.<minor>"
#   "generator":   {"name": "VisualGeometryToolkitIO", "version": "0.10.0",
#                   "caller": {"name": <the calling package>, "version": …}}    (caller optional)
#   "units":       {"length": "mm", "time": "s", "angle": "rad", "pixel": "colmap",
#                   "density": "dpi", "radiance": "relative", "length_roles": {…}}
#   "conventions": {"rotation_encoding": "quaternion_wxyz_hamilton",
#                   "transform_convention": "x_to = R * x_from + t",
#                   "matrix_layout": "rows", "grid_layout": "row_major", "index_base": 1}
#
# and, after the body, the free-form "provenance" (optional, never read). The
# units and the conventions are ALWAYS written, whole, whatever the kind holds,
# and a file missing one is refused: nothing defaults. Every number of the body
# is in a unit the `units` object declares, raised to the power its field's
# schema states (a pose covariance's rotation block rad², a print blur's
# `length²`, a sensor blur's `px²`, a focus surface's `length⁻¹`); no field
# states a unit of its own. A length is in `length` unless its field names a
# role of `length_roles` (a board's pattern in pt). `pixel` is the pixel frame
# every pixel coordinate is in; `angle` is radians and `time` seconds (fixed
# tokens, as the rotation encoding is); `radiance` the unit of a light's
# coefficients ("relative": a radiance known up to scale).
const _ROTATION_ENCODING = "quaternion_wxyz_hamilton"
const _TRANSFORM_CONVENTION = "x_to = R * x_from + t"
"""
    MATRIX_LAYOUT

How a vgio record lays out a matrix: `"rows"`, a list of its rows (the
`matrix_layout` every file's `conventions` declare). Another package's record
that holds matrices (CaliBlobs' session record) declares the same.
"""
const MATRIX_LAYOUT = "rows"
const _GRID_LAYOUT = "row_major"
const _INDEX_BASE = 1
const _TIME_UNIT = "s"
const _ANGLE_UNIT = "rad"
const _RADIANCE_UNIT = "relative"

const _PIXEL_FRAME = _Named(_is_pixel_frame_name, "a registered pixel frame (colmap, opencv, julia)")
const _LENGTH_UNIT = _Named(_is_length_name, "a registered length unit (mm, cm, m, um, inch)")
const _DENSITY_UNIT = _Named(_is_density_name, "a registered density unit (dpi)")
# A units object's length roles: role → a length unit, the role names snake_case.
struct _LengthRoles <: _Spec end
function _validate(::_LengthRoles, x, file, at)
    x isa AbstractDict || _refuse(file, at, "expected an object of length role => length unit")
    for (k, v) in x
        r = String(k)
        (occursin(r"^[a-z][a-z0-9_]*$", r) && !(r in ("length", "world", "density"))) ||
            _refuse(file, _at(at, k), "a length role is a snake_case name other than length, world and density")
        _validate(_LENGTH_UNIT, v, file, _at(at, k))
    end
    return nothing
end
_units_spec(length_spec::_Spec = _LENGTH_UNIT) = _Obj(
    :length => length_spec, :time => _Tok(_TIME_UNIT), :angle => _Tok(_ANGLE_UNIT), :pixel => _PIXEL_FRAME,
    :density => _DENSITY_UNIT, :radiance => _Tok(_RADIANCE_UNIT), :length_roles => _LengthRoles(),
)
const _UNITS_OBJECT = _units_spec()
const _CONVENTIONS_OBJECT = _Obj(
    :rotation_encoding => _Tok(_ROTATION_ENCODING), :transform_convention => _Tok(_TRANSFORM_CONVENTION),
    :matrix_layout => _Tok(MATRIX_LAYOUT), :grid_layout => _Tok(_GRID_LAYOUT), :index_base => _Tok(_INDEX_BASE),
)
const _PACKAGE_ID = _Obj(:name => _NonEmpty(), :version => _Named(s -> tryparse(VersionNumber, s) !== nothing, "a version number"))
const _GENERATOR = _Obj(Pair{Symbol, _Spec}[:name => _NonEmpty(), :version => _PACKAGE_ID.required[2].second],
                        Pair{Symbol, _Spec}[:caller => _PACKAGE_ID])
const _PROVENANCE = Pair{Symbol, _Spec}[:provenance => _Free()]

# The spec of a file of `kind`: the envelope, then the body's `required` and
# `optional` keys, then `provenance`; a cross-field `check` of the whole record.
function _kind_spec(kind::AbstractString, required::Vector{<:Pair{Symbol}}, optional::Vector{<:Pair{Symbol}} = Pair{Symbol, _Spec}[];
                    check = nothing, units::_Spec = _UNITS_OBJECT)
    o = _Obj(
        Pair{Symbol, _Spec}[:schema => _schema_tok(kind), :generator => _GENERATOR, :units => units,
                            :conventions => _CONVENTIONS_OBJECT, required...],
        Pair{Symbol, _Spec}[optional..., _PROVENANCE...],
    )
    return check === nothing ? o : _CheckedSpec(o, check)
end

# The writer's identity, every file's `generator`: this package and its version,
# and the calling package where one says so (`with_caller`).
const _CALLER = Base.ScopedValues.ScopedValue{Union{Nothing, Module}}(nothing)
"""
    with_caller(f, caller::Module)

Run `f()` with `caller` named as the calling package in the `generator` of every
vgio file written inside it: `"generator": {"name": "VisualGeometryToolkitIO",
"version": …, "caller": {"name": "<caller>", "version": "<its pkgversion>"}}`. A
package that writes vgio files (or its own kinds through [`write_record`](@ref))
wraps its writers in it, so a file records the software that wrote it.
"""
with_caller(f, caller::Module) = Base.ScopedValues.with(f, _CALLER => caller)
_package_id(m::Module) = (name = String(nameof(m)), version = string(something(pkgversion(m), v"0.0.0")))
function _generator_json()
    g = _package_id(@__MODULE__)
    c = _CALLER[]
    return (c === nothing || c === (@__MODULE__)) ? g : merge(g, (caller = _package_id(c),))
end

# The envelope a writer puts before a body of `kind` (or the schema string
# `schema`, another package's kind), from the file's style.
_envelope(kind::AbstractString, style) = (schema = _schema(kind), _envelope_tail(style)...)
_envelope_tail(style) = (generator = _generator_json(), units = _units_json(style), conventions = _CONVENTIONS_JSON)
const _CONVENTIONS_JSON = (rotation_encoding = _ROTATION_ENCODING, transform_convention = _TRANSFORM_CONVENTION,
                           matrix_layout = MATRIX_LAYOUT, grid_layout = _GRID_LAYOUT, index_base = _INDEX_BASE)

const _POSE = _Obj(:rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3))

# A covariance on the wire (VisualGeometryToolkitCore conventions/persisted-geometry.md),
# in the value's own tangent order and the file's units, its shape the storage
# the value holds: a d×d array of rows, finite and exactly symmetric (full);
# `{"diagonal": [d numbers]}` (an SDiagonal); `{"isotropic": λ}` (λ·I, a
# ScaledIdentity). Present iff the value is `Uncertain`; all zeros is `exact`.
# Positive semi-definiteness is not tested (zero is valid).
struct _Cov <: _Spec
    d::Int   # the dimension; 0: any (a size its record checks, as a target calibration's over its parameters)
end
_cov_spec(d::Int) = _Cov(d)
function _validate(s::_Cov, x, file, at)
    if x isa AbstractVector
        d = s.d > 0 ? s.d : length(x)
        _validate(_Vec(_Vec(_Real(), d), d), x, file, at)
        all(x[i][j] == x[j][i] for i in 1:d, j in 1:d) || _refuse(file, at, "covariance must be symmetric")
        return nothing
    end
    x isa AbstractDict || _refuse(file, at, "a covariance is an array of rows, {\"diagonal\": […]} or {\"isotropic\": λ}")
    return _validate(_Either(Pair{Symbol, _Spec}[
        :diagonal => _Obj(:diagonal => _Vec(_Real(), s.d > 0 ? s.d : -1)),
        :isotropic => _Obj(:isotropic => _Real()),
    ]), x, file, at)
end
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
# VGIo), and the producer re-exports. (`vgio/observations/v2`, a retired kind of
# per-frame pixels, is not the current `observations` kind, which starts at v3.)
const _RETIRED_KINDS = Dict(
    "vgio/observations/v2" => "vgio/observations/v3.0 (a new kind of the same name; an observations/v2 file predates it)",
    "vgio/point_observations/v1" => "vgio/observations/v3.0",
    "vgio/targets/v2" => "vgio/rig/v10.0", "vgio/rig/v3" => "vgio/rig/v10.0",
    "vgio/session/v3" => "the calibration pipeline's own session kind (a pipeline's checkpoint is the pipeline's, not VGIo's)",
    "vgio/correspondences/v2" => "export/correspondences/v2, an export (save_correspondences) that no reader reads",
)

# A kind renamed or folded into another → the kind its records now belong to:
# a record of the former kind reads through its version steps into the current
# one (and is never written). The non-central grid camera's records were the
# old grid kind's, then a kind of their own; the cameras store holds them now.
# The per-frame target observation kinds of IO 0.9 are one kind, `observations`.
const _RENAMED_KIND = Dict(
    "generic_camera" => "noncentral_spline", "noncentral_spline" => "cameras",
    "target_points2d" => "observations", "target_points3d" => "observations", "target_detections" => "observations",
)
# The current kind of a kind name, a former one's successor followed to the end.
_current_kind(k::AbstractString) = haskey(_RENAMED_KIND, k) ? _current_kind(_RENAMED_KIND[k]) : String(k)

function _refuse_schema(path, kind::AbstractString, s)
    if s isa AbstractString && haskey(_RETIRED_KINDS, s)
        throw(
            ArgumentError(
                "$(path) is a retired $(s) file; its successor is $(_RETIRED_KINDS[s]). There is no reader for it: " *
                    "re-export it from its producer"
            )
        )
    end
    # (a v2 transforms or trajectory file reaches its version step instead: the 2026-09 hand-eye
    # record and CaliBlobs' 2026-09 track segment read, any other is refused there)
    if s isa AbstractString && kind in ("transforms", "trajectory") && s in ("vgio/$(kind)/v1", "vgio/transform/v1")
        throw(
            ArgumentError(
                "$(path) is a retired $(s) file: its frame names are free strings, which $(_schema(kind)) " *
                    "types through the frame grammar (\"world\", \"camera\", \"view:<id>\", \"target:<id>\", " *
                    "\"body:<id>\"). There is no reader for it: re-export it from its producer"
            )
        )
    end
    # an earlier version of the kind with no version step (`_read_store` reads
    # every one that has one): no published file holds it, so nothing reads it
    if _is_earlier_version(s, kind)
        throw(
            ArgumentError(
                "$(path) is $(something(_split_schema(s)).kind == kind ? "a retired" : "an earlier or unreleased") $(s) file, which no migration reads (VisualGeometryToolkitIO reads " *
                    "$(_schema(kind)) and the earlier versions a published file holds: $(join(schema_versions(kind), ", "))): re-export it from its producer"
            )
        )
    end
    v = _split_schema(s)
    (v !== nothing && v.kind == kind && v.major > first(_schema_version(kind))) && throw(
        ArgumentError("$(path): a $(s) file is of a later major version than this VisualGeometryToolkitIO reads ($(_schema(kind))): update VisualGeometryToolkitIO")
    )
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
# record and read as one. A file of the current major at a HIGHER minor version
# is read with the keys that minor added tolerated (forward compatibility).
_read_store(path::AbstractString, kind::AbstractString, spec::_Spec) =
    _checked_store(_read_vgio_json(path), path, kind, spec)   # a non-finite number is refused by its field
_read_store(path::AbstractString, kind::AbstractString) = _read_store(path, kind, _spec(kind))
# …and the same for a file already parsed (`d`, its JSON), by a reader that
# chose the kind from its `schema`.
function _checked_store(d, path::AbstractString, kind::AbstractString, spec::_Spec = _spec(kind))
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    kind == "cameras" && (d = _one_camera_store(d, path))   # an old grid file or a noncentral_spline record: the store of its one camera
    s = get(d, :schema, nothing)
    _is_current_major(s, kind) || (_reads_through_migration(s, kind) && (d = _migrated_on_load(d, path, kind, s)))
    s2 = get(d, :schema, nothing)
    _is_current_major(s2, kind) || _refuse_schema(path, kind, s)
    _refuse_payload(path, d)
    v = _split_schema(s2)
    with(_TOLERATE_UNKNOWN => v.minor > last(_schema_version(kind))) do
        _validate(spec, d, path, "")
    end
    return d
end

# A schema string's owner, kind and version (`<owner>/<kind>/v<major>[.<minor>]`;
# a string without a minor, the form before IO 0.10, is minor 0), or `nothing`.
# `_split_schema` is vgio's own (`owner == "vgio"`), `_split_owned` any owner's.
function _split_owned(s)
    m = s isa AbstractString ? match(r"^([a-z][a-z0-9_]*)/([a-z0-9_]+)/v(\d+)(?:\.(\d+))?$", s) : nothing
    m === nothing && return nothing
    return (owner = String(m[1]::AbstractString), kind = String(m[2]::AbstractString), major = parse(Int, m[3]::AbstractString),
            minor = m[4] === nothing ? 0 : parse(Int, m[4]::AbstractString))
end
_split_schema(s) = (v = _split_owned(s); v === nothing || v.owner != "vgio" ? nothing : v)
_version(v) = (v.major, v.minor)
# Whether `s` is `kind` at its current major version (any minor).
_is_current_major(s, kind::AbstractString) = (v = _split_schema(s); v !== nothing && v.kind == kind && v.major == first(_schema_version(kind)))
# Whether `s` names an earlier version of `kind` (its own, or a former kind's).
_is_earlier_version(s, kind::AbstractString) = (v = _split_schema(s); v !== nothing &&
    (v.kind == kind ? v.major < first(_schema_version(kind)) : (haskey(_RENAMED_KIND, v.kind) && _current_kind(v.kind) == kind)))
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
    _is_current_major(reached, kind) || refuse("its version steps end at $(repr(reached))")
    @debug "$(path): a $(s) file, read through its migration to $(reached)"
    return _parse_vgio_text(JSON.json(t; allownan = true), path)   # (compact: parsed, never written)
end

# Write `body` (a kind's body, without its envelope) as a file of `kind` after
# validating the very bytes that land on disk against the kind's spec: a writer
# cannot produce a file its own reader refuses. The envelope — `schema`,
# `generator`, the `units` and `conventions` objects — is the style's, so no
# writer can serialize under a unit or pixel frame its file does not declare; a
# VGC value left in `body` is lowered on `style` (style.jl). `provenance` last.
function _write_kind(path::AbstractString, kind::AbstractString, body::NamedTuple, provenance::NamedTuple; style::VgioStyle)
    record = with_provenance(merge(_envelope(kind, style), body), provenance)
    text = _json_text(record, style)          # serialized once: these bytes are validated and written
    _validate(_spec(kind), _parse_vgio_text(text, path), path, "")   # (a NaN/Inf refused by its field)
    write(path, text)
    return nothing
end

# The same boundary for a record of another package's kind: the owner's
# record, which no spec here knows.

"""
    write_record(path, record; style = VgioStyle(), allownan = false)

Write `record` — a `NamedTuple` or `AbstractDict`, the record of a kind a
package owns, its VisualGeometryToolkitCore values left for `style` to lower —
as a JSON file in the shape every vgio file has (pretty-printed, a four-space
indent, no trailing newline), with the envelope every vgio file has: its
`schema` first (`"<owner>/<kind>/v<major>.<minor>"`, the record's own), then the
`generator` (VisualGeometryToolkitIO and the calling package, [`with_caller`](@ref)),
the `units` object `style` declares (its length unit and roles, density and
pixel frame) and the fixed `conventions`, then the record's other keys in their
order. The record states none of the envelope keys itself (an `ArgumentError`).
Every VGC value in it is written by its `JSON.lower(::VgioStyle, x)` method on
`style`. The text is parsed back before it is written, and a NaN or ±Infinity
is refused naming its field, as a vgio file's is; `allownan = true` writes one
as JSON.jl's `NaN` / `Infinity` instead (a run report's statistic over
nothing), for [`read_record`](@ref)`(path; allownan = true)`.

A kind's owner writes its records with this and reads them with `read_record`
(`VgioStyle(record)` is the style its `units` declare); this function checks
no kind's fields.
"""
function write_record(path::AbstractString, record; style::VgioStyle = VgioStyle(), allownan::Bool = false)
    pairs_ = record isa NamedTuple ? collect(Pair{Symbol, Any}, pairs(record)) : Pair{Symbol, Any}[Symbol(k) => v for (k, v) in record]
    names = first.(pairs_)
    :schema in names || throw(ArgumentError("write_record: the record states its `schema` (\"<owner>/<kind>/v<major>.<minor>\")"))
    for k in (:generator, :units, :conventions)
        k in names && throw(ArgumentError("write_record: `$(k)` is the envelope's, written from the style; the record does not state it"))
    end
    rest = [p for p in pairs_ if p.first !== :schema]
    out = merge((schema = record isa NamedTuple ? record.schema : record[first(k for k in keys(record) if Symbol(k) === :schema)],),
                _envelope_tail(style), NamedTuple(rest))
    text = _json_text(out, style)        # serialized once: these bytes are checked and written
    allownan || _parse_vgio_text(text, path)
    write(path, text)
    return nothing
end

"""
    read_record(path; allownan = false) -> JSON.Object
    read_record(io::IO; allownan = false) -> JSON.Object

The record a JSON file (or the JSON text `io` holds) is, as its kind's
CURRENT record: parsed (a `JSON.Object`, read by property access — the form
`JSON.lift(::VgioStyle, T, record)` takes) and, when its `schema` is an
earlier version that a schema owner's version steps read
([`migrate_schema`](@ref)), run through them by [`migrate_record`](@ref) in
memory; the file is not rewritten. A step's refusal (an `ArgumentError`, or a
key its version always held missing) is an `ArgumentError` naming the file
and its schema. A NaN or ±Infinity is refused naming its field;
`allownan = true` reads one (and then reads every number as a `Float64`,
JSON.jl's rule), for a file [`write_record`](@ref) wrote with
`allownan = true`.
"""
read_record(path::AbstractString; allownan::Bool = false) = _read_record_text(read(path, String), path; allownan)
read_record(io::IO; allownan::Bool = false) = _read_record_text(read(io, String), "the JSON text"; allownan)

function _read_record_text(text::AbstractString, where_::AbstractString; allownan::Bool)
    parse(t) = allownan ? _parse_json(t; allownan = true) : _parse_vgio_text(t, where_)
    d = parse(text)
    s = d isa AbstractDict ? get(d, :schema, nothing) : nothing
    (s isa AbstractString && _has_record_migration(s)) || return d
    t = try
        migrate_record(d)
    catch e
        e isa Union{ArgumentError, KeyError} || rethrow()
        throw(ArgumentError("$(where_): its $(s) record does not read through its version steps: $(sprint(showerror, e))"))
    end
    return parse(JSON.json(t; allownan = true))   # what a current file parses as
end

"""
    with_provenance(body::NamedTuple, provenance::NamedTuple) -> NamedTuple

A record's `body` with the writer's free-form `provenance` (the `provenance`
keyword of every `save`), as the writers store it: `body` itself when
`provenance` is empty, else `body` with the field `provenance` appended (the
last key). Provenance is about the file, never part of the value, and no
reader reads it back. A package writing a kind of its own through
[`write_record`](@ref) stores its provenance the same way.
"""
with_provenance(body::NamedTuple, provenance::NamedTuple) =
    isempty(provenance) ? body : merge(body, (provenance = provenance,))

# The spec of every kind, by the name its schema carries (the tables live
# beside their readers and writers; looked up at call time).
function _spec(kind::AbstractString)
    kind == "units" && return _UNITS_SPEC
    kind == "transforms" && return _TRANSFORMS_SPEC
    kind == "cameras" && return _CAMERAS_SPEC
    kind == "rig" && return _RIG_SPEC
    kind == "target" && return _TARGET_SPEC
    kind == "trajectory" && return _TRAJECTORY_SPEC
    kind == "target_poses" && return _TARGET_POSES_SPEC
    kind == "target_homographies" && return _TARGET_HOMOGRAPHIES_SPEC
    kind == "observations" && return _OBSERVATIONS_SPEC
    kind == "capture_settings" && return _CAPTURE_SETTINGS_SPEC
    kind == "calibration" && return _CALIBRATION_SPEC
    kind == "vignetting" && return _VIGNETTING_SPEC
    kind == "environment_light" && return _ENVIRONMENT_LIGHT_SPEC
    kind == "blur" && return _BLUR_SPEC
    kind == "panorama" && return _PANORAMA_SPEC
    kind == "detection" && return _DETECTION_SPEC
    kind == "sensor_noise" && return _SENSOR_NOISE_SPEC
    throw(ArgumentError("no vgio kind $(repr(kind))"))
end
_has_kind(kind::AbstractString) = haskey(_SCHEMA_VERSION, kind)

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
need more than the one file — that observations resolve in their targets —
are the exporters'.
"""
function validate(path::AbstractString)
    file = isdir(path) ? joinpath(path, UNITS_SIDECAR) : path
    s = _peek_schema(file)
    v = _split_schema(s)
    if v !== nothing
        kind = _current_kind(v.kind)
        if _has_kind(kind)
            (_is_current_major(s, kind) || _reads_through_migration(s, kind)) || _refuse_schema(file, kind, s)
            _read_store(file, kind)
            return kind
        end
    end
    s isa AbstractString && haskey(_RETIRED_KINDS, s) && _refuse_schema(file, something(v, (; kind = "")).kind, s)
    throw(ArgumentError("$(file): not a vgio file (schema $(repr(s)))"))
end
