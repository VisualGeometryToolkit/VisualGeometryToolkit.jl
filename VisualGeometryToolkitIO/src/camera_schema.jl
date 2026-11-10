# =============================================================================
# vgio/cameras/v5 — the camera store
# =============================================================================
#
# {"schema": "vgio/cameras/v5", "pixel_frame": "colmap",
#  "cameras": [{"name": "sony-fx3_tamron-20mm_f22", "width": 4240, "height": 2832,
#               "K": {"fx": …, "fy": …, "cx": …, "cy": …, "skew": 0.0},
#               "lens": {"perspective": [],
#                        "add": [{"radial": [[k1, k2, k3], [k4, k5, k6]]},
#                                {"tangential": [p1, p2]}, …]},
#               "calibration": {"tool": …, "source": …, "date": "2026-01-19",
#                               "reprojection_rms": 0.159, "n_images": 36},   (optional)
#               "lens_coverage": {"ellipse": [cx, cy, a, b, θ]}}],               (optional; or {"circle": [cx, cy, r]})
#  "provenance": {…}}                                               (optional)
#
# `lens_coverage` is the camera's LENS COVERAGE (VGC's `lens_coverage`): where light reaches the sensor, in pixels in the
# canonical frame — a Circle, or an Ellipse with its major semi-axis first. Absent, the whole sensor is lit. (v5;
# a v4 store, which has none, reads through its migration with the whole sensor lit.)
#
# WHY THIS EXISTS, given COLMAP model directories already store cameras:
# COLMAP's camera vocabulary is closed and smaller than VGC's lens families.
# The double sphere (Usenko-Demmel-Cremers 2018) has no COLMAP model at all, nor
# does COLMAP have a rational lift or a Scaramuzza lens. Storing cameras as
# COLMAP would cap what a downstream config can even refer to, and the failure
# would surface at export time — long after calibration. COLMAP stays an
# export for the expressible subset; this is the store.
#
# THE PIXEL FRAME IS DECLARED ONCE, for every K in the file, and it is GIVEN BY
# THE TYPE: VGC models a pixel-origin convention as `PixelFrame{Origin}` and
# owns the registered set behind `EuclideanMap(PixelFrame{From},
# PixelFrame{To})`, so the half pixel is a labeled rigid map between two frame
# types rather than an ad-hoc ±0.5 each reader has to remember. Every pixel
# quantity is in pixels, so no unit is stated beside it.
#
# THE LENS IS VGC'S RECORD: `lens` is VGC's `lens_record` — the structure and
# every coefficient, its arity the array lengths — and VGIo keeps no second
# list of lens names or orders to drift from VGC's. A camera body (`width`,
# `height`, `K`, `lens`) is the same object wherever a kind embeds a camera
# (a rig sensor; a calibration pipeline's session, through `camera_json`).

# ONE SOURCE OF TRUTH, and it is the type. This package keeps NO list of
# valid pixel frames: VGC owns the registered set (the offset table behind the
# `PixelFrame` conversion factory), and a second list here could only agree
# with it or drift from it. Validity is therefore decided by ASKING VGC, whose
# refusal already names the registered options. (`CANONICAL_PIXEL_FRAME`, the
# frame a `CameraModel`'s K is in, is stated once, in style.jl.)

"""
    pixel_frame_name(::Type{<:PixelFrame}) -> String

The serialized spelling of a pixel frame (`PixelFrame{:colmap}` → `"colmap"`);
[`pixel_frame_from_name`](@ref) is its inverse. Parallel to
[`frame_name`](@ref)/[`frame_from_name`](@ref) for 3D frames, deliberately
separate because the grammars differ: a 3D frame serializes prefixed
(`"view:cam1"`), a pixel frame bare (`"colmap"`).

```jldoctest
julia> VgtkIO.pixel_frame_name(PixelFrame{:opencv})
"opencv"
```
"""
pixel_frame_name(::Type{PixelFrame{O}}) where {O} = String(O)
# The origin symbol VGC's `pixel_frame` keyword takes (private).
pixel_origin(::Type{PixelFrame{O}}) where {O} = O

"""
    pixel_frame_from_name(name) -> Type{<:PixelFrame}

The pixel frame a file names (`"colmap"`, `"opencv"`, `"julia"`), the inverse
of [`pixel_frame_name`](@ref). VGC decides which pixel frames exist: the name
is checked by constructing the conversion to the canonical pixel frame
(`colmap`), and a name VGC does not know is an `ArgumentError` that quotes
VGC's list of registered frames.

```jldoctest
julia> VgtkIO.pixel_frame_from_name("julia")
PixelFrame{:julia}
```
"""
function pixel_frame_from_name(name::AbstractString)
    C = PixelFrame{Symbol(name)}
    try
        EuclideanMap(C, CANONICAL_PIXEL_FRAME)
    catch e
        throw(ArgumentError("unknown pixel frame $(repr(String(name))) — $(sprint(showerror, e))"))
    end
    return C
end

# -----------------------------------------------------------------------------
# Lenses — VGC's lens record
# -----------------------------------------------------------------------------
#
# A lens is stored as VGC's `lens_record`: every key a leaf tag or a combinator,
# every value a coefficient array whose length is the arity —
#
#   {"equidistant": [], "radial": [[k1, k2, k3, k4], []]}
#   {"perspective": [], "add": [{"radial": [[k1, k2], []]}, {"tangential": [p1, p2]}]}
#   {"lift": {"radial": [[], [k1]]}}
#
# — and read back by `lens_from_record`, which refuses any other shape by
# name. The record determines the lens type; JSON's shortest round-trip floats
# make a Float64 lens read back bitwise. VGIo keeps no list of lens names or
# orders of its own.

_lens_json(l) = applicable(lens_record, l) ? lens_record(l) :
    throw(ArgumentError("no lens record for the lens type $(typeof(l)) — the store holds VGC's Lens, LiftLens and SplineLens"))

# THE SPLINE LENS RECORD. A `SplineLens` (the central grid camera, Schöps et al.)
# is stored under its own key, in the K-normalized plane x = K⁻¹u (K is the
# camera body's, a gauge of skew 0):
#
#   {"spline": {"area": [x, y, w, h], "grid_size": [gw, gh],
#               "present": [[row, [[col_lo, col_hi], …]], …],     (optional) the held nodes; absent = every node
#               "controls": [[x, y, z], …],                      the held nodes' direction controls, row by row
#               "forward": {"area": [x, y, w, h], "grid_size": [gw, gh],
#                           "controls": [[x, y], …]}}}           (optional) the fitted forward spline, every node
#
# `area` is the backward spline's rectangle of the normalized plane, `grid_size`
# its lattice (columns, rows), the controls as stored (never re-normalized:
# a fit's controls may be off the sphere, and normalizing would change the
# spline). The forward spline (from the equidistant chart about ẑ
# to x) is stored exactly when it has been fitted (owner, 2026-11-09). Floats
# are JSON's shortest round-trip form, so a lens reads back bit for bit.
_spline_json(cs::ChartSpline, P::AbstractMatrix{Bool}) =
    merge((area = JSON.lower(VgioStyle(), cs.area), grid_size = [size(cs.controls, 2), size(cs.controls, 1)]),
          all(P) ? (;) : (present = _present_runs(P),), (controls = _controls_json(cs.controls, P),))
function _lens_json(l::SplineLens)
    b = _spline_json(l.backward, _held_mask(l.present, size(l.backward.controls)))
    f = l.forward === nothing ? (;) : (forward = _spline_json(l.forward, trues(size(l.forward.controls))),)
    return (spline = merge(b, f),)
end

# …and back, from the parsed record (`_plain`'s Dict), checked as it is read: an
# ArgumentError names what is wrong (the validator's message, `_lens_check`).
function _spline_lens_from_json(r::Dict{String, Any})
    what = "a spline lens record"
    length(r) == 1 || throw(ArgumentError("$(what) is {\"spline\": {…}} alone; got the keys $(join(sort!(collect(keys(r))), ", "))"))
    s = r["spline"]
    s isa AbstractDict || throw(ArgumentError("$(what): \"spline\" is an object"))
    b = _spline_from_json(s, Val(3), ("present", "forward"), "$(what)'s \"spline\"")
    forward = haskey(s, "forward") ? _spline_from_json(s["forward"], Val(2), (), "$(what)'s \"forward\"").spline : nothing
    return SplineLens(b.spline; forward, present = b.present)
end

# A spline record (`area`, `grid_size`, its controls of `K`-vectors, the optional
# keys `optional`, `present` among them) as `(; spline::ChartSpline, present)`,
# `present` `nothing` for every node. Validated against the spec below and the
# lattice's own check (`_lattice_problem`), named `what` in the message.
function _spline_from_json(s::AbstractDict, ::Val{K}, optional, what) where {K}
    spec = _Obj(
        Pair{Symbol, _Spec}[:area => _Vec(_Real(), 4), :grid_size => _Vec(_PosInt(), 2), :controls => _Vec(_Vec(_Real(), K))],
        Pair{Symbol, _Spec}[Symbol(k) => (k == "present" ? _RowRuns() : _Free()) for k in optional],
    )
    _validate(spec, s, what, "")
    msg = _lattice_problem(s, (:controls,))
    msg === nothing || throw(ArgumentError("$(what): $(msg)"))
    gw, gh = Int(s["grid_size"][1]), Int(s["grid_size"][2])
    P = haskey(s, "present") ? _present_from_runs(s["present"], gw, gh) : trues(gh, gw)
    a = s["area"]
    area = Rect2{Float64}(Float64(a[1]), Float64(a[2]), Float64(a[3]), Float64(a[4]))
    return (; spline = ChartSpline(area, _controls_from_json(s["controls"], P, Val(K))), present = haskey(s, "present") ? P : nothing)
end

# Parsed JSON objects (`JSON.Object`s) as the plain Dicts and Vectors the
# lens record is.
_plain(x::AbstractDict) = Dict{String, Any}(String(k) => _plain(v) for (k, v) in x)
_plain(x::NamedTuple) = Dict{String, Any}(String(k) => _plain(v) for (k, v) in pairs(x))   # a record built in memory (`_lens_json`'s)
_plain(x::AbstractVector) = Any[_plain(v) for v in x]
_plain(x) = x

function _lens_from_json(x)
    x isa Union{AbstractDict, NamedTuple} || throw(ArgumentError("a lens record is an object"))
    x = _plain(x)
    # the v1/v2 lens shape, `{"model": …, "params": …}`: read only in a v1/v2
    # file, through its version steps, never in a current record
    (haskey(x, "model") || haskey(x, :model)) && throw(
        ArgumentError(
            "a {\"model\", \"params\"} lens is a retired v1/v2 lens record, which predates the lens " *
                "families: a current record holds VGC's lens record (a vgio/cameras/v1 or v2 file is read through its migration)"
        )
    )
    # the retired "chain" (terms in sequence in application order, cameras v3 /
    # rig v5): its key and order changed together, so it is never read as "compose"
    # in a current record (a v3/v5 file is read through its migration)
    _has_chain(x) && throw(
        ArgumentError(
            "a lens record's \"chain\" (its stages in application order) is the retired vgio/cameras/v3 / " *
                "vgio/rig/v5 form, which a file of that version is read through its migration for (vgio_migrate " *
                "rewrites it): a current record lists the stages under \"compose\", in CoordinateTransformations' order, the outer first"
        )
    )
    r = x::Dict{String, Any}
    haskey(r, "spline") && return _spline_lens_from_json(r)
    return lens_from_record(r)
end
_has_chain(x) = _holds_key(x, "chain")
# whether a parsed record holds `key` at any depth
_holds_key(x::AbstractDict, key::String) = haskey(x, key) || haskey(x, Symbol(key)) || any(v -> _holds_key(v, key), values(x))
_holds_key(x::AbstractVector, key::String) = any(v -> _holds_key(v, key), x)
_holds_key(x, ::String) = false

# The lens record's own check, run by the validator on read AND write.
function _lens_check(x)
    try
        _lens_from_json(x)
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return nothing
end

const _LENS = _Checked(_Free(), _lens_check)

# -----------------------------------------------------------------------------
# A camera body: its pixel extent, K and lens, in the file's pixel frame
# -----------------------------------------------------------------------------

const _K = _Obj(:fx => _Real(), :fy => _Real(), :cx => _Real(), :cy => _Real(), :skew => _Real())
const _CAMERA_BODY_FIELDS = Pair{Symbol, _Spec}[:width => _PosInt(), :height => _PosInt(), :K => _K, :lens => _LENS]
const _CAMERA_BODY = _Obj(_CAMERA_BODY_FIELDS, Pair{Symbol, _Spec}[])

# A camera bounded by its physical `Sensor` carries the sensor's pixel pitch
# beside its body: `"pitch": [width, height]`, a length per pixel in the file's
# declared `length_unit` (px is dimensionless, so a pitch IS a length;
# `write_length`/`read_length`, decimal-exact). Absent, the body is the bare
# extent's, byte for byte. Written by the rig (sensor bodies, parametric and
# generic) and by the camera store; never dropped.
const _PITCH = _Checked(
    _Vec(_Real(), 2),
    x -> all(v -> v > 0 && isfinite(v), x) ? nothing :
        "a pixel pitch is [width, height], two positive lengths per pixel in the file's length_unit",
)
_with_pitch(body, ::Nothing, ::_LeafUnit) = body
_with_pitch(body, s::Sensor, unit::_LeafUnit) = merge(body, (pitch = [_leaf_length(s.pitch.width, unit), _leaf_length(s.pitch.height, unit)],))
# The extent, or the physical sensor when the body carries a pitch.
_bound_from_body(b, ext::Size2, u::Units) = haskey(b, :pitch) ? _sensor_from_pitch(b.pitch, ext, u) : ext
function _sensor_from_pitch(p, ext::Size2, u::Units)
    pw, ph = read_length(p[1]::Real, u), read_length(p[2]::Real, u)
    return Sensor(Size2(width = ext.width * px, height = ext.height * px), Size2(width = pw / px, height = ph / px))
end

# A CameraModel on IO's style (style.jl): its K — canonical by construction —
# with the principal point in the style's pixel frame (the file's declaration;
# a frame-typed container written in another moves it with the declaration),
# and its lens record. A body is its pixel extent and these.
JSON.lower(s::VgioStyle, model::CameraModel) = (K = _k_record(model.intrinsics.K, _pixel_frame(s)), lens = _lens_json(model.lens))
_camera_body(s::VgioStyle, model::CameraModel, extent::Size2) =
    merge((width = Int(extent.width), height = Int(extent.height)), JSON.lower(s, model))

# …and back, its K read in the style's pixel frame and moved into the canonical
# one by `_k_from_record(k, frame)` (style.jl), the exact inverse of `_k_record`
# and the one conversion of a stored K, a camera's and a non-central grid
# camera's alike (a SplineLens lens is frame-free: x = K⁻¹u).
JSON.lift(s::VgioStyle, ::Type{CameraModel}, b::_Record) =
    CameraModel(LogicalIntrinsics(_k_from_record(b.K, _pixel_frame(s))), _lens_from_json(b.lens))
_extent_from_body(b) = Size2(Int(b.width::Integer), Int(b.height::Integer))

"""
    camera_json(model::CameraModel, extent, frame = CANONICAL_PIXEL_FRAME) -> NamedTuple
    camera_from_json(body, frame) -> (model, extent)

The camera body every camera-carrying kind embeds — `width`, `height`, `K`
(`fx, fy, cx, cy, skew`) and the `lens` record — written with its principal
point in the pixel `frame` the enclosing file declares, and read back into
VGC's canonical frame. Building blocks for a package serializing its own kind.
"""
camera_json(model::CameraModel, extent::Size2, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME) =
    _camera_body(VgioStyle(; pixel_frame = frame), model, extent)
"""
    camera_from_json(body, frame) -> (model, extent)

Read a camera body (the inverse of [`camera_json`](@ref)) whose principal
point is in the pixel `frame` the enclosing file declares. The returned
`CameraModel` is in VGC's canonical pixel frame.
"""
camera_from_json(body, frame::Type{<:PixelFrame}) =
    (JSON.lift(VgioStyle(; pixel_frame = frame), CameraModel, body::_Record), _extent_from_body(body))

# -----------------------------------------------------------------------------
# The calibration block — where a stored camera's numbers came from
# -----------------------------------------------------------------------------
#
# A camera's own calibration facts are an ATTRIBUTE of the value (the metadata
# rule: envelope for the file, attributes for the value):
#
#     Attributed(bc::BoundedCamera, (; calibration = (; tool = "OpenCV calibrateCamera",
#                                                     date = "2026-05-27", reprojection_rms = 0.21)))
#
# and the camera's `calibration` block on disk, read back exactly. Its keys
# form a closed set, every one optional but at least one stated:
#
#   tool              what fitted it
#   source            the file or record it was converted from — a name, never
#                     a machine path (an absolute path is refused)
#   date              when it was fitted, ISO 8601 `YYYY-MM-DD`
#   reprojection_rms  the fit's RMS reprojection error, in pixels (≥ 0)
#   n_images          how many images the fit used
#   n_points          how many image points the fit used
#   note              free text
#   device            the hardware it was made for: `{"body": {…}, "lens": {…}}`,
#                     each a VGC `DeviceIdentity` (`make`, `model`, `serial`,
#                     each stated or omitted); in the value
#                     `(; body::DeviceIdentity, lens::DeviceIdentity)`
#
# Unlike the file's free-form `provenance`, which VGIo never reads, the block
# is part of the value: an unknown key is refused like any other. A block
# without `device` (every file before it) reads as it did.

const _CALIBRATION_KEYS = (:tool, :source, :date, :reprojection_rms, :n_images, :n_points, :note, :device)
# Each key's reader, wire value → the value's (a type converts a scalar)
const _CALIBRATION_READERS = (String, String, String, Float64, Int, Int, String, x -> _device_from_json(x))

# A DeviceIdentity on the wire: its stated fields, the unknown ones omitted.
const _DEVICE_IDENTITY_FIELDS = Pair{Symbol, _Spec}[:make => _NonEmpty(), :model => _NonEmpty(), :serial => _NonEmpty()]
const _DEVICE_IDENTITY = _Obj(Pair{Symbol, _Spec}[], _DEVICE_IDENTITY_FIELDS)
_identity_json(d::DeviceIdentity) =
    NamedTuple(k => v for (k, v) in ((:make, d.make), (:model, d.model), (:serial, d.serial)) if v !== nothing)
_identity_from_json(x) = DeviceIdentity(get(x, :make, nothing), get(x, :model, nothing), get(x, :serial, nothing))
_device_json(d) = (body = _identity_json(d.body), lens = _identity_json(d.lens))
_device_from_json(x) = (body = _identity_from_json(x.body), lens = _identity_from_json(x.lens))
const _Device = NamedTuple{(:body, :lens), <:Tuple{DeviceIdentity, DeviceIdentity}}   # a calibration's device, in the value
_is_device(d) = d isa _Device

# The checks the field types cannot make — on write and on read.
function _calibration_check(x)
    isempty(x) && return "a calibration block states at least one of $(join(_CALIBRATION_KEYS, ", "))"
    for k in keys(x)
        Symbol(k) in _CALIBRATION_KEYS || return "unknown calibration key `$(k)` (one of $(join(_CALIBRATION_KEYS, ", ")))"
    end
    s = get(x, :source, nothing)
    (s isa AbstractString && (startswith(s, '/') || startswith(s, '~') || occursin(r"^[A-Za-z]:[\\/]", s))) &&
        return "`source` names a file or record, not a machine path: got $(repr(String(s)))"
    d = get(x, :date, nothing)
    (d isa AbstractString && !occursin(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", d)) &&
        return "`date` is ISO 8601 YYYY-MM-DD, got $(repr(String(d)))"
    e = get(x, :reprojection_rms, nothing)
    (e isa Real && e < 0) && return "`reprojection_rms` is an RMS in pixels, ≥ 0; got $(e)"
    for k in (:tool, :source, :note)
        v = get(x, k, nothing)
        (v isa AbstractString && isempty(v)) && return "`$(k)` is empty — omit what is unknown"
    end
    for k in (:n_images, :n_points)
        v = get(x, k, nothing)
        (v isa Integer && v <= 0) && return "`$(k)` is a positive count, got $(v)"
    end
    d = get(x, :device, nothing)
    (x isa NamedTuple && d !== nothing && !_is_device(d)) &&
        return "`device` is `(; body::DeviceIdentity, lens::DeviceIdentity)`, got $(repr(d))"
    return nothing
end

const _CALIBRATION = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[],
        Pair{Symbol, _Spec}[
            :tool => _NonEmpty(), :source => _NonEmpty(), :date => _NonEmpty(), :reprojection_rms => _Real(),
            :n_images => _PosInt(), :n_points => _PosInt(), :note => _NonEmpty(),
            :device => _Obj(:body => _DEVICE_IDENTITY, :lens => _DEVICE_IDENTITY),
        ],
    ),
    _calibration_check,
)

# The block as the value holds it: the stated keys in declaration order, typed.
_calibration_from_json(x) = NamedTuple(k => read(x[k]) for (k, read) in zip(_CALIBRATION_KEYS, _CALIBRATION_READERS) if haskey(x, k))
# …and back: a scalar as it is, a device as its two identities
_calibration_value_json(v) = v
_calibration_value_json(d::_Device) = _device_json(d)
function _calibration_json(c::NamedTuple, what::AbstractString = "a camera's calibration block")
    msg = _calibration_check(c)
    msg === nothing || throw(ArgumentError("$(what): $(msg)"))
    return NamedTuple(k => _calibration_value_json(getfield(c, k)) for k in _CALIBRATION_KEYS if haskey(c, k))
end

# -----------------------------------------------------------------------------
# The cameras store ↔ Dict{Symbol, BoundedCamera}
# -----------------------------------------------------------------------------
#
# THE VALUE: a store of named, unposed cameras — `Dict{Symbol, …}` from the
# camera's name (the identity a config refers to) to a VGC `BoundedCamera` of a
# `CameraModel` (COLMAP's "camera" row: intrinsics, lens, image size, no pose),
# or an `Attributed` one carrying its `calibration` block. Once a pose is known,
# `BoundedCamera(PosedCamera(model(bc), pose), bc.bound; lens_coverage = lens_coverage(bc))` is the posed camera.
#
# No pixel frame is stored per camera: a model's K is in VGC's canonical frame
# by construction. On disk the store still names its pixel frame, because a
# FOREIGN file may declare `opencv` or `julia`, converted on the way in.

const _StoredCamera = Union{BoundedCamera{<:CameraModel}, Attributed{<:BoundedCamera{<:CameraModel}, <:NamedTuple{(:calibration,)}}}

# A camera's lens coverage (VisualGeometryToolkitCore's `lens_coverage`: where the lens covers the sensor) as the store's
# optional `lens_coverage`, pixels in the store's frame (moved between frames as K's principal point is):
# {"circle": [cx, cy, r]}, {"ellipse": [cx, cy, a, b, θ]} (VGC's `Ellipse` numbers, the major semi-axis first) or
# {"rect": [x, y, w, h]} (a letterboxed or padded image). The whole sensor covered writes nothing.
const _LENS_COVERAGE = _Obj(Pair{Symbol, _Spec}[],
                            Pair{Symbol, _Spec}[:circle => _Vec(_Real(), 3), :ellipse => _Vec(_Real(), 5), :rect => _Vec(_Real(), 4)])
_lens_coverage_tag(::Circle) = :circle
_lens_coverage_tag(::Ellipse) = :ellipse
_lens_coverage_tag(::Rect2) = :rect
# pixel frames differ by a translation: a Circle or an Ellipse moves under the map, a Rect by its origin
_in_frame(m, x::Union{Circle, Ellipse}) = m(x)
_in_frame(m, r::Rect2) = (o = m(Point2{Float64}(r.origin[1], r.origin[2])); Rect2{Float64}(o[1], o[2], r.widths[1], r.widths[2]))
_lens_coverage_json(s::VgioStyle, r, ext::Size2) =
    r == Rect2{Float64}(0, 0, ext.width, ext.height) ? (;) :
    (lens_coverage = NamedTuple{(_lens_coverage_tag(r),)}((JSON.lower(s, _in_frame(EuclideanMap(CANONICAL_PIXEL_FRAME, _pixel_frame(s)), r)),)),)
# A stored lens coverage's own check (a camera store's camera, a rig sensor, a non-central record), the
# message naming `what`, or `nothing`.
function _lens_coverage_problem(ic, what)
    ic === nothing && return nothing
    length(ic) == 1 || return "$(what): its lens_coverage is ONE of {\"circle\": [cx, cy, r]}, {\"ellipse\": [cx, cy, a, b, θ]} or {\"rect\": [x, y, w, h]}"
    haskey(ic, :circle) && !(ic[:circle][3] > 0) && return "$(what): the image circle's radius is positive"
    haskey(ic, :rect) && !(ic[:rect][3] > 0 && ic[:rect][4] > 0) && return "$(what): the coverage rect's width and height are positive"
    return haskey(ic, :ellipse) ? _ellipse_record_problem(ic[:ellipse], "$(what): the image ellipse") : nothing
end
_lens_coverage_from_json(::VgioStyle, ::Nothing) = nothing
function _lens_coverage_from_json(s::VgioStyle, lc)
    k = Symbol(only(keys(lc)))
    x = JSON.lift(s, k === :circle ? Circle : k === :ellipse ? Ellipse : Rect2, lc[k]::Vector{Any})
    return _in_frame(EuclideanMap(_pixel_frame(s), CANONICAL_PIXEL_FRAME), x)
end

const _CAMERAS_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("cameras"), :pixel_frame => _PIXEL_FRAME,
            :cameras => _Vec(_Obj(Pair{Symbol, _Spec}[:name => _NonEmpty(), _CAMERA_BODY_FIELDS...],
                                  Pair{Symbol, _Spec}[:calibration => _CALIBRATION, :pitch => _PITCH, :lens_coverage => _LENS_COVERAGE])),
        ],
        Pair{Symbol, _Spec}[_PROVENANCE..., :length_unit => _LENGTH_UNIT],
    ),
    function (x)
        allunique(c[:name] for c in x[:cameras]) || return "camera names must be unique — they are the identity configs refer to"
        any(c -> haskey(c, :pitch), x[:cameras]) == haskey(x, :length_unit) ||
            return "`length_unit` is declared iff a camera carries a pitch"
        for c in x[:cameras]
            p = _lens_coverage_problem(get(c, :lens_coverage, nothing), "camera $(c[:name])")
            p === nothing || return p
        end
        return nothing
    end,
)

_bounded(c::BoundedCamera) = c
_bounded(c::Attributed) = getfield(c, :value)

function _camera_entry(s::VgioStyle, name::Symbol, c::_StoredCamera)
    bc = _bounded(c)
    e = merge((name = String(name),), _with_pitch(_camera_body(s, bc.camera, extent(bc)), sensor(bc), s.length_unit),
              _lens_coverage_json(s, lens_coverage(bc), extent(bc)))
    return c isa Attributed ? merge(e, (calibration = _calibration_json(c.calibration),)) : e
end

"""
    save(path, cameras::AbstractDict{Symbol}; length_unit = mm, provenance = (;))

Write a camera store — name → `BoundedCamera` of a `CameraModel`, or an
`Attributed` one carrying its `calibration` block — as a `vgio/cameras/v5`
file, cameras sorted by name. A camera bounded by its physical `Sensor`
carries the sensor's `pitch` in `length_unit`, which the file then declares. `provenance` is the file's one free-form object,
which VGIo never reads back.

JSON only, deliberately: cameras are configuration-adjacent (referenced by
name from a TOML, hand-inspected, diffed on recalibration), the payload is
kilobytes at rig scale, and Julia prints Float64 in its shortest
round-trippable form so nothing is lost.
"""
function save(path::AbstractString, cameras::AbstractDict{Symbol}; length_unit::Units = mm, provenance::NamedTuple = (;))
    for (name, c) in cameras
        c isa _StoredCamera || throw(
            ArgumentError(
                "camera $(name): a stored camera is a BoundedCamera of a CameraModel (or one Attributed with " *
                    "`(; calibration)`); got a $(typeof(c))"
            )
        )
        isempty(String(name)) && throw(ArgumentError("a camera name is non-empty — it is the identity a config refers to"))
    end
    pitched = any(c -> sensor(_bounded(c)) !== nothing, values(cameras))
    s = VgioStyle(; length_unit)
    body = merge(
        (schema = _schema("cameras"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME)),
        pitched ? (length_unit = unit_string(length_unit),) : (;),
        (cameras = [_camera_entry(s, k, cameras[k]) for k in sort!(collect(keys(cameras)))],),
    )
    _write_store(path, _CAMERAS_SPEC, _with_provenance(body, provenance); style = s)
    return nothing
end

"""
    load(Dict{Symbol, BoundedCamera}, path) -> Dict{Symbol, Union{BoundedCamera, Attributed{<:BoundedCamera}}}

Read a `vgio/cameras/v5` store: name → `BoundedCamera` of a `CameraModel`
(its K moved from the file's declared pixel frame into the canonical one), an
`Attributed` one where the file states the camera's `calibration` block. A
v1–v4 store is read through its version steps (`migrate_record`), and a
central `vgio/generic_camera/v2` or `v3` file as the store of its one camera,
named after the file's basename (`fisheye_14mm.json` → `:fisheye_14mm`; a
`CameraModel` with a `SplineLens`: generic_camera_schema.jl).
"""
load(::Type{Dict{Symbol, BoundedCamera}}, path::AbstractString) = _load_store(_read_vgio_json(path), path)
# The store of a parsed file (`d`, its JSON).
function _load_store(d, path::AbstractString)
    _is_central_grid(d) && (d = _central_grid_store(d, path))   # a vgio/generic_camera/v2 or v3 central file: a store of its camera
    d = _checked_store(d, path, "cameras", _CAMERAS_SPEC)
    s = VgioStyle(; pixel_frame = pixel_frame_from_name(d.pixel_frame::String))
    out = Dict{Symbol, _StoredCamera}()
    for c in d.cameras::Vector{Any}
        ext = _extent_from_body(c)
        bc = BoundedCamera(JSON.lift(s, CameraModel, c::_Record), haskey(c, :pitch) ? _bound_from_body(c, ext, parse_unit(d.length_unit::String)) : ext;
                           lens_coverage = _lens_coverage_from_json(s, get(c, :lens_coverage, nothing)))
        out[Symbol(c.name::String)] = haskey(c, :calibration) ?
            Attributed(bc, (; calibration = _calibration_from_json(c.calibration::AbstractDict))) : bc
    end
    return out
end

"""
    load(BoundedCamera, path; name = nothing, file_unit = nothing) -> BoundedCamera
    load(CameraModel,  path; name = nothing) -> CameraModel

Read ONE camera from a `vgio/cameras/v5` store — the bounded camera, or just
its model (its `calibration` block, if any, is the store's to read). Without
`name` the store must hold exactly one camera: a multi-camera store is
REFUSED with the available names rather than defaulting to the first —
which camera a run uses is an identity, and silently guessing it is how a
config ends up calibrated for the wrong lens while reporting success.

The grid cameras of Schöps et al. read through the same call, by the file's
content: a central `vgio/generic_camera/v2` or `v3` file as its camera with a
`SplineLens` (its nominal pinhole the K, else the lattice's gauge; its domain
the lens coverage), a `vgio/generic_camera` non-central record as a
`BoundedCamera{<:NoncentralSplineModel}`, and the reference implementation's
`intrinsics0.yaml` as either (schoeps.jl; a non-central YAML's `point_grid`
unit is the caller's `file_unit`). Old grid files are read, never rewritten.

`load(CameraModel, path)` refuses a camera whose lens coverage is not its whole
sensor (an image circle, an old grid file's domain) rather than drop it: load
the `BoundedCamera`.
"""
function load(
        ::Type{BoundedCamera}, path::AbstractString;
        name::Union{Nothing, AbstractString, Symbol} = nothing, file_unit::Union{Nothing, Units} = nothing
    )
    if _file_form(path) !== :json                       # a Schöps intrinsics0.yaml (schoeps.jl)
        name === nothing || throw(ArgumentError("$(path): a Schöps YAML holds one camera; `name` names a camera of a store"))
        return _load_schoeps_camera(path, file_unit)
    end
    file_unit === nothing ||
        throw(ArgumentError("$(path): a vgio file declares its own length_unit; `file_unit` is for a Schöps YAML"))
    d = _read_vgio_json(path)                           # parsed once (a grid file is large), read as the kind it is
    if _is_noncentral_grid(d)                           # a vgio/generic_camera record (generic_camera_schema.jl)
        name === nothing || throw(ArgumentError("$(path) holds one non-central camera; `name` names a camera of a store"))
        return _load_noncentral(d, path)
    end
    store = _load_store(d, path)
    if name === nothing
        length(store) == 1 || throw(
            ArgumentError("$(path) holds $(length(store)) cameras; supply `name` — one of $(sort!(String.(collect(keys(store)))))")
        )
        return _bounded(only(values(store)))
    end
    haskey(store, Symbol(name)) || throw(
        ArgumentError("no camera named $(repr(String(name))); available: $(sort!(String.(collect(keys(store)))))")
    )
    return _bounded(store[Symbol(name)])
end

function load(::Type{CameraModel}, path::AbstractString; name::Union{Nothing, AbstractString, Symbol} = nothing)
    bc = load(BoundedCamera, path; name)
    m = model(bc)
    m isa CameraModel || throw(ArgumentError("$(path) holds a $(nameof(typeof(m))), not a CameraModel: load(BoundedCamera, path)"))
    lens_coverage(bc) == Rect2{Float64}(0, 0, extent(bc).width, extent(bc).height) || throw(
        ArgumentError(
            "$(path): the camera's lens coverage ($(lens_coverage(bc))) is not its whole sensor, and a CameraModel has none — " *
                "load(BoundedCamera, path) keeps it"
        )
    )
    return m
end

# THE refusal of every exporter to a foreign camera form (OpenCV's camera matrix
# and distortion, COLMAP's camera models): a SplineLens camera has none, and an
# approximation would be a different camera under its name (owner, 2026-11-09).
_refuse_spline_export(::SplineLens) = throw(
    ArgumentError(
        "a SplineLens camera has no OpenCV/COLMAP form; store it as a vgio camera record (VisualGeometryToolkitIO.save), " *
            "or fit a parametric lens to it first, `VisualGeometryToolkitEstimation.fit_camera(KannalaBrandt{4}, camera, region)` " *
            "(any lens family; the region of its image to fit over)"
    )
)
_refuse_spline_export(_) = nothing

"""
    colmap_cameras(cameras::AbstractDict{Symbol}) -> (cameras::Vector{ColmapCamera}, ids::Dict{Symbol, Int})

Project a camera store onto COLMAP camera rows, for feeding a `ColmapModel`.
Ids are assigned 1..n in name order.

TWO LOSSES, both deliberate and both surfaced rather than papered over:

1. **Names do not survive.** COLMAP has no camera name field, so the identity
   a config refers to is gone on the far side. The returned `ids` map is the
   remainder — persist it beside the model if anything downstream needs to
   get back to a name.
2. **Inexpressible cameras throw.** A lens no COLMAP model computes has no
   [`colmap_model`](@ref), and that ArgumentError is the correct outcome — a
   silently dropped or approximated lens model is worse than a failed export.
   Filter first if a partial export is what you want:
   `filter(p -> colmap_model(lens(last(p))) !== nothing, cameras)`.
"""
function colmap_cameras(cameras::AbstractDict{Symbol})
    names = sort!(collect(keys(cameras)))
    cams = [
        (bc = _bounded(cameras[n]); e = extent(bc); colmap_camera(bc.camera, i, e.width, e.height))
            for (i, n) in enumerate(names)
    ]
    return cams, Dict(n => i for (i, n) in enumerate(names))
end
