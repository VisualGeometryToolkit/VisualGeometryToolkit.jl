# =============================================================================
# vgio/cameras/v6.0 — the camera store
# =============================================================================
#
# {"schema": "vgio/cameras/v6.0", <envelope: generator, units, conventions>,
#  "cameras": [{"name": "sony-fx3_tamron-20mm_f22", "image_size": {"width": 4240, "height": 2832},
#               "K": {"fx": …, "fy": …, "cx": …, "cy": …, "skew": 0.0},
#               "lens": {"perspective": [],
#                        "add": [{"radial": [[k1, k2, k3], [k4, k5, k6]]},
#                                {"tangential": [p1, p2]}, …]},
#               "pitch": {"width": …, "height": …},                              (optional) the sensor's pixel pitch, units.length
#               "lens_coverage": {"ellipse": [cx, cy, a, b, θ]},                 (optional; or {"circle": [cx, cy, r]}, {"rect": …})
#               "calibration_record": {"tool": …, "source": …, "date": "2026-01-19",
#                                      "reprojection_rms": 0.159, "n_images": 36}},   (optional)
#              {"name": "tamron-17cm-noncentral", "type": "noncentral_spline", "image_size": {…}, "K": {…},
#               "spline": {"area": [x, y, w, h], "grid_size": {"width": gw, "height": gh}, "present": […],
#                          "controls": {"direction": [[x, y, z], …], "point": [[x, y, z], …]}}}],
#  "provenance": {…}}                                                             (optional)
#
# THE STORE HOLDS ANY BOUNDED CAMERA, central or non-central, as a rig's sensors
# do: a camera of a `CameraModel` (any lens family, a `SplineLens` included) is
# a camera body, a non-central grid camera (`NoncentralSplineModel`) the
# non-central body, told apart by its `type`. One body codec serves the store,
# a rig sensor and another package's record (`camera_json`). (The non-central
# camera's own kind, `vgio/noncentral_spline/v1`, is read as a store of its one
# camera, named after its file, and never written.)
#
# `lens_coverage` is the camera's LENS COVERAGE (VGC's `lens_coverage`): where light reaches the sensor, in pixels in
# the file's pixel frame — a Circle, an Ellipse with its major semi-axis first, or a Rect. Absent, the whole sensor is lit.
#
# WHY THIS EXISTS, given COLMAP model directories already store cameras:
# COLMAP's camera vocabulary is closed and smaller than VGC's lens families.
# The double sphere (Usenko-Demmel-Cremers 2018) has no COLMAP model at all, nor
# does COLMAP have a rational lift or a Scaramuzza lens. Storing cameras as
# COLMAP would cap what a downstream config can even refer to, and the failure
# would surface at export time — long after calibration. COLMAP stays an
# export for the expressible subset; this is the store.
#
# THE PIXEL FRAME IS DECLARED ONCE, for every K in the file (`units.pixel`), and
# it is GIVEN BY THE TYPE: VGC models a pixel-origin convention as
# `PixelFrame{Origin}` and owns the registered set behind
# `EuclideanMap(PixelFrame{From}, PixelFrame{To})`, so the half pixel is a
# labeled rigid map between two frame types rather than an ad-hoc ±0.5 each
# reader has to remember. Every pixel quantity is in pixels.
#
# THE LENS IS VGC'S RECORD: `lens` is VGC's `lens_record` — the structure and
# every coefficient, its arity the array lengths — and VGIo keeps no second
# list of lens names or orders to drift from VGC's.

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
# camera body's, a gauge of skew 0), as the ONE lattice record (validate.jl):
#
#   {"spline": {"area": [x, y, w, h], "grid_size": {"width": gw, "height": gh},
#               "present": [[row, [[col_lo, col_hi], …]], …],     (optional) the held nodes, 1-based; absent = every node
#               "controls": {"direction": [[x, y, z], …]},       the held nodes' direction controls, row by row
#               "forward": {"area": [x, y, w, h], "grid_size": {…},
#                           "controls": {"point": [[x, y], …]}}}}  (optional) the fitted forward spline, every node
#
# `area` is the backward spline's rectangle of the normalized plane, `grid_size`
# its lattice (columns, rows), the controls as stored (never re-normalized:
# a fit's controls may be off the sphere, and normalizing would change the
# spline). The forward spline (from the equidistant chart about ẑ to the
# normalized-plane point x) is stored exactly when it has been fitted (owner,
# 2026-11-09). Floats are JSON's shortest round-trip form, so a lens reads back
# bit for bit.
const _SPLINE_LATTICE = _lattice_spec([:direction => 3])
const _FORWARD_LATTICE = _lattice_spec([:point => 2])
function _lens_json(l::SplineLens)
    P = _held_mask(l.present, size(l.backward.controls))
    b = _lattice_json(l.backward.area, P, (direction = (l.backward.controls, identity),))
    f = l.forward === nothing ? (;) :
        (forward = _lattice_json(l.forward.area, trues(size(l.forward.controls)), (point = (l.forward.controls, identity),)),)
    return (spline = merge(b, f),)
end

# …and back, from the parsed record (`_plain`'s Dict), checked as it is read: an
# ArgumentError names what is wrong (the validator's message, `_lens_check`).
function _spline_lens_from_json(r::Dict{String, Any})
    what = "a spline lens record"
    length(r) == 1 || throw(ArgumentError("$(what) is {\"spline\": {…}} alone; got the keys $(join(sort!(collect(keys(r))), ", "))"))
    s = r["spline"]
    s isa AbstractDict || throw(ArgumentError("$(what): \"spline\" is an object"))
    fw = get(s, "forward", nothing)
    b = Dict{String, Any}(k => v for (k, v) in s if k != "forward")
    _validate(_SPLINE_LATTICE, b, "$(what)'s \"spline\"", "")
    L = _lattice_from_json(_namedtuple(b))
    back = ChartSpline(L.area, _controls_from_json(b["controls"]["direction"], L.P, Val(3)))
    forward = if fw === nothing
        nothing
    else
        _validate(_FORWARD_LATTICE, fw, "$(what)'s \"forward\"", "")
        F = _lattice_from_json(_namedtuple(fw))
        haskey(fw, "present") && throw(ArgumentError("$(what)'s \"forward\": a forward spline holds every node (no `present`)"))
        ChartSpline(F.area, _controls_from_json(fw["controls"]["point"], F.P, Val(2)))
    end
    return SplineLens(back; forward, present = L.present)
end

function _lens_from_json(x)
    x isa Union{AbstractDict, NamedTuple} || throw(ArgumentError("a lens record is an object"))
    x = _lower(x)                       # the plain Dicts and Vectors the lens record is (a parsed or in-memory record)
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

const _LENS = _CheckedSpec(_Free(), _lens_check)

# -----------------------------------------------------------------------------
# A bounded camera's body: its image size, K and lens (or a non-central model's
# spline), in the file's pixel frame; its sensor's pitch and its lens coverage
# -----------------------------------------------------------------------------

const _K_SPEC = _Obj(:fx => _Real(), :fy => _Real(), :cx => _Real(), :cy => _Real(), :skew => _Real())
# A camera bounded by its physical `Sensor` carries the sensor's pixel pitch
# beside its body: `"pitch": {"width", "height"}`, a length per pixel in the
# file's `units.length` (px is dimensionless, so a pitch IS a length;
# `write_length`/`read_length`, decimal-exact). Absent, the body is the bare
# extent's. Written by the rig and the camera store; never dropped.
const _PITCH = _CheckedSpec(
    _Obj(:width => _Real(), :height => _Real()),
    x -> (x[:width] > 0 && x[:height] > 0) ? nothing : "a pixel pitch is {width, height}, two positive lengths per pixel in units.length",
)
_with_pitch(body, ::Nothing, ::_LeafUnit) = body
_with_pitch(body, s::Sensor, unit::_LeafUnit) =
    merge(body, (pitch = (width = _leaf_length(s.pitch.width, unit), height = _leaf_length(s.pitch.height, unit)),))
# The extent, or the physical sensor when the body carries a pitch.
_bound_from_body(b, ext::Size2, u::Units) = haskey(b, :pitch) ? _sensor_from_pitch(b.pitch, ext, u) : ext
function _sensor_from_pitch(p, ext::Size2, u::Units)
    pw, ph = read_length(p.width::Real, u), read_length(p.height::Real, u)
    return Sensor(Size2(width = ext.width * px, height = ext.height * px), Size2(width = pw / px, height = ph / px))
end

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

# The body of a camera of a CameraModel, and of a non-central grid camera.
const _CAMERA_BODY_FIELDS = Pair{Symbol, _Spec}[:image_size => _SIZE_INT, :K => _K_SPEC, :lens => _LENS]
# The record's type: a token of the format, never a Julia type's name (a renamed type would otherwise force a new schema).
const _NONCENTRAL_TYPE = "noncentral_spline"
const _NONCENTRAL_LATTICE = _lattice_spec([:direction => 3, :point => 3])
const _NONCENTRAL_BODY_FIELDS = Pair{Symbol, _Spec}[
    :type => _Named(
        ==(_NONCENTRAL_TYPE),
        "\"$(_NONCENTRAL_TYPE)\" — a central grid camera is a camera with a spline lens (its `lens` the spline record)",
    ),
    :image_size => _SIZE_INT, :K => _K_SPEC, :spline => _NONCENTRAL_LATTICE,
]
const _BOUND_FIELDS = Pair{Symbol, _Spec}[:pitch => _PITCH, :lens_coverage => _LENS_COVERAGE]
# A bounded camera's body with the keys `extra` (required) and `extra_optional`
# of its enclosing record (a store entry's `name` and `calibration_record`): a
# camera body or a non-central one, told apart by `lens` / `type`.
_bounded_body_spec(extra = Pair{Symbol, _Spec}[], extra_optional = Pair{Symbol, _Spec}[]) = _Either(Pair{Symbol, _Spec}[
    :lens => _CheckedSpec(_Obj(Pair{Symbol, _Spec}[extra..., _CAMERA_BODY_FIELDS...], Pair{Symbol, _Spec}[_BOUND_FIELDS..., extra_optional...]),
                          x -> _lens_coverage_problem(get(x, :lens_coverage, nothing), "a camera body")),
    :type => _CheckedSpec(_Obj(Pair{Symbol, _Spec}[extra..., _NONCENTRAL_BODY_FIELDS...], Pair{Symbol, _Spec}[_BOUND_FIELDS..., extra_optional...]),
                          x -> _lens_coverage_problem(get(x, :lens_coverage, nothing), "the non-central camera")),
])
const _BOUNDED_CAMERA_BODY = _bounded_body_spec()

_image_size_json(ext::Size2) = (width = Int(ext.width), height = Int(ext.height))
_extent_from_body(b) = Size2(Int(b.image_size.width::Integer), Int(b.image_size.height::Integer))

# A CameraModel on IO's style (style.jl): its K — canonical by construction —
# with the principal point in the style's pixel frame (the file's declaration;
# a frame-typed container written in another moves it with the declaration),
# and its lens record. A body is its image size and these.
JSON.lower(s::VgioStyle, model::CameraModel) = (K = _k_record(model.intrinsics.K, _pixel_frame(s)), lens = _lens_json(model.lens))
_model_body(s::VgioStyle, model::CameraModel, extent::Size2) = merge((image_size = _image_size_json(extent),), JSON.lower(s, model))

# …and back, its K read in the style's pixel frame and moved into the canonical
# one by `_k_from_record(k, frame)` (style.jl), the exact inverse of `_k_record`
# and the one conversion of a stored K, a camera's and a non-central grid
# camera's alike (a SplineLens lens is frame-free: x = K⁻¹u).
JSON.lift(s::VgioStyle, ::Type{CameraModel}, b::Record) =
    CameraModel(LogicalIntrinsics(_k_from_record(b.K, _pixel_frame(s))), _lens_from_json(b.lens))

# A non-central camera's body on IO's style: its image size, K (in the style's
# pixel frame) and the lattice record of its two splines, laid over one
# rectangle of the K-normalized plane x = K⁻¹u — the directions and the line
# points of its observation lines (Schöps, Larsson, Pollefeys, Sattler, CVPR
# 2020) — the points (plain metres in memory) in the style's length form (the
# file's unit; the identity tree's persisted metres), or `points`, a
# migration's own numbers as its file holds them.
function _model_body(s::VgioStyle, m::NoncentralSplineModel, ext::Size2; points = nothing)
    P = _held_mask(m.present, size(m.directions.controls))
    sp = _lattice_json(m.directions.area, P, (direction = (m.directions.controls, identity),
                                               point = (m.points.controls, x -> _leaf_power(x, s.length_unit, 1))))
    points === nothing || (sp = merge(sp, (controls = merge(sp.controls, (point = points,)),)))
    return (type = _NONCENTRAL_TYPE, image_size = _image_size_json(ext), K = _k_record(m.intrinsics.K, _pixel_frame(s)), spline = sp)
end
# …and back: the model and its extent, from a validated body whose K is in the
# style's pixel frame and whose points are in its length unit.
function _noncentral_from_body(s::VgioStyle, b::Record)
    sp = b.spline
    L = _lattice_from_json(sp)
    D = ChartSpline(L.area, _controls_from_json(sp.controls.direction::Vector{Any}, L.P, Val(3)))
    X = ChartSpline(L.area, _controls_from_json(sp.controls.point::Vector{Any}, L.P, Val(3), x -> _read_power(x, s.length_unit, 1)))
    m = NoncentralSplineModel(LogicalIntrinsics(_k_from_record(b.K, _pixel_frame(s))), D, X; L.present)
    return m, _extent_from_body(b)
end

"""
    camera_json(model::CameraModel, extent, frame = CANONICAL_PIXEL_FRAME) -> NamedTuple
    camera_from_json(body, frame; length_unit = mm) -> (model, extent)

The camera body every camera-carrying kind embeds — `image_size` (`{width,
height}`), `K` (`fx, fy, cx, cy, skew`) and the `lens` record — written with its
principal point in the pixel `frame` the enclosing file declares, and read back
into VGC's canonical frame. Building blocks for a package serializing its own
kind; a bounded camera's body (its pitch and lens coverage too) is
`camera_json(camera::BoundedCamera, frame)`.
"""
camera_json(model::CameraModel, extent::Size2, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME) =
    _model_body(VgioStyle(; pixel_frame = frame), model, extent)

"""
    camera_json(model::NoncentralSplineModel, extent, frame = CANONICAL_PIXEL_FRAME; length_unit = mm) -> NamedTuple

A non-central grid camera's body, a camera store's and a rig sensor's: `type =
"noncentral_spline"`, `image_size`, the gauge `K` in the pixel `frame` and the
`spline` lattice record (the held nodes, the `direction` and `point` controls,
the line points — plain metres in memory — in `length_unit`).
[`camera_from_json`](@ref) reads it back (it tells the two bodies apart by `type`).
"""
camera_json(model::NoncentralSplineModel, extent::Size2, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME; length_unit::Units = mm) =
    _model_body(VgioStyle(; pixel_frame = frame, length_unit), model, extent)

"""
    camera_json(camera::BoundedCamera, frame = CANONICAL_PIXEL_FRAME; length_unit = mm) -> NamedTuple

A bounded camera's body — the one a camera store's entry and a rig sensor
embed: its model's body (a `CameraModel`'s of any lens, or a
`NoncentralSplineModel`'s), the sensor's `pitch` in `length_unit` when it is
bounded by a physical `Sensor`, and its `lens_coverage` in the pixel `frame` when
the lens does not cover the whole sensor. `camera_from_json(BoundedCamera, body,
frame; length_unit)` reads it back.
"""
camera_json(camera::BoundedCamera, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME; length_unit::Units = mm) =
    _bounded_body(VgioStyle(; pixel_frame = frame, length_unit), camera)

"""
    camera_from_json(body, frame; length_unit = mm) -> (model, extent)

Read a camera body (the inverse of [`camera_json`](@ref) of a model) whose
principal point is in the pixel `frame` the enclosing file declares: a
`CameraModel`, or a `NoncentralSplineModel` (a body with a `type`, its line
points in `length_unit`), in VGC's canonical pixel frame, and its extent.

A body of an earlier shape, as another package's record of an earlier version
embeds it, reads as the camera it migrates to: an IO 0.9 body (`width`,
`height`, 0-based lattice runs) through the body's 0.10 step, and an old grid
body (the body of a `vgio/generic_camera/v2`/`v3` record — RadiometricCamera's
`rc/calibration_state/v1` camera) through the same conversion as an old grid
file: a central one as a `CameraModel` with a `SplineLens`, a non-central one as
a `NoncentralSplineModel` (its line points in `length_unit`). An old grid body
is read in the colmap pixel frame (`CANONICAL_PIXEL_FRAME`), whatever `frame`
is given: the old grid code reads every old grid body's numbers in that frame.
"""
camera_from_json(body, frame::Type{<:PixelFrame}; length_unit::Units = mm) =
    _model_from_body(_current_camera_body(VgioStyle(; pixel_frame = frame, length_unit), body::Record)...)

"""
    camera_from_json(::Type{BoundedCamera}, body, frame; length_unit = mm) -> BoundedCamera

Read a bounded camera's body (the inverse of [`camera_json`](@ref) of a
`BoundedCamera`), a parsed JSON object: checked against the body's table first
(an `ArgumentError` naming the field otherwise), then its model, its bound (a
`pitch`, in `length_unit`: the physical `Sensor`; none: the pixel extent) and its
lens coverage (none: the whole sensor), in VGC's canonical pixel frame. A body of
an earlier shape reads as `camera_from_json(body, frame)` says.
"""
function camera_from_json(::Type{BoundedCamera}, body, frame::Type{<:PixelFrame}; length_unit::Units = mm)
    s, b = _current_camera_body(VgioStyle(; pixel_frame = frame, length_unit), body)   # an embedded earlier body: its camera's
    _validate(_BOUNDED_CAMERA_BODY, b, "a camera body", "")
    return _bounded_from_body(s, b::Record)
end

# A bounded camera's body on IO's style, and back: the one form every kind that stores one embeds (a camera store's
# entry, a rig sensor). The model's body, its sensor's pitch (the style's length form) and its lens coverage; read
# back, a body with a `type` is a non-central one, posed by `pose` when given (a rig sensor's).
function _bounded_body(s::VgioStyle, bc::BoundedCamera)
    m, ext = model(bc), extent(bc)
    body = _model_body(s, m, ext)                      # the model's own body: a central camera's or a non-central one's
    return merge(_with_pitch(body, sensor(bc), s.length_unit), _lens_coverage_json(s, lens_coverage(bc), ext))
end
# (The body is a current one: a record's reader gets it through its version steps, and the one entry an
# embedded body of an earlier shape reaches, `camera_from_json`, converts it first, `_current_camera_body`.)
function _model_from_body(s::VgioStyle, b::Record)
    return haskey(b, :type) ? _noncentral_from_body(s, b) : (JSON.lift(s, CameraModel, b), _extent_from_body(b))
end
function _bounded_from_body(s::VgioStyle, b::Record, pose = nothing)
    m, ext = _model_from_body(s, b)
    return BoundedCamera(_posed(m, pose), _bound_from_body(b, ext, s.length_unit); lens_coverage = _lens_coverage_from_json(s, get(b, :lens_coverage, nothing)))
end
_posed(m, ::Nothing) = m
_posed(m, X) = PosedCamera(m, X)

# -----------------------------------------------------------------------------
# The calibration record — where a stored camera's numbers came from
# -----------------------------------------------------------------------------
#
# A camera's own calibration facts are an ATTRIBUTE of the value (the metadata
# rule: envelope for the file, attributes for the value):
#
#     Attributed(bc::BoundedCamera, (; calibration_record = (; tool = "OpenCV calibrateCamera",
#                                                            date = "2026-05-27", reprojection_rms = 0.21)))
#
# and the camera's `calibration_record` block on disk (a target's is the same
# block under the same name), read back exactly. Its keys
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
#                     each stated or omitted: the ONE device-identity encoding,
#                     capture_settings' too); in the value
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

const _CALIBRATION = _CheckedSpec(
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
# THE VALUE: a store of named cameras — `Dict{Symbol, …}` from the camera's
# name (the identity a config refers to) to a VGC `BoundedCamera` (of a
# `CameraModel`: COLMAP's "camera" row, intrinsics, lens, image size; or of a
# `NoncentralSplineModel`), or an `Attributed` one carrying its
# `calibration_record`. Once a pose is known, `BoundedCamera(PosedCamera(model(bc),
# pose), bc.bound; lens_coverage = lens_coverage(bc))` is the posed camera.
#
# No pixel frame is stored per camera: a model's K is in VGC's canonical frame
# by construction. On disk the store still names its pixel frame, because a
# FOREIGN file may declare `opencv` or `julia`, converted on the way in.

const _StoredCamera = Union{BoundedCamera, Attributed{<:BoundedCamera, <:NamedTuple{(:calibration_record,)}}}

const _CAMERAS_SPEC = _kind_spec(
    "cameras",
    Pair{Symbol, _Spec}[:cameras => _Vec(_bounded_body_spec(Pair{Symbol, _Spec}[:name => _NonEmpty()],
                                                            Pair{Symbol, _Spec}[:calibration_record => _CALIBRATION]))];
    check = x -> allunique(c[:name] for c in x[:cameras]) ? nothing : "camera names must be unique — they are the identity configs refer to",
)

function _camera_entry(s::VgioStyle, name::Symbol, c::_StoredCamera)
    e = merge((name = String(name),), _bounded_body(s, value_of(c)))
    return c isa Attributed ? merge(e, (calibration_record = _calibration_json(c.calibration_record),)) : e
end

"""
    save(path, cameras::AbstractDict{Symbol}; length_unit = mm, provenance = (;))

Write a camera store — name → `BoundedCamera` (of a `CameraModel`, any lens,
or of a `NoncentralSplineModel`), or an `Attributed` one carrying its
`calibration_record` — as a `vgio/cameras/v6.0` file, cameras sorted by name.
A camera bounded by its physical `Sensor` carries the sensor's `pitch`, and a
non-central camera its line points, in `length_unit` (the file's
`units.length`). `provenance` is the file's one free-form object, which VGIo
never reads back.

JSON only, deliberately: cameras are configuration-adjacent (referenced by
name from a TOML, hand-inspected, diffed on recalibration), the payload is
kilobytes at rig scale, and Julia prints Float64 in its shortest
round-trippable form so nothing is lost.
"""
function save(path::AbstractString, cameras::AbstractDict{Symbol}; length_unit::Units = mm, provenance::NamedTuple = (;))
    for (name, c) in cameras
        c isa _StoredCamera || throw(
            ArgumentError(
                "camera $(name): a stored camera is a BoundedCamera (or one Attributed with " *
                    "`(; calibration_record)`); got a $(typeof(c))"
            )
        )
        isempty(String(name)) && throw(ArgumentError("a camera name is non-empty — it is the identity a config refers to"))
    end
    s = VgioStyle(; length_unit)
    _write_kind(path, "cameras", (cameras = [_camera_entry(s, k, cameras[k]) for k in sort!(collect(keys(cameras)))],), provenance; style = s)
    return nothing
end

"""
    save(path, camera::BoundedCamera; name, length_unit = mm, provenance = (;))

Write one bounded camera as a camera store of that camera under `name`
(`save(path, Dict(name => camera))`).
"""
save(path::AbstractString, camera::Union{BoundedCamera, Attributed{<:BoundedCamera}}; name::Union{Symbol, AbstractString},
     length_unit::Units = mm, provenance::NamedTuple = (;)) =
    save(path, Dict{Symbol, Any}(Symbol(name) => camera); length_unit, provenance)

"""
    load(Dict{Symbol, BoundedCamera}, path; intrinsics = nothing) -> Dict{Symbol, Union{BoundedCamera, Attributed{<:BoundedCamera}}}

Read a `vgio/cameras/v6.0` store: name → `BoundedCamera` (its K moved from the
file's declared pixel frame into the canonical one), an `Attributed` one where
the file states the camera's `calibration_record`. An earlier store is read
through its version steps (`migrate_record`); a `vgio/noncentral_spline/v1`
record, and an old grid file (`vgio/generic_camera/v2` or `v3`, central or
not), as the store of its one camera, named after the file's basename
(`fisheye_14mm.json` → `:fisheye_14mm`).

`intrinsics` (a `CameraCalibrationMat`, in the canonical pixel frame) supplies K
to an old grid file that stores no nominal pinhole — one whose lattice has no
complete 4 × 4 window at the gauge's pixel is refused without it — read as if
the file held it (the file is never edited); a record that holds a K refuses it.
"""
load(::Type{Dict{Symbol, BoundedCamera}}, path::AbstractString; intrinsics::Union{Nothing, CameraCalibrationMat} = nothing) =
    _load_store(_with_given_intrinsics(_read_vgio_json(path), intrinsics, path), path)
# The store of a parsed file (`d`, its JSON).
function _load_store(d, path::AbstractString)
    d = _checked_store(d, path, "cameras")   # (an old grid file or a noncentral_spline record: the store of its one camera)
    s = VgioStyle(d)
    out = Dict{Symbol, _StoredCamera}()
    for c in d.cameras::Vector{Any}
        bc = _bounded_from_body(s, c::Record)
        out[Symbol(c.name::String)] = haskey(c, :calibration_record) ?
            Attributed(bc, (; calibration_record = _calibration_from_json(c.calibration_record::AbstractDict))) : bc
    end
    return out
end

"""
    load(BoundedCamera, path; name = nothing, file_unit = nothing, intrinsics = nothing) -> BoundedCamera
    load(CameraModel,  path; name = nothing) -> CameraModel

Read ONE camera from a `vgio/cameras/v6.0` store — the bounded camera, or just
its model (its `calibration_record`, if any, is the store's to read). Without
`name` the store must hold exactly one camera: a multi-camera store is
REFUSED with the available names rather than defaulting to the first —
which camera a run uses is an identity, and silently guessing it is how a
config ends up calibrated for the wrong lens while reporting success.

The grid cameras of Schöps et al. read through the same call, by the file's
content: an old grid file (`vgio/generic_camera/v2` or `v3`) or a
`vgio/noncentral_spline/v1` record as its one camera — a central one with a
`SplineLens` (its nominal pinhole the K, else the lattice's gauge; its domain
the lens coverage), a non-central one a `BoundedCamera{<:NoncentralSplineModel}`
— and the reference implementation's `intrinsics0.yaml` as either (schoeps.jl;
a non-central YAML's `point_grid` unit is the caller's `file_unit`). Old files
are read, never rewritten.

`load(CameraModel, path)` refuses a camera whose lens coverage is not its whole
sensor (an image circle, an old grid file's domain) rather than drop it: load
the `BoundedCamera`.
"""
function load(
        ::Type{BoundedCamera}, path::AbstractString;
        name::Union{Nothing, AbstractString, Symbol} = nothing, file_unit::Union{Nothing, Units} = nothing,
        intrinsics::Union{Nothing, CameraCalibrationMat} = nothing,
    )
    if _file_form(path) !== :json                       # a Schöps intrinsics0.yaml (schoeps.jl)
        name === nothing || throw(ArgumentError("$(path): a Schöps YAML holds one camera; `name` names a camera of a store"))
        intrinsics === nothing || throw(ArgumentError("$(path): `intrinsics` supplies K to an old vgio grid file; a Schöps YAML's K is its lattice's gauge"))
        return _load_schoeps_camera(path, file_unit)
    end
    file_unit === nothing ||
        throw(ArgumentError("$(path): a vgio file declares its own units.length; `file_unit` is for a Schöps YAML"))
    store = _load_store(_with_given_intrinsics(_read_vgio_json(path), intrinsics, path), path)   # parsed once (a grid file is large)
    if name === nothing
        length(store) == 1 || throw(
            ArgumentError("$(path) holds $(length(store)) cameras; supply `name` — one of $(sort!(String.(collect(keys(store)))))")
        )
        return value_of(only(values(store)))
    end
    haskey(store, Symbol(name)) || throw(
        ArgumentError("no camera named $(repr(String(name))); available: $(sort!(String.(collect(keys(store)))))")
    )
    return value_of(store[Symbol(name)])
end

function load(
        ::Type{CameraModel}, path::AbstractString;
        name::Union{Nothing, AbstractString, Symbol} = nothing, intrinsics::Union{Nothing, CameraCalibrationMat} = nothing,
    )
    bc = load(BoundedCamera, path; name, intrinsics)
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
            "or approximate it by a parametric lens first, `VisualGeometryToolkitEstimation.approximate_camera(KannalaBrandt{4}, camera, region)` " *
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
        (bc = value_of(cameras[n]); e = extent(bc); colmap_camera(bc.camera, i, e.width, e.height))
            for (i, n) in enumerate(names)
    ]
    return cams, Dict(n => i for (i, n) in enumerate(names))
end

# A noncentral_spline record (`vgio/noncentral_spline/v1`, the non-central grid
# camera's own kind before IO 0.10), or an old grid file (vgio/generic_camera/v2
# or v3, central or not): the store of its one camera, named after the file
# (`fisheye_14mm.json` → `fisheye_14mm`), the record through its version steps.
function _one_camera_store(d, path)
    d isa AbstractDict || return d
    s = get(d, :schema, nothing)
    _is_central_grid(d) && return _central_grid_store(d, path)
    v = _split_schema(s)
    (v !== nothing && v.kind in ("noncentral_spline", "generic_camera")) || return d
    t = migrate_record(d)          # an old non-central grid file through its step to noncentral_spline/v1
    t["schema"] == "vgio/noncentral_spline/v1" ||
        throw(ArgumentError("$(path): a $(s) file, read through its migration, ends at $(repr(t["schema"])), not a non-central camera record"))
    return _parse_vgio_text(JSON.json(_noncentral_store_v6(t, first(splitext(basename(path))))), path)
end
