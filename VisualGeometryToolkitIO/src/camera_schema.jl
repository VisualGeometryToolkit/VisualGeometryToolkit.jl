# =============================================================================
# vgio/cameras/v3 — the camera store
# =============================================================================
#
# {"schema": "vgio/cameras/v3", "pixel_frame": "colmap",
#  "cameras": [{"name": "sony-fx3_tamron-20mm_f22", "width": 4240, "height": 2832,
#               "K": {"fx": …, "fy": …, "cx": …, "cy": …, "skew": 0.0},
#               "lens": {"perspective": [],
#                        "add": [{"radial": [[k1, k2, k3], [k4, k5, k6]]},
#                                {"tangential": [p1, p2]}, …]},
#               "calibration": {"tool": …, "source": …, "date": "2026-01-19",
#                               "reprojection_rms": 0.159, "n_images": 36}}],   (optional)
#  "provenance": {…}}                                               (optional)
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
# refusal already names the registered options.
#
# `CANONICAL_PIXEL_FRAME` is the single thing VGIo must state: which frame a
# `CameraModel`'s stored K is in. VGC guarantees it — every `CameraModel`
# constructor converts its `image_origin` argument to this frame on the way in
# — but exposes no accessor for it, so it is named here ONCE rather than as a
# `:colmap` literal in every writer and reader.
const CANONICAL_PIXEL_FRAME = PixelFrame{:colmap}

"""
    pixel_frame_name(::Type{<:PixelFrame}) -> String
    pixel_frame_from_name(name) -> Type{<:PixelFrame}
    pixel_origin(::Type{<:PixelFrame}) -> Symbol

The serialized spelling of a pixel frame, its inverse, and the origin symbol
VGC's `image_origin` keyword takes. Parallel to
[`frame_name`](@ref)/[`frame_from_name`](@ref) for 3D frames, deliberately
separate because the grammars differ: a 3D frame serializes prefixed
(`"view:cam1"`), a pixel frame bare (`"colmap"`).

`pixel_frame_from_name` validates by CONSTRUCTING the conversion to
[`CANONICAL_PIXEL_FRAME`](@ref) — VGC decides whether a pixel frame exists,
and its error names the registered set. This only restates that refusal as
the `ArgumentError` every corrupt record in this package raises.
"""
pixel_frame_name(::Type{PixelFrame{O}}) where {O} = String(O)
pixel_origin(::Type{PixelFrame{O}}) where {O} = O

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
    throw(ArgumentError("no lens record for the lens type $(typeof(l)) — the store holds VGC's Lens and LiftLens"))

# Parsed JSON objects (`JSON.Object`s) as the plain Dicts and Vectors the
# lens record is.
_plain(x::AbstractDict) = Dict{String, Any}(String(k) => _plain(v) for (k, v) in x)
_plain(x::AbstractVector) = Any[_plain(v) for v in x]
_plain(x) = x

function _lens_from_json(x)
    x isa AbstractDict || throw(ArgumentError("a lens record is an object"))
    # the retired v1/v2 lens shape, `{"model": …, "params": …}`: never read here
    (haskey(x, "model") || haskey(x, :model)) && throw(
        ArgumentError(
            "a {\"model\", \"params\"} lens is a retired v1/v2 lens record, which predates the lens " *
                "families; there is no reader for it: re-export the camera from its producer"
        )
    )
    return lens_from_record(_plain(x)::Dict{String, Any})
end

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

# Written from a CameraModel, whose K is canonical by construction, in the
# pixel frame the enclosing file declares (`frame`; the canonical one unless a
# frame-typed container is written in another, when its principal point moves
# with the declaration).
function _camera_body(model::CameraModel, extent::Size2, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME)
    K = model.intrinsics.K
    c = EuclideanMap(CANONICAL_PIXEL_FRAME, frame)(SVector{2, Float64}(K[1, 3], K[2, 3]))
    return (
        width = Int(extent.width), height = Int(extent.height),
        K = (fx = Float64(K[1, 1]), fy = Float64(K[2, 2]), cx = Float64(c[1]), cy = Float64(c[2]), skew = Float64(K[1, 2])),
        lens = _lens_json(model.lens),
    )
end

# A body back to (model, extent), its K read in `frame` (the file's declared
# pixel frame): `image_origin` is the CameraModel constructor's keyword for
# "the frame the K I am handing you is written in", and it converts to the
# canonical frame internally — the whole of the half-pixel handling.
function _camera_from_body(b, frame::Type{<:PixelFrame})
    k = b.K
    K = CameraCalibrationMat(
        SMatrix{3, 3, Float64}(
            Float64(k.fx::Real), 0, 0, Float64(k.skew::Real), Float64(k.fy::Real), 0,
            Float64(k.cx::Real), Float64(k.cy::Real), 1,
        )
    )
    model = CameraModel(K; lens = _lens_from_json(b.lens), image_origin = pixel_origin(frame))
    return model, Size2(Int(b.width::Integer), Int(b.height::Integer))
end

"""
    camera_json(model::CameraModel, extent, frame = CANONICAL_PIXEL_FRAME) -> NamedTuple
    camera_from_json(body, frame) -> (model, extent)

The camera body every camera-carrying kind embeds — `width`, `height`, `K`
(`fx, fy, cx, cy, skew`) and the `lens` record — written with its principal
point in the pixel `frame` the enclosing file declares, and read back into
VGC's canonical frame. Building blocks for a package serializing its own kind.
"""
camera_json(model::CameraModel, extent::Size2, frame::Type{<:PixelFrame} = CANONICAL_PIXEL_FRAME) =
    _camera_body(model, extent, frame)
camera_from_json(body, frame::Type{<:PixelFrame}) = _camera_from_body(body, frame)

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
#
# Unlike the file's free-form `provenance`, which VGIo never reads, the block
# is part of the value: an unknown key is refused like any other.

const _CALIBRATION_KEYS = (:tool, :source, :date, :reprojection_rms, :n_images, :n_points, :note)
const _CALIBRATION_TYPES = (String, String, String, Float64, Int, Int, String)

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
    return nothing
end

const _CALIBRATION = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[],
        Pair{Symbol, _Spec}[
            :tool => _NonEmpty(), :source => _NonEmpty(), :date => _NonEmpty(), :reprojection_rms => _Real(),
            :n_images => _PosInt(), :n_points => _PosInt(), :note => _NonEmpty(),
        ],
    ),
    _calibration_check,
)

# The block as the value holds it: the stated keys in declaration order, typed.
_calibration_from_json(x) = NamedTuple(
    k => convert(T, x[k]) for (k, T) in zip(_CALIBRATION_KEYS, _CALIBRATION_TYPES) if haskey(x, k)
)
function _calibration_json(c::NamedTuple)
    msg = _calibration_check(c)
    msg === nothing || throw(ArgumentError("a camera's calibration block: $(msg)"))
    return NamedTuple(k => getfield(c, k) for k in _CALIBRATION_KEYS if haskey(c, k))
end

# -----------------------------------------------------------------------------
# The cameras store ↔ Dict{Symbol, BoundedCamera}
# -----------------------------------------------------------------------------
#
# THE VALUE: a store of named, unposed cameras — `Dict{Symbol, …}` from the
# camera's name (the identity a config refers to) to a VGC `BoundedCamera` of a
# `CameraModel` (COLMAP's "camera" row: intrinsics, lens, image size, no pose),
# or an `Attributed` one carrying its `calibration` block. Once a pose is known,
# `BoundedCamera(Camera(model(bc), pose), bc.extent)` is the posed camera.
#
# No pixel frame is stored per camera: a model's K is in VGC's canonical frame
# by construction. On disk the store still names its pixel frame, because a
# FOREIGN file may declare `opencv` or `julia`, converted on the way in.

const _StoredCamera = Union{BoundedCamera{<:CameraModel}, Attributed{<:BoundedCamera{<:CameraModel}, <:NamedTuple{(:calibration,)}}}

const _CAMERAS_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("cameras"), :pixel_frame => _PIXEL_FRAME,
            :cameras => _Vec(_Obj(Pair{Symbol, _Spec}[:name => _NonEmpty(), _CAMERA_BODY_FIELDS...], Pair{Symbol, _Spec}[:calibration => _CALIBRATION])),
        ],
        _PROVENANCE,
    ),
    x -> allunique(c[:name] for c in x[:cameras]) ? nothing :
        "camera names must be unique — they are the identity configs refer to",
)

_bounded(c::BoundedCamera) = c
_bounded(c::Attributed) = getfield(c, :value)

function _camera_entry(name::Symbol, c::_StoredCamera)
    bc = _bounded(c)
    e = merge((name = String(name),), _camera_body(bc.camera, bc.extent))
    return c isa Attributed ? merge(e, (calibration = _calibration_json(c.calibration),)) : e
end

"""
    save(path, cameras::AbstractDict{Symbol}; provenance = (;))

Write a camera store — name → `BoundedCamera` of a `CameraModel`, or an
`Attributed` one carrying its `calibration` block — as a `vgio/cameras/v3`
file, cameras sorted by name. `provenance` is the file's one free-form object,
which VGIo never reads back.

JSON only, deliberately: cameras are configuration-adjacent (referenced by
name from a TOML, hand-inspected, diffed on recalibration), the payload is
kilobytes at rig scale, and Julia prints Float64 in its shortest
round-trippable form so nothing is lost.
"""
function save(path::AbstractString, cameras::AbstractDict{Symbol}; provenance::NamedTuple = (;))
    for (name, c) in cameras
        c isa _StoredCamera || throw(
            ArgumentError(
                "camera $(name): a stored camera is a BoundedCamera of a CameraModel (or one Attributed with " *
                    "`(; calibration)`); got a $(typeof(c))"
            )
        )
        isempty(String(name)) && throw(ArgumentError("a camera name is non-empty — it is the identity a config refers to"))
    end
    body = (
        schema = _schema("cameras"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME),
        cameras = [_camera_entry(k, cameras[k]) for k in sort!(collect(keys(cameras)))],
    )
    _write_store(path, _CAMERAS_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(Dict{Symbol, BoundedCamera}, path) -> Dict{Symbol, Union{BoundedCamera, Attributed{<:BoundedCamera}}}

Read a `vgio/cameras/v3` store: name → `BoundedCamera` of a `CameraModel`
(its K moved from the file's declared pixel frame into the canonical one), an
`Attributed` one where the file states the camera's `calibration` block. A v1
or v2 file is refused by name: nothing reads a retired version.
"""
function load(::Type{Dict{Symbol, BoundedCamera}}, path::AbstractString)
    d = _read_store(path, "cameras", _CAMERAS_SPEC)
    frame = pixel_frame_from_name(d.pixel_frame::String)
    out = Dict{Symbol, _StoredCamera}()
    for c in d.cameras::Vector{Any}
        bc = BoundedCamera(_camera_from_body(c, frame)...)
        out[Symbol(c.name::String)] = haskey(c, :calibration) ?
            Attributed(bc, (; calibration = _calibration_from_json(c.calibration::AbstractDict))) : bc
    end
    return out
end

"""
    load(BoundedCamera, path; name = nothing) -> BoundedCamera
    load(CameraModel,  path; name = nothing) -> CameraModel

Read ONE camera from a `vgio/cameras/v3` store — the bounded camera, or just
its model (its `calibration` block, if any, is the store's to read). Without
`name` the store must hold exactly one camera: a multi-camera store is
REFUSED with the available names rather than defaulting to the first —
which camera a run uses is an identity, and silently guessing it is how a
config ends up calibrated for the wrong lens while reporting success.
"""
function load(::Type{BoundedCamera}, path::AbstractString; name::Union{Nothing, AbstractString, Symbol} = nothing)
    store = load(Dict{Symbol, BoundedCamera}, path)
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

load(::Type{CameraModel}, path::AbstractString; name::Union{Nothing, AbstractString, Symbol} = nothing) =
    model(load(BoundedCamera, path; name))

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
        (bc = _bounded(cameras[n]); colmap_camera(bc.camera, i, bc.extent.width, bc.extent.height))
            for (i, n) in enumerate(names)
    ]
    return cams, Dict(n => i for (i, n) in enumerate(names))
end
