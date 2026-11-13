# =============================================================================
# vgio/panorama/v2.0 — a panorama's results: the camera, the frames' rotations, the vignetting and the frames' gains
# =============================================================================
#
# {"schema": "vgio/panorama/v2.0", <envelope>,
#  "camera": {bounded camera body},                                  (a camera store's entry without `name`)
#  "rotations": {"from": "camera", "to": "world", "frames": [{"frame_index": 1, "image": "DSC00001.ARW",
#                "pose": {"rotation": [w, x, y, z], "translation": [0.0, 0.0, 0.0]}}, …]},   (the trajectory body)
#  "vignetting": {vignetting body},                                  (the vgio/vignetting body)
#  "gains": [{"frame_index": 1, "gain": [g_R, g_G, g_B]}, …],        (optional: none, every frame at one exposure)
#  "provenance": {…}}                                                (optional)
#
# The results of a camera that only rotated (PanoramaStitching's): composed ONLY of bodies other kinds already write,
# each through its own lower and lift — the camera's (the camera store's and a rig sensor's), the trajectory's
# (`JSON.lower(::VgioStyle, ::Trajectory)`, the vgio/trajectory body without its envelope) and the vignetting's (the
# vgio/vignetting body). No IO body holds a rotation alone, so the rotations are the trajectory body with every
# translation zero (a camera that only rotated: its centre does not move), the rotation exact as the stored
# quaternion, the zero a length in `units.length` like any translation; a file or a value with a translation that is
# not zero is refused. A frame's gains are plain numbers, its exposure gain per colour plane (I = g·L·V), keyed by
# the frame index `frame_index`, the join key of the rotations' frames: no IO body holds a gain.
#
# The envelope declares the units once: the pixel frame of the camera's K and lens coverage and of the vignetting's
# positions (each body moves its numbers by the style), the length unit of the translations and of a sensor's pitch.
# vgio/panorama/v1 (IO 0.9.4) reads through its step to v2.0 (`_panorama_v1_to_v2`).
#
# The value has no VisualGeometryToolkitCore type of its own: it is the NamedTuple
# `(; camera, rotations, vignetting, gains)` of VGC values, written by `save_panorama` and read by `load_panorama`
# (a composite of several values, as `save_manifest`/`load_manifest`'s), and checked on the bytes by the kind's table.

const _GAIN_RECORD = _Obj(:frame_index => _Int(), :gain => _Vec(_Real()))

function _panorama_check(x)
    r = x[:rotations]
    (r[:from] == frame_name(CameraFrame) && r[:to] == frame_name(WorldFrame)) ||
        return "rotations: a panorama's rotations map the camera to the world (`from` $(repr(frame_name(CameraFrame))), `to` $(repr(frame_name(WorldFrame)))), got $(repr(r[:from])) → $(repr(r[:to]))"
    for f in r[:frames]
        all(iszero, f[:pose][:translation]) ||
            return "rotations: frame $(f[:frame_index]): a panorama's camera only rotates; its translation is zero, got $(collect(f[:pose][:translation]))"
    end
    haskey(x, :gains) || return nothing
    g = x[:gains]
    isempty(g) && return "gains: one record per frame of the rotations (none stated: omit `gains`)"
    Set(q[:frame_index] for q in g) == Set(f[:frame_index] for f in r[:frames]) == Set(1:length(g)) ||
        return "gains: one record per frame of the rotations, whose frame indices are then 1:K (row k of the gains is frame index k)"
    C = length(first(g)[:gain])
    C > 0 && all(q -> length(q[:gain]) == C, g) || return "gains: every frame has the same number ($(C) > 0) of colour planes"
    all(q -> all(>(0), q[:gain]), g) || return "gains: an exposure gain is positive"
    return nothing
end

const _PANORAMA_SPEC = _kind_spec(
    "panorama",
    Pair{Symbol, _Spec}[:camera => _BOUNDED_CAMERA_BODY, :rotations => _TRAJECTORY_BODY, :vignetting => _VignettingBody()],
    Pair{Symbol, _Spec}[:gains => _Vec(_GAIN_RECORD)];
    check = _panorama_check,
)

"""
    save_panorama(path, camera::BoundedCamera, rotations::Trajectory, vignetting::AbstractVignetting, gains;
                  length_unit = mm, provenance = (;))

Write a panorama's results — the camera of a sweep that only rotated, its frames'
rotations, its vignetting and its frames' exposure gains (PanoramaStitching's) — as a
`vgio/panorama/v2.0` file, each through the body another kind writes: the camera
store's bounded camera body (K, lens, a sensor's `pitch` in `length_unit`, its lens
coverage), the trajectory body (`rotations`: per frame index its pose `from → to`,
its `image` and `time` where the trajectory has them) and the vgio/vignetting body,
positions in the canonical (colmap) pixel frame. The rotations map the camera to the world
(`Trajectory{CameraFrame, WorldFrame}`) and every pose's translation is a zero length:
other frames, a translation that is not zero (the camera only rotates) and a plain one
(attach its unit, `EuclideanMap(R, zeros(3) .* m)`) are an `ArgumentError`, nothing
written.

`gains` is `nothing` (every frame at one exposure) or a `K × C` matrix,
`gains[k, c]` the exposure gain of frame index `k` in colour plane `c`
(PanoramaStitching's `fit_vignetting` `metadata.gains`), every one positive; the
rotations' frame indices are then `1:K`. Numbers of a narrower type (Float32) are
written exactly. [`load_panorama`](@ref) reads the file back.
"""
function save_panorama(path::AbstractString, camera::BoundedCamera, rotations::Trajectory, vignetting::AbstractVignetting,
                       gains::Union{Nothing, AbstractMatrix{<:Real}}; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    body = merge((camera = _bounded_body(s, camera), rotations = JSON.lower(s, rotations), vignetting = JSON.lower(s, vignetting)),
                 _gains_json(gains))
    _write_kind(path, "panorama", body, provenance; style = s)
    return nothing
end

_gains_json(::Nothing) = (;)
# Row k of the gains is frame index k: the table refuses gains whose frame indices are not the rotations', 1:K.
_gains_json(G::AbstractMatrix{<:Real}) = (gains = [(frame_index = k, gain = row) for (k, row) in enumerate(_matrix_rows(G))],)

"""
    load_panorama(path) -> (; camera, rotations, vignetting, gains)

Read a `vgio/panorama/v2.0` file ([`save_panorama`](@ref)'s): `camera` the
`BoundedCamera` (its K and lens coverage moved into VGC's canonical pixel frame),
`rotations` the `Trajectory` of the frames' poses (`QuatRotation`s, the stored
quaternions exactly; translations zero metres; labeled by the file's `from` and
`to`), `vignetting` the `AbstractVignetting`, rebuilt bit for bit, and `gains` the
`K × C` `Matrix{Float64}` of the frames' exposure gains (row `k` frame index `k`),
or `nothing` when the file states none. Every number reads back as the Float64 it
was written as.
"""
function load_panorama(path::AbstractString)
    d = _read_store(path, "panorama", _PANORAMA_SPEC)
    s = VgioStyle(d)
    return (
        camera = _bounded_from_body(s, d.camera::Record), rotations = JSON.lift(s, Trajectory, d.rotations::Record),
        vignetting = JSON.lift(s, AbstractVignetting, d.vignetting::Record), gains = _gains_from_json(get(d, :gains, nothing)),
    )
end

_gains_from_json(::Nothing) = nothing
function _gains_from_json(g::AbstractVector)
    return _rows_matrix(r.gain for r in sort(g; by = r -> r.frame_index::Integer))
end

# vgio/panorama/v1 (IO 0.9.4) → v2.0 (IO 0.10): the envelope (its `pixel_frame`, colmap, and pose declarations into
# `units` and `conventions`); the camera body as 0.10's (`_camera_body_v010`); the rotations' frames `t` →
# `frame_index`, `camera_optical` → `camera`; the vignetting body's step (`_vignetting_body_v010`, the vignetting kind's); the gains'
# `t` → `frame_index`. No number moves but a w < 0 quaternion's signs (the envelope step's).
function _panorama_v1_to_v2(old::AbstractDict)
    t = _envelope_v010(old, "vgio/panorama/v2.0")
    t["camera"] = _camera_body_v010(old["camera"])
    t["rotations"] = _frame_names_v010(_frames_v010(t["rotations"]))
    t["vignetting"] = _vignetting_body_v010(old["vignetting"])
    haskey(old, "gains") && (t["gains"] = Any[_renamed_key(g, "t", "frame_index") for g in old["gains"]])
    return t
end
migrate_schema(::Val{Symbol("vgio/panorama/v1")}, t::AbstractDict) = _panorama_v1_to_v2(t)
