# =============================================================================
# vgio/panorama/v1 — a panorama's results: the camera, the frames' rotations, the vignetting and the frames' gains
# =============================================================================
#
# {"schema": "vgio/panorama/v1", "pixel_frame": "colmap", <pose declarations>,
#  "camera": {bounded camera body},                                  (a camera store's entry without `name`)
#  "rotations": {"from": "camera_optical", "to": "world", "frames": [{"t": 1, "image": "DSC00001.ARW",
#                "pose": {"rotation": [w, x, y, z], "translation": [0.0, 0.0, 0.0]}}, …]},   (the trajectory body)
#  "vignetting": {vignetting body},                                  (the vgio/vignetting body)
#  "gains": [{"t": 1, "gain": [g_R, g_G, g_B]}, …],                  (optional: none, every frame at one exposure)
#  "provenance": {…}}                                                (optional)
#
# The results of a camera that only rotated (PanoramaStitching's): composed ONLY of bodies other kinds already write,
# each through its own lower and lift — the camera's (`camera_json(::BoundedCamera)`, the camera store's and a rig
# sensor's), the trajectory's (`JSON.lower(::VgioStyle, ::Trajectory)`, the vgio/trajectory body without its schema and
# declarations) and the vignetting's (the vgio/vignetting body). No IO body holds a rotation alone, so the rotations
# are the trajectory body with every translation zero (a camera that only rotated: its centre does not move), the
# rotation exact as the stored quaternion, the zero a length in `length_unit` like any translation; a file or a value
# with a translation that is not zero is refused. A frame's gains are plain numbers, its exposure gain per colour
# plane (I = g·L·V), keyed by the frame index `t`, the join key of the rotations' frames: no IO body holds a gain.
#
# The file declares its conventions once: the pixel frame (colmap, the vignetting body's own: its positions are COLMAP
# pixels, and the camera's K and lens coverage are written in it) and the pose declarations (the rotations' encoding
# and transform convention; the length unit of the translations and of a camera's pitch).
#
# The value has no VisualGeometryToolkitCore type of its own: it is the NamedTuple
# `(; camera, rotations, vignetting, gains)` of VGC values, written by `save_panorama` and read by `load_panorama`
# (a composite of several values, as `save_manifest`/`load_manifest`'s), and checked on the bytes by the kind's table.

const _GAIN_RECORD = _Obj(:t => _Int(), :gain => _Vec(_Real()))

function _panorama_check(x)
    r = x[:rotations]
    (r[:from] == frame_name(CameraFrame) && r[:to] == frame_name(WorldFrame)) ||
        return "rotations: a panorama's rotations map the camera to the world (`from` \"camera_optical\", `to` \"world\"), got $(repr(r[:from])) → $(repr(r[:to]))"
    for f in r[:frames]
        all(iszero, f[:pose][:translation]) ||
            return "rotations: frame $(f[:t]): a panorama's camera only rotates; its translation is zero, got $(collect(f[:pose][:translation]))"
    end
    haskey(x, :gains) || return nothing
    g = x[:gains]
    isempty(g) && return "gains: one record per frame of the rotations (none stated: omit `gains`)"
    Set(q[:t] for q in g) == Set(f[:t] for f in r[:frames]) == Set(1:length(g)) ||
        return "gains: one record per frame of the rotations, whose frame indices are then 1:K (row k of the gains is frame index k)"
    C = length(first(g)[:gain])
    C > 0 && all(q -> length(q[:gain]) == C, g) || return "gains: every frame has the same number ($(C) > 0) of colour planes"
    all(q -> all(>(0), q[:gain]), g) || return "gains: an exposure gain is positive"
    return nothing
end

const _PANORAMA_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("panorama"), :pixel_frame => _Tok("colmap"), _POSE_DECLARATIONS...,
            :camera => _BOUNDED_CAMERA_BODY, :rotations => _TRAJECTORY_BODY, :vignetting => _VignettingBody(),
        ],
        Pair{Symbol, _Spec}[:gains => _Vec(_GAIN_RECORD), _PROVENANCE...],
    ),
    _panorama_check,
)

"""
    save_panorama(path, camera::BoundedCamera, rotations::Trajectory, vignetting::AbstractVignetting, gains;
                  length_unit = mm, provenance = (;))

Write a panorama's results — the camera of a sweep that only rotated, its frames'
rotations, its vignetting and its frames' exposure gains (PanoramaStitching's) — as a
`vgio/panorama/v1` file, each through the body another kind writes: the camera
store's bounded camera body (K, lens, a sensor's `pitch` in `length_unit`, its lens
coverage), the trajectory body (`rotations`: per frame index its pose `from → to`,
its `image` and `time` where the trajectory has them) and the vgio/vignetting body,
positions in COLMAP pixels. The rotations map the camera to the world
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
    body = merge(
        (schema = _schema("panorama"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME)), pose_declarations(length_unit),
        (camera = _bounded_body(s, camera), rotations = JSON.lower(s, rotations), vignetting = JSON.lower(s, vignetting)),
        _gains_json(gains),
    )
    _write_store(path, _PANORAMA_SPEC, with_provenance(body, provenance); style = s)
    return nothing
end

_gains_json(::Nothing) = (;)
# Row k of the gains is frame index k: the table refuses gains whose frame indices are not the rotations', 1:K.
_gains_json(G::AbstractMatrix{<:Real}) = (gains = [(t = k, gain = row) for (k, row) in enumerate(_matrix_rows(G))],)

"""
    load_panorama(path) -> (; camera, rotations, vignetting, gains)

Read a `vgio/panorama/v1` file ([`save_panorama`](@ref)'s): `camera` the
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
    s = VgioStyle(; length_unit = parse_unit(d.length_unit::String))
    return (
        camera = _bounded_from_body(s, d.camera::Record), rotations = JSON.lift(s, Trajectory, d.rotations::Record),
        vignetting = JSON.lift(s, AbstractVignetting, d.vignetting::Record), gains = _gains_from_json(get(d, :gains, nothing)),
    )
end

_gains_from_json(::Nothing) = nothing
function _gains_from_json(g::AbstractVector)
    return _rows_matrix(r.gain for r in sort(g; by = r -> r.t::Integer))
end
