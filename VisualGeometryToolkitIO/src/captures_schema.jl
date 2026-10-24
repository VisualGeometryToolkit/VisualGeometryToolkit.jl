# =============================================================================
# The per-frame kinds of a capture — VisualGeometryToolkitCore's captures, serialized
# =============================================================================
#
#   vgio/trajectory/v3        ↔ VGC `Trajectory{S, D}`      a body's pose over time
#   vgio/target_poses/v1      ↔ VGC `TargetPoses{F}`        per frame, the poses of the targets seen
#   vgio/target_points2d/v1   ↔ VGC `TargetPoints2D{F}`     per frame, pixels of target points
#   vgio/target_points3d/v1   ↔ VGC `TargetPoints3D{F}`     per frame, 3D positions of target points
#
# Every one is a list of `frames`, each named by its index `t` — unique, the
# join key between the kinds of one capture — with, where the kind has them,
# `image` (the image taken at that instant), `sensor` (what measured it: a
# camera's name, a tracker's) and `time` (seconds since the file's
# `time_origin`, an ISO 8601 date-time without zone, stated iff a frame has a
# time). An instant not observed is ABSENT, never a NaN value. A file reads
# back as the VGC value it was written from, exactly (lengths in millimetres,
# `Uncertain` members with their covariance).
#
# The hierarchy: a RIG of targets (rig/v4) defines targets and their points; a
# measurement names a point by (target, index) — its `point_index_base`-based
# position in that target's `points` — and the measurement file names the
# targets files it resolves in by content identity (`targets`, see
# `content_id`); a trajectory is the pose of a body (`body:<id>`, a view, a
# target) in one frame.
#
# {"schema": "vgio/trajectory/v3", <pose declarations>, "from": "body:board_rig", "to": "world",
#  "time_origin": "2026-05-27T00:00:00",
#  "frames": [{"t": 17, "image": "DSC04520.JPG", "time": 47058.40,
#              "pose": {rotation, translation[, covariance]}, "recovered": true}]}
#      every pose maps `from` → `to`; `recovered` (written only when true) marks
#      a pose substituted under a stated method rather than measured
#
# {"schema": "vgio/target_poses/v1", <pose declarations>, "to": "camera_optical",
#  "frames": [{"t": 1, "image": "DSC04520.JPG", "sensor": "<camera name>", "time": 12.5,
#              "poses": [{"target": "89b", "rotation": […], "translation": […], "covariance": …}]}]}
#      each pose maps target:<target> → `to`: a detector's object → camera pose
#
# {"schema": "vgio/target_points2d/v1", "pixel_frame": "colmap", "point_index_base": 1,
#  "targets": ["<content id of a targets file>", …],
#  "frames": [{"t": 17, "image": "DSC0017.png", "sensor": "cam1", "time": …,
#              "points": [{"target": "<uid>", "index": 5, "xy": [u, v], "weight": 1.0,
#                          "covariance": [[…], […]]}]}]}
#
# {"schema": "vgio/target_points3d/v1", "length_unit": "mm", "frame": "world", "point_index_base": 1,
#  "targets": [ … ],                                                   (optional)
#  "frames": [{"t": 17, "image": …, "sensor": "mocap", "time": …,
#              "points": [{"target": "board_rig", "index": 3, "xyz": [x, y, z],
#                          "covariance": [[…3], …3 rows]}]}]}
#
# COLMAP, for orientation (no exporter): target_points2d is COLMAP's images.txt
# `points2D` with their `POINT3D_ID`, the (target, index) pair playing the
# `POINT3D_ID`; a rig of targets (rig/v4 targets) is COLMAP's `points3D`, kept as
# per-target frames (each target's points in its own TargetFrame, placed by its
# mount) with no stored tracks — the measurements are the tracks.
#
# Required per kind: trajectory — `pose`; target_poses — `image`, `sensor`;
# target_points2d — `image`, `sensor`, and at least one `targets` identity
# (the measurements cannot be resolved without it); target_points3d —
# `sensor` (its `targets` absent when the constellation is not known
# independently of these measurements — an estimate made FROM them cannot be
# named by them). The VGC values are permissive; `save` refuses a value
# missing what its kind requires.
#
# Uncertainty (VisualGeometryToolkitCore conventions/persisted-geometry.md): a pose's
# 6×6 `covariance` over [δω; δt] (rad, `length_unit`), a pixel's 2×2 (px²), a
# 3D point's 3×3 (`length_unit`²) — present iff the value is `Uncertain`.
#
# Related formats (guidance; exporters.jl emits TUM, EuRoC, BOP and TRC exactly):
# - TUM RGB-D `timestamp tx ty tz qx qy qz qw` (evo/evo/tools/file_interface.py:
#   121-174): a pose per explicit timestamp. Ours keeps the time and declares
#   its origin; quaternions stay w,x,y,z (declared); the frames are named.
# - EuRoC/ASL `#timestamp [ns], p_RS_R_x…, q_RS_w…` (file_interface.py:224-243):
#   the direction is in the NAME (`R` ← `S`); ours states `from`/`to`.
# - KITTI odometry: 3×4 row-major poses, no timestamps (file_interface.py:
#   177-220) — the implicit-index join replaced here by `t` + `image`.
# - BOP `scene_gt.json`: image id → [{cam_R_m2c, cam_t_m2c mm, obj_id}]
#   (bop_toolkit_lib/inout.py:267-300) — the per-image object list
#   target_poses adopts, model → camera; ours names the camera (`sensor`),
#   states the unit and keys targets by identity, not an integer.
# - OpenSim TRC (TRCFileAdapter.cpp:236-376) and C3D (ezc3d README:526-612):
#   per-marker X/Y/Z per frame, a missing marker empty / a negative residual.
#   target_points3d names markers by (body, index), drops an unseen one, and
#   replaces the scalar residual by a covariance when known.
# - Why target_points3d is a sibling kind and not target_points2d generalised:
#   the two share the (target, index) decomposition but not the measurement
#   space (a pixel in a pixel frame with a bundle weight, against a point in a
#   3D frame and a length unit).

# --- the per-frame facts: t, image, sensor, time ------------------------------------

const _FRAMES_CHECK = x -> something(
    _unique_check([f[:t] for f in x[:frames]], "frame indices `t`"),
    (any(f -> haskey(f::AbstractDict, :time), x[:frames]::AbstractVector) && !haskey(x, :time_origin)) ?
        "a frame has a `time` (seconds since `time_origin`) but the file states no `time_origin`" : nothing,
    Some(nothing),
)

# A frame's stamps, as written: `image` and `sensor` where the value has them
# (required ones are checked by the spec), `time` where it has one.
function _stamps_json(x, t)
    f = (t = t,)
    haskey(x.images, t) && (f = merge(f, (image = x.images[t],)))
    hasproperty(x, :sensors) && haskey(x.sensors, t) && (f = merge(f, (sensor = x.sensors[t],)))
    haskey(x.times, t) && (f = merge(f, (time = x.times[t],)))
    return f
end
_time_origin_json(x) = x.time_origin === nothing ? (;) : (time_origin = x.time_origin,)

# …and read back into the dicts a VGC capture is built from.
function _read_stamps!(images, sensors, times, f, t)
    haskey(f, :image) && (images[t] = f.image::String)
    (sensors === nothing || !haskey(f, :sensor)) || (sensors[t] = f.sensor::String)
    haskey(f, :time) && (times[t] = Float64(f.time::Real))
    return nothing
end
_time_origin(d) = haskey(d, :time_origin) ? String(d.time_origin::String) : nothing

_stamp_specs(required::Vector{Symbol}) = (
    Pair{Symbol, _Spec}[k => _NonEmpty() for k in required],
    Pair{Symbol, _Spec}[k => (k === :time ? _Real() : _NonEmpty()) for k in (:image, :sensor, :time) if !(k in required)],
)

# =============================================================================
# trajectory/v3 ↔ Trajectory
# =============================================================================

const _TRAJECTORY_FRAME = _Obj(
    Pair{Symbol, _Spec}[:t => _Int(), :pose => _UPOSE],
    Pair{Symbol, _Spec}[:image => _NonEmpty(), :time => _Real(), :recovered => _Tok(true)],
)
const _TRAJECTORY_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("trajectory"), _POSE_DECLARATIONS...,
            :from => _FRAME_NAME, :to => _FRAME_NAME,
            :frames => _Vec(_TRAJECTORY_FRAME),
        ],
        Pair{Symbol, _Spec}[:time_origin => _NonEmpty(), _PROVENANCE...],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, tr::Trajectory{S, D}; provenance = (;))

Write a VGC `Trajectory` as a `vgio/trajectory/v3` file: its frames
`from = frame_name(S)`, `to = frame_name(D)`, one record per frame index
(sorted) — the pose (a `covariance` iff `Uncertain`), and its `image`, `time`
and `recovered` flag where the trajectory has them — in millimetres.
"""
function save(path::AbstractString, tr::Trajectory{S, D}; provenance::NamedTuple = (;)) where {S, D}
    frames = map(sort!(collect(keys(tr)))) do t
        f = merge(_stamps_json(tr, t), (pose = pose_json(tr[t]),))
        t in tr.recovered ? merge(f, (recovered = true,)) : f
    end
    body = merge(
        (schema = _schema("trajectory"),), pose_declarations(mm),
        (from = frame_name(S), to = frame_name(D)), _time_origin_json(tr), (frames = frames,),
    )
    _write_store(path, _TRAJECTORY_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(Trajectory, path) -> Trajectory{S, D}

Read a `vgio/trajectory/v3` file: its frames as types (`S`, `D` from
`from`/`to`), each pose labeled `S → D` in millimetres — `Uncertain` where the
file states a covariance — with the images, times and recovered flags it
records. Type-unstable by nature (the frame types come from the file): load
once, dispatch after.
"""
function load(::Type{Trajectory}, path::AbstractString)
    d = _read_store(path, "trajectory", _TRAJECTORY_SPEC)
    unit = d.length_unit::String
    poses, images, times, recovered = Dict{Int, Any}(), Dict{Int, String}(), Dict{Int, Float64}(), Set{Int}()
    for f in d.frames::JSON3.Array
        t = Int(f.t::Integer)
        poses[t] = pose_from_json(f.pose, unit)
        _read_stamps!(images, nothing, times, f, t)
        get(f, :recovered, false) === true && push!(recovered, t)
    end
    return Trajectory(
        frame_from_name(d.from::String), frame_from_name(d.to::String), poses;
        images, times, recovered, time_origin = _time_origin(d),
    )
end

# =============================================================================
# target_poses/v1 ↔ TargetPoses
# =============================================================================

const _TARGET_POSE_RECORD = _Obj(
    Pair{Symbol, _Spec}[:target => _NonEmpty(), :rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3)],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _TARGET_POSES_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_poses"), _POSE_DECLARATIONS...,
            :to => _FRAME_NAME,
            :frames => _Vec(
                _Checked(
                    _Obj(
                        Pair{Symbol, _Spec}[:t => _Int(), _stamp_specs([:image, :sensor])[1]..., :poses => _Vec(_TARGET_POSE_RECORD)],
                        _stamp_specs([:image, :sensor])[2],
                    ),
                    f -> _unique_check([p[:target] for p in f[:poses]], "the targets of a frame"),
                )
            ),
        ],
        Pair{Symbol, _Spec}[:time_origin => _NonEmpty(), _PROVENANCE...],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, tp::TargetPoses{F}; provenance = (;))

Write a VGC `TargetPoses` as a `vgio/target_poses/v1` file declaring
`to = frame_name(F)`: per frame index (sorted), its image and sensor (both
required) and time, and every target's pose (ids sorted; a `covariance` iff
`Uncertain`), in millimetres.
"""
function save(path::AbstractString, tp::TargetPoses{F}; provenance::NamedTuple = (;)) where {F}
    frames = map(sort!(collect(keys(tp)))) do t
        poses = [merge((target = String(id),), pose_json(tp[t][id])) for id in sort!(collect(keys(tp[t])))]
        merge(_stamps_json(tp, t), (poses = poses,))
    end
    body = merge(
        (schema = _schema("target_poses"),), pose_declarations(mm), (to = frame_name(F),),
        _time_origin_json(tp), (frames = frames,),
    )
    _write_store(path, _TARGET_POSES_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(TargetPoses, path) -> TargetPoses{F}

Read a `vgio/target_poses/v1` file: `F` the frame its poses map into, every
pose in millimetres, `Uncertain` where the file states a covariance.
"""
function load(::Type{TargetPoses}, path::AbstractString)
    d = _read_store(path, "target_poses", _TARGET_POSES_SPEC)
    unit = d.length_unit::String
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::JSON3.Array
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(p.target::String) => pose_from_json(p, unit) for p in f.poses::JSON3.Array)
    end
    return TargetPoses{frame_from_name(d.to::String)}(frames; images, sensors, times, time_origin = _time_origin(d))
end

# =============================================================================
# target_points2d/v1 ↔ TargetPoints2D
# =============================================================================

const _TARGET_POINTS2D_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_points2d"), :pixel_frame => _PIXEL_FRAME,
            :point_index_base => _Tok(_POINT_INDEX_BASE),
            :targets => _Checked(_Vec(_HEX64), h -> isempty(h) ? "the measurements name at least one targets file" : _unique_check(h, "targets identities")),
            :frames => _Vec(
                _Obj(
                    Pair{Symbol, _Spec}[
                        :t => _Int(), _stamp_specs([:image, :sensor])[1]...,
                        :points => _Vec(
                            _Obj(
                                Pair{Symbol, _Spec}[:target => _NonEmpty(), :index => _PosInt(), :xy => _XY, :weight => _Real()],
                                Pair{Symbol, _Spec}[:covariance => _COV2],
                            )
                        ),
                    ],
                    _stamp_specs([:image, :sensor])[2],
                )
            ),
        ],
        Pair{Symbol, _Spec}[:time_origin => _NonEmpty(), _PROVENANCE...],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, obs::TargetPoints2D{F}; provenance = (;))

Write a VGC `TargetPoints2D` as a `vgio/target_points2d/v1` file whose
declared pixel frame is `F`, the value's own type — no pixel is converted and
none can be written under another frame's label. Every frame must name its
image and sensor, and the value must name the target definitions its indices
resolve in (`obs.targets`, their [`content_id`](@ref)s).
"""
function save(path::AbstractString, obs::TargetPoints2D{F}; provenance::NamedTuple = (;)) where {F <: PixelFrame}
    frames = map(sort!(collect(keys(obs)))) do t
        points = map(obs[t]) do o
            xy = geom(o.xy)
            p = (target = String(o.target), index = o.index, xy = [xy[1], xy[2]], weight = o.weight)
            o.xy isa Uncertain ? merge(p, (covariance = _cov_json(cov(o.xy)),)) : p
        end
        merge(_stamps_json(obs, t), (points = points,))
    end
    body = merge(
        (
            schema = _schema("target_points2d"), pixel_frame = pixel_frame_name(F),
            point_index_base = _POINT_INDEX_BASE, targets = obs.targets,
        ),
        _time_origin_json(obs), (frames = frames,),
    )
    _write_store(path, _TARGET_POINTS2D_SPEC, _with_provenance(body, provenance))
    return nothing
end

function _pixel_from_json(o)
    xy = SVector{2, Float64}(Float64(o.xy[1]::Real), Float64(o.xy[2]::Real))
    return haskey(o, :covariance) ? Uncertain(xy, CovMat(_cov_matrix(o.covariance::JSON3.Array, 2))) : xy
end

"""
    load(TargetPoints2D, path) -> TargetPoints2D{F}

Read a `vgio/target_points2d/v1` file: per frame index, the measured pixels
typed in the pixel frame `F` the file declares, exactly as stored, with the
frames' images, sensors and times and the targets identities. Change the frame
with `convert_frame`.
"""
function load(::Type{TargetPoints2D}, path::AbstractString)
    d = _read_store(path, "target_points2d", _TARGET_POINTS2D_SPEC)
    frames = Dict{Int, Vector{TargetPoint2D}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::JSON3.Array
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = TargetPoint2D[
            (target = Symbol(o.target::String), index = Int(o.index::Integer), xy = _pixel_from_json(o), weight = Float64(o.weight::Real))
                for o in f.points::JSON3.Array
        ]
    end
    return TargetPoints2D{pixel_frame_from_name(d.pixel_frame::String)}(
        frames; images, sensors, times, time_origin = _time_origin(d), targets = String[h::String for h in d.targets::JSON3.Array],
    )
end

# =============================================================================
# target_points3d/v1 ↔ TargetPoints3D
# =============================================================================

const _TARGET_POINTS3D_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_points3d"), :length_unit => _LENGTH_UNIT,
            :frame => _FRAME_NAME, :point_index_base => _Tok(_POINT_INDEX_BASE),
            :frames => _Vec(
                _Obj(
                    Pair{Symbol, _Spec}[
                        :t => _Int(), _stamp_specs([:sensor])[1]...,
                        :points => _Vec(
                            _Obj(
                                Pair{Symbol, _Spec}[:target => _NonEmpty(), :index => _PosInt(), :xyz => _XYZ],
                                Pair{Symbol, _Spec}[:covariance => _COV3],
                            )
                        ),
                    ],
                    _stamp_specs([:sensor])[2],
                )
            ),
        ],
        Pair{Symbol, _Spec}[
            :targets => _Checked(_Vec(_HEX64), h -> _unique_check(h, "targets identities")),
            :time_origin => _NonEmpty(), _PROVENANCE...,
        ],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, obs::TargetPoints3D{F}; provenance = (;))

Write a VGC `TargetPoints3D` as a `vgio/target_points3d/v1` file in frame `F`,
millimetres: per frame its sensor (required), image and time where stated; an
`Uncertain` point writes its 3×3 `covariance`.
"""
function save(path::AbstractString, obs::TargetPoints3D{F}; provenance::NamedTuple = (;)) where {F}
    frames = map(sort!(collect(keys(obs)))) do t
        points = map(obs[t]) do o
            p = geom(o.xyz)
            r = (target = String(o.target), index = o.index, xyz = [p[1], p[2], p[3]])
            o.xyz isa Uncertain ? merge(r, (covariance = _cov_json(cov(o.xyz)),)) : r
        end
        merge(_stamps_json(obs, t), (points = points,))
    end
    body = merge(
        (
            schema = _schema("target_points3d"), length_unit = unit_string(mm), frame = frame_name(F),
            point_index_base = _POINT_INDEX_BASE,
        ),
        isempty(obs.targets) ? (;) : (targets = obs.targets,),
        _time_origin_json(obs), (frames = frames,),
    )
    _write_store(path, _TARGET_POINTS3D_SPEC, _with_provenance(body, provenance))
    return nothing
end

# One wire point in millimetres (`s` mm per the file's unit).
function _point3_from_json(o, unit, s)
    p = SVector{3, Float64}(ustrip.(mm, quantity.(Float64[Float64(c::Real) for c in o.xyz::JSON3.Array], unit))...)
    xyz = haskey(o, :covariance) ? Uncertain(p, CovMat(_cov_matrix(o.covariance::JSON3.Array, 3) * s^2)) : p
    return (target = Symbol(o.target::String), index = Int(o.index::Integer), xyz = xyz)
end

"""
    load(TargetPoints3D, path) -> TargetPoints3D{F}

Read a `vgio/target_points3d/v1` file: the measurements in the frame `F` it
declares, in millimetres (`Uncertain` where it states a covariance, in mm²).
"""
function load(::Type{TargetPoints3D}, path::AbstractString)
    d = _read_store(path, "target_points3d", _TARGET_POINTS3D_SPEC)
    unit = parse_unit(d.length_unit::String)
    s = Float64(ustrip(mm, quantity(1.0, unit)))                         # mm per the file's unit
    frames = Dict{Int, Vector{TargetPoint3D}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::JSON3.Array
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = TargetPoint3D[_point3_from_json(o, unit, s) for o in f.points::JSON3.Array]
    end
    return TargetPoints3D{frame_from_name(d.frame::String)}(
        frames; images, sensors, times, time_origin = _time_origin(d),
        targets = haskey(d, :targets) ? String[h::String for h in d.targets::JSON3.Array] : String[],
    )
end
