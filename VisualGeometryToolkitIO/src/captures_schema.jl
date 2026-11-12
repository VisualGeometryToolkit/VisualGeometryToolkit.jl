# =============================================================================
# The per-frame kinds of a capture — VisualGeometryToolkitCore's captures, serialized
# =============================================================================
#
#   vgio/trajectory/v4        ↔ VGC `Trajectory{S, D}`      a body's pose over time
#   vgio/target_poses/v2      ↔ VGC `TargetPoses{F}`        per frame, the poses of the targets seen
#   vgio/target_homographies/v1 ↔ VGC `TargetHomographies{F, H}`
#                                                         per frame, each target's plane → pixel homography
#   vgio/target_points2d/v2   ↔ VGC `TargetPoints2D{F}`     per frame, pixels of target points
#   vgio/target_points3d/v2   ↔ VGC `TargetPoints3D{F}`     per frame, 3D positions of target points
#   vgio/target_detections/v2 ↔ VGC `TargetDetections{F}`   per frame, per target, index => pixel or 3D point
#                               (load(TargetDetections, …) reads the two kinds above too, converted)
#
# Every one is a list of `frames`, each named by its index `t` — unique, the
# join key between the kinds of one capture — with, where the kind has them,
# `image` (the image taken at that instant), `sensor` (what measured it: a
# camera's name, a tracker's) and `time` (seconds since the file's
# `time_origin`, an ISO 8601 date-time without zone, stated iff a frame has a
# time). An instant not observed is ABSENT, never a NaN value. A file reads
# back as the VGC value it was written from, exactly (lengths in metres in
# memory, in the declared `length_unit` on disk; `Uncertain` members with
# their covariance). A writer refuses plain numbers where a length is meant.
#
# The hierarchy: a RIG of targets (a vgio/rig file) defines targets and their points; a
# measurement names a point by (target, index) — its `point_index_base`-based
# position in that target's `points` — and the measurement file names the
# targets files it resolves in by content identity (`targets`, see
# `content_id`); a trajectory is the pose of a body (`body:<id>`, a view, a
# target) in one frame.
#
# {"schema": "vgio/trajectory/v4", <pose declarations>, "from": "body:board_rig", "to": "world",
#  "time_origin": "2026-05-27T00:00:00",
#  "frames": [{"t": 17, "image": "DSC04520.JPG", "time": 47058.40,
#              "pose": {rotation, translation[, covariance]}, "recovered": true}]}
#      every pose maps `from` → `to`; `recovered` (written only when true) marks
#      a pose substituted under a stated method rather than measured
#
# {"schema": "vgio/target_poses/v2", <pose declarations>, "to": "camera_optical",
#  "frames": [{"t": 1, "image": "DSC04520.JPG", "sensor": "<camera name>", "time": 12.5,
#              "poses": [{"target": "89b", "rotation": […], "translation": […], "covariance": …}]}]}
#      each pose maps target:<target> → `to`: a detector's object → camera pose
#
# {"schema": "vgio/target_homographies/v1", "pixel_frame": "colmap",
#  "targets": ["<content id of a targets rig or of a single target>", …],
#  "frames": [{"t": 1, "image": "DSC04520.JPG", "sensor": "<camera name>", "time": 12.5,
#              "homographies": [{"target": "89b", "type": "homography_map",
#                                "parameters": [h11, h12, h13, h21, h22, h23, h31, h32, h33],
#                                "covariance": …8×8 (optional)}]}]}
#      each `target` names a board inside one of the `targets` — a `content_id` of
#      EITHER a rig (rig/v9) of boards OR a single unposed target (target/v5); at
#      least one. Each homography is VGC's `HomographyMap` as the parametric record
#      (style.jl): its nine canonical entries (unit Frobenius norm, h33 ≥ 0), an
#      `Uncertain` one's covariance in the HomographyManifold8 chart. It maps the
#      target's plane — its own design frame, millimetres — to a pixel of the declared
#      `pixel_frame`, as `RadialHomography`'s H does: BEFORE the lens of the camera
#      `sensor` names, about that camera's centre. An H is paired with that camera by
#      VGC's `radial_homography(camera, H)`: a PROJECTIVE camera (principal point +
#      division in pixels) is the model itself; a METRIC one (K with fx == fy, no
#      skew; division in K's normalized chart) converts to λₚ = kₚ / f^(2p) about its
#      principal point — equivalently, the pixel of a plane point X is
#      project(camera, K⁻¹·H·[X; 1]). A homography has no 3D frame, so the pixel frame
#      is the TYPE parameter of the value (`TargetHomographies{F}`), as for
#      target_points2d, and `targets` its field.
#
# {"schema": "vgio/target_points2d/v2", "pixel_frame": "colmap", "point_index_base": 1,
#  "targets": ["<content id of a targets file>", …],
#  "frames": [{"t": 17, "image": "DSC0017.png", "sensor": "cam1", "time": …,
#              "points": [{"target": "<uid>", "index": 5, "xy": [u, v], "weight": 1.0,
#                          "covariance": [[…], […]]}]}]}
#
# {"schema": "vgio/target_points3d/v2", "length_unit": "mm", "frame": "world", "point_index_base": 1,
#  "targets": [ … ],                                                   (optional)
#  "frames": [{"t": 17, "image": …, "sensor": "mocap", "time": …,
#              "points": [{"target": "board_rig", "index": 3, "xyz": [x, y, z],
#                          "covariance": [[…3], …3 rows]}]}]}
#
# {"schema": "vgio/target_detections/v2", "pixel_frame": "colmap", "point_index_base": 1,
#  "targets": [ … ],                       (or "frame": "world", "length_unit": "mm"; targets optional)
#  "frames": [{"t": 17, "image": …, "sensor": "cam1", "time": …,
#              "detections": [{"target": "<uid>", "index": [1, 5], "points": [[u, v], [u, v]],
#                              "covariance": [[[…], […]], [[…], […]]], "weights": [1.0, 0.5]}]}]}
#      per target the indices (unique), its points, and per point a covariance iff
#      Uncertain and a weight iff the set has a `weights` column; required stamps as
#      target_points2d (pixels) or target_points3d (3D points). Each `targets` id is
#      the `content_id` of a rig (rig/v9) of targets OR of a single unposed target
#      (target/v5); a single target's detections are named by its design `reference`
#      (or by its content id when it has none).
#
# COLMAP, for orientation (no exporter): target_points2d is COLMAP's images.txt
# `points2D` with their `POINT3D_ID`, the (target, index) pair playing the
# `POINT3D_ID`; a rig of targets (a vgio/rig file's targets) is COLMAP's `points3D`, kept as
# per-target frames (each target's points in its own TargetFrame, placed by its
# mount) with no stored tracks — the measurements are the tracks.
#
# Required per kind: trajectory — `pose`; target_poses, target_homographies —
# `image`, `sensor`;
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
# trajectory/v4 ↔ Trajectory
# =============================================================================

const _TRAJECTORY_FRAME = _Obj(
    Pair{Symbol, _Spec}[:t => _Int(), :pose => _UPOSE],
    Pair{Symbol, _Spec}[:image => _NonEmpty(), :time => _Real(), :recovered => _Tok(true)],
)
const _TRAJECTORY_SPEC = _CheckedSpec(
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
    save(path, tr::Trajectory{S, D}; length_unit = mm, provenance = (;))

Write a physical VGC `Trajectory` as a `vgio/trajectory/v4` file: its frames
`from = frame_name(S)`, `to = frame_name(D)`, one record per frame index
(sorted) — the pose (a `covariance` iff `Uncertain`), and its `image`, `time`
and `recovered` flag where the trajectory has them — in `length_unit`. A
trajectory of plain poses is an `ArgumentError` (attach the unit, `t .* m`).
"""
function save(path::AbstractString, tr::Trajectory; length_unit::Units = mm, provenance::NamedTuple = (;))
    body = merge((schema = _schema("trajectory"),), pose_declarations(length_unit), JSON.lower(VgioStyle(; length_unit), tr))
    _write_store(path, _TRAJECTORY_SPEC, with_provenance(body, provenance); style = VgioStyle(; length_unit))
    return nothing
end

# A Trajectory on IO's style: the body of a `vgio/trajectory` file without its
# schema and its pose declarations (`pose_declarations`, the enclosing file's):
# `from`, `to`, the `time_origin` where the trajectory has one, and one record
# per frame index, sorted — its stamps and its pose, lowered on the style.
function JSON.lower(::VgioStyle, tr::Trajectory{S, D}) where {S, D}
    frames = map(sort!(collect(keys(tr)))) do t
        f = merge(_stamps_json(tr, t), (pose = tr[t],))   # the pose itself, lowered on the file's style
        t in tr.recovered ? merge(f, (recovered = true,)) : f
    end
    return merge((from = frame_name(S), to = frame_name(D)), _time_origin_json(tr), (frames = frames,))
end

"""
    load(Trajectory, path) -> Trajectory{S, D}

Read a `vgio/trajectory/v4` file: its frames as types (`S`, `D` from
`from`/`to`), each pose labeled `S → D`, lengths in metres — `Uncertain` where the
file states a covariance — with the images, times and recovered flags it
records. Type-unstable by nature (the frame types come from the file): load
once, dispatch after. An earlier version reads through its version steps; of
v2 (free frame names) only CaliBlobs' 2026-09 track segment (`track:<s>` →
`camera`), read as `target:scene_<c>` → `camera_optical` (src/migrate.jl).
"""
function load(::Type{Trajectory}, path::AbstractString)
    d = _read_store(path, "trajectory", _TRAJECTORY_SPEC)
    return JSON.lift(VgioStyle(; length_unit = parse_unit(d.length_unit::String)), Trajectory, d)
end
# …and back: a trajectory body read on the style (its lengths in the style's unit).
function JSON.lift(s::VgioStyle, ::Type{Trajectory}, d::Record)
    poses, images, times, recovered = Dict{Int, Any}(), Dict{Int, String}(), Dict{Int, Float64}(), Set{Int}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        poses[t] = JSON.lift(s, EuclideanMap{3}, f.pose::Record)
        _read_stamps!(images, nothing, times, f, t)
        get(f, :recovered, false) === true && push!(recovered, t)
    end
    return Trajectory(
        frame_from_name(d.from::String), frame_from_name(d.to::String), poses;
        images, times, recovered, time_origin = _time_origin(d),
    )
end

# =============================================================================
# target_poses/v2 ↔ TargetPoses
# =============================================================================

const _TARGET_POSE_RECORD = _Obj(
    Pair{Symbol, _Spec}[:target => _NonEmpty(), :rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3)],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _TARGET_POSES_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_poses"), _POSE_DECLARATIONS...,
            :to => _FRAME_NAME,
            :frames => _Vec(
                _CheckedSpec(
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
    save(path, tp::TargetPoses{F}; length_unit = mm, provenance = (;))

Write a physical VGC `TargetPoses` as a `vgio/target_poses/v2` file declaring
`to = frame_name(F)`: per frame index (sorted), its image and sensor (both
required) and time, and every target's pose (ids sorted; a `covariance` iff
`Uncertain`), in `length_unit`. Plain poses are an `ArgumentError`.
"""
function save(path::AbstractString, tp::TargetPoses{F}; length_unit::Units = mm, provenance::NamedTuple = (;)) where {F}
    s = VgioStyle(; length_unit)
    frames = map(sort!(collect(keys(tp)))) do t
        poses = [merge((target = String(id),), JSON.lower(s, tp[t][id])) for id in sort!(collect(keys(tp[t])))]
        merge(_stamps_json(tp, t), (poses = poses,))
    end
    body = merge(
        (schema = _schema("target_poses"),), pose_declarations(length_unit), (to = frame_name(F),),
        _time_origin_json(tp), (frames = frames,),
    )
    _write_store(path, _TARGET_POSES_SPEC, with_provenance(body, provenance); style = s)
    return nothing
end

"""
    load(TargetPoses, path) -> TargetPoses{F}

Read a `vgio/target_poses/v2` file: `F` the frame its poses map into, every
pose's lengths in metres, `Uncertain` where the file states a covariance.
"""
function load(::Type{TargetPoses}, path::AbstractString)
    d = _read_store(path, "target_poses", _TARGET_POSES_SPEC)
    s = VgioStyle(; length_unit = parse_unit(d.length_unit::String))
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(p.target::String) => JSON.lift(s, EuclideanMap{3}, p::Record) for p in f.poses::Vector{Any})
    end
    return TargetPoses{frame_from_name(d.to::String)}(frames; images, sensors, times, time_origin = _time_origin(d))
end

# =============================================================================
# target_homographies/v1 ↔ TargetHomographies
# =============================================================================

# The homography record (style.jl's parametric record): a plane homography is
# `{"type": "homography_map", "parameters": [h11, …, h33]}` — VGC's parameter
# protocol of a `HomographyMap`, its nine canonical entries row-major (unit
# Frobenius norm, H[3,3] ≥ 0) — and, for an `Uncertain` one, its 8×8 covariance
# in the `HomographyManifold8` chart about that matrix (dimensionless, in the
# storage it holds). Its frames are not written: they are given by where it
# sits (a target_homographies entry: its target's plane → the file's pixel frame).
const _HOMOGRAPHY_TAG = tag_of(PLANAR_MAP_TAGS, HomographyMap)
const _HOMOGRAPHY_PROTOTYPE = HomographyMap(HomographyMat{Float64}(one(SMatrix{3, 3, Float64, 9})))
const _HOMOGRAPHY_COV_DIMS = ntuple(_ -> 0, 8)

JSON.lower(s::VgioStyle, h::HomographyMap{3}) = _parametric_record(_HOMOGRAPHY_TAG, h, s.length_unit)
JSON.lower(s::VgioStyle, u::Uncertain{<:HomographyMap{3}}) =
    merge(JSON.lower(s, geom(u)), (covariance = _length_cov_json(cov(u), s.length_unit, _HOMOGRAPHY_COV_DIMS),))
function JSON.lift(s::VgioStyle, ::Type{HomographyMap}, x::Record)
    h = _from_parametric_record(_HOMOGRAPHY_PROTOTYPE, x, s.length_unit)
    haskey(x, :covariance) || return h
    return Uncertain(h, CovMat(_length_cov_from_json(x.covariance, 8, s.length_unit, _HOMOGRAPHY_COV_DIMS)))
end

# A stored record's parameters are canonical as read (what `parameter_vector`
# gives back for them) and not singular: VGC's protocol decides both.
function _homography_parameters_check(θ)
    h = try
        with_parameters(_HOMOGRAPHY_PROTOTYPE, Float64[Float64(x) for x in θ])
    catch e
        e isa ArgumentError || rethrow()
        return e.msg
    end
    return parameter_vector(h) == θ ? nothing :
        "the homography is not canonical: its parameters must have unit Frobenius norm and h33 ≥ 0 (parameter_vector of HomographyMap)"
end
const _HOMOGRAPHY_RECORD = _Obj(
    Pair{Symbol, _Spec}[
        :target => _NonEmpty(), :type => _Tok(_HOMOGRAPHY_TAG),
        :parameters => _CheckedSpec(_Vec(_Real(), 9), _homography_parameters_check),
    ],
    Pair{Symbol, _Spec}[:covariance => _cov_spec(8)],
)

const _TARGET_HOMOGRAPHIES_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_homographies"), :pixel_frame => _PIXEL_FRAME,
            :targets => _CheckedSpec(_Vec(_HEX64), h -> isempty(h) ? "the homographies name at least one targets identity (a rig's or a single target's `content_id`)" : _unique_check(h, "targets identities")),
            :frames => _Vec(
                _CheckedSpec(
                    _Obj(
                        Pair{Symbol, _Spec}[
                            :t => _Int(), _stamp_specs([:image, :sensor])[1]...,
                            :homographies => _Vec(_HOMOGRAPHY_RECORD),
                        ],
                        _stamp_specs([:image, :sensor])[2],
                    ),
                    f -> _unique_check([h[:target] for h in f[:homographies]], "the targets of a frame"),
                )
            ),
        ],
        Pair{Symbol, _Spec}[:time_origin => _NonEmpty(), _PROVENANCE...],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, th::TargetHomographies{F}; provenance = (;))

Write a VGC `TargetHomographies` as a `vgio/target_homographies/v1` file whose
declared pixel frame is `F`, the value's own type — no pixel is converted and
none can be written under another frame's label — and whose `targets` are the
value's `th.targets`: the [`content_id`](@ref)s of the definitions its target
ids resolve in, each the identity of a rig of boards (`vgio/rig/v9`) or of a
single unposed target (`vgio/target/v5`). Per frame index (sorted) it writes
the image and sensor (both required), the time, and every target's
homography (ids sorted) as the homography record:

    {"target": "a1f2", "type": "homography_map", "parameters": [h11, h12, h13, h21, h22, h23, h31, h32, h33]}

— VGC's parameter protocol of the `HomographyMap`: its nine entries row-major,
canonical (unit Frobenius norm, `h33 ≥ 0`, VGC `sign_normalize`), so one
placement always writes the same bytes and a stored `H` seeds
`HomographyManifold8(H)` directly; an `H` already canonical is written as it
is. An `Uncertain` homography adds its 8×8 `covariance` in that chart.

`H` maps the target's plane — its design frame, millimetres — to a pixel of
`F`, BEFORE the lens of the camera the frame's `sensor` names, about that
camera's centre: exactly `RadialHomography`'s `H`. An `H` is paired with that
camera by VGC's `radial_homography(camera, H)`: for a projective camera the camera
itself is the model; for a metric one (`fx == fy`, no skew) it converts the
division coefficients `kₚ` of `K`'s normalized chart to `λₚ = kₚ / f^(2p)` about
the principal point — equivalently, a plane point `X` lands on the pixel
`project(camera, K⁻¹·H·[X; 1])`.

A degenerate `H` (negligible norm, or non-finite), a value naming no
`targets`, and a frame with no image or sensor are an `ArgumentError`, and no
file is written.
"""
function save(path::AbstractString, th::TargetHomographies{F}; provenance::NamedTuple = (;)) where {F <: PixelFrame}
    s = VgioStyle(; pixel_frame = F)
    record(t, id) = try
        merge((target = String(id),), JSON.lower(s, th[t][id]))
    catch e
        e isa ArgumentError ? throw(ArgumentError("$(path): frame $(t), target $(id): $(e.msg)")) : rethrow()
    end
    frames = map(sort!(collect(keys(th)))) do t
        merge(_stamps_json(th, t), (homographies = [record(t, id) for id in sort!(collect(keys(th[t])))],))
    end
    body = merge(
        (schema = _schema("target_homographies"), pixel_frame = pixel_frame_name(F), targets = copy(th.targets)),
        _time_origin_json(th), (frames = frames,),
    )
    _write_store(path, _TARGET_HOMOGRAPHIES_SPEC, with_provenance(body, provenance); style = s)
    return nothing
end

"""
    load(TargetHomographies, path) -> TargetHomographies{F}
    load(TargetHomographies{F}, path) -> TargetHomographies{F}

Read a `vgio/target_homographies/v1` file: the value it was written from,
typed in the pixel frame `F` the file declares, per frame index every
target's `HomographyMap` (stored unlabeled; `homography(th, t, id)` labels it
`TargetPlaneFrame → F`) with its canonical parameters exactly as stored,
`Uncertain` where the record states a covariance, with the frames' images,
sensors and times and the `targets` identities. `H` maps the target's plane
(millimetres) to a pixel of `F`, before the lens of the camera the frame's
`sensor` names (pair them with VGC's `radial_homography(camera, H)`; see
[`save`](@ref)).

The second form asserts the frame: a file declaring another is an
`ArgumentError`. So are parameters that are not canonical or are singular.
"""
function load(::Type{TargetHomographies}, path::AbstractString)
    d = _read_store(path, "target_homographies", _TARGET_HOMOGRAPHIES_SPEC)
    s = VgioStyle(; pixel_frame = pixel_frame_from_name(d.pixel_frame::String))
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(h.target::String) => JSON.lift(s, HomographyMap, h::Record) for h in f.homographies::Vector{Any})
    end
    return TargetHomographies{_pixel_frame(s)}(
        frames; images, sensors, times, time_origin = _time_origin(d), targets = String[h::String for h in d.targets::Vector{Any}],
    )
end
_pixel_frame_type(::TargetHomographies{F}) where {F} = F
function load(::Type{<:TargetHomographies{F}}, path::AbstractString) where {F <: PixelFrame}
    th = load(TargetHomographies, path)
    th isa TargetHomographies{F} || throw(
        ArgumentError("$(path): declares pixel_frame $(repr(pixel_frame_name(_pixel_frame_type(th)))), not the requested $(repr(pixel_frame_name(F)))")
    )
    return th
end

# =============================================================================
# target_points2d/v1 ↔ TargetPoints2D
# =============================================================================

const _TARGET_POINTS2D_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("target_points2d"), :pixel_frame => _PIXEL_FRAME,
            :point_index_base => _Tok(_POINT_INDEX_BASE),
            :targets => _CheckedSpec(_Vec(_HEX64), h -> isempty(h) ? "the measurements name at least one targets file" : _unique_check(h, "targets identities")),
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
            p = (target = String(o.target), index = o.index, xy = _pixel_json(geom(o.xy)), weight = o.weight)
            o.xy isa Uncertain ? merge(p, (covariance = _cov_json(cov(o.xy)),)) : p   # px²: no length
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
    _write_store(path, _TARGET_POINTS2D_SPEC, with_provenance(body, provenance); style = VgioStyle(; pixel_frame = F))
    return nothing
end

# A pixel on the wire, as stored (px², never converted): its two numbers, and
# its 2×2 covariance where the value is `Uncertain` (`nothing` where not).
_pixel_json(xy) = Float64[xy[1], xy[2]]
function _pixel_from_json(xy, covariance)
    p = SVector{2, Float64}(Float64(xy[1]::Real), Float64(xy[2]::Real))
    return covariance === nothing ? p : Uncertain(p, CovMat(_cov_matrix(covariance, 2)))
end

# The parsed, checked record as the value (`load(TargetPoints2D, path)`'s, and
# `load(TargetDetections, path)`'s conversion of it).
function _target_points2d(d)
    frames = Dict{Int, Vector{TargetPoint2D}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = TargetPoint2D[
            (target = Symbol(o.target::String), index = Int(o.index::Integer), xy = _pixel_from_json(o.xy, get(o, :covariance, nothing)), weight = Float64(o.weight::Real))
                for o in f.points::Vector{Any}
        ]
    end
    return TargetPoints2D{pixel_frame_from_name(d.pixel_frame::String)}(
        frames; images, sensors, times, time_origin = _time_origin(d), targets = String[h::String for h in d.targets::Vector{Any}],
    )
end

"""
    load(TargetPoints2D, path; target = nothing) -> TargetPoints2D{F}

Read a `vgio/target_points2d/v1` file: per frame index, the measured pixels
typed in the pixel frame `F` the file declares, exactly as stored, with the
frames' images, sensors and times and the targets identities. Change the frame
with `convert_frame`.

A Schöps et al. camera_calibration `dataset.bin` (recognized by its
`calib_data` magic, see schoeps.jl) is read too: frame `t` is the file's
`t`-th image (its name in `images`), each detection a point of `target` —
which the file does not name, so the caller states it — with `index` the
feature id + 1 (the ids are 0-based) and weight 1, typed in
[`SCHOEPS_PIXEL_FRAME`](@ref). A `target` beside a vgio file is refused (it
names its own).
"""
function load(::Type{TargetPoints2D}, path::AbstractString; target::Union{Nothing, Symbol} = nothing)
    if _file_form(path) === :calib_data
        target === nothing && throw(
            ArgumentError("$(path): a Schöps dataset.bin does not name the target its feature ids index; state it: `target = :board`")
        )
        return _load_schoeps_dataset(path, target)
    end
    target === nothing || throw(ArgumentError("$(path): `target` is for a Schöps dataset.bin; a vgio file names its targets"))
    return _target_points2d(_read_store(path, "target_points2d", _TARGET_POINTS2D_SPEC))
end

# =============================================================================
# target_points3d/v2 ↔ TargetPoints3D
# =============================================================================

const _TARGET_POINTS3D_SPEC = _CheckedSpec(
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
            :targets => _CheckedSpec(_Vec(_HEX64), h -> _unique_check(h, "targets identities")),
            :time_origin => _NonEmpty(), _PROVENANCE...,
        ],
    ),
    _FRAMES_CHECK,
)

"""
    save(path, obs::TargetPoints3D{F}; length_unit = mm, provenance = (;))

Write a physical VGC `TargetPoints3D` as a `vgio/target_points3d/v2` file in
frame `F`, in `length_unit`: per frame its sensor (required), image and time
where stated; an `Uncertain` point writes its 3×3 `covariance` (`length_unit`²).
Plain points are an `ArgumentError` (attach the unit, `p .* m`).
"""
function save(path::AbstractString, obs::TargetPoints3D{F}; length_unit::Units = mm, provenance::NamedTuple = (;)) where {F}
    frames = map(sort!(collect(keys(obs)))) do t
        points = map(obs[t]) do o
            xyz = _lengths_json(geom(o.xyz), length_unit, "save: a 3D target point")
            r = (target = String(o.target), index = o.index, xyz = xyz)
            o.xyz isa Uncertain ? merge(r, (covariance = _point3_cov_json(cov(o.xyz), length_unit),)) : r
        end
        merge(_stamps_json(obs, t), (points = points,))
    end
    body = merge(
        (
            schema = _schema("target_points3d"), length_unit = unit_string(length_unit), frame = frame_name(F),
            point_index_base = _POINT_INDEX_BASE,
        ),
        isempty(obs.targets) ? (;) : (targets = obs.targets,),
        _time_origin_json(obs), (frames = frames,),
    )
    _write_store(path, _TARGET_POINTS3D_SPEC, with_provenance(body, provenance); style = VgioStyle(; length_unit))
    return nothing
end

# A 3D point's covariance on the wire, in the file's unit squared; and a wire
# point read back as lengths in metres (its covariance, where stated — `nothing`
# where not — in m²), in the file's unit.
_point3_cov_json(Σ, u::Units) = _length_cov_json(Σ, u, _point_cov_dims(3))
function _point3_from_json(xyz, covariance, unit::Units)
    p = _lengths_from_json(Val(3), xyz::Vector{Any}, unit)
    return covariance === nothing ? p : Uncertain(p, CovMat(_length_cov_from_json(covariance, 3, unit, _point_cov_dims(3))))
end

# The parsed, checked record as the value (as `_target_points2d`).
function _target_points3d(d)
    unit = parse_unit(d.length_unit::String)
    frames = Dict{Int, Vector{TargetPoint3D}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = TargetPoint3D[
            (target = Symbol(o.target::String), index = Int(o.index::Integer), xyz = _point3_from_json(o.xyz, get(o, :covariance, nothing), unit))
                for o in f.points::Vector{Any}
        ]
    end
    return TargetPoints3D{frame_from_name(d.frame::String)}(
        frames; images, sensors, times, time_origin = _time_origin(d),
        targets = haskey(d, :targets) ? String[h::String for h in d.targets::Vector{Any}] : String[],
    )
end

"""
    load(TargetPoints3D, path) -> TargetPoints3D{F}

Read a `vgio/target_points3d/v2` file: the measurements in the frame `F` it
declares, lengths in metres (`Uncertain` where it states a covariance, in m²).
"""
load(::Type{TargetPoints3D}, path::AbstractString) = _target_points3d(_read_store(path, "target_points3d", _TARGET_POINTS3D_SPEC))

# =============================================================================
# target_detections/v2 ↔ TargetDetections
# =============================================================================

# One target's detections in a frame: its point indices (unique, at least
# one), its points (`d` numbers each), and per point, where the value has them,
# a d×d covariance and a weight.
_detection_record_spec(d::Int) = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:target => _NonEmpty(), :index => _Vec(_PosInt()), :points => _Vec(_Vec(_Real(), d))],
        Pair{Symbol, _Spec}[:covariance => _Vec(_cov_spec(d)), :weights => _Vec(_Real())],
    ),
    function (r)
        n = length(r[:index])
        n == 0 && return "a detected target has at least one point (a target not detected is absent)"
        msg = _unique_check(r[:index], "the point indices of a target")
        msg === nothing || return msg
        for k in (:points, :covariance, :weights)
            (haskey(r, k) && length(r[k]) != n) && return "`$(k)` has $(length(r[k])) entries where `index` has $(n)"
        end
        return nothing
    end,
)
_detection_frame_spec(d::Int, required::Vector{Symbol}) = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:t => _Int(), _stamp_specs(required)[1]..., :detections => _Vec(_detection_record_spec(d))],
        _stamp_specs(required)[2],
    ),
    f -> _unique_check([r[:target] for r in f[:detections]], "the targets of a frame"),
)
# Pixels (`pixel_frame`: every frame's image and sensor, at least one targets
# identity) or 3D points (`frame` and `length_unit`: every frame's sensor).
const _TARGET_DETECTIONS_SPEC = _CheckedSpec(
    _Either(
        Pair{Symbol, _Spec}[
            :pixel_frame => _Obj(
                Pair{Symbol, _Spec}[
                    :schema => _schema_tok("target_detections"), :pixel_frame => _PIXEL_FRAME,
                    :point_index_base => _Tok(_POINT_INDEX_BASE),
                    :targets => _CheckedSpec(_Vec(_HEX64), h -> isempty(h) ? "the detections name at least one targets file" : _unique_check(h, "targets identities")),
                    :frames => _Vec(_detection_frame_spec(2, [:image, :sensor])),
                ],
                Pair{Symbol, _Spec}[:time_origin => _NonEmpty(), _PROVENANCE...],
            ),
            :frame => _Obj(
                Pair{Symbol, _Spec}[
                    :schema => _schema_tok("target_detections"), :length_unit => _LENGTH_UNIT,
                    :frame => _FRAME_NAME, :point_index_base => _Tok(_POINT_INDEX_BASE),
                    :frames => _Vec(_detection_frame_spec(3, [:sensor])),
                ],
                Pair{Symbol, _Spec}[
                    :targets => _CheckedSpec(_Vec(_HEX64), h -> _unique_check(h, "targets identities")),
                    :time_origin => _NonEmpty(), _PROVENANCE...,
                ],
            ),
        ]
    ),
    _FRAMES_CHECK,
)

# The declaration of the frame the detections are in: a pixel frame, or a 3D
# frame and the file's length unit.
_detections_frame_json(::Type{F}, ::Units) where {F <: PixelFrame} = (pixel_frame = pixel_frame_name(F),)
_detections_frame_json(::Type{F}, u::Units) where {F} = (length_unit = unit_string(u), frame = frame_name(F))

# A point and its covariance on the wire: a pixel as it is (px², never
# converted), a 3D point's lengths in `u` (plain refused) and its covariance in u².
_detection_point_json(::Type{<:PixelFrame}, p, ::Units) = _pixel_json(p)
_detection_point_json(::Type, p, u::Units) = _lengths_json(p, u, "save: a 3D target detection")
_detection_cov_json(::Type{<:PixelFrame}, Σ, ::Units) = _cov_json(Σ)
_detection_cov_json(::Type, Σ, u::Units) = _point3_cov_json(Σ, u)

function _detection_json(::Type{F}, id::Symbol, cs::Csponds, u::Units) where {F}
    pts = targets(cs)
    r = (target = String(id), index = collect(Int, sources(cs)), points = [_detection_point_json(F, geom(p), u) for p in pts])
    eltype(pts) <: Uncertain && (r = merge(r, (covariance = [_detection_cov_json(F, cov(p), u) for p in pts],)))
    cols = cs.attrs
    keys(cols) ⊆ (:weights,) || throw(
        ArgumentError("save: a target_detections file holds one attribute column, `weights`; target $(id) carries `$(first(k for k in keys(cols) if k !== :weights))`")
    )
    return haskey(cols, :weights) ? merge(r, (weights = Float64[w for w in cols.weights],)) : r
end

"""
    save(path, td::TargetDetections{F}; length_unit = mm, provenance = (;))

Write a VGC `TargetDetections` as a `vgio/target_detections/v2` file: per
frame index (sorted), its stamps, and per target (ids sorted) its point
indices, its points, a `covariance` per point iff the points are `Uncertain`,
and its `weights` column where it has one (the only column the kind holds).
Pixels declare the value's pixel frame `F` and are written as they are; every
frame must name its image and sensor, and the value its targets identities:
the [`content_id`](@ref)s of the definitions its ids resolve in, each a rig of
targets (`vgio/rig/v9`) or a single unposed target (`vgio/target/v5`, its
detections named by its design `reference`, or by its content id when it has
none). 3D points declare `frame = frame_name(F)` and are written in `length_unit`
(covariances in its square); every frame must name its sensor, and plain
points are an `ArgumentError` (attach the unit, `p .* m`).
"""
function save(path::AbstractString, td::TargetDetections{F}; length_unit::Units = mm, provenance::NamedTuple = (;)) where {F}
    frames = map(sort!(collect(keys(td)))) do t
        detections = [_detection_json(F, id, td[t][id], length_unit) for id in sort!(collect(keys(td[t])))]
        merge(_stamps_json(td, t), (detections = detections,))
    end
    body = merge(
        (schema = _schema("target_detections"),), _detections_frame_json(F, length_unit),
        (point_index_base = _POINT_INDEX_BASE,),
        (F <: PixelFrame || !isempty(td.targets)) ? (targets = td.targets,) : (;),
        _time_origin_json(td), (frames = frames,),
    )
    _write_store(path, _TARGET_DETECTIONS_SPEC, with_provenance(body, provenance); style = VgioStyle(; length_unit))
    return nothing
end

# One wire record as a detection set: indices => points (pixels as stored, 3D
# lengths in metres read in `unit`), `Uncertain` where it states covariances.
_detection_from_json(p, covariance, ::Nothing) = _pixel_from_json(p, covariance)
_detection_from_json(p, covariance, unit::Units) = _point3_from_json(p, covariance, unit)
function _detections_from_json(r, unit::Union{Nothing, Units})
    idx = Int[Int(i::Integer) for i in r.index::Vector{Any}]
    covs = haskey(r, :covariance) ? r.covariance : nothing
    pts = [_detection_from_json(p, covs === nothing ? nothing : covs[i], unit) for (i, p) in enumerate(r.points::Vector{Any})]
    cs = Csponds(idx, pts)
    return haskey(r, :weights) ? withattrs(cs; weights = Float64[Float64(w::Real) for w in r.weights::Vector{Any}]) : cs
end

# An old measurement file's value as a TargetDetections; VGC's refusal (a
# duplicate detection, a form mix) names the file.
function _converted_on_load(x, path)
    try
        return TargetDetections(x)
    catch e
        e isa ArgumentError && throw(ArgumentError("$(path): $(e.msg)"))
        rethrow()
    end
end

"""
    load(TargetDetections, path) -> TargetDetections{F}

Read a `vgio/target_detections/v2` file: per frame index, per target, its
detections typed in the frame `F` the file declares — pixels exactly as
stored, 3D points in metres — `Uncertain` where it states covariances, with
their `weights` column where stated, the frames' stamps and the targets
identities. A `vgio/target_points2d` or `vgio/target_points3d` file (any version that reads) is
read too, converted (`TargetDetections(load(TargetPoints2D, path))`), unless
it holds one of the two things those kinds allow and a `TargetDetections`
does not: a (target, index) detected twice in one frame, or plain and
`Uncertain` points in one file. Such a file is refused with an
`ArgumentError` naming the path, frame, target and index, and still loads as
what it is, `load(TargetPoints2D, path)` or `load(TargetPoints3D, path)`.
"""
function load(::Type{TargetDetections}, path::AbstractString)
    d = _read_vgio_json(path)   # parsed once; its `schema` says which kind it is read as
    k = d isa AbstractDict ? _split_schema(get(d, :schema, nothing)) : nothing
    k !== nothing && k.kind == "target_points2d" &&
        return _converted_on_load(_target_points2d(_checked_store(d, path, "target_points2d", _TARGET_POINTS2D_SPEC)), path)
    k !== nothing && k.kind == "target_points3d" &&
        return _converted_on_load(_target_points3d(_checked_store(d, path, "target_points3d", _TARGET_POINTS3D_SPEC)), path)
    d = _checked_store(d, path, "target_detections", _TARGET_DETECTIONS_SPEC)
    unit = haskey(d, :pixel_frame) ? nothing : parse_unit(d.length_unit::String)
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = Int(f.t::Integer)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(r.target::String) => _detections_from_json(r, unit) for r in f.detections::Vector{Any})
    end
    F = unit === nothing ? pixel_frame_from_name(d.pixel_frame::String) : frame_from_name(d.frame::String)
    return TargetDetections{F}(
        frames; images, sensors, times, time_origin = _time_origin(d),
        targets = haskey(d, :targets) ? String[h::String for h in d.targets::Vector{Any}] : String[],
    )
end
