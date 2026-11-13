# =============================================================================
# The per-frame kinds of a capture — VisualGeometryToolkitCore's captures, serialized
# =============================================================================
#
#   vgio/trajectory/v5.0          ↔ VGC `Trajectory{S, D}`          a body's pose over time
#   vgio/target_poses/v3.0        ↔ VGC `TargetPoses{F}`            per frame, the poses of the targets seen
#   vgio/target_homographies/v2.0 ↔ VGC `TargetHomographies{F, H}`  per frame, each target's plane → pixel homography
#   vgio/observations/v3.0        ↔ VGC `TargetObservations{F}`     per frame, per target, index => pixel or 3D point
#                                   (IO 0.9's target_points2d, target_points3d and target_detections read
#                                   through their version steps into it)
#
# Every one opens with the envelope (validate.jl) and is a list of `frames`,
# each named by its `frame_index` — unique, the join key between the kinds of
# one capture (and capture_settings) — with, where the kind has them, `image`
# (the image taken at that instant), `sensor` (what measured it: a camera's
# name, a tracker's) and `time` (in `units.time` since the file's
# `time_origin`, an ISO 8601 date-time WITH its UTC offset, stated iff a frame
# has a time). An instant not observed is ABSENT, never a NaN value. A file
# reads back as the VGC value it was written from, exactly (lengths in metres
# in memory, in the declared `units.length` on disk; `Uncertain` members with
# their covariance). A writer refuses plain numbers where a length is meant.
#
# The hierarchy: a RIG of targets (a vgio/rig file) defines targets and their
# points; an observation names a point by (target, index) — its 1-based
# position in that target's points — and the observations file names the
# targets files it resolves in by content identity (`targets`, see
# `content_id`); a trajectory is the pose of a body (`body:<id>`, a view, a
# target) in one frame.
#
# {"schema": "vgio/trajectory/v5.0", <envelope>, "from": "body:board_rig", "to": "world",
#  "time_origin": "2026-05-27T00:00:00+02:00",
#  "frames": [{"frame_index": 17, "image": "DSC04520.JPG", "time": 47058.40,
#              "pose": {rotation, translation[, covariance]}, "recovered": true}]}
#      every pose maps `from` → `to`; `recovered` (written only when true) marks
#      a pose substituted under a stated method rather than measured
#
# {"schema": "vgio/target_poses/v3.0", <envelope>, "to": "camera",
#  "frames": [{"frame_index": 1, "image": "DSC04520.JPG", "sensor": "<camera name>", "time": 12.5,
#              "poses": [{"target": "89b", "rotation": […], "translation": […], "covariance": …}]}]}
#      each pose maps target:<target> → `to`: a detector's object → camera pose
#
# {"schema": "vgio/target_homographies/v2.0", <envelope>,
#  "targets": ["<content id of a targets rig or of a single target>", …],
#  "frames": [{"frame_index": 1, "image": "DSC04520.JPG", "sensor": "<camera name>", "time": 12.5,
#              "homographies": [{"target": "89b", "type": "homography_map",
#                                "parameters": [h11, h12, h13, h21, h22, h23, h31, h32, h33],
#                                "covariance": …8×8 (optional)}]}]}
#      each `target` names a board inside one of the `targets` — a `content_id` of
#      EITHER a rig of boards OR a single unposed target; at least one. Each
#      homography is VGC's `HomographyMap` as the parametric record (style.jl): its
#      nine canonical entries (unit Frobenius norm, h33 ≥ 0), an `Uncertain` one's
#      covariance in the HomographyManifold8 chart. It maps the target's plane — its
#      own design frame, millimetres — to a pixel of the declared `units.pixel`, as
#      `RadialHomography`'s H does: BEFORE the lens of the camera `sensor` names,
#      about that camera's centre (VGC's `radial_homography(camera, H)`). A
#      homography has no 3D frame, so the pixel frame is the TYPE parameter of the
#      value (`TargetHomographies{F}`), and `targets` its field.
#
# {"schema": "vgio/observations/v3.0", <envelope>, "entity": "target_point",
#  "targets": [ … ],                       (pixels: at least one; 3D points: optional)
#  "frame": "world",                       (3D points only; pixels are in units.pixel)
#  "frames": [{"frame_index": 17, "image": …, "sensor": "cam1", "time": …,
#              "observations": [{"target": "<uid>", "index": [1, 5],
#                                "points": [{"xy": [u, v], "covariance": [[…], […]]}, {"xy": […], …}],
#                                "weights": [1.0, 0.5]}]}]}
#      THE observation kind (VGC's `TargetObservations`): per frame, per target, the
#      observed points' indices (1-based, unique), each point in the one point
#      vocabulary (`xy` a pixel, `xyz` a 3D point in `units.length`, its
#      `covariance` iff Uncertain: px², `units.length`²) and its `weights` column
#      where the set has one. `entity` says what an observation names: a
#      `target_point` (the target's id and the point's index; the only entity a
#      VGC container holds today) or a `landmark` (a scene point by its index; its
#      sets carry no `target`), declared by the schema for the containers to come.
#      Pixels need every frame's image and sensor and at least one `targets`
#      identity; 3D points every frame's sensor (their `targets` absent when the
#      constellation is not known independently of these observations).
#
# COLMAP, for orientation (no exporter): an observation is COLMAP's images.txt
# `points2D` with their `POINT3D_ID`, the (target, index) pair playing the
# `POINT3D_ID`; a rig of targets is COLMAP's `points3D`, kept as per-target
# frames with no stored tracks — the observations are the tracks.
#
# Uncertainty (VisualGeometryToolkitCore conventions/persisted-geometry.md): a pose's
# 6×6 `covariance` over [δω; δt] (rad, `units.length`), a pixel's 2×2 (px²), a
# 3D point's 3×3 (`units.length`²) — present iff the value is `Uncertain`.
#
# Related formats (guidance; exporters.jl emits TUM, EuRoC, BOP and TRC exactly):
# - TUM RGB-D `timestamp tx ty tz qx qy qz qw` (evo/evo/tools/file_interface.py:
#   121-174): a pose per explicit timestamp. Ours keeps the time and declares
#   its origin; quaternions stay w,x,y,z (declared); the frames are named.
# - EuRoC/ASL `#timestamp [ns], p_RS_R_x…, q_RS_w…` (file_interface.py:224-243):
#   the direction is in the NAME (`R` ← `S`); ours states `from`/`to`.
# - KITTI odometry: 3×4 row-major poses, no timestamps (file_interface.py:
#   177-220) — the implicit-index join replaced here by `frame_index` + `image`.
# - BOP `scene_gt.json`: image id → [{cam_R_m2c, cam_t_m2c mm, obj_id}]
#   (bop_toolkit_lib/inout.py:267-300) — the per-image object list
#   target_poses adopts, model → camera; ours names the camera (`sensor`),
#   states the unit and keys targets by identity, not an integer.
# - OpenSim TRC (TRCFileAdapter.cpp:236-376) and C3D (ezc3d README:526-612):
#   per-marker X/Y/Z per frame, a missing marker empty / a negative residual.
#   3D observations name markers by (body, index), drop an unseen one, and
#   replace the scalar residual by a covariance when known.

# --- the per-frame facts: frame_index, image, sensor, time ------------------------

# An ISO 8601 date-time with its UTC offset (`Z` or `±hh:mm`): instants written
# on different machines compare.
const _TIME_ORIGIN = _Named(
    Base.Fix1(occursin, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$"),
    "an ISO 8601 date-time with its UTC offset (2026-05-27T09:30:00+02:00, or …Z)",
)
const _FRAMES_CHECK = x -> something(
    _unique_check([f[:frame_index] for f in x[:frames]], "frame indices `frame_index`"),
    (any(f -> haskey(f::AbstractDict, :time), x[:frames]::AbstractVector) && !haskey(x, :time_origin)) ?
        "a frame has a `time` (in units.time since `time_origin`) but the file states no `time_origin`" : nothing,
    Some(nothing),
)
# `_FRAMES_CHECK` in words and in JSON Schema (json_schema.jl publishes them).
const _FRAMES_TIME_SCHEMA = Dict{String, Any}(
    "if" => Dict("required" => ["frames"], "properties" => Dict("frames" => Dict("contains" => Dict("required" => ["time"])))),
    "then" => Dict("required" => ["time_origin"]),
)
const _FRAMES_RULE = (rule = "frame indices `frame_index` are unique; a file whose frames state a `time` states its `time_origin`",
                      schema = [_FRAMES_TIME_SCHEMA])
# The optional per-file stamp keys of a capture kind.
const _CAPTURE_OPTIONAL = Pair{Symbol, _Spec}[:time_origin => _TIME_ORIGIN]

# A frame's stamps, as written: `image` and `sensor` where the value has them
# (required ones are checked by the spec), `time` where it has one.
function _stamps_json(x, t)
    f = (frame_index = t,)
    haskey(x.images, t) && (f = merge(f, (image = x.images[t],)))
    hasproperty(x, :sensors) && haskey(x.sensors, t) && (f = merge(f, (sensor = x.sensors[t],)))
    haskey(x.times, t) && (f = merge(f, (time = x.times[t],)))
    return f
end
function _time_origin_json(x)
    x.time_origin === nothing && return (;)
    occursin(_TIME_ORIGIN_RE, x.time_origin) || throw(ArgumentError(
        "save: the time_origin $(repr(x.time_origin)) states no UTC offset; an instant is compared across machines — " *
            "give it one (2026-05-27T09:30:00+02:00, or …Z)"))
    return (time_origin = x.time_origin,)
end
const _TIME_ORIGIN_RE = r"(Z|[+-]\d{2}:\d{2})$"

# …and read back into the dicts a VGC capture is built from.
function _read_stamps!(images, sensors, times, f, t)
    haskey(f, :image) && (images[t] = f.image::String)
    (sensors === nothing || !haskey(f, :sensor)) || (sensors[t] = f.sensor::String)
    haskey(f, :time) && (times[t] = Float64(f.time::Real))
    return nothing
end
_time_origin(d) = haskey(d, :time_origin) ? String(d.time_origin::String) : nothing
_frame_index(f) = Int(f.frame_index::Integer)

_stamp_specs(required::Vector{Symbol}) = (
    Pair{Symbol, _Spec}[k => _NonEmpty() for k in required],
    Pair{Symbol, _Spec}[k => (k === :time ? _Real() : _NonEmpty()) for k in (:image, :sensor, :time) if !(k in required)],
)

# =============================================================================
# trajectory/v5.0 ↔ Trajectory
# =============================================================================

const _TRAJECTORY_FRAME = _Obj(
    Pair{Symbol, _Spec}[:frame_index => _Int(), :pose => _UPOSE],
    Pair{Symbol, _Spec}[:image => _NonEmpty(), :time => _Real(), :recovered => _Tok(true)],
)
const _TRAJECTORY_SPEC = _kind_spec(
    "trajectory",
    Pair{Symbol, _Spec}[:from => _FRAME_NAME, :to => _FRAME_NAME, :frames => _Vec(_TRAJECTORY_FRAME)],
    _CAPTURE_OPTIONAL;
    check = _FRAMES_CHECK, _FRAMES_RULE...,
)

"""
    save(path, tr::Trajectory{S, D}; length_unit = mm, provenance = (;))

Write a physical VGC `Trajectory` as a `vgio/trajectory/v5.0` file: its frames
`from = frame_name(S)`, `to = frame_name(D)`, one record per frame index
(sorted) — the pose (a `covariance` iff `Uncertain`), and its `image`, `time`
and `recovered` flag where the trajectory has them — in `length_unit`. A
trajectory of plain poses is an `ArgumentError` (attach the unit, `t .* m`),
and so is a `time_origin` without its UTC offset.
"""
function save(path::AbstractString, tr::Trajectory; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    _write_kind(path, "trajectory", JSON.lower(s, tr), provenance; style = s)
    return nothing
end

# A Trajectory on IO's style: the body of a `vgio/trajectory` file without its
# envelope (the enclosing file's): `from`, `to`, the `time_origin` where the
# trajectory has one, and one record per frame index, sorted — its stamps and
# its pose, lowered on the style.
function JSON.lower(::VgioStyle, tr::Trajectory{S, D}) where {S, D}
    frames = map(sort!(collect(keys(tr)))) do t
        f = merge(_stamps_json(tr, t), (pose = tr[t],))   # the pose itself, lowered on the file's style
        t in tr.recovered ? merge(f, (recovered = true,)) : f
    end
    return merge((from = frame_name(S), to = frame_name(D)), _time_origin_json(tr), (frames = frames,))
end

"""
    load(Trajectory, path) -> Trajectory{S, D}

Read a `vgio/trajectory/v5.0` file: its frames as types (`S`, `D` from
`from`/`to`), each pose labeled `S → D`, lengths in metres — `Uncertain` where
the file states a covariance — with the images, times and recovered flags it
records. Type-unstable by nature (the frame types come from the file): load
once, dispatch after. An earlier version reads through its version steps; of
v2 (free frame names) only CaliBlobs' 2026-09 track segment (`track:<s>` →
`camera`), read as `target:scene_<c>` → `camera` (src/migrate.jl).
"""
function load(::Type{Trajectory}, path::AbstractString)
    d = _read_store(path, "trajectory")
    return JSON.lift(VgioStyle(d), Trajectory, d)
end
# …and back: a trajectory body read on the style (its lengths in the style's unit).
function JSON.lift(s::VgioStyle, ::Type{Trajectory}, d::Record)
    poses, images, times, recovered = Dict{Int, Any}(), Dict{Int, String}(), Dict{Int, Float64}(), Set{Int}()
    for f in d.frames::Vector{Any}
        t = _frame_index(f)
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
# target_poses/v3.0 ↔ TargetPoses
# =============================================================================

const _TARGET_POSE_RECORD = _Obj(
    Pair{Symbol, _Spec}[:target => _NonEmpty(), :rotation => _QUATERNION, :translation => _Vec(_Real(), 3)],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _TARGET_POSES_SPEC = _kind_spec(
    "target_poses",
    Pair{Symbol, _Spec}[
        :to => _FRAME_NAME,
        :frames => _Vec(
            _CheckedSpec(
                _Obj(
                    Pair{Symbol, _Spec}[:frame_index => _Int(), _stamp_specs([:image, :sensor])[1]..., :poses => _Vec(_TARGET_POSE_RECORD)],
                    _stamp_specs([:image, :sensor])[2],
                ),
                f -> _unique_check([p[:target] for p in f[:poses]], "the targets of a frame"); rule = "a frame poses each target once",
            )
        ),
    ],
    _CAPTURE_OPTIONAL;
    check = _FRAMES_CHECK, _FRAMES_RULE...,
)

"""
    save(path, tp::TargetPoses{F}; length_unit = mm, provenance = (;))

Write a physical VGC `TargetPoses` as a `vgio/target_poses/v3.0` file declaring
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
    _write_kind(path, "target_poses", merge((to = frame_name(F),), _time_origin_json(tp), (frames = frames,)), provenance; style = s)
    return nothing
end

"""
    load(TargetPoses, path) -> TargetPoses{F}

Read a `vgio/target_poses/v3.0` file: `F` the frame its poses map into, every
pose's lengths in metres, `Uncertain` where the file states a covariance.
"""
function load(::Type{TargetPoses}, path::AbstractString)
    d = _read_store(path, "target_poses")
    s = VgioStyle(d)
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = _frame_index(f)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(p.target::String) => JSON.lift(s, EuclideanMap{3}, p::Record) for p in f.poses::Vector{Any})
    end
    return TargetPoses{frame_from_name(d.to::String)}(frames; images, sensors, times, time_origin = _time_origin(d))
end

# =============================================================================
# target_homographies/v2.0 ↔ TargetHomographies
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
        :parameters => _CheckedSpec(_Vec(_Real(), 9), _homography_parameters_check;
                                    rule = "the canonical homography parameters (VisualGeometryToolkitCore's `parameter_vector` of a " *
                                           "HomographyMap): the 3×3 matrix's entries of unit Frobenius norm with h33 ≥ 0"),
    ],
    Pair{Symbol, _Spec}[:covariance => _cov_spec(8)],
)
const _TARGETS_IDS = _CheckedSpec(_Vec(_HEX64), h -> isempty(h) ? "at least one targets identity (a rig's or a single target's `content_id`)" : _unique_check(h, "targets identities");
                                  rule = "at least one targets identity, each once", schema = ["minItems" => 1, "uniqueItems" => true])

const _TARGET_HOMOGRAPHIES_SPEC = _kind_spec(
    "target_homographies",
    Pair{Symbol, _Spec}[
        :targets => _TARGETS_IDS,
        :frames => _Vec(
            _CheckedSpec(
                _Obj(
                    Pair{Symbol, _Spec}[:frame_index => _Int(), _stamp_specs([:image, :sensor])[1]..., :homographies => _Vec(_HOMOGRAPHY_RECORD)],
                    _stamp_specs([:image, :sensor])[2],
                ),
                f -> _unique_check([h[:target] for h in f[:homographies]], "the targets of a frame"); rule = "a frame holds each target's homography once",
            )
        ),
    ],
    _CAPTURE_OPTIONAL;
    check = _FRAMES_CHECK, _FRAMES_RULE...,
)

"""
    save(path, th::TargetHomographies{F}; provenance = (;))

Write a VGC `TargetHomographies` as a `vgio/target_homographies/v2.0` file whose
declared pixel frame (`units.pixel`) is `F`, the value's own type — no pixel is
converted and none can be written under another frame's label — and whose
`targets` are the value's `th.targets`: the [`content_id`](@ref)s of the
definitions its target ids resolve in, each the identity of a rig of boards or
of a single unposed target. Per frame index (sorted) it writes the image and
sensor (both required), the time, and every target's homography (ids sorted) as
the homography record:

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
    _write_kind(path, "target_homographies", merge((targets = copy(th.targets),), _time_origin_json(th), (frames = frames,)), provenance; style = s)
    return nothing
end

"""
    load(TargetHomographies, path) -> TargetHomographies{F}
    load(TargetHomographies{F}, path) -> TargetHomographies{F}

Read a `vgio/target_homographies/v2.0` file: the value it was written from,
typed in the pixel frame `F` the file declares (`units.pixel`), per frame index
every target's `HomographyMap` (stored unlabeled; `homography(th, t, id)`
labels it `TargetPlaneFrame → F`) with its canonical parameters exactly as
stored, `Uncertain` where the record states a covariance, with the frames'
images, sensors and times and the `targets` identities. `H` maps the target's
plane (millimetres) to a pixel of `F`, before the lens of the camera the
frame's `sensor` names (pair them with VGC's `radial_homography(camera, H)`; see
[`save`](@ref)).

The second form asserts the frame: a file declaring another is an
`ArgumentError`. So are parameters that are not canonical or are singular.
"""
function load(::Type{TargetHomographies}, path::AbstractString)
    d = _read_store(path, "target_homographies")
    s = VgioStyle(d)
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = _frame_index(f)
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
        ArgumentError("$(path): declares the pixel frame $(repr(pixel_frame_name(_pixel_frame_type(th)))), not the requested $(repr(pixel_frame_name(F)))")
    )
    return th
end

# =============================================================================
# observations/v3.0 ↔ TargetObservations
# =============================================================================

# What an observation names: a target's point (target id + point index) or a
# scene landmark (its index). VGC's `TargetObservations` holds target points.
const _ENTITIES = ("target_point", "landmark")
const _TARGET_POINT_ENTITY = "target_point"

# A pixel or a 3D point with its covariance iff Uncertain (`_TARGET_POINT`,
# rig_schema.jl: the one point vocabulary), `d` its dimension.
_observed_point_spec(d::Int) = d == 2 ?
    _Obj(Pair{Symbol, _Spec}[:xy => _XY], Pair{Symbol, _Spec}[:covariance => _COV2]) :
    _Obj(Pair{Symbol, _Spec}[:xyz => _XYZ], Pair{Symbol, _Spec}[:covariance => _COV3])
# One entity's observations in a frame: its point indices (unique, at least
# one), its points (`d` numbers each, a covariance per point iff Uncertain) and,
# where the set has one, its `weights` column; a target point's set names its
# `target`.
_observation_set_spec(d::Int) = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:index => _Vec(_PosInt()), :points => _Vec(_observed_point_spec(d))],
        Pair{Symbol, _Spec}[:target => _NonEmpty(), :weights => _Vec(_Real())],
    ),
    function (r)
        n = length(r[:index])
        n == 0 && return "an observed entity has at least one point (an entity not observed is absent)"
        msg = _unique_check(r[:index], "the point indices of a set")
        msg === nothing || return msg
        for k in (:points, :weights)
            (haskey(r, k) && length(r[k]) != n) && return "`$(k)` has $(length(r[k])) entries where `index` has $(n)"
        end
        c = count(p -> haskey(p, :covariance), r[:points])
        (c == 0 || c == n) || return "every point of a set states its `covariance`, or none does"
        return nothing
    end;
    rule = "at least one point; the point indices are unique; `points` and `weights` have one entry per index; every " *
           "point states its `covariance`, or none does",
    schema = ["properties" => Dict("index" => Dict("minItems" => 1, "uniqueItems" => true))],
)
_observations_frame_spec(d::Int, required::Vector{Symbol}) = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:frame_index => _Int(), _stamp_specs(required)[1]..., :observations => _Vec(_observation_set_spec(d))],
        _stamp_specs(required)[2],
    ),
    f -> _unique_check([r[:target] for r in f[:observations] if haskey(r, :target)], "the targets of a frame");
    rule = "a frame holds each target's observations in one set",
)
# The entity a set names agrees with the file's `entity`: a target point's set
# names its `target`, a landmark's does not.
_entity_check(x) = begin
    tp = x[:entity] == _TARGET_POINT_ENTITY
    for f in x[:frames], r in f[:observations]
        haskey(r, :target) == tp || return tp ? "an observation of a target point names its `target`" :
            "a landmark observation names no `target` (its `index` is the landmark's)"
    end
    nothing
end
# Pixels (no `frame`: every frame's image and sensor, at least one targets
# identity) or 3D points (`frame`: every frame's sensor): one of two shapes,
# told apart by whether the record states its 3D `frame` (`_IfKey`).
const _OBSERVATIONS_ENTITY = :entity => _Named(in(_ENTITIES), "an observed entity ($(join(_ENTITIES, ", ")))")
_observations_check(x) = something(_entity_check(x), _FRAMES_CHECK(x), Some(nothing))
# `_entity_check` in JSON Schema: a target point's sets name their `target`, a landmark's do not.
_sets_schema(s) = Dict("properties" => Dict("frames" => Dict("items" => Dict("properties" => Dict("observations" => Dict("items" => s))))))
const _OBSERVATIONS_RULE = (
    rule = "a target point's observation sets name their `target`, a landmark's do not; " * _FRAMES_RULE.rule,
    schema = [Dict{String, Any}("if" => Dict("properties" => Dict("entity" => Dict("const" => _TARGET_POINT_ENTITY))),
                                "then" => _sets_schema(Dict("required" => ["target"])),
                                "else" => _sets_schema(Dict("not" => Dict("required" => ["target"])))),
              _FRAMES_RULE.schema...],
)
const _OBSERVATIONS_SPEC = _IfKey(
    :frame,
    _kind_spec("observations",
        Pair{Symbol, _Spec}[_OBSERVATIONS_ENTITY, :frame => _FRAME_NAME, :frames => _Vec(_observations_frame_spec(3, [:sensor]))],
        Pair{Symbol, _Spec}[:targets => _CheckedSpec(_Vec(_HEX64), h -> _unique_check(h, "targets identities");
                                                     rule = "each targets identity once", schema = ["uniqueItems" => true]),
                            _CAPTURE_OPTIONAL...];
        check = _observations_check, _OBSERVATIONS_RULE...),
    _kind_spec("observations",
        Pair{Symbol, _Spec}[_OBSERVATIONS_ENTITY, :targets => _TARGETS_IDS, :frames => _Vec(_observations_frame_spec(2, [:image, :sensor]))],
        _CAPTURE_OPTIONAL;
        check = _observations_check, _OBSERVATIONS_RULE...),
)

# A point and its covariance on the wire: a pixel as it is (px², never
# converted), a 3D point's lengths in `u` (plain refused) and its covariance in u².
_observation_point_json(::Type{<:PixelFrame}, p, ::Units) = (xy = Float64[geom(p)[1], geom(p)[2]],)
_observation_point_json(::Type, p, u::Units) = (xyz = _lengths_json(geom(p), u, "save: a 3D target observation"),)
_observation_cov_json(::Type{<:PixelFrame}, Σ, ::Units) = _cov_json(Σ)
_observation_cov_json(::Type, Σ, u::Units) = _length_cov_json(Σ, u, _point_cov_dims(3))
_observed_point_json(::Type{F}, p, u) where {F} = p isa Uncertain ?
    merge(_observation_point_json(F, p, u), (covariance = _observation_cov_json(F, cov(p), u),)) : _observation_point_json(F, p, u)

function _observation_set_json(::Type{F}, id::Symbol, cs::Correspondences, u::Units) where {F}
    r = (target = String(id), index = collect(Int, sources(cs)), points = [_observed_point_json(F, p, u) for p in targets(cs)])
    cols = cs.attrs
    keys(cols) ⊆ (:weights,) || throw(
        ArgumentError("save: an observations file holds one attribute column, `weights`; target $(id) carries `$(first(k for k in keys(cols) if k !== :weights))`")
    )
    return haskey(cols, :weights) ? merge(r, (weights = Float64[w for w in cols.weights],)) : r
end

"""
    save(path, x::TargetObservations{F}; length_unit = mm, provenance = (;))

Write a VGC `TargetObservations` as a `vgio/observations/v3.0` file of
`target_point` observations: per frame index (sorted), its stamps, and per
target (ids sorted) its point indices, its points (each `{"xy"}` or `{"xyz"}`,
its `covariance` iff `Uncertain`), and its `weights` column where it has one
(the only column the kind holds). Pixels declare the value's pixel frame `F`
(`units.pixel`) and are written as they are; every frame must name its image
and sensor, and the value its targets identities: the [`content_id`](@ref)s of
the definitions its ids resolve in, each a rig of targets or a single unposed
target (its observations named by its design `reference`, or by its content id
when it has none). 3D points declare `frame = frame_name(F)` and are written in
`length_unit` (covariances in its square); every frame must name its sensor,
and plain points are an `ArgumentError` (attach the unit, `p .* m`).
"""
function save(path::AbstractString, x::TargetObservations{F}; length_unit::Units = mm, provenance::NamedTuple = (;)) where {F}
    frames = map(sort!(collect(keys(x)))) do t
        merge(_stamps_json(x, t), (observations = [_observation_set_json(F, id, x[t][id], length_unit) for id in sort!(collect(keys(x[t])))],))
    end
    pixels = F <: PixelFrame
    body = merge(
        (entity = _TARGET_POINT_ENTITY,), pixels ? (;) : (frame = frame_name(F),),
        (pixels || !isempty(x.targets)) ? (targets = x.targets,) : (;),
        _time_origin_json(x), (frames = frames,),
    )
    _write_kind(path, "observations", body, provenance; style = pixels ? VgioStyle(; length_unit, pixel_frame = F) : VgioStyle(; length_unit))
    return nothing
end

# One wire set as a VGC set: indices => points (pixels as stored, 3D lengths in
# metres read in `unit`), `Uncertain` where it states covariances.
function _observed_point(p, unit::Union{Nothing, Units})
    if unit === nothing
        q = SVector{2, Float64}(Float64(p.xy[1]::Real), Float64(p.xy[2]::Real))
        return haskey(p, :covariance) ? Uncertain(q, CovMat(_cov_matrix(p.covariance, 2))) : q
    end
    q = _lengths_from_json(Val(3), p.xyz::Vector{Any}, unit)
    return haskey(p, :covariance) ? Uncertain(q, CovMat(_length_cov_from_json(p.covariance, 3, unit, _point_cov_dims(3)))) : q
end
function _observation_set(r, unit::Union{Nothing, Units})
    cs = correspondences(Int[Int(i::Integer) for i in r.index::Vector{Any}], [_observed_point(p, unit) for p in r.points::Vector{Any}])
    return haskey(r, :weights) ? withattrs(cs; weights = Float64[Float64(w::Real) for w in r.weights::Vector{Any}]) : cs
end

"""
    load(TargetObservations, path; target = nothing) -> TargetObservations{F}

Read a `vgio/observations/v3.0` file of `target_point` observations: per frame
index, per target, its observations typed in the frame `F` the file declares —
pixels exactly as stored, 3D points in metres — `Uncertain` where it states
covariances, with their `weights` column where stated, the frames' stamps and
the targets identities. An IO 0.9 `vgio/target_detections`,
`vgio/target_points2d` or `vgio/target_points3d` file (any version that reads)
is read through its version steps into this kind; one holding what a
`TargetObservations` cannot (a (target, index) observed twice in one frame,
plain and `Uncertain` points in one file) is refused naming the path, frame,
target and index. A file of `landmark` observations is refused: no VGC
container holds them yet.

A Schöps et al. camera_calibration `dataset.bin` (recognized by its
`calib_data` magic, schoeps.jl) is read too: frame `t` is the file's `t`-th
image (its name in `images`), each detection a point of `target` — which the
file does not name, so the caller states it — its index the feature id + 1
(the reference's ids are 0-based; this importer converts), typed in
[`SCHOEPS_PIXEL_FRAME`](@ref). A `target` beside a vgio file is refused (it
names its own).
"""
function load(::Type{TargetObservations}, path::AbstractString; target::Union{Nothing, Symbol} = nothing)
    if _file_form(path) === :calib_data
        target === nothing && throw(
            ArgumentError("$(path): a Schöps dataset.bin does not name the target its feature ids index; state it: `target = :board`")
        )
        return _load_schoeps_dataset(path, target)
    end
    target === nothing || throw(ArgumentError("$(path): `target` is for a Schöps dataset.bin; a vgio file names its targets"))
    d = _read_store(path, "observations")
    d.entity == _TARGET_POINT_ENTITY || throw(ArgumentError(
        "$(path): a file of $(d.entity) observations; VisualGeometryToolkitCore's TargetObservations holds target points only"))
    unit = haskey(d, :frame) ? _file_length_unit(d) : nothing
    frames = Dict{Int, Dict{Symbol, Any}}()
    images, sensors, times = Dict{Int, String}(), Dict{Int, String}(), Dict{Int, Float64}()
    for f in d.frames::Vector{Any}
        t = _frame_index(f)
        _read_stamps!(images, sensors, times, f, t)
        frames[t] = Dict{Symbol, Any}(Symbol(r.target::String) => _observation_set(r, unit) for r in f.observations::Vector{Any})
    end
    F = unit === nothing ? pixel_frame_from_name(d.units.pixel::String) : frame_from_name(d.frame::String)
    try
        return TargetObservations{F}(
            frames; images, sensors, times, time_origin = _time_origin(d),
            targets = haskey(d, :targets) ? String[h::String for h in d.targets::Vector{Any}] : String[],
        )
    catch e
        e isa ArgumentError && throw(ArgumentError("$(path): $(e.msg)"))
        rethrow()
    end
end
