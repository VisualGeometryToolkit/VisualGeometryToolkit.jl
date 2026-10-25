# =============================================================================
# Exporters — our kinds in the community's exact formats (write-only)
# =============================================================================
#
# The one place VGIo emits foreign syntax, units and layouts. Each exporter
# writes the format as its reference tooling reads it, cited at the function;
# nothing here is read back by VGIo (the vgio kind is the record of truth).
#
#   save_tum_trajectory(path, tr)          TUM RGB-D   `t tx ty tz qx qy qz qw`, m, s
#   save_euroc_trajectory(path, tr)        EuRoC/ASL   `#timestamp [ns], p_RS_R_*, q_RS_*`, m, ns
#   save_bop_scene(to_frame, dir, tp, cameras; …)   BOP scene_gt.json + scene_camera.json
#   save_trc(path, obs; …)                 OpenSim TRC marker trajectories
#
# A trajectory exports as the pose of its `from` frame in its `to` frame — the
# body in the world, the direction TUM and EuRoC mean (EuRoC's `p_RS_R`: the
# sensor body S in the reference R). Times are the trajectory's (seconds since
# its `time_origin`), written as the format's timestamp.

# A physical pose's translation in metres and its quaternion; a plain one is refused.
_pose_m(X) = (g = geom(X); (_lengths_json(g.t, m, "the exported pose's translation"), _qvec(g.R)))

function _stamps(tr::Trajectory, ts, stamp::Symbol)
    stamp === :index && return Float64.(ts)
    stamp === :time || throw(ArgumentError("`stamp` is :time (the trajectory's times) or :index (its frame indices)"))
    missing_ = [t for t in ts if !haskey(tr.times, t)]
    isempty(missing_) || throw(
        ArgumentError("frame(s) $(first(missing_, 5)) have no time — export with `stamp = :index` to use the frame indices")
    )
    return [tr.times[t] for t in ts]
end

"""
    save_tum_trajectory(path, tr::Trajectory; stamp = :time) -> path

The TUM RGB-D trajectory format, as evo reads it
(evo/evo/tools/file_interface.py:121-146): one line per pose, space-separated
`timestamp tx ty tz qx qy qz qw`, translation in METRES, the quaternion x,y,z,w,
the pose of `from` in `to`. The timestamp is the frame's time in seconds since
the trajectory's `time_origin` (`stamp = :index` writes the frame index
instead). One `#` comment line names the frames and the origin (evo skips
comments, file_interface.py:77).
"""
function save_tum_trajectory(path::AbstractString, tr::Trajectory{S, D}; stamp::Symbol = :time) where {S, D}
    ts = sort!(collect(keys(tr.poses)))
    st = _stamps(tr, ts, stamp)
    open(path, "w") do io
        println(io, "# vgio trajectory $(frame_name(S)) -> $(frame_name(D)); timestamp ",
            stamp === :time ? "s since $(tr.time_origin)" : "= frame index", "; tx ty tz [m] qx qy qz qw")
        for (t, s) in zip(ts, st)
            p, q = _pose_m(tr.poses[t])
            println(io, join(string.((s, p[1], p[2], p[3], q[2], q[3], q[4], q[1])), ' '))
        end
    end
    return path
end

"""
    save_euroc_trajectory(path, tr::Trajectory; stamp = :time) -> path

The EuRoC/ASL ground-truth CSV (`state_groundtruth_estimate0/data.csv`), as
evo reads it (evo/evo/tools/file_interface.py:224-243: comma-separated, `#`
header, ≥ 8 columns, timestamp in ns, the quaternion w first): the header
`#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [],
q_RS_x [], q_RS_y [], q_RS_z []` and one row per pose — the pose of `from` (S)
in `to` (R), metres. The timestamp is the frame's time since `time_origin` in
integer nanoseconds (`stamp = :index`: the frame index).
"""
function save_euroc_trajectory(path::AbstractString, tr::Trajectory; stamp::Symbol = :time)
    ts = sort!(collect(keys(tr.poses)))
    st = _stamps(tr, ts, stamp)
    open(path, "w") do io
        println(io, "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []")
        for (t, s) in zip(ts, st)
            p, q = _pose_m(tr.poses[t])
            ns = stamp === :time ? round(Int64, s * 1.0e9) : Int64(s)
            println(io, join(vcat(string(ns), string.((p[1], p[2], p[3], q[1], q[2], q[3], q[4]))...), ','))
        end
    end
    return path
end

"""
    save_bop_scene(to_frame, dir, tp::TargetPoses{CameraFrame}, cameras; obj_ids = nothing, undistorted = false)
        -> (; scene_gt, scene_camera, obj_ids, images)

A BOP scene (bop_toolkit/docs/bop_datasets_format.md:79-112, written as
bop_toolkit_lib/inout.py `save_json` lays it out, :101-120):
`scene_gt.json` — image id → `[{cam_R_m2c (3×3 row-major), cam_t_m2c (mm),
obj_id}]`, the target → camera pose — and `scene_camera.json` — image id →
`{cam_K (row-major), depth_scale}`, K from the frame's camera in the
camera store `cameras` (by its `sensor` name), its principal point moved into `to_frame`
(BOP states no pixel origin; OpenCV's, `PixelFrame{:opencv}`, is the
toolkit's renderer's). Image ids are the frame indices `t`; `obj_id` is an
integer, so the targets are numbered — `obj_ids` (target id → Int), or by
sorted id from 1. BOP's K is a pinhole: a camera with a lens is refused unless
`undistorted = true` states that the poses were measured on images undistorted
to the camera's pinhole twin (K kept, lens dropped). Returns the two paths, the
`obj_ids` used and the image id → image name map, which BOP has no place for.
"""
function save_bop_scene(
        to_frame::Type{<:PixelFrame}, dir::AbstractString, tp::TargetPoses{CameraFrame}, cameras::AbstractDict{Symbol};
        obj_ids::Union{Nothing, AbstractDict} = nothing, undistorted::Bool = false
    )
    all_ids = sort!(unique!([id for ps in values(tp.frames) for id in keys(ps)]))
    ids = obj_ids === nothing ? Dict(id => i for (i, id) in enumerate(all_ids)) :
        Dict{Symbol, Int}(Symbol(k) => Int(v) for (k, v) in obj_ids)
    unmapped = [id for id in all_ids if !haskey(ids, id)]
    isempty(unmapped) || throw(ArgumentError("save_bop_scene: no obj_id for target(s) $(unmapped)"))
    shift = EuclideanMap(CANONICAL_PIXEL_FRAME, to_frame).t
    ts = sort!(collect(keys(tp.frames)))
    gt = Dict{String, Any}()
    cam = Dict{String, Any}()
    for t in ts
        name = Symbol(tp.sensors[t])
        haskey(cameras, name) || throw(ArgumentError("save_bop_scene: frame $(t)'s sensor $(repr(tp.sensors[t])) is not among the cameras"))
        cm = model(_bounded(cameras[name]))
        (cm.lens isa Pinhole || undistorted) || throw(
            ArgumentError(
                "save_bop_scene: camera $(name) has a lens; BOP's K is a pinhole — state `undistorted = true` " *
                    "if the poses were measured on images undistorted to its pinhole twin"
            )
        )
        K = SMatrix{3, 3, Float64}(cm.intrinsics.K)
        Kb = [K[1, 1], K[1, 2], K[1, 3] + shift[1], K[2, 1], K[2, 2], K[2, 3] + shift[2], K[3, 1], K[3, 2], K[3, 3]]
        cam[string(t)] = Dict("cam_K" => Kb, "depth_scale" => 1.0)
        gt[string(t)] = map(sort!(collect(keys(tp.frames[t])))) do id
            X = geom(tp.frames[t][id])
            R = SMatrix{3, 3, Float64}(X.R)
            t_mm = _lengths_json(X.t, mm, "save_bop_scene: target $(id)'s pose translation")
            all(isfinite, t_mm) || throw(ArgumentError("save_bop_scene: frame $(t), target $(id): `cam_t_m2c` is not finite"))
            Dict("cam_R_m2c" => vec(collect(R')), "cam_t_m2c" => t_mm, "obj_id" => ids[id])
        end
    end
    mkpath(dir)
    gt_path, cam_path = joinpath(dir, "scene_gt.json"), joinpath(dir, "scene_camera.json")
    _save_bop_json(gt_path, gt)
    _save_bop_json(cam_path, cam)
    return (; scene_gt = gt_path, scene_camera = cam_path, obj_ids = ids, images = Dict(t => tp.images[t] for t in ts))
end

# bop_toolkit's `save_json` (inout.py:101-120): `{`, one `"key": <compact JSON
# with sorted keys>` line per key in sorted-key order, `}` — image ids sorted as
# strings, as its `sorted(content.items(), key=lambda x: x[0])` sorts them.
function _save_bop_json(path, content::AbstractDict)
    ks = sort!(collect(keys(content)))
    open(path, "w") do io
        println(io, "{")
        for (i, k) in enumerate(ks)
            print(io, "  ", _json(k), ": ", _sorted_json(content[k]))
            println(io, i < length(ks) ? "," : "")
        end
        println(io, "}")
    end
    return path
end
_sorted_json(x::AbstractDict) = "{" * join(("$(_json(String(k))): $(_sorted_json(x[k]))" for k in sort!(collect(keys(x)); by = string)), ", ") * "}"
_sorted_json(x::AbstractVector) = "[" * join((_sorted_json(v) for v in x), ", ") * "]"
_sorted_json(x) = _json(x)

"""
    save_trc(path, obs::TargetPoints3D; data_rate = nothing, stamp = :time) -> (; path, markers)

OpenSim's TRC marker-trajectory text, exactly as
opensim-core/OpenSim/Common/TRCFileAdapter.cpp writes and reads it
(:236-376; reader :38-233): `PathFileType 4 (X/Y/Z) <file>`, the eight
metadata keys and their values (DataRate, CameraRate, NumFrames, NumMarkers,
Units, OrigDataRate, OrigDataStartFrame, OrigNumFrames), `Frame# Time
<marker>…`, `X1 Y1 Z1 …`, a blank line, then one row per frame — tab-separated,
each field followed by a tab. A marker is named `<target>_<index>`; a marker
not observed in a frame is written as three EMPTY fields, which the reader takes
as missing (NaN, :188-199). Units are mm. `Time` is the frame's time since
`time_origin` (`stamp = :index`: the frame index); `data_rate` (Hz) defaults
to the mean rate over that span — TRC requires one even for irregular capture.
Returns the path and the marker column order.
"""
function save_trc(path::AbstractString, obs::TargetPoints3D; data_rate::Union{Nothing, Real} = nothing, stamp::Symbol = :time)
    ts = sort!(collect(keys(obs.frames)))
    times = if stamp === :index
        Float64.(ts)
    else
        stamp === :time || throw(ArgumentError("`stamp` is :time or :index"))
        all(t -> haskey(obs.times, t), ts) || throw(ArgumentError("save_trc: a frame has no time — export with `stamp = :index`"))
        [obs.times[t] for t in ts]
    end
    markers = sort!(unique!([(o.target, o.index) for t in ts for o in obs.frames[t]]))
    col = Dict(mk => i for (i, mk) in enumerate(markers))
    rate = data_rate !== nothing ? Float64(data_rate) :
        length(times) > 1 && times[end] > times[1] ? (length(times) - 1) / (times[end] - times[1]) : 1.0
    open(path, "w") do io
        _trc_line(io, ("PathFileType", 4, "(X/Y/Z)", basename(path)))
        _trc_line(io, ("DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units", "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames"))
        _trc_line(io, (rate, rate, length(ts), length(markers), "mm", rate, 1, length(ts)))
        _trc_line(io, String["Frame#"; "Time"; [x for (tg, ix) in markers for x in ("$(tg)_$(ix)", "", "")]])
        _trc_line(io, String[""; ""; [x for i in eachindex(markers) for x in ("X$(i)", "Y$(i)", "Z$(i)")]])
        println(io)
        for (k, (t, s)) in enumerate(zip(ts, times))
            row = fill("", 3 * length(markers))
            for o in obs.frames[t]
                p = _lengths_json(geom(o.xyz), mm, "save_trc: a 3D target point")   # TRC's Units: mm
                j = 3 * (col[(o.target, o.index)] - 1)
                row[j + 1], row[j + 2], row[j + 3] = string(p[1]), string(p[2]), string(p[3])
            end
            print(io, k, '\t', s, '\t')
            _trc_line(io, row)
        end
    end
    return (; path, markers = ["$(tg)_$(ix)" for (tg, ix) in markers])
end

# One TRC line: every field followed by a tab, then the newline. The fields are
# printed one at a time rather than splatted into a `join`, which Julia cannot
# infer for a variable-length argument list.
function _trc_line(io::IO, fields)
    for x in fields
        print(io, x, '\t')
    end
    return println(io)
end
