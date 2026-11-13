# =============================================================================
# vgio/transforms/v5.0 — named rigid transforms between typed frames
# =============================================================================
#
# {"schema": "vgio/transforms/v5.0", <envelope: generator, units, conventions>,
#  "transforms": [{"from": "body:camera_rig", "to": "camera",
#                  "rotation": [w, x, y, z], "translation": [x, y, z],
#                  "covariance": [[…6], …6 rows]}],                 (covariance optional)
#  "provenance": {…}}                                               (optional)
#
# Every record names its direction (`from`, `to`) — half of every hand-eye
# confusion is "which way does this map" — and the names are FRAMES of the
# frame grammar (frame_names.jl: "world", "camera", "view:<id>",
# "target:<id>", "body:<id>"), so a free-form name is refused and a read can
# mint the labels. The unit (`units.length`), the quaternion convention and what
# a transform does to a point (`conventions`) are declared ONCE for the file, in
# its envelope (validate.jl). A pose embedded in another
# kind (a rig sensor, a target, a trajectory frame, a detection) is the bare
# `{rotation, translation[, covariance]}`: its frames are given by where it
# sits, its conventions by that file's declarations.
#
# `covariance` is present iff the pose is `Uncertain` (VisualGeometryToolkitCore
# conventions/persisted-geometry.md): 6×6 rows over [δω; δt], rotation first,
# rad and `units.length`, in the chart R ↦ exp(δω)·R, t ↦ t + δt.
#
# Related formats (guidance; ours are JSON, typed and declared — see docs/src/guide/related-formats.md):
# - ROS geometry_msgs/TransformStamped (common_msgs/geometry_msgs/msg/
#   TransformStamped.msg:1-2,8-10) names a transform by `header.frame_id` and
#   `child_frame_id`; ours names both ends as `from`/`to` and states what the
#   transform does to a point. ROS quaternions are x,y,z,w (Quaternion.msg:3-6);
#   ours are w,x,y,z, declared.
# - Kalibr's camchain `T_cn_cnm1`, `T_cam_imu` (kalibr_common/ConfigReader.py:
#   678-717, "LastCamToHere", "ImuToCam") put the direction in the NAME,
#   `T_to_from`; ours puts it in two typed fields.
# - ROS PoseWithCovariance (PoseWithCovariance.msg:5-9) orders the covariance
#   translation first; `ros_pose_covariance` converts (conventions.jl).

const _POSE_RECORD = _Obj(
    Pair{Symbol, _Spec}[
        :from => _FRAME_NAME, :to => _FRAME_NAME,
        :rotation => _QUATERNION, :translation => _Vec(_Real(), 3),
    ],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _TRANSFORMS_SPEC = _kind_spec("transforms", Pair{Symbol, _Spec}[:transforms => _Vec(_POSE_RECORD)])

# A covariance with no length in it (px², or a ratio) on the wire, in the storage it holds, and back.
_cov_json(Σ::StaticMatrix) = _cov_wire((x, n) -> x, Σ, ntuple(_ -> 0, size(Σ, 1)))
_cov_matrix(c, n::Int) = _cov_from_wire((x, k) -> Float64(x), c, n, ntuple(_ -> 0, n))

# The plain and uncertain rigid maps, any labels.
const _AnyPose = Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}

"""
    pose_json(X, unit = mm) -> NamedTuple

A physical pose — a VGC `EuclideanMap{3}` whose translation is lengths, or an
`Uncertain` one, any labels — as the embedded `{rotation, translation[,
covariance]}` of every pose-carrying kind: the rotation a w-x-y-z quaternion
with w ≥ 0 (a `QuatRotation`'s own, exactly; any other rotation converted,
reading back as a `QuatRotation`), the translation in the length `unit`
([`write_length`](@ref)), the 6×6 covariance (over `[δω; δt]`, rad and `unit`)
iff `Uncertain`. Its frames are not written: they are given by where it sits,
and its conventions (the quaternion's, the transform's, the unit) are the
enclosing file's envelope (`write_record` writes it from the style).
A plain translation is an `ArgumentError` (attach its unit,
`EuclideanMap(rotation(X), t .* m)`). The inverse is [`pose_from_json`](@ref).
"""
pose_json(X::_AnyPose, unit::_LeafUnit = mm) = JSON.lower(VgioStyle(; length_unit = unit), X)

# The pose record on IO's style (style.jl): the minimal form, a quaternion and a translation.
JSON.lower(s::VgioStyle, X::EuclideanMap{3}) =
    (rotation = collect(_qvec(X.R)), translation = _lengths_json(X.t, s.length_unit, "a pose's translation"))
JSON.lower(s::VgioStyle, u::Uncertain{<:EuclideanMap{3}}) =   # the covariance in the storage it holds
    merge(JSON.lower(s, geom(u)), (covariance = _length_cov_json(cov(u), s.length_unit, _POSE_COV_DIMS),))

"""
    pose_from_json(p, unit::AbstractString) -> EuclideanMap{3} | Uncertain

[`pose_json`](@ref)'s inverse: an embedded `{rotation, translation[,
covariance]}` read in the file's length `unit` (a unit name), returned
UNLABELED, its translation lengths in metres ([`read_length`](@ref)) —
`Uncertain` iff a covariance is stated, its translation block in m² and its
cross block in m·rad. The rotation is the stored quaternion itself, a
`QuatRotation` (not renormalized, not converted to a matrix), so the value
writes back to the same numbers. The quaternion must be unit-norm to 1e-6 and
have w ≥ 0, the declared encoding (refused otherwise, never normalized or
flipped silently).
"""
pose_from_json(p, unit::AbstractString) = JSON.lift(VgioStyle(; length_unit = parse_unit(unit)), EuclideanMap{3}, p::Record)

function JSON.lift(s::VgioStyle, ::Type{EuclideanMap{3}}, p::Record)
    q = SVector{4, Float64}(Float64[Float64(x::Real) for x in p.rotation::Vector{Any}])
    abs(norm(q) - 1) < 1.0e-6 || throw(
        ArgumentError("quaternion is not unit-norm (|q| = $(norm(q))); refusing to normalize silently")
    )
    u = s.length_unit
    q[1] >= 0 || throw(
        ArgumentError("quaternion $(Tuple(q)) has w < 0; the encoding is w ≥ 0 (q and -q are one rotation): refusing to flip it silently")
    )
    X = _quat_map(q, _lengths_from_json(Val(3), p.translation::Vector{Any}, u))
    haskey(p, :covariance) || return X
    return Uncertain(X, CovMat(_length_cov_from_json(p.covariance, 6, u, _POSE_COV_DIMS)))
end

# A labeled pose's frame names, from its labels.
function _labeled_names(X)
    S, D = VgtkCore.map_frames(X)
    (S <: UnlabeledFrame || D <: UnlabeledFrame) && throw(
        ArgumentError("a stored transform names its frames by its labels; this pose is labeled $(S) → $(D) — relabel it")
    )
    return frame_name(S), frame_name(D)
end

"""
    save(path, poses::AbstractVector; length_unit = mm, provenance = (;))

Write labeled physical rigid poses — `EuclideanMap{…, S, D}` or `Uncertain`
ones, their translations lengths — as a `vgio/transforms/v5.0` file: each
record's `from`/`to` are its labels' frame names ([`frame_name`](@ref)), its
translation in `length_unit` (declared in `units`), a `covariance` iff `Uncertain`. An
unlabeled or half-labeled pose is refused (relabel it with the frames it maps
between), and so is a plain translation (attach its unit, `t .* m`).
`provenance` is the file's free-form object.
"""
function save(path::AbstractString, poses::AbstractVector; length_unit::Units = mm, provenance::NamedTuple = (;))
    all(X -> X isa _AnyPose, poses) || throw(
        ArgumentError("save: a transforms file holds rigid poses (EuclideanMap{3}, plain or Uncertain); got $(eltype(poses))")
    )
    s = VgioStyle(; length_unit)
    records = map(poses) do X
        from, to = _labeled_names(X)
        merge((from = from, to = to), JSON.lower(s, X))
    end
    _write_kind(path, "transforms", (transforms = records,), provenance; style = s)
    return nothing
end

"""
    load(Vector{EuclideanMap}, path) -> Vector{Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}}

Read a `vgio/transforms/v5.0` file: each record a pose labeled with its
`from`/`to` frames (`frame_from_name`), its translation lengths in metres
(read in the file's `units.length`), `Uncertain` iff the file states a
covariance. Type-unstable by nature (the labels come from the
file). An earlier version reads through its version steps. A v2 file names its
frames by free strings and is refused, but for the 2026-09 hand-eye record
(`camera_rig → camera_optical`, then `printed_board_<id>` or `scene_<c>` →
`board_rig`), read as the frames the current writer names (`body:camera_rig`,
`camera`, `target:<id>`, `target:scene_<c>`, `body:board_rig`); a v1 file is refused.
"""
function load(::Type{Vector{EuclideanMap}}, path::AbstractString)
    d = _read_store(path, "transforms")
    s = VgioStyle(d)
    return _AnyPose[
        relabel(frame_from_name(r.from::String), frame_from_name(r.to::String), JSON.lift(s, EuclideanMap{3}, r::Record))
            for r in d.transforms::Vector{Any}
    ]
end
