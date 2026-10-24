# =============================================================================
# vgio/transforms/v3 — named rigid transforms between typed frames
# =============================================================================
#
# {"schema": "vgio/transforms/v3",
#  "length_unit": "mm",
#  "rotation_encoding": "quaternion_wxyz_hamilton",
#  "transform_convention": "x_to = R * x_from + t",
#  "transforms": [{"from": "body:camera_rig", "to": "camera_optical",
#                  "rotation": [w, x, y, z], "translation": [x, y, z],
#                  "covariance": [[…6], …6 rows]}],                 (covariance optional)
#  "provenance": {…}}                                               (optional)
#
# Every record names its direction (`from`, `to`) — half of every hand-eye
# confusion is "which way does this map" — and the names are FRAMES of the
# frame grammar (frame_names.jl: "world", "camera_optical", "view:<id>",
# "target:<id>", "body:<id>"), so a free-form name is refused and a read can
# mint the labels. The unit, the quaternion convention and what a transform
# does to a point are declared ONCE for the file. A pose embedded in another
# kind (a rig sensor, a target, a trajectory frame, a detection) is the bare
# `{rotation, translation[, covariance]}`: its frames are given by where it
# sits, its conventions by that file's declarations.
#
# `covariance` is present iff the pose is `Uncertain` (VisualGeometryToolkitCore
# conventions/persisted-geometry.md): 6×6 rows over [δω; δt], rotation first,
# rad and `length_unit`, in the chart R ↦ exp(δω)·R, t ↦ t + δt.
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
        :rotation => _Vec(_Real(), 4), :translation => _Vec(_Real(), 3),
    ],
    Pair{Symbol, _Spec}[:covariance => _COV6],
)
const _TRANSFORMS_SPEC = _Obj(
    Pair{Symbol, _Spec}[
        :schema => _schema_tok("transforms"), _POSE_DECLARATIONS...,
        :transforms => _Vec(_POSE_RECORD),
    ],
    _PROVENANCE,
)

"""
    pose_declarations(unit = mm) -> NamedTuple

The declarations every pose-carrying file states once — `length_unit`,
`rotation_encoding` (Hamilton quaternions, w-x-y-z, w ≥ 0) and
`transform_convention` (`x_to = R * x_from + t`) — for a file in `unit`. A
building block for a package serializing its own kind ([`pose_json`](@ref)).
"""
pose_declarations(unit::Units = mm) =
    (length_unit = unit_string(unit), rotation_encoding = _ROTATION_ENCODING, transform_convention = _TRANSFORM_CONVENTION)
_pose_declarations(unit::AbstractString) = pose_declarations(parse_unit(unit))

# A covariance as the wire's array of rows, and back.
_cov_json(Σ::StaticMatrix) = [Float64[Σ[i, j] for j in axes(Σ, 2)] for i in axes(Σ, 1)]
_cov_matrix(c, n::Int) = SMatrix{n, n, Float64}(ntuple(k -> Float64(c[(k - 1) % n + 1][(k - 1) ÷ n + 1]::Real), n * n))

# The plain and uncertain rigid maps, any labels.
const _AnyPose = Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}

_pose_frames(X::EuclideanMap{N, T, R, V, S, D}) where {N, T, R, V, S, D} = (S, D)
_pose_frames(u::Uncertain{<:EuclideanMap{3}}) = _pose_frames(geom(u))

"""
    pose_json(X) -> NamedTuple

A pose — a VGC `EuclideanMap{3}` or an `Uncertain` one, any labels — as the
embedded `{rotation, translation[, covariance]}` of every pose-carrying kind:
the rotation a w-x-y-z quaternion with w ≥ 0 (a `QuatRotation`'s own, exactly;
any other rotation converted, reading back as a `QuatRotation`), the translation in millimetres,
the 6×6 covariance (over `[δω; δt]`, rad and mm) iff `Uncertain`. Its frames
are not written: they are given by where it sits. The inverse is
[`pose_from_json`](@ref).
"""
function pose_json(X::EuclideanMap{3})
    t = ustrip.(mm, _interior_t(X))
    return (rotation = collect(_qvec(X.R)), translation = Float64[t[1], t[2], t[3]])
end
pose_json(u::Uncertain{<:EuclideanMap{3}}) =
    merge(pose_json(geom(u)), (covariance = _cov_json(SMatrix{6, 6, Float64, 36}(cov(u))),))

"""
    pose_from_json(p, unit::AbstractString) -> EuclideanMap{3} | Uncertain

[`pose_json`](@ref)'s inverse: an embedded `{rotation, translation[,
covariance]}` read in the file's length `unit` (a unit name), returned
UNLABELED in millimetres — `Uncertain` iff a covariance is stated, its
translation block moved to mm². The rotation is the stored quaternion itself,
a `QuatRotation` (not renormalized, not converted to a matrix), so the value
writes back to the same numbers. The quaternion must be unit-norm to 1e-6
(refused otherwise, never normalized silently).
"""
function pose_from_json(p, unit::AbstractString)
    q = SVector{4, Float64}(Float64.(p.rotation::Vector{Any})...)
    abs(norm(q) - 1) < 1.0e-6 || throw(
        ArgumentError("quaternion is not unit-norm (|q| = $(norm(q))); refusing to normalize silently")
    )
    t = ustrip.(mm, quantity.(Float64.(p.translation::Vector{Any}), parse_unit(unit)))
    X = _quat_map(canonical_qvec(q), t)
    return haskey(p, :covariance) ? Uncertain(X, _pose_cov_mm(_cov_matrix(p.covariance::Vector{Any}, 6), unit)) : X
end

# The covariance of a pose in `unit`, moved to millimetres (the in-memory unit):
# the translation rows and columns scale by mm-per-unit.
function _pose_cov_mm(Σ::SMatrix{6, 6, Float64}, unit::AbstractString)
    s = Float64(ustrip(mm, quantity(1.0, parse_unit(unit))))      # mm per `unit`
    d = SVector{6, Float64}(1.0, 1.0, 1.0, s, s, s)
    return CovMat(SMatrix{6, 6, Float64, 36}(Σ .* (d * d')))
end

# A labeled pose's frame names, from its labels.
function _labeled_names(X)
    S, D = _pose_frames(X)
    (S <: UnlabeledFrame || D <: UnlabeledFrame) && throw(
        ArgumentError("a stored transform names its frames by its labels; this pose is labeled $(S) → $(D) — relabel it")
    )
    return frame_name(S), frame_name(D)
end

"""
    save(path, poses::AbstractVector; provenance = (;))

Write labeled rigid poses — `EuclideanMap{…, S, D}` or `Uncertain` ones — as a
`vgio/transforms/v3` file: each record's `from`/`to` are its labels' frame
names ([`frame_name`](@ref)), its translation in millimetres, a `covariance`
iff `Uncertain`. An unlabeled or half-labeled pose is refused (relabel it with
the frames it maps between). `provenance` is the file's free-form object.
"""
function save(path::AbstractString, poses::AbstractVector; provenance::NamedTuple = (;))
    all(X -> X isa _AnyPose, poses) || throw(
        ArgumentError("save: a transforms file holds rigid poses (EuclideanMap{3}, plain or Uncertain); got $(eltype(poses))")
    )
    records = map(poses) do X
        from, to = _labeled_names(X)
        merge((from = from, to = to), pose_json(X))
    end
    body = merge((schema = _schema("transforms"),), pose_declarations(mm), (transforms = records,))
    _write_store(path, _TRANSFORMS_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(Vector{EuclideanMap}, path) -> Vector{Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}}

Read a `vgio/transforms/v3` file: each record a pose labeled with its
`from`/`to` frames (`frame_from_name`), in millimetres, `Uncertain` iff the
file states a covariance. Type-unstable by nature (the labels come from the
file). A retired v2 (free-form names) or v1 file is refused.
"""
function load(::Type{Vector{EuclideanMap}}, path::AbstractString)
    d = _read_store(path, "transforms", _TRANSFORMS_SPEC)
    unit = d.length_unit::String
    return _AnyPose[
        relabel(frame_from_name(r.from::String), frame_from_name(r.to::String), pose_from_json(r, unit))
            for r in d.transforms::Vector{Any}
    ]
end
