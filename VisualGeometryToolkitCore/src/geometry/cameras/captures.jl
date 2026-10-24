# ==============================================================================
# Captures — per-frame values of a capture: a body's trajectory and the
# measurements of targets (their poses, their points in an image or in 3D)
# ==============================================================================
#
# A capture is a sequence of instants, each named by its FRAME INDEX `t` (an
# `Int`, unique) — the join key every per-frame value of one capture shares.
# Per index a value may also record the image taken then (`images[t]`), the
# sensor that measured it (`sensors[t]`, a camera's or a tracker's name), and
# the time (`times[t]`, seconds since `time_origin`, an ISO 8601 date-time,
# stated iff any time is). An instant that was not observed is ABSENT, never a
# NaN value.
#
# The hierarchy (the COLMAP decomposition, `Rig` in rig.jl): a rig of targets
# defines the targets and their points; a MEASUREMENT names a point by
# (target id, point index) — `index` 1-based into the target's `points` — and
# is resolved against that definition; a TRAJECTORY is the pose of a body (a
# rig, a camera's view, a target) over time.
#
#   Trajectory{S, D}      the pose S → D per frame index (a body in the world)
#   TargetPoses{F}        per frame, the pose of every target seen, TargetFrame{id} → F
#   TargetPoints2D{F}     per frame, the pixels of target points, in the pixel frame F
#   TargetPoints3D{F}     per frame, the 3D positions of target points, in the 3D frame F
#
# Each may carry `Uncertain` members (conventions/persisted-geometry.md):
# a pose with its 6×6 covariance, a point with its d×d. A physical value holds
# its lengths as FlexUnits quantities in metres (covariances in m²), a plain
# one plain numbers; `ustrip(u, x)` and `with_unit(x, u)` move between them.
#
# These are the values VisualGeometryToolkitIO's `trajectory`, `target_poses`,
# `target_points2d` and `target_points3d` kinds serialize, exactly; the types
# are permissive (an in-memory value may lack its images or sensors) and a
# kind's writer states what its file requires.

# --- the per-frame facts ----------------------------------------------------------

_frame_strings(x::AbstractDict, what) = Dict{Int, String}(Int(t) => String(v) for (t, v) in x)
_frame_times(x::AbstractDict) = Dict{Int, Float64}(Int(t) => Float64(v) for (t, v) in x)

# One name for every frame, or a frame index → name map.
_frame_names(x::AbstractString, ts) = Dict{Int, String}(t => String(x) for t in ts)
_frame_names(x::AbstractDict, ts) = _frame_strings(x, "")

function _check_stamps(ts, images, times, time_origin, sensors = Dict{Int, String}())
    for (what, d) in (("images", images), ("times", times), ("sensors", sensors))
        all(in(ts), keys(d)) || throw(ArgumentError("`$(what)` names a frame index with no value"))
    end
    (isempty(times) || time_origin !== nothing) ||
        throw(ArgumentError("times are seconds since a `time_origin` — state it"))
    return nothing
end

_origin(x::Nothing) = nothing
_origin(x::AbstractString) = String(x)

# The frames of a labeled pose, unlabeled frames included.
_pose_frames(X::EuclideanMap{3}) = _em_frames(X)
_pose_frames(u::Uncertain{<:EuclideanMap{3}}) = _em_frames(geom(u))


# ------------------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------------------

"""
    Trajectory{S, D}(poses; images = Dict(), times = Dict(), time_origin = nothing, recovered = Set{Int}())
    Trajectory(S, D, poses; kwargs...)

The pose of frame `S` in frame `D` (`x_D = R x_S + t`) at each capture
instant, keyed by frame index: `tr[t]` is a labeled `EuclideanMap{…, S, D}` —
or an `Uncertain` one, its covariance kept. Both frames are named (a
trajectory is the pose of ONE body, `BodyFrame{:rig}`, a view or a target, in
one frame, `WorldFrame`): an unlabeled pose is labeled `S → D`, a labeled one
must already be. Optional, per index: `images[t]`, `times[t]` (seconds since
`time_origin`), and `recovered`, the indices whose pose was substituted under
a stated method rather than measured. Otherwise a Dict: `keys`, `pairs`,
iteration; `tr.poses` is the underlying `Dict`.
"""
struct Trajectory{S <: Frame{3}, D <: Frame{3}, P} <: AbstractDict{Int, P}
    poses::Dict{Int, P}
    images::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}
    recovered::Set{Int}

    function Trajectory{S, D}(
            poses::AbstractDict{<:Integer};
            images::AbstractDict = Dict{Int, String}(), times::AbstractDict = Dict{Int, Float64}(),
            time_origin::Union{Nothing, AbstractString} = nothing, recovered = Set{Int}()
        ) where {S <: Frame{3}, D <: Frame{3}}
        (S <: UnlabeledFrame || D <: UnlabeledFrame) &&
            throw(ArgumentError("a trajectory names both frames it maps between (e.g. BodyFrame{:rig} → WorldFrame)"))
        labeled = Dict{Int, Any}(Int(t) => _label_pose(S, D, X) for (t, X) in poses)
        P = isempty(labeled) ? EuclideanMap{3} : typejoin(map(typeof, values(labeled))...)
        rec = Set{Int}(Int(t) for t in recovered)
        imgs, tms = _frame_strings(images, "images"), _frame_times(times)
        _check_stamps(keys(labeled), imgs, tms, time_origin)
        all(in(keys(labeled)), rec) || throw(ArgumentError("`recovered` names a frame index with no pose"))
        return new{S, D, P}(Dict{Int, P}(labeled), imgs, tms, _origin(time_origin), rec)
    end
end
Trajectory(::Type{S}, ::Type{D}, poses::AbstractDict; kwargs...) where {S <: Frame{3}, D <: Frame{3}} =
    Trajectory{S, D}(poses; kwargs...)

# A pose labeled S → D: an unlabeled one is labeled, a labeled one must already be.
function _label_pose(::Type{S}, ::Type{D}, X::Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}) where {S, D}
    Xs, Xd = _pose_frames(X)
    (Xs <: UnlabeledFrame && Xd <: UnlabeledFrame) && return relabel(S, D, X)
    (Xs === S && Xd === D) || throw(ArgumentError("a pose labeled $(Xs) → $(Xd) where $(S) → $(D) is held"))
    return X
end

Base.:(==)(a::Trajectory{S, D}, b::Trajectory{S2, D2}) where {S, D, S2, D2} =
    S === S2 && D === D2 && a.poses == b.poses && a.images == b.images && a.times == b.times &&
    isequal(a.time_origin, b.time_origin) && a.recovered == b.recovered

# Units: every pose moves, the per-frame facts ride along.
has_units(tr::Trajectory) = any(has_units, values(tr.poses))
_map_poses(f, tr::Trajectory{S, D}) where {S, D} = Trajectory{S, D}(
    Dict(t => f(X) for (t, X) in tr.poses);
    images = tr.images, times = tr.times, time_origin = tr.time_origin, recovered = tr.recovered,
)
FlexUnits.ustrip(u::Units, tr::Trajectory) = _map_poses(X -> ustrip(u, X), tr)
with_unit(tr::Trajectory, u::Units) = _map_poses(X -> _with_unit_pose(X, u), tr)

# ------------------------------------------------------------------------------
# TargetPoses
# ------------------------------------------------------------------------------

"""
    TargetPoses{F}(frames; images = Dict(), sensors = Dict(), times = Dict(), time_origin = nothing)

Per frame index, the pose of every target seen, each mapping
`TargetFrame{id} → F` (`F` the measuring camera's frame, `CameraFrame` for a
detector's output): `tp[t]` is frame `t`'s target id → pose (stored
unlabeled, plain or `Uncertain`), [`pose`](@ref)`(tp, t, id)` the labeled
pose. `images[t]` names the image, `sensors[t]` the camera (one name for every
frame, or a map), `times[t]` seconds since `time_origin`. What a target IS —
its points — belongs to its definition (a target `Rig`), not to every pose of
it. Otherwise a Dict: `keys`, `pairs`, iteration.
"""
struct TargetPoses{F <: Frame{3}, P} <: AbstractDict{Int, Dict{Symbol, P}}
    frames::Dict{Int, Dict{Symbol, P}}
    images::Dict{Int, String}
    sensors::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}

    function TargetPoses{F}(
            frames::AbstractDict{<:Integer}; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing
        ) where {F <: Frame{3}}
        F <: UnlabeledFrame && throw(ArgumentError("target poses name the frame they map into (e.g. CameraFrame)"))
        fr = Dict{Int, Dict{Symbol, Any}}(
            Int(t) => Dict{Symbol, Any}(Symbol(id) => _unlabel_target_pose(F, Symbol(id), X) for (id, X) in ps)
                for (t, ps) in frames
        )
        vals = [X for ps in values(fr) for X in values(ps)]
        P = isempty(vals) ? EuclideanMap{3} : typejoin(map(typeof, vals)...)
        imgs, sens, tms = _frame_strings(images, "images"), _frame_names(sensors, keys(fr)), _frame_times(times)
        _check_stamps(keys(fr), imgs, tms, time_origin, sens)
        return new{F, P}(
            Dict{Int, Dict{Symbol, P}}(t => Dict{Symbol, P}(ps) for (t, ps) in fr), imgs, sens, tms, _origin(time_origin),
        )
    end
end

# Stored unlabeled; a labeled pose must be the one the value names.
function _unlabel_target_pose(::Type{F}, id::Symbol, X) where {F}
    S, D = _pose_frames(X)
    (S <: UnlabeledFrame && D <: UnlabeledFrame) || (S === TargetFrame{id} && D === F) ||
        throw(ArgumentError("target $(id): a pose labeled $(S) → $(D) where TargetFrame{:$(id)} → $(F) is held"))
    return unlabel(X)
end

"""
    pose(tp::TargetPoses{F}, t, id) -> EuclideanMap{…, TargetFrame{id}, F} | Uncertain

Target `id`'s pose in frame `t`, minted with its labels; an `Uncertain` one
keeps its covariance.
"""
pose(tp::TargetPoses{F}, t::Integer, id::Symbol) where {F} = relabel(TargetFrame{id}, F, tp.frames[t][id])

Base.:(==)(a::TargetPoses{F}, b::TargetPoses{G}) where {F, G} =
    F === G && a.frames == b.frames && a.images == b.images && a.sensors == b.sensors &&
    a.times == b.times && isequal(a.time_origin, b.time_origin)

has_units(tp::TargetPoses) = any(ps -> any(has_units, values(ps)), values(tp.frames))
_map_poses(f, tp::TargetPoses{F}) where {F} = TargetPoses{F}(
    Dict(t => Dict(id => f(X) for (id, X) in ps) for (t, ps) in tp.frames);
    images = tp.images, sensors = tp.sensors, times = tp.times, time_origin = tp.time_origin,
)
FlexUnits.ustrip(u::Units, tp::TargetPoses) = _map_poses(X -> ustrip(u, X), tp)
with_unit(tp::TargetPoses, u::Units) = _map_poses(X -> _with_unit_pose(X, u), tp)

# ------------------------------------------------------------------------------
# TargetPoints2D / TargetPoints3D
# ------------------------------------------------------------------------------

"""
    UncertainPixel = Uncertain{SVector{2, Float64}, CovMat{2, Float64, 4, UnlabeledFrame{2}}}

A measured pixel with its 2×2 covariance (px²).
"""
const UncertainPixel = Uncertain{SVector{2, Float64}, CovMat{2, Float64, 4, UnlabeledFrame{2}}}

"""
    UncertainPoint3 = Uncertain{SVector{3, Float64}, CovMat{3, Float64, 9, UnlabeledFrame{3}}}

A measured plain 3D point with its 3×3 covariance (in the point's units
squared). A physical one holds metres, its covariance in m².
"""
const UncertainPoint3 = Uncertain{SVector{3, Float64}, CovMat{3, Float64, 9, UnlabeledFrame{3}}}

"""
    TargetPoint2D = @NamedTuple{target::Symbol, index::Int, xy::Union{SVector{2, Float64}, UncertainPixel}, weight::Float64}

One measured pixel of a target point: point `index` (1-based into the
target's `points`) of target `target`, at `xy` in the pixel frame of the
[`TargetPoints2D`](@ref) holding it — plain, or `Uncertain` with its 2×2
covariance in px² — and `weight`, the bundle's (no default: state `1.0` where
the producer has none).
"""
const TargetPoint2D = @NamedTuple{target::Symbol, index::Int, xy::Union{SVector{2, Float64}, UncertainPixel}, weight::Float64}

"""
    TargetPoint3D = @NamedTuple{target::Symbol, index::Int, xyz::Union{SVector{3}, Uncertain{<:SVector{3}}}}

One measured 3D position of a target point (a motion-capture marker): point
`index` of target `target`, at `xyz` (in the frame of the
[`TargetPoints3D`](@ref) holding it), plain or `Uncertain`: plain numbers
stored as `Float64`, or lengths stored in metres.
"""
const TargetPoint3D = @NamedTuple{target::Symbol, index::Int, xyz::Union{SVector{3}, Uncertain{<:SVector{3}}}}

_pixel(xy::Uncertain) = Uncertain(SVector{2, Float64}(geom(xy)), CovMat(SMatrix{2, 2, Float64, 4}(Statistics.cov(xy))))
_pixel(xy) = SVector{2, Float64}(xy)
_point3(p::Uncertain) = Uncertain(_stored_point(Val(3), geom(p)), CovMat(SMatrix{3, 3, Float64, 9}(Statistics.cov(p))))
_point3(p) = _stored_point(Val(3), p)

function _target_point2d(o, t)
    hasproperty(o, :weight) || throw(
        ArgumentError("measurement $(o) in frame $(t) has no `weight` — state it (1.0 where the producer has none)")
    )
    return (target = Symbol(o.target), index = Int(o.index), xy = _pixel(o.xy), weight = Float64(o.weight))
end
_target_point3d(o, t) = (target = Symbol(o.target), index = Int(o.index), xyz = _point3(o.xyz))

"""
    TargetPoints2D{F <: PixelFrame}(frames; images, sensors, times, time_origin, targets = String[])

Per frame index, the pixels of target points — `tp[t]` a
`Vector{`[`TargetPoint2D`](@ref)`}` — every pixel in the pixel frame `F`,
part of the TYPE: the producer that knows its pixels' convention attaches it,
and only [`convert_frame`](@ref) changes it (moving the numbers). The
COLMAP points2D → points3D decomposition: a point is named by (target,
index), resolved against the target definitions — a target `Rig` per
connected component — which `targets` names by content identity
(VisualGeometryToolkitIO's `content_id`). Per frame, optionally: `images[t]`,
`sensors[t]` (one name for all frames, or a map), `times[t]`.
Otherwise a Dict: `obs[t]`, `obs[t] = v`, `keys`, `pairs`, iteration.
"""
struct TargetPoints2D{F <: PixelFrame} <: AbstractDict{Int, Vector{TargetPoint2D}}
    frames::Dict{Int, Vector{TargetPoint2D}}
    images::Dict{Int, String}
    sensors::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}
    targets::Vector{String}

    function TargetPoints2D{F}(
            frames::AbstractDict; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
            targets::AbstractVector{<:AbstractString} = String[],
        ) where {F <: PixelFrame}
        fr = Dict{Int, Vector{TargetPoint2D}}(Int(t) => TargetPoint2D[_target_point2d(o, t) for o in os] for (t, os) in frames)
        imgs, sens, tms = _frame_strings(images, "images"), _frame_names(sensors, keys(fr)), _frame_times(times)
        _check_stamps(keys(fr), imgs, tms, time_origin, sens)
        allunique(targets) || throw(ArgumentError("`targets` names a target definition twice"))
        return new{F}(fr, imgs, sens, tms, _origin(time_origin), String.(targets))
    end
end
TargetPoints2D{F}(frames::Pair...; kwargs...) where {F <: PixelFrame} = TargetPoints2D{F}(Dict(frames...); kwargs...)

"""
    TargetPoints3D{F <: Frame{3}}(frames; images, sensors, times, time_origin, targets = String[])

The 3D sibling of [`TargetPoints2D`](@ref): per frame index, the measured
positions of target points — `tp[t]` a `Vector{`[`TargetPoint3D`](@ref)`}` —
in the 3D frame `F` (`WorldFrame` for a tracker's room), millimetres. A
motion-capture system's markers: a point is (body, marker index), the body's
constellation its target definition. `sensors`, `images`, `times`,
`targets` as for [`TargetPoints2D`](@ref). A physical capture holds lengths
in metres ([`has_units`](@ref)); `ustrip(u, x)` and [`with_unit`](@ref)`(x, u)`
move every point.
"""
struct TargetPoints3D{F <: Frame{3}} <: AbstractDict{Int, Vector{TargetPoint3D}}
    frames::Dict{Int, Vector{TargetPoint3D}}
    images::Dict{Int, String}
    sensors::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}
    targets::Vector{String}

    function TargetPoints3D{F}(
            frames::AbstractDict; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
            targets::AbstractVector{<:AbstractString} = String[],
        ) where {F <: Frame{3}}
        F <: UnlabeledFrame && throw(ArgumentError("3D target points name the frame they are measured in (e.g. WorldFrame)"))
        fr = Dict{Int, Vector{TargetPoint3D}}(Int(t) => TargetPoint3D[_target_point3d(o, t) for o in os] for (t, os) in frames)
        _check_one_form([p.xyz for ps in values(fr) for p in ps], "TargetPoints3D")
        imgs, sens, tms = _frame_strings(images, "images"), _frame_names(sensors, keys(fr)), _frame_times(times)
        _check_stamps(keys(fr), imgs, tms, time_origin, sens)
        allunique(targets) || throw(ArgumentError("`targets` names a target definition twice"))
        return new{F}(fr, imgs, sens, tms, _origin(time_origin), String.(targets))
    end
end

# Equal numbers in different frames are different measurements.
for T in (:TargetPoints2D, :TargetPoints3D)
    @eval Base.:(==)(a::$T{F}, b::$T{G}) where {F, G} =
        F === G && a.frames == b.frames && a.images == b.images && a.sensors == b.sensors &&
        a.times == b.times && isequal(a.time_origin, b.time_origin) && a.targets == b.targets
end

has_units(x::TargetPoints3D) = any(ps -> any(p -> has_units(p.xyz), ps), values(x.frames))
_map_points(f, x::TargetPoints3D{F}) where {F} = TargetPoints3D{F}(
    Dict(t => [merge(p, (xyz = f(p.xyz),)) for p in ps] for (t, ps) in x.frames);
    images = x.images, sensors = x.sensors, times = x.times, time_origin = x.time_origin, targets = x.targets,
)
FlexUnits.ustrip(u::Units, x::TargetPoints3D) = _map_points(p -> _ustrip_point(u, p), x)
with_unit(x::TargetPoints3D, u::Units) = _map_points(p -> _attach_point(u, p), x)

# --- the Dict surface, frame index → value -----------------------------------------

_frames(x::Trajectory) = x.poses
_frames(x::Union{TargetPoses, TargetPoints2D, TargetPoints3D}) = x.frames

for T in (:Trajectory, :TargetPoses, :TargetPoints2D, :TargetPoints3D)
    @eval begin
        Base.length(x::$T) = length(_frames(x))
        Base.iterate(x::$T, s...) = iterate(_frames(x), s...)
        Base.getindex(x::$T, t) = _frames(x)[t]
        Base.get(x::$T, t, default) = get(_frames(x), t, default)
        Base.haskey(x::$T, t) = haskey(_frames(x), t)
        Base.keys(x::$T) = keys(_frames(x))
    end
end
Base.setindex!(x::TargetPoints2D, v, t) = (x.frames[Int(t)] = TargetPoint2D[_target_point2d(o, t) for o in v]; x)

# ------------------------------------------------------------------------------
# convert_frame — the one way to change the pixel frame of pixel data
# ------------------------------------------------------------------------------

"""
    convert_frame(G::Type{<:PixelFrame}, x) -> x in G

The ONE way to change the pixel frame of frame-typed pixel data
([`TargetPoints2D`](@ref); a package adds a method for its own frame-typed
values): every pixel is moved by `EuclideanMap(F, G)` and the result is typed
`G`, so a frame is never relabeled without its numbers moving with it. A
pixel-frame change is a translation, so a covariance does not move.
"""
function convert_frame end

_shifted(shift, xy::Uncertain) = Uncertain(SVector{2, Float64}(shift(geom(xy))), getfield(xy, :attributes))
_shifted(shift, xy) = SVector{2, Float64}(shift(xy))

function convert_frame(::Type{G}, o::TargetPoints2D{F}) where {G <: PixelFrame, F <: PixelFrame}
    shift = EuclideanMap(F, G)
    return TargetPoints2D{G}(
        Dict{Int, Vector{TargetPoint2D}}(t => TargetPoint2D[merge(p, (xy = _shifted(shift, p.xy),)) for p in ps] for (t, ps) in o.frames);
        o.images, o.sensors, o.times, o.time_origin, o.targets,
    )
end
