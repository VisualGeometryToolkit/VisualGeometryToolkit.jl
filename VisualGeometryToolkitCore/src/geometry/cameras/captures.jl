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
#   TargetDetections{F}   per frame, per target, `csponds(index, detection)` in the frame F
#                         (pixels or 3D points; its stamps a FrameStamps) — the form
#                         TargetPoints2D/3D become in 0.7
#
# Each may carry `Uncertain` members (conventions/persisted-geometry.md):
# a pose with its 6×6 covariance, a point with its d×d. A physical value holds
# its lengths as FlexUnits quantities in metres (covariances in m²), a plain
# one plain numbers; `ustrip(u, x)` and `with_unit(x, u)` move between them.
#
# These are the values VisualGeometryToolkitIO's `trajectory`, `target_poses`,
# `target_points2d`, `target_points3d` and `target_detections` kinds serialize, exactly; the types
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

"""
    FrameStamps(ts; images = Dict(), sensors = Dict(), times = Dict(), time_origin = nothing)

The per-frame stamps of a capture value whose frame indices are `ts`: per
index, optionally, `images[t]` (the image taken then), `sensors[t]` (what
measured it — one name for every frame, or a map) and `times[t]` (seconds
since `time_origin`, an ISO 8601 date-time, stated iff any time is). Each map
names only indices of `ts` (`ArgumentError`). Read as fields: `s.images`,
`s.sensors`, `s.times`, `s.time_origin`. Held by [`TargetDetections`](@ref),
which forwards the four as its own properties.
"""
struct FrameStamps
    images::Dict{Int, String}
    sensors::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}

    function FrameStamps(
            ts; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
        )
        imgs, sens, tms = _frame_strings(images, "images"), _frame_names(sensors, ts), _frame_times(times)
        _check_stamps(ts, imgs, tms, time_origin, sens)
        return new(imgs, sens, tms, _origin(time_origin))
    end
end

Base.:(==)(a::FrameStamps, b::FrameStamps) =
    a.images == b.images && a.sensors == b.sensors && a.times == b.times && isequal(a.time_origin, b.time_origin)
Base.hash(s::FrameStamps, h::UInt) = hash(s.time_origin, hash(s.times, hash(s.sensors, hash(s.images, hash(FrameStamps, h)))))

# The stamps as the keywords every capture constructor takes.
_stamp_kwargs(s::FrameStamps) = (images = s.images, sensors = s.sensors, times = s.times, time_origin = s.time_origin)

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
        _check_one_form(collect(values(labeled)))   # all plain or all physical
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
with_unit(tr::Trajectory, u::Units) = _map_poses(X -> with_unit(X, u), tr)

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
        _check_one_form(vals)   # all plain or all physical
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
with_unit(tp::TargetPoses, u::Units) = _map_poses(X -> with_unit(X, u), tp)

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
squared). A physical one holds metres, its covariance in m² — stored as given,
never rescaled from the unit the point was written in.
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
stored as `Float64`, or lengths stored in metres. An `Uncertain` physical
point's covariance is in m², stored as given (never rescaled): supply it in m².
"""
const TargetPoint3D = @NamedTuple{target::Symbol, index::Int, xyz::Union{SVector{3}, Uncertain{<:SVector{3}}}}

_pixel(xy::Uncertain) = Uncertain(SVector{2, Float64}(geom(xy)), CovMat(SMatrix{2, 2, Float64, 4}(Statistics.cov(xy))))
_pixel(xy) = SVector{2, Float64}(xy)
# the covariance as given: m² for a physical point (base units), never rescaled
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
in the 3D frame `F` (`WorldFrame` for a tracker's room); a file holds them in
its declared `length_unit`. A motion-capture system's measured markers: a
point is (constellation, marker index), the tracked body's marker
constellation its target definition, never a board mounted on the body.
Measured values only: points computed by posing a design through a pose are
derived and never stored here. `sensors`, `images`, `times`,
`targets` as for [`TargetPoints2D`](@ref). A physical capture holds lengths
in metres and an uncertain point's covariance in m², as supplied ([`has_units`](@ref)); `ustrip(u, x)` and [`with_unit`](@ref)`(x, u)`
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
        _check_one_form([p.xyz for ps in values(fr) for p in ps])
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

# ------------------------------------------------------------------------------
# TargetDetections — per frame, per target, index => detection
# ------------------------------------------------------------------------------

# The dimension of a detection, from the frame it is measured in: a pixel in a
# pixel frame, a 3D point in a named 3D frame.
_detection_dim(::Type{F}) where {F <: PixelFrame} = 2
function _detection_dim(::Type{F}) where {F <: Frame{3}}
    F <: UnlabeledFrame && throw(ArgumentError("3D target detections name the frame they are measured in (e.g. WorldFrame)"))
    return 3
end
_detection_dim(::Type{F}) where {F} =
    throw(ArgumentError("target detections are pixels in a PixelFrame or 3D points in a named 3D frame, not in $(F)"))

# One detection in its stored form (`_pixel`, `_point3`), its dimension checked.
function _detected_point(::Val{D}, p, what) where {D}
    n = length(geom(p))
    n == D || throw(ArgumentError("$(what): a detection of $(n) coordinates where the frame holds $(D)"))
    return D == 2 ? _pixel(p) : _point3(p)
end

# The attribute columns a set carries (`withattrs`), each collected into a
# vector of its own: none for a bare `csponds` StructArray or a Pair vector.
_detection_attrs(cs::Csponds) = map(collect, getfield(cs, :attrs))
_detection_attrs(_) = (;)

# One target's detections in one frame, as stored: `Csponds{Int, P}` of the
# 1-based point indices (unique) and the points, with the set's columns.
function _detection_set(::Val{D}, cs::AbstractVector{<:Pair}, t, id) where {D}
    what = "frame $(t), target $(id)"
    isempty(cs) && throw(ArgumentError("$(what): an empty detection set — a target not detected in a frame is absent"))
    idx = Int[Int(i) for i in sources(cs)]
    all(>=(1), idx) || throw(ArgumentError("$(what): point indices are 1-based (≥ 1), got $(minimum(idx))"))
    allunique(idx) || throw(ArgumentError("$(what): a point index appears twice — a point is detected once per frame"))
    pts = [_detected_point(Val(D), p, what) for p in targets(cs)]
    if !isconcretetype(eltype(pts))
        _check_one_form(pts)   # a unit mix is FlexUnits' DimensionError
        throw(ArgumentError("$(what): plain and Uncertain detections mixed — one form per container"))
    end
    return Csponds(idx, pts, _detection_attrs(cs))
end
_detection_set(::Val, cs, t, id) = throw(
    ArgumentError("frame $(t), target $(id): a detection set is a vector of index => point pairs (`csponds(indices, points)`), got $(typeof(cs))")
)
_detected_points(sets) = Any[p for cs in sets for p in targets(cs)]
_empty_detection_set(::Val{D}) where {D} =
    Csponds{Int, SVector{D, Float64}, Vector{Int}, Vector{SVector{D, Float64}}, @NamedTuple{}}

"""
    TargetDetections{F <: Frame}(frames; images, sensors, times, time_origin, targets = String[])
    TargetDetections(x::TargetPoints2D)
    TargetDetections(x::TargetPoints3D)

Per frame index, per target, the detections of the target's points: `td[t]`
is frame `t`'s target id (`Symbol`) → correspondence set, and `td[t][id]` a
[`Csponds`](@ref) `index => observation` — `index` 1-based into
[`points`](@ref)`(target)` (unique within the set), `observation` the
measured point — so the pairing with the design is
`points(target)[i] => x`. Only detected points are held (sparse, as COLMAP's
and Kalibr's detections are); a target not detected in a frame is absent.
Built from `frames = Dict(t => Dict(id => cs))`, each `cs` any vector of
`index => point` pairs ([`csponds`](@ref)`(indices, points)`), with optional
per-detection attribute columns ([`withattrs`](@ref)`(cs; weights = w)`, a
bundle's weights), per the data-arguments convention.

The dimension is the frame's: `F <: PixelFrame` holds pixels in that pixel
frame (plain `SVector{2, Float64}` or [`UncertainPixel`](@ref), covariances in
px²; [`convert_frame`](@ref) moves them), a named `F <: Frame{3}` 3D points in
it (`WorldFrame` for a tracker's room; `UnlabeledFrame` is refused) — plain
`Float64`, or lengths in metres, an `Uncertain` one's covariance in m², stored
as supplied ([`has_units`](@ref); `ustrip(u, x)` and
[`with_unit`](@ref)`(x, u)` move every point). One form per container:
every set holds the same point type and the same attribute columns
(`ArgumentError`), all physical or all plain (`DimensionError`).

POINTS only: a detection of a non-point feature (an edge, a conic) is another
type. MEASURED only, never computed by posing a design through a pose (that is
derived, a function, and not stored). A motion-capture capture's target is the
tracked body's marker constellation, never a board mounted on the body. Memory
holds metres; a file holds its declared `length_unit` (VisualGeometryToolkitIO's
`target_detections` kind).

The per-frame stamps are a [`FrameStamps`](@ref) (`td.stamps`), read as
`td.images`, `td.sensors`, `td.times` and `td.time_origin` as on
[`TargetPoints2D`](@ref); `targets` names the target definitions the indices
resolve in by content identity (VisualGeometryToolkitIO's `content_id`), unique.
Equal numbers in different frames are different detections (`==`). Otherwise a
Dict: `keys`, `pairs`, iteration.

The conversions from [`TargetPoints2D`](@ref) and [`TargetPoints3D`](@ref)
keep the frame, the stamps and the targets, group each frame's points by
target in their order, and carry the 2D bundle `weight` as a `weights` column;
a value mixing plain and `Uncertain` points is refused (one form).
"""
struct TargetDetections{F <: Frame, C} <: AbstractDict{Int, Dict{Symbol, C}}
    frames::Dict{Int, Dict{Symbol, C}}
    stamps::FrameStamps
    targets::Vector{String}

    function TargetDetections{F}(
            frames::AbstractDict; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
            targets::AbstractVector{<:AbstractString} = String[],
        ) where {F <: Frame}
        D = Val(_detection_dim(F))
        fr = Dict{Int, Dict{Symbol, Any}}(
            Int(t) => Dict{Symbol, Any}(Symbol(id) => _detection_set(D, cs, t, id) for (id, cs) in byid)
                for (t, byid) in frames
        )
        sets = Any[cs for s in values(fr) for cs in values(s)]
        _check_one_form(_detected_points(sets))   # all plain or all physical
        C = isempty(sets) ? _empty_detection_set(D) : typeof(first(sets))
        all(cs -> typeof(cs) === C, sets) || throw(
            ArgumentError(
                "one form per container: every detection set holds the same point type (all plain or all " *
                    "Uncertain) and the same attribute columns; got $(join(unique(map(typeof, sets)), ", "))"
            )
        )
        stamps = FrameStamps(keys(fr); images, sensors, times, time_origin)
        allunique(targets) || throw(ArgumentError("`targets` names a target definition twice"))
        return new{F, C}(Dict{Int, Dict{Symbol, C}}(t => Dict{Symbol, C}(s) for (t, s) in fr), stamps, String.(targets))
    end
end

# The stamps read as the value's own properties, as on the other captures.
const _STAMP_FIELDS = (:images, :sensors, :times, :time_origin)
Base.getproperty(x::TargetDetections, k::Symbol) =
    k in _STAMP_FIELDS ? getfield(getfield(x, :stamps), k) : getfield(x, k)
Base.propertynames(::TargetDetections) = (:frames, :stamps, :targets, _STAMP_FIELDS...)

# Every point moved by `f`, the indices, columns, stamps and targets kept, in frame `G`.
_map_detections(f, ::Type{G}, x::TargetDetections) where {G} = TargetDetections{G}(
    Dict(t => Dict(id => Csponds(sources(cs), map(f, targets(cs)), getfield(cs, :attrs)) for (id, cs) in s) for (t, s) in x.frames);
    _stamp_kwargs(x.stamps)..., targets = x.targets,
)

# A set's equality includes its attribute columns (a Csponds' `==` compares the pairs).
_same_detections(a::Csponds, b::Csponds) = a == b && getfield(a, :attrs) == getfield(b, :attrs)
Base.:(==)(a::TargetDetections{F}, b::TargetDetections{G}) where {F, G} =
    F === G && keys(a.frames) == keys(b.frames) &&
    all(t -> keys(a[t]) == keys(b[t]) && all(id -> _same_detections(a[t][id], b[t][id]), keys(a[t])), keys(a.frames)) &&
    a.stamps == b.stamps && a.targets == b.targets

has_units(x::TargetDetections) = any(s -> any(cs -> any(has_units, targets(cs)), values(s)), values(x.frames))
FlexUnits.ustrip(u::Units, x::TargetDetections{F}) where {F <: Frame{3}} = _map_detections(p -> _ustrip_point(u, p), F, x)
with_unit(x::TargetDetections{F}, u::Units) where {F <: Frame{3}} = _map_detections(p -> _attach_point(u, p), F, x)

# A TargetPoints2D/3D frame's points, grouped by target in their order:
# indices, points (`point(o)`) and the columns `columns(points of the target)`.
function _detections_by_target(os, point, columns)
    return Dict(
        id => (sel = [o for o in os if o.target === id]; Csponds([o.index for o in sel], [point(o) for o in sel], columns(sel)))
            for id in unique(o.target for o in os)
    )
end
_converted_detections(::Type{F}, x, point, columns) where {F} = TargetDetections{F}(
    Dict(t => _detections_by_target(os, point, columns) for (t, os) in x.frames);
    images = x.images, sensors = x.sensors, times = x.times, time_origin = x.time_origin, targets = x.targets,
)
TargetDetections(x::TargetPoints2D{F}) where {F} =
    _converted_detections(F, x, o -> o.xy, sel -> (weights = Float64[o.weight for o in sel],))
TargetDetections(x::TargetPoints3D{F}) where {F} = _converted_detections(F, x, o -> o.xyz, _ -> (;))

# --- the Dict surface, frame index → value (the methods: end of file) --------------

_frames(x::Trajectory) = x.poses
_frames(x::Union{TargetPoses, TargetPoints2D, TargetPoints3D, TargetDetections}) = x.frames

Base.setindex!(x::TargetPoints2D, v, t) = (x.frames[Int(t)] = TargetPoint2D[_target_point2d(o, t) for o in v]; x)

# ------------------------------------------------------------------------------
# convert_frame — the one way to change the pixel frame of pixel data
# ------------------------------------------------------------------------------

"""
    convert_frame(G::Type{<:PixelFrame}, x) -> x in G

The ONE way to change the pixel frame of frame-typed pixel data
([`TargetPoints2D`](@ref), a pixel [`TargetDetections`](@ref); a package adds a method for its own frame-typed
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

function convert_frame(::Type{G}, x::TargetDetections{F}) where {G <: PixelFrame, F <: PixelFrame}
    shift = EuclideanMap(F, G)
    return _map_detections(p -> _shifted(shift, p), G, x)
end

# ------------------------------------------------------------------------------
# CaptureSettings — the camera's settings per frame, and the hardware's identity
# ------------------------------------------------------------------------------
#
# Values are math; identity and provenance are attributes. A camera model, a
# sensor and a lens model hold no names: WHICH body and lens took a capture is
# a `DeviceIdentity` beside the capture, and the settings each frame was taken
# with (exposure, the lens's marked focal length and its focus) are per-frame
# values keyed by the same frame index as every other capture value. Where a
# claim came from (EXIF, a config) is the file's to record
# (VisualGeometryToolkitIO's `capture_settings` kind).

"""
    DeviceIdentity(make, model, serial)
    DeviceIdentity(; make = nothing, model = nothing, serial = nothing)

What a file says a piece of hardware — a camera body, a lens — is: its maker,
model and serial number, each a `String` or `nothing` where unknown (an empty
string is an `ArgumentError`: state `nothing`). A CLAIM, not a fact checked
here: EXIF names an adapted lens by whatever the adapter reports, so a
calibration pipeline may prefer an identity stated in its config. Held beside
the values it identifies ([`CaptureSettings`](@ref), a stored camera's
calibration record), never inside a camera or lens model.
"""
struct DeviceIdentity
    make::Union{Nothing, String}
    model::Union{Nothing, String}
    serial::Union{Nothing, String}
    function DeviceIdentity(make, model, serial)
        return new(_identity_field(make, :make), _identity_field(model, :model), _identity_field(serial, :serial))
    end
end
DeviceIdentity(; make = nothing, model = nothing, serial = nothing) = DeviceIdentity(make, model, serial)

_identity_field(::Nothing, ::Symbol) = nothing
function _identity_field(x::AbstractString, what::Symbol)
    isempty(x) && throw(ArgumentError("DeviceIdentity: `$(what)` is empty — state `nothing` where it is unknown"))
    return String(x)
end

# The second (units.jl), the unit an exposure time is checked and stripped in.
using ..VisualGeometryToolkitCore: _SECOND

# A positive, finite quantity of `u`'s dimension, stored in base form: a plain
# number is a DimensionError (its unit is not stated), another dimension a
# ConversionError.
_positive_quantity(x::Real, u, ::Symbol) = throw(DimensionError(x, u))
function _positive_quantity(q::Quantity, u, what::Symbol)
    v = float(ustrip(u, q))
    (isfinite(v) && v > 0) || throw(ArgumentError("`$(what)` is positive and finite, got $(q)"))
    return base_quantity(q)
end
function _positive_number(x::Real, what::Symbol)
    (isfinite(x) && x > 0) || throw(ArgumentError("`$(what)` is positive and finite, got $(x)"))
    return Float64(x)
end

"""
    ExposureSettings(exposure_time, f_number, iso)

The exposure a frame was taken with: `exposure_time` a time quantity
(`0.004s`, with `s = parse_unit("s")`; stored in seconds, FlexUnits' base
form), `f_number` the aperture's f-number and `iso` the ISO speed, plain
numbers stored as `Float64`. Each is positive and finite (`ArgumentError`); a
plain exposure time is a `DimensionError` (its unit is not stated), a
quantity of another dimension a `ConversionError`.
"""
struct ExposureSettings
    exposure_time::BaseQuantity
    f_number::Float64
    iso::Float64
    function ExposureSettings(exposure_time, f_number::Real, iso::Real)
        return new(
            _positive_quantity(exposure_time, _SECOND, :exposure_time),
            _positive_number(f_number, :f_number), _positive_number(iso, :iso),
        )
    end
end

"""
    LensSettings(focal_length, focus_distance = nothing)

The lens settings a frame was taken with: `focal_length`, the NOMINAL focal
length the lens is marked with or reports (a length; a zoom's setting) — not
a calibrated focal length, which is the camera model's — and
`focus_distance`, the distance the lens was focused at (a length), or
`nothing` where it is not known. Both are positive and finite
(`ArgumentError`): a lens reporting "infinity", or a marker value, is
`nothing` here, never a distance. Lengths are stored in metres.
"""
struct LensSettings
    focal_length::BaseQuantity
    focus_distance::Union{Nothing, BaseQuantity}
    function LensSettings(focal_length, focus_distance = nothing)
        return new(
            _positive_quantity(focal_length, m, :focal_length),
            focus_distance === nothing ? nothing : _positive_quantity(focus_distance, m, :focus_distance),
        )
    end
end

"""
    FrameSettings = @NamedTuple{exposure::ExposureSettings, lens_settings::LensSettings}

One frame's camera settings in a [`CaptureSettings`](@ref): its
[`ExposureSettings`](@ref) and [`LensSettings`](@ref).
"""
const FrameSettings = @NamedTuple{exposure::ExposureSettings, lens_settings::LensSettings}

function _frame_settings(x, t)
    (hasproperty(x, :exposure) && hasproperty(x, :lens_settings)) || throw(
        ArgumentError("frame $(t): its settings state `exposure` and `lens_settings`, got $(x)")
    )
    return FrameSettings((x.exposure, x.lens_settings))
end

"""
    CaptureSettings(frames; body = DeviceIdentity(), lens = DeviceIdentity())

The camera settings of a capture, per frame index: `cs[t]` is frame `t`'s
[`FrameSettings`](@ref), `(; exposure, lens_settings)` (a frame's value is
anything with those two properties). The capture's hardware rides along as
`cs.body` and `cs.lens`, [`DeviceIdentity`](@ref)s. The frame index is the
join key every per-frame value of one capture shares ([`Trajectory`](@ref),
[`TargetPoints2D`](@ref)); a frame without settings is absent. Otherwise a
Dict: `keys`, `pairs`, iteration. [`is_constant`](@ref) says whether every
frame was exposed alike.
"""
struct CaptureSettings <: AbstractDict{Int, FrameSettings}
    frames::Dict{Int, FrameSettings}
    body::DeviceIdentity
    lens::DeviceIdentity
    function CaptureSettings(frames::AbstractDict{<:Integer}; body::DeviceIdentity = DeviceIdentity(), lens::DeviceIdentity = DeviceIdentity())
        return new(Dict{Int, FrameSettings}(Int(t) => _frame_settings(x, t) for (t, x) in frames), body, lens)
    end
end

# The identities are part of the value: equal, isequal and hash all see them
# (AbstractDict's own compare the frames only).
Base.:(==)(a::CaptureSettings, b::CaptureSettings) = a.frames == b.frames && a.body == b.body && a.lens == b.lens
Base.isequal(a::CaptureSettings, b::CaptureSettings) = isequal(a.frames, b.frames) && isequal(a.body, b.body) && isequal(a.lens, b.lens)
Base.hash(x::CaptureSettings, h::UInt) = hash(x.lens, hash(x.body, hash(x.frames, hash(CaptureSettings, h))))

"""
    is_constant(cs::CaptureSettings) -> Bool

Whether every frame of `cs` has the same exposure time, f-number and ISO
(true for none or one frame). The lens settings are not compared: a fixed
exposure under a refocused lens is still one exposure. With it, a per-frame
brightness gain fitted across the capture is the light changing, not the
camera.
"""
function is_constant(cs::CaptureSettings)
    isempty(cs.frames) && return true
    e = first(values(cs.frames)).exposure
    return all(x -> x.exposure == e, values(cs.frames))   # its three fields: time, f-number, ISO
end

_frames(x::CaptureSettings) = x.frames

# ------------------------------------------------------------------------------
# The Dict surface of every per-frame value, frame index → value
# ------------------------------------------------------------------------------

for T in (:Trajectory, :TargetPoses, :TargetPoints2D, :TargetPoints3D, :TargetDetections, :CaptureSettings)
    @eval begin
        Base.length(x::$T) = length(_frames(x))
        Base.iterate(x::$T, s...) = iterate(_frames(x), s...)
        Base.getindex(x::$T, t) = _frames(x)[t]
        Base.get(x::$T, t, default) = get(_frames(x), t, default)
        Base.haskey(x::$T, t) = haskey(_frames(x), t)
        Base.keys(x::$T) = keys(_frames(x))
    end
end
