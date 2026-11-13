# ==============================================================================
# Captures — per-frame values of a capture: a body's trajectory and the
# observations of targets (their poses, their points in an image or in 3D)
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
# defines the targets and their points; an OBSERVATION names a point by
# (target id, point index) — `index` 1-based into the target's `points` — and
# is resolved against that definition; a TRAJECTORY is the pose of a body (a
# rig, a camera's view, a target) over time.
#
#   Trajectory{S, D}      the pose S → D per frame index (a body in the world)
#   TargetPoses{F}        per frame, the pose of every target seen, TargetFrame{id} → F
#   TargetHomographies{F, H} per frame, each target's plane → pixel HomographyMap (the projective TargetPoses)
#   TargetObservations{F} per frame, per target, `correspondences(index, point)` in the frame F
#                         (pixels or 3D points; its stamps a FrameStamps) — the ONE observation
#                         container (the 0.9 2D and 3D target-point containers folded into it in 0.10)
#
# Each may carry `Uncertain` members (conventions/persisted-geometry.md):
# a pose with its 6×6 covariance, a point with its d×d. A physical value holds
# its lengths as FlexUnits quantities in metres (covariances in m²), a plain
# one plain numbers; `ustrip(u, x)` and `with_unit(x, u)` move between them.
#
# These are the values VisualGeometryToolkitIO's `trajectory`, `target_poses` and
# `observations` kinds serialize, exactly; the types
# are permissive (an in-memory value may lack its images or sensors) and a
# kind's writer states what its file requires.

# --- the per-frame facts ----------------------------------------------------------

_frame_strings(x::AbstractDict, what) = Dict{Int, String}(Int(t) => String(v) for (t, v) in x)
_frame_times(x::AbstractDict) = Dict{Int, Float64}(Int(t) => Float64(v) for (t, v) in x)

# One name for every frame, or a frame index → name map.
_frame_names(x::AbstractString, ts) = Dict{Int, String}(t => String(x) for t in ts)
_frame_names(x::AbstractDict, ts) = _frame_strings(x, "")

# The per-frame stamps of a capture value, normalized and checked against its
# frame indices `ts`: (images, sensors, times).
function _stamps(ts, images, sensors, times, time_origin)
    imgs, sens, tms = _frame_strings(images, "images"), _frame_names(sensors, ts), _frame_times(times)
    _check_stamps(ts, imgs, tms, time_origin, sens)
    return imgs, sens, tms
end

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
`s.sensors`, `s.times`, `s.time_origin`. Held by [`TargetObservations`](@ref),
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

# The frames of a labeled pose, unlabeled frames included.


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
        P = isempty(labeled) ? EuclideanMap{3} : mapreduce(typeof, typejoin, values(labeled))
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
    Xs, Xd = map_frames(X)
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
        P = isempty(vals) ? EuclideanMap{3} : mapreduce(typeof, typejoin, vals)
        imgs, sens, tms = _stamps(keys(fr), images, sensors, times, time_origin)
        return new{F, P}(
            Dict{Int, Dict{Symbol, P}}(t => Dict{Symbol, P}(ps) for (t, ps) in fr), imgs, sens, tms, _origin(time_origin),
        )
    end
end

# Stored unlabeled; a labeled pose must be the one the value names.
function _unlabel_target_pose(::Type{F}, id::Symbol, X) where {F}
    S, D = map_frames(X)
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
# TargetHomographies
# ------------------------------------------------------------------------------

"""
    TargetHomographies{F <: PixelFrame}(frames; images, sensors, times, time_origin, targets = String[])

Per frame index, the homography of every target seen, each a `HomographyMap`
taking the target's plane (its chart coordinates, millimetres) to pixels of
the pixel frame `F`, part of the TYPE as in [`TargetObservations`](@ref) — the
projective counterpart of [`TargetPoses`](@ref): `th[t]` is frame `t`'s target
id → homography (stored unlabeled, plain or `Uncertain`, as `TargetPoses`
stores its poses), [`homography`](@ref)`(th, t, id)` the labeled map
[`TargetPlaneFrame`](@ref)` → F` (the plane frame is 2-D and carries no target
id: `TargetFrame{id}` is the target's 3-D frame; the id is the entry's key). A
given map is unlabeled, or labeled exactly so; any other labels are an
`ArgumentError`. An `H` is `RadialHomography`'s `H`: it
acts BEFORE the lens of the camera `sensors[t]` names, about that camera's
centre ([`radial_homography`](@ref)`(camera, H)` pairs them). An `Uncertain`
homography's covariance is 8×8, in the [`HomographyManifold8`](@ref VisualGeometryToolkitCore.HomographyManifold8) chart
about its canonical matrix (`parameter_vector`'s: unit Frobenius norm,
`H[3,3] ≥ 0`). `targets` names the target definitions the ids resolve in by
content identity (VisualGeometryToolkitIO's `content_id`: a rig's or a single
target's). `images[t]` names the image, `sensors[t]` the camera (one name for
every frame, or a map), `times[t]` seconds since `time_origin`.
The pixel frame is always stated: there is no form without `F`.
Throws `ArgumentError` for a stamp (`images`, `sensors`, `times`) naming a
frame index with no value, `times` without a `time_origin`, or a `targets`
naming a definition twice. Otherwise a Dict: `keys`, `pairs`, iteration.
"""
struct TargetHomographies{F <: PixelFrame, H} <: AbstractDict{Int, Dict{Symbol, H}}
    frames::Dict{Int, Dict{Symbol, H}}
    images::Dict{Int, String}
    sensors::Dict{Int, String}
    times::Dict{Int, Float64}
    time_origin::Union{Nothing, String}
    targets::Vector{String}

    function TargetHomographies{F}(
            frames::AbstractDict{<:Integer}; images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
            targets::AbstractVector{<:AbstractString} = String[],
        ) where {F <: PixelFrame}
        fr = Dict{Int, Dict{Symbol, Any}}(
            Int(t) => Dict{Symbol, Any}(Symbol(id) => _unlabel_target_homography(F, Symbol(id), H) for (id, H) in hs)
                for (t, hs) in frames
        )
        vals = [H for hs in values(fr) for H in values(hs)]
        H = isempty(vals) ? HomographyMap{3, Float64, HomographyMat{Float64}, UnlabeledFrame{2}, UnlabeledFrame{2}} : mapreduce(typeof, typejoin, vals)
        imgs, sens, tms = _stamps(keys(fr), images, sensors, times, time_origin)
        allunique(targets) || throw(ArgumentError("`targets` names a target definition twice"))
        return new{F, H}(
            Dict{Int, Dict{Symbol, H}}(t => Dict{Symbol, H}(hs) for (t, hs) in fr), imgs, sens, tms, _origin(time_origin), String.(targets),
        )
    end
end

# Stored unlabeled; a labeled map must be the one the value names (TargetPoses' rule).
function _unlabel_target_homography(::Type{F}, id::Symbol, h) where {F}
    h isa Union{HomographyMap{3}, Uncertain{<:HomographyMap{3}}} || throw(ArgumentError(
        "a target homography is a HomographyMap (plain or Uncertain); got a $(typeof(h)) — lift a matrix with HomographyMap(H)"))
    S, D = map_frames(h)
    (S <: UnlabeledFrame && D <: UnlabeledFrame) || (S === TargetPlaneFrame && D === F) ||
        throw(ArgumentError("target $(id): a homography labeled $(S) → $(D) where TargetPlaneFrame → $(F) is held"))
    return unlabel(h)
end

"""
    homography(th::TargetHomographies{F}, t, id) -> HomographyMap{3, …, TargetPlaneFrame, F} | Uncertain

Target `id`'s plane → pixel homography in frame `t`, minted with its labels; an
`Uncertain` one keeps its covariance.
"""
homography(th::TargetHomographies{F}, t::Integer, id::Symbol) where {F} = relabel(TargetPlaneFrame, F, th.frames[t][id])

# ------------------------------------------------------------------------------
# A plane homography's parameters: its nine entries, canonical
# ------------------------------------------------------------------------------
#
# A homography has 8 degrees of freedom but no global 8-number chart: fixing
# h33 = 1 fails for every H with h33 = 0 (a plane through the camera centre's
# image line), and any other fixed entry fails likewise. Its parameters are
# therefore its nine entries, row-major, under a DECLARED normalization —
# unit Frobenius norm and H[3,3] ≥ 0 (`sign_normalize`) — which picks one
# representative of the projective class: one placement always has the same
# nine numbers. A tangent covariance is 8×8 in `HomographyManifold8` about
# that representative (`minimal_manifold`), never over the nine.

const _HOMOGRAPHY_PARAMETER_NAMES = (:h11, :h12, :h13, :h21, :h22, :h23, :h31, :h32, :h33)
# canonical within a few ulp: `sign_normalize`'s gauge (dividing a unit-norm H by its norm can move its last bit)
_is_canonical_homography(M::StaticMatrix{3, 3, T}) where {T} = abs(norm(M) - 1) <= 4 * eps(float(T)) && M[3, 3] >= 0

"""
    parameter_names(::HomographyMap{3}) -> NTuple{9, Symbol}
    parameter_vector(h::HomographyMap{3}) -> SVector{9}
    with_parameters(h::HomographyMap{3}, θ) -> HomographyMap{3}

The parameter protocol of a plane homography: its nine entries `h11 … h33`,
row-major, of its CANONICAL matrix — unit Frobenius norm and `H[3,3] ≥ 0`
([`sign_normalize`](@ref)); a matrix already canonical (within 4 ulp) is
returned as held, so a round trip is bit for bit. Nine numbers, not eight: a
homography has 8 degrees of freedom but no global 8-number chart (fixing
`h33 = 1` fails at `h33 = 0`), so the normalization is declared instead. A
negligible or non-finite matrix is an `ArgumentError`. `with_parameters`
rebuilds the map (its frames kept) from nine row-major entries as given; a
singular one is an `ArgumentError`. No entry is a length (`length_mask`).
Its tag is `PLANAR_MAP_TAGS`' `"homography_map"`.
"""
parameter_names(::HomographyMap{3}) = _HOMOGRAPHY_PARAMETER_NAMES
function parameter_vector(h::HomographyMap{3, T}) where {T}
    M = SMatrix{3, 3, T, 9}(h.forward)
    all(isfinite, M) || throw(ArgumentError("parameter_vector(::HomographyMap): the homography has a non-finite entry"))
    if !_is_canonical_homography(M)
        Mn = sign_normalize(M)
        Mn === nothing && throw(ArgumentError("parameter_vector(::HomographyMap): the homography is degenerate (negligible norm)"))
        M = SMatrix{3, 3, T, 9}(Mn)
    end
    return SVector{9, T}(M[1, 1], M[1, 2], M[1, 3], M[2, 1], M[2, 2], M[2, 3], M[3, 1], M[3, 2], M[3, 3])
end
function with_parameters(h::HomographyMap{3, T, M, S, D}, θ::AbstractVector) where {T, M, S, D}
    length(θ) == 9 || throw(ArgumentError("with_parameters(::HomographyMap): 9 parameters (h11 … h33, row-major); got $(length(θ))"))
    U = float(eltype(θ))
    H = HomographyMat{U}(SMatrix{3, 3, U, 9}(θ[1], θ[4], θ[7], θ[2], θ[5], θ[8], θ[3], θ[6], θ[9]))
    m = HomographyMap(H)
    all(isfinite, m.inverse) || throw(ArgumentError("with_parameters(::HomographyMap): the homography is singular (it has no inverse)"))
    return relabel(S, D, m)
end
length_mask(::Type{<:HomographyMap{3}}) = SVector{9}(ntuple(_ -> false, Val(9)))
length_mask(h::HomographyMap{3}) = length_mask(typeof(h))

"""
    PLANAR_MAP_TAGS

The tags of the planar maps a record embeds by their parameters (the
parametric record `{"type": <tag>, "parameters": [θ…]}`): `"homography_map"`
→ `HomographyMap`. A [`TagRegistry`](@ref); the target calibrations, which are
also planar maps, keep their own family ([`TARGET_CALIBRATION_TAGS`](@ref)) —
a homography is an observation of a placement, not a calibration of a target.
"""
const PLANAR_MAP_TAGS = TagRegistry(:planar_map, "", "homography_map" => HomographyMap)

# Equal numbers in different pixel frames are different placements.
Base.:(==)(a::TargetHomographies{F}, b::TargetHomographies{G}) where {F, G} =
    F === G && a.frames == b.frames && a.images == b.images && a.sensors == b.sensors &&
    a.times == b.times && isequal(a.time_origin, b.time_origin) && a.targets == b.targets

# ------------------------------------------------------------------------------
# The stored point forms of TargetObservations
# ------------------------------------------------------------------------------

"""
    UncertainPixel = Uncertain{SVector{2, Float64}, CovMat{2, Float64, SMatrix{2, 2, Float64, 4}, UnlabeledFrame{2}}}

A measured pixel with its 2×2 covariance (px²), in the full form. A
[`TargetObservations`](@ref) set keeps the storage its points' covariances hold
(isotropic, diagonal or full; a set of mixed storage in the coarser one), so a
detector's isotropic noise stays one number per point; a container typed with
this alias converts a structured covariance to the full form on the way in.
"""
const UncertainPixel = Uncertain{SVector{2, Float64}, CovMat{2, Float64, SMatrix{2, 2, Float64, 4}, UnlabeledFrame{2}}}

"""
    UncertainPoint3 = Uncertain{SVector{3, Float64}, CovMat{3, Float64, SMatrix{3, 3, Float64, 9}, UnlabeledFrame{3}}}

A measured plain 3D point with its 3×3 covariance (in the point's units
squared), the full form. A physical one holds metres, its covariance in m² —
stored as given, never rescaled from the unit the point was written in.
"""
const UncertainPoint3 = Uncertain{SVector{3, Float64}, CovMat{3, Float64, SMatrix{3, 3, Float64, 9}, UnlabeledFrame{3}}}

# the covariance in Float64 and the storage it holds (full, diagonal or isotropic), unlabeled
_pixel(xy::Uncertain) = Uncertain(SVector{2, Float64}(geom(xy)), CovMat{2, Float64}(parent(Statistics.cov(xy))))
_pixel(xy) = SVector{2, Float64}(xy)
# the covariance as given: m² for a physical point (base units), never rescaled
_point3(p::Uncertain) = Uncertain(_stored_point(Val(3), geom(p)), CovMat{3, Float64}(parent(Statistics.cov(p))))
_point3(p) = _stored_point(Val(3), p)
# Uncertain points of different covariance storage, in the coarser one (`promote`): one concrete element type
function _one_storage(pts::AbstractVector)
    all(p -> p isa Uncertain, pts) || return pts
    M = mapreduce(p -> typeof(getfield(p, :attributes)), promote_type, pts)
    return [Uncertain(geom(p), convert(M, getfield(p, :attributes))) for p in pts]
end

# ------------------------------------------------------------------------------
# TargetObservations — per frame, per target, index => detection
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
# vector of its own: none for a bare `correspondences` StructArray or a Pair vector.
_detection_attrs(cs::Correspondences) = map(collect, cs.attrs)
_detection_attrs(_) = (;)

# One target's detections in one frame, as stored: `Correspondences{Int, P}` of the
# 1-based point indices (unique) and the points, with the set's columns.
function _detection_set(::Val{D}, cs::AbstractVector{<:Pair}, t, id) where {D}
    what = "frame $(t), target $(id)"
    isempty(cs) && throw(ArgumentError("$(what): an empty detection set — a target not detected in a frame is absent"))
    idx = Int[Int(i) for i in sources(cs)]
    all(>=(1), idx) || throw(ArgumentError("$(what): point indices are 1-based (≥ 1), got $(minimum(idx))"))
    allunique(idx) || throw(ArgumentError("$(what): a point index appears twice — a point is detected once per frame"))
    pts = [_detected_point(Val(D), p, what) for p in targets(cs)]
    isconcretetype(eltype(pts)) || (pts = _one_storage(pts))   # mixed covariance storage: the coarser one
    if !isconcretetype(eltype(pts))
        _check_one_form(pts)   # a unit mix is FlexUnits' DimensionError
        throw(ArgumentError("$(what): plain and Uncertain detections mixed — one form per container"))
    end
    return Correspondences(idx, pts, _detection_attrs(cs))
end
_detection_set(::Val, cs, t, id) = throw(
    ArgumentError("frame $(t), target $(id): a detection set is a vector of index => point pairs (`correspondences(indices, points)`), got $(typeof(cs))")
)
_empty_detection_set(::Val{D}) where {D} = typeof(Correspondences(Int[], SVector{D, Float64}[]))

# The private route of the per-point maps (`_map_detections`): the sets are
# built from a checked value's, so nothing is validated again.
struct _UncheckedDetections end

"""
    TargetObservations{F <: Frame}(frames; images, sensors, times, time_origin, targets = String[])
    TargetObservations{F <: Frame}(frames; stamps::FrameStamps, targets = String[])

Per frame index, per target, the observations of the target's points (detections associated with
the target's design points): `td[t]`
is frame `t`'s target id (`Symbol`) → correspondence set, and `td[t][id]` a
[`Correspondences`](@ref) `index => observation` — `index` 1-based into
[`points`](@ref)`(target)` (unique within the set), `observation` the
measured point — so the pairing with the design is
`points(target)[i] => x`. Only detected points are held (sparse, as COLMAP's
and Kalibr's detections are); a target not detected in a frame is absent.
Built from `frames = Dict(t => Dict(id => cs))`, each `cs` any vector of
`index => point` pairs ([`correspondences`](@ref)`(indices, points)`), with optional
per-detection attribute columns ([`withattrs`](@ref VisualGeometryToolkitCore.withattrs)`(cs; weights = w)`, a
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
`observations` kind).

The per-frame stamps are a [`FrameStamps`](@ref) (`td.stamps`) — given as
the four keywords or as one `stamps` value, not both (`ArgumentError`) — read as
`td.images`, `td.sensors`, `td.times` and `td.time_origin`; `targets` names the target definitions the indices
resolve in by content identity (VisualGeometryToolkitIO's `content_id`: a rig's
of targets, or a single unposed target's, whose detections are named by its
design `reference`), unique.
`==`, `isequal` and `hash` agree: the frame type, the frame keys, every set
with its attribute columns, the stamps and the targets — equal numbers in
different frames are different detections. Otherwise a Dict: `keys`, `pairs`,
iteration.

A 2D bundle's per-point weight is a `weights` column. The 0.9 containers of
2D and 3D target points, and their element aliases, are folded into this one
container (0.10): a file of the old kinds is read by VisualGeometryToolkitIO's version
steps into it.
"""
struct TargetObservations{F <: Frame, C} <: AbstractDict{Int, Dict{Symbol, C}}
    frames::Dict{Int, Dict{Symbol, C}}
    stamps::FrameStamps
    targets::Vector{String}

    function TargetObservations{F}(
            frames::AbstractDict; stamps::Union{Nothing, FrameStamps} = nothing, images::AbstractDict = Dict{Int, String}(),
            sensors::Union{AbstractString, AbstractDict} = Dict{Int, String}(),
            times::AbstractDict = Dict{Int, Float64}(), time_origin::Union{Nothing, AbstractString} = nothing,
            targets::AbstractVector{<:AbstractString} = String[],
        ) where {F <: Frame}
        D = Val(_detection_dim(F))
        # every set checked and stored; each holds one concrete point type, so
        # one point per set states the container's form
        sets = [(Int(t), Symbol(id), _detection_set(D, cs, t, id)) for (t, byid) in frames for (id, cs) in byid]
        _check_one_form([last(cs[1]) for (_, _, cs) in sets])   # all plain or all physical (`targets` is the keyword here)
        C = isempty(sets) ? _empty_detection_set(D) : typeof(sets[1][3])
        all(s -> typeof(s[3]) === C, sets) || throw(
            ArgumentError(
                "one form per container: every detection set holds the same point type (all plain or all " *
                    "Uncertain) and the same attribute columns; got $(join(unique(typeof(s[3]) for s in sets), ", "))"
            )
        )
        fr = Dict{Int, Dict{Symbol, C}}(Int(t) => Dict{Symbol, C}() for t in keys(frames))
        for (t, id, cs) in sets
            fr[t][id] = cs
        end
        if stamps === nothing
            stamps = FrameStamps(keys(fr); images, sensors, times, time_origin)
        else
            (isempty(images) && isempty(sensors) && isempty(times) && time_origin === nothing) || throw(
                ArgumentError("TargetObservations: the stamps are given as `stamps` or as `images`, `sensors`, `times` and `time_origin`, not both")
            )
            _check_stamps(keys(fr), stamps.images, stamps.times, stamps.time_origin, stamps.sensors)
        end
        allunique(targets) || throw(ArgumentError("`targets` names a target definition twice"))
        return new{F, C}(fr, stamps, String.(targets))
    end
    TargetObservations{F}(::_UncheckedDetections, frames::Dict{Int, Dict{Symbol, C}}, stamps::FrameStamps, targets::Vector{String}) where {F <: Frame, C} =
        new{F, C}(frames, stamps, targets)
end
# The value's own type names the frame; the set type follows from the sets
# (`typeof(td)(frames; …)` rebuilds one).
function TargetObservations{F, C}(frames::AbstractDict; kwargs...) where {F <: Frame, C}
    x = TargetObservations{F}(frames; kwargs...)
    x isa TargetObservations{F, C} || throw(ArgumentError("TargetObservations{$(F), $(C)}: the sets given are $(typeof(x).parameters[2])"))
    return x
end

# The stamps read as the value's own properties, as on the other captures.
const _STAMP_FIELDS = (:images, :sensors, :times, :time_origin)
Base.getproperty(x::TargetObservations, k::Symbol) =
    k in _STAMP_FIELDS ? getfield(getfield(x, :stamps), k) : getfield(x, k)
Base.propertynames(::TargetObservations) = (:frames, :stamps, :targets, _STAMP_FIELDS...)

# Every point moved by `f` (one stored point type to another), the indices,
# columns, stamps and targets kept, in frame `G`: a checked value's sets, moved
# point by point, need no validation again.
function _map_detections(f, ::Type{G}, x::TargetObservations) where {G}
    moved(cs) = Correspondences(copy(sources(cs)), map(f, targets(cs)), _detection_attrs(cs))
    sets = (cs for s in values(x.frames) for cs in values(s))
    C = isempty(sets) ? _empty_detection_set(Val(_detection_dim(G))) : typeof(moved(first(sets)))
    frames = Dict{Int, Dict{Symbol, C}}(t => Dict{Symbol, C}(id => moved(cs) for (id, cs) in s) for (t, s) in x.frames)
    return TargetObservations{G}(_UncheckedDetections(), frames, x.stamps, x.targets)
end

# A set's equality includes its attribute columns (a Correspondences' `==` compares
# its pairs only): a weighted set is not its unweighted one.
_same_detections(a::Correspondences, b::Correspondences) = a == b && a.attrs == b.attrs
_isequal_detections(a::Correspondences, b::Correspondences) = isequal(a, b) && isequal(a.attrs, b.attrs)
_same_frames(same, a::AbstractDict, b::AbstractDict) =
    keys(a) == keys(b) && all(t -> keys(a[t]) == keys(b[t]) && all(id -> same(a[t][id], b[t][id]), keys(a[t])), keys(a))

# Equal: the same frame type `F`, frame keys, per-target sets with their
# columns, stamps and targets. `isequal` and `hash` agree with it (the
# AbstractDict fallbacks would compare the frames' sets only), so a value is a
# sound Dict key.
Base.:(==)(a::TargetObservations{F}, b::TargetObservations{G}) where {F, G} =
    F === G && _same_frames(_same_detections, a.frames, b.frames) && a.stamps == b.stamps && a.targets == b.targets
Base.isequal(a::TargetObservations{F}, b::TargetObservations{G}) where {F, G} =
    F === G && _same_frames(_isequal_detections, a.frames, b.frames) && isequal(a.stamps, b.stamps) && isequal(a.targets, b.targets)
function Base.hash(x::TargetObservations{F}, h::UInt) where {F}
    # the frames are unordered: each (frame, target, set, columns) hashed alone, combined by xor
    hf = zero(UInt)
    for (t, s) in x.frames, (id, cs) in s
        hf ⊻= hash(t, hash(id, hash(cs.attrs, hash(cs))))
    end
    return hash(x.targets, hash(x.stamps, hash(hf, hash(F, hash(TargetObservations, h)))))
end

# One form per container: its first point states it.
function has_units(x::TargetObservations)
    for s in values(x.frames), cs in values(s)
        return has_units(last(cs[1]))
    end
    return false
end
FlexUnits.ustrip(u::Units, x::TargetObservations{F}) where {F <: Frame{3}} = _map_detections(p -> _ustrip_point(u, p), F, x)
with_unit(x::TargetObservations{F}, u::Units) where {F <: Frame{3}} = _map_detections(p -> _attach_point(u, p), F, x)

# --- the Dict surface, frame index → value (the methods: end of file) --------------

_frames(x::Trajectory) = x.poses
_frames(x::Union{TargetPoses, TargetHomographies, TargetObservations}) = x.frames

# ------------------------------------------------------------------------------
# convert_frame — the one way to change the pixel frame of pixel data
# ------------------------------------------------------------------------------

"""
    convert_frame(G::Type{<:PixelFrame}, x) -> x in G

The ONE way to change the pixel frame of frame-typed pixel data
(a pixel [`TargetObservations`](@ref); a package adds a method for its own frame-typed
values): every pixel is moved by `EuclideanMap(F, G)` and the result is typed
`G`, so a frame is never relabeled without its numbers moving with it. A
pixel-frame change is a translation, so a covariance does not move.
"""
function convert_frame end

_shifted(shift, xy::Uncertain) = Uncertain(SVector{2, Float64}(shift(geom(xy))), getfield(xy, :attributes))
_shifted(shift, xy) = SVector{2, Float64}(shift(xy))

function convert_frame(::Type{G}, x::TargetObservations{F}) where {G <: PixelFrame, F <: PixelFrame}
    shift = EuclideanMap(F, G)
    return _map_detections(p -> _shifted(shift, p), G, x)
end

"""
    convert_frame(G::Type{<:PixelFrame}, camera::CameraModel{<:ProjectiveIntrinsics}) -> CameraModel

The projective camera with its principal point moved from the package's
COLMAP pixel frame to `G` by `EuclideanMap(PixelFrame{:colmap}, G)`, so that
`project` of the result is `project` of the camera in `G`'s pixels. The lens
is unchanged: it acts on centred coordinates, which a translation of the pixel
origin does not touch. The result is a `CameraModel` in `G`'s convention;
the camera itself holds no frame tag, so converting it again is the caller's
bookkeeping. The element type is preserved.
"""
function convert_frame(::Type{G}, camera::CameraModel{<:ProjectiveIntrinsics{T}}) where {G <: PixelFrame, T}
    c = principal_point(camera.intrinsics)
    moved = EuclideanMap(PixelFrame{:colmap}, G)(SVector{2, T}(c[1], c[2]))
    return CameraModel(ProjectiveIntrinsics{T}(Point2{T}(moved[1], moved[2])), camera.lens)
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
[`TargetObservations`](@ref)); a frame without settings is absent. Otherwise a
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

for T in (:Trajectory, :TargetPoses, :TargetHomographies, :TargetObservations, :CaptureSettings)
    @eval begin
        Base.length(x::$T) = length(_frames(x))
        Base.iterate(x::$T, s...) = iterate(_frames(x), s...)
        Base.getindex(x::$T, t) = _frames(x)[t]
        Base.get(x::$T, t, default) = get(_frames(x), t, default)
        Base.haskey(x::$T, t) = haskey(_frames(x), t)
        Base.keys(x::$T) = keys(_frames(x))
    end
end
