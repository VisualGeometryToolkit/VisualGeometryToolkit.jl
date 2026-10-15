# ==============================================================================
# Rig — a frame with posed members (the COLMAP-style decomposition)
# ==============================================================================
#
# The COLMAP-shaped decomposition of a multi-view capture separates what a
# "view" conflates:
#
#   Rig        members with FIXED poses in a common frame — the assembly. Each
#              member is a (model, mount) pair: a camera (its model and its
#              rig-relative extrinsics) or a target (an `AbstractTarget`,
#              targets.jl, and its mount, `Mounted`). Three storages, ONE type:
#              sensors only (tuple storage, small N, static dispatch — a single
#              camera is a rig of one, `StereoRig` the 2-tuple alias), targets
#              only (dict storage, many members, labels minted on access), and
#              both (a camera rig carrying its marker constellation: the tuple
#              and the dict side by side). One interface over all three: `ids`,
#              `pose`, `extrinsics`, `length`, `haskey`, `rig[id]`.
#   frames     the rig's pose per capture instant (`world → rig` at time t) —
#              a keyed collection of `EuclideanMap`s owned by the estimator.
#              Temporal, never rigid, hundreds of entries: deliberately NOT a
#              type here (a Dict of labeled maps needs no algebra of its own).
#   image      = (frame, sensor): its camera pose is `rig→sensor ∘ world→rig`,
#              derived on demand, never stored.
#
# THE FRAME IS THE GAUGE. A rig's frame `F` is a member's own frame whenever
# the assembly is self-referenced — the reference camera of a stereo rig
# (`ViewFrame{id₁}`), the gauge board of a calibration target
# (`TargetFrame{uid}`) — and that member's pose is the identity, checked at
# construction. A rig posed in an external frame (a body frame, a tracker's
# room, the world) names that frame instead and has no gauge member.
#
# A view is `frames × sensors`. This is why "a rig of 300 views" is not
# representable on purpose: 300 views of a moving camera are a trajectory
# (frames of a 1-sensor rig), not a rigid assembly.

# ------------------------------------------------------------------------------
# Frame-parameter helpers (labels are the single source of truth)
# ------------------------------------------------------------------------------

# Raw stored extrinsics of a camera, WITHOUT the CameraExtrinsics relabel that
# the public `extrinsics(::Camera)` accessor applies. Rig sensors carry
# rig-relative labels (`F → ViewFrame{id}`) in this slot, which the accessor
# would clobber with `WorldFrame → CameraFrame`.
_raw_extrinsics(c::Camera) = c.extrinsics
_raw_extrinsics(c::BoundedCamera) = _raw_extrinsics(c.camera)
_raw_extrinsics(::AbstractCameraModel) =
    throw(ArgumentError("an unposed camera (a BoundedCamera of a model) has no extrinsics — pose it: BoundedCamera(Camera(model, X), extent)"))

# (Src, Dst) frame types of a labeled map, read off the type parameters.
_em_frames(::EuclideanMap{N, T, R, V, Src, Dst}) where {N, T, R, V, Src, Dst} =
    (Src, Dst)

# View-identity symbol of a rig sensor's extrinsics (`F → ViewFrame{id}` → `id`).
_view_id(::EuclideanMap{3, T, R, V, S, ViewFrame{b}}) where {T, R, V, S, b} = b

# Rebuild a camera with new extrinsics, preserving a BoundedCamera wrapper.
_with_extrinsics(c::Camera, X::EuclideanMap{3}) = Camera(c.model, X)
_with_extrinsics(m::AbstractCameraModel, X::EuclideanMap{3}) = Camera(m, X)   # posing an unposed camera
_with_extrinsics(c::BoundedCamera, X::EuclideanMap{3}) =
    BoundedCamera(_with_extrinsics(c.camera, X), c.extent)

# The identity check of a gauge member's pose: the rig frame IS that member's
# frame, so its stored map must be the identity to construction accuracy (a
# re-expressed bundle gauge is `inv(T) ∘ T`, exact to rounding).
_is_identity(m::EuclideanMap{3}; atol = 1.0e-6) =
    isapprox(SMatrix{3, 3, Float64}(m.R), SMatrix{3, 3, Float64}(LinearAlgebra.I); atol) &&
    isapprox(SVector{3, Float64}(m.t), zero(SVector{3, Float64}); atol)

# ------------------------------------------------------------------------------
# Mounted — a target as a rig member: the target and its mount
# ------------------------------------------------------------------------------

using ..VisualGeometryCore: Uncertain, geom   # an estimated mount (conventions/persisted-geometry.md)

"""
    Mounted(target::AbstractTarget, mount)

A target as a [`Rig`](@ref) member: the TARGET — its geometry, an
[`AbstractTarget`](@ref), independent of where it is mounted, so the same
target can sit in several rigs — and its MOUNT, the rigid map taking
target-local coordinates (the pattern plane `z = 0` for a planar target) into
the rig frame, stored unlabeled (the rig mints `TargetFrame{id} → F` on
access). The mount may be [`Uncertain`](@ref) (an estimated mount, its 6×6
covariance in the `RigidExpManifold6` chart, rotation first); the rig hands it
back uncertain, its covariance unchanged by the labels. `points(m)` are the
target's own points; `pose(m)` is the mount.
"""
struct Mounted{T <: AbstractTarget, X <: Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}}
    target::T
    mount::X
    Mounted(target::T, mount::Union{EuclideanMap{3}, Uncertain{<:EuclideanMap{3}}}) where {T <: AbstractTarget} =
        (m = unlabel(mount); new{T, typeof(m)}(target, m))
end
pose(m::Mounted) = m.mount
points(m::Mounted) = points(m.target)
Base.:(==)(a::Mounted, b::Mounted) = a.target == b.target && a.mount == b.mount

# A stored mount minted with its labels; an uncertain one keeps its covariance.
_mint(::Type{S}, ::Type{D}, m) where {S, D} = relabel(S, D, m)

# ------------------------------------------------------------------------------
# Rig
# ------------------------------------------------------------------------------

"""
    Rig{F, M}

A rigid assembly: members with FIXED poses in a common frame `F`, each a
(model, mount) pair. Three storages, one interface:

- `M <: Tuple{Vararg{AbstractCamera}}` — a physical multi-sensor rig. Sensor
  `i`'s extrinsics is the labeled map `F → ViewFrame{idᵢ}`, stored inside the
  camera (a camera IS its model and its mount); the ids, the reference frame
  and every relative transform are read off those labels — there is no cached
  inverse, no id tuple, no second copy to drift. The tuple makes the sensor
  count static: [`StereoRig`](@ref) IS the 2-tuple case, dispatchable with no
  runtime length check. Many views over time are NOT a rig — they are frames
  of one (see the header comment) — which is why a `Vector` of hundreds of
  cameras is not accepted.
- `M <: AbstractDict{Symbol, <:Mounted}` — a rig of targets: each a
  [`Mounted`](@ref) target keyed by identity, its mount stored unlabeled and
  minted `TargetFrame{id} → F` on access.
- `M <: NamedTuple{(:sensors, :targets)}` — both: a sensor tuple and a target
  dictionary posed in one frame (a camera rig carrying its motion-capture
  marker constellation, on `BodyFrame{:camera_rig}`). Built by
  `Rig{F}(sensors, targets)`; [`sensor_rig`](@ref) and [`target_rig`](@ref)
  are its two halves as rigs of their own. Ids are distinct across the two.

# Type parameters
- `F <: Frame{3}`: the frame every member is posed in. THE FRAME IS THE GAUGE:
  when `F` is a member's own frame — `ViewFrame{id₁}` for a stereo rig built by
  the [`StereoRig`](@ref) constructors, `TargetFrame{uid}` for a calibration
  target gauged on board `uid` — that member's pose is the identity, checked
  at construction. A rig posed in an external frame (a body frame, the world,
  a tracker's room) names that frame instead and has no gauge member.
- `M`: the storage, above.

# Accessors
[`ids`](@ref), `pose(rig, id)` / `extrinsics(rig, id)` (label-minting or
label-preserving per storage; sensors also by position), `rig[id]`,
`length`, `haskey`; [`view_frames`](@ref) / [`image_frames`](@ref) on sensor
rigs, [`target_frames`](@ref) and [`points`](@ref)`(rig, id)` — the target's
points in the rig frame, the WORLD POINTS of the assembly — on target rigs;
the stereo surface ([`source`](@ref), [`target`](@ref), `extrinsics(rig)`,
`pose(rig)`) on the 2-sensor alias.
"""
struct Rig{F <: Frame{3}, M}
    members::M

    # A physical rig: every sensor's extrinsics is labeled F → ViewFrame{id}.
    function Rig{F}(members::M) where {F <: Frame{3}, M <: Tuple{Vararg{AbstractCamera}}}
        _check_sensors(F, members)
        return new{F, M}(members)
    end

    # A rig of targets: Mounted targets keyed by identity, mounts unlabeled.
    # The value type is checked per member, not in the signature: a dictionary
    # mixing target types has a `where`-typed value.
    function Rig{F}(members::M) where {F <: Frame{3}, M <: AbstractDict{Symbol}}
        _check_targets(F, members)
        return new{F, M}(members)
    end

    # Both, in one frame.
    function Rig{F}(sensors::S, targets::D) where {F <: Frame{3}, S <: Tuple{Vararg{AbstractCamera}}, D <: AbstractDict{Symbol}}
        _check_sensors(F, sensors; gauge = !(F <: TargetFrame))
        _check_targets(F, targets; gauge = !(F <: ViewFrame))
        vids = map(c -> _view_id(_raw_extrinsics(c)), sensors)
        isempty(intersect(vids, keys(targets))) ||
            throw(ArgumentError("Rig: a sensor and a target share the id(s) $(collect(intersect(vids, keys(targets))))"))
        m = (sensors = sensors, targets = targets)
        return new{F, typeof(m)}(m)
    end
end

function _check_sensors(::Type{F}, members; gauge::Bool = true) where {F}
    isempty(members) && throw(ArgumentError("Rig: at least one sensor required"))
    for (i, c) in enumerate(members)
        Src, Dst = _em_frames(_raw_extrinsics(c))
        Src === F || throw(
            ArgumentError(
                "Rig: sensor $i extrinsics is labeled $Src → $Dst, but the rig's " *
                    "reference frame is $F — express every sensor in the rig frame " *
                    "(relabel/compose before construction)"
            )
        )
        Dst <: ViewFrame || throw(
            ArgumentError(
                "Rig: sensor $i extrinsics destination $Dst is not a ViewFrame — " *
                    "each sensor's frame must carry a view identity (ViewFrame{id})"
            )
        )
    end
    vids = map(c -> _view_id(_raw_extrinsics(c)), members)
    allunique(vids) || throw(ArgumentError("Rig: sensor view ids must be distinct"))
    gauge && _check_gauge(F, vids, i -> _raw_extrinsics(members[i]))
    return nothing
end

function _check_targets(::Type{F}, members; gauge::Bool = true) where {F}
    isempty(members) && throw(ArgumentError("Rig: at least one target required"))
    for (id, m) in members
        m isa Mounted || throw(
            ArgumentError("Rig: member $(repr(id)) is a $(typeof(m)), not a Mounted target (a target and its mount)")
        )
    end
    gauge && _check_gauge(F, members)
    return nothing
end

# Deduce the reference frame from the first sensor's extrinsics source label.
Rig(members::Tuple{Vararg{AbstractCamera}}) =
    Rig{_em_frames(_raw_extrinsics(members[1]))[1]}(members)

# The gauge invariant. A sensor rig whose frame is one of its views: that
# view's stored extrinsics is the identity (a rig may also be posed in a view
# frame it does not contain — nothing to check then).
function _check_gauge(::Type{ViewFrame{g}}, vids, raw) where {g}
    i = findfirst(==(g), vids)
    i === nothing && return nothing
    _is_identity(raw(i)) || throw(
        ArgumentError(
            "Rig: the rig frame is ViewFrame{$(repr(g))}, so sensor $(repr(g)) is the gauge and " *
                "its extrinsics must be the identity — express the rig relative to that view"
        )
    )
    return nothing
end
_check_gauge(::Type{<:Frame{3}}, vids, raw) = nothing
# A target rig whose frame is a target's: that target is a member, at the identity.
function _check_gauge(::Type{TargetFrame{g}}, members::AbstractDict{Symbol}) where {g}
    haskey(members, g) || throw(
        ArgumentError(
            "Rig: the rig frame is TargetFrame{$(repr(g))}, but there is no target $(repr(g)) — " *
                "a target rig gauged on a target contains it"
        )
    )
    _is_identity(geom(pose(members[g]))) || throw(
        ArgumentError(
            "Rig: the rig frame is TargetFrame{$(repr(g))}, so target $(repr(g)) is the gauge and " *
                "its pose must be the identity — express every pose relative to that target"
        )
    )
    return nothing
end
_check_gauge(::Type{<:Frame{3}}, ::AbstractDict{Symbol}) = nothing

# The mixed storage's halves, and the dispatch tags of the three storages.
const _SensorStorage = Tuple{Vararg{AbstractCamera}}
const _TargetStorage = AbstractDict{Symbol}
const _MixedStorage = NamedTuple{(:sensors, :targets)}
_sensors(rig::Rig{F, <:_SensorStorage}) where {F} = rig.members
_sensors(rig::Rig{F, <:_MixedStorage}) where {F} = rig.members.sensors
_targets(rig::Rig{F, <:_TargetStorage}) where {F} = rig.members
_targets(rig::Rig{F, <:_MixedStorage}) where {F} = rig.members.targets

"""
    sensor_rig(rig::Rig) -> Rig{F, <:Tuple}

The sensor half of a rig carrying both sensors and targets, as a rig of its
own in the same frame (the identity on a sensors-only rig). The target half is
[`target_rig`](@ref).
"""
sensor_rig(rig::Rig{F, <:_SensorStorage}) where {F} = rig
sensor_rig(rig::Rig{F, <:_MixedStorage}) where {F} = Rig{F}(rig.members.sensors)

"""
    target_rig(rig::Rig) -> Rig{F, <:AbstractDict}

The target half of a rig carrying both sensors and targets, as a rig of its
own in the same frame (the identity on a targets-only rig). The sensor half is
[`sensor_rig`](@ref).
"""
target_rig(rig::Rig{F, <:_TargetStorage}) where {F} = rig
target_rig(rig::Rig{F, <:_MixedStorage}) where {F} = Rig{F}(rig.members.targets)

"""
    reference_frame(rig::Rig{F}) -> F

The rig's frame type — the frame every member is posed in (`F → ViewFrame{idᵢ}`
for sensors, `TargetFrame{id} → F` for targets). Unexported; resolved from
the type.
"""
reference_frame(::Rig{F}) where {F} = F

"""
    ids(rig::Rig) -> NTuple{N, Symbol} | Vector{Symbol}

The member identities: a sensor rig's view-identity symbols, read off the
frame parameters of each stored sensor's labeled extrinsics (`F → ViewFrame{id}`)
— an accessor, not a field, resolved at compile time from the rig's type; a
target rig's keys, sorted for determinism; a rig of both, the sensors' then
the targets'.
"""
ids(rig::Rig{F, <:Tuple}) where {F} = _sensor_ids(rig)
ids(rig::Rig{F, <:AbstractDict}) where {F} = block_order(rig.members)
ids(rig::Rig{F, <:_MixedStorage}) where {F} = Symbol[_sensor_ids(rig)..., block_order(_targets(rig))...]
_sensor_ids(rig::Rig) = map(c -> _view_id(_raw_extrinsics(c)), _sensors(rig))

"""
    view_frames(rig::Rig) -> (ViewFrame{id1}, ViewFrame{id2}, …)

The labeled 3D per-view frame TAGS of a sensor rig's sensors — `ViewFrame{id}`
types minted from [`ids`](@ref)`(rig)`, ready to be attached at a lift
(e.g. `relabel(ViewFrame{:left}, WorldFrame, pose)`). See
[`ViewFrame`](@ref) for the instance-identity design.
"""
view_frames(rig::Rig{F, <:Union{_SensorStorage, _MixedStorage}}) where {F} = map(id -> ViewFrame{id}, _sensor_ids(rig))

"""
    image_frames(rig::Rig) -> (ImageFrame{id1}, ImageFrame{id2}, …)

The labeled 2D per-view image-frame TAGS of a sensor rig's sensors —
`ImageFrame{id}` types minted from [`ids`](@ref)`(rig)`, the slot labels for
epipolar forms over this rig. See [`ImageFrame`](@ref) for the
convention-vs-view division of labor.
"""
image_frames(rig::Rig{F, <:Union{_SensorStorage, _MixedStorage}}) where {F} = map(id -> ImageFrame{id}, _sensor_ids(rig))

"""
    target_frames(rig::Rig) -> Vector{DataType}

The labeled 3D per-target frame TAGS of a target rig — `TargetFrame{id}` types
minted from [`ids`](@ref)`(rig)`.
"""
target_frames(rig::Rig{F, <:Union{_TargetStorage, _MixedStorage}}) where {F} = [TargetFrame{id} for id in block_order(_targets(rig))]

Base.length(rig::Rig) = length(rig.members)
Base.length(rig::Rig{F, <:_MixedStorage}) where {F <: Frame{3}} = length(_sensors(rig)) + length(_targets(rig))
Base.haskey(rig::Rig{F, <:AbstractDict}, id::Symbol) where {F} = haskey(rig.members, id)
Base.haskey(rig::Rig{F, <:Tuple}, id::Symbol) where {F} = id in ids(rig)
Base.haskey(rig::Rig{F, <:_MixedStorage}, id::Symbol) where {F} = haskey(_targets(rig), id) || id in _sensor_ids(rig)
_is_target(rig::Rig{F, <:_MixedStorage}, id::Symbol) where {F} = haskey(_targets(rig), id)

# The position of sensor `id` in a sensor rig.
function _sensor_index(rig::Rig, id::Symbol)
    i = findfirst(==(id), _sensor_ids(rig))
    i === nothing && throw(KeyError(id))
    return i
end

"""
    rig[id]   (rig[i] for a sensor rig)

The member itself: a sensor (`Camera`/`BoundedCamera`, by view id or by
position) or a [`Mounted`](@ref) target (by identity).
"""
Base.getindex(rig::Rig{F, <:AbstractDict}, id::Symbol) where {F} = rig.members[id]
Base.getindex(rig::Rig{F, <:Tuple}, id::Symbol) where {F} = rig.members[_sensor_index(rig, id)]
Base.getindex(rig::Rig{F, <:Tuple}, i::Integer) where {F} = rig.members[i]
Base.getindex(rig::Rig{F, <:_MixedStorage}, id::Symbol) where {F} =
    _is_target(rig, id) ? _targets(rig)[id] : _sensors(rig)[_sensor_index(rig, id)]
Base.getindex(rig::Rig{F, <:_MixedStorage}, i::Integer) where {F} = _sensors(rig)[i]

"""
    extrinsics(rig::Rig, i::Integer | id::Symbol) -> EuclideanMap

Sensor `i`'s (or view `id`'s) stored rig-relative extrinsics, label-preserving:
the map `F → ViewFrame{idᵢ}` (NOT the `WorldFrame → CameraFrame` relabel that
`extrinsics(::Camera)` applies — a rig sensor's frame is the rig's, and the
labels are kept). For a target rig, `F → TargetFrame{id}` — common-frame
coordinates into target-local coordinates, `inv(pose(rig, id))` on demand.
"""
extrinsics(rig::Rig{F, <:Tuple}, i::Integer) where {F} = _raw_extrinsics(rig.members[i])
extrinsics(rig::Rig{F, <:Tuple}, id::Symbol) where {F} = extrinsics(rig, _sensor_index(rig, id))
extrinsics(rig::Rig{F, <:AbstractDict}, id::Symbol) where {F} = _inv_member(pose(rig, id))
extrinsics(rig::Rig{F, <:_MixedStorage}, i::Integer) where {F} = _raw_extrinsics(_sensors(rig)[i])
extrinsics(rig::Rig{F, <:_MixedStorage}, id::Symbol) where {F} =
    _is_target(rig, id) ? _inv_member(pose(rig, id)) : extrinsics(rig, _sensor_index(rig, id))
_inv_member(m::EuclideanMap{3}) = inv(m)
_inv_member(::Uncertain) = throw(
    ArgumentError(
        "extrinsics of an uncertain target: a covariance does not invert with its pose by relabeling — " *
            "invert `geom(pose(rig, id))`, or propagate the covariance explicitly"
    )
)

"""
    pose(rig::Rig, i::Integer | id::Symbol) -> EuclideanMap

A member's pose in the rig frame. For a sensor, `inv(extrinsics(rig, i))`
computed on demand (`ViewFrame{idᵢ} → F` — the typed algebra swaps the
labels); for a target, the stored map minted with its labels,
`TargetFrame{id} → F` (target-local coordinates into the rig frame).
"""
pose(rig::Rig{F, <:Tuple}, i::Integer) where {F} = inv(extrinsics(rig, i))
pose(rig::Rig{F, <:Tuple}, id::Symbol) where {F} = pose(rig, _sensor_index(rig, id))
pose(rig::Rig{F, <:AbstractDict}, id::Symbol) where {F} =
    _mint(TargetFrame{id}, F, pose(rig.members[id]))
pose(rig::Rig{F, <:_MixedStorage}, i::Integer) where {F} = inv(extrinsics(rig, i))
pose(rig::Rig{F, <:_MixedStorage}, id::Symbol) where {F} =
    _is_target(rig, id) ? _mint(TargetFrame{id}, F, pose(_targets(rig)[id])) : pose(rig, _sensor_index(rig, id))

# A target's own point lifted to 3D: a board point sits on its plane z = 0.
_lift3(p::StaticVector{2}) = SVector{3, Float64}(p[1], p[2], 0.0)
_lift3(p::StaticVector{3}) = SVector{3, Float64}(p[1], p[2], p[3])

"""
    points(rig::Rig, id::Symbol) -> Vector{SVector{3, Float64}}
    points(rig::Rig) -> Dict{Symbol, Vector{SVector{3, Float64}}}

The WORLD POINTS of a calibration target: target `id`'s pattern points
carried through its pose into the rig frame (a board's plane points lifted
to `z = 0` first) — the 3D side of a target ↔ image correspondence, what an
absolute-pose problem consumes. The one-argument form covers every target
that carries points; an unresolved one (a design reference without its points,
[`is_resolved`](@ref)) is an `ArgumentError` on its own and absent from the
dictionary.
"""
function points(rig::Rig{F, <:Union{_TargetStorage, _MixedStorage}}, id::Symbol) where {F}
    m = _targets(rig)[id]
    is_resolved(m.target) || throw(
        ArgumentError("points: target $(repr(id)) is unresolved — a design reference without its points")
    )
    P = geom(pose(rig, id))
    return [SVector{3, Float64}(P(_lift3(geom(p)))) for p in points(m)]
end
points(rig::Rig{F, <:Union{_TargetStorage, _MixedStorage}}) where {F} = Dict{Symbol, Vector{SVector{3, Float64}}}(
    id => points(rig, id) for id in block_order(_targets(rig)) if is_resolved(_targets(rig)[id].target)
)

# ------------------------------------------------------------------------------
# with_model — same geometry, new camera model
# ------------------------------------------------------------------------------

"""
    with_model(camera, model) -> Camera/BoundedCamera
    with_model(rig::Rig, i => model) -> Rig
    with_model(rig::Rig, model) -> Rig            (single-sensor convenience)

Rebuild with sensor `i`'s camera MODEL replaced — after a `fit_lens`
regression or a generic-model init — keeping everything geometric: the
labeled rig-relative extrinsics, the ids, any `BoundedCamera` extent. The
rig is reconstructed through its validating constructor (never a raw field
update, which would bypass the frame invariants); a lens/K change is
metric-neutral to the rigid transforms, so the swap is geometrically exact.

NOTE: swapping the model does NOT re-estimate poses. Poses optimized under
the old model keep their values; for an APPROXIMATING model (the fit_lens
use case) that is the intent — the approximation error is the fit report's
— while for a refinement the new model is the START of another bundle
run, not a substitute for one.
"""
with_model(c::Camera, model::AbstractCameraModel) = Camera(model, c.extrinsics)
with_model(::AbstractCameraModel, model::AbstractCameraModel) = model   # an unposed camera is its model
with_model(c::BoundedCamera, model::AbstractCameraModel) =
    BoundedCamera(with_model(c.camera, model), c.extent)

function with_model(rig::Rig{F, <:Tuple}, im::Pair{<:Integer, <:AbstractCameraModel}) where {F}
    i, m = im
    1 <= i <= length(rig.members) || throw(
        ArgumentError(
            "with_model: sensor index $i out of range for a $(length(rig.members))-sensor rig"
        )
    )
    cams = ntuple(length(rig.members)) do j
        j == i ? with_model(rig.members[j], m) : rig.members[j]
    end
    return Rig{F}(cams)
end
function with_model(rig::Rig{F, <:Tuple}, model::AbstractCameraModel) where {F}
    length(rig.members) == 1 || throw(
        ArgumentError(
            "with_model: the rig has $(length(rig.members)) sensors — supply `i => model`"
        )
    )
    return with_model(rig, 1 => model)
end
