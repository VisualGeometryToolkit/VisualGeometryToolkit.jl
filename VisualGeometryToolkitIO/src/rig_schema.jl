# =============================================================================
# vgio/rig/v4 and vgio/target/v1 — VGC's `Rig` in every storage (sensors,
# targets, both) and VGC's target model, serialized
# =============================================================================
#
# The layouts are stated beside their code below (a target's body, the target
# kind, the rig kind). Throughout:
#
# <pose declarations> = "length_unit", "rotation_encoding" and
# "transform_convention" (transform_schema.jl), stated once per file; every
# file may add a free-form "provenance" object that VGIo never reads.
#
# A rig of targets is the DEFINITION the per-frame measurements of
# target_points2d/v1 and target_points3d/v1 (captures_schema.jl) resolve
# against: a point is named by (target, index), its `point_index_base`-based
# position in that target's points, and a measurement file names the rig files
# it resolves in by their content identity ([`content_id`](@ref)).
#
# ONE AUTHORITY PER NAME: a pose's frames are given by where it sits — a
# sensor's by the rig `frame` and its `id`, a target's mount by its `id` and
# the rig `frame` — and are never restated beside it, so they cannot disagree
# with it. FRAME NAMES go through the frame grammar (frame_names.jl).
#
# UNCERTAINTY (VisualGeometryToolkitCore conventions/persisted-geometry.md): a mount
# may carry a 6×6 `covariance`, a point target's points a parallel
# `point_covariances` (one d×d per point, `length_unit`²) — each present iff
# the in-memory value is `Uncertain`. A rig sensor's extrinsics are a VGC
# `Camera`'s map, never uncertain.
#
# UNITS: a stored rig is PHYSICAL — its mounts, extrinsics and target points
# are lengths (metres in memory, the declared `length_unit` on disk); a writer
# refuses plain numbers where a length is meant.
#
# RIG SENSORS MUST BE BoundedCamera. A camera body requires the pixel extent,
# which a bare Camera does not carry — wrap with `BoundedCamera(camera,
# extent)` before saving. Refused loudly, not defaulted.

_unique_check(xs, what) = allunique(xs) ? nothing : "$(what) must be unique"

# Read the schema a file declares, before knowing which kind to validate it as.
function _peek_schema(path::AbstractString)
    d = _read_vgio_json(path)
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    return get(d, :schema, nothing)
end

# -----------------------------------------------------------------------------
# A target <-> JSON: the body every target-carrying kind embeds
# -----------------------------------------------------------------------------
#
#   {"type": "points", "points": [[x, y(, z)], …], "point_covariances": [[[…]…]…]}   PointTarget
#   {"type": "planar", "design": "<reference>", "warp": {"type": "rigid"}}            PlanarTarget of a referenced design
#   {"type": "planar", "nominal": [[x, y], …], "warp": {"type": "rigid"}}             PlanarTarget of explicit points
#
# Points and nominal points in the file's `length_unit` (written in mm unless
# the writer is given another), lengths in metres in memory, in the
# target's own frame. A referenced design's nominal points are its owner's —
# a BlobBoards board uid resolves to them — so they are NOT restated: `load`
# returns the target unresolved unless the caller supplies `resolve`, the
# design owner's resolver. The warp's record carries its `parameters` (and their
# `covariance`, `Uncertain`) for a warp that has any; `rigid` has none.

const _TARGET_POINTS = _Checked(
    _Vec(_Vec(_Real())),
    p -> (isempty(p) || (length(first(p)) in (2, 3) && all(v -> length(v) == length(first(p)), p))) ? nothing :
        "points must all be 2-vectors (a plane's) or all 3-vectors",
)
_point_covariances_check(pts, cs) = begin
    length(cs) == length(pts) || return "point_covariances must have one entry per point"
    d = isempty(pts) ? 0 : length(first(pts))
    for c in cs
        (length(c) == d && all(r -> length(r) == d, c)) || return "each point covariance is $(d)×$(d), the points' dimension"
        all(c[i][j] == c[j][i] for i in 1:d, j in 1:d) || return "a point covariance must be symmetric"
    end
    nothing
end
const _WARP = _Obj(Pair{Symbol, _Spec}[:type => _Tok("rigid")], Pair{Symbol, _Spec}[])
const _TARGET_BODY = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[:type => _NonEmpty()],
        Pair{Symbol, _Spec}[
            :points => _TARGET_POINTS, :point_covariances => _Vec(_Vec(_Vec(_Real()))),
            :design => _NonEmpty(), :nominal => _Vec(_Vec(_Real(), 2)), :warp => _WARP,
        ],
    ),
    function (t)
        ty = t[:type]
        if ty == "points"
            haskey(t, :points) || return "a points target states its `points`"
            any(k -> haskey(t, k), (:design, :nominal, :warp)) && return "a points target has `points` (and `point_covariances`) only"
            haskey(t, :point_covariances) && return _point_covariances_check(t[:points], t[:point_covariances])
            return nothing
        elseif ty == "planar"
            any(k -> haskey(t, k), (:points, :point_covariances)) && return "a planar target has `design` or `nominal`, and `warp`"
            haskey(t, :warp) || return "a planar target states its `warp`"
            (haskey(t, :design) ⊻ haskey(t, :nominal)) ||
                return "a planar target states its `design` reference or its `nominal` points — one, not both (a referenced design's points are its owner's)"
            return nothing
        end
        return "unknown target type $(repr(ty)) (points, planar)"
    end,
)

# A target's points on the wire: the geometric values, and their covariances
# iff every point is `Uncertain` (plain and uncertain points are not mixed in
# one target — the file could not say which is which).
function _points_json(what, pts, unit::Units)
    vals = [_lengths_json(geom(p), unit, "$(what): a point") for p in pts]
    n_u = count(p -> p isa Uncertain, pts)
    n_u == 0 && return (points = vals,)
    n_u == length(pts) || throw(
        ArgumentError("$(what): its points are partly Uncertain — make all of them Uncertain (exact where known) or none")
    )
    return (points = vals, point_covariances = [_length_cov_json(cov(p), unit, _point_cov_dims(length(geom(p)))) for p in pts])
end

# `points` back to static vectors of lengths (metres) of the recorded
# dimension, the form VGC composes a target's points with its mount in
# (`points(rig, id)`); `Uncertain` iff the file states their covariances
# (unit² → m²).
function _points_from_json(vals, unit::Units, covs = nothing)
    isempty(vals) && return SVector{2, Float64}[]
    n = length(first(vals)::Vector{Any})
    pts = [_lengths_from_json(Val(n), v::Vector{Any}, unit) for v in vals]
    covs === nothing && return pts
    return [Uncertain(p, CovMat(_length_cov_from_json(c, n, unit, _point_cov_dims(n)))) for (p, c) in zip(pts, covs)]
end

_warp_json(::Rigid) = (type = "rigid",)
_warp_from_json(w) = w.type == "rigid" ? Rigid() : throw(ArgumentError("unknown warp $(repr(w.type))"))

"""
    target_json(target, unit = mm) -> NamedTuple
    target_from_json(body, unit; resolve = nothing) -> AbstractTarget

The target body every target-carrying kind embeds (see rig_schema.jl), its
points (lengths) written in the length `unit`, and back from a file in length
`unit` (a unit name), as lengths in metres. A target of plain points is an
`ArgumentError` (attach the unit, `p .* m`). A `PlanarTarget` of a referenced
design is written by its reference alone and read back unresolved, unless
`resolve(design::Symbol)` — the design owner's resolver — returns its nominal
points (lengths). Building blocks for a package serializing its own kind.
"""
target_json(t::PointTarget, unit::Units = mm) = merge((type = "points",), _points_json("a points target", points(t), unit))
target_json(t::PlanarTarget, unit::Units = mm) = t.design === nothing ?
    (type = "planar", nominal = [_lengths_json(p, unit, "a planar target's nominal point") for p in t.nominal], warp = _warp_json(t.warp)) :
    (type = "planar", design = String(t.design), warp = _warp_json(t.warp))

"""
    target_from_json(body, unit; resolve = nothing) -> AbstractTarget

Read a target body (the inverse of [`target_json`](@ref)) whose lengths are in
`unit`, a unit name such as `"mm"`, as lengths in metres. A planar target that
names its design is returned unresolved unless `resolve(design::Symbol)`
returns its nominal points — lengths, as the design's owner holds them; plain
points are an `ArgumentError`.
"""
function target_from_json(t, unit::AbstractString; resolve = nothing)
    u = parse_unit(unit)
    if t.type == "points"
        return PointTarget(_points_from_json(t.points::Vector{Any}, u, haskey(t, :point_covariances) ? t.point_covariances::Vector{Any} : nothing))
    end
    warp = _warp_from_json(t.warp)
    haskey(t, :nominal) && return PlanarTarget(_points_from_json(t.nominal::Vector{Any}, u); warp)
    design = Symbol(t.design::String)
    nominal = resolve === nothing ? nothing : resolve(design)
    (nominal === nothing || isempty(nominal) || has_units(nominal)) ||
        _refuse_plain("resolve($(repr(design))): the design's nominal points")
    return PlanarTarget(design; nominal, warp)
end

# -----------------------------------------------------------------------------
# vgio/target/v1 ↔ one target (AbstractTarget, or one Attributed with its calibration block)
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/target/v1", "length_unit": "mm", "target": {target body},
#  "calibration": {…}}                                              (optional: the cameras' block)

const _TARGET_SPEC = _Obj(
    Pair{Symbol, _Spec}[:schema => _schema_tok("target"), :length_unit => _LENGTH_UNIT, :target => _TARGET_BODY],
    Pair{Symbol, _Spec}[:calibration => _CALIBRATION, _PROVENANCE...],
)

"""
    save(path, target::AbstractTarget; length_unit = mm, provenance = (;))
    save(path, Attributed(target, (; calibration)); length_unit = mm, provenance = (;))

Write one physical target as a `vgio/target/v1` file — its body (rig_schema.jl)
in `length_unit` and, for an `Attributed` one, its `calibration` block (the
same block a camera carries: who fitted it, when, from what). Plain points are
an `ArgumentError`.
"""
save(path::AbstractString, t::AbstractTarget; length_unit::Units = mm, provenance::NamedTuple = (;)) = (
    _write_store(
        path, _TARGET_SPEC,
        _with_provenance((schema = _schema("target"), length_unit = unit_string(length_unit), target = target_json(t, length_unit)), provenance),
    ); nothing
)
function save(
        path::AbstractString, t::Attributed{<:AbstractTarget, <:NamedTuple{(:calibration,)}};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    body = (
        schema = _schema("target"), length_unit = unit_string(length_unit), target = target_json(t.value, length_unit),
        calibration = _calibration_json(t.calibration),
    )
    _write_store(path, _TARGET_SPEC, _with_provenance(body, provenance))
    return nothing
end

"""
    load(AbstractTarget, path; resolve = nothing) -> AbstractTarget | Attributed{<:AbstractTarget}

Read a `vgio/target/v1` file: the target, `Attributed` with its `calibration`
block where the file states one. `resolve(design)` supplies a referenced
design's nominal points (else the target is returned unresolved).
"""
function load(::Type{AbstractTarget}, path::AbstractString; resolve = nothing)
    d = _read_store(path, "target", _TARGET_SPEC)
    t = target_from_json(d.target, d.length_unit::String; resolve)
    return haskey(d, :calibration) ? Attributed(t, (; calibration = _calibration_from_json(d.calibration::AbstractDict))) : t
end

# -----------------------------------------------------------------------------
# vgio/rig/v4 ↔ Rig, every storage
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/rig/v4", "pixel_frame": "colmap", <pose declarations>, "frame": "body:camera_rig",
#  "sensors": [{"id": "cam1", "camera": {camera body | generic body},
#               "extrinsics": {rotation, translation}}],               extrinsics: `frame` → view:<id>
#  "targets": [{"id": "<uid>", "target": {target body},
#               "pose": {rotation, translation[, covariance]}}]}       pose (the mount): target:<id> → `frame`
#
# A sensor rig has `sensors` only, a rig of targets `targets` only (a board
# rig, a motion-capture body's constellation, the boards of a calibration
# target gauged on one of them), a camera rig carrying its constellation both;
# `pixel_frame` is declared iff there are sensors. Ids are unique across the
# two. A sensor is a `BoundedCamera` (a camera body requires the pixel extent);
# its extrinsics are a plain-typed `EuclideanMap` (never `Uncertain`); a mount
# may be `Uncertain`. Every translation and point is a length.

const _SENSOR_RECORD = _Obj(
    :id => _NonEmpty(),
    :camera => _Either(Pair{Symbol, _Spec}[:lens => _CAMERA_BODY, :type => _GENERIC_BODY]),
    :extrinsics => _POSE,
)
const _MEMBER_RECORD = _Obj(Pair{Symbol, _Spec}[:id => _NonEmpty(), :target => _TARGET_BODY, :pose => _UPOSE], Pair{Symbol, _Spec}[])
const _RIG_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[:schema => _schema_tok("rig"), _POSE_DECLARATIONS..., :frame => _FRAME_NAME],
        Pair{Symbol, _Spec}[
            :pixel_frame => _PIXEL_FRAME, :sensors => _Vec(_SENSOR_RECORD), :targets => _Vec(_MEMBER_RECORD), _PROVENANCE...,
        ],
    ),
    function (x)
        ss = get(x, :sensors, Any[])::AbstractVector
        ts = get(x, :targets, Any[])::AbstractVector
        (haskey(x, :sensors) && isempty(ss)) && return "`sensors` is omitted, not empty"
        (haskey(x, :targets) && isempty(ts)) && return "`targets` is omitted, not empty"
        (isempty(ss) && isempty(ts)) && return "a rig has sensors, targets, or both"
        haskey(x, :sensors) == haskey(x, :pixel_frame) || return "`pixel_frame` is declared iff the rig has sensors"
        msg = _unique_check(vcat([s[:id] for s in ss], [t[:id] for t in ts]), "member ids")
        msg === nothing || return msg
        # A generic model is defined in the canonical pixel frame; a rig
        # declaring another could not hold one.
        (isempty(ss) || x[:pixel_frame] == pixel_frame_name(CANONICAL_PIXEL_FRAME) ||
            !any(s -> haskey((s::AbstractDict)[:camera]::AbstractDict, :type), ss)) ||
            return "a generic-model sensor is in the $(pixel_frame_name(CANONICAL_PIXEL_FRAME)) pixel frame; this rig declares $(x[:pixel_frame])"
        return nothing
    end,
)

function _sensor_json(rig, i, id, unit::Units)
    c = sensor_rig(rig).members[i]
    m = model(c)
    body = if m isa CameraModel
        c isa BoundedCamera || throw(
            ArgumentError("rig sensor $i ($(id)) carries no pixel extent — wrap it in BoundedCamera(camera, extent) before saving")
        )
        _camera_body(m, extent(c))
    else
        _generic_body(m, unit)      # generic grid model: carries its own extent
    end
    return (id = String(id), camera = body, extrinsics = pose_json(extrinsics(sensor_rig(rig), i), unit))   # <rig frame> → view:<id>
end

_has_sensors(::Rig{F, <:AbstractDict}) where {F <: Frame{3}} = false
_has_sensors(::Rig) = true
_has_targets(::Rig{F, <:Tuple}) where {F <: Frame{3}} = false
_has_targets(::Rig) = true

# The rig record without provenance — what the file holds and what its content
# identity covers. Lengths in `unit` (millimetres unless the writer is given
# another; the identity of a `Rig` value is that of its millimetre record).
function _rig_body(rig::Rig, unit::Units = mm)
    body = merge(
        (schema = _schema("rig"),),
        _has_sensors(rig) ? (pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME),) : (;),
        pose_declarations(unit), (frame = frame_name(reference_frame(rig)),),
    )
    if _has_sensors(rig)
        sr = sensor_rig(rig)
        body = merge(body, (sensors = [_sensor_json(rig, i, id, unit) for (i, id) in enumerate(ids(sr))],))
    end
    if _has_targets(rig)
        tr = target_rig(rig)
        targets = [(id = String(id), target = target_json(tr[id].target, unit), pose = pose_json(pose(tr, id), unit)) for id in ids(tr)]   # target:<id> → <rig frame>
        body = merge(body, (targets = targets,))
    end
    return body
end

"""
    save(path, rig::Rig; length_unit = mm, provenance = (;))

Write a physical VGC `Rig` — sensors, targets, or both — as a `vgio/rig/v4`
file in `length_unit`: each sensor (a `BoundedCamera`) with its extrinsics
`frame → view:<id>`, each target (sorted by id) with its body and its mount
`target:<id> → frame` (a `covariance` iff `Uncertain`). A rig of plain
translations or points is an `ArgumentError` (attach the unit, `t .* m`). A
rig's name, if it has one worth keeping, belongs in `provenance`.
"""
save(path::AbstractString, rig::Rig; length_unit::Units = mm, provenance::NamedTuple = (;)) =
    (_write_store(path, _RIG_SPEC, _with_provenance(_rig_body(rig, length_unit), provenance)); nothing)

"""
    load(Rig, path; resolve = nothing) -> Rig

Read a `vgio/rig/v4` file back into a fully labeled VGC `Rig` of the storage
it holds, its lengths in metres: sensors (each a `BoundedCamera` labeled `frame → ViewFrame{id}`),
targets (each `Mounted`, minted `TargetFrame{id} → frame` on access), or both.
`resolve(design)` supplies a referenced planar design's nominal points (see
[`target_from_json`](@ref)). Type-unstable by nature (frame types are minted
from file data) — load once, dispatch after.
"""
load(::Type{Rig}, path::AbstractString; resolve = nothing) = _rig_from_record(_read_store(path, "rig", _RIG_SPEC); resolve)

function _rig_from_record(d; resolve = nothing)
    F = frame_from_name(d.frame::String)
    unit = d.length_unit::String
    haskey(d, :targets) || return Rig{F}((_load_sensors(d, F, unit)...,))
    members = Dict{Symbol, Mounted}(
        Symbol(t.id::String) => Mounted(target_from_json(t.target, unit; resolve), pose_from_json(t.pose, unit))
            for t in d.targets::Vector{Any}
    )
    targets = Dict{Symbol, typejoin(map(typeof, values(members))...)}(members)
    haskey(d, :sensors) || return Rig{F}(targets)
    return Rig{F}((_load_sensors(d, F, unit)...,), targets)
end

function _load_sensors(d, ::Type{F}, unit::String) where {F}
    pframe = pixel_frame_from_name(d.pixel_frame::String)
    return map(d.sensors::Vector{Any}) do s
        Xl = relabel(F, ViewFrame{Symbol(s.id::String)}, pose_from_json(s.extrinsics, unit))
        if haskey(s.camera, :type)
            gm = _generic_from_body(s.camera, parse_unit(unit))
            BoundedCamera(Camera(gm, Xl), Size2(gm.width, gm.height))
        else
            cm, ext = _camera_from_body(s.camera, pframe)
            BoundedCamera(Camera(cm, Xl), ext)
        end
    end
end

"""
    content_id(targets) -> String

The identity a measurement file (`target_points2d`, `target_points3d`) names a
target definition by — `targets` a `Rig` or a `vgio/rig/v4` path: the SHA-256
of the canonical JSON of its rig record WITHOUT its provenance. A rig and the file written from it (in the
default millimetres) have the same identity, as do a file and the rig loaded from it (a path is hashed as
the record its loaded rig writes, in the file's `length_unit`, so a full-precision number that does not
survive into metres does not split the two); a relabeled provenance does not change it, a file written in
another `length_unit` has its own. A measurement value's `targets` holds
these.
"""
content_id(x::Union{Rig, AbstractString}) = _targets_hash(x)

# The identity observations name a targets file by: the SHA-256 of the
# canonical JSON (canonical.jl) of the record WITHOUT its provenance, so a
# relabeled provenance does not change which targets a file resolves against.
# The same for a file on disk and for the rig it was written from, and for a
# file and the rig loaded from it: a path is hashed as the record its loaded
# rig writes, in the file's own `length_unit`. A file's numbers are lengths read
# into metres; a 16–17 digit one need not come back (the decimal-to-metre map
# is not injective), so hashing the file's own bytes would give a loaded rig
# another identity than its file. A number of ≤ 15 significant digits comes
# back byte for byte, so such a file keeps the identity of its bytes.
function _targets_hash(path::AbstractString)
    d = _read_store(path, "rig", _RIG_SPEC)
    rig = _rig_from_record(d)                 # a referenced design stays a reference
    return _hash_record(_parse_json(_json(_rig_body(rig, parse_unit(d.length_unit::String)))))
end
_targets_hash(rig::Rig) = _hash_record(_parse_json(_json(_rig_body(rig))))
_hash_record(d) = sha256_hex(canonical_json(_lower(d; drop = :provenance)))

_lower(x::AbstractDict; drop::Union{Nothing, Symbol} = nothing) =
    Dict{String, Any}(String(k) => _lower(v) for (k, v) in x if Symbol(k) !== drop)
_lower(x::AbstractVector; drop = nothing) = Any[_lower(v) for v in x]
_lower(x; drop = nothing) = x

const _HEX64 = _Named(s -> length(s) == 64 && all(c -> c in "0123456789abcdef", s), "a SHA-256 hex digest")

# A targets argument — a Rig, a targets file, or several (one per connected
# component) — as a list, its loaded rigs, and its content hashes. A PATH is
# hashed as the record its loaded rig writes (in the file's unit), a Rig as the
# record `save` writes for it: the two agree for a file written from that rig
# and for a rig loaded from that file.
_target_list(t::Union{Rig, AbstractString}) = Any[t]
_target_list(ts::Union{AbstractVector, Tuple}) = collect(Any, ts)
_target_rigs(ts; resolve = nothing) = [_target_rig(t; resolve) for t in _target_list(ts)]
_target_rig(r::Rig; resolve = nothing) = target_rig(r)
_target_rig(p::AbstractString; resolve = nothing) = target_rig(load(Rig, p; resolve))
_target_hashes(ts) = String[_targets_hash(t) for t in _target_list(ts)]

# Every (target, index) must resolve in the rigs the file will name — checked
# at write time, where the rigs are in hand.
function _check_resolves(observations, rigs)
    seen = Set(Symbol(o.target) for (_, os) in observations for o in os)
    npts = Dict{Symbol, Int}()
    for rig in rigs, id in ids(rig)
        id in seen || continue
        t = rig[id].target
        is_resolved(t) || throw(
            ArgumentError("target $(id) is unresolved (a design reference without its points) — supply `resolve`, its design owner's resolver")
        )
        npts[id] = length(points(t))
    end
    for (t, os) in observations, o in os
        n = get(npts, Symbol(o.target), 0)
        1 <= o.index <= n || throw(
            ArgumentError(
                "measurement (target $(o.target), index $(o.index)) in frame $(t) does not resolve in the " *
                    "targets given ($(n == 0 ? "no target $(o.target) with points" : "it has $(n) points"))"
            )
        )
    end
    return nothing
end

# A per-frame name (a sensor, an image): one for every frame, or a map that must
# cover every frame — what the v1 migration requires of its caller.
_per_frame(x::AbstractString, frames, what) = Dict(t => String(x) for t in frames)
function _per_frame(x::AbstractDict, frames, what)
    missing_ = [t for t in frames if !haskey(x, t)]
    isempty(missing_) || throw(ArgumentError("`$(what)` names no value for frame(s) $(missing_)"))
    return Dict(t => String(x[t]) for t in frames)
end
