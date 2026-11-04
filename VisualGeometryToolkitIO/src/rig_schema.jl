# =============================================================================
# vgio/rig/v6 and vgio/target/v3 — VGC's `Rig` in every storage (sensors,
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
# sensor's by the rig `frame` and its `id`, a target's pose by its `id` and
# the rig `frame` — and are never restated beside it, so they cannot disagree
# with it. FRAME NAMES go through the frame grammar (frame_names.jl).
#
# UNCERTAINTY (VisualGeometryToolkitCore conventions/persisted-geometry.md): a pose
# may carry a 6×6 `covariance`, a point target's points a parallel
# `point_covariances` (one d×d per point, `length_unit`²) — each present iff
# the in-memory value is `Uncertain`. A rig sensor's extrinsics are a VGC
# `PosedCamera`'s map, never uncertain.
#
# UNITS: a stored rig is PHYSICAL — its target poses, extrinsics and target points
# are lengths (metres in memory, the declared `length_unit` on disk); a writer
# refuses plain numbers where a length is meant.
#
# RIG SENSORS MUST BE BoundedCamera. A camera body requires the pixel extent,
# which a bare PosedCamera does not carry — wrap with `BoundedCamera(camera,
# extent)` before saving. Refused loudly, not defaulted.

_unique_check(xs, what) = allunique(xs) ? nothing : "$(what) must be unique"

# Read the schema a file declares, before knowing which kind to validate it as.
function _peek_schema(path::AbstractString)
    endswith(lowercase(path), ".toml") && return get(TOML.parsefile(path), "schema", nothing)
    d = _read_vgio_json(path)
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    return get(d, :schema, nothing)
end

# -----------------------------------------------------------------------------
# A target <-> JSON: the body every target-carrying kind embeds
# -----------------------------------------------------------------------------
#
#   {"type": "points", "points": [[x, y(, z)], …], "point_covariances": [[[…]…]…]}   PointTarget
#   {"type": "planar", "reference": "<reference>", "calibration": {calibration}}     PlanarTarget of a referenced design
#   {"type": "planar", "nominal": [[x, y], …], "calibration": {calibration}}         PlanarTarget of explicit points
#   … "board": {"size": {"width": w, "height": h}, "origin": [u, v]}               (optional) its physical board
#   … "canvas": {"size": {"width": w, "height": h}, "origin": [u, v],              (optional) the printed page,
#                "texture": {"path": "<relative path>", "sha256": "<hex>",          its printed image (optional):
#                            "size": {"width": w, "height": h}, "origin": [u, v]}}  the file and the rectangle it covers
#   … "placement": {"origin": [u, v], "side": s}                                   (optional, planar only) VGC's ChartPlacement
#
# A PLACEMENT says where the calibration's dimensionless chart lies in the
# design chart (the chart point c is the design point origin + side·c): its
# origin and side are lengths in the file's `length_unit`. A placed target's
# calibration acts in that chart, so its record — nodes, parameters,
# covariance — is plain chart numbers, written and read as they are, never
# scaled by the unit (the placement carries the lengths). A body without one
# reads as before; no migration writes one.
#
# The board, the canvas and the texture are rectangles of ONE chart, the
# design chart of the nominal points (VGC's `Canvas` states it): each is its
# lower-left corner and its size, lengths in the file's `length_unit`; the
# board's placement on the page is `board.origin − canvas.origin`, derived and
# never stored. A texture's `path` is relative to the file that names it and
# `sha256` is the image file's digest; VGIo never opens the image — resolving
# and checking it is the consumer's.
#
# Points and nominal points in the file's `length_unit` (written in mm unless
# the writer is given another), lengths in metres in memory, in the
# target's own frame. A referenced design's nominal points are its owner's —
# a BlobBoards board uid resolves to them — so they are NOT restated: `load`
# returns the target unresolved unless the caller supplies `resolve`, the
# design owner's resolver.
#
# THE CALIBRATION — VGC's `TargetCalibration`, the map from the design to the
# physical target — is one record, its `type` the kind's tag in VGC's
# `TARGET_CALIBRATION_TAGS` (the one registry; snake_case of the type's name):
#
#   {"type": "rigid"}
#   {"type": "euclidean", "parameters": [angle, tx, ty]}       a planar EuclideanMap: radians, lengths
#   {"type": "scale", "parameters": [log_scale]}
#   {"type": "aspect", "parameters": [a1]}
#   {"type": "skew", "parameters": [u12]}
#   {"type": "thin_plate_spline", "nodes": [[u, v], …], "parameters": [fx1, …, fxK, fy1, …, fyK]}
#   {"type": "height_field", "nodes": [[u, v], …], "parameters": [h1, …, hK]}
#   {"type": "composition", "compose": [{outer}, …, {inner}]}
#
# `parameters` are the kind's `parameter_vector`, in its `parameter_names`
# order, lengths (by `length_mask`: a euclidean's `tx, ty`, the in-plane
# displacements, the heights) in the file's `length_unit`, plain numbers
# otherwise (an angle, the log-scale, the log-stretch, the shear). A
# euclidean's angle is the one its value stores (VGC holds the rigid member's
# rotation as an `Angle2d`), so a value read back is its file's. `nodes` are positions
# in the target's plane, lengths in the file's unit. The thin-plate kinds hold
# PURE BENDING (VGC: no affine term in the interpolant): the reader rebuilds
# each and refuses values the constructor would move (an affine displacement
# is the affine factors', a plane of heights the pose's).
#
# A composition — 2 to 6 members in VGC's canonical order
# `HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, any
# left out, each once (`Aspect ∘ Skew`, the print error with a jointly
# estimated pose; `HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew`; …),
# CoordinateTransformations' `∘` — is written by its structure, FLAT: its
# members records of their own listed under `compose` in CT's order, the outer
# map first (the key a lens record lists its stages in sequence under, in the
# same order); it reads back right-associated, VGC's one association. The
# retired `calibration_chain` record (target v2 / rig v5), whose `members` were
# in application order, and the retired `affine` kind (CT's `AffineMap`, its
# print error now `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`) are refused by name,
# naming `vgio_migrate`.
# `covariance` (optional, top level only) is present iff the calibration is
# `Uncertain`: n × n over `parameter_vector` (a composition's: every member's,
# concatenated outer first — one joint covariance, as `Uncertain(c, Σ)` holds
# it; a thin-plate kind's has rank 2K − 6 or K − 3, which is valid), each entry
# in the file's unit raised to its length dimension.
#
# THE BOARD (optional, a planar target's): the physical board the design is
# printed on, parallel to a camera's sensor — the board proper, inside the
# sheet's margin and frame included (BlobBoards' `board_size`), not the sheet —
# its `size`, a `Size2` of lengths in the established Size2 form (`width`,
# `height`), and its lower-left corner in the design chart, `origin`, every
# number a length in the file's unit; absent when the target has none.
# (`Rect(board)` is the target's domain in the design chart.)

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

# The record type of a composition, written by its structure (it has no tag in
# VGC's registry: the tag rule would spell CoordinateTransformations' type).
const _COMPOSITION_TYPE = "composition"
# The key a composition lists its members under, outer first — a calibration
# record's and a lens record's alike (VGC's `lens_record`).
const _COMPOSE_KEY = "compose"
# v2/v5's composition record (its `members` in application order): refused
const _RETIRED_COMPOSITION_TYPE = "calibration_chain"

# The keys a calibration record of each kind states, by its tag, and how many
# parameters it holds — VGC's `length(length_mask(planar_kind(K, n)))` of the
# kind `K` its registry gives for the tag (N = 2, a thin-plate kind over its `n`
# nodes), read from VGC, never restated here, so a change of a kind's parameter layout
# moves the record's counts, its covariance's order and the identity tree's
# permutation with it. (Only the v2 migrator keeps frozen counts,
# `_v2_parameter_count`: there the input is the v2 FORMAT, which does not change.)
_calibration_keys(ty) = ty == "rigid" ? () : ty in ("thin_plate_spline", "height_field") ? (:nodes, :parameters) : (:parameters,)
# (an unknown tag: VGC's ArgumentError, listing the known ones)
_record_parameter_count(ty, nodes) = length(length_mask(planar_kind(TARGET_CALIBRATION_TAGS[String(ty)], length(nodes))))
_n_parameters(x) = x[:type] == _COMPOSITION_TYPE ? sum(_n_parameters, x[:compose]; init = 0) :
    _record_parameter_count(x[:type], get(x, :nodes, ()))

# The retired composition record — `calibration_chain` with its `members` in
# application order (target v2 / rig v5) — refused by name: its key and order
# changed together, so it is never read as the current one.
const _RETIRED_COMPOSITION_MSG =
    "a \"$(_RETIRED_COMPOSITION_TYPE)\" calibration (its `members` in application order) is the retired " *
    "target v2 / rig v5 record, which a file of that version is read through its migration for (vgio_migrate " *
    "rewrites it); a current record holds a composition as " *
    "{\"type\": \"$(_COMPOSITION_TYPE)\", \"$(_COMPOSE_KEY)\": [outer, …, inner]}, CoordinateTransformations' order"
# The retired affine kind (CT's AffineMap, six parameters): its print error is
# the composition of VGC's gauge-free factors.
const _RETIRED_AFFINE_MSG =
    "an \"affine\" calibration (CoordinateTransformations' AffineMap, [a11, a12, a21, a22, bx, by]) is retired: " *
    "an affine print error is the composition EuclideanMap ∘ Scale ∘ Aspect ∘ Skew (A = s·R·D·U), " *
    "{\"type\": \"$(_COMPOSITION_TYPE)\", \"$(_COMPOSE_KEY)\": [euclidean, scale, aspect, skew]} — a target v2 / " *
    "rig v5 file holding one is read through its migration, which converts it (vgio_migrate rewrites the file)"

function _target_calibration_check(x, member::Bool)
    ty = x[:type]
    (ty == _RETIRED_COMPOSITION_TYPE || haskey(x, :members)) && return _RETIRED_COMPOSITION_MSG
    ty == "affine" && return _RETIRED_AFFINE_MSG
    ty == _COMPOSITION_TYPE && return _composition_check(x, member)
    haskey(TARGET_CALIBRATION_TAGS, ty) ||
        return "unknown calibration type $(repr(ty)) (one of $(join(keys(TARGET_CALIBRATION_TAGS), ", ")), $(_COMPOSITION_TYPE))"
    want = _calibration_keys(ty)
    for k in (:nodes, :parameters, Symbol(_COMPOSE_KEY))
        (k in want) == haskey(x, k) || return k in want ? "a $(ty) calibration states its `$(k)`" : "a $(ty) calibration has no `$(k)`"
    end
    :nodes in want && isempty(x[:nodes]) && return "a $(ty) calibration has at least one node"
    n = _n_parameters(x)
    (:parameters in want && length(x[:parameters]) != n) &&
        return "a $(ty) calibration$(:nodes in want ? " over $(length(x[:nodes])) nodes" : "") states $(n) parameters, got $(length(x[:parameters]))"
    return _covariance_check(x, ty, n, member)
end
# A composition: 2 to 6 members, each a kind of the canonical order (no Rigid,
# no composition), in that order, each once, the outer map first, as
# CoordinateTransformations orders `t1 ∘ t2`.
function _composition_check(x, member::Bool)
    member && return "a $(_COMPOSITION_TYPE)'s member is not a composition"
    for k in (:nodes, :parameters)
        haskey(x, k) && return "a $(_COMPOSITION_TYPE) has no `$(k)` (its members state theirs)"
    end
    haskey(x, :compose) || return "a $(_COMPOSITION_TYPE) states its members, `$(_COMPOSE_KEY)`"
    ms = x[:compose]
    for (i, r) in enumerate(ms)
        msg = _target_calibration_check(r, true)
        msg === nothing || return "member $(i): $(msg)"
    end
    order = map(calibration_tag, TARGET_CALIBRATION_ORDER)   # VGC's canonical order, the outer map first, by tag
    ranks = [something(findfirst(==(r[:type]), order), 0) for r in ms]
    (2 <= length(ms) <= length(order) && all(>(0), ranks) && issorted(ranks; lt = <=)) || return (
        "a $(_COMPOSITION_TYPE)'s members are 2 to 6 kinds in the canonical order $(join(nameof.(TARGET_CALIBRATION_ORDER), " ∘ ")) " *
            "([$(join(order, ", "))]), any left out, each once, the outer map first; " *
            "got [$(join((r[:type] for r in ms), ", "))]"
    )
    return _covariance_check(x, _COMPOSITION_TYPE, _n_parameters(x), member)
end
function _covariance_check(x, ty, n, member::Bool)
    if haskey(x, :covariance)
        member && return "a composition's member carries no `covariance` (the composition's covers every member's parameters)"
        n == 0 && return "a $(ty) calibration has no parameters, so no `covariance`"
        c = x[:covariance]
        (length(c) == n && all(r -> length(r) == n, c)) ||
            return "`covariance` is $(n)×$(n), over the calibration's parameters"
        all(c[i][j] == c[j][i] for i in 1:n, j in 1:n) || return "`covariance` must be symmetric"
    end
    return nothing
end
const _CALIBRATION_FIELDS = Pair{Symbol, _Spec}[:nodes => _Vec(_XY), :parameters => _Vec(_Real()), :covariance => _Vec(_Vec(_Real()))]
# (`members`, the retired record's key, is admitted here only to be refused by
# name in `_target_calibration_check`)
const _TARGET_CALIBRATION_MEMBER = _Obj(Pair{Symbol, _Spec}[:type => _NonEmpty()], vcat(_CALIBRATION_FIELDS, Pair{Symbol, _Spec}[:compose => _Vec(_Free()), :members => _Vec(_Free())]))
const _TARGET_CALIBRATION = _Checked(
    _Obj(Pair{Symbol, _Spec}[:type => _NonEmpty()], vcat(_CALIBRATION_FIELDS, Pair{Symbol, _Spec}[:compose => _Vec(_TARGET_CALIBRATION_MEMBER), :members => _Vec(_Free())])),
    x -> _target_calibration_check(x, false),
)

const _BOARD = _Obj(
    :size => _Checked(_Obj(:width => _Real(), :height => _Real()), z -> (z[:width] > 0 && z[:height] > 0) ? nothing : "a board's width and height are > 0"),
    :origin => _XY,
)

const _SIZE_WH = _Obj(:width => _Real(), :height => _Real())
const _CANVAS_TEXTURE = _Checked(
    _Obj(
        :path => _Named(p -> !isempty(p) && !isabspath(p), "a path relative to the file that names it"),
        :sha256 => _Named(s -> occursin(r"^[0-9a-f]{64}$", s), "a SHA-256 hex digest (64 lowercase hex digits)"),
        :size => _SIZE_WH, :origin => _XY,
    ),
    z -> (z[:size][:width] > 0 && z[:size][:height] > 0) ? nothing : "a canvas texture's width and height are > 0",
)
const _CANVAS = _Checked(
    _Obj(Pair{Symbol, _Spec}[:size => _SIZE_WH, :origin => _XY], Pair{Symbol, _Spec}[:texture => _CANVAS_TEXTURE]),
    z -> (z[:size][:width] >= 0 && z[:size][:height] >= 0) ? nothing : "a canvas's width and height are ≥ 0",
)

const _PLACEMENT = _Checked(
    _Obj(:origin => _XY, :side => _Real()),
    z -> (isfinite(z[:side]) && z[:side] > 0) ? nothing : "a chart placement's side is > 0",
)

const _TARGET_BODY = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[:type => _NonEmpty()],
        Pair{Symbol, _Spec}[
            :points => _TARGET_POINTS, :point_covariances => _Vec(_Vec(_Vec(_Real()))),
            :reference => _NonEmpty(), :nominal => _Vec(_Vec(_Real(), 2)), :calibration => _TARGET_CALIBRATION,
            :board => _BOARD, :canvas => _CANVAS, :placement => _PLACEMENT,
        ],
    ),
    function (t)
        ty = t[:type]
        if ty == "points"
            haskey(t, :points) || return "a points target states its `points`"
            haskey(t, :placement) && return "a points target has no `placement` (a chart placement is a planar target's: where its calibration's chart lies)"
            any(k -> haskey(t, k), (:reference, :nominal, :calibration, :board, :canvas)) && return "a points target has `points` (and `point_covariances`) only"
            haskey(t, :point_covariances) && return _point_covariances_check(t[:points], t[:point_covariances])
            return nothing
        elseif ty == "planar"
            any(k -> haskey(t, k), (:points, :point_covariances)) && return "a planar target has `reference` or `nominal`, and `calibration`"
            haskey(t, :calibration) || return "a planar target states its `calibration`"
            (haskey(t, :reference) ⊻ haskey(t, :nominal)) ||
                return "a planar target states its `reference` or its `nominal` points — one, not both (a referenced design's points are its owner's)"
            return nothing
        end
        return "unknown target type $(repr(ty)) (points, planar)"
    end,
)

# A target's points on the wire: the geometric values, and their covariances
# iff every point is `Uncertain` (plain and uncertain points are not mixed in
# one target — the file could not say which is which).
function _points_json(what, pts, unit::_LeafUnit)
    vals = [_lengths_json(geom(p), unit, "$(what): a point") for p in pts]
    n_u = count(p -> p isa Uncertain, pts)
    n_u == 0 && return (points = vals,)
    n_u == length(pts) || throw(
        ArgumentError("$(what): its points are partly Uncertain — make all of them Uncertain (exact where known) or none")
    )
    return (points = vals, point_covariances = [_length_cov_json(cov(p), unit, _point_cov_dims(length(geom(p)))) for p in pts])
end

# `points` back to static vectors of lengths (metres) of the recorded
# dimension, the form VGC composes a target's points with its pose in
# (`points(rig, id)`); `Uncertain` iff the file states their covariances
# (unit² → m²).
function _points_from_json(vals, unit::Units, covs = nothing)
    isempty(vals) && return SVector{2, Float64}[]
    n = length(first(vals)::Vector{Any})
    pts = [_lengths_from_json(Val(n), v::Vector{Any}, unit) for v in vals]
    covs === nothing && return pts
    return [Uncertain(p, CovMat(_length_cov_from_json(c, n, unit, _point_cov_dims(n)))) for (p, c) in zip(pts, covs)]
end

# A plain metre value (a calibration's stored length) as the file's number in `u`.
_metres_json(x::Real, u::_LeafUnit) = _leaf_power(Float64(x), u, 1)

# Where a calibration's lengths go: the file's unit or the identity's persisted
# metres (`_LeafUnit`), or — for a PLACED target (VGC's `ChartPlacement`),
# whose calibration acts in the board's dimensionless chart — `_CHART`: its
# nodes, values and covariance are plain chart numbers, written and read as
# they are, never scaled by `length_mask` (the placement carries the lengths).
struct _Chart end
const _CHART = _Chart()
const _CalUnit = Union{_LeafUnit, _Chart}
_metres_json(x::Real, ::_Chart) = Float64(x)
_length_cov_json(Σ::AbstractMatrix, ::_Chart, dims) = _cov_rows((x, n) -> x, Σ, dims)
_length_cov_from_json(c, n::Int, ::_Chart, dims) = SMatrix{n, n, Float64}(ntuple(k -> Float64(c[(k - 1) % n + 1][(k - 1) ÷ n + 1]::Real), n * n))
_nodes_json(c, u::_CalUnit) = [[_metres_json(p[1], u), _metres_json(p[2], u)] for p in c.system.nodes]

const _FactorKind = Union{EuclideanMap{2}, Scale, Aspect{2}, Skew{2}}
_tag(c::Union{Rigid, ThinPlateSpline, HeightField, _FactorKind}) = calibration_tag(typeof(c))
_tag(::ComposedTransformation) = _COMPOSITION_TYPE
# A calibration with lengths is stored physical: plain numbers are refused, as
# a target's plain points are. One without lengths (Rigid, Scale, Aspect,
# Skew and their compositions) has no form to refuse.
_physical(c, ::_LeafUnit) = ((any(length_mask(c)) && !has_units(c)) && _refuse_plain("a $(nameof(typeof(c))) target calibration"); c)
_physical(c, ::_Chart) = c   # a placed target's calibration is plain chart numbers (VGC refuses one with units)

_calibration_record(c::Rigid, ::_CalUnit) = (type = _tag(c),)
_calibration_record(c::Union{Scale, Aspect{2}, Skew{2}}, ::_CalUnit) = (type = _tag(c), parameters = Float64.(collect(parameter_vector(c))))
# The angle is the stored one: a target's rigid member holds its rotation as an
# `Angle2d` (VGC), whose `parameter_vector` is the angle itself, so the record
# writes it and the reader's `Angle2d` of it is the value, bit for bit.
function _calibration_record(c::EuclideanMap{2}, u::_CalUnit)
    θ = parameter_vector(_physical(c, u))
    return (type = _tag(c), parameters = [Float64(θ[1]), _metres_json(θ[2], u), _metres_json(θ[3], u)])
end
function _calibration_record(c::Union{ThinPlateSpline, HeightField}, u::_CalUnit)
    r = (type = _tag(_physical(c, u)), nodes = _nodes_json(c, u), parameters = [_metres_json(v, u) for v in parameter_vector(c)])
    _check_writes_back(c, r, u)
    return r
end
# The writer's half of the reader's pure-bending rule (`_reads_back`): a field
# pure bending neither in the file's numbers nor in metres would not read
# back, so it is refused here, naming the unit to write it in. (The identity
# tree, `_PERSISTED`, is never read.)
_check_writes_back(c, r, ::_Persisted) = nothing
function _check_writes_back(c, r, u::Union{Units, _Chart})
    _reads_back(r.nodes, r.parameters, u) === nothing || return nothing
    throw(
        ArgumentError(
            "a $(r.type) calibration that is pure bending neither in metres nor in the file's unit " *
                "$(u isa Units ? unit_string(u) : "(the placement's chart)") would not read back: pure bending (no affine term in the " *
                "interpolant) depends on the unit the nodes are in, and a unit move (with_unit, ustrip) does not re-project — " *
                "write it with the length_unit it was built in; nothing was written"
        )
    )
end
_calibration_record(c::ComposedTransformation, u::_CalUnit) =
    (type = _tag(_physical(c, u)), compose = [_calibration_record(l, u) for l in members(c)])

_calibration_record(c::Uncertain{<:TargetCalibration}, u::_CalUnit) =
    merge(_calibration_record(geom(c), u), (covariance = _length_cov_json(cov(c), u, length_mask(geom(c))),))

# A calibration's length as read: a length in metres from the file's unit, or
# (`_CHART`, a placed target's) the plain chart number as stored.
_read_metres(x, u::Units) = read_length(x::Real, u)
_read_metres(x, ::_Chart) = Float64(x::Real)
const _ReadUnit = Union{Units, _Chart}
_nodes_from_json(x, u::_ReadUnit) = [SVector(_read_metres(p[1], u), _read_metres(p[2], u)) for p in x.nodes::Vector{Any}]

# (by the kind's TYPE, which VGC's registry gives for the record's tag)
_calibration_from_record(::Type{Rigid}, x, u::_ReadUnit) = Rigid()
function _calibration_from_record(::Type{EuclideanMap}, x, u::_ReadUnit)
    p = x.parameters::Vector{Any}
    return EuclideanMap(Angle2d{Float64}(Float64(p[1])), SVector(_read_metres(p[2], u), _read_metres(p[3], u)); check = false)
end
_calibration_from_record(::Type{Scale}, x, u::_ReadUnit) = Scale(Float64(only(x.parameters)))
_calibration_from_record(::Type{Aspect}, x, u::_ReadUnit) = Aspect(Float64(only(x.parameters)))
_calibration_from_record(::Type{Skew}, x, u::_ReadUnit) = Skew(Float64(only(x.parameters)))
function _calibration_from_record(::Type{ThinPlateSpline}, x, u::_ReadUnit)
    system = _check_pure_bending(ThinPlateSpline, x.nodes, x.parameters, u)
    p = x.parameters::Vector{Any}
    K = length(x.nodes)
    return ThinPlateSpline(system, [_read_metres(p[i], u) for i in 1:K], [_read_metres(p[K + i], u) for i in 1:K]; project = false)
end
function _calibration_from_record(::Type{HeightField}, x, u::_ReadUnit)
    system = _check_pure_bending(HeightField, x.nodes, x.parameters, u)
    return HeightField(system, [_read_metres(v, u) for v in x.parameters::Vector{Any}]; project = false)
end
# A thin-plate kind holds pure bending (VGC: no affine term in its
# interpolant; its constructor projects the values it is given and keeps,
# bit for bit, values whose affine term is within 1e-12 of them). "No affine
# term" depends on the unit the nodes are in — scaling them moves a piston
# between the affine and the bending parts — and VGC's unit move
# (`with_unit`, `ustrip`) does not re-project, so a physical field is pure
# bending in the unit it was BUILT in: metres (built from lengths), or the
# unit whose plain numbers it was built from and then given (`with_unit(c,
# mm)`). The record is read, unprojected, when its values are pure bending in
# the file's own numbers (the unit it was written in; a placed target's chart
# numbers) or in metres; any other — an affine term in every form — is
# refused, never read moved. The writer refuses a field that would not read
# back (`_check_writes_back`), so anything written reads back `==`.
#
# `_reads_back(nodes, p, u)`: whether the record's nodes and parameters (the
# file's numbers; `p` a height field's K heights or a spline's 2K
# displacements, `fx` then `fy`) read back — each value block pure bending
# (VGC's `is_pure_bending`) on the nodes in the file's numbers or in metres —
# and the system the value is built on (metres, or the chart's numbers), each
# system formed once; `nothing` when they do not.
function _reads_back(nodes, p, u::_ReadUnit)
    file = ThinPlateSplineSystem([SVector(Float64(q[1]), Float64(q[2])) for q in nodes])
    pure_in_file = _pure_blocks(file, [Float64(v) for v in p])
    u isa _Chart && return pure_in_file ? file : nothing        # a placed target's chart numbers are its values'
    metres = ThinPlateSplineSystem([SVector(_metres(q[1], u), _metres(q[2], u)) for q in nodes])
    return pure_in_file || _pure_blocks(metres, [_metres(v, u) for v in p]) ? metres : nothing
end
_metres(x, u::Units) = strip_or_bare(m, read_length(x::Real, u))
_pure_blocks(system::ThinPlateSplineSystem{K}, v) where {K} =
    length(v) == K ? is_pure_bending(system, v) : is_pure_bending(system, v[1:K]) && is_pure_bending(system, v[(K + 1):end])
function _check_pure_bending(::Type{T}, nodes, p, u::_ReadUnit) where {T}
    system = _reads_back(nodes, p, u)
    system === nothing || return system
    what = T === HeightField ? "heights" : "displacements"
    throw(
        ArgumentError(
            "a $(calibration_tag(T)) calibration's $(what) carry an affine part (their thin-plate interpolant " *
                "has an affine term): a thin-plate kind holds pure bending — an affine displacement belongs to " *
                "Aspect ∘ Skew (∘ Scale ∘ EuclideanMap) composed after a thin_plate_spline, a plane of heights to the " *
                "target's pose. Neither moves out of the spline exactly (the affine factors compose after the spline, " *
                "not before it; a plane of heights is a shear, not a rigid motion), so a record holding one — a " *
                "target v2 / rig v5 spline among them — is not read or migrated: re-fit the calibration"
        )
    )
end
# A calibration record (validated) back to VGC's kind, `Uncertain` iff it
# states a covariance (unit^(lᵢ+lⱼ) → m^(lᵢ+lⱼ)).
function _target_calibration_from_json(x, u::_ReadUnit)
    c = _bare_calibration_from_json(x, u)
    haskey(x, :covariance) || return c
    dims = length_mask(c)
    return Uncertain(c, CovMat(_length_cov_from_json(x.covariance::Vector{Any}, length(dims), u, dims)))
end
# the calibration without its covariance: a kind, or a composition, its flat
# members folded right (`a ∘ (b ∘ c)`, VGC's one association)
_bare_calibration_from_json(x, u::_ReadUnit) = x.type::String == _COMPOSITION_TYPE ?
    foldr(∘, [_bare_calibration_from_json(r, u) for r in x.compose]) :
    _calibration_from_record(TARGET_CALIBRATION_TAGS[x.type::String], x, u)

"""
    target_json(target, unit = mm) -> NamedTuple

The target body every target-carrying kind embeds (see rig_schema.jl), its
points (lengths) and its calibration's lengths written in the length `unit`;
[`target_from_json`](@ref) reads it back. A target of plain points, or a plain
calibration with lengths, is an `ArgumentError` (attach the unit, `p .* m`). A
`PlanarTarget` of a referenced design is written by its reference alone.
Building blocks for a package serializing its own kind.
"""
target_json(t::PointTarget, unit::_LeafUnit = mm) = merge((type = "points",), _points_json("a points target", points(t), unit))
function target_json(t::PlanarTarget, unit::_LeafUnit = mm)
    pl = placement(t)
    cu = pl === nothing ? unit : _CHART          # a placed calibration: plain chart numbers
    return merge(
        reference(t) === nothing ?
            (type = "planar", nominal = [_lengths_json(p, unit, "a planar target's nominal point") for p in t.nominal], calibration = _calibration_record(calibration(t), cu)) :
            (type = "planar", reference = String(reference(t)), calibration = _calibration_record(calibration(t), cu)),
        _board_json(board(t), unit),
        _canvas_json(canvas(t), unit),
        _placement_json(pl, unit),
    )
end

# The placement ↔ its record (VGC's `ChartPlacement`: the chart point c is the
# design point origin + side·c): its origin and side, lengths in the file's
# unit, decimal-exactly; read back as lengths (stored plain metres).
_placement_json(::Nothing, ::_LeafUnit) = (;)
function _placement_json(P::ChartPlacement, u::_LeafUnit)
    has_units(P) || _refuse_plain("a target's chart placement")
    return (placement = (origin = [_metres_json(x, u) for x in P.origin], side = _metres_json(P.side, u)),)
end
_placement_from_json(x, u::Units) =
    ChartPlacement(SVector(read_length(x.origin[1]::Real, u), read_length(x.origin[2]::Real, u)), read_length(x.side::Real, u))

# The board ↔ its record. VGC's `Board` holds its size as lengths (metres) and
# its lower-left corner as plain metres (a physical target's; a unitless one is
# in no file). Both are written in the file's unit, the size in the Size2 form,
# decimal-exactly; read back, the size is lengths and the corner plain metres.
_board_json(::Nothing, ::_LeafUnit) = (;)
function _board_json(bd::Board, u::_LeafUnit)
    has_units(bd) || _refuse_plain("a target's board")
    size = (width = _leaf_length(bd.size.width, u), height = _leaf_length(bd.size.height, u))
    return (board = (size = size, origin = [_metres_json(x, u) for x in bd.origin]),)
end
function _board_from_json(x, u::Units)
    o = x.origin::Vector{Any}
    return Board(from_json(Size2, x.size, UnitContract(u); kind = :length), SVector(_read_power(o[1]::Real, u, 1), _read_power(o[2]::Real, u, 1)))
end

# The canvas ↔ its record: the board's rule for each rectangle (sizes by
# `write_length`/`read_length`, corners as plain metres in the file's unit),
# the texture's path and digest as stored.
_rect_json(r, u::_LeafUnit) = (
    size = (width = _leaf_length(r.size.width, u), height = _leaf_length(r.size.height, u)),
    origin = [_metres_json(x, u) for x in r.origin],
)
_rect_from_json(x, u::Units) = (
    Size2(width = read_length(x.size.width::Real, u), height = read_length(x.size.height::Real, u)),
    SVector(_read_power(x.origin[1]::Real, u, 1), _read_power(x.origin[2]::Real, u, 1)),
)
_canvas_json(::Nothing, ::_LeafUnit) = (;)
function _canvas_json(cv::Canvas, u::_LeafUnit)
    has_units(cv) || _refuse_plain("a target's canvas")
    tx = cv.texture === nothing ? (;) : (texture = merge((path = cv.texture.path, sha256 = cv.texture.sha256), _rect_json(cv.texture, u)),)
    return (canvas = merge(_rect_json(cv, u), tx),)
end
function _canvas_from_json(x, u::Units)
    tx = haskey(x, :texture) ? CanvasTexture(x.texture.path::String, x.texture.sha256::String, _rect_from_json(x.texture, u)...) : nothing
    return Canvas(_rect_from_json(x, u)...; texture = tx)
end

"""
    target_from_json(body, unit; resolve = nothing) -> AbstractTarget

Read a target body (the inverse of [`target_json`](@ref)) whose lengths are in
`unit`, a unit name such as `"mm"`, as lengths in metres, its calibration
`Uncertain` iff the body states a covariance. A planar target that names its
reference is returned unresolved unless `resolve(reference::Symbol)` returns
its nominal points — lengths, as the design's owner holds them; plain points
are an `ArgumentError`.
"""
function target_from_json(t, unit::AbstractString; resolve = nothing)
    u = parse_unit(unit)
    if t.type == "points"
        return PointTarget(_points_from_json(t.points::Vector{Any}, u, haskey(t, :point_covariances) ? t.point_covariances::Vector{Any} : nothing))
    end
    pl = haskey(t, :placement) ? _placement_from_json(t.placement, u) : nothing
    c = _target_calibration_from_json(t.calibration, pl === nothing ? u : _CHART)   # placed: plain chart numbers
    bd = haskey(t, :board) ? _board_from_json(t.board, u) : nothing
    cv = haskey(t, :canvas) ? _canvas_from_json(t.canvas, u) : nothing
    haskey(t, :nominal) && return PlanarTarget(nothing, _points_from_json(t.nominal::Vector{Any}, u), c, bd, cv, pl)
    ref = Symbol(t.reference::String)
    nominal = resolve === nothing ? nothing : resolve(ref)
    (nominal === nothing || isempty(nominal) || has_units(nominal)) ||
        _refuse_plain("resolve($(repr(ref))): the design's nominal points")
    return PlanarTarget(ref, nominal, c, bd, cv, pl)
end

# -----------------------------------------------------------------------------
# vgio/target/v3 ↔ one target (AbstractTarget, or one Attributed with its calibration record)
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/target/v3", "length_unit": "mm", "target": {target body},
#  "calibration_record": {…}}             (optional: who calibrated it — the cameras' block's keys)

const _TARGET_SPEC = _Obj(
    Pair{Symbol, _Spec}[:schema => _schema_tok("target"), :length_unit => _LENGTH_UNIT, :target => _TARGET_BODY],
    Pair{Symbol, _Spec}[:calibration_record => _CALIBRATION, _PROVENANCE...],
)

"""
    save(path, target::AbstractTarget; length_unit = mm, provenance = (;))
    save(path, Attributed(target, (; calibration_record)); length_unit = mm, provenance = (;))

Write one physical target as a `vgio/target/v3` file — its body (rig_schema.jl:
its points, and a planar target's calibration with its parameters and, when
`Uncertain`, their covariance) in `length_unit` and, for an `Attributed` one,
its `calibration_record`: who calibrated it, when, from what (the keys of a
camera's `calibration` block). Plain points are an `ArgumentError`, and so is a
`PosedTarget`: the file holds an unposed target, so save its `model`, or the
posed one inside a `Rig`.

The calibration record is not the target's calibration: `calibration(t)` is
the `TargetCalibration`, `x.calibration_record` the record of how it was
obtained.
"""
save(path::AbstractString, t::AbstractTarget; length_unit::Units = mm, provenance::NamedTuple = (;)) = (
    _write_store(
        path, _TARGET_SPEC,
        _with_provenance((schema = _schema("target"), length_unit = unit_string(length_unit), target = target_json(t, length_unit)), provenance),
    ); nothing
)
function save(
        path::AbstractString, t::Attributed{<:AbstractTarget, <:NamedTuple{(:calibration_record,)}};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    body = (
        schema = _schema("target"), length_unit = unit_string(length_unit), target = target_json(t.value, length_unit),
        calibration_record = _calibration_json(t.calibration_record, "a target's calibration record"),
    )
    _write_store(path, _TARGET_SPEC, _with_provenance(body, provenance))
    return nothing
end
# A `vgio/target/v3` file holds an UNPOSED target: a posed one is refused here,
# before anything is written, rather than deep inside the writer. Its pose is a
# rig member's, written by the rig writer.
_refuse_posed_target(path) = throw(
    ArgumentError(
        "save($(repr(path)), ::PosedTarget): a $(_schema("target")) file holds an unposed target — save `model(pt)`, " *
            "or the posed target as a member of a rig with the rig writer: `save(path, Rig{F}(Dict(id => pt)))`"
    )
)
save(path::AbstractString, ::PosedTarget; kwargs...) = _refuse_posed_target(path)
# The attribute's previous name: refused naming its successor.
save(path::AbstractString, ::Attributed{<:AbstractTarget, <:NamedTuple{(:calibration,)}}; kwargs...) = throw(
    ArgumentError(
        "save($(repr(path))): a target's record of how it was calibrated is the attribute `calibration_record` " *
            "(`Attributed(target, (; calibration_record = …))`), not `calibration`, which is the target's own " *
            "TargetCalibration; a file holding the old block is migrated with vgio_migrate"
    )
)
save(path::AbstractString, ::Attributed{<:PosedTarget, <:NamedTuple{(:calibration_record,)}}; kwargs...) = _refuse_posed_target(path)

"""
    load(AbstractTarget, path; resolve = nothing) -> AbstractTarget | Attributed{<:AbstractTarget}

Read a `vgio/target/v3` file: the target, `Attributed` with its
`calibration_record` where the file states one. `resolve(reference)` supplies
a referenced design's nominal points (else the target is returned
unresolved). A `vgio/target/v1` or `v2` file is read through its version
steps (`migrate_record`).
"""
function load(::Type{AbstractTarget}, path::AbstractString; resolve = nothing)
    d = _read_store(path, "target", _TARGET_SPEC)
    t = target_from_json(d.target, d.length_unit::String; resolve)
    return haskey(d, :calibration_record) ?
        Attributed(t, (; calibration_record = _calibration_from_json(d.calibration_record::AbstractDict))) : t
end

# -----------------------------------------------------------------------------
# vgio/rig/v6 ↔ Rig, every storage
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/rig/v6", "pixel_frame": "colmap", <pose declarations>, "frame": "body:camera_rig",
#  "sensors": [{"id": "cam1", "camera": {camera body | generic body},
#               "extrinsics": {rotation, translation}}],               extrinsics: `frame` → view:<id>
#  "targets": [{"id": "<uid>", "target": {target body},
#               "pose": {rotation, translation[, covariance]}}]}       pose: target:<id> → `frame`
#
# A sensor rig has `sensors` only, a rig of targets `targets` only (a board
# rig, a motion-capture body's constellation, the boards of a calibration
# target gauged on one of them), a camera rig carrying its constellation both;
# `pixel_frame` is declared iff there are sensors. Ids are unique across the
# two. A sensor is a `BoundedCamera` (a camera body requires the pixel extent);
# its extrinsics are a plain-typed `EuclideanMap` (never `Uncertain`); a target pose
# may be `Uncertain`. Every translation and point is a length.

# A sensor bounded by its physical `Sensor` carries the pixel pitch in its
# camera body (`_PITCH`, camera_schema.jl), in the rig's `length_unit`. Additive
# in rig/v5, which is unreleased: no version moves.
const _RIG_CAMERA_BODY = _Obj(_CAMERA_BODY_FIELDS, Pair{Symbol, _Spec}[:pitch => _PITCH])
# A generic body's `present` is optional: every node when it is absent
# (generic_camera_schema.jl), so every rig/v6 file written before reads as it
# did; the writer emits it for a model that does not hold every node.
const _RIG_GENERIC_BODY = _Checked(
    _Obj(_GENERIC_BODY_FIELDS, Pair{Symbol, _Spec}[:point_grid => _GRID, :pitch => _PITCH, _GENERIC_OPTIONAL..., _GENERIC_PRESENT...]),
    _generic_check,
)

const _SENSOR_RECORD = _Obj(
    :id => _NonEmpty(),
    :camera => _Either(Pair{Symbol, _Spec}[:lens => _RIG_CAMERA_BODY, :type => _RIG_GENERIC_BODY]),
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
        _with_pitch(_camera_body(m, extent(c)), sensor(c), unit)
    else
        _with_pitch(_generic_body(m, unit), _sensor_of(c), unit)      # generic grid model: carries its own extent
    end
    return (id = String(id), camera = body, extrinsics = pose_json(extrinsics(sensor_rig(rig), i), unit))   # <rig frame> → view:<id>
end

# a rig member's physical sensor, if it is bounded by one
_sensor_of(c::BoundedCamera) = sensor(c)
_sensor_of(c) = nothing

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

Write a physical VGC `Rig` — sensors, targets, or both — as a `vgio/rig/v6`
file in `length_unit`: each sensor (a `BoundedCamera`) with its extrinsics
`frame → view:<id>`, each target (sorted by id) with its body and its pose
`target:<id> → frame` (a `covariance` iff `Uncertain`). A rig of plain
translations or points is an `ArgumentError` (attach the unit, `t .* m`). A
rig's name, if it has one worth keeping, belongs in `provenance`.
"""
save(path::AbstractString, rig::Rig; length_unit::Units = mm, provenance::NamedTuple = (;)) =
    (_write_store(path, _RIG_SPEC, _with_provenance(_rig_body(rig, length_unit), provenance)); nothing)

"""
    load(Rig, path; resolve = nothing) -> Rig

Read a `vgio/rig/v6` file back into a fully labeled VGC `Rig` of the storage
it holds, its lengths in metres: sensors (each a `BoundedCamera` labeled `frame → ViewFrame{id}`),
targets (each a `PosedTarget`, minted `TargetFrame{id} → frame` on access), or both.
`resolve(design)` supplies a referenced planar design's nominal points (see
[`target_from_json`](@ref)). Type-unstable by nature (frame types are minted
from file data) — load once, dispatch after. A `vgio/rig/v4` or `v5` file is
read through its version steps (`migrate_record`).
"""
load(::Type{Rig}, path::AbstractString; resolve = nothing) = _rig_from_record(_read_store(path, "rig", _RIG_SPEC); resolve)

function _rig_from_record(d; resolve = nothing)
    F = frame_from_name(d.frame::String)
    unit = d.length_unit::String
    haskey(d, :targets) || return Rig{F}((_load_sensors(d, F, unit)...,))
    members = Dict{Symbol, PosedTarget}(
        Symbol(t.id::String) => PosedTarget(target_from_json(t.target, unit; resolve), pose_from_json(t.pose, unit))
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
            BoundedCamera(PosedCamera(gm, Xl), _bound_from_body(s.camera, Size2(gm.width, gm.height), parse_unit(unit)))
        else
            cm, ext = _camera_from_body(s.camera, pframe)
            BoundedCamera(PosedCamera(cm, Xl), _bound_from_body(s.camera, ext, parse_unit(unit)))
        end
    end
end

"""
    content_id(targets) -> String

The identity a measurement file (`target_points2d`, `target_points3d`) names a
target definition by — `targets` a `Rig` or a `vgio/rig/v6` path: the SHA-256
of the canonical JSON (`canonical_json`) of the rig's IDENTITY TREE. The tree
is built from the in-memory value by the record writer's own walk, under the
fixed names below — names of the identity's own, which a later record format
must keep mapping to — every length in metres (base units), the targets sorted
by id, and no schema version, declaration or provenance. The identity therefore does not change when the
file format does: a rig, every file written from it in any decimal
`length_unit`, the rig loaded back from such a file, and a migration of the
file to the next rig version have one identity. A relabeled provenance does
not change it; a target's calibration kind, its parameters, its covariance
its board and its canvas (with its texture reference) do.

The identity is of the persisted representation; a value and the file written
from it hash equal by construction. Every length enters as the number a file in
millimetres holds for it and reads back as (`read(write(x))`, idempotent), so a
quantity built in code as `v * mm`, which sits an ulp from the nearest decimal
of `v`, hashes as the file written from it. A pose's rotation enters as
the quaternion the writer stores (one conversion, shared by the writer and the
tree: w-x-y-z, w ≥ 0; a `QuatRotation` as held, any other rotation converted
once), so a rig held with `RotMatrix` poses, the file written from it and the
rig loaded back have one identity. A rotation converted matrix → quaternion →
matrix is not bit-exact, so two in-memory rigs built from the same matrix by
different routes can differ in the last ulp, and so can their identities:
mint an identity from what you save. Lengths likewise: a decimal unit (mm,
cm, m, μm) holds every metre value exactly, so its files share the value's
identity; a non-decimal unit (inch) is exact only for numbers born in it, so
an inch file of mm-born numbers reads back within an ulp and its identity
differs from the value's.

The identity tree, fixed (a test pins it for a rig of every target kind):

    rig        {"frame": <frame name>,
                "sensors": [{"id", "camera", "extrinsics": <pose>}, …]  (iff sensors, in rig order),
                "targets": [{"id", "target", "pose": <pose>}, …]        (iff targets, sorted by id)}
    pose       {"rotation": [w, x, y, z] (Hamilton, w ≥ 0), "translation": [m, m, m],
                "covariance": 6×6 rows over [δω; δt], rad and m       (iff Uncertain)}
    camera     {"intrinsics": [fx, fy, cx, cy, skew] (px, the canonical pixel frame),
                "extent": [width, height] (px), "lens": VGC's lens_record of the lens, its stages in sequence
                under "chain" in application order (the inner stage first)}
             | {"grid_model": "central" | "noncentral", "extent": [width, height],
                "calibrated_area": [min_x, min_y, max_x, max_y], "grid": [width, height],
                "directions": row-major [[x, y, z], …], "points": row-major [[m, m, m], …] (non-central),
                "nominal_intrinsics": [fx, fy, cx, cy, skew] (px)               (iff the model has one),
                "domain": {"ellipse": [cx, cy, a, b, θ]} (px, a ≥ b, rad)       (iff not the whole plane)}
    target     {"kind": "points", "points": [[m, …], …], "point_covariances": per point d×d rows, m² (iff Uncertain)}
             | {"kind": "planar", "reference": <id> | "nominal": [[m, m], …], "calibration": <calibration>,
                "board": {"size": [width, height] (m), "origin": [m, m]}  (iff the target has a board),
                "canvas": {"size": [width, height] (m), "origin": [m, m],
                           "texture": {"path", "sha256", "size": [width, height] (m), "origin": [m, m]} (iff it has one)}
                                                                          (iff the target has a canvas),
                "placement": {"origin": [m, m], "side": m}                (iff the target has a chart placement;
                                                                           its calibration's numbers are then the chart's, as held)}
    calibration {"kind": <the kind's tag> | "calibration_chain" (a composition), "parameters": parameter_vector (lengths in m),
                 "nodes": [[m, m], …] (thin-plate kinds), "members": [<calibration>, <calibration>] (a composition, in application order: the inner map first,
                 whatever order the record lists them in),
                 "covariance": rows over the parameters, m^(lᵢ+lⱼ) (iff Uncertain)}
"""
content_id(x::Union{Rig, AbstractString}) = _targets_hash(x)

_targets_hash(path::AbstractString) = _targets_hash(_rig_from_record(_read_store(path, "rig", _RIG_SPEC)))   # a referenced design stays a reference
_targets_hash(rig::Rig) = sha256_hex(canonical_json(_identity_tree(rig)))

# The identity tree (content_id's docstring). Its targets and poses are the
# writer's own records (`target_json`, `pose_json`), built with every length
# leaf as its persisted metre value (`_PERSISTED`) instead of a file's number,
# under the identity's names: `kind` for a record's `type`, a board's size as
# [width, height]. Every number is a Float64 (base units), an extent or grid
# size an Int; a name is a String.
const _Id = Dict{String, Any}

function _identity_tree(rig::Rig)
    d = _Id("frame" => frame_name(reference_frame(rig)))
    _has_sensors(rig) && (d["sensors"] = Any[_id_sensor(rig, i, id) for (i, id) in enumerate(ids(sensor_rig(rig)))])
    _has_targets(rig) && (d["targets"] = _id_targets(target_rig(rig)))
    return d
end
_id_sensor(rig, i, id) =      # (as the writer's `_sensor_json`: the rig untyped here)
    _Id("id" => String(id), "camera" => _id_camera(sensor_rig(rig).members[i]), "extrinsics" => _lower(pose_json(extrinsics(sensor_rig(rig), i), _PERSISTED)))
_id_targets(tr) = Any[
    _Id("id" => String(id), "target" => _id_order!(_id_kind!(_id_canvas!(_id_board!(_lower(target_json(tr[id].target, _PERSISTED)))))), "pose" => _lower(pose_json(pose(tr, id), _PERSISTED)))
        for id in sort!(collect(ids(tr)))
]
# The identity's ONE member order, independent of the record's: a composed
# calibration's members and a lens's stages in APPLICATION order (the inner map
# first), with a composition's covariance permuted to match — the order rig/v5
# wrote and hashed. The record lists them in CoordinateTransformations' order
# since v6 (and under new names, `composition`/`compose`); hashing the record's
# order would re-id every rig with a composed target or lens on migration,
# which an identity must not do. The identity keeps v5's names too: a
# composition is `{"kind": "calibration_chain", "members": [inner, outer]}`, a
# lens's stages its `"chain"`.
_id_order!(d::_Id) = (haskey(d, "calibration") && (d["calibration"] = _composition_v3_to_v2(d["calibration"], "kind")); d)
_id_lens(l) = _lens_v4_to_v3(_lower(_lens_json(l)))
# a record's `type` (a target's, a calibration's and its members') is the identity's `kind`.
# The identity is built from the writer's records, so a writer key renamed or
# added changes every content_id unless it is mapped back here; the pinned-tree
# test ("rig/v5: content_id is pinned on a fixed canonical tree") is the guard.
function _id_kind!(d::_Id)
    d["kind"] = pop!(d, "type")
    haskey(d, "calibration") && _id_kind!(d["calibration"])
    foreach(_id_kind!, get(d, _COMPOSE_KEY, ()))
    return d
end
# a canvas's and its texture's sizes are [width, height], as the board's
_id_size!(r) = (s = r["size"]; r["size"] = Any[s["width"], s["height"]]; r)
function _id_canvas!(d::_Id)
    haskey(d, "canvas") || return d
    _id_size!(d["canvas"])
    haskey(d["canvas"], "texture") && _id_size!(d["canvas"]["texture"])
    return d
end
# a board's size is [width, height]
function _id_board!(d::_Id)
    haskey(d, "board") || return d
    s = d["board"]["size"]
    d["board"]["size"] = Any[s["width"], s["height"]]
    return d
end

function _id_camera(c)
    mdl = model(c)
    if mdl isa CameraModel
        K = mdl.intrinsics.K
        e = extent(c)
        return _Id(
            "intrinsics" => Any[Float64(K[1, 1]), Float64(K[2, 2]), Float64(K[1, 3]), Float64(K[2, 3]), Float64(K[1, 2])],
            "extent" => Any[Int(e.width), Int(e.height)], "lens" => _id_lens(mdl.lens),
            _id_pitch(c)...,
        )
    end
    return merge(_id_grid_model(mdl), _Id(_id_pitch(c)...))
end
# the pitch iff the camera is bounded by its sensor: the writer's own `_with_pitch`,
# its leaves the persisted metre values
_id_pitch(c) = (s = _sensor_of(c); s === nothing ? () : ("pitch" => _lower(_with_pitch((;), s, _PERSISTED).pitch),))
# the held nodes, row by row (every node of a model that holds them all, as before the presence mask)
_id_grid(g, P) = Any[Any[Float64(g[gy, gx][k]) for k in 1:3] for gy in 1:size(g, 1) for gx in 1:size(g, 2) if P[gy, gx]]
_id_grid_metres(g, P) = Any[Any[_persisted(g[gy, gx][k]) for k in 1:3] for gy in 1:size(g, 1) for gx in 1:size(g, 2) if P[gy, gx]]
function _id_grid_header(gm, grid)
    return _Id(
        "extent" => Any[Int(gm.width), Int(gm.height)],
        "calibrated_area" => Any[Int(gm.calib_min_x), Int(gm.calib_min_y), Int(gm.calib_max_x), Int(gm.calib_max_y)],
        "grid" => Any[size(grid, 2), size(grid, 1)],
    )
end
_id_grid_model(gm::CentralGenericModel) = merge(
    _id_grid_header(gm, gm.grid),
    _Id("grid_model" => "central", "directions" => _id_grid(gm.grid, present_nodes(gm))), _id_grid_extras(gm),
)
_id_grid_model(gm::NoncentralGenericModel) = merge(
    _id_grid_header(gm, gm.directions),
    _Id("grid_model" => "noncentral", "directions" => _id_grid(gm.directions, present_nodes(gm)),
        "points" => _id_grid_metres(gm.points, present_nodes(gm))),   # plain metres
    _id_grid_extras(gm),
)
# the nominal pinhole, the domain and the held nodes (as runs) iff the model
# has them, under the identity's own names (fixed: a writer key renamed does
# not re-id a rig; a model holding every node has the identity it had before
# the presence mask)
function _id_grid_extras(gm::Union{CentralGenericModel, NoncentralGenericModel})
    d = _Id()
    P = present_nodes(gm)
    all(P) || (d["present"] = _present_runs(P))
    n = nominal_intrinsics(gm)
    n === nothing || (K = n.K; d["nominal_intrinsics"] = Any[Float64(K[1, 1]), Float64(K[2, 2]), Float64(K[1, 3]), Float64(K[2, 3]), Float64(K[1, 2])])
    e = gm.domain
    e isa WholePlane || (e isa Ellipse ?
        (d["domain"] = _Id("ellipse" => Any[Float64(e.center[1]), Float64(e.center[2]), Float64(e.a), Float64(e.b), Float64(e.θ)])) :
        throw(ArgumentError("content_id: a generic model's domain is an Ellipse or the whole plane; got a $(nameof(typeof(e)))")))
    return d
end

_lower(x::NamedTuple) = Dict{String, Any}(String(k) => _lower(v) for (k, v) in pairs(x))
_lower(x::AbstractDict) = Dict{String, Any}(String(k) => _lower(v) for (k, v) in x)
_lower(x::AbstractVector) = Any[_lower(v) for v in x]
_lower(x) = x

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
