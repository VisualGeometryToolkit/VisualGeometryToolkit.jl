# =============================================================================
# vgio/rig/v10.0 and vgio/target/v6.0 — VGC's `Rig` in every storage
# (sensors, targets, both) and VGC's target model, serialized
# =============================================================================
#
# The layouts are stated beside their code below (a target's body, the target
# kind, the rig kind). Every file opens with the envelope (validate.jl:
# `schema`, `generator`, the `units` and `conventions` objects) and may close
# with a free-form "provenance" object that VGIo never reads.
#
# A rig of targets is the DEFINITION the per-frame observations of
# `vgio/observations` (captures_schema.jl) resolve against: an observed target
# point is named by (target, index), its 1-based position in that target's
# points, and an observations file names the rig files it resolves in by their
# content identity ([`content_id`](@ref)).
#
# ONE AUTHORITY PER NAME: a pose's frames are given by where it sits — a
# sensor's by the rig `frame` and its `id`, a target's pose by its `id` and
# the rig `frame` — and are never restated beside it, so they cannot disagree
# with it. FRAME NAMES go through the frame grammar (frame_names.jl).
#
# UNCERTAINTY (VisualGeometryToolkitCore conventions/persisted-geometry.md): a pose
# may carry a 6×6 `covariance`, a point target's point its d×d `covariance`
# (`units.length`²) — each present iff the in-memory value is `Uncertain`. A rig sensor's extrinsics are a VGC
# `PosedCamera`'s map, never uncertain.
#
# UNITS: a stored rig is PHYSICAL — its target poses, extrinsics and target points
# are lengths (metres in memory, the declared `units.length` on disk); a writer
# refuses plain numbers where a length is meant.
#
# RIG SENSORS MUST BE BoundedCamera. A camera body requires the pixel extent,
# which a bare PosedCamera does not carry — wrap with `BoundedCamera(camera,
# extent)` before saving. Refused loudly, not defaulted.

_unique_check(xs, what) = allunique(xs) ? nothing : "$(what) must be unique"

# A TOML file's tree; a malformed file is an ArgumentError naming the path.
function _parse_toml_file(path::AbstractString)
    return try
        TOML.parsefile(path)
    catch e
        e isa TOML.ParserError ? throw(ArgumentError("$(path): not valid TOML ($(sprint(showerror, e)))")) : rethrow()
    end
end

# Read the schema a file declares, before knowing which kind to validate it as.
function _peek_schema(path::AbstractString)
    _is_toml(path) && return get(_parse_toml_file(path), "schema", nothing)
    d = _read_vgio_json(path)
    d isa AbstractDict || throw(ArgumentError("$(path): not a JSON object"))
    return get(d, :schema, nothing)
end

# -----------------------------------------------------------------------------
# A target <-> JSON: the body every target-carrying kind embeds
# -----------------------------------------------------------------------------
#
#   {"type": "point_target", "points": [{"xyz": [x, y, z], "covariance": …}, …]}           PointTarget (`xy` for a plane's)
#   {"type": "planar_target", "reference": "<reference>", "calibration": {calibration}}  PlanarTarget of a referenced design
#   {"type": "planar_target", "nominal": [[x, y], …], "calibration": {calibration}}      PlanarTarget of explicit points
#   {"type": "planar_target", "layout": {layout}, "calibration": {calibration}}         PlanarTarget(layout): a parametric design
#   … "board": {"size": {"width": w, "height": h}, "origin": [u, v]}               (optional) its physical board
#   … "canvas": {"size": {"width": w, "height": h}, "origin": [u, v],              (optional) the printed page,
#                "texture": {"path": "<relative path>", "sha256": "<hex>",          its printed image (optional):
#                            "size": {"width": w, "height": h}, "origin": [u, v]}}  the file and the rectangle it covers
#   … "placement": {"origin": [u, v], "side": s}                                   (optional, planar only) VGC's placement (a SimilarityMap{3})
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
# THE LAYOUT (target v4 / rig v7) — VGC's `AbstractTargetLayout`, the design a
# `PlanarTarget(layout; …)` was built from, which it records (`layout(t)`): a
# target built from one is written by it, its points VGC's from the layout and
# never restated. Its `type` is the layout's tag in VGC's `TARGET_LAYOUT_TAGS`
# (snake_case of the type's name), lengths in the file's `length_unit`:
#
#   {"type": "april_grid", "tag_rows": r, "tag_cols": c, "tag_size": s, "tag_spacing": g,
#    "apriltag_family": "tag36h11", "first_id": 0}                         VGC's AprilGrid (g: the gap, a length)
#   {"type": "schoeps_pattern", "squares_x": nx, "squares_y": ny, "square_length": s,
#    "star_segments": 16, "apriltag_family": "tag36h11",
#    "apriltags": [{"x", "y", "width", "height", "id"}, …]}                  VGC's SchoepsPattern (squares, y down as printed)
#
# Every key is written, defaults included. A body states its design by exactly
# one of `reference`, `nominal` and `layout`. A layout target's identity
# (`content_id`) is of its points, as a nominal target's: the layout is not
# hashed, so a layout file and a nominal file of the same design name one target.
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

# A point target's points: each `{"xy": [x, y]}` (a plane's) or `{"xyz": [x, y,
# z]}`, all of one dimension, lengths in `units.length`, and its `covariance`
# (d×d, `units.length`²) iff the point is `Uncertain` — every point or none (a
# file could not say which is which otherwise). The one point vocabulary, an
# observation's too.
const _TARGET_POINT = _Either(Pair{Symbol, _Spec}[
    :xy => _Obj(Pair{Symbol, _Spec}[:xy => _XY], Pair{Symbol, _Spec}[:covariance => _COV2]),
    :xyz => _Obj(Pair{Symbol, _Spec}[:xyz => _XYZ], Pair{Symbol, _Spec}[:covariance => _COV3]),
])
const _TARGET_POINTS = _CheckedSpec(
    _Vec(_TARGET_POINT),
    p -> isempty(p) ? nothing :
        !(all(q -> haskey(q, :xy), p) || all(q -> haskey(q, :xyz), p)) ? "points are all `xy` (a plane's) or all `xyz`" :
        !(all(q -> haskey(q, :covariance), p) || !any(q -> haskey(q, :covariance), p)) ? "every point states its `covariance`, or none does" :
        nothing,
)

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
        c isa AbstractDict && return (haskey(c, :diagonal) && length(c[:diagonal]) != n) ?
            "`covariance` is over the calibration's $(n) parameters, its diagonal has $(length(c[:diagonal]))" : nothing
        (length(c) == n && all(r -> length(r) == n, c)) ||
            return "`covariance` is $(n)×$(n), over the calibration's parameters"
        all(c[i][j] == c[j][i] for i in 1:n, j in 1:n) || return "`covariance` must be symmetric"
    end
    return nothing
end
const _CALIBRATION_FIELDS = Pair{Symbol, _Spec}[:nodes => _Vec(_XY), :parameters => _Vec(_Real()), :covariance => _Cov(0)]
# (`members`, the retired record's key, is admitted here only to be refused by
# name in `_target_calibration_check`)
const _TARGET_CALIBRATION_MEMBER = _Obj(Pair{Symbol, _Spec}[:type => _NonEmpty()], vcat(_CALIBRATION_FIELDS, Pair{Symbol, _Spec}[:compose => _Vec(_Free()), :members => _Vec(_Free())]))
const _TARGET_CALIBRATION = _CheckedSpec(
    _Obj(Pair{Symbol, _Spec}[:type => _NonEmpty()], vcat(_CALIBRATION_FIELDS, Pair{Symbol, _Spec}[:compose => _Vec(_TARGET_CALIBRATION_MEMBER), :members => _Vec(_Free())])),
    x -> _target_calibration_check(x, false),
)

const _BOARD = _Obj(
    :size => _CheckedSpec(_Obj(:width => _Real(), :height => _Real()), z -> (z[:width] > 0 && z[:height] > 0) ? nothing : "a board's width and height are > 0"),
    :origin => _XY,
)

const _SIZE_WH = _Obj(:width => _Real(), :height => _Real())
const _CANVAS_TEXTURE = _CheckedSpec(
    _Obj(
        :path => _Named(p -> !isempty(p) && !isabspath(p), "a path relative to the file that names it"),
        :sha256 => _Named(s -> occursin(r"^[0-9a-f]{64}$", s), "a SHA-256 hex digest (64 lowercase hex digits)"),
        :size => _SIZE_WH, :origin => _XY,
    ),
    z -> (z[:size][:width] > 0 && z[:size][:height] > 0) ? nothing : "a canvas texture's width and height are > 0",
)
const _CANVAS = _CheckedSpec(
    _Obj(Pair{Symbol, _Spec}[:size => _SIZE_WH, :origin => _XY], Pair{Symbol, _Spec}[:texture => _CANVAS_TEXTURE]),
    z -> (z[:size][:width] >= 0 && z[:size][:height] >= 0) ? nothing : "a canvas's width and height are ≥ 0",
)

const _PLACEMENT = _CheckedSpec(
    _Obj(:origin => _XY, :side => _Real()),
    z -> (isfinite(z[:side]) && z[:side] > 0) ? nothing : "a chart placement's side is > 0",
)

# A layout record's keys besides `type` (VGC's `TARGET_LAYOUT_TAGS` gives the
# type for the record's tag): every one is written, in this order.
_layout_keys(::Type{<:AprilGrid}) = (:tag_rows, :tag_cols, :tag_size, :tag_spacing, :apriltag_family, :first_id)
_layout_keys(::Type{<:SchoepsPattern}) = (:squares_x, :squares_y, :square_length, :star_segments, :apriltag_family, :apriltags)
const _SCHOEPS_TAG = _Obj(:x => _Int(), :y => _Int(), :width => _PosInt(), :height => _PosInt(), :id => _Int())
const _TARGET_LAYOUT = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:type => _NonEmpty()],
        Pair{Symbol, _Spec}[
            :tag_rows => _PosInt(), :tag_cols => _PosInt(), :tag_size => _Real(), :tag_spacing => _Real(), :first_id => _Int(),
            :squares_x => _PosInt(), :squares_y => _PosInt(), :square_length => _Real(), :star_segments => _PosInt(),
            :apriltags => _Vec(_SCHOEPS_TAG), :apriltag_family => _NonEmpty(),
        ],
    ),
    function (x)
        ty = x[:type]
        haskey(TARGET_LAYOUT_TAGS, ty) || return "unknown layout type $(repr(ty)) (one of $(join(keys(TARGET_LAYOUT_TAGS), ", ")))"
        want = _layout_keys(TARGET_LAYOUT_TAGS[ty])
        for k in keys(x)
            Symbol(k) in (:type, want...) || return "a $(ty) layout has no `$(k)` (its keys: $(join(want, ", ")))"
        end
        for k in want
            haskey(x, k) || return "a $(ty) layout states its `$(k)`"
        end
        return nothing
    end,
)

# A planar body's design: exactly one of these (a reference, nominal points, a layout).
# The body tags follow the tag rule, snake_case of the type's name.
const _POINT_TARGET_TAG = VgtkCore.snake_case(String(nameof(PointTarget)))     # "point_target"
const _PLANAR_TARGET_TAG = VgtkCore.snake_case(String(nameof(PlanarTarget)))   # "planar_target"
const _DESIGN_KEYS = (:reference, :nominal, :layout)

const _TARGET_BODY = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[:type => _NonEmpty()],
        Pair{Symbol, _Spec}[
            :points => _TARGET_POINTS,
            :reference => _NonEmpty(), :nominal => _Vec(_Vec(_Real(), 2)), :layout => _TARGET_LAYOUT, :calibration => _TARGET_CALIBRATION,
            :board => _BOARD, :canvas => _CANVAS, :placement => _PLACEMENT,
        ],
    ),
    function (t)
        ty = t[:type]
        if ty == _POINT_TARGET_TAG
            haskey(t, :points) || return "a point_target states its `points`"
            haskey(t, :placement) && return "a point_target has no `placement` (a chart placement is a planar target's: where its calibration's chart lies)"
            any(k -> haskey(t, k), (_DESIGN_KEYS..., :calibration, :board, :canvas)) && return "a point_target has `points` only"
            return nothing
        elseif ty == _PLANAR_TARGET_TAG
            haskey(t, :points) && return "a planar_target has `reference`, `nominal` or `layout`, and `calibration`"
            haskey(t, :calibration) || return "a planar_target states its `calibration`"
            count(k -> haskey(t, k), _DESIGN_KEYS) == 1 ||
                return "a planar target states its design by exactly one of its `reference`, its `nominal` points or its `layout` " *
                "(a referenced design's points are its owner's, a layout's are VGC's from the layout)"
            return nothing
        end
        ty in ("points", "planar") &&
            return "the target type $(repr(ty)) is the IO 0.9 tag, read only through its version step: a current record says \"$(_POINT_TARGET_TAG)\" or \"$(_PLANAR_TARGET_TAG)\" (the tag rule)"
        return "unknown target type $(repr(ty)) ($(_POINT_TARGET_TAG), $(_PLANAR_TARGET_TAG))"
    end,
)

# A target's points on the wire (`_TARGET_POINTS`): each its lengths under `xy`
# or `xyz`, by its dimension, and its covariance iff every point is `Uncertain`
# (plain and uncertain points are not mixed in one target).
_point_key(n::Int) = n == 2 ? :xy : n == 3 ? :xyz : throw(ArgumentError("a point is 2-D (`xy`) or 3-D (`xyz`), got $(n) coordinates"))
function _points_json(what, pts, unit::_LeafUnit)
    n_u = count(p -> p isa Uncertain, pts)
    (n_u == 0 || n_u == length(pts)) || throw(
        ArgumentError("$(what): its points are partly Uncertain — make all of them Uncertain (exact where known) or none")
    )
    return (points = map(pts) do p
        g = geom(p)
        r = NamedTuple{(_point_key(length(g)),)}((_lengths_json(g, unit, "$(what): a point"),))
        p isa Uncertain ? merge(r, (covariance = _length_cov_json(cov(p), unit, _point_cov_dims(length(g))),)) : r
    end,)
end

# `points` back to static vectors of lengths (metres) of the recorded
# dimension, the form VGC composes a target's points with its pose in
# (`points(rig, id)`); `Uncertain` where the file states a covariance
# (unit² → m²).
function _points_from_json(vals, unit::Units)
    isempty(vals) && return SVector{2, Float64}[]
    k = haskey(first(vals), :xy) ? :xy : :xyz
    n = k === :xy ? 2 : 3
    return map(vals) do q
        p = _lengths_from_json(Val(n), q[k]::Vector{Any}, unit)
        haskey(q, :covariance) ? Uncertain(p, CovMat(_length_cov_from_json(q.covariance, n, unit, _point_cov_dims(n)))) : p
    end
end
# A planar target's nominal points (positional `[x, y]` arrays, never uncertain) back as lengths.
_nominal_from_json(vals, unit::Units) = [_lengths_from_json(Val(2), v::Vector{Any}, unit) for v in vals]

# A plain metre value (a calibration's stored length) as the file's number in `u`.
_metres_json(x::Real, u::_LeafUnit) = _leaf_power(Float64(x), u, 1)

# Where a calibration's lengths go: the file's unit or the identity's persisted
# metres (`_LeafUnit`), or — for a PLACED target (VGC's placement, a `SimilarityMap{3}`),
# whose calibration acts in the board's dimensionless chart — `_CHART`: its
# nodes, values and covariance are plain chart numbers, written and read as
# they are, never scaled by `length_mask` (the placement carries the lengths).
struct _Chart end
const _CHART = _Chart()
const _CalUnit = Union{_LeafUnit, _Chart}
_metres_json(x::Real, ::_Chart) = Float64(x)
_length_cov_json(Σ::AbstractMatrix, ::_Chart, dims) = _cov_wire((x, n) -> x, Σ, dims)
_length_cov_from_json(c, n::Int, ::_Chart, dims) = _cov_from_wire((x, k) -> Float64(x), c, n, dims)
_nodes_json(c, u::_CalUnit) = [[_metres_json(p[1], u), _metres_json(p[2], u)] for p in c.system.nodes]

# A planar target's calibration on IO's style (style.jl), the PARAMETRIC
# RECORD: its kind's tag in VGC's `TARGET_CALIBRATION_TAGS`, its
# `parameter_vector` (lengths by `length_mask` in the style's length form), a
# thin-plate kind's `nodes` beside them; a composition its members' records,
# outer first; an `Uncertain` one its covariance beside. A calibration with
# lengths is stored physical: plain numbers are refused, as a target's plain
# points are (one without lengths — Rigid, Scale, Aspect, Skew and their
# compositions — has no form to refuse; a placed target's, `_CHART`, is plain
# chart numbers, which VGC refuses units on).
#
# A named function, called by the target's lower, and NOT a `JSON.lower`
# method: VGC's `TargetCalibration` union holds every planar `EuclideanMap`
# (a pixel-frame conversion among them) and every composition, so a `lower` on
# it would write any such map left in a body as a target calibration.
function _target_calibration_json(s::VgioStyle, c::TargetCalibration)
    u = s.length_unit
    u isa _Chart || (any(length_mask(c)) && !has_units(c) && _refuse_plain("a $(nameof(typeof(c))) target calibration"))
    c isa ComposedTransformation && return (type = _COMPOSITION_TYPE, compose = [_target_calibration_json(s, l) for l in members(c)])
    r = _parametric_record(calibration_tag(typeof(c)), c, u; structure = _calibration_structure(c, u))
    _check_writes_back(c, r, u)
    return r
end
_target_calibration_json(s::VgioStyle, c::Uncertain{<:TargetCalibration}) =
    merge(_target_calibration_json(s, geom(c)), (covariance = _length_cov_json(cov(c), s.length_unit, length_mask(geom(c))),))
# What a kind's type does not fix: a thin-plate kind's nodes.
_calibration_structure(c, u) = (;)
_calibration_structure(c::Union{ThinPlateSpline, HeightField}, u) = (nodes = _nodes_json(c, u),)
# The writer's half of the reader's pure-bending rule (`_reads_back`): a field
# pure bending neither in the file's numbers nor in metres would not read
# back, so it is refused here, naming the unit to write it in. (The identity
# tree, `_PERSISTED`, is never read.)
_check_writes_back(c, r, u) = nothing
function _check_writes_back(c::Union{ThinPlateSpline, HeightField}, r, u::Union{Units, _Chart})
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
# A calibration's length as read: a length in metres from the file's unit, or
# (`_CHART`, a placed target's) the plain chart number as stored.
_read_metres(x, u::Units) = read_length(x::Real, u)
_read_metres(x, ::_Chart) = Float64(x::Real)
const _ReadUnit = Union{Units, _Chart}

# A kind back from its record (by the kind's TYPE, which VGC's registry gives
# for the record's tag): the parametric record's `with_parameters` on the kind's
# prototype — the angle of a planar EuclideanMap is the stored one, VGC's
# `Angle2d`, so a value read back is its file's, bit for bit — and a thin-plate
# kind by its own constructor, on the nodes its record states, unprojected.
_calibration_from_record(K::Type, x, u::_ReadUnit) = _from_parametric_record(_calibration_prototype(planar_kind(K), u), x, u)
_calibration_prototype(::Type{K}, u) where {K <: Union{Rigid, Scale, Aspect{2}}} = K()
_calibration_prototype(::Type{K}, u) where {K <: Skew} = Skew{2}()
_calibration_prototype(::Type{<:EuclideanMap{2}}, u) =   # physical iff the file's numbers are lengths (not a placed chart's)
    EuclideanMap(Angle2d(0.0), u isa _Chart ? SVector(0.0, 0.0) : SVector(0.0, 0.0) .* m; check = false)
function _calibration_from_record(T::Type{<:Union{ThinPlateSpline, HeightField}}, x, u::_ReadUnit)
    system = _check_pure_bending(T, x.nodes, x.parameters, u)
    v, K = [_read_metres(p, u) for p in x.parameters::Vector{Any}], length(x.nodes)   # a spline's fx then fy; the heights
    return T === ThinPlateSpline ? ThinPlateSpline(system, v[1:K], v[(K + 1):end]; project = false) : HeightField(system, v; project = false)
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
    in_metres = ThinPlateSplineSystem([SVector(metres(q[1], u), metres(q[2], u)) for q in nodes])
    return pure_in_file || _pure_blocks(in_metres, [metres(v, u) for v in p]) ? in_metres : nothing
end
"""
    metres(x::Real, u::Units) -> Float64

The file number `x`, a length in the unit `u`, as plain metres:
[`read_length`](@ref)`(x, u)` stripped (decimal-exact for a power-of-ten
unit), what a record's reader hands a kind built on plain numbers (a
thin-plate kind's nodes and values, a non-central aperture's radii). A
number of a placed target's chart (no unit) is returned as it is.
"""
metres(x, u::Units) = strip_or_bare(m, read_length(x::Real, u))
metres(x, ::_Chart) = Float64(x::Real)
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
# states a covariance (unit^(lᵢ+lⱼ) → m^(lᵢ+lⱼ)): the target lift's, the
# inverse of `_target_calibration_json`.
function _target_calibration_from_json(s::VgioStyle, x::Record)
    u = s.length_unit
    c = _bare_calibration_from_json(x, u)
    haskey(x, :covariance) || return c
    dims = length_mask(c)
    return Uncertain(c, CovMat(_length_cov_from_json(x.covariance, length(dims), u, dims)))
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
target_json(t::PointTarget, unit::_LeafUnit = mm) = JSON.lower(VgioStyle(; length_unit = unit), t)
target_json(t::PlanarTarget, unit::_LeafUnit = mm) = JSON.lower(VgioStyle(; length_unit = unit), t)

# A target on IO's style (style.jl): its body, lengths in the style's length form.
JSON.lower(s::VgioStyle, t::PointTarget) = merge((type = _POINT_TARGET_TAG,), _points_json("a point_target", points(t), s.length_unit))
function JSON.lower(s::VgioStyle, t::PlanarTarget)
    unit, pl = s.length_unit, placement(t)
    cs = pl === nothing ? s : _with_length(s, _CHART)          # a placed calibration: plain chart numbers
    return merge(
        (type = _PLANAR_TARGET_TAG,), _design_json(s, t), (calibration = _target_calibration_json(cs, calibration(t)),),
        _board_json(board(t), unit),
        _canvas_json(canvas(t), unit),
        _placement_json(t, pl, unit),
    )
end
# The design: its reference, its layout, or its nominal points. The identity
# tree (`_PERSISTED`) takes a layout target's points, so its identity is
# geometric — a nominal target's of the same points.
function _design_json(s::VgioStyle, t::PlanarTarget)
    unit = s.length_unit
    reference(t) === nothing || return (reference = String(reference(t)),)
    (layout(t) !== nothing && !(unit isa _Persisted)) && return (layout = JSON.lower(s, layout(t)),)
    return (nominal = [_lengths_json(p, unit, "a planar target's nominal point") for p in design(t)],)
end

# A layout ↔ its record (`_TARGET_LAYOUT`), on IO's style: its tag in VGC's
# `TARGET_LAYOUT_TAGS` and every key (`_layout_keys`), each the field of that
# name — a length (`_layout_lengths`) in the file's unit through `write_length`
# / `read_length`, a Schöps tag as its five integers, any other as it is — and
# read back by the layout's keyword constructor. A layout of plain numbers is
# refused, as plain points are.
_layout_lengths(::Type{<:AprilGrid}) = (:tag_size, :tag_spacing)
_layout_lengths(::Type{<:SchoepsPattern}) = (:square_length,)
function JSON.lower(s::VgioStyle, L::AbstractTargetLayout)
    has_units(L) || _refuse_plain("a target's $(nameof(typeof(L))) layout")
    value(k) = (v = getfield(L, k); k in _layout_lengths(typeof(L)) ? write_length(v, s.length_unit) :
        k === :apriltags ? [(; t.x, t.y, t.width, t.height, t.id) for t in v] : v)
    return merge((type = tag_of(TARGET_LAYOUT_TAGS, L),), NamedTuple(k => value(k) for k in _layout_keys(typeof(L))))
end
function JSON.lift(s::VgioStyle, ::Type{AbstractTargetLayout}, x::Record)
    T = TARGET_LAYOUT_TAGS[x.type::String]
    value(k) = (v = x[k]; k in _layout_lengths(T) ? read_length(v::Real, s.length_unit) :
        k === :apriltags ? [(x = t.x::Integer, y = t.y::Integer, width = t.width::Integer, height = t.height::Integer, id = t.id::Integer) for t in v] : v)
    return T(; (k => value(k) for k in _layout_keys(T))...)
end

# The placement ↔ its record (VGC's placement, a `SimilarityMap{3}` of scale
# `side`, the identity rotation and the translation `(origin, 0)`: the chart
# point c is the design point origin + side·c): its origin and side, lengths in
# the file's unit, decimal-exactly; read back as plain metres, the placement of
# the physical target the record describes (the keys of the 0.8 record).
_placement_json(_, ::Nothing, ::_LeafUnit) = (;)
function _placement_json(t, P::SimilarityMap{3}, u::_LeafUnit)
    # the placement carries no unit: its numbers are metres only on a physical target (its points, board or canvas)
    has_units(t) || _refuse_plain("a placed target (its placement's numbers are metres only on a target with lengths)")
    o = translation(P)
    return (placement = (origin = [_metres_json(o[1], u), _metres_json(o[2], u)], side = _metres_json(scale(P), u)),)
end
function _placement_from_json(x, u::Units)
    metres(v) = ustrip(m, read_length(v::Real, u))
    return SimilarityMap(metres(x.side), RotMatrix{3}(SMatrix{3, 3}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)), SVector(metres(x.origin[1]), metres(x.origin[2]), 0.0))
end

# The board ↔ its record. VGC's `Board` holds its size as lengths (metres) and
# its lower-left corner as plain metres (a physical target's; a unitless one is
# in no file): a rectangle of the design chart, as the canvas's and its
# texture's are — the size in the Size2 form by `write_length`/`read_length`,
# the corner as plain metres in the file's unit, decimal-exactly.
_board_json(::Nothing, ::_LeafUnit) = (;)
_board_json(bd::Board, u::_LeafUnit) = (has_units(bd) || _refuse_plain("a target's board"); (board = _rect_json(bd, u),))
_board_from_json(x, u::Units) = Board(_rect_from_json(x, u)...)

# The canvas ↔ its record: the board's rule for each rectangle, the texture's
# path and digest as stored.
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
target_from_json(t, unit::AbstractString; resolve = nothing) =
    JSON.lift(VgioStyle(; length_unit = parse_unit(unit)), AbstractTarget, t::Record; resolve)

# A target body back on IO's style (the inverse of its `lower`), its lengths in
# the style's unit; `resolve` as `target_from_json`'s.
function JSON.lift(s::VgioStyle, ::Type{AbstractTarget}, t::Record; resolve = nothing)
    u = s.length_unit
    t.type == _POINT_TARGET_TAG && return PointTarget(_points_from_json(t.points::Vector{Any}, u))
    pl = haskey(t, :placement) ? _placement_from_json(t.placement, u) : nothing
    c = _target_calibration_from_json(pl === nothing ? s : _with_length(s, _CHART), t.calibration::Record)   # placed: plain chart numbers
    bd = haskey(t, :board) ? _board_from_json(t.board, u) : nothing
    cv = haskey(t, :canvas) ? _canvas_from_json(t.canvas, u) : nothing
    haskey(t, :nominal) && return PlanarTarget(nothing, _nominal_from_json(t.nominal::Vector{Any}, u), c, bd, cv, pl)
    haskey(t, :layout) && return PlanarTarget(JSON.lift(s, AbstractTargetLayout, t.layout::Record); calibration = c, board = bd, canvas = cv, placement = pl)
    ref = Symbol(t.reference::String)
    nominal = resolve === nothing ? nothing : resolve(ref)
    (nominal === nothing || isempty(nominal) || has_units(nominal)) ||
        _refuse_plain("resolve($(repr(ref))): the design's nominal points")
    return PlanarTarget(ref, nominal, c, bd, cv, pl)
end

# -----------------------------------------------------------------------------
# vgio/target/v6.0 ↔ one target (AbstractTarget, or one Attributed with its calibration record)
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/target/v6.0", <envelope>, "target": {target body},
#  "calibration_record": {…}}             (optional: who calibrated it — the cameras' block's keys)

const _TARGET_SPEC = _kind_spec("target", Pair{Symbol, _Spec}[:target => _TARGET_BODY], Pair{Symbol, _Spec}[:calibration_record => _CALIBRATION])

"""
    save(path, target::AbstractTarget; length_unit = mm, provenance = (;))
    save(path, Attributed(target, (; calibration_record)); length_unit = mm, provenance = (;))

Write one physical target as a `vgio/target/v6.0` file — its body (rig_schema.jl:
its points — a planar target's by its reference, its layout when it records
one (`layout(t)`), or its nominal points — and a planar target's calibration with its parameters and, when
`Uncertain`, their covariance) in `length_unit` and, for an `Attributed` one,
its `calibration_record`: who calibrated it, when, from what (the keys of a
camera's `calibration_record`). Plain points are an `ArgumentError`, and so is a
`PosedTarget`: the file holds an unposed target, so save its `model`, or the
posed one inside a `Rig`.

The calibration record is not the target's calibration: `calibration(t)` is
the `TargetCalibration`, `x.calibration_record` the record of how it was
obtained.
"""
save(path::AbstractString, t::AbstractTarget; length_unit::Units = mm, provenance::NamedTuple = (;)) =
    (_write_kind(path, "target", (target = t,), provenance; style = VgioStyle(; length_unit)); nothing)   # the target lowered on the file's style
function save(
        path::AbstractString, t::Attributed{<:AbstractTarget, <:NamedTuple{(:calibration_record,)}};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    body = (target = t.value, calibration_record = _calibration_json(t.calibration_record, "a target's calibration record"))
    _write_kind(path, "target", body, provenance; style = VgioStyle(; length_unit))
    return nothing
end
# A `vgio/target/v5` file holds an UNPOSED target: a posed one is refused here,
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

Read a `vgio/target/v6.0` file: the target, `Attributed` with its
`calibration_record` where the file states one. `resolve(reference)` supplies
a referenced design's nominal points (else the target is returned
unresolved). An earlier version (target v1–v5) is read through its version
steps (`migrate_record`). A Schöps pattern YAML or a Kalibr aprilgrid YAML is
read as its layout's `PlanarTarget` (see `load(PlanarTarget, path)`); the form
is decided by the file's content.
"""
function load(::Type{AbstractTarget}, path::AbstractString; resolve = nothing)
    _file_form(path) === :yaml && return _load_layout_yaml(path)
    d = _read_store(path, "target")
    t = JSON.lift(VgioStyle(d), AbstractTarget, d.target::Record; resolve)
    return haskey(d, :calibration_record) ?
        Attributed(t, (; calibration_record = _calibration_from_json(d.calibration_record::AbstractDict))) : t
end

# -----------------------------------------------------------------------------
# vgio/rig/v10.0 ↔ Rig, every storage
# -----------------------------------------------------------------------------
#
# {"schema": "vgio/rig/v10.0", <envelope>, "frame": "body:camera_rig",
#  "sensors": [{"id": "cam1", "camera": {camera body | non-central body},
#               "extrinsics": {rotation, translation}}],               extrinsics: `frame` → view:<id>
#  "targets": [{"id": "<uid>", "target": {target body},
#               "pose": {rotation, translation[, covariance]}}]}       pose: target:<id> → `frame`
#
# A sensor rig has `sensors` only, a rig of targets `targets` only (a board
# rig, a motion-capture body's constellation, the boards of a calibration
# target gauged on one of them), a camera rig carrying its constellation both.
# Ids are unique across the two. A sensor is a `BoundedCamera` (a camera body
# requires the pixel extent), its body the camera store's (camera_schema.jl:
# its pitch, its lens coverage, a non-central camera told apart by its `type`);
# its extrinsics are a plain-typed `EuclideanMap` (never `Uncertain`); a target
# pose may be `Uncertain`. Every translation and point is a length.

const _SENSOR_RECORD = _Obj(
    :id => _NonEmpty(),
    :camera => _BOUNDED_CAMERA_BODY,
    :extrinsics => _POSE,
)
const _MEMBER_RECORD = _Obj(Pair{Symbol, _Spec}[:id => _NonEmpty(), :target => _TARGET_BODY, :pose => _UPOSE], Pair{Symbol, _Spec}[])
const _RIG_SPEC = _kind_spec(
    "rig", Pair{Symbol, _Spec}[:frame => _FRAME_NAME], Pair{Symbol, _Spec}[:sensors => _Vec(_SENSOR_RECORD), :targets => _Vec(_MEMBER_RECORD)];
    check = function (x)
        ss = get(x, :sensors, Any[])::AbstractVector
        ts = get(x, :targets, Any[])::AbstractVector
        (haskey(x, :sensors) && isempty(ss)) && return "`sensors` is omitted, not empty"
        (haskey(x, :targets) && isempty(ts)) && return "`targets` is omitted, not empty"
        (isempty(ss) && isempty(ts)) && return "a rig has sensors, targets, or both"
        return _unique_check(vcat([s[:id] for s in ss], [t[:id] for t in ts]), "member ids")
    end,
)

function _sensor_json(s::VgioStyle, rig, i, id)
    c = sensor_rig(rig).members[i]
    return (id = String(id), camera = _sensor_body(s, c, i, id), extrinsics = JSON.lower(s, extrinsics(sensor_rig(rig), i)))   # <rig frame> → view:<id>
end
# A sensor's camera body: a camera's (a CameraModel's, of any lens) or a non-central
# grid camera's, its extent, its sensor's pitch and its lens coverage — the record's
# and, on the identity's style, the identity tree's.
function _sensor_body(s::VgioStyle, c, i, id)
    c isa BoundedCamera || throw(
        ArgumentError("rig sensor $i ($(id)) carries no pixel extent — wrap it in BoundedCamera(camera, extent) before saving")
    )
    return _bounded_body(s, c)
end

# a rig member's physical sensor, if it is bounded by one
_sensor_of(c::BoundedCamera) = sensor(c)
_sensor_of(c) = nothing

_has_sensors(::Rig{F, <:AbstractDict}) where {F <: Frame{3}} = false
_has_sensors(::Rig) = true
_has_targets(::Rig{F, <:Tuple}) where {F <: Frame{3}} = false
_has_targets(::Rig) = true

# The rig's body — what the file holds after its envelope, and, on the
# identity's style (`_ID_STYLE`), its identity tree — its lengths on `s`.
function _rig_body(rig::Rig, s::VgioStyle)
    body = (frame = frame_name(reference_frame(rig)),)
    if _has_sensors(rig)
        body = merge(body, (sensors = [_sensor_json(s, rig, i, id) for (i, id) in enumerate(ids(sensor_rig(rig)))],))
    end
    if _has_targets(rig)
        tr = target_rig(rig)
        targets = [(id = String(id), target = JSON.lower(s, tr[id].target), pose = JSON.lower(s, pose(tr, id))) for id in _id_order(tr)]   # target:<id> → <rig frame>
        body = merge(body, (targets = targets,))
    end
    return body
end
# The order a rig's targets are written and hashed in: by id.
_id_order(tr) = sort!(collect(ids(tr)))

"""
    save(path, rig::Rig; length_unit = mm, provenance = (;))

Write a physical VGC `Rig` — sensors, targets, or both — as a `vgio/rig/v10.0`
file in `length_unit`: each sensor (a `BoundedCamera` of a `CameraModel` — a
`SplineLens` one included — or of a `NoncentralSplineModel`, with its lens
coverage) with its extrinsics `frame → view:<id>`, each target (sorted by id)
with its body and its pose `target:<id> → frame` (a `covariance` iff
`Uncertain`). A rig of plain translations or points is an `ArgumentError`
(attach the unit, `t .* m`). A rig's name, if it has one worth keeping,
belongs in `provenance`.
"""
function save(path::AbstractString, rig::Rig; length_unit::Units = mm, provenance::NamedTuple = (;))
    s = VgioStyle(; length_unit)
    _write_kind(path, "rig", _rig_body(rig, s), provenance; style = s)
    return nothing
end

"""
    load(Rig, path; resolve = nothing, intrinsics = nothing) -> Rig

Read a `vgio/rig/v10.0` file back into a fully labeled VGC `Rig` of the storage
it holds, its lengths in metres: sensors (each a `BoundedCamera` labeled `frame → ViewFrame{id}`),
targets (each a `PosedTarget`, minted `TargetFrame{id} → frame` on access), or both.
`resolve(design)` supplies a referenced planar design's nominal points (see
[`target_from_json`](@ref)). Type-unstable by nature (frame types are minted
from file data) — load once, dispatch after. An earlier version (rig v4–v9) is
read through its version steps (`migrate_record`): a v8 sensor's old grid body
as its spline camera. `intrinsics` (sensor id → `CameraCalibrationMat`, in the
canonical pixel frame) supplies K to an old grid sensor that stores no nominal
pinhole — one whose lattice has no complete 4 × 4 window at the gauge's pixel is
refused without it — read as if the file held it (the file is never edited); a
sensor that holds its K, or an unknown id, refuses it.
"""
load(::Type{Rig}, path::AbstractString; resolve = nothing, intrinsics::Union{Nothing, AbstractDict{Symbol, <:CameraCalibrationMat}} = nothing) =
    _rig_from_record(_checked_store(_with_given_intrinsics(_read_vgio_json(path), intrinsics, path), path, "rig"); resolve)

function _rig_from_record(d; resolve = nothing)
    F = frame_from_name(d.frame::String)
    s = VgioStyle(d)
    haskey(d, :targets) || return Rig{F}((_load_sensors(d, F, s)...,))
    members = Dict{Symbol, PosedTarget}(
        Symbol(t.id::String) => PosedTarget(JSON.lift(s, AbstractTarget, t.target::Record; resolve), JSON.lift(s, EuclideanMap{3}, t.pose::Record))
            for t in d.targets::Vector{Any}
    )
    targets = Dict{Symbol, typejoin(map(typeof, values(members))...)}(members)
    haskey(d, :sensors) || return Rig{F}(targets)
    return Rig{F}((_load_sensors(d, F, s)...,), targets)
end

function _load_sensors(d, ::Type{F}, st::VgioStyle) where {F}
    return map(d.sensors::Vector{Any}) do s
        Xl = relabel(F, ViewFrame{Symbol(s.id::String)}, JSON.lift(st, EuclideanMap{3}, s.extrinsics::Record))
        _bounded_from_body(st, s.camera::Record, Xl)
    end
end

"""
    content_id(targets) -> String

The identity an observations file (`vgio/observations`, `target_homographies`)
names a target definition by — `targets` a `Rig`, a single unposed
`AbstractTarget` (`Attributed` ones by their target), or a `vgio/rig` or
`vgio/target` path (any version that reads): the SHA-256 (lowercase hex) of the
RFC 8785 (JCS) serialization ([`canonical_json`](@ref)) of its IDENTITY TREE.

The identity tree is specified once, in the docs ("Content identity"), and is
the writer's own record of the value with no envelope: a rig's is
`{"frame", "sensors"?, "targets"?}` — its body as `vgio/rig` holds it, under the
current names (the targets sorted by id, each `{"id", "target", "pose"}`, each
sensor `{"id", "camera", "extrinsics"}`) — and a single target's its `target`
body alone (so it differs from every rig's, which has `frame`). Every number is
in BASE UNITS: a length in metres, a covariance entry in m^k, a pixel in the
canonical (colmap) pixel frame, an angle in radians. Each length is normalised
to the Float64 nearest to its value rounded to 15 significant decimal digits in
metres (`_persisted`): every decimal file number of at most 15 digits, in any
decimal unit, reads back as exactly such a value, so a rig, the file written
from it in any decimal `units.length`, the rig loaded back from that file, and
a value built in code from the same decimals (`14.2mm`, an ulp off) have one
identity. No schema version, envelope or provenance is hashed: the identity does
not change when the file format does, except by a deliberate re-id (IO 0.10's,
from the 0.9 tree; `vgio_migrate_tree` repoints a tree's observation files and
keeps the id map). A planar target built from a layout enters by its nominal
points, as a target given by its points does (the layout is not hashed). A
pose's rotation enters as the quaternion the writer stores (w-x-y-z, w ≥ 0; a
`QuatRotation` as held, any other rotation converted once): mint an identity
from what you save. A `PosedTarget` is an `ArgumentError`.
"""
content_id(x::Union{Rig, AbstractTarget, Attributed{<:AbstractTarget}, AbstractString}) = _targets_hash(x)

# A path is a `vgio/rig` file or a `vgio/target` file (by its schema); a referenced design stays a reference.
function _targets_hash(path::AbstractString)
    v = _split_schema(_peek_schema(path))
    v !== nothing && v.kind == "target" && return _targets_hash(load(AbstractTarget, path))
    return _targets_hash(_rig_from_record(_read_store(path, "rig")))
end
_targets_hash(rig::Rig) = sha256_hex(canonical_json(_identity_tree(rig)))
_targets_hash(t::Attributed{<:AbstractTarget}) = _targets_hash(t.value)   # who calibrated it is not part of what it is
_targets_hash(t::AbstractTarget) = sha256_hex(canonical_json(_identity_tree(t)))
_targets_hash(::PosedTarget) = throw(
    ArgumentError("content_id: a PosedTarget has a pose, which an unposed target's identity does not hold — name `model(pt)`, or the rig it sits in")
)

# The identity tree (content_id's docstring; the docs' "Content identity"): the
# writer's own record, on the identity's style — every length leaf its persisted
# metre value (`_PERSISTED`), the canonical pixel frame — lowered to plain JSON.
const _ID_STYLE = VgioStyle(; length_unit = _PERSISTED)
_identity_tree(rig::Rig) = _lower(JSON.parse(JSON.json(_rig_body(rig, _ID_STYLE); style = _ID_STYLE)))
_identity_tree(t::AbstractTarget) = _lower(JSON.parse(JSON.json(JSON.lower(_ID_STYLE, t); style = _ID_STYLE)))
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
function _target_rig(p::AbstractString; resolve = nothing)
    v = _split_schema(_peek_schema(p))
    v !== nothing && v.kind == "target" && return _target_rig(load(AbstractTarget, p; resolve))
    return target_rig(load(Rig, p; resolve))
end
# A single unposed target as a one-member rig, under the name its measurements
# carry: its design `reference`, or its content id with none (the manifest's rule).
_target_rig(t::Attributed{<:AbstractTarget}; resolve = nothing) = _target_rig(t.value; resolve)
function _target_rig(t::AbstractTarget; resolve = nothing)
    name = t isa PlanarTarget && reference(t) !== nothing ? reference(t) : Symbol(content_id(t))
    return Rig{TargetFrame{name}}(Dict(name => PosedTarget(t, EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0) .* m))))
end
_target_hashes(ts) = String[_targets_hash(t) for t in _target_list(ts)]

# Every (target, index) must resolve in the rigs the file will name — checked
# at write time, where the rigs are in hand.
function _check_resolves(obs::TargetObservations, rigs)
    seen = Set(id for (_, sets) in obs for id in keys(sets))
    npts = Dict{Symbol, Int}()
    for rig in rigs, id in ids(rig)
        id in seen || continue
        t = rig[id].target
        is_resolved(t) || throw(
            ArgumentError("target $(id) is unresolved (a design reference without its points) — supply `resolve`, its design owner's resolver")
        )
        npts[id] = length(points(t))
    end
    for (t, sets) in obs, (id, cs) in sets, i in sources(cs)
        n = get(npts, id, 0)
        1 <= i <= n || throw(
            ArgumentError(
                "observation (target $(id), index $(i)) in frame $(t) does not resolve in the " *
                    "targets given ($(n == 0 ? "no target $(id) with points" : "it has $(n) points"))"
            )
        )
    end
    return nothing
end
