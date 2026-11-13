# ==============================================================================
# Targets — the calibration target as a MODEL, the analogue of a camera model
# ==============================================================================
#
# A target is what a camera observes and an observation names: a set of points
# in its own frame, `TargetFrame{id}`. Its geometry is independent of where it
# is mounted — the same target can sit in several rigs (`PosedTarget`, rig.jl) —
# and of who calibrated it (a stored target's `calibration` block is an
# attribute of the value, as a camera's is).
#
#   AbstractTarget        the contract: `points(target)`, the target's points in
#                         its own frame — 2-vectors on the plane z = 0 for a
#                         planar target, 3-vectors for a solid one — plain or
#                         `Uncertain`. That is all regression and evaluation call.
#   PointTarget           explicit points, nominal or estimated (`Uncertain`
#                         points): a motion-capture marker constellation, or a
#                         board whose calibration was resolved to measured points.
#   PlanarTarget{C}       a planar DESIGN — its nominal points, or the reference
#                         they are resolved from — and a CALIBRATION `C` taking
#                         them to the physical target: `points = C(nominal)`.
#
# THE DESIGN REFERENCE. A content-addressed design (a BlobBoards board uid) is
# named by `design`; VGC holds the reference and, once resolved, the nominal
# points, and never parses a design file — the package that owns the design
# (BlobBoards) resolves a reference into nominal points ([`resolve`](@ref)). An
# unresolved `PlanarTarget` (a pose-only board) has a reference and no points;
# asking for its points is an error naming the reference.
#
# THE CALIBRATION. What a target calibration estimates about the physical
# target, as a camera calibration estimates intrinsics and a lens: a
# `TargetCalibration` (target_calibration.jl). `Rigid` — no parameters — is
# exactly a printed board taken as designed.
#
# CALIBRATABLE TARGETS. A target whose geometry an estimator may refine
# implements a parameter protocol, the target analogue of the lens leaf
# protocol (lens_chart.jl), on the same manifold contract the
# block-coordinate-descent and LM machinery consume:
#
#   parameter_vector(t)         the parameter vector θ (an SVector; empty for Rigid)
#   parameter_names(t)          its names, so a configuration frees or fixes them by name
#   parameter_bounds(t)         its bounds (unbounded unless the calibration says otherwise)
#   with_parameters(t, θ)       the same target rebuilt from θ
#   parameters(t)               both, as a NamedTuple
#   minimal_manifold(t)         a `TargetManifold`: θ's offsets from an anchor;
#                               `action_jacobian(ch, x)` = ∂C(x)/∂θ at a nominal point x
#   regularizer(t)              the prior's residual rows r(θ) (‖r‖² the penalty,
#                               e.g. a thin-plate spline's bending energy); empty for Rigid
#
# A `PlanarTarget` forwards the protocol to its calibration, whose own protocol is
# `parameter_vector`, `parameter_names`, `with_parameters(c, θ)`,
# `transform_deriv_params(c, x)` and `regularizer(c)`.

"""
    AbstractTarget

A calibration target as a model: [`points`](@ref)`(target)` returns its points
in its own frame `TargetFrame{id}` — 2-vectors on the plane `z = 0` for a planar
target, 3-vectors for a solid one or a planar design lifted off its plane by a
[`HeightField`](@ref) — plain or `Uncertain`. See the header of
targets.jl for the calibratable-target protocol.

Placement is a trait, [`is_posed`](@ref), as for cameras: a target type such
as [`PlanarTarget`](@ref) or [`PointTarget`](@ref) is unposed, is its own
`model`, and refuses `pose` and `extrinsics` with an `ArgumentError`; a
[`PosedTarget`](@ref) is posed. The calibratable-target protocol and
[`TargetManifold`](@ref) are a target MODEL's, as the lens chart is a camera
model's: on a `PosedTarget` they are an `ArgumentError` naming `model(pt)`.
"""
abstract type AbstractTarget end

# A target point stored: plain numbers as Float64, lengths in metres (FlexUnits
# base form); an `Uncertain` one keeps its covariance as given — m² for a
# physical point (units.md: values in base units), never rescaled here.
_target_point(p::StaticVector{2}) = _stored_point(Val(2), p)
_target_point(p::StaticVector{3}) = _stored_point(Val(3), p)
_target_point(p::AbstractVector) = length(p) == 2 ? _stored_point(Val(2), p) : length(p) == 3 ? _stored_point(Val(3), p) :
    throw(ArgumentError("a target point is a 2-vector (a plane's) or a 3-vector; got $(length(p)) entries"))
# the covariance in Float64 and the storage it holds (full, diagonal or isotropic), unlabeled
_target_point(p::Uncertain) = (v = _target_point(geom(p)); Uncertain(v, CovMat{length(v), Float64}(parent(Statistics.cov(p)))))

# ------------------------------------------------------------------------------
# PointTarget
# ------------------------------------------------------------------------------

"""
    PointTarget(points)

A target given by its points, explicitly, in its own frame: 2-vectors (a
plane's) or 3-vectors, all of one dimension, plain or `Uncertain` (an
estimated constellation with its covariances). A motion-capture body's
marker constellation, or a board whose calibration was resolved to measured points.

A physical target's points are lengths, stored in metres, and an `Uncertain`
physical point's covariance is in m²: values are held in base units, and the
covariance is stored as given, never rescaled — supply it in m², not in the
squared unit the point was written in. Plain points are stored as `Float64`. All are one form or the other
([`has_units`](@ref)); `ustrip(u, t)` and [`with_unit`](@ref)`(t, u)` move
between them.
"""
struct PointTarget{P} <: AbstractTarget
    points::Vector{P}
    function PointTarget(pts::AbstractVector)
        ps = map(_target_point, pts)
        isconcretetype(eltype(ps)) || (ps = _one_storage(ps))   # mixed covariance storage: the coarser one (captures.jl)
        isempty(ps) || allequal(length(geom(p)) for p in ps) ||
            throw(ArgumentError("a target's points are all 2-vectors or all 3-vectors"))
        _check_one_form(ps)
        P = isempty(ps) ? SVector{3, Float64} : mapreduce(typeof, typejoin, ps)
        return new{P}(Vector{P}(ps))
    end
end

points(t::PointTarget) = t.points
Base.:(==)(a::PointTarget, b::PointTarget) = a.points == b.points
has_units(t::PointTarget) = any(has_units, points(t))
FlexUnits.ustrip(u::Units, t::PointTarget) = PointTarget([_ustrip_point(u, p) for p in points(t)])
with_unit(t::PointTarget, u::Units) = PointTarget([_attach_point(u, p) for p in points(t)])

# ------------------------------------------------------------------------------
# The placement — a calibration's dimensionless chart in the target's frame
# ------------------------------------------------------------------------------
#
# A thin-plate calibration is pure bending only on the nodes it is built on
# (target_calibration.jl, "no affine term"): its radial basis U(r) = r² log r is not
# scale-invariant, so the same field on nodes in metres and on nodes in
# millimetres differ by a piston. A target whose calibration is defined on a
# DIMENSIONLESS chart of its board — RadiometricCamera's page in units of its
# width, [0, 1] × [0, H/W] — says where that chart lies with a PLACEMENT, fixed
# data: Core's `SimilarityMap{3}` `S₃` from the chart's 3-D frame (the chart
# plane and the height along its normal) to the target's, of scale `side`,
# the identity rotation and the translation `(origin, 0)`. The target's point
# of the chart point `c` is then the composition
#
#     S₃ ∘ lift ∘ C (c)          (lift: (c, 0) for a planar calibration C)
#
# so the side scales the plane and the heights alike, and the design point `p`
# is the chart point `S₂(c) = origin + side·c` (S₂ the placement's in-plane
# part). The placement is plain numbers in the target's own form (metres for a
# physical target, as a Board's corner is stored): it carries no unit, and a
# unit move rescales it.

# The checked placement: a similarity of the chart plane into the target's,
# no rotation and no offset along the normal (a reflected chart is built on
# reflected nodes instead: a reflection keeps a thin-plate field's bending pure).
function _check_placement(S::SimilarityMap{3})
    s, t = S.s, S.t
    (isfinite(s) && s > 0) || throw(ArgumentError("a placement's scale (the chart's side) is positive and finite, got $s"))
    all(isfinite, t) || throw(ArgumentError("a placement's translation is finite, got $(Tuple(t))"))
    iszero(t[3]) || throw(ArgumentError("a placement lies in the target's plane: its translation's z is 0, got $(t[3])"))
    SMatrix{3, 3}(S.R) == SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1) ||
        throw(ArgumentError("a placement has the identity rotation (a chart that runs down the design's y is built on reflected nodes)"))
    return S
end
# The stored placement's one concrete type (plain Float64, unlabeled frames), so a
# placed target's points and surface are inferred.
const _Placement = SimilarityMap{3, Float64, Float64, RotMatrix{3, Float64, 9}, SVector{3, Float64}, UnlabeledFrame{3}, UnlabeledFrame{3}}
_stored_placement(::Nothing) = nothing
_stored_placement(S::SimilarityMap{3}) =
    (_check_placement(S); _Placement(Float64(S.s), RotMatrix{3, Float64}(SMatrix{3, 3}(S.R)), SVector{3, Float64}(S.t)))
# the placement's in-plane part S₂ (in the placement's element type, unchecked:
# a device path), and the identity placement of an unplaced target's chart map
_placement_origin(S::SimilarityMap{3}) = (t = S.t; SVector(t[1], t[2]))
@inline function _placement_plane(S::SimilarityMap{3})
    T = eltype(S.t)
    return _similarity(S.s, SMatrix{2, 2, T}(1, 0, 0, 1), _placement_origin(S))
end
@inline _identity_placement(::Type{T}) where {T} = _similarity(one(T), SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1), zero(SVector{3, T}))
# the placement after a unit move of the target's numbers, `f` one length's conversion
_move_placement(S::SimilarityMap{3}, f) = SimilarityMap(f(S.s), S.R, map(f, S.t))
_move_placement(::Nothing, _) = nothing

# ------------------------------------------------------------------------------
# PlanarTarget
# ------------------------------------------------------------------------------

"""
    AbstractTargetLayout

A parametric design of a planar target — [`AprilGrid`](@ref) or
[`SchoepsPattern`](@ref): the few numbers its nominal points follow from, and
whatever else the printed design states (its tag family and ids). `PlanarTarget(layout; …)`
is the target of its points, and it records the layout ([`layout`](@ref)`(t)`)
so that a writer can store the design by its layout rather than by its points.
"""
abstract type AbstractTargetLayout end

# A layout is its design: two are equal iff they are the same layout type with
# the same design (lengths compared as stored, in metres). Each layout declares
# its design beside its struct (`_layout_design`, an accessor, not reflection).
Base.:(==)(a::AbstractTargetLayout, b::AbstractTargetLayout) =
    nameof(typeof(a)) === nameof(typeof(b)) && _layout_design(a) == _layout_design(b)
Base.hash(L::AbstractTargetLayout, h::UInt) = hash(_layout_design(L), hash(nameof(typeof(L)), h))

# The construction routes of a PlanarTarget (private): checked, or a unit move.
struct _Checked end
struct _UnitMove end

"""
    PlanarTarget(nominal; calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing)                  a design given by its points
    PlanarTarget(design::Symbol; nominal = nothing, calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing)   a content-addressed design

A planar target: a DESIGN — its nominal plane points, and/or the reference
`design` they are resolved from (a BlobBoards board uid) — and a
[`TargetCalibration`](@ref) `C` taking the design to the physical target,
`points(t) = C.(nominal)`. An unresolved target (a reference, no nominal
points yet) is a pose-only board; [`resolve`](@ref) supplies its points, and
its [`points`](@ref) is an error until then. VGC never parses a design file:
the design's package resolves the reference.

The calibration may be [`Uncertain`](@ref) — `Uncertain(c, CovMat(Σ))`, `Σ`
over [`parameter_vector`](@ref)`(c)` — as a [`PosedTarget`](@ref)'s pose may
be; the type parameter `C` is then the calibration kind it wraps.
[`calibration`](@ref)`(t)` returns it as stored, [`design`](@ref)`(t)` the
design and [`reference`](@ref)`(t)` the reference (or `nothing`).

The nominal points are plain numbers (stored as `Float64`) or lengths (stored
in metres), all one form ([`has_units`](@ref)); a calibration with lengths
(a rigid member's translation, a thin-plate spline's nodes and displacements, a
height field's nodes and heights) is in the same form, or the constructor
raises FlexUnits' `DimensionError`. `ustrip(u, t)` and [`with_unit`](@ref)`(t, u)`
move the points and the calibration together. An unresolved target holds no
points: it is neither form, and both move only its calibration.

`board` is the board the design is printed on, a [`Board`](@ref) (its size and
its corner in the design chart), or `nothing` — the target's analogue of a
camera's [`Sensor`](@ref): `Rect(board)` is the printed domain in the design
chart, and [`surface`](@ref) bounds the target's primitive by it. The board
is in the points' form — lengths for physical points, plain numbers for plain
ones (FlexUnits' `DimensionError` otherwise) — and the points lie inside
`Rect(board)` to within a rounding margin of 16 eps of its coordinates, the
margin `surface` grows the board by (an `ArgumentError` otherwise). Being in the design chart, the
board is the same rectangle under any calibration. [`board`](@ref)`(t)`
reads it; `ustrip` and `with_unit` move it with the points.

`canvas` is the printed page the board lies on, a [`Canvas`](@ref) (its size,
its corner in the design chart and optionally the printed image, a
[`CanvasTexture`](@ref)), or `nothing`: the record of what was printed, which
the points, the calibration and [`surface`](@ref) never read. It is in the
points' form like the board, and the board lies on it to within the same
rounding margin (an `ArgumentError` otherwise). [`canvas`](@ref)`(t)` reads it;
`ustrip` and `with_unit` move it with the points.

`placement` is a `SimilarityMap{3}`, or `nothing`: where the
calibration's dimensionless chart lies in the target's frame (scale the
chart's side, the identity rotation, the translation `(origin, 0)`; see
"The placement" in targets.jl). With one, the
calibration acts in that chart — a [`HeightField`](@ref) or
[`ThinPlateSpline`](@ref) built on the chart's nodes (RadiometricCamera's
page `[0, 1]²`), its values in chart units — and is plain numbers whatever
the points' form (an `ArgumentError` for a calibration with units); the
placement is plain numbers in the points' unit (metres for a physical
target): `points(t) = (S₃ ∘ lift ∘ C ∘ S₂⁻¹).(nominal)`, a height scaled by
the placement's side.
The calibration's parameters, manifold and regularizer are its own: the
placement is fixed data and adds none. A unit move moves the placement and
leaves the calibration as it is. [`placement`](@ref)`(t)` reads it.

A target built from a parametric design, `PlanarTarget(layout; calibration,
board, canvas, placement)` (an [`AbstractTargetLayout`](@ref):
[`AprilGrid`](@ref), [`SchoepsPattern`](@ref)), holds the layout's points as its
nominal points and records the layout, [`layout`](@ref)`(t)`; every other
construction records none.
"""
struct PlanarTarget{C <: TargetCalibration, N <: Union{Nothing, Vector{<:SVector{2}}}, S <: Union{C, Uncertain{C}}} <: AbstractTarget
    reference::Union{Nothing, Symbol}
    nominal::N
    calibration::S
    board::Union{Nothing, Board}
    canvas::Union{Nothing, Canvas}
    placement::Union{Nothing, _Placement}
    layout::Union{Nothing, AbstractTargetLayout}
    # The private `_UnitMove()` route skips the board containment check: the
    # points are known to lie on the board — a unit move of a target that held
    # them there (`ustrip`, `with_unit`), whose rounding must not refuse it.
    # Every public construction is `_Checked()`. A `layout` is given only with
    # the points it builds (`PlanarTarget(layout; …)`, a unit move of one,
    # `with_parameters`): rebuilding them here would make the point type
    # uninferable and redo the layout on every parameter update in an LM loop.
    function PlanarTarget(
            reference::Union{Nothing, Symbol}, nominal, calibration::Union{TargetCalibration, Uncertain{<:TargetCalibration}},
            board::Union{Nothing, Board}, canvas::Union{Nothing, Canvas}, placement::Union{Nothing, SimilarityMap{3}},
            route::Union{_Checked, _UnitMove}, layout::Union{Nothing, AbstractTargetLayout} = nothing
        )
        (reference === nothing && nominal === nothing) &&
            throw(ArgumentError("a planar target has a design reference, nominal points, or both"))
        n = nominal === nothing ? nothing : _nominal_points(nominal)
        _check_calibration(n, calibration)
        # the points, the calibration's lengths (or, placed, the placement's) and the board: one form
        placement = _stored_placement(placement)
        placement === nothing || _check_chart_calibration(geom(calibration))
        _check_one_form((n === nothing || isempty(n) ? nothing : first(n), placement === nothing ? geom(calibration) : nothing, board, canvas))
        _check_board(n, board, calibration, route isa _Checked)
        route isa _Checked && _check_canvas(board, canvas)
        return new{typeof(geom(calibration)), typeof(n), typeof(calibration)}(reference, n, calibration, board, canvas, placement, layout)
    end
end
# The public positional form (the board, the canvas and the placement
# optional): always checked, the calibration first (`_stored_calibration`,
# target_calibration.jl: one of the kinds or an accepted composition,
# right-associated, a rigid member's lengths in metres).
PlanarTarget(reference::Union{Nothing, Symbol}, nominal, calibration,
             board::Union{Nothing, Board} = nothing, canvas::Union{Nothing, Canvas} = nothing,
             placement::Union{Nothing, SimilarityMap{3}} = nothing) =
    PlanarTarget(reference, nominal, _stored_calibration(calibration), board, canvas, placement, _Checked())
# A placed target's calibration acts in the dimensionless chart: plain numbers.
_check_chart_calibration(c) = _unit_form(c) === :physical && throw(
    ArgumentError(
        "PlanarTarget: with a placement the calibration acts in its dimensionless chart, so it is plain numbers " *
            "(nodes and values in chart units); got one with units — strip it (`ustrip(m, c)`) and build it on the chart's nodes"
    )
)
# The board lies on the canvas, within the rounding margin of `_grown_rect`.
function _check_canvas(bd, cv)
    (bd === nothing || cv === nothing) && return nothing
    b, r = Rect(bd), _grown_rect(Rect(cv))
    (_in_rect(r, b.origin) && _in_rect(r, b.origin + b.widths)) || throw(
        ArgumentError("PlanarTarget: the board, $(Tuple(b.origin)) to $(Tuple(b.origin + b.widths)), lies off the canvas, $(Tuple(r.origin)) to $(Tuple(r.origin + r.widths))")
    )
    return nothing
end
# The board is in the points' form, and they lie on it (in the design chart:
# metres when physical).
# An unresolved target's board is in its calibration's form when that carries
# lengths. A point on the edge is on the board within the rounding margin of
# `_grown_rect` (target_surfaces.jl) — the rectangle `surface` bounds the
# target's primitive by, so a point the constructor accepts is one a ray meets.
function _check_board(nominal, bd, c, on_board::Bool = true)
    (bd === nothing || nominal === nothing || isempty(nominal) || !on_board) && return nothing
    r = _grown_rect(Rect(bd))
    for p in nominal
        q = has_units(bd) ? strip_metres(p) : p
        _in_rect(r, q) || throw(
            ArgumentError("PlanarTarget: the nominal point $(Tuple(q)) lies off the board, $(Tuple(r.origin)) to $(Tuple(r.origin + r.widths))")
        )
    end
    return nothing
end
# The nominal points stored (an empty design is a plain one, as before).
function _nominal_points(nominal)
    ps = _check_one_form([_stored_point(Val(2), geom(p)) for p in nominal])
    return isempty(ps) ? SVector{2, Float64}[] : ps
end
# An uncertain calibration's covariance is over its parameters.
function _check_calibration(nominal, c)
    c isa Uncertain || return nothing
    n = length(parameter_vector(geom(c)))
    size(Statistics.cov(c)) == (n, n) || throw(
        DimensionMismatch("an uncertain calibration's covariance is $(n) × $(n) (its parameters); got $(size(Statistics.cov(c)))")
    )
    return nothing
end
PlanarTarget(nominal::AbstractVector; calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing) =
    PlanarTarget(nothing, nominal, calibration, board, canvas, placement)
PlanarTarget(reference::Symbol; nominal = nothing, calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing) =
    PlanarTarget(reference, nominal, calibration, board, canvas, placement)

"""
    AprilGrid(; tag_rows, tag_cols, tag_size, tag_spacing, apriltag_family = "tag36h11", first_id = 0)

The layout of an AprilGrid, Kalibr's grid of AprilTags (the `aprilgrid` target
of Kalibr's target YAML: `tagRows`, `tagCols`, `tagSize`, `tagSpacing`):
`tag_rows × tag_cols` square tags of side `tag_size`, `tag_spacing` apart — the
gap between two neighbouring tags, a length like the side (Kalibr's
`tagSpacing` is the ratio `tag_spacing / tag_size`). Both are lengths
(`90mm`, `27mm`) or both plain numbers. The tags are of the AprilTag family
`apriltag_family` (Kalibr's grids are `tag36h11`), with ids from `first_id`
(≥ 0) on. Tag `i` (0-based: the family id less the grid's first id) sits in
row `i ÷ tag_cols`, column `i % tag_cols`.

`PlanarTarget(grid; calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing)`
is the target of its tag corners (recording the grid, [`layout`](@ref)) in Kalibr's point order and frame: the `2tag_rows × 2tag_cols`
corners row by row, point `r·2tag_cols + c` (0-based, `r` and `c` 0-based) at

    x = (c ÷ 2)(tag_size + tag_spacing) + (c % 2)·tag_size
    y = (r ÷ 2)(tag_size + tag_spacing) + (r % 2)·tag_size

— tag 0's first corner at the origin, x along the columns and y along the
rows: the design chart (`TargetFrame`), x right and y up as the print is seen
from the front. [`point_index`](@ref)`(grid, tag, corner)` is the point of a
tag's corner. The points are in the layout's form (lengths, stored in metres,
or plain numbers); a `board` or `canvas` is in that form too, and a page whose
lower-left corner is not tag 0's first corner says so by its origin
(`Canvas(size, origin)`).
"""
struct AprilGrid{T <: Number} <: AbstractTargetLayout
    tag_rows::Int
    tag_cols::Int
    tag_size::T
    tag_spacing::T
    apriltag_family::String
    first_id::Int
    function AprilGrid{T}(
            tag_rows::Integer, tag_cols::Integer, tag_size::T, tag_spacing::T,
            apriltag_family::AbstractString = _DEFAULT_APRILTAG_FAMILY, first_id::Integer = 0
        ) where {T <: Number}
        (tag_rows >= 1 && tag_cols >= 1) || throw(ArgumentError("AprilGrid: $(tag_rows) × $(tag_cols) tags; a grid has at least one"))
        zero(tag_size) < tag_size || throw(ArgumentError("AprilGrid: the tag size is $(tag_size), not positive"))
        zero(tag_spacing) <= tag_spacing || throw(ArgumentError("AprilGrid: the tag spacing is $(tag_spacing), negative"))
        _check_apriltag_family("AprilGrid", apriltag_family)
        first_id >= 0 || throw(ArgumentError("AprilGrid: the first tag id is $(first_id), negative"))
        return new{T}(Int(tag_rows), Int(tag_cols), tag_size, tag_spacing, String(apriltag_family), Int(first_id))
    end
end
_layout_design(g::AprilGrid) = (g.tag_rows, g.tag_cols, g.tag_size, g.tag_spacing, g.apriltag_family, g.first_id)
# The AprilTag family a layout's tags are of when it does not say: Kalibr's and
# Schöps' generators both print tag36h11.
const _DEFAULT_APRILTAG_FAMILY = "tag36h11"
_check_apriltag_family(what, f) =
    isempty(f) ? throw(ArgumentError("$(what): the AprilTag family is empty (the generators' family is \"$(_DEFAULT_APRILTAG_FAMILY)\")")) : nothing
function AprilGrid(;
        tag_rows::Integer, tag_cols::Integer, tag_size::Number, tag_spacing::Number,
        apriltag_family::AbstractString = _DEFAULT_APRILTAG_FAMILY, first_id::Integer = 0
    )
    (tag_size isa Quantity) == (tag_spacing isa Quantity) ||
        throw(ArgumentError("AprilGrid: the tag size and the tag spacing are both lengths or both plain numbers; got $(tag_size) and $(tag_spacing)"))
    if tag_size isa Quantity
        all(q -> dimension(q) == dimension(mm), (tag_size, tag_spacing)) ||
            throw(ArgumentError("AprilGrid: the tag size and spacing are lengths; got $(tag_size) and $(tag_spacing)"))
    end
    s, g = tag_size isa Quantity ? (base_quantity(tag_size), base_quantity(tag_spacing)) : float.(promote(tag_size, tag_spacing))
    return AprilGrid{promote_type(typeof(s), typeof(g))}(tag_rows, tag_cols, promote(s, g)..., apriltag_family, first_id)
end

"""
    point_index(grid::AprilGrid, tag, corner) -> Int

The 1-based index, in `PlanarTarget(grid)`'s points, of corner `corner` of tag
`tag` (both 0-based: Kalibr's tag id less the grid's first id, and its corner
order 0 (min column, min row), 1 (max column, min row), 2 (max column, max
row), 3 (min column, max row)) — Kalibr's `GridCalibrationTargetAprilgrid`
point of a detected corner.
"""
function point_index(grid::AprilGrid, tag::Integer, corner::Integer)
    (0 <= tag < grid.tag_rows * grid.tag_cols) || throw(ArgumentError("point_index: no tag $(tag) in a $(grid.tag_rows) × $(grid.tag_cols) grid (0-based)"))
    (0 <= corner <= 3) || throw(ArgumentError("point_index: a tag's corners are 0 … 3, not $(corner)"))
    ncols = 2 * grid.tag_cols
    base = 2 * (tag ÷ grid.tag_cols) * ncols + 2 * (tag % grid.tag_cols)
    return 1 + base + (0, 1, ncols + 1, ncols)[corner + 1]
end

function _layout_points(grid::AprilGrid)
    pitch = grid.tag_size + grid.tag_spacing
    along(i) = (i ÷ 2) * pitch + (i % 2) * grid.tag_size
    return [SVector(along(c), along(r)) for r in 0:(2 * grid.tag_rows - 1) for c in 0:(2 * grid.tag_cols - 1)]
end
has_units(g::AprilGrid) = g.tag_size isa Quantity

"""
    ustrip(u::Units, grid::AprilGrid) -> AprilGrid
    with_unit(grid::AprilGrid, u::Units) -> AprilGrid

The grid's tag size and spacing moved between the forms: a physical grid's
lengths as plain numbers in `u`, a plain grid's numbers as lengths in `u`. The
grid already in the form asked for is an `ArgumentError`.
"""
FlexUnits.ustrip(u::Units, g::AprilGrid{<:Quantity}) = AprilGrid(;
    g.tag_rows, g.tag_cols, tag_size = strip_or_bare(u, g.tag_size), tag_spacing = strip_or_bare(u, g.tag_spacing), g.apriltag_family, g.first_id
)
with_unit(g::AprilGrid{<:Real}, u::Units) = AprilGrid(;
    g.tag_rows, g.tag_cols, tag_size = _attach_length(g.tag_size, u), tag_spacing = _attach_length(g.tag_spacing, u), g.apriltag_family, g.first_id
)

# The target of a layout's points, recording the layout (`layout(t)`).
PlanarTarget(L::AbstractTargetLayout; calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing) =
    PlanarTarget(nothing, _layout_points(L), _stored_calibration(calibration), board, canvas, placement, _Checked(), L)

const _SchoepsTag = @NamedTuple{x::Int, y::Int, width::Int, height::Int, id::Int}   # a Schöps tag: in squares, y down; the family id

"""
    SchoepsPattern(; squares_x, squares_y, square_length, apriltags, star_segments = 16, apriltag_family = "tag36h11")

The layout of a tagged star pattern of Schöps et al.'s camera_calibration
(their `create_calibration_pattern.py`; the pattern YAML's `squares_x`,
`squares_y`, `square_length_in_meters`, `num_star_segments` and `apriltags`): a `squares_x ×
squares_y` grid of squares of side `square_length` — a length (`8mm`) or a
plain number — with a star feature of `star_segments` segments (even, ≥ 4) at every inner corner, and AprilTags of the family
`apriltag_family`, each
covering `width × height` squares from square `(x, y)` (the YAML's `tag_x`,
`tag_y`, `width`, `height`: 0-based squares, x to the right and y DOWN as the
pattern is printed) and carrying the family id `id` (the YAML's `index`).
`apriltags` is a collection of `(; x, y, width, height)` or `(; x, y, width,
height, id)` — without ids the tags are numbered 0, 1, … in order — stored as
`(; x, y, width, height, id)`; each tag lies on the pattern, and the ids are
distinct and ≥ 0.

`PlanarTarget(pattern; calibration = Rigid(), board = nothing, canvas = nothing, placement = nothing)`
is the target (recording the pattern, [`layout`](@ref)) of
its star features in Schöps' feature order
(`FeatureDetectorTaggedPattern::GetCorners`): the inner corners `(gx, gy)`,
`gx ∈ 0:squares_x − 2`, `gy ∈ 0:squares_y − 2`, row by row (`gy` outer), the
corners next to a tag left out (`IsValidFeatureCoord`: `x − 1 ≤ gx ≤ x − 1 +
width` and `y − 1 ≤ gy ≤ y − 1 + height` for some tag), so for a YAML's
first pattern (pattern index 0) feature id `k` (0-based, Schöps' and a
`dataset.bin`'s) is point `k + 1`. `GetCorners` numbers the features of a
YAML with several patterns across them, in order: pattern `i`'s feature id
`k` is point `k + 1 − (n₀ + … + nᵢ₋₁)` of its own target, `nⱼ` the number of
points of pattern `j`'s target. Each point is at

    square_length · (gx + 1, squares_y − 1 − gy)

— the design chart (`TargetFrame`): the pattern's lower-left corner at the
origin, x right and y up as the print is seen from the front. Schöps' own
geometry (`KnownGeometry`: `cell_length_in_meters · (gx, gy, 0)`, y down) is the
same points under a half turn about x and a shift of one square, a rigid
motion the pose absorbs. The points are in the layout's form (lengths, stored
in metres, or plain numbers); a `board` or `canvas` is in that form too, and a
page whose lower-left corner is not the pattern's says so by its origin
(`Canvas(size, origin)`).
"""
struct SchoepsPattern{T <: Number} <: AbstractTargetLayout
    squares_x::Int
    squares_y::Int
    square_length::T
    apriltags::Vector{_SchoepsTag}
    star_segments::Int
    apriltag_family::String
    function SchoepsPattern{T}(
            squares_x::Integer, squares_y::Integer, square_length::T, apriltags::Vector{_SchoepsTag},
            star_segments::Integer = 16, apriltag_family::AbstractString = _DEFAULT_APRILTAG_FAMILY
        ) where {T <: Number}
        (squares_x >= 2 && squares_y >= 2) ||
            throw(ArgumentError("SchoepsPattern: $(squares_x) × $(squares_y) squares; a pattern has an inner corner, at least 2 × 2"))
        zero(square_length) < square_length || throw(ArgumentError("SchoepsPattern: the square length is $(square_length), not positive"))
        for (; x, y, width, height, id) in apriltags
            (width >= 1 && height >= 1 && x >= 0 && y >= 0 && x + width <= squares_x && y + height <= squares_y) || throw(
                ArgumentError("SchoepsPattern: a $(width) × $(height) tag at square ($(x), $(y)) does not lie on the $(squares_x) × $(squares_y) pattern")
            )
            id >= 0 || throw(ArgumentError("SchoepsPattern: a tag's id is $(id), negative"))
        end
        allunique(t.id for t in apriltags) || throw(ArgumentError("SchoepsPattern: the tags' ids $([t.id for t in apriltags]) are not distinct"))
        (star_segments >= 4 && iseven(star_segments)) ||
            throw(ArgumentError("SchoepsPattern: a star of $(star_segments) segments; a star has an even number, at least 4"))
        _check_apriltag_family("SchoepsPattern", apriltag_family)
        return new{T}(Int(squares_x), Int(squares_y), square_length, apriltags, Int(star_segments), String(apriltag_family))
    end
end
_layout_design(p::SchoepsPattern) = (p.squares_x, p.squares_y, p.square_length, p.apriltags, p.star_segments, p.apriltag_family)
function SchoepsPattern(;
        squares_x::Integer, squares_y::Integer, square_length::Number, apriltags,
        star_segments::Integer = 16, apriltag_family::AbstractString = _DEFAULT_APRILTAG_FAMILY
    )
    if square_length isa Quantity
        dimension(square_length) == dimension(mm) || throw(ArgumentError("SchoepsPattern: the square length is a length; got $(square_length)"))
    end
    s = square_length isa Quantity ? base_quantity(square_length) : float(square_length)
    tags = _SchoepsTag[(x = Int(t.x), y = Int(t.y), width = Int(t.width), height = Int(t.height), id = Int(hasproperty(t, :id) ? t.id : i - 1))
        for (i, t) in enumerate(apriltags)]
    return SchoepsPattern{typeof(s)}(squares_x, squares_y, s, tags, star_segments, apriltag_family)
end
has_units(p::SchoepsPattern) = p.square_length isa Quantity

"""
    ustrip(u::Units, pattern::SchoepsPattern) -> SchoepsPattern
    with_unit(pattern::SchoepsPattern, u::Units) -> SchoepsPattern

The pattern's square length moved between the forms: a physical pattern's as a
plain number in `u`, a plain pattern's number as a length in `u`. The pattern
already in the form asked for is an `ArgumentError`.
"""
FlexUnits.ustrip(u::Units, p::SchoepsPattern{<:Quantity}) =
    SchoepsPattern(; p.squares_x, p.squares_y, square_length = strip_or_bare(u, p.square_length), p.apriltags, p.star_segments, p.apriltag_family)
with_unit(p::SchoepsPattern{<:Real}, u::Units) =
    SchoepsPattern(; p.squares_x, p.squares_y, square_length = _attach_length(p.square_length, u), p.apriltags, p.star_segments, p.apriltag_family)

# A canvas, its texture and a layout move their unit once: stripping plain numbers or
# attaching a unit twice is refused (target_calibration.jl's refusals, by type name).
FlexUnits.ustrip(::Units, x::Union{CanvasTexture{<:Real}, Canvas{<:Real}, AprilGrid{<:Real}, SchoepsPattern{<:Real}}) = _refuse_strip(x)
with_unit(x::Union{CanvasTexture{<:Quantity}, Canvas{<:Quantity}, AprilGrid{<:Quantity}, SchoepsPattern{<:Quantity}}, ::Units) = _refuse_attach(x)

# The inner corners next to a tag (Schöps' IsValidFeatureCoord refuses them):
# `covered[gx + 1, gy + 1]`, each tag marking its own range once.
function _schoeps_covered(p::SchoepsPattern)
    nx, ny = p.squares_x - 1, p.squares_y - 1
    covered = falses(nx, ny)
    for t in p.apriltags
        covered[max(t.x, 1):min(t.x + t.width, nx), max(t.y, 1):min(t.y + t.height, ny)] .= true
    end
    return covered
end

function _layout_points(p::SchoepsPattern)
    s, covered = p.square_length, _schoeps_covered(p)
    return [SVector((gx + 1) * s, (p.squares_y - 1 - gy) * s) for gy in 0:(p.squares_y - 2) for gx in 0:(p.squares_x - 2) if !covered[gx + 1, gy + 1]]
end

"""
    TARGET_LAYOUT_TAGS

The [`TagRegistry`](@ref) of the target layouts ([`AbstractTargetLayout`](@ref)):
`"april_grid"` ([`AprilGrid`](@ref)) and `"schoeps_pattern"`
([`SchoepsPattern`](@ref)), `snake_case` of the type's name — the `type` of
VisualGeometryToolkitIO's target `layout` record. (Kalibr's own YAML spells
its target type `aprilgrid`; that is the foreign file's word, read by its
reader, not a tag.)
"""
const TARGET_LAYOUT_TAGS = TagRegistry(:target_layout, "", "april_grid" => AprilGrid, "schoeps_pattern" => SchoepsPattern)

"""
    calibration(t::PlanarTarget) -> TargetCalibration or Uncertain

The target's calibration as stored: a [`TargetCalibration`](@ref), or an
[`Uncertain`](@ref) one carrying its covariance (`geom` unwraps it).
VisualGeometryToolkitIO's record of who calibrated a target (the tool, the
date, the RMS) is a different thing with another name, the attribute
`Attributed(t, (; calibration_record = …))`.
"""
calibration(t::PlanarTarget) = t.calibration

"""
    reference(t::PlanarTarget) -> Union{Nothing, Symbol}

The content-addressed design the target's points are resolved from (a
BlobBoards board uid), or `nothing` for a design given by its points alone.
"""
reference(t::PlanarTarget) = t.reference

"""
    design(t::PlanarTarget) -> Vector{SVector{2}} or Symbol

The target's design: its nominal plane points when it is resolved
([`is_resolved`](@ref)), the reference they are resolved from (a `Symbol`)
when it is not. The physical target is `calibration(t)` applied to it.
"""
design(t::PlanarTarget) = something(t.nominal, t.reference)

"""
    board(t::PlanarTarget) -> Union{Nothing, Board}

The board the target's design is printed on ([`Board`](@ref)), or `nothing`
when the target does not carry one. `Rect(board(t))` is its domain in the
design chart, plain metres.
"""
board(t::PlanarTarget) = t.board

"""
    canvas(t::PlanarTarget) -> Union{Nothing, Canvas}

The page the target's board is printed on ([`Canvas`](@ref), with its printed
image when it carries one), or `nothing` when the target does not carry one.
"""
canvas(t::PlanarTarget) = t.canvas

"""
    placement(t::PlanarTarget) -> Union{Nothing, SimilarityMap{3}}

Where the target's calibration chart lies in its design chart
(a `SimilarityMap{3}`: the chart's side, the identity rotation, the
translation `(origin, 0)`, plain numbers in the target's form), or `nothing` when the calibration acts on the
design chart itself.
"""
placement(t::PlanarTarget) = t.placement

"""
    layout(t::PlanarTarget) -> Union{Nothing, AbstractTargetLayout}

The parametric design the target's nominal points were built from — an
[`AprilGrid`](@ref) or a [`SchoepsPattern`](@ref), given to
`PlanarTarget(layout; …)` — or `nothing` for a target given by its points or
by a reference. A unit move moves it with the points and a new calibration
(`with_parameters`) keeps it; [`resolve`](@ref), which supplies other points,
drops it.
"""
layout(t::PlanarTarget) = t.layout

"""
    resolve(t::PlanarTarget, nominal) -> PlanarTarget

The target with its design's nominal points supplied — what the design's
owner returns for the reference `reference(t)`.
"""
resolve(t::PlanarTarget, nominal::AbstractVector) = PlanarTarget(t.reference, nominal, t.calibration, t.board, t.canvas, t.placement)

"""
    is_resolved(t) -> Bool

Whether a target carries its points (a `PlanarTarget` holding only a design
reference does not).
"""
is_resolved(::AbstractTarget) = true
is_posed(::AbstractTarget) = false   # an unposed target; `PosedTarget` (rig.jl) answers true

# The camera parallel (cameras.jl `_unposed_only`): an unposed target is its own
# model, and the posed-only verbs refuse it by name.
_pose_hint(::AbstractTarget) = ("target", "`PosedTarget(model, X)`")
model(t::AbstractTarget) = is_posed(t) ? throw(MethodError(model, (t,))) : t
pose(t::AbstractTarget) = _unposed_only(pose, "pose", t)
extrinsics(t::AbstractTarget) = _unposed_only(extrinsics, "extrinsics", t)
is_resolved(t::PlanarTarget) = t.nominal !== nothing

@noinline _refuse_unresolved(t) = throw(
    ArgumentError("points: the planar target $(repr(t.reference)) is unresolved — its design's owner resolves the reference (`resolve`)")
)
"""
    points(t::PlanarTarget) -> Vector{SVector{2}} or Vector{SVector{3}}

The physical target's points in its own frame, `calibration(t)` applied to the
design: `SVector{2}`s on the plane `z = 0` when the calibration
[`is_planar`](@ref), `SVector{3}`s when a [`HeightField`](@ref) lifts them off
it — decided by the calibration's type, so the result type is inferred. A
placed target ([`placement`](@ref)) applies its calibration in the chart,
`P(C(P⁻¹(p)))`. An `ArgumentError` naming the reference for an unresolved
target.
"""
points(t::PlanarTarget) = t.nominal === nothing ? _refuse_unresolved(t) : _calibrated_points(geom(t.calibration), t.placement, t.nominal)
_calibrated_points(c, ::Nothing, nominal) = [c(p) for p in nominal]
# placed: the design point's chart point through the composed map S₃ ∘ lift ∘ C, in the points' form
function _calibrated_points(c, P::SimilarityMap{3}, nominal)
    M = _target_map(P, c)
    S₂⁻¹ = inv(_placement_plane(P))
    return [_restore_point(_points_dims(c, M(S₂⁻¹(_plain_point(p)))), p) for p in nominal]
end
# a planar calibration's points stay in the plane
_points_dims(c, x::StaticVector{3}) = is_planar(typeof(c)) ? SVector(x[1], x[2]) : x
Base.:(==)(a::PlanarTarget, b::PlanarTarget) =
    a.reference === b.reference && isequal(a.nominal, b.nominal) && a.calibration == b.calibration && a.board == b.board &&
    a.canvas == b.canvas && a.placement == b.placement && a.layout == b.layout
Base.hash(t::PlanarTarget, h::UInt) =
    hash(t.layout, hash(t.placement, hash(t.canvas, hash(t.board, hash(t.calibration, hash(t.nominal, hash(t.reference, hash(:PlanarTarget, h))))))))

# Units: a resolved target's nominal points move, and with them a calibration
# that has lengths; an unresolved one moves only its calibration.
# The board moves with them, by the same rule (an unresolved target's board
# moves when it is in the form being left). A placed target moves its
# placement instead of its calibration, which is plain chart numbers.
has_units(t::PlanarTarget) = is_resolved(t) ? has_units(t.nominal) : (lp = _lengths_part(t); lp !== nothing && has_units(lp))
# the part that carries the calibration's lengths: the calibration, or none when placed (the placement is plain)
_lengths_part(t::PlanarTarget) = t.placement === nothing ? geom(t.calibration) : nothing
# The calibration as a unit move `f` leaves it: moved, or — placed — as it is
# (a placed calibration is plain chart numbers; the placement carries the lengths).
_calibration_unit_move(t::PlanarTarget, f) = t.placement === nothing ? f(t.calibration) : t.calibration
FlexUnits.ustrip(u::Units, t::PlanarTarget) = _unit_moved(
    t, _move_board(ustrip, u, t.layout, true), p -> _ustrip_point(u, p),
    _calibration_unit_move(t, c -> _ustrip_calibration(u, c)),
    _move_board(ustrip, u, t.board, true), _move_board(ustrip, u, t.canvas, true), _move_placement(t.placement, x -> _from_metres(x, u))
)
_unit_attached(u, x) = with_unit(x, u)   # `with_unit` in `ustrip`'s argument order, for `_move_board`
with_unit(t::PlanarTarget, u::Units) = _unit_moved(
    t, _move_board(_unit_attached, u, t.layout, false), p -> _attach_point(u, p),
    _calibration_unit_move(t, c -> _attach_calibration(u, c)),
    _move_board(_unit_attached, u, t.board, false), _move_board(_unit_attached, u, t.canvas, false),
    _move_placement(t.placement, x -> _to_metres(x, u))
)
# A layout target's points are derived from its moved layout by the constructor,
# so they stay the layout's points exactly.
_unit_moved(t, L, move_point, c, bd, cv, pl) = PlanarTarget(
    t.reference, L !== nothing ? _layout_points(L) : is_resolved(t) ? [move_point(p) for p in t.nominal] : nothing,
    c, bd, cv, pl, _UnitMove(), L
)
# The one-form rule's view of a planar target (`_check_one_form`): its nominal
# points', calibration's (or placement's), board's and canvas's forms, which its constructor
# already holds to one — read from the stored parts, never by applying the
# calibration. An unresolved target whose calibration, board or canvas carries
# lengths has their form, so a plain pose on it is refused as on a resolved one.
function _unit_form(t::PlanarTarget)
    n = t.nominal
    forms = (n === nothing || isempty(n) ? :none : _unit_form(first(n)), _unit_form(_lengths_part(t)), _unit_form(t.board), _unit_form(t.canvas))
    return :physical in forms ? :physical : :plain in forms ? :plain : :none
end
_form_sample(t::PlanarTarget) = _form_sample(
    t.nominal !== nothing && !isempty(t.nominal) ? first(t.nominal) :
        _unit_form(_lengths_part(t)) !== :none ? _lengths_part(t) : t.board !== nothing ? t.board : t.canvas
)
_move_board(_, _, ::Nothing, _) = nothing
_move_board(f, u, bd::Union{Board, Canvas, AbstractTargetLayout}, from_physical::Bool) = has_units(bd) == from_physical ? f(u, bd) : bd

# --- the calibratable-target protocol ------------------------------------------------

# A calibratable target's parameters are its calibration's. A new parameter
# vector is a new calibration: `with_parameters` drops an uncertain one's
# covariance, which described the old parameters.
parameter_vector(t::PlanarTarget) = parameter_vector(geom(t.calibration))

parameter_names(t::PlanarTarget) = parameter_names(geom(t.calibration))
with_parameters(t::PlanarTarget, θ::AbstractVector) = PlanarTarget(
    t.reference, t.nominal, _stored_calibration(with_parameters(geom(t.calibration), θ)),
    t.board, t.canvas, t.placement, _Checked(), t.layout
)
parameters(t::PlanarTarget) = NamedTuple{parameter_names(t)}(Tuple(parameter_vector(t)))
regularizer(t::PlanarTarget) = regularizer(geom(t.calibration))
is_planar(::PlanarTarget{C}) where {C} = is_planar(C)

"""
    TargetManifold(target; known = ()) <: Manifold

A calibratable target's manifold anchored at `target`: `minimal_manifold(target; known)`.
No calibration kind has a gauge (the affine factors leave the pose's
directions out of the composition, the thin-plate kinds hold no affine or
plane part), so the manifold is every direction the calibration's parameters
may move in, and with the target's pose estimated jointly every one of them is
observable.

That holds for the print error a jointly estimated pose leaves,
`HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew`. A rigid member
(`EuclideanMap`) duplicates the pose's rotation and translation, and a
[`Scale`](@ref) the pose's scale (its translation's length): with the pose
estimated jointly they are the pose's gauges, and the manifold REFUSES them
(`ArgumentError`) unless the caller states what fixes them — `known` lists
`:length` (a measured length, a ruler: `Scale` is observable) or `:pose` (a
known pose, a fixture or motion capture: `EuclideanMap` and `Scale` both
are). Otherwise drop them from the calibration, or absorb them into the pose,
before a joint refinement. (A `vgio_migrate` of a retired `"affine"` record
keeps all four factors `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`, the same map;
that record takes `known = (:pose,)` or that reduction.)

The chart is linear: `retract(ch, v) = with_parameters(target, θ₀ + B v)`,
`B` an `n × m` matrix with orthonormal columns, so `tangent_size(ch) = m` and
`retract_jacobian(ch) = B`. For the affine factors `B` is the identity and the
coordinates are the parameters' offsets, named by [`parameter_names`](@ref).
For a [`ThinPlateSpline`](@ref) (a [`HeightField`](@ref)) `B` spans the
`2K − 6` (`K − 3`) pure-bending directions, the null space of the
interpolant's affine-coefficient rows — the complement of an orthonormal basis
of those rows' span, from a Householder QR
with column pivoting (LAPACK, so one reproducible basis per node set and
parameter layout); a composition's the complement of its members'
constraints together.
`action_jacobian(ch, x) = transform_deriv_params(c, x) · B` at the design point
`x`; `local_coordinates(ch, t) = Bᵀ (θ(t) − θ₀)`. [`restrict`](@ref)`(ch,
names)` keeps only the parameters `names` free, the others held.

NO GAUGE WITH THE POSE, AND AN OBLIQUE COMPLEMENT. The tangent has no
direction a change of the jointly estimated pose reproduces: the pose's
nodal fields (a piston and two tilts of a height field, an in-plane
translation and rotation of a thin-plate spline) are affine, and the
thin-plate kinds' free directions are pure bending, their interpolant's
affine part zero. That complement of the pose's directions is OBLIQUE, not
orthogonal: the basis is orthogonal to the interpolant's affine ROWS (the
pose's fields in the metric `K⁻¹` of the radial basis), not to the plane
fields themselves, so a bending direction correlates with a piston at a set
of sampled points while no combination of them is one. The deformation's and
the pose's spans intersect trivially — the stacked point Jacobian has full
rank (test/mapped_surface_tests.jl pins its condition number and the smallest
principal angle between the spans per composition) — which is the
gauge-freedom statement; a Euclidean orthogonality in point space is not a
property any parametrization can have. The spans can come close: with a
4 × 4 node grid reaching past the detections (nodes at ±0.6, samples to
±0.5 × ±0.4), the nearest bending direction is a nearly constant field of
nodal heights — a bowl whose curvature lies outside the data, almost a piston
on it — at about 5° (0.095 rad) from the pose's span, which inflates the
variance along that one mode about 110×; nodes covered by detections (0.36 rad
with the grid inside the samples) and the bending regularizer control it. A
`Scale` and an in-plane `EuclideanMap` ARE pose directions, hence refused
without `known`.

THE BASIS IS FIXED AT THE ANCHOR, and `recenter` keeps it: the constrained
directions are constants of the nodes, so the slice is the same wherever it
is anchored, and the manifold's type does not change.

`B` is an `SMatrix` while `n ≤ 18` and a `Matrix{Float64}`
above: a static basis compiles in time that grows steeply with its size (the
first call of the manifold's verbs, cold: n = 18, 0.6 s static against 0.8 s
as a `Matrix`; n = 32, 2.3 s against 0.7 s; a 5 × 5 in-plane spline, n = 50,
9.9 s against 0.7 s; 7 × 7, n = 98, 100 s against 2.6 s). In the large case `retract`'s step,
`action_jacobian` and `local_coordinates` are a `Vector` or a `Matrix`, and
`tangent_size` is a run-time value. Zero-dimensional for a `Rigid` target.
"""
struct TargetManifold{T <: AbstractTarget, B <: AbstractMatrix{Float64}} <: Manifold
    target::T
    basis::B                      # n × m, orthonormal columns: the free directions of θ
    fixed::Vector{Symbol}         # the parameters held fixed (`restrict`)
end
TargetManifold(t::PlanarTarget; known::Tuple{Vararg{Symbol}} = ()) = _target_manifold(_check_pose_members(t, known), Symbol[])

# A rigid member and a scale are the jointly estimated pose's gauges: refused
# unless `known` states the pose (both) or a length (the scale).
const _TARGET_KNOWN = (:length, :pose)
function _check_pose_members(t::PlanarTarget, known::Tuple{Vararg{Symbol}})
    for k in known
        k in _TARGET_KNOWN || throw(ArgumentError("TargetManifold: known = $(repr(known)) — each entry is :length (a measured length) or :pose (a known pose)"))
    end
    ms = members(geom(t.calibration))
    gauges = String[]
    any(x -> x isa EuclideanMap, ms) && !(:pose in known) && push!(gauges, "EuclideanMap (the pose's rotation and translation)")
    any(x -> x isa Scale, ms) && isempty(known) && push!(gauges, "Scale (the pose's scale)")
    isempty(gauges) || throw(
        ArgumentError(
            "TargetManifold: the calibration holds $(join(gauges, " and ")), a gauge of the target's jointly estimated pose — " *
                "drop it from the calibration (or absorb it into the pose) for a joint refinement, whose print error is " *
                "Aspect ∘ Skew; or state what fixes it: known = (:length,) for a measured length (Scale), known = (:pose,) for a known pose"
        )
    )
    return t
end

# The largest parameter count with a static basis: below it a static basis
# compiles about as fast as a Matrix one, above it far slower (the docstring's
# observations).
const _STATIC_BASIS_MAX = 18

# B: the directions E y of the free parameter axes E (every axis but the held
# parameters) that keep the calibration's constraint, Dᵀ E y = 0, D its
# constraint columns (`_constraint_columns`: the thin-plate kinds' plane
# fields, orthonormal) — y in the complement of range(Eᵀ D), from the full
# Householder QR of Eᵀ D with column pivoting, its rank where |Rᵢᵢ| is above
# √eps of the largest (D is orthonormal, so the cutoff is scale-free).
function _target_manifold(t::PlanarTarget, fixed::Vector{Symbol})
    c = geom(t.calibration)
    names = parameter_names(c)
    n = length(names)
    free_axes = [i for i in 1:n if !(names[i] in fixed)]
    f = length(free_axes)
    E = zeros(n, f)                                                    # the free parameter axes
    foreach(((j, i),) -> E[i, j] = 1.0, enumerate(free_axes))
    B = _free_basis(E, _constraint_columns(c))
    return TargetManifold(t, _basis(B, Val(n <= _STATIC_BASIS_MAX)), fixed)
end
# THE null-space step: the directions E y of the axes E that keep Dᵀ E y = 0 (the
# manifold's basis, and a scattered fit's unknowns, HeightField(nodes, data)).
function _free_basis(E::Matrix{Float64}, D::AbstractMatrix)
    f = size(E, 2)
    (size(D, 2) == 0 || f == 0) && return E
    X = E' * D                                                         # f × g
    F = LinearAlgebra.qr(X, LinearAlgebra.ColumnNorm())
    d = abs.(diag(F.R))
    r = isempty(d) ? 0 : count(>(sqrt(eps(Float64)) * maximum(d)), d)
    Q = F.Q * Matrix{Float64}(LinearAlgebra.I, f, f)
    return E * Q[:, (r + 1):f]
end
_basis(B::Matrix{Float64}, ::Val{true}) = SMatrix{size(B, 1), size(B, 2), Float64}(B)
_basis(B::Matrix{Float64}, ::Val{false}) = B

minimal_manifold(t::PlanarTarget; known::Tuple{Vararg{Symbol}} = ()) = TargetManifold(t; known)
ambient_coordinates(t::PlanarTarget) = parameter_vector(t)
tangent_size(ch::TargetManifold) = size(ch.basis, 2)
model(ch::TargetManifold) = ch.target
retract(ch::TargetManifold, v) = with_parameters(ch.target, parameter_vector(ch.target) + ch.basis * v)
recenter(ch::TargetManifold, v) = TargetManifold(retract(ch, v), ch.basis, ch.fixed)
retract_jacobian(ch::TargetManifold) = ch.basis
retract_jacobian(ch::TargetManifold, ::AbstractVector) = ch.basis
action_jacobian(ch::TargetManifold{<:PlanarTarget}, x) = _placed_action(ch.target.placement, geom(ch.target.calibration), x, ch.basis)
# A placed target's calibration acts at the design point's chart point, and its
# in-plane rows and height row come back scaled by the placement's side
# (`x = p + side·(y − c)`, `side·y₃`: ∂x/∂θ = side·∂y/∂θ).
_placed_action(::Nothing, c, x, B) = _calibration_action(c, x, B)
# placed: through the members, ∂/∂θ S₃(lift(C(S₂⁻¹ x))) = ∂S₃ · lift · ∂C/∂θ at S₂⁻¹ x
function _placed_action(P::SimilarityMap{3}, c, x, B)
    y = inv(_placement_plane(P))(_plain_point(x))
    A = _calibration_action(c, y, B)
    JS = transform_deriv(P, SVector(y[1], y[2], zero(eltype(y))))
    return size(A, 1) == 2 ? JS[StaticArrays.SOneTo(2), StaticArrays.SOneTo(2)] * A : JS * A
end
# ∂C(x)/∂θ · B, multiplying only the rows and columns the calibration's Jacobian
# can have nonzero: a height field moves only z, a thin-plate spline's two rows
# each depend on one of its two value blocks.
_calibration_action(c::TargetCalibration, x, B) = transform_deriv_params(c, x) * B
# A composition t1 ∘ t2, θ = (θ₁, θ₂): ∂C/∂θ · B = ∂t1/∂θ₁ · B₁ + ∂t1/∂y · (∂t2/∂θ₂ · B₂)
# at y = t2(x), B₁ and B₂ the rows of B over each member's parameters — each
# member's own (sparse) action on its own row block, chained through the point
# Jacobian, never the dense Jacobian of the whole. Each member acts on the
# point in its own form — a physical member (a rigid member's translation in
# metres) on the physical point — and its Jacobians come back plain, as CT's
# chain rule over the members' `transform_deriv(_params)` has them.
function _calibration_action(c::_CalibrationComposition, x, B)
    n1, n2 = _parameter_count(typeof(c.t1)), _parameter_count(typeof(c.t2))
    y = c.t2(x)
    return _calibration_action(c.t1, y, _row_block(B, 0, Val(n1))) + transform_deriv(c.t1, y) * _calibration_action(c.t2, x, _row_block(B, n1, Val(n2)))
end
# (Each kept row is a 1 × K matrix times the K rows of B it meets: the product
# kernel the full Jacobian's rows went through, so the entries keep their bits.)
function _calibration_action(c::HeightField{K}, x, B) where {K}
    z = SMatrix{1, K}(Tuple(nodal_weights(c.system, _plain_point(x)))) * B
    return vcat(zero(z), zero(z), z)
end
function _calibration_action(c::ThinPlateSpline{K}, x, B) where {K}
    n = SMatrix{1, K}(Tuple(nodal_weights(c.system, _plain_point(x))))
    return vcat(n * _row_block(B, 0, Val(K)), n * _row_block(B, K, Val(K)))
end
_row_block(B::StaticMatrix{R, M}, o, ::Val{K}) where {R, M, K} = B[SVector{K}(ntuple(i -> o + i, Val(K))), StaticArrays.SOneTo(M)]
_row_block(B::AbstractMatrix, o, ::Val{K}) where {K} = B[(o + 1):(o + K), :]
local_coordinates(ch::TargetManifold, t::AbstractTarget) = ch.basis' * (parameter_vector(t) - parameter_vector(ch.target))

"""
    restrict(ch::TargetManifold, names::Tuple{Vararg{Symbol}}) -> TargetManifold

The target manifold on the parameters `names` ([`parameter_names`](@ref)'
vocabulary), every other parameter held at the anchor — a calibration's
complexity as a manifold, as `restrict` of a [`LensManifold`](@ref). The
thin-plate kinds' constraint still holds: the free directions are those of
the named parameters that keep the values pure bending (no affine term in the
interpolant), so
a height field restricted to `m ≤ 3` of its heights has no direction left
(three plane conditions on `m` unknowns), and to `m` heights in general
position `m − 3`.
"""
function restrict(ch::TargetManifold{<:PlanarTarget}, names::Tuple{Symbol, Vararg{Symbol}})
    all_names = parameter_names(ch.target)
    for n in names
        n in all_names || throw(
            ArgumentError("restrict: the target has no parameter $(repr(n)); its parameters are $(join(String.(all_names), ", "))")
        )
    end
    return _target_manifold(ch.target, Symbol[n for n in all_names if !(n in names)])
end
