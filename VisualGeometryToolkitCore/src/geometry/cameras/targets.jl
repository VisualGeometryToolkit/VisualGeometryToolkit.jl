# ==============================================================================
# Targets — the calibration target as a MODEL, the analogue of a camera model
# ==============================================================================
#
# A target is what a camera observes and a measurement names: a set of points
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
# `calibration_jacobian(c, x)` and `regularizer(c)`.

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
# base form); an `Uncertain` one keeps its covariance (m² for a physical point).
_target_point(p::StaticVector{2}) = _stored_point(Val(2), p)
_target_point(p::StaticVector{3}) = _stored_point(Val(3), p)
_target_point(p::AbstractVector) = length(p) == 2 ? _stored_point(Val(2), p) : length(p) == 3 ? _stored_point(Val(3), p) :
    throw(ArgumentError("a target point is a 2-vector (a plane's) or a 3-vector; got $(length(p)) entries"))
_target_point(p::Uncertain) = (v = _target_point(geom(p)); Uncertain(v, CovMat(SMatrix{length(v), length(v), Float64}(Statistics.cov(p)))))

# ------------------------------------------------------------------------------
# PointTarget
# ------------------------------------------------------------------------------

"""
    PointTarget(points)

A target given by its points, explicitly, in its own frame: 2-vectors (a
plane's) or 3-vectors, all of one dimension, plain or `Uncertain` (an
estimated constellation with its covariances). A motion-capture body's
marker constellation, or a board whose calibration was resolved to measured points.

A physical target's points are lengths, stored in metres (its covariances in
m²); plain points are stored as `Float64`. All are one form or the other
([`has_units`](@ref)); `ustrip(u, t)` and [`with_unit`](@ref)`(t, u)` move
between them.
"""
struct PointTarget{P} <: AbstractTarget
    points::Vector{P}
    function PointTarget(pts::AbstractVector)
        ps = map(_target_point, pts)
        isempty(ps) || allequal(length(geom(p)) for p in ps) ||
            throw(ArgumentError("a target's points are all 2-vectors or all 3-vectors"))
        _check_one_form(ps)
        P = isempty(ps) ? SVector{3, Float64} : typejoin(map(typeof, ps)...)
        return new{P}(Vector{P}(ps))
    end
end

points(t::PointTarget) = t.points
Base.:(==)(a::PointTarget, b::PointTarget) = a.points == b.points
has_units(t::PointTarget) = any(has_units, points(t))
FlexUnits.ustrip(u::Units, t::PointTarget) = PointTarget([_ustrip_point(u, p) for p in points(t)])
with_unit(t::PointTarget, u::Units) = PointTarget([_attach_point(u, p) for p in points(t)])

# ------------------------------------------------------------------------------
# PlanarTarget
# ------------------------------------------------------------------------------

# The construction routes of a PlanarTarget (private): checked, or a unit move.
struct _Checked end
struct _UnitMove end

"""
    PlanarTarget(nominal; calibration = Rigid(), board = nothing)                  a design given by its points
    PlanarTarget(design::Symbol; nominal = nothing, calibration = Rigid(), board = nothing)   a content-addressed design

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
(an `Affine`'s translation, a thin-plate spline's nodes and displacements, a
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
"""
struct PlanarTarget{C <: TargetCalibration, N <: Union{Nothing, Vector{<:SVector{2}}}, S <: Union{C, Uncertain{C}}} <: AbstractTarget
    reference::Union{Nothing, Symbol}
    nominal::N
    calibration::S
    board::Union{Nothing, Board}
    # The private `_UnitMove()` route skips the board containment check: the
    # points are known to lie on the board — a unit move of a target that held
    # them there (`ustrip`, `with_unit`), whose rounding must not refuse it.
    # Every public construction is `_Checked()`.
    function PlanarTarget(
            reference::Union{Nothing, Symbol}, nominal, calibration::Union{TargetCalibration, Uncertain{<:TargetCalibration}},
            board::Union{Nothing, Board}, route::Union{_Checked, _UnitMove}
        )
        (reference === nothing && nominal === nothing) &&
            throw(ArgumentError("a planar target has a design reference, nominal points, or both"))
        n = nominal === nothing ? nothing : _nominal_points(nominal)
        _check_calibration(n, calibration)
        # the points, the calibration's lengths and the board: one form
        _check_one_form((n === nothing || isempty(n) ? nothing : first(n), geom(calibration), board))
        _check_board(n, board, calibration, route isa _Checked)
        return new{typeof(geom(calibration)), typeof(n), typeof(calibration)}(reference, n, calibration, board)
    end
end
# The public positional form (the board optional): always checked.
PlanarTarget(reference::Union{Nothing, Symbol}, nominal, calibration::Union{TargetCalibration, Uncertain{<:TargetCalibration}},
             board::Union{Nothing, Board} = nothing) = PlanarTarget(reference, nominal, calibration, board, _Checked())
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
PlanarTarget(nominal::AbstractVector; calibration = Rigid(), board = nothing) = PlanarTarget(nothing, nominal, calibration, board)
PlanarTarget(reference::Symbol; nominal = nothing, calibration = Rigid(), board = nothing) =
    PlanarTarget(reference, nominal, calibration, board)

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
    resolve(t::PlanarTarget, nominal) -> PlanarTarget

The target with its design's nominal points supplied — what the design's
owner returns for the reference `reference(t)`.
"""
resolve(t::PlanarTarget, nominal::AbstractVector) = PlanarTarget(t.reference, nominal, t.calibration, t.board)

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
it — decided by the calibration's type, so the result type is inferred. An
`ArgumentError` naming the reference for an unresolved target.
"""
points(t::PlanarTarget) = _calibrated_points(t, geom(t.calibration), t.nominal)
_calibrated_points(t, _, ::Nothing) = _refuse_unresolved(t)
_calibrated_points(_, c, nominal) = [c(p) for p in nominal]
Base.:(==)(a::PlanarTarget, b::PlanarTarget) =
    a.reference === b.reference && isequal(a.nominal, b.nominal) && a.calibration == b.calibration && a.board == b.board
Base.hash(t::PlanarTarget, h::UInt) =
    hash(t.board, hash(t.calibration, hash(t.nominal, hash(t.reference, hash(:PlanarTarget, h)))))

# Units: a resolved target's nominal points move, and with them a calibration
# that has lengths; an unresolved one moves only its calibration.
# The board moves with them, by the same rule (an unresolved target's board
# moves when it is in the form being left).
has_units(t::PlanarTarget) = is_resolved(t) ? has_units(t.nominal) : has_units(geom(t.calibration))
FlexUnits.ustrip(u::Units, t::PlanarTarget) = PlanarTarget(
    t.reference, is_resolved(t) ? [_ustrip_point(u, p) for p in t.nominal] : nothing, _ustrip_calibration(u, t.calibration),
    _move_board(ustrip, u, t.board, true), _UnitMove()
)
with_unit(t::PlanarTarget, u::Units) = PlanarTarget(
    t.reference, is_resolved(t) ? [_attach_point(u, p) for p in t.nominal] : nothing, _attach_calibration(u, t.calibration),
    _move_board((u, bd) -> with_unit(bd, u), u, t.board, false), _UnitMove()
)
# The one-form rule's view of a planar target (`_check_one_form`): its nominal
# points', calibration's and board's forms, which its constructor already
# holds to one — read from the stored parts, never by applying the
# calibration. An unresolved target whose calibration or board carries lengths
# has their form, so a plain pose on it is refused as on a resolved one.
function _unit_form(t::PlanarTarget)
    n = t.nominal
    forms = (n === nothing || isempty(n) ? :none : _unit_form(first(n)), _unit_form(geom(t.calibration)), _unit_form(t.board))
    return :physical in forms ? :physical : :plain in forms ? :plain : :none
end
_form_sample(t::PlanarTarget) = _form_sample(
    t.nominal !== nothing && !isempty(t.nominal) ? first(t.nominal) :
        _unit_form(geom(t.calibration)) !== :none ? geom(t.calibration) : t.board
)
_move_board(_, _, ::Nothing, _) = nothing
_move_board(f, u, bd::Board, from_physical::Bool) = has_units(bd) == from_physical ? f(u, bd) : bd

# --- the calibratable-target protocol ------------------------------------------------

# A calibratable target's parameters are its calibration's. A new parameter
# vector is a new calibration: `with_parameters` drops an uncertain one's
# covariance, which described the old parameters.
parameter_vector(t::PlanarTarget) = parameter_vector(geom(t.calibration))

parameter_names(t::PlanarTarget) = parameter_names(geom(t.calibration))
with_parameters(t::PlanarTarget, θ::AbstractVector) = PlanarTarget(t.reference, t.nominal, with_parameters(geom(t.calibration), θ), t.board)
parameters(t::PlanarTarget) = NamedTuple{parameter_names(t)}(Tuple(parameter_vector(t)))
regularizer(t::PlanarTarget) = regularizer(geom(t.calibration))
is_planar(::PlanarTarget{C}) where {C} = is_planar(C)
gauge_names(t::PlanarTarget) = gauge_names(geom(t.calibration))

"""
    TargetManifold(target; free = ()) <: Manifold

A calibratable target's manifold anchored at `target`, with its GAUGES held
fixed: the directions of [`parameter_vector`](@ref)`(target)` that images
cannot see because the target's pose absorbs them ([`gauge_names`](@ref)) —
an [`Affine`](@ref)'s translation, in-plane rotation and scale, a
[`ThinPlateSpline`](@ref)'s affine part, a [`HeightField`](@ref)'s piston and
two tilts. `minimal_manifold(target)` is this manifold, and it means "the
pose is estimated jointly". `free` names gauge directions to leave free, one
`Symbol` or a tuple: with a known pose (motion capture, a fixture), free them
all; with a measured length (a ruler), `TargetManifold(t; free = :scale)`.

The chart is linear: `retract(ch, v) = with_parameters(target, θ₀ + B v)`,
`B` an `n × m` matrix whose orthonormal columns span the directions of θ
orthogonal to the fixed ones, so `tangent_size(ch) = m` and
`retract_jacobian(ch) = B`. With nothing fixed `B` is the identity and the
coordinates are the parameters' offsets, named by [`parameter_names`](@ref);
with a gauge fixed they are coordinates in that orthonormal basis.
`action_jacobian(ch, x) = calibration_jacobian(c, x) · B` at the design point
`x`; `local_coordinates(ch, t) = Bᵀ (θ(t) − θ₀)`. [`restrict`](@ref)`(ch,
names)` keeps only the parameters `names` free (the others fixed, as a
`TargetManifold` with the same gauges).

THE BASIS IS FIXED AT THE ANCHOR. The gauge directions are evaluated once,
where the manifold is built (an `Affine`'s rotation and scale directions
depend on `A₀`), and `recenter` keeps `B`: a fit moves on one linear slice
through its starting parameters (for an `Affine`, the components of `A` along
`A₀` and `J A₀` stay where they started), and the manifold's type does not
change. A covariance estimated on it is in that slice's gauge; building a new
`minimal_manifold` from the result re-anchors the slice and so changes the
gauge.

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
    fixed_gauges::Vector{Symbol}  # the gauge directions held fixed
    fixed::Vector{Symbol}         # the parameters held fixed (`restrict`)
end
function TargetManifold(t::PlanarTarget; free::Union{Symbol, Tuple{Vararg{Symbol}}} = ())
    gauges = gauge_names(t)
    freed = free isa Symbol ? (free,) : free
    for g in freed
        g in gauges || throw(
            ArgumentError("TargetManifold: no gauge direction $(repr(g)) to free; the target's are $(isempty(gauges) ? "none" : join(gauges, ", "))")
        )
    end
    return _target_manifold(t, Symbol[g for g in gauges if !(g in freed)], Symbol[])
end

# The largest parameter count with a static basis: below it a static basis
# compiles about as fast as a Matrix one, above it far slower (the docstring's
# measurements).
const _STATIC_BASIS_MAX = 18

# B: within the free parameter axes E (every axis but the fixed parameters),
# the orthogonal complement of the fixed gauge combinations that LIE in
# span(E) — span(E) ∩ span(D), D the fixed gauge directions. A gauge with only
# a component on the free axes is not a gauge there and fixes nothing. With
# Q an orthonormal basis of span(D), the combinations are Q c with
# ‖(I − EEᵀ) Q c‖ ≤ tol ‖c‖ — the right singular vectors of (I − EEᵀ) Q whose
# singular values are ≤ tol, tol = 1e-6 (loose, so a near-gauge counts).
const _GAUGE_IN_SPAN_RTOL = 1.0e-6
function _target_manifold(t::PlanarTarget, fixed_gauges::Vector{Symbol}, fixed::Vector{Symbol})
    c = geom(t.calibration)
    names = parameter_names(c)
    n = length(names)
    free_axes = [i for i in 1:n if !(names[i] in fixed)]
    E = zeros(n, length(free_axes))                                    # the free parameter axes
    foreach(((j, i),) -> E[i, j] = 1.0, enumerate(free_axes))
    B = if isempty(fixed_gauges) || isempty(free_axes)
        E
    else
        gn = gauge_names(c)
        D = gauge_directions(c)[:, [findfirst(==(g), gn) for g in fixed_gauges]]
        # unit columns: the rank cutoff below is then scale-free (a height
        # field's tilt columns are its node coordinates, centimetres in metres)
        for j in axes(D, 2)
            nj = norm(view(D, :, j))
            nj > 0 && (D[:, j] ./= nj)
        end
        FD = svd(D)
        Q = FD.U[:, FD.S .> sqrt(eps(Float64)) * maximum(FD.S)]          # span(D), orthonormal
        inside = if length(free_axes) == n
            Q                                                           # nothing fixed: every gauge is in span(E)
        else
            outside = Q[setdiff(1:n, free_axes), :]                     # (I − EEᵀ) Q, in the fixed rows
            FO = svd(outside; full = true)
            σ = [FO.S; zeros(size(Q, 2) - length(FO.S))]
            Q * FO.V[:, σ .<= _GAUGE_IN_SPAN_RTOL]                      # the gauge combinations in span(E)
        end
        if size(inside, 2) == 0
            E
        else
            FI = svd(E' * inside; full = true)
            r = count(>(sqrt(eps(Float64))), FI.S)
            E * FI.U[:, (r + 1):end]
        end
    end
    return TargetManifold(t, _basis(B, Val(n <= _STATIC_BASIS_MAX)), fixed_gauges, fixed)
end
_basis(B::Matrix{Float64}, ::Val{true}) = SMatrix{size(B, 1), size(B, 2), Float64}(B)
_basis(B::Matrix{Float64}, ::Val{false}) = B

minimal_manifold(t::PlanarTarget) = TargetManifold(t)
ambient_coordinates(t::PlanarTarget) = parameter_vector(t)
tangent_size(ch::TargetManifold) = size(ch.basis, 2)
model(ch::TargetManifold) = ch.target
retract(ch::TargetManifold, v) = with_parameters(ch.target, parameter_vector(ch.target) + ch.basis * v)
recenter(ch::TargetManifold, v) = TargetManifold(retract(ch, v), ch.basis, ch.fixed_gauges, ch.fixed)
retract_jacobian(ch::TargetManifold) = ch.basis
retract_jacobian(ch::TargetManifold, ::AbstractVector) = ch.basis
action_jacobian(ch::TargetManifold{<:PlanarTarget}, x) = _calibration_action(geom(ch.target.calibration), x, ch.basis)
# ∂C(x)/∂θ · B, multiplying only the rows and columns the calibration's Jacobian
# can have nonzero: a height field moves only z, a thin-plate spline's two rows
# each depend on one of its two value blocks.
_calibration_action(c::TargetCalibration, x, B) = calibration_jacobian(c, x) * B
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
complexity as a manifold, as `restrict` of a [`LensManifold`](@ref). Of the
fixed gauges, only the combinations that lie inside the span of the free
parameters stay fixed: a gauge direction with a component on a held parameter
cannot move there, so it constrains nothing (restricting an [`Affine`](@ref)
to `(:a11,)` keeps that one direction free).
"""
function restrict(ch::TargetManifold{<:PlanarTarget}, names::Tuple{Symbol, Vararg{Symbol}})
    all_names = parameter_names(ch.target)
    for n in names
        n in all_names || throw(
            ArgumentError("restrict: the target has no parameter $(repr(n)); its parameters are $(join(String.(all_names), ", "))")
        )
    end
    return _target_manifold(ch.target, ch.fixed_gauges, Symbol[n for n in all_names if !(n in names)])
end
