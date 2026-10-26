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
        _check_one_form(ps, "PointTarget")
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

"""
    PlanarTarget(nominal; calibration = Rigid())                  a design given by its points
    PlanarTarget(design::Symbol; nominal = nothing, calibration = Rigid())   a content-addressed design

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
design.

The nominal points are plain numbers (stored as `Float64`) or lengths (stored
in metres), all one form ([`has_units`](@ref)); a calibration with lengths
(an `Affine`'s translation, a thin-plate spline's nodes and displacements, a
height field's nodes and heights) is in the same form, or the constructor
raises FlexUnits' `DimensionError`. `ustrip(u, t)` and [`with_unit`](@ref)`(t, u)`
move the points and the calibration together. An unresolved target holds no
points: it is neither form, and both move only its calibration.
"""
struct PlanarTarget{C <: TargetCalibration, N <: Union{Nothing, Vector{<:SVector{2}}}, S <: Union{C, Uncertain{C}}} <: AbstractTarget
    design::Union{Nothing, Symbol}
    nominal::N
    calibration::S
    function PlanarTarget(design::Union{Nothing, Symbol}, nominal, calibration::Union{TargetCalibration, Uncertain{<:TargetCalibration}})
        (design === nothing && nominal === nothing) &&
            throw(ArgumentError("a planar target has a design reference, nominal points, or both"))
        n = nominal === nothing ? nothing : _nominal_points(nominal)
        _check_calibration(n, calibration)
        return new{typeof(geom(calibration)), typeof(n), typeof(calibration)}(design, n, calibration)
    end
end
# The nominal points stored (an empty design is a plain one, as before).
function _nominal_points(nominal)
    ps = _check_one_form([_stored_point(Val(2), geom(p)) for p in nominal], "PlanarTarget")
    return isempty(ps) ? SVector{2, Float64}[] : ps
end
# An uncertain calibration's form covers its parameters; a calibration with
# lengths is in the form of the points it moves.
function _check_calibration(nominal, c)
    if c isa Uncertain
        n = length(parameter_vector(geom(c)))
        size(Statistics.cov(c)) == (n, n) || throw(
            DimensionMismatch("an uncertain calibration's covariance is $(n) × $(n) (its parameters); got $(size(Statistics.cov(c)))")
        )
    end
    (nominal === nothing || isempty(nominal) || !_has_lengths(geom(c))) && return nothing
    has_units(first(nominal)) == has_units(geom(c)) && return nothing
    throw(DimensionError(first(nominal), _length_sample(geom(c))))
end
PlanarTarget(nominal::AbstractVector; calibration = Rigid()) = PlanarTarget(nothing, nominal, calibration)
PlanarTarget(design::Symbol; nominal = nothing, calibration = Rigid()) = PlanarTarget(design, nominal, calibration)

"""
    calibration(t::PlanarTarget) -> TargetCalibration or Uncertain

The target's calibration as stored: a [`TargetCalibration`](@ref), or an
[`Uncertain`](@ref) one carrying its covariance (`geom` unwraps it).
"""
calibration(t::PlanarTarget) = t.calibration

"""
    design(t::PlanarTarget) -> Vector{SVector{2}} or Symbol

The target's design: its nominal plane points when it is resolved
([`is_resolved`](@ref)), the reference they are resolved from (a `Symbol`)
when it is not. The physical target is `calibration(t)` applied to it.
"""
design(t::PlanarTarget) = something(t.nominal, t.design)

"""
    resolve(t::PlanarTarget, nominal) -> PlanarTarget

The target with its design's nominal points supplied — what the design's
owner returns for the reference `t.design`.
"""
resolve(t::PlanarTarget, nominal::AbstractVector) = PlanarTarget(t.design, nominal, t.calibration)

"""
    is_resolved(t) -> Bool

Whether a target carries its points (a `PlanarTarget` holding only a design
reference does not).
"""
is_resolved(::AbstractTarget) = true
is_posed(::AbstractTarget) = false   # an unposed target; `PosedTarget` (rig.jl) answers true

# The camera parallel (cameras.jl `_unposed_only`): an unposed target is its own
# model, and the posed-only verbs refuse it by name. A posed target type answers
# them with its own methods, so on one that lacks them this is the MethodError
# it would have been.
@noinline _refuse_unposed_target(what, t) = throw(
    ArgumentError(
        "$(what): the target is unposed (a $(nameof(typeof(t))), `is_posed` is false) — pose it: " *
            "`PosedTarget(model, X)`"
    )
)
_unposed_target_only(f, what, t, args...) =
    is_posed(t) ? throw(MethodError(f, (t, args...))) : _refuse_unposed_target(what, t)
model(t::AbstractTarget) = is_posed(t) ? throw(MethodError(model, (t,))) : t
pose(t::AbstractTarget) = _unposed_target_only(pose, "pose", t)
extrinsics(t::AbstractTarget) = _unposed_target_only(extrinsics, "extrinsics", t)
is_resolved(t::PlanarTarget) = t.nominal !== nothing

@noinline _refuse_unresolved(t) = throw(
    ArgumentError("points: the planar target $(repr(t.design)) is unresolved — its design's owner resolves the reference (`resolve`)")
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
    a.design === b.design && isequal(a.nominal, b.nominal) && a.calibration == b.calibration

# Units: a resolved target's nominal points move, and with them a calibration
# that has lengths; an unresolved one moves only its calibration.
has_units(t::PlanarTarget) = is_resolved(t) ? has_units(t.nominal) : has_units(geom(t.calibration))
FlexUnits.ustrip(u::Units, t::PlanarTarget) = PlanarTarget(
    t.design, is_resolved(t) ? [_ustrip_point(u, p) for p in t.nominal] : nothing, _ustrip_calibration(u, t.calibration)
)
with_unit(t::PlanarTarget, u::Units) = PlanarTarget(
    t.design, is_resolved(t) ? [_attach_point(u, p) for p in t.nominal] : nothing, _attach_calibration(u, t.calibration)
)

# --- the calibratable-target protocol ------------------------------------------------

# A calibratable target's parameters are its calibration's. A new parameter
# vector is a new calibration: `with_parameters` drops an uncertain one's
# covariance, which described the old parameters.
parameter_vector(t::PlanarTarget) = parameter_vector(geom(t.calibration))

parameter_names(t::PlanarTarget) = parameter_names(geom(t.calibration))
with_parameters(t::PlanarTarget, θ::AbstractVector) = PlanarTarget(t.design, t.nominal, with_parameters(geom(t.calibration), θ))
parameters(t::PlanarTarget) = NamedTuple{parameter_names(t)}(Tuple(parameter_vector(t)))
regularizer(t::PlanarTarget) = regularizer(geom(t.calibration))
is_planar(::PlanarTarget{C}) where {C} = is_planar(C)
gauge_names(t::PlanarTarget) = gauge_names(geom(t.calibration))

"""
    TargetManifold(target; free = ()) <: Manifold

A calibratable target's manifold anchored at `target`, with its GAUGES held
fixed: the directions of [`parameter_vector`](@ref)`(target)` that images
cannot see because the target's pose absorbs them ([`gauge_names`](@ref)) —
an [`Affine`](@ref)'s uniform scale, a [`HeightField`](@ref)'s piston and two
tilts. `minimal_manifold(target)` is this manifold; `free` names gauge
directions to leave free (`TargetManifold(t; free = (:scale,))` when a
measured length — a ruler, motion capture — pins the scale).

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

The gauge directions are evaluated ONCE, at the anchor the manifold is built
on, and `recenter` keeps `B`: a fit moves on one linear slice through its
starting parameters (for an `Affine`, the component of `A` along the anchor's
`A₀` stays where it started), and the manifold's type does not change.
Zero-dimensional for a `Rigid` target.
"""
struct TargetManifold{T <: AbstractTarget, B <: SMatrix} <: Manifold
    target::T
    basis::B                      # n × m, orthonormal columns: the free directions of θ
    fixed_gauges::Vector{Symbol}  # the gauge directions held fixed
    fixed::Vector{Symbol}         # the parameters held fixed (`restrict`)
end
function TargetManifold(t::PlanarTarget; free::Tuple{Vararg{Symbol}} = ())
    gauges = gauge_names(t)
    for g in free
        g in gauges || throw(
            ArgumentError("TargetManifold: no gauge direction $(repr(g)) to free; the target's are $(isempty(gauges) ? "none" : join(gauges, ", "))")
        )
    end
    return _target_manifold(t, Symbol[g for g in gauges if !(g in free)], Symbol[])
end

# B: the free parameter axes E (every axis but the fixed parameters), then
# within them the orthogonal complement of the fixed gauge directions D
# projected on them (an SVD of EᵀD, each direction normalized first). The span
# of B is exactly the complement of span(fixed axes ∪ D).
function _target_manifold(t::PlanarTarget, fixed_gauges::Vector{Symbol}, fixed::Vector{Symbol})
    c = geom(t.calibration)
    names = parameter_names(c)
    n = length(names)
    E = Matrix{Float64}(LinearAlgebra.I, n, n)[:, [i for i in 1:n if !(names[i] in fixed)]]
    B = if isempty(fixed_gauges)
        E
    else
        gn = gauge_names(c)
        D = gauge_directions(c)[:, [findfirst(==(g), gn) for g in fixed_gauges]]
        Dp = E' * (D ./ max.(LinearAlgebra.norm.(eachcol(D))', floatmin(Float64)))
        F = svd(Dp; full = true)
        r = count(>(sqrt(eps(Float64)) * max(1.0, maximum(F.S; init = 0.0))), F.S)
        E * F.U[:, (r + 1):end]
    end
    return TargetManifold(t, SMatrix{n, size(B, 2), Float64}(B), fixed_gauges, fixed)
end

minimal_manifold(t::PlanarTarget) = TargetManifold(t)
ambient_coordinates(t::PlanarTarget) = parameter_vector(t)
tangent_size(ch::TargetManifold) = size(ch.basis, 2)
model(ch::TargetManifold) = ch.target
retract(ch::TargetManifold, v) = with_parameters(ch.target, parameter_vector(ch.target) + ch.basis * v)
recenter(ch::TargetManifold, v) = TargetManifold(retract(ch, v), ch.basis, ch.fixed_gauges, ch.fixed)
retract_jacobian(ch::TargetManifold) = ch.basis
retract_jacobian(ch::TargetManifold, ::StaticVector) = ch.basis
action_jacobian(ch::TargetManifold{<:PlanarTarget}, x) = calibration_jacobian(geom(ch.target.calibration), x) * ch.basis
local_coordinates(ch::TargetManifold, t::AbstractTarget) = ch.basis' * (parameter_vector(t) - parameter_vector(ch.target))

"""
    restrict(ch::TargetManifold, names::Tuple{Vararg{Symbol}}) -> TargetManifold

The target manifold on the parameters `names` ([`parameter_names`](@ref)'
vocabulary), every other parameter held at the anchor, its gauges still fixed
— a calibration's complexity as a manifold, as `restrict` of a
[`LensManifold`](@ref).
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
