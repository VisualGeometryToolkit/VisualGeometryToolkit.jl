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
#                         board whose warp was resolved to measured points.
#   PlanarTarget{W}       a planar DESIGN — its nominal points, or the reference
#                         they are resolved from — and a warp `W` taking them to
#                         the physical target: `points = W(nominal)`.
#
# THE DESIGN REFERENCE. A content-addressed design (a BlobBoards board uid) is
# named by `design`; VGC holds the reference and, once resolved, the nominal
# points, and never parses a design file — the package that owns the design
# (BlobBoards) resolves a reference into nominal points ([`resolve`](@ref)). An
# unresolved `PlanarTarget` (a pose-only board) has a reference and no points;
# asking for its points is an error naming the reference.
#
# THE WARP. `Rigid` — the identity, no parameters — is exactly a printed board
# taken as designed. The slot is the extension point for a deforming target: an
# `Affine` warp, a `ThinPlateSpline{K}` with a bending-energy regularizer.
# Adding one is purely additive: a new `AbstractWarp` implementing the warp
# protocol below.
#
# CALIBRATABLE TARGETS. A target whose geometry an estimator may refine
# implements a parameter protocol, the target analogue of the lens leaf
# protocol (lens_chart.jl), on the same manifold contract the
# block-coordinate-descent and LM machinery consume:
#
#   parameter_vector(t)         the parameter vector θ (an SVector; empty for Rigid)
#   parameter_names(t)          its names, so a configuration frees or fixes them by name
#   parameter_bounds(t)         its bounds (unbounded unless the warp says otherwise)
#   with_parameters(t, θ)       the same target rebuilt from θ
#   parameters(t)               both, as a NamedTuple
#   minimal_manifold(t)         a `TargetManifold`: θ's offsets from an anchor;
#                               `action_jacobian(ch, x)` = ∂W(x)/∂θ at a nominal point x
#   regularizer(t)              the prior's residual rows r(θ) (‖r‖² the penalty,
#                               e.g. a thin-plate spline's bending energy); empty for Rigid
#
# A `PlanarTarget` forwards the protocol to its warp, whose own protocol is
# `parameter_vector`, `parameter_names`, `with_parameters(w, θ)`,
# `warp_jacobian(w, x)` (2 × n) and `regularizer(w)`.

"""
    AbstractTarget

A calibration target as a model: [`points`](@ref)`(target)` returns its points
in its own frame `TargetFrame{id}` — 2-vectors on the plane `z = 0` for a planar
target, 3-vectors for a solid one — plain or `Uncertain`. See the header of
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
marker constellation, or a board whose warp was resolved to measured points.

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
# Warps
# ------------------------------------------------------------------------------

"""
    AbstractWarp

The deformation of a planar target's nominal design: a map of plane points
`w(p)`, with the warp protocol `parameter_vector`, `parameter_names`,
`with_parameters(w, θ)`, `warp_jacobian(w, x)` (∂w(x)/∂θ, 2 × n) and
`regularizer(w)`. [`Rigid`](@ref) is the one shipped.
"""
abstract type AbstractWarp end

"""
    Rigid() <: AbstractWarp

The identity warp: the target is its design, exactly. No parameters, no
regularizer.
"""
struct Rigid <: AbstractWarp end
(::Rigid)(p) = p

# The warp's parameters (the protocol of lens_chart.jl); empty for `Rigid`.
parameter_vector(::Rigid) = SVector{0, Float64}()

parameter_names(::Rigid) = ()
with_parameters(w::Rigid, θ::AbstractVector) =
    (isempty(θ) || throw(ArgumentError("Rigid takes no parameters; got $(length(θ))")); w)

"""
    warp_jacobian(w, x) -> SMatrix{2, n}

∂w(x)/∂θ at the nominal plane point `x`.
"""
warp_jacobian(::Rigid, x) = zero(SMatrix{2, 0, Float64})

"""
    regularizer(w | target) -> SVector

The prior's residual rows `r(θ)` at the current parameters: `‖r‖²` is the
penalty a calibrating estimator adds (a thin-plate spline's bending energy, a
shrinkage toward the design). Empty for [`Rigid`](@ref).
"""
regularizer(::Rigid) = SVector{0, Float64}()

# ------------------------------------------------------------------------------
# PlanarTarget
# ------------------------------------------------------------------------------

"""
    PlanarTarget(nominal; warp = Rigid())                  a design given by its points
    PlanarTarget(design::Symbol; nominal = nothing, warp = Rigid())   a content-addressed design

A planar target: a DESIGN — its nominal plane points, and/or the reference
`design` they are resolved from (a BlobBoards board uid) — and a warp `W`
taking the design to the physical target, `points(t) = W.(nominal)` (2-vectors
on the plane `z = 0`). An unresolved target (a reference, no nominal points
yet) is a pose-only board; [`resolve`](@ref) supplies its points, and its
[`points`](@ref) is an error until then. VGC never parses a design file: the
design's package resolves the reference.

The nominal points are plain numbers (stored as `Float64`) or lengths (stored
in metres), all one form ([`has_units`](@ref)); `ustrip(u, t)` and
[`with_unit`](@ref)`(t, u)` move between them. An unresolved target holds no
lengths: it is neither form, and both return it unchanged.
"""
struct PlanarTarget{W <: AbstractWarp, N <: Union{Nothing, Vector{<:SVector{2}}}} <: AbstractTarget
    design::Union{Nothing, Symbol}
    nominal::N
    warp::W
    function PlanarTarget(design::Union{Nothing, Symbol}, nominal, warp::W) where {W <: AbstractWarp}
        (design === nothing && nominal === nothing) &&
            throw(ArgumentError("a planar target has a design reference, nominal points, or both"))
        n = nominal === nothing ? nothing : _nominal_points(nominal)
        return new{W, typeof(n)}(design, n, warp)
    end
end
# The nominal points stored (an empty design is a plain one, as before).
function _nominal_points(nominal)
    ps = _check_one_form([_stored_point(Val(2), geom(p)) for p in nominal], "PlanarTarget")
    return isempty(ps) ? SVector{2, Float64}[] : ps
end
PlanarTarget(nominal::AbstractVector; warp::AbstractWarp = Rigid()) = PlanarTarget(nothing, nominal, warp)
PlanarTarget(design::Symbol; nominal = nothing, warp::AbstractWarp = Rigid()) = PlanarTarget(design, nominal, warp)

"""
    resolve(t::PlanarTarget, nominal) -> PlanarTarget

The target with its design's nominal points supplied — what the design's
owner returns for the reference `t.design`.
"""
resolve(t::PlanarTarget, nominal::AbstractVector) = PlanarTarget(t.design, nominal, t.warp)

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

function points(t::PlanarTarget)
    t.nominal === nothing && throw(
        ArgumentError("points: the planar target $(repr(t.design)) is unresolved — its design's owner resolves the reference (`resolve`)")
    )
    return [t.warp(p) for p in t.nominal]
end
Base.:(==)(a::PlanarTarget, b::PlanarTarget) = a.design === b.design && isequal(a.nominal, b.nominal) && a.warp == b.warp

# Units: a resolved target's nominal points move; an unresolved one has none.
has_units(t::PlanarTarget) = is_resolved(t) && has_units(t.nominal)
FlexUnits.ustrip(u::Units, t::PlanarTarget) =
    is_resolved(t) ? PlanarTarget(t.design, [_ustrip_point(u, p) for p in t.nominal], t.warp) : t
with_unit(t::PlanarTarget, u::Units) =
    is_resolved(t) ? PlanarTarget(t.design, [_attach_point(u, p) for p in t.nominal], t.warp) : t

# --- the calibratable-target protocol ------------------------------------------------

# A calibratable target's parameters are its warp's.
parameter_vector(t::PlanarTarget) = parameter_vector(t.warp)

parameter_names(t::PlanarTarget) = parameter_names(t.warp)
with_parameters(t::PlanarTarget, θ::AbstractVector) = PlanarTarget(t.design, t.nominal, with_parameters(t.warp, θ))
parameters(t::PlanarTarget) = NamedTuple{parameter_names(t)}(Tuple(parameter_vector(t)))
regularizer(t::PlanarTarget) = regularizer(t.warp)

"""
    TargetManifold(target) <: Manifold

A calibratable target's manifold anchored at `target`: its
[`parameter_vector`](@ref)' coordinates as offsets from the anchor (linear);
`action_jacobian(ch, x)` is ∂points/∂θ at the nominal point `x`
(`warp_jacobian`). `minimal_manifold(target)` is this manifold. Zero-dimensional
for a `Rigid` target.
"""
struct TargetManifold{T <: AbstractTarget} <: Manifold
    target::T
end
minimal_manifold(t::PlanarTarget) = TargetManifold(t)
ambient_coordinates(t::PlanarTarget) = parameter_vector(t)
tangent_size(ch::TargetManifold) = length(parameter_vector(ch.target))
model(ch::TargetManifold) = ch.target
retract(ch::TargetManifold, v) = with_parameters(ch.target, parameter_vector(ch.target) + v)
recenter(ch::TargetManifold, v) = TargetManifold(retract(ch, v))
retract_jacobian(ch::TargetManifold) = (n = tangent_size(ch); SMatrix{n, n, Float64}(LinearAlgebra.I))
retract_jacobian(ch::TargetManifold, ::StaticVector) = retract_jacobian(ch)
action_jacobian(ch::TargetManifold{<:PlanarTarget}, x) = warp_jacobian(ch.target.warp, x)
local_coordinates(ch::TargetManifold, t::AbstractTarget) = parameter_vector(t) - parameter_vector(ch.target)
