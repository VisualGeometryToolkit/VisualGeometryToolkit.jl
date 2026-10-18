# ==============================================================================
# Targets — the calibration target as a MODEL, the analogue of a camera model
# ==============================================================================
#
# A target is what a camera observes and a measurement names: a set of points
# in its own frame, `TargetFrame{id}`. Its geometry is independent of where it
# is mounted — the same target can sit in several rigs (`Mounted`, rig.jl) —
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
# protocol (lens_parameters.jl), on the same manifold contract the
# block-coordinate-descent and LM machinery consume:
#
#   target_parameters(t)        the parameter vector θ (an SVector; empty for Rigid)
#   target_parameter_names(t)   its names, so a configuration frees or fixes them by name
#   with_parameters(t, θ)       the same target rebuilt from θ
#   parameters(t)               both, as a NamedTuple
#   minimal_manifold(t)         a `TargetManifold`: θ's offsets from an anchor;
#                               `action_jacobian(ch, x)` = ∂W(x)/∂θ at a nominal point x
#   regularizer(t)              the prior's residual rows r(θ) (‖r‖² the penalty,
#                               e.g. a thin-plate spline's bending energy); empty for Rigid
#
# A `PlanarTarget` forwards the protocol to its warp, whose own protocol is
# `warp_parameters`, `warp_parameter_names`, `with_parameters(w, θ)`,
# `warp_jacobian(w, x)` (2 × n) and `regularizer(w)`.

"""
    AbstractTarget

A calibration target as a model: [`points`](@ref)`(target)` returns its points
in its own frame `TargetFrame{id}` — 2-vectors on the plane `z = 0` for a planar
target, 3-vectors for a solid one — plain or `Uncertain`. See the header of
targets.jl for the calibratable-target protocol.
"""
abstract type AbstractTarget end

_target_point(p::StaticVector{2}) = SVector{2, Float64}(p)
_target_point(p::StaticVector{3}) = SVector{3, Float64}(p)
_target_point(p::AbstractVector) = length(p) == 2 ? SVector{2, Float64}(p) : length(p) == 3 ? SVector{3, Float64}(p) :
    throw(ArgumentError("a target point is a 2-vector (a plane's) or a 3-vector; got $(length(p)) entries"))
_target_point(p::Uncertain) = (v = _target_point(geom(p)); Uncertain(v, CovMat(SMatrix{length(v), length(v), Float64}(Statistics.cov(p)))))

# ------------------------------------------------------------------------------
# PointTarget
# ------------------------------------------------------------------------------

"""
    PointTarget(points)

A target given by its points, explicitly, in its own frame: 2-vectors (a
plane's) or 3-vectors, all of one dimension, plain or `Uncertain` (an
estimated constellation, its covariances in mm²). A motion-capture body's
marker constellation, or a board whose warp was resolved to measured points.
"""
struct PointTarget{P} <: AbstractTarget
    points::Vector{P}
    function PointTarget(pts::AbstractVector)
        ps = map(_target_point, pts)
        isempty(ps) || allequal(length(geom(p)) for p in ps) ||
            throw(ArgumentError("a target's points are all 2-vectors or all 3-vectors"))
        P = isempty(ps) ? SVector{3, Float64} : typejoin(map(typeof, ps)...)
        return new{P}(Vector{P}(ps))
    end
end

points(t::PointTarget) = t.points
Base.:(==)(a::PointTarget, b::PointTarget) = a.points == b.points

# ------------------------------------------------------------------------------
# Warps
# ------------------------------------------------------------------------------

"""
    AbstractWarp

The deformation of a planar target's nominal design: a map of plane points
`w(p)`, with the warp protocol `warp_parameters`, `warp_parameter_names`,
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

"""
    warp_parameters(w) -> SVector

The warp's parameters, in [`warp_parameter_names`](@ref)' order; empty for
[`Rigid`](@ref).
"""
warp_parameters(::Rigid) = SVector{0, Float64}()

"""
    warp_parameter_names(w) -> NTuple{N, Symbol}

The names of [`warp_parameters`](@ref).
"""
warp_parameter_names(::Rigid) = ()
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
"""
struct PlanarTarget{W <: AbstractWarp, N <: Union{Nothing, Vector{SVector{2, Float64}}}} <: AbstractTarget
    design::Union{Nothing, Symbol}
    nominal::N
    warp::W
    function PlanarTarget(design::Union{Nothing, Symbol}, nominal, warp::W) where {W <: AbstractWarp}
        (design === nothing && nominal === nothing) &&
            throw(ArgumentError("a planar target has a design reference, nominal points, or both"))
        n = nominal === nothing ? nothing : SVector{2, Float64}[SVector{2, Float64}(geom(p)) for p in nominal]
        return new{W, typeof(n)}(design, n, warp)
    end
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
is_resolved(t::PlanarTarget) = t.nominal !== nothing

function points(t::PlanarTarget)
    t.nominal === nothing && throw(
        ArgumentError("points: the planar target $(repr(t.design)) is unresolved — its design's owner resolves the reference (`resolve`)")
    )
    return [t.warp(p) for p in t.nominal]
end
Base.:(==)(a::PlanarTarget, b::PlanarTarget) = a.design === b.design && isequal(a.nominal, b.nominal) && a.warp == b.warp

# --- the calibratable-target protocol ------------------------------------------------

"""
    target_parameters(t) -> SVector

A calibratable target's parameter vector (a `PlanarTarget`'s: its warp's).
"""
target_parameters(t::PlanarTarget) = warp_parameters(t.warp)

"""
    target_parameter_names(t) -> NTuple{N, Symbol}

The names of [`target_parameters`](@ref).
"""
target_parameter_names(t::PlanarTarget) = warp_parameter_names(t.warp)
with_parameters(t::PlanarTarget, θ::AbstractVector) = PlanarTarget(t.design, t.nominal, with_parameters(t.warp, θ))
parameters(t::PlanarTarget) = NamedTuple{target_parameter_names(t)}(Tuple(target_parameters(t)))
regularizer(t::PlanarTarget) = regularizer(t.warp)

"""
    TargetManifold(target) <: Manifold

A calibratable target's manifold anchored at `target`: its
[`target_parameters`](@ref)' coordinates as offsets from the anchor (linear);
`action_jacobian(ch, x)` is ∂points/∂θ at the nominal point `x`
(`warp_jacobian`). `minimal_manifold(target)` is this manifold. Zero-dimensional
for a `Rigid` target.
"""
struct TargetManifold{T <: AbstractTarget} <: Manifold
    target::T
end
minimal_manifold(t::PlanarTarget) = TargetManifold(t)
_ambient_coordinates(t::PlanarTarget) = target_parameters(t)
tangent_size(ch::TargetManifold) = length(target_parameters(ch.target))
model(ch::TargetManifold) = ch.target
retract(ch::TargetManifold, v) = with_parameters(ch.target, target_parameters(ch.target) + v)
recenter(ch::TargetManifold, v) = TargetManifold(retract(ch, v))
retract_jacobian(ch::TargetManifold) = (n = tangent_size(ch); SMatrix{n, n, Float64}(LinearAlgebra.I))
retract_jacobian(ch::TargetManifold, ::StaticVector) = retract_jacobian(ch)
action_jacobian(ch::TargetManifold{<:PlanarTarget}, x) = warp_jacobian(ch.target.warp, x)
local_coordinates(ch::TargetManifold, t::AbstractTarget) = target_parameters(t) - target_parameters(ch.target)
