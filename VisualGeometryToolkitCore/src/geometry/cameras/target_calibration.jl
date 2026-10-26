# ==============================================================================
# Target calibration — what a target calibration estimates about a planar target
# ==============================================================================
#
# A `PlanarTarget` (targets.jl) is a DESIGN — its nominal plane points — and a
# CALIBRATION taking the design to the physical target, the way a camera model
# is its intrinsics and its lens. A calibration is a map of design points
# `c(p)` carrying its parameters, with the protocol every kind implements (the
# target analogue of the lens chart, lens_chart.jl):
#
#   parameter_vector(c)            θ, an SVector (empty for Rigid)
#   parameter_names(c)             θ's names, so a configuration frees or fixes them
#   parameter_bounds(c)            unbounded (lens_chart.jl's default)
#   with_parameters(c, θ)          the same kind rebuilt from θ
#   calibration_jacobian(c, x)     ∂c(x)/∂θ at a design point x
#   regularizer(c)                 the prior's residual rows r(θ), ‖r‖² the penalty
#
# LENGTHS. A calibration's lengths are in the form of the design points it
# moves — plain numbers, or lengths stored as plain METRES with `has_units(c)`
# true — and `ustrip(u, c)` / `with_unit(c, u)` move between the forms as they
# do for points. Applied to a physical point it strips the point in metres,
# computes in plain numbers, and attaches metres again; its parameters and
# Jacobians are plain numbers, the lengths among them in metres.

"""
    TargetCalibration

What a target calibration estimates about a planar target: a map `c(p)` of
its design points to the physical target, with the parameter protocol
[`parameter_vector`](@ref), [`parameter_names`](@ref),
[`parameter_bounds`](@ref), [`with_parameters`](@ref)`(c, θ)`,
[`calibration_jacobian`](@ref)`(c, x)` and [`regularizer`](@ref)`(c)`.
[`Rigid`](@ref) is the design taken exactly. A downstream kind subtypes this
and implements the protocol.
"""
abstract type TargetCalibration end

"""
    calibration_jacobian(c::TargetCalibration, x) -> SMatrix

`∂c(x)/∂θ` at the design point `x`, one column per entry of
[`parameter_vector`](@ref)`(c)`. Plain numbers; for a physical calibration
the point and the length parameters are in metres.
"""
function calibration_jacobian end

"""
    regularizer(c | target) -> SVector

The prior's residual rows `r(θ)` at the current parameters: `‖r‖²` is the
penalty a calibrating estimator adds (a thin-plate spline's bending energy, a
shrinkage toward the design). Empty for [`Rigid`](@ref).
"""
function regularizer end

"""
    Rigid() <: TargetCalibration

The design, exactly: the identity, no parameters, no regularizer, no lengths
(so it sits with plain and physical designs alike).
"""
struct Rigid <: TargetCalibration end
(::Rigid)(p) = p

parameter_vector(::Rigid) = SVector{0, Float64}()
parameter_names(::Rigid) = ()
with_parameters(c::Rigid, θ::AbstractVector) =
    (isempty(θ) || throw(ArgumentError("Rigid takes no parameters; got $(length(θ))")); c)
calibration_jacobian(::Rigid, x) = zero(SMatrix{2, 0, Float64})
regularizer(::Rigid) = SVector{0, Float64}()

# --- lengths ---------------------------------------------------------------------
#
# `_has_lengths(c)`: whether a kind carries lengths at all (Rigid does not);
# `has_units(c)`: whether they are physical. `_length_sample(c)`: one of them in
# its own form, to name the mismatch in a `DimensionError`. `_length_mask(c)`:
# which entries of θ are lengths (they scale with a unit change; the rest are
# dimensionless).
_has_lengths(::Rigid) = false
has_units(::Rigid) = false
_length_mask(::Rigid) = SVector{0, Bool}()
FlexUnits.ustrip(::Units, c::Rigid) = c
with_unit(c::Rigid, ::Units) = c

# A calibration's form moves with its unit: an uncertain one's covariance
# scales with it, each length entry by `f` (`ustrip`: `f` = units of `u` per
# metre) — Σᵢⱼ by f^(lᵢ + lⱼ), a precision the opposite way.
_ustrip_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? ustrip(u, c) : c
_attach_calibration(u::Units, c::TargetCalibration) = _has_lengths(c) ? with_unit(c, u) : c
function _ustrip_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(ustrip(u, g), _scale_length_form(getfield(c, :attributes), _length_mask(g), _metre_factor(u), *))
end
function _attach_calibration(u::Units, c::Uncertain{<:TargetCalibration})
    g = geom(c)
    _has_lengths(g) || return c
    return Uncertain(with_unit(g, u), _scale_length_form(getfield(c, :attributes), _length_mask(g), _metre_factor(u), /))
end
function _scale_length_form(M::CovMat{N, T, L, C}, mask, f, op) where {N, T, L, C}
    return CovMat{N, T, L, C}(ntuple(k -> _length_entry(M[k], k, N, mask, f, op), Val(L)))
end
function _scale_length_form(M::PrecMat{N, T, L, C}, mask, f, op) where {N, T, L, C}
    return PrecMat{N, T, L, C}(ntuple(k -> _length_entry(M[k], k, N, mask, f, _opposite(op)), Val(L)))
end
@inline function _length_entry(x, k, N, mask, f, op)
    n = mask[mod1(k, N)] + mask[(k - 1) ÷ N + 1]
    return n == 0 ? x : op(x, n == 1 ? f : f * f)
end

# A point in plain numbers (a physical one in metres), and a plain result put
# back in the form of the point it came from.
@inline _plain_point(p::StaticVector{N, <:Quantity}) where {N} = SVector{N}(map(q -> _strip_length(m, q), Tuple(p)))
@inline _plain_point(p::StaticVector) = SVector(Tuple(p))
@inline _restore_point(x::StaticVector, ::StaticVector{N, <:Quantity}) where {N} = SVector(map(v -> _attach_length(v, m), Tuple(x)))
@inline _restore_point(x::StaticVector, ::StaticVector) = x
