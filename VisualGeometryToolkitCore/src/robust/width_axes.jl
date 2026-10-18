# =============================================================================
# Width axes — which width quantity a tuned or deployed value is
# =============================================================================
#
# The three width quantities of the width vocabulary (conventions/naming.md, rule 10), as
# dispatch tags: a tuned value, a grid axis or a persisted kind names its
# quantity by one of these types, never by a bare symbol. The map from a scale
# multiple to a method's NATIVE width is `to_native`, one method per native
# axis: `LossWidth` through the truncated loss `TLSShape`, `OutlierHalfwidth`
# through the Gaussian-uniform mixture `GauInlier`; the two native axes convert
# at an operating scale multiple (`outlier_halfwidth_to_loss_width`).
#
# Uses: TLSShape, decision_width (robust/losses.jl); GauInlier (robust/mixture.jl,
# read at call time).
# =============================================================================

"""
    WidthAxis

The width quantity a value is expressed in — the axis it is tuned on and
persisted under:

- [`ScaleMultiple`](@ref): a width as a multiple of the estimated scale ŝ,
  dimensionless (`width = scale_multiple · ŝ`), partition-anchored;
- [`LossWidth`](@ref): a robust loss's scale parameter (`loss_width`), the
  residual scale where the loss leaves its quadratic regime — a threshold
  score's threshold is its truncated loss's width;
- [`OutlierHalfwidth`](@ref): the Student-t marginal's uniform outlier
  half-width.

[`width_axis`](@ref) reads one from its persisted name and
[`width_axis_name`](@ref) writes it; [`to_native`](@ref) maps a scale multiple
onto a native axis.
"""
abstract type WidthAxis end

"""
    ScaleMultiple()
    ScaleMultiple(m::Real)

A width as a multiple of the estimated scale (see [`WidthAxis`](@ref)). Bare,
it is the AXIS (`ScaleMultiple() === ScaleMultiple(nothing)`); with a value it
is a local optimizer's width MODE: the loss width `m·ŝ`, `ŝ` the scorer's
consensus scale (in the loss's own currency, see [`deployed_scale_multiple`](@ref)).
"""
struct ScaleMultiple{T} <: WidthAxis
    value::T
    ScaleMultiple{T}(value) where {T} = new{T}(value)
end
ScaleMultiple() = ScaleMultiple{Nothing}(nothing)
ScaleMultiple(::Nothing) = ScaleMultiple()
ScaleMultiple(m::Real) = ScaleMultiple{Float64}(Float64(m))

"""
    LossWidth()
    LossWidth(w::Real)
    LossWidth(scale_multiple::Real, s2::Real)

A robust loss's scale parameter, its `loss_width` (see [`WidthAxis`](@ref)).
Bare, it is the AXIS; with a value it is a local optimizer's ABSOLUTE width
mode, stored FACTORED as `(scale_multiple, s2)` so that the loss deploys at
exactly `deploy(shape, s2; scale_multiple)` — `LossWidth(w)` is `(1.0, w²)`
(`sq_loss_width = 1.0²·w² === w²`), and `LossWidth(m, s2)` keeps the two
factors apart for a reader of `s2` alone (a GNC entry, Barron's α fit).
"""
struct LossWidth{T} <: WidthAxis
    value::T
    LossWidth{T}(value) where {T} = new{T}(value)
end
LossWidth() = LossWidth{Nothing}(nothing)
LossWidth(::Nothing) = LossWidth()
LossWidth(w::Real) = LossWidth{NTuple{2, Float64}}((1.0, Float64(w)^2))
LossWidth(scale_multiple::Real, s2::Real) = LossWidth{NTuple{2, Float64}}((Float64(scale_multiple), Float64(s2)))

"""
    OutlierHalfwidth()
    OutlierHalfwidth(a::Real)

The Student-t marginal's uniform outlier half-width (see [`WidthAxis`](@ref)).
Bare, it is the AXIS; with a value it is the width a scoring is built at
(`with_width(scoring, OutlierHalfwidth(a))`). Never a local optimizer's width.
"""
struct OutlierHalfwidth{T} <: WidthAxis
    value::T
    OutlierHalfwidth{T}(value) where {T} = new{T}(value)
end
OutlierHalfwidth() = OutlierHalfwidth{Nothing}(nothing)
OutlierHalfwidth(::Nothing) = OutlierHalfwidth()
OutlierHalfwidth(a::Real) = OutlierHalfwidth{Float64}(Float64(a))

"The value a width carries (`nothing` for the bare axis)."
width_value(w::WidthAxis) = w.value

"""
    loss_width(w::LossWidth) -> Float64

The absolute loss width a [`LossWidth`](@ref) value carries,
`scale_multiple·√s2` of its factored `(scale_multiple, s2)` — `w` itself for
`LossWidth(w)` (`1.0·√(w²) == w` in IEEE arithmetic).
"""
loss_width(w::LossWidth{NTuple{2, Float64}}) = w.value[1] * sqrt(w.value[2])

const _WIDTH_AXES = (
    :scale_multiple => ScaleMultiple(),
    :loss_width => LossWidth(),
    :outlier_halfwidth => OutlierHalfwidth(),
)

"""
    width_axis(name::Union{AbstractString,Symbol}) -> WidthAxis

The width axis a persisted name denotes: `"scale_multiple"` →
[`ScaleMultiple`](@ref), `"loss_width"` → [`LossWidth`](@ref),
`"outlier_halfwidth"` → [`OutlierHalfwidth`](@ref) (a string, as TOML and CLI
values read, or a symbol, as grid keys are). Any other name is an
`ArgumentError` — an abbreviation is never read as a width. The inverse of
[`width_axis_name`](@ref).
"""
function width_axis(name::Symbol)
    for (n, axis) in _WIDTH_AXES
        n === name && return axis
    end
    throw(ArgumentError(
        "width_axis: unknown width axis $(repr(String(name))) (expected " *
            join((repr(String(n)) for (n, _) in _WIDTH_AXES), ", ", " or ") * ")"
    ))
end
width_axis(name::AbstractString) = width_axis(Symbol(name))

"""
    width_axis_name(axis::WidthAxis) -> Symbol

The persisted name of a width axis — the key of its grid, its sidecar kind, its
TOML and CLI value: `:scale_multiple`, `:loss_width`, `:outlier_halfwidth`
(`String` of it where a string is written), whatever value the width carries.
The inverse of [`width_axis`](@ref) on a bare axis:
`width_axis(width_axis_name(a)) === a`.
"""
width_axis_name(::ScaleMultiple) = :scale_multiple
width_axis_name(::LossWidth) = :loss_width
width_axis_name(::OutlierHalfwidth) = :outlier_halfwidth

"""
    to_native(axis::WidthAxis, scale_multiple, scale; d_g) -> Float64

A scale multiple as a method's NATIVE width at the estimated `scale`: the width
on `axis` whose inlier/outlier decision flips at the residual
`scale_multiple · scale` — one conversion, [`decision_width`](@ref), of the
axis's decision model. Each native axis's method lives with its model:

- [`LossWidth`](@ref) (here): the truncated loss `TLSShape()`, which flips at
  its width, so the native width is `scale_multiple · scale` itself (MSAC,
  RANSAC, GaU, MAGSAC++, PoseLib's truncation);
- [`OutlierHalfwidth`](@ref): through `GauInlier` (the Student-t marginal's
  crossover).

`d_g` is the residual codimension of the problem (1 for E and F, 2 for H). It
has NO default: a native grid and the window cut inside it must be cut at the
same `d_g`, so every caller states it. The loss-width axis does not depend on
it; the outlier half-width does.
"""
function to_native end

to_native(::LossWidth, scale_multiple::Real, scale::Real; d_g::Integer) =
    decision_width(TLSShape(), Float64(scale_multiple) * Float64(scale))

"""
    to_native(::OutlierHalfwidth, scale_multiple, scale; d_g) -> Float64

A scale multiple as the Student-t marginal's native width: the outlier
half-width whose MAP inlier/outlier decision flips at the residual
`scale_multiple · scale` — [`decision_width`](@ref) of the Gaussian-uniform
mixture [`GauInlier`](@ref)`(scale²; d_g)`, the stack-wide definition of the
crossover (moved from RansacScoringDoneRight). `scale` is the estimated inlier
scale (linear, in the units of the residuals), and the result is in those units.

`d_g` is the residual codimension (1 for E and F, 2 for H) and has no default:
it is the EXPONENT of the outlier density, so it moves the outlier half-width
by decades (at `scale = 1`, a scale multiple of 6 maps to 8.2293e7 at
`d_g = 1` and to 1.0156e4 at `d_g = 2`).
"""
function to_native(::OutlierHalfwidth, scale_multiple::Real, scale::Real; d_g::Integer)
    s = Float64(scale)
    return decision_width(GauInlier(s^2; d_g), Float64(scale_multiple) * s)
end

"""
    native_map(axis::Union{LossWidth, OutlierHalfwidth}; d_g) -> (scale_multiple, scale) -> Float64

Curried [`to_native`](@ref) on `axis` at codimension `d_g` — THE per-axis map
closure every window and coverage grid is cut with. `d_g` is required for both
axes (the loss-width axis ignores it), so a grid and the window cut inside it
are always cut at the same codimension.
"""
native_map(axis::Union{LossWidth, OutlierHalfwidth}; d_g::Integer) =
    (scale_multiple, scale) -> to_native(axis, scale_multiple, scale; d_g)

# The half-width at σ = 1 whose boundary sits at `scale_multiple`: the
# crossover's GAIN per unit σ. At a fixed scale multiple the half-width is
# linear in σ, so this one scalar is the entire content of both conversions
# below, which are therefore inverses by construction. It depends on the
# CODIMENSION: the outlier density is (2h)^-d_g and the inlier Gaussian's
# normaliser (2πσ²)^-d_g/2 (at scale multiple 2.5 the gain is 28.53 for d_g = 1
# and 5.979 for d_g = 2).
_unit_halfwidth(scale_multiple::Real, d_g::Integer) =
    decision_width(GauInlier(1.0; d_g), Float64(scale_multiple))

"""
    outlier_halfwidth_to_loss_width(outlier_halfwidth; operating_scale_multiple, d_g) -> Float64

The truncation loss width at the SAME effective decision boundary as the
Student-t marginal's `outlier_halfwidth`, holding the operating scale multiple
fixed: `ŝ_implied = outlier_halfwidth / _unit_halfwidth(operating_scale_multiple, d_g)`,
then `loss_width = operating_scale_multiple·ŝ_implied`. `d_g` is required (the
crossover moves with it). `operating_scale_multiple` is caller policy — a
PROTOCOL choice (compare methods, not parameterisations).
"""
outlier_halfwidth_to_loss_width(outlier_halfwidth::Real; operating_scale_multiple::Real, d_g::Integer) =
    Float64(operating_scale_multiple) * Float64(outlier_halfwidth) /
    _unit_halfwidth(operating_scale_multiple, d_g)

"""
    loss_width_to_outlier_halfwidth(loss_width; operating_scale_multiple, d_g) -> Float64

Inverse of [`outlier_halfwidth_to_loss_width`](@ref): the marginal's outlier
half-width whose decision boundary matches a truncation at `loss_width`, at the
operating scale multiple. The literal reciprocal around the same
`_unit_halfwidth`, so the two are inverses by construction. `d_g` is required.
"""
loss_width_to_outlier_halfwidth(loss_width::Real; operating_scale_multiple::Real, d_g::Integer) =
    _unit_halfwidth(operating_scale_multiple, d_g) * Float64(loss_width) /
    Float64(operating_scale_multiple)
