# =============================================================================
# The robust-estimation contract's scale hand-off (owner rulings Q1, Q2)
# =============================================================================
#
# Every robust estimator's result records ONE scale — the scale it ran with, or
# the one it estimated — and where it came from. A stage that is handed a scale
# (`scale = σ` or `scale = <an upstream result>`) records it unchanged; only a
# scale-estimating stage (a RANSAC scorer's consensus partition, an EM fit's
# E-step) produces a new one; a stage with no robust loss and no scale records
# none. What a stage was given is the previous stage's record, so there is no
# separate input field.
#
# A metadata type carries two concretely typed fields, `scale::T` and
# `scale_source::ScaleSource`; `noise_scale` and `scale_source` read them
# through the FitResult. `ScaleNone`'s scale is NaN — never `nothing` — and
# `noise_scale` refuses it, so a NaN never reaches the next stage silently.
# =============================================================================

"""
    ScaleSource

Where a robust result's `scale` came from — one of:

- `ScaleGiven`: a number the caller gave (`scale = σ`);
- `ScaleUpstream`: an earlier stage's result (`scale = result`, its [`noise_scale`](@ref));
- `ScaleEstimated`: estimated by this stage — a RANSAC scorer's consensus
  partition, an EM fit's E-step;
- `ScaleNone`: no scale (a plain least-squares stage given none); the scale is `NaN`.

An `@enum` (isbits): a metadata that carries it stays concretely typed.
"""
@enum ScaleSource::UInt8 ScaleGiven ScaleUpstream ScaleEstimated ScaleNone

@doc "`ScaleGiven`: a [`ScaleSource`](@ref) — the scale is a number the caller gave (`scale = σ`)." ScaleGiven
@doc "`ScaleUpstream`: a [`ScaleSource`](@ref) — the scale is an earlier stage's [`noise_scale`](@ref) (`scale = result`)." ScaleUpstream
@doc "`ScaleEstimated`: a [`ScaleSource`](@ref) — this stage estimated the scale (a scorer's consensus partition, an EM fit's E-step)." ScaleEstimated
@doc "`ScaleNone`: a [`ScaleSource`](@ref) — no scale was recorded; the scale is `NaN` and [`noise_scale`](@ref) refuses it." ScaleNone

"""
    ScaleInput

The scale input of one robust stage, parsed once by [`scale_input`](@ref):
`scale` (the noise scale σ, `NaN` when a width was given) or `width` (the loss
width, `NaN` when a scale was given) — exactly one is finite, in the caller's
float type — and the
[`ScaleSource`](@ref) the stage records.
"""
struct ScaleInput{T <: AbstractFloat}
    scale::T
    width::T
    source::ScaleSource
end

"""
    scale_input(scale, width) -> ScaleInput

The scale input of a robust estimator, from its two keywords — EXACTLY ONE of:

- `scale = σ`: a positive, finite noise scale (`ScaleGiven`);
- `scale = result`: an earlier robust result, read by [`noise_scale`](@ref)
  (`ScaleUpstream`; a result that recorded no scale is refused);
- `width = w`: a positive, finite loss width (`ScaleGiven`; the scale is `NaN`).

Neither, or both, is an `ArgumentError` naming both keywords. A stage with no
robust loss (plain least squares) takes neither, and refuses one it would
ignore — that check is the stage's, which knows its loss.
"""
scale_input(::Nothing, ::Nothing) = throw(ArgumentError(
    "a robust loss needs its scale: give `scale = σ` (or an upstream result) or `width = w`"))
scale_input(scale, width) = throw(ArgumentError(
    "give `scale` or `width`, not both (got scale = $(repr(scale)), width = $(repr(width)))"))
scale_input(scale::Real, ::Nothing) = (v = _positive_scale(scale, "scale"); ScaleInput(v, oftype(v, NaN), ScaleGiven))
scale_input(::Nothing, width::Real) = (v = _positive_scale(width, "width"); ScaleInput(oftype(v, NaN), v, ScaleGiven))
scale_input(r::FitResult, ::Nothing) = (v = noise_scale(r); ScaleInput(v, oftype(v, NaN), ScaleUpstream))

function _positive_scale(x::Real, name::AbstractString)
    v = float(x)
    (isfinite(v) && v > 0) || throw(ArgumentError("`$name` must be positive and finite, got $x"))
    return v
end

"""
    noise_scale(result) -> T

The scale a robust result recorded — the one it ran with, or the one it
estimated (its metadata's `scale`; [`scale_source`](@ref) says which). The
hand-off: a downstream stage takes it as `scale = result`. An `ArgumentError`
for a result whose source is `ScaleNone` (a plain least-squares stage given
no scale), so its `NaN` is never handed on.

A metadata type answers through its fields `scale` and `scale_source`; one that
records the scale otherwise adds a method on its metadata type.
"""
noise_scale(r::FitResult) = noise_scale(r.metadata)
noise_scale(::Nothing) = throw(ArgumentError(
    "a result without metadata records no scale; give the next stage its own `scale` or `width`"))
function noise_scale(m)
    scale_source(m) == ScaleNone && throw(ArgumentError(
        "this $(nameof(typeof(m))) result recorded no scale (ScaleNone: a plain least-squares stage given none); " *
            "give the next stage its own `scale` or `width`"))
    return m.scale
end

"""
    scale_source(result) -> ScaleSource

Where a robust result's [`noise_scale`](@ref) came from ([`ScaleSource`](@ref)).
"""
scale_source(r::FitResult) = scale_source(r.metadata)
scale_source(m) = m.scale_source
scale_source(::Nothing) = ScaleNone
