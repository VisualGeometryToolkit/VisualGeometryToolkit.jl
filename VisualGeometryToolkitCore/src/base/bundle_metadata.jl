# =============================================================================
# BundleMetadata — the metadata of a bundle refinement (after scale_source.jl:
# it records the scale the bundle ran with, as every robust result does)
# =============================================================================

"""
    BundleMetadata(weights, residuals, fixed_frames, dof, loss_width, before, after; scale, scale_source)

Diagnostics of a bundle refinement: the final per-observation `weights` its
robust loss left and `residuals` (each observation's residual NORM, in the
bundle's order — what an inlier classification of the refined model reads; a
weight is the loss's guide to its optimizer, never inverted back into a
residual), the gauge-fixing `fixed_frames`, the degrees of freedom `dof`, the
robust loss's frozen `loss_width` (`NaN` under plain least squares — a LOSS
WIDTH, not a noise scale), and `before`/`after` residual summaries as concrete
`(; rms, median, p90)` NamedTuples — the FitResult's own `rms` mirrors
`after.rms` first-class.

The bundle's scale record (the robust-estimation contract), required: `scale`,
the noise scale it ran with (a LINEAR scale; `NaN` with `ScaleNone`, plain
least squares), and `scale_source` ([`ScaleSource`](@ref)), read by
[`noise_scale`](@ref) and [`scale_source`](@ref). Its termination record is the
FitResult's core ([`retcode`](@ref), [`stop_rule`](@ref)): build the result as
`FitResult(estimate, code, rule, iterations, rms, n, metadata)`.
"""
struct BundleMetadata{W <: AbstractVector{Float64}}
    weights::W
    residuals::Vector{Float64}
    fixed_frames::Vector{Int}
    dof::Int
    loss_width::Float64
    before::@NamedTuple{rms::Float64, median::Float64, p90::Float64}
    after::@NamedTuple{rms::Float64, median::Float64, p90::Float64}
    scale::Float64
    scale_source::ScaleSource
    function BundleMetadata(weights::W, residuals::AbstractVector{<:Real}, fixed_frames::AbstractVector{<:Integer},
                            dof::Integer, loss_width::Real, before, after;
                            scale::Real, scale_source::ScaleSource) where {W <: AbstractVector{Float64}}
        length(residuals) == length(weights) || throw(ArgumentError(
            "BundleMetadata: $(length(residuals)) residuals for $(length(weights)) weights — one each per observation"))
        return new{W}(weights, residuals, fixed_frames, dof, loss_width, before, after, Float64(scale), scale_source)
    end
end
