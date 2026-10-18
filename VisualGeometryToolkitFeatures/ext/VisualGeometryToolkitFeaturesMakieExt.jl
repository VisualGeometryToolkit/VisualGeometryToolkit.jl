module VisualGeometryToolkitFeaturesMakieExt
    import VisualGeometryToolkitCore as VgtkCore

# =============================================================================
# Makie extension: correspondence visualization
# =============================================================================
#
# Extends VGC's canonical `matches!`/`matches` for Cspond-typed inputs and
# existing Makie infrastructure (`imshow!`, `poly!`).
#
# Both functions dispatch on cspond endpoint type:
#   - plain `Pair{<:AbstractVector, <:AbstractVector}` — color-coded markers.
#   - stochastic `Pair{<:Uncertain{<:Any,<:PrecMat{2}}, …}` — markers plus
#     `sigma`-σ confidence ellipses (one per endpoint), color-coded so the
#     source ellipse, target ellipse, and matching markers share a hue —
#     useful for sanity-checking the matcher's confidence calibration.
#
# The `sigma`-σ ellipse comes from VGC's
# `Ellipse(::Uncertain{<:AbstractVector,<:PrecMat{2}}; sigma)` constructor,
# which delegates to `Ellipse(center, Σ::Symmetric)` with `Σ = inv(Λ)`.

using VisualGeometryToolkitCore: VisualGeometryToolkitCore, Uncertain, PrecMat, CovMat,
    Point2, Ellipse, Cspond, sources, targets, feature_frame
import VisualGeometryToolkitFeatures: detections!
using VisualGeometryToolkitCore: AbstractDetection   # a detection of a frame or of an uncertain frame
import VisualGeometryToolkitCore: matches, matches!, ellipses!

using Makie: Makie, Figure, Axis, cgrad,
    hidedecorations!, hidespines!, DataAspect

# Endpoint type alias for dispatch readability. A stochastic endpoint is VGC's
# `Uncertain` point (a 2×2 PrecMat or CovMat); VGC defines an
# `Ellipse(d; sigma)` for either.
const _PlainEndpoint = AbstractVector

# -----------------------------------------------------------------------------
# matches! (two-axis Cspond forms) — color-coded endpoint primitives
# -----------------------------------------------------------------------------

#     matches!(ax_A, ax_B, cs::AbstractVector{<:Cspond{<:AbstractVector,<:AbstractVector}};
#             colormap=:hsv, markersize=4.0, marker=:cross)
#
# Plain-point overload: scatter color-coded endpoints on each axis.  `cs` may
# be a `StructArray{Cspond{S,T}}` (built via `csponds(src, tgt)`) — its
# `first`/`second` columns are accessed directly.
function matches!(
        ax_A, ax_B,
        cs::AbstractVector{<:Cspond{<:_PlainEndpoint, <:_PlainEndpoint}};
        colormap = :hsv,
        markersize::Real = 4.0,
        marker = :cross
    )
    n = length(cs)
    cmap = cgrad(colormap, n, categorical = true)
    cols = [cmap[i] for i in 1:n]
    src = [Point2{Float64}(p[1], p[2]) for p in sources(cs)]
    tgt = [Point2{Float64}(p[1], p[2]) for p in targets(cs)]
    Makie.scatter!(ax_A, src; color = cols, markersize = markersize, marker = marker)
    Makie.scatter!(ax_B, tgt; color = cols, markersize = markersize, marker = marker)
    return nothing
end

#     matches!(ax_A, ax_B, cs::AbstractVector{<:Cspond{<:StochasticEndpoint, ...}};
#             sigma=3.0, colormap=:hsv, max_display=500,
#             ellipse_strokewidth=1.0, markersize=3.0, alpha=0.6)
#
# Stochastic-detection overload: in addition to color-coded center markers,
# draws a `sigma`-σ confidence ellipse per endpoint via VGC's `poly!` and
# `Ellipse(d; sigma)`.  Endpoint attributes may be `PrecMat{2}` (precision)
# or `CovMat{2}` (covariance) — VGC's `Ellipse` handles both.
#
# `max_display` caps the number of csponds drawn (subsamples evenly).
function matches!(
        ax_A, ax_B,
        cs::AbstractVector{
            <:Cspond{
                <:Uncertain{<:AbstractVector, <:Union{PrecMat{2}, CovMat{2}}},
                <:Uncertain{<:AbstractVector, <:Union{PrecMat{2}, CovMat{2}}},
            },
        };
        sigma::Real = 3.0,
        colormap = :hsv,
        max_display::Int = 500,
        ellipse_strokewidth::Real = 1.0,
        markersize::Real = 3.0,
        alpha::Real = 0.6
    )
    max_display >= 1 || throw(ArgumentError("max_display must be ≥ 1, got $max_display"))
    n = length(cs)
    stride = max(1, cld(n, max_display))   # ⌈n/max_display⌉: at most max_display drawn
    idxs = 1:stride:n
    m = length(idxs)

    cmap = cgrad(colormap, m, categorical = true)
    cols_solid = [cmap[i] for i in 1:m]
    cols_alpha = [(cmap[i], Float32(alpha)) for i in 1:m]

    sub_src = view(sources(cs), idxs)
    sub_tgt = view(targets(cs), idxs)
    src_e = [Ellipse(d; sigma = sigma) for d in sub_src]
    tgt_e = [Ellipse(d; sigma = sigma) for d in sub_tgt]
    for k in 1:m
        VgtkCore.poly!(
            ax_A, src_e[k];
            color = :transparent,
            strokecolor = cols_alpha[k],
            strokewidth = ellipse_strokewidth
        )
        VgtkCore.poly!(
            ax_B, tgt_e[k];
            color = :transparent,
            strokecolor = cols_alpha[k],
            strokewidth = ellipse_strokewidth
        )
    end

    src_pts = [Point2{Float64}(d.value[1], d.value[2]) for d in sub_src]
    tgt_pts = [Point2{Float64}(d.value[1], d.value[2]) for d in sub_tgt]
    Makie.scatter!(
        ax_A, src_pts;
        color = cols_solid, markersize = markersize, marker = :cross
    )
    Makie.scatter!(
        ax_B, tgt_pts;
        color = cols_solid, markersize = markersize, marker = :cross
    )
    return nothing
end

# -----------------------------------------------------------------------------
# matches — figure-level convenience over two side-by-side image axes
# -----------------------------------------------------------------------------

#     matches(img_A, img_B, cs; axis=(;), figure=(;), kwargs...)
#         -> (Figure, (axis_A, axis_B))
#
# Build a Figure with two axes side-by-side, `imshow!` each image, then call
# `matches!` (which dispatches on cspond endpoint type).  Trailing kwargs are
# forwarded to `matches!`.
function matches(
        img_A, img_B,
        cs::AbstractVector{<:Pair};
        axis = (;),
        figure = (;),
        kwargs...
    )
    fig = Figure(; figure...)
    ax_A = Axis(fig[1, 1]; aspect = DataAspect(), axis...)
    ax_B = Axis(fig[1, 2]; aspect = DataAspect(), axis...)
    VgtkCore.imshow!(ax_A, img_A)
    VgtkCore.imshow!(ax_B, img_B)
    hidedecorations!(ax_A); hidespines!(ax_A)
    hidedecorations!(ax_B); hidespines!(ax_B)

    matches!(ax_A, ax_B, cs; kwargs...)
    return fig, (ax_A, ax_B)
end


# =============================================================================
# detections! — Detection-metadata-aware overlay (composes VGC ellipses!)
# =============================================================================

function detections!(
        ax, dets::AbstractVector{<:AbstractDetection};
        cutoff::Real = 1.0,
        color_by = nothing,
        colormap = :viridis,
        strokecolor = :cyan,
        kwargs...
    )
    isempty(dets) && return nothing
    shapes = [feature_frame(d) for d in dets]
    color = color_by === nothing ? nothing :
        color_by === :response ? Float64[abs(d.metadata.response) for d in dets] :
        color_by isa AbstractVector ? color_by :
        throw(ArgumentError("color_by must be nothing, :response, or a value vector"))
    return ellipses!(ax, shapes; cutoff, color, colormap, strokecolor, kwargs...)
end

end # module VisualGeometryToolkitFeaturesMakieExt
