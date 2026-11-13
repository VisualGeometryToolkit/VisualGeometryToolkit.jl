module VisualGeometryToolkitFeaturesMakieExt
    import VisualGeometryToolkitCore as VgtkCore

# =============================================================================
# Makie extension: correspondence visualization
# =============================================================================
#
# Extends VGC's canonical `matches!`/`matches` for Correspondence-typed inputs and
# existing Makie infrastructure (`imshow!`, `poly!`).
#
# Both functions dispatch on correspondence endpoint type:
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
    Point2, Ellipse, Correspondence, sources, targets, feature_frame, Attributed, weight_of, value_of
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
# matches! (two-axis Correspondence forms) — color-coded endpoint primitives
# -----------------------------------------------------------------------------

#     matches!(ax_A, ax_B, cs::AbstractVector{<:Correspondence{<:AbstractVector,<:AbstractVector}};
#             colormap=:hsv, markersize=4.0, marker=:cross)
#
# Plain-point overload: scatter color-coded endpoints on each axis.  `cs` may
# be a `StructArray{Correspondence{S,T}}` (built via `correspondences(src, tgt)`) — its
# `first`/`second` columns are accessed directly. A correspondence may carry its
# weight, `Attributed(src => tgt, (; weight))`: the weight (`weight_of`, in
# [0, 1]) is its markers' alpha; a bare correspondence is drawn opaque.
const _PlainCorrespondence = Correspondence{<:_PlainEndpoint, <:_PlainEndpoint}
function matches!(
        ax_A, ax_B,
        cs::AbstractVector{<:Union{_PlainCorrespondence, Attributed{<:_PlainCorrespondence}}};
        colormap = :hsv,
        markersize::Real = 4.0,
        marker = :cross
    )
    n = length(cs)
    cmap = cgrad(colormap, n, categorical = true)
    cols = [(cmap[i], Float32(weight_of(cs[i]))) for i in 1:n]
    src = [Point2{Float64}(first(value_of(c))[1], first(value_of(c))[2]) for c in cs]
    tgt = [Point2{Float64}(last(value_of(c))[1], last(value_of(c))[2]) for c in cs]
    Makie.scatter!(ax_A, src; color = cols, markersize = markersize, marker = marker)
    Makie.scatter!(ax_B, tgt; color = cols, markersize = markersize, marker = marker)
    return nothing
end

#     matches!(ax_A, ax_B, cs::AbstractVector{<:Correspondence{<:StochasticEndpoint, ...}};
#             sigma=3.0, colormap=:hsv, max_display=500,
#             ellipse_strokewidth=1.0, markersize=3.0, alpha=0.6)
#
# Stochastic-detection overload: in addition to color-coded center markers,
# draws a `sigma`-σ confidence ellipse per endpoint via VGC's `poly!` and
# `Ellipse(d; sigma)`.  Endpoint attributes may be `PrecMat{2}` (precision)
# or `CovMat{2}` (covariance) — VGC's `Ellipse` handles both.
#
# `max_display` caps the number of correspondences drawn (subsamples evenly).
function matches!(
        ax_A, ax_B,
        cs::AbstractVector{
            <:Correspondence{
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
# `matches!` (which dispatches on correspondence endpoint type).  Trailing kwargs are
# forwarded to `matches!`.
function matches(
        img_A, img_B,
        cs::AbstractVector{<:Union{Pair, Attributed{<:Pair}}};
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

# =============================================================================
# compare_images — the stack's one image viewer: linked axes over a menu of entries
# =============================================================================

import VisualGeometryToolkitFeatures: compare_images
using Makie: Menu, Observable, on, linkaxes!, reset_limits!

# An entry's images: the tuple itself, or what its loader returns (called when the entry is chosen).
_entry_images(x::Tuple) = x
function _entry_images(f)
    x = f()
    x isa Tuple || throw(ArgumentError("compare_images: an entry's loader returns a tuple of images, got a $(typeof(x))"))
    return x
end

function compare_images(entries::AbstractVector{<:Pair}; titles = nothing, default = nothing, size = (1800, 1000))
    isempty(entries) && throw(ArgumentError("compare_images: no entries"))
    names = [String(first(e)) for e in entries]
    shown = something(default, first(names))
    shown in names || throw(ArgumentError("compare_images: the default $(repr(shown)) is not an entry ($(join(names, ", ")))"))
    lookup = Dict(zip(names, last.(entries)))
    images = _entry_images(lookup[shown])
    n = length(images)
    labels = titles === nothing ? ntuple(_ -> "", n) : Tuple(titles)
    length(labels) == n || throw(ArgumentError("compare_images: $(length(labels)) titles for $n images an entry"))
    fig = Figure(; size)
    menu = Menu(fig[1, 1]; options = names, default = shown, tellwidth = false)
    shown_images = [Observable{AbstractMatrix}(images[i]) for i in 1:n]
    axes = [Axis(fig[i + 1, 1]; title = String(labels[i])) for i in 1:n]
    for (ax, img) in zip(axes, shown_images)
        hidedecorations!(ax)
        VgtkCore.imshow!(ax, img)                 # Core's orientation rule
    end
    n > 1 && linkaxes!(axes)                      # Makie's vector method: the image count is data
    on(menu.selection) do name
        next = _entry_images(lookup[name])
        length(next) == n || throw(ArgumentError("compare_images: entry $(repr(name)) has $(length(next)) images, the others $n"))
        foreach(i -> shown_images[i][] = next[i], 1:n)
        reset_limits!(first(axes))
    end
    return fig
end

end # module VisualGeometryToolkitFeaturesMakieExt
