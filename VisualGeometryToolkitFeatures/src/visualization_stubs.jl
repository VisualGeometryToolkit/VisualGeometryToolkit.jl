# =============================================================================
# Visualization stubs — implemented in VisualGeometryToolkitFeaturesMakieExt
# =============================================================================
#
# The canonical correspondence primitives are VGC's `matches!`/`matches`;
# this package EXTENDS them by dispatch for feature-typed inputs (`Correspondence`
# vectors: two-axis color-coded endpoints, image-pair figures) rather than
# exporting competing names: `matches!(ax_A, ax_B, cs)` and `matches(img_A, img_B, cs)`.
#
# VGF-owned inspection sugar:
#   detections!  — Detection-metadata-aware overlay over VGC's `ellipses!`

"""
    detections!(ax, dets::AbstractVector{<:Detection};
                cutoff=1.0, color_by=nothing, colormap=:viridis, kwargs...)

Overlay detections as batched ellipse boundaries (VisualGeometryToolkitCore's `ellipses!`).

`color_by` selects the stroke coloring:
- `nothing` — constant `strokecolor`
- `:response` — `|metadata.response|` through `colormap`
- an `AbstractVector{<:Real}` — explicit per-detection values

Requires Makie.
"""
function detections! end

"""
    compare_images(entries; titles = nothing, default = first name, size = (1800, 1000)) -> Figure

The stack's one image viewer: a menu over named entries, each a tuple of
images to compare (a pair, typically) or a function returning one (called
when the entry is chosen, so a large set loads lazily), drawn on one axis per
image with the axes linked (they pan and zoom together, pixel for pixel).
Every image is drawn by VisualGeometryToolkitCore's orientation rule
(`imshow!`: x right, y down, COLMAP pixels, row 1 at the top), whatever its
element type (a `srgb_image` display image, a real matrix through a colormap).
`entries` is a vector of `name => images` pairs, every entry the same number of
images; `titles` names the axes (one per image), and `default` the entry
shown first. Display it with a Makie backend (`using GLMakie; display(fig)`).
An empty `entries`, a `default` not among them or a title count other than the
images' is an `ArgumentError`.

Requires Makie.
"""
function compare_images end
