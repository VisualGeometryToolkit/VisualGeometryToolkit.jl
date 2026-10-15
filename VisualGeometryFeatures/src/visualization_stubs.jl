# =============================================================================
# Visualization stubs — implemented in VisualGeometryFeaturesMakieExt
# =============================================================================
#
# The canonical correspondence primitives are VGC's `matches!`/`matches`;
# this package EXTENDS them by dispatch for feature-typed inputs (`Cspond`
# vectors: two-axis color-coded endpoints, image-pair figures) rather than
# exporting competing names. The historical `cspond!`/`matchlines` names
# are gone — use `matches!(ax_A, ax_B, cs)` and `matches(img_A, img_B, cs)`.
#
# VGF-owned inspection sugar:
#   detections!  — Detection-metadata-aware overlay over VGC's `ellipses!`

"""
    detections!(ax, dets::AbstractVector{<:Detection};
                cutoff=1.0, color_by=nothing, colormap=:viridis, kwargs...)

Overlay detections as batched ellipse boundaries (VGC `ellipses!`).

`color_by` selects the stroke coloring:
- `nothing` — constant `strokecolor`
- `:response` — `|metadata.response|` through `colormap`
- an `AbstractVector{<:Real}` — explicit per-detection values

Requires Makie.
"""
function detections! end
