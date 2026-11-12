# =============================================================================
# ImageSmooth backend of detect_image_circle — the optional L0 smoothing
# =============================================================================
#
# The per-pixel maximum's L0 gradient smoothing (ImageSmooth's `L0Smooth`, weight λ) before its 8-bit levels, used only
# when `detect_image_circle(…; smoothing_weight > 0)` (src/image_circle.jl, step 2).
# =============================================================================

module VisualGeometryToolkitFeaturesImageSmoothExt

using ImageSmooth: L0Smooth, smooth!
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures

VisualGeometryToolkitFeatures._l0_smooth!(levels, upper, λ::Float64) = smooth!(levels, upper, L0Smooth(; λ))

end
