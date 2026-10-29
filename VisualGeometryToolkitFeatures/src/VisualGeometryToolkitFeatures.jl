"""
    VisualGeometryToolkitFeatures

Feature detection and descriptor matching.

- **Detection:** ASIFT (IMAS-25) on KernelAbstractions. [`asift_detect`](@ref) runs on the
  CPU or any KernelAbstractions GPU backend and returns [`Detection`](@ref)s (a frame plus
  metadata). VisualGeometryToolkitIO writes and reads them as JSON (`json_value`, `from_json`).
- **Matching:** nearest-neighbour descriptor matching, one KernelAbstractions
  implementation on the CPU or any GPU backend ([`nn_match`](@ref), `match`), and a
  stacked matcher over many galleries ([`nn_match_stacked`](@ref)).
- **LoMa:** the learned DeDoDe descriptors and LoMa matchers, from the authors' ONNX
  exports ([`LoMaDescriptor`](@ref), [`LoMaMatcher`](@ref)).
- **Extensions:** JLD2 backs [`FileCache`](@ref); Makie draws detections and correspondences.
"""
module VisualGeometryToolkitFeatures

# =============================================================================
# Dependencies
# =============================================================================

using StaticArrays: SVector, SMatrix
using ImageCore: channelview
using StructArrays: StructArray
using FixedPointNumbers: N0f8
using Colors: Gray, Colorant, RGB, gray
using ImageMorphology: MaxTree, areas   # the component tree behind detect_mser
using Distributions: FDist, quantile   # the F-test of detect_foerstner
using PNGFiles: PNGFiles     # load_linear_image: a PNG read without gamma re-encoding

# VisualGeometryToolkitCore (geometry primitives, units, the correspondence layer)
import KernelAbstractions as KA   # KA devices are the backend dispatch tokens
using VisualGeometryToolkitCore: Point2
using VisualGeometryToolkitCore: Csponds  # #21 correspondence layer
using VisualGeometryToolkitCore: default_device   # the family's one KA backend resolver — never a hardcoded KA.CPU()
using VisualGeometryToolkitCore: AbstractDetection, FeatureFrame, OrientedDisc, AffineFrame, PointFrame
using VisualGeometryToolkitCore: characteristic_scale
using VisualGeometryToolkitCore: CovMat, UnlabeledFrame, Uncertain, implicit_covariance, propagate   # keypoint uncertainty
import VisualGeometryToolkitCore: cov, geom   # extended for Detections of an Uncertain frame
# Extended here with Detection methods:
import VisualGeometryToolkitCore: Attributed, Disc, EllipticFrame, Ellipse
import VisualGeometryToolkitCore: centroid, moment_matrix, feature_frame, region


# =============================================================================
# INCLUDED SOURCE FILES
# =============================================================================
#
#   detection_types.jl  Detection{F,M}, polarity traits
#   image_input.jl      gray_f32 / gray_u8 input ladders
#   mser.jl             detect_mser, MserRegion, region_mask, boundary_contours
#   foerstner.jl        detect_foerstner (Förstner/Gülch, Köthe; after ip_fop.m)
#   batching.jl         batched_rows chunking for the describe backends
#   asift/              KernelAbstractions ASIFT (IMAS-25) — THE detector.
#                       Identity-only tilt schedule ≡ plain DoG-SIFT.
#   matching.jl         CspondSet (Cspond and csponds come from VGC)
#   nn_match.jl         NN/MutualNN/KnnMatch strategies, metrics, match
#   matching_ka.jl      the one matcher (KernelAbstractions) behind match and nn_match; pairwise_matrix
#   matching_stacked_ka.jl  nn_match_stacked over many galleries
#   described_feature.jl  DescribedFeature, descriptor_matrix
#   visualization_stubs.jl  detections! stub; matches!/matches Cspond overloads live in the Makie ext
#   cache.jl            FileCache (JLD2 backend in VisualGeometryToolkitFeaturesJLD2Ext)
#
# REMOVED 2026-07: the legacy scale-space detection stack (scalespace.jl,
# responses.jl, extrema.jl, detector.jl, nms.jl — FeatureDetector, ScaleSpace,
# Hessian/DoG/Laplacian response methods, suppress_*) is superseded by
# src/asift/. `bbox` moved to VisualGeometryToolkitCore. hungarian.jl and
# synthetic_correspondences.jl were dead code.

include("detection_types.jl")
include("image_input.jl")    # gray_f32 / gray_u8 — the input ladders (hoisted from BlobBoards)
include("mser.jl")           # detect_mser — VLFeat's MSER selection on ImageMorphology's MaxTree
include("linear_image.jl")   # load_linear_image — a PNG as linear radiance (hoisted from RadiometricCamera)
include("batching.jl")       # batched_rows — the describe backends' chunking skeleton
include("asift/pyramid.jl")
include("asift/warp.jl")
include("asift/backproject.jl")
include("asift/crosswarp.jl")
include("asift/uncertainty.jl")   # FeatureCovariance: the keypoint covariance tables and per-keypoint math
include("asift/driver.jl")
include("asift/api.jl")
include("matching.jl")
include("nn_match.jl")
include("matching_ka.jl")   # the one matcher: KA kernels on the CPU and any GPU
include("foerstner.jl")      # detect_foerstner — the Förstner interest operator (after ip_fop.m); after asift/ and matching_ka.jl, whose upsampling and upload it uses
include("matching_stacked_ka.jl")   # stacked multi-gallery matcher (hoisted from BlobBoards)
include("described_feature.jl")
include("loma.jl")
include("visualization_stubs.jl")

include("cache.jl")

# =============================================================================
# Backend-dispatched matching
# =============================================================================

"""
    nn_match(desc1, desc2, strategy = MutualNN(); device = nothing, kw...) -> Vector{NNMatch}

[`match`](@ref) on `device`, any KernelAbstractions backend (`device = nothing`
is `KA.CPU()`). Every backend runs the same GEMM and selection kernels, so a GPU
(e.g. `CUDABackend()`) returns what the CPU returns, with every keyword `match`
accepts: `metric`, `dm`, `ratio_threshold`, `min_similarity`, `max_distance`.
"""
function nn_match(
        desc1, desc2, strategy::NNStrategy = MutualNN();
        device::Union{Nothing, KA.Backend} = nothing, kw...
    )
    return _match(desc1, desc2, strategy, something(device, KA.CPU()); kw...)
end

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================
#
# Every export and `public` declaration of the package is here; the source
# files define names but do not export them.

# -----------------------------------------------------------------------------
# 1. DETECTIONS - the Detection alias, its accessor, polarity helpers
# -----------------------------------------------------------------------------
export Detection
export feature_frame            # VGC's accessor; the Detection method is added here
export is_light, is_dark
# The Holy trait behind is_light/is_dark: a metadata type opts in by extending
# `polarity_trait`.
public HasPolarityTrait, HasPolarity, NoPolarity, polarity_trait

# -----------------------------------------------------------------------------
# 2. ASIFT (IMAS-25) DETECTOR - KernelAbstractions, CPU or any GPU backend
# -----------------------------------------------------------------------------
export asift_detect, asift_detect!, asift_detections
export AsiftPlan, ImasFeatureKA
export clear_asift_plans!, set_max_cached_plans!
export FeatureCovariance            # keypoint uncertainty: asift_detect(img; uncertainty = FeatureCovariance())
export gray_f32, gray_u8        # the input ladders every detector-side entry point uses
export load_linear_image        # a PNG as linear radiance: no gamma re-encoding, optional sRGB decode, box downsample
public srgb_to_linear, box_downsample, gaussian_blur   # gaussian_blur: the (masked) separable Gaussian of a linear image
export batched_rows             # the describe backends' chunking skeleton
export detect_mser, region_mask   # MSER (VLFeat's selection on ImageMorphology's MaxTree)
public MserRegion, boundary_contours
export detect_foerstner        # the Förstner interest operator: labelled corner / circular points

# -----------------------------------------------------------------------------
# 3. MATCHING - strategies, metrics, NN matching on any KA backend
# -----------------------------------------------------------------------------
export NNStrategy, NN, MutualNN, KnnMatch
export DescriptorMetric, DescriptorSimilarity, DescriptorDistance, CosineSimilarity, L2
export NNMatchMeta, NNMatch
export nn_match, pairwise_matrix
# `match` is public, not exported: exported, it clashed with Base.match, so an
# unqualified match(r"...", s) was ambiguous in every `using VisualGeometryToolkitFeatures` module.
public match
export resolve_one_to_one, resolve_one_to_one!
export StackedDescriptorIndex, nn_match_stacked
export Cspond, csponds          # VGC's pair alias and builder; the NNMatch method is added here
export CspondSet                # N-way correspondence
# point_csponds deleted (#21) — use VGC points/planar_lift/lift

# -----------------------------------------------------------------------------
# 4. DESCRIBED FEATURES - Detection + descriptor, StructArray-backed
# -----------------------------------------------------------------------------
export DescribedFeature, described_features, descriptor_matrix
export LoMaDescriptor, LoMaMatcher   # LoMa (ONNX); models cached in the scratch space

# -----------------------------------------------------------------------------
# 5. PERSISTENT CACHE - FileCache (JLD2 backend in VisualGeometryToolkitFeaturesJLD2Ext)
# -----------------------------------------------------------------------------
export FileCache, open_cache, close_cache, cache_key
# `get!(f, cache::FileCache, key)` (JLD2 extension) extends Base.get! — nothing to export.

# -----------------------------------------------------------------------------
# 6. VISUALIZATION - stubs implemented in VisualGeometryToolkitFeaturesMakieExt
# -----------------------------------------------------------------------------
export detections!

include("precompile.jl")   # last: the workload calls the API defined above

end # module VisualGeometryToolkitFeatures
