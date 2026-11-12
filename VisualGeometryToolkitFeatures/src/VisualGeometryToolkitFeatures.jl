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
- **Camera RAW:** the stack's one RAW path. [`load_raw`](@ref) reads a Bayer RAW file as
  exact linear values on the 2×2 Bayer cells (the green-cell image for geometry and
  radiometry, or R G B), with saturation flags, true sample positions and the exact frame
  map onto the raw ([`raw_frame_map`](@ref)); a display-only demosaic for figures.
- **Extensions:** JLD2 backs [`FileCache`](@ref); Makie draws detections and correspondences;
  LibRaw_jll is the RAW decoder.
"""
module VisualGeometryToolkitFeatures

# =============================================================================
# Dependencies
# =============================================================================

using StaticArrays: SVector, SMatrix
using ImageCore: channelview
using StructArrays: StructArray
using FixedPointNumbers: N0f8, FixedPoint
using Colors: Gray, Colorant, RGB, gray
using ImageMorphology: MaxTree, areas   # the component tree behind detect_mser
using Distributions: FDist, quantile   # the F-test of detect_foerstner
using PNGFiles: PNGFiles     # load_linear_image: a PNG read without gamma re-encoding

# VisualGeometryToolkitCore (geometry primitives, units, the correspondence layer)
import KernelAbstractions as KA   # KA devices are the backend dispatch tokens
using VisualGeometryToolkitCore: Point2
using VisualGeometryToolkitCore: Csponds  # #21 correspondence layer
using VisualGeometryToolkitCore: default_device   # the family's one KA backend resolver — never a hardcoded KA.CPU()
using VisualGeometryToolkitCore: device_free_bytes, reclaim_device!   # the accelerator's free memory and its pool handed back (base/device.jl, the CUDA extension's queries)
using VisualGeometryToolkitCore: ka_upload, on_device, device_zeros   # the family's one upload to a device (base/device.jl)
using VisualGeometryToolkitCore: AbstractDetection, FeatureFrame, OrientedDisc, AffineFrame, PointFrame
using VisualGeometryToolkitCore: characteristic_scale
using VisualGeometryToolkitCore: image_warp, image_warp!, AffineMap, Rect   # LoMaDescriptor's resampling to the graph's input (loma.jl)
using VisualGeometryToolkitCore: DeviceIdentity, ExposureSettings, LensSettings, CaptureSettings, parse_unit, mm, m   # a RAW's EXIF, typed (raw_image.jl)
using VisualGeometryToolkitCore: Frame, PixelFrame, SimilarityMap, LinearMap, Translation, relabel, scale, translation   # the RAW cell grid's frame map (raw_image.jl)
using LinearAlgebra: UniformScaling, isposdef
using VisualGeometryToolkitCore: CovMat, UnlabeledFrame, Uncertain   # keypoint uncertainty
import VisualGeometryToolkitCore: cov, geom   # extended for Detections of an Uncertain frame
# Extended here with Detection methods:
import VisualGeometryToolkitCore: Attributed, Disc, EllipticFrame, Ellipse
import VisualGeometryToolkitCore: centroid, shape_matrix, feature_frame, region


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
include("raw_image.jl")      # load_raw — the one RAW path: a camera RAW mosaic as exact linear values (the decoder is the LibRaw_jll extension)
include("batching.jl")       # batched_rows — the describe backends' chunking skeleton
include("asift/pyramid.jl")
include("asift/warp.jl")
include("asift/backproject.jl")
include("asift/crosswarp.jl")
include("asift/uncertainty.jl")   # FeatureCovariance: the keypoint covariance tables and per-keypoint math
include("asift/driver.jl")
include("asift/api.jl")
include("variance_stabilization.jl")   # variance_stabilize — a RAW's green cells with noise σ = 1 everywhere; ASIFT on it at k noise σ, saturation masked
include("matching.jl")
include("nn_match.jl")
include("matching_ka.jl")   # the one matcher: KA kernels on the CPU and any GPU
include("foerstner.jl")      # detect_foerstner — the Förstner interest operator (after ip_fop.m); after asift/ and matching_ka.jl, whose upsampling and upload it uses
include("matching_stacked_ka.jl")   # stacked multi-gallery matcher (hoisted from BlobBoards)
include("described_feature.jl")
include("onnx_runtime.jl")   # ONNX Runtime sessions: the execution provider (the CUDA 12 libraries) and the thread pools
include("loma.jl")
include("spread.jl")         # spread_strongest — SSC (Bailo et al. 2018): N detections spread over the image, on KernelAbstractions
include("visualization_stubs.jl")

include("cache.jl")

# =============================================================================
# Backend-dispatched matching
# =============================================================================

"""
    nn_match(desc1, desc2, strategy = MutualNN(); device = default_device(), kw...) -> Vector{NNMatch}

[`match`](@ref) on `device`, any KernelAbstractions backend
(VisualGeometryToolkitCore's `default_device()` when none is supplied: an
accelerator when one is functional, else `KA.CPU()`). Every backend runs the same GEMM and selection kernels, so a GPU
(e.g. `CUDABackend()`) returns what the CPU returns, with every keyword `match`
accepts: `metric`, `dm`, `ratio_threshold`, `min_similarity`, `max_distance`.
"""
function nn_match(
        desc1, desc2, strategy::NNStrategy = MutualNN();
        device::KA.Backend = default_device(), kw...
    )
    return _match(desc1, desc2, strategy, device; kw...)
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
export tilt_schedule             # ASIFT's tilts by name: :imas25 (the default), :untilted (the identity alone)
export AsiftPlan, ImasFeatureKA
export clear_asift_plans!, set_max_cached_plans!
export FeatureCovariance            # keypoint uncertainty: asift_detect(img; uncertainty = FeatureCovariance())
export gray_f32, gray_u8        # the input ladders every detector-side entry point uses
export load_linear_image        # a PNG as linear radiance: no gamma re-encoding, optional sRGB decode, box downsample
export srgb_image               # the one display encoding: exposure, white balance, NaN → background, tone, clamp, sRGB, N0f8/N0f16
public srgb_to_linear, linear_to_srgb, box_downsample, gaussian_blur   # the sRGB decode and encode; gaussian_blur: the (masked) separable Gaussian of a linear image
export load_raw, load_raw_mosaic   # a camera RAW file as linear values (:green_cells, :bin2; :demosaic for display), and its decoded mosaic
export LinearRaw, RawMosaic, RawDecoder, LibRawDecoder
export variance_stabilize, StabilizedImage   # the detection image of a RAW: the generalized Anscombe transform (noise σ = 1), saturation masked
export BayerCellFrame, raw_frame_map, bayer_plane_map, green_cell_sites   # a plane's frame, the cell grid's and each plane's exact map onto the raw, a green cell's footprint
export raw_coordinates          # THE accessor of an image's sensor-pixel coordinates (raw_frame_map applied)
public raw_provenance        # decoder, mode, LibRaw version and frame map, as plain data
public cfa_string, photosite_colors   # the pattern as letters; each photosite's colour index (the pattern tiled)
export load_raw_metadata        # a RAW file's EXIF fields alone (no photosites decoded)
public exposure_settings, lens_settings, body_identity, lens_identity, capture_time   # a RAW's EXIF as Core's typed values
export extract_capture_settings # frame index → RAW metadata in, Core's CaptureSettings out
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
public LOMA_IMAGE_SIZE              # the descriptor graph's input side
export spread_strongest             # about n of the strongest detections spread over the image (SSC), in detector order
public loma_provenance              # the model, the execution provider used and the threads, as plain data
public onnx_cuda_status, OnnxCudaStatus, resolve_execution_provider   # whether ONNX Runtime's CUDA provider runs here (why not); :auto's answer
public onnx_session                 # an ONNX Runtime session as the package builds its own: provider, threads, CUDA arena cap

# -----------------------------------------------------------------------------
# 5. PERSISTENT CACHE - FileCache (JLD2 backend in VisualGeometryToolkitFeaturesJLD2Ext)
# -----------------------------------------------------------------------------
export FileCache, open_cache, close_cache, cache_key
# `get!(f, cache::FileCache, key)` (JLD2 extension) extends Base.get! — nothing to export.

# -----------------------------------------------------------------------------
# 6. VISUALIZATION - stubs implemented in VisualGeometryToolkitFeaturesMakieExt
# -----------------------------------------------------------------------------
export detections!, compare_images   # detections over an image; the one image viewer (Makie extension)

include("precompile.jl")   # last: the workload calls the API defined above

end # module VisualGeometryToolkitFeatures
