# =============================================================================
# Pools — the benchmark pool record and the raw-pair interface
# =============================================================================
#
# The light pool record `PoolPair` (one image pair's precomputed minimal-sample
# models, pose errors and ground truth, stored in NormalizedImageFrame) with its
# accessors, and `StereoRawPair`, the interface a dataset reader's raw pair
# satisfies before a pool is drawn from it. The record lives here, the lowest
# common parent of its readers (RansacScoringDoneRight's pool scorers), its
# writer (VisualGeometryToolkitEstimation's pool-pair constructor) and its IO
# (VisualGeometryToolkitEval). The residual matrices are NOT part of the record: they
# are a separate file bound to it by a content key (VisualGeometryToolkitEval).
#
# The pool is the stack's one declared exception to "frame in = frame out":
# every point datum and model of a record is in NormalizedImageFrame, and
# every accessor that crosses the boundary names it.
#
# DEPENDENCY GRAPH:
#
#   pool_pair.jl   (PoolPair, pool_normalization(size1, size2), accessors)
#        ↓          Uses: CspondNormalization, DiagonalNormalization,
#        ↓                FramedAffine, similarity_scale, StereoRig
#   raw_pair.jl    (StereoRawPair + accessors; pool_normalization of a raw pair)
#                   Uses: pool_normalization(size1, size2)
#
# =============================================================================

"""
    Pools

The benchmark pool record and the raw-pair interface (`using
VisualGeometryToolkitCore.Pools`):

- [`PoolPair`](@ref): one image pair of a scoring pool — correspondences,
  minimal-sample models and their pose errors, ground truth — in
  [`NormalizedImageFrame`](@ref), with the per-image maps it was normalized by;
- its accessors: [`num_points`](@ref), [`num_models`](@ref),
  [`pool_imc`](@ref), [`gt_pose`](@ref), [`calibration_matrices`](@ref),
  [`pool_normalization`](@ref), [`pool_frame_maps`](@ref),
  [`pool_frame_calibrations`](@ref), [`pool_pixel_scale`](@ref),
  [`pixel_csponds`](@ref) and [`rig`](@ref);
- [`StereoRawPair`](@ref): what a dataset reader's pair provides before a pool
  is drawn from it (pixel correspondences, calibration, image sizes, GT).

The record holds no residuals: a pool's residual matrices are a separate file,
bound to the record by a content key, read by the evaluation package.
"""
module Pools

using StaticArrays: SMatrix, SVector
using CoordinateTransformations: AffineMap
using ..VisualGeometryToolkitCore: HomographyMat, CspondNormalization, DiagonalNormalization,
    source_map, target_map, to_homogeneous, ImageFrame, NormalizedImageFrame, csponds, sources, targets
using ..VisualGeometryToolkitCore.Cameras: FramedAffine, StereoRig, similarity_scale

# =============================================================================
# EXPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. THE POOL RECORD - one pair of a scoring pool, in NormalizedImageFrame
# -----------------------------------------------------------------------------
export PoolPair
export num_points, num_models, pool_imc, gt_pose, calibration_matrices
# The pool frame, named at every boundary: the record's two-sided normalization,
# its framed maps (ImageFrame → NormalizedImageFrame), the pool-frame cameras,
# the pool units per pixel, and the correspondences back in pixels.
export pool_normalization, pool_frame_maps, pool_frame_calibrations, pool_pixel_scale, pixel_csponds
public rig   # the GT StereoRig, built on access (not exported: `rig` is every caller's local name)

# -----------------------------------------------------------------------------
# 2. THE RAW-PAIR INTERFACE - a dataset reader's pair, in pixels
# -----------------------------------------------------------------------------
export StereoRawPair
export pair_csponds, pair_snn, pair_K1, pair_K2, pair_size1, pair_size2,
    pair_R_gt, pair_t_gt, pair_gt_inliers, pair_scene

# =============================================================================
# INCLUDE ORDER
# =============================================================================

include("pool_pair.jl")   # PoolPair + accessors; pool_normalization(size1, size2)
include("raw_pair.jl")    # StereoRawPair + accessors (uses pool_normalization(size1, size2))

end # module Pools
