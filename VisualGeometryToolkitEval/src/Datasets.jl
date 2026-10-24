# =============================================================================
# Datasets — the raw dataset readers: each returns concrete VisualGeometryToolkitCore
# `StereoRawPair`s (pixels; correspondences, SNN, calibration, image sizes,
# GT) from which a pool is drawn.
#
#   PhotoTourism  load_pt_pairs, PtRawTrees, PT_SCENES (HDF5
#                 matches + COLMAP recon through VisualGeometryToolkitIO)
#   HEB           load_heb_pairs (`<scene>_homographies.h5`; the scene list is
#                 configs/splits.toml `[splits.heb]`)
#   RoMa / SP+LG  load_roma_pairs (consolidated JLD2)
#
# and the producer of those consolidated files from the depth-validated stereo
# datasets (ETH3D, ScanNet1500, KITTI, SevenScenes): `stereo_pairs`, the
# matcher's sparse-match files (`load_sparse_matches`), GT validation against
# depth (`load_validation_data`, `gt_config`) and `consolidate_csponds`.
#
# The subsample sizes and seed of a scene are the caller's (its pool
# configuration); the read thresholds default to configs/default.toml
# `[pool.<dataset>]`. Raw pairs are never persisted.
# =============================================================================
"""
    Datasets

Raw dataset readers. Each returns VisualGeometryToolkitCore `StereoRawPair`s in
pixels, from which a scoring pool is drawn: PhotoTourism
([`load_pt_pairs`](@ref)), the homography benchmark HEB
([`load_heb_pairs`](@ref)) and consolidated RoMa or SuperPoint+LightGlue
correspondences ([`load_roma_pairs`](@ref)). The module also produces those
consolidated files from depth-validated stereo datasets ([`ETH3D`](@ref),
[`ScanNet1500`](@ref), [`KITTI`](@ref), [`SevenScenes`](@ref)) through
[`stereo_pairs`](@ref), [`load_sparse_matches`](@ref),
[`load_validation_data`](@ref) and [`consolidate_csponds`](@ref).

The read thresholds default to `configs/default.toml` (`[pool.<dataset>]`).
"""
module Datasets

import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
using Random: shuffle
using LinearAlgebra: I
using DelimitedFiles: readdlm
using HDF5: HDF5, h5open
using NPZ: npzread
using PNGFiles: PNGFiles
using JLD2: JLD2
using StaticArrays: SMatrix, SVector
using StructArrays: StructArrays, StructArray
using Rotations: QuatRotation
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, Size2, FundMat, EssentialMat, sampson_distance, intrinsics,
    centroid, model, extent, extrinsics, source, target, project, unproject, csponds, Point3, Vec3,
    Camera, BoundedCamera, CameraModel, CameraCalibrationMat, EuclideanMap, StereoRig,
    validate_gt_inliers, VgcQuantity, m, mm, ustrip
using VisualGeometryToolkitCore.Pools: Pools, StereoRawPair
import VisualGeometryToolkitIO
using VisualGeometryToolkitIO: UnknownScale, camera_model, UnitContract, ColmapModel, ColmapCamera, ColmapImage
import .._config_default   # configs/default.toml (config_defaults.jl)
using ..Stats: stable_seed, POOL_RNG

export PtPair, PtRawTrees, PT_SCENES, load_pt_pairs, pt_scene_files,
    HebPair, load_heb_pairs,
    RomaPair, load_roma_pairs,
    StereoPair, GTConfig, AbstractStereoDataset, ETH3D, ScanNet1500, KITTI, SevenScenes,
    stereo_pairs, load_validation_data, gt_config, eth3d_pairs, scannet1500_pairs, kitti_pairs,
    seven_scenes_pairs, load_sparse_matches, consolidate_csponds, CONSOLIDATED_FORMAT

include("datasets/raw_pairs.jl")
include("datasets/phototourism.jl")
include("datasets/heb.jl")
include("datasets/stereo_datasets.jl")
include("datasets/eth3d.jl")
include("datasets/scannet1500.jl")
include("datasets/kitti.jl")
include("datasets/seven_scenes.jl")
include("datasets/sparse_matches.jl")
include("datasets/consolidate.jl")
include("datasets/roma.jl")

end # module Datasets
