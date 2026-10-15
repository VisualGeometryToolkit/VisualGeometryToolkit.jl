# =============================================================================
# Pools — THE scoring-pool IO of the stack: the in-memory pool (`ScoringPool`
# of VisualGeometryCore `PoolPair` records) and its residual matrices
# (`PoolResiduals`), bound by the content key `pool_key`; the one writer and
# reader of each (format 4 light files, format 1 residuals files: PLAIN data,
# no stack type serialized); the per-scene loaders; the generic scene-pool
# orchestration `write_scene_pool`; the PnP pool file; and the synthetic E pool
# pair of the engine tests; and THE minimal-sample boundaries of a pool
# (`sample_starts`), the one place a sample count is reconstructed.
#
# A pool is in NormalizedImageFrame (`frame(::ScoringPool)`); a PnP pool in
# ImageFrame. Nothing here converts a frame.
# =============================================================================
module Pools

using LinearAlgebra: I, normalize
using Printf: @sprintf
using Random: AbstractRNG, MersenneTwister
using StaticArrays: SMatrix, SVector, SA
using JLD2: JLD2
using SHA: SHA
using Rotations: RotY, RotZ
using VisualGeometryCore: FundMat, HomographyMat, EssentialMat, NormalizedImageFrame, csponds, sources,
    targets, sampson_distance
using VisualGeometryCore.Synthetic: optimistic_pose_error
using VisualGeometryCore.Pools: PoolPair, num_points, num_models, pair_scene
import .._save_artifact, .._check_artifact_format, .._format_key   # artifact_format.jl
using ..Stats: stable_seed, POOL_RNG
import ..Tuning: sampling_unit
using ..Tuning: PairUnit

export ScoringPool, PoolResiduals, pool_key, load_pool, save_pool, load_pool_residuals,
    save_pool_residuals, residual_matrix, residuals_path, scene_loaders, write_scene_pool,
    save_pnp_pool, load_pnp_pool, sample_starts, sample_ids, derive_sample_starts, save_sample_starts,
    load_sample_starts

public frame, GT_CONVENTION, synthetic_e_pool_pair

include("pools/pool_io.jl")
include("pools/sample_starts.jl")
include("pools/write_scene_pool.jl")
include("pools/pnp_pool.jl")
include("pools/synthetic_pool.jl")

end # module Pools
