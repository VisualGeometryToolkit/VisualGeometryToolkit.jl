# =============================================================================
# Pools — the scoring-pool IO of the stack: the in-memory pool (`ScoringPool`
# of VisualGeometryToolkitCore `PoolPair` records) and its residual matrices
# (`PoolResiduals`), bound by the content key `pool_key`; the one writer and
# reader of each (format 4 light files, format 1 residuals files: PLAIN data,
# no stack type serialized); the per-scene loaders; the generic scene-pool
# orchestration `write_scene_pool`; the PnP pool file; and the synthetic E pool
# pair of the engine tests; and the minimal-sample boundaries of a pool
# (`sample_starts`), the one place a sample count is reconstructed.
#
# A pool is in NormalizedImageFrame (`frame(::ScoringPool)`); a PnP pool in
# ImageFrame. Nothing here converts a frame.
# =============================================================================
"""
    Pools

Scoring-pool input and output. A pool ([`ScoringPool`](@ref)) holds
VisualGeometryToolkitCore `PoolPair` records; its residual matrices
([`PoolResiduals`](@ref)) are bound to it by the content key
[`pool_key`](@ref). The module has one writer and one reader of each
([`save_pool`](@ref), [`load_pool`](@ref)), the per-scene loaders
([`scene_loaders`](@ref)), the scene-pool producer
([`write_scene_pool`](@ref)), the perspective-n-point (PnP) pool file
([`save_pnp_pool`](@ref)) and the minimal-sample boundaries of a pool
([`sample_starts`](@ref)).

A scoring pool is stored in `NormalizedImageFrame` ([`frame`](@ref)) and a PnP
pool in `ImageFrame`. Nothing here converts a frame.
"""
module Pools

using LinearAlgebra: I, normalize
using Printf: @sprintf
using Random: AbstractRNG, MersenneTwister
using StaticArrays: SMatrix, SVector, SA
using JLD2: JLD2
using Dates: Dates
using SHA: SHA
using Rotations: RotY, RotZ
using VisualGeometryToolkitCore: FundMat, HomographyMat, EssentialMat, NormalizedImageFrame, csponds, sources,
    targets, sampson_distance
using VisualGeometryToolkitCore.Synthetic: optimistic_pose_error
using VisualGeometryToolkitCore.Pools: PoolPair, num_points, num_models, pair_scene, pixel_csponds
import .._save_artifact, .._check_artifact_format, .._format_key   # artifact_format.jl
using ..Stats: stable_seed, POOL_RNG
import ..Tuning: sampling_unit
using ..Tuning: PairUnit

export ScoringPool, PoolResiduals, pool_key, load_pool, save_pool, load_pool_residuals,
    save_pool_residuals, residual_matrix, residuals_path, scene_loaders, write_scene_pool,
    save_pnp_pool, load_pnp_pool, sample_starts, sample_ids, derive_sample_starts, save_sample_starts,
    load_sample_starts, export_pixel_inputs

public frame, GT_CONVENTION, synthetic_e_pool_pair

include("pools/pool_io.jl")
include("pools/sample_starts.jl")
include("pools/write_scene_pool.jl")
include("pools/pnp_pool.jl")
include("pools/pixel_inputs.jl")
include("pools/synthetic_pool.jl")

end # module Pools
