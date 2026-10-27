# =============================================================================
# Stats — the shared summary statistics: the finite median and its lower-order
# variant, the validation reference scale and its robust extremes, the eligibility-ruled mAA with its
# outcome counts, the BCa bootstrap CI, the mean-of-scene-median row
# emitter, the stable seed derivation of every random sample and `POOL_RNG`,
# the stream algorithm of every pool. mAA itself is VgtkCore's
# `Synthetic.mean_average_accuracy`, called directly. Loaded FIRST: every other
# module (Evaluation's tuner included) takes its statistics from here.
# =============================================================================
"""
    Stats

Summary statistics of the evaluation: [`finite_median`](@ref) and
[`lower_median`](@ref), the validation [`reference_scale`](@ref) and its
robust extremes ([`scale_bounds`](@ref)), the mean average accuracy (mAA) under
the eligibility rule ([`eligible_maa`](@ref), [`pair_outcomes`](@ref)),
bias-corrected and accelerated (BCa) bootstrap intervals
([`bootstrap_ci`](@ref)), means of per-scene medians
([`scene_mean_median`](@ref), [`scene_mean_median_rows!`](@ref)) and the seed
derivation of every random sample ([`stable_seed`](@ref), [`POOL_RNG`](@ref)),
the quantile of a possibly empty sample ([`quantile_or`](@ref)) and the replay
of persisted improvement traces ([`ImprovementTrace`](@ref),
[`trace_error_at`](@ref), [`anytime_curve`](@ref)).
"""
module Stats

using Statistics: median, mean, quantile
using Random: AbstractRNG
using StableRNGs: StableRNG
import .._config_default   # configs/default.toml (config_defaults.jl)
using SHA: sha256
import Random
using Bootstrap: bootstrap, BasicSampling, BCaConfInt, confint
using VisualGeometryToolkitCore.Synthetic: mean_average_accuracy

export finite_median, lower_median, reference_scale, maa_threshold, eligible_maa, pair_outcomes,
       bootstrap_ci, scene_mean, scene_mean_median, scene_mean_median_rows!, scene_ranges,
       scale_bounds, stable_seed, quantile_or, ImprovementTrace, trace_error_at, anytime_curve

public POOL_RNG

include("stats/stats.jl")

end # module Stats
