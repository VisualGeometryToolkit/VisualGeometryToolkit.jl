# =============================================================================
# Stats — THE shared summary statistics: the finite median and its lower-order
# variant, the validation reference scale and its robust extremes, the eligibility-ruled mAA with its
# outcome counts, the BCa bootstrap CI, the mean-of-scene-median row
# emitter, THE stable seed derivation of every random sample and `POOL_RNG`,
# the stream algorithm of every pool. mAA itself is VGC's
# `Synthetic.mean_average_accuracy`, called directly. Loaded FIRST: every other
# module (Evaluation's tuner included) takes its statistics from here.
# =============================================================================
module Stats

using Statistics: median, mean, quantile
using Random: AbstractRNG
using StableRNGs: StableRNG
import .._config_default   # configs/default.toml (config_defaults.jl)
using SHA: sha256
import Random
using Bootstrap: bootstrap, BasicSampling, BCaConfInt, confint
using VisualGeometryCore.Synthetic: mean_average_accuracy

export finite_median, lower_median, reference_scale, maa_threshold, eligible_maa, pair_outcomes,
       bootstrap_ci, scene_mean, scene_mean_median, scene_mean_median_rows!, scene_ranges,
       scale_bounds, stable_seed

public POOL_RNG

include("stats/stats.jl")

end # module Stats
