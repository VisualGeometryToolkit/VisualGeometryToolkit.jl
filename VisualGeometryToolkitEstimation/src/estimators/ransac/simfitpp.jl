# =============================================================================
# SIMFIT++ — less biased noise-scale (loss-width) estimation for threshold-
# robust RANSAC (Edstedt, "Less Biased Noise Scale Estimation for Threshold-
# Robust RANSAC", CVPRW 2025). A step-for-step port of the reference
# implementation, github.com/Parskatt/simfitpp (src/simfitpp/simfitpp.py,
# scale_estimators.py, noise_models.py), kept in the reference's structure:
#
#   noise model        χ²_{d_g} on the squared residual (reference: χ²₁ for
#                      the scalar Sampson residual of F / E)
#   UnbiasedMed        `_unbiased_median_variance`: the median of the squared
#                      residuals BELOW the current loss width, de-biased for
#                      that truncation by a five-step fixed-point iteration on
#                      the quantile the truncated median corresponds to, then
#                      divided by the χ² quantile
#   SIMFITPP.__call__  `solve(EstimationProblem(LossWidth, estimator, p), SimFitPP())`: `max_iter` rounds of a random
#                      train/validation split; the geometry is estimated on the
#                      train split at the current loss width (fitting bias
#                      removed by scoring held-out residuals); the validation
#                      residuals give the scale, hence the new loss width
#                      loss_width = sqrt(ppf(alpha)·σ²); out-of-bounds proposals are
#                      skipped; accepted proposals enter a running mean; the
#                      loop stops on `ftol` or on a repeated (3-decimal)
#                      value; the final estimate is bounds-checked and, in the
#                      multi-pair setting, filtered in log space with the
#                      `num_previous_est` earlier estimates.
#
# The geometry estimator is a callback pair (`fit`, `residuals_sq`) so the
# same loop runs on a full RANSAC (the reference: PoseLib at the loss width
# with refinement; PanoramaStitching: a rotation refit per verified pair) or on
# a hypothesis pool (`EstimationProblem(PoolSelection, …)`: the MSAC-best pool hypothesis scored on
# the train split — the paper's fixed-pool protocol, no refinement).
#
# Moved from RANSACScoringDoneRight (src/simfitpp.jl) on 2026-11-11, unchanged
# but for the train-split MSAC sum, which is now VisualGeometryToolkitCore's one
# threshold `score!` (the same per-point quality, summed in the same order).
# =============================================================================

"""
    SimFitPP(; alpha, max_iter, ftol, train_fraction, min_loss_width, max_loss_width, dg)

SIMFIT++ ([Edstedt2025](@citet); the reference calls the loss width its threshold), a randomized
[`FitKind`](@extref VisualGeometryToolkitCore.FitKind) (`supports_solve_rng` is true: `rng` is a solve keyword):

    solve(EstimationProblem(LossWidth, (; fit, residuals_sq, n), (; initial_loss_width[, num_previous_est])), SimFitPP(); rng)
    solve(EstimationProblem(PoolSelection, residuals::AbstractMatrix, (; initial_loss_width[, …])), SimFitPP(); rng)
    solve(EstimationProblem(PoolSelection, pair::PoolPair, (; initial_loss_width[, …])), SimFitPP(); rng)

Every field defaults to configs/default.toml's `[simfitpp]` table, which holds the reference's constructor defaults.
`alpha` is the inlier coverage the loss width targets (loss_width² = ppf_{χ²_dg}(alpha)·σ²);
`min_loss_width` and `max_loss_width` bound an acceptable loss width in the
residual unit of the residuals handed to the estimator (pixels in the
reference); `dg` is the residual codimension (the χ² degrees of freedom).

**The loss width** (`EstimationProblem(LossWidth, estimator, p)`, VisualGeometryToolkitCore's `LossWidth` as the model
tag): the reference's `SIMFITPP.__call__` over `n` correspondences. `fit(loss_width, train)` estimates the geometry
from the correspondences indexed by `train` at `loss_width` and returns a model handle (or `nothing` on failure);
`residuals_sq(model, val)` returns the squared residuals of that model on the held-out indices `val` (or `nothing`).
The context: `initial_loss_width` seeds the iteration, and `num_previous_est = p` (default 0) earlier estimates behind
it filter a successful estimate as `exp(log(loss_width)/(p+1) + p·log(initial_loss_width)/(p+1))`. `u` is the loss
width, `initial_loss_width` unchanged when the estimate fails: `successful_retcode(result)` is the reference's
`success` (the estimate in bounds). Its `stop_rule`: `IncrementSize` (an accepted proposal moved the width by less than
`ftol`), `NoImprovement` (a repeated 3-decimal proposal) or `IterationCap` (`max_iter` rounds); `stats.nsteps` the
rounds drawn, `n` the correspondences.

**A pool's selection** (`EstimationProblem(PoolSelection, …)`): SIMFIT++ on a fixed hypothesis pool, `residuals` the
pool's n×M residual matrix (points × models) in the unit `initial_loss_width` and the bounds are expressed in. The
geometry "estimator" is the MSAC-best pool hypothesis on the train split at the current loss width, scored by its
validation-split residuals; `u` is the MSAC argmax over ALL points at the final loss width (`0` for an empty pool),
`metadata.loss_width` that width, and the record the loss width's. The pair form is PIXELS in, pixels out: it runs on
VisualGeometryToolkitCore's `Pools.pixel_residuals(pair)`, so `initial_loss_width`, the bounds (the reference's pixel
defaults, 0.25–8 px) and the loss width are all pixels; to work in the pool frame's unit, solve the matrix form on the
pair's stored residual matrix with bounds in that unit ([`with_residual_unit`](@ref)).
"""
Base.@kwdef struct SimFitPP <: AbstractEstimationAlgorithm
    alpha::Float64          = _dflt("simfitpp", "alpha")
    max_iter::Int           = _dflt("simfitpp", "max_iter")
    ftol::Float64           = _dflt("simfitpp", "ftol")
    train_fraction::Float64 = _dflt("simfitpp", "train_fraction")
    min_loss_width::Float64 = _dflt("simfitpp", "min_loss_width")
    max_loss_width::Float64 = _dflt("simfitpp", "max_loss_width")
    dg::Int                 = _dflt("simfitpp", "dg")
end
estimator_kind(::SimFitPP) = FitKind()
supports_solve_rng(::AbstractEstimationProblem, ::SimFitPP) = true   # randomized without being a find: its train/validation splits

"""
    PoolSelection

The model TAG of a hypothesis pool's selection (abstract, never instantiated): `EstimationProblem(PoolSelection,
residuals_or_pair, (; initial_loss_width))` estimates which pool model to keep (`u` its index), solved by
[`SimFitPP`](@ref).
"""
abstract type PoolSelection end

"""
    with_residual_unit(cfg::SimFitPP, unit_scale) -> SimFitPP

The same configuration with the loss-width bounds expressed in another residual
unit. `min_loss_width`/`max_loss_width` are multiplied by `unit_scale` (e.g.
VisualGeometryToolkitCore's `Pools.pool_pixel_scale(pair)` to carry the
reference's pixel bounds into a pool stored in the pool frame).
"""
with_residual_unit(cfg::SimFitPP, unit_scale::Real) =
    SimFitPP(cfg.alpha, cfg.max_iter, cfg.ftol, cfg.train_fraction,
             cfg.min_loss_width * unit_scale, cfg.max_loss_width * unit_scale, cfg.dg)

# The reference's `UnbiasedMed` ([Edstedt2025](@citet)): the noise variance σ² from
# the squared residuals below `sq_loss_width`, corrected for the truncation. With
# `med` the median of the kept residuals and `ratio = sq_loss_width / med`, the
# quantile `q` that the truncated median corresponds to under χ²_dg solves
# `q = F(ratio · F⁻¹(q)) / 2`, iterated five times from `q = 1/2`; the estimate
# is `med / F⁻¹(q)`. For `dg == 1` a ratio below 5 has no useful fixed point and
# the naive `med / F⁻¹(1/2)` is returned (the reference's `min_ratio`). NaN
# when no residual lies below the loss width.
function _unbiased_median_variance(residuals_sq::AbstractVector{<:Real}, sq_loss_width::Real; dg::Integer = 1)
    dist = Chisq(dg)
    min_ratio = dg == 1 ? 5.0 : 0.0
    kept = Float64[r for r in residuals_sq if r < sq_loss_width]
    isempty(kept) && return NaN
    med = Statistics.median(kept)
    ratio = sq_loss_width / med
    q = 0.5
    σ²0 = med / Statistics.quantile(dist, 0.5)             # the naive estimate: the χ²_dg median
    ratio < min_ratio && return σ²0
    for _ in 1:5
        q = cdf(dist, ratio * Statistics.quantile(dist, q)) / 2
    end
    return med / Statistics.quantile(dist, q)
end

# The SIMFIT++ context: the initial loss width, and the earlier estimates behind it.
_simfitpp_context(p) = VgtkCore.read_context(p, (:initial_loss_width,), (; num_previous_est = 0), "SimFitPP")

function __solve(prob::EstimationProblem{<:LossWidth, <:NamedTuple{(:fit, :residuals_sq, :n)}}, cfg::SimFitPP;
                 rng::Random.AbstractRNG = Random.default_rng())
    (; initial_loss_width, num_previous_est) = _simfitpp_context(prob.p)
    (; fit, residuals_sq, n) = prob.f
    final, success, rule, rounds = _simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg, num_previous_est, rng)
    return FitResult(final, success ? ReturnCode.Success : ReturnCode.Failure, rule, SolveStats(; nsteps = rounds), NaN, n, nothing)
end

# The reference's loop: (final loss width, success, the rule that ended it, the rounds drawn).
function _simfitpp_loss_width(fit::F, residuals_sq::G, n::Integer, initial_loss_width::Real, cfg::SimFitPP,
                              num_previous_est::Integer, rng::Random.AbstractRNG) where {F, G}
    inner_num_success = 0
    previous_loss_width = Float64(initial_loss_width)
    seen = Set{Float64}()
    coverage_chisq_quantile = Statistics.quantile(Chisq(cfg.dg), cfg.alpha)
    loss_width = NaN
    rule = IterationCap
    rounds = 0
    for _ in 1:cfg.max_iter
        rounds += 1
        perm = Random.randperm(rng, n)
        train_end = floor(Int, n * cfg.train_fraction)      # the reference's int()
        train = @view perm[1:train_end]
        val   = @view perm[train_end + 1:end]
        model = fit(previous_loss_width, train)
        r2 = model === nothing ? nothing : residuals_sq(model, val)
        σ_sq = r2 === nothing ? NaN : _unbiased_median_variance(r2, previous_loss_width * previous_loss_width; dg = cfg.dg)
        proposed_loss_width = sqrt(coverage_chisq_quantile * σ_sq)
        (proposed_loss_width >= cfg.min_loss_width && proposed_loss_width <= cfg.max_loss_width) || continue    # NaN fails both
        loss_width = proposed_loss_width / (inner_num_success + 1) +
             previous_loss_width * inner_num_success / (inner_num_success + 1)
        abs(loss_width - previous_loss_width) < cfg.ftol && (rule = IncrementSize; break)
        key = round(loss_width; digits = 3)                  # the reference's 3-decimal key
        key in seen && (rule = NoImprovement; break)
        push!(seen, key)
        inner_num_success += 1
        previous_loss_width = loss_width
    end
    final = loss_width
    out_of_bounds = isnan(final) || final <= cfg.min_loss_width || final >= cfg.max_loss_width
    success = !out_of_bounds
    if !success
        final = Float64(initial_loss_width)
    else
        p = num_previous_est
        final = exp(log(final) / (p + 1) + p / (p + 1) * log(Float64(initial_loss_width)))
    end
    return final, success, rule, rounds
end

_msac_at(loss_width::Real) = MsacScoring{1, Float64}(Float64(loss_width))
function _msac_best_over(distances::AbstractMatrix, rows, loss_width::Real;
                         ws = scoring_workspace(_msac_at(loss_width), length(rows), 1),
                         scores::AbstractVector{Float64} = Vector{Float64}(undef, length(rows)))
    sc = _msac_at(loss_width)
    m = length(rows)
    return best_model_index(j -> score!(ws, sc, scores, view(distances, rows, j), m), size(distances, 2))
end

function __solve(prob::EstimationProblem{PoolSelection, <:AbstractMatrix}, cfg::SimFitPP;
                 rng::Random.AbstractRNG = Random.default_rng())
    (; initial_loss_width, num_previous_est) = _simfitpp_context(prob.p)
    D = prob.f
    n = size(D, 1)
    (n == 0 || size(D, 2) == 0) &&
        return FitResult(0, ReturnCode.Failure, NoRule, SolveStats(), NaN, 0, (; loss_width = Float64(initial_loss_width)))
    ws = scoring_workspace(_msac_at(1.0), n, 1)
    scores = Vector{Float64}(undef, n)
    fit(loss_width, train) = (j = _msac_best_over(D, train, loss_width; ws, scores); j == 0 ? nothing : j)
    residuals_sq(j, val) = square_distances!(Vector{Float64}(undef, length(val)),
                                             view(D, val, j), length(val))
    loss_width, success, rule, rounds = _simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg, num_previous_est, rng)
    return FitResult(_msac_best_over(D, 1:n, loss_width; ws, scores), success ? ReturnCode.Success : ReturnCode.Failure, rule,
                     SolveStats(; nsteps = rounds), NaN, n, (; loss_width))
end
__solve(prob::EstimationProblem{PoolSelection, <:PoolPair}, cfg::SimFitPP; rng::Random.AbstractRNG = Random.default_rng()) =
    __solve(EstimationProblem(PoolSelection, pixel_residuals(prob.f), prob.p), cfg; rng)
