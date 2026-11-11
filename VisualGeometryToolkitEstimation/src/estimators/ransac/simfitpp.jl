# =============================================================================
# SIMFIT++ — less biased noise-scale (loss-width) estimation for threshold-
# robust RANSAC (Edstedt, "Less Biased Noise Scale Estimation for Threshold-
# Robust RANSAC", CVPRW 2025). A step-for-step port of the reference
# implementation, github.com/Parskatt/simfitpp (src/simfitpp/simfitpp.py,
# scale_estimators.py, noise_models.py), kept in the reference's structure:
#
#   noise model        χ²_{d_g} on the squared residual (reference: χ²₁ for
#                      the scalar Sampson residual of F / E)
#   UnbiasedMed        `unbiased_median_variance`: the median of the squared
#                      residuals BELOW the current loss width, de-biased for
#                      that truncation by a five-step fixed-point iteration on
#                      the quantile the truncated median corresponds to, then
#                      divided by the χ² quantile
#   SIMFITPP.__call__  `simfitpp_loss_width`: `max_iter` rounds of a random
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
# a hypothesis pool (`simfitpp_select`: the MSAC-best pool hypothesis scored on
# the train split — the paper's fixed-pool protocol, no refinement).
#
# Moved from RANSACScoringDoneRight (src/simfitpp.jl) on 2026-11-11, unchanged
# but for the train-split MSAC sum, which is now VisualGeometryToolkitCore's one
# threshold `score!` (the same per-point quality, summed in the same order).
# =============================================================================

"""
    SimFitPP(; alpha = 0.99, max_iter = 4, ftol = 0.01, train_fraction = 0.5,
               min_loss_width = 0.25, max_loss_width = 8.0, dg = 1)

SIMFIT++ configuration ([Edstedt2025](@citet); the reference's constructor
defaults; the reference calls the loss width its threshold). `alpha` is the
inlier coverage the loss width targets (loss_width² = ppf_{χ²_dg}(alpha)·σ²);
`min_loss_width` and `max_loss_width` bound an acceptable loss width in the
residual unit of the residuals handed to the estimator (pixels in the
reference); `dg` is the residual codimension (the χ² degrees of freedom).
"""
Base.@kwdef struct SimFitPP
    alpha::Float64          = 0.99
    max_iter::Int           = 4
    ftol::Float64           = 0.01
    train_fraction::Float64 = 0.5
    min_loss_width::Float64 = 0.25
    max_loss_width::Float64 = 8.0
    dg::Int                 = 1
end

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

"""
    unbiased_median_variance(residuals_sq, sq_loss_width; dg = 1) -> σ²

The reference's `UnbiasedMed` ([Edstedt2025](@citet)): the noise variance from
the squared residuals below `sq_loss_width`, corrected for the truncation. With
`med` the median of the kept residuals and `ratio = sq_loss_width / med`, the
quantile `q` that the truncated median corresponds to under χ²_dg solves
`q = F(ratio · F⁻¹(q)) / 2`, iterated five times from `q = 1/2`; the estimate
is `med / F⁻¹(q)`. For `dg == 1` a ratio below 5 has no useful fixed point and
the naive `med / F⁻¹(1/2)` is returned (the reference's `min_ratio`). Returns
`NaN` when no residual lies below the loss width.

It returns σ², not σ, as its name says (VisualGeometryToolkitCore's `s2`
convention: a squared scale is named as one).
"""
function unbiased_median_variance(residuals_sq::AbstractVector{<:Real}, sq_loss_width::Real; dg::Integer = 1)
    dist = Chisq(dg)
    min_ratio = dg == 1 ? 5.0 : 0.0
    kept = Float64[r for r in residuals_sq if r < sq_loss_width]
    isempty(kept) && return NaN
    med = Statistics.median(kept)
    ratio = sq_loss_width / med
    q = 0.5
    σ²0 = med / Statistics.quantile(dist, q)
    ratio < min_ratio && return σ²0
    for _ in 1:5
        q = cdf(dist, ratio * Statistics.quantile(dist, q)) / 2
    end
    return med / Statistics.quantile(dist, q)
end

"""
    simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg = SimFitPP();
                        num_previous_est = 0, rng = Random.default_rng())
        -> (loss_width, success::Bool)

The reference's `SIMFITPP.__call__` ([Edstedt2025](@citet)) over `n`
correspondences. `fit(loss_width, train)` estimates the geometry from the
correspondences indexed by `train` at `loss_width` and returns a model handle
(or `nothing` on failure); `residuals_sq(model, val)` returns the squared
residuals of that model on the held-out indices `val` (or `nothing`).
`initial_loss_width` seeds the iteration and is returned unchanged when the
estimate fails (`success == false`); with `num_previous_est = p > 0` earlier
estimates behind `initial_loss_width`, the successful estimate is filtered as
`exp(log(loss_width)/(p+1) + p·log(initial_loss_width)/(p+1))`.
"""
function simfitpp_loss_width(fit, residuals_sq, n::Integer, initial_loss_width::Real,
                             cfg::SimFitPP = SimFitPP();
                             num_previous_est::Integer = 0,
                             rng::Random.AbstractRNG = Random.default_rng())
    inner_num_success = 0
    previous_loss_width = Float64(initial_loss_width)
    seen = Set{Float64}()
    coverage_chisq_quantile = Statistics.quantile(Chisq(cfg.dg), cfg.alpha)
    loss_width = NaN
    for _ in 1:cfg.max_iter
        perm = Random.randperm(rng, n)
        train_end = floor(Int, n * cfg.train_fraction)      # the reference's int()
        train = @view perm[1:train_end]
        val   = @view perm[train_end + 1:end]
        model = fit(previous_loss_width, train)
        r2 = model === nothing ? nothing : residuals_sq(model, val)
        σ_sq = r2 === nothing ? NaN : unbiased_median_variance(r2, previous_loss_width * previous_loss_width; dg = cfg.dg)
        proposed_loss_width = sqrt(coverage_chisq_quantile * σ_sq)
        (proposed_loss_width >= cfg.min_loss_width && proposed_loss_width <= cfg.max_loss_width) || continue    # NaN fails both
        loss_width = proposed_loss_width / (inner_num_success + 1) +
             previous_loss_width * inner_num_success / (inner_num_success + 1)
        abs(loss_width - previous_loss_width) < cfg.ftol && break
        key = round(loss_width; digits = 3)                  # the reference's 3-decimal key
        key in seen && break
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
    return final, success
end

# -----------------------------------------------------------------------------
# Hypothesis-pool adapter
# -----------------------------------------------------------------------------

# The MSAC-best pool model over the rows `rows` of the n×M residual matrix
# `distances` at `loss_width`: VisualGeometryToolkitCore's pool argmax
# (`Pools.best_model_index`, 0 for an empty pool) of its threshold `score!`
# (MSAC at that width) on each column's rows — no minimal sample (the
# reference's fixed-pool protocol). `ws` and `scores` are that scorer's
# workspace and per-point scratch, sized for at least `length(rows)` points.
# The scorer is built by its type (`MsacScoring{1, Float64}`), not by the
# `threshold` keyword, whose `Union{Nothing, Real}` would make it a union.
_msac_at(loss_width::Real) = MsacScoring{1, Float64}(Float64(loss_width))
function _msac_best_over(distances::AbstractMatrix, rows, loss_width::Real;
                         ws = scoring_workspace(_msac_at(loss_width), length(rows), 1),
                         scores::AbstractVector{Float64} = Vector{Float64}(undef, length(rows)))
    sc = _msac_at(loss_width)
    m = length(rows)
    return best_model_index(j -> score!(ws, sc, scores, view(distances, rows, j), m), size(distances, 2))
end

"""
    simfitpp_select(residuals::AbstractMatrix, initial_loss_width, cfg = SimFitPP();
                    num_previous_est = 0, rng = default_rng())
        -> (best_idx, loss_width, success)
    simfitpp_select(pair::PoolPair, initial_loss_width_px, cfg = SimFitPP(); ...)

SIMFIT++ ([Edstedt2025](@citet)) on a fixed hypothesis pool: `residuals` is the
pool's n×M residual matrix (points × models) in the unit `initial_loss_width`
and the bounds of `cfg` are expressed in. The geometry "estimator" is the
MSAC-best pool hypothesis on the train split at the current loss width, scored
by its validation-split residuals; the returned index is the MSAC argmax over
ALL points at the final loss width (`0` for an empty pool).

The pair method is PIXELS in, pixels out: it runs on VisualGeometryToolkitCore's
`Pools.pixel_residuals(pair)`, the pool models' residuals in pixels, so
`initial_loss_width_px`, the bounds of `cfg` (the reference's pixel defaults,
0.25–8 px) and the returned loss width are all pixels. To work in the pool
frame's unit instead, call the matrix method on the pair's stored residual
matrix with bounds in that unit ([`with_residual_unit`](@ref)).
"""
function simfitpp_select(D::AbstractMatrix, initial_loss_width::Real, cfg::SimFitPP = SimFitPP();
                         num_previous_est::Integer = 0, rng::Random.AbstractRNG = Random.default_rng())
    n = size(D, 1)
    (n == 0 || size(D, 2) == 0) && return (0, Float64(initial_loss_width), false)
    ws = scoring_workspace(_msac_at(1.0), n, 1)
    scores = Vector{Float64}(undef, n)
    fit(loss_width, train) = (j = _msac_best_over(D, train, loss_width; ws, scores); j == 0 ? nothing : j)
    residuals_sq(j, val) = square_distances!(Vector{Float64}(undef, length(val)),
                                             view(D, val, j), length(val))
    loss_width, success = simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg;
                                              num_previous_est, rng)
    return (_msac_best_over(D, 1:n, loss_width; ws, scores), loss_width, success)
end
simfitpp_select(pair::PoolPair, initial_loss_width_px::Real, cfg::SimFitPP = SimFitPP();
                num_previous_est::Integer = 0, rng::Random.AbstractRNG = Random.default_rng()) =
    simfitpp_select(pixel_residuals(pair), initial_loss_width_px, cfg; num_previous_est, rng)
