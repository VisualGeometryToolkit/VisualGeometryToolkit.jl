# =============================================================================
# fit_residual_mixture — the Student-t + uniform mixture of a held model's
# residuals: the scorer's partition seeds it, EM fits it, MAP classifies
# =============================================================================
#
# Regression and classification apart. Whatever estimated the model — a robust
# bundle under a loss whose weights only guide its optimizer — its residuals are
# FIXED here, and the inlier/outlier decision is a fit of the noise model to
# them: the consensus set of the scorer (its posterior seeded at that set),
# then EM of the mixture with the model held
# (`PosteriorIrlsMethod` under `KeepModel`, VGC's unchanged
# `block_coordinate_descent!`), then each residual's MAP label (VGC
# `is_map_inlier`).
# =============================================================================

"""
    MixtureEmConfig(; model_update, nuisance, stop, max_iter)

The EM of the Student-t + uniform mixture: the model update
([`WeightedRefine`](@ref) or [`KeepModel`](@ref)`(tol)`, whose `tol` is the
relative mixture-NLL decrease below which the EM has converged), the nuisance
parts' update policies `nuisance = (; scale, inlier_prior, nu)` (each
`Estimated()`, or held: VisualGeometryToolkitCore's `FixedS2(s2)`,
`FixedPrior(π)` and `FixedNu(ν)`, and for ν also `ConsensusDof()` — held at the consensus seed's
posterior-predictive dof), the stop rule `stop` ([`NoImprovementStop`](@ref): the
incumbent's cost, returning the incumbent; [`ParameterChangeStop`](@ref)`(rtol)`:
the relative change of the parameters, returning the last iterate) and the
iteration cap. Every default reads the `[mixture_em]` table of
`configs/default.toml` (every key is required there: none is defaulted in code);
[`mixture_em_from_toml`](@ref) loads a caller's table and [`config_dict`](@ref)
writes one back.
"""
struct MixtureEmConfig{U, S, P, N, T}
    model_update::U
    nuisance::@NamedTuple{scale::S, inlier_prior::P, nu::N}
    stop::T
    max_iter::Int
    function MixtureEmConfig(model_update::U, nuisance::@NamedTuple{scale::S, inlier_prior::P, nu::N},
                             stop::T, max_iter::Integer) where {U, S, P, N, T}
        max_iter >= 1 || throw(ArgumentError("MixtureEmConfig: max_iter must be ≥ 1, got $max_iter"))
        return new{U, S, P, N, T}(model_update, nuisance, stop, Int(max_iter))
    end
end
MixtureEmConfig(model_update, nuisance::NamedTuple, stop, max_iter::Integer) =
    MixtureEmConfig(model_update, (; scale = nuisance.scale, inlier_prior = nuisance.inlier_prior, nu = nuisance.nu),
                    stop, max_iter)
MixtureEmConfig(;
    model_update = _model_update_from_toml(_dflt("mixture_em", "model_update"), "mixture_em.model_update"),
    nuisance = _nuisance_from_toml(_dflt("mixture_em", "nuisance"), "mixture_em.nuisance"),
    stop = _stop_from_toml(_dflt("mixture_em", "stop"), "mixture_em.stop"),
    max_iter::Integer = _dflt("mixture_em", "max_iter"),
) = MixtureEmConfig(model_update, nuisance, stop, max_iter)

# The EM's block-coordinate method on the refit workspace `refine_ws` (`nothing`
# when the model is held, or when the problem carries its own refit).
PosteriorIrlsMethod(refine_ws, config::MixtureEmConfig) = PosteriorIrlsMethod(refine_ws, config.model_update, config.stop)

# A held model's distances as a problem: the residual block of the iteration
# re-reads them through the protocol's distance hook (`≥ 0`), and nothing
# refines them (there is no `refine!`).
struct _FixedDistances
    distances::Vector{Float64}
end
data_size(p::_FixedDistances) = length(p.distances)
mahalanobis_distances!(d::AbstractVector, p::_FixedDistances, _held) = copyto!(d, p.distances)

# The robust-estimation contract's result of an EM fit (§2, §12): its per-datum
# weights are the E-step's posterior inlier probabilities (`PosteriorWeights`; a
# posterior below eps stored as an exact 0, and a datum the fit did not take a
# weight of 0), its scale the E-step's σ̂ (`ScaleEstimated`), `n` the support
# (the data of positive weight) and `rms` the weight-weighted root mean square —
# the find_* results' own rule, `_posterior_support_rms!`.

"""
    ResidualMixtureMetadata

The `metadata` of a [`fit_residual_mixture`](@ref) result: the mixture's
negative log-likelihood `nll` of the distances at the fitted noise model (the
quantity the EM lowered), each distance's posterior inlier probability
`weights` ([`PosteriorWeights`](@extref VisualGeometryToolkitCore.PosteriorWeights):
`inlier_mask(result)` is the MAP rule, posterior > ½; a posterior below
`eps()` is an exact 0) and the E-step's scale `scale` = ŝ (the distances' unit,
a LINEAR scale; `scale_source` `ScaleEstimated`), read by `weights(result)`
and `noise_scale(result)`.
"""
struct ResidualMixtureMetadata
    nll::Float64
    weights::Vector{Float64}
    scale::Float64
    scale_source::ScaleSource
end
weight_kind(::Type{<:ResidualMixtureMetadata}) = PosteriorWeights()

"""
    fit_residual_mixture(distances; weighting, model_dof, config = MixtureEmConfig())
        -> Union{FitResult{PosteriorWeightState, ResidualMixtureMetadata}, Nothing}

Fit the Student-t + uniform mixture to a held model's `distances` (`≥ 0`,
unsquared: the Mahalanobis distances, the residual NORMS of a codimension-2
observation — what `mahalanobis_distances!` fills) and classify each one. A
negative entry is a signed residual, not a distance, and is an
`ArgumentError` (supply `abs.(r)`).

`weighting` is the EM's weighting, the Student-t + uniform E-step: a scorer with
a uniform outlier model (`outlier_halfwidth`) whose per-point scores are the
squared residuals, as a sort-and-sweep scorer's are (VisualGeometryToolkitCore's
`StudentTUniformScoring`). A fit takes no `scoring`: the former positional
`scoring` argument and a `scoring` keyword are an `ArgumentError` naming
`weighting` (the robust-estimation contract, since 0.9).

1. The weighting takes their consensus set — `score!` at its codimension and
   scale prior — and the set's Student-t (ŝ² at the residual dof
   `(k* − model_dof)·d_g`) seeds the posterior (`PosteriorWeightState` on the
   weighting's `consensus_order`).
2. EM with the model held — `config.model_update` must be [`KeepModel`](@ref) and
   `config.stop` [`NoImprovementStop`](@ref) (a held model's parameters never move) —
   under VisualGeometryToolkitCore's `block_coordinate_descent!`: each iteration the
   nuisance parts (`config.nuisance`) then the E-step, until `config.stop` or
   `config.max_iter`.
3. Each residual's MAP label: inlier iff `is_map_inlier` of its posterior.

`model_dof` counts the parameters the held model spent per correspondence
(a bundle's free parameters over the codimension, rounded up). Returns
`nothing` when the partition is too thin to carry a scale.

The result: `estimate` the fitted mixture, a `PosteriorWeightState` — its
`s_hat_sq`, `student_t_dof` and `inlier_prior`, and per residual its posterior
`inlier_weight(estimate)[i]`, whose MAP label is VisualGeometryToolkitCore's
`is_inlier(estimate, i)`. The contract's record (`metadata`, a
[`ResidualMixtureMetadata`](@ref)): `weights(result)` the posteriors
(`PosteriorWeights`, so `inlier_mask(result)` is the MAP labels) and
`noise_scale(result)` = ŝ, estimated (`ScaleEstimated`). `n` is the support, the
distances of positive weight, and `rms` the weight-weighted root mean square
`√(Σwᵢrᵢ² / Σwᵢ)` (the MAP inliers' count is `count(inlier_mask(result))`);
`retcode` and `iterations` the EM's.
"""
function fit_residual_mixture(distances::AbstractVector{<:Real}; weighting::AbstractScoring,
                              model_dof::Int, config::MixtureEmConfig = MixtureEmConfig(),
                              scoring = RetiredKeyword())   # refused, with the way to say it
    scoring isa RetiredKeyword || throw(_scoring_is_weighting("fit_residual_mixture"))
    config.model_update isa KeepModel || throw(ArgumentError(
        "fit_residual_mixture: the distances are a held model's, so the model update must be " *
            "KeepModel, got $(config.model_update)"))
    # a held model never moves: a parameter-change stop would end the EM at iteration 1, the nuisance unconverged
    config.stop isa NoImprovementStop || throw(ArgumentError(
        "fit_residual_mixture: the distances are a held model's, whose parameters never move, so the EM stops on its " *
            "cost: `stop` must be NoImprovementStop (`{ type = \"no_improvement\" }`), got $(config.stop)"))
    i = findfirst(<(0), distances)
    i === nothing || throw(ArgumentError(
        "fit_residual_mixture: distances[$i] = $(distances[i]) is negative — a signed residual, not a distance; " *
            "supply its magnitude, `abs.(r)`"))
    r = collect(Float64, distances)
    n = length(r)
    rws = _consensus_seeded_posterior(r, n, weighting; model_dof)
    rws === nothing && return nothing
    lh = PosteriorLikelihood(rws, n; scale_update = config.nuisance.scale,
                             prior_update = config.nuisance.inlier_prior, nu_update = config.nuisance.nu)
    # The held "model" is the residual vector itself: the incumbent the loop
    # tracks, never moved (`KeepModel`).
    state = PosteriorIrlsState(rws, copy(r), copy(r), r, mixture_nll(rws, r, n))
    res = block_coordinate_descent!(state, lh, PosteriorIrlsMethod(nothing, config),
                                    _FixedDistances(r), r; max_iter = config.max_iter)
    w = collect(Float64, view(inlier_weight(rws), 1:n))
    support, wrms = _posterior_support_rms!(w, (abs2(x) for x in r))
    return FitResult(rws, retcode(res), stop_rule(res), iterations(res), wrms, support,
                     ResidualMixtureMetadata(state.current_cost, w, sqrt(s_hat_sq(rws)), ScaleEstimated))
end

# The former positional scorer: refused, naming the keyword.
fit_residual_mixture(::AbstractVector{<:Real}, ::AbstractScoring; kw...) = throw(_scoring_is_weighting("fit_residual_mixture"))

# A fit takes no `scoring` (the robust-estimation contract): its Student-t + uniform E-step is its `weighting`.
_scoring_is_weighting(f::AbstractString) = ArgumentError(
    "$f: `scoring` is no keyword (nor argument) of a fit since 0.9 — the Student-t + uniform E-step is the fit's `weighting` " *
        "(`weighting = StudentTUniformScoring(…)`, or a `weighting` table read by `scoring_from_toml`)")

# The posterior seeded from the scorer's consensus set of residuals `r[1:n]`:
# `score!` (the squared residuals into the state's own buffer), then the set's
# Student-t under the scorer's prior (the pre-sorted `PosteriorWeightState`).
# `nothing` when the set is too thin to carry a scale (k* ≤ model_dof).
function _consensus_seeded_posterior(r::AbstractVector, n::Int, scoring; model_dof::Int)
    d_g = codimension(scoring)
    W = scoring_workspace(scoring, n, d_g)
    sq_r = Vector{Float64}(undef, n)
    score!(W, scoring, sq_r, r, n)
    k = inlier_count(scoring, W)
    k > model_dof || return nothing
    return PosteriorWeightState(
        sq_r, n, model_dof, k; order = consensus_order(scoring, W), prior = scale_prior(scoring),
        w = Vector{Float64}(undef, n), inlier_posterior = Vector{Float64}(undef, n), sq_r,
        log_two_outlier_halfwidth = log(2 * required_outlier_halfwidth(scoring)), codimension = d_g
    )
end
