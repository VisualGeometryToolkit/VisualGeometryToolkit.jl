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
    MixtureEmConfig(; model_update, nuisance, max_iter)

The EM of the Student-t + uniform mixture: the model update
([`WeightedRefine`](@ref) or [`KeepModel`](@ref)`(tol)`, whose `tol` is the
relative mixture-NLL decrease below which the EM has converged), the nuisance
parts' update policies `nuisance = (; scale, inlier_prior, nu)` (each
`Estimated()`, or held: VisualGeometryCore's `FixedS2(s2)`,
`FixedPrior(π)` and `FixedNu(ν)`, and for ν also `ConsensusDof()` — held at the consensus seed's
posterior-predictive dof) and the iteration cap. Every default reads the `[mixture_em]`
table of `configs/default.toml`; [`mixture_em_from_toml`](@ref) loads a
caller's table and [`config_dict`](@ref) writes one back.
"""
struct MixtureEmConfig{U, S, P, N}
    model_update::U
    nuisance::@NamedTuple{scale::S, inlier_prior::P, nu::N}
    max_iter::Int
    function MixtureEmConfig(model_update::U, nuisance::@NamedTuple{scale::S, inlier_prior::P, nu::N},
                             max_iter::Integer) where {U, S, P, N}
        max_iter >= 1 || throw(ArgumentError("MixtureEmConfig: max_iter must be ≥ 1, got $max_iter"))
        return new{U, S, P, N}(model_update, nuisance, Int(max_iter))
    end
end
MixtureEmConfig(model_update, nuisance::NamedTuple, max_iter::Integer) =
    MixtureEmConfig(model_update, (; scale = nuisance.scale, inlier_prior = nuisance.inlier_prior, nu = nuisance.nu),
                    max_iter)
MixtureEmConfig(;
    model_update = _model_update_from_toml(_dflt("mixture_em", "model_update"), "mixture_em.model_update"),
    nuisance = _nuisance_from_toml(_dflt("mixture_em", "nuisance"), "mixture_em.nuisance"),
    max_iter::Integer = _dflt("mixture_em", "max_iter"),
) = MixtureEmConfig(model_update, nuisance, max_iter)

# A held model's residuals as a problem: the residual block of the iteration
# re-reads them, and nothing refines them (there is no `refine!`).
struct _FixedResiduals
    residuals::Vector{Float64}
end
data_size(p::_FixedResiduals) = length(p.residuals)
residuals!(r::AbstractVector, p::_FixedResiduals, _held) = copyto!(r, p.residuals)

"""
    ResidualMixtureMetadata

The `metadata` of a [`fit_residual_mixture`](@ref) result: the mixture's
negative log-likelihood `nll` of the residuals at the fitted noise model (the
quantity the EM lowered).
"""
struct ResidualMixtureMetadata
    nll::Float64
end

"""
    fit_residual_mixture(residuals, scoring; model_dof,
                         config = MixtureEmConfig())
        -> Union{FitResult{PosteriorWeightState, ResidualMixtureMetadata}, Nothing}

Fit the Student-t + uniform mixture to a held model's `residuals` (unsquared;
the residual NORMS of a codimension-2 measurement) and classify each one.

1. The scorer takes their consensus set — `score!` of `scoring` (any scorer with
   a uniform outlier model, `outlier_halfwidth`, whose per-point scores are the
   squared residuals, as a sort-and-sweep scorer's are) at its codimension and
   scale prior — and the set's Student-t (ŝ² at the residual dof
   `(k* − model_dof)·d_g`) seeds the posterior (`PosteriorWeightState` on the
   scorer's `consensus_order`).
2. EM with the model held — `config.model_update` must be [`KeepModel`](@ref) —
   under VisualGeometryCore's `block_coordinate_descent!`: each iteration the
   nuisance parts (`config.nuisance`) then the E-step, until the mixture NLL
   stops falling (`tol`) or `config.max_iter`.
3. Each residual's MAP label: inlier iff `is_map_inlier` of its posterior.

`model_dof` counts the parameters the held model spent per correspondence
(a bundle's free parameters over the codimension, rounded up). Returns
`nothing` when the partition is too thin to carry a scale.

The result: `estimate` the fitted mixture, a `PosteriorWeightState` — its
`s_hat_sq`, `student_t_dof` and `inlier_prior`, and per residual its posterior
`inlier_weight(estimate)[i]`, whose MAP label is VisualGeometryCore's
`is_inlier(estimate, i)` (`is_map_inlier` of the posterior: the state is a
`NormalizedWeightPolicy`; `inlier_mask(estimate)` labels them all); `rms` and
`n` the root-mean-square residual over the MAP inliers and their count, so
`rms` is the reprojection error of record; `retcode` and `iterations` the EM's;
`metadata` a [`ResidualMixtureMetadata`](@ref).
"""
function fit_residual_mixture(residuals::AbstractVector{<:Real}, scoring::AbstractScoring;
                              model_dof::Int, config::MixtureEmConfig = MixtureEmConfig())
    config.model_update isa KeepModel || throw(ArgumentError(
        "fit_residual_mixture: the residuals are a held model's, so the model update must be " *
            "KeepModel, got $(config.model_update)"))
    r = collect(Float64, residuals)
    n = length(r)
    rws = _consensus_seeded_posterior(r, n, scoring; model_dof)
    rws === nothing && return nothing
    lh = PosteriorLikelihood(rws, n; scale_update = config.nuisance.scale,
                             prior_update = config.nuisance.inlier_prior, nu_update = config.nuisance.nu)
    # The held "model" is the residual vector itself: the incumbent the loop
    # tracks, never moved (`KeepModel`).
    state = PosteriorIrlsState(rws, copy(r), copy(r), r, mixture_nll(rws, r, n))
    res = block_coordinate_descent!(state, lh, PosteriorIrlsMethod(nothing, config.model_update),
                                    _FixedResiduals(r), r; max_iter = config.max_iter)
    ss, k = 0.0, 0
    for i in 1:n
        is_inlier(rws, i) || continue
        ss += r[i]^2
        k += 1
    end
    return FitResult(rws, retcode(res), iterations(res), k > 0 ? sqrt(ss / k) : NaN, k,
                     ResidualMixtureMetadata(state.current_cost))
end

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
