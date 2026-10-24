# =============================================================================
# The posterior-weight state of the Student-t + uniform mixture — the EM state
# every posterior-weighted estimator of the stack runs (moved from
# RANSACScoringDoneRight's posterior_weighting.jl): its seeds, its M-step
# blocks and their update policies, the ECM cycle, and the partition mixture's
# weights.
# =============================================================================
#
# `update_scale!` here is the state's OWN 2-arg scale block, on
# `PosteriorWeightState` — NOT the 1-arg policy verb `update_nuisance!` that
# `block_coordinate_descent!` drives. Across the downstream policies that verb
# moves ŝ², ν, π, α or nothing at all; this function does the narrow thing its
# name claims: ŝ² from the weighted RSS, one block; ν is the separate block
# `update_nu!`. VisualGeometryToolkitEstimation's `PosteriorLikelihood` is the adapter that
# IS a policy, and its `update_nuisance!` calls the parts in order.
# =============================================================================

"""
    PosteriorWeightState

Mutable state for computing robust IRLS weights from a Student-t inlier +
uniform outlier mixture model. Holds the sufficient statistics (ν, ŝ²,
log_two_outlier_halfwidth, the inlier prior π) and per-point buffers
(w, inlier_posterior, r²).

The `Posterior` in the name is the posterior over the *inlier* indicator: the
weights `w`/`inlier_posterior` are `P(inlier | r)`. The noise scale `σ²` also
has a posterior, but it is *marginalized* away, so it is held here only as the
scalar `ν`/`ŝ²`, never as a posterior.

The per-point buffers are supplied by the caller (typically a workspace's);
construction-time allocation is performed only by the `n`-constructor and
[`median_seeded_posterior`](@ref).
"""
# <: NormalizedWeightPolicy: the Student-t + uniform mixture IS a proper
# normalized probability model — the state itself is the likelihood-grade
# weight policy.
mutable struct PosteriorWeightState <: NormalizedWeightPolicy
    w::Vector{Float64}
    inlier_posterior::Vector{Float64}
    sq_residuals::Vector{Float64}
    ν::Float64
    s_hat_sq::Float64
    log_two_outlier_halfwidth::Float64
    d_g::Int  # constraint codimension (1 for scalar, 2 for 2D reprojection)
    prior::IGPrior  # IG scale prior (IGPrior(0,0) ⇒ Jeffreys), the seed's
    inlier_prior::Float64  # the mixing proportion: the prior probability that a
    # residual is an inlier. 1/2 is the fixed prior this
    # field generalises; only then is the E-step unchanged.
end

# 8-argument form: the inlier prior stays at 1/2.
PosteriorWeightState(w, inlier_posterior, sq_r, ν, s_hat_sq, log_two_outlier_halfwidth, d_g, prior) =
    PosteriorWeightState(w, inlier_posterior, sq_r, ν, s_hat_sq, log_two_outlier_halfwidth, d_g, prior, 0.5)

"""
    StudentTMixture(pws::PosteriorWeightState)

The mixture the state's E-step evaluates: its current `ν`, `ŝ²`, outlier
support, codimension and inlier prior.
"""
StudentTMixture(pws::PosteriorWeightState) =
    StudentTMixture(pws.ν, pws.s_hat_sq, pws.log_two_outlier_halfwidth, pws.d_g, pws.inlier_prior)

"""
    sq_residuals(pws::PosteriorWeightState) -> Vector{Float64}

The squared residuals the E-step weighs — the buffer `update_scale!` and
`update_weights!` read, refilled by the caller between iterations.
"""
sq_residuals(pws::PosteriorWeightState) = pws.sq_residuals

"""
    s_hat_sq(pws::PosteriorWeightState) -> Float64

The current squared scale estimate `ŝ²` of the Student-t inlier density (a
SQUARE; see [`fit_scale`](@ref)).
"""
s_hat_sq(pws::PosteriorWeightState) = pws.s_hat_sq

"""
    student_t_dof(pws::PosteriorWeightState) -> Float64

The Student-t degrees of freedom `ν` of the inlier density — not the model's
DOF (`minimal_sample_size`), and not Distributions' `dof`.
"""
student_t_dof(pws::PosteriorWeightState) = pws.ν

"""
    inlier_prior(pws::PosteriorWeightState) -> Float64

The mixing proportion `π` the E-step weighs with: the prior probability that a
residual is an inlier (1/2 unless estimated or held elsewhere).
"""
inlier_prior(pws::PosteriorWeightState) = pws.inlier_prior

"""
    PosteriorWeightState(sq_residuals, n, model_dof, k; order, prior, w, inlier_posterior, sq_r,
                         log_two_outlier_halfwidth, codimension = 1)

The state seeded from a hard partition: the first `k` entries of `order` (a
scorer's [`consensus_order`](@ref) over these same squared residuals), with no
fresh sort. The seed is the partition's Student-t under `prior`
([`partition_student_t`](@ref)), so it equals the scoring predictive over
`order[1:k]`; the E-step runs once.

Precondition: `order[1:k]` is a partition of `sq_residuals[1:n]` and
`k > model_dof`.
"""
function PosteriorWeightState(sq_residuals::AbstractVector{Float64}, n::Int, model_dof::Int,
                              k_prev::Int;
                              order::AbstractVector{Int},
                              prior::IGPrior,
                              w::Vector{Float64},
                              inlier_posterior::Vector{Float64},
                              sq_r::Vector{Float64},
                              log_two_outlier_halfwidth::Float64,
                              codimension::Int = 1)
    pws = PosteriorWeightState(w, inlier_posterior, sq_r, NaN, NaN, log_two_outlier_halfwidth, codimension, prior)
    return seed_posterior!(pws, sq_residuals, n, model_dof, k_prev; order, prior, log_two_outlier_halfwidth, codimension)
end

"""
    seed_posterior!(pws, sq_residuals, n, model_dof, k; order, prior, log_two_outlier_halfwidth,
                    codimension = 1) -> pws

The partition seeding of [`PosteriorWeightState`](@ref)`(sq_residuals, n, model_dof, k; …)`
IN PLACE, on `pws`'s own buffers: the same squared residuals, partition
Student-t and first E-step, the inlier prior back at 1/2. What a caller that
keeps one state across calls (a local optimizer's workspace) re-seeds with.
"""
function seed_posterior!(pws::PosteriorWeightState, sq_residuals::AbstractVector{Float64}, n::Int,
                         model_dof::Int, k_prev::Int;
                         order::AbstractVector{Int}, prior::IGPrior,
                         log_two_outlier_halfwidth::Float64, codimension::Int = 1)
    w, inlier_posterior, sq_r = pws.w, pws.inlier_posterior, pws.sq_residuals
    _check_state_buffers(w, inlier_posterior, sq_r, n)
    _copy_sq_residuals!(sq_r, sq_residuals, n)
    ν, s_hat_sq = partition_student_t(prior, _partition_rss(sq_r, order, k_prev), k_prev;
                                      model_dof, codimension)
    # Initial soft weights from the seeded (ν, ŝ²)
    soft_posterior_weights!(w, inlier_posterior, sq_r, ν, s_hat_sq, log_two_outlier_halfwidth, n, codimension)
    pws.ν = ν; pws.s_hat_sq = s_hat_sq; pws.log_two_outlier_halfwidth = log_two_outlier_halfwidth
    pws.d_g = codimension; pws.prior = prior; pws.inlier_prior = 0.5
    return pws
end

# The caller-owned buffers of a seeded state hold at least `n` entries.
function _check_state_buffers(w, inlier_posterior, sq_r, n::Int)
    length(w) >= n || throw(ArgumentError("w buffer too small: $(length(w)) < $n"))
    length(inlier_posterior) >= n || throw(
        ArgumentError(
            "inlier_posterior buffer too small: $(length(inlier_posterior)) < $n"
        )
    )
    length(sq_r) >= n || throw(ArgumentError("sq_r buffer too small: $(length(sq_r)) < $n"))
    return nothing
end

# Seed the state's own buffer from the caller's squared residuals (a view the
# state does not own); nothing to do when they are the same buffer.
function _copy_sq_residuals!(sq_r, sq_residuals, n::Int)
    sq_r === sq_residuals && return sq_r
    @inbounds for i in 1:n
        sq_r[i] = sq_residuals[i]
    end
    return sq_r
end

"""
    PosteriorWeightState(n; ν, s_hat_sq, log_two_outlier_halfwidth, codimension = 1, inlier_prior = 1/2,
                         prior = JeffreysPrior())

A state with FRESH length-`n` buffers at the given mixture, for a caller that
fills the residuals itself and runs the E-step (e.g. [`mixture_ecm!`](@ref))
before reading any weight. The buffers are uninitialised; `ŝ²` is floored at
eps and the inlier prior clamped to `[1e-3, 1 − 1e-3]` (the one clamp,
`update_inlier_prior!`'s).
"""
function PosteriorWeightState(n::Int; ν::Real, s_hat_sq::Real, log_two_outlier_halfwidth::Real,
                              codimension::Int = 1, inlier_prior::Real = 0.5,
                              prior::IGPrior = JeffreysPrior())
    return PosteriorWeightState(
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Float64(ν), max(Float64(s_hat_sq), eps()), Float64(log_two_outlier_halfwidth), codimension, prior,
        _clamp_inlier_prior(inlier_prior)
    )
end

"""
    reset_posterior!(pws; ν, s_hat_sq, log_two_outlier_halfwidth, codimension = 1,
                     inlier_prior = 0.5, prior = JeffreysPrior()) -> pws

[`PosteriorWeightState`](@ref)`(n; …)` IN PLACE: the same parameters (ŝ² floored
at eps, π clamped) on `pws`'s existing buffers, whose contents are left as
they are — that constructor's buffers are uninitialized, so nothing may read
them before writing.
"""
function reset_posterior!(pws::PosteriorWeightState; ν::Real, s_hat_sq::Real, log_two_outlier_halfwidth::Real,
                          codimension::Int = 1, inlier_prior::Real = 0.5, prior::IGPrior = JeffreysPrior())
    pws.ν = Float64(ν); pws.s_hat_sq = max(Float64(s_hat_sq), eps())
    pws.log_two_outlier_halfwidth = Float64(log_two_outlier_halfwidth); pws.d_g = codimension
    pws.prior = prior; pws.inlier_prior = _clamp_inlier_prior(inlier_prior)
    return pws
end

"""
    median_seeded_posterior(sq, n; outlier_halfwidth, codimension = 2, ν = 1e4,
                            prior = IGPrior()) -> PosteriorWeightState

A `PosteriorWeightState` seeded from the MEDIAN residual, for a caller with no
partition to seed from. Every point starts an inlier candidate at the scale
`ŝ² = median(r²)/median(χ²_d)` (floored at eps), with `ν` large (a
near-Gaussian start) and the E-step run once. Allocates its own buffers
(`sq[1:n]` is copied). `codimension` is 1 or 2; `median(χ²_d)` is
[`chisq_median`](@ref).
"""
function median_seeded_posterior(sq::AbstractVector{Float64}, n::Int;
                                 outlier_halfwidth::Real, codimension::Int = 2,
                                 ν::Real = 1.0e4, prior::IGPrior = IGPrior(0.0, 0.0))
    codimension in (1, 2) || throw(ArgumentError("codimension must be 1 or 2"))
    m_d = chisq_median(codimension)
    pws = PosteriorWeightState(
        n; ν, s_hat_sq = Statistics.median(view(sq, 1:n)) / m_d,
        log_two_outlier_halfwidth = log(2 * Float64(outlier_halfwidth)), codimension, prior
    )
    _copy_sq_residuals!(pws.sq_residuals, sq, n)
    update_weights!(pws, n)
    return pws
end

# Σ_{i ≤ n} P(inlier | rᵢ) of the E-step — the soft inlier count both M-step
# blocks that read it (`update_scale!`'s dof, `update_inlier_prior!`'s mean)
# share. Float64, left to right from 0.0.
@inline function _sum_inlier_posterior(pws, n::Int)
    s = 0.0
    @inbounds for i in 1:n
        s += pws.inlier_posterior[i]
    end
    return s
end

"""
    update_scale!(pws, n)

M-step, scale block: overwrite ŝ² from the E-step's weights and the current
squared residuals, and nothing else. Call after the residual block fills
`sq_residuals(pws)`; the ν block is [`update_nu!`](@ref), called after this.

ŝ² is the ECM step of the d_g-variate Student-t with scale ŝ²·I:

    ŝ²_new = Σ πᵢ uᵢ rᵢ² / (d_g · Σ πᵢ),     uᵢ = (ν + d_g)/(ν + rᵢ²/ŝ²),

with πᵢ the inlier posteriors and rᵢ² squared NORMS. The E-step's weight is
precision-scaled, `wᵢ = πᵢ·(ν + d_g)/(νŝ² + rᵢ²) = πᵢuᵢ/ŝ²`, so the numerator is
`ŝ²·Σwᵢrᵢ²` at the ŝ² that E-step used, and the inverse-gamma prior enters as
`fit_scale(prior, ŝ²·Σwᵢrᵢ², d_g·Σπᵢ)`. Returns ŝ².
"""
function update_scale!(pws::PosteriorWeightState, n::Int)
    sum_wr2 = weighted_rss(pws.w, pws.sq_residuals, n)
    return pws.s_hat_sq = fit_scale(
        pws.prior, pws.s_hat_sq * sum_wr2,
        pws.d_g * _sum_inlier_posterior(pws, n)
    )
end

"""
    update_nu!(pws, n)

M-step, ν block: overwrite the Student-t dof ν with [`next_nu`](@ref) (the
safeguarded Newton MLE) from the E-step's inlier posteriors, the current
squared residuals and the CURRENT ŝ², and nothing else. Call after
[`update_scale!`](@ref), which it reads. Returns ν.
"""
function update_nu!(pws::PosteriorWeightState, n::Int)
    return pws.ν = next_nu(
        pws.ν, pws.sq_residuals, pws.s_hat_sq, pws.inlier_posterior, n;
        d_g = pws.d_g
    )
end

# The mixing proportion is kept strictly inside (0, 1): the E-step reads it
# through log((1 − π)/π), which is infinite at either end. The ONE clamp: the
# M-step below and the allocating constructor both apply it.
const _INLIER_PRIOR_BOUNDS = (1.0e-3, 1 - 1.0e-3)
@inline _clamp_inlier_prior(π) = clamp(Float64(π), _INLIER_PRIOR_BOUNDS...)

"""
    update_inlier_prior!(pws, n)

M-step, inlier-prior block: overwrite the mixing proportion `π` with the mean
inlier posterior of the E-step, and nothing else. The mean `Σπᵢ/n` is clamped to
`[1e-3, 1 − 1e-3]` (the E-step reads it through `log((1 − π)/π)`). Call after
[`update_weights!`](@ref) filled the posteriors it reads. Returns π.
"""
function update_inlier_prior!(pws::PosteriorWeightState, n::Int)
    return pws.inlier_prior = _clamp_inlier_prior(_sum_inlier_posterior(pws, n) / n)
end

# =============================================================================
# The nuisance parts' update policies — estimated, or held at a value
# =============================================================================
#
# The nuisance block of the mixture has three parts — the scale ŝ², the inlier
# prior π and the Student-t dof ν — and each part's M-step either ESTIMATES it
# or HOLDS it at a value. The policy is a value, and each part's update
# dispatches on it: `update_scale!(pws, n, policy)`, `update_inlier_prior!`,
# `update_nu!`. Every method writes its one part and nothing else.

"""
    Estimated()

The update policy of a nuisance part that is ESTIMATED: its M-step block runs
(`update_scale!(pws, n)`, `update_inlier_prior!(pws, n)`, `update_nu!(pws, n)`).
One value for every part.
"""
struct Estimated end

"""
    FixedS2(s2)

The update policy of a scale HELD at the squared scale `s2` (positive, finite):
the ŝ² block writes it.
"""
struct FixedS2
    s2::Float64
    function FixedS2(s2::Real)
        (isfinite(s2) && s2 > 0) || throw(
            ArgumentError("FixedS2: a held scale must be a positive finite ŝ², got $s2")
        )
        return new(Float64(s2))
    end
end
scale_s2(policy::FixedS2) = policy.s2

"""
    FixedPrior(inlier_prior)

The update policy of an inlier prior HELD at `inlier_prior ∈ (0, 1)`: the π
block writes it (through the E-step's one clamp, `[1e-3, 1 − 1e-3]`).
"""
struct FixedPrior
    inlier_prior::Float64
    function FixedPrior(inlier_prior::Real)
        0 < inlier_prior < 1 ||
            throw(ArgumentError("FixedPrior: the inlier prior must lie in (0, 1), got $inlier_prior"))
        return new(Float64(inlier_prior))
    end
end

"""
    FixedNu(ν)

The update policy of a Student-t dof HELD at `ν` (positive, finite): the ν
block writes it.
"""
struct FixedNu
    ν::Float64
    function FixedNu(ν::Real)
        (isfinite(ν) && ν > 0) || throw(ArgumentError("FixedNu: ν must be positive and finite, got $ν"))
        return new(Float64(ν))
    end
end

"""
    ConsensusDof()

The update policy of a Student-t dof HELD at the value its consensus seed gave
it: the posterior-predictive dof of the scorer's consensus set,
`ν = 2α₀ + (k* − model_dof)·d_g` ([`partition_student_t`](@ref), with its
`ν ≥ 2` floor), where `α₀` is the scale prior's shape (0 under Jeffreys). The
seed sets it once; the ν block never moves it afterwards.

Why hold it there: a Gaussian whose scale carries an inverse-gamma posterior
marginalizes to a Student-t with exactly that dof — ν measures how well the
consensus set DETERMINES σ, not the shape of the inlier tail. Estimating ν
instead lets the fit trade one for the other: on a contaminated set a heavy
inlier tail (ν → 2) absorbs the outliers — the identifiability collapse this
policy prevents.
"""
struct ConsensusDof end

"""
    update_scale!(pws, n, policy) -> ŝ²

The scale block under its update policy. `Estimated()` is
[`update_scale!`](@ref)`(pws, n)`; [`FixedS2`](@ref) holds ŝ².
"""
update_scale!(pws::PosteriorWeightState, n::Int, ::Estimated) = update_scale!(pws, n)
update_scale!(pws::PosteriorWeightState, ::Int, policy::FixedS2) = pws.s_hat_sq = scale_s2(policy)

"""
    update_inlier_prior!(pws, n, policy) -> π

The inlier-prior block under its update policy: `Estimated()` is
[`update_inlier_prior!`](@ref)`(pws, n)`; [`FixedPrior`](@ref) holds π.
"""
update_inlier_prior!(pws::PosteriorWeightState, n::Int, ::Estimated) = update_inlier_prior!(pws, n)
update_inlier_prior!(pws::PosteriorWeightState, ::Int, policy::FixedPrior) =
    pws.inlier_prior = _clamp_inlier_prior(policy.inlier_prior)

"""
    update_nu!(pws, n, policy) -> ν

The ν block under its update policy: `Estimated()` is [`update_nu!`](@ref)`(pws, n)`;
[`FixedNu`](@ref) holds ν; [`ConsensusDof`](@ref) keeps the seed's.
"""
update_nu!(pws::PosteriorWeightState, n::Int, ::Estimated) = update_nu!(pws, n)
update_nu!(pws::PosteriorWeightState, ::Int, policy::FixedNu) = pws.ν = policy.ν
update_nu!(pws::PosteriorWeightState, ::Int, ::ConsensusDof) = pws.ν

"""
    update_weights!(pws, n)

E-step: recompute posterior inlier probabilities and combined IRLS weights
from current ν, ŝ², and squared residuals.
"""
function update_weights!(pws::PosteriorWeightState, n::Int)
    return soft_posterior_weights!(
        pws.w, pws.inlier_posterior, pws.sq_residuals,
        pws.ν, pws.s_hat_sq, pws.log_two_outlier_halfwidth, n, pws.d_g, pws.inlier_prior
    )
end

"""
    mixture_ecm!(pws, r, n)

One ECM cycle of the Student-t + uniform mixture with the mixing proportion
ESTIMATED (ν held fixed), from the residuals `r[1:n]` (unsquared). It squares them
into [`sq_residuals`](@ref)`(pws)`, then runs the E-step ([`update_weights!`](@ref)),
the scale block ([`update_scale!`](@ref)) and the inlier-prior block
([`update_inlier_prior!`](@ref)), in that order. Each block is an exact
conditional maximisation, so the mixture NLL ([`mixture_nll`](@ref)) never
rises. Returns `nothing`.
"""
function mixture_ecm!(pws::PosteriorWeightState, r::AbstractVector, n::Int)
    square_residuals!(pws.sq_residuals, r, n)
    update_weights!(pws, n)
    update_scale!(pws, n)
    update_inlier_prior!(pws, n)
    return nothing
end

"""
    mixture_nll(pws::PosteriorWeightState, r, n) -> Float64

The mixture negative log-likelihood `Σ_{i ≤ n} −log[π·t_ν(rᵢ; ŝ²) + (1−π)·p_out]`
of the residuals `r[1:n]` (UNSQUARED) at the state's current `(ν, ŝ², π)`.
It sums the per-point `mixture_nll(m, rᵢ²)` of [`StudentTMixture`](@ref)`(pws)`
in Float64. Reads the state; writes nothing.
"""
function mixture_nll(pws::PosteriorWeightState, r::AbstractVector, n::Int)
    m = StudentTMixture(pws)
    tot = 0.0
    @inbounds for i in 1:n
        tot += mixture_nll(m, r[i] * r[i])
    end
    return tot
end

# The weight-policy reading contract. `inlier_posterior` is the membership
# probability the mixture computes; `w` is that times the Student-t's own IRLS
# factor `(nu + d_g) / (RSS + r^2)`, which is what the normal equations want.
# Confusing them is a silent scale error — `w` is not bounded by one.
inlier_weight(pws::PosteriorWeightState) = pws.inlier_posterior
robust_weight(pws::PosteriorWeightState) = pws.w

# =============================================================================
# The partition mixture — weights of a hard partition's Student-t
# =============================================================================

# The Student-t mixture of a hard partition `order[1:k]`: its posterior
# predictive, `partition_student_t` — the SAME (ν, ŝ²) rule a
# `PosteriorWeightState` seeds from. The RSS is summed in Float64 regardless of
# the residuals' storage eltype. The inlier prior is ½.
@inline function _partition_mixture(sq_residuals::AbstractVector,
                                    order::AbstractVector{Int}, k::Int,
                                    dg::Int, model_dof::Int, log_two_outlier_halfwidth::Float64,
                                    prior::IGPrior)
    ν, s_hat_sq = partition_student_t(
        prior, _partition_rss(sq_residuals, order, k), k;
        model_dof, codimension = dg
    )
    return StudentTMixture(ν, s_hat_sq, log_two_outlier_halfwidth, dg)
end

"""
    posterior_weights!(w, sq_residuals, order, k; model_dof, log_two_outlier_halfwidth,
                       codimension = 1, prior = JeffreysPrior())

Student-t posterior weights `w[i] = P(inlier | rᵢ) · uᵢ` of the hard partition
`order[1:k]` of the squared residuals: the partition's Student-t
([`partition_student_t`](@ref) under `prior`: `ν = (k − model_dof)·d_g + 2α₀`,
at least 2, and `ŝ² = (RSS + 2β₀)/((k − model_dof)·d_g + 2α₀)`) against the
uniform outlier support, `uᵢ = (ν + d_g)/(ν·ŝ² + r²ᵢ)`. Per point this is
[`posterior_weight`](@ref) of that [`StudentTMixture`](@ref). Fills every
entry of `w`.
"""
function posterior_weights!(w::AbstractVector, sq_residuals::AbstractVector,
                            order::AbstractVector{Int}, k::Int;
                            model_dof::Int, log_two_outlier_halfwidth::Float64,
                            codimension::Int = 1, prior::IGPrior = JeffreysPrior())
    m = _partition_mixture(sq_residuals, order, k, codimension, model_dof, log_two_outlier_halfwidth, prior)
    @inbounds for i in eachindex(w)
        w[i] = posterior_weight(m, Float64(sq_residuals[i]))
    end
    return w
end

"""
    posterior_inlier_probs!(w, sq_residuals, order, k; model_dof, log_two_outlier_halfwidth,
                            codimension = 1, prior = JeffreysPrior())

Posterior inlier probabilities `w[i] ∈ [0, 1]` of the hard partition
`order[1:k]`, without the IRLS weight factor — the same mixture as
[`posterior_weights!`](@ref), so that is this times `uᵢ`.
"""
function posterior_inlier_probs!(w::AbstractVector, sq_residuals::AbstractVector,
                                 order::AbstractVector{Int}, k::Int;
                                 model_dof::Int, log_two_outlier_halfwidth::Float64,
                                 codimension::Int = 1, prior::IGPrior = JeffreysPrior())
    m = _partition_mixture(sq_residuals, order, k, codimension, model_dof, log_two_outlier_halfwidth, prior)
    @inbounds for i in eachindex(w)
        w[i] = inlier_posterior(m, Float64(sq_residuals[i]))
    end
    return w
end
