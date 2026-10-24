# =============================================================================
# The Student-t inlier + uniform outlier mixture — its scalar E-step, the ν
# M-step, and the MAP crossover of the Gaussian/uniform pair (moved from
# RANSACScoringDoneRight's posterior_weighting.jl and scoring.jl).
# =============================================================================
#
# The EM iteration of the mixture alternates:
#   E-step: soft_posterior_weights!  →  inlier_posterior = P(inlier | r),
#           and the combined weight w = inlier_posterior · u_i / ŝ² (precision-scaled)
#   M-step: update_scale!  →  ŝ² (ECM), then update_nu!  →  ν
#           (`next_nu`, Newton MLE for ν at the new ŝ²)
#
# The ν update uses fixed E-step weights (not recomputed), matching the paper's
# EM derivation and guaranteeing Q-monotonicity.
# =============================================================================

"""
    StudentTMixture(ν, s_hat_sq, log_two_outlier_halfwidth, d_g[, inlier_prior = 1/2])
    StudentTMixture{T}(m::StudentTMixture)

The Student-t inlier + uniform outlier mixture of one E-step, with every
per-residual constant precomputed. Its parameters are `ν` degrees of
freedom, squared scale `ŝ² = s_hat_sq`, outlier support of half-width
`outlier_halfwidth` per axis (`log_two_outlier_halfwidth = log(2·outlier_halfwidth)`,
density `p_out = (2·outlier_halfwidth)^-d_g`), inlier mixing proportion
`π = inlier_prior`. The `loggamma`s are evaluated HERE, once.

An immutable `isbits` value, so it can be an argument of a KernelAbstractions kernel, and
the three scalar functions of it — [`inlier_posterior`](@ref),
[`posterior_weight`](@ref), [`mixture_nll`](@ref) — are allocation-free plain
functions of a squared residual, callable inside a kernel as VGC's `ρ` is.
`StudentTMixture{Float32}(m)` converts the constants for a Float32 kernel.

These are THE formulas of the mixture: `soft_posterior_weights!` (the E-step
over a buffer) and `posterior_weights!`/`posterior_inlier_probs!` (over a
partition) build one of these and call the same scalar kernel.
"""
struct StudentTMixture{T <: AbstractFloat}
    s2::T            # ŝ² (read only by the ν = Inf Gaussian limit)
    νs2::T           # ν·ŝ²
    νdg::T           # ν + d_g
    half_νdg::T      # (ν + d_g)/2
    log_C::T         # log of the Student-t normaliser Γ((ν+d)/2) / (Γ(ν/2)·(νπŝ²)^{d/2})
    log_odds_out::T  # log p_out + log((1−π)/π): the outlier's log-odds against a unit-density inlier
    log_mix_out::T   # log((1−π)·p_out): the outlier term of the mixture density
end

# log of the d-variate Student-t normaliser Γ((ν+d)/2) / (Γ(ν/2)·(νπŝ²)^{d/2}) —
# and at ν = Inf its Gaussian limit (2πŝ²)^{−d/2}, which the finite form cannot
# reach (Inf·π·ŝ² in the log). THE one statement: the mixture's E-step below and
# the MAP crossover's `GauInlier` both read it.
@inline function _student_t_log_normaliser(ν, s_hat_sq, d::Integer)
    isinf(ν) && return -(d / 2) * log(2 * π * s_hat_sq)
    return loggamma((ν + d) / 2.0) - loggamma(ν / 2.0) - (d / 2.0) * log(ν * π * s_hat_sq)
end

function StudentTMixture(ν::Real, s_hat_sq::Real, log_two_outlier_halfwidth::Real, d_g::Integer,
                         inlier_prior::Real = 0.5)
    ν = Float64(ν); s_hat_sq = Float64(s_hat_sq); log_two_outlier_halfwidth = Float64(log_two_outlier_halfwidth)
    inlier_prior = Float64(inlier_prior)
    # Spelled as the E-step always spelled them, so the vector paths built on
    # this are bit-identical to the loops they replace.
    half_νdg = (ν + d_g) / 2.0
    log_C = _student_t_log_normaliser(ν, s_hat_sq, d_g)
    log_pout = -d_g * log_two_outlier_halfwidth                          # uniform density (2·outlier_halfwidth)^{-d_g}
    # The inlier prior enters ONLY as an additive shift of the log-odds: the
    # outlier's prior log-odds −logit(π). At inlier_prior = 1/2 the shift is
    # `-logit(0.5) === -0.0` and `x - 0.0 === x`.
    log_odds_out = log_pout - logit(inlier_prior)
    log_mix_out = log_pout + log1p(-inlier_prior)
    return StudentTMixture{Float64}(s_hat_sq, ν * s_hat_sq, ν + d_g, half_νdg, log_C,
                                    log_odds_out, log_mix_out)
end

StudentTMixture{T}(m::StudentTMixture) where {T <: AbstractFloat} =
    StudentTMixture{T}(T(m.s2), T(m.νs2), T(m.νdg), T(m.half_νdg), T(m.log_C),
                       T(m.log_odds_out), T(m.log_mix_out))

# The d-variate Student-t kernel of a squared residual, from the precomputed
# (ν + d)/2, ν + d, νŝ² and ŝ², and at ν = Inf its Gaussian limit, which the
# finite forms cannot reach (Inf·0 in the log, Inf/Inf in the weight). THE one
# statement of each: the E-step below and the MAP crossover's `GauInlier` both
# read them.
#   log kernel  −((ν+d)/2)·log1p(sq_r/(νŝ²))      →  −sq_r/(2ŝ²)
#   weight      (ν+d)/(νŝ² + sq_r)                 →  1/ŝ²     (−2·∂/∂sq_r of it)
@inline _student_t_log_kernel(half_νdg, νs2, s2, sq_r) =
    isinf(νs2) ? -sq_r / (2 * s2) : -half_νdg * log1p(sq_r / νs2)
@inline _student_t_weight(νdg, νs2, s2, sq_r) =
    isinf(νs2) ? inv(s2) : νdg / (νs2 + sq_r)

# log t_ν(r; ŝ²): the Student-t inlier log-density of a squared residual
@inline _log_inlier_density(m::StudentTMixture, sq_r) =
    m.log_C + _student_t_log_kernel(m.half_νdg, m.νs2, m.s2, sq_r)

# (P(inlier | r), w): the responsibility and the IRLS weight, the pair the
# E-step writes. ONE kernel: the public scalars and every vector path call it.
@inline function _posterior_pair(m::StudentTMixture{T}, sq_r) where {T}
    p = one(T) / (one(T) + exp(m.log_odds_out - _log_inlier_density(m, sq_r)))
    return p, p * _student_t_weight(m.νdg, m.νs2, m.s2, sq_r)
end

"""
    inlier_posterior(m::StudentTMixture, sq_r) -> P(inlier | r)

The posterior probability that a residual with squared norm `sq_r` is an
inlier, `π·t / (π·t + (1−π)·p_out)`. Scalar and kernel-callable.
"""
@inline inlier_posterior(m::StudentTMixture, sq_r) = first(_posterior_pair(m, sq_r))

"""
    posterior_weight(m::StudentTMixture, sq_r) -> w

The IRLS weight of the mixture at `sq_r`: `P(inlier | r) · (ν + d_g)/(ν·ŝ² + r²)`,
the responsibility times the Student-t's own weight. Scalar and kernel-callable.
"""
@inline posterior_weight(m::StudentTMixture, sq_r) = last(_posterior_pair(m, sq_r))

"""
    posterior_weight_derivative(m::StudentTMixture, sq_r) -> dw/d(sq_r)

The slope of [`posterior_weight`](@ref) in the squared residual,
`−w/(ν·ŝ² + r²) · (1 + (ν + d_g)/2 · (1 − P(inlier | r)))` — the Student-t
weight's own decay plus the responsibility's. `w` is strictly decreasing in
`sq_r`, so this is negative wherever `w > 0`; it is what a Newton inversion of
the weight (a tabulated half-quadratic conjugate, LocalOptimizationDoneRight's `TabulatedLift`)
needs. Scalar and kernel-callable.
"""
@inline function posterior_weight_derivative(m::StudentTMixture, sq_r)
    p, w = _posterior_pair(m, sq_r)
    # ν = Inf: the Gaussian weight is flat, only the responsibility decays
    isinf(m.νs2) && return -w * (1 - p) / (2 * m.s2)
    return -w / (m.νs2 + sq_r) * (1 + m.half_νdg * (1 - p))
end

"""
    mixture_nll(m::StudentTMixture, sq_r) -> −log(π·t_ν(r; ŝ²) + (1−π)·p_out)

The negative log-density of the mixture at one residual — the per-point summand
of the mixture NLL. Computed against the outlier term in log space,
`−(log((1−π)p_out) + softplus(log π·t − log((1−π)p_out)))`, so it is finite for
an infinite residual (it is the outlier term's) and for an exact inlier.
Scalar and kernel-callable.
"""
@inline mixture_nll(m::StudentTMixture, sq_r) =
    -(m.log_mix_out + log1pexp(_log_inlier_density(m, sq_r) - m.log_odds_out))

"""
    soft_posterior_weights!(w, inlier_posterior, sq_residuals, ν, s_hat_sq, log_two_outlier_halfwidth, n, d_g=1[, inlier_prior=1/2])

E-step: compute posterior inlier probability and combined IRLS weight.

Stores `inlier_posterior[i] = P(inlier | r_i)` separately for the ν M-step,
and `w[i] = inlier_posterior[i] · u_i` (combined weight for the θ solve).

`d_g` is the constraint codimension (1 for scalar residuals, 2 for 2D).

The separation is critical: the ν Newton update must use fixed E-step weights
(not recomputed from the changing ν), otherwise ν and `inlier_posterior`
couple in a positive feedback loop that drives ν → ∞.
"""
function soft_posterior_weights!(w::AbstractVector, inlier_posterior::AbstractVector,
                                 sq_residuals::AbstractVector,
                                 ν::Float64, s_hat_sq::Float64, log_two_outlier_halfwidth::Float64, n::Int,
                                 d_g::Int = 1, inlier_prior::Float64 = 0.5)
    m = StudentTMixture(ν, s_hat_sq, log_two_outlier_halfwidth, d_g, inlier_prior)
    @inbounds for i in 1:n
        inlier_posterior[i], w[i] = _posterior_pair(m, sq_residuals[i])
    end
    return nothing
end

"""
    next_nu(ν, sq_residuals, s_hat_sq, w_fixed, n; d_g=1, kwargs...) -> ν_new

M-step: Newton update for ν maximizing the EM Q-function with fixed E-step weights.

Maximizes Q(ν) = Σ w_i · log t_{d_g}(r_i; ν, ŝ²) w.r.t. ν, where w_i are the
fixed posterior inlier probabilities from the E-step (not recomputed), r_i² the
squared NORM of a d_g-vector residual and ŝ² the per-component scale:

  log t_d(r; ν, ŝ²) = logΓ((ν+d)/2) − logΓ(ν/2) − (d/2)·log(νπŝ²)
                      − ((ν+d)/2)·log(1 + z),     z = r²/(νŝ²)

  ∂Q/∂ν = Σ w_i · [½ψ((ν+d)/2) − ½ψ(ν/2) − d/(2ν)
                     − ½log(1 + z_i) + (ν+d)z_i/(2ν(1+z_i))]

  ∂²Q/∂ν² = Σ w_i · [¼ψ₁((ν+d)/2) − ¼ψ₁(ν/2) + d/(2ν²)
                       + z_i/(2ν(1+z_i)) − z_i(ν + 2d + d·z_i)/(2ν²(1+z_i)²)]

with `z_i` the SCALE-NORMALISED squared residual.

Up to `max_newton` Newton steps from the current ν, each clamped to
`[ν_min, ν_max]`. Newton is taken only where Q is concave (∂²Q < 0), where its
step is uphill; elsewhere the step is geometric, uphill along the gradient —
ν ← 2ν if ∂Q/∂ν > 0, ν ← ν/2 otherwise.
"""
function next_nu(ν::Float64, sq_residuals::AbstractVector, s_hat_sq::Float64,
                 w_fixed::AbstractVector, n::Int;
                 d_g::Int = 1, max_newton::Int = 5, ν_min::Float64 = 2.0, ν_max::Float64 = 1.0e4)
    for _ in 1:max_newton
        dQ, d2Q, sum_w = _nu_derivatives(ν, sq_residuals, s_hat_sq, w_fixed, n, d_g)
        sum_w < 1.0e-10 && break
        # Ascent safeguard: Newton only where Q is concave (its step is then
        # uphill); a convex point steps geometrically along the gradient.
        ν_new = d2Q < 0.0 ? ν - dQ / d2Q : (dQ > 0.0 ? 2.0 * ν : 0.5 * ν)
        ν = clamp(ν_new, ν_min, ν_max)
        abs(dQ / sum_w) < 1.0e-8 && break
    end
    return ν
end

# (∂Q/∂ν, ∂²Q/∂ν², Σwᵢ) of `next_nu`'s Q at ν — its docstring states both.
# Points with wᵢ < 1e-15 are skipped, as outliers.
function _nu_derivatives(ν::Float64, sq_residuals::AbstractVector, s_hat_sq::Float64,
                         w_fixed::AbstractVector, n::Int, d_g::Int)
    dg = Float64(d_g)
    ψ_half_νdg = digamma((ν + dg) / 2.0)
    ψ_half_ν = digamma(ν / 2.0)
    ψ1_half_νdg = trigamma((ν + dg) / 2.0)
    ψ1_half_ν = trigamma(ν / 2.0)
    dQ = 0.0
    d2Q_data = 0.0
    sum_w = 0.0
    inv_ν = 1.0 / ν
    inv_ν2 = inv_ν * inv_ν
    @inbounds for i in 1:n
        wi = w_fixed[i]
        wi < 1.0e-15 && continue
        z_i = sq_residuals[i] / (ν * s_hat_sq)   # scale-normalised r²
        inv_1pz = 1.0 / (1.0 + z_i)
        dq_i = 0.5 * ψ_half_νdg - 0.5 * ψ_half_ν - 0.5 * dg * inv_ν -
            0.5 * log1p(z_i) + (ν + dg) * z_i * inv_1pz * 0.5 * inv_ν
        dQ += wi * dq_i
        sum_w += wi
        d2q_data_i = 0.5 * z_i * inv_1pz * inv_ν -
            0.5 * z_i * (ν + dg + dg + dg * z_i) * inv_1pz * inv_1pz * inv_ν2
        # ν + 2d + d·z, spelled so that d = 1 rounds exactly as the old ν+d+1+z
        d2Q_data += wi * d2q_data_i
    end
    d2Q_nu_only = sum_w * (0.25 * ψ1_half_νdg - 0.25 * ψ1_half_ν + 0.5 * dg * inv_ν2)
    return dQ, d2Q_nu_only + d2Q_data, sum_w
end

# =============================================================================
# The MAP crossover of the Gaussian(-t)/uniform pair
# =============================================================================
#
# The mixture has TWO parameterisations of one boundary: the outlier half-width
# that the marginal family carries, and the decision threshold — the residual at
# which the MAP rule flips — that a truncation at that boundary would carry as
# its loss width. Converting between them IS the MAP rule solved for one unknown
# or the other, so there are exactly two functions:
#
#     decision_threshold(m, outlier_halfwidth)  -> the residual the rule flips at
#     decision_width(m, threshold)              -> the outlier half-width
# =============================================================================

"""
    GauInlier(s2; nu = Inf, gamma = 1/2, d_g)

Everything the Gaussian-uniform mixture needs EXCEPT the outlier half-width —
the half both directions of the boundary share.

- `s2`   inlier variance sigma^2
- `nu`   effective dof: `Inf` is the Gaussian inlier, a finite value the
         scale-marginalised (Student-t) one. The Gaussian IS the nu -> Inf
         limit, and the two closed forms below agree there.
- `gamma` the label prior `Pr(z_i = 1)`; a general gamma adds
         `log(gamma/(1-gamma))` to the boundary.
- `d_g`  the codimension, REQUIRED. It enters the outlier density as the
         EXPONENT `(2·outlier_halfwidth)^-d_g`, so no value of it is a safe
         default.
"""
struct GauInlier{T <: Real}
    s2::T
    nu::T
    gamma::T
    d_g::Int
end
GauInlier(s2::Real; nu::Real = Inf, gamma::Real = 0.5, d_g::Integer) =
    GauInlier(promote(float(s2), float(nu), float(gamma))..., Int(d_g))

# log-odds of the label prior (the inlier's); 0 at gamma = 1/2
@inline _label_logodds(m::GauInlier) = logit(m.gamma)

# log of the inlier density's peak, Student-t or its nu -> Inf Gaussian limit
@inline _log_peak(m::GauInlier) = _student_t_log_normaliser(m.nu, m.s2, m.d_g)

"""
    decision_threshold(m::GauInlier, outlier_halfwidth) -> Float64

The residual at which the MAP rule flips between inlier and outlier: `z* = 1`
iff `gamma*f_in > (1-gamma)*f_out` holds exactly when `norm(r)` is below it.

Returns `0.0` when the outlier component already explains every residual at
least as well (the mixture admits nothing), rather than a negative root.
"""
function decision_threshold(m::GauInlier, outlier_halfwidth::Real)
    d, L = m.d_g, _label_logodds(m)
    # log(gamma*peak / ((1-gamma)*(2·outlier_halfwidth)^-d))
    rhs = L + _log_peak(m) + d * log(2 * outlier_halfwidth)
    if isinf(m.nu)
        t2 = 2 * m.s2 * rhs
        return t2 <= 0 ? 0.0 : sqrt(t2)
    end
    t = expm1(2 * rhs / (m.nu + d))
    return t <= 0 ? 0.0 : sqrt(m.nu * m.s2 * t)
end

"""
    decision_width(m::GauInlier, threshold) -> outlier_halfwidth

The mixture's method of [`decision_width`](@ref), and the inverse of
[`decision_threshold`](@ref): the per-axis outlier half-width that puts the
MAP boundary at `threshold`. A HALF-WIDTH and not a radius — the outlier
support is a box, a product of intervals, never a ball, which is why the
codimension enters as the exponent.
"""
function decision_width(m::GauInlier, threshold::Real)
    d, L = m.d_g, _label_logodds(m)
    lhs = -_student_t_log_kernel((m.nu + d) / 2, m.nu * m.s2, m.s2, threshold^2)
    return 0.5 * exp((lhs - L - _log_peak(m)) / d)
end

