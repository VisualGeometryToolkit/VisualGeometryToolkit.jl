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
# Generic in the float type (a Float32 kernel computes in Float32: the halves are
# `/ 2`, exact in every float type), with the log-gamma supplied (a kernel supplies
# its device's). At Float64 it is bit for bit the `/ 2.0` spelling it replaced.
@inline function _student_t_log_normaliser(ν::T, s_hat_sq::T, d::Integer, lgamma::F = loggamma) where {T <: Real, F}
    isinf(ν) && return -(T(d) / 2) * log(2 * T(π) * s_hat_sq)
    return lgamma((ν + T(d)) / 2) - lgamma(ν / 2) - (T(d) / 2) * log(ν * T(π) * s_hat_sq)
end
_student_t_log_normaliser(ν::Real, s_hat_sq::Real, d::Integer) =
    _student_t_log_normaliser(promote(float(ν), float(s_hat_sq))..., d)

function StudentTMixture(ν::Real, s_hat_sq::Real, log_two_outlier_halfwidth::Real, d_g::Integer,
                         inlier_prior::Real = 0.5)
    return StudentTMixture{Float64}(ν, s_hat_sq, log_two_outlier_halfwidth, d_g, inlier_prior)
end

"""
    StudentTMixture{T}(ν, s_hat_sq, log_two_outlier_halfwidth, d_g[, inlier_prior = 1/2]; lgamma = loggamma)

The mixture built IN `T` (every constant computed in `T`, not in Float64 and
rounded), with the log-gamma function supplied: kernel-lawful, so a device
kernel whose ν and ŝ² change per work item builds one per item and calls
[`inlier_posterior`](@ref) / [`posterior_weight`](@ref) on it. `lgamma` is the
device's (CUDA's `__nv_lgamma`, Core's `lgamma_lanczos` elsewhere); the host
default is SpecialFunctions' `loggamma`. `StudentTMixture(…)` is
`StudentTMixture{Float64}(…)`.
"""
function StudentTMixture{T}(ν::Real, s_hat_sq::Real, log_two_outlier_halfwidth::Real, d_g::Integer,
                            inlier_prior::Real = T(0.5); lgamma::F = loggamma) where {T <: AbstractFloat, F}
    ν = T(ν); s_hat_sq = T(s_hat_sq); log_two_outlier_halfwidth = T(log_two_outlier_halfwidth)
    inlier_prior = T(inlier_prior); dg = T(d_g)
    # Spelled as the E-step always spelled them, so the vector paths built on
    # this are bit-identical to the loops they replace.
    half_νdg = (ν + dg) / 2
    log_C = _student_t_log_normaliser(ν, s_hat_sq, d_g, lgamma)
    log_pout = -dg * log_two_outlier_halfwidth                          # uniform density (2·outlier_halfwidth)^{-d_g}
    # The inlier prior enters ONLY as an additive shift of the log-odds: the
    # outlier's prior log-odds −logit(π). At inlier_prior = 1/2 the shift is
    # `-logit(0.5) === -0.0` and `x - 0.0 === x`.
    log_odds_out = log_pout - logit(inlier_prior)
    log_mix_out = log_pout + log1p(-inlier_prior)
    return StudentTMixture{T}(s_hat_sq, ν * s_hat_sq, ν + dg, half_νdg, log_C, log_odds_out, log_mix_out)
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
    soft_posterior_weights!(w, inlier_posterior, sq_distances, ν, s_hat_sq, log_two_outlier_halfwidth, n, d_g=1[, inlier_prior=1/2])

E-step: compute posterior inlier probability and combined IRLS weight.

Stores `inlier_posterior[i] = P(inlier | r_i)` separately for the ν M-step,
and `w[i] = inlier_posterior[i] · u_i` (combined weight for the θ solve).

`d_g` is the constraint codimension (1 for scalar residuals, 2 for 2D).

The separation is critical: the ν Newton update must use fixed E-step weights
(not recomputed from the changing ν), otherwise ν and `inlier_posterior`
couple in a positive feedback loop that drives ν → ∞.
"""
function soft_posterior_weights!(w::AbstractVector, inlier_posterior::AbstractVector,
                                 sq_distances::AbstractVector,
                                 ν::Float64, s_hat_sq::Float64, log_two_outlier_halfwidth::Float64, n::Int,
                                 d_g::Int = 1, inlier_prior::Float64 = 0.5)
    m = StudentTMixture(ν, s_hat_sq, log_two_outlier_halfwidth, d_g, inlier_prior)
    @inbounds for i in 1:n
        inlier_posterior[i], w[i] = _posterior_pair(m, sq_distances[i])
    end
    return nothing
end

"""
    next_nu(ν, sq_distances, s_hat_sq, w_fixed, n; d_g=1, kwargs...) -> ν_new

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
function next_nu(ν::Float64, sq_distances::AbstractVector, s_hat_sq::Float64,
                 w_fixed::AbstractVector, n::Int;
                 d_g::Int = 1, max_newton::Int = 5, ν_min::Float64 = 2.0, ν_max::Float64 = 1.0e4)
    for _ in 1:max_newton
        dQ, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν = _nu_sums(ν, sq_distances, s_hat_sq, w_fixed, n, d_g)
        ν, converged = next_nu_from_sums(ν, dQ, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν; d_g, ν_min, ν_max)
        converged && break
    end
    return ν
end

"""
    nu_point_terms(z, ν, ψ_half_νdg, ψ_half_ν; d_g = 1) -> (dq, d2q_data)

One point's terms of [`next_nu`](@ref)'s Q-derivatives at ν (its docstring
states both): `dq` its summand of `∂Q/∂ν` and `d2q_data` the data part of its
summand of `∂²Q/∂ν²` (the ν-only part is the sum's, `Σwᵢ` times a constant,
added by [`next_nu_from_sums`](@ref)), at the scale-normalised squared
residual `z = r²/(ν·ŝ²)`, with `ψ_half_νdg = ψ((ν + d_g)/2)` and
`ψ_half_ν = ψ(ν/2)` the caller's digammas. Generic in the float type and
allocation-free: a device kernel sums `wᵢ·dq` and `wᵢ·d2q_data` over its
points (a workgroup reduction) and hands the sums to `next_nu_from_sums`, the
same arithmetic as the host loop of `next_nu`.
"""
@inline function nu_point_terms(z::T, ν::T, ψ_half_νdg::T, ψ_half_ν::T; d_g::Integer = 1) where {T <: Real}
    dg = T(d_g)
    h = T(0.5)
    inv_ν = one(T) / ν
    inv_1pz = one(T) / (one(T) + z)
    dq = h * ψ_half_νdg - h * ψ_half_ν - h * dg * inv_ν -
        h * log1p(z) + (ν + dg) * z * inv_1pz * h * inv_ν
    # ν + 2d + d·z, spelled so that d = 1 rounds exactly as the old ν+d+1+z
    d2q_data = h * z * inv_1pz * inv_ν - h * z * (ν + dg + dg + dg * z) * inv_1pz * inv_1pz * (inv_ν * inv_ν)
    return dq, d2q_data
end

"""
    next_nu_from_sums(ν, dQ, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν; d_g = 1, ν_min = 2, ν_max = 1e4) -> (ν_new, converged)

One safeguarded Newton update of [`next_nu`](@ref) from the reduced sums
`dQ = Σwᵢ·dqᵢ`, `d2Q_data = Σwᵢ·d2q_dataᵢ` ([`nu_point_terms`](@ref)) and
`sum_w = Σwᵢ`, with `ψ1_half_νdg = ψ₁((ν + d_g)/2)` and `ψ1_half_ν = ψ₁(ν/2)`
the caller's trigammas. `∂²Q = sum_w·(¼ψ₁((ν+d)/2) − ¼ψ₁(ν/2) + d/(2ν²)) +
d2Q_data`; Newton is taken only where Q is concave (`∂²Q < 0`, its step then
uphill), elsewhere the step is geometric along the gradient (`2ν` or `ν/2`);
the result is clamped to `[ν_min, ν_max]`. `converged` is `true` when the
weights vanish (`sum_w < 1e-10`, ν returned unchanged) or `|dQ/sum_w| < 1e-8`.
Generic in the float type, allocation-free: the ν update of a device kernel.
"""
@inline function next_nu_from_sums(ν::T, dQ::T, d2Q_data::T, sum_w::T, ψ1_half_νdg::T, ψ1_half_ν::T;
                                   d_g::Integer = 1, ν_min::Real = 2.0, ν_max::Real = 1.0e4) where {T <: Real}
    sum_w < T(1.0e-10) && return ν, true
    d2Q = _nu_curvature(ν, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν, d_g)
    # Ascent safeguard: Newton only where Q is concave (its step is then
    # uphill); a convex point steps geometrically along the gradient.
    ν_new = d2Q < zero(T) ? ν - dQ / d2Q : (dQ > zero(T) ? 2 * ν : T(0.5) * ν)
    return clamp(ν_new, T(ν_min), T(ν_max)), abs(dQ / sum_w) < T(1.0e-8)
end

# The reduced sums of `next_nu`'s Q at ν, and the trigammas its curvature reads:
# (Σwᵢdqᵢ, Σwᵢd2q_dataᵢ, Σwᵢ, ψ₁((ν+d)/2), ψ₁(ν/2)). Points with wᵢ < 1e-15 are
# skipped, as outliers.
function _nu_sums(ν::Float64, sq_distances::AbstractVector, s_hat_sq::Float64,
                  w_fixed::AbstractVector, n::Int, d_g::Int)
    dg = Float64(d_g)
    ψ_half_νdg = digamma((ν + dg) / 2.0)
    ψ_half_ν = digamma(ν / 2.0)
    dQ = 0.0
    d2Q_data = 0.0
    sum_w = 0.0
    @inbounds for i in 1:n
        wi = w_fixed[i]
        wi < 1.0e-15 && continue
        z_i = sq_distances[i] / (ν * s_hat_sq)   # scale-normalised r²
        dq_i, d2q_data_i = nu_point_terms(z_i, ν, ψ_half_νdg, ψ_half_ν; d_g)
        dQ += wi * dq_i
        sum_w += wi
        d2Q_data += wi * d2q_data_i
    end
    return dQ, d2Q_data, sum_w, trigamma((ν + dg) / 2.0), trigamma(ν / 2.0)
end

# (∂Q/∂ν, ∂²Q/∂ν², Σwᵢ) of `next_nu`'s Q at ν — its docstring states both.
function _nu_derivatives(ν::Float64, sq_distances::AbstractVector, s_hat_sq::Float64,
                         w_fixed::AbstractVector, n::Int, d_g::Int)
    dQ, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν = _nu_sums(ν, sq_distances, s_hat_sq, w_fixed, n, d_g)
    return dQ, _nu_curvature(ν, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν, d_g), sum_w
end

# ∂²Q/∂ν²: the ν-only part, Σwᵢ times its constant, plus the data part.
@inline function _nu_curvature(ν::T, d2Q_data::T, sum_w::T, ψ1_half_νdg::T, ψ1_half_ν::T, d_g::Integer) where {T}
    inv_ν = one(T) / ν
    return sum_w * (T(0.25) * ψ1_half_νdg - T(0.25) * ψ1_half_ν + T(0.5) * T(d_g) * (inv_ν * inv_ν)) + d2Q_data
end

# =============================================================================
# The MAP crossover of the Gaussian(-t)/uniform pair
# =============================================================================
#
# The mixture has TWO parameterisations of one boundary: the outlier half-width
# that the marginal family carries, and the decision boundary — the residual at
# which the MAP rule flips — that a truncation at that boundary would carry as
# its loss width. Converting between them IS the MAP rule solved for one unknown
# or the other, so there are exactly two functions:
#
#     decision_boundary(m, outlier_halfwidth)  -> the residual the rule flips at
#     decision_width(m, boundary)               -> the outlier half-width
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
    decision_boundary(m::GauInlier, outlier_halfwidth) -> Float64

The residual at which the MAP rule flips between inlier and outlier: `z* = 1`
iff `gamma*f_in > (1-gamma)*f_out` holds exactly when `norm(r)` is below it.

Returns `0.0` when the outlier component already explains every residual at
least as well (the mixture admits nothing), rather than a negative root.
"""
function decision_boundary(m::GauInlier, outlier_halfwidth::Real)
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
    decision_width(m::GauInlier, boundary) -> outlier_halfwidth

The mixture's method of [`decision_width`](@ref), and the inverse of
[`decision_boundary`](@ref): the per-axis outlier half-width that puts the
MAP boundary at `boundary`. A HALF-WIDTH and not a radius — the outlier
support is a box, a product of intervals, never a ball, which is why the
codimension enters as the exponent.
"""
function decision_width(m::GauInlier, boundary::Real)
    d, L = m.d_g, _label_logodds(m)
    lhs = -_student_t_log_kernel((m.nu + d) / 2, m.nu * m.s2, m.s2, boundary^2)
    return 0.5 * exp((lhs - L - _log_peak(m)) / d)
end

