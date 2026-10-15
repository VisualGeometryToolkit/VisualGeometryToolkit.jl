# =============================================================================
# The scale estimators of the robust-statistics toolkit (moved from
# RansacScoringDoneRight's sweep.jl and posterior_weighting.jl): the
# inverse-gamma scale prior, the inverse-gamma MAP scale, a hard partition's
# Student-t, and the residual sums every estimator in the stack reads.
# =============================================================================

"""
    IGPrior(alpha0, beta0)

Inverse-Gamma prior on σ²: `π(σ²) ∝ (σ²)^{-(α₀+1)} exp(-β₀/σ²)`. Conjugate
to the Gaussian inlier likelihood, so the marginal reduces to a closed-form
sweep update.

The Jeffreys prior `π(σ²) ∝ 1/σ²` is the limiting `α₀ = β₀ = 0` case, built by
[`JeffreysPrior`](@ref)`()` (or `IGPrior()`); one struct handles both.

# Fields
- `alpha0::Float64` — shape hyperparameter (0 ⇒ Jeffreys)
- `beta0::Float64`  — scale hyperparameter (0 ⇒ Jeffreys)
"""
struct IGPrior
    alpha0::Float64
    beta0::Float64
end

# Zero-arg → Jeffreys (improper, non-informative)
IGPrior() = IGPrior(0.0, 0.0)

"""
    JeffreysPrior() -> IGPrior

The Jeffreys scale prior `π(σ²) ∝ 1/σ²`: `IGPrior(0.0, 0.0)`, the default
prior of every scale estimator and scorer in the stack. A function returning
the value, not a second name of the type — test `alpha0 == 0` rather than
dispatching on a name.
"""
JeffreysPrior() = IGPrior(0.0, 0.0)

"""
    fit_scale(prior::IGPrior, rss, dof) -> s2

Fit the squared scale ŝ² as the inverse-gamma MAP, `(rss + 2β₀)/(dof + 2α₀)`.

Returns ŝ², the SQUARED scale — never a linear σ̂.

ŝ² is PER COMPONENT: `rss` sums squared NORMS of d_g-vector residuals, so `dof`
counts components — `ν_data = (k − model_dof)·d_g` for a hard partition, or
`d_g·Σπᵢ` in the IRLS M-step (`update_scale!`). The `2α₀`/`2β₀` pseudo-counts
make a refinement inherit the scoring prior (`β₀` floors ŝ² at low inlier
counts); `IGPrior(0, 0)` (Jeffreys) gives `rss/dof`.

The first argument is THE ESTIMATOR: `IGPrior` selects the inverse-gamma MAP,
its prior being that estimator's parameters. The residuals are TAKEN as
inliers, no initial guess is needed, and no inlier decision is made here.

The result is floored at `eps()`, so a zero `rss` under Jeffreys returns
`eps()`, not a NaN. A caller that would rather HOLD its previous scale in that
case must test the dof itself at the call site.
"""
@inline fit_scale(prior::IGPrior, rss::Real, dof::Real) =
    max((rss + 2 * prior.beta0) / max(dof + 2 * prior.alpha0, eps()), eps())

"""
    partition_student_t(prior::IGPrior, rss, k; model_dof, codimension = 1) -> (ν, ŝ²)

The Student-t of a hard partition of `k` points with residual sum of squares
`rss`. It is the scoring marginal's posterior predictive over it under `prior`, and
THE seed of every Student-t in the stack: counted dof
`ν_data = (k − model_dof)·codimension`, then

    ν  = max(ν_data + 2α₀, 2),     ŝ² = fit_scale(prior, rss, ν_data).

The `ν ≥ 2` clamp keeps the seed's variance finite (a Student-t has one only for
ν > 2) at a thin partition, and `fit_scale` floors the QUOTIENT at eps. One rule
for every caller: both [`PosteriorWeightState`](@ref) seeds, the partition
mixture of [`posterior_weights!`](@ref)/[`posterior_inlier_probs!`](@ref), the
consensus scale [`consensus_s2`](@ref) and a RANSAC result's σ̂.
`model_dof` is the caller's: the sweep's `minimal_sample_size` for the scoring
marginal, one more for a local-optimization seed that also estimates the scale.
"""
@inline function partition_student_t(prior::IGPrior, rss::Real, k::Integer;
                                     model_dof::Integer, codimension::Integer = 1)
    ν_data = Float64((k - model_dof) * codimension)
    return max(ν_data + 2 * prior.alpha0, 2.0), fit_scale(prior, rss, ν_data)
end

# Σ_{j ≤ k} f(xs[order[j]]) in Float64, left to right from 0.0 — the one partition
# loop behind `_partition_rss` (squared residuals in) and `partition_rss`
# (residuals in, squared here). Left-to-right from 0.0, so every caller is
# bit-identical to the loop it replaced.
@inline function _partition_sum(f::F, xs::AbstractVector, order::AbstractVector{Int},
                                k::Int) where {F}
    s = 0.0
    @inbounds for j in 1:k
        s += f(xs[order[j]])
    end
    return s
end

# RSS of a partition from SQUARED residuals: Σ_{j ≤ k} r²_{order[j]}, in Float64
# whatever the storage eltype.
@inline _partition_rss(sq_residuals::AbstractVector, order::AbstractVector{Int}, k::Int) =
    _partition_sum(Float64, sq_residuals, order, k)

"""
    partition_rss(order, residuals, k) -> Float64

The residual sum of squares of the partition `order[1:k]`:
`Σ_{j ≤ k} residuals[order[j]]²`, from UNSQUARED residuals (signed scalars or
norms), accumulated in Float64 whatever their storage eltype. `order` is a
scorer's [`consensus_order`](@ref) (or a copy of it).
"""
partition_rss(order::AbstractVector{Int}, residuals::AbstractVector, k::Int) =
    _partition_sum(r -> Float64(r)^2, residuals, order, k)

"""
    weighted_rss(w, sq_residuals, n) -> Float64

The weighted residual sum of squares `Σ_{i ≤ n} wᵢ·rᵢ²` of SQUARED residuals,
accumulated in Float64 left to right from 0.0. A zero-weight term contributes
exactly 0: an excluded point may carry an infinite squared residual (the
Sampson degeneracy rule — a vanishing projected variance), and `0·Inf = NaN`
would poison the sum and everything read from it (the ŝ² M-step of
[`update_scale!`](@ref), an IRLS cost and its stop rule).
"""
@inline function weighted_rss(w::AbstractVector, sq_residuals::AbstractVector, n::Int)
    s = 0.0
    @inbounds for i in 1:n
        s += weighted_sq(w[i], sq_residuals[i])
    end
    return s
end

"""
    weighted_sq(w, sq_r) -> w·sq_r, exactly 0 at w = 0

One term of [`weighted_rss`](@ref), for a caller summing weights it computes on
the fly rather than holding in a buffer (an ECM scale over a loss's own IRLS
weights). A zero weight contributes exactly 0 even against an infinite squared
residual, where `0·Inf = NaN`.
"""
@inline weighted_sq(w, sq_r) = iszero(w) ? zero(w * sq_r) : w * sq_r

"""
    square_residuals!(sq, r, n) -> sq

`sq[i] = r[i]²` for `i ≤ n`, squared in `sq`'s OWN precision (`x = eltype(sq)(r[i])`,
then `x·x`): a Float64 buffer squares a Float32 residual exactly. THE one
residual-squaring loop of the stack — the refill of a
[`PosteriorWeightState`](@ref)'s [`sq_residuals`](@ref), the per-point scores
a scorer sweeps, and the pool staging.
"""
function square_residuals!(sq::AbstractVector{T}, r::AbstractVector, n::Int) where {T}
    @inbounds for i in 1:n
        x = T(r[i])
        sq[i] = x * x
    end
    return sq
end

# A minimal sample's points satisfy their constraints exactly: zero their
# entries of `v` (a score or a penalty buffer). Bounds-checked on purpose (a
# handful of writes): an invalid sample index THROWS rather than writing outside
# `v` or being skipped. `nothing` = no minimal sample.
zero_sample!(v::AbstractVector, ::Nothing) = v
function zero_sample!(v::AbstractVector, sample_idx)
    for idx in sample_idx
        v[idx] = zero(eltype(v))
    end
    return v
end
