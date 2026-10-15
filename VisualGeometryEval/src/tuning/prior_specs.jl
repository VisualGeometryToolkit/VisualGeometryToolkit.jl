# =============================================================================
# prior_specs.jl — the scale-PRIOR strategy layer of a tuning: the
# ScalePriorFamily × PriorFitting axes (`PriorSpec`, `fit_prior`,
# `family_tag`/`fitting_tag`, `prior_label`, `ALL_PRIOR_SPECS`), fitting
# VisualGeometryCore's `IGPrior` from per-pair σ̂² samples. Moved from
# RansacScoringDoneRight's eval_strategy.jl (K+, 2026-10-09): it is tuning
# policy, consumed by the paper projects' tunings.
# =============================================================================

# =============================================================================
# Prior family × fitting strategy — two orthogonal axes
# =============================================================================
#
# The old `PriorStrategy` axis conflated *what kind of distribution we're
# parameterizing* with *how its parameters are derived from data*. They're
# separate concerns:
#
#   `ScalePriorFamily` — the distribution. Jeffreys (improper, parameterless)
#                        or InverseGamma (proper, two hyperparameters).
#
#   `PriorFitting`     — the algorithm. Unfitted for parameterless families;
#                        RobustFit (median-based) or MLE (closed-form on
#                        Gamma(1/σ²)) for InverseGamma.
#
# `PriorSpec(family, fitting)` bundles them. `fit_prior(spec, σ²s)` dispatches
# to the right (family, fitting) combination and returns an `IGPrior`. Illegal
# combinations (e.g. Jeffreys + MLE) error at dispatch time.

"""
    ScalePriorFamily

Abstract supertype for the scale-prior distribution family. Concrete subtypes
identify the *kind* of prior (its parameter shape and admissible values);
the *values* come from a [`PriorFitting`](@ref) applied to σ̂² data.
"""
abstract type ScalePriorFamily end

"""
    Jeffreys()

Improper Jeffreys prior — parameterless, equivalent to `IGPrior(0, 0)`.
"""
struct Jeffreys <: ScalePriorFamily end

"""
    InverseGamma()

Proper inverse-gamma prior `IG(α₀, β₀)`; hyperparameters come from a fitting.
"""
struct InverseGamma <: ScalePriorFamily end

"""
    PriorFitting

Abstract supertype for "how do we derive `(α₀, β₀)` from the per-pair σ̂²
data?" Concrete subtypes implement [`fit_prior`](@ref) for one or more
[`ScalePriorFamily`](@ref) values.
"""
abstract type PriorFitting end

"""
    Unfitted()

Parameterless families take `Unfitted` — there's nothing to fit.
"""
struct Unfitted <: PriorFitting end

"""
    RobustFit(alpha0 = 1.0)

Robust median-based fitting of an inverse-gamma prior:
`β₀ = α₀ · median(σ²s)` with fixed shape `α₀` (default 1.0).
Median is robust to heavy-tailed σ̂² distributions (HEB).
"""
struct RobustFit <: PriorFitting
    alpha0::Float64
end
RobustFit() = RobustFit(1.0)

"""
    MLE()

Closed-form MLE on the inverse-gamma family via the Gamma reparameterisation
`1/X ~ Gamma(α, 1/β)`. Requires `length(σ²s) ≥ 2`.
"""
struct MLE <: PriorFitting end

"""
    PriorSpec(family::ScalePriorFamily, fitting::PriorFitting)

Compound spec: the prior family + the algorithm that turns σ̂² data into its
parameters. The named specs are the constructor functions
[`JeffreysPriorSpec`](@ref) and [`IGRobustSpec`](@ref); the set a sweep
iterates is [`ALL_PRIOR_SPECS`](@ref).
"""
struct PriorSpec{F<:ScalePriorFamily, S<:PriorFitting}
    family::F
    fitting::S
end

"""Improper Jeffreys prior — equivalent to `PriorSpec(Jeffreys(), Unfitted())`."""
JeffreysPriorSpec() = PriorSpec(Jeffreys(), Unfitted())

"""Robust IG prior with shape `α₀` and scale fit from `median(σ̂²)`."""
IGRobustSpec(alpha0::Float64=1.0) = PriorSpec(InverseGamma(), RobustFit(alpha0))

# fit_prior dispatches on (family, fitting). Defined here at the header (no body
# yet — the InverseGamma methods need `fit_mle(Gamma, …)` from Distributions,
# imported later in the file).
function fit_prior end
"""
    prior_label(spec::PriorSpec) -> String

The display label of a legal prior spec: `"Jeffreys"`, `"IG-Robust"`, `"IG-MLE"`.
"""
function prior_label end

# Parameterless family — Jeffreys returns IGPrior(0, 0) ignoring data.
fit_prior(::Jeffreys, ::Unfitted, _) = IGPrior(0.0, 0.0)

# Illegal combinations — caught at dispatch time rather than via runtime checks.
fit_prior(::Jeffreys, fit::PriorFitting, _) =
    error("Jeffreys family is parameterless; got fitting=$(typeof(fit)). Use Unfitted().")
fit_prior(::InverseGamma, ::Unfitted, _) =
    error("InverseGamma family requires a non-trivial fitting (RobustFit or MLE).")

# Convenience: dispatch on a PriorSpec to delegate.
fit_prior(spec::PriorSpec, sigma2s) = fit_prior(spec.family, spec.fitting, sigma2s)

# Display labels — keyed on the (family, fitting) combination via PriorSpec
# so labels stay synchronised with the spec roster.
prior_label(::PriorSpec{Jeffreys,     Unfitted})     = "Jeffreys"
prior_label(::PriorSpec{InverseGamma, RobustFit}) = "IG-Robust"
prior_label(::PriorSpec{InverseGamma, MLE})       = "IG-MLE"

# =============================================================================
# Per-axis tags + bracket-key dispatch
# =============================================================================
# The two prior axes have their own tag accessors — `family_tag` and
# `fitting_tag` — so either axis can be identified independently.

"""
    family_tag(family::ScalePriorFamily) -> Symbol

Compact symbol identifying the prior family. `:jeffreys` (parameterless) or
`:ig` (inverse-gamma). Used in display + sidecar-key construction.
"""
family_tag(::Jeffreys)     = :jeffreys
family_tag(::InverseGamma) = :ig
family_tag(s::PriorSpec)   = family_tag(s.family)

"""
    fitting_tag(fitting::PriorFitting) -> Symbol

Compact symbol identifying the fitting algorithm. `:none` for parameterless
families, `:robust` for median-based, `:mle` for closed-form MLE.
"""
fitting_tag(::Unfitted)  = :unfitted
fitting_tag(::RobustFit) = :robust
fitting_tag(::MLE)       = :mle
fitting_tag(s::PriorSpec) = fitting_tag(s.fitting)

# NOTE: IG-Robust / IG-MLE dropped for the fair (oracle-free) comparison — both
# fit the IG prior from GT σ̂² (empirical Bayes), an asymmetric advantage the
# threshold baselines don't get.  Jeffreys (α₀=β₀=0) is the oracle-free IG: its
# scale is the data's own consensus RMS (RSS/ν), selected by pose-error
# free-argmin like every other method.  Re-add the two specs here to restore the
# empirical-Bayes variants (then label them as GT-annotated references).
"""
    ALL_PRIOR_SPECS :: Tuple of canonical (family, fitting) specs probed + evaluated.

Single source of truth for the prior specs a sweep iterates.
"""
const ALL_PRIOR_SPECS = (JeffreysPriorSpec(),)

"""
    fit_prior(::InverseGamma, ::MLE, σ²s) -> IGPrior

MLE of the inverse-gamma shape/scale (α, β). Distributions.jl does not
implement `fit_mle(InverseGamma, ·)` directly, so we use the closed-form
transformation

    X ~ InverseGamma(α, β)  ⇔  1/X ~ Gamma(shape = α, scale = 1/β)

and call `fit_mle(Gamma, 1 ./ x)`. Returns `nothing` for n < 2.
"""
function fit_prior(::InverseGamma, ::MLE, sigma2s)
    length(sigma2s) < 2 && return nothing
    g = fit_mle(Gamma, 1 ./ sigma2s)
    α = shape(g)
    β = 1 / scale(g)
    return IGPrior(α, β)
end

"""
    fit_prior(::InverseGamma, fit::RobustFit, σ²s) -> IGPrior

Robust empirical prior: fixed shape `α₀ = fit.alpha0`, scale `β₀ = α₀ · median(σ̂²)`.
The IG prior mean is `β₀/(α₀−1)`; with `α₀=1` the mean is undefined but the
mode `β₀/(α₀+1) = α₀ · median(σ̂²) / 2` anchors near the data median.

Used for HEB-like datasets where per-pair `σ̂²` estimates are heavy-tailed
(low-`n_in` pairs and non-planar GT-inlier sets contaminate MLE Gamma fits).
The median is unaffected by individual outlier `σ̂²` values, so the prior
location stays stable even with single-digit val-pool sizes.
"""
function fit_prior(::InverseGamma, fit::RobustFit, sigma2s)
    isempty(sigma2s) && return nothing
    β = fit.alpha0 * median(sigma2s)
    return IGPrior(fit.alpha0, β)
end
