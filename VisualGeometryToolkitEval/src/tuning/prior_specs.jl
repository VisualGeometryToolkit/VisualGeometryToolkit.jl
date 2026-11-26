# =============================================================================
# prior_specs.jl — the scale-PRIOR strategy layer of a tuning: the
# ScalePriorFamily × PriorFitting axes (`PriorSpec`, the closed-form algorithm
# that fits VisualGeometryToolkitCore's `IGPrior` from per-pair σ̂² samples,
# `family_tag`/`fitting_tag`, `prior_label`). Moved from
# RANSACScoringDoneRight's eval_strategy.jl (K+, 2026-10-09): it is tuning
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
# `PriorSpec(family, fitting)` bundles them, and is the estimation algorithm:
# `solve(EstimationProblem(IGPrior, σ²s), spec)` dispatches on the (family,
# fitting) combination and returns a FitResult whose `u` is the `IGPrior`.
# Illegal combinations (e.g. Jeffreys + MLE) are refused where the spec is built (ArgumentError).

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
data?" Each concrete subtype is one [`PriorSpec`](@ref)'s fitting for one or
more [`ScalePriorFamily`](@ref) values.
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

The scale-prior fit, a VisualGeometryToolkitCore `ClosedFormKind` algorithm: the prior
family and the fitting that turns σ̂² data into its parameters.
`solve(EstimationProblem(IGPrior, σ²s), spec)` returns a `FitResult` whose `u` is the fitted
`IGPrior` (VisualGeometryToolkitCore):

- `PriorSpec(Jeffreys(), Unfitted())` — `IGPrior(0, 0)`; the data are not read (`n = 0`; they may be `nothing`);
- `PriorSpec(InverseGamma(), RobustFit(α₀))` — fixed shape `α₀`, scale `β₀ = α₀ · median(σ̂²)`;
- `PriorSpec(InverseGamma(), MLE())` — the inverse-gamma MLE through the Gamma reparameterisation.

`n` is the number of σ̂² values a fit read, `rms` NaN. Too few data for a fitting (none for `RobustFit`, fewer than
two for `MLE`) is degenerate: `u = nothing`, `(InitialFailure, NoRule)`, `n = 0`. An illegal combination
(`Jeffreys` with a fitting, `InverseGamma` unfitted) is refused where the spec is built (an `ArgumentError`). An `IGPrior` problem has no
default algorithm: the spec is the experiment's decision.

"""
struct PriorSpec{F<:ScalePriorFamily, S<:PriorFitting} <: AbstractEstimationAlgorithm
    family::F
    fitting::S
    function PriorSpec(family::F, fitting::S) where {F<:ScalePriorFamily, S<:PriorFitting}
        family isa Jeffreys && !(fitting isa Unfitted) && throw(ArgumentError(
            "Jeffreys family is parameterless; got fitting=$(typeof(fitting)). Use Unfitted()."))
        family isa InverseGamma && fitting isa Unfitted && throw(ArgumentError(
            "InverseGamma family requires a non-trivial fitting (RobustFit or MLE)."))
        return new{F, S}(family, fitting)
    end
end
estimator_kind(::PriorSpec) = ClosedFormKind()

"""
    prior_label(spec::PriorSpec) -> String

The display label of a legal prior spec: `"Jeffreys"`, `"IG-Robust"`, `"IG-MLE"`.
"""
function prior_label end

# The fits, one `__solve` per (family, fitting). The result type of a fitting that can be degenerate.
const _MaybePrior = Union{IGPrior, Nothing}

# Parameterless family — Jeffreys returns IGPrior(0, 0) and reads no data.
__solve(::EstimationProblem{IGPrior}, ::PriorSpec{Jeffreys, Unfitted}) = closed_form_result(IGPrior(0.0, 0.0), 0)

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

# MLE of the inverse-gamma shape/scale (α, β). Distributions.jl does not
# implement `fit_mle(InverseGamma, ·)` directly, so we use the closed-form
# transformation
#
#     X ~ InverseGamma(α, β)  ⇔  1/X ~ Gamma(shape = α, scale = 1/β)
#
# and call `fit_mle(Gamma, 1 ./ x)`. Degenerate (`u = nothing`) for n < 2.
function __solve(prob::EstimationProblem{IGPrior}, ::PriorSpec{InverseGamma, MLE})
    sigma2s = prob.f
    length(sigma2s) < 2 && return closed_form_result(_MaybePrior, nothing, 0)
    g = fit_mle(Gamma, 1 ./ sigma2s)
    α = shape(g)
    β = 1 / scale(g)
    return closed_form_result(_MaybePrior, IGPrior(α, β), length(sigma2s))
end

# Robust empirical prior: fixed shape `α₀ = fit.alpha0`, scale `β₀ = α₀ · median(σ̂²)`.
# The IG prior mean is `β₀/(α₀−1)`; with `α₀=1` the mean is undefined but the
# mode `β₀/(α₀+1) = α₀ · median(σ̂²) / 2` anchors near the data median.
#
# Used for HEB-like datasets where per-pair `σ̂²` estimates are heavy-tailed
# (low-`n_in` pairs and non-planar GT-inlier sets contaminate MLE Gamma fits).
# The median is unaffected by individual outlier `σ̂²` values, so the prior
# location stays stable even with single-digit val-pool sizes.
function __solve(prob::EstimationProblem{IGPrior}, alg::PriorSpec{InverseGamma, RobustFit})
    sigma2s = prob.f
    isempty(sigma2s) && return closed_form_result(_MaybePrior, nothing, 0)
    β = alg.fitting.alpha0 * median(sigma2s)
    return closed_form_result(_MaybePrior, IGPrior(alg.fitting.alpha0, β), length(sigma2s))
end
