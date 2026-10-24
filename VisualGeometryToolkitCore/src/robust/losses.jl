# =============================================================================
# Robust-loss family — one loss, three facets (see LOSSES.md)
# =============================================================================
#
# A LOSS SHAPE is a normalized SCALE FAMILY of losses
#
#     ρ_w(r) = w²·ρ₁(r/w),
#
# and `loss_width` is its scale parameter w: the residual scale at which the
# loss leaves the quadratic regime every shape shares at the origin. In this
# file's convention (facet B below) ρ is written in the squared residual
# s = r² with ρ′(0) = w_max, so ρ₁(u) = w_max·u² + o(u²) (w_max = 1 for every
# shape but smooth-TLS's 2). The classical M-estimation normalization
# ρ₁(u) = u²/2 + o(u²) differs from it by a constant factor only — a factor
# that moves neither the minimizer nor the width (`cost_scale` carries the one
# shape, Welsch, whose tracked cost keeps the ½).
#
# Every closed form below takes the SQUARED width `sq_loss_width` = w², the
# quantity the formulas are written in; a deployed loss answers
# `sq_loss_width(loss)` and `loss_width(loss)` itself.
#
# The same family is also written ρ(r) = Δ·g(r²/w²): a NORMALIZED SHAPE g
# (sup g = 1 for bounded shapes) and a depth Δ (log-contrast; the density floor
# is e^{−Δ}). The three historical taxonomies are facets of this family under
# different parameter conventions:
#
#   A. M-loss: coupled dials, Δ = w²/(2g′(0)) so ρ″(0) = 1. Historical — its
#      types no longer exist in this package (see LOSSES.md).
#   B. Geman–Reynolds dual (the half-quadratic lift): ρ′(0) = w_max
#      convention, Δ = w_max·w²/g′(0). The lift_* verbs below implement it;
#      VisualGeometryToolkitEstimation's lift envelope tests are their numerical gate.
#   C. Inlier density (RANSACScoringDoneRight inlier_potentials.jl): the
#      unified family itself — Δ swept, e^{−ρ} normalized over a truncated
#      domain by Z̃. RSDR's formula layer defines its methods ON these tags.
#
# Two INDEPENDENT traits (they differ — Cauchy is the witness):
#   liftable(shape) — finite w_max AND closed-form GR dual shipped here.
#   bounded(shape)  — sup g = 1 (ρ(∞) = Δ < ∞): redescends to a floor, hence NOT
#                     normalizable on ℝ (evidence only via the truncated Z̃; the
#                     associated weight policy is never a NormalizedWeightPolicy).
# =============================================================================

"""
    AbstractLossShape

Tag-type root of the unified robust-loss family: a normalized scale family
`ρ_w(r) = w²·ρ₁(r/w)` whose scale parameter `w` is the [`loss_width`](@ref),
the residual scale at which the loss leaves the shared quadratic regime
(see LOSSES.md and the header of robust/losses.jl). A shape carries no width;
[`deploy`](@ref) gives it one.
"""
abstract type AbstractLossShape end

# The shape-math extension point. Unexported (these are not names a caller of
# the one-argument verbs ever needs) but PUBLIC: a downstream shape declares
# itself by adding methods to them.
public ρ, ψ, weight, conjugate, conjugate_curvature, zblock, cost, cost_scale, band_signal

# Unexported (`sq_loss_width` and `loss_width` are also natural keyword and
# variable names at every call site) but intended API: VisualGeometryToolkitEstimation
# imports `sq_loss_width` by name.
public sq_loss_width, loss_width
# The χ²_d median behind every median-seeded scale (ŝ² = median(r²)/chisq_median(d));
# RANSACScoringDoneRight imports it by name.
public chisq_median

"""
    AbstractMEstimator <: AbstractLossShape

A robust loss whose WIDTH is set by asymptotic efficiency at the Gaussian.
Subtyping this is a promise, and the promise costs exactly one method.

# Interface

A subtype MUST define one method:

    efficiency_constant(::Type{MyShape}) -> Float64

its tuning constant at codimension 1 — the solution of `eff(m, 1) = 0.95` for
that shape's psi. It takes the TYPE, not an instance: the constant is a
property of the shape, and a deployment is built FROM it. Failing to define it
is a `MethodError` on the first call, deliberately NOT caught by a catch-all,
so `hasmethod(efficiency_constant, (Type{MyShape},))` answers honestly.

From that single number every other method is inherited and guaranteed:

| inherited                         | what it gives                              |
|-----------------------------------|--------------------------------------------|
| `efficiency_constant(K, d)`       | the constant at codimension `d`             |
| `codimension_factor(K, d)`        | its ratio to the d = 1 constant             |
| `sq_loss_width(loss)`           | `scale_multiple^2 * s2`, the squared width it runs at    |
| `loss_width(loss)`                   | `scale_multiple * sqrt(s2)`, derived                     |

Note `sq_loss_width`/`loss_width` are methods of a DEPLOYMENT (`deploy(shape, s2; scale_multiple)`),
not of the shape: a bare shape has no width.

The noise level is an INPUT. This interface turns one into a width; it does
not estimate the noise, and nothing here should. How a strategy obtains its
scale is a separate axis with its own vocabulary — see `ScaleInterface`
(`ConsensusAnchored`, `FixedScale`, `FollowsScorer`, `SelfScaling`, `NoScale`) in
robust/estimation.jl — and the estimator is a real choice (MAD, RMS,
median-of-norms, Qn, …) that belongs to the caller, not to the loss.

A subtype MAY also define `_solved_constants(::Type{MyShape})`, a tuple of the
constants for `d = 1, 2, …` (entry 1 equal to its `efficiency_constant`) at
each codimension where `eff(m, d) = 0.95` has actually been SOLVED, replacing
the generic chi lift `sqrt((d+2)/3)` with those solved numbers. It is an
optimisation of accuracy, never a requirement — see the note under
`codimension_factor`.

# Who is deliberately NOT here

Losses that efficiency does not parameterise stay directly under
`AbstractLossShape`, and calling `loss_width` on one is a `MethodError` by design:
`GaussianShape` has no width to set, `TLSShape` is a hard truncation with no
influence function, and `BarronShape`'s shape is itself fitted — deriving a
fixed constant for it would contradict the fit.

`SmoothTLSShape` is the one shape that sits outside this hierarchy yet DOES
carry constants. It is a truncation, so it ships no default `scale_multiple` and subtyping
`AbstractMEstimator` would wrongly promise one — but `eff(m, d) = 0.95` is
genuinely solvable for its psi, so `efficiency_constant(SmoothTLSShape)` and
its whole solved column exist and are supported. Read the subtype relation as
"has a default `scale_multiple`", NOT as "has a constant": a caller wanting the
efficiency-matched truncation asks `efficiency_constant` directly. This is why
`efficiency_constant(K, d)` and `codimension_factor(K, d)` below take
`Type{<:AbstractLossShape}` and not `Type{<:AbstractMEstimator}`.
"""
abstract type AbstractMEstimator <: AbstractLossShape end

"Gaussian/L2 shape g(u) = u — unbounded, the Δ→∞ anchor of the depth sweep."
struct GaussianShape <: AbstractLossShape end

"Welsch shape g(u) = 1 − e^{−u} — Gaussian core, soft exponential tail."
struct WelschShape <: AbstractMEstimator end

"Geman–McClure shape g(u) = u/(1+u) — steep core, soft polynomial tail."
struct GemanMcclureShape <: AbstractMEstimator end

"Cauchy shape g(u) = ln(1+u) — unbounded (ρ(∞)=∞) yet liftable (finite w_max)."
struct CauchyShape <: AbstractMEstimator end

"Smooth-TLS shape g(u) = 2u − u² capped at 1 — Gaussian core, hard cutoff."
struct SmoothTLSShape <: AbstractLossShape end

"""
    TLSShape <: AbstractLossShape

HARD truncated-least-squares tag: ρ(r) = min(r², c²) — the discontinuous-influence
truncation (shape g(u) = min(u, 1)). Facet tag only: NOT liftable (the truncated
quadratic has no Geman–Reynolds dual — its graduation is the Black–Rangarajan
outer bound, `gnc_tls_weight` below, not a width anneal), so it ships no lift_*
closed forms. Names the objective of the truncated/GNC-TLS strategy vocabulary
(e.g. `Gnc{Irls{TLSShape}}` in VisualGeometryToolkitEstimation).
"""
struct TLSShape <: AbstractLossShape end

"""
    HuberShape <: AbstractLossShape

Huber tag: quadratic core with a LINEAR tail past the elbow — monotone ψ
(influence constant, never vanishing, beyond the elbow), hence unbounded and
not redescending; not liftable (w_max is finite but the GR dual adds nothing
over plain IRLS). Facet tag only — names the objective of the static-IRLS
strategy vocabulary (`Irls{HuberShape}`).
"""
struct HuberShape <: AbstractMEstimator end

"""
    liftable(shape::AbstractLossShape) -> Bool

Whether the loss has a finite inlier-limit weight `w_max = ω(0)` and ships
the closed-form Geman–Reynolds dual (`lift_conjugate`/`lift_zblock`) — i.e. the
joint-(θ,z) half-quadratic lift machinery applies. NOT the same axis as
[`bounded`](@ref): Cauchy is liftable but unbounded.
"""
liftable(::AbstractLossShape) = false
liftable(::WelschShape) = true
liftable(::GemanMcclureShape) = true
liftable(::CauchyShape) = true
liftable(::SmoothTLSShape) = true

"""
    bounded(shape::AbstractLossShape) -> Bool

Whether the shape is bounded (`sup g = 1`, so ρ(∞) = Δ): the potential
redescends to a positive density floor `e^{−Δ}` and is therefore NOT
normalizable on an unbounded domain — usable as evidence only through a
truncated normalizer Z̃, and never likelihood-grade at frozen scale (aligns
with the `NormalizedWeightPolicy` boundary in robust/estimation.jl).
"""
bounded(::AbstractLossShape) = false
bounded(::WelschShape) = true
bounded(::GemanMcclureShape) = true
bounded(::SmoothTLSShape) = true
bounded(::TLSShape) = true   # ρ(∞) = c² (hard truncation)

# -----------------------------------------------------------------------------
# Facet B — the Geman–Reynolds dual (half-quadratic lift closed forms).
# Every redescending loss below admits
#
#     ρ(r) = min_{w ∈ (0, w_max]} [ w·r² + conjugate(w) ],        s ≡ r²,
#
# solved by the loss's own IRLS weight w*(s) = ω(s) = ρ′(s), w_max = ω(0).
# The lift substitutes w = w_max·z² (w̄ = z² ∈ (0,1]) and runs a joint-(θ,z)
# Gauss-Newton on (√w̄·rᵢ, κ(w̄)), κ²(w̄) = conjugate(w_max·w̄)/w_max.
#
# Closed forms (s = r²; in these formula comments c is the loss width, c² =
# `sq_loss_width`, c̄² the target one; the conjugate is ρ(s(w)) − w·s(w)):
#
#   Welsch          ρ = c²(1−e^{−s/c²})      ω = e^{−s/c²}          w_max = 1
#                   s(w) = −c²·ln w           conjugate = c²(w·ln w − w + 1)
#   Geman–McClure   ρ = c²s/(c²+s)           ω = c⁴/(c²+s)²         w_max = 1
#                   s(w) = c²(w^{−1/2}−1)     conjugate = c²(√w − 1)²
#   Cauchy          ρ = c²·ln(1+s/c²)        ω = 1/(1+s/c²)         w_max = 1
#                   s(w) = c²(1/w − 1)        conjugate = c²(w − 1 − ln w)
#   smooth-TLS      ρ = 2s−s²/c² (s≤c²), c²  ω = 2(1−s/c²)₊         w_max = 2
#                   s(w) = c²(1−w/2)          conjugate = c²(1−w/2)²
# -----------------------------------------------------------------------------

"""
    lift_wmax(loss) -> Float64

Inlier-limit weight `w_max = ω(0)` of the loss's IRLS weight function — the
upper end of the Geman–Reynolds domain `w ∈ (0, w_max]`. The lifting variable
is normalized: `w̄ = z² = w/w_max ∈ (0, 1]`.
"""
lift_wmax(::WelschShape) = 1.0
lift_wmax(::GemanMcclureShape) = 1.0
lift_wmax(::CauchyShape) = 1.0
lift_wmax(::SmoothTLSShape) = 2.0

# PRECISION. The shape math below is generic in the argument's float type:
# `ρ(shape, 1f0, 1f0)` is Float32, and a Float64 call is bit-identical to the
# historical Float64-literal forms (integer literals convert exactly, and
# `w / 2` is the same correctly-rounded halving as `0.5 * w`). Where a shape
# needs a CONSTANT of the computation's type — a constant weight, a Barron `α`
# stored as Float64, a series coefficient — `_shape_float` names it: the promoted float
# type of the arguments, but Float64 for anything that is not an
# `AbstractFloat` (a ForwardDiff `Dual`, an integer), so those paths keep the
# plain Float64 constants they always had.
_shape_float(::Type{T}) where {T <: AbstractFloat} = T
_shape_float(::Type) = Float64
@inline _shape_float(x, y) = _shape_float(promote_type(typeof(x), typeof(y)))

"""
    ρ(loss, s, sq_loss_width) -> Float64

Loss value ρ at squared residual `s = r²`, scale `sq_loss_width = c²` — the LOSS a
residual pays, in the convention `ρ′(0) = w_max`.

TOTAL over the family. For the liftable shapes ρ is also the Geman–Reynolds
envelope `min_w [w·s + conjugate(w)]`, but nothing about a loss REQUIRES a
lift — `GaussianShape` is `s`, `HuberShape` is the monotone elbow, `TLSShape`
is the hard truncation `min(s, sq_loss_width)`. Scoping ρ to the liftable ones made
`ρ(GaussianShape(), …)` throw while `ψ(GaussianShape(), …)` answered, which
is incoherent: ψ is ρ′. Only [`conjugate`](@ref) and [`zblock`](@ref) stay
lift-only, because only they need the auxiliary variable to exist.

The bounded shapes saturate at `sq_loss_width` — the well depth — so `1 - ρ/sq_loss_width` is a
normalised quality in [0, 1]. `GaussianShape` and `HuberShape` are unbounded
and have no depth.
"""
ρ(::WelschShape, s, sq_loss_width) = -sq_loss_width * expm1(-s / sq_loss_width)
ρ(::GemanMcclureShape, s, sq_loss_width) = sq_loss_width * s / (sq_loss_width + s)
ρ(::CauchyShape, s, sq_loss_width) = sq_loss_width * log1p(s / sq_loss_width)
ρ(::SmoothTLSShape, s, sq_loss_width) = s <= sq_loss_width ? 2 * s - s * s / sq_loss_width : sq_loss_width
# Not liftable, but still losses. TLS is the hard truncation the smooth one
# approximates; its ρ′ jumps at sq_loss_width, which is exactly why it has no lift.
ρ(::GaussianShape, s, sq_loss_width) = float(s)
ρ(::TLSShape, s, sq_loss_width) = min(float(s), float(sq_loss_width))
ρ(::HuberShape, s, sq_loss_width) = s <= sq_loss_width ? float(s) : 2 * sqrt(sq_loss_width * s) - sq_loss_width
# NO CATCH-ALL, deliberately — the same policy `efficiency_constant` states above
# and that `weight`, `conjugate` and `zblock` already follow. A shape that does
# not define ρ must be a `MethodError` at dispatch, so `hasmethod(ρ, (MyShape,
# Float64, Float64))` answers honestly and the failure names the missing method.
# An `ArgumentError` catch-all here made `hasmethod` lie and gave a downstream
# shape one error type from ρ and another from its siblings.

"""
    conjugate(loss, w, sq_loss_width) -> Float64

Geman–Reynolds penalty conjugate(w) on the UNNORMALIZED weight `w ∈ (0, w_max]`, so
that `ρ(r) = min_w [w·r² + conjugate(w)]` (verified numerically in the RVG tests).

!!! note "Two psis, and this is not the one"
    The half-quadratic literature (Geman–Reynolds, Black–Rangarajan) writes
    this penalty ψ; M-estimation writes ψ for the INFLUENCE FUNCTION
    `ρ′(r)`. This package needs both, so the influence function keeps the
    symbol — it is the one `efficiency_constant` is solved against — and the
    penalty is named for what it IS: the Legendre conjugate of ρ in `w`.

    Tell them apart by the argument: `conjugate(shape, w, sq_loss_width)` takes a WEIGHT,
    [`ψ`](@ref) and [`weight`](@ref) take a residual and a squared residual.
"""
conjugate(::WelschShape, w, sq_loss_width) = sq_loss_width * (xlogx(w) - w + 1)   # xlogx(0) = 0: the limit, not NaN
conjugate(::GemanMcclureShape, w, sq_loss_width) = sq_loss_width * (sqrt(w) - 1)^2
conjugate(::CauchyShape, w, sq_loss_width) = sq_loss_width * (w - 1 - log(w))
conjugate(::SmoothTLSShape, w, sq_loss_width) = sq_loss_width * (1 - w / 2)^2

"""
    conjugate_curvature(loss, w, sq_loss_width) -> Float64

`w·κ²″(w)`, the scaled second derivative of the Geman–Reynolds
[`conjugate`](@ref) `κ²(w)` in the unnormalized weight `w` — the z-block
curvature of a Newton lift in `w = z²`. Since `κ²′(w) = −s(w)` (`s` the squared
residual whose weight is `w`), it is `−w/ω′(s(w))`. Closed forms, each pinned
against a central difference of `conjugate`:

| shape | `κ²(w)` | `w·κ²″(w)` |
|---|---|---|
| Welsch | `c²(w ln w − w + 1)` | `c²` |
| Geman–McClure | `c²(√w − 1)²` | `c²/(2√w)` |
| Cauchy | `c²(w − 1 − ln w)` | `c²/w` |
| smooth TLS | `c²(1 − w/2)²` | `c²w/2` |
| Barron(α) | see [`BarronShape`](@ref) | `2c²·w^{2/(α−2)}` |

Defined for the liftable shapes only, as `conjugate` is.
"""
conjugate_curvature(::WelschShape, w, sq_loss_width) = sq_loss_width * one(w)
conjugate_curvature(::GemanMcclureShape, w, sq_loss_width) = sq_loss_width / (2 * sqrt(w))
conjugate_curvature(::CauchyShape, w, sq_loss_width) = sq_loss_width / w
conjugate_curvature(::SmoothTLSShape, w, sq_loss_width) = sq_loss_width * w / 2

"""
    weight(loss, s, sq_loss_width) -> Float64

IRLS weight `ω(s) = dρ/ds` (unnormalized; `ω(0) = w_max`). TOTAL over the
family, like [`ρ`](@ref) it differentiates. For the LIFTABLE shapes it is
additionally the argmin of the Geman–Reynolds envelope at `s`, and the
weight-from-z map of the lift is `w = w_max·z²`; for the other three there is
no envelope, only the derivative.
"""
weight(::WelschShape, s, sq_loss_width) = exp(-s / sq_loss_width)
weight(::GemanMcclureShape, s, sq_loss_width) = (sq_loss_width / (sq_loss_width + s))^2
weight(::CauchyShape, s, sq_loss_width) = sq_loss_width / (sq_loss_width + s)
# The clamp is an `ifelse`, not `max`: Julia's `max` propagates NaN at the cost
# of several ops per call, which the GPU scorers pay per residual. A NaN `s`
# therefore gets weight 0, an OUTLIER, the stack's rule for a non-finite
# residual; every finite `s` is bit-identical to the `max` form.
weight(::SmoothTLSShape, s, sq_loss_width) = (v = 2 * (1 - s / sq_loss_width); ifelse(v > zero(v), v, zero(v)))
weight(::GaussianShape, s, sq_loss_width) = one(_shape_float(s, sq_loss_width))
weight(::TLSShape, s, sq_loss_width) = (T = _shape_float(s, sq_loss_width); s < sq_loss_width ? one(T) : zero(T))
weight(::HuberShape, s, sq_loss_width) = s <= sq_loss_width ? one(_shape_float(s, sq_loss_width)) : sqrt(sq_loss_width / s)

"""
    ψ(loss, r, sq_loss_width) -> Float64

INFLUENCE FUNCTION ρ′(r) at residual `r` (NOT squared) and scale `sq_loss_width = c²` —
ψ in the M-estimation sense (Huber, Hampel, Maronna–Martin–Yohai).

This is the function the width contract is defined against: a shape's
[`efficiency_constant`](@ref) is the `c` solving `eff(m, 1) = 0.95` for

    eff(m, d) = E[ψ′ + (d-1)ψ/r]² / (d · E[ψ²]),   r ~ chi_d

so the solved table (Welsch 2.9846, Geman-McClure 3.7874, Cauchy 2.3849, …)
is a claim ABOUT THIS FUNCTION, and exposing it is what makes that claim
checkable rather than a comment.

Derived, never a second closed form per shape, via the defining relation of
IRLS: the weight IS `w(r) = ψ(r)/r`, so `ψ(r) = r·w(r)`. Built directly on
[`weight`](@ref), which covers the whole family.

Constant factors do not matter here: `eff` is invariant under
`ψ → αψ` (the numerator is squared, the denominator is `E[ψ²]`), which is why
the table holds despite `ρ` carrying this package's depth convention rather
than the classical one.

Not to be confused with [`conjugate`](@ref), the half-quadratic penalty that
the Geman–Reynolds literature also calls ψ — that one takes a WEIGHT.
"""
@inline ψ(shape::AbstractLossShape, r, sq_loss_width) = r * weight(shape, r * r, sq_loss_width)

# =============================================================================
# Loss width — DERIVED from the noise, not tuned
# =============================================================================
#
# Every M-estimator cuts at a WIDTH, and that width factors into three
# independent parts:
#
#     width = efficiency_constant(loss) * codimension_factor(loss, d) * sigma
#
# Naming them separately is the whole point, because only ONE of the three is
# data:
#
#   sigma                the noise.                       ESTIMATED per problem.
#   efficiency_constant  the loss's identity at d = 1.   DERIVED, once, here.
#   codimension_factor   the residual's geometry.          DERIVED from d.
#
# So a width is never a hyperparameter. Cross-validating one is cross-validating
# a noise estimate wearing a disguise, and it silently trades off against the
# outlier half-width, which IS a real degree of freedom.
#
# THE CONSTANT is fixed by asymptotic efficiency at the Gaussian: it solves
#
#     eff(m, d) = E[psi'(r) + (d-1) psi(r)/r]^2 / (d * E[psi(r)^2]) = 0.95,
#     r ~ chi_d
#
# which reduces to the classic (E psi')^2 / E psi^2 at d = 1.
#
# THE CODIMENSION FACTOR exists because a d-dimensional residual is scored
# through its NORM, whose square is chi^2_d, and that distribution's mass moves
# outward with d (mean ~ sqrt(d)) — the same multiple of sigma therefore retains
# a different fraction of the inlier population at d = 2 than at d = 1. It is
# the same correction the scale seeding already carries as the chi^2_d median
# (`chisq_median`), and as 1.1774 (Rayleigh) rather than 0.6745 (half-normal).
#
# THE CONSTANTS ARE PARAMETERISATION-DEPENDENT, which is a live trap: published
# Welsch constants appear as BOTH 2.985 and 2.110, because exp(-r^2/c^2) and
# exp(-r^2/2c^2) are both in use and 2.9846/sqrt(2) = 2.1104. The numbers below
# are for THIS FILE'S parameterisation (`lift_weight` above) and must be
# re-derived if a loss's form ever changes.
#
# PROVENANCE: solved numerically from the integral above. Cauchy (2.3849) and
# Welsch (2.9846) reproduce the published constants to four figures, which is
# what validates the machinery; Huber carries its published 1.345 rather than
# the solver's 1.3371, the 0.6% being quadrature error at psi's kink.
# Geman-McClure has no widely published constant — 3.7874 is derived here and
# cross-checks against nothing, so treat it as the weakest entry.

"""
    efficiency_constant(K::Type{<:AbstractMEstimator}) -> Float64
    efficiency_constant(K::Type{<:AbstractLossShape}, codim::Integer) -> Float64

The loss's width as a MULTIPLE OF THE NOISE SIGMA. The one-argument form is
the shape's own constant at codimension 1 — a property of its psi, like its
influence function — and is what a loss must define to join this interface.
The two-argument form lifts it to codimension `d` through `codimension_factor`.

TAKES THE TYPE, not an instance. The constant is a property of the SHAPE, and a
deployed loss carries `scale_multiple` as a field — so an instance method would be
circular: you would need a loss to obtain the constant used to build one.
(Contrast `lift_*`, which must take an instance: `BarronShape`'s `α` is a
field, so its type carries strictly less information than its value.)

The DEFAULT for a deployed `scale_multiple`, not the only legitimate value. It solves
`eff(m, d) = 0.95` at the GAUSSIAN — the width that minimises variance
inflation on CLEAN data. A stage running under real contamination may
legitimately cross-validate its own operating point, and will land tighter:
VisualGeometryToolkitEstimation deploys a scale multiple ≈ 1.6 per model class where this returns ≈ 3.25,
trading efficiency for breakdown. Both are right for their stage.

Defined only for losses that efficiency parameterises: `GaussianShape` has no width and `BarronShape`'s shape is itself fitted, so
neither has one; those raise a `MethodError` by design. `TLSShape` likewise.
`SmoothTLSShape` is a truncation rather than an influence function and ships
no default `scale_multiple` in its constructor, but `eff(m, d) = 0.95` IS solvable for its
psi, so it does carry a tabulated constant for a caller that wants the
efficiency-matched truncation.
"""
function efficiency_constant end

# The SOLVED constants, carried verbatim as the numbers the integral returned.
#
# PROVENANCE (2026-09-20): eff(m, d) = 0.95 solved by bisection on adaptive
# Gauss-Kronrod quadrature (QuadGK, rtol 1e-12, order 15) over r ~ chi_d, with
# an explicit breakpoint at the solved multiple so the kink in Huber's and smooth-TLS's psi is not
# integrated across. The derivation script is scratch, not a package dep.
#
# VALIDATION: at d = 1 it returns Huber 1.3450, Cauchy 2.3849, Welsch 2.9846 and
# GM 3.7874 — the published constants to four figures, which is what validates
# the machinery. The breakpoint matters: the same solver without it reproduces
# only 1.3371 at d = 1 against a published 1.345.
#
# A loss with no entry at `d` inherits the chi lift in `efficiency_constant`.
# Prefer a solved value: at d = 2 the lift gives Welsch 3.4463 against the solved
# 3.2510 (6% high), and the gap widens with d.
#
# ONE tuple per shape, indexed by `d`. `efficiency_constant(K, d)` takes `d` at
# RUNTIME (RSDR calls it with a runtime codimension), so a tuple index is as
# fast as any per-`d` dispatch, and extending to d = 7 is one number per row.
#
# Smooth-TLS is a truncation, not an influence function, so it ships no default
# `scale_multiple` in its constructor — but eff(m, d) = 0.95 is still solvable for its psi,
# and the solved values are tabulated for a caller that wants the
# efficiency-matched truncation.
_solved_constants(::Type{<:AbstractLossShape}) = nothing

#                                                d = 1   d = 2   d = 3   d = 4   d = 5   d = 6
_solved_constants(::Type{HuberShape}) = (1.345, 1.5011, 1.628, 1.7311, 1.812, 1.8693)
_solved_constants(::Type{CauchyShape}) = (2.3849, 2.5486, 2.6656, 2.7516, 2.8154, 2.8624)
_solved_constants(::Type{WelschShape}) = (2.9846, 3.251, 3.4691, 3.6552, 3.8186, 3.9646)
_solved_constants(::Type{GemanMcclureShape}) = (3.7874, 4.0956, 4.3373, 4.5354, 4.7024, 4.846)
_solved_constants(::Type{SmoothTLSShape}) = (3.6732, 4.0231, 4.3204, 4.5826, 4.819, 5.0358)

# The d = 1 constant of every tabulated shape is its table's first entry. The
# method names the shapes explicitly (no catch-all), so `hasmethod` stays honest
# for a shape without a table.
efficiency_constant(::Type{K}) where {
    K <: Union{HuberShape, CauchyShape, WelschShape, GemanMcclureShape, SmoothTLSShape},
} = _solved_constants(K)[1]

function efficiency_constant(K::Type{<:AbstractLossShape}, d::Integer)
    d == 1 && return efficiency_constant(K)
    solved = _solved_lookup(K, d)
    solved === nothing || return solved
    return efficiency_constant(K) * sqrt((Int(d) + 2) / 3)
end

# `d` indexes the tuple directly. A shape with no table, or a `d` past its last
# solved column, returns `nothing` and inherits the chi lift.
@inline function _solved_lookup(K::Type{<:AbstractLossShape}, d::Integer)
    t = _solved_constants(K)
    (t === nothing || d < 1 || d > length(t)) && return nothing
    return @inbounds t[d]
end

"""
    codimension_factor(K::Type{<:AbstractLossShape}, codim::Integer) -> Float64

How much wider the width must be for a `codim`-dimensional residual than for a
scalar one, at equal efficiency: `efficiency_constant(K, d) / efficiency_constant(K)`,
and `1.0` at codimension 1 by construction.

DERIVED, not stored — `efficiency_constant(K, d)` is the primitive, because the
constant is what gets solved. Where `eff(m, d) = 0.95` has actually been solved
the ratio comes out of that solved number; everywhere else it is the
chi-distribution rule `sqrt((d+2)/3)`, which is exact at d = 1, a good
approximation elsewhere, and makes a new loss work from its 1-D constant
alone. The solved values run 3.8-7.5% under the generic 1.1547 at d = 2
(Huber 1.1109, GM 1.0814, Welsch 1.0893, Cauchy 1.0686).
"""
codimension_factor(K::Type{<:AbstractLossShape}, d::Integer) =
    efficiency_constant(K, d) / efficiency_constant(K)

# =============================================================================
# ONE convenience scale estimator — NOT part of the interface above
# =============================================================================
#
# A loss width takes sigma as an input, and the choice of estimator is
# the caller's: MAD, RMS, a median of norms, Qn/Sn, an IRLS-updated scale, or a
# value carried in from calibration (`FixedScale`). The two functions below are
# the median-of-norms option, provided because it is the one that pairs with the
# chi lift in `codimension_factor`. They are deliberately UNEXPORTED — call them
# qualified — so that reaching for them is a decision rather than a default.

"""
    chisq_median(codim::Integer) -> Float64

Median of `chi^2_d`, for ANY codimension: `0.4549` at d = 1, `2 log 2 = 1.3863`
at d = 2. Its square root is `0.6745` at d = 1 and `1.1774` (the Rayleigh
median) at d = 2.

`0.6745` is `Phi^-1(0.75)`, the SAME constant that scales a MAD — but sharing a
constant is not being the same estimator; see `noise_sigma`.

Answers for every `d` on purpose — a hand-tabulated d ∈ {1, 2} would throw on
the very call pattern this interface exists for, `noise_sigma(norms,
codimension(problem))`, where a hand-eye problem reports d = 6.
"""
chisq_median(d::Integer) = Distributions.quantile(Distributions.Chisq(d), 0.5)

"""
    noise_sigma(norms, codim::Integer) -> Float64

Robust noise sigma from residual NORMS: `median(norms) / sqrt(chisq_median(d))`.
If `r ~ N(0, sigma^2 I_d)` then `norm(r)/sigma ~ chi_d`, so dividing the median
norm by that distribution's median inverts to `sigma`.

NOT A MAD, though it shares MAD's `0.6745` at d = 1. A MAD centres first —
`median(abs(x - median(x)))` — so a bias cancels; this takes the median of the
norms THEMSELVES and has no centring step. That is the right choice for
residuals, which are already deviations from a fitted model (and for d >= 2 a
norm is non-negative, so there is no centre to subtract). The cost is that it
measures BIAS AS NOISE: on residuals centred at 0.9 with sigma = 0.37 it
returns 1.33 where a MAD still returns 0.37. Feed it residuals of a model that
fits, not of one that is offset.

Robust to OUTLIERS, which is the point — the median has 50% breakdown, so an
RMS would be inflated by exactly the gross errors the loss exists to reject.

The codimension is not optional. A 2-D residual's norm is Rayleigh, so its
median is `1.1774 sigma`; a scalar residual's is `0.6745 sigma`. Reading 2-D
norms as d = 1 returns 1.745x the true sigma, and every downstream width
inherits it.
"""
noise_sigma(norms, d::Integer) = Statistics.median(norms) / sqrt(chisq_median(d))

# =============================================================================
# CONSTRUCTING a deployed loss, and reading its width
# =============================================================================
#
# A loss is a SHAPE plus the width it runs at, and the width factors as
#
#     loss_width = scale_multiple * sigma          (the TARGET width)
#
# with `scale_multiple` the dimensionless multiple of the noise scale (a
# DEPLOYMENT choice, defaulting to the shape's efficiency constant) and `sigma`
# the noise scale (DATA, always supplied by the caller — this package never
# estimates it; see `ScaleInterface` in robust/estimation.jl for the axis that
# decides how).
#
# The two are stored SEPARATELY, never their product: a `SelfScaling` strategy
# re-estimates sigma every iteration while `scale_multiple` is fixed, so
# `rescale` replaces one field and leaves the other alone. Collapsed into a
# width you would have to recover `scale_multiple` to do it.

"""
    deploy(shape, s2; codim = 1, scale_multiple = efficiency_constant(typeof(shape), codim))

A SHAPE at the width it runs at — the pairing every consumer was assembling by
hand. Returns a [`DeployedLoss`](@ref) whose [`loss_width`](@ref) is
`scale_multiple * sqrt(s2)`.

`s2` is the noise VARIANCE (sigma squared), not sigma. The whole robust layer
speaks squared scale — `LossWidth(m, s2)`, `FixedS2(s2)`, `scale_s2(cs)`, and
every shape closed form takes a squared width — so storing the square is what
makes `sq_loss_width` exact: `scale_multiple^2 * s2` is the arithmetic the call
sites already do, bit for bit. Storing a linear sigma instead would force a
sqrt in and a square out, and the two disagree in the last ulps on 61% of
inputs.

It is the CALLER's: this package does not estimate it (see `ScaleInterface` in
robust/estimation.jl for the axis that decides how). From a linear sigma, square
it at the call: `deploy(shape, noise_sigma(norms, 2)^2; codim = 2)`.

`scale_multiple` is the width as a dimensionless multiple of the noise scale,
defaulting to the shape's efficiency constant at `codim` and overridden by a
caller that cross-validates its own operating point.

The two are kept APART, never multiplied into a stored width: a `SelfScaling`
strategy re-estimates the variance every iteration while `scale_multiple` is
fixed, so [`rescale`](@ref) replaces one and leaves the other.

```julia
deploy(WelschShape(), 0.37^2; codim = 2)                         # from efficiency -> 3.2510
deploy(WelschShape(), 0.37^2; codim = 2, scale_multiple = 1.597) # a CV'd operating point
```
"""
deploy(
    shape::AbstractMEstimator, s2::Real; codim::Integer = 1,
    scale_multiple::Real = efficiency_constant(typeof(shape), codim)
) =
    Attributed(shape, (scale_multiple = Float64(scale_multiple), s2 = Float64(s2)))

# A shape with no efficiency constant — a truncation, or Barron's fitted alpha —
# has nothing to default `scale_multiple` from, so it is required rather than
# raising a MethodError from inside a default argument.
deploy(shape::AbstractLossShape, s2::Real; scale_multiple::Real) =
    Attributed(shape, (scale_multiple = Float64(scale_multiple), s2 = Float64(s2)))

"""
    DeployedLoss

A shape together with the width it runs at: `Attributed(shape, (scale_multiple,
s2))`, the multiple of the noise scale and the noise VARIANCE. This is the
homotopy's TARGET — [`anneal`](@ref) wraps one to move along the path, and
[`target_loss`](@ref) unwraps back to it.

`isbits` and 16 bytes: the shape is a zero-byte tag, so a deployment is two
Float64s and costs a stack write.
"""
const DeployedLoss = Attributed{<:AbstractLossShape, <:NamedTuple{(:scale_multiple, :s2)}}

"""
    AnnealedLoss

A [`DeployedLoss`](@ref) wearing its position on the graduation path — what
[`anneal`](@ref) returns. Nests, so [`target_loss`](@ref) is an unwrap.
"""
const AnnealedLoss = Attributed{<:DeployedLoss, <:NamedTuple{(:μ,)}}

"""
    AnyLoss

A loss that KNOWS ITS OWN WIDTH — deployed, or deployed and annealed.

The one-argument verbs below are written against this, ONCE. Every one of them
is `<shape math>(loss_shape(loss), …, sq_loss_width(loss))`, and both members
answer those two accessors, so nothing in a verb body is per-type.

A future wrapper (censored, per-point-scaled) joins the family by defining
`loss_shape` and `sq_loss_width`, not by copying nine one-liners.
"""
const AnyLoss = Union{DeployedLoss, AnnealedLoss}

"The shape a [`DeployedLoss`](@ref) deploys — `g`, without a width."
loss_shape(loss::DeployedLoss) = loss.value
loss_shape(shape::AbstractLossShape) = shape
"The noise variance `s2` a [`DeployedLoss`](@ref) runs at (see [`scale_s2`](@ref))."
scale_s2(loss::DeployedLoss) = loss.s2

"""
    sq_loss_width(loss) -> Float64

The squared [`loss_width`](@ref), `scale_multiple^2 * s2` — the quantity the
shape's closed forms are written in, and the one this type exists to make exact.
"""
@inline sq_loss_width(loss::DeployedLoss) = loss.scale_multiple^2 * loss.s2

"""
    loss_width(loss) -> Float64

The loss's scale parameter in the residual's own units,
`scale_multiple * sqrt(s2)`: the residual scale at which the loss leaves the
quadratic regime (see [`AbstractLossShape`](@ref)). DERIVED; prefer
[`sq_loss_width`](@ref), which is what every closed form consumes and which
needs no square root.
"""
@inline loss_width(loss::DeployedLoss) = loss.scale_multiple * sqrt(loss.s2)

"""
    rescale(loss, s2) -> DeployedLoss

The same shape and scale multiple at a new noise VARIANCE — the `SelfScaling`
update, and what a moving-scale lift arm calls once per iteration.
`scale_multiple` is not a function of the data and does not move.

Goes through [`deploy`](@ref) rather than rebuilding the attribute tuple, so
the `(scale_multiple, s2)` schema is written in exactly one place.
"""
@inline rescale(loss::DeployedLoss, s2::Real) =
    deploy(loss_shape(loss), s2; scale_multiple = loss.scale_multiple)

"""
    retune(loss, scale_multiple) -> DeployedLoss

The same shape and noise variance at a new scale multiple — a caller replacing
the derived efficiency constant with a cross-validated operating point.
"""
@inline retune(loss::DeployedLoss, scale_multiple::Real) =
    deploy(loss_shape(loss), loss.s2; scale_multiple)

# =============================================================================
# The public vocabulary: the loss knows its own width
# =============================================================================
#
# The SHAPE MATH — the closed forms, written against an explicit squared width.
# They keep taking `sq_loss_width` because one shape is evaluated at several widths during
# a graduation, and they are PUBLIC (see the `public` declaration below)
# because THEY ARE THE EXTENSION POINT: a package defining its own loss shape
# joins the family by adding methods here, and cannot do it through the
# one-argument verbs, which read a width the new shape does not yet have.

# Everything a caller touches is below and takes only the loss and the
# residual: the width is the loss's, not an argument threaded through every
# call site.

"""
    lift_rho(loss, s) -> Float64

Loss value ρ at squared residual `s = r²`, at the loss's own width.
"""
@inline lift_rho(loss::AnyLoss, s) = ρ(loss_shape(loss), s, sq_loss_width(loss))

"""
    lift_conjugate(loss, w) -> Float64

Geman–Reynolds penalty conjugate(w) at the loss's own width, so that
`ρ(r) = min_w [w·r² + conjugate(w)]`.
"""
@inline lift_conjugate(loss::AnyLoss, w) = conjugate(loss_shape(loss), w, sq_loss_width(loss))

"""
    lift_weight(loss, s) -> Float64

IRLS weight `ω(s) = ρ′(s)` at the loss's own width — the factor multiplying a
residual in the weighted normal equations, for EVERY shape. For the liftable
shapes it is additionally the argmin of the Geman–Reynolds envelope at squared
residual `s`; for Huber and Gaussian it is just the derivative.

The single deployed-width weight verb.
"""
@inline lift_weight(loss::AnyLoss, s) = weight(loss_shape(loss), s, sq_loss_width(loss))

"""
    lift_zblock(loss, w̄) -> (rinv2, hreg)

The joint-(θ,z) Gauss–Newton z-block at the loss's own width. The one closed
form the lift core (LocalOptimizationDoneRight) consumes.
"""
@inline lift_zblock(loss::AnyLoss, w̄) = zblock(loss_shape(loss), w̄, sq_loss_width(loss))

"""
    lift_cost(loss, r, n) -> Float64

Total lifted cost over `n` residuals `r`, at the loss's own width.
"""
lift_cost(loss::AnyLoss, r, n::Int) = cost(loss_shape(loss), r, sq_loss_width(loss), n)

# Shape facets read through the deployment.
lift_wmax(loss::AnyLoss) = lift_wmax(loss_shape(loss))
liftable(loss::AnyLoss) = liftable(loss_shape(loss))
bounded(loss::AnyLoss) = bounded(loss_shape(loss))
gnc_surrogate(loss::AnyLoss) = gnc_surrogate(loss_shape(loss))

# -----------------------------------------------------------------------------
# Annealing — a point ALONG the homotopy, not a third field
# -----------------------------------------------------------------------------
#
# GNC deforms a convex surrogate into the target objective, and THE LOSS IS
# THE TARGET: a deployed loss with no μ attribute is where the path ends. A
# point along the path is that same loss wearing its position.
#
# An attribute rather than a field because μ is NOT ONE QUANTITY — a width
# multiplier for Welsch (μ·c̄², down to 1), a convexity parameter for GM, a
# Black–Rangarajan band parameter for TLS running the other way to 1e6. One
# field would force one name and one meaning onto three different things;
# `gnc_surrogate` says how to read it. Most losses never anneal at all.

"""
    anneal(loss, μ) -> AnnealedLoss

The loss at position `μ` on its graduation path. `μ = 1` is the target for the
μ-scaled families; see [`gnc_surrogate`](@ref) for what `μ` means to a shape.
"""
@inline anneal(loss::DeployedLoss, μ::Real) = Attributed(loss, (μ = Float64(μ),))

"""
    target_loss(loss) -> DeployedLoss

The homotopy's endpoint. A deployed loss IS the target; an annealed one
unwraps to it. GNC graduates at [`sq_loss_width`](@ref) but best-tracks against this.

NOT `target`: that name is the stereo-rig accessor (`target(::StereoRig)`,
geometry/cameras/stereo.jl), exported since 0.25 and meaning the second camera
of a pair — a different concept that must not share a generic with this one.
"""
target_loss(loss::DeployedLoss) = loss
target_loss(annealed::AnnealedLoss) = annealed.value

# The TWO accessors that make an annealed loss a member of `AnyLoss`.
# Defining these is the whole cost of joining: every one-argument verb above is
# written against `loss_shape` + `sq_loss_width` and needs no annealed method of
# its own.
loss_shape(annealed::AnnealedLoss) = loss_shape(annealed.value)
@inline sq_loss_width(annealed::AnnealedLoss) = _annealed_sq_loss_width(gnc_surrogate(annealed), annealed)

# `loss_width` is the one verb that is genuinely per-type, because the
# association matters: `scale_multiple * sqrt(s2)`, never
# `sqrt(scale_multiple^2 * s2)` — the two disagree on ~30% of inputs. The
# annealed width is the target's scaled by the family's own factor, taken from
# the trait rather than recovered by dividing two `sq_loss_width`s (the ratio
# is not bitwise `μ`, and costs a divide and two roots).
@inline loss_width(annealed::AnnealedLoss) = loss_width(annealed.value) * _annealed_loss_width_factor(gnc_surrogate(annealed), annealed)

rescale(annealed::AnnealedLoss, s2::Real) = Attributed(rescale(annealed.value, s2), annealed.attributes)
retune(annealed::AnnealedLoss, scale_multiple::Real) = Attributed(retune(annealed.value, scale_multiple), annealed.attributes)

# THE w̄ → 1 LIMIT of the z-block Hessian. With ε = 1 − w̄, both `rinv2` (≈ κ̄²″(1)·ε)
# and κ̄² (≈ κ̄²″(1)·ε²/2) vanish, but their ratio does not: `hreg → 2κ̄²″(1)`,
# which is 2c² for Welsch and Cauchy and 4c²·b/(2 − α) (4c² for α < 2) for
# Barron. This is Zach & Bourmaud's prescription (ECCV 2018, Eq. 10): the
# indeterminate (γ′)²/γ entry of the lifted Gauss-Newton block is replaced by
# its Taylor limit 2γ″(1) (γ = κ̄²/2, and our Φ is twice theirs). Every shape
# returns it AT w̄ = 1, and the lift's upper z clamp puts every confident
# inlier there EXACTLY (measured: 16% of the z-block evaluations of RVG's
# default E lift and 34% of F's, and every LO run), so this is the common case,
# not an edge. The direct ratio also cancels catastrophically near 1, so for
# |ε| below `_zblock_series_cut` each shape evaluates the ratio from its Taylor
# series in ε with the ε² cancelled analytically. The series is two-sided, as
# is the paper's expansion (in ±Δv, γ continued past 1): it is the ratio's
# expansion at w̄ = 1, accurate for small ε of either sign. The lift's z window
# keeps w̄ ≤ 1, but `zblock` is correct on the continuation w̄ > 1 too, rather
# than taking the series for every w̄ > 1 (ε < 0 < cut) where it is wrong.
#
# The cut is where the truncated series (error ~ε⁶) and the direct form
# (relative error ~eps/|ε|) meet: |ε| = eps^(1/6) — 2.5e-3 in Float64, 0.07 in
# Float32 — so neither side is used where it is worse than the other.
@inline _zblock_series_cut(::Type{T}) where {T} = cbrt(sqrt(eps(T)))

"""
    zblock(shape, w̄, sq_loss_width) -> (rinv2, hreg)

The two z-block quantities of the joint-(θ,z) Gauss-Newton step at normalized
weight `w̄ = z²` (the lift keeps `w̄ ∈ (0, 1]`; the forms are also valid
on the continuation `w̄ > 1`), scale `sq_loss_width = c²`:

- `rinv2 = −κ̄²′(w̄)` — the inverse-weight map `r²(w̄)` (≥ 0 on w̄ ≤ 1), where
  `κ̄²(w̄) = conjugate(w_max·w̄)/w_max` is the normalized conjugate;
  enters the z-gradient `gz = z(r² − rinv2)`.
- `hreg = w̄·rinv2²/κ̄²(w̄)` — the GN z-block Hessian `(dκ/dz)²`;
  enters the Schur pivot `dᵢ = 1/(r² + hreg + μ)`.

Losses with a cancelling ratio return the simplified closed form
(Geman–McClure: `hreg = c²` constant; smooth-TLS: `hreg = 2c²w̄`). For Welsch,
Cauchy and Barron both `rinv2` and κ̄² vanish at `w̄ = 1` while `hreg` tends to
a POSITIVE limit, `2κ̄²″(1)` (2c² for Welsch and Cauchy, 4c²·b/(2 − α) for
Barron); near `w̄ = 1` they evaluate the ratio from its series, so `hreg` is
accurate there and exact at `w̄ = 1`, where the lift's z clamp puts every
confident inlier. That value is Zach & Bourmaud's Taylor limit `2γ″(1)` of the
lifted Gauss-Newton block (ECCV 2018, Eq. 10; `γ = κ̄²/2`), the same for every
shape, so Barron's anchors (α → −∞ ≡ Welsch, α = −2 ≡ Geman–McClure, α = 0 ≡
Cauchy) hold for `zblock` at the clamp as everywhere else.
"""
@inline function zblock(shape::WelschShape, w̄, sq_loss_width)
    rinv2 = -sq_loss_width * log(w̄)                                # r²(w̄) = −c²·ln w̄
    return rinv2, sq_loss_width * w̄ * _welsch_hreg_ratio(w̄, _shape_float(w̄, sq_loss_width))
end
# ln²(w̄)/κ̄²-ratio of Welsch, (ln w̄)² / (w̄·ln w̄ − w̄ + 1), as a function of
# ε = 1 − w̄; → 2 at ε = 0. Below the cut: (Σ_{k≥1} ε^{k−1}/k)² / Σ_{k≥2} ε^{k−2}/(k(k−1)).
@inline function _welsch_hreg_ratio(w̄, ::Type{T}) where {T}
    ε = one(w̄) - w̄
    if abs(ε) < _zblock_series_cut(T)
        a = 1 + ε * (T(1 // 2) + ε * (T(1 // 3) + ε * (T(1 // 4) + ε * (T(1 // 5) + ε * T(1 // 6)))))
        b = T(1 // 2) + ε * (T(1 // 6) + ε * (T(1 // 12) + ε * (T(1 // 20) + ε * (T(1 // 30) + ε * T(1 // 42)))))
        return a * a / b
    end
    # w̄·ln w̄ + ε: terms of size ε (ε = 1 − w̄ is exact for w̄ ≥ ½, and below ½
    # nothing cancels), where w̄·ln w̄ − w̄ + 1 would cancel terms of size 1
    L = log(w̄)
    return L * L / (w̄ * L + ε)
end
@inline function zblock(::GemanMcclureShape, w̄, sq_loss_width)
    rinv2 = sq_loss_width * (1 / sqrt(w̄) - 1)                      # r²(w̄) = c²(w̄^{−1/2}−1)
    return rinv2, sq_loss_width                                    # w̄·rinv2²/κ̄² = c² (constant)
end
@inline function zblock(shape::CauchyShape, w̄, sq_loss_width)
    rinv2 = sq_loss_width * (1 / w̄ - 1)                            # r²(w̄) = c²(1/w̄−1)
    # w̄·rinv2²/κ² = c² / (w̄·S), S = (−ε − ln(1 − ε))/ε² = Σ_{k≥2} ε^{k−2}/k → ½
    return rinv2, sq_loss_width / (w̄ * _cauchy_conjugate_ratio(w̄, _shape_float(w̄, sq_loss_width)))
end
@inline function _cauchy_conjugate_ratio(w̄, ::Type{T}) where {T}
    ε = one(w̄) - w̄
    abs(ε) < _zblock_series_cut(T) &&
        return T(1 // 2) + ε * (T(1 // 3) + ε * (T(1 // 4) + ε * (T(1 // 5) + ε * (T(1 // 6) + ε * T(1 // 7)))))
    return (w̄ - 1 - log(w̄)) / (ε * ε)
end
@inline function zblock(::SmoothTLSShape, w̄, sq_loss_width)
    rinv2 = sq_loss_width * (1 - w̄)                                # r²(w̄) = c²(1−w̄)  [w̄ = w/2]
    return rinv2, 2 * sq_loss_width * w̄                            # w̄·rinv2²/κ̄² = 2c²w̄ (const curvature)
end

"""
    cost_scale(shape) -> Float64

The shape's historical COST CONVENTION: the constant its tracked objective is
scaled by, `1.0` for everything but Welsch, whose tracker carries a ½.

A pure rescaling — monotone-equivalent, so it cannot change which model wins —
and therefore a property of the SHAPE's history, not of its math. It is a trait
of its own precisely because it is not deducible from anything else, and it is
per-SHAPE so that no graduation family inherits Welsch's ½: a future
width-annealed shape carries its own convention, not Welsch's.

Multiplying by ½ is exact in binary floating point.
"""
cost_scale(::AbstractLossShape) = 1.0
cost_scale(::WelschShape) = 0.5

"""
    cost(loss, r, sq_loss_width, n) -> Float64

Loss M-estimator cost `Σᵢ ρ(rᵢ; c)` over `r[1:n]` — the best-tracking
objective of the lifted LO, scaled by the shape's [`cost_scale`](@ref).

ONE loop for every shape; Welsch's ½ comes from `cost_scale`, so the Welsch ρ
closed form is written once.
"""
function cost(shape::AbstractLossShape, r, sq_loss_width::Real, n::Int)
    tot = zero(float(promote_type(eltype(r), typeof(sq_loss_width))))
    @inbounds for i in 1:n
        tot += ρ(shape, r[i] * r[i], sq_loss_width)
    end
    return oftype(tot, cost_scale(shape)) * tot     # ½ and 1 are exact in every float type
end

"""
    decision_width(m, decision_threshold) -> Float64

The native width of `m` whose inlier/outlier decision flips at the residual
`decision_threshold` — THE one conversion behind every "same decision point"
comparison across methods (the evaluation's fairness axis, where
`decision_threshold = scale_multiple · ŝ`). Each owner defines its own method:

- a loss shape: the [`loss_width`](@ref) at which the shape's NORMALIZED IRLS
  weight at `decision_threshold` is exactly ½, `ω(t²; w²)/w_max = ½` (the
  half-weight decision of a reweighting loss);
- `TLSShape`: the identity — a hard truncation decides at its own width;
- RANSACScoringDoneRight's `GauInlier`: the Student-t marginal's outlier
  half-width whose MAP boundary sits at `decision_threshold`.

Closed forms for the shapes (`t` the decision threshold, `w` the loss width):
- Welsch:        `e^{−t²/w²} = ½`        ⇒ `w = t/√(ln 2)`
- Geman–McClure: `(1+t²/w²)^{−2} = ½`    ⇒ `w = t/√(√2 − 1)`
- Cauchy:        `(1+t²/w²)^{−1} = ½`    ⇒ `w = t`
- smooth-TLS:    `1 − t²/w² = ½`         ⇒ `w = √2·t` (hard cutoff at `√2·t`)
- TLS:           `w = t`
"""
function decision_width end

decision_width(::WelschShape, decision_threshold::Real) = Float64(decision_threshold) / sqrt(log(2.0))
decision_width(::GemanMcclureShape, decision_threshold::Real) = Float64(decision_threshold) / sqrt(sqrt(2.0) - 1.0)
decision_width(::CauchyShape, decision_threshold::Real) = Float64(decision_threshold)
decision_width(::SmoothTLSShape, decision_threshold::Real) = sqrt(2.0) * Float64(decision_threshold)
decision_width(::TLSShape, decision_threshold::Real) = Float64(decision_threshold)

# -----------------------------------------------------------------------------
# GNC surrogate facet — the graduated (μ-parameterized) surrogate closed forms
# (Yang, Antonante, Tzoumas & Carlone, RA-L 2020; Black & Rangarajan, IJCV 96).
#
# The GM and Welsch GNC surrogates are NOT new closed forms: at rung μ they
# are exactly this file's facet-B verbs at the ANNEALED squared scale μ·c̄²,
#
#     w_gnc-gm(x; c̄², μ)     ≡ lift_weight(anneal(gm_loss, μ), x)
#     w_gnc-welsch(x; c̄², μ) ≡ lift_weight(anneal(welsch_loss, μ), x)
#
# (and likewise their ρ's via `lift_rho` at μ·c̄²) — engines MUST route
# through `lift_weight` rather than re-deriving them. Only the TLS surrogate
# has no facet-B counterpart (the truncated quadratic is not liftable in the
# Geman–Reynolds sense — its graduation is the Black–Rangarajan outer bound,
# not a width anneal), so its closed forms live here.
# -----------------------------------------------------------------------------

"""
    gnc_tls_weight(x, sq_loss_width, μ) -> Float64

The Black–Rangarajan GNC-TLS surrogate weight at squared residual `x = r²`,
truncation bound `sq_loss_width = c̄²`, continuation parameter `μ`:

    w = 1                          x ≤ (μ/(μ+1))·c̄²
    w = c̄·√(μ(μ+1))/|r| − μ        in the middle band
    w = 0                          x ≥ ((μ+1)/μ)·c̄²

Continuous across the band edges (w → 1 below, w → 0 above). μ → ∞ recovers
hard truncated least squares; the middle band is the graduation. Points with
`0 < w < 1` are the OPEN band — a GNC engine keeps graduating only while the
band is populated (the `open` observation of the `LoSchedule` contract).
"""
@inline function gnc_tls_weight(x::Float64, sq_loss_width::Float64, μ::Float64)
    x <= (μ / (μ + 1.0)) * sq_loss_width && return 1.0
    x >= ((μ + 1.0) / μ) * sq_loss_width && return 0.0
    return sqrt(sq_loss_width * μ * (μ + 1.0)) / sqrt(x) - μ
end

# The TLS target loss (μ = ∞, non-surrogate) is `min(x, c̄²)`, which is exactly
# `ρ(TLSShape(), x, c̄²)`; `gnc_target` reaches it through ρ like every other
# shape. The surrogate `gnc_tls_weight` is its own function because the
# Black–Rangarajan band is genuinely its own closed form, with no counterpart
# in the shape math.

# =============================================================================
# GncSurrogate — how a shape graduates, as a trait
# =============================================================================
#
# How a shape graduates is a property of the SHAPE, so it is a trait on the
# loss: a new shape declares one method, and a shape that declares none is a
# MethodError at dispatch rather than silently taking a branch.
#
# The three are genuinely different, which is why they are three types:
#
#   WidthAnnealed      mu scales the squared width (c2_mu = mu*sq_loss_width), mu -> 1.
#                      The only family for which mu IS a width, hence the only
#                      one that accepts a numeric entry-breadth.
#   ConvexityAnnealed  mu enters the weight the same way but MEANS convexity;
#                      mu0 = 2*r2max/sq_loss_width rather than r2max/sq_loss_width.
#   BandAnnealed       Black-Rangarajan band parameter with its own closed form
#                      (`gnc_tls_weight`), running the OTHER WAY, to the mu cap.

"""
    GncSurrogate

How a loss shape graduates under GNC — the trait `gnc_weight`, `gnc_target`,
`gnc_mu0` and `gnc_entry_mu0` dispatch on. See [`gnc_surrogate`](@ref).
"""
abstract type GncSurrogate end

"""
    MuScaled <: GncSurrogate

The families whose `μ` multiplies the squared width (`c²_μ = μ·c̄²`). They
differ in what `μ` MEANS — a width for Welsch, convexity for Geman–McClure —
and so in `gnc_mu0` and `gnc_target`, but not in how it enters the weight, so
that half of the trait is written once against this supertype.

An ABSTRACT TYPE, not a `Union`: a downstream shape whose `μ` scales the
squared width joins by subtyping this, the way a downstream M-estimator joins
by subtyping [`AbstractMEstimator`](@ref). A `Union` would have made the new
extension point closed — joining would mean editing a `Union` inside VGC.
"""
abstract type MuScaled <: GncSurrogate end

"GNC: `μ` scales the squared width, `c²_μ = μ·c̄²`, annealing down to `μ = 1`."
struct WidthAnnealed <: MuScaled end

"GNC: `μ` enters as `μ·c̄²` but denotes CONVEXITY, not a width (Geman–McClure)."
struct ConvexityAnnealed <: MuScaled end

"GNC: Black–Rangarajan band parameter with its own closed form, annealing UP."
struct BandAnnealed <: GncSurrogate end

"GNC: this shape has no graduation path."
struct NoSurrogate <: GncSurrogate end

"""
    gnc_surrogate(loss) -> GncSurrogate

The loss's graduation family. A shape that does not graduate answers
[`NoSurrogate`](@ref), and the `gnc_*` verbs then raise a `MethodError` rather
than silently taking a branch.
"""
gnc_surrogate(::AbstractLossShape) = NoSurrogate()
gnc_surrogate(::WelschShape) = WidthAnnealed()
gnc_surrogate(::GemanMcclureShape) = ConvexityAnnealed()
gnc_surrogate(::TLSShape) = BandAnnealed()

"""
    band_signal(surrogate, w) -> Int

The per-residual rung-advance signal a GNC engine counts: `1` when the weight
lies in the OPEN band `0 < w < 1`, `0` otherwise — and identically `0` for
every family whose graduation has no band.

A TRAIT VERB rather than an `isa BandAnnealed` test inside the accumulation
loops (LocalOptimizationDoneRight's lift core). Those loops are generic over the family, so they
name no family: a new surrogate family declares its own signal instead of
editing the loop.

Returns an `Int` so the loops accumulate unconditionally; where the answer is
the constant `0` the whole term folds away.
"""
band_signal(::GncSurrogate, w) = 0
band_signal(::BandAnnealed, w) = ifelse(0.0 < w < 1.0, 1, 0)

# The annealed squared width, per family. Width- and convexity-annealed shapes
# both enter as μ·c̄²; a band-annealed one does not anneal a width at all.
_annealed_sq_loss_width(::MuScaled, annealed) = annealed.μ * sq_loss_width(annealed.value)
# A band parameter is genuinely not a width, so a band-annealed shape answers
# with the target's. A shape with NO graduation path gets no method at all: it
# must not accept a μ and return it inert.
_annealed_sq_loss_width(::BandAnnealed, annealed) = sq_loss_width(annealed.value)

# `loss_width(::AnnealedLoss)`'s factor, same split: √μ where μ scales the width,
# 1 where it is a band parameter. Exact, unlike recovering μ from a ratio.
_annealed_loss_width_factor(::MuScaled, annealed) = sqrt(annealed.μ)
_annealed_loss_width_factor(::BandAnnealed, annealed) = 1.0

# The closed forms a GNC engine consumes, dispatched on `gnc_surrogate`:
# surrogate weights at rung μ, target (non-surrogate) losses, and the
# Yang–Carlone μ₀ entry policies. GM/Welsch route through the facet-B verbs
# at the annealed squared scale μ·c̄² (the identity documented above); TLS is
# the Black–Rangarajan band. `x = r²`.

"""
    gnc_weight(loss, x, μ) -> Float64

GNC surrogate weight at squared residual `x`, terminal bound `c̄²`, rung `μ`:
The μ-scaled families are `lift_weight` at the annealed width `μ·c̄²`; a band-
annealed shape is
[`gnc_tls_weight`](@ref).
"""
@inline gnc_weight(loss::DeployedLoss, x::Float64, μ::Float64) =
    _gnc_weight(gnc_surrogate(loss), loss, x, μ)
# The GM/Welsch surrogate IS the loss's own weight at the annealed width —
# written as that identity so `anneal` stays the single definition of what μ does.
@inline _gnc_weight(::MuScaled, loss, x, μ) = lift_weight(anneal(loss, μ), x)
@inline _gnc_weight(::BandAnnealed, loss, x, μ) = gnc_tls_weight(x, sq_loss_width(loss), μ)

"""
    gnc_target(loss, x) -> Float64

TARGET (non-surrogate) loss the GNC engine best-tracks on: the shape's own ρ at
the terminal width `c̄²`, scaled by its [`cost_scale`](@ref) — which is `1.0`
for every shape but Welsch, whose tracker carries a ½ (exact; the ×½ scaling is
bitwise).

ONE expression for all three families. The ½ is Welsch's cost convention, a
property of the shape (`cost_scale`), not of its graduation family; TLS routes
through ρ like every other shape.

The dispatch exists ONLY to refuse: a [`NoSurrogate`](@ref) shape gets no
method and raises a `MethodError`, like every other `gnc_*` verb, rather than
best-tracking a graduation it does not have.
"""
@inline gnc_target(loss::DeployedLoss, x::Float64) =
    _gnc_target(gnc_surrogate(loss), loss, x)
@inline _gnc_target(::Union{MuScaled, BandAnnealed}, loss, x) = _gnc_target_value(loss, x)
# `loss::AnyLoss`, not `loss`: untyped, `loss_shape` also accepts a bare
# shape, which widens the inferred type enough that `lift_rho` picks up a
# spurious `::AbstractLossShape` union split (a JET no-method report).
@inline _gnc_target_value(loss::AnyLoss, x) = cost_scale(loss_shape(loss)) * lift_rho(loss, x)

"μ cap: the effectively-hard TLS surrogate (also the GNC μ-ladder's :up cap, LocalOptimizationDoneRight)."
const _GNC_MU_CAP = 1.0e6

"""
    gnc_mu_cap() -> Float64

The μ at which the TLS band surrogate is effectively hard: the cap
[`gnc_mu0`](@ref) guards to, and the cap of an ascending GNC μ-ladder.
"""
gnc_mu_cap() = _GNC_MU_CAP

"""
    gnc_mu0(loss, r2max) -> Float64

Yang–Carlone μ₀ from the SEED residual extreme `r2max = r²_max`:
`ConvexityAnnealed` (GM) μ₀ = max(2r²_max/c̄², 1) (convex start; already ≤ 1 ⇒
true GM); `WidthAnnealed` (Welsch) μ₀ = max(r²_max/c̄², 1) (c₀² = r²_max ⇒ min
weight e^{−1}); `BandAnnealed` (TLS) μ₀ = c̄²/(2r²_max − c̄²), guarded to the μ
cap when r²_max ≤ c̄²/2 (no graduation needed — start effectively hard).
"""
@inline gnc_mu0(loss::DeployedLoss, r2max::Float64) =
    _gnc_mu0(gnc_surrogate(loss), loss, r2max)
@inline _gnc_mu0(::ConvexityAnnealed, loss, r2max) =
    max(2.0 * r2max / sq_loss_width(loss), 1.0)            # already ≤1 ⇒ start at true GM
@inline _gnc_mu0(::WidthAnnealed, loss, r2max) =
    max(r2max / sq_loss_width(loss), 1.0)                  # c₀² = r²_max ⇒ min weight e^{-1}
@inline function _gnc_mu0(::BandAnnealed, loss, r2max)
    target = sq_loss_width(loss)                      # c̄², the target squared width
    r2max <= target / 2.0 && return _GNC_MU_CAP       # no graduation needed: hard TLS
    return target / (2.0 * r2max - target)
end

"""
    gnc_entry_mu0(loss, r2max, entry) -> Float64

Entry-breadth policy (smooth-and-lift dose axis 1): `entry = NaN` is the
seed-extreme Yang–Carlone init ([`gnc_mu0`](@ref) — the r_max entry); a
numeric `entry = m` starts the anneal at m× the TARGET WIDTH, i.e.
μ₀ = max(m², 1) (μ multiplies the SQUARED scale c²_μ = μ·c̄²). m = 1 is the
static limit. Only [`WidthAnnealed`](@ref) consumes numeric entries — for the
other families μ is a convexity or band parameter, not a width.
"""
@inline gnc_entry_mu0(loss::DeployedLoss, r2max::Float64, entry::Float64) =
    isnan(entry) ? gnc_mu0(loss, r2max) : _entry_mu0(gnc_surrogate(loss), entry)
# Only an annealed-WIDTH family has a width for `entry` to be a multiple OF; the
# others get no method, so the refusal is dispatch's rather than hand-rolled.
@inline _entry_mu0(::WidthAnnealed, entry) = max(entry * entry, 1.0)

# -----------------------------------------------------------------------------
# Barron family — the α-parameterized general robust loss (Barron CVPR 2019)
# as a facet-B member: ONE continuous retirement-steepness dial with the
# existing losses embedded as parameter limits (smooth-and-lift α-dial).
#
# Facet-B normalization (ρ′(0) = w_max = 1, squared scale sq_loss_width):
#
#     ρ_α(s; c²) = (2c²b/α)·((s/(c²b) + 1)^{α/2} − 1),   b = |α − 2|,
#     w_α(s; c²) = (s/(c²b) + 1)^{α/2−1},                w_max = 1,
#
# α = 0 is the removable Cauchy limit (closed forms special-cased). EXACT
# anchor identities (the α-dial's consistency gates; convention-mapped c):
#
#     α = −2      ≡ Geman–McClure at 4c²
#     α → −∞      ≡ Welsch        at 2c²
#     α = 0       ≡ Cauchy        at 2c²
#
# They hold for ρ, weight, conjugate and zblock everywhere, the z clamp w̄ = 1
# included (every shape takes Zach & Bourmaud's limit 2κ̄²″(1) there).
#
# Liftable for every α < 2 (ρ concave in s = r²; the α case-splitting is
# removable parameter limits — the lift never sees it); BOUNDED only for
# α < 0 with ρ(∞) = 2c²b/|α| (the liftable/bounded trait split reappears
# INSIDE the family as the α = 0 threshold, Cauchy the boundary case).
# -----------------------------------------------------------------------------

"""
    BarronShape(α) <: AbstractLossShape

Barron's general robust loss at fixed shape `α < 2` (facet-B convention
above). `α` is the retirement-steepness dial: −∞ = Welsch, −2 = GM,
0 = Cauchy, 2 = L2 (excluded — not liftable). Bounded (redescending to a
floor) iff `α < 0`.
"""
struct BarronShape <: AbstractLossShape
    α::Float64      # the ONLY field: α is shape, which is why it is fitted and
    # why this family has no efficiency constant to derive.
    function BarronShape(α::Real)
        Float64(α) < 2.0 || throw(ArgumentError("BarronShape: α must be < 2 (α = 2 is L2 — not liftable)"))
        return new(Float64(α))
    end
end

liftable(::BarronShape) = true
bounded(shape::BarronShape) = shape.α < 0.0

lift_wmax(::BarronShape) = 1.0

@inline _barron_b(α::Real) = abs(α - 2)

# All forms below are written in cancellation-free expm1/log1p style so the
# α → −∞ limit is numerically exact to gate tolerance (the naive power forms
# lose the Welsch limit at |α| ≳ 10⁶ through (1+x)-rounding and the b/α
# amplification of w^{α/(α−2)} cancellations).
function ρ(shape::BarronShape, s, sq_loss_width)
    α = _shape_float(s, sq_loss_width)(shape.α)                              # α in the computation's precision
    iszero(α) && return ρ(CauchyShape(), s, 2 * sq_loss_width)           # the Cauchy limit at 2c²
    b = _barron_b(α)
    return (2 * sq_loss_width * b / α) * expm1((α / 2) * log1p(s / (sq_loss_width * b)))
end

function weight(shape::BarronShape, s, sq_loss_width)
    α = _shape_float(s, sq_loss_width)(shape.α)
    iszero(α) && return weight(CauchyShape(), s, 2 * sq_loss_width)        # Cauchy weight at 2c²
    b = _barron_b(α)
    return exp((α / 2 - 1) * log1p(s / (sq_loss_width * b)))
end

# Geman–Reynolds dual: s(w) = c²b·expm1((2/(α−2))·ln w);
# conjugate(w) = c²b[(2/α)·expm1((α/(α−2))·ln w) − w·expm1((2/(α−2))·ln w)]
# (conjugate(1) = 0, conjugate′ = −s(w); algebraically equal to
# c²b[(2/α−1)w^{α/(α−2)} + w − 2/α], but stable ∀α).
function conjugate(shape::BarronShape, w, sq_loss_width)
    α = _shape_float(w, sq_loss_width)(shape.α)
    iszero(α) && return conjugate(CauchyShape(), w, 2 * sq_loss_width)           # Cauchy conjugate at 2c²
    b = _barron_b(α)
    lw = log(w)
    return sq_loss_width * b * (
        (2 / α) * expm1((α / (α - 2)) * lw) -
            w * expm1((2 / (α - 2)) * lw)
    )
end

# w·κ²″ = −c²b·(2/(α−2))·w^{2/(α−2)} = 2c²·w^{2/(α−2)} (b = 2 − α); at α = 0 it
# is the Cauchy form at 2c², 2c²/w, without a branch.
function conjugate_curvature(shape::BarronShape, w, sq_loss_width)
    α = _shape_float(w, sq_loss_width)(shape.α)
    return 2 * sq_loss_width * exp((2 / (α - 2)) * log(w))
end

@inline function zblock(shape::BarronShape, w̄, sq_loss_width)
    α = _shape_float(w̄, sq_loss_width)(shape.α)
    iszero(α) && return zblock(CauchyShape(), w̄, 2 * sq_loss_width)       # Cauchy closed form at 2c²
    b = _barron_b(α)
    p = 2 / (α - 2)
    rinv2 = sq_loss_width * b * expm1(p * log(w̄))                             # s(w̄) ≥ 0
    ε = one(w̄) - w̄
    T = _shape_float(w̄, sq_loss_width)
    # the series converges like (|p|·ε)^k: the cut shrinks with |p| (α → 2⁻)
    if abs(ε) < _zblock_series_cut(T) / max(one(T), abs(p))
        # r̃ = s/(c²b·ε) = Σ_{k≥1} (−1)^k p⁽ᵏ⁾/k! ε^{k−1},  κ̃ = κ²/(c²b·ε²) = Σ_{k≥1} (−1)^k p⁽ᵏ⁾/(k+1)! ε^{k−1}
        # (p⁽ᵏ⁾ the falling factorial); hreg = w̄·c²b·r̃²/κ̃ → −2c²b·p
        r̃ = zero(T); κ̃ = zero(T); F = one(T); kf = one(T); εk = one(T); sgn = one(T)
        for kk in 1:7
            F *= p - (kk - 1); kf *= kk; sgn = -sgn
            r̃ += sgn * F / kf * εk
            κ̃ += sgn * F / (kf * (kk + 1)) * εk
            εk *= ε
        end
        return rinv2, w̄ * sq_loss_width * b * r̃ * r̃ / κ̃
    end
    kap2 = conjugate(shape, w̄, sq_loss_width)                                       # κ̄² = κ² (w_max = 1)
    return rinv2, w̄ * rinv2 * (rinv2 / kap2)                         # ordered: rinv2² overflows as w̄ → 0 for α near 2
end

# Half-weight decision: w(t²; c²) = ½ at the decision threshold t ⇒
# c = t/√(b·(2^{2/(2−α)} − 1)) (no 1/α — regular at α = 0, where it gives t/√2 =
# the Cauchy-at-2c² identity: the effective Cauchy width is √2·c = t).
function decision_width(shape::BarronShape, decision_threshold::Real)
    b = _barron_b(shape.α)
    return Float64(decision_threshold) / sqrt(b * (2.0^(2.0 / (2.0 - shape.α)) - 1.0))
end

# =============================================================================
# Loss identity: the loss FAMILY a shape belongs to (a figure roster's row)
# and its display short-name (a strategy label's prefix) — what a
# loss-parametric strategy's `loss_family` and label derive from. The smooth
# surrogate of TLS shares the TLS row; its TYPE records the surrogate.
# =============================================================================
loss_family(::Type{GaussianShape}) = :l2
loss_family(::Type{WelschShape}) = :welsch
loss_family(::Type{GemanMcclureShape}) = :gm
loss_family(::Type{CauchyShape}) = :cauchy
loss_family(::Type{<:Union{SmoothTLSShape, TLSShape}}) = :tls
loss_family(::Type{HuberShape}) = :huber
loss_family(::Type{BarronShape}) = :barron

loss_label(::Type{GaussianShape}) = raw"$\ell_2$"
loss_label(::Type{WelschShape}) = "Welsch"
loss_label(::Type{GemanMcclureShape}) = "GM"
loss_label(::Type{CauchyShape}) = "Cauchy"
loss_label(::Type{<:Union{SmoothTLSShape, TLSShape}}) = "TLS"
loss_label(::Type{HuberShape}) = "Huber"
loss_label(::Type{BarronShape}) = "Barron"
