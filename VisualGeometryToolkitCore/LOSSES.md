# LOSSES.md — one robust-loss family, three facets

The design note behind `src/robust/losses.jl`: the stack's robust losses are
ONE family of shapes (`AbstractLossShape`), read under three historical
parameter conventions (facets). Sections 1–4 describe the shipped design.

## 0. Vocabulary — one name per quantity

| name | quantity |
|---|---|
| loss shape | a normalized SCALE FAMILY `ρ_w(r) = w²·ρ₁(r/w)`; a tag type (`WelschShape`, …) under `AbstractLossShape` |
| `loss_width` | the family's scale parameter `w`: the residual scale at which the loss leaves the quadratic regime every shape shares at the origin |
| `sq_loss_width` | `loss_width²`, the argument every closed form takes |
| `scale_multiple` | `loss_width / ŝ`: the width as a dimensionless multiple of the estimated noise scale (NOT a quantile) |
| `scale`, `s2` | the noise scale `ŝ` and its square — data, never a width |
| `outlier_halfwidth` | the half-width of a Student-t marginal's uniform outlier component (RansacScoringDoneRight) |
| `decision_width(m, decision_threshold)` | the native width of `m` whose inlier/outlier decision flips at the residual `decision_threshold` |

"Kernel" is not a name for any of these. In the stack it is kept only where it
means something else: a GPU (KernelAbstractions) kernel, a convolution or
filter kernel, a minimal-solver kernel (the positional function under a
`MinimalSolver`), the kernel (null space) of a matrix, or the kernel of a
probability density (its unnormalized part).

## 1. The family: a normalized scale family

A loss shape is a scale family on the residual `r`,

```
ρ_w(r) = w² · ρ₁(r / w),        w = loss_width,
```

normalized so that every member shares the same quadratic regime at the
origin. In the facet-B convention this file ships (ρ written in the squared
residual `x = r²` with `ρ′(0) = w_max`), `ρ₁(u) = w_max·u² + o(u²)`, with
`w_max = 1` for every shape but smooth-TLS (`w_max = 2`). The classical
M-estimation normalization `ρ₁(u) = u²/2 + o(u²)` differs by a constant factor
only, which moves neither the minimizer nor the width; the one shape whose
tracked cost keeps the ½ (Welsch) says so through `cost_scale`.

The same family is written with a depth Δ and a normalized shape `g`:

```
ρ(r) = Δ · g(x / w²),        x = r²,
```

- `g(0) = 0`, `g` nondecreasing and concave in `x` (this concavity is what
  makes both the IRLS weight and the Geman–Reynolds dual exist);
- for **bounded** shapes, `sup g = 1` — the depth Δ is then exactly `ρ(∞)`, a
  dimensionless log-contrast (the density floor is `e^{−Δ}`);
- for **unbounded** shapes (Gaussian/L2 `g(u) = u`, Cauchy `g(u) = ln(1+u)`,
  Huber), `sup g = ∞` and Δ is a plain multiplier.

The three dials: shape `g` (tail law), depth `Δ` (evidence contrast / well
depth), width `w` (`loss_width`). Derived quantities, all facet-independent:

| quantity | definition | role |
|---|---|---|
| influence weight | `ω(x) = ρ'ₓ = (Δ/w²)·g'(x/w²)` | IRLS / θ-solve weight |
| inlier-limit weight | `w_max = ω(0) = Δ·g'(0)/w²` | GR domain end, lift normalizer |
| liftable | `g'(0) < ∞` (⇒ `w_max < ∞`) | GR dual exists |
| bounded | `sup g = 1` (`ρ(∞) = Δ < ∞`) | redescends to a floor; NOT normalizable on ℝ |
| GR dual | `conjugate(w) = ρ(x(w)) − w·x(w)`, `x(w) = ω⁻¹(w)` | z-block of the joint-(θ,z) lift |
| normalizer | `Z̃(w, Δ) = ∫_{−r_max}^{r_max} e^{−ρ(r)} dr` | truncated-density evidence (scoring) |

### The two traits are NOT the same axis (Cauchy)

`liftable(shape)` (finite `w_max`) and `bounded(shape)` (`ρ(∞) < ∞`) differ:
`ρ_Cauchy = Δ·ln(1+x/w²)` has `w_max = Δ/w² < ∞` (liftable — `CauchyShape`
ships a working GR dual) but `ρ(∞) = ∞`. Boundedness is the axis that matters
for *evidence*: `e^{−ρ}` of a bounded shape has a positive floor, so it is not
normalizable on an unbounded domain, scoring with it needs the truncated
normalizer `Z̃` over `[−r_max, r_max]` (RansacScoringDoneRight's construction),
and the corresponding weight policy is never a `NormalizedWeightPolicy`.

The shipped tags, by quadrant (`liftable` means the closed-form GR dual ships
HERE):

- `bounded ∧ liftable`: Welsch, Geman–McClure, smooth-TLS; Barron for α < 0.
- `¬bounded ∧ liftable`: Cauchy; Barron for 0 ≤ α < 2.
- `bounded ∧ ¬liftable`: hard truncation (TLS). Its `g'` is discontinuous, so
  its lift exists only through the smooth surrogate smooth-TLS.
- `¬bounded ∧ ¬liftable`: Huber and Gaussian/L2. Deployed, both answer
  `lift_rho`/`lift_weight`/`lift_cost` (through the total `ρ`/`weight`), but
  neither has the Geman–Reynolds dual.

## 2. The three facets

Each facet is the same `(g, Δ, w)` triple under a different convention.

### Facet A — M-estimation (classical, not shipped)

Curvature-normalized core (`ρ ≈ r²/2` near 0); one dial sets both width and
depth: `Δ = w²/(2 g'(0))`, so `ρ''(0) = 1`. Classical M-estimation has no
independent contrast dial. The efficiency constants (`efficiency_constant`) are
quoted in this literature's terms and are invariant under the constant factor
that separates it from facet B.

### Facet B — Geman–Reynolds dual (the `lift_*` verbs)

`ρ'(x = 0) = w_max`, `Δ = w_max · w² / g'(0)`. E.g. `WelschShape` at width `w`:
`ρ = w²(1 − e^{−x/w²})`, `Δ = w²`, `w_max = 1`.

- `lift_wmax = Δ·g'(0)/w²`
- `lift_rho(x) = Δ·g(x/w²)`
- `lift_weight(x) = ω(x)` (unnormalized, `ω(0) = w_max`)
- `lift_conjugate(w) = ρ(x(w)) − w·x(w)`
- `lift_zblock(w̄)` on the normalized `w̄ = w/w_max`
- `lift_cost` (scaled by `cost_scale`)
- `decision_width(shape, t)`: the width at which the NORMALIZED weight at the
  residual `t` is ½ — the half-weight decision.

| shape | g(u) | w_max | Δ (at width w) |
|---|---|---|---|
| `WelschShape` | `1−e^{−u}` | 1 | `w²` |
| `GemanMcclureShape` | `u/(1+u)` | 1 | `w²` |
| `CauchyShape` | `ln(1+u)` | 1 | `w²` (unbounded) |
| `SmoothTLSShape` | `2u−u²` capped at 1 | 2 | `w²` |
| `BarronShape(α)` | α-family: Welsch (α→−∞), GM (α=−2), Cauchy (α=0) | 1 | `2w²\|α−2\|/\|α\|` for α<0 (unbounded for 0≤α<2) |

### Facet C — inlier density (RansacScoringDoneRight, inlier_potentials.jl)

Already the unified family: `Δ·g(x/w²)` with `sup g = 1` for the bounded tags,
plus the truncated normalizer `log Z̃(w², Δ)` over the fixed domain
`[0, r_max]`, which makes `e^{−ρ}` usable as an inlier density despite
`bounded ⇒ ¬normalizable` on ℝ. RansacScoringDoneRight defines its formula
layer (`_rho`, `_default_depth`, …) as methods ON the VGC tags;
there is no second set of names.

## 3. The shipped tags and their deployment

```julia
abstract type AbstractLossShape end

# Shapes whose width is set by asymptotic efficiency at the Gaussian. One method
# — `efficiency_constant(::Type{S})` at codimension 1 — supplies the default
# `scale_multiple` a deployment uses.
abstract type AbstractMEstimator <: AbstractLossShape end

struct WelschShape       <: AbstractMEstimator end     # g(u) = 1 − e^{−u}
struct GemanMcclureShape <: AbstractMEstimator end     # g(u) = u/(1+u)
struct CauchyShape       <: AbstractMEstimator end     # g(u) = ln(1+u)
struct HuberShape        <: AbstractMEstimator end     # quadratic core, linear tail

# Efficiency does not parameterise these: no width (Gaussian), a truncation
# rather than an influence function (TLS), or a shape that is itself fitted.
struct GaussianShape     <: AbstractLossShape end   # g(u) = u
struct SmoothTLSShape    <: AbstractLossShape end   # g(u) = 2u−u², capped at 1
struct TLSShape          <: AbstractLossShape end   # g(u) = min(u, 1) — hard truncation
struct BarronShape       <: AbstractLossShape       # α-family, α < 2
    α::Float64
end
```

### A shape carries no width: deployment composes it on

```julia
deploy(shape, s2; codim = 1, scale_multiple = efficiency_constant(typeof(shape), codim))
const DeployedLoss = Attributed{<:AbstractLossShape, <:NamedTuple{(:scale_multiple, :s2)}}
const AnnealedLoss = Attributed{<:DeployedLoss,      <:NamedTuple{(:μ,)}}
loss_width(loss)    == loss.scale_multiple * sqrt(loss.s2)
sq_loss_width(loss) == loss.scale_multiple^2 * loss.s2
```

`s2` is the noise VARIANCE, not a linear sigma: the whole robust layer speaks
squared scale, so storing the square makes `sq_loss_width` exact.
`scale_multiple` and `s2` are stored APART, never as their product:

| verb | moves | driver |
| --- | --- | --- |
| `rescale(loss, s2)` | the variance | `SelfScaling`, per iteration |
| `retune(loss, scale_multiple)` | the scale multiple | deploying a CV'd operating point |
| `anneal(loss, μ)` | the homotopy position | a GNC rung |

Annealing is an ATTRIBUTE: `μ` is not one quantity across the families (a
width multiplier for Welsch, a convexity parameter for GM, a Black–Rangarajan
band for TLS — see `GncSurrogate`). The deployed loss IS the homotopy's
target, so `target_loss` is an unwrap.

### Two vocabularies, deliberately

| form | takes | for |
| --- | --- | --- |
| `lift_weight(loss, x)` | a deployed/annealed loss | callers; the loss knows its width |
| `weight(shape, x, sq_loss_width)` | a bare shape + an explicit squared width | the SHAPE MATH, and the extension point |

The shape math (`ρ`, `ψ`, `weight`, `conjugate`, `zblock`, `cost`,
`cost_scale`, `band_signal`) is `public` but unexported and generic in the
argument's float type. It takes an explicit `sq_loss_width` because a
graduation evaluates ONE shape at MANY widths, and because a downstream package
defines its own shape there.

## 4. The decision point: `decision_width`

Comparing methods at "the same decision" means comparing them at the residual
where each one's inlier/outlier decision flips. `decision_width(m, t)` returns
the native width of `m` whose decision flips at the residual `t`
(`t = scale_multiple · ŝ` on the evaluation's fairness axis). One
generic, one method per owner, each owner's own math:

| `m` | owner | decision | width returned |
|---|---|---|---|
| Welsch, GM, Cauchy, smooth-TLS, Barron | VGC | normalized IRLS weight = ½ | `loss_width` |
| `TLSShape` | VGC | the truncation itself | `t` (the identity) |
| `GauInlier` | RansacScoringDoneRight | Student-t/uniform MAP boundary | `outlier_halfwidth` |

The MAP decision on a POSTERIOR membership weight is `is_map_inlier`
(robust/estimation.jl): posterior > ½. It never applies to a loss weight.
