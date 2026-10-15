# Scoring Functions

This page describes how RANSAC scores a model hypothesis, the baseline
scorers the core ships, and how a scorer from another package plugs in.

## The scoring contract

The RANSAC loop reads a scorer only through VisualGeometryCore's scoring
contract (`robust/scoring_contract.jl`). A scorer `S <: AbstractScoring`
implements:

| verb | role |
|:--|:--|
| `scoring_workspace(scoring, n, d_g) -> W` | its scratch, built once per composed workspace |
| `score!(W, scoring, scores, residuals, n; sample, bound) -> Float64` | the score of a model from its residuals (higher is better) |
| `inlier_count(scoring, W)` | the consensus size ``k^*`` of the last `score!` (drives the adaptive stop) |
| `consensus_order(scoring, W)` | an order whose first ``k^*`` entries are the consensus set |

and optionally `is_predictive`/`scored_model` (predictive scorers),
`scale_prior` (default Jeffreys), `outlier_halfwidth` (the uniform outlier
model, if it has one), `inlier_threshold` with the static trait
`has_inlier_threshold` (its decision point, which a local optimizer with no
width follows) and `check_workspace`. VisualGeometryCore derives the rest once
for every scorer: `score_model!` (residuals → `score!` → consensus mask, the
one scoring path of the loop and of every local optimizer), `consensus_set`
and `consensus_s2` (the consensus set's Student-t scale ŝ², at `model_dof = m
+ 1` for local optimization and `m` for the result's σ̂), and
`inlier_probabilities!` (the result's soft weights: the Student-t + uniform
posterior when the scorer has an outlier model, 0/1 on the consensus set
otherwise).

## MSAC and RANSAC (the core default)

VisualGeometryCore's `MsacScoring` and `RansacScoring` at a threshold `τ` are
the baseline scorers. MSAC sums the truncated quadratic
``1 - \min(r_i^2, \tau^2)/\tau^2`` over the points; RANSAC counts
``r_i^2 < \tau^2``. Both are O(N) — no sort; the consensus set is a stable
partition of the indices below the threshold — and both define
`inlier_threshold = τ`.

```julia
using VisualGeometryCore: MsacScoring, RansacScoring

scoring = MsacScoring(problem; threshold = 2.0)    # codimension from the problem
scoring = RansacScoring(problem; threshold = 2.0)
result  = ransac(problem, scoring)
```

Every `find_*` builds `MsacScoring(problem; threshold)` when no `scoring` is
given, at the caller's `threshold` or the class's `[pipelines.<class>]` value
in `configs/default.toml`. The thresholds are in the problem's residual unit —
pixels for every pixel-data class, E and PnP included, whose residuals are
whitened into pixels by the cameras — and are conventions, not tuned values
(H, F, radial homography, radial camera 3 px; E 1 px; PnP 8 px). Hand-eye's
is a fraction of the characteristic length, `0.01·L`.

## Composing a scorer from another package

Any type implementing the contract is a `scoring`. The paper scorer of the
stack is RansacScoringDoneRight's `StudentTUniformScoring`: a scale-free
Student-t + uniform marginal likelihood that profiles the inlier set by
sort-and-sweep and marginalises σ² under a scale prior (Jeffreys by default,
`IGPrior(α₀, β₀)` optional). Its one width parameter is the outlier
half-width `a`, the half-width of the uniform outlier support:

```julia
using RansacScoringDoneRight: StudentTUniformScoring
using VisualGeometryCore: IGPrior

scoring = StudentTUniformScoring(problem, 20.0)                              # Jeffreys
scoring = StudentTUniformScoring(problem, 20.0; prior = IGPrior(2.0, 1.0))   # weakly informative
result  = find_homography(pairs; scoring)
```

The outlier half-width is the scorer's parameter, never a pipeline keyword:
`find_*(…; outlier_halfwidth = …)` is an `ArgumentError` that names this form.
The marginal score, its derivation and its sweep are documented with
RansacScoringDoneRight.

## Predictive scoring

A predictive scorer (`is_predictive(scoring) == true`, e.g.
`StudentTUniformScoring{Predictive}`) scores a hypothesis with its model
uncertainty: the loop asks `scored_model(scoring, problem, θ, sample)` for an
`Uncertain(θ, Σ̃_θ)`, the minimal solver's covariance from the problem's
`solver_covariance` (see [Covariance and Gauge Invariance](covariance.md)):

```math
\tilde\Sigma_{g_i} = \partial_x g_i\, \tilde\Sigma_{x_i}\, (\partial_x g_i)^\top
    + \partial_\theta g_i\, \tilde\Sigma_\theta\, (\partial_\theta g_i)^\top
```

A problem type that defines no solver covariance
(`provides_solver_covariance(P) == false`) cannot be scored predictively; the
fallback throws an `ArgumentError` naming the type.

**When to use**: NOT by default — it is an option. Reach for it when you want
the score to account for the conditioning of the minimal sample (high-leverage
points get wider acceptance bands), and measure it on your own data first.

**Measured verdict (2026-09-22)**: as a hypothesis criterion IN ISOLATION it
does not pay for itself. Over 320 tasks with thresholds tuned per scoring and
per stage and paired seeds, pooled ΔAUC@10° (predictive − certain) is a tie for
H (−0.003) and pose (−0.001); F gains at selection (+0.011) but loses after
local optimization (−0.029); E loses at selection (−0.010) and gains after it
(+0.007), its one clear win being E after LO at N = 60 (+0.016). The loss is
largest where the predicted variance is most heterogeneous across inliers
(sd log V: H 2.30, F 1.39, pose 1.29, E 0.97). It costs 13–82× per hypothesis
at N = 60 and 68–565× at N = 250 with RansacScoringDoneRight's kinetic sweep.
The search itself is EXACT: this is a property of the FIRST-ORDER predictive
model, not of the implementation. A post-release experiment (oracle covariance
against the first-order one) will settle whether linearization is the cause.
The tables are in the scratchpad under
`exp-certain-vs-predictive/` (`condensed_tables.md`, `results_table.txt`,
`rhoN_table.txt`, `diag_table.txt`, `bench_cost.jls`).

## Whitening Contract and Scoring Dispatch

The scoring layer dispatches on a single axis: **model type**.

| Model type | ``q_i`` | When |
|------------|---------|------|
| `model::M` (certain) | ``r_i^2`` | every model-certain scorer (MSAC, RANSAC, the marginal) |
| `model::Uncertain{M}` | ``r_i^2 / \tilde{v}_i`` | a predictive scorer (`is_predictive`), e.g. `StudentTUniformScoring{Predictive}`, with ``\Sigma_\theta \neq 0`` |

The key design invariant is the **whitening contract**: every `residuals!`
implementation must return whitened residuals such that ``r_i^2`` equals the
Mahalanobis distance ``g_i^\top \tilde\Sigma_{g_i}^{-1} g_i``. Whether the
problem has isotropic or heteroscedastic measurement noise is the problem's
responsibility — it handles the whitening inside `residuals!`. For example:

- **Isotropic** (``\tilde\Sigma_{x_i} = I``): the Sampson distance
  ``r_i = g_i / \sqrt{c_i}`` is already whitened, where ``c_i = \|\partial_x g_i\|^2``.
- **Heteroscedastic** (``\tilde\Sigma_{x_i}`` varies): `residuals!` must
  apply ``r_i = L_i^{-1} g_i`` where ``L_i L_i^\top = \tilde\Sigma_{g_i} = \partial_x g_i \, \tilde\Sigma_{x_i} \, (\partial_x g_i)^\top``.

Because `residuals!` handles whitening, a scorer only needs to square the
residuals (model-certain path) or additionally inflate by the predictive
variance (model-uncertain path). For the Student-t + uniform marginal the
measurement-space covariance penalty is zero on both paths (whitened outlier
formulation — the ``|\tilde\Sigma_{g_i}|^{-1/2}`` factors cancel between the
inlier Gaussian and the outlier uniform density in the whitened space); the
model-uncertain path adds only the model-uncertainty penalty
``\ell_i = \log|V_i|`` of the unit-scale predictive variance
``V_i = I + \partial_\theta g_i\, \tilde\Sigma_\theta\, \partial_\theta g_i^\top``.

## Local Optimization (LO-RANSAC)

See the [RANSAC guide](ransac.md#local-optimization-lo-ransac) for the local
optimization contract, the core strategies and the two-level gating that
wraps them.
