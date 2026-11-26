# RANSAC

The RANSAC implementation is fully modular: the scorer, the local
optimization strategy and the sampler are swapped independently. The loop
calls only VisualGeometryToolkitCore's contract verbs, so a scorer or a strategy from
another package composes with every problem here.

## Basic Usage

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: correspondences, SA, MsacScoring, TLSShape, EstimationProblem, solve
using Random: Xoshiro

src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

f = HomographyProblem(correspondences(src, dst))
alg = Ransac(; scoring = MsacScoring(f; threshold = 2.0),    # pixels
             local_optimization_strategy = Irls{TLSShape}())
result = solve(EstimationProblem(f), alg; rng = Xoshiro(1))
```

`init(EstimationProblem(f), alg; rng)` → `solve!(cache)` is the allocation-free
form: the cache holds the composed `RansacWorkspace` and the result buffers, and
`reinit!(cache; f = HomographyProblem(other_cs))` points it at new data, keeping
them (rebuilt only when the new data outgrow them); `reinit!(cache; alg, rng)` gives
it a new configuration of the same `Ransac` type (a scorer at another width) or a new
rng (of the same type), re-resolved against the kept workspace. The result's `resid` and
weights ARE the cache's buffers, overwritten by the next `solve!`: copy what you keep. `Ransac(Float64; …)`
stores the per-point buffers in `Float64` (the default is `Float32`).

## Scoring Functions

A scoring is any implementation of VisualGeometryToolkitCore's scoring contract.
The core ships MSAC and RANSAC at a threshold (VisualGeometryToolkitCore's
`MsacScoring`, `RansacScoring`); the paper scorer, the Student-t + uniform
marginal `StudentTUniformScoring`, is VisualGeometryToolkitCore's (its research outlier models and grids RANSACScoringDoneRight's). See the
[Scoring Functions guide](scoring.md).

```julia
scoring = MsacScoring(problem; threshold = 2.0)            # the class default's form (MSAC)

using VisualGeometryToolkitCore: StudentTUniformScoring, Predictive, JeffreysPrior, FlatLabelPrior
scoring = StudentTUniformScoring(problem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())            # outlier half-width, pixels
scoring = StudentTUniformScoring{Predictive}(problem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
```

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop.
Whenever a minimal-sample model beats the current best global score
``S_{\mathrm{gl}}``, the loop takes the scorer's consensus set
(`consensus_set`) and hands it to the strategy's `locally_optimize!`
(VisualGeometryToolkitCore's local-optimization contract); the refined model is
re-scored, and a second gate ``S_{\mathrm{lo}}`` tracks the best locally
optimized score — the returned model is always the best LO result.

`Ransac()`'s `local_optimization_strategy = ClassDefault()` reads the problem
class's from `configs/default.toml` (`[pipelines.<class>]`: `Irls{TLSShape}()`
for most classes); a problem without a class takes its scorer's,
`default_local_optimization_strategy(scoring)` (`nothing`, no local
optimization, for the core scorers).

### Strategy: `Irls{K}`

`Irls{K}` is static iteratively reweighted least squares on a
VisualGeometryToolkitCore loss shape `K` (`TLSShape`, `WelschShape`, `HuberShape`,
`GemanMcclureShape`, `CauchyShape`, `SmoothTLSShape`, `GaussianShape`), its
loss width frozen at the start of the call by its width mode:

| width mode | constructor | loss width |
|:--|:--|:--|
| follow the scorer (default) | `Irls{K}()` | the loss's decision point at the scorer's `decision_boundary` (`decision_width(K(), τ)`; τ itself for `TLSShape`) |
| consensus-anchored | `Irls{K}(efficiency_constant = m)` | `m·ŝ`, ŝ the consensus set's Student-t scale |
| absolute | `Irls{K}(loss_width = w)` | `w` |

Following a scorer with no `decision_boundary` is refused at entry
(`check_composition`) with the way to give a width. `with_width(lo,
EfficiencyConstant(m))` / `with_width(lo, LossWidth(w))` set the width of an
existing strategy.

```julia
Ransac(; scoring, local_optimization_strategy = Irls{TLSShape}())                         # least squares on the inliers
Ransac(; scoring, local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5))
```

Each iteration computes the per-point weights ``w_i`` of the deployed loss at
the current residuals, then refits by the problem's
`refine!(ws, problem, θ, weights)`, routed by its `refit_kind` trait — the
chart problems are `NonlinLsq`: a Levenberg–Marquardt refit on a minimal chart
about the current model (tabled below). It stops when the weighted cost fails
to decrease, the refit fails, or after `max_iter` iterations, returning the
lowest-cost model.

Two weight rules hold for every shape:

- **A non-finite residual weighs 0.** A datum whose squared residual is `Inf`
  or `NaN` (an out-of-reach point, an infinite Sampson distance) gets weight 0
  at the seed and at every refit, so it can neither make the cost
  ``\sum_i w_i r_i^2`` infinite — which would stop the iteration at its seed —
  nor enter the refit. The robust shapes already weigh an infinite residual 0;
  the rule adds the Gaussian shape and the `NaN` residual.
- **The Gaussian shape refits the consensus set.** `Irls{GaussianShape}` is
  least squares, which no outlier may enter: it weighs the scorer's consensus
  set (the first `inlier_count` of its `consensus_order`, the set
  `consensus_mask!` marks) at unit weight and every other datum 0 — the
  LO-RANSAC least squares on the inliers. No threshold of its own: the set is
  the scorer's decision. The robust shapes weigh every datum by their loss.

### Strategy: `Triggs{K}`

`Triggs{K}` is the robustified chart LM: one Levenberg–Marquardt solve from the
hypothesis on the chart problems' rows, the loss `K` at its width carried by
NLLSsolver's Triggs correction (no reweighting cycles). The two rules fix its
per-datum weights, 0 or 1, once, at the hypothesis, for that one solve: a datum
whose residual at the hypothesis is not finite weighs 0 under every shape (its
row would make the LM's cost infinite at its start), and `Triggs{GaussianShape}`
weighs only the scorer's consensus set. With neither rule excluding a datum,
every row is at unit weight, as before. An LM increment that puts a weighed datum
out of reach has a non-finite cost and is rejected; a row that is non-finite in
Float64 stays so at weight 0 (`0·Inf`), as in every chart refit. The inliers are
the loop's re-score's, as for `Irls{K}`.

### M-step dispatch

The M-step is implemented by `refine!(ws, problem, θ, weights)`, routed by the
problem's `refit_kind` trait (`NonlinLsq` for every problem below) to its
`_refine!` method — there is no fit-strategy marker. Each IRLS iteration runs one
full Levenberg–Marquardt solve, anchored at the incoming ``\theta``, at that
iteration's weights ``w_i``. Every one is VisualGeometryToolkitCore's one
`LevenbergMarquardt`: the problem's refit workspace (`refit_workspace(problem)`, a
`RefitSlot`) holds the summary-free refine cache
`init(RefineProblem(p, zero(M)), LevenbergMarquardt(; optimization_strategy = nothing); weights, summary = nothing)`,
and each refit re-points it, `solve!(reinit!(c, θ; f = p, weights = w)).u`. The
Triggs local optimization holds a second cache at
`LevenbergMarquardt(; optimization_strategy = Triggs{K}())` with
`width = deployed_loss`. The relative-pose bundle is a block problem, which Core
builds and solves as a multi-block LM.

**A zero-weight row is exactly 0.** Every chart refit — this M-step, a
`LevenbergMarquardt` refine's least squares and both Triggs paths — builds the
rows through Core's contract generics
[`refit_chart`](@extref VisualGeometryToolkitCore.refit_chart)`(f, θ, weights)`
(default `model_chart`) and
[`refit_rows`](@extref VisualGeometryToolkitCore.refit_rows)`(f, chart, sqrtw)`
(default `chart_rows`), which this package implements as its `_refit_chart` and
`_refit_state`, and forms them as
``\sqrt{w_i}\,e_i`` in one place. A datum at weight 0 whose own residual is
`Inf` or `NaN` would give ``0 \cdot \infty = \mathrm{NaN}`` and stop the LM at
its start, so the rows (and Jacobian rows) at a zero ``\sqrt{w}`` are set to 0
whatever the problem's row is. A zero weight is the local optimization's or
the problem's own (a PnP point out of the camera's reach at the anchor). A row
at a nonzero weight is the problem's own, bit for bit.

| Problem | M-step body | Notes |
|---------|-------------|-------|
| `HomographyProblem` | LM on the minimal 8-dof sl(3) chart ``H(x) = H_0 (I + \sum_q x_q G_q)`` (``G_q`` a traceless basis) — minimizes ``\sum_i w_i \|L_i^{-1}(\pi(H x_{1i}) - x_{2i})\|^2`` with ``L_i L_i^\top = \tilde\Sigma_{g_i}`` the Sampson covariance shape of the transfer error at ``H_0``. VisualGeometryToolkitCore's chart LM, the summary-free `LevenbergMarquardt` refine on the problem's `RefitSlot`. | Analytic Jacobian (``\partial H / \partial x_q = H_0 G_q`` is constant). ``\tilde\Sigma_{g_i}`` is computed from the incoming ``\theta`` and frozen for the solve. |
| `FundMatProblem` | LM on the 7-dof Bartoli–Sturm chart ``F \sim U \, \mathrm{diag}(1, \sigma, 0) \, V^\top`` (MRP rotations of ``U`` and ``V``, additive ``\sigma``) — minimizes the weighted signed Sampson distance ``\sum_i w_i \, (g_i / \|\partial_x g_i\|)^2``. Same core. | Analytic Jacobian. ``F`` is rank 2 by construction; the one-dimensional gauge of near-essential seeds (``\sigma \to 1``) is absorbed by the LM damping. |
| `AbsolutePoseProblem` | LM on the 6-dof left-perturbation chart ``R = \exp(\delta\omega) R_0``, ``t = \exp(\delta\omega) t_0 + \delta t`` (VGC's `LeftRigidManifold6`) — calibrated reprojection residual via VGC's `constraint_with_jacobians(::`CameraExtrinsics`, X, ray)` (for a distorting lens, the raw-pixel residual of `constraint_with_jacobians(model, pose, X, u)`). The same chart LM. | Analytic 2×6 Jacobian: the projection block of `constraint_with_jacobians` times the chart's closed-form `action_jacobian` (``[-[R X + t]_\times \; I_3]``). |
| `HandEyeProblem` | LM on the 6-dof decoupled chart of ``X`` (``R = \exp(\delta\omega) R_0``, ``t = t_0 + \delta t``; VGC's `RigidExpManifold6`, the chart of its constraint and covariance) — the stacked 6-vector hand-eye residuals ``g_i``. The same chart LM. Each hypothesis is Tsai–Lenz fitted on its two motions (`sample_hypothesis`). | ForwardDiff Jacobian of the explicit residual: the ``G_\theta`` of `constraint_with_jacobians` is only a near-optimum approximation of it. |

### Configuration

```julia
Irls{K}(;
    max_iter = 25,         # maximum IRLS iterations per hypothesis ([local_optimization_strategy.irls])
    efficiency_constant = m,    # or loss_width = w; neither: follow the scorer
)
```

### Strategies from other packages

The paper's strategies are LocalOptimizationDoneRight's and implement the
same contract: the half-quadratic lifts `HQ{K}` (over VisualGeometryToolkitCore's
chart-rows contract, which the chart problems here provide), `Gnc{S}`, the
Student-t + uniform IRLS `StudentTUniform` (on this package's
`PosteriorIrlsMethod` / `PosteriorLikelihood`), the score-matched
block-coordinate ascent and the PoseLib refiners:

```julia
using LocalOptimizationDoneRight: HQ, StudentTUniform
using VisualGeometryToolkitCore: WelschShape

Ransac(; scoring, local_optimization_strategy = HQ{WelschShape}(efficiency_constant = 1.6))
Ransac(; scoring = StudentTUniformScoring(f, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
       local_optimization_strategy = StudentTUniform())
```

### Two-Level Gating

The RANSAC loop maintains two independent best-so-far scores:

1. **Global gate** (cheap): score the minimal-sample model (`score_model!`),
   producing ``S_g``. Reject if ``S_g \le S_g^*``.
2. **Local gate** (expensive): run LO-RANSAC refinement + re-score, producing
   ``S_l``. Accept only if ``S_l > S_l^*``.

Global scores compete with global scores, local with local. The global gate
filters unpromising trials before the expensive LO inner loop.

## Samplers

| Sampler | Description |
|---------|-------------|
| `UniformSampler(n)` | Uniform random sampling from `1:n` |
| `ProsacSampler(sorted_indices, m; T_N)` | Progressive sampling using match quality ordering |
| `ExhaustiveSampler(n, k)` | Every `k`-subset of `1:n` exactly once; no adaptive stop |

The correspondence problems build their sampler themselves: uniform by
default, PROSAC when per-match `scores` are given (`sampler =` overrides).

The sampler also owns the run's schedule (`trial_budget`, `adapts`). A random
sampler runs `config.max_trials` trials, shortened adaptively once a good
consensus is seen. The `ExhaustiveSampler` runs exactly `binomial(n, k)`
trials whatever the `RansacConfig`: for small data, where every minimal sample
is cheap, all of them are tried.

## Configuration

`RansacConfig` controls the RANSAC loop. `RansacConfig()` is the `[ransac]`
table of `configs/default.toml` (the values below); keywords override single
fields, and `RansacConfig(path)` loads another file's `[ransac]` over it:

```julia
config = RansacConfig(;
    success_prob   = 0.99,     # η: desired probability of an all-inlier sample
    max_trials = 10000,
    min_trials = 100,      # overrides adaptive stopping
)
result = solve(EstimationProblem(f), Ransac(; scoring, config); rng = Xoshiro(1))
```

### Selecting a strategy in TOML

A local-optimization strategy can be selected by tag, for instance from a
configuration file. `optimization_strategy_from_toml` reads one strategy table.
The core registry `VisualGeometryToolkitEstimation.OPTIMIZATION_STRATEGY_TAGS` holds the tags
`"none"` (no local optimization, `nothing`), `"irls"`, `"triggs"` and
`"terminal_lo"` (the terminal placement,
`{ type = "terminal_lo", inner = { … } }`: the race on raw scores, then one LO of
its incumbent by the inner strategy):

```julia
using VisualGeometryToolkitEstimation: optimization_strategy_from_toml

lo = optimization_strategy_from_toml(Dict("type" => "irls", "loss" => "welsch", "efficiency_constant" => 2.5);
    class = :homography)
```

A table without a width key follows the scorer; `efficiency_constant` or
`loss_width` sets the width. Keys that are absent inherit the shipped
`configs/default.toml`, and an unknown key or tag is an `ArgumentError`.
Strategies from another package load through an explicit merge of its
registry, for example LocalOptimizationDoneRight's:

```julia
import LocalOptimizationDoneRight
using VisualGeometryToolkitEstimation: OPTIMIZATION_STRATEGY_TAGS

lo = optimization_strategy_from_toml(Dict("type" => "hq", "loss" => "welsch", "efficiency_constant" => 1.6);
    class = :homography, tags = merge(OPTIMIZATION_STRATEGY_TAGS, LocalOptimizationDoneRight.LODR_STRATEGY_TAGS))
```

## Result Type

A `Ransac` (and a `LocalOptimization`) solve returns a `RansacEstimate`, read by
field and, for the cross-type quantities, by function:

```julia
using VisualGeometryToolkitCore: inlier_mask, weight_kind, noise_scale, scale_source, dof, found_model, successful_retcode

found_model(result)          # false ⇔ NO estimate (result.retcode == ReturnCode.Failure)
successful_retcode(result)   # strict: the confidence reached (or every sample drawn); false at the max_trials cap
result.retcode, result.stop_rule   # the termination record: a ReturnCode and a StopRule
result.u               # estimated model (e.g., 3×3 matrix); `nothing` when !found_model
result.resid           # each datum's residual at the model (storage precision)
weights(result)        # per-point weights in [0, 1]: 1/0 membership (MSAC/RANSAC), the posterior (Student-t)
weight_kind(result)    # BinaryWeights() or PosteriorWeights(): the scorer's, from the result's type
inlier_mask(result)    # BitVector of inliers, cut by the kind: w > 0, or the MAP rule w > ½
result.n, result.rms   # the weights' support count(w > 0), and the weighted rms √(Σwr²/Σw)
result.metadata.quality   # quality score
result.stats.nsteps    # for RANSAC, the number of trials (hypotheses drawn)
```

The scale is the one the scorer ORIGINATES, from its consensus partition:

```julia
noise_scale(result)    # ŝ of the consensus set (consensus_s2 at the model's dof; √(RSS/ν) under the
                       # Jeffreys prior, floored at √eps); a LINEAR scale. The hand-off: a later
                       # stage takes it as `scale = result`
scale_source(result)   # ScaleEstimated; ScaleNone (no model, or a set too thin to scale) and
                       # noise_scale refuses it
dof(result)            # ν = d_g·(n_partition − p)
```
