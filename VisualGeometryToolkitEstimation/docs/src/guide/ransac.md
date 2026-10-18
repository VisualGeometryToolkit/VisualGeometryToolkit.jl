# RANSAC

The RANSAC implementation is fully modular: the scorer, the local
optimization strategy and the sampler are swapped independently. The loop
calls only VisualGeometryToolkitCore's contract verbs, so a scorer or a strategy from
another package composes with every problem here.

## Basic Usage

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: csponds, SA, estimate, iterations, MsacScoring, TLSShape

src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

problem = HomographyProblem(csponds(src, dst))
scoring = MsacScoring(problem; threshold = 2.0)    # pixels
result  = ransac(problem, scoring; local_optimization = Irls{TLSShape}())
```

`ransac!(result_buffers, workspace, problem, scoring; …)` is the
allocation-free form: build `RansacWorkspace(problem, scoring, lo)` and
`RansacResultBuffers(workspace)` once and reuse them across calls.

## Scoring Functions

A scoring is any implementation of VisualGeometryToolkitCore's scoring contract.
The core ships MSAC and RANSAC at a threshold (VisualGeometryToolkitCore's
`MsacScoring`, `RansacScoring`); the paper scorer, the Student-t + uniform
marginal `StudentTUniformScoring`, is RansacScoringDoneRight's. See the
[Scoring Functions guide](scoring.md).

```julia
scoring = MsacScoring(problem; threshold = 2.0)            # the find_* default

using RansacScoringDoneRight: StudentTUniformScoring, Predictive
scoring = StudentTUniformScoring(problem, 20.0)            # outlier half-width, pixels
scoring = StudentTUniformScoring{Predictive}(problem, 20.0)
```

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop.
Whenever a minimal-sample model beats the current best global score
``S_{\mathrm{gl}}``, the loop takes the scorer's consensus set
(`consensus_set`) and hands it to the strategy's `locally_optimize!`
(VisualGeometryToolkitCore's local-optimization contract); the refined model is
re-scored, and a second gate ``S_{\mathrm{lo}}`` tracks the best locally
optimized score — the returned model is always the best LO result.

`ransac(problem, scoring)` defaults to `NoLocalOptimization()`; the `find_*`
pipelines read theirs from `configs/default.toml` — `Irls{TLSShape}()` for
every class.

### Strategy: `Irls{K}`

`Irls{K}` is static iteratively reweighted least squares on a
VisualGeometryToolkitCore loss shape `K` (`TLSShape`, `WelschShape`, `HuberShape`,
`GemanMcclureShape`, `CauchyShape`, `SmoothTLSShape`, `GaussianShape`), its
loss width frozen at the start of the call by its width mode:

| width mode | constructor | loss width |
|:--|:--|:--|
| follow the scorer (default) | `Irls{K}()` | the loss's decision point at the scorer's `inlier_threshold` (`decision_width(K(), τ)`; τ itself for `TLSShape`) |
| consensus-anchored | `Irls{K}(scale_multiple = m)` | `m·ŝ`, ŝ the consensus set's Student-t scale |
| absolute | `Irls{K}(loss_width = w)` | `w` |

Following a scorer with no `inlier_threshold` is refused at entry
(`check_composition`) with the way to give a width. `with_width(lo,
ScaleMultiple(m))` / `with_width(lo, LossWidth(w))` set the width of an
existing strategy.

```julia
result = ransac(problem, scoring; local_optimization = Irls{TLSShape}())                        # least squares on the inliers
result = ransac(problem, scoring; local_optimization = Irls{WelschShape}(scale_multiple = 2.5))
```

Each iteration computes the per-point weights ``w_i`` of the deployed loss at
the current residuals, then refits by the problem's
`refine!(ws, problem, θ, weights)`, routed by its `refit_kind` trait — the
chart problems are `NonlinLsq`: a Levenberg–Marquardt refit on a minimal chart
about the current model (tabled below). It stops when the weighted cost fails
to decrease, the refit fails, or after `max_iter` iterations, returning the
lowest-cost model.

### M-step dispatch

The M-step is implemented by `refine!(ws, problem, θ, weights)`, routed by the
problem's `refit_kind` trait (`NonlinLsq` for every problem below) to its
`_refine!` method — there is no fit-strategy marker. Each IRLS iteration runs one
full Levenberg–Marquardt solve, anchored at the incoming ``\theta``, at that
iteration's weights ``w_i``.

| Problem | M-step body | Notes |
|---------|-------------|-------|
| `HomographyProblem` | LM on the minimal 8-dof sl(3) chart ``H(x) = H_0 (I + \sum_q x_q G_q)`` (``G_q`` a traceless basis) — minimizes ``\sum_i w_i \|L_i^{-1}(\pi(H x_{1i}) - x_{2i})\|^2`` with ``L_i L_i^\top = \tilde\Sigma_{g_i}`` the Sampson covariance shape of the transfer error at ``H_0``. VisualGeometryToolkitCore's chart LM (FastLevenbergMarquardt's `lmsolve!`) on a `ManifoldLmWorkspace{8}`. | Analytic Jacobian (``\partial H / \partial x_q = H_0 G_q`` is constant). ``\tilde\Sigma_{g_i}`` is computed from the incoming ``\theta`` and frozen for the solve. |
| `FundMatProblem` | LM on the 7-dof Bartoli–Sturm chart ``F \sim U \, \mathrm{diag}(1, \sigma, 0) \, V^\top`` (MRP rotations of ``U`` and ``V``, additive ``\sigma``) — minimizes the weighted signed Sampson distance ``\sum_i w_i \, (g_i / \|\partial_x g_i\|)^2``. Same core, on a `ManifoldLmWorkspace{7}`. | Analytic Jacobian. ``F`` is rank 2 by construction; the one-dimensional gauge of near-essential seeds (``\sigma \to 1``) is absorbed by the LM damping. |
| `AbsolutePoseProblem` | LM on the 6-dof left-perturbation chart ``R = \exp(\delta\omega) R_0``, ``t = \exp(\delta\omega) t_0 + \delta t`` (VGC's `LeftRigidManifold6`) — calibrated reprojection residual via VGC's `constraint_with_jacobians(::Extrinsics3, X, ray)` (for a distorting lens, the raw-pixel residual of `constraint_with_jacobians(model, pose, X, u)`). The same chart LM, on a `ManifoldLmWorkspace{6}`. | Analytic 2×6 Jacobian: the projection block of `constraint_with_jacobians` times the chart's closed-form `action_jacobian` (``[-[R X + t]_\times \; I_3]``). |
| `HandEyeProblem` | LM on the 6-dof decoupled chart of ``X`` (``R = \exp(\delta\omega) R_0``, ``t = t_0 + \delta t``; VGC's `RigidExpManifold6`, the chart of its constraint and covariance) — the stacked 6-vector hand-eye residuals ``g_i``. The same chart LM. Each hypothesis is Tsai–Lenz fitted on its two motions (`fit_hypothesis`). | ForwardDiff Jacobian of the explicit residual: the ``G_\theta`` of `constraint_with_jacobians` is only a near-optimum approximation of it. |

### Configuration

```julia
Irls{K}(;
    max_iter = 25,         # maximum IRLS iterations per hypothesis ([local_optimization.irls])
    scale_multiple = m,    # or loss_width = w; neither: follow the scorer
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

result = ransac(problem, scoring; local_optimization = HQ{WelschShape}(scale_multiple = 1.6))
result = ransac(problem, StudentTUniformScoring(problem, 20.0); local_optimization = StudentTUniform())
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

The correspondence problems build their sampler themselves: uniform by
default, PROSAC when per-match `scores` are given (`sampler =` overrides).

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
result = ransac(problem, scoring; config=config)
```

## Result Type

`ransac` returns a `RansacEstimate`:

```julia
converged(result)      # false ⇔ retcode(result) === :no_model: NO estimate
estimate(result)       # estimated model (e.g., 3×3 matrix); a zero(M) placeholder when !converged
result.inlier_mask  # BitVector of inliers
result.quality      # quality score
iterations(result)  # for RANSAC, the number of trials (hypotheses drawn)
```

Additional fields available from `RansacMetadata` (all forward as properties of the result):

```julia
result.sigma        # σ̂ of the consensus set (consensus_s2 at the model's dof; √(RSS/ν) under the
                    # Jeffreys prior, floored at √eps); a LINEAR sigma
result.dof          # ν = n_inliers - p
```
