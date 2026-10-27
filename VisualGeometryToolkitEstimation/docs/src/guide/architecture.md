# Architecture

## Type Hierarchy

### Block Coordinate Descent

```
AbstractBlockCoordinateMethod (VGC)          model block of block_coordinate_descent!
├── PosteriorIrlsMethod                      weighted-refit IRLS (Irls{K}, fit_residual_mixture, …)
└── (LocalOptimizationDoneRight)             joint-(θ,z) loss lifts (HQ{K}) and GNC (Gnc{S})

AbstractWeightPolicy (VGC)                   nuisance + weight updates driven by block_coordinate_descent!
└── PosteriorLikelihood                      Student-t + uniform mixture (one of several policies)
```

The generic `block_coordinate_descent!` loop (VisualGeometryToolkitCore `robust/block_coordinate_descent.jl`)
dispatches on an `AbstractBlockCoordinateMethod` and a weight policy
(`AbstractWeightPolicy`). `PosteriorIrlsMethod` is the weighted-refit plug-in:
paired with `PosteriorLikelihood` and VisualGeometryToolkitCore's
`PosteriorWeightState` it is the Student-t + uniform EM of
`fit_residual_mixture` and `fit_division_camera` (and of
LocalOptimizationDoneRight's `StudentTUniform`); `Irls{K}` runs the same
method under a loss weight policy. The half-quadratic lifts and GNC are
LocalOptimizationDoneRight's and bring their own methods and policies.

### RANSAC Problems

```
AbstractRansacProblem{K}                     RANSAC problem interface, K = minimal sample size
├── LineFittingProblem
├── InhomLineFittingProblem
├── HandEyeProblem, RadialCameraProblem, RelposeBundleProblem
└── AbstractCspondProblem{K}                 Correspondence-based problems
    └── MinimalSolveProblem{M, K, T, ...}    the one parametric problem struct
        ├── HomographyProblem{T, ...}        alias: M = HomographyMat{T}, K = 4
        ├── FundMatProblem{T, ...}           alias: M = FundMat{T},       K = 7
        ├── EssentialProblem{T, ...}         alias: M = EssentialMat{T},  K = 5
        ├── AbsolutePoseProblem{...}         alias: M = Extrinsics3,      K = 3 (3D-2D)
        └── RadialHomographyProblem{T, K, ...}
```

`HomographyProblem`, `FundMatProblem`, `EssentialProblem`,
`AbsolutePoseProblem` and `RadialHomographyProblem` are exported `const`
aliases of the single parametric `MinimalSolveProblem{M, K, T, ...}` struct
(defined in `src/estimators/ransac/problems/cspond.jl`). The model matrix type `M`
selects per-model behaviour (codimension, minimal solver, Sampson whitening)
via Holy traits like `_model_codim(::Type{<:HomographyMat}) = 2`.

RANSAC problems implement: `minimal_sample_size`, `data_size`, `model_type`,
`codimension`, `solve`, `residuals!`, and optionally `test_sample`,
`test_model`, `test_consensus`, `refine!`, `constraint_with_jacobians` and, for
predictive scoring, `provides_solver_covariance` (the covariance itself is RVG's
one `solver_covariance`, the constraint sandwich over an index set).

### Scoring Functions

```
AbstractScoring (VGC)                        the scoring contract
├── MsacScoring{DG, T}, RansacScoring{DG, T} VGC's baseline scorers at a threshold
│                                            (MSAC is every find_*'s default)
└── StudentTUniformScoring{P, DG}            RANSACScoringDoneRight's Student-t + uniform marginal
                                             (one downstream implementation)
```

The loop reads a scorer only through VisualGeometryToolkitCore's scoring contract:
`scoring_workspace(scoring, n, d_g)` builds its scratch once, `score!` scores a
model's residuals, `inlier_count` and `consensus_order` give its consensus
set; the optional verbs (`is_predictive`, `scored_model`, `scale_prior`,
`outlier_halfwidth`, `inlier_threshold`, `check_workspace`) default in VGC. The
derived verbs `score_model!`, `consensus_set`, `consensus_s2` and
`inlier_probabilities!` are stated once in VGC and serve every scorer. See
[Scoring Functions](scoring.md).

### The composed workspace

```
RansacWorkspace{K, M, T, W, L}             built once, reused across trials and calls
├── buf::ScoreBuffers{T, W}                residuals, per-point scores, consensus mask,
│                                          the scorer's workspace W, a freshness epoch
├── lo_ws::L                               lo_workspace(P, T, lo, n): for the core strategies a
│                                          LoBuffers (refine_workspace(P, n) + posterior buffers)
└── best_* / sample_indices                the incumbent (model, residuals, scores, mask, order)
```

`RansacWorkspace(problem, scoring, local_optimization)` (or
`RansacWorkspace(P, n_max, scoring, local_optimization; T)`) composes it; the
no-scoring form `RansacWorkspace(P, n)` builds the default MSAC scorer's. All
dispatch on `W` and `L` is static.

## Design decisions

**The loop knows contracts, not types.** The RANSAC loop
(`src/estimators/ransac/loop.jl`) calls only VisualGeometryToolkitCore's contract
verbs: `scored_model`, `score_model!`, `inlier_count`, `consensus_order`,
`consensus_set`, `consensus_s2`, `inlier_probabilities!`, `is_predictive` and
`locally_optimize!`. It names no scorer or strategy type. Any scorer therefore
composes with any local optimizer and any problem, including the ones that
live in other packages.

**Dispatch is static.** `RansacWorkspace{K, M, T, W, L}` carries the type of
the scorer's workspace `W` and of the local optimizer's workspace `L`. The
loop compiles once for each combination of problem, scorer and local
optimizer, with no runtime dispatch. On a reused workspace, `ransac!` does not
allocate.

**A width is resolved once.** A strategy that carries a loss width, such as
`Irls{K, W}`, has one of three width modes. With no width it follows the
scorer's `inlier_threshold`. `ScaleMultiple(m)` anchors the width at `m` times
the consensus set's scale, and `LossWidth(w)` fixes it. VisualGeometryToolkitCore's
`loss_at` resolves the mode once per call. Following a scorer that defines no
threshold is refused at entry by `check_composition`.

**Local optimization goes through `refine!`.** A problem implements
VisualGeometryToolkitCore's `refine!(ws, problem, θ, weights)` on the workspace
`refine_workspace(P, n)`. It is the weighted refit from the current model, and
the problem's `refit_kind` trait routes it. For the homography, fundamental
matrix, essential matrix, absolute pose and hand-eye problems, the refit is
Levenberg–Marquardt on a minimal chart about the current model. The chart is
sl(3) with 8 degrees of freedom for a homography, Bartoli–Sturm with 7 for a
fundamental matrix, and so(3)⊕R³ with 6 for the two poses. The same chart
problems provide VisualGeometryToolkitCore's chart-rows contract
(`rows_provider`), which the half-quadratic lifts of
LocalOptimizationDoneRight consume.

**Cardinality and uncertainty are in the types.** The number of solutions a
minimal solver returns is carried by `FixedModels{N, M}`. A model with a
covariance is an `Uncertain{M}`, which the predictive scorers read. Covariances
are propagated back through the problem's normalization, so they come back in
the caller's units.

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop through
VisualGeometryToolkitCore's local-optimization contract:
`locally_optimize!(L, lo, problem, θ, cs::ConsensusSet, buf::ScoreBuffers,
scoring)` on the strategy's workspace `L`, returning `lo_estimate(θ′, retcode;
iterations, residuals_current)`. The core strategies are:

- `NoLocalOptimization` — no local optimization anywhere; the default of
  `ransac(problem, scoring)`. When selected, `_finalize` does **not** run LO
  either — the contract is "no LO, period."
- `Irls{K}` — static IRLS on a VGC loss shape `K` at a frozen width.
  `Irls{TLSShape}()` with no width (it follows the scorer's
  `inlier_threshold`) is the default of every `find_*` pipeline
  (`[pipelines.<class>]` in `configs/default.toml`).

Every other strategy — the half-quadratic lifts `HQ{K}`, `Gnc{S}`, the
Student-t + uniform IRLS `StudentTUniform`, the score-matched
block-coordinate ascent, the Barron arms and the PoseLib refiners — is
LocalOptimizationDoneRight's and plugs into the same contract.

### Block-coordinate-descent extension interface

Downstream packages (e.g., BlobBoards) implement their own methods by
extending a small set
of `public`-declared interface methods: `init_state!`, `update_model!`,
`update_residuals!`, `update_nuisance!`, `update_weights!`, `post_iteration!`,
`stopping_criterion` (with `stop_reason` for a new criterion), `block_coordinate_result`. The loop `block_coordinate_descent!(state, likelihood,
method, problem, θ₀)` dispatches on the method and calls through these hooks
generically. (There is no trust-region hook: each model update carries its
own damping.)

### Workspace buffers

Problems that support LO-RANSAC implement `refine!(ws, problem, θ, weights)`,
the local optimization's one step from the current model. The generic `refine!`
routes through the problem's `refit_kind` trait to `_refine!(::NonlinLsq, …)` — there is no
fit-strategy marker. Each `NonlinLsq` refit is Levenberg–Marquardt on a
minimal chart anchored at `θ`:

- `HomographyProblem`: 8-dof sl(3) chart, whitened transfer residual with
  the Sampson shape `Σ̃_{gᵢ}` computed from the current `θ` and frozen for
  the solve; VisualGeometryToolkitCore's chart LM, NLLSsolver through `lmsolve_chart!` (workspace:
  `ManifoldLmWorkspace{8}`, `refine_workspace(P, n)`, held in the local
  optimizer's workspace).
- `FundMatProblem`: 7-dof Bartoli–Sturm chart, signed Sampson residual; the
  same core (workspace: `ManifoldLmWorkspace{7}`).
- `AbsolutePoseProblem`: 6-dof left-perturbation chart (VGC `LeftRigidManifold6`),
  reprojection residual; the same core (workspace: `ManifoldLmWorkspace{6}`).
- `HandEyeProblem`: 6-dof decoupled chart (VGC `RigidExpManifold6`, the chart
  of its constraint and covariance); the same core with a ForwardDiff Jacobian
  (workspace: `ManifoldLmWorkspace{6}`). Each hand-eye hypothesis is Tsai–Lenz
  fitted on its two motions (`fit_hypothesis`).

The posterior-weight buffers (`posterior_w`, `posterior_inlier`,
`posterior_sq_r`) live on the core strategies' `LoBuffers`, beside the refit
workspace, so a posterior-seeded local optimization is zero-allocation per
hypothesis.

## Dependency on VisualGeometryToolkitCore

VisualGeometryToolkitEstimation imports geometry types, minimal solvers, the robust-loss
shapes (the `AbstractLossShape` family: `WelschShape`, `GemanMcclureShape`,
`CauchyShape`, `SmoothTLSShape`, `TLSShape`, `HuberShape`, `GaussianShape`,
`BarronShape`) and the generic block-coordinate-descent loop (`block_coordinate_descent!`,
`robust/block_coordinate_descent.jl`) from VGC. It extends VGC's `test_model`,
`test_sample`, `test_consensus`, `constraint_with_jacobians`, `solve` and the
loop's extension verbs with new methods. The loss shapes and the
local-optimization trait vocabulary (`ScaleInterface`, `optimization_path`, …)
are VGC's, exported by VGC; RVG re-exports nothing. The problem protocol
(`data_size`, `codimension`, `minimal_sample_size`, `residuals!`, `refine!`,
`refine_workspace`), the scoring and local-optimization contracts, the baseline
scorers and the Student-t / posterior-weighting toolkit are VGC's too; RVG
implements them for its problems and strategies.

Geometry-level model feasibility checks (`test_model` for `HomographyMat`,
`FundMat`) live in VGC and dispatch on model type. RVG problem wrappers
extract sample points and delegate to these VGC methods.

```
VisualGeometryToolkitCore (types, solvers, loss shapes, contracts, toolkit, MSAC/RANSAC scorers)
        ↓                                   ↓                         ↓
VisualGeometryToolkitEstimation              RANSACScoringDoneRight     LocalOptimizationDoneRight
(the RANSAC loop, problems,       (paper scorers; VGC only)  (paper strategies; VGC + RVG)
 NoLocalOptimization, Irls{K})
```

The user composes them: `find_*(…; scoring = <a RANSACScoringDoneRight scorer>,
local_optimization = <a LocalOptimizationDoneRight strategy>)`. VisualGeometryToolkitEstimation depends on
neither downstream package.
