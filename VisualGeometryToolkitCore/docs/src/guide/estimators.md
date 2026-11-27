# Robust Estimation

!!! note "Division of labor"
    VisualGeometryToolkitCore owns the *contracts* every robust estimator composes
    through — the problem protocol, the scoring contract, the
    local-optimization contract — together with the robust-statistics
    toolkit, the loss family and the baseline scorers. The one generic RANSAC
    loop, the problems and the core local optimizations live in
    [VisualGeometryToolkitEstimation.jl](@extref VisualGeometryToolkitEstimation :doc:`index`);
    the paper scorers in RANSACScoringDoneRight and the paper local
    optimizers in LocalOptimizationDoneRight implement the contracts. See
    [The composable RANSAC](@ref) below.

## Which function for which task

Every estimator is `solve(problem, algorithm)` ([The estimation interface](@ref)):
the problem names what is estimated from which data — [`EstimationProblem`](@ref)`(M, data[, p])` from scratch,
[`MinimalProblem`](@ref)`(M, data, sample[, p])` for a minimal sample, [`RefineProblem`](@ref)`(f, u0)` from a guess
— and the algorithm, a typed value, how; its [`EstimatorKind`](@ref) states how it can fail. A closed form returns
a [`FitResult`](@ref) whose `u` is the model (`nothing` when the data are degenerate), a minimal solver every root
(`FixedModels`). Iterative fits, robust searches (`Ransac`) and refinements from a guess ([`LevenbergMarquardt`](@ref))
live in VisualGeometryToolkitEstimation. Data go in as correspondences (`a => b` pairs, a
[`Correspondences`](@ref) preferred), with per-correspondence weights as its `weights` column.

| Task | Function | Kind | Package |
|------|----------|------|---------|
| Homography from ≥ 4 point correspondences | `solve(EstimationProblem(HomographyMat, cs), `[`DLT`](@ref)`())` | closed form | VGC |
| Fundamental matrix from ≥ 8 correspondences | `solve(EstimationProblem(FundMat, cs), DLT())` | closed form | VGC |
| Essential matrix from ≥ 8 bearing pairs | `solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))` (PoseLib backend) | closed form | VGC |
| Rotation from vector pairs (`R·a ≈ b`) | `solve(EstimationProblem(RotMatrix{3}, cs), `[`Kabsch`](@ref)`())` | closed form | VGC |
| Mean of a set of rotations | `solve(EstimationProblem(RotMatrix{3}, Rs), `[`ChordalMean`](@ref)`())` | closed form | VGC |
| Absolute rotations from relative ones | `solve(EstimationProblem(`[`AbsoluteRotations`](@ref)`, rel), `[`SpectralAveraging`](@ref)`())` | closed form | VGC |
| Focal lengths from plane homographies | `solve(EstimationProblem(LogicalIntrinsics, Hs, (; center[, pitch])), `[`Zhang`](@ref)`())` | closed form | VGC |
| Line through 2D points, with its covariance | `solve(EstimationProblem(Line2D, points), `[`TotalLeastSquares`](@ref)`())` | closed form | VGC |
| The point nearest to N lines (`origin => direction` pairs) | `solve(EstimationProblem(Point, lines), `[`NearestPoint`](@ref)`())` | closed form | VGC |
| A 3D point from a bearing pair and the relative pose | `solve(EstimationProblem(Point3, f₁ => f₂, pose), `[`Triangulation`](@ref)`())` | closed form | VGC |
| Ellipse through 2D points (algebraic, always an ellipse) | `solve(EstimationProblem(Ellipse, points), `[`DirectEllipse`](@ref)`())` | closed form | VGC |
| A pure-bending height field through scattered heights (`point => height` pairs, the plane dropped) | [`HeightField`](@ref)`(nodes, points .=> heights; smoothing)` | one-step | VGC |
| Normalize points for a solve (centroid, or a known centre) | `AffineMap(`[`HartleyNormalization`](@ref)`(; center), points)` | normalization | VGC |
| Normalize a correspondence set for a model class | [`CorrespondenceNormalization`](@ref VisualGeometryToolkitCore.CorrespondenceNormalization)`(M, cs; center)` | normalization | VGC |
| Store image points resolution-independently (pools) | `AffineMap(`[`DiagonalNormalization`](@ref)`(size))` | normalization | VGC |
| Homography from a 4-point sample | `solve(MinimalProblem(HomographyMat, cs, idx), `[`Homography4pt`](@ref)`())` | minimal | VGC |
| Radially distorted homography from 5 points | `solve(MinimalProblem(RadialHomography, cs, idx), `[`Homography5ptRadial`](@ref)`())` | minimal | VGC |
| Fundamental matrix from a 7-point sample | `solve(MinimalProblem(FundMat, cs, idx), `[`Fundamental7pt`](@ref)`())` | minimal | VGC |
| Essential matrix from 5 bearing pairs | `solve(MinimalProblem(EssentialMat, cs, idx), `[`Essential5pt`](@ref)`())` | minimal | VGC |
| Absolute pose from 3 point–bearing pairs | `solve(MinimalProblem(CameraExtrinsics, cs, idx), `[`P3P`](@ref)`())` | minimal | VGC |
| Circle through 3 points | `solve(MinimalProblem(Circle, points, idx), `[`Circle3pt`](@ref)`())` | minimal | VGC |
| Ellipse through 5 points | `solve(MinimalProblem(Ellipse, points, idx), `[`Ellipse5pt`](@ref)`())` | minimal | VGC |
| Radial camera from 5 plane–pixel pairs | `solve(MinimalProblem(RadialCamera, cs, idx), `[`RadialCamera5pt`](@ref)`(; center))` | minimal | VGC |
| A model from contaminated correspondences | `solve(EstimationProblem(HomographyProblem(cs), p), Ransac(); rng)` (likewise `AbsolutePoseProblem`, `RadialCameraProblem`, `CircleProblem`, `EllipseProblem`, …) | find (RANSAC) | RVG |
| Radial camera of one board, whitened iterations | `RadialCameraFit` | fit | RVG |
| Division camera (focal and distortion) from boards | `DivisionCameraFit` (a stacked linear focal solve per iteration) | fit | RVG |
| A camera model re-expressed in another family (a lens family, a `SplineLens` and its forward spline) over a region | `CameraRegression` | fit | RVG |
| A model refined from a given one (no scorer: the caller's scale or width) | `solve(RefineProblem(HomographyProblem(cs), H₀), `[`LevenbergMarquardt`](@ref)`(); scale)` (any family problem) | refine | RVG |
| Robot–world / hand–eye from a guess | `LevenbergMarquardt` on its block problem | refine | RVG |

The kind fixes how a call can fail: a closed form is one deterministic computation
and can only be degenerate; a fit can fail to converge; a find is randomized;
a refine can end in a local minimum.

### Finds and refines

VisualGeometryToolkitEstimation's `Ransac` is RANSAC's sampling around one local-optimization step: on a
family problem it scores hypotheses and ORIGINATES a scale (its result's `noise_scale`). The
[`LevenbergMarquardt`](@ref) refine of the same family problem starts from a given model
(`RefineProblem(f, u0)`) and estimates no scale: it runs no scorer, and its robust
loss — `optimization_strategy = Triggs{Shape}` (the default, Welsch: one
robustified LM, no reweighting cycles; an `Irls{Shape}` is refused) or
`nothing` for least squares — runs at the caller's `scale`
(a number, or an upstream result such as the find's) times the loss's
[`efficiency_constant`](@ref), or at a given `width`. It returns the refined
model with per-datum loss weights and the scale it ran with. The two take one
family of [`AbstractOptimizationStrategy`](@ref) types under two keywords
(`local_optimization_strategy` of `Ransac`); the RANSAC-family seeded step,
with a scorer and its consensus set, is VisualGeometryToolkitEstimation's
`LocalOptimization`.

### What a problem implements to be refined

```@meta
CurrentModule = VisualGeometryToolkitCore
```

`LevenbergMarquardt` is the one host of every refine; a problem `f` opts in
through one of two contracts.

- **The chart contract** (one model on a minimal chart):
  [`model_chart`](@ref)`(f, θ)` the chart about the current model,
  [`chart_rows`](@ref)`(f, θ)` its √w-weighted residual rows,
  [`chart_model`](@ref)`(f, ch, θ)` the model back from the chart,
  [`chart_size`](@ref) its dimension, and the class's defaults
  [`default_optimization_strategy`](@ref) and [`default_stops`](@ref). A guess
  in a wider type than the problem's model is mapped in by [`guess_model`](@ref)
  and the estimate back by [`solution_model`](@ref) (the frame labels and units
  of the guess). The refine is `solve(RefineProblem(f, θ), LevenbergMarquardt(;
  optimization_strategy); stops, scale | width)`: plain least squares, or one robustified LM under a
  `Triggs{K}` strategy; `init` keeps its workspace across `reinit!(c, θ′; weights)`.
- **The block contract** (several variable blocks, a bundle):
  [`AbstractBlockProblem`](@ref) and its hooks: the problem at the guess
  ([`block_variables`](@ref), [`residual_blocks`](@ref), [`fixed_variables`](@ref); or
  [`block_structure`](@ref), the bundle stated once with what its result keeps of it),
  the summary it asks for ([`residual_summary`](@ref)) and its result
  ([`block_result`](@ref)), each residual block type answering
  [`with_robustifier`](@ref). Core builds and solves the multi-block LM, and
  tells a bundle that evaluates its blocks together (a device batch) the
  variable vectors each evaluation reads ([`prepare_evaluation!`](@ref); a
  no-op by default).

The robust loss is parsed once, from the solve's `scale`/`width` and the
algorithm's strategy, by [`bundle_loss`](@ref); the result records the scale it
ran with ([`recorded_scale`](@ref)) and its per-datum loss weights
([`loss_weights`](@ref), normalized by [`normalized_loss_weight`](@ref)) in a
[`RefineMetadata`](@ref), an [`AbstractFitMetadata`](@ref).
[`weighted_support`](@ref) and [`weighted_support_rms`](@ref) are the weight
mass of the data the loss kept and its residual RMS. A context `p` an algorithm
reads is checked by [`read_context`](@ref) (required keys, optional keys with
defaults, any other key refused), and a result's residual variance is
[`s2`](@ref)`(r)`.

## The robust-loss family

One unified family of loss SHAPES ([`AbstractLossShape`](@ref)) replaces
per-package loss-function copies. A loss shape is a normalized scale family
`ρ_w(r) = w²·ρ₁(r/w)`: every member shares the quadratic regime at the origin,
and its scale parameter `w`, the [`loss_width`](@ref VisualGeometryToolkitCore.loss_width), is the residual scale at
which it leaves that regime (`LOSSES.md` at the repository root states the
normalization). The shapes: [`GaussianShape`](@ref),
[`WelschShape`](@ref), [`GemanMcclureShape`](@ref),
[`CauchyShape`](@ref), [`SmoothTLSShape`](@ref), [`TLSShape`](@ref),
[`HuberShape`](@ref), and the Barron α-family dial
([`BarronShape`](@ref)). Traits classify them ([`liftable`](@ref),
[`bounded`](@ref)). A shape is paired with the width it runs at by
[`deploy`](@ref) — `loss_width = efficiency_constant · ŝ`, the width as a multiple of
the noise scale — and the resulting [`DeployedLoss`](@ref) answers the
one-argument verbs that read that width: [`lift_weight`](@ref),
[`lift_rho`](@ref), [`lift_conjugate`](@ref), [`lift_cost`](@ref),
[`lift_zblock`](@ref). [`decision_width`](@ref) converts a decision threshold
(the residual where a method's inlier/outlier decision flips) into each
shape's native width, and [`is_map_inlier`](@ref) is the MAP decision on a
posterior membership weight.

The GNC surrogate facet supplies the annealed closed form
([`gnc_tls_weight`](@ref)) and the engine vocabulary ([`gnc_weight`](@ref),
[`gnc_target`](@ref), [`gnc_mu0`](@ref)), all
dispatched on the [`gnc_surrogate`](@ref) trait.

## The composable RANSAC

A robust search is a composition of three parts, each behind a contract this
package owns: a **problem** (data, residuals, minimal solver, weighted refit), a
**scoring** (which hypothesis wins, and its consensus set), and a **local
optimization** (how the winner is refined). VisualGeometryToolkitEstimation's loop calls
only the contract verbs, so a scorer or an optimizer from any package plugs in
as a keyword:

```julia
using VisualGeometryToolkitEstimation, LocalOptimizationDoneRight
using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior
prob = EstimationProblem(HomographyProblem(cs))
solve(prob, Ransac(); rng)   # core default: the class's scorer (MSAC) + Irls{TLSShape} at its decision point
solve(prob, Ransac(; scoring = StudentTUniformScoring(HomographyProblem, 20.0;   # outlier half-width, pixels
                        scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
                    local_optimization_strategy = HQ{WelschShape}()); rng)       # an optimizer (LODR)
```

Every dispatch is static: the composed workspace is typed on the scorer's and
the optimizer's workspaces, so one RANSAC call compiles per (problem, scoring,
local optimization).

**The problem protocol** ([`AbstractConstraintProblem`](@ref
VisualGeometryToolkitCore.AbstractConstraintProblem)): `data_size`, `codimension`,
`minimal_sample_size` and `constraint_with_jacobians`, from which the signed
`whitened_residual` and the distances `mahalanobis_distances!` are derived (a
problem may supply a faster `residual_context` / `mahalanobis_distance` path); optionally `solver_covariance` for predictive
scoring; and the weighted refit `refine!(R, problem, θ, w)` on
`refit_workspace(P, n)` that local optimizations call.

**The scoring contract** (`AbstractScoring`). A scorer implements four verbs:

| verb | meaning |
|---|---|
| `scoring_workspace(scoring, n, d_g)` | its scratch, built once per composed workspace |
| `score!(W, scoring, scores, residuals, n; sample, bound)` | the score of one model (higher is better) |
| `inlier_count(scoring, W)` | the size `k*` of the consensus set of the last `score!` |
| `consensus_order(scoring, W)` | an order whose first `k*` entries are the consensus set |

Optional verbs have defaults: `is_predictive` and `scored_model` (a scorer that
scores a model with its covariance), `scale_prior` (Jeffreys),
`outlier_halfwidth` (a uniform outlier model, `nothing` by default),
`decision_boundary` with its static trait `has_decision_boundary` (the decision
point a width-free optimizer follows), `score_bound` (a bounded-sum scorer may
stop a score early) and `check_workspace`. The derived verbs are stated once and
never overridden: `consensus_set` (the `ConsensusSet`, the inlier
set and its Student-t scale that local optimization starts from),
`inlier_probabilities!` and `score_model!`, the one scoring path over a
`ScoreBuffers`. The consensus set's scale alone is an estimator, `ConsensusScale(scoring)`
on `EstimationProblem(SquaredScale, residuals, (; order, inlier_count, model_dof, codimension))`
(or on a family problem and a model, `(; model)`). `with_width(scoring, w)` and `width_axis(scoring)` rebuild a
scorer at another width on its native axis.

The baseline scorers live here: [`RansacScoring`](@ref
VisualGeometryToolkitCore.RansacScoring) (inlier count) and [`MsacScoring`](@ref
VisualGeometryToolkitCore.MsacScoring) (truncated quadratic). `MsacScoring(problem;
threshold = τ)` is a loop scorer at a threshold in the problem's residual unit;
the width-free `MsacScoring()` is the grid family that pool selection sweeps
over a width grid. Both are [`AbstractThresholdScoring`](@ref
VisualGeometryToolkitCore.AbstractThresholdScoring)s: a family at a threshold
supplies only its per-point `residual_quality` (and, where it is proven, its
`score_bound`) and inherits the one threshold `score!` with its consensus set,
so a downstream loss-width family shares the loop.

**The local-optimization contract** (`AbstractOptimizationStrategy`):

```julia
lo_workspace(::Type{P}, ::Type{T}, lo, n) -> L     # built once; default nothing
locally_optimize!(L, lo, problem, θ, cs::ConsensusSet, buf::ScoreBuffers, scoring)
    -> lo_estimate(θ′, code, rule; iterations, residuals_current)
```

On entry `residuals(buf)` holds θ's residuals and `cs` is the consensus set of
the score that just ran; the optimizer may overwrite the buffers and call
`score_model!`, but never scores for acceptance (the loop re-scores and keeps
the incumbent). A width-carrying optimizer has a width mode, read by `lo_width`
and set by `with_width(lo, w)`: `nothing` follows the scorer's
`decision_boundary`, [`EfficiencyConstant`](@ref VisualGeometryToolkitCore.EfficiencyConstant)`(m)`
anchors the width at `m` times the consensus scale, and [`LossWidth`](@ref
VisualGeometryToolkitCore.LossWidth)`(w)` is an absolute width. `loss_at` is the one
resolution of a mode into a [`DeployedLoss`](@ref), and `check_composition`
refuses, once at entry, an optimizer that follows a scorer with no threshold.
The label trait `scale_interface` is derived from the mode
(`FollowsScorer`, `ConsensusAnchored`, `FixedScale`).

**External refiners.** A strategy that runs outside the loop (a third-party
solver) declares `RefitFamily(lo) = ExternalRefit()` and implements
`external_workspace(lo, problem, frame, scoring)`, optionally
`prepare_external!`, and `external_refit!(ws, lo, problem, θ, width, scoring) ->
FitResult` (its `u` the refined model), a pure model-to-model refine at an absolute `LossWidth` (a
`FixedS2(s2)` noise scale is the width `√s2` unless the refiner reads it
itself). In-loop strategies are `NativeRefit()`, the default.

**The chart-rows contract** (`rows_provider`, `lift_init`, `lift_refresh!`,
`lift_jacobian!`, `lift_retract`, `lift_model`, `lift_valid`, `lift_rows`) —
what a lifted local optimizer reads of a chart problem: its whitened rows,
their Jacobian and per-point norms in a `RowsWorkspace`. Problems provide it
(VisualGeometryToolkitEstimation); the half-quadratic lifts and their continuation
schedules that consume it are LocalOptimizationDoneRight's.

**Pool selection** over a benchmark pool record (`Pools.PoolPair`):
`best_pool_model!(ws, scoring, pair, R)` is the per-pair argmax of `score!` over the
stored models; `select_on_grid!(idxs, P, scoring, grid, staged)` the
selections at every width of a grid, after `selection_staging(scoring)` and
`stage_selection(key, pairs, residuals; backend)`. The generic method is a CPU
fallback (per column `with_width`, then `best_pool_model!`); a grid family with
device kernels (RANSACScoringDoneRight) implements its own staging and
selection.

**Width axes.** One name per width quantity, each a value carrier and, bare, an
axis tag: [`EfficiencyConstant`](@ref VisualGeometryToolkitCore.EfficiencyConstant),
[`LossWidth`](@ref VisualGeometryToolkitCore.LossWidth) and
[`OutlierHalfwidth`](@ref VisualGeometryToolkitCore.OutlierHalfwidth) (a scorer's
width, never an optimizer's). `width_axis(name)` / `width_axis_name(axis)` map
the persisted names, and `to_native` / `native_map` convert an efficiency constant
into an axis's native width at the same decision point.

**Configuration tags and quality gates.** A TOML tag that selects a type goes
through a `TagRegistry`, whose tags are `snake_case` of the type name with the
family's suffix removed (`TLSShape` → `"tls"`, `StudentTUniform` →
`"student_t_uniform"`); a renamed tag fails with its successor's name. The
quality gates (`test_quality_gates`) check every registry, every config tag, the
stack dependency allow-list, and that a package adds methods to another stack
package's functions only through that owner's `CONTRACT_GENERICS`. The
conformance helpers `test_scoring_contract`, `test_local_optimization_contract`
and `test_constraint_problem` (loaded with `Test`) check an implementation.

## Interfaces and loops

- **Scoring/LO interface hierarchy** ([`AbstractScoring`](@ref),
  [`AbstractOptimizationStrategy`](@ref), [`RefitFamily`](@ref),
  [`AbstractWeightPolicy`](@ref)) — the type hierarchy of the contracts above.
- **Generic block-coordinate-descent loop** ([`BlockCoordinateDescent`](@ref),
  `solve(RefineProblem(prob, θ₀), BlockCoordinateDescent(method; likelihood); stops)`,
  [`AbstractBlockCoordinateMethod`](@ref) and its extension verbs) — the
  model/nuisance/weights loop (EM, IRLS, GNC, Barron-adaptive) stated once, extended by downstream methods.
- **Chart-rows contract** (`rows_provider`, `lift_rows`, `RowsWorkspace`) —
  what a lifted local optimizer reads of a chart problem: its whitened rows,
  their Jacobian and per-point norms. The lifts and their continuation
  schedules are LocalOptimizationDoneRight's.
- **The one LM** ([`LevenbergMarquardt`](@ref), `LmStops`; [`StaticLevenbergMarquardt`](@ref),
  `StaticLmStops`) — the stack's Levenberg–Marquardt: NLLSsolver (Marquardt's DᵀD damping by default,
  `LevenbergDamping` opt-in; the damping is the algorithm's field, not a stop setting) through ONE
  internal adapter, solved by `solve`/`init` on a `RefineProblem`: of a row problem
  ([`AbstractRowProblem`](@ref): a linearization state's `residual_rows!` / `residual_jacobian_rows!`), of a chart
  problem (the chart contract, robustified per datum under `Triggs{K}`), or of a bundle (the block contract:
  heterogeneous variables and residual blocks). The cache `init` returns keeps the problem and the solver's internal
  data across `reinit!`. Its stop settings have ONE owner, `LmStops` (configs/default.toml
  `[levenberg_marquardt.stops]`). Every solve has ONE success rule (Core 0.11.0):
  a tolerance stop (`ftol`, `xtol`, `xtol_rel`) is converged, and the
  consecutive-rejection cap is judged by the stationarity postcondition at the
  returned point (`is_stationary`), a cost below the final point's rounding
  ((16ε)²·Σᵢ sᵢ² over the residuals' operand scales, floored by the data's
  `residual_resolution`²) an exact fit. A kernel's fixed-N solve is the one device
  LM, `solve(RefineProblem(fj, x₀; lb, ub), StaticLevenbergMarquardt(); stops)`, under `StaticLmStops` and its own rule (stationary and accepted). No package calls
  NLLSsolver's solver API directly: `check_duplication` fails it. The Schur
  complement (NLLSsolver pull request #60, still
  open) is used when the loaded NLLSsolver has it; see "The
  Schur complement of the LM driver" in [Getting Started](getting-started.md).

## Model manifolds and covariance

Model manifolds ([`Manifold`](@ref): `EssentialManifold5`,
[`FundMatManifold7`](@ref), `HomographyManifold8`, `RigidManifold6`, …) provide
local parameterizations of the model manifolds with first-order
covariance transport between charts and the ambient space
([`transform_covariance`](@ref), the ambient chart
`EuclideanManifold(ch)`); [`implicit_covariance`](@ref) derives the
solver covariance by the implicit function theorem — of a minimal solve, and of
the whitened least-squares estimator on an overdetermined sample. Image-domain confidence regions
([`confidence_ellipse`](@ref), [`epipolar_band`](@ref),
[`LineConfidenceBand`](@ref)) turn those covariances into point-transfer
ellipses and Hartley–Zisserman epipolar-line bands [HartleyZisserman2003](@citep).

## API reference

See the [Robust Estimation API](../api/estimators.md) for the complete
reference.
