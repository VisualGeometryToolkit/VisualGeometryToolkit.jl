# Robust Estimation

!!! note "Division of labor"
    VisualGeometryToolkitCore owns the *contracts* every robust estimator composes
    through — the problem protocol, the scoring contract, the
    local-optimization contract — together with the robust-statistics
    toolkit, the loss family and the baseline scorers. The one generic RANSAC
    loop, the problems and the core local optimizations live in
    [VisualGeometryToolkitEstimation.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitEstimation);
    the paper scorers in RansacScoringDoneRight and the paper local
    optimizers in LocalOptimizationDoneRight implement the contracts. See
    [The composable RANSAC](@ref) below.

## Which function for which task

An estimator's name states its kind: a one-step algorithm is called by its
algorithm's name; a minimal solver runs through the `solve(solver, cs, idx)`
protocol; iterative fits (`fit_*`), robust searches (`find_*`) and
refinements from a guess (`refine_*`) live in VisualGeometryToolkitEstimation. Data go
in as correspondences (`a => b` pairs, a [`Csponds`](@ref) preferred), with
per-correspondence weights as its `weights` column.

| Task | Function | Kind | Package |
|------|----------|------|---------|
| Homography from ≥ 4 point correspondences | [`homography_dlt`](@ref)`(cs)` | one-step | VGC |
| Fundamental matrix from ≥ 8 correspondences | [`fundamental_dlt`](@ref)`(cs)` | one-step | VGC |
| Essential matrix from ≥ 8 bearing pairs | [`essential_dlt`](@ref)`(cs)` (PoseLib backend) | one-step | VGC |
| Rotation from vector pairs (`R·a ≈ b`) | [`kabsch`](@ref)`(cs)` | one-step | VGC |
| Mean of a set of rotations | [`chordal_mean`](@ref)`(Rs)` | one-step | VGC |
| Focal lengths from plane homographies | [`zhang_intrinsics`](@ref)`(Hs[, pitch])` | one-step | VGC |
| Line through 2D points, with its covariance | [`tls_line`](@ref)`(points)` | one-step | VGC |
| Normalize points for a solve (centroid, or a known centre) | `AffineMap(`[`HartleyNormalization`](@ref)`(; center), points)` | normalization | VGC |
| Normalize a correspondence set for a model class | [`CspondNormalization`](@ref VisualGeometryToolkitCore.CspondNormalization)`(M, cs; center)` | normalization | VGC |
| Store image points resolution-independently (pools) | `AffineMap(`[`DiagonalNormalization`](@ref)`(size))` | normalization | VGC |
| Homography from a 4-point sample | `solve(`[`Homography4pt`](@ref)`(), cs, idx)` | minimal | VGC |
| Radially distorted homography from 5 points | `solve(`[`Homography5ptRadial`](@ref)`(), cs, idx)` | minimal | VGC |
| Fundamental matrix from a 7-point sample | `solve(`[`Fundamental7pt`](@ref)`(), cs, idx)` | minimal | VGC |
| Essential matrix from 5 bearing pairs | `solve(`[`Essential5pt`](@ref)`(), cs, idx)` | minimal | VGC |
| Absolute pose from 3 point–bearing pairs | `solve(`[`P3P`](@ref)`(), cs, idx)` | minimal | VGC |
| Radial camera from 5 plane–pixel pairs | `solve(`[`RadialCamera5pt`](@ref)`(; center), cs, idx)` | minimal | VGC |
| A model from contaminated correspondences | `find_homography`, `find_absolute_pose`, `find_radial_camera`, … | find (RANSAC) | RVG |
| Radial camera of one board, whitened iterations | `fit_radial_camera` | fit | RVG |
| Division camera (focal and distortion) from boards | `fit_division_camera` (over `division_camera_step`) | fit | RVG |
| A lens model re-expressed in another family | `fit_lens` | fit | RVG |
| Line through points carrying covariances | `fit_line` | fit | RVG |
| Robot–world / hand–eye from a guess | `refine_robot_world_hand_eye` | refine | RVG |

The kind fixes how a call can fail: an algorithm is one deterministic computation
and can only be degenerate; a fit can fail to converge; a find is randomized;
a refine can end in a local minimum.

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
[`deploy`](@ref) — `loss_width = scale_multiple · ŝ`, the width as a multiple of
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
using VisualGeometryToolkitEstimation, RansacScoringDoneRight, LocalOptimizationDoneRight
find_homography(cs)       # core default: MsacScoring + Irls{TLSShape} at the scorer's threshold
find_homography(cs; scoring = StudentTUniformScoring(HomographyProblem, 20.0),  # a scorer (RSDR)
                    local_optimization = HQ{WelschShape}())                     # an optimizer (LODR)
```

Every dispatch is static: the composed workspace is typed on the scorer's and
the optimizer's workspaces, so one RANSAC call compiles per (problem, scoring,
local optimization).

**The problem protocol** ([`AbstractConstraintProblem`](@ref
VisualGeometryToolkitCore.AbstractConstraintProblem)): `data_size`, `codimension`,
`minimal_sample_size` and `constraint_with_jacobians`, from which `residuals!`
is derived (a problem may supply a faster `residual_context` /
`whitened_residual` path); optionally `solver_covariance` for predictive
scoring; and the weighted refit `refine!(R, problem, θ, w)` on
`refine_workspace(P, n)` that local optimizations call.

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
`inlier_threshold` with its static trait `has_inlier_threshold` (the decision
point a width-free optimizer follows), `score_bound` (a bounded-sum scorer may
stop a score early) and `check_workspace`. The derived verbs are stated once and
never overridden: `consensus_s2`, `consensus_set` (the `ConsensusSet`, the inlier
set and its Student-t scale that local optimization starts from),
`inlier_probabilities!` and `score_model!`, the one scoring path over a
`ScoreBuffers`. `with_width(scoring, w)` and `width_axis(scoring)` rebuild a
scorer at another width on its native axis.

The baseline scorers live here: [`RansacScoring`](@ref
VisualGeometryToolkitCore.RansacScoring) (inlier count) and [`MsacScoring`](@ref
VisualGeometryToolkitCore.MsacScoring) (truncated quadratic). `MsacScoring(problem;
threshold = τ)` is a loop scorer at a threshold in the problem's residual unit;
the width-free `MsacScoring()` is the grid family that pool selection sweeps
over a width grid.

**The local-optimization contract** (`AbstractLocalOptimization`):

```julia
lo_workspace(::Type{P}, ::Type{T}, lo, n) -> L     # built once; default nothing
locally_optimize!(L, lo, problem, θ, cs::ConsensusSet, buf::ScoreBuffers, scoring)
    -> lo_estimate(θ′, retcode; iterations, residuals_current)
```

On entry `residuals(buf)` holds θ's residuals and `cs` is the consensus set of
the score that just ran; the optimizer may overwrite the buffers and call
`score_model!`, but never scores for acceptance (the loop re-scores and keeps
the incumbent). A width-carrying optimizer has a width mode, read by `lo_width`
and set by `with_width(lo, w)`: `nothing` follows the scorer's
`inlier_threshold`, [`ScaleMultiple`](@ref VisualGeometryToolkitCore.ScaleMultiple)`(m)`
anchors the width at `m` times the consensus scale, and [`LossWidth`](@ref
VisualGeometryToolkitCore.LossWidth)`(w)` is an absolute width. `loss_at` is the one
resolution of a mode into a [`DeployedLoss`](@ref), and `check_composition`
refuses, once at entry, an optimizer that follows a scorer with no threshold.
The label trait `scale_interface` is derived from the mode
(`FollowsScorer`, `ConsensusAnchored`, `FixedScale`).

**External refiners.** A strategy that runs outside the loop (a third-party
solver) declares `RefitFamily(lo) = ExternalRefit()` and implements
`external_workspace(lo, problem, frame, scoring)`, optionally
`prepare_external!`, and `refine_external!(ws, lo, problem, θ, width, scoring) ->
θ′ | nothing`, a pure model-to-model refine at an absolute `LossWidth` (a
`FixedS2(s2)` noise scale is the width `√s2` unless the refiner reads it
itself). In-loop strategies are `NativeRefit()`, the default.

**The chart-rows contract** (`rows_provider`, `lift_init`, `lift_refresh!`,
`lift_jacobian!`, `lift_retract`, `lift_model`, `lift_valid`, `lift_rows`) —
what a lifted local optimizer reads of a chart problem: its whitened rows,
their Jacobian and per-point norms in a `RowsWorkspace`. Problems provide it
(VisualGeometryToolkitEstimation); the half-quadratic lifts and their continuation
schedules that consume it are LocalOptimizationDoneRight's.

**Pool selection** over a benchmark pool record (`Pools.PoolPair`):
`best_pool_model(scoring, pair, R)` is the per-pair argmax of `score!` over the
stored models; `select_on_grid!(idxs, P, scoring, grid, staged)` the
selections at every width of a grid, after `selection_staging(scoring)` and
`stage_selection(key, pairs, residuals; backend)`. The generic method is a CPU
fallback (per column `with_width`, then `best_pool_model`); a grid family with
device kernels (RansacScoringDoneRight) implements its own staging and
selection.

**Width axes.** One name per width quantity, each a value carrier and, bare, an
axis tag: [`ScaleMultiple`](@ref VisualGeometryToolkitCore.ScaleMultiple),
[`LossWidth`](@ref VisualGeometryToolkitCore.LossWidth) and
[`OutlierHalfwidth`](@ref VisualGeometryToolkitCore.OutlierHalfwidth) (a scorer's
width, never an optimizer's). `width_axis(name)` / `width_axis_name(axis)` map
the persisted names, and `to_native` / `native_map` convert a scale multiple
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
  [`AbstractLocalOptimization`](@ref), [`RefitFamily`](@ref),
  [`AbstractWeightPolicy`](@ref)) — the type hierarchy of the contracts above.
- **Generic block-coordinate-descent loop** ([`block_coordinate_descent!`](@ref),
  [`AbstractBlockCoordinateMethod`](@ref) and its extension verbs) — the
  model/nuisance/weights loop (EM, IRLS, GNC, Barron-adaptive) stated once, extended by downstream methods.
- **Chart-rows contract** (`rows_provider`, `lift_rows`, `RowsWorkspace`) —
  what a lifted local optimizer reads of a chart problem: its whitened rows,
  their Jacobian and per-point norms. The lifts and their continuation
  schedules are LocalOptimizationDoneRight's.
- **Chart LM** (`ManifoldLmWorkspace`) — the stack's Levenberg–Marquardt:
  FastLevenbergMarquardt's in-place `lmsolve!` on a linearization state's
  `residual_rows!` / `residual_jacobian_rows!`, in buffers allocated once.

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
ellipses and Hartley–Zisserman epipolar-line bands.

## API reference

See the [Robust Estimation API](../api/estimators.md) for the complete
reference.
