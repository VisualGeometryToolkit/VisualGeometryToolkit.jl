# Fitting Algorithms

Every estimator of the package is `solve(problem, algorithm)`. The algorithm's
KIND ([`estimator_kind`](@extref VisualGeometryToolkitCore.estimator_kind))
states how it computes and so how it can fail:

| kind | problem | algorithms here |
|:--|:--|:--|
| find (randomized; decides the inliers) | `EstimationProblem(f, p)`, `f` a family problem | [`Ransac`](@ref) |
| refine (from a guess; can end in a local minimum) | `RefineProblem(f, θ₀[, p])` | VisualGeometryToolkitCore's `LevenbergMarquardt`; [`LocalOptimization`](@ref) |
| fit (seeds itself; can fail to converge) | `EstimationProblem(M, data[, p])` | [`ReweightedLeastSquares`](@ref), [`RotationAveraging`](@ref), [`FilteredRotationAveraging`](@ref), [`ResidualMixture`](@ref), [`CameraRegression`](@ref), [`PSpline`](@ref), [`ForwardSpline`](@ref), [`OpticalAxis`](@ref), [`RadialCameraFit`](@ref), [`DivisionCameraFit`](@ref), [`SimFitPP`](@ref), [`PooledScale`](@ref) |

One-step algebraic algorithms (closed forms and minimal solvers) are
VisualGeometryToolkitCore's: `DLT`, `Kabsch`, `TotalLeastSquares`, `Homography4pt`, ….

## Finds: `Ransac` on a family problem

A family problem holds the data and its own construction context (a camera, a
distortion centre, the hand-eye motions); the find's context `p` is the
optional validity region, `(; extent)`:

| family problem | data | default MSAC `threshold` |
|:--|:--|:--|
| [`HomographyProblem`](@ref)`(cs)`, [`FundMatProblem`](@ref)`(cs)` | pixel correspondences | 3 px |
| [`RadialHomographyProblem`](@ref)`(cs; center)`, [`RadialCameraProblem`](@ref)`(cs; center)` | board ↔ pixel correspondences | 3 px |
| [`EssentialProblem`](@ref)`(camera1, camera2, cs)`, [`RotationProblem`](@ref)`(camera1, camera2, cs)` | pixel correspondences, back-projected by the cameras | 1 px |
| [`AbsolutePoseProblem`](@ref)`(camera, X .=> u)` | world point ↔ pixel | 8 px |
| [`HandEyeProblem`](@ref)`(A .=> B; motion_pairs, char_length)` | rig-pose ↔ pattern-pose pairs | `0.01·L` (a fraction of the characteristic length) |
| [`CircleProblem`](@ref)`(points)`, [`EllipseProblem`](@ref)`(points)` | image points | 1 px |

`Ransac()` reads its defaults from the problem's class, the
`[pipelines.<class>]` table of `configs/default.toml`: the `scoring` table
(read by [`scoring_from_toml`](@ref VisualGeometryToolkitEstimation.scoring_from_toml) — every class's is MSAC,
`{ type = "msac", threshold = … }`) and an `Irls{TLSShape}()` local optimization
following the scorer's decision point. The thresholds are conventions, not
tuned values. `Ransac(; scoring = …)` replaces the scorer and
`local_optimization_strategy = …` the strategy; a bare `threshold` or
`outlier_halfwidth` keyword is refused, a width being its scorer's.

```julia
f = HomographyProblem(correspondences(src, dst))
r = solve(EstimationProblem(f), Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 2.0)); rng = Xoshiro(1))
H = r.u                      # the model; `nothing` when found_model(r) is false
inliers = inlier_mask(r)
```

A find originates a scale: its consensus partition's ŝ, the result's
`noise_scale`. A family problem holds any number of data; on fewer than a
minimal sample the find draws no trial and returns the no-model result (`(Failure, TrialBudget)`).

## Refines: `LevenbergMarquardt` on the same problem

A refine starts from the model the caller gives and NEVER estimates a scale: it
runs no scorer, forms no consensus set and drops no datum. Its robust loss runs
over every datum at the scale the caller gives, frozen:

```julia
r = solve(EstimationProblem(f), Ransac(); rng = Xoshiro(1))         # the model, its inliers, its scale ŝ
h = solve(RefineProblem(f, r.u), LevenbergMarquardt(); scale = r)   # Triggs{Welsch} at efficiency_constant(Welsch, 2) · ŝ
noise_scale(h) == noise_scale(r)                                     # the scale it ran with, handed on (ScaleUpstream)
```

The strategy is the algorithm's `optimization_strategy`: `Triggs{Shape}` (the
default, `ClassDefault()`: `configs/default.toml [refine.<class>]`), one
robustified chart LM, the one fixed-scale robust path (an `Irls{Shape}` is
refused, naming its `Triggs{Shape}`); or `nothing`, plain least squares. A
robust strategy needs EXACTLY ONE of the solve keywords `scale = σ` (a number,
or an upstream result whose `noise_scale` it takes) and `width = w` (the loss
width itself); least squares takes neither. The loss width is the strategy's
efficiency constant times the scale — VisualGeometryToolkitCore's
`efficiency_constant(Shape, d)` at the problem's codimension unless the strategy
states one (TLS has none: state one, or give a width). The LM's stops are the
solve keyword `stops`.

The result is a `FitResult` with VisualGeometryToolkitCore's `RefineMetadata`:
the refined model `u`, the per-datum distances at it `resid`, the per-datum
`weights` (the loss's ρ′; all ones under least squares) and the scale record
(`noise_scale`, `scale_source`, `loss_width`). `n` is the weights' support and
`rms` the weighted one. There is no inlier mask: a loss weight has no cut of its
own (`inlier_mask(result, cut)` states one). On fewer data than a minimal sample
the guess comes back unrefined, `(InitialFailure, NoRule)`.

The RANSAC-family seeded refine — a scorer, its consensus set, the strategy
anchored on it, the consensus test on the result — is
[`LocalOptimization`](@ref)`(strategy; scoring)` on `RefineProblem(f, θ₀[, (; extent)])`,
with any strategy the problem hosts.

The absolute pose also has a problem with no find:
[`BearingPoseProblem`](@ref)`(X .=> f)`, a camera given by its rays, possibly
with per-ray origins (a non-central camera, for which there is no minimal
solver) — refined by `LevenbergMarquardt` like any other.

The block problems — [`CameraPosesProblem`](@ref), [`RobotWorldHandEyeProblem`](@ref),
[`RelativePoseProblem`](@ref), [`CameraRegressionProblem`](@ref) — are
refined by the same `LevenbergMarquardt` through VisualGeometryToolkitCore's
block contract, with the same strategy, default and scale input.

## Units and frames at the boundary

The absolute-pose, bearing-pose and hand-eye problems take plain or physical
data (lengths), all one form: physical data are stripped in metres once at
construction, and the solve's model comes back in metres (a find's, and a
refine's from a physical guess). A refine's pose keeps its guess's frame labels.
The plane problems (homography, radial homography, radial camera) refuse
physical plane points: the model cannot record their unit.

## Robust fits: one strategy, one fixed scale

The reweighted fits — [`ReweightedLeastSquares`](@ref) on a linear model, a
rotation from direction pairs, a rigid alignment and a robot-world
calibration; [`RotationAveraging`](@ref) — keep their own objective (a least
squares, a Kabsch, the staged hand-eye solve). Their only robustness is
`optimization_strategy = Irls{K}(…)`, an M-estimator weighting at ONE loss
width, set from the solve's scale input and never re-estimated;
`optimization_strategy = nothing` is the plain least squares. The scale input is
exactly one of the solve keywords:

- `scale = σ` — the residuals' noise scale (`ScaleGiven`);
- `scale = result` — an upstream robust result's `noise_scale(result)`
  (`ScaleUpstream`), e.g. a find's consensus scale handed down;
- `width = w` — the loss width itself (the result then records `ScaleNone`).

The width is `w`, or `c · σ` with `c` the strategy's efficiency constant, by
default `efficiency_constant(K, d)` at the fit's residual codimension `d`.
Neither, both, a scale given to a least-squares fit, a refine's `Triggs`, or a
shape with no efficiency constant and no width is an `ArgumentError`. The
result records `scale`/`scale_source` (`noise_scale(result)` hands it on), its
`loss_width`, and per-datum `weights(result)`: the loss weights `ρ′` normalized
to 1 (`IrlsWeights`; `inlier_mask(result, cut)` needs a cut), or lazy unit
weights (`BinaryWeights`) for least squares.

```julia
R = solve(EstimationProblem(RotMatrix{3}, cs), ReweightedLeastSquares(); scale = rotation_find_result)
align = solve(EstimationProblem(EuclideanMap{3}, centres .=> truth),
              ReweightedLeastSquares(; optimization_strategy = Irls{WelschShape}()); scale = 0.005)
```

## Camera regression

A camera model is regressed onto a source camera over a region by
[`CameraRegression`](@ref) (one linear solve, no guess) on a
[`CameraRegressionProblem`](@ref), and refined from a guess by
`LevenbergMarquardt` on the same problem; [`PSpline`](@ref) is the spline
targets' penalized solve on a given lattice, [`ForwardSpline`](@ref) a
`SplineLens` camera's forward spline, and [`OpticalAxis`](@ref) the radially
symmetric camera closest to a camera. A fit then a refine is the two-solve
recipe:

```julia
start = solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, camera, region)), CameraRegression()).u   # the linear fit
kb = solve(RefineProblem(CameraRegressionProblem(camera, region), start), LevenbergMarquardt())                    # its exact pixel least squares
```

## Line fitting

The line problems ([`LineFittingProblem`](@ref), [`InhomLineFittingProblem`](@ref))
have no class of record: name the scorer.

```julia
using VisualGeometryToolkitCore: MsacScoring
f = LineFittingProblem(uncertain_points)
r = solve(EstimationProblem(f), Ransac(; scoring = MsacScoring(f; threshold = 1.0)); rng = Xoshiro(1))
```

The line problem's local-optimization refit is an iterative
covariance-weighted orthogonal fit, seeded from an unweighted scatter, that
reweights each point by its variance along the normal (σᵢ² = n⊤Σᵢn) and refits;
the one-step unweighted total-least-squares line is VisualGeometryToolkitCore's
`TotalLeastSquares`.

See the [RANSAC](ransac.md) page for the full configuration surface (samplers,
adaptive trial counts, width modes, the cache).
