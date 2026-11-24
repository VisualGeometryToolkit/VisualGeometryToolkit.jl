# Getting Started

## Installation

VisualGeometryToolkitEstimation is registered in the General registry (Julia
1.13+). The examples use the geometry types of VisualGeometryToolkitCore, so
add both:

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitEstimation
```

## Problem and algorithm

Every estimator is `solve(problem, algorithm)` (CommonSolve's verb, the one
estimation verb of the stack; VisualGeometryToolkitCore owns the front end).
The PROBLEM says what is estimated from which data; the ALGORITHM says how, its
configuration in its fields; the solve's keywords are run settings only (an
`rng`, a `scale` or `width`, the `stops` of an LM).

- A **find** searches contaminated data for a model:
  `solve(EstimationProblem(f, p), Ransac(…); rng)`. `f` is a family problem —
  [`HomographyProblem`](@ref)`(cs)`, [`EssentialProblem`](@ref)`(camera1,
  camera2, cs)`, [`AbsolutePoseProblem`](@ref)`(camera, cs)`, … — holding the
  data; `p = (; extent)` optionally validates the model over a region.
- A **refine** polishes a guess:
  `solve(RefineProblem(f, θ₀), LevenbergMarquardt(…); scale | width)` on the
  same family problem, VisualGeometryToolkitCore's one host
  Levenberg–Marquardt.
- A **fit** seeds itself (a reweighted least squares, a rotation averaging, a
  camera regression): `solve(EstimationProblem(…), Algorithm(…))` with the fit's
  own algorithm type ([Fitting Algorithms](fitting.md)).

## RANSAC Homography

Estimate a homography from point correspondences with outliers. The example
builds 40 correspondences from a known homography and replaces 8 of them with
random points; the docs build runs it (a doctest):

```jldoctest homography
julia> using VisualGeometryToolkitEstimation, Random

julia> using VisualGeometryToolkitCore: HomographyMat, HomographyMap, Point2, correspondences, to_euclidean, MsacScoring,
           inlier_mask, EstimationProblem, RefineProblem, LevenbergMarquardt, solve

julia> H_true = HomographyMat([1.02 -0.02 12.0; 0.01 0.98 -5.0; 1e-5 2e-5 1.0]);

julia> rng = Xoshiro(1);

julia> src = [Point2(640rand(rng), 480rand(rng)) for _ in 1:40];   # source points, pixels

julia> dst = [to_euclidean(HomographyMap(H_true)(x)) for x in src];   # their images under H_true

julia> for i in 1:8; dst[i] = Point2(640rand(rng), 480rand(rng)); end   # 8 outliers

julia> problem = EstimationProblem(HomographyProblem(correspondences(src, dst)));

julia> result = solve(problem, Ransac(); rng = Xoshiro(2));    # MSAC at 3 px, the class default

julia> count(inlier_mask(result))                       # the 32 true correspondences
32

julia> H = Matrix(result.u); H ./ H[3, 3] ≈ Matrix(H_true)
true

julia> result = solve(problem,                          # a scorer overrides the class's
              Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 2.0)); rng = Xoshiro(2));
```

`Ransac()` reads its defaults from the problem's class, the
`[pipelines.homography]` table of `configs/default.toml`: VisualGeometryToolkitCore's
`MsacScoring(problem; threshold)` and an `Irls{TLSShape}()` local optimization
(least squares on the inliers, its truncation following the scorer's
threshold). Every family problem works the same way; the default thresholds are
pixel conventions (3 px for H and F, 1 px for E, 8 px for PnP), not tuned
values. Read a result by field — `result.u` (the model, `nothing` when the
search found none: test `found_model(result)`), `result.retcode`,
`result.stats.nsteps` (the trials drawn), `result.resid` — and the cross-type
quantities by their functions (`inlier_mask`, `weights`, `noise_scale`).

## Refining the result

The refine takes the find's model as its guess and the find's result as its
noise scale (`noise_scale(result)`, the scale the scorer estimated on the
consensus set):

```julia
refined = solve(RefineProblem(HomographyProblem(correspondences(src, dst)), result.u),
                LevenbergMarquardt(); scale = result)
```

Its default strategy is the class's (`[refine.homography]`: one robustified LM,
`Triggs{WelschShape}` at the efficiency constant times the scale);
`LevenbergMarquardt(; optimization_strategy = nothing)` is plain least squares
and takes no `scale`. A refine estimates no scale and forms no consensus set.

## Composing a scorer and a local optimizer

The loop runs on VisualGeometryToolkitCore's scoring and local-optimization
contracts, so a scorer and a strategy from other packages plug in as `Ransac`'s
fields. The Student-t + uniform scorer is VisualGeometryToolkitCore's; the paper
strategies are LocalOptimizationDoneRight's:

```julia
using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior
using LocalOptimizationDoneRight: HQ, StudentTUniform, WELSCH_HQ_EFFICIENCY_CONSTANT_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

f = HomographyProblem(correspondences(src, dst))

result = solve(EstimationProblem(f),
    Ransac(; scoring = StudentTUniformScoring(f, 20.0;    # outlier half-width, pixels
                                              scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
             local_optimization_strategy = HQ{WelschShape}(efficiency_constant = WELSCH_HQ_EFFICIENCY_CONSTANT_HOMOGRAPHY));
    rng = Xoshiro(1))
```

A bare `threshold` or `outlier_halfwidth` keyword is refused by `Ransac`
(`ArgumentError`): a width is its scorer's — a threshold scorer's `threshold`,
the Student-t + uniform scorer's outlier half-width — so it goes into the
scorer.

## Reusing a workspace

`init` builds the composed workspace once; `solve!` runs it and `reinit!` points
it at new data of the same problem type, keeping its buffers (the result's
`resid` and weights ARE the cache's buffers, overwritten by the next `solve!`):

```julia
using VisualGeometryToolkitCore: init, solve!, reinit!

cache = init(EstimationProblem(HomographyProblem(cs₁)), Ransac(); rng = Xoshiro(1))
r₁ = solve!(cache)
reinit!(cache; f = HomographyProblem(cs₂))
r₂ = solve!(cache)
```

A local optimization from a caller's seed — a scorer, its consensus set and the
strategy anchored on it, no sampling — is [`LocalOptimization`](@ref):

```julia
r = solve(RefineProblem(f, H₀), LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(f; threshold = 2.0)))
```

See the [RANSAC](ransac.md) page for the full configuration surface.

## Next Steps

- [Architecture](architecture.md) — type hierarchy and dispatch model
- [Scoring Functions](scoring.md) — the scoring contract, MSAC, composing a scorer
- [RANSAC](ransac.md) — scoring, local optimization, samplers, configuration
- [Fitting Algorithms](fitting.md) — every family problem and fit algorithm
- [Extending](extending.md) — implement your own `AbstractRansacProblem`
