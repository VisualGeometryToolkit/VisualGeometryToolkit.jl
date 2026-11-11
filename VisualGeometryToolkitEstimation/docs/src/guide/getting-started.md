# Getting Started

## Installation

VisualGeometryToolkitEstimation is registered in the General registry (Julia
1.12+). The examples use the geometry types of VisualGeometryToolkitCore, so
add both:

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitEstimation
```

## RANSAC Homography

Estimate a homography from point correspondences with outliers using the
convenience pipeline `find_homography`. The example builds 40 correspondences
from a known homography and replaces 8 of them with random points; the docs
build runs it (a doctest):

```jldoctest homography
julia> using VisualGeometryToolkitEstimation, Random

julia> using VisualGeometryToolkitCore: HomographyMat, HomographyMap, Point2, csponds, estimate, to_euclidean, MsacScoring

julia> H_true = HomographyMat([1.02 -0.02 12.0; 0.01 0.98 -5.0; 1e-5 2e-5 1.0]);

julia> rng = Xoshiro(1);

julia> src = [Point2(640rand(rng), 480rand(rng)) for _ in 1:40];   # source points, pixels

julia> dst = [to_euclidean(HomographyMap(H_true)(x)) for x in src];   # their images under H_true

julia> for i in 1:8; dst[i] = Point2(640rand(rng), 480rand(rng)); end   # 8 outliers

julia> result = find_homography(csponds(src, dst));    # MSAC at 3 px, the default

julia> count(result.inlier_mask)                        # the 32 true correspondences
32

julia> H = Matrix(estimate(result)); H ./ H[3, 3] ≈ Matrix(H_true)
true

julia> result = find_homography(csponds(src, dst);   # a scorer overrides the class's
              scoring = MsacScoring(HomographyProblem; threshold = 2.0));
```

`find_homography` constructs a [`HomographyProblem`](@ref), wires up the core
defaults — VisualGeometryToolkitCore's `MsacScoring(problem; threshold)` and
`Irls{TLSShape}()` local optimization (least squares on the inliers, its
truncation following the scorer's threshold), both from the
`[pipelines.homography]` table of `configs/default.toml` — and runs the RANSAC
loop. The same pattern works for every `find_*`; the default thresholds are
pixel conventions (3 px for H and F, 1 px for E, 8 px for PnP), not tuned
values.

## Composing a scorer and a local optimizer

The loop runs on VisualGeometryToolkitCore's scoring and local-optimization
contracts, so a scorer and a strategy from other packages plug in by keyword.
The Student-t + uniform scorer is VisualGeometryToolkitCore's; the paper strategies are
LocalOptimizationDoneRight's:

```julia
using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior
using LocalOptimizationDoneRight: HQ, StudentTUniform, WELSCH_HQ_EFFICIENCY_CONSTANT_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

cs = csponds(src, dst)
problem = HomographyProblem(cs)

result = find_homography(cs;
    scoring = StudentTUniformScoring(problem, 20.0;    # outlier half-width, pixels
                                     scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
    local_optimization_strategy = HQ{WelschShape}(efficiency_constant = WELSCH_HQ_EFFICIENCY_CONSTANT_HOMOGRAPHY))
```

A bare `threshold` or `outlier_halfwidth` keyword is refused by every RANSAC
entry (`ArgumentError`): a width is its scorer's — a threshold scorer's
`threshold`, the Student-t + uniform scorer's `outlier_halfwidth` — so it goes
into the scorer.

## RANSAC manually

To control every piece, build the problem and the scorer and call `ransac`:

```julia
using VisualGeometryToolkitCore: MsacScoring, TLSShape, WelschShape, IGPrior

problem = HomographyProblem(csponds(src, dst))

# the core baseline
result = ransac(problem, MsacScoring(problem; threshold = 2.0);
                local_optimization_strategy = Irls{TLSShape}())

# a Welsch IRLS anchored at the consensus set's scale
result = ransac(problem, MsacScoring(problem; threshold = 2.0);
                local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5))

# the marginal scorer with a weakly-informative scale prior, or its predictive variant
# scoring = StudentTUniformScoring(problem, 20.0; scale_prior = IGPrior(2.0, 1.0), label_prior = FlatLabelPrior())
# scoring = StudentTUniformScoring{Predictive}(problem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
```

`ransac(problem, scoring)` with no `local_optimization_strategy` runs none
(`nothing`). The two-level RANSAC loop gates entry into local
optimization on the scorer's score, then re-scores the locally refined
hypothesis. See the [RANSAC](ransac.md) page for the full configuration
surface.

## Next Steps

- [Architecture](architecture.md) — type hierarchy and dispatch model
- [Scoring Functions](scoring.md) — the scoring contract, MSAC, composing a scorer
- [RANSAC](ransac.md) — scoring, local optimization, samplers, configuration
- [Fitting Algorithms](fitting.md) — `find_homography`, `find_fundmat`, `find_absolute_pose`
- [Extending](extending.md) — implement your own `AbstractRansacProblem`
