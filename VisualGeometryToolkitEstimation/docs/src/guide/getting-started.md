# Getting Started

## Installation

VisualGeometryToolkitEstimation is registered in the General registry (Julia 1.12+):

```julia
using Pkg
Pkg.add("VisualGeometryToolkitEstimation")
```

## RANSAC Homography

Estimate a homography from noisy point correspondences with outliers using
the convenience pipeline `find_homography`:

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: csponds, estimate
using StaticArrays

# Point correspondences (source → target)
src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

result = find_homography(csponds(src, dst))                    # MSAC at 3 px
result = find_homography(csponds(src, dst); threshold = 2.0)   # a keyword overrides

H       = estimate(result)         # 3×3 homography matrix
inliers = result.inlier_mask       # BitVector
println("Inliers: $(sum(inliers)) / $(length(inliers))")
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
The paper scorer is RansacScoringDoneRight's; the paper strategies are
LocalOptimizationDoneRight's:

```julia
using RansacScoringDoneRight: StudentTUniformScoring
using LocalOptimizationDoneRight: HQ, StudentTUniform, WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

cs = csponds(src, dst)
problem = HomographyProblem(cs)

result = find_homography(cs;
    scoring = StudentTUniformScoring(problem, 20.0),   # outlier half-width, pixels
    local_optimization = HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY))
```

An `outlier_halfwidth` keyword is refused by the core entries: the half-width
is the marginal scorer's parameter, so it goes into the scorer. Giving both
`threshold` and `scoring` is an `ArgumentError` too.

## RANSAC manually

To control every piece, build the problem and the scorer and call `ransac`:

```julia
using VisualGeometryToolkitCore: MsacScoring, TLSShape, WelschShape, IGPrior

problem = HomographyProblem(csponds(src, dst))

# the core baseline
result = ransac(problem, MsacScoring(problem; threshold = 2.0);
                local_optimization = Irls{TLSShape}())

# a Welsch IRLS anchored at the consensus set's scale
result = ransac(problem, MsacScoring(problem; threshold = 2.0);
                local_optimization = Irls{WelschShape}(scale_multiple = 2.5))

# the marginal scorer with a weakly-informative scale prior, or its predictive variant
# scoring = StudentTUniformScoring(problem, 20.0; prior = IGPrior(2.0, 1.0))
# scoring = StudentTUniformScoring{Predictive}(problem, 20.0)
```

`ransac(problem, scoring)` with no `local_optimization` runs none
(`NoLocalOptimization()`). The two-level RANSAC loop gates entry into local
optimization on the scorer's score, then re-scores the locally refined
hypothesis. See the [RANSAC](ransac.md) page for the full configuration
surface.

## Next Steps

- [Architecture](architecture.md) — type hierarchy and dispatch model
- [Scoring Functions](scoring.md) — the scoring contract, MSAC, composing a scorer
- [RANSAC](ransac.md) — scoring, local optimization, samplers, configuration
- [Fitting Algorithms](fitting.md) — `find_homography`, `find_fundmat`, `find_absolute_pose`
- [Extending](extending.md) — implement your own `AbstractRansacProblem`
