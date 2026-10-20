# VisualGeometryToolkitEstimation.jl

[![CI](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/dev/)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl/branch/main/graph/badge.svg?flag=VisualGeometryToolkitEstimation)](https://app.codecov.io/gh/VisualGeometryToolkit/VisualGeometryToolkit.jl?flags%5B0%5D=VisualGeometryToolkitEstimation)

Robust estimation for geometric vision: ONE generic RANSAC loop on
VisualGeometryToolkitCore's scoring and local-optimization contracts, the RANSAC
problems of the stack (lines, homographies, fundamental and essential
matrices, absolute pose (P3P), radial homographies and radial cameras,
relative-pose bundle refinement, hand-eye / robot-world calibration), and the
core local optimizations `NoLocalOptimization` and `Irls{K}`.

The loop calls only contract verbs, so any scorer composes with any local
optimizer and any problem. Out of the box every `find_*` runs MSAC
(VisualGeometryToolkitCore's `MsacScoring`) with least squares on the inliers
(`Irls{TLSShape}`, its truncation at the scorer's threshold). The paper scorers
and local optimizers live in their own packages and plug in by keyword:
[RansacScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RansacScoringDoneRight.jl)
(the Student-t + uniform marginal score) and
[LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl)
(half-quadratic lifts, GNC, the Student-t + uniform IRLS, PoseLib refiners).
VisualGeometryToolkitEstimation depends on neither.

Depends only on [VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore)
for geometry types, solvers, the problem protocol, the scoring and
local-optimization contracts and the robust-statistics toolkit. The evaluation
layer (datasets, pools, tuning, the evaluation engine) is
**VisualGeometryToolkitEval.jl**.

## Installation

```julia
using Pkg
Pkg.add("VisualGeometryToolkitEstimation")
```

Requires Julia 1.12. Documentation:
<https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/>.

## Quick Examples

### RANSAC Homography (the core default)

```julia
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitCore: estimate     # a result's model: estimate(result)
using StructArrays, StaticArrays

# Point correspondences (source → target) as a Pair vector
pairs = StructArray{Pair{SVector{2,Float64}, SVector{2,Float64}}}((src, dst))

# MSAC at the class's threshold + least squares on the inliers (Irls{TLSShape}),
# both read from configs/default.toml [pipelines.homography]
result = find_homography(pairs)                    # threshold = 3.0 px
result = find_homography(pairs; threshold = 2.0)   # a keyword overrides

H = estimate(result)                         # HomographyMat
inliers = result.inlier_mask                 # BitVector
println("Inliers: $(sum(inliers)) / $(length(inliers))")
```

The default thresholds are pixel conventions (OpenCV's), not tuned values:
homography, fundamental matrix, radial homography and radial camera 3 px,
essential matrix 1 px, absolute pose 8 px; hand-eye uses `0.01·L`, a fraction
of the problem's characteristic length (a convention of this package).

### Composition: bring a scorer and a local optimizer

A scorer from RansacScoringDoneRight and a strategy from
LocalOptimizationDoneRight are supplied as `scoring` and `local_optimization`:

```julia
using RansacScoringDoneRight: StudentTUniformScoring
using LocalOptimizationDoneRight: HQ, StudentTUniform, Gnc, WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY
using VisualGeometryToolkitCore: WelschShape

problem = HomographyProblem(pairs)
result = find_homography(pairs;
    scoring = StudentTUniformScoring(problem, 20.0),     # outlier half-width, pixels
    local_optimization = HQ{WelschShape}(scale_multiple = WELSCH_HQ_SCALE_MULTIPLE_HOMOGRAPHY))

# any pairing composes: the Student-t + uniform IRLS on the same scorer
result = find_homography(pairs;
    scoring = StudentTUniformScoring(problem, 20.0),
    local_optimization = StudentTUniform())
```

A marginal scorer's outlier half-width belongs to that scorer: the core
entries refuse an `outlier_halfwidth` keyword (`ArgumentError`), and
`threshold` together with `scoring` is refused too. A width-carrying strategy
given no width follows the scorer's decision point (`inlier_threshold`);
`scale_multiple = m` anchors it to the consensus set's scale, `loss_width = w`
fixes it.

### Low-level RANSAC

```julia
using VisualGeometryToolkitCore: MsacScoring, TLSShape

problem = HomographyProblem(pairs)
scoring = MsacScoring(problem; threshold = 2.0)
result  = ransac(problem, scoring; local_optimization = Irls{TLSShape}())

# zero-allocation form: a workspace and result buffers built once, reused
ws  = VisualGeometryToolkitEstimation.RansacWorkspace(problem, scoring, Irls{TLSShape}())
buf = VisualGeometryToolkitEstimation.RansacResultBuffers(ws)
result = ransac!(buf, ws, problem, scoring; local_optimization = Irls{TLSShape}())
```

### RANSAC Fundamental Matrix

```julia
result = find_fundmat(pairs; threshold = 1.5)
F = estimate(result)
```

### Hand-Eye / Robot-World Calibration (AX = XB)

```julia
# Synchronized pose streams (e.g. mocap + PnP), one rigid EuclideanMap{3} per frame
A = [ ... ]   # T_WR (rig in world)
B = [ ... ]   # T_CP (pattern in camera)

result = find_hand_eye(A .=> B)       # one A_i => B_i per frame, time order
X = estimate(result)                 # T_RP, the rigid pattern-in-rig transform
```

`find_hand_eye` eliminates the nuisance camera-in-world transform `Y = T_WC`
via the AX = XB form, uses Tsai–Lenz as the minimal solver, and runs
`Irls{TLSShape}` for local optimization, whose refit is Levenberg–Marquardt
on the decoupled exp(δω)·R ⊕ t chart of X (VisualGeometryToolkitCore's chart LM,
FastLevenbergMarquardt's `lmsolve!`, ForwardDiff Jacobian).

### Robust Line Fitting

```julia
using VisualGeometryToolkitCore: Point2, estimate, MsacScoring

points = [Point2(1.0, 2.1), Point2(2.0, 4.0), ...]  # with outliers

# Build the RANSAC problem (points, or Uncertain{Point2} with per-point covariances)
problem = LineFittingProblem(points)
result  = ransac(problem, MsacScoring(problem; threshold = 0.5))

line    = estimate(result)       # Line2D (Hesse normal form)
inliers = result.inlier_mask     # BitVector
```

### Strategies from TOML

A strategy is selected by tag. The core registry `VisualGeometryToolkitEstimation.LO_TAGS`
holds `"no_local_optimization"` and `"irls"`; a downstream registry is merged
explicitly:

```julia
using VisualGeometryToolkitEstimation: local_optimization_from_toml, LO_TAGS
import LocalOptimizationDoneRight

lo = local_optimization_from_toml(Dict("type" => "hq", "loss" => "welsch", "scale_multiple" => 1.6);
    class = :homography, tags = merge(LO_TAGS, LocalOptimizationDoneRight.LO_TAGS))
```

## Architecture

```
AbstractRansacProblem{K} <: VGC AbstractConstraintProblem   RANSAC problem interface
├── LineFittingProblem                        + InhomLineFittingProblem
├── HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem,
│   RadialHomographyProblem                   aliases of MinimalSolveProblem{M, K, …}
├── RadialCameraProblem
├── RelposeBundleProblem                      relative-pose bundle refinement
└── HandEyeProblem                            AX = XB (Tsai–Lenz + so(3)⊕R³ chart-LM refit)
    (BoundedProblem(inner, extent) wraps any of the above with a validity region)

AbstractScoring (VGC)                        the scoring contract: scoring_workspace, score!,
│                                            inlier_count, consensus_order (+ optional verbs)
├── MsacScoring, RansacScoring (VGC)         the baseline scorers; MSAC is the find_* default
└── StudentTUniformScoring (RansacScoringDoneRight)   one example of a downstream scorer

AbstractLocalOptimization (VGC)              the LO contract: lo_workspace, locally_optimize!
├── NoLocalOptimization                      no refit; the default of ransac(problem, scoring)
├── Irls{K, W}                               static IRLS on VGC loss shape K; the find_* default
│                                            is Irls{TLSShape} (least squares on the inliers)
└── HQ, Gnc, StudentTUniform, … (LocalOptimizationDoneRight)   downstream strategies

RansacWorkspace{K, M, T, W, L}               the composed workspace, built once
  ├── buf::ScoreBuffers{T, W}                residuals, scores, consensus mask, the scorer's workspace W
  ├── lo_ws::L                               the local optimizer's workspace (lo_workspace(P, T, lo, n))
  └── best_* / sample_indices                the incumbent and the minimal sample
```

### Key design decisions

- **Contracts, not concrete types.** The loop (`estimators/ransac/loop.jl`)
  calls only VisualGeometryToolkitCore's contract verbs — `scored_model`,
  `score_model!`, `inlier_count`, `consensus_order`, `consensus_set`,
  `consensus_s2`, `inlier_probabilities!`, `is_predictive` and
  `locally_optimize!` — so no scorer or strategy type is named in it.
- **Static dispatch.** `RansacWorkspace{K, M, T, W, L}` carries the scorer's
  workspace type `W` and the local optimizer's `L`, so `_ransac_loop`
  compiles once per (problem, scorer, local optimizer) with no runtime
  dispatch, and `ransac!` runs allocation-free on a reused workspace.
- **Width modes.** A width-carrying strategy (`Irls{K, W}`) has no width
  (`nothing`, follow the scorer's `inlier_threshold`), `ScaleMultiple(m)`
  (anchored at the consensus set's scale) or `LossWidth(w)` (absolute),
  resolved once by VGC's `loss_at`. Following a scorer that defines no
  threshold is refused at entry (`check_composition`).
- **Block-coordinate-descent extension interface**: `init_state!`, `update_model!`,
  `update_residuals!`, `update_nuisance!`, `update_weights!`, `post_iteration!`,
  `stopping_criterion`, `stop_reason`, `block_coordinate_result` are VGC's and
  `public` — downstream packages plug into the generic
  `block_coordinate_descent!` loop. RVG keeps the posterior-weight iteration
  (`PosteriorIrlsMethod`, `PosteriorLikelihood`) that `fit_residual_mixture`
  and `fit_division_camera` run, and downstream strategies build on.
- **LO-RANSAC via `refine!`**: Problems implement VGC's
  `refine!(ws, problem, θ, weights)` on `refine_workspace(P, n)`, the local
  optimization's weighted refit from the current model, routed by the
  `refit_kind` trait. For `HomographyProblem`, `FundMatProblem`,
  `EssentialProblem`, `AbsolutePoseProblem` and `HandEyeProblem` it is
  `NonlinLsq`: Levenberg–Marquardt on a minimal chart about θ — sl(3) (8 dof)
  for H, Bartoli–Sturm (7 dof) for F, so(3)⊕R³ (6 dof) for the two poses. The
  chart problems also provide VGC's chart-rows contract (`rows_provider`),
  which the half-quadratic lifts of LocalOptimizationDoneRight consume.
- **Type dispatch**: Solver cardinality via `FixedModels{N,M}`; model
  uncertainty via `Uncertain{M}` (predictive scorers).
- **Uncertainty quantification**: Full covariance propagation through
  Hartley normalization.

## Documentation

Online at <https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/>. To
build locally (into `docs/build/`):

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes
`Project.toml`, they re-resolve the environment (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## Contributors

VisualGeometryToolkitEstimation is written by James Pritts, with contributions from
Philipp Böll ([@phib2b](https://github.com/phib2b)): opt-in joint scale
estimation for up-to-scale pose pairs in robot-world calibration.

## Citing

VisualGeometryToolkitEstimation is part of VisualGeometryToolkit. Cite the toolkit; its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff) is the source of this entry (GitHub's "Cite this
repository" button exports it):

```bibtex
@software{VisualGeometryToolkit,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  year   = {2026}
}
```

Cite also the papers behind the methods you use, listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).

## License

MIT
