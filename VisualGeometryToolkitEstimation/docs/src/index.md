# VisualGeometryToolkitEstimation.jl

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

VisualGeometryToolkitEstimation provides robust estimation for geometric
vision: one generic RANSAC loop and the estimation problems of the
VisualGeometryToolkit packages, from lines, homographies, fundamental and
essential matrices and absolute pose to radial cameras and hand-eye
calibration. Every `find_*` pipeline runs MSAC scoring and refits the model on
the inliers by default (least squares for most problems), and other scorers
and local optimizers plug in by keyword.

## What it provides

- **One generic RANSAC loop** on VisualGeometryToolkitCore's scoring and
  local-optimization contracts: any scorer composes with any local optimizer
  and any problem, with static dispatch throughout, and `ransac!` runs on a
  reused [`RansacWorkspace`](@ref). Two-level gating, adaptive trial counts,
  PROSAC ordered sampling. See [RANSAC](guide/ransac.md).
- **Core defaults**: every `find_*` runs VisualGeometryToolkitCore's MSAC
  (`[pipelines.<class>] scoring`, MSAC at the class's threshold) and a local optimization read from
  `configs/default.toml`: least squares on the inliers
  ([`Irls`](@ref)`{TLSShape}`, following the scorer's decision point) for most
  problems, the robustified chart LM [`Triggs`](@ref) for the circle and the
  ellipse. The per-class thresholds are conventions (homography, fundamental
  matrix and the radial classes 3 px; essential matrix, rotation, circle and
  ellipse 1 px; absolute pose 8 px; hand-eye `0.01·L`). See
  [Fitting Algorithms](guide/fitting.md).
- **Composition**: bring a scorer and a strategy from other packages by
  keyword — `scoring = StudentTUniformScoring(problem, a)` from
  RANSACScoringDoneRight, `local_optimization = HQ{WelschShape}(…)`,
  `StudentTUniform()`, `Gnc{…}(…)` from LocalOptimizationDoneRight. See
  [Getting Started](guide/getting-started.md).
- **Core local optimizations**: [`NoLocalOptimization`](@ref) and the static
  IRLS [`Irls`](@ref)`{K}` on any VisualGeometryToolkitCore loss shape, with its width
  following the scorer, anchored at the consensus scale (`ScaleMultiple(m)`)
  or absolute (`LossWidth(w)`). The refit is Levenberg–Marquardt on a minimal
  chart about the current model: sl(3) for `HomographyProblem`, Bartoli–Sturm
  for `FundMatProblem`, so(3)⊕R³ for `AbsolutePoseProblem` and `HandEyeProblem`.
- **Posterior-weight iteration** ([`PosteriorIrlsMethod`](@ref),
  [`PosteriorLikelihood`](@ref)) on VisualGeometryToolkitCore's generic
  block-coordinate-descent loop: the machinery of
  [`fit_residual_mixture`](@ref) and `fit_division_camera`, and the base the
  downstream Student-t + uniform strategy builds on.
- **Problem implementations**: 2D line fitting (Hesse, inhomogeneous), homography,
  fundamental and essential matrices, radial homography and radial camera,
  3D-2D pose estimation (P3P + LO), relative-pose bundle refinement, and
  hand-eye / robot-world calibration (AX = XB, Tsai–Lenz minimal solver +
  so(3)⊕R³ chart-LM refit). The chart problems provide VisualGeometryToolkitCore's
  chart-rows contract for lifted local optimizers.
- **Solver backends**: pure-Julia `NativeSolver` or PoseLib C++ bindings via `PoseLibSolver`.

The design — one loop on contracts, problems that own their normalization
(frame in = frame out), configuration read from one TOML file — is explained
in [Architecture](guide/architecture.md).

## Installation

```julia-repl
pkg> add VisualGeometryToolkitCore VisualGeometryToolkitEstimation
```

Julia 1.12 or later; every dependency is in the General registry.

## A first example

The example builds 40 correspondences from a known homography, replaces 8 of
them with random points, and recovers the homography and the 32 inliers.

```julia
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation, Random

H_true = HomographyMat([1.02 -0.02 12.0; 0.01 0.98 -5.0; 1e-5 2e-5 1.0])

rng = Xoshiro(1)
src = [Point2(640rand(rng), 480rand(rng)) for _ in 1:40]    # source points, pixels
dst = [to_euclidean(HomographyMap(H_true)(x)) for x in src] # their images under H_true
for i in 1:8                                                # replace 8 with outliers
    dst[i] = Point2(640rand(rng), 480rand(rng))
end

result = find_homography(csponds(src, dst))   # MSAC at 3 px, the default
count(result.inlier_mask)                     # 32, the true correspondences

H = Matrix(estimate(result))
H ./ H[3, 3] ≈ Matrix(H_true)                 # true

result_2px = find_homography(csponds(src, dst);   # a scorer overrides the class's
                           scoring = VisualGeometryToolkitCore.MsacScoring(HomographyProblem; threshold = 2.0))
```

## Documentation

**Start here.** [Getting Started](guide/getting-started.md): a first
pipeline, composing a scorer and a local optimizer, and calling `ransac`
directly.

**How to.**
[Fitting Algorithms](guide/fitting.md) (every `find_*` and `fit_*` with its
default threshold),
[RANSAC](guide/ransac.md) (local optimization, samplers, configuration, the
result),
[Scoring Functions](guide/scoring.md),
[Rig-to-Pattern Hand–Eye Calibration](guide/hand-eye.md),
[Robot–World Hand–Eye Calibration](guide/robot-world-hand-eye.md) and
[Extending](guide/extending.md) (a new RANSAC problem or block-coordinate
method).

**Explanation.**
[Architecture](guide/architecture.md),
[Covariance and Gauge Invariance](guide/covariance.md),
[Ray Density Correction](guide/ray-density.md),
[Homography Decomposition](guide/homography-decomposition.md) and
[Homography Cheirality](guide/cheirality.md).

**Reference.** The API pages, from [RANSAC](api/ransac.md) and the
[Scoring & LO Contracts](api/scoring.md) to each problem family, and the
[References](references.md).

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitEstimation is the robust estimation layer of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack: the
RANSAC loop, problem implementations, and the core local optimizations. It
depends only on
[VisualGeometryToolkitCore.jl](@extref VisualGeometryToolkitCore :doc:`index`):
geometry types, minimal solvers, the problem protocol, the scoring and
local-optimization contracts, the robust-loss shapes, the robust-statistics
toolkit and the generic block-coordinate-descent loop. Paper scorers and local
optimizers live outside it and implement those contracts:
[RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl)
(the Student-t + uniform marginal score) and
[LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl)
(half-quadratic lifts, GNC, the Student-t + uniform IRLS). The
evaluation, tuning and plotting layer is VisualGeometryToolkitEval.jl.

All toolkit documentation is aggregated at the portal:
<https://visualgeometrytoolkit.github.io>.

## Citing

VisualGeometryToolkitEstimation is written by James Pritts, with contributions
from Philipp Böll ([@phib2b](https://github.com/phib2b)): opt-in joint scale
estimation for up-to-scale pose pairs in robot-world calibration. Cite the
toolkit. Its
[CITATION.cff](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/blob/main/CITATION.cff)
is the source of this entry, and GitHub's "Cite this repository" button exports it:

```bibtex
@software{pritts2026vgtk,
  author = {Pritts, James and Sauer, Hendrik},
  title  = {{VisualGeometryToolkit.jl}: Julia Packages for Multiple-View Geometry},
  url    = {https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl},
  doi    = {10.5281/zenodo.23548372},
  year   = {2026}
}
```

Cite also the papers behind the methods you use, listed on
[the portal](https://visualgeometrytoolkit.github.io/#publications).
