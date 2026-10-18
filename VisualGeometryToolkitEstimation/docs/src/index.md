# VisualGeometryToolkitEstimation.jl

Robust estimation algorithms for geometric vision.

## Features

- **One generic RANSAC loop** on VisualGeometryToolkitCore's scoring and
  local-optimization contracts: any scorer composes with any local optimizer
  and any problem, with static dispatch throughout and an allocation-free
  `ransac!` on a reused [`RansacWorkspace`](@ref). Two-level gating, adaptive
  trial counts, PROSAC ordered sampling.
- **Core defaults**: every `find_*` runs VisualGeometryToolkitCore's MSAC
  (`MsacScoring(problem; threshold)`) with least squares on the inliers
  ([`Irls`](@ref)`{TLSShape}`, following the scorer's threshold). The
  per-class thresholds are pixel conventions (homography, fundamental matrix
  and the radial classes 3 px, essential matrix 1 px, absolute pose 8 px;
  hand-eye `0.01·L`).
- **Composition**: bring a scorer and a strategy from other packages by
  keyword — `scoring = StudentTUniformScoring(problem, a)` from
  RansacScoringDoneRight, `local_optimization = HQ{WelschShape}(…)`,
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

Depends only on [VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore):
geometry types, minimal solvers, the problem protocol, the scoring and
local-optimization contracts, the robust-loss shapes (`AbstractLossShape`),
the robust-statistics toolkit and the generic block-coordinate-descent loop
(`block_coordinate_descent!`).

## Contents

```@contents
Pages = [
    "guide/getting-started.md",
    "guide/architecture.md",
    "guide/scoring.md",
    "guide/ransac.md",
    "guide/fitting.md",
    "guide/extending.md",
]
Depth = 2
```

## Role in the VisualGeometryToolkit stack

VisualGeometryToolkitEstimation is the **robust estimation** layer of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack: the
RANSAC loop, problem implementations, and the core local optimizations. It
builds on
[VisualGeometryToolkitCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore)
(geometric types and the contracts). Paper scorers and local optimizers live
outside it and implement those contracts:
[RansacScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RansacScoringDoneRight.jl)
(the Student-t + uniform marginal score) and
[LocalOptimizationDoneRight.jl](https://github.com/VisualGeometryToolkit/LocalOptimizationDoneRight.jl)
(half-quadratic lifts, GNC, the Student-t + uniform IRLS). The
evaluation/tuning/plotting layer lives in VisualGeometryToolkitEval.jl.

All toolkit documentation is aggregated at the portal:
<https://visualgeometrytoolkit.github.io>.
