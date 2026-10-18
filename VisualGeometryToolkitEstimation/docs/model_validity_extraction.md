# Model-validity interface extraction (VGC) + RansacScoring split

> **Historical.** Step 1 landed; the package split of "Step 2+" was superseded
> by the composable RANSAC: VisualGeometryToolkitCore owns the problem protocol
> (`AbstractConstraintProblem`), the scoring and local-optimization contracts
> and the robust-statistics toolkit, and VisualGeometryToolkitEstimation depends on
> VisualGeometryToolkitCore alone. RansacScoringDoneRight implements the scoring
> contract downstream.

Goal: extract the RANSAC **scoring** layer into a standalone `RansacScoring.jl`
that depends on `VisualGeometryToolkitCore` (VGC), with `VisualGeometryToolkitEstimation` (RVG)
depending on `RansacScoring`. A prerequisite, tracked here, is relocating the
**model-validity predicates** (`test_sample` / `test_model` / `test_consensus`)
from RVG problem methods into VGC as a model-class interface — because they are
properties of the geometry, not of the RANSAC harness or the scoring layer.

## Target package graph (acyclic)

```
VisualGeometryToolkitCore   model types + minimal solvers + constraint_with_jacobians
                     + model-validity interface (test_sample/test_model/test_consensus per class)
      ▲
RansacScoring.jl     scoring math + the residual-level problem supertype (never sees a sample)
      ▲
VisualGeometryToolkitEstimation RANSAC loop + problems (validity = one-line delegations to VGC) + fitting
```

`eval/` becomes a script environment in the RansacScoring repo:
- evaluation (small/large val): deps RansacScoring + VGC
- dataset generation: deps VGC + VisualGeometryToolkitFeatures (RoMA) only — **no RVG**

## Step 1 — VGC model-validity interface (this change)

Contract (all methods): caller supplies correspondences + model in the **same,
Hartley-normalized frame**; VGC does not normalize internally (tolerances are
calibrated for normalized coords). `test_sample` dispatches on the model
**Type** (pre-solve, no instance yet); `test_model`/`test_consensus` dispatch
on the model **instance**.

New file `src/geometry/model_validity.jl` (included after `fundmat.jl`):
- moved helpers (from RVG): `_orient` (2D/3D), `thin_ratio`,
  `_orientation_preserving`, `_not_collinear`, `_are_points_distinct`,
  `_points_distinct_idx`
- `test_sample(::Type{<:HomographyMat}, cs, idx)` — bow-tie ∧ distinct ∧ ¬collinear
- `test_sample(::Type{<:Union{FundMat,EssentialMat}}, cs, idx)` — distinct
- `test_model(H::HomographyMat, cs, idx)` = `test_conditioning(H) && test_cheirality(H, cs, idx)`
  (F's `test_model` already exists in `fundmat.jl` — oriented epipolar)
- `test_consensus(H::HomographyMat, cs, mask)` = `test_cheirality(H, cs, mask)`
- `test_consensus(::Union{FundMat,EssentialMat}, cs, mask) = true`

Exports added: `test_sample`, `test_consensus` (`test_model` already exported).

## Step 1 — RVG delegations

- `problems/homography.jl`: delete `_orient`, `thin_ratio`,
  `_orientation_preserving`, `_not_collinear`; `test_sample(p::HomographyProblem,idx)`
  → `test_sample(HomographyMat, p.cs_hartley, idx)`; `_test_model_norm(...H_n...)`
  → `test_model(H_n, p.cs_hartley, idx)`; `test_consensus(...H...)` → `test_consensus(H, p.cs, mask)`.
- `problems/cspond.jl`: delete `_are_points_distinct`;
  `test_sample(p::RelativeProblem{<:AbstractFundMat},idx)` →
  `test_sample(<model class>, p.cs_hartley, idx)`.
- `VisualGeometryToolkitEstimation.jl`: add `test_sample`, `test_consensus` to the VGC
  import list (RVG extends them for the *problem* type).

## Step 2+ — package extraction (not in step 1)

RansacScoring owns: `IGPrior`, `ScoringWorkspace`, sweep kernels, Student-t EM,
`StudentTUniformScoring{P,DG}` + `score!`/`mask!`/`_predictive_score_penalty`,
`_wrap_*_cov`, `_mahalanobis`/`_get_Σ_x`, and the residual-level problem supertype +
interface stubs (`data_size`, `codimension`, `minimal_sample_size`,
`solver_covariance`, `fit_param_covariance`). `constraint_with_jacobians` stays in
VGC. RVG `AbstractRansacProblem{K}` subtypes that supertype,
keeps `RansacWorkspace`-typed forwarders, re-exports scoring names.

Release order (Cau): VGC minor bump → register RansacScoring → RVG compat bump
+ delegations + dep on RansacScoring → move eval/ with two Project.toml envs.
