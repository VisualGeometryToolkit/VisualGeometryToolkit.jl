docs-pass: 2dc3ac60

# Wave 10 docs pass: VisualGeometryToolkitEstimation 0.11.0, VisualGeometryToolkitEval 0.8.0, PoseLib 0.3.0 (julia-docs-pass) — branch w10-est-eval

Theme est-eval of the breaking wave 10 (the IO wave, the terminology ruling, NO DUPLICATIONS). Baseline: the wave-113
docs pass (7c550857 / 2f994d2b). Julia 1.13.3; gate 2630a051 (2dc3ac60 + the temporary Features compat line on the gate
branch only). Records-only commits after 2dc3ac60 change CHANGELOG / comments, no code.

## 1. Public API since the baseline (each with its CHANGELOG entry)

- [x] Estimation, removed: `LiftableProblem` (→ `MinimalSolveProblem`), `AbstractCspondProblem` (→ `AbstractCorrespondenceProblem`),
  `image_size` (→ Core's `extent`), `check_penalty_weight(…; allow_nothing)`, `refine_camera_and_poses(…; refit_forward,
  forward_tolerance)` — each refused by name (RETIRED_NAMES or `refuse_retired_keyword`); `[chart_lm]` and
  `[relpose_bundle] stationarity_tol` config keys retired.
- [x] Estimation, added: `device` on `fit_camera(L, …)` and `optical_axis`; `support` on every camera fit (Core's
  `check_region_support`); `optical_axis(camera, region)`; `[optical_axis] radial_terms`; `stop_rule` methods.
- [x] Eval, removed: `Stats.POOL_RNG`, `consolidate_csponds` (→ `consolidate_correspondences`), `min_csponds` (→
  `min_correspondences`), `_toml_hash` (→ Core's `canonical_json`); pool format 6, consolidated format 3, tuning format 8.
- [x] Eval, added: `DataArtifacts.published_name`, `Render.write_tsv(path, rows::AbstractVector{<:NamedTuple})`,
  `Tuning.WindowExtension` + `extend(::EfficiencyConstantWindow, …)`, `Reproduce.stage_each`, `PlannedPhase`; runner graph keys
  `array.from`/`array.each`, `after = ["stage:item"]`, `phases`, `nice`, `julia_flags`, `precompile`; site key `precompile_time`.
- [x] PoseLib 0.3.0: on Core 0.10 (`Correspondence`), `InitialFailure` below the minimal sample, retired keywords through Core.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | every new public name has a docstring on its package's API page (`@autodocs` / `@docs`); docs builds strict (no warnonly) green ×3 (gate 2630a051) |
| stale mentions | grep of README/docs/src for every removed name: none outside CHANGELOGs; no private path in any CHANGELOG, config or test comment |
| CHANGELOG | Estimation 0.11.0, Eval 0.8.0, PoseLib 0.3.0 (Breaking / Added / Fixed / Internal) |
| versions | Estimation 0.11.0, Eval 0.8.0 (on Estimation 0.11.0, IO 0.10.0, Core 0.10.0), PoseLib 0.3.0 |
| README examples | the readme items run from the install line (green in the gate) |

## 3. Gate (2630a051): 20/20 green — see ~/logs/dgx-agent/wave10/est-eval.md (the table) — and check_duplication,
check_naming, check_splat 0 on all three.
