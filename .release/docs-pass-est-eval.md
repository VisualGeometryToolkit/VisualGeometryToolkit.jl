docs-pass: 2d48ca86

# Wave 10 docs pass: VisualGeometryToolkitEstimation 0.11.0, VisualGeometryToolkitEval 0.8.0, PoseLib 0.3.0 (julia-docs-pass) — branch w10-est-eval

Theme est-eval of the breaking wave 10 (the IO wave, the terminology ruling, NO DUPLICATIONS). Baseline: the wave-113
docs pass (7c550857 / 2f994d2b). Julia 1.13.3; gate 2630a051 (2dc3ac60 + the temporary Features compat line on the gate
branch only). Records-only commits after 2dc3ac60 change CHANGELOG / comments, no code. This pass covers the code at
2d48ca86: the four code changes after 2630a051 are listed in §4 (d90bc914, 28db5516 + 76d0c6aa, 3bc4eb10, 2d48ca86).

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

## 4. The code after gate 2630a051, up to 2d48ca86 (each with its CHANGELOG entry, Eval 0.8.0)

- [x] **d90bc914 — the runner's precompile phase keys nothing.** A CPU-only `Pkg.precompile()` is environment state, not
  a result: it is left out of the stage's run key, and the site's `precompile_time` is recorded in the manifest as a
  measurement (in no key). Docs: reproduce.md (the precompile phase and `precompile_time`). Evidence: dev job 14470.
- [x] **28db5516 + 76d0c6aa — `write_provenance(…; sensor_model)`** (new keyword: a `SensorModel` or one per frame): the
  run record states the sensor model's provenance under `sensor_model`, per distinct model with its frame count, read
  from VisualGeometryToolkitCore's `component_sources` (its one provenance view; no copy in Eval). `sensor_model` is a
  reserved provenance field. Docs: `write_provenance`'s docstring. The test binds no "measured" (R14). Evidence: dev jobs
  14495, 14509.
- [x] **3bc4eb10 — a run of record of a REGISTERED package's code is accepted** (the port of Eval 0.7.4, patch-eval074
  7271049c): `code_trees` identifies a depot install by the tree hash the active environment records for it, through
  Pkg's API (`Pkg.dependencies()`: name, version, uuid, git-tree-sha1), the one source of that hash; the check before it
  is a file-system look (no git run per call, inside the memoised `code_trees`). Docs: `code_trees`' docstring.
  **Open (a docs change, not in this records commit):** reproduce.md at 2d48ca86 still says code "in no git checkout" is
  refused; patch-eval074's accedc52 (its page edit: a registered install's identity, the refusal narrowed) was not
  ported with 3bc4eb10. Evidence: dev job 14543 (616/616).
- [x] **2d48ca86 — `code_trees` binds the active project to a local before its `nothing` test** (JET infers the
  narrowing; the port of patch-eval074's 4209e867). Evidence: dev job 14590 (quality_gates, reproduce_runner,
  experiments_data: 662/662).

Gate of the code at 2d48ca86: the release integrator's final full gate (Estimation, Eval, PoseLib at the release
commit, which contains 2d48ca86) — cited here when it reports. The up-v13 snapshot (a8f4aa02, which merges 2d48ca86)
loaded all six packages (job 14592).

