# Wave 10 duplication checklist: est-eval (Estimation 0.11.0, Eval 0.8.0, PoseLib 0.3.0)

Every concept this theme added or folded, the stack searched BY CONCEPT first (Core, IO, Features, Estimation, Eval,
PoseLib, BB, RSDR, LODR, PS, CB; RC read-only). A box is ticked when the final gate (2630a051) is green on it.

| # | concept | considered | done | status |
|---|---|---|---|---|
| 1 | a second name for a problem family / an RNG | `LiftableProblem`, `POOL_RNG` | deleted, RETIRED_NAMES (no alias) | [x] |
| 2 | the chart-LM refit prologue ×3 | interface.jl `_refine!`, _refine.jl `_refine_run`, manifold_providers.jl `_triggs_lm` | ONE `_chart_refit` | [x] |
| 3 | bundle LMs beside Core's adapter | `_solve_bundle`, `_bundle_optimize`, `_bundle_retcode`, `_robust_stationary`, stationarity-row copies | deleted; Core's `LmProblem` / `lmsolve_chart!` / `LmStops` | [x] |
| 4 | per-site stop literals | `[chart_lm]`, `_CHART_LM_*`, relpose xtol/ftol literals, `_regress_lens` (no ftol) | Core's `[lm]` via `LmStops`; site overrides from own tables | [x] |
| 5 | a private stop-rule map | `_stop_rule(stop, model_update)` | methods of Core's `stop_rule(criterion, method)` | [x] |
| 6 | the lens regression's launch + reduce | refine_camera.jl hand unpack | Core's `reduce_normal_equations` / `reduce_cost` (patch 822a01cf, a4978736) | [x] |
| 7 | a region clip with a warning | Estimation's clip-past-the-fold | deleted; Core's `check_region_support` refuses | [x] |
| 8 | an alias of Core's `extent` | `image_size`, Eval `_pixel_size` | deleted; `extent` | [x] |
| 9 | a retired-keyword refusal | Estimation `_refuse_held_spellings`, PoseLib `_check_refine_keywords`, `_refuse_forward_refit` | Core's `refuse_retired_keyword` | [x] |
| 10 | a second canonical encoding | Eval `_toml_hash` (sorted TOML) | Core's `canonical_json` (JCS); the v6/v7 recipe kept only to verify stored hashes | [x] |
| 11 | a TSV writer | Eval `verify_assets`' hand writer, RC `write_rows` | Render's one `write_tsv` (+ NamedTuple rows) | [x] |
| 12 | a window widening | CB icra2026 `WindowWidening` | Eval `WindowExtension` + `extend` (CB deletes its copy) | [x] |
| 13 | a second SHA-256 stack | Eval code_identity.jl | declared in DUPLICATION.toml (bare-julia bootstrap) | [x] |
| 14 | the precompile job beside a phase mechanism | `_precompile_command` | the runner's `precompile` phase | [x] |
| 15 | data-length splats | 3 Estimation, 10 Eval findings | fixed; 4 fixed-arity `[[splat]]` entries | [x] |
| — | E11 `refine_width`, E12 `assert_unit_bearings` (PoseLib/Estimation copies) | handed to core10 (handoffs.md) | open with core10 | [ ] |

check_duplication / check_splat (gate prep 14383, verbatim):
```
== check_duplication(VisualGeometryToolkitEstimation 0.11.0): 0 finding(s)
== check_duplication(VisualGeometryToolkitEval 0.8.0): 0 finding(s)
== check_duplication(PoseLib 0.3.0): 0 finding(s)
== check_splat(VisualGeometryToolkitEstimation 0.11.0): 0 finding(s)
== check_splat(VisualGeometryToolkitEval 0.8.0): 0 finding(s)
== check_splat(PoseLib 0.3.0): 0 finding(s)
```
