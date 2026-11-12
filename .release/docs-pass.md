# 0.9 docs pass (julia-docs-pass) — final RC

Packages: VisualGeometryToolkitCore 0.9.0, VisualGeometryToolkitIO 0.9.0, VisualGeometryToolkitEstimation 0.9.0,
VisualGeometryToolkitFeatures 0.7.0, VisualGeometryToolkitEval 0.7.0, PoseLib 0.2.0. Baseline: 3eabba46 (the last
docs-reviewed candidate). Final RC: the version bump 1d816a22 plus this file's commit (no code).

The pass has two parts:
- **Static** (done on dd9eeec5, read-only): the API diff, stale mentions, docs coverage, make.jl gates, detached
  docstrings.
- **Executed** (the final gate, in Slurm): the strict docs builds and the suite items that hold the docs true. These
  are `docs` (`undocumented_public`, `signature_mismatches`, `detached_docstrings`), `README`, `agent docs`, `quality`
  and `naming`.

## 1. Public API since 3eabba46

| Package | added | removed / renamed |
|---|---|---|
| Core | the strategy layer, moved from Estimation: `AbstractOptimizationStrategy`, `Irls`, `Triggs`, `EfficiencyConstant`, `LOSS_TAGS`, `STRATEGY_TAGS`, `@lo_registry`, `fixed_width`, `strategy_loss`, `lo_*` |  |
| | the termination record: `ReturnCode`, `StopRule` (18 rules), `stop_rule`, `found_model`, `retcode_symbol`, `retcode_from_symbol`, `successful_retcode`, `worst_retcode`, `nlls_termination`, `test_termination_record` | `nlls_retcode` → `nlls_termination`; `lm_retcode` removed |
| | hoists: `RetiredKeyword`, `refuse_retired_keyword`, `bearing_csponds`, `prefix_sweep`, `install_artifact`, `huggingface_token`, `free_device_buffers!`, `free_parameter_names`, `map_estimate`, `reproject_so3`, `s2_tangent_basis_matrix`, `sample_point_held`, `unit_weights`, `@threshold_scoring_constructors` |  |
| | | `AbstractLocalOptimization` → `AbstractOptimizationStrategy`; `chisq_median`, `ScaleMultiple` and `deployed_scale_multiple` removed |
| IO | none | none |
| Features | `raw_coordinates` | none |
| Estimation | `OPTIMIZATION_STRATEGY_TAGS`, `RefineMetadata`, `RigidAlignmentMetadata`, `RotationCenterRefineMetadata`, `RotationCenterViews`, `rotation_center_pose`, `ViewAngleLimitError`, `approximate_camera`, `bundle_loss`, `default_student_t_uniform_scoring`, `fit_filtered_rotation_averaging`, `fit_rigid_alignment`, `irls_termination`, `lo_resolve_efficiency_constant`, `n_without_bearing`, `without_bearing`, `optimization_strategy_from_toml`, `pooled_noise_scale`, `refine_camera`, `refine_relative_pose`, `uncertain_rotation`, `with_extent` | renamed: `bundle_refine_relpose` → `refine_relative_pose`, `filtered_rotation_averaging` → `fit_filtered_rotation_averaging`, `local_optimization_from_toml` → `optimization_strategy_from_toml`, `lo_resolve_scale_multiple` → `lo_resolve_efficiency_constant`, `LO_TAGS` → `OPTIMIZATION_STRATEGY_TAGS`. Removed: `NoLocalOptimization`, `deployed_shape`, `plain_scale_multiple`, `lo_scale_multiple_shape`. Moved to Core: the strategy layer above |
| Eval | `REPRODUCE_SCRIPT_TEMPLATE`, `UNPUBLISHED_TIER`, `catalog_view`, `catalog_view_mount`, `code_tree_changes`, `code_trees`, `fetch_data!`, `file_sha256`, `inspect_input`, `mount_dir`, `project_relative_path`, `provenance_token`, `refuse_uncommitted_code`, `reproduce_script`, `scratch_record`, `slurm_queue_states`, `stage_markers`, `stage_markers_dir`, `stage_run_state`, `tier_data_root`, `tier_variable`, `tree_sha256`, `write_reproduce_script` | none |
| PoseLib | `LeZach`, `POSELIB_STRATEGY_TAGS`, `PoseLibRefineMetadata` | `LeZachTruncation` → `LeZach` |

Every removal and rename is in its package's CHANGELOG (Breaking).

## 2. Static findings, and what was done

| check | result |
|---|---|
| Removed or renamed names still in docs, README, docstrings, examples or configs | One hit, fixed in fc3bf6dd: PoseLib/CLAUDE.md:62 `LeZachTruncation()` → `LeZach()`. The retired-name lists in check_naming are deliberate. |
| Estimation prose that passes `Irls` to a `refine_*` (refused since dup-item-6) | None. The refine prose says `Triggs` or `nothing` and names the refusal; the remaining `Irls` mentions are `fit_*` or ransac LO. One ambiguous comment (configs/default.toml:88, `[pipelines.bearing_pose]`: "the default refine") now reads "the default local optimization" (fc3bf6dd). |
| Added public names: docstring present, file covered by `@autodocs`/`@docs` | All present and covered by static reading. Confirmed by `checkdocs = :public` in the strict builds. |
| docs/make.jl gates (`checkdocs = :public`, `doctest = true`, linkcheck, no `warnonly`) | All six. Exception: Core make.jl sets `warnonly` only when `VGTK_DOCS_INVENTORY_ONLY=1`. That is the Slurm docs gate's inventory-only rebuild after a failed strict Core build (ci/slurm/docs.sbatch). It is never set for the strict build that gates. Reason at the site (make.jl comment). |
| `linkcheck_ignore` | host-level only (github.com, doi.org, publishers, the local inventory server), each with its reason |
| Detached docstrings (150 changed src files) | none |
| README install lines and pinned versions | none pinned; `julia = "1.12"` everywhere; no README example uses a removed name |
| docs and README test items | all six packages have `docs_tests`, `readme_tests` and agent-docs items |
| In-repo compat for the bumps | Eval and Eval/test on Estimation "0.8" → "0.9", Features examples and validation on Features "0.6" → "0.7": in the version-bump commit 1d816a22 |

## 3. Executed on the final RC

| Package | strict docs build | docs item | README item | agent docs | quality / naming |
|---|---|---|---|---|---|
| Core | `<exit>` | | | | |
| IO | | | | | |
| Features | | | | | |
| Estimation | | | | | |
| Eval | | | | | |
| PoseLib | | | | | |

README items are known to fail until the 0.9 stack is registered: the fresh environment from the install line
resolves from the registry. They pass after registration. This is the only accepted failure.

## 4. Before General (cau-release runs it)

`bash ~/.claude/skills/julia-docs-pass/general_preflight.sh <package dir>`, per package, after the owner's go. Then
read every MANUAL line against general-checklist.md.
