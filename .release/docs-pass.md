# 0.9.3 docs pass (julia-docs-pass) — release-0.9.3

The packages released: VisualGeometryToolkitCore 0.9.3 (one public name added), VisualGeometryToolkitIO 0.9.2 and
VisualGeometryToolkitFeatures 0.8.1, all three patches. The others are unchanged. The baseline is 9dc838a1 (the 0.9.2
release); the earlier passes are below.

## 1. Public API since 9dc838a1

Core adds the public `require_extension` (base/utils.jl, docstring on api/core.md through its `base/utils.jl`
`@autodocs`): the stack's one check for a package extension, hoisted from Features' and BlobBoards' private copies.
Nothing else is added, removed or renamed in any package. Core changes: the MappedSurface root computes with explicit fused
products, its bits now the same at every call site (a fix); the private `_chart_affine` is deleted. IO: an embedded old
grid camera body is now read (a fix).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `require_extension`: a docstring with both signatures and an example, on api/core.md (that page is ~177 KiB rendered; the docstring adds ~2 KiB). Changed in place: docstrings changed in place: `camera_from_json` (IO: the old embedded body, read in colmap), the IO kinds page |
| CHANGELOG | Core 0.9.3 and IO 0.9.2 sections; the registered headings dated 2026-11-12 (Core 0.9.2, IO 0.9.1, Estimation 0.9.1, Features 0.8.0) |
| versions | Core 0.9.3, IO 0.9.2, Features 0.8.1 (its Core floor "0.9.3" in package, docs and test environments) |
| in-repo compat | none needs moving: every package's Core compat admits 0.9.3, and Eval's IO compat "0.9" admits 0.9.2 |

## 3. Executed on the final RC

To be filled from the final gate (Julia 1.13 and 1.12; GPU six, CPU six, docs).

---

# 0.9.2 docs pass (julia-docs-pass) — release-0.9.2

The packages released: VisualGeometryToolkitCore 0.9.2, VisualGeometryToolkitIO 0.9.1, VisualGeometryToolkitFeatures
0.8.0 (Breaking) and VisualGeometryToolkitEstimation 0.9.1. Eval 0.7.0 and PoseLib 0.2.0 are unchanged. The baseline
is d1a5f837 (the registered stack). The 0.9 pass is below.

## 1. Public API since d1a5f837

| Package | added | removed / no longer exported |
|---|---|---|
| Core | exported: `RawMosaic`, `LinearRaw`, `BayerCellFrame`, `raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `raw_coordinates`, `channelview` methods, `photositeview`, `linearview`, `saturationview`. Public: `bayer_channels`, `LinearRawView`, `SaturationView`, `PhotositeColors`, `raw_provenance`, `cfa_string`, `photosite_colors`, `signed_area` (was `_shoelace_signed_area`), `normalized_point` (was `_normalized_point`), `bilinear_weights` (was `_bilinear_setup`, its arguments reordered), `julia_blocks`. New methods: `with_parameters(::MappedSurface, θ)`, `pixel_mask(region, frame_map, w, h)`, the `SimilarityMap` actions on `Rect` and `RegionIntersection` | none |
| IO | public `metres` (was `_metres`), `with_provenance` (was `_with_provenance`) | none |
| Estimation | public `image_size` (was `_image_size`; in `CONTRACT_GENERICS`) | none |
| Features | exported `detect_image_circle` | no longer exported: the sensor image moved to Core (`RawMosaic`, `LinearRaw`, `BayerCellFrame`, `raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `raw_coordinates`, `raw_provenance`, `cfa_string`, `photosite_colors`), and the old re-exports of Core's `feature_frame`, `Cspond` and `csponds` |

Every change is in its package's CHANGELOG. Features 0.8.0 lists each dropped name.

## 2. Static findings, and what was done

| check | result |
|---|---|
| Removed or no-longer-exported names in docs, README, examples, configs | No build or example failure. Prose in the Features guides now names Core as the owner of `feature_frame` and `csponds`. The raw-images snippet loads Features for `load_raw`. |
| Added public names: docstring, file covered by `@autodocs`/`@docs` | All covered. The sensor image is on a new Core page, `api/raw.md`. The other additions are on primitives, cameras, warping, lenses and stack_contracts in Core, on native in IO, on radial in Estimation, and on detection in Features. |
| Page sizes (200 KiB hard limit) | `api/raw.md` is new. `api/lenses` gains ~3 KiB (`pixel_mask`), estimated ~185 KiB. The strict build is the check (§3). |
| docs/make.jl gates | `checkdocs = :public`, `doctest = true`, linkcheck, no `warnonly` (Core's inventory-only rebuild excepted, as in 0.9). The new pages are wired. |
| Detached docstrings in changed files | none |
| README install lines | unpinned; no README example uses a dropped name |
| Versions, headings, in-repo compat | Core 0.9.2, IO 0.9.1, Features 0.8.0, Estimation 0.9.1, each with an "(unreleased)" CHANGELOG section. The Core floor is "0.9.2" in Estimation's and Features' package, docs and test environments. Features' examples and validation compat is "0.8". No other package depends on Features. |

## 3. Executed on the final RC

The final RC is 76efbdbb. The gate ran on two fresh clones, one per Julia version: a `git clone` with no Manifest, its
workspace resolved from scratch. Each suite includes its `docs`, README, agent-docs, quality and naming items. Counts
are pass / fail / error / broken.

| Package | 1.13.3 GPU (10211) | 1.13.3 CPU (10212) | 1.12.7 GPU (10215, Features rerun 10279) | 1.12.7 CPU (10216) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| Core | 380433 / 0 / 0 / 2 | 378855 / 0 / 0 / 30 | 380430 / 0 / 0 / 2 | 378852 / 0 / 0 / 30 | exit 0 / exit 0 |
| IO | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | exit 0 / exit 0 |
| Estimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | exit 0 / exit 0 |
| Features | 3069 / 0 / 0 / 2 | 3012 / 0 / 0 / 8 | 3069 / 0 / 0 / 2 (rerun) | 3012 / 0 / 0 / 8 | exit 0 / exit 0 |
| Eval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | exit 0 / exit 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | exit 0 / exit 0 |

The first 1.12.7 GPU run of Features (10215) had 1 failure and 1 error. The single rerun on the same tree (10279) is
green.
- Aqua "persistent tasks" was not a persistent task. Its subprocess could not load the package: another job was
  precompiling it in the same depot ("Being precompiled by another process", then a cache miss for ImageMorphology).
  Aqua's own message says "a precompilation error, not a persistent task".
- "pairwise_matrix: exact for any descriptors" (`vgf_tests.jl:692`, the default device's Float32 L2 matrix `==` the
  CPU's) failed once and passed on the rerun and at the released d1a5f837 (10278). This batch does not touch the test
  or the kernel. It is listed as a post-release follow-up: an intermittent GPU/CPU bit difference.

---

# 0.9 docs pass (julia-docs-pass) — final RC

Packages: VisualGeometryToolkitCore 0.9.0, VisualGeometryToolkitIO 0.9.0, VisualGeometryToolkitEstimation 0.9.0,
VisualGeometryToolkitFeatures 0.7.0, VisualGeometryToolkitEval 0.7.0, PoseLib 0.2.0. Baseline: 3eabba46 (the last
docs-reviewed candidate). Final RC: 9c74e264 (the version bump 1d816a22 and the release records).

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

The final RC is 9c74e264. The gate ran on a fresh clone (/home/jpr/src/vgtk-final-9c74e264): a `git clone` with no
Manifest, its workspace resolved from scratch with Julia 1.13.3. Each suite includes its `docs`, README, agent-docs,
quality and naming items. Counts are pass / fail / error / broken.

| Package | strict docs build (9765/9766) | GPU suite (9763) | CPU suite (9764) |
|---|---|---|---|
| Core | exit 0 | 342441 / 0 / 0 / 2 | 342175 / 0 / 0 / 25 |
| IO | exit 0 | 30522 / 0 / 0 / 3 | 30522 / 0 / 0 / 3 |
| Estimation | exit 0 | 20956 / 0 / 0 / 3 | 20946 / 0 / 0 / 3 |
| Features | exit 0 | 15520 / 0 / 0 / 2 | 15146 / 0 / 0 / 9 |
| Eval | exit 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 |
| PoseLib | exit 0 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 |

There are no failures or errors anywhere, the README items included. The CPU run's extra broken counts are the GPU
items, which are recorded as broken when no device is present. Core's api/estimators page, which had been 201.15 KiB
(over the 200 KiB limit), is now under the 150 KiB warning size since the split in 541e1681.

## 4. Before General (cau-release runs it)

`bash ~/.claude/skills/julia-docs-pass/general_preflight.sh <package dir>`, per package, after the owner's go. Then
read every MANUAL line against general-checklist.md.
