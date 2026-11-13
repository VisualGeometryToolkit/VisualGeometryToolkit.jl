docs-pass: 09018e60

# Wave 10 docs pass (julia-docs-pass) — VisualGeometryToolkitCore 0.10.0, branch w10-core10

Baseline: the wave-113 release, 2f994d2b (Core 0.9.5). The breaking wave of 2026-11-13. The API is listed in full in
Core's CHANGELOG `## 0.10.0`, with its Breaking and New lists. The function list, with file:line and the stack functions
each one considered, is in ~/logs/dgx-agent/wave10/core10.md. Every public interface change was posted in
~/logs/dgx-agent/wave10/interfaces.md before its commit.

## 1. Public API since the baseline

- **Breaking**:
  - the terminology renames, `Cspond`/`Csponds`/`csponds` → `Correspondence`/`Correspondences`/`correspondences`, and the
    rest (no aliases; each in `check_naming`'s retired names);
  - `TargetDetections` → `TargetObservations`, with `TargetPoints2D/3D` and `TargetPoint2D/3D` deleted;
  - one LM: `LmStops`, `StaticLmStops`, `LmProblem`, `lmsolve_chart!(…, stops)`, one success rule, `nlls_options` deleted;
  - the FeatureFrame interface, with every verb once;
  - the sensor model, which replaces wave 10's interim sensor-noise names;
  - `LinearRawView{T, A, D, L}` (D14);
  - `RelativeNoise`, which is specified only;
  - `block_coordinate_result`, which takes a stop rule;
  - the weak floors, restored.
- **New, public**:
  - FeatureFrame: `center(::Uncertain{<:FeatureFrame})`, `frame_dof`, `transfer(m, ::FeatureFrame)`;
  - the GPU test tier: `gpu_tier`/`gpu_problem`/`require_gpu`;
  - LM: `stop_rule`;
  - the quality gates: `check_splat`/`SplatViolation`;
  - the sensor model:
    - `SensorModel`, `SensorCharacterisation`, `sensor_model`, `content_digest`, `component_sources`;
    - its components and `SensorSource`;
    - its estimators: `photon_transfer`, `master_dark`, `defect_onsets`;
    - `poisson_slope(L, model[, plane])`, `green_plane`;
  - `Statistics.var(::Uncertain{<:Real})`;
  - `canonical_json`, hoisted from IO;
  - regions: `check_region_support`/`RegionBeyondSupport` and the one `⊆`;
  - `BinnedRadialProfile`/`radial_coordinate`;
  - `TargetPlaneFrame{id}`;
  - `iou_below`, `page_examples`, `install_lines`, `refine_width`, `assert_unit_bearings`;
  - forward-ported from Core 0.9.6–0.9.9 (they reach the wave through these ports):
    - 0.9.6: `reduce_normal_equations`/`reduce_cost`;
    - 0.9.7: their launch-and-reduce forms;
    - 0.9.8: `device_resident_threads`;
    - 0.9.9: `pointwise_kernel`.

## 2. Static findings

| check | result |
|---|---|
| every public name has a docstring | `test_docs`: 0 undocumented beyond the allow-lists. The `SensorSource` enum's type and four values each have their own docstring (12e1bc48); there is no exception entry. |
| docstrings attached, signatures match | `detached_docstrings` 0; `signature_mismatches` 0 (the docs item in the gate's CPU group) |
| every public docstring in the manual | `checkdocs = :public` green. The new sensor-model, region and LM pages go through `@autodocs` `Pages`, and `SensorSource`'s members through one `@docs` block in api/raw.md. |
| cross-references | the `SensorSource` docstring's references are qualified (Documenter resolves a bare `@ref` inside the enum's module; 4cbd4285) |
| stale mentions | retired names: none outside the CHANGELOG (`check_naming` R9/R14: 0) |
| CHANGELOG | Core `## 0.10.0` (unreleased), with Breaking, Fixed and New |
| version | VisualGeometryToolkitCore 0.10.0 |

## 3. Executed

| run | Core at | result |
|---|---|---|
| docs build (Julia 1.13.3, `--startup-file=no`, docs env), job 14852 | 09018e60 | green: no warning beyond Documenter's HTML-size notices |
| up-v13 integration gate a8f4aa02 (jobs 14786–14790; all six packages, wave stack paths) | 12e1bc48 | 48/48 green. For Core: CPU 413524 pass / 32 broken over 8 groups; GPU 54319 / 0 over 4 groups; docs green; prep lint, agent_docs, floor --latest, audit 0; `check_duplication` 0 on all six packages |
| load and the pointwise items, job 14851 | 09018e60 | 2937 / 1 broken (the patch agent's GPU run of the same items: 2973 / 2973) |

12e1bc48..09018e60 is the Core 0.9.9 forward port (`pointwise_kernel`, 5006841e and 09018e60). It has its own load, test
and docs runs above, and it goes into up-v14.

---

docs-pass: 80b9fb45
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 -> 6674c98 prepared locally (/raid/data/jpr/vgtk-landing-wave113, branch wave-113: "The packages require Julia 1.13"; not pushed)

# Wave 10 docs pass: VisualGeometryToolkitIO 0.10.0 (julia-docs-pass) — branch w10-io10-docs 80b9fb45 (w10-io10 0d6e90e5 merged)

The IO breaking wave (specs/2026-11-13-io-breaking-wave.md): one envelope (`schema`, `generator`, one `units` object, one
`conventions` object), every kind a new major `v<M>.<m>`, the observation kind, the detection / sensor_noise /
radial_profile / panorama v2 kinds, one camera store, one lattice record, one frame codec, the deliberate re-id.
Baseline: the IO 0.9.3 docs pass (7c550857). Julia 1.13.3, dev jobs on dgx01 (2 CPUs).

## 1. Public API since IO 0.9.3

- Added (public, unexported): `schema_versions`, `with_caller`, `validate_record`, `migrate_envelope`, `migrate_units`,
  `json_schema`; `VgioStyle(record)`, `VgioStyle(; …, density, length_roles)`; the `load`/`save` methods of
  `vgio/detection/v1.0`, `vgio/sensor_noise/v1.0`, `vgio/radial_profile/v1.0`; `TargetObservations` (`vgio/observations/v3.0`).
- Renamed: `migrate_opencv`/`migrate_matlab` → `convert_opencv_calibration`/`convert_matlab_calibration`; ColmapDatabase
  `keypoints` → `detections`; `save_manifest(…; kind)` → `result_kind`.
- Removed: `ScalarOrQuantity`, `pose_declarations`, `json_value`, `from_json`, `read_quantity`, `write_quantity`, the
  `TargetPoints2D`/`TargetPoints3D`/`TargetDetections` `load`/`save` methods (types gone from Core). Every mention deleted from
  README, docs/src and docstrings (grep: none left outside CHANGELOG and the version-step prose that names them as read).

## 2. Static findings, and what was done

| check | result |
|---|---|
| docstring gates (Core's `test_docs`) | undocumented 0, detached 0 (was 1: `frame_from_name`, consts between docstring and definition, fixed), signature mismatches 0; no exceptions, no backlog |
| docstrings | `frame_from_name` (attached, a doctest), `migrate_envelope` (its `density` keyword), `validate` (no "v2 file"), `VgioStyle` (the retired `JSON.lower(style, contract)` claim), `vgio_migrate_tree` (manifest rebinding, the id map's `rewritten`), OpenCV/MATLAB readers (`target → camera`) |
| manual | kinds.md rewritten: the envelope and minor versions, every kind at its 0.10 version, observations, the camera store with non-central cameras and the one lattice record, targets (tags, point vocabulary, placement), the version table GENERATED from `schema_versions()` (`@eval`), IO 0.10's step, the cameras v1 doctest now migrating to `vgio/cameras/v6.0`, the re-id and `vgio_migrate_tree`, the building blocks (the three `VgioStyle` forms, the quantity/Size2/frame codec, blur planes, `write_record` writing the envelope, `migrate_envelope`, `migrate_units`, `validate_record`); schemas.md, conventions.md, getting-started.md (the printed 0.10 file; a DocTestFilter on the generator version), third-party.md (`convert_*`), related-formats.md, serialization.md (its DocTestSetup was missing), index.md and README (the envelope; links to JSON Schemas and Content Identity) |
| API pages | native.md: the deleted noncentral_spline_schema.jl dropped, units_sidecar.jl added; third-party.md: migrate_v010.jl added |
| docs env | docs/Project.toml gains FlexUnits (the serialization doctest names `Quantity`) |
| terminology (R14) | no "measurement", "keypoint", "sighting", `cspond` in docs/src, README or docstrings (COLMAP's own `keypoints` table name is a foreign token) |

## 3. Executed

| run | result |
|---|---|
| `docs/make.jl` strict (checkdocs = :public, doctest, linkcheck, no warnonly) + Core's `test_docs`, job 13003, a detached clone of 80b9fb45 | BUILD OK; undocumented 0, detached 0, mismatches 0 |
| test items README, docs, agent docs, panorama, json_schema (13004, the same clone) | 443/443 |
| quality items (12878, 49bd5f38; not this slice's) | naming: R9 retired names in test/ only (calibration_manifest, covariance_storage, units_roundtrip, v2_validation tests); Aqua: JSON.lift ambiguities (tests slice fixing core_json.jl); ExplicitImports: `Base.unwrap_unionall` (migrate.jl:286, `schema_versions`); JET 10 > 0 |

## 4. Open

- `VgioStyle(record)` accepts an `AbstractDict` but reads `record.units` by property access: a `Dict` record fails (io10).
- Core's conventions/units.md still describes per-file `length_unit` and the `{"world", "density"}` units layout (core10).
- README/install tests resolving from the registry: known pre-registration failure until the wave registers.

# Wave 113 docs pass (julia-docs-pass) — wave-113 (Core 0.9.5, IO 0.9.3, Features 0.8.3, Estimation 0.10.1, Eval 0.7.2, PoseLib 0.2.1)

The wave: the whole stack on Julia 1.13 only, every [compat] floor raised to the gated resolution, the Downgrade
workflow deleted; Core 0.9.5 = the SIFT landing plus `farthest_radius`, the masked `bending_rows`, `backproject` of a
non-central model at any number type, `plane_map` for every camera model, `transform_deriv` of the 2-D homography map,
the CUDA default device with `always_inline`; Estimation 0.10.1 = `held_parameters`, `pixel_whitening_defect`; the
private forwards and aliases deleted (the owner's no-shim order). The "cuda" device-tag rename was stopped as breaking
(Features' `execution_provider`), by the coordinator's ruling left as it is. Baseline: sift-landing 8f979b38.

## 1. Public API since the baseline

- Core, public, not exported: `farthest_radius`, `plane_map`; new methods `bending_rows(mask; spacing)`,
  `transform_deriv(PerspectiveMap() ∘ ::HomographyMap{3}, ::StaticVector{2})`, `project(::NoncentralSplineModel, ::StaticVector{3, <:Dual})`;
  `Ray3{T <: Real}` (was `AbstractFloat`); `backproject(::NoncentralSplineModel)` eltype-generic. Nothing removed.
- Estimation, public, not exported: `held_parameters`, `pixel_whitening_defect`. Nothing removed.
- IO, Features, Eval, PoseLib: no API change.
- The function list with file:line, the stack functions considered and the deleted shims: ~/logs/dgx-agent/wave113-functions.md.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | Core: `farthest_radius` (cameras/image_circle.jl) and `plane_map` (cameras/cameras.jl) on api/lenses.md and api/cameras.md through their `@autodocs`; `bending_rows(mask)` on api/batched.md (numerics/bspline_energy.jl); the 2-D homography map's derivative is a commented CT method (no docstring on a foreign function, as `transform_deriv(::PerspectiveMap, X)`). Estimation: `pixel_whitening_defect` in api/bearings.md's `@docs`, `held_parameters` in api/radial.md's, both named in the rotation-centre prose. |
| stale mentions | the deleted private names: none outside CHANGELOGs and earlier release records (grep in wave113-functions.md). "Julia 1.12 or later": every README, docs index and getting-started page reads 1.13; CLAUDE.md's workspace line too. |
| CHANGELOG | Core 0.9.5, IO 0.9.3, Features 0.8.3, Estimation 0.10.1, Eval 0.7.2, PoseLib 0.2.1 |
| versions | as above |
| in-repo compat | every member's floors at the gated resolution (wave113-floors.txt, 164 entries; floor_compat `--raise --check` on a copy: 0 more); stack siblings at their in-wave versions |
| README / landing | the landing page's "require Julia 1.12" line prepared as 6674c98 (above) |

## 3. Executed on the final RC (7c550857)

Gate (~/logs/dgx-agent/scripts/wave113/gate.sbatch, submit.sh; jobs 11013–11020): a fresh clone per task of `wave-113`, no
Manifest, General a GitHub clone in the wave's own depot (JULIA_PKG_SERVER=""), prep without a GPU (resolve, precompile,
lint, floor, audit, dup), then the CPU tiers and docs without `--gres` and each GPU tier on its own GPU, `--depwarn=error`,
JULIA_NUM_THREADS = the task's CPUs, Core at 3 ReTestItems workers. Julia 1.13.3 only. Counts are pass / broken.

| | CPU | GPU | docs |
|---|---|---|---|
| prep: registration_lint --subsequent, agent_docs, floor_compat --latest, audit_manifests | 0 errors, 0 warnings; rc 0; rc 0; all up to date | | |
| check_duplication (all six, stack paths set) | 0 findings each | | |
| Core | 410185 / 30 | 411768 / 2 | 0 |
| IO | 30541 / 4 | 30541 / 4 | 0 |
| Features | 3063 / 9 | 3133 / 2 | 0 |
| Estimation | 8813 / 2 | 8823 / 2 | 0 |
| Eval | 2906 / 0 | 2907 / 0 | 0 |
| PoseLib | 373 / 1 | 373 / 1 | 0 |

The duplication item is the same without and with VGTK_STACK_PATHS (the idiom cap counts the repository's own
definitions; job 10988: 1889/1889 both ways).

---

docs-pass: 94a303ff (sift-landing)
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 (no change: its Features line names ASIFT detection, not a refinement option)

# 0.9.5 / 0.8.3 docs pass (julia-docs-pass) — sift-landing (VisualGeometryToolkitCore 0.9.5, VisualGeometryToolkitFeatures 0.8.3)

The packages released: VisualGeometryToolkitCore 0.9.5 and VisualGeometryToolkitFeatures 0.8.3, both additive patches (the
owner's SIFT landing rulings of 2026-11-13). Estimation, IO, Eval and PoseLib are unchanged. The baseline is main c39b4a2d
(Core's tree its registered 0.9.4's, Features' its 0.8.2's).

## 1. Public API since Core 0.9.4 / Features 0.8.2

- Core, public, not exported: `static_lm` (the former private `_spline_lm`, at static N), `static_lm_termination`,
  `cubic_bspline_weight_derivatives` (the former private `_cubic_bspline_weights_deriv`, with the second derivative),
  `spline_window_sums`, `cubic_bspline_prefilter_taps`; `is_positive_definite` gains its static-N method. A generic
  docstring for `score_sq_residuals!`. Nothing removed from the public API.
- Features, exported: `ContinuousRefinement`; `asift_detect`/`asift_detect!` take the keyword `refinement` (default
  `nothing`: unchanged). Every other new name is private.
- The function list with each new name's file:line and its concept-level duplication check:
  ~/logs/dgx-agent/sift-landing-functions.md (in the owner's hands before posting).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | Core: each new public name has a docstring on its page's `@autodocs` (cameras/spline_lens.jl, cameras/chart_spline.jl, primitives/feature_frames.jl, robust/student_t_uniform_scoring.jl); `static_lm`'s docstring sits on the function (the comment block above it, not between). Features: `ContinuousRefinement` in an `@docs` block on api/detection.md (continuous_refine.jl is not in the page's `@autodocs`: its other docstrings are private); the `asift_detect` docstring's "Continuous refinement (opt-in)" paragraph; the guide's section "Continuous refinement (opt-in)" with the measured numbers. No `@ref` from a public docstring to a private name. |
| stale mentions | `_spline_lm`, `_spline_lm_increment`, `_cubic_bspline_weights_deriv`: none outside CHANGELOGs and the historical plan under docs/superpowers. The private Features names (`_DogSplineWindow`, …) appear in no doc page. |
| CHANGELOG | Core 0.9.5, Features 0.8.3 |
| versions | Core 0.9.5, Features 0.8.3 |
| in-repo compat | Features' package, test and docs environments: VisualGeometryToolkitCore "0.9.5". The other packages' "0.9" / "0.9.2" admit it. |
| README / landing | no README or landing text names the refinement: unchanged |

## 3. Executed on the final RC (94a303ff)

Gate (~/logs/dgx-agent/scripts/siftland/gate.sh; jobs 10745–10748): a fresh clone per run of `sift-landing`, no
Manifest, General a GitHub clone in the gate's own depot (holding Core 0.9.4, IO 0.9.2, Features 0.8.2, Estimation 0.10.0,
Eval 0.7.1), CauRegistry from the shared depot; every suite strict (`--depwarn=error`), Core with RETESTITEMS_NWORKERS=3.
Counts are pass / fail / error / broken.

| | 1.13.3 GPU (10745) | 1.13.3 CPU (10746) | 1.12.7 GPU (10747) | 1.12.7 CPU (10748) |
|---|---|---|---|---|
| registration_lint --subsequent | rc 0 | | rc 0 | |
| floor_compat --latest | rc 0 | | rc 0 | |
| check_duplication(Core), (Features) | 0, 0 findings | | 0, 0 findings | |
| Core (--depwarn=error) | 411597 / 0 / 0 / 2 | 410019 / 0 / 0 / 30 | 411594 / 0 / 0 / 2 | 410016 / 0 / 0 / 30 |
| Features (--depwarn=error; JET, Aqua, ExplicitImports, duplication in-suite) | 3133 / 0 / 0 / 2 | 3063 / 0 / 0 / 9 | 3133 / 0 / 0 / 2 | 3063 / 0 / 0 / 9 |
| Estimation (--depwarn=error) | 8817 / 0 / 0 / 2 | 8807 / 0 / 0 / 2 | 8805 / 0 / 0 / 2 | 8795 / 0 / 0 / 2 |
| Eval (--depwarn=error) | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 |
| Core docs, Features docs (strict make.jl) | 0, 0 | | 0, 0 | |

Two earlier gates were red and fixed before this one: 12ab92d0 (jobs 10732–10735) on a test of mine (the Float32 static-LM
case built a Float64 residual) and on the SplineLens optimization-level item, whose `M.stored` read in the item's world is
an `UndefVarError` under `--depwarn=error` on Julia ≥ 1.12 — main c39b4a2d's strict Core suite fails it too (job 10744);
94a303ff reads the Core.eval'd functions inside `invokelatest`. 5377c3b5 (job 10729) had two stale DUPLICATION.toml
entries (removed in 12ab92d0) and my test syntax.

---

docs-pass: f6a3e7f2
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 (no change: it names neither the SplineLens refine nor the forward fit)

# 0.10.0 docs pass (julia-docs-pass) — release-chrome-095 (VisualGeometryToolkitEstimation 0.10.0, Features 0.8.2, Eval 0.7.1)

The packages released: VisualGeometryToolkitEstimation 0.10.0 (breaking), VisualGeometryToolkitFeatures 0.8.2 and
VisualGeometryToolkitEval 0.7.1 (compat-only patches). Core, IO and PoseLib are unchanged. The baseline is the
Estimation 0.9.2 release (edab2655, below).

## 1. Public API since 0.9.2

No name added or removed. Changed behaviour of the exported `refine_camera_and_poses` on a SplineLens camera: it takes
`fixed = (:controls,)` only; the default `fixed = ()` (the image-space refine of the controls) and a `penalty` are
refused by name. `CameraPosesRefineMetadata.spline` loses the retired route's fields (CHANGELOG 0.10.0).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `refine_camera_and_poses`: its SplineLens signature line (`fixed = (:controls,)`, required), the `camera` argument, the `fixed` keyword and the whole "# A SplineLens camera" section rewritten for the held route; `CameraPosesRefineMetadata`'s `spline` and `free` for a SplineLens camera. api/radial.md: the camera-and-poses paragraph (the held route, the refusal, the retired `free = ()` spelling replaced by `fixed`) and the GCV paragraph (no longer names the SplineLens refine). configs/default.toml's `forward_tolerance` comment names `fit_forward_spline`. |
| stale mentions | the retired route's metadata fields, `penalty = :gcv` and the window/rebuild/GCV prose: none left outside CHANGELOGs and the plan notes under docs/superpowers (historical plans, not documentation). Core's spline_lens.jl comment that `_projection_jacobians` serves "a SplineLens camera's bundle refinement" is Core's (unchanged tree; Core is not released here) — noted for Core's next release. |
| CHANGELOG | Estimation 0.10.0 (Breaking, Changed), Features 0.8.2, Eval 0.7.1 |
| versions | Estimation 0.10.0, Features 0.8.2, Eval 0.7.1 |
| in-repo compat | Features (weak dependency) and Eval, and their test environments: Estimation "0.10" |
| README / landing | no README or landing text names the SplineLens refine's route: unchanged |

## 3. Executed on the final RC

The first 0.10.0 gate (66900bd9; jobs 10693–10696) was green on both Julia versions for every gated item; Features' strict
run (recorded, not gated there) errored one item, the L0 smoothing, on a third-party deprecation: ImageSmooth 0.1.1, its
latest release (2021, no newer upstream commit), calls ImageBase's deprecated `fdiff!` inside `L0Smooth`; Features calls
only ImageSmooth's public `smooth!`. f6a3e7f2 runs that one item's check in a child Julia with `--depwarn=yes` when the
suite is strict (the coordinator's option 2, the comment at the site names the cause); every other item stays strict, and
Features' strict run gates. No docs change.

The final gate on a0016dbb (jobs 10706–10709): two fresh clones per Julia version, no Manifest, General cloned from
GitHub (the package server bypassed), CauRegistry from the shared depot. Counts are pass / fail / error / broken.

| | 1.13.3 GPU (10706) | 1.13.3 CPU (10707) | 1.12.7 GPU (10708) | 1.12.7 CPU (10709) |
|---|---|---|---|---|
| registration_lint --subsequent | 0 errors, 0 warnings | | 0 errors, 0 warnings | |
| floor_compat --latest | rc 0 | | rc 0 | |
| check_duplication(Estimation) | 0 findings | | 0 findings | |
| Estimation (--depwarn=error) | 8817 / 0 / 0 / 2 | 8807 / 0 / 0 / 2 | 8805 / 0 / 0 / 2 | 8795 / 0 / 0 / 2 |
| Features (--depwarn=error) | 3071 / 0 / 0 / 2 | | 3071 / 0 / 0 / 2 | |
| Eval (--depwarn=error) | 2907 / 0 / 0 / 0 | | 2907 / 0 / 0 / 0 | |
| Estimation docs (strict make.jl) | 0 | | 0 | |

Estimation's count fell from 20979 (0.9.2) to 8817 with the retired route's items (174 → 171 items: the window
blocks' finite-difference Jacobian item, the smoothness/GCV item, the free-route refine items and the opt-in benchmark
item removed or converted; two refusal / held-camera items added).

---

# 0.9.2 docs pass (julia-docs-pass) — release-chrome-095 (VisualGeometryToolkitEstimation 0.9.2)

The package released: VisualGeometryToolkitEstimation 0.9.2 alone, a patch. The others are unchanged (Core's tree is
its registered 0.9.4's). The baseline is VisualGeometryToolkitEstimation-v0.9.1 (on main 540c2591 its tree differs from
the tag by the CHANGELOG heading only).

## 1. Public API since VisualGeometryToolkitEstimation-v0.9.1

One public name added, not exported: `fit_forward_spline(camera::BoundedCamera; tolerance, region, device)`
(fitting/fit_camera.jl; docstring on api/radial.md in its `@docs` block, beside `fit_camera`, and a paragraph of that
page's prose). Nothing removed, renamed or re-signatured. Changed behaviour, no API change: a held SplineLens camera is
projected whole (`refine_camera_and_poses` with `fixed = (:controls,)`).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `fit_forward_spline`: a docstring (signature, what is kept, how it differs from `fit_camera(SplineLens, …)`, the result, the `ArgumentError`), on api/radial.md. `refine_camera_and_poses`: its "A SplineLens camera" section said the forward is refitted by `fit_camera(SplineLens, …)`; it now names `fit_forward_spline` (72085327). Its held-camera paragraph (no window, the whole camera through `project`) came with the fix. |
| stale mentions | `fit_camera(SplineLens, …)` as a held camera's forward: none left in the monorepo. PanoramaStitching's rotation_bundle.jl docstring and sampling.jl comment named it: fixed on PS's ps-0.6.2. |
| CHANGELOG | an Estimation 0.9.2 section above 0.9.1 |
| versions | Estimation 0.9.2; no other package moves |
| in-repo compat | none needs moving: Features (weak), Eval and their test environments declare Estimation "0.9" |
| README / landing | no README or landing text names the refine's SplineLens route or the forward fit: unchanged |

## 3. Executed on the final RC

The first gate (61d44aa3; jobs 10673–10676, Julia 1.13.3 and 1.12.7) failed one Estimation item on both versions under
`--depwarn=error`: `ransac_weights`' `@test_throws ErrorException r.sigma` (a missing field is a `FieldError` since
Julia 1.12; Test deprecates asserting `ErrorException` for it). d06f1098 asserts `FieldError`; no docs change. The
final gate's results are recorded by its records commit.

---

# 0.9.4 docs pass (julia-docs-pass) — release-0.9.4

The package released: VisualGeometryToolkitCore 0.9.4 alone, a patch (RadiometricCamera 0.8.0 waits for it). The others
are unchanged. The baseline is 1c7581ec (release-0.9.3's head; 8ae2ea67 plus a test-only fix); the earlier passes are below.

## 1. Public API since 1c7581ec

Core adds the one public name `is_inside_texel_centres` (transforms/warp.jl, docstring on api/warping.md through its
`@autodocs`): whether a position lies in the closed rectangle of the texel centres of `bilinear_sample`'s half-texel
chart, exactly where `bilinear_weights` does not extend the edge. And one method of the exported `raw_frame_map`,
`raw_frame_map(::Type{LinearRaw{mode}})` (raw/linear_raw.jl, on api/raw.md through its `@autodocs`): a cell mode's frame
map without an image. And one method of the public `raw_provenance`, `raw_provenance(::Type{LinearRaw{mode}})`
(raw/linear_raw.jl, on api/raw.md): the `(; mode, frame_map)` an image's provenance holds, by mode. No new exported name; nothing removed or renamed in any package. `bilinear_weights` and the cell
images' maps and records are unchanged bit for bit (they now read the private chart `_texel_chart`/`_clamp_to_centres`, the
mode method and `_frame_map_record` respectively).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `is_inside_texel_centres`: a docstring, on api/warping.md, and a line in that page's prose beside `bilinear_weights`. `raw_frame_map(::Type{LinearRaw{mode}})`: a docstring, on api/raw.md, a line in that page's intro and a row in the guide's raw_frame_map table (guide/raw-images.md). `raw_provenance(::Type{LinearRaw{mode}})`: a docstring, on api/raw.md, and the same intro line |
| CHANGELOG | a Core 0.9.4 section above 0.9.3 |
| versions | Core 0.9.4; no other package moves |
| in-repo compat | none needs moving: every Core compat in the repo ("0.9", "0.9.2", "0.9.3") admits 0.9.4, and no in-repo package uses either addition |

## 3. Executed on the final RC

The final RC is 4055b67b (0.9.4 on b7ad736e, the pushed 0.9.3). The gate ran on two fresh clones, one per Julia version: a
`git clone` with no Manifest, its workspace resolved from scratch. Counts are pass / fail / error / broken.

| Package | 1.13.3 GPU (10623) | 1.13.3 CPU (10624) | 1.12.7 GPU (10632) | 1.12.7 CPU (10633) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| VisualGeometryToolkitCore | 411320 / 0 / 0 / 2 | 409742 / 0 / 0 / 30 | 411317 / 0 / 0 / 2 | 409739 / 0 / 0 / 30 | 0 / 0 (10625, 10634) |
| VisualGeometryToolkitIO | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 0 / 0 (10626, 10635) |
| VisualGeometryToolkitEstimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | 0 / 0 |
| VisualGeometryToolkitFeatures | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 0 / 0 |
| VisualGeometryToolkitEval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 0 / 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 0 / 0 |

No failures or errors anywhere. The previous RC 2133c287 (the provenance method with an untyped map argument) failed Core's JET
gate on two of its four runs (13 reports against the bound 12: `translation(S::Any)`, union-split onto `Disc`/`OrientedDisc`);
4055b67b types the argument. Earlier RCs: eff6ff34 (before the provenance method), all green.

---|---|---|---|---|---|
| VisualGeometryToolkitCore | 410869 / 0 / 0 / 2 | 409291 / 0 / 0 / 30 | 410866 / 0 / 0 / 2 | 409288 / 0 / 0 / 30 | 0 / 0 (10513, 10517) |
| VisualGeometryToolkitIO | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 0 / 0 (10514, 10518) |
| VisualGeometryToolkitEstimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | 0 / 0 |
| VisualGeometryToolkitFeatures | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 0 / 0 |
| VisualGeometryToolkitEval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 0 / 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 0 / 0 |

No failures or errors anywhere. The 0.9.4 delta's own Core suites on the worktree, on the pre-rebase base 9762b9b4 (CPU 10425/10429,
GPU 10426/10430), failed only the base's `require_extension` world-age item that 1c7581ec fixes; the new items passed on all four.

---

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

The final RC is 1c7581ec. The gate ran on two fresh clones, one per Julia version: a `git clone` with no Manifest, its
workspace resolved from scratch. Counts are pass / fail / error / broken.

| Package | 1.13.3 GPU (10494) | 1.13.3 CPU (10495) | 1.12.7 GPU (10498) | 1.12.7 CPU (10499) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| Core 0.9.3 | 396965 / 0 / 0 / 2 | 395387 / 0 / 0 / 30 | 396962 / 0 / 0 / 2 | 395384 / 0 / 0 / 30 | exit 0 / exit 0 |
| IO 0.9.2 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | exit 0 / exit 0 |
| Estimation 0.9.1 | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | exit 0 / exit 0 |
| Features 0.8.1 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | exit 0 / exit 0 |
| Eval 0.7.0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | exit 0 / exit 0 |
| PoseLib 0.2.0 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | exit 0 / exit 0 |

There are no failures or errors anywhere, the README items included.

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
