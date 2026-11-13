docs-pass: 94a303ff
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
