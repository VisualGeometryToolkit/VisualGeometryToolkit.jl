# SplineLens Fold-in Implementation Plan (plan 1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the central grid camera an ordinary `CameraModel(K, SplineLens(...))`. Delete `CentralGenericModel`, `SplineProjection` and `nominal_intrinsics` along with their roughly 70 parallel methods. Rebuild `NoncentralSplineModel` on the same `ChartSpline` machinery. Port IO and Estimation, so that Core/IO/Estimation 0.9 ships one grid implementation.

**Architecture:** `SplineLens` is a backward lens, like `LiftLens`. It works in K-normalized coordinates x = K⁻¹u. Its `backward::ChartSpline{3}` maps x to a direction. Its optional `forward::ChartSpline{2}` maps the equidistant chart about the lens axis to x. It has a presence mask. K is a gauge: skew = 0, frozen in every fit, and chosen so that pp is the pixel imaging ẑ. Every pixel-unit constant of the old pixel-space LM is rescaled by (fx, fy), so the new code reproduces the old outputs. Golden fixtures generated at BASE prove this.

**Tech Stack:** Julia 1.12+, VisualGeometryToolkit monorepo (Core, IO, Estimation), ReTestItems, kaimon `run_tests`.

**Spec:** the design relayed and agreed in session 60680b91 (2026-11-09). Its summary is the "SplineLens design" paragraph of the ledger `.superpowers/sdd/2026-11-09-bounded-camera-region/progress.md`. Surveys of the Core internals, the CameraModel audit and the IO/Estimation/RC/PS usage were taken at `e33764a9`. Where this plan and that design disagree, the design wins.

## Global Constraints

- **No duplication (owner rule, extreme vigilance).** One implementation per concept. A refactor deletes special-case copies instead of porting them. No aliases, no forwarding wrappers, no shims. Every task ends with a duplication gate (below).
- **Retired names** go into `RETIRED_NAMES` in `VisualGeometryToolkitCore/ext/VisualGeometryToolkitCoreTestExt/check_naming.jl`, each with a comment naming its replacement: `CentralGenericModel`, `NoncentralGenericModel`, `SplineProjection`, `nominal_intrinsics`, `with_nominal`, `with_domain`, `generic_lattice`, `GenericModelWindow`, `GenericGridSpec`, `central_generic`.
- **New public names, exactly these:** `SplineLens`, `NoncentralSplineModel`, `spline_lattice`, `SplineLensSpec`, and `CameraModelSpec(:spline; …)`. `window`, `node_mask`, `free_nodes`, `present_nodes`, `with_present`, `lattice_spacing`, `spline_value`, `spline_window` and `ChartSpline` keep their names.
- **skew = 0** is required for every SplineLens camera. Constructors and readers throw `ArgumentError` otherwise.
- **K is frozen in every fit** of a SplineLens camera: the gauge. Parameters are the lens controls only.
- **User-facing settings stay in pixels:** `grid_size`, `calib_area`, `tolerance`, `region`, and the stored records' `calibration_min_*`.
- **Monorepo commits carry NO AI attribution:** no Co-Authored-By line.
- The naming check R7 bans the word "pass" (and "passes", "passed", "passing") in code, comments and docs.
- No version bumps. The branch already carries Core/IO 0.9.0. Estimation's 0.9.0 bump comes with the release.
- Nothing is pushed.

**Duplication gate (every task, before its commit).** Grep the monorepo for parallel implementations of what the task touched:
- the same algorithm under another name;
- a pixel-space copy next to a normalized-space one;
- the old grid's helper kept alongside the new lens's;
- validators or JSON helpers copied between files.

Consolidate in the same task. The report names what was grepped and what was deleted.

## Review Focus

1. **A grid whose ẑ ray falls outside its calibrated area** (an off-axis or cropped calibration). The gauge falls back to the calibrated-rect centre, and loading never throws. Pinned in Task 1.
2. **An old central record with an elliptical `domain`.** Loaded as a `BoundedCamera`, the domain becomes `lens_coverage`. Loaded as a bare `CameraModel`, it throws `ArgumentError` naming `BoundedCamera` and never drops the domain silently. Pinned in Task 6.
3. **`project` with a forward spline, on a ray outside the forward's fitted chart.** It returns NaN, not an extrapolation. Inside, it agrees with the LM project within the fit tolerance. Pinned in Task 2.
4. **Re-gauging or reframing a SplineLens camera** (`CameraModel(m, K′)`, IO pixel_frame colmap↔opencv, a crop). `backproject` per pixel is unchanged to 1e-12. Pinned in Tasks 1 and 6.
5. **Float32 conversion and device adaptation** (`convert_eltype`, `adapt`) of a SplineLens camera. Batched `backproject`/`project` on the adapted camera equals the CPU scalar result to 1e-5. Pinned in Task 2.

---

### Task 0: Golden fixtures from the old grid (before anything is deleted)

**Files:**
- Create: `VisualGeometryToolkitCore/test/fixtures/spline_lens_golden.jl` (generated, literal arrays)
- Create: `VisualGeometryToolkitCore/test/fixtures/make_spline_lens_golden.jl` (the generator, kept for the record)

**Interfaces:**
- Produces: `const GOLDEN = (; central, noncentral, ...)`, a NamedTuple of literal arrays read by the Task 1–4 equivalence tests via `include`.

- [ ] **Step 1: Write the generator.**
  - It builds the same models the Core tests use: `CentralGenericModel(KB_FISHEYE, (1000, 1000); grid_size = (24, 24))` with KB coefficients `(0.2468, 0.1505, -0.0214, 0.1308)`, f = 300 px, pp = (500, 500), plus a masked copy via `with_present` (drop the 3 outermost node rings).
  - It records, at a fixed 17×17 pixel lattice over the calibrated area plus 8 pixels outside it:
    - `backproject` (NaN kept);
    - `backproject_with_jacobian`;
    - `project` of the backprojected rays, and of 8 rays outside the domain;
    - batched `project`/`backproject`;
    - `undistort` of a 64×64 checker image with `Kr` = nominal and a 48×48 rect.
  - For the noncentral copy (directions from the central model, points `0.01·d` m), it records `backproject` rays and `project` pixels.
  - It also loads the stored IO fixtures `VisualGeometryToolkitIO/test/fixtures/old/generic_camera_v2.json`, `.../units/generic_camera.json` and `.../units/generic_camera_masked.json` with the old IO reader, and records `backproject` on the same lattice. Task 6 checks its migrations against these values.
  - For a `SplineProjection` fitted by Estimation's `fit_camera(SplineProjection, grid, Rect(0,0,1000,1000); tolerance = 0.01)`, it records the control grid and area and `project` at 64 rays.
  - Floats are printed with `repr` so they round-trip exactly.
- [ ] **Step 2: Run it at the current HEAD (old code) once:** `ex(e = "include(\"…/make_spline_lens_golden.jl\")")` in the Core session. Expected: the fixture file is written. It is ≤ 300 KB.
- [ ] **Step 3: Commit:** `git commit -m "VisualGeometryToolkitCore: golden outputs of the central and noncentral grid models, before the SplineLens fold-in"`.

### Task 1: `SplineLens` (backward), its gauge, and `CameraModel(SplineLens, source, size)`

**Files:**
- Create: `VisualGeometryToolkitCore/src/geometry/cameras/spline_lens.jl` (the lens: struct, backward evaluation, presence, valid domain, gauge)
- Modify: `VisualGeometryToolkitCore/src/geometry/cameras/chart_spline.jl` (keep `ChartSpline`; the `SplineProjection` section is deleted in Task 4)
- Modify: `.../camera_models.jl:676-680` (`_rescale_lens` method for SplineLens)
- Modify: `.../cameras.jl` (include, exports)
- Test: `VisualGeometryToolkitCore/test/spline_lens_tests.jl`

**Interfaces:**
- Produces:
  - `SplineLens{B<:ChartSpline{3}, F<:Union{Nothing, ChartSpline{2}}, P<:Union{Nothing, AbstractMatrix{Bool}}} <: AbstractLensModel`, with fields `backward::B`, `forward::F`, `present::P`.
  - The constructor `SplineLens(backward; forward = nothing, present = nothing)`. An all-true `present` is stored as `nothing`; an all-false one throws (the `_present_mask` rule, moved here, not copied). `present` must have `size(backward.controls)`.
  - `backproject(lens::SplineLens, x::SVec2) -> SVec3`: a unit vector, NaN outside the backward area's half-open box or in an absent window. Never throws.
  - `CameraModel(::Type{SplineLens}, source, size::NTuple{2,Int}; grid_size = default_generic_grid(size), calib_area = (0, 0, W-1, H-1), init = :interpolate, sampling_margin = 0.97, unit_controls = init !== :quasi_interpolate) -> CameraModel{LogicalIntrinsics, <:SplineLens}`. This replaces `CentralGenericModel(source, size; …)`. Same `init` modes and helpers, re-hosted, not copied.
  - `spline_lattice(::Type{SplineLens}, source, size; grid_size, calib_area)`: replaces `generic_lattice`.
  - `_spline_gauge(pixel_direction, rect::Rect2) -> LogicalIntrinsics`: the ONE gauge derivation, used by the source constructor (Function sources), by IO migration (Task 6) and by Estimation (Task 7). A `CameraModel` source keeps its own K (skew ≠ 0 → ArgumentError).
  - `valid_domain(::CameraModel{<:Any, <:SplineLens})`: the backward area mapped through K, ∩ the bounding Rect of present nodes' support. It returns a `Rect2{Float64}` (concrete).
  - `_rescale_lens(::SplineLens, sx, sy)`: remaps both splines' areas, and the forward's output, by the diagonal change of x. Exact.

- [ ] **Step 1: Write the failing tests** in `spline_lens_tests.jl` (`@testitem "SplineLens: backward lens equals the old central grid"`, with `include` of the golden fixture):
  - `cam = CameraModel(SplineLens, KB_FISHEYE, (1000, 1000); grid_size = (24, 24))`.
  - `@test backproject(cam, u) ≈ GOLDEN.central.backproject[i] atol = 1e-12` at every golden pixel, and `isnan` exactly where the golden value is NaN.
  - The same for the masked copy (`with_present`).
  - `@test calibration_matrix(cam) == calibration_matrix(KB_FISHEYE)` (a CameraModel source keeps its K).
  - `@test_throws ArgumentError CameraModel(SplineLens, skewed_cam, (1000, 1000))`.
  - **Gauge from a Function source** (owner-agreed 2026-11-09, tolerances MEASURED at e33764a9). Set-up: the 4256×2848 KB fisheye `CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))`, sampled as a pixel→ray Function. The pp is the projection of ẑ; `fx = 1/J[1,1]` and `fy = 1/J[2,2]` of `J = ∂(d_xy/d_z)/∂u` at pp.
    - Measured on the old grid: 24×16 gives a pp error of 9.4e-5 px and an f error of 7.6e-5 relative; the default 70×48 gives 9.1e-7 px and 5.5e-7; 105×75 gives 1.7e-7 px and 5.8e-8. The off-diagonal of J is ≤ 2e-11·f.
    - Test, at a 10× margin: 24×16 gives pp ≤ 1e-3 px and f ≤ 1e-3 relative; 70×48 gives pp ≤ 1e-5 px and f ≤ 1e-5 relative.
  - **Review Focus 1:** a Function source whose ẑ lies outside `calib_area = (600, 600, 999, 999)` gives pp = the calibrated-rect centre and does not throw.
  - **Review Focus 4:** `backproject(CameraModel(cam, K′), u′) ≈ backproject(cam, u)` to 1e-12 for `K′` = K scaled by (0.5, 0.5) with pp scaled, `u′` the scaled pixel.
  - `@inferred backproject(cam, u)` and `@inferred project(cam, X)` hold for a camera with and without `forward`/`present`.
  - `@inferred valid_domain(cam)` returns a `Rect2{Float64}` equal to the old `_calibrated_rect` when there is no mask.
- [ ] **Step 2: Run** `run_tests(pattern = "SplineLens: backward", session = core)`. Expected: FAIL, `SplineLens not defined`.
- [ ] **Step 3: Implement.**
  - **Coordinate change.** The backward area is `K⁻¹` of the pixel half-open rect `[min, max+1)`. The grid coordinate `1 + (g−3)/(max+1−min)·(x−min)` becomes the same affine map in x, so uniform knots map exactly.
  - **Moved, not copied, into `spline_lens.jl`:** `_spline_window`, `_window_present`, `_eval_window`, `_unit_or_normalize`, `_present_mask`, `_check_present`, and the init helpers (`default_generic_grid`, `_prefilter_matrix`, `_sampling_radius`, `_directions_from`, `quasi_interpolant`, `_source_direction`, `_quasi_interpolation_directions`, `_init_directions`). Their old homes are deleted in Task 4. Until then, the old grid calls the moved helpers.
- [ ] **Step 4: Run the same pattern.** Expected: PASS. Then run the full Core suite; it must stay green, since the old types still exist.
- [ ] **Step 5: Duplication gate, then commit:** `"VisualGeometryToolkitCore: SplineLens, the central grid as a CameraModel lens (backward)"`.

### Task 2: `project` through a SplineLens (LM in normalized coordinates; forward spline), batched and device forms

**Files:**
- Modify: `spline_lens.jl`, `chart_spline.jl` (`_in_fitted_chart` stays and serves the forward)
- Test: `spline_lens_tests.jl` (new item `"SplineLens: project"`)

**Interfaces:**
- Consumes: Task 1's `SplineLens` and helpers.
- Produces:
  - `project(lens::SplineLens, X::SVec3) -> SVec2`, NaN where it does not image. When `forward !== nothing`: `c = equidistant_chart(_lens_axis(lens))(X)`; NaN unless `_in_fitted_chart(forward, c)`; then `x = forward(c)`, NaN unless x is in the presence-masked backward box. Otherwise LM.
  - `_lens_axis(lens) = normalize(backward(0, 0))`: the ray of pp. Computed, not stored.
  - `_spline_lm(residual, x₀, lb, ub, s)`: the ONE LM, the old `_pixel_lm` with its constants rescaled by `s = (fx, fy)`. The noncentral model uses it too (Task 3). The rescaled constants: `ϵx = 1e-10 ./ s`; bounds `lb = area.origin`, `ub = area.origin + area.widths − 0.001 ./ s` (= pixel `max+0.999`); DtD floor `1e-12 .* s.^2`, cap `1e16 .* s.^2`. λ schedule unchanged: ×1/13, ×3, clamp `[1e-28, 1e32]`, start 1e-6. `_PROJECT_MAXITER = 100`, `_PROJECT_TOL = 1e-6` unchanged. Start at the area centre.
  - Batched `project(cam, rays, p₀s; iters = 10, tol)` and `backproject(cam, pixels)` on `CameraModel{<:Any, <:SplineLens}`. The backward evaluation is `ChartSpline`'s batched kernel (chart_spline.jl:75–88) after K⁻¹, NOT a second kernel.
  - `Adapt.adapt_structure(to, ::SplineLens)` and `convert_eltype(T, ::SplineLens)`.

- [ ] **Step 1: Write the failing tests:**
  - **Without a forward:** `project(cam, X) ≈ GOLDEN.central.project[i]` to 1e-9 px. NaN where golden is NaN.
  - **With a forward:** the golden SplineProjection's controls become `forward` (same chart: `_lens_axis` equals the golden chart's axis to 1e-12). `project(cam_fwd, X) ≈ GOLDEN.splineprojection.project[i]` to 1e-9 px.
  - **Review Focus 3:**
    - `isnan(project(cam_fwd, X_out))` for a ray 2 knot cells beyond the forward's chart;
    - `norm(project(cam_fwd, X) - project(cam, X)) ≤ 0.01` px on the 64 golden rays;
    - batched forms equal the golden batched outputs.
  - **Review Focus 5:** `c32 = convert_eltype(Float32, cam)`; batched `backproject(adapt(Array, c32), pixels)` ≈ scalar `backproject(cam, ·)` to 1e-5.
- [ ] **Step 2: Run** `pattern = "SplineLens: project"`. Expected: FAIL (MethodError `project(::SplineLens, …)`).
- [ ] **Step 3: Implement as in Interfaces.** `CameraModel`'s generic `project` (camera_models.jl:278) wraps the lens; no camera-level project method is added.
- [ ] **Step 4: Run.** Expected: PASS. Run the full Core suite: green.
- [ ] **Step 5: Duplication gate.** `_pixel_lm` and the old batched kernels now have a replacement; they are deleted in Task 4, and the gate lists them as pending there. Commit: `"VisualGeometryToolkitCore: project through a SplineLens: one LM in normalized coordinates, the forward spline on the lens axis"`.

### Task 3: `NoncentralSplineModel` on `ChartSpline`

**Files:**
- Modify: `generic_models.jl`, renamed by `git mv` to `noncentral_spline_model.jl` (what is left of it after Task 4)
- Modify: `cameras.jl:679-690`, `oriented_camera.jl:45-46`, `support.jl:126-133`
- Test: `VisualGeometryToolkitCore/test/noncentral_spline_model_tests.jl` (`git mv` the noncentral parts of `generic_models_tests.jl`)

**Interfaces:**
- Produces: `NoncentralSplineModel{I<:LogicalIntrinsics, S<:ChartSpline{3}, P} <: AbstractCamera`.
  - Fields: `intrinsics::I`, `directions::S`, `points::S` (plain metres), `present::P`.
  - No `width`/`height`/`calib_*`: a stored or rig camera carries its extent in its `BoundedCamera`.
  - Constructors: `NoncentralSplineModel(K, directions, points; present = nothing)`, and `NoncentralSplineModel(source, size; grid_size, calib_area)` (directions from the central constructor, points zero).
  - Methods: `backproject` (a `Ray3`), `backproject_with_jacobian`, `project` (via `_spline_lm`, tangent-plane residual), `central_model(m) -> CameraModel{…, SplineLens}`, `with_points`, `with_directions`, `with_present`, `PosedCamera`, `ustrip`, `with_unit`, support hooks.
  - Direction evaluation and windows are SplineLens's helpers (Task 1), shared, not copied.
- [ ] **Step 1: Write the failing tests** against `GOLDEN.noncentral`:
  - `backproject` ray origin and direction to 1e-12;
  - `project` to 1e-9 px;
  - `central_model(m)` backprojects as the golden central;
  - `@test_throws ArgumentError` for skew ≠ 0.
- [ ] **Step 2: Run.** Expected: FAIL (`NoncentralSplineModel not defined`).
- [ ] **Step 3: Implement.** The noncentral residual uses ForwardDiff as before.
- [ ] **Step 4: Run.** Expected: PASS. Full Core suite green.
- [ ] **Step 5: Duplication gate, then commit:** `"VisualGeometryToolkitCore: NoncentralSplineModel, the non-central grid on ChartSpline in normalized coordinates"`.

### Task 4: Delete the old grid; port Core's remaining grid users and tests

**Files:**
- Delete:
  - all `CentralGenericModel`/`NoncentralGenericModel`/`_GenericModel`/`_GenericLattice`/`GenericModelWindow` code in `generic_models.jl` (including `_pixel_lm`, the batched kernels, `nominal_intrinsics`, `with_nominal`, `with_domain`, `_default_nominal`, `_default_domain`, `undistort(::Central …)`, `_undistort_remap`, `_remap_sample`, `_march_node!`, `_rectified_view(::Central …)`, the adapt/convert_eltype copies);
  - the `SplineProjection` section of `chart_spline.jl`.
- Modify:
  - `window`: now `window(cam::CameraModel{<:Any,<:SplineLens}, rows, cols)` returning the same camera type over a sub-`ChartSpline` (one window type: the camera);
  - `lens_spec.jl:302-449` (`GenericGridSpec` → `SplineLensSpec`, `:central_generic` → `:spline`; the TOML reader refuses `"central_generic"` with an ArgumentError naming `"spline"`);
  - `precompile.jl`, `interface.jl`, `bounded_map.jl` comments, exports in `VisualGeometryToolkitCore.jl` and `cameras.jl`, `check_naming.jl` `RETIRED_NAMES` (the list in Global Constraints);
  - docs `guide/lenses.md`, `api/generic_models.md` → `api/spline_lens.md`, `guide/estimators.md`, `conventions/units.md`.
- Test: port `generic_models_tests.jl`, `generic_model_window_tests.jl`, `generic_backproject_jacobian_tests.jl`, `generic_present_nodes_tests.jl`, `chart_spline_tests.jl`, `support_tests.jl`, `lens_chart_tests.jl`, `downstream_hoists_tests.jl`, `input_guards_tests.jl`, `equirectangular_tests.jl`, `stereo_rectify_tests.jl`, `rig_tests.jl`, and the single-line users (`cameras_tests`, `view_overlap_tests`, `units_boundaries_tests`, `bounded_camera_region_tests`, `lens_contract_tests`). Rename the files with `git mv` to `spline_*`.

- [ ] **Step 1: Write the failing test** `@testitem "SplineLens: undistort and rectify through the generic camera path"`:
  - `undistort(cam, checker, Kr, rect) ≈ GOLDEN.central.undistort` (max abs difference ≤ 1e-6) through the GENERIC `CameraModel` undistort;
  - the stereo-rectify test on two SplineLens cameras.
- [ ] **Step 2: Run.** Expected: FAIL, or the generic path is missing a method. Note which.
- [ ] **Step 3: Delete and port.** If the generic undistort is measurably slower than `_march_node!` (> 3× on the 1000² image), keep the speed as a method of the GENERIC undistort, not as a grid special case. Ledger it as a ruling.
- [ ] **Step 3b: KernelAbstractions (owner, 2026-11-10).** The generic undistort/rectify of a SplineLens camera runs through Core's KernelAbstractions warp (`image_warp` → `_warp_kernel!` on `get_backend(dst)`). The lens's `project`, with and without a forward spline, must therefore be device-safe inside that kernel: no allocation, no throw, isbits after `adapt`.
  - Test it on the KernelAbstractions CPU backend: the result `==` the plain CPU result.
  - Test it on CUDA when available (skip otherwise): ≤ 1e-4 relative.
  - The grid's own KA kernels (`_row_march_kernel!`, `_remap_dense_kernel!`, the old `_project_kernel!`/`_backproject_kernel!`) are DELETED. There is one KA warp path for every camera.
- [ ] **Step 4: Run** the full Core suite plus `Aqua`/naming check (`run_tests` Core, all items). Expected: green. `grep -rn "CentralGenericModel\|NoncentralGenericModel\|SplineProjection\|nominal_intrinsics\|GenericModelWindow\|generic_lattice" VisualGeometryToolkitCore` returns CHANGELOG lines only.
- [ ] **Step 5: Duplication gate.** Confirm there is exactly one LM, one window-evaluation path, one presence-mask rule and one batched kernel. Commit: `"VisualGeometryToolkitCore: the central grid is a CameraModel; CentralGenericModel, SplineProjection and nominal_intrinsics are gone"`.

### Task 5: K-as-physics audit (the CameraModel methods that read K)

**Files:**
- Modify: `lens_chart.jl:135-247, 529-582`; `camera_models.jl` (rows below); `utilities.jl`, `image_circle.jl`, `view_overlap.jl`, `calibration.jl`, `homography.jl`, `rectify.jl`; `src/plotting/rectification.jl`; Estimation `fitting/refine_camera_and_poses.jl:194`.
- Test: `VisualGeometryToolkitCore/test/spline_lens_gauge_tests.jl`; one Estimation item in `refine_camera_and_poses_tests.jl`.

**Interfaces:**
- Produces: the trait `_intrinsics_free(lens)::Bool`, true by default and false for `SplineLens`. The ONE switch that `_bound_kinds`, `parameters`, `parameter_names`, `parameter_vector`, `parameter_bounds`, `with_parameters`, `minimal_manifold`, `ambient_coordinates`, `_parameter_jacobian` and `refine_camera_and_poses`' default `free` read. There is no SplineLens copy of any of them.
- Rulings per bucket-B method, for a SplineLens camera:
  - **Keep, docstring updated:** K is the gauge pinhole at the optical-axis pixel. These methods have a meaning because pp images ẑ.
    - `calibration_matrix`, `intrinsics`, `_principal_point`, `pinhole`, `distortion_map`, `undistort`/`undistort!` (all 6), `_frame_walk`, `valid_rect`;
    - `angle_of_view` (utilities.jl:118), `image_circle(::CameraModel, ::Sensor, d)`, `_footprint`, `rectify`'s default K, `_induced_plane_homography`, the plotting panels.
  - **Unchanged; these work as is:** `fov_radius` (it reads only f), `undistorted_rect`'s `max_fov`, `_default_max_fov` (a constant). The audit misfiled them.
  - **Horizon from the grid (owner, 2026-11-09):** `monotone_radius(::SplineLens)` is the largest normalized radius about x = 0 whose disc lies inside the backward area ∩ the present nodes' support (where the grid is defined and one-to-one). `horizon_radius = f·monotone_radius` then needs no change. This matches the parametric lenses, whose `monotone_radius` reads their first radial piece and documents that decentering terms do not enter.
  - **`RadialCamera(::PosedCamera{<:CameraModel})` refuses (owner, 2026-11-09)** a SplineLens AND any `Lens` whose distortion holds a decentering piece (`Tangential`, `ThinPrism`, `Tilt`, alone or inside an `Additive`/`∘` composition). It throws `ArgumentError`, because the radial camera is exact only for a radially symmetric lens. One type-level check over the distortion pieces, using the existing stage walk (`_stages`), not a new one. This closes a gap: such lenses used to be silently accepted. Breaking; note it in the CHANGELOG.
  - **Re-gauge:** `CameraModel(::CameraModel, K′)` uses Task 1's `_rescale_lens`. Done there.
  - **Unreachable** (Division/Pinhole/Projective-only): no change.
  - `_default_nominal` was deleted in Task 4.
- [ ] **Step 1: Write the failing tests:**
  - `parameter_names(cam)` lists only lens controls (no `:f`, `:cx`);
  - `with_parameters(cam, θ)` keeps `calibration_matrix` identical;
  - `refine_camera_and_poses(cam, poses, views)` leaves K bit-identical and lowers the reprojection cost on a synthetic 3-view fixture;
  - `horizon_radius(cam)` is finite and within 1% of the KB source's `horizon_radius` clipped to the grid's inscribed radius; `fov_radius(cam, θ) == f·tan θ`;
  - `@test_throws ArgumentError RadialCamera(PosedCamera(cam, X))`;
  - `@test_throws ArgumentError RadialCamera(PosedCamera(brown_conrady_with_tangential, X))` and the same for a KB `Additive(Radial, Tangential)`. `RadialCamera(PosedCamera(KB_FISHEYE, X))` still works;
  - `angle_of_view(BoundedCamera(cam, Size2(1000, 1000)))` is within 0.5° of the KB source's.
- [ ] **Step 2: Run.** Expected: FAIL (K parameters present; `horizon_radius` MethodError, not ArgumentError).
- [ ] **Step 3: Implement the trait and the refusals.**
- [ ] **Step 4: Run** the Core and Estimation suites. Expected: green.
- [ ] **Step 5: Duplication gate, then commit:** `"VisualGeometryToolkitCore, VisualGeometryToolkitEstimation: a SplineLens camera's K is a frozen gauge; radial-only methods refuse it"`.

### Task 6: IO — the SplineLens record, generic_camera v4 (non-central only), migrations, rig, Schöps, exporters

**Files:**
- Modify:
  - `VisualGeometryToolkitIO/src/camera_schema.jl` (`_lens_json`/`_lens_from_json` gain kind `"spline"`; cameras stays v5 because v5 is unreleased);
  - `generic_camera_schema.jl` (v4: non-central only; the central type string is refused, and the reader points at the migration);
  - `migrate.jl` (`_generic_camera_v3_to_v4` and the file migration of a central v2/v3 record into a `vgio/cameras/v5` record);
  - `rig_schema.jl` (rig v8 → v9: a central generic sensor body becomes a camera body with a spline lens; v9 also stores the per-sensor `lens_coverage`, which retires the plan-1 refusal in `_sensor_json`);
  - `schoeps.jl` (central YAML → `BoundedCamera{CameraModel{…,SplineLens}}`, non-central → `BoundedCamera{NoncentralSplineModel}`);
  - `validate.jl` (`"generic_camera" => 4`, `"rig" => 9`, migrations table);
  - `conventions.jl:141`, `assemble.jl:157`, `external_calibration.jl:169` (refuse SplineLens through ONE shared helper: `ArgumentError("a SplineLens camera has no OpenCV/COLMAP form; store it as a vgio camera record (VisualGeometryToolkitIO.save), or fit a parametric lens to it first (VisualGeometryToolkitEstimation.fit_camera)")`, owner 2026-11-09);
  - `VisualGeometryToolkitIO.jl` imports;
  - docs `guide/kinds.md`; CHANGELOG 0.9.0.
- Move: `_present_runs`/`_present_from_runs` and `_RowRuns` out of `generic_camera_schema.jl` into `style.jl` (shared by the lens record and the non-central record; one copy).
- Test: port `generic_present_tests`, `generic_nominal_domain_tests` (→ `spline_lens_record_tests.jl`), `schoeps_tests`, `rig_schema_tests`, `rig_pitch_tests`, `units_roundtrip_tests`, `v2_validation_tests`, `schema_versions_tests`, `refusals_tests`, `target_record_tests:421-438`, and `camera_store_region_tests` (the rig item now round-trips the coverage). Regenerate the fixtures with the migration code, never by hand. The old v2/v3 fixtures stay as they are, as migration inputs.

**Interfaces:**
- Consumes: `SplineLens`, `NoncentralSplineModel`, `_spline_gauge` (Task 1), `with_lens_coverage` (plan 1).
- Produces:
  - **The lens record** `{"spline": {"area": [x, y, w, h], "controls": [[x,y,z],…], "grid_size": [gw, gh], "present": runs?, "forward": {"area", "controls", "grid_size"}?}}`, in normalized coordinates. K lives in the camera's usual intrinsics block.
  - `load(BoundedCamera, old_central_v2_or_v3.json)`:
    - extent = `width × height`;
    - K = stored `nominal_intrinsics` (skew must be 0) else `_spline_gauge`;
    - `lens_coverage` = stored `domain` else the sensor Rect.
    - Read migration only (owner, 2026-11-09): old v2/v3 files load forever and are never rewritten. The published `rc_baselines` grids and Ties's grids stay as they are on disk.
    - The forward spline is stored exactly when it has been fitted (owner, 2026-11-09).
  - `load(CameraModel, …)` on a record with a domain throws (Review Focus 2).
- [ ] **Step 1: Write the failing tests:**
  - **Old records load and match the golden values:** `test/fixtures/old/generic_camera_v2.json` (Noncentral) and `fixtures/units/generic_camera_masked.json` (Central v3) load. Their `backproject` equals `GOLDEN.io_fixtures` (Task 0) to 1e-12.
  - **A domain becomes coverage:** a v3 central record with `"domain": {"ellipse": …}` loads as a `BoundedCamera` whose `lens_coverage(bc)` is that Ellipse. `@test_throws ArgumentError load(CameraModel, p)`.
  - **Round trip:** a SplineLens camera with mask and forward round-trips through `save`/`load` bit-exactly (controls `==`).
  - **Review Focus 4:** the colmap→opencv frame conversion moves only K; `backproject` per pixel is unchanged to 1e-12.
  - **Rig migration:** `fixtures/v4/rig_canonical_m.json` migrates to v9; its sensors backproject as before.
  - **Rig coverage:** a rig holding a covered sensor round-trips its coverage.
  - **Exporters refuse:** `@test_throws ArgumentError opencv_parameters(cam)`, `colmap_cameras(Dict(:c => bc_spline))` and `save_opencv_calibration(…)`.
  - **Schöps:** `fixtures/schoeps/central_intrinsics0.yaml` loads as a SplineLens BoundedCamera.
- [ ] **Step 2: Run** with `run_tests(project_path = IO)`. Expected: FAIL.
- [ ] **Step 3: Implement.** `content_id`'s `_id_grid_model` is deleted; a SplineLens camera is identified by `_id_camera`'s lens branch (`_id_lens(::SplineLens)`). The target_record test pins the new keys.
- [ ] **Step 4: Run** the full IO suite. Expected: green, except the README fresh-env test (it needs Core 0.9 registered; already ruled).
- [ ] **Step 5: Duplication gate.**
  - There is one presence-runs codec, one ellipse/rect JSON codec (plan 1's `_lens_coverage_json`, reused for the old `domain`), and one schema-step mechanism.
  - Delete `_GENERIC_*` specs left without users.

  Commit: `"VisualGeometryToolkitIO: the SplineLens lens record; central grid files migrate to camera records; generic_camera v4 and rig v9"`.

### Task 7: Estimation — `fit_camera(SplineLens, …)` fills backward and forward

**Files:**
- Modify: `VisualGeometryToolkitEstimation/src/fitting/fit_camera.jl:57-83, 447-822`, `src/VisualGeometryToolkitEstimation.jl:225-227`, CHANGELOG 0.9.0, CLAUDE.md file map
- Test: `test/fit_camera_tests.jl`, `test/fit_camera_testsetups.jl`

**Interfaces:**
- Consumes: Task 1's source constructor and `spline_lattice`; Task 2's forward semantics.
- Produces:
  - `fit_camera(::Type{SplineLens}, source, region; grid_size, tolerance, forward = true) -> FitResult{<:CameraModel{LogicalIntrinsics,<:SplineLens}, CameraFitMetadata}`.
    - The backward comes from `pspline_grid`, unless the source is already a SplineLens camera, in which case its backward is kept exactly and only the forward is fitted.
    - The forward is the old `_spline_projection_fit` on `equidistant_chart(_lens_axis(lens))`, to `tolerance` px.
    - The metadata `family` is the union of the old central and SplineProjection fields.
  - `fit_camera(::Type{NoncentralSplineModel}, source, region; …)`.
  - `pspline_grid(SplineLens | NoncentralSplineModel, source, region, grid_size)`.
  - The `CentralGenericModel`, `NoncentralGenericModel` and `SplineProjection` targets are deleted. `_image_size` reads the `BoundedCamera` extent only.
- [ ] **Step 1: Port the tests to the new targets, keeping their numeric expectations:**
  - grid RMS and max per region;
  - SplineProjection tolerance met → forward tolerance met;
  - `forward = false` gives `lens(cam).forward === nothing`;
  - refitting a SplineLens source keeps `backward.controls` `==`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full Estimation suite. Expected: green. Also run the Features and Eval suites (they must stay green; they hold no references).
- [ ] **Step 5: Duplication gate.** There is one quadrature, one spacing search and one gauge source. Commit: `"VisualGeometryToolkitEstimation: fit_camera(SplineLens) fits the backward grid and the forward spline of one camera"`.

### Task 8: Whole-monorepo sweep

- [ ] **Step 1:** Run `grep -rn` over the monorepo (excluding CHANGELOGs and the plans) for every retired name and for "generic model"/"generic camera" wording in docs and docstrings. Expected: no hits.
- [ ] **Step 2:** Run every package's full suite and `ci/registration_lint.jl --subsequent`. Expected: green, except the README fresh-env items.
- [ ] **Step 3:** Fill the CHANGELOG 0.9.0 "Breaking" sections for Core, IO and Estimation. Name every retired name and its replacement, and the record migrations. Commit: `"VisualGeometryToolkit: the SplineLens fold-in, CHANGELOGs"`.

After Task 8 the controller runs `/simplify` and then `/code-review` on `origin/main...HEAD`. It then sends the final names and signatures to caliblobs-jl-73.
