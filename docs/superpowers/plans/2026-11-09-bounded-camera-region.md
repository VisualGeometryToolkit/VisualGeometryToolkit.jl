# BoundedCamera bound + lit region (Core and IO 0.9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `BoundedCamera` carries the region of the sensor that light reaches (a geometric primitive), so every consumer asks one question — `imaged_region(camera)` — instead of passing its own image circle.

**Architecture:** `BoundedCamera{C, B, R}` replaces its two bound fields (`extent`, `sensor`, which duplicate the extent when a sensor is given) with one `bound::B` (`Size2{Int}`, or the physical `Sensor` whose resolution IS the extent) and adds `region::R`, the lit region (`Rect2{Float64}`, `Circle{Float64}` or `Ellipse{Float64}`; default the sensor's own `Rect`, so there is no "none" value). `imaged_region(bc) = Rect(extent) ∩ region ∩ valid_domain(model)`; a grid model's `domain` keeps only its "where the grid is defined" role, answered by `valid_domain(grid)`. IO's camera store goes to `vgio/cameras/v5` with an optional per-camera `image_circle`.

**Tech Stack:** Julia 1.12+, VisualGeometryToolkitCore / IO (monorepo `~/src/vgtk-bounded-camera-region`, branch `bounded-camera-region`), GeometryBasics (`Rect2`, `Circle` = `HyperSphere{2}`), ReTestItems.

**Spec:** the design agreed with the owner on 2026-11-09, recorded in `~/.claude/projects/-home-jbpritts-src-radiometric-camera/memory/image-circle-on-bounded-camera.md`. Plans 2 (PanoramaStitching) and 3 (RadiometricCamera) adopt this release; they are separate documents.

## Global Constraints

- TYPE STABLE (owner's condition): unions appear only as type-parameter bounds; every constructed `BoundedCamera` has concrete field types; accessors dispatch on the field type (no `isa` branches); `@inferred` holds for `extent`, `sensor`, `lit_region`, `imaged_region`, `project`, `backproject` over every bound × region combination.
- `isbits(BoundedCamera(CameraModel(...), …))` stays true for an isbits wrapped camera, for each bound × region combination (it travels into GPU kernels as a value).
- The lit region is a geometric primitive only: `Rect2{Float64}`, `Circle{Float64}` (GeometryBasics), `Ellipse{Float64}` (Core). Never `nothing`, never `WholePlane` in the field. Other element types are converted to `Float64` by the constructor.
- `WholePlane` stays where it is: the answer of `valid_domain` for a lens with no limit.
- Breaking release: Core 0.8.0 → 0.9.0, IO 0.8.0 → 0.9.0; in-repo dependents' compat on them moves in the same hand commit (cau-release "breaking monorepo bump"). The downstream cascade is mandatory and is NOT part of this plan.
- No AI attribution in any commit message (monorepo CLAUDE.md). The stack's naming conventions (`VisualGeometryToolkitCore/conventions/`) apply to every new name.

## Review Focus

1. A lit region larger than the sensor or cut by it (the Sigma 14 mm's disc reaches past the top and bottom edges): `imaged_region` is clipped by the sensor `Rect`, `Rect2(imaged_region(bc))` is a finite box inside `[0, W] × [0, H]`. — test in Task 2.
2. `convert_eltype(Float32, bc)`, `ustrip`, `with_unit`, `with_model`, re-posing (`_with_extrinsics`) keep the bound AND the region bit for bit. — test in Task 1.
3. A `Vector{BoundedCamera}` whose cameras have DIFFERENT region types (one `Circle`, one `Ellipse`) is accepted by `view_overlap` and gives each camera's own region. — test in Task 2.
4. A `vgio/cameras/v4` file (no `image_circle`, with and without `pitch`) loads with the sensor `Rect` as region and the same bound as before. — test in Task 3.
5. An `Attributed` stored camera (with `calibration`) round-trips its region through v5. — test in Task 3.

---

### Task 1: `BoundedCamera{C, B, R}` — bound and lit region

**Files:**
- Modify: `VisualGeometryToolkitCore/src/geometry/cameras/cameras.jl:425-560` (struct, constructors, `sensor`, `_rebound`, docstring), `:652` (`extent(::HasBounds, bc)`)
- Modify: `VisualGeometryToolkitCore/src/geometry/cameras/rig.jl:45-60, 561` (`_raw_extrinsics`, `_with_extrinsics`, `with_model` rebuild through `_rebound`), `VisualGeometryToolkitCore/src/.../units` file holding `ustrip`/`with_unit` for `BoundedCamera` (grep `ustrip(u::Units, c::BoundedCamera)`)
- Modify: `VisualGeometryToolkitCore/src/VisualGeometryToolkitCore.jl` (export `lit_region`, `with_lit_region`)
- Test: `VisualGeometryToolkitCore/test/bounded_camera_region_tests.jl` (new); update `VisualGeometryToolkitCore/test/sensor_bound_tests.jl:20` (`bs.extent` → `extent(bs)`)

**Interfaces:**
- Produces:
  - `struct BoundedCamera{C <: AbstractCamera, B <: Union{Size2{Int}, Sensor{BaseQuantity, BaseQuantity}}, R <: Union{Rect2{Float64}, Circle{Float64}, Ellipse{Float64}}} <: AbstractCamera` with fields `camera::C`, `bound::B`, `region::R`
  - `BoundedCamera(camera, extent::Size2; region = nothing)`, `BoundedCamera(camera, sensor::Sensor; region = nothing)` — `region === nothing` means `Rect2{Float64}(0, 0, extent...)`; a `Circle`/`Ellipse`/`Rect2` of any real element type is converted to `Float64`; anything else is an `ArgumentError` naming the three accepted types
  - `extent(bc::BoundedCamera) -> Size2{Int}` (via the existing `BoundsTrait` path, `_extent(bc.bound)`), `sensor(bc) -> Union{Sensor, Nothing}` (via `_sensor(bc.bound)`), `lit_region(bc) -> R`, `with_lit_region(bc, region) -> BoundedCamera` (same camera and bound, new region, converted as in the constructor)
  - `_rebound(bc, camera)` keeps `bound` AND `region`

- [ ] **Step 1: Write the failing tests** in `bounded_camera_region_tests.jl`, one `@testitem "BoundedCamera: bound and lit region"`:
  - for `bound ∈ (Size2(4256, 2848), Sensor(Size2(4256px, 2848px), Size2(5.95μm/px, 5.95μm/px)))` × `region ∈ (nothing, Circle(Point2(2131.1, 1407.9), 1680.1), Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777))`, with `m = CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))`:
    - `bc = BoundedCamera(m, bound; region)`; `@test isbits(bc)`; `@test isconcretetype(typeof(bc))`
    - `@test @inferred(extent(bc)) == Size2(4256, 2848)`; `@test @inferred(sensor(bc)) === (bound isa Sensor ? sensor(bc) : nothing)` and `sensor(bc) isa (bound isa Sensor ? Sensor : Nothing)`
    - `@test @inferred(lit_region(bc)) == (region === nothing ? Rect2{Float64}(0, 0, 4256, 2848) : convert-of(region))`
    - `u = SVector(2500.0, 1500.0)`; `@test @inferred(backproject(bc, u)) == backproject(m, u)`; `@test @inferred(project(bc, backproject(m, u))) ≈ u`
    - `@test lit_region(convert_eltype(Float32, bc)) == lit_region(bc) && extent(convert_eltype(Float32, bc)) == extent(bc)`; same for `with_model(bc, m)` and a re-posed camera (`BoundedCamera(PosedCamera(m, X), bound; region)` → `_with_extrinsics` keeps the region)
  - `@test_throws ArgumentError BoundedCamera(m, Size2(10, 10); region = WholePlane())`
  - `@test lit_region(BoundedCamera(m, Size2(10, 10); region = Circle(Point2(5, 5), 4))) isa Circle{Float64}` (integer input converted)
  - `@test lit_region(with_lit_region(BoundedCamera(m, Size2(10, 10)), Circle(Point2(5.0, 5.0), 4.0))) == Circle(Point2(5.0, 5.0), 4.0)`

- [ ] **Step 2: Run to verify it fails**

  Run (kaimon, Core test env): `run_tests(pattern = "BoundedCamera: bound and lit region")`
  Expected: FAIL — `lit_region` not defined / `BoundedCamera` has no field `bound`.

- [ ] **Step 3: Implement** the struct, constructors, `_extent`/`_sensor` dispatch methods (`_extent(s::Sensor) = Size2{Int}(Rect(s).widths...)` — whatever `BoundedCamera(camera, s::Sensor)` computes today; `_extent(e::Size2) = e`; `_sensor(s::Sensor) = s`; `_sensor(::Size2) = nothing`), `lit_region`, `with_lit_region`, region conversion (`_region(::Nothing, ext)`, `_region(c::Circle, _)`, `_region(e::Ellipse, _)`, `_region(r::Rect2, _)`, fallback `ArgumentError`), `_rebound` keeping both fields. Replace every direct `bc.extent` / `bc.sensor` read in Core `src/` (cameras.jl:513, 652; docstrings :441, :454, :464, :479, :534, :547 — the hints now read `BoundedCamera(PosedCamera(model(bc), X), bc.bound; region = lit_region(bc))`). Update the docstring: "# The bound", "# The lit region", "# Field access" (`bc.camera`, `bc.bound`, `bc.region`).

- [ ] **Step 4: Run to verify it passes**, then the whole Core suite (`run_tests()` on the Core test env). Expected: PASS, no other item regresses (`sensor_bound_tests.jl` updated in this task).

- [ ] **Step 5: Commit** — `git commit -m "VisualGeometryToolkitCore: a BoundedCamera's bound (an extent or its Sensor) and its lit region"`

### Task 2: `imaged_region(bc)`, `valid_domain(grid)`, `view_overlap` on the camera's own region

**Files:**
- Modify: `VisualGeometryToolkitCore/src/geometry/cameras/image_circle.jl:24-40, 96-98, 177-194` (`imaged_region` docstring and methods), `VisualGeometryToolkitCore/src/geometry/cameras/generic_models.jl:129-137, 320, 382-400, 1470` (grid `imaged_region` → `valid_domain`), `VisualGeometryToolkitCore/src/geometry/cameras/view_overlap.jl:43-213`, `VisualGeometryToolkitCore/src/geometry/cameras/chart_spline.jl:100` (docstring wording), the two export comments (`VisualGeometryToolkitCore.jl:330`, `cameras.jl:1068`)
- Modify: every Core caller of `imaged_region(m)` on a grid model and of `imaged_region(camera, circle)` (grep both in `src/`, `ext/`, `test/`)
- Test: `VisualGeometryToolkitCore/test/bounded_camera_region_tests.jl` (second item), and the existing `view_overlap` / generic-model items updated to the new calls

**Interfaces:**
- Consumes: Task 1's `BoundedCamera{C, B, R}`, `lit_region`, `extent`
- Produces:
  - `imaged_region(bc::BoundedCamera) = Rect2{Float64}(0, 0, extent(bc)...) ∩ lit_region(bc) ∩ valid_domain(model(bc))` — the ONLY `imaged_region` method; the two-argument `imaged_region(camera, circle)` is removed (0.9 is breaking; its callers use `with_lit_region`)
  - `valid_domain(m::Union{CentralGenericModel, NoncentralGenericModel}) = _calibrated_rect(m) ∩ m.domain` — today's `imaged_region(m)` body; `imaged_region(m)` on a bare grid model is removed
  - `view_overlap(cameras::AbstractVector{<:BoundedCamera}; kwargs...)` and `view_overlap(a::BoundedCamera, b::BoundedCamera; kwargs...)` — each camera's footprint is its own `imaged_region`; the `circle`/`circles` positional arguments are removed

- [ ] **Step 1: Write the failing tests** (`@testitem "imaged_region: sensor ∩ lit region ∩ valid domain"`):
  - the Review Focus case 1: `bc = BoundedCamera(m, Size2(4256, 2848); region = Circle(Point2(2140.8, 1375.8), 1742.2))` (the 14 mm's measured lit disc, cut at top and bottom); `r = @inferred imaged_region(bc)`; `b = Rect2(r)`; `@test minimum(b) ≥ [0, 0] && maximum(b) ≤ [4256, 2848]`; `@test Point2(2140.8, 1375.8) ∈ r && Point2(10.0, 10.0) ∉ r`
  - a grid model `g` with `domain = Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777)` bounded with that larger lit circle: `@test Point2(2133.9 + 1650.0, 1423.7) ∉ imaged_region(BoundedCamera(g, Size2(4256, 2848); region = Circle(Point2(2140.8, 1375.8), 1742.2)))` (lit but outside the grid: excluded) and `∈` for `+1550.0`
  - `@test valid_domain(g) == _calibrated_rect(g) ∩ g.domain`-equivalent membership on a few points, and a parametric `CameraModel` with no lens limit: `imaged_region(BoundedCamera(m, Size2(100, 80)))` contains every pixel centre of `Rect2(0, 0, 100, 80)`
  - Review Focus case 3: `view_overlap([BoundedCamera(…; region = Circle(…)), BoundedCamera(…; region = Ellipse(…))])` runs and equals the overlap computed with each camera's region (compare against the 0.8 two-argument form's result for the same circles, recorded once as the expected matrix in the test)

- [ ] **Step 2: Run to verify it fails** — `run_tests(pattern = "imaged_region")`; expected FAIL (`imaged_region(::BoundedCamera)` with one argument has no method).

- [ ] **Step 3: Implement** the methods above; move the grid body to `valid_domain`; rewrite `view_overlap`'s `_footprint(camera, circle, n)` as `_footprint(camera, n)` reading `imaged_region(camera)`; update every caller and docstring found by `grep -rn "imaged_region\|view_overlap(" VisualGeometryToolkitCore/`.

- [ ] **Step 4: Run** the new item, then the whole Core suite. Expected: PASS.

- [ ] **Step 5: Commit** — `"VisualGeometryToolkitCore: imaged_region(camera) is the sensor ∩ its lit region ∩ the model's valid domain; a grid's domain is its valid_domain"`

### Task 3: IO — `vgio/cameras/v5` with the lit region

**Files:**
- Modify: `VisualGeometryToolkitIO/src/camera_schema.jl:2-20` (header), `:313-435` (store write/read: `bc.extent` at :349 and :458 → `extent(bc)`; write `image_circle`; read it), the `_CAMERAS_SPEC` key spec, the migration that reads v1/v2 (add v4 → v5), `VisualGeometryToolkitIO/src/VisualGeometryToolkitIO.jl:138` (kind table), `VisualGeometryToolkitIO/src/calibration_manifest.jl:8` (kind string example)
- Modify: `VisualGeometryToolkitIO/test/external_calibration_tests.jl:375, 387` (`c0.extent` → `extent(c0)`), every IO test or fixture naming `vgio/cameras/v4`
- Test: `VisualGeometryToolkitIO/test/camera_store_region_tests.jl` (new)

**Interfaces:**
- Consumes: Task 1's constructor keyword `region`, `lit_region`, `extent`, `sensor`
- Produces: `vgio/cameras/v5` — per camera an OPTIONAL `"image_circle"`, written only when `lit_region(bc)` is not the sensor `Rect`: `{"circle": [cx, cy, r]}` or `{"ellipse": [cx, cy, a, b, θ]}` (px in the store's pixel frame, the same `lower`/`lift` pair and pixel-frame conversion as the grid record's `domain`, `rig_schema.jl:964`); absent → the sensor `Rect`. A `Rect2` region other than the sensor's is an `ArgumentError` on write ("only a Circle or an Ellipse is stored as an image circle"). `load(Dict{Symbol, BoundedCamera}, path)` reads v5 and v4 (v4 through the migration: no `image_circle`, so the sensor `Rect`); `save` writes v5.

- [ ] **Step 1: Write the failing tests** (`@testitem "camera store v5: the lit region"`):
  - round trip in a temp dir for each region (`nothing`, a `Circle`, an `Ellipse`) × bound (`Size2`, `Sensor`): `save(p, Dict(:cam => bc))`; `bc2 = load(BoundedCamera, p)`; `@test lit_region(bc2) ≈ lit_region(bc)` (element-wise on centre/axes, rtol 1e-12) `&& extent(bc2) == extent(bc) && (sensor(bc2) === nothing) == (sensor(bc) === nothing)`; and the file's JSON has `"image_circle"` iff the region is not the sensor `Rect`; `"schema" == "vgio/cameras/v5"`
  - Review Focus case 5: the same round trip through `Attributed(bc, (; calibration = …))` keeps the region and the calibration
  - Review Focus case 4: a checked-in v4 fixture (copy an existing v4 test fixture, with and without `pitch`) loads with `lit_region == Rect2{Float64}(0, 0, extent...)` and the bound it had before
  - a COLMAP export of a v5 store with an ellipse region still writes the same `cameras.txt` row as before (the region is not COLMAP's)
  - `@test_throws ArgumentError save(p, Dict(:cam => with_lit_region(bc, Rect2(10.0, 10.0, 100.0, 100.0))))`

- [ ] **Step 2: Run to verify it fails** — `run_tests(pattern = "camera store v5")` on the IO test env; expected FAIL (schema is v4, no `image_circle`).

- [ ] **Step 3: Implement** the v5 writer/reader and the v4 → v5 migration; replace the `bc.extent` reads; update the kind table and doc strings naming `vgio/cameras/v4`.

- [ ] **Step 4: Run** the new item, then the whole IO suite. Expected: PASS.

- [ ] **Step 5: Commit** — `"VisualGeometryToolkitIO: vgio/cameras/v5 stores a camera's image circle (its lit region); v4 reads through the migration"`

### Task 4: The rest of the monorepo, the version bump and CHANGELOGs

**Files:**
- Modify: any remaining caller across `VisualGeometryToolkitEstimation`, `VisualGeometryToolkitFeatures`, `VisualGeometryToolkitEval`, `PoseLib` (`src/`, `ext/`, `test/`, `docs/`, `examples/`, `benchmark/`) of: `bc.extent`, `bc.sensor`, `BoundedCamera{C, S}` two-parameter type annotations, `imaged_region(…, circle)`, `imaged_region(grid_model)`, `view_overlap(…, circle…)`. (`VisualGeometryToolkitEval/src/datasets/roma.jl:108` reads `cam.extent` of a LEGACY deserialized struct — leave it unless `cam` is a Core `BoundedCamera`.)
- Modify: `VisualGeometryToolkitCore/Project.toml` and `VisualGeometryToolkitIO/Project.toml` versions `0.8.0` → `0.9.0`; every in-repo member's `[compat]` on Core / IO `"0.8"` → `"0.9"` (root, each package, `test/`, `docs/`, `benchmark/`, `examples/`, `validation/`, `ci/`)
- Modify: `VisualGeometryToolkitCore/CHANGELOG.md`, `VisualGeometryToolkitIO/CHANGELOG.md` — a `## 0.9.0` entry each, "Breaking:" lines for: the `BoundedCamera` fields (`extent`/`sensor` → `bound`; new `region`; three type parameters), `imaged_region(camera, circle)` removed, `imaged_region(grid)` → `valid_domain(grid)`, `view_overlap`'s circle arguments removed, `vgio/cameras/v5`

**Interfaces:**
- Consumes: Tasks 1–3.

- [ ] **Step 1:** `grep -rnE '\.(extent|sensor)\b|BoundedCamera\{[^}]*,[^}]*\}|imaged_region\(|view_overlap\(' --include=*.jl` over the five packages; fix each Core-`BoundedCamera` use as in Tasks 1–3.
- [ ] **Step 2:** Bump the two versions and every in-repo `[compat]` entry on Core and IO; resolve the workspace (`Pkg.update(; workspace = true)` in a kaimon session of the worktree root) — it must resolve with Core 0.9.0 and IO 0.9.0 from the working trees.
- [ ] **Step 3:** Run every package's suite in the worktree (Core, IO, Estimation, Features, Eval, PoseLib) and `ci/registration_lint.jl --subsequent`. Expected: all PASS.
- [ ] **Step 4: Commit** the call-site fixes, then `"Bump version: VisualGeometryToolkitCore 0.8.0 → 0.9.0, VisualGeometryToolkitIO 0.8.0 → 0.9.0"` with the compat moves (cau-release: a breaking monorepo bump is made by hand with the in-repo dependents' compat; the release leg then runs with `old == new`).

### Task 5: `/simplify`, `/code-review`, then hand to the release

- [ ] **Step 1:** `/simplify` on the branch diff (`origin/main...bounded-camera-region`); land its accepted changes as their own commit(s); re-run the touched packages' suites.
- [ ] **Step 2:** `/code-review` on the same diff; fix confirmed findings in their own commit(s); re-run the touched suites.
- [ ] **Step 3:** Stop. The release (cau-release, `LEG_PHASES="prep gate"`, General registration with `Release notes:` naming the breaking changes, then the mandatory downstream cascade) waits for the owner's go-ahead and for plans 2 and 3 to have proven the API downstream on temporary `[sources]`.
