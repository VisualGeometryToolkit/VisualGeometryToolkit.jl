docs-pass: 4c910232
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the portal names packages and links their docs; none of these fixes touches what it says)

# Core 0.9.6 / Estimation 0.10.2 / Eval 0.7.3 docs pass (julia-docs-pass) — patch-core096-eval073

Urgent patches from the catch-up review (wave10/review-monorepo.md C1, E1), with the additions the owner asked for:
Core's `reduce_normal_equations`/`reduce_cost` (Estimation's hand unpack moved there), the weak CUDA floor that made
0.9.5 unresolvable beside cuDNN 1, the chart LM's `ftol` in `refine_camera`'s regression, and `optical_axis(camera,
region)` with every bounded-source fit over `region ∩ imaged_region`. Baseline: 2f994d2b (wave 113), main eeb82edf
(IO 0.9.4) merged in. Core 0.9.6 was released from ccf9e2af (its tree as here).

## 1. Public API since the baseline

- Core, public, not exported: `reduce_normal_equations(partials, Val(N))`, `reduce_cost(partials)` (robust/lm_core.jl;
  docstrings each, on api/robust_statistics.md through its `@autodocs` of robust/lm_core.jl). `axis_gauged`'s
  BoundedCamera method: no signature change; its docstring's bounded-camera line now says a posed one's extrinsics turn.
  Weak `[compat] CUDA` "5.11.3" → "5.11 - 5". Nothing removed.
- Estimation: `optical_axis(camera::BoundedCamera, region = imaged_region(camera))` (a new optional argument; docstring
  and docs/src/api/radial.md); `fit_camera`/`refine_camera`/`approximate_camera` of a bounded source fit over
  `region ∩ imaged_region(source)` (docstrings say so; `fit_camera(SplineLens, …)` unchanged: its region is the
  lattice's support). The private hand unpack deleted. `[compat] VisualGeometryToolkitCore = "0.9.6"`.
- Eval: no API change. `PlannedStage`'s docstring (the run key folds the upstream stages' keys) and the reproduce page
  (docs/src/reproduce.md, "Run manifest, reuse") updated.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | the two new Core names documented and in the manual (Core docs build strict); signatures checked by the `docs` items |
| stale mentions | none: the deleted Estimation unpack had no docs; the runner's old key rule appears only in CHANGELOGs |
| CHANGELOG | Core 0.9.6, Estimation 0.10.2, Eval 0.7.3 (Eval's names the one-time rerun of manifests from 0.7.2) |
| versions | Core 0.9.6, Estimation 0.10.2, Eval 0.7.3; IO 0.9.4 (main), Features 0.8.3, PoseLib 0.2.1 unchanged |
| in-repo compat | Estimation (package, test, docs) on Core "0.9.6"; every other member as it was |
| README / landing | unchanged (no install line or example moved) |

## 3. Executed on the release candidate (ccf9e2af)

Gate (~/logs/dgx-agent/scripts/patch096/gate.sbatch, submit.sh): a fresh `git clone --shared`, the sha detached, no
Manifest, General a GitHub clone in the patch's depot (JULIA_PKG_SERVER=""), a CPU prep that precompiles every test env
(the GPU code included) with the tasks' flags (--check-bounds=yes --depwarn=error), 2-CPU single-worker test groups,
each GPU group on its own GPU; Julia 1.13.3 only. Three runs; a package's results carry by its tree hash
(`git rev-parse <sha>:<dir>`): Core, IO, PoseLib from ccf9e2af (= 4c910232's trees), Features from 562d197e (= 4c910232's),
Estimation and Eval from 4c910232. Counts are pass / broken.

| | CPU | GPU | docs | run |
|---|---|---|---|---|
| prep: registration_lint --subsequent, agent_docs, floor_compat --latest, audit_manifests | 0 errors, 0 warnings; 0; rc 0; up to date | | | each run |
| check_duplication (all six, stack paths set) | 0 findings each | | | each run |
| GPU stacks: the six beside cuDNN 1; bb-063 and cb-053 with them | resolve (cuDNN 1.4.7, CUDA 5.11.1, CUDA_Runtime_Discovery 1.0.0) | | | each run |
| Core 0.9.6 | 410198 / 30 (8 groups) | 53121 / 0 (4) | 0 | ccf9e2af 12587–12591 |
| IO 0.9.4 | 30756 / 4 (6) | — (no device file) | 0 | ccf9e2af |
| Features 0.8.3 (tests of 562d197e) | 3063 / 9 (6) | 2011 / 0 (2) | 0 | 562d197e 12703–12706 |
| Estimation 0.10.2 | 8830 / 2 (6) | 55 / 0 (1) | 0 | 4c910232 13140–13143 |
| Eval 0.7.3 | 2913 / 0 (6) | 359 / 0 (1) | 0 | 4c910232 |
| PoseLib 0.2.1 | 373 / 1 (suite) | — | 0 | ccf9e2af |

ccf9e2af's Features GPU group 25 failed (47, `vgf_tests.jl` "the one matcher ≡ the reference rules": a GPU matrix
against the CPU reference, once another item had loaded CUDA); the same failure on the released 2f994d2b (job 12586).
562d197e's test-only fix (feat09's) is green.

---

docs-pass: ce8acb3c
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the portal names packages and links their docs; it lists no vgio kind or IO function)

# IO 0.9.4 docs pass (julia-docs-pass) — io-0.9.4 (VisualGeometryToolkitIO 0.9.4 alone; every other package unchanged)

Additive: the `vgio/panorama/v1` kind and `PointFrame` in the detection codec. Baseline: 2f994d2b (wave 113, IO 0.9.3).

## 1. Public API since the baseline

- New public, not exported: `save_panorama(path, camera::BoundedCamera, rotations::Trajectory, vignetting::AbstractVignetting,
  gains; length_unit = mm, provenance = (;))`, `load_panorama(path) -> (; camera, rotations, vignetting, gains)`
  (src/panorama_schema.jl).
- New methods of the public `json_value` / `from_json`: `PointFrame`, and a detection of an `Uncertain{PointFrame}`
  (src/core_json.jl). Nothing removed or re-signatured.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `save_panorama`, `load_panorama` on api/native.md (its `@autodocs` Pages gain panorama_schema.jl); `json_value`/`from_json` docstrings name `PointFrame` and its covariance encoding |
| guides | kinds.md (the kinds table: the panorama row), schemas.md (the current versions: panorama v1), serialization.md (the frames table's `PointFrame` row, the point frame's paragraph and a doctest) |
| module header | VisualGeometryToolkitIO.jl: the current versions list and the I/O surface comment name the kind |
| CHANGELOG | VisualGeometryToolkitIO 0.9.4 |
| versions | IO 0.9.4 (Project.toml and its benchmark env's compat on itself); in-repo dependents' compat unchanged (a patch) |
| floors | floor_compat --latest and --raise --check on the gate's resolve: 0 raises (gate prep 11552) |

## 3. Gate

Fresh-clone gate on Julia 1.13.3, ce8acb3c, all green (prep 11552; tests 11553, 11554; docs 11554_5): IO strict 30760 tests (4 broken: the report-mode stack gates and the pycolmap skip, as at 0.9.3), Eval strict 2906, IO docs (checkdocs = :public, doctest, linkcheck). Table: ~/logs/dgx-agent/io094-ps064.md.
