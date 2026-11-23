# Core 0.14 and the monorepo callers: the implementation spec

The working spec of branch `core-0.14/solve-interface`. The design record behind it is
`docs/design/sciml-mapping.md` (the SciML study and the owner's rulings of 2026-11-23). Every agent working on the
branch reads both before editing, and adds to §9 every decision it takes that this spec does not already state.

## 0. Rules that override everything

- **One way per quantity, zero duplication.** No alias, shim, deprecation, second path, `const` rebinding or wrapper
  of a deleted name. Search exhaustively (grep + read the hits) before adding anything.
- **Every deleted public name** goes into `RETIRED_NAMES` (`VisualGeometryToolkitCore/ext/VisualGeometryToolkitCoreTestExt/check_naming.jl`)
  with its replacement in the comment, and into the package's CHANGELOG.
- **Bit identity.** Every replacement must replay the golden records (`ci/golden/`, §8) with `===`. A numeric change
  is a defect unless the owner rules it in.
- **Kinds of keyword.** Problem context is `p` (a type parameter); solve keywords are run settings only (`rng`,
  `scale`/`width`, `stops`, `bound`); algorithm configuration is the algorithm's fields. No `workspace =` keyword
  anywhere: workspace reuse is `init`/`solve!`/`reinit!` only.
- **No AI attribution** in commits. Never force-push, never rewrite history; push only the branch.
- Spelling: American in identifiers, British in prose.

## 1. The interface (Core, `src/base/solve_interface.jl`)

Built. Read it. In short:

- `AbstractEstimationProblem`, `AbstractEstimationAlgorithm`, `NullParameters`.
- The problems:
  - `EstimationProblem(f, p = NullParameters())`: `f` a problem-interface object whose `model_type(f)` is the model.
  - `EstimationProblem(M::Type, data, p = NullParameters())`: a closed form or fit on raw data.
  - `RefineProblem(f, u0, p = NullParameters(); lb, ub)`.
  - `MinimalProblem(M::Type, data, sample, p = NullParameters())`.
- The kinds: `EstimatorKind` = `MinimalKind | ClosedFormKind | FitKind | FindKind | RefineKind`.
- The traits: `estimator_kind(alg)` (every algorithm type defines one), `needs_initial_model(alg)`,
  `decides_inliers(alg)` and `supports_solve_rng(prob, alg)`. The last is derived (`FindKind`) and overridable per
  algorithm; it replaces the prototype's `is_randomized`.
- The front end:
  - `solve(prob, alg; kw...)` = `check_kind`, then `__solve(prob, alg; kw...)`, then `with_problem(result, prob, alg)`.
  - `init(prob, alg; kw...)` = `check_kind`, then `__init(prob, alg; kw...)`.
  - `solve(prob)` runs `default_algorithm(prob)`.
  - `reinit!(cache[, u0]; p, f)` is defined by the cache's owner.
  - `__solve` defaults to `solve!(__init(…))`.
- **Packages add `__solve`/`__init` methods only, for their own algorithm types.** A `CommonSolve.solve`/`init`
  method outside Core is a violation (R16).
- **`default_algorithm(prob)`** is declared per problem type: closed form over fit, LM for a `RefineProblem`, the
  model's one minimal solver for a `MinimalProblem`, NONE for a find. Without a declaration it throws an
  `ArgumentError` listing the `__solve`/`__init` methods that apply.

## 2. The solution (`FitResult`, `src/base/fit_result.jl`)

Built.

- `FitResult{U, M, R, Prob, Alg, Orig}` has the fields `u, resid, retcode, stats::SolveStats, prob, alg, original,
  stop_rule, rms, n, metadata`.
- The constructor is `FitResult(u, retcode, stop_rule, stats, rms, n, metadata; resid, prob, alg, original)`.
- `SolveStats(; nsteps, nf, njacs, nfactors, nsolve)`, immutable.
- `with_problem(r, prob, alg)` sets `prob` and `alg`; only the front end calls it.
- **Read fields by field:** `r.u`, `r.retcode`, `r.stats.nsteps`, `r.stop_rule`, `r.rms`, `r.n`, `r.resid`,
  `r.metadata.<field>`. The success test is `successful_retcode(r)`.
- **Read cross-type contract quantities by their function ONLY:** `noise_scale`, `scale_source`, `weights`,
  `inlier_mask`, `dof`, `s2`, `found_model`, `retcode_symbol` (and `weight_kind`).
  - A FitResult forwards no property into its metadata.
  - `r.metadata.weights`, `r.metadata.dof`, `r.metadata.scale`, `r.metadata.scale_source`, `r.metadata.s2` and
    `r.metadata.inlier_mask` are forbidden outside the contract methods that read them (check_naming R21).
- **Retired:** `estimate`, `retcode(r)`, `iterations`, `converged` (function and property), `stop_rule(r)`,
  `rms(r::FitResult)` (`rms(residuals, n)`, the computation, stays), `metadata(r)`, `nested_metadata`, and every
  property forwarded into the metadata (`r.weights`, `r.inlier_mask`, `r.quality`, …).
- **`resid` holds the per-datum whitened residual distances,** computed once at the solution.
  - Where a result kept them in its metadata (`RefineMetadata.residuals`, `RansacMetadata.residuals`), they move to
    `resid` and the metadata field is deleted.
  - A result that computes none (a closed form) has `resid = nothing`. No new residual computation is added: that
    would break the runtime gate.
- **Closed forms return a `FitResult`** (`closed_form_result`, `src/geometry/solvers/closed_forms.jl`):
  - success: `(Success, ClosedForm)`, `n` the data used, `rms` NaN;
  - degenerate: `u = nothing`, `(InitialFailure, NoRule)`, `n = 0`;
  - malformed input: the old `ArgumentError`, unchanged.

## 3. Core's estimators: old → new

| old | new |
|---|---|
| `homography_dlt(cs; mask, svd_ws)` | `solve(EstimationProblem(HomographyMat, cs), DLT())`. A mask is data selection: `cs[mask]`. `svd_ws` is gone from the public form; the inner loop keeps `homography_dlt!(A, N, cs; mask, svd_ws)` |
| `fundamental_dlt(cs; …)` | `solve(EstimationProblem(FundMat, cs), DLT())`; inner loop `fundamental_dlt!` |
| `essential_dlt([backend,] cs)` | `solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))`. The backend kernel is the contract generic `essential_8pt(backend, cs)`, which PoseLib implements |
| `kabsch(cs)` | `solve(EstimationProblem(RotMatrix{3}, cs), Kabsch())` (private kernel `_kabsch`) |
| `chordal_mean(Rs)` | `solve(EstimationProblem(RotMatrix{3}, Rs), ChordalMean())` |
| `spectral_rotation_averaging(rel)` | `solve(EstimationProblem(AbsoluteRotations, rel), SpectralAveraging())`. `AbsoluteRotations` is a model TAG (abstract, never instantiated) |
| `zhang_intrinsics(Hs[, pitch]; center)` | `solve(EstimationProblem(LogicalIntrinsics, Hs, (; center[, pitch])), Zhang())` |
| `tls_line(points[, weights])` | `solve(EstimationProblem(Line2D, points), TotalLeastSquares())`; weighted: `EstimationProblem(Line2D, (; points, weights))` |
| `direct_ellipse(points)` | `solve(EstimationProblem(Ellipse, points), DirectEllipse())` |
| `nearest_point(lines)` | `solve(EstimationProblem(Point, lines), NearestPoint())` |
| `triangulate(pose, f₁, f₂) -> (X, λ₁, λ₂)` | `solve(EstimationProblem(Point3, f₁ => f₂, pose), Triangulation())`: `u = X`, `metadata.depths = (λ₁, λ₂)`. Inner loops call the private `_triangulate` |
| `p3p(model, cs)` | `solve(MinimalProblem(Extrinsics3, cs, 1:3, model), P3P())`: the context `p` is the camera, which back-projects the pixels |
| `solve(solver, cs, idx)` | `solve(MinimalProblem(M, cs, idx), solver)`, or `solve(MinimalProblem(M, cs, idx))` (the declared default) |
| `photon_transfer(groups, darks; defects, block, normalize, significance)` | `solve(EstimationProblem(PoissonGaussian, groups, (; darks, defects)), PhotonTransfer(; block, normalize, significance))`; the `ReadNoise`-dict form: `p = (; darks, sensor_model)`; the single-mosaic form: `EstimationProblem(PoissonGaussian, mosaic, (; sensor_model))`, its `u` the `PoissonGaussian` |
| `master_dark(frames; exposure_band, false_defects)` | `solve(EstimationProblem(Dark, frames, (; exposure_band)), MasterDark(; false_defects))`: `u = (; dark, read_noise, defects)` |
| `fit_scale(prior, rss, dof)` | `solve(EstimationProblem(SquaredScale, (; rss, dof)), ScaleMAP(prior))`: `u` is ŝ². `SquaredScale` is a model TAG; inner loops call the private `_ig_map_scale` |
| `static_lm(fj, x₀, lb, ub, stops[, bound])` | `solve(RefineProblem(fj, x₀; lb, ub), StaticLevenbergMarquardt(); stops, bound)`: `u = x`, `stats.nsteps` the iteration count (now returned), `metadata = (; jacobian = J, evaluation = e)`; isbits, runs in a kernel |

The algorithm docstrings carry the old functions' documentation. The kernels keep their numerics and become private
(`_kabsch`, `_tls_line`, …), except the inner-loop public machinery (`homography_dlt!`, `fundamental_dlt!`,
`tls_line_statistics`, the positional minimal kernels `homography_4pt`, `p3p(rays, X)`, …).

## 4. The Levenberg–Marquardt (decisions 3, 10; rulings Q4, Q6)

- **`LevenbergMarquardt(; optimization_strategy = ClassDefault(), damping = MarquardtDamping())`** is Core's ONE host
  LM.
  - Damping is its field. `LmStops` loses its `damping` keyword and field.
  - Stops are the solve keyword `stops = LmStops(…)`; `scale` or `width` are solve keywords too.
- **It is generic over the chart-LM interface of `f`.** The generic refine body moves from Estimation
  (`fitting/_refine.jl`, `manifold_providers.jl`: `_refine_pipeline`, `_refine_spec`, `_chart_refit`,
  `_triggs_lm`, `_ZeroWeightRows`, `RefineMetadata`) into Core.
  - Estimation's problems implement the chart contract as public Core contract generics: `model_chart(f, model)`,
    `chart_rows(f, chart, sqrtw)`, `chart_model(f, chart)`, `chart_size(f)`.
  - The names are final once §9 records them. Core already exports `chart` for `MappedSurface`, so that name is taken.
  - A multi-block family (the bundles) implements the block contract the same way, over Core's `LmProblem`.
- **`StaticLevenbergMarquardt()`** is the device LM (`static_lm`'s numerics).

## 5. VisualGeometryToolkitEstimation

- **Finds.** `Ransac(; scoring, local_optimization_strategy, config)` owns `__solve(::EstimationProblem{<:Any,
  <:AbstractRansacProblem}, ::Ransac; rng)` and `__init` → `RansacCache` / `solve!`.
  - `p = (; extent)` replaces `extent =`.
  - The family problems (`HomographyProblem(cs)`, …) are the public `f`s, with public constructors.
  - `solve(problem, idx)` on a RANSAC problem is renamed `sample_models(problem, idx)`.
- **Refines.** `solve(RefineProblem(XProblem(data), θ₀), LevenbergMarquardt(); scale | width)` runs through Core's
  generic LM.
- **Fits.** One algorithm type per fit, per the plan's table (`~/src/_wt-solve-proto/migration-plan.md` §2), each
  with its `estimator_kind`. `SimFitPP` answers `supports_solve_rng = true`.
- **`locally_optimize`** becomes `RefineProblem(f, θ₀)` → `LocalOptimization(strategy; scoring, score_guard)`.
- **Contract hooks with reserved prefixes** are renamed:
  - `fit_hypothesis` → `sample_hypothesis`;
  - `locally_optimize_rescored!` → `rescore_and_optimize!`;
  - Core's `refine_external!` → `external_refit!`, `refine_workspace` → `refit_workspace` and `refine_width` →
    `refit_width`.
  - LODR implements the renamed hooks in its own wave.

## 6. PoseLib, Eval, Features, IO

- **PoseLib:** `PoseLibLM(; max_iter, optimization_strategy)` on
  `RefineProblem(XProblem-equivalent f, θ₀, (; K1, K2…))`. The backend kernels implement Core's contract generics,
  incl. `essential_8pt`. The raw C++ result goes in `original`.
- **Eval:** `fit_prior(spec, σ²s)` → `solve(EstimationProblem(IGPrior, σ²s), spec)`. `PriorSpec` is the algorithm
  (`ClosedFormKind`).
- **Features:** the internal callers migrate. `detect_image_circle` keeps its name (a task verb). Its internals call
  the new circle and ellipse finds.
- **IO:** `fit_record_json` → `fit_result_json`, `fit_record_from_json` → `fit_result_from_json`. These serialize a
  `FitResult` (the `fit_` prefix is reserved). The persisted record keeps its schema keys; a version step maps
  `estimate`/`iterations` if the schema changes.

## 7. Decision 9, 11, 13 items

These are done in Core 0.14, each with its own commit:

- `centroid` follows its input;
- `origin`-as-centre deleted;
- one pixel-camera constructor plus `BoundedCamera(f, sensor; lens)`;
- the `px` K forms check focal positivity;
- the LM record's cost → `objective` (SciML's name; `sum_squares` only where the value is not the minimized
  objective);
- American identifiers (`SensorCharacterization` and its TOML key);
- one `s2`/`dof` spelling;
- the two 1-ulp tolerances;
- TOML tables renamed after algorithms and problems, with a versioned converter per package;
- PrecompileTools workloads.

## 8. Golden records (`ci/golden/`)

- The recorder runs at 3d8a06da (`record.jl`, `record/<group>.jl`). The replay runs on the branch (`replay.jl`,
  `replay/<group>.jl`, `replay!(cases, id, fixture, run; map = FieldMap(…))`). Both build their fixtures from the same
  `fixtures/<group>.jl`.
- `FieldMap(renames, new_only, retyped, ignored)` states the field mapping explicitly:
  - FitResult → FitResult: renames `"u" => "estimate"`, `"stats.nsteps" => "iterations"`,
    `"resid" => "metadata.residuals"` where the residuals moved; `new_only = ["prob", "alg", "original",
    "stats.nf", "stats.njacs", "stats.nfactors", "stats.nsolve", "resid"]` (unless mapped); `retyped = ["",
    "metadata"]` where the types changed.
  - A closed form that returned a bare model → FitResult: renames `"u" => ""`, every other field `new_only`.
  - `ignored = ["exception.msg"]` only where the old message named the old keywords (the type is always compared).
- A fixture that holds an upstream result (a find's result seeding a refine) is rebuilt with the new API, its
  `fixture_map` stating the mapping.
- Run: `JULIA_LOAD_PATH="@:$PWD:@stdlib" julia -t 1 --startup-file=no --project=VisualGeometryToolkitEstimation/test
  ci/golden/replay.jl [group]` from the worktree root. 100 % is the gate.

## 9. Decisions taken during the migration

(Append: date, decision, reason.)

- 2026-11-23: Closed forms keep their malformed-input `ArgumentError`s, and degenerate is `u = nothing`. Reason: bit
  identity of the recorded exceptions; a closed form "can only be degenerate".
- 2026-11-23: `MinimalProblem` has only the `(M, data, sample, p)` form. Reason: no caller needs a family-problem
  form; Estimation's internal sample solve is `sample_models`.
- 2026-11-23: `mask` and `svd_ws` leave the public DLT. `mask` is data selection (`cs[mask]`); `svd_ws` would be a
  `workspace` keyword (decision 5).
- 2026-11-23 (Core agent): the kind law's minimal refusal is two methods (`EstimationProblem`, `RefineProblem`), not one
  on their `Union`. Reason: the `Union` method was ambiguous with `_check_kind(::RefineProblem, ::EstimatorKind)` and
  `_check_kind(::EstimationProblem, ::EstimatorKind)`.
- 2026-11-23 (Core agent): `LmStops` has no damping; the damping of every host LM run is `p.damping` of the
  `LmProblem` it solves (the chart LMs' kept runs included). `LmProblem` gains a type parameter `D <: LmDamping` (was an
  abstract `damping::LmDamping` field), so the NLLSsolver options stay inferred and allocation-free. The chart forms of
  `lmsolve_chart!` take the damping as a trailing positional argument, `lmsolve_chart!(ews, lm[, robustifier, Val(C)],
  stops = LmStops(), damping = MarquardtDamping())`: the smallest change, and what `LevenbergMarquardt`'s generic
  `__solve` will pass (`alg.damping`) when the refine pipeline moves into Core. The 0.10.4 refusal of a stops of
  another damping than the problem's is gone (the stops carry none). Bit identity: the options are built from the same
  stops and the same damping type as before.
- 2026-11-23 (Core agent): `LevenbergMarquardt` and `ClassDefault` are defined in Core now (`ClassDefault` in
  solve_interface.jl, `LevenbergMarquardt{S, D}` in lm_core.jl with `default_algorithm(::RefineProblem)`); no
  `__solve` for it yet (the refine pipeline moves in a later step), so `solve(RefineProblem(…))` is a `MethodError`
  until then.
- 2026-11-23 (Core agent): the device LM. The kernel is the private `_static_lm`, returning
  `(x, converged, J, e, rule, iterations)` (the iteration count appended; numerics unchanged). The hot device loops —
  the parametric lens inverse (lens_rays.jl), the spline lenses' projection (spline_lens.jl), the mapped surfaces' ray
  root (mapped_surface.jl) and the non-central projection (noncentral_spline_model.jl) — call `_static_lm` directly:
  they read only `x` (and `ok`, the evaluation), and a FitResult, a problem record and a keyword call per ray buy
  them nothing. `StaticLevenbergMarquardt()`'s `__solve` wraps the same kernel: `stops` is a REQUIRED keyword (its
  `scale` is the problem's; no default exists); `lb`/`ub` of `nothing` are `∓Inf` (bit-identical to an explicit
  infinite box); `rms`/`n` are over the final residual `r = e[1]` (`rms(r)`, a reduction of a value the kernel
  returned, not a new residual evaluation); `resid = nothing`; `metadata = (; jacobian, evaluation)`; the result is
  built by the positional inner constructor (no keyword call on a device). Tested in a KernelAbstractions CPU kernel
  against `_static_lm`, `===` per ray.
- 2026-11-23 (Core agent): `ScaleMAP(prior)` is a `ClosedFormKind`; `default_algorithm` of a `SquaredScale` problem is
  `ScaleMAP(JeffreysPrior())` (the stack's default prior); its result has `n = 0` (it reads a sum, not data). The inner
  loops (`update_scale!`, `partition_student_t`) call `_ig_map_scale`.
- 2026-11-23 (Core agent): `PhotonTransfer(; block, normalize, significance)` is a `FitKind` (iterative reweighting)
  and the declared default of a `PoissonGaussian` problem; `MasterDark(; false_defects)` a `ClosedFormKind`, the
  default of a `Dark` problem from frames. Their configuration is checked in their constructors (`ArgumentError`, as
  before, now at construction). The context keys are checked: a series with `Darks` reads `(:darks, :defects)`, with
  a `ReadNoise` dict `(:darks, :sensor_model)`, one frame `(:sensor_model,)` (or `NullParameters()`); any other key is an
  `ArgumentError` (a misspelt key is never ignored); an absent `defects`/`sensor_model` is `NoDefects()`/`SensorModel()`,
  the old keyword defaults. The one-frame form now returns a `FitResult` (`u` the `PoissonGaussian`; `(Success,
  IncrementSize)` — it throws otherwise, as before; `stats.nsteps` the reweightings; `n` the points; `rms` NaN, no new
  residual summary; metadata `nothing`). `MasterDark`'s `u` is `(; dark, read_noise, defects)`, `n` the frames.
- 2026-11-23 (Core agent): an essential matrix declares NO default (the only 8-point is PoseLib's); `DLT()` (native)
  on it is an `ArgumentError` naming `DLT(PoseLibSolver())`. The bare `essential_8pt(cs)` form PoseLib used to add
  (`essential_dlt(cs)`) is not part of the contract: the generic is `essential_8pt(backend, cs)` only.
- 2026-11-23 (Core agent): the camera form of P3P, `MinimalProblem(Extrinsics3, cs, idx, camera)`, takes a
  `CameraModel` or a `BoundedCamera{<:CameraModel}` (its camera back-projects): it replaces both `p3p(model, cs)` and
  the `MacroTools.@forward` of `p3p` onto `BoundedCamera`, which no longer exists (the kernels `p3p(rays, X)` stay).
- 2026-11-23 (Core agent): errors that named the old function now name the algorithm (`DirectEllipse requires…`,
  `NearestPoint: …`, `TotalLeastSquares requires…`, `SpectralAveraging: …`, `PhotonTransfer: …`, `MasterDark: …`,
  `Zhang needs…`); the exception TYPES are unchanged. The golden replay maps them through `ignored = ["exception.msg"]`.
- 2026-11-23 (Core agent): exports. Exported: the problems, `NullParameters`, `AbstractEstimationAlgorithm`,
  `ClassDefault`, `solve`, `init`, `solve!`, `reinit!`, the six kind types, `SolveStats`, `successful_retcode` (moved from
  public), every algorithm type and the model tags `AbsoluteRotations`, `SquaredScale`. Public, not exported: the
  extension points `__solve`, `__init`; the declarations `estimator_kind`, `default_algorithm`, `supports_solve_rng`;
  the readings `needs_initial_model`, `decides_inliers`; `check_kind`, `with_problem`, `closed_form_result`;
  `essential_8pt`, `static_lm_termination`, `map_estimate`. Reason: users call `solve`; the rest is for algorithm and
  problem authors. `rms` stays exported (the computation), `stop_rule` stays public (the block-coordinate-descent
  hook `stop_rule(criterion, method)`, a different method).
- 2026-11-23 (Core agent): `CONTRACT_GENERICS` gains `__solve, __init, solve!, reinit!, estimator_kind,
  default_algorithm, supports_solve_rng, essential_8pt` and loses `solve` (CommonSolve's now: no package but Core adds
  a method, R16), `essential_dlt` and `nested_metadata`.
- 2026-11-23 (Core agent): `RETIRED_NAMES` holds the deleted FUNCTION names (the closed forms, the estimators,
  `static_lm`, `nested_metadata`). The retired ACCESSORS `estimate`, `iterations`, `converged`, `retcode`, `stop_rule`,
  `rms`, `metadata` cannot be retired by the rule's text match: each is also a field (`r.retcode`, `r.stop_rule`,
  `r.rms`, `r.metadata`), a kept function (`stop_rule(criterion, method)`, `rms(residuals, n)`), a NamedTuple field
  (`sol.iterations`) or a local/English word. Their retirement needs a syntax-aware rule (a call `f(r)` on a
  FitResult), not a RETIRED_NAMES entry — for the owner/the rules agent (R21's family).
- 2026-11-23 (Core agent): the retired-name rule forbids `tls_line.jl`, `nearest_point.jl` and `direct_ellipse.jl` as
  file names (`.` is no word boundary for it): renamed `total_least_squares.jl`, `point_from_lines.jl`,
  `algebraic_ellipse.jl`; test items named after a retired function are renamed after the algorithm
  (`"TotalLeastSquares"`, `"DirectEllipse"`, …), their files kept (`*_tests.jl` does not match).
- 2026-11-23 (Core agent), OPEN — needs the owner: §0 (retire every deleted name) collides with §7 (TOML tables named
  after the algorithm, snake_case) wherever the algorithm's snake name IS the deleted function's: `kabsch`,
  `direct_ellipse`, `nearest_point`, `photon_transfer`, `master_dark`. Today one instance exists: Core's
  `configs/default.toml` table `[photon_transfer]` (and its reader `toml_default(_CORE_DEFAULTS, "photon_transfer",
  …)`), which the naming gate flags once `photon_transfer` is retired. Not renamed here: the table's name is §7's
  versioned-converter step, and a name that is neither the algorithm's nor the old one would be a third spelling.
- 2026-11-23 (Core agent): `map_estimate` calls `f` once for a concrete `u` (the lead's draft called it twice).
- 2026-11-23 (Core agent): `lo_estimate` keeps its keyword `iterations` (the LO contract's caller-side count) and
  records it as `stats.nsteps`.
- 2026-11-23 (Core agent): Core's `BundleMetadata.residuals` (per-observation residual norms) is NOT moved to `resid`
  here: its producers are Estimation's bundles, whose migration decides it with `RefineMetadata.residuals`.
- 2026-11-23 (Core agent): no `check_algorithm_traits` gate exists yet; the docstrings that named it now describe the
  contract, and Core's suite checks every Core algorithm's `estimator_kind` (solve_interface_tests.jl).
