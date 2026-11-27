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
  - `init(prob, alg; kw...)` = `check_kind`, then `EstimationCache(__init(prob, alg; kw...), prob, alg)`: Core's
    `solve!(::EstimationCache)` runs the owner's `solve!(cache)` and records `prob`/`alg` (`with_problem`), and its
    `reinit!` forwards to the owner's and records the new problem — one recording place for both paths.
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
| `tls_line(points[, weights])` | `solve(EstimationProblem(Line2D, points), TotalLeastSquares())`; weighted: the points' `weights` attribute column, `EstimationProblem(Line2D, withattrs(points; weights))` (as Kabsch, NearestPoint and DLT read theirs) |
| `direct_ellipse(points)` | `solve(EstimationProblem(Ellipse, points), DirectEllipse())` |
| `nearest_point(lines)` | `solve(EstimationProblem(Point, lines), NearestPoint())` |
| `triangulate(pose, f₁, f₂) -> (X, λ₁, λ₂)` | `solve(EstimationProblem(Point3, f₁ => f₂, pose), Triangulation())`: `u = X`, `metadata.depths = (λ₁, λ₂)`. Inner loops call the private `_triangulate` |
| `p3p(model, cs)` | `solve(MinimalProblem(Extrinsics3, cs, 1:3, model), P3P())`: the context `p` is the camera, which back-projects the pixels |
| `solve(solver, cs, idx)` | `solve(MinimalProblem(M, cs, idx), solver)`, or `solve(MinimalProblem(M, cs, idx))` (the declared default) |
| `photon_transfer(groups, darks; defects, block, normalize, significance)` | `solve(EstimationProblem(PoissonGaussian, groups, (; darks, defects)), PhotonTransfer(; block, normalize, significance))`; the `ReadNoise`-dict form: `p = (; darks, sensor_model)`; the single-mosaic form: `EstimationProblem(PoissonGaussian, mosaic, (; sensor_model))`, its `u` the `PoissonGaussian` |
| `master_dark(frames; exposure_band, false_defects)` | `solve(EstimationProblem(Dark, frames, (; exposure_band)), MasterDark(; false_defects))`: `u` the `Dark`, `metadata = (; read_noise, defects)` |
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

- **PoseLib:** `LevenbergMarquardt(PoseLibSolver(); optimization_strategy)` (Core's one host LM with PoseLib's
  backend; coordinator's and owner's rulings, §9) on Core's `RefineProblem(cs, guess[, (; K1, K2) | (; K)])`, the guess a
  `HomographyMat`, `FundMat` or `EuclideanMap{3}`, its budget the solve keyword `stops = LmStops(; max_iter)`; Core
  forwards it to the backend contract generic `lm_backend_refine`, which PoseLib implements. The backend kernels
  implement Core's contract generics, incl. `essential_8pt`. The raw C++ result goes in `original`.
- **Eval:** `fit_prior(spec, σ²s)` → `solve(EstimationProblem(IGPrior, σ²s), spec)`. `PriorSpec` is the algorithm
  (`ClosedFormKind`).
- **Features:** the internal callers migrate. `detect_image_circle` keeps its name (a task verb). Its internals call
  the new circle and ellipse finds.
- **IO:** `fit_record_json` and `fit_record_from_json` are deleted with no named replacement (owner ruling
  2026-11-25, §9: a `fit_result_*` name would need an R15 exception, and there are none). A `FitResult` goes through
  IO's style codec, as every embedded value: `JSON.lower(::VgioStyle, r::FitResult)` and `JSON.lift(::VgioStyle,
  FitResult, record)`; a kind's body holds the result itself. The persisted record keeps its schema keys
  (`iterations` is `r.stats.nsteps`); a version step would map them if the schema changed.

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
- 2026-11-23 (Core agent, owner's rulings relayed by the lead): naming.md item 5 now reads: "`update_model!` is the
  model-block update of an iterative algorithm, with weights and nuisance held fixed: its method of the
  block-coordinate-descent hook (items 3 and 8), public only as that hook; the computation inside it is internal,
  named for what it computes (its local Δ is item 6's `<method>_increment`). No identifier contains the word `step`,
  except Manopt.jl's `step_solver!` and SciMLBase's iteration counter `nsteps`." Reason: with no `fit_*` names the
  `<fit>_step` form has no anchor; `update_model!` is already the hook items 3 and 8 name, so no new vocabulary is
  added. R5 follows: its `<fit>_step` exception (`_is_fit_update` and the `fits` set it read) is deleted, and
  `nsteps` is exempt in every binding (`_SCIML_COUNTER_NAMES`, beside Manopt's `_MANOPT_CYCLE_NAMES`). Consequence
  downstream: Estimation's `division_camera_step` (internal or not) must be renamed by what it computes.
- 2026-11-23 (Core agent): the mechanism-token scope is written into naming.md ("Mechanism tokens name steps, never
  estimators"). Core's `check_naming` has NO mechanism-token rule at this commit (no rule fires on `lm`/`gn`/`irls`), so
  there was no check to scope; the rules agent that adds one implements the scope (estimators: problem and algorithm
  types and estimator functions; contract steps exempt: `posterior_irls!`, `lmsolve_chart!`,
  `block_coordinate_descent!`).
- 2026-11-23 (Core agent): `mad_scale` → `solve(EstimationProblem(NoiseScale, residuals), MedianNormScale())`
  (`ClosedFormKind`, `n` the residuals, the numerics `_mad_scale_norms` unchanged). `NoiseScale` declares NO default:
  the old docs made reaching for this estimator "a decision rather than a default", so it stays named. The five
  internal names are `_partition_student_t`, `_next_nu`, `_posterior_weights!`, `_posterior_inlier_probs!`,
  `_soft_posterior_weights!` (an `_` prefix: the stack's mark of a non-public name, which the private-name gate
  enforces across packages); all six old names are in RETIRED_NAMES.
- 2026-11-23 (Core agent, owner's ruling on the R9/§7 collision): an algorithm's snake-case tag is its ONE tag. R9 exempts
  a retired-name hit that equals `snake_case(T)` (the tag rule's function) of an estimation-algorithm type the scanned
  package defines (`struct T … <: AbstractEstimationAlgorithm`, or `<: MinimalSolver{…}`, found in the parsed code: no
  hand-written list), where a tag is spelled: anywhere on a TOML line (table, key, tag value), and in a Julia file only
  when every hit on the line is inside a string literal (a reader of the tag's table). Reading taken: the narrow one —
  every other R9 coverage (identifiers, comments, docstrings, docs, a string that is not an algorithm's tag) is kept, as
  conventions/naming.md item 9 states; if the ruling meant to drop R9 from comments and strings altogether, that is a
  one-line widening of `_algorithm_tag`.
- 2026-11-23 (Core agent): R21, on the parsed AST (Meta.parseall, as R3–R14): it flags (a) a call of `estimate`,
  `iterations`, `converged`, `retcode`, `metadata` or `stop_rule` with exactly one positional argument (qualified or
  not), and any call of `nested_metadata`; (b) a definition of any of those, or of `rms`, whose positional argument is
  annotated `::FitResult`; (c) a read `x.metadata.<f>`, `f` in `weights, dof, scale, scale_source, s2, inlier_mask`,
  outside a method of `weights`, `dof`, `noise_scale`, `scale_source`, `s2`, `inlier_mask` or `weight_kind`. It cannot
  see types, so two retired forms are out of its reach and stated: a call `rms(r)` on a result (the same syntax as the
  kept `rms(residuals)`), and a dotted `r.converged`/`r.iterations` (the same syntax as a NamedTuple's field, e.g. the
  multi-block LM's `sol.iterations`); both fail at run time anyway (no method, no field), since a FitResult forwards
  nothing.
- 2026-11-23 (Core agent, decision 9): the two last-bit tests compare within a stated tolerance — `inv(::Uncertain
  {EuclideanMap})`: the rotation exactly, the translation within 4 ulp of its norm (StaticArrays' `muladd`s fused by LLVM
  in one compiled copy and not another); PhotonTransfer of one frame vs 0.10.0's printed numbers: each within 4 ulp
  (the weighted line's LAPACK inverse under another BLAS/LAPACK build). Nothing else loosened.
- 2026-11-23 (Core agent, simplify review): `init` returns a Core `EstimationCache(cache, prob, alg)` (mutable, its
  `prob` replaced by `reinit!`); `solve!` on it records the problem and algorithm through `with_problem`, as `solve`
  does (`with_problem(x, prob, alg) = x` for a non-FitResult replaces `_record`). A package's `__init`/`solve!`/`reinit!`
  methods stay on its own cache type.
- 2026-11-23 (Core agent): ONE FitResult constructor, `FitResult{U}(u, code, rule, stats, rms, n, metadata; resid, prob,
  alg, original)` (and `FitResult(u, …)` taking `U` from `u`), and ONE rebuild, `_rebuild(r, U, u, prob, alg)`
  (`with_problem`, `map_estimate`). The device LM calls `FitResult{U}(…)` without keywords: a positional call, as a
  kernel runs it.
- 2026-11-23 (Core agent): context `p` is read by ONE helper, `_solve_context(p, required, optional, who)` (any key
  order; one `ArgumentError` naming the missing and the unknown keys; `NullParameters()` holds none): Zhang
  (`center`, optional `pitch`), MasterDark (`exposure_band`), PhotonTransfer (`darks` with `defects` or
  `sensor_model` by the darks' type; one frame: optional `sensor_model`). Algorithms that read no context take only
  `NullParameters` (`EstimationProblem{M, D, NullParameters}`, `_MinimalNoContext` for the minimal solvers): a stray
  context has no method. Zhang's default is declared only for the contexts it solves (`_ZhangContext`). A Zhang
  problem without its centre is now an `ArgumentError` (was a `MethodError`).
- 2026-11-23 (Core agent): the kind law's two minimal refusals were NOT merged over a `Union`: the merged method is
  ambiguous with `_check_kind(::RefineProblem, ::EstimatorKind)` and `_check_kind(::EstimationProblem, ::EstimatorKind)`
  (the ambiguity that split them); both are one line calling `_refuse_minimal`.
- 2026-11-23 (Core agent): the algorithm docstrings are the one copy of their documentation; the private kernels carry
  a `# Kernel of [Algorithm]` comment (the triangulation kernel also keeps its recorded study and negative result).
- 2026-11-23 (Core agent): `check_algorithm_traits(M)` (TestExt; public stub) is the decision-7 gate, run by Core's
  quality item: every concrete algorithm has `estimator_kind`; every declared default is not a find, overrides no
  `supports_solve_rng`, is a refine exactly for a `RefineProblem` and a minimal solver exactly for a `MinimalProblem`,
  and is a closed form wherever a closed form of the module solves the same problem (from inferred results: no
  problem is built). The no-default message is held by solve_interface_tests.jl.
- 2026-11-23 (Core agent, decision-13 gate on the device LM): the front end is `@inline` (`solve(prob, alg; kw...)`
  and `with_problem` on a FitResult), so a caller's `solve(RefineProblem(fj, x₀; lb, ub), StaticLevenbergMarquardt();
  stops, bound)` inlines down to the kernel: the kind law and the termination record fold, and the fields a caller
  does not read (the recorded problem, `rms`) are never built. Measured: the old `static_lm` vs the new `solve`,
  interleaved processes, medians — the lens inverse within +0.7 % (full result) and +0.9 % (reading `u`, the record and
  `J`); the N = 3 smooth problem within +0.6 % / +0.3 %; the old runs' A/A spread 1.1 %. Before the inlining the full
  result cost +3–7 % (a 288-byte result materialized through a non-inlined `solve`).
- 2026-11-23 (Core agent, code review): Core's version is 0.14.0 now: a sibling that still names a deleted function
  fails to resolve rather than load against a broken Core, and gets its compat bump when it migrates. PoseLib's
  `essential_dlt` → `essential_8pt` rename is the PoseLib phase's.
- 2026-11-23 (Core agent): `reinit!(::EstimationCache, …)` builds the new problem first and refuses one of another
  concrete type (`ArgumentError`) before the owner's `reinit!` runs, so a refusal leaves both sides as they were.
  `reinit!` is public, not exported (SciMLBase exports its own).
- 2026-11-23 (Core agent): Triangulation of parallel rays is recorded `(InitialFailure, NoRule)`, `n = 0`, its `u` the
  kernel's NaN point (the one degenerate closed form whose `u` is not `nothing`: the type stays a concrete `Point3`
  and the golden mapping of `u` holds).
- 2026-11-23 (Core agent): `default_algorithm(::RefineProblem{<:AbstractConstraintProblem}) = LevenbergMarquardt()`
  (declared beside `AbstractConstraintProblem`, robust/problem.jl): only a refine whose `f` is a constraint problem —
  the chart-LM interface's implementers — has the host LM as its default; a residual closure has none, and its
  no-default error names `StaticLevenbergMarquardt`. The refine pipeline's move into Core completes this with the LM's
  `__solve`.
- 2026-11-23 (Core agent): the device LM's `stats.nsteps` counts ACCEPTED steps (SolveStats' meaning, the host LM's
  count), by a counter of its own; the loop's `iter` and its cap are unchanged.
- 2026-11-23 (Core agent): `MasterDark`'s `u` is the `Dark`, `metadata = (; read_noise, defects)`, so
  `r.u isa model_type(r.prob)`.
- 2026-11-23 (Core agent): TotalLeastSquares' weights are the points' `weights` attribute column
  (`withattrs(points; weights)`, read by `weights_column`, as Kabsch, NearestPoint and DLT read theirs; `withattrs` of a
  plain point vector is a `PointSet`); the `(; points, weights)` form is gone.
- 2026-11-23 (owner, dedupe agent): ONE storage-boundary policy for the frame-tagged maps (`EuclideanMap`,
  `SimilarityMap`, `RotationMap`, `HomographyMap`), implemented once at the end of `homography_map.jl`: `convert` into a
  fully parameterized slot of the same map type (implicit in `setindex!`, a typed field, `push!`) may LABEL an
  unlabeled map or keep its frames, never relabel or erase one, and keeps the number type `T` — a different `T` is a
  `MethodError`, as `EuclideanMap`'s always was (`SimilarityMap`'s used to convert it). Reason: the Float32-storage /
  Float64-statistics boundary; a change of precision is explicit (`HomographyMap{N, T}(h)`, the maps' constructors):
  HomographyMap's eltype-changing `convert(HomographyMap{N, T}, h)` is removed (its one user was a test, now pinning
  the refusal); every map refuses with the same `MethodError`.
  `relabel`/`unlabel` of the four maps, plain or `Uncertain`, are one method each. The 4-parameter `EuclideanMap{N, T,
  R, V}` convert (a frame-preserving shim with no caller) is removed.
- 2026-11-23 (code review before the Core 0.14 commit): no default is declared for a problem no method solves yet or
  where a default would guess. `RefineProblem` of a constraint problem: `LevenbergMarquardt()` is declared with its
  `__solve` by the refine pipeline's move. `MinimalProblem(RadialCamera, …)`: none, because the pencil's centre is
  the problem's geometry and `(0, 0)` is a guess. `ScaleMAP` reads its sums through `_solve_context` (any order, one
  naming error). `Triangulation` is declared for a plain (unitless) pose only, the kernel's domain. R9's algorithm-tag
  exemption covers a TOML table, key or value, never a comment, and in Julia only a whole string literal, never
  prose. The `core_lie` bits are pinned by the golden replay from the Estimation milestone on; until then the suite
  pins nothing for them (coordinator, 2026-11-23).
- 2026-11-23 (Estimation agent): THE REFINE MOVE. Core's `robust/refine.jl` holds the one host LM refine:
  `__solve(::RefineProblem{<:AbstractConstraintProblem}, ::LevenbergMarquardt; scale, width, stops = LmStops())` over
  the CHART CONTRACT — `model_chart(f, model)` (the chart at a model; was Estimation's `_chart ∘ _as_model`),
  `chart_rows(f, chart, √w)` (the rows the LM descends; was `_triggs_state`), `chart_model(f, chart)` (was `_model_at`),
  `chart_size(f) -> (rows, Val(dof))` (was `_lm_size(problem)`), `guess_model(f, u0)` / `solution_model(f, θ, u0)` (the
  guess into the inner loop's model and the refined model back into the guess's form: a physical guess's metres, a
  labelled guess's frames; defaults `convert(model_type(f), u0)` and `θ`), `default_optimization_strategy(f)` (what
  `ClassDefault()` resolves to) — all contract generics (CONTRACT_GENERICS). Not `chart` (MappedSurface's). Moved from
  Estimation with their numerics unchanged: `_refine_pipeline`, `_refine_spec`, `_refine_run`, `_refine_too_few`,
  `_ZeroWeightRows`, `RefineMetadata` (now `<: AbstractFitMetadata{L}`, its `residuals` moved to the result's `resid`),
  the scale input (`bundle_loss`, same public name, now Core's), `_chart_refit` → public `chart_refit(ws, f, ch, rows,
  weights; robustifier, stops, damping)`, `_triggs_lm` → public `triggs_refit(ws, shape, f, model, loss; weights, stops,
  damping)`, `_lm_workspace(ws, rows, Val)` → public `lm_workspace` (the one reuse policy). They are public because
  Estimation's local optimizations call them (the private-name gate forbids `VgtkCore._x`). Also public, each the one
  copy: `normalized_loss_weight(shape, s, c²)` (was Estimation's `_irls_weight`), `loss_weights` (was `_loss_weights`),
  `weighted_support(w, sq)` and `weighted_support_rms(w, sq)` (were Estimation's `_weighted_support(_rms)`),
  `recorded_scale(input)` (was `_recorded_scale`), `sq_loss_width(::Nothing) = NaN` (replaces `_sq_loss_width`), and
  `read_context` (was Core's `_solve_context`; Estimation's fits read their keyed contexts through it; not `solve_context`: R19). A `RefineProblem`
  with `lb`/`ub` is refused by the host LM (`ArgumentError` naming `StaticLevenbergMarquardt`). The data hold fewer than a
  minimal sample: the guess back unrefined, `(InitialFailure, NoRule)`, `resid` NaN, as `_guarded_refine` did.
- 2026-11-23 (Estimation agent): THE BLOCK CONTRACT. `AbstractBlockProblem` (Core) is the parent of a multi-block refine's
  `f` (the bundles, the camera regression's refine, the relative-pose bundle); `__solve(::RefineProblem{<:AbstractBlock
  Problem}, ::LevenbergMarquardt; scale, width, stops = default_stops(f))` parses the strategy and scale input
  (`bundle_loss` at `codimension(f)`, 2 by default) and calls the author's `block_refit(f, u0, loss, stops, damping)`,
  which builds its blocks and variables and solves them by Core's `block_lm(blocks, vars0, unknowns, loss, stops;
  damping)` (Estimation's `_solve_bundle`, moved, numerics unchanged; `solve_bundle` would break R19), the blocks
  robustified through the contract generic `with_robustifier(block, sqrt_weight, robustifier)` (was `_with_robust`).
  `default_stops(f)` (a contract generic, default `LmStops()`) carries a class's stops of record. `default_algorithm` of
  both refine forms is `LevenbergMarquardt()`.
- 2026-11-23 (Estimation agent): `AbstractFitMetadata{L}` (Core, `L` the producer) with ONE `weight_kind` method; the
  producer of the posterior-weight iteration is `PosteriorWeightState` (`weight_kind(::Type{<:PosteriorWeightState}) =
  PosteriorWeights()`), a least-squares fit's `Nothing`. RansacMetadata{T, S} <: AbstractFitMetadata{S}; its
  `residuals` moved to the result's `resid`.
- 2026-11-23 (Estimation agent): `Ransac([T = Float32]; scoring = ClassDefault(), local_optimization_strategy =
  ClassDefault(), config = RansacConfig(), score_guard = false, on_improvement = nothing, on_hypothesis = nothing)`. The
  storage precision `T` is the optional leading positional type (signatures.md: a type is never a keyword); it replaces
  the old `workspace = RansacWorkspace(T, …)`. The trace hooks and the score guard are configuration (fields). The
  width keywords `threshold`/`outlier_halfwidth` and the retired `local_optimization` are still refused by the
  constructor with the way to say them (`ArgumentError`, as the find pipelines did). `ClassDefault()` resolves through
  the problem's class (`_problem_class`: `[pipelines.<class>]`); a problem without a class has no default scorer
  (`ArgumentError`) and its default strategy is the scorer's (`default_local_optimization_strategy(scoring)`, the old
  `ransac` default). `max_trials`/`min_trials` are `config`'s only.
- 2026-11-23 (Estimation agent): a find's context is `p = (; extent)` (or `NullParameters()`); any other key is an
  `ArgumentError`, and so is an extent on a problem whose model class has no extent check (circle, ellipse, F, E,
  rotation, hand-eye, lines: was a `MethodError` of the find, which took no `extent`). A physical problem's extent is a
  physical rectangle (stripped in metres); a plain one with a physical problem, or the reverse, is a `MethodError` (was a
  keyword `TypeError`).
- 2026-11-23 (Estimation agent): the family problems hold ANY number of data: the constructors' "Need at least K
  correspondences" refusal is gone, so a solve on too few data returns the find's `:no_model` (no trial drawn) and the
  refine's unrefined guess, bit-identical to the old guarded pipelines. Units: `AbsolutePoseProblem`, `BearingPoseProblem`
  and `HandEyeProblem` accept physical data (lengths), strip them in metres once at construction and record the unit (a
  field of their context/problem); a find's model comes back in metres (`_with_units`), a refine's through
  `solution_model` from a physical guess; a plain guess with a physical problem (or the reverse) is a `MethodError`.
  The plane problems (`HomographyProblem`, `RadialHomographyProblem`, `RadialCameraProblem`) refuse physical plane points
  at construction (was the find/refine wrappers'). The bearing pose's guess-coincidence refusal is `guess_model`'s.
- 2026-11-23 (Estimation agent): `LocalOptimization(strategy; scoring, score_guard = false)` is a `RefineKind` on
  `RefineProblem(f, θ₀[, (; extent)])`, `θ₀` a model or a `FixedModels`; `init` gives the same `RansacCache` as a find
  (`reinit!(cache[, θ₀]; f, p)`). The cache's `solve!` returns a result whose `resid` and weights ARE the cache's buffers
  (the old `ransac!` contract); a fresh `solve` builds fresh ones. The inner sample solve is `sample_models(problem,
  idx)`, through Core's `solve(MinimalProblem(model_type(solver), data, idx), solver)`.
- 2026-11-23 (Estimation agent): `pooled_noise_scale` → `PooledScale()` (`ClosedFormKind`) on `EstimationProblem(NoiseScale,
  results)` or `EstimationProblem(NoiseScale, scales => dofs)`; same numbers and record.
- 2026-11-24 (owner): SUPERSEDES the next entry. The math is deterministic: a kernel whose bits depend on JIT vs pkgimage
  compilation is a defect. Core's workload entries are restored. The comparison (the golden replay with the workloads, against `--pkgimages=no`) found ONE kernel: the
  homography's composition back from the normalized frame (`∘` of an affine or `HomographyMap` with a `HomographyMat`,
  `geometry/forms.jl`), a StaticArrays `SMatrix * SMatrix` whose `muladd`s LLVM fused per entry differently in the two
  copies (the JIT copy: `A ∘ H` fuses the last term of columns 1 and 3 and none of column 2, `H ∘ B` the middle term of
  entry (2, 2) only; the pkgimage copy: the last term of every entry). Each entry's arithmetic is now stated
  (`_collineation_left`, `_collineation_right`: `fma` where the records fused, `*` and `+` where they did not), which
  reproduces the recorded bits in both modes. The pattern is the records', not a design: a later change to it is a
  numeric change with intended-change entries.
  STOPPED, NEEDS THE OWNER (2026-11-24): an Estimation workload (the homography find and refine: first find 107 ms
  against 5.8 s, first refine 0.3 ms against 1.5 s, Estimation's load +50 ms) moves the homography chart LM's Jacobian
  rows (Core `base/sampson.jl` `_jacobian_terms`, the codimension-2 path: `L⁻¹·Gθ`, the factor rate,
  `_along_columns(·, B) = A·Bᵀ`) in its pkgimage copy: 167 homography, 146 conic-group and 11 noise-scale cases leave
  their records (`find/h_scene_120+60/lo_classdefault/seed1`: recorded estimate digest 91c59225…, pkgimage 5a46dd9f…;
  the rows' digest d4bf4c5d… against the JIT copy's ba24b230…). That kernel is shared by every Sampson class (F, E,
  circle, radial camera, rotation, PnP), and the records hold each class's own JIT lane pattern, so stating one pattern
  in the shared kernel is not a clean pin. Options: (a) no Estimation workload (decision 13's gate holds without it:
  first find −17 %, first refine −25 %, loads +0.2 %); (b) state the kernel's arithmetic uniformly (`fma` throughout)
  and re-make the LO/refine leaves of every Sampson class as intended changes; (c) both workloads, and the records
  re-made under them (bits that then depend on what a package image holds — against this ruling).
- 2026-11-23 (Estimation agent), NEEDS THE OWNER: decision 13's PrecompileTools workload broke bit identity. Core 0.14's
  `precompile.jl` compiled the DLT, minimal, Kabsch and TLS entries into the pkgimage, and that copy fuses StaticArrays'
  `muladd`s differently from the JIT copy the golden records were made with (verified: `--pkgimages=no`, or the workload
  removed, reproduces the records bit for bit; with it, `DLT` on `htrue_100+30` differs in the last bits). Those
  workload entries are removed (the rest of the workload is unchanged), and Estimation has no workload. A workload that
  covers golden-pinned arithmetic needs either the records re-made under it (an intended change of every leaf it moves)
  or the kernels rewritten with explicit `fma` (`_fma_dot`), which is itself a numeric change: the owner's call.
- 2026-11-23 (Estimation agent): golden framework: in `compare_record` a rename takes precedence over a `new_only` prefix
  (`stats` is new, `stats.nsteps` is renamed onto `iterations`), and a `retyped` STRUCT is compared as "still a struct"
  (its fields one by one; the old test compared the type name and could never pass). `replay!` gained `fixture_view`
  (the core-replay agent) for fixtures that hold arrays of upstream results.
- 2026-11-23 (Estimation agent): the triangulation kernel is public inner-loop machinery, `triangulate_bearings(pose, f₁,
  f₂) -> (X, λ₁, λ₂)` (`@inline`), beside `homography_dlt!`: the relative-pose bundle seeds its points with it inline, as
  it called the public `triangulate` at 3d8a06da (the LLVM fusion of its `muladd`s follows the caller; a call through the
  `@noinline` copy `Triangulation`'s solve keeps for its own bits moved the bundle's last bits). `s2(r::FitResult)`
  (public, a contract generic beside `dof`) is the one reading of a recorded residual variance (R21 lists it).
- 2026-11-23 (Estimation agent): the chart and block contract generics have throwing fallbacks (an `ArgumentError`
  naming the missing method), so a call on an `f` that implements none is a stated refusal and JET's no-method reports
  on Core's generic refine body stay out of the bound.
- 2026-11-23 (naming-rules agent, relayed): R15 covers a package's own `src/`/`ext/` (functions defined, exported or
  declared public beginning `fit_`/`find_`/`refine_`; `_` internals exempt; `TASK_VERBS` = `find_boards`, `find_board`,
  `find_board_cameras`, each with its reason); R19 is literal (bindings beginning `solve_`; `_` internals and quoted
  symbols exempt) — so Core's keyed-context reader is `read_context`, and four Core test locals were renamed
  (`linear_condition`, `constant_at_efficiency`, `p3p_allocated`, `static_lm_rays!`); R20's algorithm types are computed
  from the code; R16–R18 run in `check_algorithm_traits`; a retired name ending in `(` is matched without a right word
  boundary (`ransac(p, s)` is caught, the word and `Ransac` are not).
- 2026-11-23 (Estimation agent): whole-case intended changes. `intended_changes.toml` entries with `field = "*"` state a
  case whose outcome changed KIND (an exception now a value, or the reverse): each side is `{type = "exception", value =
  <type>}` or `{type = "record", value = <SHA-256 of the whole record>}`, both exact (`replay.jl`'s
  `whole_case_change`; `ci/golden/digest.jl` prints the digests). Used for: a DLT index mask with repeated in-range
  indices (now a valid selection, spec's mask ruling) and a `RotationProblem` of bearings without their Jacobian
  columns (refused by the constructor whatever the count; the old guard returned before building it).
- 2026-11-23 (camera-fits agent, relayed): ONE `f` for every camera-from-camera regression, `CameraRegressionProblem(target,
  source, region; support, fixed, weight, device) <: AbstractBlockProblem` (no target for the refine, `OpticalAxis` and
  `ForwardSpline`); `support`/`fixed`/`weight`/`device` are problem settings (refused at the solve where a target does
  not read them), the spline-target knobs `CameraRegression`'s fields (`tolerance`, `grid_size`, `lattice_spacing`,
  `forward`; `nothing` = not stated). Algorithms: `CameraRegression` (fit), `PSpline(; grid_size)` (closed form),
  `ForwardSpline(; tolerance)`, `OpticalAxis()`, `RadialCameraFit(; iterations)` (the plan's `WhitenedIterations`),
  `DivisionCameraFit(; ncoef, weighting, mixture_em, verbose)` on `EstimationProblem(CameraModel, (radial_cameras,
  views), (; image_size, focal_seed, k))`. The refine of a regression is `LevenbergMarquardt` through the block contract
  (least squares only; `default_stops` from `[camera_regression_problem]`). `approximate_camera` is deleted (a fit, then
  its refine); the deferred fit/refine duplicate is one helper (`_bounded_part`). The camera-and-poses bundles are
  `CameraPosesProblem(correspondences; fixed, box, held_boards, held_views, residual) <: AbstractBlockProblem`, the guess
  a NamedTuple of the result's form (`(; camera, poses[, board_poses])` or `(; camera, views::RotationCenterViews)`), the
  LM settings `stops` (`[camera_poses_problem]`). `resid` stays `nothing` for these results (quadrature samples; signed
  rows, D12); the bundles' metadata `s²`/`ν` are `s2`/`dof`, read by `s2(r)`/`dof(r)`. `n_without_bearing`,
  `without_bearing`, `without_projection` are retired (read `r.metadata.<field>`; field names, not RETIRED_NAMES).
  `division_camera_step` is internal (`_keyed_focal_solve`, the inner `_stacked_focal_solve`: no `step`). TOML tables
  renamed to the algorithm's or problem's tag: `[division_camera_fit]`, `[camera_poses_problem]`,
  `[camera_regression_problem]` (`[optical_axis]` is `OpticalAxis`'s tag). Source and test files renamed so no file name
  holds a retired name.
- 2026-11-23 (other-fits agent, relayed): ONE `ReweightedLeastSquares(; optimization_strategy = ClassDefault(), tol)` for
  the fixed-scale reweighted fits (a rotation on `EstimationProblem(RotMatrix{3}, cs)`, a rigid alignment on
  `EuclideanMap{3}`, a linear model on the tag `LinearModel` with `(A, y[, prior])`, the robot-world calibration on
  `EstimationProblem(RobotWorldHandEyeCalibration, markers, (; σ_rot, σ_trans, min_rel_angle))`) — `Irls{K}` cannot be
  the algorithm (it is an `AbstractOptimizationStrategy`), and the plan's separate robot-world fit is the same algorithm;
  `scale`/`width` are solve keywords; the class tables `[reweighted_least_squares.<rotation|linear|rigid_alignment>]`.
  `RotationAveraging(; seed, …)` and `FilteredRotationAveraging(; cycle_tol, orientation_tol)` on
  `EstimationProblem(AbsoluteRotations, graph)` (the filtered graph in `metadata.graph`); `ResidualMixture(; weighting)`
  on `EstimationProblem(PosteriorWeightState, distances, (; model_dof))` (the mechanism-token rule: not `MixtureEM`);
  `SimFitPP` (`supports_solve_rng = true`) on `EstimationProblem(LossWidth, …)` and the tag `PoolSelection`;
  `RobotWorldHandEyeProblem(camera, markers)` and `RelativePoseProblem(K1, K2, cs)` are block problems (the latter's `u`
  is `nothing` on failure; it refuses a robust strategy and a damping other than Marquardt). The fits' residuals stay in
  their metadata (`resid = nothing`). The `_degenerate` trio is one `_reweighted_failure`. `fit_line` is internal
  (`_heteroscedastic_line`). `irls_termination`'s keyword `solve_failed` is `linear_solve_failed` (R19; BlobBoards calls
  it). The tests' shared helpers live in `test/fixtures/reweighted_fits.jl`.
- 2026-11-24 (Estimation agent): decision 11, versioned configurations. ONE converter per package: Core's
  `ConfigVersions(package, migrations::ConfigMigration...)` and `migrate_config(d, versions; source)`
  (`base/config_toml.jl`), each package's converter constant (Core's `CONFIG_VERSIONS`, Estimation's `ESTIMATION_CONFIG_VERSIONS`: one name per owner) beside its `configs/default.toml`, which states
  `config_version` (absent = 1: the files before versioning). `parse_defaults(path; versions)` reads the shipped file
  through it, so the code sees the current table names only. A migration lists every table of the version it reads (an
  unknown one is refused, naming it) and renames whole tables by dotted path; a current file holding an earlier name is
  refused naming the current one. Core: version 1, no migration. Estimation: version 2, one migration (the eight
  estimator tables, §9's fits entries). The version-1 table names are stored-format tokens in `check_naming`'s
  `FOREIGN_FORMAT_TOKENS` (accepted in string literals and in `test/fixtures/stored_configs/`, the converter's real
  stored inputs: Core 0.13.3's and Estimation 0.12.2's shipped files, comments dropped).
- 2026-11-24 (Estimation agent): `rand(rng, Line2D)` / `rand(rng, Plane3D)` go through Random's sampler interface
  (`rand(rng, ::SamplerType{Line2D{T}})`, `Sampler(…, ::Type{Line2D}, …)`), never a `rand(::AbstractRNG,
  ::Type{Line2D})` method, which is ambiguous with every RNG package's `rand(rng::R, T::Type)` (CUDA's). A sampler takes
  no keywords: `max_rho` and `bounds` are gone (no caller in the stack or its downstream snapshots), the ranges fixed at
  their defaults ([0, 1), [-1, 1)).
- 2026-11-24 (owner): no public `mask` anywhere. A hard selection is `cs[idx]`, a soft influence the `weights` attribute;
  repeated data is valid (a duplicated row acts as weight 2 in a least squares; conventions `data-arguments.md`).
  Index lists exist only inside the inner kernels (`homography_dlt!`, `fundamental_dlt!`, the RANSAC gates), which keep
  refusing one longer than their buffer: the three `fundmat_dlt/guard_8/refused_long_index_mask` cases replay through
  `fundamental_dlt!` (`DimensionMismatch`, as recorded) and their whole-case entries are gone.
  `CorrespondenceNormalization(M, cs; mask)` and `AffineMap(::HartleyNormalization, points, mask, w)` lose the mask.
- 2026-11-24 (owner, option (b) of the STOPPED entry above): the Sampson Jacobian states its arithmetic. Every
  product `_jacobian_terms` reaches (`L⁻¹·Gθ`, `L⁻¹·g`, `G_x·Σ·G_xᵀ`, the factor rate's `W·K`, the chart directions
  `A·Bᵀ` and `B·v`, the default `_normal_rate` as `_fma_tmul`, the essential rate's `M·∇`) is the explicit `fma` chain of
  `base/fused_products.jl` (`_fma_mul`, `_fma_tmul`: the one owner of stated products, StaticArrays' order); the
  Estimation workload is restored. The pinned code is identical in pkgimage and `--pkgimages=no` mode (all 4598 replay
  cases and their fixtures). 683 records move (cases, and fixtures holding an upstream find as `<case>#fixture`), 1554
  leaves, each an intended change with its case's size (max relative difference over its float leaves against the
  unpinned code): 523 at most 1e-10 (425 at most 1e-12), 95 in (1e-10, 1e-6], 39 in (1e-6, 1e-3], 25 above 1e-3, one
  trace that changed length; 28 change a discrete leaf (an iteration count, a stop rule). The large ones are the
  essential refines on `stereo_ecam_140+60`/`epix_cam800` (the model's sign representative flips around a ~5e-10 move,
  or the refine stops at another iterate) and the rotation averagings on `edges5_*` (an ill-conditioned five-edge graph
  carrying the last-bit change to a different solution). ACCEPTED (owner, 2026-11-24): the fma pin and its 683 listed
  records, not re-recorded; the unfused form and every per-product mix were tested and reproduce fewer (the recorded JIT
  copy fused per lane); paper projects rerun within their documented tolerance (migration plan §5).
- 2026-11-24 (PoseLib agent, coordinator's naming ruling): NO `PoseLibLM`. PoseLib's refines are Core's one host LM with
  a backend type parameter, `LevenbergMarquardt{B <: AbstractSolverBackend, S, D}` (backend first: the review round), constructed
  `LevenbergMarquardt([backend = NativeSolver()]; optimization_strategy, damping = default_damping(backend))` — the
  design of `DLT(PoseLibSolver())`. `LevenbergMarquardt()` is unchanged (`NativeSolver()`, `MarquardtDamping()`), so
  every native refine's bits are too. Core's two refines of a problem (chart and block contract) take
  `LevenbergMarquardt{NativeSolver}` only (another backend on them has no method); the type is spelled
  in each signature, never aliased (R20).
- 2026-11-24 (PoseLib agent): `default_damping(backend)` is a Core contract generic (public, `CONTRACT_GENERICS`):
  `MarquardtDamping()` for `NativeSolver()`, `LevenbergDamping()` for `PoseLibSolver()` (PoseLib's method; its LM damps by
  λ·I, `BundleOptions`' `LEVENBERG`); a backend whose package is not loaded is an `ArgumentError` (one message,
  `_no_backend_lm`, shared with `lm_backend_refine`'s fallback; the review round, below). PoseLib refuses any other damping (`ArgumentError`).
- 2026-11-24 (owner, "Or our core types" / "We have types for all of these"; route (a) of the R16 finding): PoseLib's
  refines use NO new type. The problem is Core's `RefineProblem(data, guess[, p])` with the stack's model types —
  `RefineProblem(cs, HomographyMat(H))`, `RefineProblem(cs, FundMat(F))`, `RefineProblem(cs, pose::EuclideanMap{3}, (;
  K1, K2))`, `RefineProblem(bearings, pose)`, `RefineProblem(X => u, pose, (; K))` — the guess's type and the data's
  shape select PoseLib's residual. Dispatch follows the stack's backend pattern exactly: Core owns ONE `__solve(prob::
  RefineProblem{<:AbstractVector}, alg::LevenbergMarquardt; scale, width,
  stops = LmStops())`, which refuses a box and forwards to the new backend contract generic
  `lm_backend_refine(backend, u0, data, p, alg; stops, scale, width)` (public, `CONTRACT_GENERICS`, an `ArgumentError`
  fallback naming the backend, the `essential_8pt` pattern); PoseLib implements `lm_backend_refine(::PoseLibSolver, …)`
  and adds NO `__solve`, so `check_algorithm_traits(PoseLib)` (R16, now in PoseLib's quality test) holds. Reason: R16
  refused the earlier PoseLib `__solve` methods (11 violations measured: a package adds `__solve` only for algorithm types
  it owns, and `LevenbergMarquardt` is Core's); the minimal-solver kernels were sanctioned only by `CONTRACT_GENERICS`.
  A refine of raw data by the native backend is the fallback's `ArgumentError`. PoseLib's own methods refuse a stray
  context or another model with the tuple's `MethodError` (a PoseLib fallback, so Core's message never claims PoseLib
  lacks a refine). `PoseLibProblem` and `PoseLibStops` were never released: deleted, no RETIRED_NAMES entry.
- 2026-11-24 (owner): PoseLib's stops are Core's `LmStops`. PoseLib applies `max_iter` (default 1000, `LmStops()`'s) and
  REFUSES every other field that is "set": a field is set when it is not `===` to the same field of `LmStops()` (the
  defaults of record, read once into a module constant); the `ArgumentError` names it (`ftol`, `xtol`, `xtol_rel`,
  `stationarity_tol`, `max_time`, `residual_resolution`). A caller who passes a default explicitly is not told apart
  (it is the default). `max_iter` must fit a C int.
- 2026-11-24 (owner, approved intended entries): the refine's guess is typed, so a non-3×3 seed is refused where the
  `HomographyMat`/`FundMat` is built (`DimensionMismatch`): `poselib` `homography_refine/refused/seed_2x2` and
  `fundamental_refine/refused/seed_2x3` (were the refiner's `ArgumentError`).
- 2026-11-24 (PoseLib agent): `original` is the C wrapper's raw output: `(; iterations::Cint, H | F)` or
  `(; iterations, R, t)`, the return value (PoseLib's `BundleStats.iterations`) and the output buffers as written
  (`SVector`s, column-major). The wrapper returns no more of `BundleStats`; widening it needs a `PoseLib_jll` rebuild,
  not done. Below the minimal sample no C call is made and `original` is `nothing`, so a solve infers to a two-member
  `Union` of `FitResult`s (never `Any`); the suite's `@inferred FitResult` holds it. A pose refine returns the unlabeled
  `EuclideanMap(RotMatrix{3, Float64}(R), Vec3{Float64}(t); check = false)` (the C buffers' bits, unchecked as before).
- 2026-11-24 (PoseLib agent): `essential_8pt(::PoseLibSolver, cs)` is PoseLib's method for bearing correspondences
  only; any other data (calibrated 2-vectors) is the pair's `MethodError`, thrown by a PoseLib fallback method — the
  recorded `MethodError` replays. Its too-few message names `DLT(PoseLibSolver())`.
- 2026-11-24 (owner, R20 scope): check_naming R20 also refuses a `const` whose name takes no parameters and whose right
  side names ONE concrete instantiation of a stack type: the type applied to as many parameters as it has free (every
  free variable of an alias's body counted, so `PlainEuclideanMap{3, Float64}`, whose `R` and `V` stay free, is a
  UnionAll and no hit), none bounded (`<:`) and no `where`. Parametric aliases with free variables stay
  (`Line2D{T}`, `PlainEuclideanMap{N, T}`, `CameraExtrinsics{T, R, V}`, `HomographyProblem`, …). The stack types are the
  scanned package's own declarations (parsed) and every type of a loaded stack package. Regression tests in Core's
  `quality_naming_tests.jl`. Every hit is fixed, with no allow-list: public `Pinhole` (→ `Lens(Perspective())`, type
  `Lens{Perspective, Nothing}`), `Extrinsics3` (→ `CameraExtrinsics`, concrete `CameraExtrinsics{Float64, RotMatrix{3,
  Float64, 9}, Vec3{Float64}}`; the planar-target pose is `CameraExtrinsics(H, intrinsics)`), `UncertainPixel`,
  `UncertainPoint3` (Core), `Rigid3` (Estimation; PoseLib's went with the redesign), `NNMatch` (Features: built
  `Attributed(Correspondence{Int, Int}(i, j), NNMatchMeta(score, ratio))`), `SCHOEPS_PIXEL_FRAME`,
  `CANONICAL_PIXEL_FRAME` (IO: `PixelFrame{:colmap}`) — each in RETIRED_NAMES with its replacement ("Pinhole" also a
  stored-format token: IO's version-1 lens model name); private `_Placement`, `_BundleOwnedData`, `_CameraPosePose`,
  `_FrameRotation`, `_BoardPose`, `_RwPose`, `_ViewRotationChart`, `_FrameCov`, `_FoerstnerDetection`, `_SfopDetection`,
  `_CAMERA_RIG`, `_BOARD_RIG`, and test constants (`OPENCV3`, IO's `colmap`, the golden fixture's `ECF_OPENCV3`) inlined.
  `CircleProblem`, `EllipseProblem`, `RotationProblem` and `RadialCameraProblem` are NOT hits (their type keeps free
  parameters). OPEN (rule limit, for the owner): a type computed by a call, `const X = typeof(…)`, is not seen statically —
  `_AXIS_FAMILY`, `_UncertainRotation`, `_BundlePose`, `_UnitWeights`, `_Metres` (Estimation), `_PoolCorrespondences`
  (Core), the tests' `DSL`/`EQ`/`DECENTERED`, and `BaseQuantity` (FlexUnits' base form, which conventions/units.md names
  as a computed constant) — the same kind of name for one concrete type.
- 2026-11-24 (owner, Rigid3 storage): the hand-eye problem stores its motions as `HandEyeProblem{S, U, E}`
  (`motions::Vector{Tuple{E, E}}`, `model_type` `E`, `FixedModels{1, E}`), the robot-world calibration as
  `RobotWorldHandEyeCalibration{K, P}` with `P` the concrete type built (its constructor's dictionaries are built
  with the inferred concrete type of the motions they hold). The hand-eye methods on a model read `X::E`, `E` bounded to the UNLABELED plain maps
  (`EuclideanMap{3, Float64, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}`): a bound of
  `PlainEuclideanMap{3, Float64}` alone also admits the labeled `CameraExtrinsics`, which widened inference of
  `sample_hypothesis(::Any, θ, idx)` for the pose problems (seen as a container's element type in the golden
  replay, bits unchanged) — the old `X::Rigid3` admitted the unlabeled form only. Writing `CameraExtrinsics`' concrete
  type out made the absolute and bearing pose problems' two `_chart` bodies long enough for the duplication gate: one
  method on `Union{AbsolutePoseProblem, BearingPoseProblem}` now (bearing_pose.jl). The one conversion into the stored form is the private `_plain_motion(x) =
  EuclideanMap(RotMatrix{3, Float64}(rotation(x)), Vec3{Float64}(translation(x)); check = false)` (Estimation, one
  definition for both files: the `convert(Rigid3, x)` it replaces, bit for bit) — no type alias and no public name.
- 2026-11-24 (golden framework): a recorded type printed through a retired alias names the identical type: a FieldMap's
  `aliases` (default `RETIRED_ALIAS_PRINTS`, `ci/golden/bits.jl`: each alias's exact recorded print `=>` its type's
  exact print now) rewrite the TYPE text of a recorded leaf (a struct's, a singleton's, a type value's, an array's
  element type) before comparison; values (bits, digests, strings) are never rewritten.
- 2026-11-24 (PoseLib agent): `check_naming` retires `refine_relpose` and `refine_fundamental` and accepts
  `refine_homography`, `refine_fundamental`, `refine_relpose` as PoseLib's C++ functions in a C source
  (`FOREIGN_FORMAT_TOKENS`, `:c_source`, as `bundle_adjust`). PoseLib's internal `_refine_width`/`_refine_result` are
  `_deployed_width`/`_poselib_result`.
- 2026-11-24 (PoseLib agent): PoseLib 0.4.0's compat is Core 0.14 only; it has no Estimation dependency (none is
  needed). The replay environment is Estimation's test project with the workspace root on the load path (its `[deps]`
  hold PoseLib), as for every group; `replay/core_minimal.jl` and `replay/core_closed_forms.jl` named PoseLib by a wrong
  UUID, so their PoseLib cases could not run until now.
- 2026-11-24 (PoseLib agent, gates of this milestone): golden replay 4905/4905 `===` in package-image and
  `--pkgimages=no` modes over poselib, core_closed_forms, core_minimal, core_sensor_lm, core_lie, homography,
  noise_scales and every Estimation group, with 13 poselib intended entries (the 11 retired-keyword `MethodError`s, D5,
  and the 2 approved seed refusals) and the alias prints of `RETIRED_ALIAS_PRINTS`. The est_camera_fits record and replay
  name `Lens{Perspective, Radial{Even, 3, 0, Float64}}` where the fixture constant was (same type, same bits).
  VisualGeometryToolkitIO, -Features and -Eval do not load against Core 0.14 yet (their own migrations): their alias
  removals are source edits the static checks cover (check_naming finds no R20 hit and no retired alias in them), not
  run by a suite.
- 2026-11-24 (owner, the computed aliases; supersedes the OPEN note above): R20 also runs on the loaded module.
  `check_naming(M::Module)` (and so `test_naming(M)`, every stack package's quality test) adds
  `_concrete_type_bindings(M)`: a module-level `const` of `M` or a submodule whose value is a concrete `DataType` of a
  stack package (its root module a key of `stack_dependencies()`) under a name that is not the type's own — a computed
  `const X = typeof(…)` included. Regression test in Core's `quality_naming_tests.jl`. Fixed: Estimation's `_AXIS_FAMILY`
  (now the value `_AXIS_LENS`, its type `typeof(_AXIS_LENS)` where it is read), `_UncertainRotation` (the assertion spells
  `Uncertain{RotMatrix{3, Float64, 9}, CovMat{3, Float64, SMatrix{3, 3, Float64, 9}, UnlabeledFrame{3}}}`), `_BundlePose`
  (`EssentialManifold{S2TangentT{Float64}, Float64}`), and the test constants `DSL`, `EQ`, `DECENTERED` and the golden
  fixture's `ECF_DSL`, `ECF_EQ`, `ECF_DECENTERED`, `ECF_RAT_TP` (inlined, in the fixture, its record and its replay).
  Not hits, by the rule's stack-type criterion and with no special case: `BaseQuantity` (FlexUnits'
  `Quantity{Float64, Dimensions{FixRat32}}`), `_Metres` (FlexUnits' `Units`), `_UnitWeights` (FillArrays' `Fill`),
  `_PoolCorrespondences` (a StructArrays `StructVector`) — names for one concrete type of a package outside the stack.
  Owner ruling (2026-11-24): R20 stays limited to stack types; these four stay. A `const` naming a concrete type of
  another package names a type that has no name of its own there, so it is no second name for one of ours
  (conventions/naming.md, R20).
- 2026-11-24 (PoseLib agent, the milestone's review round; code-review and simplify findings applied):
  - `LevenbergMarquardt{B, S, D}`: the backend parameter first, so a method on one backend is
    `LevenbergMarquardt{NativeSolver}` (Core's two refines of a problem, `_refine_spec`); Core's forwarder of a raw-data
    refine takes `alg::LevenbergMarquardt` and defaults `stops = LmStops()`; PoseLib types `alg::LevenbergMarquardt` (the
    backend is dispatched on positionally). No record prints the type: the replay is unchanged.
  - A backend whose package is not loaded is ONE `ArgumentError` (`_no_backend_lm`, naming the backend and the two
    contract generics its package implements, no downstream package named) from `default_damping`'s new fallback and
    `lm_backend_refine`'s; `LevenbergMarquardt(PoseLibSolver())` without `using PoseLib` raises it.
  - `init` on a raw-data refine is the `MethodError` of `__init` (left as it is, as for the native LM, which has no
    `__init` either).
  - R20, parsed: the arity table is keyed by `(root package, type name)` and holds only types whose full application is
    concrete (a loaded struct, or an alias whose body is a non-abstract struct, its count every free variable of the
    body; the scanned package's own `struct`s); an abstract type, a `Union` and the package's own parametric aliases
    are left to the run-time check. R20, run time: also a concrete instantiation bound under the type's own name
    (`v !== v.name.wrapper`). Not seen (a limit): a partial application, a `UnionAll` such as `const X =
    EuclideanMap{3, Float64}`.
  - One unit-bearing gate: Estimation's `_assert_unit_bearings` (EssentialProblem, RotationProblem) and PoseLib's copy
    are deleted for Core's `assert_unit_bearings(fs, which, who; remedy)`. PoseLib's recorded messages are identical;
    Estimation's finite-bearing message loses its trailing " before building the problem" (no record holds it).
  - Estimation's hand-eye bound is the parametric alias `_UnlabeledMap3{R, V}` (free variables: no R20 hit).
  - Features 0.10.0 and IO 0.12.0 (public names removed: `NNMatch`; `SCHOEPS_PIXEL_FRAME`, `CANONICAL_PIXEL_FRAME`);
    their Core compat stays until their own migrations (neither loads against Core 0.14); IO's writer fixtures name
    their generator 0.12.0. Eval removes no public name: unchanged. `ci/Project.toml`'s Core compat is "0.14".
- 2026-11-25 (IO agent, the IO milestone):
  - §6's `fit_result_json` / `fit_result_from_json` are refused by R15 (the `fit_` prefix). Owner ruling (2026-11-25):
    no exception; the record goes through IO's existing generic mechanism. So `fit_record_json` and
    `fit_record_from_json` are deleted with no named replacement: `JSON.lower(::VgioStyle, r::FitResult)` and
    `JSON.lift(::VgioStyle, ::Type{FitResult}, record::Record)`, the style's codec every embedded value uses (a camera, a
    trajectory, a region); a kind's body holds the result itself and `write_kind` lowers it. RETIRED_NAMES lists the two
    old names and the never-released interim `fit_result_json`, `fit_result_from_json`, each with the replacement.
    The codec reads the fields (`r.retcode`, `r.stop_rule`, `r.stats.nsteps`, `r.rms`, `r.n`); the reader builds
    `SolveStats(; nsteps)` (the other counters `0`: never stored). The persisted record keeps its schema keys
    byte for byte, `{"retcode", "stop_rule", "iterations", "rms", "n"}` (`iterations` on disk is `stats.nsteps`; a
    test pins the on-disk record), so no version step is needed. `estimate` was never part of the record.
  - IO's suite had never run since the PoseLib milestone's source-only edits; fixed: two `using VisualGeometryToolkitIO:`
    lines left empty and a tautological `PixelFrame{:colmap} === PixelFrame{:colmap}` test (schoeps_tests), three
    items missing a `PixelFrame` import (target_homographies_tests), broken markdown (``` ``PixelFrame{:colmap}`` ```,
    ``TargetObservations{`PixelFrame{:colmap}`}``, a `[`PixelFrame{:colmap}`](@ref)`) and two stale comments; the
    test fixtures' undefined `Chain` import removed.
  - Found by this first run, older than the milestone: the SplineLens export refusal named the retired
    `approximate_camera` (now Estimation's `solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, camera,
    region)), CameraRegression())`); the duplication checker's similarity rule found `_kalibr_field` ⇔ `_schoeps_field`
    (now one `_yaml_value`, schoeps.jl); two docstrings `@ref`'d `opencv_lens`, which is Core's since 0.12.0 (IO's docs
    hold no Core docstrings: named in prose); the non-finite-length test edited the first `2.0` in the file, which the
    0.12.0 generator version now holds (it targets the translation's number).
  - Compat: IO (and its test, docs, benchmark projects) Core "0.14"; IO's benchmark IO "0.12". Eval keeps IO "0.11"
    (its Core compat is still 0.13, so "0.12" could not resolve; Eval's milestone moves both). The stale Core "0.13" of
    PoseLib's docs and Core's two benchmark projects is "0.14".
  - Review round (code-review, simplify): the FitResult codec sits in core_json.jl beside the other Core-value codecs;
    the lift is on `::Type{<:FitResult}` (a concrete `FitResult{Nothing, Nothing}` reaches it) and reads both enums
    through the name tuples (type-stable; an unknown name is an `ArgumentError` listing the names); `_yaml_value` throws
    its own refusal (the Schöps and Kalibr readers are one-line wrappers), and an indented key's indent is spaces or
    tabs only (`[ \t]+`, was `\s+`, which could cross a blank line).
- 2026-11-25 (Features agent, the Features milestone):
  - VisualGeometryToolkitFeatures 0.10.0 requires Core "0.14" and (weak) Estimation "0.13"; its test, docs, examples and
    validation environments too (examples/validation also Features "0.10", stale since the 0.10.0 bump).
  - `detect_image_circle` keeps its name, keywords and result; its fit (VisualGeometryToolkitFeaturesEstimationExt) is
    `solve(EstimationProblem(EllipseProblem(candidates) | CircleProblem(candidates)), Ransac(; scoring); rng)`, the
    scorer unchanged (Estimation's `default_student_t_uniform_scoring` on the problem type). `res.fit` is the RANSAC's
    FitResult, read by field (`res.fit.u`, was `estimate(res.fit)`).
  - The continuous DoG refinement calls Core's device LM through `solve(RefineProblem(fj, u₀; lb, ub),
    StaticLevenbergMarquardt(); stops = StaticLmStops(t))` inside its KernelAbstractions kernel (CPU and CUDA items
    pass; allocations unchanged), replacing the retired `static_lm`/`static_lm_termination` pair; the rim's MAD scale
    is `MedianNormScale` (was `mad_scale`). Same kernels, same bits.
  - Features defines no algorithm type, so check_algorithm_traits does not apply. check_naming's first run on Features
    under Core 0.14 found R9 `static_lm` (a config comment), R9 `photon_transfer` (an error message, a test comment)
    and R15 on a local closure `fit_on` (renamed `consensus_on`); all fixed.
  - Golden: replay/features_image_circle.jl, 141/141 `===` in package-image and --pkgimages=no modes, no intended
    entry.
- 2026-11-25 (lead): check_naming's binding walks skip deprecated bindings (`_bindings`; a read is an error under
  `--depwarn=error`, found by the LODR port), and R9 leaves a foreign package's function alone: a retired name a file
  imports explicitly from a non-stack package (`using M: name`) or qualifies by one (`M.name`) is that package's
  (DelaunayTriangulation's, in RC's experiments). RETIRED_NAMES gains RadiometricCamera 0.4.0's retired names.
- 2026-11-26 (owner, option (i)): the complete fma pin.
  - Why the 2026-11-24 pin was incomplete:
    - It stated `_jacobian_terms`' products only.
    - Its `_fma_*` helpers never inlined: `Base.fma`'s typed body keeps the `have_fma` branch and the `fma_emulated`
      call, so the inliner priced each fma as a call. The three calls per essential row kept the Jacobian loop from
      vectorizing, and the essential IRLS LO ran 2× slower than in 0.12.
    - The constraint's own StaticArrays `muladd`s still fused per compiled context: the scorer's residual and the LM
      row's residual differed in their last bits on 243 of 300 bearing pairs.
  - What the complete pin does:
    - Every helper inlines (`Expr(:meta, :inline)`).
    - Every product the Sampson layer reaches is a stated fma chain: each model's `constraint_with_jacobians` (F in the
      plane, E on bearings, both transfer forms, ellipse, conic, radial camera).
    - One whitening, `_whitening_parts` (`G·(Σ·Gᵀ)`, Cholesky, the closed-form `L⁻¹` for dg ≤ 2, the triangular solve
      beyond, so hand-eye does not move), serves the Sampson factor, the Jacobian terms, the noise-whitened residual,
      `whiten` and `whitening_factor`.
    - The scorer's and the Jacobian row's residuals are `===` on every row (test/sampson_determinism_tests.jl, with a
      typed-code audit: no `muladd_float`, no product call).
    - The essential IRLS LO is back at 0.12's runtime.
  - What moved, against the 2026-11-24 pin (identical in both modes):
    - 923 golden cases (3658 leaves): 94 ≤ 1e-15, 455 in (1e-15, 1e-12], 162 in (1e-12, 1e-10], 112 in (1e-10, 1e-6],
      34 in (1e-6, 1e-3], 66 > 1e-3.
    - 23 change a discrete leaf: 14 iteration counts or stop rules, and 9 noise_scales pooled scales whose upstream
      homography find changed an inlier count. One anytime trace changes length.
    - The classes above 1e-3: the essential sign representative, the ellipse angle representative, the contaminated
      ellipse consensus, rotation averaging on `edges5_*`, refines stopping one iteration apart, `radialcam/board1`.
    - Each moved leaf is an intended change (reason "the complete fma pin (owner, 2026-11-26)"). The 503 entries of
      the 2026-11-24 pin whose leaf came back to its record are gone.
  - Not pinned: the PnP, camera-model and hand-eye algebra; Estimation's mapped-homography product (Estimation cannot
    reach Core's private helpers); the ellipse's `sincos`; GPU contraction.
- 2026-11-26 (owner, then Eval agent): SciML's cache reuse, `reinit!(cache[, u0]; f, p, alg, rng)`.
  - `alg` must be the cache's concrete algorithm type, for type stability. Any other type is an `ArgumentError`, raised
    before anything changes.
  - The owner re-resolves the scorer, strategy and config against the kept workspace, and rebuilds it only for capacity.
  - Results alias the cache's buffers until the next `solve!`.
  - Core forwards `alg` and settings only when they are given.
  - The configuration-dependent workspace check (`check_workspace`, which uses reflection) stays where every run is, at
    `solve!`'s `_entry_workspace`, not in `reinit!`. It cost about 57 allocations per swap.
- 2026-11-26 (Eval agent): Eval's refit layer keeps one `LocalOptimization` cache per (pair, problem variant, scoring
  template, algorithm type, seed type).
  - The cache is built with `init` once, outside the timed bracket. Each call is `reinit!(cache, seed; alg)`.
  - Its refits are bit-identical to a fresh solve, with 0.12's allocation profile: 5.2 KB and 79 allocations per call.
  - `PairCtx.lo_ws` is a `Dict{Any,Any}`.
  - Owner ruling: `IGPrior` has no default algorithm (2026-11-26).
- 2026-11-26 (rename agent), decision 11, the characterisation family:
  - The kind is renamed, not re-versioned: `vgio/sensor_characterisation` → `vgio/sensor_characterization` v3.0, through
    `_RENAMED_KIND` and a v2→v3 step, which keeps every published file readable.
  - Owner ruling (a)'s v1.0 write ends with the rename (coordinator ruling 2026-11-26, decision 9 renames the key): the
    written-versions machinery is removed. A reader older than IO 0.12 cannot read a `sensor_characterization` record.
  - `vgio_migrate` refuses a former-kind file that holds a dark map (the IO 0.10.3 hazard: rasters lost with the staging
    directory); such a file reads in place.
  - `content_digest(::SensorModel)` keeps hashing `"NoCharacterisation"`: a rename moves no identity.
  - RC's `rc/run_parameters` gets v4.0 with a v3→v4 step. RC's old option key is migrated through RC's ConfigVersions
    v1→v2 converter, not refused (coordinator ruling: a renamed key is a converter step, never a refusal).
  - Recorded data names stay as data (`FOREIGN_FORMAT_TOKENS`).
- 2026-11-26 (sensor-builder agent): `DarksAndFlats` is added in `raw/sensor_estimators.jl`. It adds code and changes none.
  - Its output is bit-identical to RC's hand-assembled `characterise_sensor` (`test/darks_and_flats_tests.jl`). It uses
    PhotonTransfer's `Dict`-of-`ReadNoise` plus `sensor_model` form, as RC does.
  - Its fields are `dark` and `series`; `photon_transfer` and `master_dark` are retired names (R9). `false_defects` is
    configuration, so it lives in the `dark` field.
  - It is the problem's `default_algorithm`, being its only algorithm.
- 2026-11-26 (1A, lead's ruling on G1–G4): the RANSAC local optimization reaches Core's chart LM only through the front
  end. G1: the chart refine's cache setting `summary = nothing` (beside `weights`, A-4) is the LO refit — the problem's
  `refit_chart(f, θ, weights)` and `refit_rows(f, ch, sqrtw)` (new contract generics, defaults `model_chart`,
  `chart_rows`), one chart LM, the model only (`FitResult{Union{Nothing, M}}`, no residual sweep); a `Triggs{K}` refit
  reads `model_chart`/`chart_rows`, as the Triggs kernel did (PnP's pixel rows: 1B's finding); the weights' type is
  fixed at `init` (a Union field would box a view). G2: `width` takes a
  `DeployedLoss` of the strategy's shape, used as deployed (bit for bit with `lo_loss`). G3: a seeded
  `BlockCoordinateDescent` run gives its state as the problem's `p`; the one-shot solve builds no cache. G4: the cache
  IS the workspace (SciML): no `workspace` keyword (C15) and no public LM buffer (R22); `LoBuffers{C}` holds the
  `init`'d refit cache, built once per RANSAC solve behind a function barrier from the `zero(M)` placeholder guess (the
  stack's existing placeholder of a model type).
- 2026-11-26 (1A): `are_points_distinct(pts::NTuple; tol_px = COINCIDENT_PX)` is public, the stack's one
  minimal-sample coincidence gate on a whole fixed-arity tuple (the homography's 4, P3P's 3). It is checked in the
  caller's data frame; the problems' `test_sample` gates read it, so no package keeps its own copy.
- 2026-11-26 (1A, lead's rulings G8, G9): a row refine's result carries `stationary()` in its metadata, the closure the
  chart-problem refine already returns (one convention; Estimation's OpticalAxis gate reads it). `chart_size(f, u0)`
  is a method of the chart contract's generic (default `chart_size(f)`), read once per `init` of a chart-problem refine:
  a dof that depends on the guess (the camera regression's family, lens size, `fixed`) costs at most one function
  barrier at `init`. Option (b), a size in the refine's spelling, was rejected: it changes the public refine.
- 2026-11-26 (1A, lead's rulings A1, A2): a `DeployedLoss` as the `width` is accepted under every `Triggs{K}`,
  `Triggs{GaussianShape}` included (the LO's ℓ2-Triggs at its deployed width keeps the robustified LM, bit for bit), in
  the summary and the summary-free refine alike; the shape-only Gaussian spelling stays refused. SciML's
  `solve ≡ solve!(init(…))`: the one-shot chart refine takes `weights` and `summary`, on the shared task workspace (a
  per-trial refit allocates no workspace of its own).
- 2026-11-26 (1A, lead's ruling A4): the summary-free refit's `u0` is the problem's own model, read as
  `convert(model_type(f), u0)` (the RANSAC's `_as_model`); the summary refine keeps `guess_model` for a user's guess.
- 2026-11-26 (1A, lead's ruling A3): the one-shot block refine keeps its multi-block LM per task and kind (the bundle's,
  variables' and blocks' element, loss and damping types), in `task_buffers`, never per size: the same structure
  (variable types and sizes, held variables, block types, `varindices`, row counts, in order — what NLLSsolver's
  `OptimizeCache` and variable–cost map assume) re-seeds the kept problem in place (its variables and NLLSsolver's
  per-type cost vectors overwritten, every check before the first write); another rebuilds it. NLLSsolver is used as
  its `OptimizeCache` documents (the costs' data changed in place, the structure kept); no upstream change. The kept
  object holds the last call's blocks only, never the bundle or its guess; a reentrant call (from a callback) builds
  its own problem.
  A test reads the policy through the public query `task_kept_blocks(F)` (lead: no reach of the internal holder).
- 2026-11-26 (1A, lead's ruling C1): a zero-dimensional chart (`chart_size` `(rows, Val(0))`) dispatches at the one
  chart-problem `solve!` (one-shot and cache): no NLLSsolver run, `u` the empty chart vector, `(Success, ClosedForm)`,
  `nsteps` 0, `cost = ‖r‖²` at it (the `metadata.cost` convention: twice `residual_cost!`), `stationary() = true`, then
  `chart_result` as usual.
- 2026-11-26 (1A, lead's ruling C2): the TOML encoding of an uncapped count is Core's, one documented public pair in
  `base/config_toml.jl` (`toml_unbounded`, `unbounded_from_toml`); a config's `toml_field` hook and reader call it.
  The stored spelling stays `"unbounded"` (no converter step). BlobBoards deletes its `toml_field` branch and
  `_max_detections` in favour of the pair, and its DUPLICATION.toml `toml_field` pair goes with them.
- 2026-11-26 (1A, C3): `board!` is a retired name (BlobBoards' recipe is `projectedboard!`); `board` is not.
- 2026-11-26 (1A, 1B set 2 into the breaking set): `BackendRefit(alg, stops; cheirality_vote_points)` takes any
  `RefineKind` algorithm and carries its algorithm's identity traits (a LM's: its strategy's, scale `FixedScale()`; the
  traits' defaults cover an `AbstractEstimationAlgorithm` as they cover a strategy);
  in the breaking set its `external_refit!` returns the refine's `FitResult` (the external refit's 0.14 contract), not
  `.u` as on the non-breaking line.
- 2026-11-26 (1A, C4): `CalibrationState` is a retired name (RC b348450: `calibration_problem(…)`, a
  `RefineProblem(CalibrationData, CalibrationParameters)`); the RC entries' replacements follow RC's
  `RadiometricRefinement`.
- 2026-11-26 (1B, owner ruling G-j): penalty-weight and hyperparameter selection are estimators. ONE problem tag,
  Estimation's `Hyperparameter` (abstract): `EstimationProblem(Hyperparameter, objective, p)`.
  - `GeneralizedCrossValidation(; log10_bracket, points)` (Estimation, FitKind) replaces `gcv_select`: `u = β`,
    `metadata = (; λ, gcv, rss, dof, log10_β_bracket, at_bracket_end)`, `stats.nf` the score evaluations; a minimum at
    the bracket end is `(MaxIters, IterationCap)`. `check_penalty_weight` takes the algorithm value (no `:gcv`).
  - Eval's `TuningScheme <: AbstractEstimationAlgorithm`: `CoarseFine`/`JointCoarseFine` (FitKind) on
    `(; cells)`, `u` the selection per key, `metadata.records` each key's record; `WindowedBootstrap` (ClosedFormKind)
    the per-draw selection on `ValidationErrors`; `SelectionBootstrap(criterion; n_resamples)` (ClosedFormKind,
    `supports_solve_rng`) the bootstrap of a selection. `Factorial` has `estimator_kind` only (no solve until a
    consumer needs one). The scalar forms (`coarse_fine_cv`, `probe_curve`, …), `tune`/`evaluate`,
    `WindowedGridMethod` and `bootstrap_argmin` are deleted.
- 2026-11-26 (1B, G-m): `ConsensusScale(scoring)` (ClosedFormKind) on `EstimationProblem(SquaredScale, problem,
  (; model))` replaces `consensus_s2!`; `u = ŝ²` (the SQUARED scale: Core's `SquaredScale`, not `NoiseScale`, whose `u`
  is linear), degenerate `(InitialFailure, NoRule)` on a consensus too thin; an init/reinit! cache owns the workspace.
- 2026-11-26 (1B, G-n, owner ruling Q3): `ScoreSelection(scoring)` (ClosedFormKind) on `EstimationProblem(PoolSelection,
  pair => R | R)`: the best-scoring stored model (Core's `best_model_index` rule: first strictly best, non-finite
  never wins). Streaming through `reinit!(c; p = (; budget))`: a budget ≥ the folded count continues, a smaller one or
  new data starts over; the scratch is sized by the data `init` saw and grows only for more points. Eval's sample-budget
  `ScoreSelector` reads it at each checkpoint.
- 2026-11-26 (1B, G-o): `LikelihoodRatioSelection(refine::LocalOptimization)` (RefineKind) on `RefineProblem(f, (θ₁,
  θ₂, …), p)`: each candidate refined by the same seeded LO (`solve`, recorded), the highest `metadata.quality` kept,
  the first on a tie; `metadata = (; candidates, selected)` (`selected = 0` with no found model).
- 2026-11-26 (1B, G-l): a staged algorithm calls `solve` (never `__solve`) for each stage, so every stage result is
  recorded by the front end; it returns one stage's result as is (the front end relabels it with the outer prob/alg)
  or its own FitResult with the stage results in its metadata, never an upstream result rebuilt. The first instance:
  `CameraRegression(; refine = LevenbergMarquardt())`, the fit then its refine (the deleted `fit_then_refine` helper).
- 2026-11-26 (1B, E4): no algorithm calls another `__solve`; a units or graph adapter of one estimation and the
  estimation itself share one private kernel (rotation averaging, robot-world, rigid alignment, SimFitPP's pool form).
- 2026-11-26 (1B, owner ruling Q7): `draw_hypotheses`/`Hypotheses` are internal (`_draw_hypotheses`): the generator
  solves minimal problems, so it is an estimator, and only Estimation uses it (`PoolPair(rng, problem, raw; …)`).
- 2026-11-26 (1B, E11, owner-approved): a camera regression's sample density is a type, `ImageMeasure()`,
  `SolidAngleMeasure()`, `UniformMeasure()` <: `AbstractImageMeasure` (was a Symbol), in `CameraFitMetadata.weight`.
  The golden FieldMap gains `retagged` (`path => suffix`): a recorded Symbol equals the new singleton's tag by the tag
  rule, exactly (ci/golden/framework_tests.jl pins it). One retagged path, `metadata.weight`, 143 est_camera_fits cases.
- 2026-11-26 (1B, E14, owner ruling): `irls_termination`'s tuple → `irls_retcode(converged, closed_form)` and
  `irls_stop_rule(converged, closed_form)`, one public owner of the reweighted loops' record (BlobBoards calls both).
- 2026-11-26 (1B, V6, owner-approved): an illegal `PriorSpec` is an `ArgumentError` from its constructor (was an
  `ErrorException` at the solve), the same message; five eval_noise intended entries (type only).
- 2026-11-26 (1B, V1/V5/V8/E15): Eval's `refit_model`, `JeffreysPriorSpec`, `IGRobustSpec` and `ALL_PRIOR_SPECS` are
  deleted; the refit problem is `refit_on = PoolFrameProblem() | ReprojectionBundle()` (was `bundle::Bool`);
  `CountedLO.call_records` is a `Vector` of model-free FitResults (the replay maps them to the recorded shape).
- 2026-11-26 (1B, DUPLICATION ruling): Eval's `Pools` submodule is `PoolFiles` (one `Pools`: Core's).
- 2026-11-26 (1B, F1/F2): Features' continuous refinement holds its response as a type (`_FaridSimoncelliResponse`,
  `_SplineResponse`), and its kernel step returns its `_DogExtremum` record, not a tuple.
- 2026-11-26 (1B, G-p): the LO timing bracket of one refit is `solve!(reinit!(cache, θ))`, the workspace built by
  `init` outside it (LocalOptimization's docstring).
- KNOWN REMAINING (owner ruling G-f): PoseLib's family-problem refines cover HomographyProblem, FundMatProblem and
  RelativePoseProblem; EssentialProblem is left out (its unit bearings would take PoseLib's bearing C route and move
  LODR's recorded bits): an E refine on pixels is the RelativePoseProblem form, the cheirality seed the caller's step.
- 2026-11-26 (1A): the refine caches (`init`/`solve!`/`reinit!`) for the chart, row and block LMs. A chunked
  continuation is the caller's loop of `solve!` and `reinit!(c, u′; stops)` (no callback type). `weights` is a
  cache setting of the chart refine (`reinit!(c, θ; weights)`), not a solve keyword. `callback` joins §0's keyword list
  (`LmIterationState`; `iteration` is 1-based, SciML's).
- 2026-11-26 (1A): stops types. BlockCoordinateDescent's, IRLS's and the LO strategies' `stops` reuse
  `StoppingCriterion`, run after the method's own criterion (`StopWhenSolveFailed() | stopping_criterion(method) |
  stops`). Krylov solves and bisections use ONE type, `ToleranceStops(; max_iter, rtol)` (Q1).
- 2026-11-26 (1A): `EnsembleProblem` has two forms, an isbits `prob_func` (one KA launch) and an algorithm owner's batch
  value (BB's IRLS, Q2). Core adds no segmented container.
- 2026-11-26 (1A): configuration version 2 (C16), `[levenberg_marquardt.stops]` and `[static_levenberg_marquardt.stops]`.
- 2026-11-26 (1A): no reflection on a hot path. The transitional `block_refit` route is chosen by a dispatch sentinel
  (`block_variables`' default), not by `which`, which cost 28 allocations per solve.
- 2026-11-26 (sensor-constructor agent): the sensor components and `SensorCharacterization` take their fields as
  keywords (`ChannelBlack`, `Dark`, `Darks`, `ReadNoise`, `PoissonGaussian`, `RelativeNoise`, `DefectList`,
  `SensorCharacterization`), each forwarding to its positional form's checks. IO's reader builds every one through
  `construct_from_table`, so the `:sensor_component` exemption in `DUPLICATION.toml` is removed, with no new entry.
  - Nested components: the caller builds the children first, in its `from_toml` hook (a dark's channel black, a
    `Darks`' Darks, an entry's slots). There is no per-family builder in Core, so nothing is duplicated. An omitted
    black is the keyword form's `FileBlack()` default, so IO never calls `FileBlack()`.
  - The one Greek ⇄ ASCII key table gains `σ` ⇄ `sigma` and `σ_rel` ⇄ `sigma_rel`. No config struct has a field of
    either name, so no stored config key moves.
  - The IO reader keeps only the keys a body's spec declares (`_declares`). A newer minor's key, which the validator
    tolerates, would otherwise reach the constructor as an unknown keyword. Tested.
- 2026-11-26 (1A, lead's ruling C5): Phase 2b's retired names (RSDR, LODR, VRTK) are in RETIRED_NAMES; of VRTK's only
  the non-math alias names. The papers' math symbols (𝔭, 𝔣, t̂, 𝔱, 𝛂, 𝛈, ρ_c, 𝔟, π̂⁻¹, and the footprint ℐ_𝐮, its entry
  removed 2026-11-27 by the lead's ruling) stay legitimate notation in prose,
  docstrings and field names (their bindings are gone: a code use is already an `UndefVarError`).
- 2026-11-26 (lead, LODR L10): a configuration value that changes FORM (LODR's `adapt_scale::Bool` becoming the
  `scale_interface` tag) is a step of the package's versioned converter: `ConfigMigration`'s `values`, one owner in
  Core, run after the renames. Never a refusal; the old form in a current file is refused with its version named.
- 2026-11-26 (1B set 2, coordinator's rulings for the LODR port): PoseLib's PoseLib-specific arms are Core's
  `BackendRefit` over `LevenbergMarquardt(PoseLibSolver(); optimization_strategy)`, through PoseLib's weak extension
  `PoseLibEstimationExt` (on VisualGeometryToolkitEstimation; the allow-list row adds it): `lm_backend_refine` on a
  `HomographyProblem`/`FundMatProblem` forwards to the raw-data refine on `problem.cs` (a mapped homography refused),
  and the `EssentialProblem` route is `external_workspace` (`frame.Kc1`, `frame.Kc2`, `frame.cs`) plus `external_refit!`
  — the cheirality vote on a strided subsample (`BackendRefit`'s `cheirality_vote_points`, default 256 =
  LocalOptimizationDoneRight's `[optimization_strategy.poselib] cheirality_vote_points`), then the camera-aware pose
  refine — LODR's E body, bit for bit.
- 2026-11-26 (1B set 2): `BackendRefit{A <: AbstractEstimationAlgorithm}` is THE external refit of any refine
  algorithm: construction refuses an algorithm whose `estimator_kind` is not `RefineKind()` (`ArgumentError`). Its
  `external_refit!` returns `solve(…).u` whatever the record: a budgeted `(MaxIters, IterationCap)` stop is normal,
  and the scoring decides whether the model is kept (it returned the `FitResult`, a contract bug). In the breaking
  set (Q10) the external refit's contract is a `FitResult`: `external_refit!` returns the refine's `FitResult` there.
- 2026-11-26 (1B set 2): `honoured_stops(::PoseLibSolver) = (:max_iter,)`; Core's `check_honoured_stops` is the one
  refusal of an unhonoured `LmStops` field, and PoseLib's own (`_unapplied_stop`) is deleted.
- 2026-11-26 (1B set 2, item 6): a `BackendRefit`'s identity traits (`optimization_path`, `scale_interface`,
  `loss_family`, `lo_label`, `lo_marker`) are its algorithm's, a `LevenbergMarquardt`'s its `optimization_strategy`'s,
  and its `scale_interface` is `FixedScale()` (the refit is given its loss width); Eval's labels and markers read them
  (before, the defaults: `path_marker(:unspecified)` threw). The traits' defaults (`:unspecified`, `UnspecifiedScale()`,
  `"LO"`) cover a refine algorithm (`AbstractEstimationAlgorithm`) as they cover a strategy, so a package's refine that
  declares none falls to them through its `BackendRefit`. PoseLib's `LeZach()` reports the path `:gnc` (it reported
  `:irls`), its loss family `:tls` and label `TLS-LZ`, as LocalOptimizationDoneRight's TLS-LZ arm does.
- 2026-11-26 (1B set 2, item 7): `cheiral_pose_fast(E, cs::AbstractVector{<:Correspondence}; vote_points::Integer)` is Core's one
  cheirality seed of a relative-pose refit: the vote on the strided subsample `1:max(1, n ÷ vote_points):n` of `cs`'s
  bearing pairs, returned as `EuclideanMap(R0, Vec3{Float64}(t0))`, bit for bit the inline form PoseLibEstimationExt's
  essential route and LocalOptimizationDoneRight's adaptive arm each wrote. `BackendRefit`'s default
  `cheirality_vote_points` is 256, LocalOptimizationDoneRight's `[optimization_strategy.poselib]` of record.
- 2026-11-26 (1B set 2, item 8, lead's ruling): `check_duplication`'s `[similar]` check reads a ONE-EXPRESSION body's
  callees (every call's function in source order; a recursion as `<self>`): two such bodies that call no function in
  common (both non-empty) are no candidate pair, since a shingle normalizes identifiers and every one-line forward of
  one shape is token-identical. Narrower than "the same callees": a shared operator, constructor or recursion keeps a
  pair (RC's `plain` beside Core's `toml_value` stays a finding). Stack scan before/after (monorepo + BB, RC, LODR,
  RSDR, VRTK): 127 → 123 findings, the 4 removed all one-expression pairs with disjoint callees, none added.
- 2026-11-26 (lead, BB/CB Q1): a configuration holding another package's tables delegates them to the owner's
  converter (`ConfigVersions`' `delegates`), the owner's version recorded in the top-level `[config_versions]` table by
  package name. One owner per table: CaliBlobs never restates BlobBoards' steps.
- 2026-11-26 (1B, breaking port): `CameraPosesProblem(cs; residual = ReprojectionResidual())` on `RotationCenterViews`
  is the pixel refine from its guess alone (one refine per solve); the bearing-then-pixel flow is two solves, the
  second's guess the first's estimate, bit-identical to the retired fused form (G7 ruling (b)).
- 2026-11-26 (1B): the type form `refit_workspace(P, n[, local_optimization_strategy])` (a pre-allocated
  `RansacWorkspace(P, n, …)`) holds the same concrete cache types as the problem form, inferred from `P`
  (`Base.promote_op` of the cache builders, the stack's precedent in Core), so its refits allocate nothing. An `Any`
  slot boxed every refit (1808 bytes warm, `RotationProblem`).
- 2026-11-26 (1B): every `retcode_symbol(r) === :sym` test site is the exact `(retcode, stop_rule)` record each run
  produced (an instrumented run of the retired map logged them); a disjunction of symbols is the observed record. The
  `DivisionCameraFit` failed-leg predicate (`_failed_leg`) is exactly the records the retired map named
  `:solve_failed`, so the leg audit is unchanged.
- 2026-11-26 (1B): a setting a chart-problem refine does not take (`scale`/`width` on a `CameraRegressionProblem`
  refine) is a `MethodError`, Core's rule for any setting a solve does not take; it was an `ArgumentError`. A physical
  `RelativePoseProblem` seed is an `ArgumentError` naming `ustrip(m, pose)` (units.md point 3: an up-to-scale pose
  records no unit), where Core's absent-contract refusal would name `block_variables`.
- 2026-11-26 (joint landing, lead's ruling): `block_structure(f, u0) -> (vars, blocks, stated)` is a contract generic
  (default `(block_variables(f, u0), residual_blocks(f, u0), nothing)`), read ONCE per one-shot `solve`, `init` and
  `reinit!`; `stated` reaches `block_result` as the summary's field `stated` (G6's channel; no field for `nothing`). A
  `reinit!` that keeps the blocks of a bundle with no `stated` reads `block_variables` alone. Estimation's
  `CameraPosesProblem` and `RobotWorldHandEyeProblem` implement it: their pipelines state the bundle as a
  `_BundleStatement` (blocks, variables, and the assembly `finish(sol, loss)`, each wrapper's reading composed onto
  it), concrete, so `block_result` never re-states (the 1B port ran the pipeline three times per solve, through a
  `Ref{Any}` record: CameraPoses ~1.5× slower, the robot-world refine not inferred). Bit-identical.
- 2026-11-27 (1C), decision 11, wave 2: every British-spelled identifier in the monorepo is American; the inventory
  is the 1C helper's `inventory.tsv`.
  - Public renames, no aliases: Core `pixel_center`, `pixel_centers`, `strip_meters`, `truncated_log_normalizer`,
    `is_inside_texel_centers`, `PolygonRegion.center`; IO `meters`; Eval `Experiments.canceled` (and `live_runs`'
    `canceled` keyword). The released function names are in `RETIRED_NAMES`. A one-word function is listed by its
    call form (`"metres("`, `"cancelled("`) because the bare word is prose. A field name is not listed.
  - `honoured_stops`/`check_honoured_stops` (1B set 2, above) were never released: they are renamed
    `honored_stops`/`check_honored_stops` in place, with no `RETIRED_NAMES` entry (the `PoseLibProblem` precedent).
  - The stored key: the region body's polygon `"centre"` → `"center"`. No vgio kind embeds a polygon, so no vgio
    schema steps. IO's public `migrate_region(body)` is the one step body for an embedding kind (RC's
    `rc/initialization_region`, `rc/image_region`: each owner adds its version step). The RC records found (236
    `rc/initialization_region/v1.0` files) all hold a rect.
  - Kept as recorded data: the golden case labels (`units_40_metres`, `centre_board_raw`, …, in `ci/golden/records`)
    and wave 1's `FOREIGN_FORMAT_TOKENS`. Kept as foreign: KaimonGate's `is_cancelled`. `Disc`/`OrientedDisc`
    (and the stored `"disc"`/`"oriented_disc"` tags) are left as accepted American spellings (lead's call).
  - No content identity moves: no hashed tree holds a renamed name. No golden record stores one.
- 2026-11-27 (joint landing, runtime A/B 11930): the task-kept block problem's structure check (`_reseed_block_lm!`)
  compares each block in one call behind a function barrier (`_same_block_structure`, which reads the kept block
  itself), so an abstract vector of blocks boxes nothing: CameraPoses' least-squares refine measured +3.9% (gate 3%) with
  600 boxed allocations a solve there; now 72777 allocations against the base's 73241. The row-refine test item reads
  `CircleLmScene`'s circle rows (one owner): the duplication gate reads tracked files, so the copy surfaced only once
  `lo_refit_front_end_tests.jl` was committed.
- 2026-11-27 (set 3, C9): `ToleranceStops(; max_iter, rtol, atol = 0.0)` stays the ONE stops type; every solve that
  takes it stops at `quantity ≤ max(atol, rtol · scale)` through one private rule, `_stop_tolerance` (CGLS and CG:
  the right-hand side's norm; PathFollowing's Newton: `1 + ‖v‖`; a downstream bisection: its own stated scale).
  `atol = 0` is bit-identical (`max(0.0, x) === x` for `x ≥ 0` and NaN). PathFollowing honours `atol` too, so the field
  is inert for no consumer.
- 2026-11-27 (set 3, C10): R9's tag exemption reads the same algorithm set as R20 — the package's declarations and
  every LOADED `AbstractEstimationAlgorithm` subtype (`_loaded_algorithm_names`), computed once. The old test that
  wrote `[photon_transfer]` "without the algorithm" now uses a retired name no loaded algorithm's tag spells.
- 2026-11-27 (set 3, C12): `VgtkIO.termination_from_legacy(retcode::AbstractString, rule = NoRule)` holds Core 0.13's
  eleven rows verbatim (`rule` fills `converged`/`stalled`). The spellings are NOT added to `FOREIGN_FORMAT_TOKENS`:
  that list holds retired names a foreign or stored format owns, and none of the eleven is retired or banned, so an
  entry would be consulted by no rule (an inert entry); the stored spellings are documented by the reader's table.
- 2026-11-27 (set 3): `has_nlls_schur()` (public, lm_core.jl) is the stack's one Schur-system detection, a literal
  `Bool` fixed at compile time (`@eval`), so `_RelposeRun{has_nlls_schur(), …}` stays concrete; Core's
  `_nlls_features`/`_NLLS_HAS_SCHUR` and Estimation's `_BUNDLE_SCHUR` are deleted (private: no RETIRED_NAMES entry).
  Named in the stack's `has_<thing>` form (`has_units`, `has_distortion`).
- 2026-11-27 (set 3, BB #2): `StopWhenIncrementLess{T}(rtol)` is BB's per-blob test bit for bit: summed in index order
  in `eltype(θ)`, the guard `T(1e-300)`, strict `<` against `rtol` of its own type (a Float32 kernel compares in
  Float32). θ and θ_old are indexable real vectors (an SVector, a column view); `θ = nothing` continues
  (`StopWhenSolveFailed`'s). It is not Estimation's `StopWhenParameterChangeLess` (old-model reference, `eps()` guard,
  `ParameterChange`), which stays.
- 2026-11-27 (set 3, C14): the spec is silent on a `BoundedProblem` built directly over an essential problem. Chosen:
  `_check_extent(::WithExtent{<:HyperRectangle{2}}, _, ::EssentialMat, _) = true` (the radial camera's rule: no image
  region to validate, the inner consensus decides), so the bounded local optimization is the unbounded one bit for
  bit (LODR's "validation decorator" test). The context form `p = (; extent)` on an essential problem stays an
  `ArgumentError`, as the 2026-11-23 entry states.
- 2026-11-27 (set 4, C17, lead's ruling): `prepare_evaluation!(f::AbstractBlockProblem, variables, varnext)` is a
  contract generic (no public accessor of the block cache): Core's block refine hands the bundle NLLSsolver's two
  variable vectors before the LM's first evaluation and again before each evaluation at the solution (after a run that
  ended on its best point another vector may hold the solution), from `_lmsolve_chart!`'s multi-block form, which now
  takes the bundle. Inside the LM loop NLLSsolver evaluates at those two only (it swaps them on an accepted step; its
  best-point swap happens only on a non-finite cost, which ends the run), so a bundle that locates a request's vector
  by identity sees exactly what it read from `LmProblem.variables`/`varnext` before R22 (RadiometricCamera's batched
  window and view batches). `varnext` exists before the first call: the problem's `OptimizeCache` is built with it. No
  end hook: RC's registration is torn down by its own `try … finally` around `solve!`. The default is a no-op (0
  allocations, inferred); bits `===`.
- 2026-11-27 (lead, C15): `stop_tolerance(stops::ToleranceStops, scale)` is public, the one owner of a ToleranceStops run's
  bound; a solver outside Core reads it rather than restating `max(atol, rtol · scale)`.
- 2026-11-27 (coordinator's ruling 1, the `rw_l2_M20` runtime gate): the row-refine benchmark at M = 20 (the
  GC-dominated path; 5 steps, 5360 residuals) is gated on the minimum of a batch and on allocations, not on the median.
  Its median is set by when a collection lands (≈107–112 MB a solve), and the kept problem of A3 (the 2026-11-26 entry)
  keeps more alive between solves (62.4 MB live after a GC against 57.5 MB), which is the designed trade-off of keeping
  one problem per task. Measured in one session, base then new, two rounds (`_wt-joint-base` against `_wt-joint2`):
  minimum 231.9 / 171.5 ms (base) against 173.2 / 173.9 ms (new), the best of each +1.0 % (gate 3 %); bytes a solve
  112 027 928 → 106 742 616; allocations a solve 6079 → 5848; bits `===` (the same `u` hash, `d879b5b6808a0bd7`, in
  both). Every other touched path stays on the median gate.
- 2026-11-27 (coordinator's ruling 3, audit L11): a LODR local-optimization strategy's budget `max_iter` is a field of
  the strategy, as an algorithm argument should be, and becomes the stops where the strategy solves:
  `solve(RefineProblem(problem, model, state), BlockCoordinateDescent(method); stops = StopAfterIteration(lo.max_iter))`
  (LODR `strategies/irls.jl`, `strategies/essential.jl`, `lift/lifts.jl`). StuTuBCD's `tol` (the least score gain that
  keeps it iterating) is a setting of its BCD method (`ScoreBcdMethod`), the method's own progress test, as Core's
  block-coordinate descent places it. No second stops type and no keyword bag exist, so L11 is closed as satisfied by
  this design.
