# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Once per clone: let git regenerate the Manifest whenever a pull, checkout or
# rebase moves Project.toml (compat bumps from a release cut elsewhere).
# Without this every such pull leaves the Manifest illegal. See .githooks/.
git config core.hooksPath .githooks

# Run all tests — ReTestItems: every @testitem in test/*_tests.jl, in parallel
# worker processes (RETESTITEMS_NWORKERS overrides the count in test/runtests.jl;
# 0 runs serially in-process, which is what you want when debugging)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run only the items whose name matches a regex. An item is named after its
# file: ransac_homography_tests.jl -> "ransac_homography". There is no
# include-one-file workflow any more — a @testitem cannot be `include`d
# standalone, it must be run through ReTestItems.
julia --project=. -e 'using Pkg; Pkg.test(; test_args = ["ransac_homography|lift"])'

# Run tests interactively with Revise (ReTestItems and the item's test deps
# must be loadable, e.g. from the global environment)
julia --project=.
julia> using Revise, ReTestItems, VisualGeometryToolkitEstimation
julia> runtests("test/ransac_homography_tests.jl")          # one file
julia> runtests(VisualGeometryToolkitEstimation; name = r"^lift")      # items by name

# Strict run: turns any upstream deprecation into a test failure at the
# commit that introduces it rather than at the release that removes it.
# Run it before a release, and fix what it reports in the commit that owns the code.
julia --project=. -e 'using Pkg; Pkg.test("VisualGeometryToolkitEstimation"; julia_args=["--depwarn=error"])'

# Docs (from the monorepo root; --startup-file=no is required)
julia --project=VisualGeometryToolkitEstimation/docs --startup-file=no VisualGeometryToolkitEstimation/docs/make.jl
```

`test/quality_gates_tests.jl` holds the Aqua / ExplicitImports / JET ratchets.
Each bound there is the value the package meets TODAY and may only ever move
DOWN; a change that trips one is a defect in the change, not in the bound.
`test/readme_tests.jl` runs the README's example verbatim in a fresh
environment, and `test/docs_tests.jl` the docstring gates.

## The estimation interface here

Every estimator is VisualGeometryToolkitCore's `solve(problem, algorithm)`. This
package adds the problems' `f` and its own algorithm types, never a `solve`
method (it adds `__solve`/`__init` for the algorithms it owns):

```julia
f = HomographyProblem(cs)                                          # a family problem: the data and its context
r = solve(EstimationProblem(f, (; extent)), Ransac(); rng)         # a find (FindKind)
h = solve(RefineProblem(f, r.u), LevenbergMarquardt(); scale = r)  # a refine (Core's one host LM)
c = init(EstimationProblem(f), Ransac(); rng); solve!(c); reinit!(c; f = f2)   # a reused workspace (RansacCache)
lo = solve(RefineProblem(f, θ₀), LocalOptimization(Irls{TLSShape}(); scoring))  # one seeded local optimization
```

- **Finds** are `Ransac(; scoring, local_optimization_strategy, config, score_guard, on_improvement, on_hypothesis)`
  on a family problem (`estimators/ransac/solve.jl`); each keyword defaults to
  `ClassDefault()`, its problem class's `[pipelines.<class>]` table.
- **Refines** of a family problem are Core's `LevenbergMarquardt`: the problems
  implement Core's chart contract (`model_chart`, `chart_rows`, `chart_model`,
  `chart_size`, `default_optimization_strategy`, `default_stops`); the bundles
  (`CameraPosesProblem`, `CameraRegressionProblem`, `RobotWorldHandEyeProblem`,
  `RelativePoseProblem`) its block contract (`block_refit`).
- **Fits** are one algorithm type each: `ReweightedLeastSquares` (on `RotMatrix{3}`,
  `EuclideanMap{3}`, `LinearModel`, `RobotWorldHandEyeCalibration`),
  `RotationAveraging`, `FilteredRotationAveraging`, `ResidualMixture`,
  `RadialCameraFit`, `DivisionCameraFit`, `CameraRegression`, `PSpline`,
  `ForwardSpline`, `OpticalAxis`, `SimFitPP`, `PooledScale`, `GeneralizedCrossValidation`,
  `ConsensusScale`.
- **Results** are read by field (`r.u`, `r.retcode`, `r.stop_rule`,
  `r.stats.nsteps`, `r.resid`, `r.rms`, `r.n`, `r.metadata.…`); a cross-type
  quantity by its Core function (`inlier_mask`, `weights`, `noise_scale`,
  `scale_source`, `dof`, `s2`, `found_model`, `successful_retcode`).
- **Units**: a problem built from physical data strips it in metres at
  construction (`units.jl`) and the solve reattaches the unit; a model that cannot
  record a unit (a plane homography, a radial or division camera, a relative
  pose) refuses physical input.

## Configuration (TOML is the authority, kwargs override)

`configs/default.toml` is the config of record: `RansacConfig()`, every core
local-optimization strategy's keyword defaults, every problem class's
`[pipelines.<class>]` find defaults (`scoring` — MSAC at a `threshold`: a
per-class convention, not a tuned value — and `local_optimization_strategy`),
every `[refine.<class>]` and block problem's refine defaults, and the fits'
tables (`[division_camera_fit]`, `[robot_world_hand_eye]`, `[rotation_averaging]`,
`[reweighted_least_squares.*]`, `[camera_poses_problem]`,
`[camera_regression_problem]`, `[simfitpp]`, …) all READ that file. There is no
second set of Julia literals — `src/config_defaults.jl` parses it at precompile
time (`include_dependency`, so an edit forces a recompile; restart a Revise
session to see one) and exposes `_dflt(table, key)`, which every default
expression calls. A default that is not in the file throws BY DESIGN.
The file states `config_version = 2`; `ESTIMATION_CONFIG_VERSIONS` (Core's `ConfigVersions`) reads a stored
version-1 file (0.12's table names) at version 2 through `migrate_config`. A table rename is a new
version: a `ConfigMigration` appended there, never a silent rename.

- **Override one value:** a field of the algorithm —
  `Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 2.0))`,
  `RansacConfig(max_trials = 500)`, `Irls{WelschShape}(efficiency_constant = 1.2, max_iter = 40)`.
  `outlier_halfwidth` is no field of `Ransac` (`ArgumentError`): it belongs to a
  marginal scorer supplied as `scoring`.
- **Use another file:** `RansacConfig(path)` loads a file's `[ransac]` table and
  `optimization_strategy_from_toml(table; class, tags = OPTIMIZATION_STRATEGY_TAGS)` a strategy table; absent keys
  inherit the shipped file, present keys override, UNKNOWN keys/tags are an
  `ArgumentError`, never ignored.
- **Select a strategy in TOML:** `local_optimization_strategy = { type = "irls", loss =
  "tls" }` (no width key: follow the scorer; or `efficiency_constant = m` /
  `loss_width = w`) — tags in the `TagRegistry` values `OPTIMIZATION_STRATEGY_TAGS` / `LOSS_TAGS`
  (`src/config_toml.jl`, `scoring_glue.jl`). The core registry holds
  `"none"` (`nothing`), `"irls"`, `"triggs"` and `"terminal_lo"` (`TerminalLO`, the
  terminal placement: `{ type = "terminal_lo", inner = { … } }`); LocalOptimizationDoneRight's
  strategies load through an explicit merge,
  `tags = merge(VisualGeometryToolkitEstimation.OPTIMIZATION_STRATEGY_TAGS, LocalOptimizationDoneRight.LODR_STRATEGY_TAGS)`,
  which also resolves `efficiency_constant = "cross_validated"` (the core refuses
  it, naming the merge). A loss the mechanism cannot run (a Barron shape
  without its α) is refused at load.
- **Write one back:** `config_dict` is derived from `fieldnames`, and
  `test/config_toml_tests.jl` asserts completeness + the strategy round trip.

Adding a tunable means: field on the struct → key in `default.toml` → default
expression `_dflt(...)`. The completeness test fails until all three exist.

Revise.jl caveats:
- Compact one-liner redefinitions sometimes not picked up; use `VisualGeometryToolkitEstimation.eval(quote ... end)` as workaround
- Struct changes always require REPL restart
- Full test suite via `Pkg.test()` in fresh process avoids Revise issues

## Architecture

VisualGeometryToolkitEstimation is the composable RANSAC: ONE generic loop on
VisualGeometryToolkitCore's scoring and local-optimization contracts, the RANSAC
problems of the stack, and the fits built on them. The core local optimizations
`Irls{K}` and `Triggs{K}` are VisualGeometryToolkitCore's (`nothing` is no local
optimization). Out of the box `Ransac()` runs its problem class's scorer
(`[pipelines.<class>] scoring`: VGC's MSAC) with `Irls{TLSShape}` following the
scorer's threshold; per-class thresholds are conventions (H/F/radial 3 px,
E 1 px, PnP 8 px, hand-eye 0.01·L). Paper scorers (VisualGeometryToolkitCore's
`StudentTUniformScoring`) and strategies (LocalOptimizationDoneRight's `HQ`,
`Gnc`, `StudentTUniform`, …) implement the contracts downstream and are supplied
as `Ransac(; scoring = …, local_optimization_strategy = …)`. Depends ONLY on
VisualGeometryToolkitCore (geometry types, solvers, the estimation interface, the
problem protocol, the contracts, the robust-statistics toolkit, robust losses,
the BCD and LM cores); it must never depend on RANSACScoringDoneRight or
LocalOptimizationDoneRight.

### Module Structure

The include order in `src/VisualGeometryToolkitEstimation.jl` is the authority; this tree
mirrors it. The generic block-coordinate-descent loop, the problem protocol,
the scoring and local-optimization contracts, the MSAC/RANSAC scorers, the
chart and block contracts of the refine and the Student-t / posterior-weighting
toolkit are NOT here — they are VisualGeometryToolkitCore's
(`robust/block_coordinate_descent.jl`, `robust/lm_core.jl`, `robust/refine.jl`,
`robust/problem.jl`, `robust/scoring_contract.jl`, `robust/lo_contract.jl`,
`robust/baseline_scorers.jl`, `robust/rows.jl`, `robust/scale.jl`,
`robust/mixture.jl`, `robust/posterior.jl`, `robust/losses.jl`). What remains here
is the RANSAC loop and its algorithm types, the problem implementations (with
their `refine!`, chart and rows providers), and the fits.

```
src/
├── VisualGeometryToolkitEstimation.jl    # Entry point, imports, exports, include ordering
├── interface.jl               # Shared function stubs
├── config_defaults.jl         # configs/default.toml → `_dflt` (before every config-carrying struct); ESTIMATION_CONFIG_VERSIONS, its converter
├── units.jl                   # the estimators' one strip at entry and one reattachment at exit
├── estimators/ransac/
│   ├── types.jl               # RansacConfig, RansacMetadata/RansacEstimate, PooledScale, DltWorkspace, LoBuffers, RansacWorkspace{K,M,T,W,L}, RansacResultBuffers
│   ├── samplers.jl            # UniformSampler, ProsacSampler, GaussianSampler, ExhaustiveSampler
│   ├── interface.jl           # AbstractRansacProblem{K} API, refit_workspace, lo_workspace, RansacWorkspace constructors
│   ├── scoring_glue.jl        # LO strategy types + width modes, PosteriorLikelihood, OPTIMIZATION_STRATEGY_TAGS
│   ├── loop.jl                # the contract-only RANSAC loop (_ransac!, _try_model!, _finalize)
│   ├── local_optimization.jl  # the hosted strategies' locally_optimize!, PosteriorIrlsMethod, consensus_posterior, rescore_and_optimize!
│   ├── hypotheses.jl          # _draw_hypotheses: the RANSAC trial, repeated, no scoring
│   ├── extent_traits.jl       # ExtentTrait / BoundedProblem validity-region wrapper
│   ├── problems/
│   │   ├── correspondence.jl      # AbstractCorrespondenceProblem, MinimalSolveProblem + its aliases (Homography/FundMat/Essential/AbsolutePoseProblem)
│   │   ├── p3p.jl                 # AbsolutePoseProblem gates, residuals, PnP chart rows
│   │   ├── bearing_pose.jl        # BearingPoseProblem: pose from unit bearings (+ ray origins, pixel whitening), refine-only if non-central
│   │   ├── homography.jl          # HomographyProblem
│   │   ├── radial_homography.jl   # RadialHomographyProblem
│   │   ├── radial_camera.jl       # RadialCameraProblem (distortion-free board verification)
│   │   ├── fundmat.jl             # FundMatProblem
│   │   ├── essential.jl           # EssentialProblem
│   │   ├── circle.jl              # CircleProblem: Circle3pt + the point-on-circle Sampson rows
│   │   ├── ellipse.jl             # EllipseProblem: Ellipse5pt + the orthogonal-distance rows
│   │   ├── rotation.jl            # RotationProblem: the pure rotation on bearings; uncertain_rotation
│   │   ├── sampson_lm.jl          # the chart-LM rows of every Sampson-constraint problem (F, E, H, circle, radial camera, rotation)
│   │   ├── relpose_bundle.jl      # RelposeBundleProblem; RelativePoseProblem, the two-view bundle (block problem)
│   │   ├── line.jl                # LineFittingProblem, InhomLineFittingProblem
│   │   ├── hand_eye.jl            # HandEyeProblem
│   │   ├── pool.jl                # the pool-frame problems of a pool record; PoolPair(rng, problem, raw)
│   │   └── manifold_providers.jl  # ManifoldLiftProvider: the chart problems' VGC chart-rows contract (included after covariance.jl)
│   ├── simfitpp.jl            # SimFitPP, PoolSelection: the SIMFIT++ loss width and its pool selection
│   ├── covariance.jl          # solver_covariance: the one constraint-sandwich covariance (after the problems)
│   └── solve.jl               # Ransac, LocalOptimization, RansacCache (init/solve!/reinit!), the class defaults
├── fitting/
│   ├── line.jl                # the covariance-weighted line fit and its step `_weighted_line`
│   ├── radial_camera_fit.jl   # RadialCameraFit
│   ├── division_camera_stack.jl  # DivisionCameraProblem, DivisionCameraState, DivisionCameraError: one iteration of DivisionCameraFit
│   ├── gcv.jl                 # Hyperparameter, GeneralizedCrossValidation, gcv_curve, gcv_score, check_penalty_weight
│   ├── camera_regression.jl   # CameraRegressionProblem, CameraRegression, PSpline, ForwardSpline, CameraFitMetadata
│   ├── camera_regression_refine.jl  # the regression's LevenbergMarquardt refine (block_refit), OpticalAxis
│   ├── pose.jl                # the absolute-pose and bearing-pose physical forms (strip at construction, reattach at solve)
│   ├── irls_weights.jl        # ReweightedLeastSquares and the fixed-scale IRLS of the reweighted fits
│   ├── robot_world_hand_eye.jl          # RobotWorldHandEyeCalibration, its staged ReweightedLeastSquares fit
│   ├── robot_world_hand_eye_refine.jl   # RobotWorldHandEyeProblem: its reprojection refine (block problem)
│   ├── camera_poses.jl        # CameraPosesProblem: a camera and one pose per view (block problem)
│   ├── rotation_center_views.jl  # RotationCenterViews: CameraPosesProblem's rotation-centre views
│   ├── camera_poses_spline.jl # its SplineLens route
│   ├── reweighted_rotation.jl        # ReweightedLeastSquares on RotMatrix{3} (direction pairs)
│   ├── reweighted_linear.jl          # LinearModel; ReweightedLeastSquares on its coefficients
│   ├── reweighted_rigid_alignment.jl # ReweightedLeastSquares on EuclideanMap{3} (ATE alignment)
│   ├── rotation_averaging.jl  # RotationAveraging, FilteredRotationAveraging
│   ├── residual_mixture.jl    # ResidualMixture, MixtureEmConfig (a held model's residuals → mixture EM → MAP labels)
│   └── division_camera_fit.jl # DivisionCameraFit (after residual_mixture.jl: its MixtureEmConfig)
├── synthetic_scenes.jl        # synthetic_scenes: entry point (the scenes are the Test extension)
├── graphs/                    # ViewGraph, ObservationGraph
├── config_toml.jl             # TOML loaders, OPTIMIZATION_STRATEGY_TAGS/LOSS_TAGS registries, writer (then the refine strategies of record)
└── precompile.jl              # the PrecompileTools workload: the homography refine, not the find (its scoring loop runs slower from a package image) (included LAST)
```

### Include Order

Include order in `src/VisualGeometryToolkitEstimation.jl` is dependency-aware:

1. **interface.jl**, **config_defaults.jl** (every keyword default reads it), **units.jl**
2. **estimators/ransac/** — framework (types → samplers → interface →
   scoring_glue → loop → local_optimization → hypotheses → extent_traits)
3. **estimators/ransac/problems/** — the problem implementations, then
   `simfitpp.jl`, `covariance.jl` (its `solver_covariance` dispatches on the
   problem types), `problems/manifold_providers.jl` (needs the per-problem
   chart/residual helpers) and `solve.jl` (the algorithms; after the problems,
   whose classes it reads)
4. **fitting/*.jl** — the fits; `division_camera_stack.jl` precedes
   `division_camera_fit.jl`, which follows `residual_mixture.jl` (its
   `MixtureEmConfig`); `irls_weights.jl` precedes every `ReweightedLeastSquares`
   problem; `robot_world_hand_eye.jl` precedes `robot_world_hand_eye_refine.jl`
5. **config_toml.jl** — its loaders and writer dispatch on every config-carrying type; then the refine
   strategies of record, and **precompile.jl** LAST (the workload calls everything above)

`scoring_glue.jl` loads before the problem files. Problem-specific
specializations go in `problems/`.

## Code Conventions

- **Functions**: `snake_case` (`mahalanobis_distances!`, `minimal_sample_size`, `sample_models`)
- **Types**: `CamelCase` (`HomographyProblem`, `RansacWorkspace`, `RansacEstimate`)
- **Constants**: `UPPERCASE_SNAKE_CASE`
- **Section dividers**: `# =============================================================================`
- **Private helpers**: Prefix with `_` (`_try_model!`, `_adaptive_count`, `_seed_posterior_state`)

## Stack Conventions

The conventions every stack package follows — naming and the estimation
interface, signatures, data arguments, coordinate frames and normalization,
pixel origins, bearings, Jacobians, units and the stack contracts — live ONCE,
in VisualGeometryToolkitCore's `conventions/` folder (`conventions/README.md`
indexes it). Read them before changing code here. This file carries no copy of
them, only how they land in this package. Where the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

**How they land here.**
- The RANSAC family is the signatures rule's problem-first exception:
  `sample_models(problem, idx)`, `sample_hypothesis(problem, idx)`,
  `test_model(problem, model, idx)`, `mahalanobis_distances!(d, problem, model)`,
  `refine!(ws, problem, θ, weights)`, `solver_covariance(problem, model, idx)`.
  `RansacWorkspace([T = Float32,] P, n_data)` takes its storage eltype as an
  optional leading positional.
- A RANSAC problem owns its normalization: it is built once at construction
  (`CorrespondenceNormalization(M, cs)`), the gates and the minimal solves run on
  the normalized view, and `refine!(ws, problem, θ, w)` uses the problem's
  normalization. It never calls a public estimator, or a one-step algorithm that
  normalizes again. A solve returns residuals, scales, inlier sets and
  covariances in the caller's units: px, or scene length for hand-eye.
- An `rng` keyword appears only on the finds' `__solve`/`__init` and on
  `SimFitPP`'s.

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Export Organization

Export the user API only — the algorithm types, the problems a user builds, the
fits' problem tags. The machinery (workspaces, samplers, tag tables, plumbing
types, extension points, research arms, metadata types) is `public`. The package
re-exports nothing: a name is exported only by the package that owns it
(VisualGeometryToolkitCore's `solve`, `EstimationProblem`, `RefineProblem`,
`LevenbergMarquardt`, `Irls`, `Triggs`, …). Numbered category subsections with `-`
dividers:

```julia
# -----------------------------------------------------------------------------
# 1. FINDS AND REFINES — `solve(EstimationProblem(f, p), Ransac(…); rng)`, …
# -----------------------------------------------------------------------------
export Ransac, LocalOptimization, RansacConfig, RansacEstimate

# -----------------------------------------------------------------------------
# PUBLIC, not exported — the machinery
# -----------------------------------------------------------------------------
public DltWorkspace, RansacWorkspace, ...
```

Group related names together (types, then functions). Add brief comments where the purpose isn't obvious.

## Benchmarking

Use `@btime` or `@benchmark` from BenchmarkTools.jl. Never use `@time` for benchmarking (includes JIT compilation on first run).

```julia
using BenchmarkTools
@btime my_function($input)  # $ interpolates to avoid benchmarking global access
```
