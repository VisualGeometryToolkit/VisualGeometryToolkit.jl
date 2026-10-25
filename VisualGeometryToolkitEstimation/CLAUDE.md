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
# Clean as of 2026-09-21 — run it before a release, and fix what it
# reports in the commit that owns the code.
julia --project=. -e 'using Pkg; Pkg.test("VisualGeometryToolkitEstimation"; julia_args=["--depwarn=error"])'
```

`test/quality_gates_tests.jl` holds the Aqua / ExplicitImports / JET ratchets.
Each bound there is the value the package meets TODAY and may only ever move
DOWN; a change that trips one is a defect in the change, not in the bound.

## Configuration (TOML is the authority, kwargs override)

`configs/default.toml` is the config of record, adopted from BlobBoards'
mechanism (2026-09-03): `RansacConfig()`, every core local-optimization
strategy's keyword defaults, every `find_*` pipeline's MSAC `threshold` (a
per-class convention, not a tuned value) and `local_optimization`, and
`fit_division_camera`/`fit_robot_world_hand_eye`'s knobs all READ that file. There is
no second set of Julia literals — `src/config_defaults.jl` parses it at
precompile time (`include_dependency`, so an edit forces a recompile; restart a
Revise session to see one) and exposes `_dflt(table, key)`, which every
default expression calls. A default that is not in the file throws BY DESIGN.

- **Override one value:** a keyword — `find_homography(p; threshold = 2.0)`,
  `RansacConfig(max_trials = 500)`, `Irls{WelschShape}(scale_multiple = 1.2, max_iter = 40)`.
  `outlier_halfwidth` is no keyword of the core entries (`ArgumentError`): it
  belongs to a marginal scorer supplied as `scoring`.
- **Use another file:** `RansacConfig(path)` loads a file's `[ransac]` table and
  `local_optimization_from_toml(table; class, tags = LO_TAGS)` a strategy table; absent keys
  inherit the shipped file, present keys override, UNKNOWN keys/tags are an
  `ArgumentError`, never ignored. (There is no whole-file loader: the
  `PipelineConfig`/`RvgConfig` pair, `load_config` and `write_config` were
  deleted 2026-09-21 — only tests used them.)
- **Select a strategy in TOML:** `local_optimization = { type = "irls", loss =
  "tls" }` (no width key: follow the scorer; or `scale_multiple = m` /
  `loss_width = w`) — tags in the `TagRegistry` values `LO_TAGS` / `LOSS_TAGS`
  (`src/config_toml.jl`, `scoring_glue.jl`). The core registry holds
  `"no_local_optimization"`, `"irls"` and `"terminal_lo"` (`TerminalLO`, the
  terminal placement: `{ type = "terminal_lo", inner = { … } }`); LocalOptimizationDoneRight's
  strategies load through an explicit merge,
  `tags = merge(VisualGeometryToolkitEstimation.LO_TAGS, LocalOptimizationDoneRight.LO_TAGS)`,
  which also resolves `scale_multiple = "cross_validated"` (the core refuses
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
problems of the stack, and the core local optimizations `NoLocalOptimization`
and `Irls{K}`. Out of the box every `find_*` runs VGC's `MsacScoring(problem;
threshold)` with `Irls{TLSShape}` following the scorer's threshold; per-class
thresholds are conventions (H/F/radial 3 px, E 1 px, PnP 8 px, hand-eye
0.01·L). Paper scorers (RANSACScoringDoneRight's `StudentTUniformScoring`)
and strategies (LocalOptimizationDoneRight's `HQ`, `Gnc`, `StudentTUniform`,
…) implement the contracts downstream and are supplied as `scoring = …,
local_optimization = …`. Depends ONLY on VisualGeometryToolkitCore (geometry types,
solvers, the problem protocol, the contracts, the robust-statistics toolkit,
robust losses, the BCD and chart-LM cores); it must never depend on
RANSACScoringDoneRight or LocalOptimizationDoneRight.

### Module Structure

The include order in `src/VisualGeometryToolkitEstimation.jl` is the authority; this tree
mirrors it. The generic block-coordinate-descent loop, the problem protocol,
the scoring and local-optimization contracts, the MSAC/RANSAC scorers, the
chart-rows contract and the Student-t / posterior-weighting toolkit are NOT
here — they are VisualGeometryToolkitCore's (`robust/block_coordinate_descent.jl`,
`robust/lm_core.jl`, `robust/problem.jl`, `robust/scoring_contract.jl`,
`robust/lo_contract.jl`, `robust/baseline_scorers.jl`, `robust/rows.jl`,
`robust/scale.jl`, `robust/mixture.jl`, `robust/posterior.jl`,
`robust/losses.jl`). What remains here is the RANSAC loop, the problem
implementations (with their `refine!` and chart-rows providers), the core
strategies and the fitting pipelines.

```
src/
├── VisualGeometryToolkitEstimation.jl    # Entry point, imports, exports, include ordering
├── interface.jl               # Shared function stubs
├── config_defaults.jl         # configs/default.toml → `_dflt` (before every config-carrying struct)
├── estimators/ransac/
│   ├── types.jl               # RansacConfig, RansacMetadata/RansacEstimate, DltWorkspace, LoBuffers, RansacWorkspace{K,M,T,W,L}, RansacResultBuffers
│   ├── samplers.jl            # UniformSampler, ProsacSampler, GaussianSampler
│   ├── interface.jl           # AbstractRansacProblem{K} API, refine_workspace, lo_workspace, RansacWorkspace constructors
│   ├── scoring_glue.jl        # core strategy types (NoLocalOptimization, Irls{K,W}) + width modes, PosteriorLikelihood, LO_TAGS
│   ├── loop.jl                # the contract-only RANSAC loop: ransac/ransac!, _try_model!, locally_optimize, _finalize
│   ├── local_optimization.jl  # the core strategies' locally_optimize!, PosteriorIrlsMethod, consensus_posterior
│   ├── hypotheses.jl          # draw_hypotheses: the RANSAC trial, repeated, no scoring
│   ├── extent_traits.jl       # ExtentTrait / BoundedProblem validity-region wrapper
│   ├── problems/
│   │   ├── cspond.jl              # AbstractCspondProblem, MinimalSolveProblem + its aliases (Homography/FundMat/Essential/AbsolutePoseProblem)
│   │   ├── p3p.jl                 # AbsolutePoseProblem gates, residuals, PnP chart-LM refit
│   │   ├── homography.jl          # HomographyProblem
│   │   ├── radial_homography.jl   # RadialHomographyProblem
│   │   ├── radial_camera.jl       # RadialCameraProblem (distortion-free board verification)
│   │   ├── epipolar_lm.jl         # Shared dof-generic Sampson LM plumbing
│   │   ├── fundmat.jl             # FundMatProblem
│   │   ├── essential.jl           # EssentialProblem
│   │   ├── relpose_bundle.jl      # RelposeBundleProblem, SchurSolver
│   │   ├── line.jl                # LineFittingProblem, InhomLineFittingProblem
│   │   ├── hand_eye.jl             # HandEyeProblem
│   │   ├── pool.jl                # the pool-frame problems of a pool record; PoolPair(rng, problem, raw)
│   │   └── manifold_providers.jl  # ManifoldLiftProvider: the chart problems' VGC chart-rows contract (rows_provider, lift_*) (included after covariance.jl)
│   └── covariance.jl          # solver_covariance: the one constraint-sandwich covariance (included after problems/hand_eye.jl)
├── fitting/
│   ├── _pipeline.jl           # _ransac_pipeline — shared body of every find_*
│   ├── line.jl                # fit_line
│   ├── homography.jl          # find_homography
│   ├── radial_homography.jl   # find_radial_homography
│   ├── radial_camera.jl       # find_radial_camera
│   ├── fit_radial_camera.jl   # fit_radial_camera
│   ├── division_camera_step.jl # division_camera_step, DivisionCameraState, DivisionCameraError
│   ├── fit_division_camera.jl # fit_division_camera
│   ├── fit_lens.jl            # fit_lens
│   ├── essential.jl           # find_essential
│   ├── pose.jl                # find_absolute_pose
│   ├── fundmat.jl             # find_fundmat
│   ├── hand_eye.jl             # find_hand_eye
│   ├── robot_world_hand_eye.jl          # fit_robot_world_hand_eye, RobotWorldHandEyeCalibration
│   ├── refine_robot_world_hand_eye.jl  # refine_robot_world_hand_eye
│   └── residual_mixture.jl    # fit_residual_mixture, MixtureEmConfig (a held model's residuals → mixture EM → MAP labels)
└── config_toml.jl             # TOML loaders, LO_TAGS/LOSS_TAGS registries, writer (included LAST)
```

### Include Order

Include order in `src/VisualGeometryToolkitEstimation.jl` is dependency-aware:

1. **interface.jl** — shared function stubs
2. **config_defaults.jl** — parses configs/default.toml; every keyword default reads it
3. **estimators/ransac/** — framework (types → samplers → interface →
   scoring_glue → loop → local_optimization → hypotheses → extent_traits)
4. **estimators/ransac/problems/** — RANSAC problem implementations (cspond →
   p3p → homography → radial_homography → radial_camera → epipolar_lm →
   fundmat → essential → relpose_bundle → line → hand_eye → pool), then
   **estimators/ransac/covariance.jl** (its `solver_covariance` dispatches on
   the problem types above), then **problems/manifold_providers.jl** (needs
   the per-problem chart/residual helpers)
5. **fitting/*.jl** — _pipeline, then the pipelines and
   fits (depend on the RANSAC problems); `division_camera_step.jl` precedes
   `fit_division_camera.jl`, `robot_world_hand_eye.jl` precedes `refine_robot_world_hand_eye.jl`
6. **config_toml.jl** — LAST: its loaders and writer dispatch on every config-carrying type

`scoring_glue.jl` loads before the problem files. Problem-specific
specializations go in `problems/`.

## Code Conventions

- **Functions**: `snake_case` (`residuals!`, `minimal_sample_size`, `locally_optimize`)
- **Types**: `CamelCase` (`HomographyProblem`, `RansacWorkspace`, `RansacEstimate`)
- **Constants**: `UPPERCASE_SNAKE_CASE`
- **Section dividers**: `# =============================================================================`
- **Private helpers**: Prefix with `_` (`_try_model!`, `_adaptive_trials`, `_seed_posterior_state`)

## Function Signature Conventions

The rule is the stack's (`conventions/signatures.md` gives the rationale): **the
model or shape first, then the data it acts on or is measured against, then
parameters.**

- `project(camera, point3d)`, `sampson_distance(H, u₁, u₂)`,
  `distance(line, point)`
- An accessor takes the one object: `minimal_sample_size(problem)`, `model_type(problem)`
- A GIVEN model leads in task verbs and estimators too:
  `find_absolute_pose(camera, pairs)`, `AbsolutePoseProblem(camera, pairs)`.
- A REQUIRED tool applied to the data leads (a kernel, scorer, network, the
  method being evaluated); a policy with a default value stays last
  (`ransac(problem, scoring; local_optimization = …)`).

**Mutation, types and RNGs follow Julia's style guide exactly** (the stack's
rule; `conventions/signatures.md` quotes the manual). Base's order: (1) a function argument, (2) an
IO — an RNG takes this slot, (3) the mutated input(s), (4) a type, (5) inputs not
mutated, (6) key, (7) value, (8) everything else, (9) varargs, (10) keywords.
The model-first rule above, and the RANSAC problem-first exception below, order
the inputs within (5) only.
- A function that modifies a Julia object argument ends in `!`, and only a
  function argument, an IO or an RNG precede what it mutates:
  `residuals!(r, problem, model)`, `score!(W, scoring, scores, residuals, n)`,
  `refine!(ws, problem, θ, weights)`, `block_coordinate_descent!(state, likelihood, method, prob, θ₀)`,
  `consensus_s2!(ws, problem, scoring, model)`. An output the caller reads
  back is the leading mutated argument of a `!` function
  (`score_model!(buf, scoring, problem, θ)`: the caller reads `buf`).
- Not mutation: advancing an RNG or a RANSAC sampler's state (so `ransac`,
  `locally_optimize` and the `find_*` pipelines keep their names), and a scratch
  buffer supplied as a KEYWORD whose contents don't matter afterwards (their
  `workspace`; Base's `sort(v; scratch)`).
- A stopping criterion is a pure query (`stop_reason`): per-iteration mutation
  (best-tracking, a schedule's rung advance, the counters a criterion reads)
  goes in `post_iteration!`, which the loop runs first.
- A type argument is positional, never a keyword: it leads a non-mutating call
  and follows the mutated inputs of a `!` call: `RansacWorkspace(P, n_data)`,
  the problem type first. An optional type is an optional positional, written
  as separate methods: `RansacWorkspace(P, n_data[, T = Float32])` (the
  per-point storage eltype; there are no per-family `*_workspace` wrappers).
- An RNG leads: `draw_sample!(rng, indices, problem)`.
- A reorder or rename makes every old call fail loudly (`MethodError` /
  `UndefVarError`) and adds no method ambiguity.

**Exception — the RANSAC family is problem-first** among the inputs it does
not mutate: the problem (or context) carries the data, the model is the
hypothesis after it; a mutated workspace or state still leads. This includes
`locally_optimize` and every wrapper that holds a problem.
- `solve(problem, sample_indices)`
- `test_model(problem, model, sample_indices)`,
  `constraint_with_jacobians(problem, model, i)`,
  `solver_covariance(problem, model, sample_indices)`
- `refine!(ws, problem, θ, weights)` — the local-optimization step from the
  current model (excluded points have `weights[i] = 0`): the model right after
  the problem, like every sibling
- `locally_optimize(problem, θ₀)`, `ransac(problem, scoring)`

The other exceptions are VGC's: a method on a function owned elsewhere keeps the
owner's order; a symmetric relation takes either order.

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization ("frame in =
frame out"), pixel origins, bearings, Jacobians, units and the stack
contracts — live ONCE, in VisualGeometryToolkitCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

## Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

**In this package:**

A RANSAC problem owns its normalization. It is built once at construction
(`CspondNormalization(M, cs)`), the gates and the minimal solves run on the
normalized view, and `refine!(ws, problem, θ, w)` uses the problem's
normalization. It never calls a public estimator, or a one-step form that
normalizes again. `find_*` returns residuals, scales, inlier sets and
covariances in the caller's units: px, or scene length for hand-eye.

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Export Organization

Export the user API only — the pipelines, the RANSAC entry points, the problems
a user builds, the LO strategies a user picks. The machinery (workspaces,
samplers, tag tables, plumbing types, extension points, research arms) is
`public`. RVG re-exports nothing: a name is exported only by the package that
owns it (VisualGeometryToolkitCore). Numbered category
subsections with `-` dividers:

```julia
# -----------------------------------------------------------------------------
# 1. PIPELINES — robust estimation end to end
# -----------------------------------------------------------------------------
export find_homography, find_fundmat, find_essential, ...

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

