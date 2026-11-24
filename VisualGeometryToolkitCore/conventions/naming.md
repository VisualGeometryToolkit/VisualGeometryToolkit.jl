## Naming: the kind states how an estimator can fail

An estimator is a PROBLEM solved by an ALGORITHM: `solve(problem, algorithm)`
(CommonSolve's verb, the one estimation verb). The problem names WHAT IS
ESTIMATED — the model type, or a problem-interface object `f` (a
VisualGeometryToolkitEstimation family problem: `HomographyProblem(cs)`) — and
the algorithm's KIND ([`estimator_kind`](@ref)) states how it computes, and so how
it can fail. Which kind an algorithm is follows its input and its design:

1. It needs a starting guess → a refine ([`RefineKind`](@ref)), solving a
   [`RefineProblem`](@ref)`(f, u0)`.
2. It samples hypotheses and verifies them (RANSAC) → a find ([`FindKind`](@ref)).
3. It is one deterministic algebraic step — a minimal sample, or least squares
   with FIXED weights → a minimal solver or a closed form ([`MinimalKind`](@ref),
   [`ClosedFormKind`](@ref)): [`DLT`](@ref), [`Kabsch`](@ref), [`Zhang`](@ref),
   [`TotalLeastSquares`](@ref).
4. Otherwise (iterative, reweighted, repeated, staged) → a fit ([`FitKind`](@ref)).

| Kind ([`EstimatorKind`](@ref)) | Input | Needs a guess | Decides inliers | Returns | Must not | Examples |
|------|-------|---------------|-----------------|---------|----------|----------|
| minimal solver ([`MinimalKind`](@ref)) | a minimal sample: `solve(MinimalProblem(M, cs, idx), solver)` | no | no | every solution (`FixedModels`) | iterate, weigh, reject | VGC (`Homography4pt`, `P3P`, `RadialCamera5pt`) |
| closed form ([`ClosedFormKind`](@ref)) | an overdetermined set taken as inliers, optionally with FIXED weights: `solve(EstimationProblem(M, data), algorithm)` | no | no | the model, plus any statistic a consumer reads | iterate, reweight, reject | VGC (`DLT`, `Kabsch`, `ChordalMean`, `Zhang`, `TotalLeastSquares`) |
| fit ([`FitKind`](@ref)) | an overdetermined set taken as inliers: `solve(EstimationProblem(…), algorithm)` | no — it seeds itself | soft weights at most (an IRLS) | one model with residual statistics | sample hypotheses | VisualGeometryToolkitEstimation (`DivisionCameraFit`, `CameraRegression`, `RotationAveraging`, `ReweightedLeastSquares`) |
| find ([`FindKind`](@ref)) | contaminated data: `solve(EstimationProblem(f, p), Ransac(…); rng)` | no | yes: hypothesize, verify, locally optimize | model + inlier set | require a guess | VisualGeometryToolkitEstimation's `Ransac` on its family problems (`HomographyProblem`, `AbsolutePoseProblem`, `RadialCameraProblem`) |
| refine ([`RefineKind`](@ref)) | a model and its observations: `RefineProblem(f, u0)` | yes, required | no; continuous reweighting at most | the polished model, same family | re-estimate from scratch; change the family | VGC's [`LevenbergMarquardt`](@ref) (and the device [`StaticLevenbergMarquardt`](@ref)); VisualGeometryToolkitEstimation's `LocalOptimization`; BlobBoards' board bundle |

The kind fixes how a call can fail: a closed form is one deterministic
computation and can only be degenerate; a fit can fail to converge; a find
is randomized; a refine can end in a local minimum. That is the point of
naming by kind rather than by what comes back — the caller's error handling
follows the kind. In code the kind is the algorithm's [`estimator_kind`](@ref), and
`solve` refuses a problem of another kind before any method runs.

The kind follows the DESIGN, not a knob: a radial-camera fit with one
iteration is still a fit. **[`solve`](@ref) is the ONE estimation verb** —
CommonSolve's `solve(problem, algorithm)`, Core's front end for every problem —
never an estimator's name; a package adds `__solve` methods for its own
algorithms. A RANSAC problem's sample solve, returning every root, is
VisualGeometryToolkitEstimation's `sample_models(problem, idx)`. The RANSAC local-optimization step is a
refine: `refine!(ws, problem, θ, weights)` starts from the current model θ (a
linear problem runs its one-step algorithm under the weights and ignores θ).
The noise-model estimators (VGC's [`ScaleMAP`](@ref), VisualGeometryToolkitEval's
prior fit) are outside this rule.

**The estimation interface's rules, enforced by `check_naming` (R15–R21).**
- R15: `fit_`, `find_` and `refine_` name kinds, never functions: no function a
  package's own code (`src/`, `ext/`) defines, exports or declares public begins
  with one, except a task verb listed in `TASK_VERBS` with its reason
  (`find_boards`); a name beginning `_` is an internal step and exempt.
- R16: Core owns the front end: no package but VisualGeometryToolkitCore adds a
  method to `solve` or `init`; a package adds `__solve`/`__init` methods only for
  algorithm types it owns.
- R17: every concrete algorithm type has an [`estimator_kind`](@ref) method.
- R18: an `rng` keyword appears only on a `__solve`/`__init` method whose
  (problem, algorithm) pair [`supports_solve_rng`](@ref) — a find, or an algorithm
  that draws and says so (`SimFitPP`).
- R19: no identifier begins with `solve_` (an iterative algorithm's block update
  is `update_<block>!`).
- R20: no `const` aliases an algorithm type or a solve verb, and none holds an
  algorithm value: one name per thing, an algorithm's configuration its fields.
  Nor does a `const` name ONE concrete instantiation of a stack type (`const X =
  EuclideanMap{3, Float64, …}` with every parameter given, `const X = typeof(…)`):
  spell the type, or write a parametric alias with free variables (`const
  Line2D{T} = Hyperplane{2, 3, T}`), which stays allowed. The parsed rule sees a
  type applied to all its free parameters; the module form, `check_naming(M)`
  (every package's quality test), sees the loaded bindings, a computed one
  included. A type of a package outside the stack is not a stack type.
- R21: a [`FitResult`](@ref) is read by field (`r.u`, `r.retcode`,
  `r.stats.nsteps`, `r.stop_rule`, `r.rms`, `r.metadata`) and
  `successful_retcode(r)`, its cross-type quantities only by their functions
  (`weights(r)`, `dof(r)`, `noise_scale(r)`, `scale_source(r)`,
  `inlier_mask(r)`); no retired accessor and no `r.metadata.<contract field>`
  outside the contract method.

**Mechanism tokens name steps, never estimators (owner, 2026-11-23).** An ESTIMATOR — a problem or algorithm type, or an estimator function — is named for what it estimates and how it can fail, never by an abbreviated mechanism token (`lm`, `gn`, `irls`, …) that hides its kind: `StaticLevenbergMarquardt`, not a `StaticLm`. A contract STEP — a function a loop or a contract calls, which does one mechanism by definition — may name its mechanism, because that is what it is: `posterior_irls!`, `lmsolve_chart!`, `block_coordinate_descent!`.

**Detection, correspondence, observation — the terminology taxonomy (owner, 2026-11-13), enforced by `check_naming` (R14).**
- A **detection** is a detector's output in one image, unassociated: `Attributed{Uncertain{<:FeatureFrame}}`, a [`PointFrame`](@ref) as first-class as any scaled frame (feature-frames). A keypoint IS a detection; with a descriptor it is VisualGeometryToolkitFeatures' `DescribedFeature`.
- A **correspondence** is a hypothesised pairing ([`Correspondence`](@ref), [`Correspondences`](@ref), [`correspondences`](@ref)). A **match** comes from descriptors; an **inlier** is a verified correspondence.
- An **observation** is a detection associated with a model entity (a target point, a landmark), in the form an estimator consumes: [`TargetObservations`](@ref), VisualGeometryToolkitIO's `observations` kind (its entity `target_point` or `landmark`), and a detection's point observation [`center`](@ref VisualGeometryToolkitCore.center)`(u)`. A **track** is the observations of one entity.
- **measured** is a provenance adjective and value only (`SensorSource.measured`, `"measured"`: the source of a sensor-model component such as a [`ChannelBlack`](@ref), a covariance's), never a bound name. No identifier — function, type, field, keyword, local, constant, quoted symbol, TOML key — holds the word "measurement", the old pairing abbreviation, "sighting" or "keypoint" (R14); each old public name is in `RETIRED_NAMES` with its replacement (R9). A published artifact's name that spells an old word is data: it is listed in `FOREIGN_FORMAT_TOKENS` and accepted only in string literals, quoted TOML values and data fixtures.

**Iterative-estimator vocabulary — no exceptions, enforced by a quality test in every stack package.**
1. An **iteration** is one cycle of `block_coordinate_descent!` through every block. Its counters are `iter`, `max_iter` and `iterations`; its hook is `post_iteration!`.
2. A **block** is one component of the iteration state the loop overwrites: the model, the residuals, the nuisance and each of its parts (scale, inlier prior, ν, α), and the weights.
3. **`update_<block>!`** overwrites that one block in place and nothing else. Every name that begins `update_` or `_update_` ends in `!` and writes exactly one block; a composite block's update (`update_nuisance!`) only calls its parts' updates. A function that writes two blocks is split.
4. **`next_<block>`** (or `_next_<block>`) is pure: it returns a block's new value and mutates nothing.
5. **`update_model!`** is the model-block update of an iterative algorithm, with weights and nuisance held fixed: its method of the block-coordinate-descent hook (items 3 and 8), public only as that hook; the computation inside it is internal, named for what it computes (its local Δ is item 6's `<method>_increment`). No identifier contains the word `step`, except Manopt.jl's `step_solver!` (one cycle of `block_coordinate_descent!`) and SciMLBase's iteration counter `nsteps` (a [`SolveStats`](@ref) field).
6. **`<method>_increment`** is a local model's Δ (`_newton_increment`, `_lm_increment`).
7. A fixed number of reweight-and-solve cycles is `iterations`; a repetition of a whole estimator after changing its inputs is an `outer_iteration` (`outer_iterations` for the count). A random hypothesis draw — a RANSAC or PROSAC sample-and-verify cycle, a sampler draw, a random LO inner sample — is a *trial* (`trial`, `trials`, `max_trials`, `min_trials`, `n_trials`; its cap is the trial budget), never an iteration, and a deterministic cycle is never a trial; the one exception is VisualGeometryToolkitCore's `FitResult` counter `r.stats.nsteps` (SciML's name for the iterations), which for a RANSAC counts its trials. A test, gate or check that holds *succeeds* (`succeeded`, `n_succeeded`), and one that does not *fails*; a caller *supplies* an argument. No other word is used for any of these: the banned words and phrases are listed once, in `VisualGeometryToolkitCore.check_naming`, which every stack package's quality test runs.
8. The hooks of `block_coordinate_descent!` are exactly `init_state!`, `update_model!`, `update_residuals!`, `update_nuisance!`, `update_weights!`, `post_iteration!`, `stopping_criterion`, `stop_reason` and `block_coordinate_result`; its steps carry Manopt.jl's names, `initialize_solver!`, `step_solver!` and `stop_solver!`, and its termination tests are `StoppingCriterion` values with Manopt's `Stop…` names.
9. The words are matched as whole `_`-separated tokens (CamelCase split into words) of every identifier the stack binds: functions, types, fields, keywords, locals and constants, in src, ext, test, experiments, scripts and docs/src. A method added to a function the stack does not own keeps the owner's name (Base's `step(r)`). Retired names — listed in `check_naming` — may not appear anywhere, including comments, docstrings and docs, except a `CHANGELOG.md`, whose job is to record what each release removed (banned words are still banned there).

**Width vocabulary — one name per quantity, enforced by `check_naming` (R10).** A robust loss is a LOSS (`AbstractLossShape`, `DeployedLoss`), never a "kernel"; "kernel" means only a GPU, convolution, minimal-solver, null-space or probability-density kernel. A width is `efficiency_constant` (the width as a multiple of the estimated scale ŝ — not a quantile), `loss_width` (a loss shape's scale parameter, the residual scale where it leaves the shared quadratic regime; `sq_loss_width` is its square, and a scorer's threshold is its loss's `loss_width`) or `outlier_halfwidth` (the half-width of the Student-t marginal's uniform outlier component); the noise scale stays `scale` / `s2`. A single letter or abbreviation (`k`, `a`, `c`, `τ`, `thr`, a `*_ref` or `*_grid` short form) is never a keyword, field, type parameter, argument, TOML key, sidecar key or CLI flag for them; a paper's symbol appears once, in the paper project's symbol map. `decision_width(m, boundary)` is the one conversion to the native width whose inlier/outlier decision flips at a given residual (its inverse, and a scorer's current one, is `decision_boundary`), and `is_map_inlier(posterior_weight)` the one MAP rule (posterior > ½). Each scorer takes its own width and only it: a threshold scorer (RANSAC, MSAC, the loss-width families) a `threshold`, the Student-t + uniform scorer an `outlier_halfwidth` — it has no threshold, and no name, key, comment or doc calls its half-width one. R12 flags that word close to the scorer's type or its TOML tag (six tokens); the contract's decision point is `decision_boundary` (a threshold scorer's constant τ; the Student-t scorer's MAP crossover at the current consensus scale, re-estimated with every score).

**Images or a session in → a task verb**, naming the job: `find_boards` (per
board the answer IS a RANSAC's model and inliers), `track_boards`,
`detect_*`, `extract_*`, `load_*`. A pipeline composes the estimators above
and does no estimation of its own; an estimate from scratch buried inside a
refine (a P3P inside a polish) is a violation to move, not to name around. A
task verb that begins with a kind word is listed, with its reason, in
`check_naming`'s `TASK_VERBS` (R15).

**Nouns are accessors, constructors and algorithm names only** —
`centroid(frame)`, `blobs(bundle)`, `cutoff(bundle)`,
`PatternConfig(...)`, `Kabsch()` — never pipelines: a pipeline's result's
natural variable name is the noun, and `detections = detections(img)`
makes the name local inside a function (an `UndefVarError`) and rebinds the
global at top level. An algorithm name carries no such hazard: it is a type, and the
estimate of `solve(EstimationProblem(RotMatrix{3}, cs), Kabsch())` is named `R`.

**Package aliases: one short name per stack package.** The registered names
carry the full `VisualGeometryToolkit` prefix (General already holds other
`VisualGeometry*` packages). Code that qualifies a stack package's name binds
its alias once per module — `import VisualGeometryToolkitCore as VgtkCore` — and
writes `VgtkCore.foo`, never the long form. The aliases are exactly `VgtkCore`,
`VgtkIO`, `VgtkFeatures`, `VgtkEval` and `VgtkEstimation`; PoseLib keeps its
name. The long name stays where Julia or a reader needs the real one:
`Project.toml`, `using`/`import` paths (`using VisualGeometryToolkitCore.Pools`),
a package's references to itself in its own `src/` and `docs/make.jl`, URLs,
and user-facing install snippets. A package's own tests use its alias like
any other code.
"Vgtk" is internal shorthand only, never a registered name.

See [Robust Estimation](../docs/src/guide/estimators.md) for which function to
reach for.
