## Naming: the verb states the kind

An estimator's name states its KIND — how it computes, and so how it can fail —
and the object names WHAT IS ESTIMATED. Which family a function belongs to is
decided by its input.

**Correspondences or a model in → decide in this order:**

1. It needs a starting guess → `refine_*`.
2. It samples hypotheses and verifies them (RANSAC) → `find_*`.
3. It is one deterministic algebraic step — a minimal sample, or least squares
   with FIXED weights → an ALGORITHM NAME, no verb ([`homography_dlt`](@ref),
   [`kabsch`](@ref), [`zhang_intrinsics`](@ref), [`tls_line`](@ref)).
4. Otherwise (iterative, reweighted, repeated, staged) → `fit_*`.

| Kind | Input | Needs a guess | Decides inliers | Returns | Must not | Lives in |
|------|-------|---------------|-----------------|---------|----------|----------|
| minimal solver | a minimal sample: `solve(solver, cs, idx)` | no | no | every solution (`FixedModels`) | iterate, weigh, reject | VGC (`Homography4pt`, `P3P`, `RadialCamera5pt`) |
| one-step solver (algorithm name) | an overdetermined set taken as inliers, optionally with FIXED weights | no | no | the model, plus any statistic a consumer reads | iterate, reweight, reject | VGC (`homography_dlt`, `fundamental_dlt`, `kabsch`, `chordal_mean`, `zhang_intrinsics`, `tls_line`) |
| `fit` | an overdetermined set taken as inliers | no — it seeds itself | soft weights at most (an IRLS) | one model with residual statistics | sample hypotheses | RVG (`fit_division_camera`, `fit_radial_camera`, `fit_lens`, `fit_line`) |
| `find` | contaminated data | no | yes: hypothesize, verify, locally optimize | model + inlier set | require a guess | RVG (`find_homography`, `find_absolute_pose`, `find_radial_camera`) |
| `refine` | a model and its observations | yes, required | no; continuous reweighting at most | the polished model, same family | re-estimate from scratch; change the family | RVG (`refine_robot_world_hand_eye`); PoseLib.jl's `refine_*` bindings; BlobBoards' board bundle |

The kind fixes how a call can fail: an algorithm is one deterministic
computation and can only be degenerate; a `fit` can fail to converge; a `find`
is randomized; a `refine` can end in a local minimum. That is the point of
naming by kind rather than by what comes back — the caller's error handling
follows the verb.

The name follows the DESIGN, not a knob: `fit_radial_camera(iterations = 1)` is
still a fit. `find` is RESERVED for the sample-and-verify search.
**[`solve`](@ref) is ONLY the minimal-solver protocol** — VGC's
`solve(solver, cs, idx)` and RVG's `solve(problem, idx)`, both returning every
root — never an estimator's name. The RANSAC local-optimization step is a
refine: `refine!(ws, problem, θ, weights)` starts from the current model θ (a
linear problem runs its one-step algorithm under the weights and ignores θ).
The noise-model estimators (VGC's `fit_scale`, VisualGeometryToolkitEval's
`fit_prior`) are outside this rule.

**Iterative-estimator vocabulary — no exceptions, enforced by a quality test in every stack package.**
1. An **iteration** is one cycle of `block_coordinate_descent!` through every block. Its counters are `iter`, `max_iter` and `iterations`; its hook is `post_iteration!`.
2. A **block** is one component of the iteration state the loop overwrites: the model, the residuals, the nuisance and each of its parts (scale, inlier prior, ν, α), and the weights.
3. **`update_<block>!`** overwrites that one block in place and nothing else. Every name that begins `update_` or `_update_` ends in `!` and writes exactly one block; a composite block's update (`update_nuisance!`) only calls its parts' updates. A function that writes two blocks is split.
4. **`next_<block>`** (or `_next_<block>`) is pure: it returns a block's new value and mutates nothing.
5. **`<fit>_step`** (`<fit>_step!` if it mutates) is the public model-block update of the fit `fit_<fit>`, with weights and nuisance held fixed; that fit's `update_model!` calls it. `fit_<fit>` must exist. No other identifier contains the word `step`, except Manopt.jl's `step_solver!` (one cycle of `block_coordinate_descent!`).
6. **`<method>_increment`** is a local model's Δ (`_newton_increment`, `_lm_increment`).
7. A fixed number of reweight-and-solve cycles is `iterations`; a repetition of a whole estimator after changing its inputs is an `outer_iteration` (`outer_iterations` for the count). A random hypothesis draw — a RANSAC or PROSAC sample-and-verify cycle, a sampler draw, a random LO inner sample — is a *trial* (`trial`, `trials`, `max_trials`, `min_trials`, `n_trials`; its cap is the trial budget), never an iteration, and a deterministic cycle is never a trial; the one exception is VisualGeometryToolkitCore's `FitResult.iterations` field and its `iterations(result)` accessor, which for a RANSAC count its trials. A test, gate or check that holds *succeeds* (`succeeded`, `n_succeeded`), and one that does not *fails*; a caller *supplies* an argument. No other word is used for any of these: the banned words and phrases are listed once, in `VisualGeometryToolkitCore.check_naming`, which every stack package's quality test runs.
8. The hooks of `block_coordinate_descent!` are exactly `init_state!`, `update_model!`, `update_residuals!`, `update_nuisance!`, `update_weights!`, `post_iteration!`, `stopping_criterion`, `stop_reason` and `block_coordinate_result`; its steps carry Manopt.jl's names, `initialize_solver!`, `step_solver!` and `stop_solver!`, and its termination tests are `StoppingCriterion` values with Manopt's `Stop…` names.
9. The words are matched as whole `_`-separated tokens (CamelCase split into words) of every identifier the stack binds: functions, types, fields, keywords, locals and constants, in src, ext, test, experiments, scripts and docs/src. A method added to a function the stack does not own keeps the owner's name (Base's `step(r)`). Retired names — listed in `check_naming` — may not appear anywhere, including comments, docstrings and docs.

**Width vocabulary — one name per quantity, enforced by `check_naming` (R10).** A robust loss is a LOSS (`AbstractLossShape`, `DeployedLoss`), never a "kernel"; "kernel" means only a GPU, convolution, minimal-solver, null-space or probability-density kernel. A width is `scale_multiple` (the width as a multiple of the estimated scale ŝ — not a quantile), `loss_width` (a loss shape's scale parameter, the residual scale where it leaves the shared quadratic regime; `sq_loss_width` is its square, and a scorer's threshold is its loss's `loss_width`) or `outlier_halfwidth` (the half-width of the Student-t marginal's uniform outlier component); the noise scale stays `scale` / `s2`. A single letter or abbreviation (`k`, `a`, `c`, `τ`, `thr`, a `*_ref` or `*_grid` short form) is never a keyword, field, type parameter, argument, TOML key, sidecar key or CLI flag for them; a paper's symbol appears once, in the paper project's symbol map. `decision_width(m, decision_threshold)` is the one conversion to the native width whose inlier/outlier decision flips at a given residual, and `is_map_inlier(posterior_weight)` the one MAP rule (posterior > ½).

**Images or a session in → a task verb**, naming the job: `find_boards` (per
board the answer IS a RANSAC's model and inliers), `track_boards`,
`detect_*`, `extract_*`, `load_*`. A pipeline composes the estimators above
and does no estimation of its own; an estimate from scratch buried inside a
`refine` (a P3P inside a polish) is a violation to move, not to name around.

**Nouns are accessors, constructors and algorithm names only** —
`centroid(frame)`, `blobs(bundle)`, `cutoff(bundle)`,
`PatternConfig(...)`, `kabsch(cs)` — never pipelines: a pipeline's result's
natural variable name is the noun, and `detections = detections(img)`
makes the name local inside a function (an `UndefVarError`) and rebinds the
global at top level. An algorithm name carries no such hazard: the result of
`kabsch(cs)` is named `R`, not `kabsch`.

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
