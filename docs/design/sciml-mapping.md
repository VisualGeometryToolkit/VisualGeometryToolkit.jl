# SciML → VisualGeometryToolkit: the mapping for Core 0.14

Status: the design record of the Core 0.14 migration (branch `core-0.14/solve-interface`, from main 3d8a06da, Core
0.13.3), with the owner's rulings of 2026-11-23 folded in (§8).

Sources were read, not run, from shallow clones at these commits:

| repo | commit | date | version read |
|---|---|---|---|
| CommonSolve.jl | 396f80d | 2026-10-17 | 0.2.14 (the version already in the stack's Manifest) |
| SciMLBase.jl | f75f097 | 2026-11-03 | 3.59.1 |
| NonlinearSolve.jl, incl. `lib/NonlinearSolveBase`, `lib/SimpleNonlinearSolve`, `lib/NonlinearSolveFirstOrder` | cee51ef | 2026-11-20 | |
| Optimization.jl, incl. `lib/OptimizationBase` | 11fd5d9 | 2026-11-12 | |

File references below are relative to each repository's root.

## 0. Findings that change the plan

**F1. CommonSolve's own rule.** `CommonSolve.jl/src/CommonSolve.jl`, the `solve` docstring: "Extensions must dispatch the first positional argument on a type that they own." SciML follows it in two layers:

- the package that owns the problem type defines the public `solve`/`init`. NonlinearSolveBase defines `solve(prob::AbstractNonlinearProblem, args...)` (`lib/NonlinearSolveBase/src/solve.jl:76`); OptimizationBase defines `solve(prob::OptimizationProblem, …)`.
- that front end checks keywords (`checkkwargs` against `allowedkeywords`, `SciMLBase.jl/src/errors.jl:1`), routes the default algorithm and concretizes the problem (`solve_up` → `solve_call`).
- it then calls the extension points `SciMLBase.__solve(prob, alg; kw...)` and `SciMLBase.__init(prob, alg; kw...)`. Algorithm packages add methods there for the algorithm types they own (`NonlinearSolve.jl/src/default.jl:25,40`).

SimpleNonlinearSolve, the kernel solvers, is the one exception: it defines `CommonSolve.solve(prob::NonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm)` directly, "to bypass the high-level checks" (`lib/SimpleNonlinearSolve/src/SimpleNonlinearSolve.jl:78`).

Consequence for us. The prototype's `solve(::EstimationProblem{<:HomographyMat}, ::Ransac)` in Estimation is not Aqua piracy, because Estimation owns `Ransac`. It does break CommonSolve's stated rule, though: the first argument is a Core type. The SciML-faithful shape is:

- Core owns the front end: `CommonSolve.solve(::AbstractEstimationProblem, alg; kw...)` and `init`. The front end runs the kind law and the default-algorithm routing, then calls `__solve`/`__init`.
- Each package implements `__solve`/`__init` (Core's extension points, `public`, not exported) for its own algorithm types.
- R16 becomes: no stack package except Core adds a `CommonSolve.solve`/`init` method. Others add `__solve`/`__init` methods whose algorithm argument they own.

**F2. Decision 3 vs. who owns `f`. Ruled 2026-11-23: option A (§8).** SciML's problem carries its `f`: `NonlinearLeastSquaresProblem(f, u0, p; lb, ub)` (`SciMLBase.jl/src/problems/nonlinear_problems.jl:349`). The algorithm is generic over `f`'s interface, and `f` is a value its author owns. In the stack, the chart-LM row protocol (`residual_rows!`, `residual_jacobian_rows!`) is implemented in Estimation, on Estimation's own family problem types: `ChartLm{<:…, <:SampsonChartProblem}`, `AbsolutePoseProblem`, `EllipseProblem`, `RadialHomographyProblem`, `HandEyeProblem`, `BearingPoseProblem` (`VisualGeometryToolkitEstimation/src/estimators/ransac/problems/*.jl`). The generic refine body `_refine_pipeline(problem, spec, θ₀)` is already generic over those types (`fitting/_refine.jl`). Each `refine_X` is only "build `XProblem(data)`, then run the generic body".

The constraint follows from that:

- The host `LevenbergMarquardt` is a Core type, so the generic `__solve(::RefineProblem, ::LevenbergMarquardt)` lives in Core.
- If the problem is keyed by the model type (the prototype's `RefineProblem(H₀, cs)`), Core must call a Core-owned generic to turn `(HomographyMat, Correspondences)` into Estimation's `HomographyProblem`.
- Estimation adding that method on Core's function for Core's types is piracy. That is exactly the `treat_as_own` the owner rejected.

For RANSAC there is no such problem: Estimation owns `Ransac` and can own the mapping function too.

The options put to the owner (A was recommended and chosen):

- **A. Problems carry `f`, as in SciML.**
  - Signatures: `RefineProblem(f, u0, p = NullParameters(); lb, ub)`; `EstimationProblem(f, p)` for finds and fits; `MinimalProblem(f, sample, p)`.
  - `f` is the problem-interface object: Estimation's family problems (`HomographyProblem(cs)` …) for the robust and refine classes, a residual closure for the device LM, a Core data container for Core's closed forms.
  - Every algorithm has ONE generic `__solve` over `f`'s interface, and every interface method is on its author's own type: zero piracy.
  - User code: `solve(EstimationProblem(HomographyProblem(cs)), Ransac(); rng)` and `solve(RefineProblem(HomographyProblem(cs), H₀), LevenbergMarquardt(); scale = r)`. DLT stays `solve(EstimationProblem(HomographyMat, cs), DLT())`, all Core types. That is a different `f` (the algebraic error, not the geometric constraint), so it is not a second way.
  - Cost: the family problem constructors become the public `f`s. This reverses the plan's "pixel constructors become unexported machinery".
- **B. Move the family problems into Core.** That is ~4.7 k lines (`estimators/ransac/problems/`): data, constraints, rows, solver covariance.
  - Model-type-keyed problems then work everywhere without piracy.
  - But each family's RANSAC-contract methods (Estimation functions) would sit on Core types, splitting every family file across two packages. Core grows by a whole layer.
- **C. A separate host LM type in Estimation.** This contradicts decision 3 (one host LM, distinct types only for distinct implementations).
- **D. `treat_as_own`.** Rejected by the owner.

**F3. SciML has no `default_algorithm` function.** `solve(prob)` means `alg = nothing`. The front end sends `nothing` to `__solve(prob, ::Nothing)`, defined by the problem family's package (`NonlinearSolve.jl/src/default.jl:40`, a polyalgorithm chosen from `prob`: bounds, a user Jacobian, `u0 isa StaticArray`). When nothing is defined, SciML throws `NoDefaultAlgorithmError` (`SciMLBase.jl/src/errors.jl:241`, thrown at `NonlinearSolveBase/src/solve.jl:645`).

Decision 7's declared `default_algorithm(problem)` is therefore a deviation, justified in §5 D3. The routing itself follows SciML: `solve(prob)` → `solve(prob, default_algorithm(prob))`.

**F4. SciML already has the randomized trait and the `rng` keyword.**

- `rng` is in SciMLBase's `allowedkeywords` (`errors.jl:50`).
- The trait is `supports_solve_rng(prob, alg) -> Bool` (`alg_traits.jl:553–568`), defined on the PAIR, default `false`. A solver that returns `true` must consume the supplied rng.
- Decision 7's `is_randomized` should take SciML's name and signature: `supports_solve_rng(prob, alg)`. R18 then reads that trait.
- SimFitPP (randomized, not a find) is covered for free: the trait is per algorithm, not derived from the kind.

**F5. Stop settings are solve keywords, damping is an algorithm field.** In SciML, `abstol`, `reltol`, `maxiters`, `maxtime` and `termination_condition` are `solve` keywords, on the host and the kernel solvers alike (`SimpleNonlinearSolve/src/raphson.jl:44`). The LM's damping lives inside the algorithm value:

- `LevenbergMarquardt(; damping_initial, damping_increase_factor, …)` builds `GeneralizedFirstOrderAlgorithm(; descent = DampedNewtonDescent(; damping_fn = LevenbergMarquardtDampingFunction(…)))` (`lib/NonlinearSolveFirstOrder/src/levenberg_marquardt.jl:39–68`).
- Its damping scales by `DᵀD` floored at `min_damping_D = 1e-8`, the Marquardt form. That matches the stack's default `MarquardtDamping`.
- Decisions 6 and 10 agree with SciML. The prototype kept `stops` inside `LevenbergMarquardt` and the damping inside `LmStops`; both move.

**F6. Kernel solvers: SciML's precedent and naming.**

- SciML keeps the non-allocating, kernel-capable solvers in a separate subpackage under the `Simple` prefix (`SimpleNewtonRaphson`, `SimpleTrustRegion`, `SimpleGaussNewton`; it has no Simple LM).
- They run on the SAME problem type (`NonlinearLeastSquaresProblem`, or the isbits `ImmutableNonlinearProblem`, `nonlinear_problems.jl:703–718`), with `maxiters`/`abstol` as solve keywords.
- They return `stats = nothing`: no counters on the kernel path (`raphson.jl:56,81`).

Decision 3's `StaticLevenbergMarquardt` fits that precedent. The SciML spelling would be `SimpleLevenbergMarquardt` (ruled: `StaticLevenbergMarquardt`, §8). SciML also ships an alias, `const SimpleGaussNewton = SimpleNewtonRaphson` (`raphson.jl:31`). The stack's no-alias rule is stricter: deviation D9.

**F7. Neither `remake` nor `reinit!(cache, data)` exists in SciML as we sketched it.** SciML's `reinit!(cache[, u0]; p)` resets a cache for a new initial point and/or new parameters (`NonlinearSolveBase/src/abstract_types.jl:730–736`). Data that changes is new `p` in SciML, because SciML's data lives in `p` or in `f`'s closure. `remake(prob; kw...)` rebuilds a problem with replaced fields (`remake.jl:33`).

## 1. Problems

| SciML | stack (Core 0.14) | note |
|---|---|---|
| `AbstractSciMLProblem` | `AbstractEstimationProblem` | Core-owned supertype; the front end dispatches on it (F1) |
| `NonlinearLeastSquaresProblem(f, u0, p = NullParameters(); lb, ub, kwargs...)` | `RefineProblem(f, u0, p = NullParameters(); lb, ub)` | field names `f, u0, p, lb, ub` as SciML's. The prototype's `bounds = (lb, ub)` becomes SciML's `lb`/`ub`. There is no constructor without `u0` |
| `LinearProblem(A, b, p; u0 = nothing)`: a problem defined by its data, guess optional | `EstimationProblem(f, p = NullParameters())` | from-scratch problems: closed-form, fit and find. `f` per F2 |
| — | `MinimalProblem(f, sample, p = NullParameters())` | SciML has no minimal-sample problem. Extension, kept from the prototype |
| `p`: a type parameter `P`, default `NullParameters()` (`problem_utils.jl:210`, errors on indexing) | `p`, a type parameter of every problem (decision 6); `NullParameters()` mirrored under SciML's name | the prototype's `extent` keyword and the context (`center`, `cameras`, `char_length`, …) go in `p` |
| `kwargs` stored on the problem and merged into the solve | not adopted | D6 |
| `isinplace` / `{iip}` | not adopted | no in-place/out-of-place `f` duality: the row protocol always writes caller buffers |
| `ImmutableNonlinearProblem` (isbits, for kernels) | `RefineProblem` itself is immutable, so it is isbits for isbits `f`/`u0`/`p` | SciML's `NonlinearProblem` is mutable only "temporarily" for Enzyme (`nonlinear_problems.jl:292`). One immutable type serves host and device |
| `remake(prob; kw...)` | not adopted in 0.14 | D7 |

## 2. Algorithms

| SciML | stack | note |
|---|---|---|
| `AbstractSciMLAlgorithm`, `AbstractNonlinearAlgorithm` | `AbstractEstimationAlgorithm` | |
| an algorithm value carries its configuration as fields (`LevenbergMarquardt(; damping_…)`, `linsolve`, `autodiff`) | same: `Ransac(; scoring, local_optimization_strategy, config)`, `LevenbergMarquardt(; optimization_strategy, damping = MarquardtDamping())`, `DLT()`, `Kabsch()`, `Irls{K}(…)` | damping as a field (decision 10, F5) |
| host `LevenbergMarquardt` vs kernel `Simple*` | host `LevenbergMarquardt` (NLLSsolver adapter) vs device `StaticLevenbergMarquardt` (`static_lm`, made private) | F6; ruled §8 Q6 |
| algorithm traits: `allowsbounds`, `requiresbounds`, `supports_solve_rng(prob, alg)`, `has_init`, … (`alg_traits.jl`) | `estimator_kind(alg)` (`MinimalKind`/`ClosedFormKind`/`FitKind`/`FindKind`/`RefineKind`), `needs_initial_model`, `supports_solve_rng(prob, alg)` (in place of `is_randomized`, F4), `decides_inliers` | SciML has no kind taxonomy: extension E1 |
| `_check_opt_alg` → `IncompatibleOptimizerError` (OptimizationBase `solve.jl`), `check_prob_alg_pairing` | the kind law in the front end: an `ArgumentError` naming the problem the algorithm solves | same role, the stack's wording |
| `solve(prob)` ≡ `__solve(prob, nothing)` → the package's polyalgorithm; else `NoDefaultAlgorithmError` | `solve(prob)` → `default_algorithm(prob)`, declared per problem type; with none, an error that lists the algorithms (generated from the traits) | F3, D3 |
| `allowedkeywords` + `checkkwargs` (one global list) | keywords are declared on each `__solve` method; an unknown one is a `MethodError` | D5 |

Defaults (decision 7) on this mapping:

- a `RefineProblem` → `LevenbergMarquardt()`;
- a `MinimalProblem` → its unique minimal solver, declared by `f`'s model type;
- a closed-form `EstimationProblem` → its closed form;
- a find problem → NONE: robustness is named explicitly. SciML has no analogue; its `NonlinearProblem` default is adaptive (D3).

## 3. Solve, init, solve!, reinit!

| SciML / CommonSolve | stack | note |
|---|---|---|
| `solve(args...) = solve!(init(args...))` (CommonSolve fallback) | used as is: an algorithm with workspaces implements `__init` + `solve!(cache)`, and gets `solve` free | |
| `solve(prob, alg; kw...)` defined by the problem's owner → `__solve` | Core: `solve(::AbstractEstimationProblem, alg; kw...)` → `__solve(prob, alg; kw...)` | F1 |
| `init(prob, alg; kw...)` → `__init` → a cache | Core: `init` → `__init`; e.g. `RansacCache` from Estimation | decision 5: the only workspace reuse |
| `solve!(cache)` | `solve!(cache)`, defined by the cache's owner | |
| `reinit!(cache[, u0]; p)` | `reinit!(cache[, u0]; p)` with SciML's signature, plus `f` where the data change: `reinit!(cache; f)`, a new family problem on new data (streams, BlobBoards' per-frame finds) | F7, D8 |
| `step!(cache)` | not implemented in 0.14 (CommonSolve's generic, optional) | |
| `rng` solve keyword + `supports_solve_rng` | `rng` only on algorithms with `supports_solve_rng(prob, alg) == true` (R18) | F4 |
| `maxiters`, `abstol`, `reltol`, `maxtime`, `termination_condition` solve keywords | `stops = LmStops(…)` / `StaticLmStops(…)` as the solve keyword, plus `scale`/`width` | D4 |
| `@compile_workload` in each package (CommonSolve itself has one) | decision 13: workloads in Core and Estimation for the common `solve` entries | same |

## 4. Solutions: `FitResult` field by field

SciML's `NonlinearSolution` (`solutions/nonlinear_solutions.jl:65`) has the fields `u, resid, prob, alg, retcode, original, left, right, stats, trace`. `OptimizationSolution` (`optimization_solutions.jl:80`) has `u, cache, alg, objective, retcode, original, stats, dual`. Today's `FitResult{P, M}` has `estimate, retcode, stop_rule, iterations, rms, n, metadata`.

| SciML field | today | Core 0.14 | golden mapping |
|---|---|---|---|
| `u` | `estimate` | `u` (`nothing` only for a find with no model, or a degenerate closed form: D10) | `estimate` → `u` |
| `retcode::ReturnCode.T` | `retcode` | unchanged. The stack's `ReturnCode` already mirrors SciMLBase's names AND integer values (checked against SciMLBase 3.59.1: Default 0, Success 1, Terminated 2, MaxIters 4, InitialFailure 8, Failure 10, MaxTime 15, Stalled 18, InternalLinearSolveFailed 20) | identity |
| `stats` (`NLStats(nf, njacs, nfactors, nsolve, nsteps)`, mutable; `OptimizationStats(iterations, time, fevals, gevals, hevals)`, 0 = unavailable) | `iterations` | `stats::SolveStats`, immutable and isbits, SciML's `NLStats` field names (`nsteps` the accepted iterations or a RANSAC's trials; `nf`, `njacs`, `nfactors`, `nsolve` 0 where not counted, documented per algorithm as OptimizationStats does). D11 | `iterations` → `stats.nsteps` |
| `resid` | `metadata.residuals` where present (refine, RANSAC: per-datum whitened distances, not the signed rows) | `resid`: the per-datum whitened distances, computed once at the solution (ruled §8 Q2; D12) | `metadata.residuals` → `resid` |
| `prob` | — | added: the problem solved (cheap: a reference; isbits on the device) | new |
| `alg` | — | added: the algorithm value | new |
| `original` | — | added: the wrapped backend's raw return, as SciML defines it (PoseLib's C++ result; `nothing` for native solvers) | new |
| `objective` (Optimization) | the LM record's `cost` | decision 9: SciML's name for the minimized value is `objective`. Recommendation: `objective` where the value is the objective the LM minimized; `sum_squares` only where it is a plain ‖r‖² that is not the objective | `cost` → `objective` |
| `left`, `right`, `trace`, `dual`, `cache` | — | not adopted (bracketing, tracing, duals) | |
| — | `stop_rule` | extension E2 (SciML has only the retcode) | identity |
| — | `rms`, `n` | extension E3 (n is the support, `count(w > 0)`, not `length(resid)`) | identity |
| — | `metadata::M` | extension E4: the concrete per-family type parameter; no Union fields (decision 4) | per family, field by field |

Type stability: `FitResult{U, R, S, Prob, Alg, Orig, M}`, every parameter concrete (as `NonlinearSolution` parameterizes every field). The one Union stays `Union{M, Nothing}` for a find's `u`.

Accessors (ruled, §8 Q1): SciML's interface is FIELD access (`sol.u`, `sol.retcode`, `sol.stats`, `sol.resid`) plus one function, `successful_retcode(sol)` (`retcodes.jl:468–497`). The stack exposes accessor functions as its API and holds two success tests (`converged(r)` reading `successful_retcode(retcode)`). Following decision 4 strictly:

- the SciML-defined part is read by field: `r.u`, `r.retcode`, `r.stats.nsteps`, `r.resid`;
- the success test is `successful_retcode(r)`;
- `estimate`, `retcode` (the function), `iterations` and `converged` are retired;
- the extensions keep their accessor functions: `stop_rule`, `rms`, `noise_scale`, `scale_source`, `weights`, `inlier_mask`, `dof`, `s2`.

The property forwarding into `metadata` stays (an extension).

`FitResult` will not subtype SciML's `AbstractNoTimeSolution`, which is an `AbstractArray` forwarding to `u` (D2).

## 5. Deviations from SciML, each with its reason

- **D1. No dependency on SciMLBase. Mirror its names.**
  - SciMLBase 3.59.1 has 31 direct dependencies (RecursiveArrayTools, SymbolicIndexingInterface, RuntimeGeneratedFunctions, SciMLOperators, FunctionWrappersWrappers, …). None is in the stack's Manifest today; CommonSolve already is (via Roots).
  - Measured 2026-11-23 (Julia 1.13.4; SciMLBase 3.58.0, the newest registered version; cold processes, 5 runs each): `using SciMLBase` alone takes 148–151 ms (50 packages). Loaded after VisualGeometryToolkitCore 0.13.3 (2006–2029 ms) it still adds 174–176 ms, **+8.7 %** on Core's load, against decision 13's +5 % budget. `using CommonSolve` takes 14.4 ms.
  - The stack's `ReturnCode` and `successful_retcode` are already this kind of mirror (same names, same values).
  - `solve`/`init`/`solve!` ARE shared (CommonSolve), so a user who also loads SciML gets one `solve`.
- **D2. `FitResult` is not an `AbstractArray`.** `u` is a matrix model, a rigid map or `nothing`, so array semantics on the result would be meaningless or wrong.
- **D3. A declared `default_algorithm(prob)` instead of `__solve(prob, ::Nothing)`.** The quality gate must enumerate every default and check it is deterministic and guess-free (decision 7), and the no-default error lists the qualifying algorithms from the traits. A `nothing` method can do neither. SciML's defaults are adaptive polyalgorithms; the stack forbids that: robustness and randomness are always named.
- **D4. One stop-settings struct (`stops = LmStops(…)`) instead of SciML's flat `maxiters`/`abstol`/`reltol`/`maxtime` plus `termination_condition`.**
  - `LmStops` is the single owner of the LM's stops and their TOML defaults (configs/default.toml `[lm]`). Its knobs (`ftol`, `xtol`, `xtol_rel`, `stationarity_tol`, `residual_resolution`) have no SciML names, and the stack's success rule is not SciML's.
  - (Ruled §8 Q4: `stops`.)
- **D5. No global `allowedkeywords` list.** SciML needs one because keywords pass through many layers into `prob.kwargs`. The stack declares keywords on each `__solve` method, so an unsupported one (e.g. `rng` on DLT) is a `MethodError`, as the prototype showed.
- **D6. No `kwargs` stored on problems.** A problem holds data and context (`p`). Run settings are solve keywords only (decision 6), so there is one place for each.
- **D7. No `remake` in 0.14.** Its SciML uses (symbolic indexing, AD through solves, ensembles) are absent here, and a problem constructor plus `reinit!` already cover replacing a guess or parameters. A second constructor path would break zero duplication. It can be added later, generically, through Accessors (already a Core dependency).
- **D8. `reinit!(cache; f)` where SciML would change `p`.** In SciML, data lives in `p` or in `f`'s closure. In the stack it lives in the family problem `f` (F2 option A), and `p` is context only (decision 6). New data is therefore a new `f`.
- **D9. No aliases.** SciML ships `const SimpleGaussNewton = SimpleNewtonRaphson`; the stack refuses aliases (R20).
- **D10. `u = nothing` for a find with no model and a degenerate closed form.** SciML always returns some `u` (the last iterate or `u0`) with a failing retcode. A find or a closed form has no `u0`, so there is nothing truthful to return. The inline Union is measured at 48 vs 40 bytes and is never sent to the device (fit_result.jl's header).
- **D11. `SolveStats` is immutable and isbits; SciML's `NLStats` is mutable.** A device solve must return an isbits result, and a host cache accumulates its counts in the cache, then freezes them in the solution. The field names are SciML's.
- **D12. `resid` holds per-datum whitened distances, not the stacked signed rows f(u).** Every consumer reads the distances, and the rows are chart- and √w-dependent internals (lm_core.jl's "two currencies").

## 6. Extensions (SciML lacks them)

- **E1.** `estimator_kind` and the kind law: closed-form, fit, find, refine, minimal. SciML has capability traits, but no task taxonomy.
- **E2.** `stop_rule`: which rule fired. SciML stores only the verdict.
- **E3.** `rms` and `n` (the support), first-class.
- **E4.** `metadata::M`: concrete per family, with property forwarding.
- **E5.** The noise-scale hand-off: `scale = r` reads `noise_scale(r)`. SciML has nothing like it.
- **E6.** `MinimalProblem` and `FixedModels` (every root of a minimal solve).
- **E7.** `ClassDefault()`: an algorithm field left at its problem class's TOML default.

## 7. TOML (decision 11) on this mapping

SciML has no configuration files. Its configuration lives in algorithm fields and solve keywords, so the tables follow them:

- `[<algorithm>]` for an algorithm's field defaults (`[levenberg_marquardt] damping`, `[ransac] …`);
- `[<algorithm>.stops]` for its stop keywords;
- `[<problem f>.<algorithm>]` for a problem-dependent default of an algorithm field (today's `[refine.homography] optimization_strategy` → `[homography_problem.levenberg_marquardt] optimization_strategy`). SciML's analogue: `__solve(prob, nothing)` deriving the polyalgorithm's options from `prob`.

Exact keys are fixed with each package's converter.

## 8. The owner's rulings (2026-11-23)

- **Q1 accessors:** SciML's fields only — `r.u`, `r.retcode`, `r.stats` (`stats.nsteps`) and `successful_retcode(r)`;
  `estimate`, `retcode()`, `iterations` and `converged` are retired. The stack's extensions are further fields.
- **Q2 `resid`:** the per-datum whitened residual distances, computed once at the solution (D12).
- **Q3 `f` (F2):** option A. Problems carry `f`; Estimation's family problems are the public `f`s and their constructors
  stay public: `solve(RefineProblem(HomographyProblem(cs), H₀), LevenbergMarquardt(); scale = r)`. One generic solve per
  algorithm; no piracy, no `treat_as_own`.
- **Q4:** `stops` (the stack's `LmStops`/`StaticLmStops`; D4).
- **Q5:** `__solve`/`__init`. Core owns `solve`/`init` (keyword checks, the default algorithm, the kind law); packages add
  `__solve`/`__init` methods for their own algorithm types, declared public in `CONTRACT_GENERICS`.
- **Q6:** `StaticLevenbergMarquardt` for the device LM (a written deviation from SciML's `Simple` prefix).
- F1, F3–F7 and D1–D12 accepted as written: `default_algorithm` stays (D3), routed through `solve(prob)`;
  `supports_solve_rng` replaces `is_randomized`; stops are solve keywords and the damping a field of the LM algorithm;
  `reinit!(cache[, u0]; p)` plus `reinit!(cache; f)`, no `remake` in 0.14; no SciMLBase dependency (D1, measured).
