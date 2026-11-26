# The Estimation Interface

```@meta
CurrentModule = VisualGeometryToolkitCore
```

SciML's problem + algorithm with CommonSolve's verbs: the problems, the kinds of algorithm, the front end
(`solve`, `init`, `solve!`, `reinit!`) and the solution every estimator returns, the [`FitResult`](@ref). The
algorithms themselves are on their subjects' pages (the closed forms on Solvers, the LMs on the LM core, the sensor
estimators on Raw Images). The design record is `docs/design/sciml-mapping.md` in the repository.

## The estimation interface

Every estimator of the stack is a typed ALGORITHM solving a PROBLEM, under SciML's interface and CommonSolve.jl's
verbs: `solve(problem, algorithm; settings...)`, or `init(problem, algorithm; settings...)` and `solve!(cache)` for
an algorithm that keeps workspaces (`reinit!(cache[, u0]; p, f, alg, rng)` points the cache at new data, a new
configuration of the same algorithm type or a new rng; a result may alias the cache's buffers until the next
`solve!`). The problem says
what is estimated, from which data and in which context `p` — [`EstimationProblem`](@ref) (from scratch),
[`RefineProblem`](@ref) (from a required guess), [`MinimalProblem`](@ref) (every root of a minimal sample); the
algorithm says how, its configuration its fields ([`DLT`](@ref)`()`, [`LevenbergMarquardt`](@ref)`()`, a minimal
solver); the keywords are run settings only (`rng`, `scale` or `width`, `stops`).

```julia
r = solve(EstimationProblem(HomographyMat, cs), DLT())   # a FitResult: r.u is the homography (nothing if degenerate)
models = solve(MinimalProblem(HomographyMat, cs, 1:4))   # FixedModels: Homography4pt, the declared default
```

An algorithm's [`EstimatorKind`](@ref) (`MinimalKind`, `ClosedFormKind`, `FitKind`, `FindKind`, `RefineKind`) fixes
how it can fail and which problem it solves; `solve` checks that law ([`check_kind`](@ref)) before any method runs.
A package adds `__solve`/`__init` methods for its own algorithm types, never a `solve` method.

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["base/solve_interface.jl"]
```

`solve!(cache)` is CommonSolve's verb, run on the cache an algorithm's `init` returns (its owner adds the method);
`solve(problem, algorithm)` is `solve!(init(problem, algorithm))` for an algorithm that holds workspaces.

## Fit Results

Every estimator but a minimal solver returns a [`FitResult`](@ref): SciML's solution fields — the estimate `u`, the residuals `resid`,
the verdict `retcode`, the work counters `stats` ([`SolveStats`](@ref)), the problem `prob` and algorithm `alg`
solved, a wrapped backend's `original` return — then the stack's `stop_rule`, the residual `rms` over `n`, and
solver-specific `metadata` selected by type parameter. Read its own data by field (`r.u`, `r.retcode`,
`r.stats.nsteps`, `r.metadata.depths`); success is [`successful_retcode`](@ref)`(r)`. The cross-type contract
quantities are read by their functions only (`weights(r)`, `dof(r)`, `noise_scale(r)`, …): a result forwards no
property into its metadata, and a name that is not a field is an error, never `nothing`.

```@docs
FitResult
SolveStats
BundleMetadata
with_problem
rms
weights
dof
s2
map_estimate
```

A robust result records ONE scale — the one it ran with, or the one it
estimated — and where it came from: its metadata's `scale` and
`scale_source`, read by [`noise_scale`](@ref) and [`scale_source`](@ref). A
downstream stage takes the hand-off as `scale = result`; [`scale_input`](@ref)
parses a stage's `scale`/`width` keywords, exactly one of which it requires.

```@docs
ScaleSource
ScaleGiven
ScaleUpstream
ScaleEstimated
ScaleNone
noise_scale
scale_source
ScaleInput
scale_input
```

Why a fit stopped is its termination record, a [`ReturnCode`](@ref) and a [`StopRule`](@ref) in the result's core: see
the Termination Record page.


Its per-datum weights in [0, 1] are [`weights`](@ref)`(result)`, a plain vector;
what they mean — and so where [`inlier_mask`](@ref)`(result)` cuts them — is
the producer type's [`WeightKind`](@ref), read by [`weight_kind`](@ref)
(Estimation Interfaces).

## Index

```@index
Pages = ["estimation.md"]
```
