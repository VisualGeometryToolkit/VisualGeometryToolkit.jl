# Extending

<!-- R22: contract guide -->

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## Implementing a Custom RANSAC Problem

To add a new geometry type to the RANSAC framework, implement a subtype of
`AbstractRansacProblem` with the required methods.

### Required Methods

```julia
# The problem interface is split across its owners: extend each verb from the
# package that owns it (`import`, so the methods below extend rather than shadow).
using VisualGeometryToolkitEstimation: AbstractRansacProblem
import VisualGeometryToolkitEstimation: sample_models
import VisualGeometryToolkitCore: minimal_sample_size, data_size, codimension, mahalanobis_distances!,
    model_type, test_sample, test_model, test_consensus, refine!, refit_workspace
using VisualGeometryToolkitCore: FixedModels

struct MyProblem <: AbstractRansacProblem
    data::MyDataType
end

# Number of points needed for a minimal sample
minimal_sample_size(::MyProblem) = 3

# Total number of data points
data_size(p::MyProblem) = length(p.data)

# Type of the model being estimated
model_type(::MyProblem) = SMatrix{3,3,Float64,9}

# Codimension: number of independent constraint equations per point
codimension(::MyProblem) = 1

# Every model of a minimal sample (indices into data)
function sample_models(p::MyProblem, sample_indices::AbstractVector{Int})
    # Return FixedModels{N,M}: count 0 on failure
end

# Compute the Mahalanobis distance (≥ 0) of every data point given a model.
# MUST satisfy d[i]^2 = squared Mahalanobis distance (the whitening contract).
# For isotropic problems (Σ_x = I), this is just the geometric distance.
function mahalanobis_distances!(d::Vector{Float64}, p::MyProblem, model)
    for i in eachindex(d)
        d[i] = compute_distance(p.data[i], model)
    end
    return d
end
```

### Optional Methods

```julia
# Validate a minimal sample before solving (e.g., reject degenerate configs)
test_sample(p::MyProblem, indices::AbstractVector{Int}) = true

# Validate a model after solving (e.g., reject degenerate solutions)
# Receives the sample indices used to fit the model for point-specific checks
test_model(p::MyProblem, model, indices::AbstractVector{Int}) = true

# Validate model against its consensus (inlier) set
test_consensus(p::MyProblem, model, mask::BitVector) = true

# The local optimization's one step: refine!(ws, p, θ, weights) refines the
# model from the current θ under per-datum weights. Its generic method routes
# through refit_kind(p) to _refine!(::NonlinLsq, ws, p, θ, weights) — the
# built-in steps are Levenberg–Marquardt on a minimal chart from θ — and a
# problem that defines refine!(ws, ::MyProblem, θ, weights) itself, as below,
# bypasses the router. A LINEAR problem's step is its one-step algorithm under
# the weights, θ unread (the weighted DLT for H/F when handed a DltWorkspace;
# the line problem instead refits with its iterative covariance-weighted line on
# the points of positive weight). `ws` is the refit workspace refit_workspace(typeof(p), n)
# (a RefitSlot holding the summary-free LevenbergMarquardt refine cache for the
# chart-LM problems, `nothing` for a problem
# whose refit needs none); define a method for MyProblem when it needs one.
# Type θ with the model type, so a call in the wrong order is a MethodError
# rather than a silent bind.
function refine!(ws, p::MyProblem, θ::SMatrix{3,3,Float64,9}, weights::AbstractVector)
    # Return refined model or nothing
end

# sample_models returns FixedModels{N,M}:
# single-solution FixedModels{1,M}; multi-solution FixedModels{N,M}
```

### Usage

A problem without a class of record names its scorer (`Ransac()`'s
`ClassDefault()` scorer is refused for it):

```julia
using VisualGeometryToolkitCore: MsacScoring, TLSShape, EstimationProblem, solve
using Random: Xoshiro

f = MyProblem(data)
result = solve(EstimationProblem(f),
               Ransac(; scoring = MsacScoring(f; threshold = 2.0),
                      local_optimization_strategy = Irls{TLSShape}());   # needs refine!
               rng = Xoshiro(1))
```

A chart problem is refined by VisualGeometryToolkitCore's `LevenbergMarquardt`
(`solve(RefineProblem(f, θ₀), LevenbergMarquardt(…); scale | width)`) once it
implements Core's chart contract on its own type: `model_chart(f, model)` (the
chart at a model), `chart_rows(f, chart, √w)` (the √w-weighted rows, in the
residual's own unit), `chart_model(f, chart)`, `chart_size(f)`, and
`default_optimization_strategy(f)` for its `ClassDefault()`. The family
problems here implement it once for every `AbstractRansacProblem` with a chart
(`chart_dof`), so a new chart problem provides the per-problem facts the family
uses (`_chart`, the rows on `ChartLm`, `_model_at`). A RANSAC local
optimization's refit reads the chart and rows through Core's
[`refit_chart`](@extref VisualGeometryToolkitCore.refit_chart)`(f, θ, weights)`
(default `model_chart`) and
[`refit_rows`](@extref VisualGeometryToolkitCore.refit_rows)`(f, chart, sqrtw)`
(default `chart_rows`); a problem whose refit differs from its refine
customizes these two.

The problem now composes with every scorer and strategy of the stack —
VisualGeometryToolkitCore's conformance helpers (`test_constraint_problem`,
`test_scoring_contract`, `test_local_optimization_contract`, loaded with
`Test`) check an implementation against the contracts. A chart problem that
also provides VisualGeometryToolkitCore's chart-rows contract (`rows_provider`,
`lift_init`, `lift_refresh!`, `lift_jacobian!`, `lift_retract`, `lift_model`,
`lift_valid`) runs LocalOptimizationDoneRight's half-quadratic lifts.

## Implementing a Custom Block-Coordinate Method

For a non-RANSAC EM / iteratively reweighted least squares, implement
`AbstractBlockCoordinateMethod` and pair it with a likelihood (`PosteriorLikelihood`). The generic
loop, VisualGeometryToolkitCore's
[`BlockCoordinateDescent`](@extref VisualGeometryToolkitCore.BlockCoordinateDescent)`(method; likelihood)`,
handles iteration, convergence, and the best-cost ratchet — your concrete
method only needs to provide:

```julia
struct MyMethod <: AbstractBlockCoordinateMethod
    # ... per-call configuration / pre-allocated buffers
end

# Returns whatever state your method needs to track between iterations.
init_state!(method::MyMethod, problem, θ₀) = ...

# The model block: one refit given the current weights; return the new θ.
update_model!(state, method::MyMethod, problem, θ) = ...

# Recompute residuals and the bookkeeping cost after a refit.
update_residuals!(state, method::MyMethod, problem, θ) = ...

# Optional: keep best-so-far (all per-iteration mutation goes here — the loop
# runs it before the stop test), decide convergence (a VGC StoppingCriterion
# value: a pure query of the state), return the final result.
post_iteration!(state, method::MyMethod, problem, θ, iter) = ...
struct StopWhenMine <: VisualGeometryToolkitCore.StoppingCriterion end
# The termination record: a (ReturnCode.T, StopRule) pair, CONTINUE while the run goes on.
VisualGeometryToolkitCore.stop_reason(::StopWhenMine, state, method, problem, θ, θ_old, iter) =
    ... ? (ReturnCode.Success, IncrementSize) : CONTINUE
stopping_criterion(::MyMethod) = StopWhenMine()   # compose several with `|`
block_coordinate_result(state, method::MyMethod, problem, θ, code::ReturnCode.T, final_iter, rule::StopRule) = ...
```

A user runs the method through the front end:

```julia
r = solve(RefineProblem(problem, θ₀), BlockCoordinateDescent(MyMethod(…); likelihood);
          stops = StopAfterIteration(max_iter))
r.retcode, r.stop_rule
```

A seeded state is the problem's context `p` (`RefineProblem(problem, θ₀, state)`):
the loop runs on it and does not call `init_state!`.

The protocol is exercised by [`PosteriorIrlsMethod`](@ref), which `Irls{K}`
runs inside the RANSAC LO inner loop and [`ResidualMixture`](@ref) runs on held
residuals — read `src/estimators/ransac/local_optimization.jl` for a complete
worked example.
