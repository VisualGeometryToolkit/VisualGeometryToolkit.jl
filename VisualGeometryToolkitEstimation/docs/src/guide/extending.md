# Extending

## Implementing a Custom RANSAC Problem

To add a new geometry type to the RANSAC framework, implement a subtype of
`AbstractRansacProblem` with the required methods.

### Required Methods

```julia
# The problem interface is split across its owners: extend each verb from the
# package that owns it (`import`, so the methods below extend rather than shadow).
using VisualGeometryToolkitEstimation: AbstractRansacProblem
import VisualGeometryToolkitCore: minimal_sample_size, data_size, codimension, residuals!,
    model_type, solve, test_sample, test_model, test_consensus, refine!, refine_workspace
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

# Solve from a minimal sample (indices into data)
function solve(p::MyProblem, sample_indices::AbstractVector{Int})
    # Return a model or FixedModels{N,M}, or nothing on failure
end

# Compute whitened residuals for all data points given a model.
# MUST satisfy r[i]^2 = Mahalanobis distance (the whitening contract).
# For isotropic problems (Σ_x = I), this is just the signed distance.
function residuals!(residuals::Vector{Float64}, p::MyProblem, model)
    for i in eachindex(residuals)
        residuals[i] = compute_distance(p.data[i], model)
    end
    return residuals
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
# the line problem instead refits with its iterative fit_line on the points of
# positive weight). `ws` is the refit workspace refine_workspace(typeof(p), n)
# (a ManifoldLmWorkspace{N} for the chart-LM problems, `nothing` for a problem
# whose refit needs none); define a method for MyProblem when it needs one.
# Type θ with the model type, so a call in the wrong order is a MethodError
# rather than a silent bind.
function refine!(ws, p::MyProblem, θ::SMatrix{3,3,Float64,9}, weights::AbstractVector)
    # Return refined model or nothing
end

# solve() must return FixedModels{N,M} or nothing
# Single-solution: FixedModels{1,M}; multi-solution: FixedModels{N,M}
```

### Usage

```julia
using VisualGeometryToolkitCore: MsacScoring, TLSShape

problem = MyProblem(data)
result = ransac(problem, MsacScoring(problem; threshold = 2.0);
                local_optimization = Irls{TLSShape}())   # needs refine!
```

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
loop in [`block_coordinate_descent!`](@ref) handles iteration, convergence, and the best-cost
ratchet — your concrete method only needs to provide:

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
VisualGeometryToolkitCore.stop_reason(::StopWhenMine, state, method, problem, θ, θ_old, iter) =
    ... ? :converged : :continue
stopping_criterion(::MyMethod) = StopWhenMine()   # compose several with `|`
block_coordinate_result(state, method::MyMethod, problem, θ, stop_reason, final_iter) = ...
```

The protocol is exercised by [`PosteriorIrlsMethod`](@ref), which `Irls{K}`
runs inside the RANSAC LO inner loop and `fit_residual_mixture` runs on held
residuals — read `src/estimators/ransac/local_optimization.jl` for a complete
worked example.
