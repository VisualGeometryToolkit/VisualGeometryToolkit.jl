# =============================================================================
# RANSAC Interface — AbstractRansacProblem API and method stubs
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Problem Type
# -----------------------------------------------------------------------------

"""
    AbstractRansacProblem

Abstract type for RANSAC problem definitions.

A concrete subtype encapsulates the data, sampling strategy, minimal solver,
and constraint derivatives for a specific estimation task.

# Required Methods
- `minimal_sample_size(p)::Int` — minimum data points for a minimal solve
- `data_size(p)::Int` — total number of data points
- `codimension(p)::Int` — constraint codimension d_g
- `model_type(::Type{<:P})::Type` — concrete model type `M` (for workspace
  pre-allocation; defined on the TYPE, so `RansacWorkspace(P, n)` can read it)
- `solve(p, indices)` — fit model(s) from sample indices; returns `FixedModels{N,M}` with `count` ∈ 0..N (0 means failure)
- `constraint_with_jacobians(p, model, i)` — `(g, ∂ₓg, ∂_θg)` at data point `i`

# Derived Automatically
- `residuals!(r, p, model)` — Mahalanobis distance `√(gᵀ Σ̃_g⁻¹ g)` from `constraint_with_jacobians`

# Optional Methods
- `draw_sample!(rng, indices, p)` — custom sampling (default: uniform without replacement)
- `test_sample(p, indices)` — degeneracy check (default: `true`)
- `test_model(p, model, indices)` — feasibility check at sample points (default: `true`)
- `test_consensus(p, model, mask)` — consensus set validation (default: `true`)
- `refine!(ws, p, θ, weights)` — the local optimization's one weighted step from the current model θ (default: `nothing`)
- `solver_covariance(p, model, indices)` + `provides_solver_covariance(::Type{<:P}) = true` — model parameter
  covariance for predictive scoring; `nothing` for a degenerate sample (the hypothesis is skipped). A problem
  without a provider is refused by a predictive scorer.
- `residuals!(r, p, model)` — override the default if a faster batch implementation exists

# Example
```julia
struct MyLineProblem <: AbstractRansacProblem
    points::Vector{SVector{2,Float64}}
end

minimal_sample_size(::MyLineProblem) = 2
data_size(p::MyLineProblem) = length(p.points)
codimension(::MyLineProblem) = 1
model_type(::Type{MyLineProblem}) = SVector{2,Float64}  # (a, b) for y = a + bx

function solve(p::MyLineProblem, idx)
    p1, p2 = p.points[idx[1]], p.points[idx[2]]
    b = (p2[2] - p1[2]) / (p2[1] - p1[1])
    a = p1[2] - b * p1[1]
    model = SVector(a, b)
    FixedModels{1, typeof(model)}(1, (model,))
end

@inline function constraint_with_jacobians(p::MyLineProblem, model, i)
    pt = p.points[i]; a, b = model
    g = pt[2] - a - b * pt[1]           # scalar constraint
    ∂ₓg = SA[one(eltype(pt))]             # ∂g/∂y = 1 (only y has noise)
    ∂_θg = SA[-one(eltype(pt)), -pt[1]]  # ∂g/∂(a,b)
    return (g, ∂ₓg, ∂_θg)
end
```
"""
abstract type AbstractRansacProblem{K} <: AbstractConstraintProblem end

# -----------------------------------------------------------------------------
# Required Method Stubs
# -----------------------------------------------------------------------------

"""
    minimal_sample_size(problem::AbstractRansacProblem{K}) -> K

Return the minimum number of data points needed for a minimal solve.
Derived from the type parameter `K` of `AbstractRansacProblem{K}`.

E.g., 2 for line fitting, 4 for homography, 7 for fundamental matrix.
"""
minimal_sample_size(::AbstractRansacProblem{K}) where {K} = K
minimal_sample_size(::Type{<:AbstractRansacProblem{K}}) where {K} = K

# The "enough points" predicates — one owner each. `m` is an `Int`, or a problem
# / problem type (read by `minimal_sample_size`).

"""
    _holds_minimal_sample(m, n) -> Bool

THE rule for "enough data for a minimal sample": `n` points hold one of `m`.
Every constructor's refusal, every `find_*` pipeline's early `nothing`, the
covariance's degenerate-set exit and a weighted refit's over-determination
(`m + 1` points) ask this one question.
"""
@inline _holds_minimal_sample(m::Integer, n::Integer) = n >= m
@inline _holds_minimal_sample(P, n::Integer) = _holds_minimal_sample(minimal_sample_size(P), n)

"""
    _holds_stable_partition(m, k) -> Bool

THE rule for a partition (consensus set) of `k` points being stable for a
minimal sample of `m`: strictly more than `m + 1` points — the thin-partition
and stable-consensus gates.
"""
@inline _holds_stable_partition(m::Integer, k::Integer) = k > m + 1
@inline _holds_stable_partition(P, k::Integer) = _holds_stable_partition(minimal_sample_size(P), k)

# ONE code path for codimension and model_type: problems define the ::Type
# method; the instance form falls back through it.
codimension(p::AbstractRansacProblem) = codimension(typeof(p))
model_type(p::AbstractRansacProblem) = model_type(typeof(p))

# data_size(problem) is declared in interface.jl alongside the AbstractRobustProblem
# interface. RANSAC problems also implement it.

# `model_type` and `solve` are VisualGeometryToolkitCore generics (#21 solver layer) —
# imported in VisualGeometryToolkitEstimation.jl; problem types add methods:
#
#   model_type(::Type{<:Problem}) -> Type
#       Concrete model type being estimated (the instance form falls back). Parameterizes
#       `RansacWorkspace{K,M,T}` for type-stable storage.
#
#   solve(problem, indices::AbstractVector{Int}) -> FixedModels{N,M}
#       Fit model(s) from a minimal sample. `count` ∈ 0..N; a failed/degenerate
#       solve is `count == 0` — NEVER `nothing` (the RANSAC loop iterates the
#       result directly). Problem-level methods delegate to the VGC
#       MinimalSolver verbs, which enforce the same contract.
#
#   refine!(ws, problem, θ, weights) -> model | nothing
#       The local optimization's one weighted step, from the current model θ
#       (a chart LM from θ; a linear problem's one-step algorithm, θ unread).
#       The step's covariance is not read off `ws`: it is the constraint
#       covariance (`solver_covariance`) on the consensus it was refined on.

# `residuals!` (the default that derives the whitened distance from
# `constraint_with_jacobians`) is VisualGeometryToolkitCore's constraint-problem
# protocol, which `AbstractRansacProblem` implements, so that default applies
# here. RANSAC problems still override `residuals!` for a faster batch path
# (those methods extend the imported `residuals!`).

# -----------------------------------------------------------------------------
# Optional Method Stubs with Defaults
# -----------------------------------------------------------------------------

# `codimension(problem)` (the constraint codimension d_g) is owned by
# VisualGeometryToolkitCore's problem protocol (imported above) — each RANSAC problem type implements
# it (e.g. d_g=1 for F/E/line, d_g=2 for homography / PnP). No default.

"""
    residual_dof(problem_or_type, n_in::Int) -> Int

Residual degrees of freedom on a fitted model with `n_in` inliers:

    ν = (n_in − m) · d_g

where `m = minimal_sample_size(P)` and `d_g = codimension(P)`.

Single source of truth for the per-pair scale-estimate denominator
`σ̂²_p = RSS / ν` and for the score-equation `ν`.  Use this anywhere the residual DOF appears so
that the model class drives the formula rather than scattered magic
constants (which historically drifted: an earlier `Union{E, F}`
σ̂²_p dispatch hard-coded `m = 5`, biasing F-matrix σ̂²_p down by
factor (n_in − 7)/(n_in − 5)).

Returns 0 when `n_in ≤ m` so callers can `ν > 0 || return NaN` cleanly.
"""
residual_dof(::Type{P}, n_in::Int) where {P <: AbstractRansacProblem} =
    max(0, (n_in - minimal_sample_size(P)) * codimension(P))
residual_dof(p::AbstractRansacProblem, n_in::Int) = residual_dof(typeof(p), n_in)

"""
    oracle_residual_dof(problem_or_type, n::Int) -> Int

Residual DOF when residuals are measured against an **oracle** model that was
NOT fitted to these `n` points (e.g. the COLMAP ground-truth pose evaluated on a
pair's SfM-track keypoints): no parameters are consumed, so

    ν = n · d_g          (vs `residual_dof`'s `(n − m)·d_g` for a *fitted* model)

Use this for σ̂²_p = RSS/ν under the GT geometry. Subtracting the model DOF (as a
fit would) biases σ̂²_p *up* by `n/(n−m)` — a DLT remnant, not warranted for the
oracle.
"""
oracle_residual_dof(::Type{P}, n::Int) where {P <: AbstractRansacProblem} =
    n * codimension(P)
oracle_residual_dof(p::AbstractRansacProblem, n::Int) = oracle_residual_dof(typeof(p), n)

"""
    draw_sample!(rng::Random.AbstractRNG, indices::AbstractVector{Int},
                 problem::AbstractRansacProblem)

Fill `indices` with a random sample of data point indices (without
replacement), drawn from `rng`.

Default: delegates to `draw!(rng, indices, sampler(problem))`.
Problems with a stored sampler just need to define `sampler(p::MyProblem) = p._sampler`.
The rng-leading form is the only form, and a problem overriding `draw_sample!`
must define it: routing every draw through an explicit `rng` is what makes a
seeded `ransac(...; rng = Xoshiro(seed))` run bit-reproducible, and what keeps
a threaded caller off the task-local `Random.default_rng()`.
"""
function draw_sample!(
        rng::Random.AbstractRNG, indices::AbstractVector{Int},
        problem::AbstractRansacProblem
    )
    draw!(rng, indices, sampler(problem))
    return nothing
end

"""
    sampler(problem::AbstractRansacProblem) -> AbstractSampler

Return the sampler used by this problem.

Default: `UniformSampler(data_size(p))`.
Override for problems with a stored sampler field:

    sampler(p::MyProblem) = p._sampler
"""
sampler(p::AbstractRansacProblem) = UniformSampler(data_size(p))

"""
    test_sample(problem::AbstractRansacProblem, indices::AbstractVector{Int}) -> Bool

Check whether a sample is non-degenerate (e.g., points not collinear).

Default: `true` (no degeneracy check). Override for problem-specific checks.
"""
test_sample(::AbstractRansacProblem, ::AbstractVector{Int}) = true

# =============================================================================
# RansacWorkspace convenience constructor (needs AbstractRansacProblem)
# =============================================================================

"""
    refine_workspace(::Type{P}, n_data) -> ManifoldLmWorkspace | nothing

The scratch of `refine!` for a RANSAC problem of type `P`: a sized
`ManifoldLmWorkspace{N}` for every chart problem (from `_lm_size`, so from its
`chart_dof`), `nothing` for a problem with no chart (VisualGeometryToolkitCore's
problem-protocol verb).
"""
refine_workspace(::Type{P}, n::Int) where {P <: AbstractRansacProblem} = _lm_workspace(_lm_size(P, n))

# Every local-optimization strategy this package hosts runs on the same
# workspace: the problem's refit workspace plus the posterior-weight buffers.
lo_workspace(::Type{P}, ::Type{T}, ::AbstractLocalOptimization, n::Int) where {P <: AbstractRansacProblem, T} =
    LoBuffers{T}(refine_workspace(P, n), n, model_type(P))

"""
    RansacWorkspace(problem, scoring[, local_optimization = NoLocalOptimization()][, T = Float32])
    RansacWorkspace(P, n_max, scoring[, local_optimization]; T = Float32)

The composed workspace for `problem` (or problems of type `P` with up to
`n_max` points): the scorer's workspace (`scoring_workspace(scoring, n,
codimension)` — its tables built for this scoring) and the local optimizer's
(`lo_workspace(P, T, local_optimization, n)`). `K` is the type parameter of
`AbstractRansacProblem{K}` — compile-time constant, no runtime dispatch.
"""
# The workspace of the DEFAULT scorer (its tables depend on the scale prior and
# the profile flag only, not on the half-width) — `RansacWorkspace(P, n_data[, T])`,
# what a caller pre-allocates for the `find_*` pipelines.
RansacWorkspace(::Type{P}, n::Int, ::Type{T} = Float32) where {P <: AbstractRansacProblem, T <: AbstractFloat} =
    RansacWorkspace(P, n, _default_scoring_at(P, 1.0), NoLocalOptimization(); T)
RansacWorkspace(problem::AbstractRansacProblem, scoring, ::Type{T} = Float32) where {T <: AbstractFloat} =
    RansacWorkspace(problem, scoring, NoLocalOptimization(), T)
RansacWorkspace(problem::AbstractRansacProblem, scoring, lo::AbstractLocalOptimization, ::Type{T} = Float32) where {T <: AbstractFloat} =
    RansacWorkspace(typeof(problem), data_size(problem), scoring, lo; T)
function RansacWorkspace(::Type{P}, n::Int, scoring::AbstractScoring, lo::AbstractLocalOptimization = NoLocalOptimization();
                         T::Type{<:AbstractFloat} = Float32) where {P <: AbstractRansacProblem}
    return RansacWorkspace(
        n, minimal_sample_size(P), model_type(P), T,
        scoring_workspace(scoring, n, codimension(P)), lo_workspace(P, T, lo, n)
    )
end

"""
    test_model(problem::AbstractRansacProblem, model, sample_indices::AbstractVector{Int}) -> Bool

Check whether a fitted model is feasible (e.g., positive focal length, non-degenerate
local geometry at the sample points).

The `sample_indices` are the minimal sample indices used to fit the model, enabling
point-specific validation (e.g., Jacobian determinant bounds at sample locations).

Default: `true` (no feasibility check). Override for problem-specific checks.
"""
test_model(::AbstractRansacProblem, _model, _indices::AbstractVector{Int}) = true

"""
    test_consensus(problem::AbstractRansacProblem, model, mask::BitVector) -> Bool

Validate a model against its consensus (inlier) set. Called after scoring and
mask computation, before accepting a candidate as the new best model.

Use this for checks that require the inlier set, e.g., the oriented epipolar
constraint for fundamental matrices. The `mask` is a `BitVector` where `true`
entries are classified inliers.

Default: `true` (no consensus check). Override for problem-specific checks.
"""
test_consensus(::AbstractRansacProblem, _model, _mask::BitVector) = true

"""
    refine!(lo_ws, problem, θ, weights) -> model or nothing

The local optimization's ONE step: the problem's model refined from the current
model `θ` under per-datum `weights`, used by LO-RANSAC's `update_model!` and by
direct callers (unit tests of the refit paths). The model follows the problem
(the RANSAC-family order); `lo_ws` leads as the mutated workspace.

Routed by the `refit_kind(p)` Holy trait — NOT by the model's nominal type —
through `_refine!(::RefitKind, ws, p, θ, weights)` (see `refit_kind` /
`RefitKind` below):

- `NonlinLsq` — every chart problem (`chart_dof`: H, F, E, the absolute pose,
  hand-eye, the radial homography and the radial camera) → THE chart LM from θ
  (`_refine!` above), on the problem's own chart and rows.
- A problem whose step is not a chart LM carries its own `refine!`: the
  relative-pose bundle (a Schur-reduced full-3D BA, relpose_bundle.jl); a
  LINEAR problem's one-step algorithm under the weights, θ unread —
  `LineFittingProblem` (line.jl), a `RelativeProblem` handed a `DltWorkspace`
  (the weighted DLT, cspond.jl).
- `NoRefit` (no chart) → `nothing` (no step in this shape).

`lo_ws` is the refit workspace (`refine_workspace`; a local optimizer holds it):
a `ManifoldLmWorkspace{N}` for the chart problems (every `_lm_size`; used
when its size matches, otherwise the one task-local reuse policy of
`_lm_workspace`), a `DltWorkspace` for the algebraic DLT, or `nothing` for
problems that allocate nothing ahead (line fitting). It is accepted for
signature uniformity where unused.

Excluded points are encoded as `weights[i] = 0` — there is no separate `mask`
argument. Every step short-circuits (returns `nothing`) when too few points
have non-zero weight to over-determine the model.
"""
function refine! end

# The chart-LM protocol `residual_rows!` / `residual_jacobian_rows!` is VGC's
# (robust/lm_core.jl); every chart problem adds its method pair on `ChartLm`
# (below), dispatched on the problem type.

# The shared prologue of every weighted step (the chart-LM `_refine!`, the DLT
# and the line step). ONE threshold: a point counts toward over-determining
# the model when its weight is positive — exactly when its residual rows, scaled
# by √w, carry signal. (Hand-eye and P3P used `> 1e-12` while the rest used
# `> 0` until 2026-09-21.)
@inline _underdetermined(p, weights::AbstractVector) =
    !_holds_minimal_sample(minimal_sample_size(p) + 1, count(>(zero(eltype(weights))), weights))

# =============================================================================
# The chart problems — ONE trait, ONE linearization state, ONE refit
# =============================================================================
#
# A chart problem refines its model on a minimal chart of the model (VGC's
# `Manifold`): the chart-LM refit (`refine!`) and the HQ/GNC loss lifts
# (problems/manifold_providers.jl) run on it. What differs per problem is a
# handful of FACTS, each stated once beside the problem:
#
#   chart_dof(P)                 the chart's dof (type level: a workspace is sized
#                                before a model exists) — the trait itself
#   _chart(p, θ)                 the chart at a model (default `minimal_manifold(θ)`),
#                                the ONE seeding of the refit and the lift
#   residual_rows! /             the √w-weighted rows and their Jacobian on
#   residual_jacobian_rows!      `ChartLm{C, <:P}` (VGC's LM protocol)
#
# and, where the default does not hold: `_refit_chart(p, θ, w)` (a weight-aware
# seed: the radial homography's k, the radial camera's guard), `_lm_state(p, ch,
# sqrtw)` (a state that needs an `aux`: H's whitening, PnP's pixel block),
# `_refit_state` (a refit-only weight adjustment: PnP's out-of-reach points),
# `_model_at(p, ch)` (the model a chart stands for, in the problem's frame and
# type; default `model(ch)`). The residual row count is `codimension(P)` rows
# per datum, so `_lm_size`, the lift's `(D, C)` and `refit_kind` all DERIVE from
# `chart_dof` and `codimension` — none is restated per problem.

"""
    chart_dof(P::Type) -> Int | nothing
    chart_dof(problem)

The dof of the minimal chart a problem refines its model on — the trait that
makes a problem a CHART problem (a chart-LM `refine!` and, at codimension ≤ 2,
the HQ/GNC lifts). `nothing` (the default) for a problem with no chart.
"""
chart_dof(::Type{<:AbstractRansacProblem}) = nothing
chart_dof(p::AbstractRansacProblem) = chart_dof(typeof(p))

"""
    ChartLm(ch, problem, sqrtw[, aux])

THE linearization state of every chart LM and chart lift: the chart `ch` the
iterate lives in (anchored at the current model), the problem whose rows it
evaluates, the per-row √weights and the one per-problem extra `aux` (`nothing`
unless the rows need one — H's frozen transfer whitening, PnP's pixel block and
participation buffer). Each chart problem adds VGC's `residual_rows!` /
`residual_jacobian_rows!` on `ChartLm{<:Any, <:ItsProblemType}`.
"""
struct ChartLm{C, P, S, A}
    ch::C
    problem::P
    sqrtw::S
    aux::A
end
ChartLm(ch, problem, sqrtw) = ChartLm(ch, problem, sqrtw, nothing)

# A chart re-anchored at its own origin — its `recenter` policy applied to the
# anchor itself (the unit-Frobenius representative, a re-projected rotation).
@inline _reanchored(ch) = recenter(ch, VgtkCore.zero_tangent(ch))

# The per-problem facts, with their defaults (see the section header).
_chart(_p, θ) = minimal_manifold(θ)
_refit_chart(p, θ, _weights) = _chart(p, θ)
_lm_state(p, ch, sqrtw) = ChartLm(ch, p, sqrtw)
_refit_state(p, ch, sqrtw) = _lm_state(p, ch, sqrtw)
_model_at(_p, ch) = model(ch)

# =============================================================================
# LO buffers — ONE reuse policy, ONE chart-LM skeleton
# =============================================================================
#
# Every local optimization draws its buffers the same way: the caller-owned
# `RansacWorkspace.lo_ws` when it is the right type AND size, otherwise a
# task-local buffer set keyed by its type and sizes — built once per (task,
# size), reused by every later LO call of that size on that task. A
# caller-pooled workspace is sized to the LARGEST problem it serves and cannot
# be trimmed in place (`J` is a Matrix), so without the memo an oversized
# `lo_ws` fell through to a fresh allocation on every LO call (~100× per
# RANSAC; measured 2026-08-14 on the BlobBoards engine). Task-local so the
# engine's threaded per-board fits never share (`Threads.@threads` bodies are
# tasks). The E and radial-camera lifts used to park theirs on `ws.lo_ws`,
# the other lifts allocated per call, and the chart LMs split between a memo
# and a fresh allocation; this is the one policy for all of them — VGC's
# `task_buffers` (robust/lift_core.jl), which the lift workspace draws from too.

"""
    _lm_size(P::Type, n_data) -> (rows, Val(dof)) | nothing
    _lm_size(problem)

The shape of a problem type's chart-LM refit at `n_data` points — its residual
row count (`codimension(P)` rows per datum) and its `chart_dof`, derived from
those two facts: `_alloc_lo_workspace` allocates from it and the chart-LM
refit (`_refine!`) sizes its buffers and row weights from it.
`nothing` for a problem with no chart. Type-level, so a workspace can be sized
before a problem exists.
"""
_lm_size(::Type{P}, n::Int) where {P <: AbstractRansacProblem} = _lm_size(chart_dof(P), P, n)
_lm_size(::Nothing, _P, ::Int) = nothing
_lm_size(D::Int, ::Type{P}, n::Int) where {P} = (codimension(P) * n, Val(D))
_lm_size(p::AbstractRansacProblem) = _lm_size(typeof(p), data_size(p))

# THE constructor of a chart-LM workspace from an `_lm_size`.
_lm_workspace(::Nothing) = nothing
_lm_workspace(sz::Tuple{Int, Val}) = ManifoldLmWorkspace(sz[1], sz[2])

# The chart LM's workspace under the one reuse policy (the concrete type of
# the N-dof workspace, so the refit stays inferred: VGC `lm_workspace_type`).
@inline function _lm_workspace(lo_ws, rows::Int, v::Val{N}) where {N}
    W = lm_workspace_type(v)
    lo_ws isa W && length(sqrt_weights(lo_ws)) == rows && return lo_ws
    return task_buffers(W, () -> _lm_workspace((rows, v)), (:rvg_lm_ws, rows))
end

# The chart LM's iteration cap, stall tolerance and cost tolerance
# (configs/default.toml [chart_lm]), read once: the refit runs per LO
# iteration, so it does not look the file up per call. The stall tolerance is
# every chart LM's (the LO refit, `fit_lens`) and the stall postcondition of
# `refine_robot_world_hand_eye`'s NLLSsolver LM (`is_stationary`); the cost
# tolerance `ftol` is the LO refit's alone.
const _CHART_LM_MAX_ITER = _dflt("chart_lm", "max_iter")::Int
const _CHART_LM_STATIONARITY_TOL = _dflt("chart_lm", "stationarity_tol")::Float64
const _CHART_LM_FTOL = _dflt("chart_lm", "ftol")::Float64

# The refit-kind trait's two classes (the router and `refit_kind` below).
abstract type RefitKind end
struct NonlinLsq <: RefitKind end
struct NoRefit <: RefitKind end

"""
    _refine!(::NonlinLsq, lo_ws, problem, θ, weights) -> model | nothing

THE chart-LM refit every chart problem runs: the underdetermined guard, the
chart at θ (`_refit_chart`), the workspace (one reuse policy), the √weights for
each datum's rows, the state (`_refit_state`) and VGC's chart LM
(`lmsolve_chart!`: FastLevenbergMarquardt's `lmsolve!` on the state's rows), the
NaN guard, and the exit — the chart re-anchored at the solution by its own
`recenter` (the one gauge every model the stack returns is in: the lift
re-anchors every step the same way) and read as a model (`_model_at`). The LM's
retcode is not read: a `:stalled` refit's model still lowered the cost from θ,
and the LO's own acceptance (the scoring) decides whether it is kept.
"""
function _refine!(::NonlinLsq, lo_ws, p::AbstractRansacProblem, θ, weights::AbstractVector)
    _underdetermined(p, weights) && return nothing
    ch = _refit_chart(p, θ, weights)
    ch === nothing && return nothing
    rows, dof = _lm_size(p)
    n = data_size(p)
    ews = _lm_workspace(lo_ws, rows, dof)
    lm = _refit_state(p, ch, fill_sqrt_weights!(sqrt_weights(ews), weights, n, rows ÷ n))
    x, _, _ = lmsolve_chart!(ews, lm; maxit = _CHART_LM_MAX_ITER, stationarity_tol = _CHART_LM_STATIONARITY_TOL, ftol = _CHART_LM_FTOL)
    any(isnan, x) && return nothing
    return _model_at(p, recenter(ch, x))
end

# `lmsolve_chart!` (FastLevenbergMarquardt's `lmsolve!` on one linearization
# state's rows) and `fill_sqrt_weights!` are VGC's (robust/lm_core.jl).

# =============================================================================
# The stopping criterion of the posterior-weight iteration (VGC
# `StoppingCriterion`): pure, it reads the verdict `post_iteration!` keeps in
# the state. Public: LocalOptimizationDoneRight's score-matched BCD reuses it.
# =============================================================================

"""
    StopWhenNoImprovement(patience, first_iteration)

`:converged` once `patience` consecutive iterations have failed to improve the
incumbent ([`iterations_without_improvement`](@ref)), from iteration
`first_iteration` on.
"""
struct StopWhenNoImprovement <: StoppingCriterion
    patience::Int
    first_iteration::Int
end
@inline stop_reason(c::StopWhenNoImprovement, state, method, prob, θ, θ_old, iter) =
    iter >= c.first_iteration && iterations_without_improvement(state) >= c.patience ? :converged : :continue

"""
    iterations_without_improvement(state) -> Int

The consecutive iterations that have failed to improve the incumbent, as
[`StopWhenNoImprovement`](@ref) reads it. The default: a state that keeps only
whether the LAST iteration improved (its `improved` field) counts 0 or 1; a
state that counts further adds its method.
"""
@inline iterations_without_improvement(state) = state.improved ? 0 : 1

# =============================================================================
# Refit-kind trait — routes the LO step `refine!` by problem CLASS, not by the
# model's nominal type. Replaces the old `θ::AbstractHomMat` dispatch, which
# (mis)used the transform-vs-bilinear-form hierarchy as a stand-in for "DLT-
# refittable" and so silently excluded F and E (both <: AbstractFundMat, a
# sibling of AbstractHomMat) — their LO refit fell through to `nothing`.
#
#   NonlinLsq    — a chart problem (`chart_dof`) → the chart LM from θ
#   NoRefit      — no refit in this shape
#
# DERIVED from `chart_dof`, never stated per problem. A problem whose step is not
# a chart LM (the line's one-step algorithm, the relative-pose bundle, the DLT
# workspace form) carries its own `refine!`, strictly more specific than the
# router.
# =============================================================================

refit_kind(p::AbstractRansacProblem) = chart_dof(p) === nothing ? NoRefit() : NonlinLsq()

"""
    fit_hypothesis(problem, θ, sample_indices) -> model | nothing

The hypothesis a minimal sample contributes, given a minimal solution `θ` of it
(`solve`). For a problem whose minimal sample exactly determines the model this
is `θ` itself (the default). A problem whose minimal sample OVER-determines it
(hand-eye: two motions, 12 rows for 6 dof) fits the sample, seeded by `θ`, so
the hypothesis is the sample's least-squares model — the estimator its
`solver_covariance` describes. `solve` stays the pure minimal solver; `nothing`
skips the hypothesis, as a failed solve does.
"""
fit_hypothesis(::AbstractRansacProblem, θ, ::AbstractVector{Int}) = θ

# The router. Concrete problems carrying their own `refine!(ws, p, θ, w)` method
# (the line step, the relative-pose bundle, the DLT workspace form) are
# strictly more specific and bypass this.
refine!(ws, p::AbstractRansacProblem, θ, weights::AbstractVector) =
    _refine!(refit_kind(p), ws, p, θ, weights)

# The order is `refine!(ws, p, θ, weights)`. A call in the old `(weights, θ)` order
# is a MethodError wherever θ is typed: every refit method types it (a
# StaticMatrix, an EuclideanMap, a RadialCamera — none an AbstractVector). This
# untyped fallback is the one place a vector-valued model (the `SVector{2}` line
# problems) could bind swapped, so it checks the length instead: loud there too.
function _refine!(::NoRefit, _ws, p::AbstractRansacProblem, _θ, weights::AbstractVector)
    length(weights) >= data_size(p) || throw(
        DimensionMismatch(
            "refine!(ws, problem, θ, weights): $(length(weights)) weights for " *
                "$(data_size(p)) data — θ and weights supplied in the old order?"
        )
    )
    return nothing
end

# `solver_covariance(problem, model, indices)` and the
# `provides_solver_covariance` trait are owned by VisualGeometryToolkitCore
# (predictive scoring reads the minimal sample's through the solver covariance; the
# loop wraps a refit and a seed itself, on their consensus set, loop.jl
# `_covariance_scored`): a problem without a solver-covariance provider is refused
# by predictive scoring, and a provider's `nothing` marks a degenerate sample
# (skipped). RVG's `solver_covariance` is ONE function for every provider (H,
# F, E, the absolute pose, hand-eye, the inhomogeneous line): VGC's
# `implicit_covariance` on the stacked constraint rows (estimators/ransac/
# covariance.jl).


# `constraint_with_jacobians(problem::AbstractRansacProblem, model, i::Int)` is the
# problem-level convenience added by RVG: it looks up the i-th data point on
# `problem` and forwards to VGC's `constraint_with_jacobians(model, ...)`. RVG
# imports VGC's `constraint_with_jacobians` at module top level, so the methods
# below extend the same function — there is no separate RVG-side function with
# the same name. Concrete `(problem, model, i::Int)` methods live in each
# problem file (cspond.jl, line.jl).


# NOTE: SvdWorkspace and svd_nullvec! are defined in VisualGeometryToolkitCore
# (imported at module top level) since VGC's own solvers also need them.

# =============================================================================
# Problem Lifecycle: prepare! (called before RANSAC main loop)
# =============================================================================

"""
    _prepare!(problem::AbstractRansacProblem)

Called once before the RANSAC main loop starts.

Default: resets the problem's sampler via `reset!(sampler(problem))`. The
two-argument form, which `ransac` calls, also hands the sampler the run's
trial budget (`reset!(sampler(problem), budget)`).
"""
_prepare!(p::AbstractRansacProblem) = (reset!(sampler(p)); nothing)
_prepare!(p::AbstractRansacProblem, budget::Int) = (reset!(sampler(p), budget); nothing)

# =============================================================================
# Shared Utilities: Sampler Builder
# =============================================================================

"""
    _build_sampler(n, m, scores) -> AbstractSampler

Build a RANSAC sampler from `n` data points and minimal sample size `m`.

- `scores === nothing` → `UniformSampler(n)`
- `scores::AbstractVector{<:Real}` → `ProsacSampler` ordered by descending score
"""
@inline _build_sampler(n::Int, ::Int, ::Nothing) = UniformSampler(n)

function _build_sampler(n::Int, m::Int, scores::AbstractVector{<:Real})
    length(scores) == n || throw(
        ArgumentError(
            "scores length $(length(scores)) does not match $n correspondences"
        )
    )
    order = sortperm(scores; rev = true)
    return ProsacSampler(order, m)
end
