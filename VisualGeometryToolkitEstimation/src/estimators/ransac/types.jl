# =============================================================================
# RANSAC Types — Configuration, workspace, result types
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

"""
    RansacConfig

Configuration parameters for the RANSAC algorithm.

Field names match [posebench](https://github.com/PoseLib/posebench) conventions.

`RansacConfig()` IS `configs/default.toml`'s `[ransac]` table (see
config_defaults.jl); `RansacConfig(path)` loads another file's `[ransac]` over
it. Keywords override single values.

# Fields
- `success_prob::Float64`: Desired probability of finding an all-inlier sample,
  in `(0, 1]` (`1` runs the full `max_trials` budget once a model is found)
- `max_trials::Int`: Maximum number of random trials, `≥ 1`
- `min_trials::Int`: Minimum number of trials (overrides adaptive stopping), `≥ 0`

Out-of-range values are an `ArgumentError` at construction, for every sampler
alike.

# Example
```julia
config = RansacConfig(success_prob=0.999, max_trials=50_000)
result = solve(EstimationProblem(problem), Ransac(; scoring, config); rng)
```
"""
Base.@kwdef struct RansacConfig
    success_prob::Float64 = _dflt("ransac", "success_prob")
    max_trials::Int = _dflt("ransac", "max_trials")
    min_trials::Int = _dflt("ransac", "min_trials")
    function RansacConfig(success_prob::Real, max_trials::Integer, min_trials::Integer)
        0 < success_prob <= 1 || throw(ArgumentError("success_prob must be in (0, 1], got $success_prob"))
        max_trials >= 1 || throw(ArgumentError("max_trials must be ≥ 1, got $max_trials"))
        min_trials >= 0 || throw(ArgumentError("min_trials must be ≥ 0, got $min_trials"))
        return new(success_prob, max_trials, min_trials)
    end
end

# -----------------------------------------------------------------------------
# Result Types — FitResult with RANSAC / LO metadata
# -----------------------------------------------------------------------------
#
# Every estimator here returns VisualGeometryToolkitCore's `FitResult`: SciML's
# fields (`u`, `resid`, `retcode`, `stats`, …) and the stack's (`stop_rule`,
# `rms`, `n`) plus one concrete metadata type selected at compile time. RANSAC
# carries its consensus diagnostics in `RansacMetadata`; a local-optimization
# refit is VisualGeometryToolkitCore's lean `lo_estimate` result. A result's own
# data are read by field (`r.u`, `r.metadata.quality`); the per-datum weights,
# their kind, the inlier mask and the scale through Core's accessors (`weights`,
# `weight_kind`, `inlier_mask`, `noise_scale`, `scale_source`).

"""
    RansacMetadata{T, S}

RANSAC-specific diagnostics, the `metadata` of a [`RansacEstimate`](@ref):
`T` the per-point storage precision, `S` the SCORER's type — the producer
whose `weight_kind` gives the weights their meaning (a threshold scorer's
`BinaryWeights`, the Student-t + uniform scorer's `PosteriorWeights`).

The result's own fields carry the rest: the termination record (`r.retcode`,
`r.stop_rule`; below), `r.stats.nsteps` the number of trials (hypotheses
drawn — SciML's counter), `r.resid` the final residuals of every data point
(storage precision), and `n` and `rms` the weights' support and weighted
root-mean-square residual:

- `n = count(>(0), weights)`;
- `rms = √(Σᵢ wᵢ rᵢ² / Σᵢ wᵢ)`.

For a threshold scorer (binary weights) these are the consensus set's size and
its rms, so `rms` is not outlier-dominated. For a posterior scorer they span
every point of non-negligible posterior, each counted at its weight.

# Fields
- `weights::Vector{T}`: per-point weights in [0, 1] (storage precision), read by
  `weights(result)`: the consensus membership 1/0 for a threshold scorer, the
  posterior inlier probability for the Student-t + uniform scorer (a value
  below `eps(T)` is stored as an exact 0). `inlier_mask(result)` cuts them by
  the scorer's kind (`w > 0`; the MAP rule `w > ½`)
- `quality::Float64`: total model quality (higher = better) — the marginal score
- `scale::Float64`: the scale the scorer ORIGINATES, read by `noise_scale(result)`:
  the scale MAP `√ŝ²` (Core's `ScaleMAP(prior)` on `(RSS, ν)`) of the consensus partition — the inverse-gamma
  MAP (VisualGeometryToolkitCore's `consensus_s2`) under the scoring's scale
  prior, `√((RSS + 2β₀)/(ν + 2α₀))`, so `√(RSS/ν)` under the default Jeffreys
  prior, floored at `√eps` (a noise-free partition, `RSS = 0`, gives ≈ 1.49e-8,
  not 0). A LINEAR scale, not a variance and not a loss width. `NaN` when there
  is no partition to scale (`dof == 0`, or no model)
- `scale_source::ScaleSource`: `ScaleEstimated` (the scorer's partition), or
  `ScaleNone` with a `NaN` scale, which `noise_scale` refuses
- `dof::Int`: residual degrees of freedom `ν = d_g·(n_in − p)` of the consensus
  partition (`n_in` its size, `p` the minimal-sample dof); 0 when it is too thin
- `sample_acceptance_rate::Float64`: fraction of trials that succeeded at `test_sample`
  (effective trials / `iterations`); low values mean frequent sample degeneracy
- `lo_retcode::ReturnCode.T`, `lo_stop_rule::StopRule`: the termination record of
  the best model's local optimization, the core of its
  [`lo_estimate`](@extref VisualGeometryToolkitCore.lo_estimate) result (the same record as
  every result's): `(Success, ClosedForm)` for no local optimization (`nothing`),
  `(InitialFailure, NoRule)` for a consensus too thin to refine, an iterative
  optimizer's own; the refit's code with the rule `SeedKept` when the score guard
  kept the seed; `(ReturnCode.Default, NoRule)` — SciMLBase's "not run" — when no
  local optimization ran for the best model (no model, or the seed kept unscored)
- `nonfinite_scale::Bool`: the consensus partition's scale came out non-finite (an
  overflowing RSS) and was dropped: `scale_source` `ScaleNone`, `scale` `NaN`, `dof`
  the partition's. An estimated scale is finite: `ScaleEstimated` never carries a
  non-finite one

`quality` and `scale` are `Float64` regardless of `T`: scalars carry no
footprint argument, and these are the numbers downstream thresholds compare.
"""
struct RansacMetadata{T, S} <: AbstractFitMetadata{S}
    weights::Vector{T}
    quality::Float64
    scale::Float64
    scale_source::ScaleSource
    dof::Int
    sample_acceptance_rate::Float64
    lo_retcode::ReturnCode.T
    lo_stop_rule::StopRule
    nonfinite_scale::Bool
end
# A result whose scale was not dropped (every result before the field, and every search with no model).
RansacMetadata{T, S}(weights, quality, scale, scale_source, dof, sample_acceptance_rate, lo_retcode, lo_stop_rule) where {T, S} =
    RansacMetadata{T, S}(weights, quality, scale, scale_source, dof, sample_acceptance_rate, lo_retcode, lo_stop_rule, false)

# The local-optimization record of a model no local optimization ran for: SciMLBase's "not run".
const _LO_NOT_RUN = (ReturnCode.Default, NoRule)

# The weights' kind is the scorer's, the producer (Core's one `AbstractFitMetadata` method).

"""
    RansacEstimate{M, T, S} = FitResult{Union{M, Nothing}, RansacMetadata{T, S}}

The result of [`Ransac`](@ref) and [`LocalOptimization`](@ref): `M` the model, `T` the storage precision, `S` the
scorer's type. Read by field — `r.u` (the model, `nothing` for a search that found none), `r.retcode`, `r.stop_rule`,
`r.stats.nsteps` (for RANSAC the number of trials, hypotheses drawn), `r.resid`, `r.rms`, `r.n`, `r.metadata.quality`
— and the cross-type quantities by their functions: `found_model(r)`, `weights(r)`, `weight_kind(r)`,
`inlier_mask(r)`, `noise_scale(r)`, `scale_source(r)`.

The termination record of the search:

| stop | `retcode` | `stop_rule` |
|:--|:--|:--|
| no hypothesis held a consensus (no estimate) | `Failure` | `TrialBudget` |
| the adaptive trial count drawn (the confidence `success_prob` reached) | `Success` | `AdaptiveCount` |
| an exhaustive sampler drew every sample | `Success` | `TrialBudget` |
| `max_trials` drawn without reaching the confidence | `MaxIters` | `TrialBudget` |
| a selection among fixed candidates, or a seed kept unrefined | `Success` | `ClosedForm` |
| a seed refined by a local optimization | that refit's record | |

`found_model(r)` is THE "a model was found" predicate; `successful_retcode(r)` is strict (false for a search that drew
`max_trials` without reaching its confidence, which still found a model). A failed search — `found_model(r) == false`,
`r.retcode == ReturnCode.Failure` — has no estimate: its `u` is `nothing` (so `r.u` is `Union{M, Nothing}`, Base's
`findfirst` idiom), its `quality` is `-Inf`, its weights all zero (an all-false `inlier_mask`), `rms` NaN and its scale
`ScaleNone`.
"""
const RansacEstimate{M, T, S} = FitResult{Union{M, Nothing}, RansacMetadata{T, S}}

"""
    inlier_ratio(result::RansacEstimate) -> Float64

Fraction of data points that are inliers, `count(inlier_mask(result)) / length(weights(result))`.
"""
inlier_ratio(r::RansacEstimate) = count(VgtkCore.inlier_mask(r)) / length(weights(r))

"""
    PooledScaleMetadata

The `metadata` of a [`PooledScale`](@ref) result: `scale` the pooled noise scale (read by `noise_scale`),
`scale_source` `ScaleEstimated`, `dof` the pooled degrees of freedom `Σνᵢ`, `pooled` the number of entries that entered
the pool and `left_out` the number that did not (no scale, or no degrees of freedom).
"""
struct PooledScaleMetadata
    scale::Float64
    scale_source::ScaleSource
    dof::Int
    pooled::Int
    left_out::Int
end

"""
    PooledScale()

A [`ClosedFormKind`](@extref VisualGeometryToolkitCore.ClosedFormKind): ONE noise scale for a set of results that share
one noise model (one per view, board or photograph of a set), the degrees-of-freedom-weighted pool of their scales,

    σ̂ = √(Σᵢ νᵢ ŝᵢ² / Σᵢ νᵢ),   ŝᵢ = noise_scale(rᵢ),   νᵢ = dof(rᵢ),

the pooled variance of their consensus partitions:

- `solve(EstimationProblem(NoiseScale, results), PooledScale())`: the results themselves; a result without a scale
  (`ScaleNone`: a failed search) or without degrees of freedom (`dof == 0`) is left out and counted;
- `solve(EstimationProblem(NoiseScale, scales => dofs), PooledScale())`: the numbers a result records — each one's noise
  scale `ŝᵢ` and its degrees of freedom `νᵢ`, index-aligned (a record read back from disk) — an entry whose scale is not
  finite (`NaN`: none recorded) or whose `νᵢ ≤ 0` left out and counted; a `DimensionMismatch` when the lengths differ.

`u` is `σ̂`, also the metadata's `scale` (`ScaleEstimated`), so a downstream stage takes it as `scale = result` and
records `ScaleUpstream`; `n` the entries pooled; its termination record a closed form's (`Success`, `ClosedForm`), or
`(InitialFailure, NoRule)` (`u` and scale `NaN`, `ScaleNone`) when nothing enters the pool.
"""
struct PooledScale <: AbstractEstimationAlgorithm end
estimator_kind(::PooledScale) = ClosedFormKind()

__solve(prob::EstimationProblem{NoiseScale, <:AbstractVector{<:FitResult}, NullParameters}, ::PooledScale) =
    _pooled_scale([scale_source(r) == ScaleNone ? NaN : Float64(noise_scale(r)) for r in prob.f], [dof(r) for r in prob.f])
__solve(prob::EstimationProblem{NoiseScale, <:Pair{<:AbstractVector, <:AbstractVector}, NullParameters}, ::PooledScale) =
    _pooled_scale(first(prob.f), last(prob.f))

function _pooled_scale(scales::AbstractVector, dofs::AbstractVector)
    length(scales) == length(dofs) ||
        throw(DimensionMismatch("PooledScale: $(length(scales)) scales, $(length(dofs)) degrees of freedom"))
    num, den, pooled, left_out = 0.0, 0, 0, 0
    for (s, ν) in zip(scales, dofs)
        if ν <= 0 || !isfinite(s)
            left_out += 1
            continue
        end
        num += ν * Float64(s)^2; den += ν; pooled += 1
    end
    pooled == 0 && return FitResult(NaN, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, PooledScaleMetadata(NaN, ScaleNone, 0, 0, left_out))
    σ = sqrt(num / den)
    return FitResult(σ, ReturnCode.Success, ClosedForm, SolveStats(), NaN, pooled, PooledScaleMetadata(σ, ScaleEstimated, den, pooled, left_out))
end

# The lean result of one local-optimization refit is VisualGeometryToolkitCore's
# `lo_estimate(model, retcode; iterations, residuals_current)` (the LO contract).

# -----------------------------------------------------------------------------
# DltWorkspace — pre-allocated DLT solver workspace
# -----------------------------------------------------------------------------

"""
    DltWorkspace{C, T}

Pre-allocated workspace for SVD-based DLT solvers (the no-θ algebraic
`refine!(ws::DltWorkspace, p, weights)` form — the LO refits are NonlinLsq
manifold LM on a `RefitSlot` since the AlgebraicLsq excision).

Holds the design matrix buffer (`buf`) and the LAPACK SVD workspace (`svd`).
Sized at construction by the column count `C` (= 9 for 3×3 homogeneous models
like H/F, = 12 for the 3×4 camera matrix case) and the maximum row count.

# Type Parameters
- `C`: DLT column count (= number of model parameters, e.g. 9 for 3×3, 12 for 3×4)
- `T`: scalar float type

# Constructor
```julia
ws = DltWorkspace{9, Float64}(n_rows)
```
"""
struct DltWorkspace{C, T <: AbstractFloat}
    buf::FixedSizeArray{T, 2, Memory{T}}
    svd::SvdWorkspace{T}
end

DltWorkspace{C, T}(n_rows::Int) where {C, T <: AbstractFloat} =
    DltWorkspace{C, T}(FixedSizeArray{T}(undef, n_rows, C), SvdWorkspace{T}(n_rows, C))

# `buf` (design matrix) and `svd` already take runtime row counts via
# `view(buf, 1:2k, :)` / `svd_nullvec!(..., 2k, ...)` — nothing to resize.
_set_n!(::DltWorkspace, ::Int) = nothing
_set_n!(::Nothing, ::Int) = nothing

# -----------------------------------------------------------------------------
# Workspace (Pre-allocated Buffers)
# -----------------------------------------------------------------------------

"""
    LoBuffers{R, T, S}

The local-optimization workspace of every strategy this package hosts
(VisualGeometryToolkitCore's `lo_workspace`), built once per composed workspace: the
problem's refit workspace `refine` ([`refit_workspace`](@ref): a
`RefitSlot` for the chart problems, its summary-free refine cache; `nothing` for a problem whose
refit pre-allocates nothing) and the per-point buffers of the posterior-weight
iteration (`posterior_w`, `posterior_inlier`, `posterior_sq_r`, Float64) and of
the IRLS residual block (`lo_residuals`, `lo_best_residuals`, storage `T`).
The per-call states live here too, reset by every call instead of built:
`posterior`, the `PosteriorWeightState` on the posterior buffers (re-seeded by
[`consensus_posterior`](@ref)), and `irls::S`, the posterior-weight
iteration's `PosteriorIrlsState` on the residual buffers.
"""
struct LoBuffers{R, T <: AbstractFloat, S}
    refine::R
    posterior_w::Vector{Float64}
    posterior_inlier::Vector{Float64}
    posterior_sq_r::Vector{Float64}
    lo_residuals::Vector{T}
    lo_best_residuals::Vector{T}
    posterior::PosteriorWeightState
    irls::S
end
function LoBuffers{T}(refine::R, n::Int, ::Type{M}) where {R, T, M}
    w, π, sq = Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    r, rbest = Vector{T}(undef, n), Vector{T}(undef, n)
    posterior = PosteriorWeightState(w, π, sq, NaN, NaN, NaN, 1, JeffreysPrior())
    irls = PosteriorIrlsState{M, T}(posterior, r, rbest)
    return LoBuffers{R, T, typeof(irls)}(refine, w, π, sq, r, rbest, posterior, irls)
end

function _set_n!(L::LoBuffers, n::Int)
    resize!(L.posterior_w, n); resize!(L.posterior_inlier, n); resize!(L.posterior_sq_r, n)
    resize!(L.lo_residuals, n); resize!(L.lo_best_residuals, n)
    _set_n!(L.refine, n)
    return L
end

"""
    RansacWorkspace{K, M, T, W, L}

The composed workspace of one RANSAC run, built once and reused across trials
and calls: the scoring buffers `buf` (VisualGeometryToolkitCore's `ScoreBuffers{T, W}`
— residuals, per-point scores, consensus mask, the scorer's workspace `W` and
the evaluation epoch), the local optimizer's workspace `lo_ws::L`, the
incumbent's buffers (`best_*`) and the minimal sample's indices. Every
parameter is concrete, so the loop, the scorer and the local optimizer are all
statically dispatched.

# Type Parameters
- `K`: Minimal sample size (compile-time constant)
- `M`: Model type
- `T`: Per-point STORAGE element type (default `Float32`). Residuals and soft
  weights only — the per-point scores and every statistic accumulated from them
  stay `Float64` regardless of `T`.
- `W`: the scorer's workspace type (`scoring_workspace(scoring, n, d_g)`)
- `L`: the local optimizer's workspace type (`lo_workspace(P, T, lo, n)`)

# Constructors
`RansacWorkspace([T = Float32,] problem, scoring[, local_optimization_strategy])` and
its siblings, documented on the constructor.

# Reuse
Reuse one through `init(problem, Ransac(…); rng)` → `solve!` → `reinit!` to avoid
allocation across calls; a workspace sized for a larger problem is trimmed to
each call's data size.
"""
mutable struct RansacWorkspace{K, M, T <: AbstractFloat, W, L}
    const buf::ScoreBuffers{T, W}
    const lo_ws::L
    # Best-so-far (mutated in-place): the incumbent's residuals, per-point
    # scores, consensus mask and order — `_finalize` reads its scale and soft
    # weights from them with no re-score.
    best_model::M
    const best_residuals::Vector{T}
    const best_scores::Vector{Float64}
    const best_mask::BitVector
    const best_order::Vector{Int}
    best_k::Int
    # Sampling buffer (length = minimal_sample_size, statically sized)
    const sample_indices::MVector{K, Int}
    has_best::Bool
    best_lo_retcode::ReturnCode.T   # the current best model's local-optimization record (surfaced on the result)
    best_lo_stop_rule::StopRule
end

function RansacWorkspace(
        n::Int, k::Int, ::Type{M}, ::Type{T}, scoring_ws::W, lo_ws::L
    ) where {M, T <: AbstractFloat, W, L}
    # Uninitialized model placeholder — has_best = false guards against reading it
    return RansacWorkspace{k, M, T, W, L}(
        ScoreBuffers{T}(n, scoring_ws), lo_ws,
        Ref{M}()[], Vector{T}(undef, n), Vector{Float64}(undef, n), falses(n), Vector{Int}(undef, n), 0,
        MVector{k, Int}(undef), false, _LO_NOT_RUN...,
    )
end

"""
    RansacResultBuffers{T}(n)

The per-point arrays of a [`RansacEstimate`](@ref) — its `residuals` and
`weights` — preallocated, so that a [`RansacCache`](@ref)'s `solve!` writes its
result into them with no allocation. The returned estimate's metadata holds THESE
arrays: the next `solve!` of the same cache overwrites it.
"""
struct RansacResultBuffers{T <: AbstractFloat}
    residuals::Vector{T}
    weights::Vector{T}
end
RansacResultBuffers{T}(n::Int) where {T} = RansacResultBuffers{T}(Vector{T}(undef, n), Vector{T}(undef, n))
RansacResultBuffers(ws::RansacWorkspace{<:Any, <:Any, T}) where {T} = RansacResultBuffers{T}(length(ws.best_residuals))

_set_n!(r::RansacResultBuffers, n::Int) = (VgtkCore.resize_buffers!(n, r.residuals, r.weights); r)

"""
    set_n!(ws::RansacWorkspace, n::Int) -> ws

In-place trim every internal length-`data_size` buffer of `ws` to `n`. Used by
every entry when an oversized shared workspace is supplied via `workspace=`,
so all consumers see vectors whose `length` equals the current problem's
`data_size(problem)`. `resize!` is O(1) on shrink.

It cannot grow past the size the scorer's workspace was built for when that
workspace cannot grow (the sort-and-sweep's `log Γ` table): `n` beyond it is
an `ArgumentError`, thrown before any buffer is touched.
"""
function set_n!(ws::RansacWorkspace, n::Int)
    # The scorer's workspace first: it refuses an `n` it cannot hold before
    # anything else is touched.
    resize!(ws.buf.scoring_ws, n)
    resize!(ws.buf, n)
    resize!(ws.best_residuals, n)
    resize!(ws.best_scores, n)
    resize!(ws.best_mask, n)
    resize!(ws.best_order, n)
    _set_n!(ws.lo_ws, n)
    return ws
end
