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
result = ransac(problem, scoring; config)
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
# Every estimator here returns VisualGeometryToolkitCore's `FitResult{P, M}`: the
# fixed core (`estimate`, `retcode`, `iterations`, `rms`, `n`) plus one
# concrete metadata type selected at compile time. RANSAC carries its
# consensus diagnostics in `RansacMetadata`; a local-optimization refit is
# VisualGeometryToolkitCore's lean `lo_estimate` result. Metadata fields forward as properties,
# so `r.inlier_mask` reads exactly as it did on the retired
# `RansacAttributes` — only `.value → .estimate`, `.stop_reason → .retcode`
# and `.trials → .iterations` changed spelling.

"""
    RansacMetadata{T}

RANSAC-specific diagnostics, the `metadata` of a [`RansacEstimate`](@ref).
The core fields carry the rest: `retcode` is `:converged` or `:no_model` (no
estimate: test `converged(result)`),
`iterations` the number of trials (hypotheses drawn — RANSAC's loop count;
VGC's shared `FitResult` field, no second name), `rms` the root-mean-square residual over
the INLIER set and `n` that set's size — so `rms` is not outlier-dominated
(`length(inlier_mask)` is the data size).

# Fields
- `inlier_mask::BitVector`: consensus set (true = inlier)
- `residuals::Vector{T}`: final residuals for all data points (storage precision)
- `weights::Vector{T}`: posterior inlier weights (storage precision)
- `quality::Float64`: total model quality (higher = better) — the marginal score
- `sigma::Float64`: residual scale `σ = √(fit_scale(prior, RSS, ν))` when
  `dof > 0`, else `NaN` — the inverse-gamma MAP (VisualGeometryToolkitCore's
  `consensus_s2`) under the scoring's scale prior, `√((RSS + 2β₀)/(ν + 2α₀))`, so `√(RSS/ν)` under the
  default Jeffreys prior, floored at `√eps` (a noise-free partition, `RSS = 0`,
  gives ≈ 1.49e-8, not 0). A LINEAR σ, not a variance and not a loss width —
  the robust layer's own scale port speaks `s2` (the consensus set's,
  `consensus_s2`)
- `dof::Int`: residual degrees of freedom `ν = d_g·(n_in − p)` (`n_in` inliers,
  `p` the minimal-sample dof); 0 when scored at a supplied loss width
- `sample_acceptance_rate::Float64`: fraction of trials that succeeded at `test_sample`
  (effective trials / `iterations`); low values mean frequent sample degeneracy
- `lo_retcode::Symbol`: why the local-optimization loop stopped for the best
  model (see [`lo_estimate`](@ref)); `:not_run` when no LO ran

`quality` and `sigma` are `Float64` regardless of `T`: scalars carry no
footprint argument, and these are the numbers downstream thresholds compare.
"""
struct RansacMetadata{T}
    inlier_mask::BitVector
    residuals::Vector{T}
    weights::Vector{T}
    quality::Float64
    sigma::Float64
    dof::Int
    sample_acceptance_rate::Float64
    lo_retcode::Symbol
end

"""
    RansacEstimate{M, T} = FitResult{M, RansacMetadata{T}}

The return type of [`ransac`](@ref) and every `find_*` pipeline. The model and
the core diagnostics are read through VisualGeometryToolkitCore's accessors —
`estimate(result)`, `converged(result)`, `iterations(result)` (for RANSAC the
number of trials, hypotheses drawn) —
and the RANSAC diagnostics of [`RansacMetadata`](@ref) forward as properties:
`result.inlier_mask`, `result.quality`.

`converged(result)` is THE success predicate. A failed search —
`retcode(result) === :no_model`, `converged(result) == false` — is not an
estimate: its `estimate` is a `zero(M)` placeholder (the type stays concrete),
its `quality` is `-Inf`, its `inlier_mask` all false and `rms`/`sigma` NaN. Every
`find_*` pipeline returns this type and inherits the rule.
"""
const RansacEstimate{M, T} = FitResult{M, RansacMetadata{T}}

"""
    inlier_ratio(result::RansacEstimate) -> Float64

Fraction of data points classified as inliers.
"""
inlier_ratio(r::RansacEstimate) = sum(r.inlier_mask) / length(r.inlier_mask)

# The lean result of one local-optimization refit is VisualGeometryToolkitCore's
# `lo_estimate(model, retcode; iterations, residuals_current)` (the LO contract).

# -----------------------------------------------------------------------------
# DltWorkspace — pre-allocated DLT solver workspace
# -----------------------------------------------------------------------------

"""
    DltWorkspace{C, T}

Pre-allocated workspace for SVD-based DLT solvers (the no-θ algebraic
`refine!(ws::DltWorkspace, p, weights)` form — the LO refits are NonlinLsq
manifold LM on a `ManifoldLmWorkspace` since the AlgebraicLsq excision).

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
problem's refit workspace `refine` ([`refine_workspace`](@ref): a
`ManifoldLmWorkspace{N}` for the chart problems, `nothing` for a problem whose
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
```julia
ws = RansacWorkspace(problem, scoring[, local_optimization][, T])
ws = RansacWorkspace(P, n_max, scoring, local_optimization; T = Float32)
```

# Reuse
Supply a workspace to `ransac(; workspace = ws)` (or `ransac!`) to avoid
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
    best_lo_retcode::Symbol   # LO stop reason for the current best model (surfaced on the result)
end

function RansacWorkspace(
        n::Int, k::Int, ::Type{M}, ::Type{T}, scoring_ws::W, lo_ws::L
    ) where {M, T <: AbstractFloat, W, L}
    # Uninitialized model placeholder — has_best = false guards against reading it
    return RansacWorkspace{k, M, T, W, L}(
        ScoreBuffers{T}(n, scoring_ws), lo_ws,
        Ref{M}()[], Vector{T}(undef, n), Vector{Float64}(undef, n), falses(n), Vector{Int}(undef, n), 0,
        MVector{k, Int}(undef), false, :not_run,
    )
end

"""
    RansacResultBuffers{T}(n)

The per-point arrays of a [`RansacEstimate`](@ref) — `inlier_mask`,
`residuals` and the soft `weights` — preallocated, so that
[`ransac!`](@ref) writes its result into them with no allocation. The returned
estimate's metadata holds THESE arrays: the next `ransac!` into the same
buffers overwrites it.
"""
struct RansacResultBuffers{T <: AbstractFloat}
    inlier_mask::BitVector
    residuals::Vector{T}
    weights::Vector{T}
end
RansacResultBuffers{T}(n::Int) where {T} = RansacResultBuffers{T}(falses(n), Vector{T}(undef, n), Vector{T}(undef, n))
RansacResultBuffers(ws::RansacWorkspace{<:Any, <:Any, T}) where {T} = RansacResultBuffers{T}(length(ws.best_residuals))

function _set_n!(r::RansacResultBuffers, n::Int)
    resize!(r.inlier_mask, n); resize!(r.residuals, n); resize!(r.weights, n)
    return r
end

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
