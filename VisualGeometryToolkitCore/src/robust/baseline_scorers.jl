# =============================================================================
# The baseline scorers — RANSAC and MSAC (moved from RansacScoringDoneRight's
# loss-width families, not rewritten) — and their scoring-contract glue.
# =============================================================================
#
# One type per scorer, two uses:
#   * WIDTH-FREE (`MsacScoring()`, `MsacScoring(dg)`, threshold `nothing`): the
#     grid family RansacScoringDoneRight sweeps a loss width over (its
#     `grid_evaluate`, `best_pool_model`, `pool_model_score` methods);
#   * AT A THRESHOLD (`MsacScoring(dg; threshold = τ)`): the RANSAC-loop scorer of
#     the scoring contract. Its score is the SAME expression as RSDR's
#     `pool_model_score` on one residual vector: Σ `residual_quality(sc, d², τ)`
#     in index order, the minimal sample's points entering at zero residual.
#
# MSAC's loss is truncated least squares (`loss_shape(::MsacScoring) ===
# TLSShape()`): profiling the labels is a min of a quadratic against a constant.
# RANSAC is the same shape under the other reduction, the inlier count
# `d² < τ²` — TLS's `weight`.
# =============================================================================

"""
    RansacScoring{DG, T}

RANSAC inlier counting at codimension `DG`: a model's score is the number of
residuals below the threshold (see [`residual_quality`](@ref)).

- `RansacScoring()`, `RansacScoring(dg)`, `RansacScoring(problem)`: the
  width-free grid family (`T = Nothing`);
- `RansacScoring(dg; threshold = τ)`, `RansacScoring(problem; threshold = τ)`:
  the loop scorer at threshold `τ` (in the problem's residual unit — pixels for
  the pixel-data problems), `T = Float64`.
"""
struct RansacScoring{DG, T} <: AbstractScoring
    threshold::T
    RansacScoring{DG, T}(threshold) where {DG, T} = new{DG, T}(threshold)
end

"""
    MsacScoring{DG, T}

MSAC at codimension `DG`: the truncated quadratic `1 − ρ(d², τ²)/τ²` at the
threshold `τ`, summed over the residuals (see [`residual_quality`](@ref)).
Constructors as [`RansacScoring`](@ref): width-free `MsacScoring()` /
`MsacScoring(dg)` / `MsacScoring(problem)`, or the loop scorer
`MsacScoring(dg; threshold = τ)`.
"""
struct MsacScoring{DG, T} <: AbstractScoring
    threshold::T
    MsacScoring{DG, T}(threshold) where {DG, T} = new{DG, T}(threshold)
end

const _ThresholdScoring{DG, T} = Union{RansacScoring{DG, T}, MsacScoring{DG, T}}
const _ScoringAtThreshold = Union{RansacScoring{<:Any, Float64}, MsacScoring{<:Any, Float64}}

for S in (:RansacScoring, :MsacScoring)
    @eval begin
        $S{DG}() where {DG} = $S{DG, Nothing}(nothing)
        $S() = $S{1}()
        $S(dg::Integer; threshold::Union{Nothing, Real} = nothing) = _at_threshold($S{Int(dg)}, threshold)
        $S(problem::AbstractConstraintProblem; threshold::Union{Nothing, Real} = nothing) =
            _at_threshold($S{codimension(problem)}, threshold)
        $S(::Type{P}; threshold::Union{Nothing, Real} = nothing) where {P <: AbstractConstraintProblem} =
            _at_threshold($S{codimension(P)}, threshold)
    end
end
_at_threshold(::Type{S}, ::Nothing) where {S} = S{Nothing}(nothing)
function _at_threshold(::Type{S}, τ::Real) where {S}
    (isfinite(τ) && τ > 0) || throw(ArgumentError("$(nameof(S)): threshold must be positive and finite, got $τ"))
    return S{Float64}(Float64(τ))
end

codimension(::_ThresholdScoring{DG}) where {DG} = DG
loss_shape(::MsacScoring) = TLSShape()

# The width a threshold scorer is built at is its truncated loss's width: the
# loss-width axis, set by `with_width`.
width_axis(::_ThresholdScoring) = LossWidth()
with_width(sc::RansacScoring{DG}, w::LossWidth{NTuple{2, Float64}}) where {DG} = RansacScoring(DG; threshold = loss_width(w))
with_width(sc::MsacScoring{DG}, w::LossWidth{NTuple{2, Float64}}) where {DG} = MsacScoring(DG; threshold = loss_width(w))

"""
    residual_quality(scoring, d2::Float64, loss_width::Float64) -> Float64

The score contribution of ONE residual with squared value `d2` under a
loss-width family `scoring` at `loss_width`; a model's score is the sum over its
residuals, and higher is better:

- RANSAC: the inlier indicator `d² < loss_width²` (TLS's `weight`);
- MSAC: `1 − ρ(TLSShape(), d², loss_width²)/loss_width²`.

RansacScoringDoneRight adds the methods of its paper baselines (MLESAC,
MAGSAC++, the loss costs). Plain Float64 functions, callable inside a kernel. A
non-finite `d2` is an outlier and is skipped by every caller; it is not an input.
"""
function residual_quality end

@inline residual_quality(::RansacScoring, d2::Float64, loss_width::Float64) =
    weight(TLSShape(), d2, loss_width * loss_width)

@inline function residual_quality(sc::MsacScoring, d2::Float64, loss_width::Float64)
    sq_loss_width = loss_width * loss_width
    return 1.0 - ρ(loss_shape(sc), d2, sq_loss_width) / sq_loss_width
end

# =============================================================================
# The scoring contract at a threshold
# =============================================================================

"""
    ThresholdScoringWorkspace

The workspace of a threshold scorer ([`MsacScoring`](@ref) / [`RansacScoring`](@ref)
at a threshold): the consensus order of the last [`score!`](@ref) (inliers
first, each part in index order — a stable partition, no sort) and its size.
"""
mutable struct ThresholdScoringWorkspace
    order::Vector{Int}
    k::Int
end
scoring_workspace(::_ScoringAtThreshold, n::Int, _d_g::Int) = ThresholdScoringWorkspace(Vector{Int}(undef, n), 0)
Base.resize!(W::ThresholdScoringWorkspace, n::Integer) = (resize!(W.order, n); W)

inlier_count(::_ScoringAtThreshold, W::ThresholdScoringWorkspace) = W.k
consensus_order(::_ScoringAtThreshold, W::ThresholdScoringWorkspace) = W.order
inlier_threshold(sc::_ScoringAtThreshold, _buf) = sc.threshold
has_inlier_threshold(::Type{<:_ScoringAtThreshold}) = true
score_bound(::_ScoringAtThreshold) = 1.0

# Whether `i` is one of the minimal sample's points (a handful of comparisons).
@inline _in_sample(::Nothing, _i) = false
@inline function _in_sample(sample, i)
    for s in sample
        s == i && return true
    end
    return false
end

# The conservative margin of the early exit: a sum of n terms each ≤ b, summed
# left to right in Float64, can exceed its exact value by at most n·eps·(|s| + n·b);
# the exit fires only when even that cannot reach the bound.
@inline _exit_margin(n, s, b) = n * eps(Float64) * (abs(s) + n * b) + eps(Float64)

function score!(W::ThresholdScoringWorkspace, sc::_ScoringAtThreshold, scores::AbstractVector,
                residuals::AbstractVector, n::Int; sample = nothing, bound::Float64 = -Inf)
    τ = sc.threshold
    τ2 = τ * τ
    q0 = residual_quality(sc, 0.0, τ)
    ns = sample === nothing ? 0 : length(sample)
    b = score_bound(sc)
    # The pool score's arithmetic (RansacScoringDoneRight `pool_model_score`):
    # Σ quality over every point in index order, a non-finite residual an
    # outlier (quality 0), then each sample point corrected to zero residual.
    s = 0.0
    k = 0; lo = 0
    order = W.order
    @inbounds for i in 1:n
        d = residuals[i]
        d2 = isfinite(d) ? Float64(d * d) : Inf
        q = isfinite(d2) ? residual_quality(sc, d2, τ) : 0.0
        s += q
        scores[i] = q
        if isfinite(bound) && s + (n - i + ns) * b < bound - _exit_margin(n + ns, s, b)
            W.k = 0
            return -Inf
        end
    end
    if sample !== nothing
        for i in sample   # bounds-checked: an invalid sample index throws (a handful of reads)
            d = residuals[i]
            s += q0 - (isfinite(d) ? residual_quality(sc, Float64(d * d), τ) : 0.0)
            scores[i] = q0
        end
    end
    # The consensus set: d² < τ² (strict, the grids' test), sample points in it;
    # a stable partition, inliers first.
    @inbounds for i in 1:n
        d = residuals[i]
        (_in_sample(sample, i) || (isfinite(d) && Float64(d * d) < τ2)) || continue
        k += 1
        order[k] = i
    end
    lo = k
    @inbounds for i in 1:n
        d = residuals[i]
        (_in_sample(sample, i) || (isfinite(d) && Float64(d * d) < τ2)) && continue
        lo += 1
        order[lo] = i
    end
    W.k = k
    return s
end
