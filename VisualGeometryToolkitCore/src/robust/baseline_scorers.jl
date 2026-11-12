# =============================================================================
# The threshold scorers — the abstract type whose one loop scores every
# loss-width family at a threshold, and Core's two baselines on it, RANSAC and
# MSAC (moved from RANSACScoringDoneRight's loss-width families, not rewritten).
# =============================================================================
#
# One type per scorer, two uses:
#   * WIDTH-FREE (`MsacScoring()`, `MsacScoring(dg)`, threshold `nothing`): the
#     grid family RANSACScoringDoneRight sweeps a loss width over (its
#     `grid_evaluate`, `best_pool_model`, `pool_model_score` methods);
#   * AT A THRESHOLD (`MsacScoring(dg; threshold = τ)`): the RANSAC-loop scorer of
#     the scoring contract. Its score is Σ `residual_quality(sc, d², τ)` in index
#     order, the minimal sample's points entering at zero residual.
#
# The loop is written once, on `AbstractThresholdScoring{Float64}`: a family
# at a threshold supplies its `residual_quality` (and, where it is proven, its
# `score_bound`), and inherits the rest of the scoring contract.
#
# MSAC's loss is truncated least squares (`loss_shape(::MsacScoring) ===
# TLSShape()`): profiling the labels is a min of a quadratic against a constant.
# RANSAC is the same shape under the other reduction, the inlier count
# `d² < τ²` — TLS's `weight`.
# =============================================================================

"""
    AbstractThresholdScoring{T} <: AbstractScoring

A scorer whose score is a sum of per-point qualities at one threshold `τ` (its
loss width): `S = Σᵢ `[`residual_quality`](@ref)`(scoring, dᵢ², τ)`, higher is
better. `T` is the threshold's type: `Float64` at a threshold (a scorer of the
scoring contract), `Nothing` for the width-free form, a grid family that a loss
width is swept over and that implements no scoring contract.

A subtype at a threshold implements [`residual_quality`](@ref) and holds its
threshold in a field `threshold` (or adds an [`decision_boundary`](@ref) method),
and inherits the rest from `AbstractThresholdScoring{Float64}`:

- [`score!`](@ref) on a [`ThresholdScoringWorkspace`](@ref): the one threshold
  loop. Σ quality over every point in index order, a non-finite squared residual
  an outlier (quality 0; a finite `d` whose square overflows too), then each of
  the minimal sample's points corrected to zero residual; the consensus set is
  `d² < τ²` (strict) with the sample's points in it.
- [`scoring_workspace`](@ref), [`inlier_count`](@ref), [`consensus_order`](@ref),
  [`decision_boundary`](@ref) (`τ`) and [`has_decision_boundary`](@ref) (`true`).
- [`score_bound`](@ref): `nothing` (no early exit) by default. A subtype states a
  bound `b` only where it holds `0 ≤ residual_quality(scoring, d², τ) ≤ b` for
  every finite `d² ≥ 0`, to within a few ulps of `b` (the exit's margin absorbs
  those). Both sides matter: the sample's points are corrected after the sum, so
  a negative quality could let the corrected score exceed the bound the exit
  tested.

Core's subtypes are [`RansacScoring`](@ref) and [`MsacScoring`](@ref); a
downstream package adds its own loss-width families (RANSACScoringDoneRight's
paper baselines) the same way.
"""
abstract type AbstractThresholdScoring{T} <: AbstractScoring end

"""
    RansacScoring{DG, T} <: AbstractThresholdScoring{T}

RANSAC inlier counting at codimension `DG`: a model's score is the number of
residuals below the threshold (see [`residual_quality`](@ref)).

- `RansacScoring()`, `RansacScoring(dg)`, `RansacScoring(problem)`: the
  width-free grid family (`T = Nothing`);
- `RansacScoring(dg; threshold = τ)`, `RansacScoring(problem; threshold = τ)`:
  the loop scorer at threshold `τ` (in the problem's residual unit — pixels for
  the pixel-data problems), `T = Float64`.
"""
struct RansacScoring{DG, T} <: AbstractThresholdScoring{T}
    threshold::T
    RansacScoring{DG, T}(threshold) where {DG, T} = new{DG, T}(threshold)
end

"""
    MsacScoring{DG, T} <: AbstractThresholdScoring{T}

MSAC at codimension `DG`: the truncated quadratic `1 − ρ(d², τ²)/τ²` at the
threshold `τ`, summed over the residuals (see [`residual_quality`](@ref)).
Constructors as [`RansacScoring`](@ref): width-free `MsacScoring()` /
`MsacScoring(dg)` / `MsacScoring(problem)`, or the loop scorer
`MsacScoring(dg; threshold = τ)`.
"""
struct MsacScoring{DG, T} <: AbstractThresholdScoring{T}
    threshold::T
    MsacScoring{DG, T}(threshold) where {DG, T} = new{DG, T}(threshold)
end

# The scoring families' registry, `SCORING_TAGS`, is robust/student_t_uniform_scoring.jl's (it names that
# file's `StudentTUniformScoring` too).

const _ThresholdScoring{DG, T} = Union{RansacScoring{DG, T}, MsacScoring{DG, T}}

# A scorer built for a problem TYPE reads its codimension here: the one call every problem-aware scorer constructor
# shares (these and StudentTUniformScoring's), the problem protocol's `codimension` implemented by the problem type.
_problem_codimension(::Type{P}) where {P <: AbstractConstraintProblem} = codimension(P)

"""
    @threshold_scoring_constructors S

The constructors and the width setter of a threshold-scorer family `S{DG, T} <:
`[`AbstractThresholdScoring`](@ref)`{T}` (`DG` its codimension, its threshold in
a field `threshold`, an inner constructor `S{DG, T}(threshold)`), as
[`RansacScoring`](@ref) and [`MsacScoring`](@ref) have them: `S{DG}()` and
`S()` (codimension 1) the width-free form; `S(dg; threshold)`,
`S(problem; threshold)` and `S(P; threshold)` (a problem, or a problem TYPE,
gives the codimension) at a threshold, refused unless positive and finite
(`ArgumentError`), or width-free without one; and
[`with_width`](@ref)`(::S{DG}, w::LossWidth)`, the same family at `w`'s loss
width. A downstream family (RANSACScoringDoneRight's loss-width baselines) is
given them by this one line after its struct, exactly as Core's two are.
"""
macro threshold_scoring_constructors(S)
    at, cod, pcod = GlobalRef(@__MODULE__, :_at_threshold), GlobalRef(@__MODULE__, :codimension), GlobalRef(@__MODULE__, :_problem_codimension)
    ACP, WW, LW, lw = GlobalRef(@__MODULE__, :AbstractConstraintProblem), GlobalRef(@__MODULE__, :with_width),
        GlobalRef(@__MODULE__, :LossWidth), GlobalRef(@__MODULE__, :loss_width)
    return esc(quote
        $S{DG}() where {DG} = $S{DG, Nothing}(nothing)
        $S() = $S{1}()
        $S(dg::Integer; threshold::Union{Nothing, Real} = nothing) = $at($S{Int(dg)}, threshold)
        $S(problem::$ACP; threshold::Union{Nothing, Real} = nothing) = $at($S{$cod(problem)}, threshold)
        $S(::Type{P}; threshold::Union{Nothing, Real} = nothing) where {P <: $ACP} = $at($S{$pcod(P)}, threshold)
        $WW(sc::$S{DG}, w::$LW{NTuple{2, Float64}}) where {DG} = $S(DG; threshold = $lw(w))
        nothing
    end)
end
_at_threshold(::Type{S}, ::Nothing) where {S} = S{Nothing}(nothing)
function _at_threshold(::Type{S}, τ::Real) where {S}
    (isfinite(τ) && τ > 0) || throw(ArgumentError("$(nameof(S)): threshold must be positive and finite, got $τ"))
    return S{Float64}(Float64(τ))
end
@threshold_scoring_constructors RansacScoring
@threshold_scoring_constructors MsacScoring

codimension(::_ThresholdScoring{DG}) where {DG} = DG
loss_shape(::MsacScoring) = TLSShape()

# The width a threshold scorer is built at is its truncated loss's width: the
# loss-width axis, set by `with_width` (`@threshold_scoring_constructors`).
width_axis(::_ThresholdScoring) = LossWidth()

"""
    residual_quality(scoring, d2::Float64, loss_width::Float64) -> Float64

The score contribution of ONE residual with squared value `d2` under a
loss-width family `scoring` at `loss_width`; a model's score is the sum over its
residuals, and higher is better:

- RANSAC: the inlier indicator `d² < loss_width²` (TLS's `weight`);
- MSAC: `1 − ρ(TLSShape(), d², loss_width²)/loss_width²`.

RANSACScoringDoneRight adds the methods of its paper baselines (MLESAC,
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

The workspace of a threshold scorer (an [`AbstractThresholdScoring`](@ref) at a
threshold): the consensus order of the last [`score!`](@ref) (inliers first,
each part in index order — a stable partition, no sort) and its size.
"""
mutable struct ThresholdScoringWorkspace
    order::Vector{Int}
    k::Int
end
scoring_workspace(::AbstractThresholdScoring{Float64}, n::Int, _d_g::Int) =
    ThresholdScoringWorkspace(Vector{Int}(undef, n), 0)
Base.resize!(W::ThresholdScoringWorkspace, n::Integer) = (resize!(W.order, n); W)

inlier_count(::AbstractThresholdScoring{Float64}, W::ThresholdScoringWorkspace) = W.k
consensus_order(::AbstractThresholdScoring{Float64}, W::ThresholdScoringWorkspace) = W.order
decision_boundary(sc::AbstractThresholdScoring{Float64}, _buf) = sc.threshold
has_decision_boundary(::Type{<:AbstractThresholdScoring{Float64}}) = true
# Each quality is in [0, 1]: RANSAC's an indicator, MSAC's `1 − min(d², τ²)/τ²`.
score_bound(::_ThresholdScoring{<:Any, Float64}) = 1.0

# Whether `i` is one of the minimal sample's points (a handful of comparisons).
@inline _in_sample(::Nothing, _i) = false
@inline function _in_sample(sample, i)
    for s in sample
        s == i && return true
    end
    return false
end

# Whether a minimal-sample point's residual `d` is held (its square finite:
# `sample_point_held`), and whether point `i` is in the consensus set: a held
# sample point, or a finite square below τ².
@inline _held_sample_point(d) = sample_point_held(Float64(d * d))
@inline _in_consensus(sample, d, i, τ2) =
    (_in_sample(sample, i) && _held_sample_point(d)) || (isfinite(d) && Float64(d * d) < τ2)

# One point's quality: `d` squared in its own precision, a non-finite square an
# outlier (quality 0) — a non-finite `d`, and a finite one whose square
# overflows, alike.
@inline function _threshold_quality(sc, d, τ::Float64)
    d2 = Float64(d * d)
    return isfinite(d2) ? residual_quality(sc, d2, τ) : 0.0
end

# The conservative margin of the early exit: a sum of n terms each ≤ b, summed
# left to right in Float64, can exceed its exact value by at most n·eps·(|s| + n·b);
# the exit fires only when even that cannot reach the bound.
@inline _exit_margin(n, s, b) = n * eps(Float64) * (abs(s) + n * b) + eps(Float64)

# Whether a partial sum `s` with `rest` points still to come, each adding at
# most `b`, provably cannot reach `bound`. Without a bound (`nothing`), never.
@inline _cannot_reach(::Nothing, _s, _rest, _bound, _m) = false
@inline _cannot_reach(b::Float64, s, rest, bound, m) = s + rest * b < bound - _exit_margin(m, s, b)

function score!(W::ThresholdScoringWorkspace, sc::AbstractThresholdScoring{Float64}, scores::AbstractVector,
                residuals::AbstractVector, n::Int; sample = nothing, bound::Float64 = -Inf)
    τ = decision_boundary(sc, nothing)
    τ2 = τ * τ
    q0 = residual_quality(sc, 0.0, τ)
    ns = sample === nothing ? 0 : length(sample)
    b = score_bound(sc)
    # Σ quality over every point in index order (a non-finite square an outlier,
    # quality 0), then each sample point corrected to zero residual.
    s = 0.0
    k = 0; lo = 0
    order = W.order
    @inbounds for i in 1:n
        q = _threshold_quality(sc, residuals[i], τ)
        s += q
        scores[i] = q
        if isfinite(bound) && _cannot_reach(b, s, n - i + ns, bound, n + ns)
            W.k = 0
            return -Inf
        end
    end
    if sample !== nothing
        for i in sample   # bounds-checked: an invalid sample index throws (a handful of reads)
            d = residuals[i]
            _held_sample_point(d) || continue      # a non-finite square stays an outlier (`sample_point_held`)
            s += q0 - _threshold_quality(sc, d, τ)
            scores[i] = q0
        end
    end
    # The consensus set: d² < τ² (strict, the grids' test), the held sample points
    # in it; a stable partition, inliers first.
    @inbounds for i in 1:n
        _in_consensus(sample, residuals[i], i, τ2) || continue
        k += 1
        order[k] = i
    end
    lo = k
    @inbounds for i in 1:n
        _in_consensus(sample, residuals[i], i, τ2) && continue
        lo += 1
        order[lo] = i
    end
    W.k = k
    return s
end
