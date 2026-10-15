# =============================================================================
# The scoring contract — what a RANSAC loop and a local optimizer may ask of a
# scorer. A scorer implements the REQUIRED verbs; the optional ones have
# defaults here; the DERIVED ones are stated once, here, and scorers do not
# override them.
# =============================================================================
#
#   REQUIRED
#     scoring_workspace(scoring, n, d_g) -> W        built ONCE per composed workspace
#     score!(W, scoring, scores, residuals, n; sample = nothing, bound = -Inf) -> Float64
#     inlier_count(scoring, W) -> Int                 k* of the last score!
#     consensus_order(scoring, W) -> AbstractVector{Int}   its first k* entries ARE the consensus set
#   OPTIONAL (defaults here)
#     score!(W, scoring, scores, residuals, problem, θ::Uncertain, n; sample)   predictive scorers only
#     is_predictive(scoring)                   [false]
#     scored_model(scoring, problem, θ, sample) [θ]   predictive: Uncertain(θ, Σ̃θ) or nothing
#     scale_prior(scoring)                     [JeffreysPrior()]
#     outlier_halfwidth(scoring)               [nothing]   the uniform-outlier model, if any
#     inlier_threshold(scoring, buf)           [none]      the decision point of the last score!
#     has_inlier_threshold(S)                  [false]     static trait: whether the method exists
#     score_bound(scoring)                     [nothing]   a per-point upper bound of a bounded-sum score
#     check_workspace(scoring, W)              [nothing]   once at entry
#   DERIVED (one statement each)
#     consensus_s2, consensus_set, inlier_probabilities!, score_model!
#
# Scores: higher is better, comparable within one scoring only. `scores`
# receives the per-point score the scorer sorts or sums (the squared residual
# for a sort-and-sweep scorer).
# =============================================================================

"""
    scoring_workspace(scoring, n::Int, d_g::Int) -> W

The scorer's own scratch for up to `n` points of codimension `d_g`, built ONCE
per composed workspace and reused by every [`score!`](@ref). `scoring_workspace(buf)`
reads it back from a [`ScoreBuffers`](@ref).
"""
function scoring_workspace end

"""
    score!(W, scoring, scores, residuals, n; sample = nothing, bound = -Inf) -> Float64
    score!(W, scoring, scores, residuals, problem, θ::Uncertain, n; sample = nothing, bound = -Inf) -> Float64

Score a model from its residuals `residuals[1:n]` (model-certain), or — a
predictive scorer only — from the problem and the model with its covariance
(filling `residuals` with the whitened residuals of the same constraint
evaluation). Fills `scores[1:n]` with the per-point score and caches in `W`
what [`inlier_count`](@ref) and [`consensus_order`](@ref) read. `sample`: the
minimal sample's indices (or `nothing`), whose points are exact by
construction. Higher is better.

`bound`: the score a caller must beat (strictly) for the result to matter. A
bounded-sum scorer ([`score_bound`](@ref)) MAY stop early once the candidate
provably cannot exceed it; it then returns a value `≤ bound` and its cached
state is undefined until the next full `score!`. The default `-Inf` never
stops early.
"""
function score! end

"""
    consensus_order(scoring, W) -> AbstractVector{Int}

The order of the points after the most recent [`score!`](@ref): its first
[`inlier_count`](@ref)`(scoring, W)` entries are the consensus set. A view of
the scorer's buffer, valid until the next `score!`.
"""
function consensus_order end

"""
    is_predictive(scoring) -> Bool

Whether the scorer scores a model WITH its covariance (`θ::Uncertain`). Default
`false`.
"""
is_predictive(::AbstractScoring) = false

"""
    scored_model(scoring, problem, θ, sample) -> θ′ | nothing

The form in which a model is scored: the plain model (default), or for a
predictive scorer `Uncertain(θ, Σ̃θ)` with the covariance of the model fit on
the points `sample`; `nothing` when there is none (a degenerate sample, which is
skipped like a failed solve).
"""
scored_model(::AbstractScoring, _problem, θ, _sample) = θ

"""
    scale_prior(scoring) -> IGPrior

The scale prior the scorer's consensus scale is fit under. Default
[`JeffreysPrior`](@ref)`()`.
"""
scale_prior(::AbstractScoring) = JeffreysPrior()

"""
    outlier_halfwidth(scoring) -> Float64 | nothing

The half-width of the scorer's uniform outlier model, if it has one (the
Student-t + uniform posterior weights read it); `nothing` otherwise (default).
"""
outlier_halfwidth(::AbstractScoring) = nothing

"""
    inlier_threshold(scoring, buf::ScoreBuffers) -> Float64

The scorer's decision point for the most recent [`score!`](@ref) on `buf`: the
residual below which a point is an inlier. A threshold scorer returns its
threshold; a scale-free scorer derives one from its consensus set. A local
optimizer with no width of its own follows it ([`loss_at`](@ref)). Only scorers
with [`has_inlier_threshold`](@ref) define it.
"""
function inlier_threshold end

"""
    has_inlier_threshold(::Type{S}) -> Bool

Whether scorer type `S` defines [`inlier_threshold`](@ref) — a static trait,
`false` by default, declared `true` beside the method.
"""
has_inlier_threshold(::Type{<:AbstractScoring}) = false
has_inlier_threshold(scoring::AbstractScoring) = has_inlier_threshold(typeof(scoring))

"""
    score_bound(scoring) -> Float64 | nothing

An upper bound of one point's contribution to a BOUNDED-SUM score (each point
adds a quality in `[−∞, b]` to the total), or `nothing` (default) for a scorer
whose score is not such a sum. A bounded-sum scorer may stop a [`score!`](@ref)
early against its `bound` keyword.
"""
score_bound(::AbstractScoring) = nothing

"""
    check_workspace(scoring, W) -> nothing

Refuse, at entry, a workspace built for a different scorer (`ArgumentError`).
The default refuses a workspace the scorer cannot read — no
[`inlier_count`](@ref)`(scoring, W)` method: another scorer's
`scoring_workspace` built it — and names the scorers it was built for. A scorer
whose workspace also depends on its configuration (a prior's tables) adds a
method that checks that as well. Called once per entry, never in the inner loop.
"""
function check_workspace(scoring::AbstractScoring, W)
    applicable(inlier_count, scoring, W) && return nothing
    owners = _workspace_owners(W)
    throw(ArgumentError(
        "this workspace ($(nameof(typeof(W)))) was built for " *
        "$(isempty(owners) ? "no scorer" : join(owners, " / ")); build it with " *
        "`scoring_workspace(scoring, n, d_g)` for $(nameof(typeof(scoring)))"))
end

# The scorer types whose `inlier_count` reads a workspace like `W` (the error
# path of `check_workspace` only).
function _workspace_owners(W)
    owners = String[]
    for m in methods(inlier_count)
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) == 3 || continue
        S, WT = sig.parameters[2], sig.parameters[3]
        (WT isa Type && typeof(W) <: WT) || continue
        for s in Base.uniontypes(S)
            s = Base.unwrap_unionall(s)
            s isa DataType && s <: AbstractScoring && push!(owners, string(nameof(s)))
        end
    end
    return unique!(owners)
end

"""
    with_width(x, w::WidthAxis) -> typeof(x) with the width `w`

The width setter shared by scorings and local optimizations: a scoring rebuilt
at a native width (`with_width(scoring, OutlierHalfwidth(a))`), a local
optimizer given an explicit width mode (`with_width(lo, ScaleMultiple(m))`,
`with_width(lo, LossWidth(w))`). The owner of the type implements it.
"""
function with_width end
# The refusal: a scorer whose owner implements no width of that axis.
with_width(x::AbstractScoring, w::WidthAxis) = throw(ArgumentError("$(nameof(typeof(x))) takes no width $(nameof(typeof(w)))"))

# =============================================================================
# The buffers one composed workspace scores into
# =============================================================================

"""
    ScoreBuffers{T, W}(n, scoring_ws)

The scoring buffers of one composed workspace, built once and handed to a local
optimizer by reference: the residuals (`residuals(buf)`, storage precision
`T`), the per-point scores (`scores(buf)`, Float64 — a derived statistic:
squaring a Float32 residual into Float64 is exact), the consensus mask
(`consensus_mask(buf)`), the scorer's workspace (`scoring_workspace(buf)`) and
the evaluation `epoch` that stamps every [`ConsensusSet`](@ref) built from them
(`check_fresh(cs, buf)`).
"""
mutable struct ScoreBuffers{T <: AbstractFloat, W}
    residuals::Vector{T}
    scores::Vector{Float64}
    mask::BitVector
    scoring_ws::W
    epoch::Int
end
ScoreBuffers{T}(n::Int, scoring_ws::W) where {T, W} =
    ScoreBuffers{T, W}(zeros(T, n), zeros(Float64, n), falses(n), scoring_ws, 0)

"The residuals of a [`ScoreBuffers`](@ref)."
residuals(buf::ScoreBuffers) = buf.residuals
"The per-point scores of a [`ScoreBuffers`](@ref)."
scores(buf::ScoreBuffers) = buf.scores
"The consensus mask of a [`ScoreBuffers`](@ref): the set of the last scored model."
consensus_mask(buf::ScoreBuffers) = buf.mask
scoring_workspace(buf::ScoreBuffers) = buf.scoring_ws
check_fresh(cs::ConsensusSet, buf::ScoreBuffers) = check_fresh(cs, buf.epoch)
is_fresh(cs::ConsensusSet, buf::ScoreBuffers) = is_fresh(cs, buf.epoch)

# Trim every buffer to `n` (a shared workspace sized for a larger problem).
function Base.resize!(buf::ScoreBuffers, n::Integer)
    resize!(buf.residuals, n); resize!(buf.scores, n); resize!(buf.mask, n)
    return buf
end

# =============================================================================
# DERIVED — one statement each
# =============================================================================

"""
    consensus_s2(scoring, residuals, order, k; model_dof, d_g) -> Float64

The Student-t scale ŝ² of the consensus set `order[1:k]` under the scorer's
[`scale_prior`](@ref): [`partition_student_t`](@ref) of its RSS
([`partition_rss`](@ref), from the plain `residuals`). `model_dof` is the
caller's — `m + 1` for a local optimizer's consensus set (the scale's extra
dof), `m` for a result's σ̂.
"""
consensus_s2(scoring, residuals::AbstractVector, order::AbstractVector{Int}, k::Int;
             model_dof::Integer, d_g::Integer) =
    last(
    partition_student_t(
        scale_prior(scoring), partition_rss(order, residuals, k), k;
        model_dof, codimension = d_g
    )
)

"""
    consensus_set(scoring, W, residuals; model_dof, d_g, epoch, score = NaN) -> ConsensusSet

The consensus set of the most recent [`score!`](@ref): its size
[`inlier_count`](@ref), its scale [`consensus_s2`](@ref) over
[`consensus_order`](@ref), stamped with `epoch`, carrying that evaluation's
`score` when the caller has it.
"""
function consensus_set(scoring, W, residuals::AbstractVector; model_dof::Integer, d_g::Integer, epoch::Integer,
                       score::Real = NaN)
    k = inlier_count(scoring, W)
    s2 = consensus_s2(scoring, residuals, consensus_order(scoring, W), k; model_dof, d_g)
    return ConsensusSet(; count = k, s2, epoch, score)
end

"""
    inlier_probabilities!(w, scoring, scores, order, k; model_dof, d_g) -> w

Per-point inlier probabilities of the consensus set `order[1:k]`: under a
scorer with an outlier model ([`outlier_halfwidth`](@ref) a number) the
Student-t + uniform posterior of the set's Student-t
([`posterior_inlier_probs!`](@ref) on the squared per-point `scores`, under the
scorer's prior); otherwise 0/1 membership of `order[1:k]`.
"""
inlier_probabilities!(w::AbstractVector, scoring, scores::AbstractVector, order::AbstractVector{Int}, k::Int;
                      model_dof::Integer, d_g::Integer) =
    _inlier_probabilities!(outlier_halfwidth(scoring), w, scoring, scores, order, k, model_dof, d_g)

_inlier_probabilities!(a::Float64, w, scoring, scores, order, k, model_dof, d_g) =
    posterior_inlier_probs!(
    w, scores, order, k; model_dof = Int(model_dof), log_two_outlier_halfwidth = log(2 * a),
    codimension = Int(d_g), prior = scale_prior(scoring)
)
function _inlier_probabilities!(::Nothing, w, _scoring, _scores, order, k, _model_dof, _d_g)
    fill!(w, zero(eltype(w)))
    @inbounds for j in 1:k
        w[order[j]] = one(eltype(w))
    end
    return w
end

"""
    consensus_mask!(buf, scoring) -> buf

Set `consensus_mask(buf)` to the consensus set of the most recent
[`score!`](@ref): the first [`inlier_count`](@ref) entries of
[`consensus_order`](@ref).
"""
function consensus_mask!(buf::ScoreBuffers, scoring)
    W = buf.scoring_ws
    order = consensus_order(scoring, W)
    k = inlier_count(scoring, W)
    # Word-wise clear (fill! on a BitVector writes 64 bits per store) instead of
    # a per-bit loop — this runs once per accepted candidate model.
    fill!(buf.mask, false)
    @inbounds for j in 1:k
        buf.mask[order[j]] = true
    end
    return buf
end

"""
    score_model!(buf, scoring, problem, θ; sample = nothing, reuse_residuals = false,
                 mask = true, bound = -Inf) -> (S, k)

THE one scoring path of a model on a composed workspace: bump the epoch (every
[`ConsensusSet`](@ref) issued before is stale), compute `θ`'s residuals into
`residuals(buf)` (skipped under `reuse_residuals` — the caller left them there —
for a model-certain scorer), [`score!`](@ref), and set the consensus mask
(`mask = false` defers it: [`consensus_mask!`](@ref) sets it when needed).
`θ` is the SCORED form ([`scored_model`](@ref)): an `Uncertain` under a
predictive scorer, which then fills the residuals itself. Returns the score and
[`inlier_count`](@ref).
"""
function score_model!(buf::ScoreBuffers, scoring, problem, θ;
                      sample = nothing, reuse_residuals::Bool = false,
                      mask::Bool = true, bound::Float64 = -Inf)
    buf.epoch += 1
    n = data_size(problem)
    S = _score_model!(buf, scoring, problem, θ, n, sample, reuse_residuals, bound)
    k = inlier_count(scoring, buf.scoring_ws)
    mask && consensus_mask!(buf, scoring)
    return (S, k)
end

function _score_model!(buf, scoring, problem, θ, n, sample, reuse_residuals, bound)
    (reuse_residuals && !is_predictive(scoring)) || residuals!(buf.residuals, problem, θ)
    return score!(buf.scoring_ws, scoring, buf.scores, buf.residuals, n; sample, bound)
end
_score_model!(buf, scoring, problem, θ::Uncertain, n, sample, _reuse_residuals, bound) =
    score!(buf.scoring_ws, scoring, buf.scores, buf.residuals, problem, θ, n; sample, bound)
