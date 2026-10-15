# =============================================================================
# Selection over a benchmark pool — the scoring contract applied to a pool
# record's stored models (`Pools.PoolPair` with its residual matrix).
#
#   best_pool_model(scoring, pair, R)        THE per-pair argmax, generic over `score!`
#   selection_staging(scoring) -> key        scorings sharing a key share one staged batch
#   stage_selection(key, pairs, Rs; backend) the staged batch of a group of pairs
#   select_on_grid!(idxs, scoring, grid, staged; problem_type)
#                                            the selections at every width of a grid
#
# A grid family (a scoring that is swept over a width grid on a device) owns
# its staging and its kernels (RansacScoringDoneRight); the generic method here
# is the CPU fallback every scoring with a width has: per grid column
# `with_width` then `best_pool_model`.
# =============================================================================

"""
    best_pool_model(scoring, pair::PoolPair, R; ws, scores) -> Int

The index of the best-scoring stored model of `pair` under `scoring` (a scorer
at its width), scored on the pair's residual matrix `R` (`num_points(pair) ×
num_models(pair)`, the pool frame's units): per model, [`score!`](@ref) of its
residual column with its minimal sample (the model's sample points enter at
zero residual, as in every scorer). The FIRST strictly best model wins; a
non-finite score never wins; `0` (never a valid index) means no model scores
above −Inf. A matrix of another shape is a `DimensionMismatch`. `ws` and
`scores` are the scorer's workspace and per-point scratch, allocated for `pair`
when omitted.
"""
function best_pool_model(scoring::AbstractScoring, pair::Pools.PoolPair, R::AbstractMatrix;
                         ws = scoring_workspace(scoring, Pools.num_points(pair), codimension(scoring)),
                         scores::AbstractVector{Float64} = Vector{Float64}(undef, Pools.num_points(pair)))
    n, M = Pools.num_points(pair), Pools.num_models(pair)
    size(R) == (n, M) || throw(DimensionMismatch(
        "residual matrix of size $(size(R)) for a pool pair of $n points and $M models " *
            "(expected num_points × num_models)"))
    best_j = 0; best = -Inf
    for j in 1:M
        s = score!(ws, scoring, scores, view(R, 1:n, j), n; sample = view(pair.sample_indices, :, j))
        s = isfinite(s) ? Float64(s) : -Inf
        s > best && (best = s; best_j = j)
    end
    return best_j
end

"""
    selection_staging(scoring) -> key

The staging a scoring's grid selection reads ([`stage_selection`](@ref)):
scorings with the same key share ONE staged batch of a group of pairs. Default
`:pool` (the pairs and their residual matrices as they are); a grid family
with device kernels declares its own.
"""
selection_staging(_scoring) = :pool

"""
    stage_selection(key, pairs, residuals; backend) -> staged

The staged batch of `pairs` (light pool records) with their residual matrices
for the scorings of staging `key` ([`selection_staging`](@ref)), on the compute
`backend`. The `:pool` staging is `(; pairs, residuals)`.
"""
stage_selection(key::Symbol, pairs, residuals; backend = nothing) = stage_selection(Val(key), pairs, residuals; backend)
stage_selection(::Val{:pool}, pairs, residuals; backend = nothing) = (; pairs, residuals)

"""
    select_on_grid!(idxs::AbstractMatrix{Int32}, P, scoring, grid, staged; kw...) -> idxs

The selections of `scoring` at every width of `grid` (its native axis,
[`width_axis`](@ref)) over a staged batch of pairs: `idxs[p, t]` the stored
model selected for pair `p` at `grid[t]` (`0`: none). `P` is the pairs'
problem type (a family that carries neither its minimal sample size nor its
codimension reads them from it). Further keywords are a family's own
(a randomized family's per-(pair, column) generator `rng_of`) and are ignored by
the others. The generic method is the CPU
fallback on the `:pool` staging — per column `with_width(scoring, axis(grid[t]))`
then [`best_pool_model`](@ref); a grid family implements its own.
"""
function select_on_grid!(idxs::AbstractMatrix{Int32}, ::Type, scoring, grid::AbstractVector, staged; _kw...)
    pairs, Rs = staged.pairs, staged.residuals
    axis = width_axis(scoring)
    for t in eachindex(grid)
        sc = with_width(scoring, _width_of(axis, Float64(grid[t])))
        Threads.@threads for p in eachindex(pairs)
            idxs[p, t] = best_pool_model(sc, pairs[p], Rs[p])
        end
    end
    return idxs
end
_width_of(::LossWidth, w::Float64) = LossWidth(w)
_width_of(::OutlierHalfwidth, a::Float64) = OutlierHalfwidth(a)
_width_of(::ScaleMultiple, m::Float64) = ScaleMultiple(m)
