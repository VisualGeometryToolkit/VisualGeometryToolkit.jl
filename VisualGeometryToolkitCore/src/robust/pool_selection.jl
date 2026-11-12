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
# its staging (RANSACScoringDoneRight). On the `:pool` staging, a loss-width
# family with a per-residual quality (`has_residual_quality`) selects through
# the exact kernels (pool_grid_kernels.jl) on the staging's backend
# (`default_device()` unless the caller supplies one), the same kernels a grid
# family runs; any other scoring with a width selects per grid column by
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
    Pools.check_residual_matrix(pair, R)
    n = Pools.num_points(pair)
    return Pools.best_model_index(Pools.num_models(pair)) do j
        score!(ws, scoring, scores, view(R, 1:n, j), n; sample = view(pair.sample_indices, :, j))
    end
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
`backend` (`nothing`: the staging's default). The `:pool` staging is
`(; pairs, residuals, backend)`, its backend [`default_device`](@ref)`()` unless one
is supplied: the device its exact kernels run on.
"""
stage_selection(key::Symbol, pairs, residuals; backend = nothing) = stage_selection(Val(key), pairs, residuals; backend)
stage_selection(::Val{:pool}, pairs, residuals; backend = nothing) =
    (; pairs, residuals, backend = something(backend, default_device()))

"""
    sorted_squared_residuals(pair::PoolPair, R) -> Matrix{Float32}

Per stored model of `pair` (a column), its squared residuals from the residual
matrix `R` (`num_points(pair) × num_models(pair)`) with the model's minimal
sample at zero (as every scorer enters it), sorted ascending: what the exact
selection kernels read, and RANSACScoringDoneRight's staging. In the residuals'
precision (Float32).
"""
function sorted_squared_residuals(pair::Pools.PoolPair, R::AbstractMatrix)
    Pools.check_residual_matrix(pair, R)
    n = Pools.num_points(pair); M = Pools.num_models(pair)
    sorted_sq = Matrix{Float32}(undef, n, M)
    for j in 1:M
        col = view(sorted_sq, 1:n, j)
        square_distances!(col, view(R, 1:n, j), n)   # in the residuals' own precision
        zero_sample!(col, view(pair.sample_indices, :, j))
        sort!(col)
    end
    return sorted_sq
end

"""
    select_on_grid!(idxs::AbstractMatrix{Int32}, P, scoring, grid, staged; kw...) -> idxs

The selections of `scoring` at every width of `grid` (its native axis,
[`width_axis`](@ref)) over a staged batch of pairs: `idxs[p, t]` the stored
model selected for pair `p` at `grid[t]` (`0`: none). `P` is the pairs'
problem type (a family that carries neither its minimal sample size nor its
codimension reads them from it). Further keywords are a family's own
(a randomized family's per-(pair, column) generator `rng_of`) and are ignored by
the others. The generic method reads the `:pool` staging: a loss-width family
with a per-residual quality ([`has_residual_quality`](@ref)) is scored by
[`exact_pm_score_kernel!`](@ref) and selected by
[`argmax_models_kernel!`](@ref) on the staging's backend (per pair, every width in
one launch, on [`sorted_squared_residuals`](@ref)); any other scoring per column by
`with_width(scoring, axis(grid[t]))` then [`best_pool_model`](@ref). Either way
the FIRST strictly best model wins. A grid family implements its own.
"""
function select_on_grid!(idxs::AbstractMatrix{Int32}, ::Type, scoring, grid::AbstractVector, staged; _kw...)
    pairs, Rs = staged.pairs, staged.residuals
    axis = width_axis(scoring)
    has_residual_quality(scoring) && axis isa LossWidth &&
        return _select_exact!(idxs, staged.backend, scoring, Float64.(grid), pairs, Rs)
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
_width_of(::EfficiencyConstant, m::Float64) = EfficiencyConstant(m)

# The exact selection of a residual-quality family, pair by pair on `backend`: the score kernel
# over the pair's models at every width, then the argmax kernel over its widths (one work item
# per workgroup, so the CPU backend spreads the items over its threads). The kernels' inputs go to
# `backend` through `on_device` (the host arrays themselves on the CPU) and the selections come
# back to the host. A pair without a model selects 0.
function _select_exact!(idxs::AbstractMatrix{Int32}, backend::KA.Backend, scoring, loss_widths::Vector{Float64}, pairs, Rs)
    T = length(loss_widths)
    support = support_sq_loss_width(scoring, maximum(loss_widths))
    score_models! = exact_pm_score_kernel!(backend, (1, 1))
    pick_models! = argmax_models_kernel!(backend, (1, 1))
    widths = on_device(Float64, backend, loss_widths)
    offset = on_device(Int64, backend, Int64[0])
    errors = device_zeros(Float64, backend, 1, T); best = device_zeros(Int32, backend, 1, T)
    for p in eachindex(pairs)
        n, M = Pools.num_points(pairs[p]), Pools.num_models(pairs[p])
        if M == 0
            idxs[p, :] .= 0
            continue
        end
        flat = on_device(Float32, backend, vec(sorted_squared_residuals(pairs[p], Rs[p])))
        scores = KA.allocate(backend, Float64, 1, M, T); fill!(scores, -Inf)
        n_per = on_device(Int32, backend, Int32[n]); M_per = on_device(Int32, backend, Int32[M])
        score_models!(scores, flat, offset, n_per, M_per, widths, Int32(T), Int32(0), scoring, support;
                      ndrange = (1, M))
        KA.synchronize(backend)
        pick_models!(errors, best, scores, M_per, device_zeros(Float64, backend, M), offset, Int32(0); ndrange = (1, T))
        KA.synchronize(backend)
        idxs[p, :] .= vec(on_device(KA.CPU(), best))
    end
    return idxs
end
