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

# The exact selection of a residual-quality family on `backend`: every pair's sorted squared residuals staged end to
# end (`sorted_squared_residuals`, one column per model), then ONE launch of the score kernel over (pair, model) at
# every width and ONE of the argmax kernel over (pair, width), at one item per workgroup through `OnBackend` (every
# thread on the CPU, one warp per workgroup on a device). The staged arrays go to `backend` through `on_device` (the
# host arrays themselves on the CPU) and the selections come back to the host. A pair without a model selects 0.
function _select_exact!(idxs::AbstractMatrix{Int32}, backend::KA.Backend, scoring, loss_widths::Vector{Float64}, pairs, Rs)
    T = length(loss_widths); P = length(pairs)
    ns = Int32[Pools.num_points(pair) for pair in pairs]; Ms = Int32[Pools.num_models(pair) for pair in pairs]
    M_max = P == 0 ? 0 : Int(maximum(Ms))
    M_max == 0 && (fill!(idxs, 0); return idxs)
    # pair p's model m reads its residuals from pair_offset[p] + (m − 1)·n_p; its (unread) error from model_offset[p] + m
    pair_offset = zeros(Int64, P); model_offset = zeros(Int64, P)
    for p in 2:P
        pair_offset[p] = pair_offset[p - 1] + Int64(ns[p - 1]) * Ms[p - 1]
        model_offset[p] = model_offset[p - 1] + Ms[p - 1]
    end
    flat = Vector{Float32}(undef, pair_offset[P] + Int64(ns[P]) * Ms[P])
    Threads.@threads for p in 1:P
        Ms[p] == 0 && continue
        copyto!(flat, pair_offset[p] + 1, sorted_squared_residuals(pairs[p], Rs[p]), 1, Int(ns[p]) * Ms[p])
    end
    n_d = on_device(Int32, backend, ns); M_d = on_device(Int32, backend, Ms)
    scores = KA.allocate(backend, Float64, P, M_max, T); fill!(scores, -Inf)
    OnBackend(backend; workgroupsize = 1)(exact_pm_score_kernel!)(
        scores, on_device(Float32, backend, flat), on_device(Int64, backend, pair_offset), n_d, M_d,
        on_device(Float64, backend, loss_widths), Int32(T), Int32(0), scoring, support_sq_loss_width(scoring, maximum(loss_widths));
        ndrange = (P, M_max))
    errors = KA.allocate(backend, Float64, P, T); best = KA.allocate(backend, Int32, P, T)
    OnBackend(backend; workgroupsize = 1)(argmax_models_kernel!)(
        errors, best, scores, M_d, device_zeros(Float64, backend, model_offset[P] + Ms[P] + 1),
        on_device(Int64, backend, model_offset), Int32(0); ndrange = (P, T))
    KA.synchronize(backend)
    copyto!(idxs, on_device(KA.CPU(), best))
    for p in 1:P
        Ms[p] == 0 && (idxs[p, :] .= 0)
    end
    return idxs
end
