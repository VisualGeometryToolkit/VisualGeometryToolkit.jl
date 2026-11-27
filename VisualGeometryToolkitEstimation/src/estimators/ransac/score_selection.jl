# =============================================================================
# ScoreSelection — the best-scoring model of a stored hypothesis pool
# =============================================================================

"""
    ScoreSelection(scoring::AbstractScoring)

A [`ClosedFormKind`](@extref VisualGeometryToolkitCore.ClosedFormKind) (a selection among fixed candidates): the
stored model of a hypothesis pool that scores best under `scoring` (a scorer at its width),
`solve(EstimationProblem(PoolSelection, pool), ScoreSelection(scoring))`. The pool is

- `pair => R`: a pool record (VisualGeometryToolkitCore's `Pools.PoolPair`) and its residual matrix
  (`num_points(pair) × num_models(pair)`, the pool frame's units), each model scored with its minimal sample entering
  at zero residual, as in every scorer; or
- `R`, a residual matrix alone (points × models), each column scored as it is.

Each model is scored with VisualGeometryToolkitCore's `score!` on its residual column, bounded by the running best (an
exact early stop: the selection is the full scores' strict argmax). The FIRST strictly best model wins and a
non-finite score never wins (Core's `best_model_index` rule). `u` is the selected model's index, `n` the points,
`stats.nsteps` the models scored and `metadata = (; score)` its score; `(Success, ClosedForm)`, or `u = 0` with
`(Failure, NoRule)` when no model scores above −Inf.

Streaming over a sampling budget (the sample-budget engine's checkpoints): `p = (; budget = k)` scores only the first
`k` models. A cache, `c = init(EstimationProblem(PoolSelection, pool, (; budget = k)), ScoreSelection(scoring))`,
keeps the running best, so `solve!(reinit!(c; p = (; budget = k′)))` with `k′ ≥ k` CONTINUES, scoring models
`k + 1 … k′` only. A smaller budget, or new data (`reinit!(c; f)`), starts over. The workspace and the per-point scratch
are the cache's, sized for the points of the data `init` saw: new data with no more points reuse them (a pool of many
pairs is best initialized on its largest), more points grow them.

# The batched grid form

A staged batch of pools selected at every width of a grid (an evaluation's selection sweep) is the ensemble of the
(pool, width) selections, one launch:

    r = solve(EnsembleProblem(SelectionGrid(problem_type, grid, staged), npools * length(grid)), ScoreSelection(scoring),
              EnsembleKernel(backend))

`scoring` is then the grid's selection scoring at any width (its native axis, `width_axis`): a scorer, or a grid family
of another package (a scoring with VisualGeometryToolkitCore's `select_on_grid!`, which a grid family implements); see
[`SelectionGrid`](@ref). The FIRST strictly best model wins in every cell, as in the per-pool form.
"""
struct ScoreSelection{S} <: AbstractEstimationAlgorithm
    scoring::S
end
estimator_kind(::ScoreSelection) = ClosedFormKind()

# The pool's residual matrix and the minimal sample of its model `j` (`nothing`: a bare matrix has none).
_pool_residuals(pool::AbstractMatrix) = pool
_pool_residuals(pool::Pair{<:PoolPair, <:AbstractMatrix}) = last(pool)
_pool_sample(::AbstractMatrix, _j) = nothing
_pool_sample(pool::Pair{<:PoolPair, <:AbstractMatrix}, j) = view(first(pool).sample_indices, :, j)
_check_pool(::AbstractMatrix) = nothing
_check_pool(pool::Pair{<:PoolPair, <:AbstractMatrix}) = check_residual_matrix(first(pool), last(pool))
_pool_budget(::NullParameters, M::Int) = M
_pool_budget(p::NamedTuple{(:budget,)}, M::Int) = min(Int(p.budget), M)

mutable struct ScoreSelectionCache{F, S, W}
    pool::F
    scoring::S
    ws::W   # sized for `length(scores)` points
    scores::Vector{Float64}
    budget::Int
    folded::Int
    best_idx::Int
    best_score::Float64
end

function __init(prob::EstimationProblem{PoolSelection, <:Union{AbstractMatrix, Pair{<:PoolPair, <:AbstractMatrix}}},
                alg::ScoreSelection{<:AbstractScoring})
    _check_pool(prob.f)
    R = _pool_residuals(prob.f)
    n = size(R, 1)
    ws = scoring_workspace(alg.scoring, n, codimension(alg.scoring))
    return ScoreSelectionCache(prob.f, alg.scoring, ws, Vector{Float64}(undef, n), _pool_budget(prob.p, size(R, 2)),
                               0, 0, -Inf)
end

function reinit!(c::ScoreSelectionCache{F, S}; f = c.pool, p = (; budget = c.budget), alg = nothing) where {F, S}
    f isa F || throw(ArgumentError("reinit!: the cache holds a $F pool; init a new cache for a $(typeof(f))"))
    R = _pool_residuals(f)
    alg === nothing || (c.scoring = (alg::ScoreSelection{S}).scoring)
    if size(R, 1) > length(c.scores)                                  # more points than the scratch holds: grow it
        c.ws = scoring_workspace(c.scoring, size(R, 1), codimension(c.scoring))
        c.scores = Vector{Float64}(undef, size(R, 1))
    end
    budget = _pool_budget(p, size(R, 2))
    if f !== c.pool || alg !== nothing || budget < c.folded
        f === c.pool || _check_pool(f)
        c.pool = f
        c.folded = 0; c.best_idx = 0; c.best_score = -Inf          # new data, scorer or a smaller budget: start over
    end
    c.budget = budget
    return c
end

function solve!(c::ScoreSelectionCache)
    R = _pool_residuals(c.pool)
    n = size(R, 1)
    @inbounds for j in (c.folded + 1):c.budget
        sc = score!(c.ws, c.scoring, c.scores, view(R, 1:n, j), n; sample = _pool_sample(c.pool, j), bound = c.best_score)
        (isfinite(sc) && sc > c.best_score) && (c.best_score = Float64(sc); c.best_idx = j)
    end
    c.folded = max(c.folded, c.budget)
    found = c.best_idx != 0
    return FitResult(c.best_idx, found ? ReturnCode.Success : ReturnCode.Failure, found ? ClosedForm : NoRule,
                     SolveStats(; nsteps = c.folded), NaN, n, (; score = c.best_score))
end

# =============================================================================
# The batched grid form: every (pool, width) selection of a staged batch
# =============================================================================

"""
    SelectionGrid(problem_type, grid, staged; rng_of = nothing)

The batch of an [`EnsembleProblem`](@extref VisualGeometryToolkitCore.EnsembleProblem) of pool selections:
the pools of a staged batch (VisualGeometryToolkitCore's `stage_selection(selection_staging(scoring), pairs,
residuals; backend)`, its pairs of type `problem_type`, whose minimal sample size and codimension a grid family that
carries neither reads) selected at every width of `grid` (the scoring's native axis, VisualGeometryToolkitCore's
`width_axis`). `rng_of(fingerprint, t)` is a randomized family's generator per (pool content, column), so a grid is
independent of the thread schedule (SIMFIT++'s halves); `nothing` for a deterministic one.

`solve(EnsembleProblem(SelectionGrid(P, grid, staged), npools * length(grid)), ScoreSelection(scoring),
EnsembleKernel(backend))` returns a VisualGeometryToolkitCore `EnsembleResult` over the cells in column-major order —
pool `p` at width `grid[t]` is item `p + npools·(t − 1)`, so `reshape(r.u, npools, length(grid))` is the selection
matrix: `u` the selected model's index (`Int32`, `0` none), `retcode`/`stop_rule` `(Success, ClosedForm)` or `(Failure,
NoRule)` with no model, `nsteps` 0 and `rms` NaN. The selections are VisualGeometryToolkitCore's `select_on_grid!` of
the scoring — the exact kernels of a residual-quality loss-width family, a per-column best-pool selection otherwise,
or a grid family's own — on the staging's device: the `EnsembleKernel`'s backend is the one the batch was staged on.
"""
struct SelectionGrid{P, G <: AbstractVector, S, R}
    problem_type::Type{P}
    grid::G
    staged::S
    rng_of::R
end
SelectionGrid(problem_type::Type, grid::AbstractVector, staged; rng_of = nothing) =
    SelectionGrid(problem_type, grid, staged, rng_of)

function __solve(ens::EnsembleProblem{<:SelectionGrid}, alg::ScoreSelection, e::EnsembleKernel)
    g = ens.f
    T = length(g.grid)
    npools, rest = T == 0 ? (0, ens.n) : divrem(ens.n, T)
    rest == 0 || throw(DimensionMismatch(string("SelectionGrid: ", ens.n, " items are no whole number of pools at ", T,
        " widths (npools · length(grid))")))
    hasproperty(g.staged, :backend) && g.staged.backend != e.backend && throw(ArgumentError(string(
        "SelectionGrid: the batch was staged on ", g.staged.backend, ", the ensemble launches on ", e.backend,
        ": give EnsembleKernel the staging's backend")))
    idxs = Matrix{Int32}(undef, npools, T)
    _select_on_grid!(idxs, g.problem_type, alg.scoring, g.grid, g.staged, g.rng_of)
    u = vec(idxs)
    found = u .!= 0
    return EnsembleResult(u, [f ? ReturnCode.Success : ReturnCode.Failure for f in found],
                          [f ? ClosedForm : NoRule for f in found], zeros(Int, length(u)), fill(NaN, length(u)))
end
_select_on_grid!(idxs, P, scoring, grid, staged, ::Nothing) = select_on_grid!(idxs, P, scoring, grid, staged)
_select_on_grid!(idxs, P, scoring, grid, staged, rng_of) = select_on_grid!(idxs, P, scoring, grid, staged; rng_of)
