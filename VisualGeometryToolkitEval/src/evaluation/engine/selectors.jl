# =============================================================================
# evaluation/engine/selectors.jl — the sample-budget engine's running-best
# selectors.
#
# Each selector folds models into a running best as samples accumulate;
# `checkpoint` reads off its imc pose error; `lo_error!` refines the running
# best through the same refit layer (`refit_imc!`) as every other consumer. The
# oracle is just another "method" column, not a special-cased array.
#
# All pose errors are `model_imc` (degrees), dispatched on the pool model's type:
# for E/F the COHERENT max(Δr, Δt) over the single best decomposition (VgtkCore
# `imc_pose_error`); for H the decomposition rotation error.
# =============================================================================

"""
    SampleSelector

A running-best selector of the sample-budget engine.
It folds pool models into a running best ([`update!`](@ref)), reports that
best's pose error at a checkpoint ([`checkpoint`](@ref)) and refits it through
the shared refit layer ([`lo_error!`](@ref)).
"""
abstract type SampleSelector end

"""
    ScoreSelector(name, scoring, width, pool; lo = nothing, refinement_width = nothing)

Scoring-based selector: the running best of `scoring`'s score over the pool models, read at each checkpoint —
VisualGeometryToolkitEstimation's `ScoreSelection(scoring)` on `EstimationProblem(PoolSelection, pair => R, (; budget))`,
its cache continued over the models seen so far (`reinit!(cache; p = (; budget))`), so every model is scored once.
The selector holds the cache `init` returns; the fold and the checkpoint read allocate nothing once the front end's
`reinit!` does (a Core gap, owned by Phase 1A).
`pool` (`pair => R`) sizes the cache: the largest pair of the evaluation. `width` is the native width `scoring`
selects at (its outlier half-width, or its threshold). Its running best is refit by the config's LO (`lo = nothing`)
at the config's native (`nat_ref`, else `width`), or by its own LO `lo` at `refinement_width` (else the config's
`nat_ref`, else `width`) — the MSAC + truncated-refiner columns.
"""
mutable struct ScoreSelector{S, C, L} <: SampleSelector
    name::Symbol
    scoring::S
    width::Float64
    cache::C
    lo::L
    refinement_width::Union{Nothing,Float64}
    seen::Int        # the models of the current pair folded in by `update!` (the selection's budget)
end
ScoreSelector(name::Symbol, scoring, width::Real, pool::Pair{<:PoolPair, <:AbstractMatrix};
              lo = nothing, refinement_width::Union{Nothing,Real} = nothing) =
    ScoreSelector(name, scoring, Float64(width), init(EstimationProblem(PoolSelection, pool, (; budget = 0)), ScoreSelection(scoring)),
                  lo, refinement_width === nothing ? nothing : Float64(refinement_width), 0)

"""
    OracleSelector(name, outlier_halfwidth)

Oracle selector: the ground-truth-best SINGLE model by [`model_imc`](@ref).
Recomputes the imc per model (pruned by VisualGeometryToolkitCore's `pool_imc`, the
stored lower bound on that imc) and refines that one model under LO (at a
representative outlier half-width, the first method's, unless the config
names the arm's refinement native).
"""
mutable struct OracleSelector <: SampleSelector
    name::Symbol
    outlier_halfwidth::Float64
    best_idx::Int
    best_imc::Float64        # running min imc pose error [deg]
end
OracleSelector(name::Symbol, outlier_halfwidth::Real) =
    OracleSelector(name, Float64(outlier_halfwidth), 0, Inf)

# Methods of VisualGeometryToolkitEstimation's `reset!` (its samplers' reset): fresh running-best state.
reset!(s::ScoreSelector)  = (s.seen = 0; nothing)
reset!(s::OracleSelector) = (s.best_idx = 0; s.best_imc = Inf; nothing)

"""
    update!(s::SampleSelector, geom::PairGeom, pair, R, j, col_j, n)

Fold pool model `j` (its residual column `R[1:n, j]`, minimal sample `col_j`)
into the selector's running best.
"""
function update! end

# Model `j` of the pair joins the selection's budget; the first model of a pair points the cache at its pool. The
# models are scored when the selection is read (`_selected_model`, at a checkpoint), each once: ScoreSelection's bounded
# `score!` fold, the stack's rule (the minimal sample's points at zero residual, the first strictly best model wins).
@inline function update!(s::ScoreSelector, geom::PairGeom, pair, R::AbstractMatrix, j::Integer, col_j, n::Integer)
    j == 1 && reinit!(s.cache; f = pair => R, p = (; budget = 0))
    s.seen = j
    return nothing
end
# The selection over the models seen so far (its pool index, 0 for none): the cache continued to the budget `s.seen`.
@inline _selected_model(s::ScoreSelector) = solve!(reinit!(s.cache; p = (; budget = s.seen))).u

@inline function update!(s::OracleSelector, geom::PairGeom, pair, R::AbstractMatrix, j::Integer, col_j, n::Integer)
    # Prune: the stored lower bound on the imc (VgtkCore `pool_imc`) — a model whose
    # bound can't beat the running best needs no decomposition.
    lb = pool_imc(pair, j)
    (isfinite(lb) && lb < s.best_imc) || return nothing
    imc = model_imc(pair.models[j], geom)
    imc < s.best_imc && (s.best_imc = imc; s.best_idx = j)
    return nothing
end

"""
    checkpoint(s::SampleSelector, geom::PairGeom, pair) -> (imc::Float32, best_idx::Int32)

The running best's imc pose error [deg] and pool index, recorded at a
checkpoint; `(NaN32, 0)` before any model was folded in.
"""
function checkpoint end

@inline function checkpoint(s::ScoreSelector, geom::PairGeom, pair)
    j = _selected_model(s)
    return j == 0 ? (NaN32, Int32(0)) : (Float32(model_imc(pair.models[j], geom)), Int32(j))
end
@inline checkpoint(s::OracleSelector, geom::PairGeom, pair) = s.best_idx == 0 ?
    (NaN32, Int32(0)) :
    (Float32(s.best_imc), Int32(s.best_idx))

"""
    lo_error!(ctx::PairCtx, s::SampleSelector, cfg) -> (imc_error::Float32, lo_ns::Int64)

LO refit of the running best through the shared refit layer
([`refit_imc!`](@ref)). `lo_ns` is the bracketed optimizer wall-clock (0 when
there is no model to refine).
"""
function lo_error! end

lo_error!(ctx::PairCtx, s::ScoreSelector, cfg) = _lo_error!(ctx, s, s.lo, cfg)
lo_error!(ctx::PairCtx, s::OracleSelector, cfg) = _config_lo_error!(ctx, s.best_idx, s.outlier_halfwidth, cfg)

# A selector refit by the config's LO.
_lo_error!(ctx, s::ScoreSelector, ::Nothing, cfg) = _config_lo_error!(ctx, _selected_model(s), s.width, cfg)

function _config_lo_error!(ctx, best_idx::Int, width::Float64, cfg)
    best_idx == 0 && return (NaN32, Int64(0))
    seed = refit_seed(ctx, ctx.pair.models[best_idx])
    # PROTOCOL: outlier-half-width arms refit at their tuned native;
    # efficiency-constant arms supply nat_ref = nothing (consensus from selection,
    # width in the LO). `consumed_native` maps the incoming native (on
    # `nat_axis`) to the refiner's consumed axis — identity for a native
    # refiner, outlier half-width → loss width for an external one — so every
    # refiner sees the same operating point as the ladder and the tuning. The
    # refit scoring is the context's template AT nat′ (refit_imc!'s default),
    # never the selector's selection scoring.
    nat′ = consumed_native(cfg.lo_strategy, cfg.nat_axis, something(cfg.nat_ref, width);
                           d_g = codimension(ctx.problem_type))
    imc, lo_ns = refit_imc!(ctx, cfg.lo_strategy, seed, nat′)
    return (Float32(imc), lo_ns)
end

# A selector with its own refiner (MSAC + a truncated refiner) refits at its
# refinement width (else the config's `nat_ref`, else its selection width).
function _lo_error!(ctx, s::ScoreSelector, lo, cfg)
    j = _selected_model(s)
    j == 0 && return (NaN32, Int64(0))
    seed = refit_seed(ctx, ctx.pair.models[j])
    refit_width = something(s.refinement_width, cfg.nat_ref, s.width)
    imc, lo_ns = refit_imc!(ctx, lo, seed, refit_width)
    return (Float32(imc), lo_ns)
end
# A selection-only column (a LossWidthSpec without a refiner, `lo = nothing`):
# the selected model's own `model_imc`, as its batch checkpoint records it; no
# refit. Its selector carries `_SelectionOnly()`, since a selector's `nothing`
# means "the config's LO".
struct _SelectionOnly end
_lo_error!(ctx, s::ScoreSelector, ::_SelectionOnly, cfg) = (checkpoint(s, ctx.geom, ctx.pair)[1], Int64(0))
