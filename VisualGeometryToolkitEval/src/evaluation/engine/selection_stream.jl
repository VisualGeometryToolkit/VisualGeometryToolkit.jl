# =============================================================================
# evaluation/engine/selection_stream.jl — the scene stream of every
# selection+LO experiment: stage → the experiment's selection grids
# (VisualGeometryToolkitCore's `select_on_grid!`, a grid family's own kernels) → one
# apply_lo_refits call over `refits_of(idxs_of, scene_index)` (PairCtx built per
# (pair, spec) for ALL refit specs).
#
# Selection reads a pool's residual matrices; the refits read the light
# records. The loaders are VisualGeometryToolkitEval's `scene_loaders(dir, scenes;
# residuals = true)`.
# =============================================================================

# The base seed of a randomized grid family's per-(pair, column) generators
# (SIMFIT++'s train/validation halves): configs/default.toml `[selection]
# simfitpp_seed`, drawn through `stable_seed`, so a grid is independent of the
# thread schedule and of Julia's version.
const _SIMFITPP_SEED = Int(_config_default("selection", "simfitpp_seed"))
_selection_rng(fingerprint, t) = StableRNG(stable_seed(_SIMFITPP_SEED, "simfitpp", fingerprint..., t))

"""
    stream_selections(refits_of, T, problem_type, loaders, grids, scoring, backend, batch;
                      solo_scoring = scoring)
        -> (M::Dict{label, Vector{Matrix{T}}}, s2_by_scene, np_per)

The scene stream of every selection+LO experiment. `loaders[i]()` returns
scene `i`'s `(pool, residuals)` (VisualGeometryToolkitEval `scene_loaders(dir,
scenes; residuals = true)`). Per scene: stage each `batch` of pairs once per
staging kind (VisualGeometryToolkitCore `stage_selection` on `backend`, shared by the
grids whose scorings declare the same `selection_staging`), evaluate every
selection grid of `grids` (`key => (widths, scoring)`: the scoring's native
widths and the scoring itself — VisualGeometryToolkitCore `select_on_grid!`) on it,
then one [`apply_lo_refits`](@ref) call over `refits_of(idxs_of, scene_index)`
(`idxs_of[key]` = that grid's per-(pair, column) selections) on the light
records. `M[label]` holds each refit's per-scene matrix in element type `T`;
`s2_by_scene` the records' σ̂², `np_per` the pairs per scene. `scoring` and
`solo_scoring` are the refits' scoring templates.
"""
function stream_selections(refits_of::Function, ::Type{T}, problem_type::Type, loaders,
                           grids::AbstractVector{<:Pair}, scoring, backend,
                           batch::Int; solo_scoring = scoring) where {T}
    nscene = length(loaders)
    M = Dict{String,Vector{Matrix{T}}}()
    s2_by_scene = Vector{Vector{Float64}}(undef, nscene)
    np_per = Vector{Int}(undef, nscene)
    # Prefetch pipeline: while scene `si` is processed, a worker task loads
    # scene `si+1`. Only the timing of each load moves; what is staged, swept and
    # refit, and in which order, is unchanged. Depth 1: at most two scenes are
    # resident and never two loads run at once.
    prefetch(i) = Threads.@spawn (loaders[i])()
    nxt = nscene >= 1 ? prefetch(1) : nothing
    for si in 1:nscene
        pool, res = fetch(nxt)
        nxt = si < nscene ? prefetch(si + 1) : nothing
        pp = pool.pairs
        P = length(pp)
        np_per[si] = P
        s2_by_scene[si] = Float64[p.gt_variance for p in pp]
        idxs_of = Dict(k => Matrix{Int32}(undef, P, length(widths)) for (k, (widths, _)) in grids)
        bs = 1
        while bs <= P
            be = min(bs + batch - 1, P)
            sub = pp[bs:be]
            Rsub = [residual_matrix(res, i) for i in bs:be]
            staged = Dict{Symbol,Any}()
            for (k, (widths, sc)) in grids
                key = selection_staging(sc)
                st = get!(() -> stage_selection(key, sub, Rsub; backend), staged, key)
                idx = Matrix{Int32}(undef, length(sub), length(widths))
                select_on_grid!(idx, problem_type, sc, widths, st; rng_of = _selection_rng)
                @views idxs_of[k][bs:be, :] .= idx
            end
            staged = nothing
            bs = be + 1
        end
        res = nothing
        specs = refits_of(idxs_of, si)
        out = apply_lo_refits(problem_type, specs, pp; scoring, solo_scoring)
        for spec in specs
            haskey(M, spec.label) || (M[spec.label] = Vector{Matrix{T}}(undef, nscene))
            M[spec.label][si] = T.(out[spec.label])
        end
        pool = nothing; pp = nothing; GC.gc()
    end
    return M, s2_by_scene, np_per
end
