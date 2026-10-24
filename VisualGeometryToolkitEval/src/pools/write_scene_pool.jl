# =============================================================================
# src/pools/write_scene_pool.jl — the generic scene-pool orchestration: the
# threaded, seeded build of one scene's pool from its raw pairs and the one
# write of its two files. It knows no problem type: the caller's `build` draws
# and assembles a pair (VisualGeometryToolkitEstimation's pool-pair constructor, wrapped
# in a few lines of project glue).
# =============================================================================

"""
    write_scene_pool(path, raws, build; problem, residual_kind, seed,
                     skip_existing = true, provenance = Dict{String,Any}(),
                     scene = <basename of path>) -> path | nothing

Build one scene's pool from its raw pairs `raws` (VisualGeometryToolkitCore
`StereoRawPair`s, in pixels) and write it: the light file `path`
([`save_pool`](@ref)) and its residuals ([`save_pool_residuals`](@ref) at
[`residuals_path`](@ref)`(path)`).

`build(rng, raw)` builds one pair and returns either `nothing` — the pair is
refused (e.g. fewer correspondences than the problem's minimal sample) and
dropped — or a NamedTuple `(; pair::PoolPair, residuals::Matrix{Float32},
samples::Integer, solutions::Integer)`: the record, its `(num_points,
num_models)` residual matrix, and the accepted minimal samples and real
solutions its draw counted. Pair `i` of `raws` draws from its OWN stream
`POOL_RNG(stable_seed(seed, i))` — keyed by its index in `raws`, not by the
thread — so the pool is a pure function of `(seed, raws)`, independent of
thread count, scheduling and Julia version. Pairs are built under
`Threads.@threads`.

`problem` (`:E`, `:F`, `:H`) and `residual_kind` (`:sampson` |
`:tangent_sampson`, what `build`'s matrices hold) are stamped into the files,
with the scene, the per-pair `pair_scene` labels (else `scene`), the sample and
solution totals, `pool_rng`, the Julia version and the caller's `provenance`.
With `skip_existing` (the default: a pool is hours of compute) a scene whose
two files both exist is skipped. Returns `path`, or `nothing` when no pair
survives.
"""
function write_scene_pool(path::AbstractString, raws::AbstractVector, build;
                          problem::Symbol, residual_kind::Symbol, seed::Integer,
                          skip_existing::Bool = true,
                          provenance::AbstractDict = Dict{String,Any}(),
                          scene::AbstractString = first(splitext(basename(path))))
    rpath = residuals_path(path)
    if skip_existing && isfile(path) && isfile(rpath)
        @info "  $scene: skip (already exists: $path)"
        return path
    end
    n = length(raws)
    built = Vector{Any}(nothing, n)
    t0 = time()
    Threads.@threads for i in 1:n
        built[i] = build(POOL_RNG(stable_seed(seed, i)), raws[i])
    end
    elapsed = time() - t0
    kept = findall(!isnothing, built)
    n_dropped = n - length(kept)
    n_dropped > 0 && @warn "build refused $n_dropped pair(s)" scene problem
    isempty(kept) && (@warn "No usable pairs in $scene"; return nothing)

    pairs = [built[i].pair for i in kept]
    total_samples = sum(Int(built[i].samples) for i in kept)
    total_solutions = sum(Int(built[i].solutions) for i in kept)
    mean_sol = total_samples == 0 ? 0.0 : total_solutions / total_samples
    pool = ScoringPool(pairs, problem, scene;
        scene_names = String[something(pair_scene(raws[i]), String(scene)) for i in kept],
        stamps = (gt_convention = GT_CONVENTION, pool_rng = string(nameof(POOL_RNG)),
                  julia_version = string(VERSION), total_samples, total_solutions,
                  mean_solutions_per_sample = mean_sol,
                  provenance = Dict{String,Any}(String(k) => v for (k, v) in provenance)))
    res = PoolResiduals(pool.pool_key, residual_kind, [built[i].residuals for i in kept])
    @info @sprintf("  %s: %d pairs, %.1fs  [avg %.0f models/pair, mean %.2f real/sample]",
                   scene, length(kept), elapsed, sum(num_models, pairs) / length(kept), mean_sol)
    # Residuals first: the light file, whose key binds them, completes the pair.
    save_pool_residuals(rpath, res)
    save_pool(path, pool)
    @info "  → saved $path"
    return path
end
