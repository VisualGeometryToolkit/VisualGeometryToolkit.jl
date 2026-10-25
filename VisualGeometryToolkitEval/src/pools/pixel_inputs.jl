# =============================================================================
# pools/pixel_inputs.jl — the raw-PIXEL input slice of a pool for the external
# estimators (SupeRANSAC, PoseLib's Python peers) that read plain arrays.
# =============================================================================

# N×4 `[x₁ y₁ x₂ y₂]` rows of a pair's pixel csponds — the external estimators' input layout.
_match_matrix(cs) = [reduce(hcat, collect.(sources(cs)))' reduce(hcat, collect.(targets(cs)))']

"""
    export_pixel_inputs(path, pool, idx; scene = pool.scene, created = string(Dates.now())) -> path

Write the raw-pixel inputs of the pairs `idx` (strictly increasing pool
indices) of `pool` to the JLD2 file `path`, in the layout the external
estimators' workers read: per pair `matches` (an N×4 `[x₁ y₁ x₂ y₂]` matrix of
pixel correspondences, back through the record's own maps,
`VisualGeometryToolkitCore.Pools.pixel_csponds`), the stored Lowe ratios `snn`, the
pixel calibrations `K1` and `K2` and the ground-truth `R_gt` and `t_gt`; plus
`scene`, `n_pairs`, `pair_indices`, `pool_size`, the `pool_key` the slice was
read from, a `provenance` string and `created`. The parent directory is
created and an existing file is overwritten.
"""
function export_pixel_inputs(path::AbstractString, pool::ScoringPool, idx::AbstractVector{<:Integer};
                             scene::AbstractString = pool.scene, created::AbstractString = string(Dates.now()))
    isempty(idx) && throw(ArgumentError("export_pixel_inputs: no pair indices for $scene"))
    issorted(idx; lt = <=) || throw(ArgumentError("export_pixel_inputs: pair indices of $scene must be strictly increasing"))
    indices = collect(Int, idx)
    ps = pool.pairs[indices]
    mkpath(dirname(abspath(path)))
    JLD2.jldsave(path; scene = String(scene), n_pairs = length(ps), pair_indices = indices,
                 pool_size = length(pool), pool_key = pool.pool_key,
                 matches = [Matrix{Float64}(_match_matrix(pixel_csponds(p))) for p in ps],
                 snn = [Float64.(p.snn) for p in ps],
                 K1 = [Matrix{Float64}(p.K1) for p in ps], K2 = [Matrix{Float64}(p.K2) for p in ps],
                 R_gt = [Matrix{Float64}(p.R_gt) for p in ps], t_gt = [Vector{Float64}(p.t_gt) for p in ps],
                 provenance = "load_pool($(scene)) pool_key $(pool.pool_key); pixel_csponds + snn + K/R/t " *
                              "as stored; pool pairs $(indices); julia $(VERSION)",
                 created = String(created))
    return path
end
