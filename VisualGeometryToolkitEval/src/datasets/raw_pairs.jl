# =============================================================================
# src/datasets/raw_pairs.jl — what every raw reader shares: The SNN filter,
# The seeded per-scene pair subsample and the row-major [R | t] parser.
#
# A raw pair (a concrete VisualGeometryToolkitCore `StereoRawPair`) carries the
# matched correspondences, calibration, image sizes and GT of one image pair in
# PIXELS, before any pool is drawn from it. Raw pairs are never persisted.
# =============================================================================

"""
    _snn_filtered_matches(m, snn, snn_threshold) -> (csponds, snn_kept, kept)

The SNN filter of every raw reader: keep the matches with `snn <
snn_threshold` (SNN = Lowe's d_NN/d_2NN, low = good; a match AT the threshold
is dropped). `m` holds one match per column with the pixel coordinates in rows
1–4 (`x₁ y₁ x₂ y₂`); returns the kept pixel csponds, their SNN, and the kept
column indices (for any per-match column the reader carries along).
"""
function _snn_filtered_matches(m::AbstractMatrix, snn::AbstractVector{Float32}, snn_threshold::Real)
    kept = findall(<(Float32(snn_threshold)), snn)
    src = [SVector{2,Float64}(m[1, i], m[2, i]) for i in kept]
    dst = [SVector{2,Float64}(m[3, i], m[4, i]) for i in kept]
    return StructArray(src .=> dst), snn[kept], kept
end

"""
    _seeded_key_subsample(keys, tag; max_pairs_per_scene, subsample_overdraw, subsample_seed) -> keys′

The per-scene pair subsample of the raw readers: when the scene has more pair
keys than `max_pairs_per_scene`, a seeded `subsample_overdraw ×` overdraw
(`shuffle` under `POOL_RNG(stable_seed(subsample_seed, tag))`, order preserved)
that the reader trims to the cap after its own rejections; otherwise every
key. Stable across Julia versions: the seed is a fixed SHA-256 derivation and
the stream StableRNGs'.
"""
function _seeded_key_subsample(keys::AbstractVector, tag::AbstractString;
                               max_pairs_per_scene::Integer, subsample_overdraw::Integer,
                               subsample_seed::Integer)
    n = length(keys)
    max_pairs_per_scene < n || return keys
    rng = POOL_RNG(stable_seed(subsample_seed, tag))
    order = shuffle(rng, 1:n)[1:min(n, subsample_overdraw * max_pairs_per_scene)]
    return keys[sort!(order)]
end

"""
    _row_major_pose(v) -> (R::SMatrix{3,3,Float64,9}, t::SVector{3,Float64})

The rotation block and the last column of a 3×4 matrix `[R | t]` stored ROW-MAJOR
as 12 numbers (`R₁₁ R₁₂ R₁₃ t₁ R₂₁ … t₃`) — the layout of the HEB `pose_*`
datasets (and of KITTI's `calib.txt`/`poses/*.txt` and ScanNet-1500's
`rel_pose`). The one parser of it.
"""
function _row_major_pose(v::AbstractVector)
    length(v) == 12 || throw(ArgumentError("a row-major 3×4 needs 12 values, got $(length(v))"))
    M = SMatrix{4,3,Float64,12}(v)'          # column-major 4×3 of the row-major stream = [R | t]ᵀ
    return SMatrix{3,3,Float64,9}(M[:, 1:3]), SVector{3,Float64}(M[:, 4])
end
