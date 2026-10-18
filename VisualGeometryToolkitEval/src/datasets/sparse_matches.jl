# =============================================================================
# src/datasets/sparse_matches.jl — the per-pair sparse-match HDF5 files of the
# match extractor (one `<pair id>.h5` per pair): RoMa v2 (`model.sample()`
# pixel coordinates with a per-point precision) and SuperPoint + LightGlue
# (pixel coordinates and a match score). The file's `matcher` attribute picks
# the layout; a file without one is RoMa v2's (it predates the attribute).
#
#   RoMa v2   kpts_A, kpts_B (N, 2) Float32 pixels; precision_A/_B (N, 2, 2); confidence (N,)
#   SP+LG     kpts_A, kpts_B (N, 2) Float32 pixels; scores (N,); kpt_scores_A/_B
#
# Python writes row-major, so HDF5.jl hands each (N, 2) back as (2, N): one
# match per column. The readers return pixel correspondences and the
# per-match confidence (higher = better); RoMa's precision is not carried.
# =============================================================================

# The `confidence` dataset of each layout, by the file's `matcher` attribute.
const _MATCH_CONFIDENCE = Dict("roma_v2" => "confidence", "superpoint+lightglue" => "scores")

"""
    load_sparse_matches(path) -> (; csponds, confidence)

The correspondences (pixels, `Float32` as written) and the per-match
confidence (higher = better) of one sparse-match HDF5 file of the match
extractor, RoMa v2 or SuperPoint + LightGlue by its `matcher` attribute.
"""
function load_sparse_matches(path::AbstractString)
    h5open(path, "r") do f
        tag = haskey(HDF5.attrs(f), "matcher") ? String(HDF5.attrs(f)["matcher"]) : "roma_v2"
        key = get(_MATCH_CONFIDENCE, tag) do
            error("$path: unknown matcher $(repr(tag)); known: $(join(sort!(collect(keys(_MATCH_CONFIDENCE))), ", "))")
        end
        a, b = read(f["kpts_A"])::Matrix{Float32}, read(f["kpts_B"])::Matrix{Float32}   # (2, N)
        confidence = read(f[key])::Vector{Float32}
        src = [SVector{2,Float32}(a[1, i], a[2, i]) for i in axes(a, 2)]
        tgt = [SVector{2,Float32}(b[1, i], b[2, i]) for i in axes(b, 2)]
        return (; csponds = StructArray(src .=> tgt), confidence)
    end
end

# The extractor's file name of a pair: `<scene>__<source stem>__<target stem>`,
# with a `/` in the scene (7-Scenes `chess/seq-03`) flattened to `__`.
_pair_file_id(p) = "$(replace(p.scene, '/' => "__"))__$(first(splitext(p.source_name)))__$(first(splitext(p.target_name)))"
