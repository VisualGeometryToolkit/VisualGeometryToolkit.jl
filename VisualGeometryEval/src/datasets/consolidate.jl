# =============================================================================
# src/datasets/consolidate.jl — THE producer of the consolidated csponds files
# `load_roma_pairs` reads: one `<scene>.jld2` per scene of a stereo dataset,
# each pair's sparse matches with their GT inlier mask from depth.
#
# The record is plain data in pixels (no stack type is serialized, so a file
# outlives any change of a camera or rig type):
#
#   pairs :: Vector of (; scene, source_name, target_name,
#                         csponds    :: StructArray of SVector{2,Float32} pixel pairs,
#                         confidence :: Vector{Float32}   (higher = better),
#                         inlier_mask :: BitVector,
#                         K1, K2 :: SMatrix{3,3,Float64}, size1, size2 :: Size2{Int},
#                         R_gt :: SMatrix{3,3,Float64}, t_gt :: SVector{3,Float64})
#   format = CONSOLIDATED_FORMAT, matcher, gt_method, metric_tol_mm, rel_tol, bidirectional
#
# Every scene needs GT depth for both views of every pair: the inlier mask is
# what makes a pool's ground truth. A scene whose depth is unavailable is
# filtered out of the dataset's pairs beforehand (a missing depth file throws).
# =============================================================================

"The version of the consolidated csponds record [`consolidate_csponds`](@ref) writes."
const CONSOLIDATED_FORMAT = 2

# The pixel calibration of a camera (as stored by its model) and its extent.
_pixel_K(cam) = SMatrix{3,3,Float64,9}(intrinsics(model(cam)).K)
_pixel_size(cam) = (e = extent(cam); Size2(Int(e.width), Int(e.height)))

function _consolidated_record(p, matches, inlier_mask::BitVector)
    E = extrinsics(p.rig)                                   # source camera → target camera
    return (; scene = p.scene, source_name = p.source_name, target_name = p.target_name,
            csponds = matches.csponds, confidence = matches.confidence, inlier_mask,
            K1 = _pixel_K(source(p.rig)), K2 = _pixel_K(target(p.rig)),
            size1 = _pixel_size(source(p.rig)), size2 = _pixel_size(target(p.rig)),
            R_gt = SMatrix{3,3,Float64,9}(E.R), t_gt = SVector{3,Float64}(E.t))
end

"""
    consolidate_csponds(ds::AbstractStereoDataset, matcher::Symbol; match_dir, out_dir,
                        scenes = nothing) -> Vector{String}

Consolidate the sparse matches of every pair of `ds` (one
[`load_sparse_matches`](@ref) file `<match_dir>/<scene>__<src>__<tgt>.h5` per
pair; a pair without one is skipped) into `<out_dir>/<scene>.jld2`, one file
per scene (a `/` in a scene id becomes `__`), each pair's GT inlier mask from
[`load_validation_data`](@ref) under [`gt_config`](@ref)`(ds, matcher)`.
`scenes`, when given, restricts the run to those scene ids. Returns the paths
written. The record is documented in src/datasets/consolidate.jl.
"""
function consolidate_csponds(ds::AbstractStereoDataset, matcher::Symbol;
                             match_dir::AbstractString, out_dir::AbstractString, scenes = nothing)
    cfg = gt_config(ds, matcher)
    by_scene = Dict{String,Vector{StereoPair}}()
    for p in stereo_pairs(ds)
        scenes === nothing || p.scene in scenes || continue
        push!(get!(by_scene, p.scene, StereoPair[]), p)
    end
    mkpath(out_dir)
    written = String[]
    for scene in sort!(collect(keys(by_scene)))
        pairs = by_scene[scene]
        records = Vector{Any}(nothing, length(pairs))
        Threads.@threads for j in eachindex(pairs)
            p = pairs[j]
            h5 = joinpath(match_dir, "$(_pair_file_id(p)).h5")
            isfile(h5) || continue
            matches = load_sparse_matches(h5)
            v = load_validation_data(ds, p, matches.csponds)
            mask = validate_gt_inliers(v.rig, v.csponds, v.depth_src, v.depth_tgt;
                                       rel_tol = cfg.rel_tol, metric_tol = cfg.metric_tol,
                                       bidirectional_depth = cfg.bidirectional)
            records[j] = _consolidated_record(p, matches, BitVector(mask))
        end
        kept = [r for r in records if r !== nothing]
        out = joinpath(out_dir, "$(replace(scene, '/' => "__")).jld2")
        JLD2.jldsave(out; pairs = kept, format = CONSOLIDATED_FORMAT, matcher,
                     gt_method = cfg.gt_method, metric_tol_mm = Float64(ustrip(mm, cfg.metric_tol)),
                     rel_tol = cfg.rel_tol, bidirectional = cfg.bidirectional)
        @info "consolidated csponds" scene pairs = length(kept) missing = length(pairs) - length(kept) out
        push!(written, out)
    end
    return written
end
