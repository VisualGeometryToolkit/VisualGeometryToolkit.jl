# =============================================================================
# src/datasets/heb.jl — the raw ETH HEB reader (`<scene>_homographies.h5`).
#
# Per image pair: pixel csponds, SNN ratios, GT inlier flags, K1/K2, the
# COLMAP relative pose; the SNN hard filter `snn < snn_threshold`
# (`_snn_filtered_matches`) and the pair drop below `min_csponds`.
# =============================================================================

"""
    HebPair <: StereoRawPair

One raw HEB pair, in pixels: csponds, SNN ratios and the dataset's GT inlier
flags (post-SNN filter), the pixel calibrations, image sizes and the extrinsic
GT relative pose `x₂ = R·x₁ + t`.
"""
struct HebPair <: StereoRawPair
    src_id::String; dst_id::String; sub::String
    csponds::StructArray
    snn::Vector{Float32}
    gt_inliers::BitVector
    K1::SMatrix{3,3,Float64,9}
    K2::SMatrix{3,3,Float64,9}
    size1::Size2{Float64}
    size2::Size2{Float64}
    R_gt::SMatrix{3,3,Float64,9}
    t_gt::SVector{3,Float64}
end

function _parse_pair_key(key::AbstractString)
    parts = split(key, "_")
    length(parts) == 8 || return nothing
    (src=join(parts[2:4], "_"), dst=join(parts[5:7], "_"), sub=parts[8])
end

function _read_heb_pair(f, key::AbstractString, ids; snn_threshold::Real)
    c = read(f[key])                               # (10, N) float64: x₁ y₁ x₂ y₂ … snn gt
    csponds, snn, kept = _snn_filtered_matches(c, Float32.(c[9, :]), snn_threshold)
    gt_inliers_bits = BitVector(c[10, kept] .== 1.0)

    # HEB stores the relative pose row-major (numpy/h5py) as the 3×4 [R | t]
    # of the extrinsic convention x₂ = R·x₁ + t. HDF5.jl hands a row-major
    # buffer back transposed — 4×3 = [Rᵀ; tᵀ] — so the rotation block needs
    # the same transpose K gets below. Until 2026-09-21 it was read without
    # one and t formed as −Rᵀ·t: the pools stored the inverse motion
    # (Rᵀ, −Rᵀt) and both H pose metrics decomposed Hᵀ to compensate.
    # `vec` of that transposed buffer IS the row-major 3×4 sequence.
    R_gt, t_gt = _row_major_pose(vec(read(f["pose_$(ids.src)_$(ids.dst)_$(ids.sub)"])))
    K1 = SMatrix{3,3,Float64,9}(read(f["K_$(ids.src)"])')
    K2 = SMatrix{3,3,Float64,9}(read(f["K_$(ids.dst)"])')

    # H5 stores image size as a 2-vector (width, height) per image.
    raw_size1 = read(f["size_$(ids.src)"])
    raw_size2 = read(f["size_$(ids.dst)"])
    size1 = Size2(width=Float64(raw_size1[1]), height=Float64(raw_size1[2]))
    size2 = Size2(width=Float64(raw_size2[1]), height=Float64(raw_size2[2]))

    HebPair(ids.src, ids.dst, ids.sub, csponds, snn, gt_inliers_bits,
             K1, K2, size1, size2, R_gt, t_gt)
end

"""
    load_heb_pairs(h5_path; min_csponds, max_pairs_per_scene, subsample_overdraw,
                   subsample_seed, snn_threshold = [pool.heb] snn_threshold) -> Vector{HebPair}

Iterate all `corr_*` groups of the raw HEB scene file `h5_path` that have a
pose and both calibrations, in a seeded subsample ([`_seeded_key_subsample`](@ref),
keyed by the file name) of at most `max_pairs_per_scene` kept pairs; apply
the SNN threshold and drop pairs with fewer than `min_csponds` correspondences.
The subsample sizes and seed are the caller's (its pool configuration).
"""
function load_heb_pairs(h5_path::AbstractString;
                        min_csponds::Integer,
                        max_pairs_per_scene::Integer,
                        subsample_overdraw::Integer,
                        subsample_seed::Integer,
                        snn_threshold::Real = _config_default("pool.heb", "snn_threshold"))
    # First sweep: enumerate candidate keys (cheap — metadata only) then
    # subsample indices BEFORE reading heavy per-pair data.
    h5open(h5_path, "r") do f
        corr_keys = sort!(filter(k -> startswith(k, "corr_"), keys(f)))
        # Keep only keys with a matching pose/K so we never waste a slot on an
        # invalid pair.  Cheap hash lookup in HDF5.
        valid_keys = String[]
        for key in corr_keys
            ids = _parse_pair_key(key); ids === nothing && continue
            haskey(f, "pose_$(ids.src)_$(ids.dst)_$(ids.sub)") || continue
            haskey(f, "K_$(ids.src)") && haskey(f, "K_$(ids.dst)") || continue
            push!(valid_keys, key)
        end

        # Seeded subsample of the valid-key list (the overdraw absorbs the
        # min_csponds rejections that only show after the read).
        valid_keys = _seeded_key_subsample(valid_keys, basename(h5_path); max_pairs_per_scene,
                                           subsample_overdraw, subsample_seed)

        pairs = HebPair[]
        for key in valid_keys
            ids = _parse_pair_key(key)
            p = _read_heb_pair(f, key, ids; snn_threshold)
            length(p.csponds) >= min_csponds || continue  # need enough for meaningful sampling
            push!(pairs, p)
            length(pairs) >= max_pairs_per_scene && break
        end
        return pairs
    end
end
