# =============================================================================
# src/datasets/phototourism.jl — the raw IMC2020 PhotoTourism reader.
#
# 21-scene set: 12 RANSAC-Tutorial scenes (val/sacre_coeur, val/st_peters_square
# and 10 train scenes minus the 2 UBC-removed-for-inconsistency entries
# prague_old_town_square + westminster_abbey) + 9 IMC-PT 2020 TestData scenes
# (reichstag, florence_cathedral_side, british_museum, lincoln_memorial_statue,
# milan_cathedral, mount_rushmore, piazza_san_marco, sagrada_familia,
# st_pauls_cathedral). All cameras are PINHOLE (verified across 23,644
# entries).
#
# Per pair:
#   - Read pixel csps + SNN (matches.h5 + match_conf.h5).
#   - Filter csps with `match_conf < snn_threshold`
#     (PT inverts SNN — 0 good, 1 bad — so we keep LOW values).
#   - Drop the pair if kept csps < `min_csponds`.
#   - Read the GT relative pose from the scene's COLMAP recon
#     (`<recon_dir>/{cameras,images,points3D}.bin`):
#       - K from `cameras.bin` per-image cam_id (PINHOLE: fx, fy, cx, cy).
#       - R, t from `images.bin` quaternion + translation per image
#         (world→cam, composed to relative cam2←cam1 the same way the
#         tutorial-shipped R.h5/T.h5 did).
#   - SfM-track-paired SIFT detections: from `points3D.bin` track structure,
#     pick the (kpt_A, kpt_B) observations of every 3D point that both
#     images observe. These are the ORIGINAL SIFT detections in images.bin
#     (xy is never modified by COLMAP BA — verified directly in COLMAP source
#     `bundle_adjustment_ceres.cc`, where point2D.xy is baked into the
#     cost-functor as a constant constructor argument and never added as a
#     parameter block). They are the σ̂²_p noise-scale source of a pool,
#     decoupled from matches.h5 RootSIFT (a different detector pipeline).
#   - The `gt_inliers` mask (a pair-eligibility gate downstream): matches.h5
#     csps with pixel Sampson under the GT F below `gt_inlier_sampson_px`.
# =============================================================================

"""
    PtRawTrees(; matches_tutorial_val, matches_tutorial_train, matches_ef_test,
                 recon_tutorial, recon_ef_test, recon_special = Dict{String,String}())

The raw PhotoTourism trees, SUPPLIED BY THE CALLER. This package ships no data
directory to default to — the rule the HEB and RoMa readers follow (their
files are arguments), and which PT alone used to break: these five trees were
once absolute paths on one machine. Two consequences, both real: a registered package carried one
machine's paths, so PT pools could not be generated anywhere else; and a pool
producer cell could not DECLARE its raw inputs in `_provenance.toml`, because
no config key named them — every other input of the accv2026 campaign is
declared and hashed, and PT-E's was not.

`matches_*` hold `<scene>/{matches.h5,match_conf.h5}`; `recon_*` hold
`<scene>/dense/sparse` COLMAP models. `recon_special` maps a scene to a
COMPLETE recon directory (it already ends at the model), for the IMC-PT
TestData scenes whose model does not sit under `recon_ef_test`.
"""
Base.@kwdef struct PtRawTrees
    matches_tutorial_val::String
    matches_tutorial_train::String
    matches_ef_test::String
    recon_tutorial::String
    recon_ef_test::String
    recon_special::Dict{String,String} = Dict{String,String}()
end

"""
    PT_SCENES :: Vector{Tuple{String,Symbol}}

The PhotoTourism scenes with both matches (`matches.h5` + `match_conf.h5`)
and a COLMAP reconstruction on disk, each with its source tag: which tree of
[`PtRawTrees`](@ref) holds them. The scene SPLITS are configs/splits.toml's.
"""
const PT_SCENES = [
    # tutorial val (matches at ransac_tutorial/val/, recon at _track_verified_recons/)
    ("sacre_coeur",                :tutorial_val),
    ("st_peters_square",           :tutorial_val),
    # tutorial train (10 scenes — minus prague_old_town_square + westminster_abbey,
    # which UBC removed in 2022 for "data inconsistencies")
    ("brandenburg_gate",           :tutorial_train),
    ("buckingham_palace",          :tutorial_train),
    ("colosseum_exterior",         :tutorial_train),
    ("grand_place_brussels",       :tutorial_train),
    ("notre_dame_front_facade",    :tutorial_train),
    ("palace_of_westminster",      :tutorial_train),
    ("pantheon_exterior",          :tutorial_train),
    ("taj_mahal",                  :tutorial_train),
    ("temple_nara_japan",          :tutorial_train),
    ("trevi_fountain",             :tutorial_train),
    # IMC-PT 2020 TestData (9 scenes)
    ("reichstag",                  :ef_test),
    ("florence_cathedral_side",    :ef_test),
    ("british_museum",             :ef_test),
    ("lincoln_memorial_statue",    :ef_test),
    ("milan_cathedral",            :ef_test),
    ("mount_rushmore",             :ef_test),
    ("piazza_san_marco",           :ef_test),
    ("sagrada_familia",            :ef_test),
    ("st_pauls_cathedral",         :ef_test),
]

# The validation/test split of these scenes is configs/splits.toml
# [splits.phototourism] (the as-run alphabetical 10/11 partition), the one
# scene list of record, read through Tuning `registered_split`; nothing here
# restates it. (An earlier Shekhovtsov-aligned partition — RANSAC-Scoring
# load_data.py:51-67, minus the scenes UBC withdrew in 2022 — was retired
# 2026-07-09 in favour of the partition every production run used.)

function _scene_paths(scene::AbstractString, source::Symbol, t::PtRawTrees)
    if source === :tutorial_val
        (joinpath(t.matches_tutorial_val, scene),
         joinpath(t.recon_tutorial, scene, "dense", "sparse"))
    elseif source === :tutorial_train
        (joinpath(t.matches_tutorial_train, scene),
         joinpath(t.recon_tutorial, scene, "dense", "sparse"))
    elseif source === :ef_test
        recon = get(t.recon_special, scene,
                    joinpath(t.recon_ef_test, scene, "dense", "sparse"))
        (joinpath(t.matches_ef_test, scene), recon)
    else
        error("Unknown PT scene source: $source")
    end
end

"""
    pt_scene_files(scene, trees::PtRawTrees) -> Vector{String}

The raw files [`load_pt_pairs`](@ref) opens for PhotoTourism `scene` (one of
[`PT_SCENES`](@ref)), under the caller's `trees`: its `matches.h5` and
`match_conf.h5` and the COLMAP model's `cameras.bin`, `images.bin` and
`points3D.bin`. The one statement of the reader's path rule outside the
reader (the data-artifact catalog derives its file lists from it).
"""
function pt_scene_files(scene::AbstractString, trees::PtRawTrees)
    i = findfirst(p -> first(p) == scene, PT_SCENES)
    i === nothing && throw(ArgumentError("pt_scene_files: $(repr(scene)) is not one of PT_SCENES"))
    mdir, rdir = _scene_paths(scene, last(PT_SCENES[i]), trees)
    return [joinpath(mdir, "matches.h5"), joinpath(mdir, "match_conf.h5"),
            (joinpath(rdir, "$b.bin") for b in ("cameras", "images", "points3D"))...]
end

# =============================================================================
# COLMAP reconstruction: VisualGeometryIo's reader, scale unknown
# =============================================================================
# PhotoTourism's COLMAP models are SfM reconstructions with an ARBITRARY world
# scale; only the relative pose's direction is used, so the model is read
# under VisualGeometryIo's explicit `UnknownScale()` contract and `tvec` stays a
# unitless number.

# Pair-keyed index from points3D: (image_id_A, image_id_B) → vector of
# (kpt_idx_A, kpt_idx_B), only for image pairs that share at least one 3D
# point. Symmetric pairs are stored once with image_id_A < image_id_B; the
# query side flips kpts when reading back.
function _build_track_index(pts)
    idx = Dict{Tuple{Int32, Int32}, Vector{Tuple{Int32, Int32}}}()
    for p in pts
        T = length(p.track)
        T < 2 && continue
        for i in 1:T-1, j in i+1:T
            (a_id, a_k) = p.track[i]
            (b_id, b_k) = p.track[j]
            if a_id < b_id
                key = (a_id, b_id); val = (a_k, b_k)
            elseif a_id > b_id
                key = (b_id, a_id); val = (b_k, a_k)
            else
                continue
            end
            push!(get!(idx, key, Tuple{Int32, Int32}[]), val)
        end
    end
    idx
end

# The pixel K of a COLMAP camera row, through VisualGeometryIo's
# `camera_model`. All 21 IMC2020 scenes are PINHOLE; any other model is an
# explicit error rather than silently dropped distortion.
function _pinhole_K(cam)
    cam.model === :PINHOLE || error(
        "COLMAP camera model $(cam.model) not supported — only PINHOLE. All IMC2020 " *
        "scenes are pure PINHOLE; a distorted source needs undistortion first.")
    SMatrix{3,3,Float64,9}(intrinsics(camera_model(cam)).K)
end

# World→camera rotation of a COLMAP image row (wxyz quaternion; QuatRotation
# normalizes, so a slightly off-unit stored quaternion still gives SO(3)).
_colmap_R(im) = SMatrix{3,3,Float64,9}(QuatRotation(im.qvec[1], im.qvec[2], im.qvec[3], im.qvec[4]))

# Pair-eligibility mask: matches.h5 csps with pixel Sampson < threshold under
# the GT F (the oracle fundamental of the COLMAP pose). Used downstream only as
# an eligibility gate; NOT the σ̂²_p source.
function _sampson_px_inlier_mask(F_px, csponds; threshold_px::Float64)
    N = length(csponds)
    mask = BitVector(undef, N)
    @inbounds for i in 1:N
        d = sampson_distance(F_px, csponds.first[i], csponds.second[i])
        mask[i] = isfinite(d) && d < threshold_px
    end
    return mask
end

"""
    PtPair <: StereoRawPair

Raw PhotoTourism pair data (post-SNN filter), in pixels: csps from matches.h5,
COLMAP intrinsics, the composed relative pose, the GT-inlier mask (pixel
Sampson under the GT F), and the SfM-track-paired SIFT detections (from
points3D.bin) — the σ̂²_p noise-scale source.
"""
struct PtPair <: StereoRawPair
    pair_key::String
    csponds::StructArray                       # matches.h5 RootSIFT csps (RANSAC test set)
    snn::Vector{Float32}                       # SNN ratio (Lowe's d_NN/d_2NN; low = good)
    gt_inliers::BitVector                      # pixel Sampson under the GT F < gt_inlier_sampson_px
    K1::SMatrix{3,3,Float64,9}
    K2::SMatrix{3,3,Float64,9}
    size1::Size2{Float64}
    size2::Size2{Float64}
    R_gt::SMatrix{3,3,Float64,9}
    t_gt::SVector{3,Float64}
    track_kpts_A::Vector{SVector{2,Float64}}   # SIFT detections from images.bin, on tracks shared with B
    track_kpts_B::Vector{SVector{2,Float64}}   # paired with track_kpts_A (same point3D_id)
end
function _split_pair_key(key::AbstractString)
    idx = findlast('-', key)
    idx === nothing && error("Unrecognized pair key format: $key")
    (String(SubString(key, 1, idx - 1)), String(SubString(key, idx + 1, lastindex(key))))
end

"""
    _load_pt_pair(matches_f, conf_f, recon, pair_key; snn_threshold, min_csponds,
                 gt_inlier_sampson_px = [pool.phototourism] gt_inlier_sampson_px)
        -> PtPair | nothing

Load one pair from already-open matches/conf HDF5 handles plus the per-scene
`recon = (cameras, images, track_index)` (built once per scene). Apply the
SNN filter (`match_conf < snn_threshold`; 0=good, 1=bad), extract
SfM-track-paired SIFT detections, compose the GT relative pose from the
COLMAP per-image (q, t), and mark the GT inliers (pixel Sampson under the GT
F below `gt_inlier_sampson_px`).
"""
function _load_pt_pair(matches_f::HDF5.File, conf_f::HDF5.File,
                      recon::NamedTuple, pair_key::String;
                      snn_threshold::Real,
                      min_csponds::Integer,
                      gt_inlier_sampson_px::Real = _config_default("pool.phototourism", "gt_inlier_sampson_px"))
    img1, img2 = _split_pair_key(pair_key)
    haskey(recon.images, img1) || return nothing
    haskey(recon.images, img2) || return nothing
    eA = recon.images[img1]; eB = recon.images[img2]
    haskey(recon.cameras, eA.camera_id) || return nothing
    haskey(recon.cameras, eB.camera_id) || return nothing

    m       = read(matches_f[pair_key]::HDF5.Dataset)   # (4, N) Float64 (Python (N,4) → Julia transposed)
    snn_all = Float32.(read(conf_f[pair_key]::HDF5.Dataset))
    size(m, 2) == length(snn_all) || return nothing
    cs, snn_kept, _ = _snn_filtered_matches(m, snn_all, snn_threshold)
    length(cs) < min_csponds && return nothing

    # SfM-track-paired SIFT detections for σ̂²_p: walk the precomputed
    # track index for this image pair.  Track index is keyed by canonical
    # (low-id, high-id) ordering; flip kpts when reading back if our (A, B)
    # ordering is the reverse.
    pair_key_canon = eA.image_id < eB.image_id ?
        (eA.image_id, eB.image_id) : (eB.image_id, eA.image_id)
    flip_kpts = eA.image_id > eB.image_id
    raw_track = get(recon.track_index, pair_key_canon, Tuple{Int32, Int32}[])
    n_tk = length(raw_track)
    track_A = Vector{SVector{2,Float64}}(undef, n_tk)
    track_B = Vector{SVector{2,Float64}}(undef, n_tk)
    @inbounds for k in 1:n_tk
        kA, kB = flip_kpts ? (raw_track[k][2], raw_track[k][1]) : raw_track[k]
        # COLMAP kpt indices in points3D.bin track are 0-based.
        xA, yA, _ = eA.points2D[kA + 1]; xB, yB, _ = eB.points2D[kB + 1]
        track_A[k] = SVector(xA, yA)
        track_B[k] = SVector(xB, yB)
    end

    camA = recon.cameras[eA.camera_id]; camB = recon.cameras[eB.camera_id]
    K1 = _pinhole_K(camA)
    K2 = _pinhole_K(camB)
    size1 = Size2(width = Float64(camA.width), height = Float64(camA.height))
    size2 = Size2(width = Float64(camB.width), height = Float64(camB.height))

    R1 = _colmap_R(eA); t1 = SVector{3,Float64}(eA.tvec)    # unitless (UnknownScale)
    R2 = _colmap_R(eB); t2 = SVector{3,Float64}(eB.tvec)
    R_rel = R2 * R1'
    t_rel = t2 - R_rel * t1

    R_gt = SMatrix{3,3,Float64,9}(R_rel)
    F_px = FundMat(EssentialMat(R_gt, t_rel), K1, K2)
    mask = _sampson_px_inlier_mask(F_px, cs; threshold_px = Float64(gt_inlier_sampson_px))
    PtPair(pair_key, cs, snn_kept, mask, K1, K2, size1, size2, R_gt, t_rel, track_A, track_B)
end

"""
    load_pt_pairs(scene, source; trees::PtRawTrees, min_csponds, max_pairs_per_scene,
                  subsample_overdraw, subsample_seed,
                  snn_threshold = [pool.phototourism] snn_threshold,
                  min_track_pairs = [pool.phototourism] min_track_pairs,
                  gt_inlier_sampson_px = [pool.phototourism] gt_inlier_sampson_px) -> Vector{PtPair}

Resolve the scene's (matches_dir, recon_dir) via `_scene_paths`, read the
scene's COLMAP recon once, then iterate over `matches.h5` keys (with the
seeded subsample [`_seeded_key_subsample`](@ref) to `max_pairs_per_scene`)
and assemble PtPairs (`_load_pt_pair`). Drops pairs with too few
SfM-track-paired observations to give a stable σ̂²_p (`min_track_pairs`). The
subsample sizes and seed are the caller's (its pool configuration); the read
thresholds default to configs/default.toml `[pool.phototourism]`.
"""
function load_pt_pairs(scene::AbstractString, source::Symbol;
                       trees::PtRawTrees,
                       min_csponds::Integer,
                       max_pairs_per_scene::Integer,
                       subsample_overdraw::Integer,
                       subsample_seed::Integer,
                       snn_threshold::Real = _config_default("pool.phototourism", "snn_threshold"),
                       min_track_pairs::Integer = _config_default("pool.phototourism", "min_track_pairs"),
                       gt_inlier_sampson_px::Real = _config_default("pool.phototourism", "gt_inlier_sampson_px"))
    matches_dir, recon_dir = _scene_paths(scene, source, trees)
    matches_path = joinpath(matches_dir, "matches.h5")
    conf_path    = joinpath(matches_dir, "match_conf.h5")
    for p in (matches_path, conf_path)
        isfile(p) || error("Missing $p")
    end

    mdl = VisualGeometryIo.load(VisualGeometryIo.ColmapModel, recon_dir; contract = UnknownScale())
    cameras = Dict(c.camera_id => c for c in mdl.cameras)
    images  = Dict(replace(im.name, r"\.(jpg|jpeg|png)$"i => "") => im for im in mdl.images)
    recon   = (cameras = cameras, images = images, track_index = _build_track_index(mdl.points3D))

    h5open(matches_path, "r") do matches_f
        h5open(conf_path, "r") do conf_f
            all_keys = collect(keys(matches_f))
            sort!(all_keys)

            want_keys = _seeded_key_subsample(all_keys, scene; max_pairs_per_scene,
                                              subsample_overdraw, subsample_seed)

            pairs = PtPair[]
            sizehint!(pairs, min(max_pairs_per_scene, length(want_keys)))
            for k in want_keys
                p = _load_pt_pair(matches_f, conf_f, recon, k;
                                  snn_threshold, min_csponds, gt_inlier_sampson_px)
                p === nothing && continue
                length(p.track_kpts_A) >= min_track_pairs || continue
                push!(pairs, p)
                length(pairs) >= max_pairs_per_scene && break
            end
            return pairs
        end
    end
end
