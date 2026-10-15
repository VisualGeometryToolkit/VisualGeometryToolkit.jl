# =============================================================================
# src/datasets/seven_scenes.jl — Microsoft 7-Scenes (Kinect): SuperANSAC's
# frame pairs (tests/datasets/seven_scenes.py) of each split's sequences,
# `frame-XXXXXX.color.png` with its 4×4 camera-to-world `frame-XXXXXX.pose.txt`
# and its `frame-XXXXXX.depth.png` (millimetres, invalid 65535), every scene
# under the one shared Kinect colour calibration of `[datasets.seven_scenes]`.
# =============================================================================

# The Kinect colour calibration (OpenCV origin) and frame size of every scene.
function _seven_scenes_K()
    f = Float64(_config_default("datasets.seven_scenes", "focal"))
    cx, cy = Float64.(_config_default("datasets.seven_scenes", "principal_point"))
    return SMatrix{3,3,Float64,9}(f, 0, 0, 0, f, 0, cx, cy, 1)
end
_seven_scenes_extent() = Size2(_config_default("datasets.seven_scenes", "image_size")...)

# A split's sequences, `"<scene>/<sequence>"`, in the order of record.
function _seven_scenes_sequences(split::Symbol)
    seqs = split === :test ? _config_default("datasets.seven_scenes", "test_sequences") :
           split === :train ? _config_default("datasets.seven_scenes", "train_sequences") :
           throw(ArgumentError("split must be :train or :test, got $split"))
    return String.(seqs)
end

function _parse_seven_scenes_pose(path::AbstractString)
    M = readdlm(path, Float64)::Matrix{Float64}
    size(M) == (4, 4) || error("$path: expected a 4×4 pose, got $(size(M))")
    return SMatrix{4,4,Float64,16}(M)
end

"""
    seven_scenes_pairs(root; split = :test, scene = :all, interval, stride) -> Vector{StereoPair}

The `(frame[i], frame[i + interval])` pairs, `i = 1, 1 + stride, …`, of every
sequence of `split` (`:test`, `:train`) under `root`, for every scene or the
one named by `scene`. The scene id of a pair is `"<scene>/<sequence>"`.
"""
function seven_scenes_pairs(root::AbstractString; split::Symbol = :test,
                            scene::Union{Symbol,AbstractString} = :all,
                            interval::Integer = _config_default("datasets.seven_scenes", "interval"),
                            stride::Integer = _config_default("datasets.seven_scenes", "stride"))
    K, extent = _seven_scenes_K(), _seven_scenes_extent()
    sequences = _seven_scenes_sequences(split)
    scene === :all || any(s -> first(splitpath(s)) == String(scene), sequences) ||
        throw(ArgumentError("7-Scenes: no $split sequence of scene $scene"))
    pairs = StereoPair[]
    for scene_seq in sequences
        scene === :all || first(splitpath(scene_seq)) == String(scene) || continue
        seq_dir = joinpath(root, scene_seq)
        isdir(seq_dir) || (@warn "7-Scenes: no sequence directory, skipped" sequence = scene_seq; continue)
        files = sort(filter(endswith("color.png"), readdir(seq_dir)))
        pose(f) = _parse_seven_scenes_pose(joinpath(seq_dir, replace(f, "color.png" => "pose.txt")))
        for (i, j) in _consecutive_pose_pairs(length(files); interval, stride)
            push!(pairs, _make_stereo_pair(scene_seq, files[i], files[j], joinpath(seq_dir, files[i]),
                                           joinpath(seq_dir, files[j]), K, pose(files[i]), pose(files[j]), extent))
        end
    end
    return pairs
end

# Both frames' depth (colour resolution already; invalid 65535).
function _load_seven_scenes_depths(root::AbstractString, scene::AbstractString,
                                   source_name::AbstractString, target_name::AbstractString)
    load(name) = _decode_depth_png(joinpath(root, scene, replace(name, "color.png" => "depth.png")), typemax(UInt16))
    return load(source_name), load(target_name)
end

stereo_pairs(ds::SevenScenes) =
    seven_scenes_pairs(ds.root; split = ds.split, scene = ds.scene, interval = ds.interval, stride = ds.stride)
function load_validation_data(ds::SevenScenes, p::NamedTuple, csponds)
    depth_src, depth_tgt = _load_seven_scenes_depths(ds.root, p.scene, p.source_name, p.target_name)
    return (; rig = p.rig, csponds, depth_src, depth_tgt)
end
