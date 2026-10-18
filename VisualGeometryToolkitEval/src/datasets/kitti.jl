# =============================================================================
# src/datasets/kitti.jl — KITTI odometry: consecutive left-grayscale (`image_0`)
# frame pairs of the sequences with poses (SuperANSAC's tests/datasets/kitti.py),
# and a sparse depth map per frame from the velodyne scan.
#
# `calib.txt` holds `P0` (= K [I | 0] for cam0) and `Tr` (velodyne → cam0), each
# a row-major 3×4; `poses/<seq>.txt` one row-major camera-to-world 3×4 per frame.
# =============================================================================

# The 3×4 row-major matrix on the `tag` line (`"P0:"`, `"Tr:"`) of a sequence's calib.txt.
function _kitti_calib_row(seq_dir::AbstractString, tag::AbstractString)
    path = joinpath(seq_dir, "calib.txt")
    for line in eachline(path)
        startswith(line, tag) || continue
        vals = parse.(Float64, split(strip(line[length(tag)+1:end])))
        length(vals) == 12 || error("$path: expected 12 values on $tag, got $(length(vals))")
        return _row_major_pose(vals)
    end
    error("$path: no $tag line")
end

# The cam0 calibration: `P0 = K [I | 0]`, so K is its left 3×3.
_parse_kitti_calib(seq_dir::AbstractString) = first(_kitti_calib_row(seq_dir, "P0:"))

# The velodyne → cam0 (rectified) rigid map of `Tr`.
function _parse_kitti_tr(seq_dir::AbstractString)
    R, t = _kitti_calib_row(seq_dir, "Tr:")
    return EuclideanMap(R, Vec3{Float64}(t); project = true)
end

# The camera-to-world poses of `poses/<seq>.txt`, as 4×4.
function _parse_kitti_poses(path::AbstractString)
    M = readdlm(path, Float64)::Matrix{Float64}
    size(M, 2) == 12 || error("$path: a pose row needs 12 values, got $(size(M, 2))")
    bottom = SMatrix{1,4,Float64,4}(0, 0, 0, 1)
    return [vcat(hcat(_row_major_pose(view(M, i, :))...), bottom) for i in axes(M, 1)]
end

# A velodyne scan (`[x y z intensity]` Float32 records) as its points, intensity dropped.
function _load_kitti_velodyne(path::AbstractString)
    raw = read(path)
    length(raw) % (4 * sizeof(Float32)) == 0 || error("$path: velodyne scan not divisible by 4 floats")
    return [Point3{Float64}(p[1], p[2], p[3]) for p in reinterpret(SVector{4,Float32}, raw)]
end

# The cam0 extent of a sequence (most are 1241×376; 04 is 1226×370): its first frame's size.
function _kitti_cam0_extent(seq_dir::AbstractString)
    img_dir = joinpath(seq_dir, "image_0")
    files = sort(filter(endswith(".png"), readdir(img_dir)))
    isempty(files) && error("no image_0 frames in $img_dir")
    h, w = size(PNGFiles.load(joinpath(img_dir, first(files))))
    return Size2(w, h)
end

# The sparse depth map of a scan: each point in front of the camera projected
# through the OpenCV-origin `cam`; the nearest return wins a pixel; NaN elsewhere.
function _kitti_velo_to_depth(points::AbstractVector{<:Point3}, cam::CameraModel,
                              velo_to_cam::EuclideanMap, extent::Size2)
    w, h = Int(extent.width), Int(extent.height)
    depth = fill(NaN32, h, w)
    @inbounds for P_v in points
        P_v[1] > 0.0 || continue                     # the velodyne x axis points forward
        P_c = velo_to_cam(P_v)
        P_c[3] > 0 || continue
        u = project(cam, P_c)                        # 0-based pixel centres
        col, row = Int(round(u[1])) + 1, Int(round(u[2])) + 1
        (1 <= col <= w && 1 <= row <= h) || continue
        z = Float32(P_c[3])
        cur = depth[row, col]
        (isnan(cur) || z < cur) && (depth[row, col] = z)
    end
    return depth
end

# Both frames' velodyne depth maps; the scene id is `kitti_<seq>`, the names `NNNNNN.png`.
function _load_kitti_depths(root::AbstractString, scene::AbstractString,
                            source_name::AbstractString, target_name::AbstractString)
    seq_dir = joinpath(root, "sequences", String(last(split(scene, '_'))))
    cam = CameraModel(CameraCalibrationMat(_parse_kitti_calib(seq_dir)); image_origin = :opencv)
    velo_to_cam = _parse_kitti_tr(seq_dir)
    extent = _kitti_cam0_extent(seq_dir)
    scan(name) = _load_kitti_velodyne(joinpath(seq_dir, "velodyne", replace(name, ".png" => ".bin")))
    return (_kitti_velo_to_depth(scan(source_name), cam, velo_to_cam, extent),
            _kitti_velo_to_depth(scan(target_name), cam, velo_to_cam, extent))
end

"""
    kitti_pairs(root; sequences = [datasets.kitti] sequences) -> Vector{StereoPair}

The cam0 frame pairs of each KITTI odometry sequence under `root` with poses,
by the `[datasets.kitti]` pair rule (consecutive frames). A sequence missing
its directory or pose file is skipped with a warning.
"""
function kitti_pairs(root::AbstractString; sequences = String.(_config_default("datasets.kitti", "sequences")))
    pairs = StereoPair[]
    for s in sequences
        seq_dir, poses_path = joinpath(root, "sequences", s), joinpath(root, "poses", "$(s).txt")
        isdir(seq_dir) || (@warn "KITTI: no sequence directory, skipped" sequence = s; continue)
        isfile(poses_path) || (@warn "KITTI: no pose file, skipped" sequence = s; continue)
        K = _parse_kitti_calib(seq_dir)
        poses = _parse_kitti_poses(poses_path)
        img_dir = joinpath(seq_dir, "image_0")
        files = sort(filter(endswith(".png"), readdir(img_dir)))
        n = min(length(files), length(poses))
        extent = _kitti_cam0_extent(seq_dir)
        for (i, j) in _consecutive_pose_pairs(n; interval = _config_default("datasets.kitti", "interval"),
                                              stride = _config_default("datasets.kitti", "stride"))
            push!(pairs, _make_stereo_pair("kitti_$(s)", files[i], files[j], joinpath(img_dir, files[i]),
                                           joinpath(img_dir, files[j]), K, poses[i], poses[j], extent))
        end
    end
    return pairs
end

stereo_pairs(ds::KITTI) = kitti_pairs(ds.root; sequences = ds.sequences)
function load_validation_data(ds::KITTI, p::NamedTuple, csponds)
    depth_src, depth_tgt = _load_kitti_depths(ds.root, p.scene, p.source_name, p.target_name)
    return (; rig = p.rig, csponds, depth_src, depth_tgt)
end
