# =============================================================================
# src/datasets/eth3d.jl — ETH3D high-resolution multi-view stereo (Schöps et
# al., CVPR 2017): the co-visible pairs of the UNDISTORTED DSLR images (what a
# matcher sees) and the laser depth of the DISTORTED ones (where the depth lives).
#
# Calibration is COLMAP text, read by VisualGeometryToolkitIO with the world unit
# stated (ETH3D is metric). The rig is built in METRES on the file's own
# numbers — the unit of the laser depth and of the metric GT tolerance.
# Depth files `ground_truth_depth/dslr_images/<name>` are raw little-endian
# Float32 in row-major `(H, W)`; Inf (no laser return) reads as NaN.
# =============================================================================

const _ETH3D_CONTRACT = UnitContract(world = m)

# The ETH3D COLMAP model of `<scene_dir>/<calib>` (`dslr_calibration_jpg` or `…_undistorted`).
_eth3d_model(scene_dir::AbstractString, calib::AbstractString) =
    VgtkIO.load(ColmapModel, joinpath(scene_dir, calib); contract = _ETH3D_CONTRACT)

# A COLMAP camera and image → a bounded camera, world→camera in metres: the
# file's `tvec` is in the contract's world unit, metres, read as such.
function _eth3d_camera(cam::ColmapCamera, im::ColmapImage)
    q, t = im.qvec, im.tvec
    pose = EuclideanMap(QuatRotation(q[1], q[2], q[3], q[4]), Vec3{Float64}(t[1], t[2], t[3]) .* m)
    return BoundedCamera(Camera(camera_model(cam), pose), Size2(cam.width, cam.height))
end

# ETH3D's raw depth (row-major Float32, `(height, width)`) as a `(H, W)` matrix, NaN invalid.
function _load_eth3d_depth(path::AbstractString, width::Int, height::Int)
    floats = Vector{Float32}(undef, width * height)
    read!(path, floats)
    arr = permutedims(reshape(floats, (width, height)), (2, 1))
    return replace!(x -> isfinite(x) ? x : NaN32, arr)
end

# One scene's undistorted calibration: the cameras by image name, the names
# present on disk (sorted) and each image's triangulated point ids; `nothing`
# when the scene has no undistorted calibration.
function _walk_eth3d_scene(scene_path::AbstractString)
    img_dir = joinpath(scene_path, "images", "dslr_images_undistorted")
    calib = joinpath(scene_path, "dslr_calibration_undistorted")
    (isdir(img_dir) && isfile(joinpath(calib, "cameras.txt")) && isfile(joinpath(calib, "images.txt"))) ||
        return nothing
    mdl = _eth3d_model(scene_path, "dslr_calibration_undistorted")
    cameras = Dict(c.camera_id => c for c in mdl.cameras)
    images = Dict(basename(im.name) => im for im in mdl.images)   # names are relative to <scene>/images/
    names = sort!([n for n in readdir(img_dir) if haskey(images, n)])
    cams = Dict(n => _eth3d_camera(cameras[images[n].camera_id], images[n]) for n in names)
    # co-visibility on triangulated detections only (point3D_id −1: never triangulated)
    visible = Dict(n => Set(Int64(pid) for (_, _, pid) in images[n].points2D if pid != -1) for n in names)
    return (; scene = basename(scene_path), img_dir, cams, names, visible)
end

"""
    eth3d_pairs(scenes_root; min_covisibility = [datasets.eth3d] min_covisibility) -> Vector{StereoPair}

Every unordered pair of undistorted DSLR images of every scene under
`scenes_root` with at least `min_covisibility` co-visible triangulated points
(SuperANSAC's ETH3D protocol), its rig from the undistorted calibration.
"""
function eth3d_pairs(scenes_root::AbstractString;
                     min_covisibility::Int = _config_default("datasets.eth3d", "min_covisibility"))
    pairs = StereoPair[]
    for scene_path in readdir(scenes_root; join = true)
        isdir(scene_path) || continue
        s = _walk_eth3d_scene(scene_path)
        s === nothing && continue
        for i in 1:length(s.names)-1, j in i+1:length(s.names)
            ni, nj = s.names[i], s.names[j]
            length(intersect(s.visible[ni], s.visible[nj])) >= min_covisibility || continue
            push!(pairs, StereoPair((s.scene, ni, nj, joinpath(s.img_dir, ni), joinpath(s.img_dir, nj),
                                     StereoRig(s.cams[ni], s.cams[nj]))))
        end
    end
    return pairs
end

# An undistorted pixel → the distorted pixel that observes the same ray: both
# cameras share the world pose; only the lens differs.
function _undistorted_to_distorted(cam_undist, cam_dist, p)
    X = unproject(model(cam_undist), SVector{2,Float64}(p[1], p[2]), 1.0)
    return project(model(cam_dist), X)
end

# The distorted-image rig (lens models of `dslr_calibration_jpg`), the
# correspondences moved into it, and the two laser depth maps.
function _eth3d_validation_data(scenes_root::AbstractString, p, csponds_undist::AbstractVector)
    scene_dir = joinpath(scenes_root, p.scene)
    mdl = _eth3d_model(scene_dir, "dslr_calibration_jpg")
    cameras = Dict(c.camera_id => c for c in mdl.cameras)
    images = Dict(im.name => im for im in mdl.images)
    im_a, im_b = images["dslr_images/$(p.source_name)"], images["dslr_images/$(p.target_name)"]
    ca, cb = cameras[im_a.camera_id], cameras[im_b.camera_id]
    rig = StereoRig(_eth3d_camera(ca, im_a), _eth3d_camera(cb, im_b))
    depth_src = _load_eth3d_depth(joinpath(scene_dir, "ground_truth_depth", im_a.name), Int(ca.width), Int(ca.height))
    depth_tgt = _load_eth3d_depth(joinpath(scene_dir, "ground_truth_depth", im_b.name), Int(cb.width), Int(cb.height))
    src = [_undistorted_to_distorted(source(p.rig), source(rig), centroid(c.first)) for c in csponds_undist]
    tgt = [_undistorted_to_distorted(target(p.rig), target(rig), centroid(c.second)) for c in csponds_undist]
    return (; rig, csponds = csponds(src, tgt), depth_src, depth_tgt)
end

stereo_pairs(ds::ETH3D) = eth3d_pairs(ds.scenes_root; min_covisibility = ds.min_covisibility)
load_validation_data(ds::ETH3D, p::NamedTuple, csponds) = _eth3d_validation_data(ds.scenes_root, p, csponds)
