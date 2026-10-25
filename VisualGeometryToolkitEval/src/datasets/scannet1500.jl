# =============================================================================
# src/datasets/scannet1500.jl — ScanNet-1500: the 1500 evaluation pairs of
# `test.npz` (the pair list of LoFTR and RoMa) and Kinect depth resampled to
# the colour resolution.
#
# `test.npz`: `name` (1500 × 4 UInt16: scene, sub-scene, frame a, frame b) and
# `rel_pose` (1500 × 12 Float32: the row-major source → target 3×4, metres).
# The rig is built in metres (`t .* m`), the depth plain metres.
# =============================================================================

# Every ScanNet colour frame is 1296 × 968; depth is 640 × 480.
const _SCANNET_COLOR_SIZE = Size2(1296, 968)

"""
    scannet1500_pairs(root) -> Vector{StereoPair}

The 1500 ScanNet evaluation pairs of `<root>/test.npz`, each scene's
calibration from `scans_test/<scene>/intrinsic/intrinsic_color.txt`.
"""
function scannet1500_pairs(root::AbstractString)
    npz = npzread(joinpath(root, "test.npz"))
    name = npz["name"]::Matrix{UInt16}
    rel_pose = npz["rel_pose"]::Matrix{Float32}
    (size(name, 2) == 4 && size(rel_pose, 2) == 12 && size(name, 1) == size(rel_pose, 1)) ||
        error("$(joinpath(root, "test.npz")): expected name (N, 4) and rel_pose (N, 12)")
    models = Dict{String,CameraModel}()
    identity_pose = EuclideanMap(SMatrix{3,3,Float64,9}(I), zero(Vec3{Float64}) .* m)
    pairs = Vector{StereoPair}(undef, size(name, 1))
    for i in axes(name, 1)
        scene = "scene$(lpad(Int(name[i, 1]), 4, '0'))_$(lpad(Int(name[i, 2]), 2, '0'))"
        scene_dir = joinpath(root, "scans_test", scene)
        cm = get!(models, scene) do
            K = readdlm(joinpath(scene_dir, "intrinsic", "intrinsic_color.txt"), Float64)::Matrix{Float64}
            CameraModel(CameraCalibrationMat(SMatrix{3,3,Float64,9}(K[1:3, 1:3])); image_origin = :opencv)
        end
        R, t = _row_major_pose(view(rel_pose, i, :))
        rig = StereoRig(BoundedCamera(Camera(cm, identity_pose), _SCANNET_COLOR_SIZE),
                        BoundedCamera(Camera(cm, EuclideanMap(R, Vec3{Float64}(t) .* m; project = true)), _SCANNET_COLOR_SIZE))
        src, tgt = "$(Int(name[i, 3])).jpg", "$(Int(name[i, 4])).jpg"
        pairs[i] = StereoPair((scene, src, tgt, joinpath(scene_dir, "color", src), joinpath(scene_dir, "color", tgt), rig))
    end
    return pairs
end

# Both frames' depth (Kinect: invalid 0), nearest-resampled to the colour size.
function _load_scannet1500_depths(root::AbstractString, scene::AbstractString,
                                  source_name::AbstractString, target_name::AbstractString)
    depth_dir = joinpath(root, "scans_test", scene, "depth")
    hw = (Int(_SCANNET_COLOR_SIZE.height), Int(_SCANNET_COLOR_SIZE.width))
    load(name) = _nearest_resize(_decode_depth_png(joinpath(depth_dir, "$(first(splitext(name))).png"), UInt16(0)), hw)
    return load(source_name), load(target_name)
end

stereo_pairs(ds::ScanNet1500) = scannet1500_pairs(ds.root)
function load_validation_data(ds::ScanNet1500, p::NamedTuple, csponds)
    depth_src, depth_tgt = _load_scannet1500_depths(ds.root, p.scene, p.source_name, p.target_name)
    return (; rig = p.rig, csponds, depth_src, depth_tgt)
end
