# =============================================================================
# src/datasets/stereo_datasets.jl — the depth-validated stereo datasets: the
# pair enumeration, ground-truth depth and GT-validation tolerances behind
# `consolidate_csponds`, which writes the consolidated csponds files that
# `load_roma_pairs` reads.
#
#   ETH3D        co-visible undistorted DSLR pairs; laser depth
#   ScanNet1500  the 1500 evaluation pairs of `test.npz`; Kinect depth
#   KITTI        consecutive cam0 frames of the odometry sequences; velodyne
#   SevenScenes  SuperANSAC's (interval, stride) frame pairs; Kinect depth
#
# A dataset is a value holding its on-disk root (the caller's, never a path in
# code); `stereo_pairs`, `load_validation_data` and `gt_config` dispatch on it.
# Every pixel coordinate here is in the image the matcher saw; every depth is in
# metres, invalid as NaN (VisualGeometryToolkitCore's convention).
#
# Units: a pair's `rig` is a public physical value, so the readers build it with
# FlexUnits quantities in metres (each dataset's poses are metric, so `t .* m`
# reads the file's numbers as metres, exactly: the scale is 1). Depth maps, a depth
# validator's inputs, stay plain metres, as VisualGeometryToolkitCore's
# `validate_gt_inliers` takes them; the consolidated record is plain data and
# strips the rig once (`consolidate.jl`).
# =============================================================================

"""
    StereoPair

One image pair of a [`AbstractStereoDataset`](@ref): the `scene`, the two image
names (`source_name`, `target_name`) and paths (`source_img_path`,
`target_img_path`, under the dataset's root), and the GT `rig`
(`VisualGeometryToolkitCore.StereoRig`) in the frame of the images the matcher sees,
its translations FlexUnits lengths in metres.
"""
const StereoPair = @NamedTuple{
    scene::String,
    source_name::String,
    target_name::String,
    source_img_path::String,
    target_img_path::String,
    rig::StereoRig,
}

# -----------------------------------------------------------------------------
# GT-validation tolerances
# -----------------------------------------------------------------------------

"""
    GTConfig(; gt_method, metric_tol, rel_tol, bidirectional)
    GTConfig(table::AbstractString)

How correspondences are validated against ground-truth depth and pose
(VisualGeometryToolkitCore `validate_gt_inliers`): `gt_method` (a provenance label,
`:depth_warp_eth3d_laser`), `metric_tol` (the 3D disagreement at the surface
below which a correspondence is an inlier, a length quantity: metric, so one
number is portable across resolutions and focal lengths), `rel_tol` (the
relative depth-consistency check applied first) and `bidirectional` (check
both warp directions).

`GTConfig(table)` reads the values of record of `[<table>]` in
configs/default.toml (`"gt.eth3d.splg"`; `metric_tol_mm` in millimetres).
"""
Base.@kwdef struct GTConfig
    gt_method     :: Symbol
    metric_tol    :: BaseQuantity
    rel_tol       :: Float64
    bidirectional :: Bool
end

GTConfig(table::AbstractString) =
    GTConfig(gt_method = Symbol(_config_default(table, "gt_method")),
             metric_tol = read_length(_config_default(table, "metric_tol_mm"), mm),
             rel_tol = _config_default(table, "rel_tol"),
             bidirectional = _config_default(table, "bidirectional"))

# -----------------------------------------------------------------------------
# The datasets
# -----------------------------------------------------------------------------

"""
    AbstractStereoDataset

A depth-validated stereo benchmark: a value carrying its on-disk root, on
which [`stereo_pairs`](@ref), [`load_validation_data`](@ref) and
[`gt_config`](@ref) dispatch, and which [`consolidate_csponds`](@ref) drives.
"""
abstract type AbstractStereoDataset end

"""
    ScanNet1500(root)

The ScanNet-1500 evaluation split (1500 pairs over 100 scenes; the pair list of
LoFTR and RoMa). `root` holds `test.npz` and `scans_test/<scene>/{color,depth,intrinsic}`.
"""
struct ScanNet1500 <: AbstractStereoDataset
    root::String
end

"""
    ETH3D(scenes_root; min_covisibility = [datasets.eth3d] min_covisibility)

The ETH3D high-resolution multi-view benchmark with laser depth. `scenes_root`
holds one directory per scene with `images/dslr_images_undistorted/`,
`dslr_calibration_undistorted/`, `dslr_calibration_jpg/` and
`ground_truth_depth/dslr_images/`. A pair needs `min_covisibility` co-visible
triangulated points.
"""
Base.@kwdef struct ETH3D <: AbstractStereoDataset
    scenes_root      :: String
    min_covisibility :: Int = _config_default("datasets.eth3d", "min_covisibility")
end

"""
    KITTI(root; sequences = [datasets.kitti] sequences)

The KITTI odometry benchmark, consecutive left-grayscale (`image_0`) frames.
`root` holds `sequences/<seq>/{image_0,velodyne,calib.txt}` and `poses/<seq>.txt`.
"""
Base.@kwdef struct KITTI <: AbstractStereoDataset
    root      :: String
    sequences :: Vector{String} = String.(_config_default("datasets.kitti", "sequences"))
end

"""
    SevenScenes(root; split = :test, scene = :all, interval, stride)

Microsoft 7-Scenes (Kinect). `root` holds `<scene>/<sequence>/frame-*.{color,depth}.png`
and `frame-*.pose.txt`. `split` is `:test` or `:train` (SuperANSAC's sequence
lists, `[datasets.seven_scenes]`); `scene` is `:all` or one scene name; the
pairs are `(frame[i], frame[i + interval])` for `i = 1, 1 + stride, …`.
"""
Base.@kwdef struct SevenScenes <: AbstractStereoDataset
    root     :: String
    split    :: Symbol = :test
    scene    :: Union{Symbol,String} = :all
    interval :: Int = _config_default("datasets.seven_scenes", "interval")
    stride   :: Int = _config_default("datasets.seven_scenes", "stride")
end

"""
    stereo_pairs(ds::AbstractStereoDataset) -> Vector{StereoPair}

Every evaluation pair of the dataset, by the dataset's own pair rule.
"""
function stereo_pairs end

"""
    load_validation_data(ds::AbstractStereoDataset, p::StereoPair, csponds)
        -> (; rig, csponds, depth_src, depth_tgt)

What GT validation of `csponds` (pixels of the images `p` names) needs: the rig
and correspondences in the frame of the depth maps, and the two depth maps
(metres, NaN invalid). Only ETH3D changes frame — its laser depth lives in the
distorted DSLR images, its matches in the undistorted ones; every other dataset
returns `p.rig` and `csponds` unchanged.
"""
function load_validation_data end

"""
    gt_config(ds::AbstractStereoDataset, matcher::Symbol) -> GTConfig

The GT-validation tolerances of record for `(dataset, matcher)` (`:roma`,
`:splg`): configs/default.toml `[gt.<dataset>.<matcher>]`. A combination
without a calibrated table is a `MethodError`.
"""
function gt_config end

gt_config(ds::AbstractStereoDataset, matcher::Symbol) = gt_config(ds, Val(matcher))
gt_config(::ScanNet1500, ::Val{:roma}) = GTConfig("gt.scannet1500.roma")
gt_config(::ScanNet1500, ::Val{:splg}) = GTConfig("gt.scannet1500.splg")
gt_config(::ETH3D,       ::Val{:roma}) = GTConfig("gt.eth3d.roma")
gt_config(::ETH3D,       ::Val{:splg}) = GTConfig("gt.eth3d.splg")
gt_config(::SevenScenes, ::Val{:roma}) = GTConfig("gt.seven_scenes.roma")
gt_config(::KITTI, ::Union{Val{:roma},Val{:splg}}) = GTConfig("gt.kitti")   # one tolerance for every matcher

# -----------------------------------------------------------------------------
# Shared readers
# -----------------------------------------------------------------------------

# A pair from absolute camera-to-world poses `T_src`, `T_tgt` (4×4, metres) and
# one shared calibration `K` (OpenCV pixel origin): the source camera at the
# identity, the target at `inv(T_tgt) · T_src` (re-orthonormalized: text-file
# poses drift ~1e-6 from SO(3)); the rig in metres.
function _make_stereo_pair(scene::AbstractString, src_name::AbstractString, tgt_name::AbstractString,
                           src_path::AbstractString, tgt_path::AbstractString,
                           K::SMatrix{3,3,Float64,9}, T_src::SMatrix{4,4,Float64,16},
                           T_tgt::SMatrix{4,4,Float64,16}, extent::Size2)
    cm = CameraModel(CameraCalibrationMat(K); image_origin = :opencv)
    cam_src = BoundedCamera(PosedCamera(cm, EuclideanMap(SMatrix{3,3,Float64,9}(I), zero(Vec3{Float64}) .* m)), extent)
    T = inv(T_tgt) * T_src
    cam_tgt = BoundedCamera(PosedCamera(cm, EuclideanMap(SMatrix{3,3,Float64,9}(T[1:3, 1:3]),
                                                    Vec3{Float64}(T[1:3, 4]) .* m; project = true)), extent)
    return StereoPair((scene, src_name, tgt_name, src_path, tgt_path, StereoRig(cam_src, cam_tgt)))
end

# The 1-based `(i, i + interval)` frame pairs for `i = 1, 1 + stride, …` while
# `i + interval ≤ n_frames` (SuperANSAC's `range(0, n - interval, stride)`).
function _consecutive_pose_pairs(n_frames::Integer; interval::Integer, stride::Integer)
    interval > 0 || throw(ArgumentError("interval must be > 0, got $interval"))
    stride > 0 || throw(ArgumentError("stride must be > 0, got $stride"))
    return [(i, i + interval) for i in 1:stride:(n_frames - interval)]
end

# A 16-bit depth PNG in millimetres as metres, the dataset's invalid sentinel
# (ScanNet 0, 7-Scenes 65535) as NaN32.
function _decode_depth_png(path::AbstractString, invalid::UInt16)
    raw = reinterpret(UInt16, PNGFiles.load(path))
    return @. ifelse(raw == invalid, NaN32, Float32(raw) / 1000.0f0)
end

# Nearest-neighbour resampling of an image to `(h, w)` with the outer pixel
# corners mapped onto each other (pixel `i` covers `i ± 1/2`; sample position
# `s·i + 1/2 − s/2`, `s = old/new`, clamped to the image, rounded half up).
function _nearest_resize(A::AbstractMatrix, (h, w)::Tuple{Int,Int})
    function index_map(n_old, n_new)
        s = n_old / n_new
        off = 1 - 0.5 - s * (1 - 0.5)
        return [floor(Int, clamp(s * i + off, 1, n_old) + 0.5) for i in 1:n_new]
    end
    rows, cols = index_map(size(A, 1), h), index_map(size(A, 2), w)
    return A[rows, cols]
end
