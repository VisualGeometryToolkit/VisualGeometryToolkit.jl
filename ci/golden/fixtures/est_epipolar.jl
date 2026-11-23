# Fixtures of the epipolar family (F, E, the pure rotation, the relative-pose bundle): the scenes of
# VisualGeometryToolkitEstimation's suite — scenes_testsetups.jl's FindScenes (stereo_scene, TooFewData),
# ransac_fundmat_tests' fundmat_scene, fitting_pipelines_tests' calibrated F/E scenes, ransac_essential_tests'
# _e_pixel_scene and its PROSAC-scored pairs, ransac_no_model_tests' one-source pairs, ransac_rotation_tests'
# RotationScene (a fisheye past 90°) and relpose_bundle_tests' _scene/_seed/_harness_scene — built from the shared
# synthetic scenes and data constructors only.
using Test, StableRNGs, StaticArrays, StructArrays, Random, LinearAlgebra
using Rotations: RotMatrix, RotXYZ, RotY, AngleAxis
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _EP_E = VgtkEstimation.synthetic_scenes().EpipolarScenes
const _EP_P = VgtkEstimation.synthetic_scenes().PoseScenes
const EP_ECAM = _EP_E.ECAM
const EP_CAM800 = VgtkCore.CameraModel(800.0 * VgtkCore.px, SA[960.0 * VgtkCore.px, 540.0 * VgtkCore.px])   # ransac_essential_tests

Base.include(@__MODULE__, joinpath(@__DIR__, "common.jl"))   # the find_* scenes, too-few sets and pixel-pose scenes

# --- ransac_essential_tests' _e_pixel_scene ------------------------------------------------------------------------
function ep_e_pixel_scene(rng, cam₁, cam₂; n_in = 150, n_out = 60, σ = 0.5)
    R = RotY(0.2); t = normalize(SA[0.8, 0.1, 0.3])
    c₁ = SA[960.0, 540.0]; c₂ = SA[900.0, 560.0]
    a, b = _EP_E.stereo_pixel_pairs(
        rng, cam₁, cam₂, R, t; n = n_in, σ₁ = σ, σ₂ = σ, point = _EP_E.box_point, n_out,
        outlier = rng -> (c₁ + SA[800 * (rand(rng) - 0.5), 500 * (rand(rng) - 0.5)], c₂ + SA[800 * (rand(rng) - 0.5), 500 * (rand(rng) - 0.5)])
    )
    return VgtkCore.Point2{Float64}.(a), VgtkCore.Point2{Float64}.(b)
end

# The fundamental-matrix scenes: name => correspondences.
function ep_fundmat_scenes()
    u₁, u₂ = stereo_scene(PCAM)
    fs = _EP_E.fundmat_scene(StableRNG(42); n_inliers = 100, n_outliers = 30, σ = 0.5)
    rng = Xoshiro(5)                                   # ransac_no_model_tests: every source one point
    src = fill(VgtkCore.Point2(10.0, 20.0), 20)
    dst = [VgtkCore.Point2(100 * rand(rng), 100 * rand(rng)) for _ in 1:20]
    return [
        "stereo_pcam_140+60" => StructArray(u₁ .=> u₂),
        "fundmat_scene_100+30" => VgtkCore.correspondences(fs.source, fs.target),
        "stereo_pcam_pairs" => u₁ .=> u₂,
        "one_source_20" => VgtkCore.correspondences(src, dst),
    ]
end

# The essential scenes: name => (camera1, camera2, pixel correspondences).
function ep_essential_scenes()
    v₁, v₂ = stereo_scene(EP_ECAM; seed = 14)
    a, b = ep_e_pixel_scene(StableRNG(2034), EP_CAM800, EP_CAM800)
    rng = StableRNG(2033)
    a2, b2 = ep_e_pixel_scene(rng, EP_CAM800, EP_CAM800; n_out = 20)
    s = rand(rng, length(a2))
    rng = StableRNG(17)                                # fitting_pipelines: calibrated + 0.002, then 50 outliers, in ECAM pixels
    c₁, c₂ = _EP_E.calibrated_pairs(rng, RotY(0.2), normalize(SA[0.8, 0.1, 0.3]); n = 120, σ = 0.002)
    for _ in 1:50
        push!(c₁, VgtkCore.Point2{Float64}(randn(rng), randn(rng)))
        push!(c₂, VgtkCore.Point2{Float64}(randn(rng), randn(rng)))
    end
    return [
        "stereo_ecam_140+60" => (EP_ECAM, EP_ECAM, VgtkCore.correspondences(v₁, v₂)),
        "epix_cam800_150+60" => (EP_CAM800, EP_CAM800, VgtkCore.correspondences(a, b)),
        "epix_cam800_scored" => (EP_CAM800, EP_CAM800, VgtkCore.Correspondences(a2, b2; score = s)),
        "calib_ecam_120+50" => (EP_ECAM, EP_ECAM, VgtkCore.correspondences(_EP_E.topx.(c₁), _EP_E.topx.(c₂))),
    ]
end

# --- RotationScene (`FISHEYE`, `rotation_scene`): the suite's fixture file ------------------------------------
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "rotation_scene.jl"))   # `FISHEYE`, `rotation_scene`

# The bearing form of a pixel set: unit bearings with their back-projection Jacobians as the M₁/M₂ columns.
function ep_bearings(camera1, camera2, cs)
    fm₁ = map(u -> VgtkCore.backproject_with_jacobian(camera1, u), VgtkCore.sources(cs))
    fm₂ = map(u -> VgtkCore.backproject_with_jacobian(camera2, u), VgtkCore.targets(cs))
    return VgtkCore.withattrs(VgtkCore.Correspondences(first.(fm₁), first.(fm₂)); M₁ = last.(fm₁), M₂ = last.(fm₂))
end

# The rotation scenes: name => (camera1, camera2, pixel correspondences).
ep_rotation_scenes() = [
    "fisheye_120_outl30" => (FISHEYE, FISHEYE, first(rotation_scene(Xoshiro(35), RotMatrix(RotXYZ(0.1, -0.2, 0.3)), 120; outl = 0.3))),
    "fisheye_300_outl30" => (FISHEYE, FISHEYE, first(rotation_scene(Xoshiro(8), RotMatrix(RotXYZ(0.2, -0.3, 0.4)), 300; outl = 0.3))),
]

# --- relpose_bundle_tests ------------------------------------------------------------------------------------------

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "relpose_scenes.jl"))   # `CAM1`, `CAM2`, `KC1`, `KC2`, `relpose_scene`, `relpose_seed`, `KP1`, `KP2`, `harness_scene`

# The relative-pose bundle fixtures: name => (seed pose, K1, K2, correspondences).
function ep_relpose_fixtures()
    rigid = _EP_P.rigid
    out = Pair{String, Any}[]
    for N in (100, 1000)
        R0, t0, u1, u2 = harness_scene(N)
        push!(out, "harness_$N" => (rigid(R0, t0), KP1, KP2, VgtkCore.correspondences(u1, u2)))
    end
    rng = StableRNG(3); Rgt, tgt, u1, u2 = relpose_scene(rng, 30; σ = 0.0)
    R0, t0 = relpose_seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    push!(out, "exact_30" => (rigid(R0, t0), KC1, KC2, VgtkCore.correspondences(u1, u2)))
    rng = StableRNG(5); Rgt, tgt, u1, u2 = relpose_scene(rng, 40; σ = 0.002)
    R0, t0 = relpose_seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    push!(out, "noisy_40" => (rigid(R0, t0), KC1, KC2, VgtkCore.correspondences(u1, u2)))
    rng = StableRNG(6); Rgt, tgt, u1, u2 = relpose_scene(rng, 30; σ = 0.002)   # a point at infinity, weighted 0 and 1
    d = normalize(SVector(0.1, -0.05, 1.0))
    push!(u1, SVector{2, Float64}(VgtkCore.project(CAM1, d))); push!(u2, SVector{2, Float64}(VgtkCore.project(CAM2, Rgt * d)))
    R0, t0 = relpose_seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    for w_far in (0.0, 1.0)
        push!(out, "infinity_31_wfar$(Int(w_far))" =>
            (rigid(R0, t0), KC1, KC2, VgtkCore.Correspondences(u1, u2; weights = vcat(ones(30), w_far))))
    end
    rng = StableRNG(7); Rgt, tgt, u1, u2 = relpose_scene(rng, 20; σ = 0.002)
    t0 = VgtkCore.Vec3{Float64}(tgt)
    push!(out, "drifted_seed_20" => (rigid(1.001 * SMatrix{3, 3, Float64, 9}(Rgt), t0), KC1, KC2, VgtkCore.correspondences(u1, u2)))
    w = ones(20); w[4] = -0.5
    push!(out, "negative_weight_20" => (rigid(Rgt, t0), KC1, KC2, VgtkCore.Correspondences(u1, u2; weights = w)))
    w = ones(20); w[4] = NaN
    push!(out, "nan_weight_20" => (rigid(Rgt, t0), KC1, KC2, VgtkCore.Correspondences(u1, u2; weights = w)))
    push!(out, "too_few_4" => (rigid(Rgt, t0), KC1, KC2, VgtkCore.correspondences(u1[1:4], u2[1:4])))
    seed = rigid(RotMatrix(RotXYZ(0.01, 0.02, 0.0)), SVector(1.0, 0.0, 0.0))   # units_boundary_tests: a physical seed
    K = SMatrix{3, 3, Float64}(500, 0, 0, 0, 500, 0, 320, 240, 1)
    push!(out, "physical_seed_10" => (VgtkCore.with_unit(seed, VgtkCore.m), K, K,
        [SVector(100.0 + i, 100.0 + 2i) => SVector(110.0 + i, 100.0 + 2i) for i in 1:10]))
    return out
end
