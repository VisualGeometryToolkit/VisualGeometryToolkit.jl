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
const EP_PCAM = VgtkCore.CameraModel(; focal = 800.0, pp = (480.0, 360.0))           # FindScenes' PCAM
const EP_ECAM = _EP_E.ECAM
const EP_CAM800 = VgtkCore.CameraModel(800.0 * VgtkCore.px, SA[960.0 * VgtkCore.px, 540.0 * VgtkCore.px])   # ransac_essential_tests
const EP_FISHEYE = VgtkCore.CameraModel(300.0 * VgtkCore.px, SA[600.0 * VgtkCore.px, 600.0 * VgtkCore.px];
                                        lens = VgtkCore.KannalaBrandt(-0.02))   # RotationScene's FISHEYE

# --- TooFewData ----------------------------------------------------------------------------------------------------
ep_few_points(n) = [VgtkCore.Point2(100.0 + 10i, 50.0 + 7i^2) for i in 1:n]
ep_few_correspondences(n) = VgtkCore.correspondences(ep_few_points(n), ep_few_points(n) .+ Ref(VgtkCore.Point2(1.0, 2.0)))

# --- FindScenes.stereo_scene -----------------------------------------------------------------------------------------
function ep_stereo_scene(cam; seed = 13)
    u₁, u₂ = _EP_E.stereo_pixel_pairs(
        StableRNG(seed), cam, cam, RotMatrix(RotXYZ(0.05, 0.12, -0.03)), SVector{3, Float64}(0.9, 0.05, 0.1);
        n = 140, σ₂ = 0.4, n_out = 60,
        point = rng -> SA[2.0 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), 4.0 + 3.0 * rand(rng)],
        outlier = rng -> (SA[960 * rand(rng), 720 * rand(rng)], SA[960 * rand(rng), 720 * rand(rng)])
    )
    return u₁, u₂
end

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
    u₁, u₂ = ep_stereo_scene(EP_PCAM)
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
    v₁, v₂ = ep_stereo_scene(EP_ECAM; seed = 14)
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

# --- RotationScene.rotation_scene ----------------------------------------------------------------------------------
function ep_rotation_scene(rng, R, n; σ = 0.5, outl = 0.0, θmax = deg2rad(100))
    u₁ = SVector{2, Float64}[]; u₂ = SVector{2, Float64}[]
    while length(u₁) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        (acos(clamp(d[3], -1, 1)) < θmax && acos(clamp((R * d)[3], -1, 1)) < θmax) || continue
        a = VgtkCore.project(EP_FISHEYE, d); b = VgtkCore.project(EP_FISHEYE, R * d)
        (all(isfinite, a) && all(isfinite, b)) || continue
        push!(u₁, a + σ * randn(rng, SVector{2, Float64})); push!(u₂, b + σ * randn(rng, SVector{2, Float64}))
    end
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for i in bad
        while true
            c = SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1))
            all(isfinite, VgtkCore.backproject_with_jacobian(EP_FISHEYE, c)[1]) && (u₂[i] = c; break)
        end
    end
    return VgtkCore.Correspondences(u₁, u₂)
end

# The bearing form of a pixel set: unit bearings with their back-projection Jacobians as the M₁/M₂ columns.
function ep_bearings(camera1, camera2, cs)
    fm₁ = map(u -> VgtkCore.backproject_with_jacobian(camera1, u), VgtkCore.sources(cs))
    fm₂ = map(u -> VgtkCore.backproject_with_jacobian(camera2, u), VgtkCore.targets(cs))
    return VgtkCore.withattrs(VgtkCore.Correspondences(first.(fm₁), first.(fm₂)); M₁ = last.(fm₁), M₂ = last.(fm₂))
end

# The rotation scenes: name => (camera1, camera2, pixel correspondences).
ep_rotation_scenes() = [
    "fisheye_120_outl30" => (EP_FISHEYE, EP_FISHEYE, ep_rotation_scene(Xoshiro(35), RotMatrix(RotXYZ(0.1, -0.2, 0.3)), 120; outl = 0.3)),
    "fisheye_300_outl30" => (EP_FISHEYE, EP_FISHEYE, ep_rotation_scene(Xoshiro(8), RotMatrix(RotXYZ(0.2, -0.3, 0.4)), 300; outl = 0.3)),
]

# --- relpose_bundle_tests ------------------------------------------------------------------------------------------
const EP_CAM1 = VgtkCore.CameraModel(; focal = (1.1, 1.05), pp = (0.5, 0.5))
const EP_CAM2 = VgtkCore.CameraModel(; focal = (1.2, 1.15), pp = (0.48, 0.52))
const EP_KC1 = SMatrix{3, 3, Float64, 9}(VgtkCore.calibration_matrix(EP_CAM1))
const EP_KC2 = SMatrix{3, 3, Float64, 9}(VgtkCore.calibration_matrix(EP_CAM2))
const EP_KP1 = SMatrix{3, 3, Float64, 9}(800.0, 0, 0, 0, 800.0, 0, 640, 360, 1)
const EP_KP2 = SMatrix{3, 3, Float64, 9}(820.0, 0, 0, 0, 815.0, 0, 630, 370, 1)

function ep_relpose_scene(rng, N; σ = 0.0)
    Rgt = RotMatrix(RotXYZ(0.1, -0.15, 0.08)); tgt = normalize(SVector(0.7, 0.2, 0.6))
    u1, u2 = _EP_E.stereo_pixel_pairs(rng, EP_CAM1, EP_CAM2, Rgt, tgt; n = N, σ₁ = σ, σ₂ = σ,
                                      point = rng -> VgtkCore.Point3(2randn(rng), 2randn(rng), 5 + 2rand(rng)))
    return Rgt, tgt, u1, u2
end
ep_relpose_seed(rng, Rgt, tgt; δ, s) = (SMatrix{3, 3, Float64, 9}(Rgt * RotXYZ(δ...)),
    VgtkCore.Vec3{Float64}(normalize(tgt + s * SVector(randn(rng), randn(rng), randn(rng)))))

function ep_harness_scene(N; seed = 1, σ = 0.5)
    rng = StableRNG(seed)
    Rgt = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(8.0), normalize(SVector(0.1, 1.0, 0.05))...)))
    tgt = normalize(SVector(-1.0, 0.05, 0.1))
    u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
    while length(u1) < N
        X = SVector(8 * (rand(rng) - 0.5), 6 * (rand(rng) - 0.5), 6 + 6 * rand(rng))
        Y = Rgt * X + tgt
        Y[3] <= 0.5 && continue
        w1 = EP_KP1 * X; w2 = EP_KP2 * Y
        p1 = SVector(w1[1], w1[2]) / w1[3]; p2 = SVector(w2[1], w2[2]) / w2[3]
        (0 <= p1[1] <= 1280 && 0 <= p1[2] <= 720 && 0 <= p2[1] <= 1280 && 0 <= p2[2] <= 720) || continue
        push!(u1, p1 + σ * SVector(randn(rng), randn(rng)))
        push!(u2, p2 + σ * SVector(randn(rng), randn(rng)))
    end
    a = normalize(SVector(randn(rng), randn(rng), randn(rng)))
    R0 = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(1.0), a...))) * Rgt
    p = normalize(cross(tgt, SVector(randn(rng), randn(rng), randn(rng))))
    t0 = normalize(SVector{3, Float64}(RotMatrix(AngleAxis(deg2rad(2.0), p...)) * tgt))
    return R0, t0, u1, u2
end

# The relative-pose bundle fixtures: name => (seed pose, K1, K2, correspondences).
function ep_relpose_fixtures()
    rigid = _EP_P.rigid
    out = Pair{String, Any}[]
    for N in (100, 1000)
        R0, t0, u1, u2 = ep_harness_scene(N)
        push!(out, "harness_$N" => (rigid(R0, t0), EP_KP1, EP_KP2, VgtkCore.correspondences(u1, u2)))
    end
    rng = StableRNG(3); Rgt, tgt, u1, u2 = ep_relpose_scene(rng, 30; σ = 0.0)
    R0, t0 = ep_relpose_seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    push!(out, "exact_30" => (rigid(R0, t0), EP_KC1, EP_KC2, VgtkCore.correspondences(u1, u2)))
    rng = StableRNG(5); Rgt, tgt, u1, u2 = ep_relpose_scene(rng, 40; σ = 0.002)
    R0, t0 = ep_relpose_seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    push!(out, "noisy_40" => (rigid(R0, t0), EP_KC1, EP_KC2, VgtkCore.correspondences(u1, u2)))
    rng = StableRNG(6); Rgt, tgt, u1, u2 = ep_relpose_scene(rng, 30; σ = 0.002)   # a point at infinity, weighted 0 and 1
    d = normalize(SVector(0.1, -0.05, 1.0))
    push!(u1, SVector{2, Float64}(VgtkCore.project(EP_CAM1, d))); push!(u2, SVector{2, Float64}(VgtkCore.project(EP_CAM2, Rgt * d)))
    R0, t0 = ep_relpose_seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    for w_far in (0.0, 1.0)
        push!(out, "infinity_31_wfar$(Int(w_far))" =>
            (rigid(R0, t0), EP_KC1, EP_KC2, VgtkCore.Correspondences(u1, u2; weights = vcat(ones(30), w_far))))
    end
    rng = StableRNG(7); Rgt, tgt, u1, u2 = ep_relpose_scene(rng, 20; σ = 0.002)
    t0 = VgtkCore.Vec3{Float64}(tgt)
    push!(out, "drifted_seed_20" => (rigid(1.001 * SMatrix{3, 3, Float64, 9}(Rgt), t0), EP_KC1, EP_KC2, VgtkCore.correspondences(u1, u2)))
    w = ones(20); w[4] = -0.5
    push!(out, "negative_weight_20" => (rigid(Rgt, t0), EP_KC1, EP_KC2, VgtkCore.Correspondences(u1, u2; weights = w)))
    w = ones(20); w[4] = NaN
    push!(out, "nan_weight_20" => (rigid(Rgt, t0), EP_KC1, EP_KC2, VgtkCore.Correspondences(u1, u2; weights = w)))
    push!(out, "too_few_4" => (rigid(Rgt, t0), EP_KC1, EP_KC2, VgtkCore.correspondences(u1[1:4], u2[1:4])))
    seed = rigid(RotMatrix(RotXYZ(0.01, 0.02, 0.0)), SVector(1.0, 0.0, 0.0))   # units_boundary_tests: a physical seed
    K = SMatrix{3, 3, Float64}(500, 0, 0, 0, 500, 0, 320, 240, 1)
    push!(out, "physical_seed_10" => (VgtkCore.with_unit(seed, VgtkCore.m), K, K,
        [SVector(100.0 + i, 100.0 + 2i) => SVector(110.0 + i, 100.0 + 2i) for i in 1:10]))
    return out
end
