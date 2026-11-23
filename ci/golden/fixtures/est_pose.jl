# Fixtures of the pose family (the absolute pose in its pixel and bearing forms, hand-eye, the radial homography and
# the radial camera): the scenes of VisualGeometryToolkitEstimation's suite — scenes_testsetups.jl's FindScenes
# (pnp_scene, hand_eye_scene, radial_homography_scene), RadialCameraScene and TooFewData; pnp_lens_tests' lens
# boards; units_boundary_tests' PnP and hand-eye sets (plain, metres, millimetres, an extent); refine_absolute_pose_tests'
# PixelPoseScenes (pinhole, a fisheye past 90°, bearings with Jacobians, ray origins, contaminated rays);
# zero_weight_rows_tests; hand_eye_tests' _synth_dataset; radial_homography_tests' and radial_camera_tests' draws;
# refine_contract_tests' too-few classes; ransac_no_model_tests' degenerate sets — built from the shared synthetic
# scenes and data constructors only.
using Test, StableRNGs, StaticArrays, StructArrays, Random, LinearAlgebra
using Rotations: RotMatrix, RotXYZ, RotX, RotZ, RotationVec
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _PO = VgtkEstimation.synthetic_scenes()
const _PO_P = _PO.PoseScenes
const _PO_C = _PO.Contamination
const _PO_D = _PO.DivisionBoards
const _PO_R = _PO.RadialScenes
const po_m = VgtkCore.m
const po_mm = VgtkCore.mm

Base.include(@__MODULE__, joinpath(@__DIR__, "common.jl"))   # the find_* scenes, too-few sets and pixel-pose scenes

# =================================================================================================================
# The absolute pose, pixel form: name => (camera, X => u pairs[, extent])
# =================================================================================================================
function po_units_pnp()   # units_boundary_tests: 80 frustum points, every fifth pixel moved (20 % outliers)
    rng = StableRNG(9)
    obs = [Pair(_PO_P.frustum_point(rng, PCAM, R_PNP, T_PNP)...) for _ in 1:80]
    obs = [(k % 5 == 0 ? first(p) => last(p) + SVector(40.0, -25.0) : p) for (k, p) in enumerate(obs)]
    return first.(obs), last.(obs)
end
po_lens_camera(lens) = VgtkCore.CameraModel(; focal = 950.0, pp = (2120.0, 1416.0), lens)
function po_lens_scenes()   # pnp_lens_tests: a board through a division fisheye and an OpenCV polynomial lens
    cams = ("division2" => (po_lens_camera(VgtkCore.Division(-0.3, -0.08)), RotMatrix(RotXYZ(0.35, -0.35, 0.2)), VgtkCore.Vec3{Float64}(850.0, 250.0, 350.0)),
            "opencv5" => (po_lens_camera(VgtkCore.Lens(VgtkCore.Perspective(), VgtkCore.Additive(VgtkCore.Radial((-0.25, 0.06, -0.01)),
                                                                                    VgtkCore.Tangential(0.001, -0.0005)))),
                          RotMatrix(RotXYZ(0.35, -0.55, 0.2)), VgtkCore.Vec3{Float64}(280.0, -60.0, 420.0)))
    return [name => (cam, (s = _PO_P.lens_board_scene(StableRNG(21), cam, R, t); s[1] .=> s[2])) for (name, (cam, R, t)) in cams]
end
function po_abspose_scenes()
    X, u = po_units_pnp()
    s = collect(range(1.0, 0.1; length = length(X)))
    return [
        "pnp_120+60" => (PCAM, pnp_scene()),
        "units_80_pairs" => (PCAM, X .=> u),
        "units_80_metres" => (PCAM, [x .* po_m for x in X] .=> u),
        "units_80_millimetres" => (PCAM, [(1000 .* x) .* po_mm for x in X] .=> u),
        "units_80_plain_mm" => (PCAM, [1000 .* x for x in X] .=> u),
        "units_80_scored" => (PCAM, VgtkCore.Correspondences(X, u; score = s)),
        "units_80_scored_metres" => (PCAM, VgtkCore.Correspondences([x .* po_m for x in X], u; score = s)),
        po_lens_scenes()...,
    ]
end
po_abspose_extent() = VgtkCore.HyperRectangle(VgtkCore.Vec2(-2.0, -2.0), VgtkCore.Vec2(4.0, 4.0))
po_abspose_extent_m() = VgtkCore.HyperRectangle(VgtkCore.Vec2(-2.0, -2.0) .* po_m, VgtkCore.Vec2(4.0, 4.0) .* po_m)
function po_abspose_degenerate()   # ransac_no_model_tests: three collinear world points
    X = [SVector(0.1i, 0.0, 2.0) for i in 1:3]
    u = [VgtkCore.project(PCAM, x) for x in X]
    return ["collinear_3" => (PCAM, X .=> u), "collinear_3_metres" => (PCAM, [x .* po_m for x in X] .=> u),
            "too_few_2" => (PCAM, [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]] .=> [SA[400.0, 300.0], SA[500.0, 350.0]]),
            "too_few_2_metres" => (PCAM, [SA[0.1, 0.2, 2.0] .* po_m, SA[-0.3, 0.1, 2.5] .* po_m] .=> [SA[400.0, 300.0], SA[500.0, 350.0]])]
end

# --- refine_absolute_pose_tests' PixelPoseScenes ------------------------------------------------------------------
const PO_PIN = VgtkCore.CameraModel(; focal = 800.0, pp = (640.0, 480.0))
const PO_FISH = VgtkCore.CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = VgtkCore.Lens(VgtkCore.Equidistant()))

# The pixel-form refine fixtures: name => (guess, camera, pairs).
function po_pixel_refine_fixtures()
    E = true_pose()
    out = Pair{String, Any}[]
    for (lbl, cam, θmax) in (("pinhole", PO_PIN, 0.5), ("fisheye110", PO_FISH, deg2rad(110)))
        cs = pixel_pairs(cam, E, scene_points(E, θmax))
        push!(out, "$(lbl)_perturbed" => (perturbed(E), cam, cs))
        push!(out, "$(lbl)_near_noisy" => (E, cam, [X => u + SVector(0.3 * sin(7i), 0.3 * cos(11i)) for (i, (X, u)) in enumerate(cs)]))
    end
    cs = pixel_pairs(PO_PIN, E, scene_points(E, 0.6; n = 12))
    push!(out, "pinhole12_labeled" => (VgtkCore.relabel(VgtkCore.WorldFrame, VgtkCore.CameraFrame, perturbed(E)), PO_PIN, cs))
    push!(out, "pinhole12_metres" => (VgtkCore.EuclideanMap(perturbed(E).R, perturbed(E).t .* po_m), PO_PIN, [X .* po_m => u for (X, u) in cs]))
    push!(out, "pinhole12_too_few_2" => (E, PO_PIN, cs[1:2]))
    push!(out, "too_few_2_metres" => (VgtkCore.with_unit(VgtkCore.Extrinsics3(one(RotMatrix{3, Float64}), VgtkCore.Vec3{Float64}(0.0, 0.0, 0.0)), po_m),
        PCAM, [SA[0.1, 0.2, 2.0] .* po_m, SA[-0.3, 0.1, 2.5] .* po_m] .=> [SA[400.0, 300.0], SA[500.0, 350.0]]))
    return out
end

# The bearing-form refine fixtures: name => (guess, bearing correspondences).
function po_bearing_refine_fixtures()
    E = true_pose()
    out = Pair{String, Any}[]
    # pinhole, pixel-whitened (with M), noisy, and without M
    Xs = scene_points(E, 0.5)
    us = [SVector{2, Float64}(VgtkCore.project(PO_PIN, E.R * X + E.t)) for X in Xs]
    fM = [VgtkCore.backproject_with_jacobian(PO_PIN, u) for u in us]
    push!(out, "pinhole_M" => (perturbed(E), VgtkCore.withattrs(VgtkCore.Correspondences(Xs, first.(fM)); M = last.(fM))))
    noise = [SVector(0.3 * sin(7i), 0.3 * cos(11i)) for i in eachindex(us)]
    fMn = [VgtkCore.backproject_with_jacobian(PO_PIN, u + δ) for (u, δ) in zip(us, noise)]
    push!(out, "pinhole_noisy_M" => (E, VgtkCore.withattrs(VgtkCore.Correspondences(Xs, first.(fMn)); M = last.(fMn))))
    push!(out, "pinhole_noisy_angle" => (E, VgtkCore.Correspondences(Xs, first.(fMn))))
    # a fisheye past 90° with a ray behind the camera; the same rays through a fisheye camera's Jacobians
    Xs = scene_points(E, deg2rad(110))
    fs = [normalize(E.R * X + E.t) for X in Xs]
    Xb = E.R' * (SVector(0.3, 0.1, -2.5) - E.t)
    push!(Xs, Xb); push!(fs, normalize(E.R * Xb + E.t))
    push!(out, "fisheye110_behind" => (perturbed(E), VgtkCore.Correspondences(Xs, fs)))
    keep = [i for i in 1:(length(Xs) - 1) if all(isfinite, VgtkCore.project(PO_FISH, E.R * Xs[i] + E.t))]
    fM = [VgtkCore.backproject_with_jacobian(PO_FISH, SVector{2, Float64}(VgtkCore.project(PO_FISH, E.R * Xs[i] + E.t))) for i in keep]
    push!(out, "fisheye_camera_M" => (perturbed(E), VgtkCore.withattrs(VgtkCore.Correspondences(Xs[keep], first.(fM)); M = last.(fM))))
    # a generalized camera (ray origins), and its central form on the same rays
    Xs = scene_points(E, deg2rad(100))
    os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]
    fs = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    push!(out, "generalized_origins" => (perturbed(E), VgtkCore.withattrs(VgtkCore.Correspondences(Xs, fs); origins = os)))
    push!(out, "generalized_as_central" => (perturbed(E), VgtkCore.Correspondences(Xs, fs)))
    push!(out, "origin_on_point" => (E, VgtkCore.withattrs(VgtkCore.Correspondences(Xs, fs);
                                                           origins = [i == 1 ? E.R * Xs[1] + E.t : o for (i, o) in enumerate(os)])))
    # contaminated rays with origins and 800 px Jacobians, from a near guess
    rng = Xoshiro(3)
    Bof(f) = (b = VgtkCore.s2_tangent_basis(f); hcat(SVector{3}(b[1]), SVector{3}(b[2])))
    fs = [(f = normalize(E.R * X + E.t - o); normalize(f + Bof(f) * (0.3 / 800 * SVector(randn(rng), randn(rng))))) for (X, o) in zip(Xs, os)]
    bad = randperm(rng, length(Xs))[1:10]
    for i in bad
        fs[i] = normalize(SVector(randn(rng), randn(rng), randn(rng)))
    end
    Ms = [Bof(f) / 800 for f in fs]
    guess = VgtkCore.EuclideanMap(RotMatrix{3, Float64}(VgtkCore.exp_so3(SVector(0.002, -0.0015, 0.002)) * E.R), E.t + SVector(0.004, -0.003, 0.003))
    push!(out, "contaminated_rays" => (guess, VgtkCore.withattrs(VgtkCore.Correspondences(Xs, fs); origins = os, M = Ms)))
    # refusals, frames, units
    Xs = scene_points(E, 0.6; n = 12)
    fs = [normalize(E.R * X + E.t) for X in Xs]
    push!(out, "chart_points" => (E, VgtkCore.Correspondences(Xs, [f / f[3] for f in fs])))
    Mbad = [hcat(normalize(SVector(1.0, 0.0, 0.0) - f[1] * f) + f, normalize(SVector(0.0, 1.0, 0.0) - f[2] * f)) for f in fs]
    push!(out, "nontangent_M" => (E, VgtkCore.withattrs(VgtkCore.Correspondences(Xs, fs); M = Mbad)))
    push!(out, "too_few_2" => (E, VgtkCore.Correspondences(Xs[1:2], fs[1:2])))
    push!(out, "labeled_12" => (VgtkCore.relabel(VgtkCore.WorldFrame, VgtkCore.CameraFrame, perturbed(E)), VgtkCore.Correspondences(Xs, fs)))
    os = [SVector(0.01, 0.0, 0.0) for _ in Xs]
    fo = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    push!(out, "metres_origins_12" => (VgtkCore.EuclideanMap(perturbed(E).R, perturbed(E).t .* po_m),
        VgtkCore.withattrs(VgtkCore.Correspondences([X .* po_m for X in Xs], fo); origins = [o .* po_m for o in os])))
    # zero_weight_rows_tests: a point at the camera centre at caller weight 0
    E₀ = VgtkCore.EuclideanMap(RotMatrix(RotXYZ(0.33, -0.18, 0.12)), SVector(0.13, -0.08, 2.05))
    Xs = [E.R' * (SVector(0.3sin(k), 0.3cos(2k), 2.0 + 0.5sin(3k)) - E.t) for k in 1:30]
    fs = [normalize(E.R * X + E.t) for X in Xs]
    Xc = -(E.R' * E.t)
    w = ones(31); w[31] = 0.0
    push!(out, "centre_point_weight0" => (E₀, VgtkCore.withattrs(VgtkCore.Correspondences(vcat(Xs, [Xc]), vcat(fs, [SVector(0.0, 0.0, 1.0)])); weights = w)))
    # refine_contract / ransac_no_model too few: plain and metres
    X2 = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    pose = VgtkCore.Extrinsics3(one(RotMatrix{3, Float64}), VgtkCore.Vec3{Float64}(0.0, 0.0, 0.0))
    push!(out, "contract_too_few_2" => (pose, VgtkCore.Correspondences(X2, [normalize(x) for x in X2])))
    push!(out, "contract_too_few_2_metres" => (VgtkCore.with_unit(pose, po_m), VgtkCore.Correspondences([x .* po_m for x in X2], [normalize(x) for x in X2])))
    return out
end

# =================================================================================================================
# Hand-eye: name => the A => B motions
# =================================================================================================================
function po_synth_dataset(rng; N = 60, σ_R = 0.001, σ_t = 0.001, n_outliers = 0)   # hand_eye_tests' _synth_dataset
    sc = _PO_P.robot_world_hand_eye_scene(rng; N, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π / 3, t_scale = 0.5))
    Xgt, Ygt, A = only(sc.X), sc.Y, sc.A
    B = [_PO_P.marker_pose(Ygt, A[i], Xgt) for i in 1:N]
    A_obs = [_PO_P.perturb_pose(rng, A[i]; σ_R, σ_t) for i in 1:N]
    B_obs = [_PO_P.perturb_pose(rng, B[i]; σ_R, σ_t) for i in 1:N]
    for k in Random.randperm(rng, N)[1:n_outliers]
        B_obs[k] = _PO_P.rand_pose(rng; ang_max = π, t_scale = 5.0)
    end
    return A_obs .=> B_obs
end
function po_slow_rotation_set()   # hand_eye_tests: every consecutive motion turns 3°
    r7 = StableRNG(7)
    sc = _PO_P.robot_world_hand_eye_scene(r7; N = 1, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π, t_scale = 0.5))
    Xgt, Ygt, A = only(sc.X), sc.Y, sc.A
    for _ in 2:30
        ax = randn(r7, 3); ax ./= norm(ax)
        Δ = VgtkEstimation.Rigid3(RotMatrix{3, Float64}(RotMatrix(RotationVec((deg2rad(3) .* ax)...))), SVector{3, Float64}((0.2 .* randn(r7, 3))...))
        push!(A, A[end] ∘ Δ)
    end
    return A .=> [_PO_P.marker_pose(Ygt, a, Xgt) for a in A]
end
function po_units_hand_eye()   # units_boundary_tests' hand_eye_data
    rng = StableRNG(11)
    sc = _PO_P.robot_world_hand_eye_scene(rng; N = 40, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π / 3, t_scale = 0.5))
    A = [_PO_P.perturb_pose(rng, a; σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    B = [_PO_P.perturb_pose(rng, _PO_P.marker_pose(sc.Y, a, only(sc.X)); σ_R = 1.0e-3, σ_t = 1.0e-3) for a in sc.A]
    return A, B
end
po_phys(E, u = po_m) = VgtkCore.with_unit(E, u)
function po_hand_eye_scenes()
    rng = StableRNG(20260520)
    noiseless = po_synth_dataset(rng; N = 30, σ_R = 0.0, σ_t = 0.0)
    moderate = po_synth_dataset(rng; N = 80, σ_R = 1.0e-3, σ_t = 1.0e-3)
    outliers = po_synth_dataset(rng; N = 120, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 15)
    allpairs = po_synth_dataset(rng; N = 20, σ_R = 1.0e-3, σ_t = 1.0e-3)
    mmset = po_synth_dataset(StableRNG(252); N = 80, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 8)
    tomm(p) = VgtkEstimation.Rigid3(VgtkCore.rotation(p), SVector{3, Float64}((1000 .* VgtkCore.translation(p))...))
    A, B = po_units_hand_eye()
    Amm = po_phys.([VgtkCore.EuclideanMap(VgtkCore.rotation(a), 1000 .* VgtkCore.translation(a)) for a in A], Ref(po_mm))
    Bmm = po_phys.([VgtkCore.EuclideanMap(VgtkCore.rotation(b), 1000 .* VgtkCore.translation(b)) for b in B], Ref(po_mm))
    return [
        "find_scene_30" => first(hand_eye_scene()),
        "noiseless_30" => noiseless,
        "moderate_80" => moderate,
        "outliers_120+15" => outliers,
        "allpairs_20" => allpairs,
        "mixed_80+8_metres" => mmset,
        "mixed_80+8_plain_mm" => tomm.(first.(mmset)) .=> tomm.(last.(mmset)),
        "units_40" => A .=> B,
        "units_40_metres" => po_phys.(A) .=> po_phys.(B),
        "units_40_millimetres" => Amm .=> Bmm,
    ]
end
po_hand_eye_parallel() = [VgtkCore.EuclideanMap(RotMatrix(RotZ(0.35i)), SVector(0.1i, 0.05i^2, 0.2)) =>
                          VgtkCore.EuclideanMap(RotMatrix(RotX(0.35i)), SVector(0.2, 0.1i, -0.03i)) for i in 1:8]
function po_hand_eye_contract()   # refine_contract_tests: three motions
    rigid(R, t) = VgtkCore.EuclideanMap(RotMatrix{3, Float64}(R), SVector{3, Float64}(t))
    A = [rigid(RotXYZ(0.3i, -0.2i, 0.1i^2), SVector(0.1i, -0.05i, 0.02i)) for i in 1:3]
    Xhe = rigid(RotXYZ(0.1, 0.2, -0.3), SVector(0.05, 0.0, 0.1))
    return Xhe, [Ai => inv(Xhe) ∘ Ai ∘ Xhe for Ai in A]
end

# =================================================================================================================
# The radial homography: name => (pairs, center)
# =================================================================================================================
function po_radial_homography_scenes()
    c = (RW / 2, RH / 2)
    m_gt = _PO_R.radial_homography_truth(-0.25)
    rng = StableRNG(1234)   # radial_homography_tests' draws, in its order
    clean20 = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 20, n_out = 0))
    exact70a = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 70, n_out = 30))
    exact70b = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 70, n_out = 30))
    noisy120 = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 120, n_out = 20, σ = 0.4))
    noisy40 = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 40, n_out = 0, σ = 0.3))
    ext60 = first(_PO_R.radial_homography_pairs(rng, m_gt; n_in = 60, n_out = 15))
    four80 = first(_PO_R.radial_homography_pairs(StableRNG(77), m_gt; n_in = 80, n_out = 25, σ = 0.3))
    return [
        "find_scene_70+30" => (first(radial_homography_scene()), c),
        "clean_20" => (clean20, c), "exact_70+30_a" => (exact70a, c), "exact_70+30_b" => (exact70b, c),
        "noisy_120+20" => (noisy120, c), "noisy_40" => (noisy40, c), "extent_60+15" => (ext60, c), "four_80+25" => (four80, c),
    ]
end

# =================================================================================================================
# The radial camera: name => (pairs, center)
# =================================================================================================================
po_radial_camera_scenes() = [
    "find_scene_81+20" => (first(radial_camera_pairs()), RC_CENTER),
    "board_exact_seed7" => (_PO_D.radial_board(StableRNG(7))[2], _PO_D.RADIAL_BOARD_CENTER),
    "board_noisy0.3_seed3" => (_PO_D.radial_board(StableRNG(3); σ = 0.3)[2], _PO_D.RADIAL_BOARD_CENTER),
    "board_noisy0.2_seed11" => (_PO_D.radial_board(StableRNG(11); σ = 0.2)[2], _PO_D.RADIAL_BOARD_CENTER),
]
