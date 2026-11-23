# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The contaminated scenes of every find_* class and the too-few sets (scenes_testsetups.jl's TooFewData, FindScenes and
# RadialCameraScene, each of which includes this file and exports its own names, and the golden records of
# ci/golden/fixtures/est_{conics_ransac,epipolar,pose}.jl), built from the shared synthetic scenes
# (`synthetic_scenes()`, the Test extension). Data constructors only — no estimator is called; include-once safe
# (functions, and constants of immutable values that a repeated include rebinds to an identical value).
using Random: rand, randn
using StaticArrays: SA, SVector, SMatrix
using StructArrays: StructArray
using StableRNGs: StableRNG
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: Point2, correspondences, CameraModel, Vec3, RadialHomography, HomographyMat, radial_transfer
import VisualGeometryToolkitEstimation as VgtkEstimation

# --- TooFewData: `n` plane points on a parabola, and their correspondences under a pure shift ------------------------
few_points(n) = [Point2(100.0 + 10i, 50.0 + 7i^2) for i in 1:n]
few_correspondences(n) = correspondences(few_points(n), few_points(n) .+ Ref(Point2(1.0, 2.0)))

# --- FindScenes: a homography's pairs, a stereo pair's pixels, a PnP set, a radial homography's pairs, a hand-eye set --
# the homography's pairs and the number of inliers
function h_scene()
    (; homography_pairs, H_PX, gaussian_source) = VgtkEstimation.synthetic_scenes().HomographyScenes
    src, dst = homography_pairs(
        StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3,
        source = gaussian_source, target = rng -> randn(rng, 2) .* 400 .+ 300
    )
    return StructArray(src .=> dst), 120
end
# the two views' pixels and the number of inliers
function stereo_scene(cam; seed = 13)
    u₁, u₂ = VgtkEstimation.synthetic_scenes().EpipolarScenes.stereo_pixel_pairs(
        StableRNG(seed), cam, cam, RotMatrix(RotXYZ(0.05, 0.12, -0.03)), SVector{3, Float64}(0.9, 0.05, 0.1);
        n = 140, σ₂ = 0.4, n_out = 60,
        point = rng -> SA[2.0 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), 4.0 + 3.0 * rand(rng)],
        outlier = rng -> (SA[960 * rand(rng), 720 * rand(rng)], SA[960 * rand(rng), 720 * rand(rng)])
    )
    return u₁, u₂, length(u₁) - 60
end
const PCAM = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
const R_PNP = RotMatrix(RotXYZ(0.15, -0.25, 0.1))
const T_PNP = Vec3{Float64}(0.3, -0.2, 0.6)
# the PnP set, drawn from `rng` (the suite's StableRNG(19) by default)
function pnp_scene(rng = StableRNG(19))
    (; frustum_point) = VgtkEstimation.synthetic_scenes().PoseScenes
    pairs, _ = VgtkEstimation.synthetic_scenes().Contamination.contaminated_pairs(
        rng, (rng, _) -> Pair(frustum_point(rng, PCAM, R_PNP, T_PNP)...), 120; σ = 0.5, n_out = 60,
        outlier = (rng, _) -> first(frustum_point(rng, PCAM, R_PNP, T_PNP)) => SVector(40.0 + 880.0 * rand(rng), 40.0 + 640.0 * rand(rng))
    )
    return StructArray(first.(pairs) .=> last.(pairs))
end
const RW, RH = 1200.0, 900.0
const M_RH = RadialHomography{Float64}(
    HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
    -0.25 / hypot(RW, RH)^2, SVector(RW / 2, RH / 2)
)
# 70 inliers and `n_out` outliers, drawn from `rng` (the suite's StableRNG(17) by default). The truth is an argument (default M_RH), never the constant captured by the inlier closure: a captured constant is
# folded into the compiled transfer, which then rounds some targets one ulp apart from the same model at run time.
function radial_homography_scene(rng = StableRNG(17), M = M_RH; n_out = 30)
    function inlier(rng, _)
        while true
            s = SVector(200 * rand(rng), 150 * rand(rng))
            d = radial_transfer(M, s)
            d === nothing && continue
            (0 < d[1] < RW && 0 < d[2] < RH) && return s => SVector(d[1], d[2])
        end
    end
    return VgtkEstimation.synthetic_scenes().Contamination.contaminated_pairs(
        rng, inlier, 70; σ = 0.3, n_out, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(RW * rand(rng), RH * rand(rng))
    )
end
# the motions A .=> B and the hand-eye truth X
function hand_eye_scene()
    (; robot_world_hand_eye_scene, perturb_pose, marker_pose) = VgtkEstimation.synthetic_scenes().PoseScenes
    rng = StableRNG(29)
    sc = robot_world_hand_eye_scene(rng; N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                    A = (ang_max = π / 3, t_scale = 0.5))
    X, Y, A = only(sc.X), sc.Y, sc.A
    B = [perturb_pose(rng, marker_pose(Y, A[i], X); σ_R = 0.001, σ_t = 0.001) for i in eachindex(A)]
    return A .=> B, X
end

# --- RadialCameraScene: a division-lens board seen from one pose, its pixel pairs with 20 outliers, the distortion centre
const RC_CENTER = SVector(2000.0, 1500.0)
# drawn from `rng` (the suite's StableRNG(19) by default)
function radial_camera_pairs(rng = StableRNG(19))
    (; DivisionBoards, Contamination) = VgtkEstimation.synthetic_scenes()
    cam = DivisionBoards.board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f = 900.0, ks = (-2.0e-7,), center = RC_CENTER)
    return Contamination.contaminated_pairs(
        rng, DivisionBoards.board_pixels(cam; grid = -100.0:25.0:100.0); σ = 0.2, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(4000 * rand(rng), 3000 * rand(rng))
    )
end
