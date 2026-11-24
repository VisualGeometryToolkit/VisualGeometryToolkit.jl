# Fixtures of the conic family (circle, ellipse) and of the RANSAC core (ransac, ransac!, locally_optimize in its
# model and FixedModels forms, locally_optimize_rescored!, fit_hypothesis, and the minimal solves solve(problem, idx)
# of every RANSAC problem type): the scenes of VisualGeometryToolkitEstimation's suite — scenes_testsetups.jl's
# ConicScenes, TooFewData, FindScenes (h_scene); ransac_circle/ellipse_tests, ellipse_fitting_tests, triggs_tests,
# core_defaults_tests' conic draws; ransac_tests' make_line_data; line_fitting_tests' contaminated_line;
# inhom_line_tests' MSAC scene; ransac_homography_tests' make_homography_data, PROSAC scores and thin incumbent;
# ransac_fundmat_tests' fundmat_scene — built from the shared synthetic scenes and data constructors only. The solve
# fixtures are each problem type's data and a fixed list of minimal samples (StableRNG(1)).
using Test, StableRNGs, StaticArrays, StructArrays, Random, LinearAlgebra
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation

const _CR = VgtkEstimation.synthetic_scenes()
const _CR_H = _CR.HomographyScenes
const _CR_E = _CR.EpipolarScenes
const _CR_P = _CR.PoseScenes
const _CR_C = _CR.Contamination
const _CR_D = _CR.DivisionBoards
const _CR_R = _CR.RadialScenes

Base.include(@__MODULE__, joinpath(@__DIR__, "common.jl"))   # the find_* scenes and too-few sets

# --- ConicScenes ---------------------------------------------------------------------------------------------------
cr_collinear() = [VgtkCore.Point2(Float64(i), 2.0 * i + 1) for i in 1:10]

function cr_circle_scenes()
    rng46 = StableRNG(46)   # triggs_tests: 60 points of a circle, then a straight run
    pts46 = [VgtkCore.Point2{Float64}(100 + 50cos(t) + 0.3randn(rng46), 80 + 50sin(t) + 0.3randn(rng46)) for t in range(0, 2π, length = 61)[1:60]]
    append!(pts46, [VgtkCore.Point2{Float64}(10.0i, 3.0i) for i in 1:20])
    rng51 = StableRNG(51)
    C, R = SVector(200.0, 150.0), 80.0
    pts51 = [VgtkCore.Point2{Float64}(C + R * SVector(cos(t), sin(t)) + 0.3 * randn(rng51, SVector{2})) for t in 2π .* rand(rng51, 200)]
    return [
        "outl40_1000" => first(circle_scene(Xoshiro(32), 1000)),
        "outl30_300" => first(circle_scene(Xoshiro(33), 300; outl = 0.3)),
        "outl40_600" => first(circle_scene(Xoshiro(5), 600)),
        "outl20_200" => first(circle_scene(Xoshiro(3), 200; outl = 0.2)),
        "run_60+20" => pts46,
        "clean_noisy0.3_200" => pts51,
    ]
end
function cr_ellipse_scenes()
    return [
        "outl40_1000" => first(ellipse_scene(Xoshiro(32), 1000)),
        "outl30_300" => first(ellipse_scene(Xoshiro(34), 300; outl = 0.3)),
        "outl40_600" => first(ellipse_scene(Xoshiro(6), 600)),
    ]
end
# The refine fixtures from a given guess: name => (guess, points).
function cr_circle_refine_fixtures()
    rng51 = StableRNG(51)
    C, R = SVector(200.0, 150.0), 80.0
    pts51 = [VgtkCore.Point2{Float64}(C + R * SVector(cos(t), sin(t)) + 0.3 * randn(rng51, SVector{2})) for t in 2π .* rand(rng51, 200)]
    return [
        "perturbed_outl40_800" => (VgtkCore.Circle(VgtkCore.Point2(C₀ + SVector(2.0, -1.5)), R₀ + 3.0), first(circle_scene(Xoshiro(41), 800))),
        "truth_clean_200" => (VgtkCore.Circle(VgtkCore.Point2(C₀), R₀), first(circle_scene(Xoshiro(43), 200; σ = 0.0, outl = 0.0))),
        "perturbed_outl30_300" => (VgtkCore.Circle(VgtkCore.Point2(C₀ + SVector(1.0, -1.0)), R₀ + 1.0), first(circle_scene(Xoshiro(33), 300; outl = 0.3))),
        "perturbed_noisy0.3_200" => (VgtkCore.Circle(VgtkCore.Point2(C + SVector(1.0, -0.8)), R + 1.2), pts51),
        "too_few_2" => (VgtkCore.Circle(VgtkCore.Point2(0.0, 0.0), 1.0), few_points(2)),
        "too_few_1" => (VgtkCore.Circle(VgtkCore.Point2(0.0, 0.0), 1.0), [VgtkCore.Point2(1.0, 0.0)]),
    ]
end
# ellipse_fitting_tests' points on an ellipse: `ellipse_boundary_points`, VisualGeometryToolkitCore's test fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitCore", "test", "fixtures", "ellipse_points.jl"))
const CR_e0 = VgtkCore.Ellipse(VgtkCore.Point2(120.0, 80.0), 30.0, 12.0, 0.6)
function cr_ellipse_refine_fixtures()
    p30 = Point2{Float64}.(ellipse_boundary_points(StableRNG(1), CR_e0, 30))
    p60 = Point2{Float64}.(ellipse_boundary_points(StableRNG(1), CR_e0, 60; σ = 0.5))
    p60arc = Point2{Float64}.(ellipse_boundary_points(StableRNG(1), CR_e0, 60; σ = 0.5, arc = π))
    return [
        "perturbed_outl40_1000" => (VgtkCore.Ellipse(VgtkCore.Point2(515.0, 399.0), 303.0, 177.0, 0.62), first(ellipse_scene(Xoshiro(42), 1000))),
        "truth_clean_200" => (E₀, first(ellipse_scene(Xoshiro(44), 200; σ = 0.0, outl = 0.0))),
        "e0_exact_30" => (CR_e0, p30),
        "direct_exact_30" => (VgtkCore.solve(VgtkCore.EstimationProblem(VgtkCore.Ellipse, p30), VgtkCore.DirectEllipse()).u, p30),
        "direct_noisy_60" => (VgtkCore.solve(VgtkCore.EstimationProblem(VgtkCore.Ellipse, p60), VgtkCore.DirectEllipse()).u, p60),
        "direct_noisy_arc_60" => (VgtkCore.solve(VgtkCore.EstimationProblem(VgtkCore.Ellipse, p60arc), VgtkCore.DirectEllipse()).u, p60arc),
        "e0_too_few_4" => (CR_e0, Point2{Float64}.(ellipse_boundary_points(StableRNG(1), CR_e0, 4))),
        "too_few_4" => (VgtkCore.Ellipse(VgtkCore.Point2(0.0, 0.0), 2.0, 1.0, 0.0), few_points(4)),
    ]
end

# --- the RANSAC core's problems' data -----------------------------------------------------------------------------
function cr_make_line_data(; n_inliers = 100, n_outliers = 50, noise = 0.1, seed = 42)   # ransac_tests
    rng = StableRNG(seed)
    inliers = VgtkCore.Point2{Float64}[]
    for _ in 1:n_inliers
        x = 10 * rand(rng) - 5
        y = 2x + 1 + noise * randn(rng)
        push!(inliers, SA[x, y])
    end
    outliers = VgtkCore.Point2{Float64}[]
    for _ in 1:n_outliers
        push!(outliers, SA[30 * rand(rng) - 10, 30 * rand(rng) - 10])
    end
    return vcat(inliers, outliers)
end
# Contamination.contaminated_line's points and seed: 57 on y = 0 (σ = 0.01), 3 outliers at y = 0.5.
function cr_contaminated_line(rng)
    pts = [VgtkCore.Point2{Float64}(Float64(i), 0.01 * randn(rng)) for i in 0:56]
    append!(pts, [VgtkCore.Point2{Float64}(Float64(x), 0.5) for x in (10, 30, 50)])
    return pts, VgtkCore.Line2D{Float64}(SVector(0.0, 1.0, -0.025))
end
function cr_inhom_line_points()   # inhom_line_tests: MSAC end-to-end
    rng = StableRNG(999)
    a_true, b_true = 5.0, 2.0
    pts_in = [SA[x, a_true + b_true * x + 0.5 * randn(rng)] for x in range(-10, 10, length = 50)]
    pts_out = [SA[20.0 * rand(rng) - 10.0, 50.0 * rand(rng) - 25.0] for _ in 1:20]
    return vcat(pts_in, pts_out)
end
function cr_inhom_line_clean()   # line_fitting_tests: the failed refit
    rng = StableRNG(59)
    return [SVector(Float64(i), 1.0 + 0.5i + 0.01 * randn(rng)) for i in 1:40]
end
cr_make_homography_data(; n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42) =
    _CR_H.homography_pairs(StableRNG(seed), _CR_H.H_TRUE; n_in = n_inliers, n_out = n_outliers, σ = noise)
function cr_homography_scored()
    src, dst = cr_make_homography_data()
    rng = StableRNG(99)
    scores = [i <= 100 ? 0.8 + 0.2 * rand(rng) : 0.1 * rand(rng) for i in eachindex(src)]
    return VgtkCore.Correspondences(src, dst; score = scores)
end
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "thin_homography.jl"))
cr_homography_thin() = thin_homography_correspondences()   # termination_record / ransac_homography: 5 exact pairs and 12 random ones
function cr_fundmat_data()
    fs = _CR_E.fundmat_scene(StableRNG(42); n_inliers = 100, n_outliers = 30, σ = 0.5)
    return VgtkCore.correspondences(fs.source, fs.target)
end

# --- the solve fixtures: per problem type, its data (the problem's constructor arguments) ---------------------------
cr_essential_pixels() = (sc = stereo_scene(_CR_E.ECAM; seed = 14); VgtkCore.correspondences(sc[1], sc[2]))
function cr_bearing_pose()   # central bearings of the PnP set's points, through the pinhole
    cs = pnp_scene()
    fM = [VgtkCore.backproject_with_jacobian(PCAM, u) for u in VgtkCore.targets(cs)]
    return VgtkCore.withattrs(VgtkCore.Correspondences(collect(VgtkCore.sources(cs)), first.(fM)); M = last.(fM))
end
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitEstimation", "test", "fixtures", "rotation_scene.jl"))   # `FISHEYE`, `rotation_scene`
cr_rotation_pixels() = first(rotation_scene(Xoshiro(35), RotMatrix(RotXYZ(0.1, -0.2, 0.3)), 120))   # outlier-free
cr_hand_eye_motions() = first(hand_eye_scene())
cr_radial_homography_pairs() = first(_CR_R.radial_homography_pairs(StableRNG(17), _CR_R.radial_homography_truth(-0.25); n_in = 70, n_out = 30, σ = 0.3))
cr_radial_board_pairs() = _CR_D.radial_board(StableRNG(7); σ = 0.2)[2]

# n minimal samples of size m over 1:N (StableRNG(1)).
cr_samples(N, m; n = 50) = (rng = StableRNG(1); [randperm(rng, N)[1:m] for _ in 1:n])
