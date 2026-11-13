# =============================================================================
# Shared synthetic scenes — ReTestItems `@testsetup` modules
# =============================================================================
#
# ONE definition of every scene more than one test item draws from: the
# scenes are VisualGeometryToolkitEstimation's `synthetic_scenes()` (its Test extension,
# ext/VisualGeometryToolkitEstimationTestExt.jl), which every stack package's tests
# share. An item names the family it needs (`@testitem "x" setup = [EpipolarScenes]
# begin`) and gets its exports; each setup below is that family, re-exported.
# =============================================================================

@testsetup module EpipolarScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().EpipolarScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:EpipolarScenes])...))
end

@testsetup module HomographyScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().HomographyScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:HomographyScenes])...))
end

@testsetup module PoseScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().PoseScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:PoseScenes])...))
end

@testsetup module DivisionBoards
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().DivisionBoards
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:DivisionBoards])...))
end

@testsetup module P3PScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().P3PScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:P3PScenes])...))
end

@testsetup module Cameras
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().Cameras
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:Cameras])...))
end

@testsetup module Contamination
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().Contamination
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:Contamination])...))
end

# The circle and the ellipse of the RANSAC conic tests (ransac_circle_tests.jl,
# ransac_ellipse_tests.jl, core_defaults_tests.jl, ransac_no_model_tests.jl): n
# points on the curve with isotropic noise σ, then a fraction `outl` replaced,
# half by a straight run (a floor edge), half by uniform clutter.
@testsetup module ConicScenes
using Random: AbstractRNG, randn, rand, randperm
using StaticArrays: SVector
using VisualGeometryToolkitCore: Point2, Circle, Ellipse, boundary_point
export circle_scene, ellipse_scene, C₀, R₀, E₀

const C₀ = SVector(512.3, 401.7)
const R₀ = 300.0
const E₀ = Ellipse(Point2(512.3, 401.7), 300.0, 180.0, 0.6)   # axes ratio 0.6, rotated

function contaminated_boundary(rng::AbstractRNG, curve, n; σ, outl)
    pts = [Point2{Float64}(SVector(boundary_point(curve, t)...) + σ * randn(rng, SVector{2, Float64})) for t in 2π .* rand(rng, n)]
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for (k, i) in enumerate(bad)
        pts[i] = isodd(k) ? Point2{Float64}(250.0 + 500 * rand(rng), 760.0 + randn(rng)) :
            Point2{Float64}(1024 * rand(rng), 800 * rand(rng))
    end
    return pts, bad
end
circle_scene(rng, n; σ = 0.5, outl = 0.4) = contaminated_boundary(rng, Circle(Point2(C₀), R₀), n; σ, outl)
ellipse_scene(rng, n; σ = 0.5, outl = 0.4) = contaminated_boundary(rng, E₀, n; σ, outl)
end

# The radial camera's contaminated board scene (core_defaults_tests.jl,
# contract_conformance_tests.jl): a division-lens board seen from one pose, its
# pixel pairs with 20 outliers, and the distortion centre the problems take.
@testsetup module RadialCameraScene
using Test: Test                                       # loads the extension
using StaticArrays: SVector
using Rotations: RotXYZ
using StableRNGs: StableRNG
using VisualGeometryToolkitEstimation: synthetic_scenes
export radial_camera_pairs, RC_CENTER

const RC_CENTER = SVector(2000.0, 1500.0)
function radial_camera_pairs()
    (; DivisionBoards, Contamination) = synthetic_scenes()
    cam = DivisionBoards.board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f = 900.0, ks = (-2.0e-7,), center = RC_CENTER)
    return Contamination.contaminated_pairs(
        StableRNG(19), DivisionBoards.board_pixels(cam; grid = -100.0:25.0:100.0); σ = 0.2, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(4000 * rand(rng), 3000 * rand(rng))
    )
end
end

@testsetup module RadialScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: synthetic_scenes
const _Family = synthetic_scenes().RadialScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:RadialScenes])...))
end

# Too few data for a minimal sample (ransac_no_model_tests.jl, refine_contract_tests.jl): `n` plane points on a
# parabola, and their correspondences under a pure shift.
@testsetup module TooFewData
using VisualGeometryToolkitCore: Point2, correspondences
export few_points, few_correspondences
few_points(n) = [Point2(100.0 + 10i, 50.0 + 7i^2) for i in 1:n]
few_correspondences(n) = correspondences(few_points(n), few_points(n) .+ Ref(Point2(1.0, 2.0)))
end

# The contaminated scene of every find_* class (core_defaults_tests.jl, termination_record_tests.jl): a homography's
# pairs, a stereo pair's pixels, a PnP set, a radial homography's pairs and a robot-world hand-eye set, each with its
# outliers and the number of inliers or the truth.
@testsetup module FindScenes
    using Test: Test                                       # loads the extension
    using Random: rand, randn
    using StaticArrays: SA, SVector, SMatrix
    using StructArrays: StructArray
    using StableRNGs: StableRNG
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore: CameraModel, Vec3, RadialHomography, HomographyMat, radial_transfer
    using VisualGeometryToolkitEstimation: synthetic_scenes
    const _H = synthetic_scenes().HomographyScenes
    const _E = synthetic_scenes().EpipolarScenes
    const _P = synthetic_scenes().PoseScenes
    const _C = synthetic_scenes().Contamination
    using ._H: homography_pairs, H_PX, gaussian_source
    using ._E: stereo_pixel_pairs
    using ._P: frustum_point, robot_world_hand_eye_scene, perturb_pose, marker_pose
    using ._C: contaminated_pairs
    export h_scene, stereo_scene, PCAM, R_PNP, T_PNP, pnp_scene, RW, RH, M_RH, radial_homography_scene, hand_eye_scene

    function h_scene()
        src, dst = homography_pairs(
            StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3,
            source = gaussian_source, target = rng -> randn(rng, 2) .* 400 .+ 300
        )
        return StructArray(src .=> dst), 120
    end
    function stereo_scene(cam; seed = 13)
        u₁, u₂ = stereo_pixel_pairs(
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
    function pnp_scene()
        pairs, _ = contaminated_pairs(
            StableRNG(19), (rng, _) -> Pair(frustum_point(rng, PCAM, R_PNP, T_PNP)...), 120; σ = 0.5, n_out = 60,
            outlier = (rng, _) -> first(frustum_point(rng, PCAM, R_PNP, T_PNP)) => SVector(40.0 + 880.0 * rand(rng), 40.0 + 640.0 * rand(rng))
        )
        return StructArray(first.(pairs) .=> last.(pairs))
    end
    const RW, RH = 1200.0, 900.0
    const M_RH = RadialHomography{Float64}(
        HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
        -0.25 / hypot(RW, RH)^2, SVector(RW / 2, RH / 2)
    )
    function radial_homography_scene()
        function inlier(rng, _)
            while true
                s = SVector(200 * rand(rng), 150 * rand(rng))
                d = radial_transfer(M_RH, s)
                d === nothing && continue
                (0 < d[1] < RW && 0 < d[2] < RH) && return s => SVector(d[1], d[2])
            end
        end
        return contaminated_pairs(
            StableRNG(17), inlier, 70; σ = 0.3, n_out = 30, shuffle = true,
            outlier = (rng, _) -> SVector(200 * rand(rng), 150 * rand(rng)) => SVector(RW * rand(rng), RH * rand(rng))
        )
    end
    function hand_eye_scene()
        rng = StableRNG(29)
        sc = robot_world_hand_eye_scene(rng; N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                        A = (ang_max = π / 3, t_scale = 0.5))
        X, Y, A = only(sc.X), sc.Y, sc.A
        B = [perturb_pose(rng, marker_pose(Y, A[i], X); σ_R = 0.001, σ_t = 0.001) for i in eachindex(A)]
        return A .=> B, X
    end

end
