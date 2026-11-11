# =============================================================================
# Shared synthetic scenes — ReTestItems `@testsetup` modules
# =============================================================================
#
# ONE definition of every scene more than one test item draws from: the
# scenes are VisualGeometryToolkitEstimation's `test_scenes()` (its Test extension,
# ext/VisualGeometryToolkitEstimationTestExt.jl), which every stack package's tests
# share. An item names the family it needs (`@testitem "x" setup = [EpipolarScenes]
# begin`) and gets its exports; each setup below is that family, re-exported.
# =============================================================================

@testsetup module EpipolarScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().EpipolarScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:EpipolarScenes])...))
end

@testsetup module HomographyScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().HomographyScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:HomographyScenes])...))
end

@testsetup module PoseScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().PoseScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:PoseScenes])...))
end

@testsetup module DivisionBoards
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().DivisionBoards
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:DivisionBoards])...))
end

@testsetup module P3PScenes
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().P3PScenes
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:P3PScenes])...))
end

@testsetup module Cameras
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().Cameras
using ._Family
eval(Expr(:export, setdiff(names(_Family), [:Cameras])...))
end

@testsetup module Contamination
using Test: Test                                       # loads the extension
using VisualGeometryToolkitEstimation: test_scenes
const _Family = test_scenes().Contamination
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
using VisualGeometryToolkitEstimation: test_scenes
export radial_camera_pairs, RC_CENTER

const RC_CENTER = SVector(2000.0, 1500.0)
function radial_camera_pairs()
    (; DivisionBoards, Contamination) = test_scenes()
    cam = DivisionBoards.board_camera(RotXYZ(0.35, -0.25, 0.15), SVector(40.0, -30.0, 600.0); f = 900.0, ks = (-2.0e-7,), center = RC_CENTER)
    return Contamination.contaminated_pairs(
        StableRNG(19), DivisionBoards.board_pixels(cam; grid = -100.0:25.0:100.0); σ = 0.2, n_out = 20, shuffle = true,
        outlier = (rng, _) -> SVector(200 * rand(rng) - 100, 200 * rand(rng) - 100) => SVector(4000 * rand(rng), 3000 * rand(rng))
    )
end
end
