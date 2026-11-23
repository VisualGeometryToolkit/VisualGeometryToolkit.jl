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
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "conic_scenes.jl"))   # shared with the golden records
export circle_scene, ellipse_scene, C₀, R₀, E₀
end

# The radial camera's contaminated board scene (core_defaults_tests.jl,
# contract_conformance_tests.jl): a division-lens board seen from one pose, its
# pixel pairs with 20 outliers, and the distortion centre the problems take.
@testsetup module RadialCameraScene
using Test: Test                                       # loads the extension
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "find_scenes.jl"))   # shared with the golden records
export radial_camera_pairs, RC_CENTER
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
using Test: Test                                       # loads the extension
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "find_scenes.jl"))   # shared with the golden records
export few_points, few_correspondences
end

# The contaminated scene of every find_* class (core_defaults_tests.jl, termination_record_tests.jl): a homography's
# pairs, a stereo pair's pixels, a PnP set, a radial homography's pairs and a robot-world hand-eye set, each with its
# outliers and the number of inliers or the truth.
@testsetup module FindScenes
    using Test: Test                                       # loads the extension
    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "find_scenes.jl"))   # shared with the golden records
    export h_scene, stereo_scene, PCAM, R_PNP, T_PNP, pnp_scene, RW, RH, M_RH, radial_homography_scene, hand_eye_scene
end
