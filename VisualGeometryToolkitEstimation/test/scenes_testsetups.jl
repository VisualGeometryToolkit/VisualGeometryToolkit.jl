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
