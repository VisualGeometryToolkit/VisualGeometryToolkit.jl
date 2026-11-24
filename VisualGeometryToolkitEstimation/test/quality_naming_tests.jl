# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (CLAUDE.md, "Function Verb Conventions"), checked by VisualGeometryToolkitCore's
# `test_naming` against this package. The checker and its rule-by-rule tests
# are VisualGeometryToolkitCore's; this item only asserts that RVG complies.

@testitem "quality_naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
    VgtkCore.test_naming(VisualGeometryToolkitEstimation)
end

@testitem "quality: the estimation interface's declarations" begin
    # check_naming's run-time rules (R16–R18) and the declared defaults (decision 7): every algorithm has its kind,
    # every solve method is a `__solve`/`__init` of an algorithm this package owns, an `rng` only where the pair draws.
    import VisualGeometryToolkitCore as VgtkCore
    using Test, VisualGeometryToolkitEstimation
    found = VgtkCore.check_algorithm_traits(VisualGeometryToolkitEstimation)
    isempty(found) || foreach(println, found)
    @test isempty(found)
end

@testitem "quality: no duplication" begin
    # One implementation per concept (VisualGeometryToolkitCore's check_duplication): names
    # an upstream stack package owns, restated function bodies, copied files and retired
    # patterns; the other stack repositories in VGTK_STACK_PATHS are compared too.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitEstimation
    VgtkCore.test_duplication(VisualGeometryToolkitEstimation)
end
