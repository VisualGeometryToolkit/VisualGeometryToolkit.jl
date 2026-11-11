# quality_naming_tests.jl — the stack's iterative-estimator vocabulary,
# checked by VisualGeometryToolkitCore's `check_naming` on this package. The rules
# and the lists of banned words and retired names live in VisualGeometryToolkitCore
# (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which also tests the checker itself.

@testitem "quality: naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    VgtkCore.test_naming(VisualGeometryToolkitIO)
end

@testitem "quality: no duplication" begin
    # One implementation per concept (VisualGeometryToolkitCore's check_duplication): names
    # an upstream stack package owns, restated function bodies, copied files and retired
    # patterns; the other stack repositories in VGTK_STACK_PATHS are compared too.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitIO
    VgtkCore.test_duplication(VisualGeometryToolkitIO)
end
