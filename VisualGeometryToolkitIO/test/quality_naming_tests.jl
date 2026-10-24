# quality_naming_tests.jl — the stack's iterative-estimator vocabulary,
# checked by VisualGeometryToolkitCore's `check_naming` on this package. The rules
# and the lists of banned words and retired names live in VisualGeometryToolkitCore
# (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which also tests the checker itself.

@testitem "quality: naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    found = VgtkCore.check_naming(pkgdir(VisualGeometryToolkitIO))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
