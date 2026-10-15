# quality_naming_tests.jl — the stack's iterative-estimator vocabulary,
# checked by VisualGeometryCore's `check_naming` on this package. The rules
# and the lists of banned words and retired names live in VisualGeometryCore
# (ext/VisualGeometryCoreTestExt/check_naming.jl, loaded with Test), which also tests the checker itself.

@testitem "quality: naming" begin
    using Test
    using VisualGeometryCore
    found = VisualGeometryCore.check_naming(pkgdir(VisualGeometryIo))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
