# =============================================================================
# quality_naming_tests.jl — the stack's iterative-estimator vocabulary, checked
# by VisualGeometryToolkitCore's `check_naming` against this package: every name it
# binds and every line of its stack text (src, test, docs, the README and
# CLAUDE.md). The checker, its word lists and its scope live in
# VisualGeometryToolkitCore; this item only runs it here.
# =============================================================================

@testitem "quality_naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitEval
    import VisualGeometryToolkitCore

    found = VgtkCore.check_naming(pkgdir(VisualGeometryToolkitEval))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
