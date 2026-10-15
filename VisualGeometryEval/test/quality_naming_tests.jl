# =============================================================================
# quality_naming_tests.jl — the stack's iterative-estimator vocabulary, checked
# by VisualGeometryCore's `check_naming` against this package: every name it
# binds and every line of its stack text (src, test, docs, the README and
# CLAUDE.md). The checker, its word lists and its scope live in
# VisualGeometryCore; this item only runs it here.
# =============================================================================

@testitem "quality_naming" begin
    using Test
    using VisualGeometryEval
    import VisualGeometryCore

    found = VisualGeometryCore.check_naming(pkgdir(VisualGeometryEval))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
