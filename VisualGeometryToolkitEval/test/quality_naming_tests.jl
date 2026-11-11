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

    VgtkCore.test_naming(VisualGeometryToolkitEval)
end

@testitem "quality: no duplication" begin
    # One implementation per concept (VisualGeometryToolkitCore's check_duplication): names
    # an upstream stack package owns, restated function bodies, copied files and retired
    # patterns; the other stack repositories in VGTK_STACK_PATHS are compared too.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitEval
    VgtkCore.test_duplication(VisualGeometryToolkitEval)
end
