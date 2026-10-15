# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (CLAUDE.md, "Function Verb Conventions"), checked by VisualGeometryCore's
# `test_naming` against this package. The checker and its rule-by-rule tests
# are VisualGeometryCore's; this item only asserts that RVG complies.

@testitem "quality_naming" begin
    using Test, VisualGeometryCore, RobustVisualGeometry
    VisualGeometryCore.test_naming(RobustVisualGeometry)
end
