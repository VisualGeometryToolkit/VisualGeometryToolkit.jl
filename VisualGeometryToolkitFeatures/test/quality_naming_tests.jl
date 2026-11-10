# quality_naming_tests.jl — the stack's iterative-estimator vocabulary,
# checked by VisualGeometryToolkitCore's `check_naming` on this package. The rules
# and the lists of banned words and retired names live in VisualGeometryToolkitCore
# (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which also tests the checker itself.

@testitem "quality: naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    found = VgtkCore.check_naming(pkgdir(VisualGeometryToolkitFeatures))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end

# The stack's one RAW path, checked by VisualGeometryToolkitCore's `check_raw_decoding`:
# outside this package's raw module (src/raw_image.jl, its LibRaw_jll extension and
# their tests), no decoding library, demosaic or CFA interpolation.
@testitem "quality: one RAW path" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    found = VgtkCore.check_raw_decoding(pkgdir(VisualGeometryToolkitFeatures))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
