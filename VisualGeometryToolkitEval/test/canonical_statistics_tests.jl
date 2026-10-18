# =============================================================================
# The package's canonical statistics: `finite_median` skips non-finite values
# and is NaN on an empty (or all-NaN) sample.
# =============================================================================

@testitem "canonical_statistics" begin
    using Test
    using VisualGeometryToolkitEval.Stats: finite_median

    @testset "canonical statistics" begin
        @test isnan(finite_median(Float64[]))
        @test finite_median([1.0, NaN, 3.0]) == 2.0
    end
end
