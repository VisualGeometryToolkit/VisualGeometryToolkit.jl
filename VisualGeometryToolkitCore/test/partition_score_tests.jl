# `partition_score` — the ONE per-partition score of the marginal sweep, public
# so that a sweep outside this package (BlobBoards' per-blob device sweep, whose
# `partition_score` restated it) calls it instead of a copy: precision-generic and
# plain arithmetic, so it runs inside a KernelAbstractions kernel.
@testitem "partition_score" begin
    using Test
    using SpecialFunctions: loggamma
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore, partition_score
    using VisualGeometryToolkitCore: IGPrior
    @test Base.ispublic(VisualGeometryToolkitCore, :partition_score)
    # Jeffreys (α₀ = β₀ = 0): S = lgamma(ν/2) − (ν/2)·log(π·RSS) − outlier cost
    for (ν½, RSS, out) in ((1.5, 0.37, 2.0), (4.0, 12.5, 0.0), (0.5, 1.0e-3, 7.25))
        @test partition_score(IGPrior(), loggamma(ν½), ν½, RSS, out) ≈ loggamma(ν½) - ν½ * log(π * RSS) - out
    end
    # a proper prior: (ν/2 + α₀)·log(RSS/2 + β₀) in place of (ν/2)·log(RSS/2)
    p = IGPrior(2.0, 0.5)
    @test partition_score(p, 0.0, 3.0, 4.0, 1.0) ≈ -3 * log(2π) - 5 * log(2.5) - 1
    # precision-generic: Float32 in, Float32 out (the device sweeps' Ta)
    s32 = partition_score(IGPrior(), 0.5f0, 1.5f0, 0.37f0, 2.0f0)
    @test s32 isa Float32 && s32 ≈ Float32(partition_score(IGPrior(), 0.5, 1.5, 0.37, 2.0))
end
