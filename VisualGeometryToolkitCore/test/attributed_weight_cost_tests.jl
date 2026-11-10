# The per-datum attributes the stages read off an `Attributed` value: its weight
# (`weight_of`, 1.0 for a bare value or one without `:weight`), its value
# (`value_of`) and, for the best
# iterate of a descent, its cost (`keep_best` over `Attributed(θ, (; cost))`).
@testitem "weight_of, value_of: an Attributed datum's weight (1.0 otherwise) and value" begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: Attributed, CovMat, weight_of, value_of

    p = SVector(1.0, 2.0)
    @test weight_of(Attributed(p, (; weight = 2.5))) === 2.5
    @test weight_of(Attributed(p, (; weight = 3))) === 3.0                  # integer weights read as Float64
    @test weight_of(Attributed(p, (; weight = 0.5f0, score = 9.0))) === 0.5
    @test weight_of(Attributed(p, (; score = 9.0))) === 1.0                 # no :weight
    @test weight_of(Attributed(p, 0.83)) === 1.0                           # a bare-score attribute
    @test weight_of(Attributed(p, CovMat{2, Float64}(1.0, 0.0, 0.0, 1.0))) === 1.0
    @test weight_of(p) === 1.0                                             # a bare value
    @test weight_of(7) === 1.0
    @test value_of(Attributed(p, (; weight = 2.0))) === p
    @test value_of(p) === p
end

@testitem "keep_best: the best iterate is Attributed(θ, (; cost))" begin
    using Test
    using VisualGeometryToolkitCore: Attributed, keep_best

    inc = Attributed([1.0, 2.0], (; cost = 5.0))
    @test inc.cost === 5.0 && inc.value == [1.0, 2.0]
    inc2, improved = keep_best(inc, [0.0, 0.0], 4)                         # strictly lower: kept, cost as Float64
    @test improved && inc2.value == [0.0, 0.0] && inc2.cost === 4.0
    @test typeof(inc2) === typeof(inc)
    inc3, improved = keep_best(inc2, [9.0, 9.0], 4.0)                      # a tie keeps the incumbent
    @test !improved && inc3 === inc2
end
