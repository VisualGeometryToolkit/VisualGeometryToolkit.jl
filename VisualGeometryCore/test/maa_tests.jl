# Synthetic.mean_average_accuracy / maa_score — the continuous mAA@T, whose
# threshold `T` the caller always supplies (synthetic/evaluation.jl).

@testitem "maa" begin
    using Test
    using VisualGeometryCore.Synthetic: mean_average_accuracy, maa_score, maa
    using Statistics: mean

    e = [0.0, 2.5, 5.0, 10.0, 20.0, NaN, Inf]
    @testset "the threshold is required" begin
        @test_throws UndefKeywordError mean_average_accuracy(e)
        @test_throws UndefKeywordError maa_score(1.0)
    end
    @testset "the continuous AUC and its per-item form" begin
        @test maa_score(0.0; T = 10) == 1.0
        @test maa_score(2.5; T = 10) == 0.75
        @test maa_score(10.0; T = 10) == 0.0
        @test maa_score(NaN; T = 10) == 0.0
        @test mean_average_accuracy(e; T = 10) == (1.0 + 0.75 + 0.5) / 7
        @test mean_average_accuracy(e; T = 20) ≈ mean(x -> maa_score(x; T = 20), e)
        @test isnan(mean_average_accuracy(Float64[]; T = 10))
        @test maa(e, 10) == 3 / 7
    end
end
