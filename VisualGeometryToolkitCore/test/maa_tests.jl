# Synthetic.mean_average_accuracy / maa_score — the continuous mAA@T, whose
# threshold `T` the caller always supplies (synthetic/evaluation.jl).

@testitem "maa" begin
    using Test
    using VisualGeometryToolkitCore.Synthetic: mean_average_accuracy, maa_score, maa
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

@testitem "perturbed_pose" begin
    using Test
    using StableRNGs, StaticArrays
    using Rotations: RotZ, RotY
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore.Synthetic: perturbed_pose

    # Expected values recorded from VisualGeometryToolkitEval's `perturbed_pose`
    # (the body this one is moved from, verbatim) with the same inputs; to 1e-14,
    # since a runner's libm differs from the recording machine's by an ulp.
    R = SMatrix{3, 3, Float64}(RotZ(0.3) * RotY(-0.2))
    t = SVector(0.6, 0.1, -0.7)
    Rp, tp = perturbed_pose(StableRNG(1), R, t, 0.1)
    @test isapprox(Rp, [0.9112397858850927 -0.372247141313633 -0.1762785250786902; 0.37526993893126315 0.9267570477025991 -0.01714197970155395; 0.16974841841565536 -0.050531577428282605 0.9841910557041215]; rtol = 1.0e-14)
    @test isapprox(tp, [0.5748344741822867, -0.11302012837107411, -0.8104269108775696]; rtol = 1.0e-14)
    @test norm(tp) ≈ 1
    # a caller-seeded rng reproduces the seed
    @test perturbed_pose(StableRNG(1), R, t, 0.1) == (Rp, tp)
end
