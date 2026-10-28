# =============================================================================
# Equirectangular (geometry/cameras/equirectangular.jl): the axes, the round
# trip both ways, scale-freeness, the poles and the refusals.
# =============================================================================

@testitem "Equirectangular: axes, round trips, poles" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: Equirectangular, project, backproject, angle_between

    E = Equirectangular()
    @testset "the axes: forward, right, up, back" begin
        @test project(E, SVector(0.0, 0.0, 1.0)) == SVector(0.0, 0.0)
        @test project(E, SVector(1.0, 0.0, 0.0)) ≈ SVector(π / 2, 0.0)
        @test project(E, SVector(-1.0, 0.0, 0.0)) ≈ SVector(-π / 2, 0.0)
        @test project(E, SVector(0.0, -1.0, 0.0)) ≈ SVector(0.0, π / 2)     # up = −y
        @test project(E, SVector(0.0, 1.0, 0.0)) ≈ SVector(0.0, -π / 2)
        @test project(E, SVector(0.0, 0.0, -1.0)) ≈ SVector(π, 0.0)
        @test backproject(E, SVector(π / 2, 0.0)) ≈ SVector(1.0, 0.0, 0.0) atol = 1.0e-16
        @test backproject(E, SVector(0.0, π / 2)) ≈ SVector(0.0, -1.0, 0.0) atol = 1.0e-16
    end
    @testset "round trips, and the length of X does not enter" begin
        rng = Xoshiro(3)
        for _ in 1:1000
            X = randn(rng, SVector{3, Float64})
            p = project(E, X)
            @test -π < p[1] <= π && -π / 2 <= p[2] <= π / 2
            @test angle_between(backproject(E, p), X) < 1.0e-14
            @test project(E, 7.5 * X) ≈ p rtol = 1.0e-14
            @test norm(backproject(E, p)) ≈ 1 rtol = 1.0e-15
            q = SVector(2π * rand(rng) - π, π * rand(rng) - π / 2)
            @test project(E, backproject(E, q)) ≈ q atol = 1.0e-13
        end
        @test backproject(E, SVector(0.3 + 2π, 0.2)) ≈ backproject(E, SVector(0.3, 0.2)) atol = 1.0e-15
    end
    @testset "a zero or non-finite direction is NaN; Float32 stays Float32" begin
        @test all(isnan, project(E, SVector(0.0, 0.0, 0.0)))
        @test all(isnan, project(E, SVector(NaN, 0.0, 1.0)))
        @test all(isnan, project(E, SVector(0.0, Inf, 1.0)))
        @test eltype(project(E, SVector(1.0f0, 0.0f0, 1.0f0))) === Float32
        @test eltype(backproject(E, SVector(0.1f0, 0.2f0))) === Float32
    end
end
