# Synthetic.projective_distance — the distance between two models known only up
# to scale and sign (synthetic/evaluation.jl).
@testitem "projective_distance" begin
    using Test, LinearAlgebra, Random, StaticArrays
    using VisualGeometryCore
    using VisualGeometryCore.Synthetic: projective_distance

    rng = MersenneTwister(5)
    A = SMatrix{3, 3, Float64, 9}(randn(rng, 9))
    B = SMatrix{3, 3, Float64, 9}(randn(rng, 9))

    # blind to the representative: any nonzero scale, either sign
    @test projective_distance(A, A) == 0
    for c in (3.7, -0.2, -1.0)
        @test projective_distance(A, c * A) ≈ 0 atol = 1.0e-15
        @test projective_distance(A, c * B) ≈ projective_distance(A, B) rtol = 1.0e-12
    end
    # symmetric, bounded by √2 (orthogonal representatives)
    @test projective_distance(A, B) ≈ projective_distance(B, A) rtol = 1.0e-12
    e1 = SVector(1.0, 0.0, 0.0); e2 = SVector(0.0, 1.0, 0.0)
    @test projective_distance(e1, e2) ≈ sqrt(2) rtol = 1.0e-15
    # the closer of the two signs, against the definition by cases
    â = vec(Matrix(A)) / norm(A); b̂ = vec(Matrix(B)) / norm(B)
    @test projective_distance(A, B) ≈ min(norm(â - b̂), norm(â + b̂)) rtol = 1.0e-12
    # small values are the angle between the representatives: 1 − |cos| = d²/2
    θ = 1.0e-3
    u = SVector(cos(θ), sin(θ), 0.0)
    @test projective_distance(e1, u) ≈ 2sin(θ / 2) rtol = 1.0e-12
    # any shape with the same number of entries; a mismatch is refused
    @test projective_distance(SMatrix{2, 3}(1.0:6.0...), -2 * SMatrix{2, 3}(1.0:6.0...)) ≈ 0 atol = 1.0e-15
    @test_throws DimensionMismatch projective_distance(e1, SVector(1.0, 2.0))
end
