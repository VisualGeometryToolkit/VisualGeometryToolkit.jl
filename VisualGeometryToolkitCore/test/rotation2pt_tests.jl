# =============================================================================
# Rotation2pt (geometry/solvers/solver_layer.jl): exact on two clean pairs at
# any field angle (past 90° included), scale-free, the parallel refusal.
# =============================================================================

@testitem "Rotation2pt: exact on two pairs, degenerate refused" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    using VisualGeometryToolkitCore: Rotation2pt, solve, sample_size, max_solutions, Csponds

    rng = Xoshiro(5)
    @test sample_size(Rotation2pt()) == 2 && max_solutions(Rotation2pt()) == 1
    for _ in 1:200
        R = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
        u = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:2]   # anywhere on the sphere: past 90° of any axis
        v = [R * x for x in u]
        sol = solve(Rotation2pt(), Csponds(3.0 .* u, 0.5 .* v), [1, 2])
        @test sol.count == 1
        @test norm(SMatrix{3, 3}(first(sol)) - R) < 1.0e-12
    end
    u = [SVector(1.0, 0.0, 0.0), SVector(-2.0, 0.0, 0.0)]                     # antiparallel in the source view
    @test solve(Rotation2pt(), Csponds(u, u), [1, 2]).count == 0
    a = SVector(0.0, 0.0, 1.0)
    @test solve(Rotation2pt(), Csponds([a, SVector(0.0, 1.0, 0.0)], [a, a]), [1, 2]).count == 0   # parallel in the target view
    @test solve(Rotation2pt(), Csponds([a, SVector(NaN, 0.0, 1.0)], [a, a]), [1, 2]).count == 0
end
