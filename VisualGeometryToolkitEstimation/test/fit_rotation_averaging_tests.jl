# fit_rotation_averaging — robust rotation averaging on a view graph: exact on
# consistent rotations, an outlier edge down-weighted, noise, prior weights, the
# gauge, a disconnected graph refused.

@testitem "fit_rotation_averaging: exact, robust, gauged" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: fit_rotation_averaging
    using VisualGeometryToolkitCore: estimate, converged, CauchyShape, TLSShape

    rng = Xoshiro(17)
    n = 8
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel(i, j) = R[j]' * R[i]
    err(e) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(e[k])' * R[k])), 1:n)
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    @testset "consistent edges: exact" begin
        res = fit_rotation_averaging(Dict(e => rel(e...) for e in edges))
        @test converged(res)
        @test err(estimate(res)) < 1.0e-10
        @test estimate(res)[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12
    end
    @testset "noise and one outlier edge: down-weighted" begin
        noisy = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
        bad = (2, 4); noisy[bad] = RotMatrix(RotationVec(0.5, -0.3, 0.2)) * noisy[bad]
        res = fit_rotation_averaging(noisy; loss = CauchyShape())
        @test err(estimate(res)) < 5.0e-3
        @test res.weights[bad] < 0.01
        @test res.residuals[bad] > 0.3
        # a zero prior weight drops the edge outright
        p = Dict(e => (e == bad ? 0.0 : 1.0) for e in keys(noisy))
        @test err(estimate(fit_rotation_averaging(noisy, p))) < 5.0e-3
    end
    @test_throws ArgumentError fit_rotation_averaging(Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4)))
    @test_throws ArgumentError fit_rotation_averaging(Dict((1, 2) => rel(1, 2)); loss = TLSShape())
end
