# SphereManifold: the chart of a model known up to scale and sign, and the
# radial camera's minimal manifold through it — the contract (basis, retraction
# differential, inverse, re-anchoring) and the covariance machinery on it.
@testitem "sphere_manifold" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: SphereManifold, RadialCamera, retract, recenter, retract_jacobian,
        local_coordinates, tangent_size, model, minimal_manifold, manifold_constraints,
        constraint_jacobians, implicit_covariance

    @testset "the chart: basis, retraction differential, inverse, re-anchoring" begin
        # generic, q₁ < 0, and the Householder's hardest inputs ±e₁
        for q in (SVector(0.3, -0.5, 0.2, 0.7), SVector(-0.9, 0.1, 0.2, 0.1), SVector(1.0, 0.0, 0.0, 0.0), SVector(-1.0, 1.0e-9, 0.0, 0.0))
            ch = SphereManifold(3 * q)                           # normalized on construction
            @test tangent_size(ch) == 3
            @test norm(ch.q) ≈ 1 atol = 1.0e-15
            @test ch.B' * ch.B ≈ I atol = 1.0e-15
            @test maximum(abs, ch.B' * ch.q) < 1.0e-15
            @test model(ch) == ch.q
            v = SVector(0.1, -0.2, 0.05)
            @test norm(retract(ch, v)) ≈ 1 atol = 1.0e-15
            @test retract_jacobian(ch, v) ≈ ForwardDiff.jacobian(w -> retract(ch, w), v) atol = 1.0e-15
            @test retract_jacobian(ch) ≈ ch.B atol = 1.0e-15
            @test local_coordinates(ch, retract(ch, v)) ≈ v atol = 1.0e-14
            @test local_coordinates(ch, -3 * retract(ch, v)) ≈ v atol = 1.0e-14    # scale and sign invariant
            @test retract(recenter(ch, v), zero(v)) ≈ retract(ch, v) atol = 1.0e-15
        end
    end

    @testset "the radial camera's minimal manifold: S⁵ on vec(P), the centre riding along" begin
        rng = Xoshiro(2)
        P = SMatrix{2, 3}(1.0, 0.1, -0.2, 0.9, 30.0, -20.0); c = Point2(640.0, 480.0)
        m = RadialCamera(P, c)
        ch = minimal_manifold(m)
        @test ch isa SphereManifold
        m0 = retract(ch, zero(SVector{5, Float64}))
        @test m0 isa RadialCamera{Float64} && m0.center == c
        @test m0.P ≈ P / norm(P) atol = 1.0e-15
        v = SVector(0.01, -0.02, 0.005, 0.01, -0.03)
        @test local_coordinates(ch, RadialCamera(-7 * retract(ch, v).P, c)) ≈ v atol = 1.0e-13
        # plane points and pixels on their radial lines: the constraints vanish at the
        # anchor, their chart Jacobian matches central differences, and a minimal
        # sample's implicit covariance is the Gauss–Newton sandwich
        cs = [(X = Point2(100randn(rng), 100randn(rng)); d = P * SVector(X[1], X[2], 1.0); X => Point2(c[1] + 3d[1], c[2] + 3d[2])) for _ in 1:5]
        z = zero(SVector{5, Float64})
        @test maximum(abs, manifold_constraints(ch, z, cs)) < 1.0e-9
        A, B = constraint_jacobians(ch, z, cs)
        @test size(A) == (5, 5) && size(B) == (5, 10)
        Afd = FiniteDiff.finite_difference_jacobian(w -> manifold_constraints(ch, w, cs), z, Val(:central))
        @test A ≈ Afd rtol = 1.0e-8
        Σi = implicit_covariance(ch, cs)                        # the square solve
        blk(i) = (A[i:i, :], SMatrix{1, 2}(B[i, (2i - 1):(2i)]))   # … and the sandwich over its blocks
        @test Σi ≈ implicit_covariance(blk, 1:5) rtol = 1.0e-10
    end
end
