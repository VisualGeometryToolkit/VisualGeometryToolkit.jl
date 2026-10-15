# Line2DManifold and the point-on-line constraint: the chart contract, single-
# point data in the covariance machinery, and the Monte Carlo calibration of the
# line's Gauss–Newton covariance against total least squares.
@testitem "line_manifold" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using VisualGeometryCore
    using VisualGeometryCore: Line2DManifold, retract, recenter, retract_jacobian, action_jacobian,
        local_coordinates, tangent_size, model, minimal_manifold, constraint_with_jacobians,
        implicit_covariance, constraint_jacobians, coeffs, normal

    l = Line2D(0.6, 0.8, -5.0)
    ch = minimal_manifold(l)
    t = SVector(-0.8, 0.6); p0 = SVector(3.0, 4.0)                 # on the line, along it

    @testset "the chart" begin
        @test ch isa Line2DManifold && tangent_size(ch) == 2 && model(ch) == l
        v = SVector(0.2, -0.3)
        lv = retract(ch, v)
        @test normal(lv) ≈ SMatrix{2, 2}(cos(0.2), sin(0.2), -sin(0.2), cos(0.2)) * normal(l) atol = 1.0e-15
        @test coeffs(lv)[3] ≈ -5.3 atol = 1.0e-15
        @test retract_jacobian(ch, v) ≈ ForwardDiff.jacobian(w -> coeffs(retract(ch, w)), v) atol = 1.0e-15
        @test local_coordinates(ch, lv) ≈ v atol = 1.0e-14
        @test local_coordinates(ch, Line2D((-coeffs(lv))...)) ≈ v atol = 1.0e-14    # (−n, −d) is the same line
        @test model(recenter(ch, v)) == lv
    end

    @testset "the point constraint and its chart row" begin
        x = SVector(1.5, -2.0)
        g, G_x, G_θ = constraint_with_jacobians(l, x)
        @test g ≈ 0.6 * 1.5 + 0.8 * -2.0 - 5.0 atol = 1.0e-15
        @test G_x == normal(l) && G_θ == SVector(1.5, -2.0, 1.0)
        @test action_jacobian(ch, x) ≈ transpose(G_θ) * retract_jacobian(ch) atol = 1.0e-15      # (t·x, 1)
        @test action_jacobian(ch, x) ≈ SMatrix{1, 2}(dot(t, x), 1.0) atol = 1.0e-15
    end

    @testset "covariance on single-point data: implicit at 2, the sandwich at n" begin
        rng = Xoshiro(3)
        pts = [Point2(p0 + 10randn(rng) * t) for _ in 1:30]
        A, B = constraint_jacobians(ch, zero(SVector{2, Float64}), pts)
        @test size(A) == (30, 2) && size(B) == (30, 60)
        # isotropic unit noise, unit normal: W₀ = I and the sandwich is (AᵀA)⁻¹ — rows (tᵢ, 1)
        @test implicit_covariance(ch, pts) ≈ inv(A' * A) rtol = 1.0e-12
        # two points: the line through them, differentiated against the implicit covariance
        through(z) = (d = SVector(z[3] - z[1], z[4] - z[2]); n = SVector(-d[2], d[1]) / norm(d);
            local_coordinates(ch, Line2D(n[1], n[2], -(n[1] * z[1] + n[2] * z[2]))))
        z0 = SVector(pts[1][1], pts[1][2], pts[2][1], pts[2][2])
        Jfd = ForwardDiff.jacobian(through, z0)
        @test implicit_covariance(ch, pts[1:2]) ≈ Jfd * Jfd' rtol = 1.0e-10
        # Monte Carlo: total least squares (the ML line for isotropic noise) is
        # calibrated against σ²·(AᵀA)⁻¹ in the chart
        σ = 0.5
        Σ = σ^2 * implicit_covariance(ch, pts)
        L = cholesky(Symmetric(Σ)).L
        nees = map(1:2000) do _
            q = [p + σ * SVector(randn(rng), randn(rng)) for p in pts]
            c = sum(q) / length(q)
            M = sum((p - c) * (p - c)' for p in q)
            n = eigen(Symmetric(Matrix(M))).vectors[:, 1]
            v = local_coordinates(ch, Line2D(n[1], n[2], -(n[1] * c[1] + n[2] * c[2])))
            sum(abs2, L \ v)
        end
        @test abs(sum(nees) / 2000 - 2) < 4 * sqrt(2 * 2 / 2000)
    end
end
