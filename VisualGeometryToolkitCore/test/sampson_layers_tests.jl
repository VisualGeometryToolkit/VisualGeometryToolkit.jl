# The two layers over one constraint (base/sampson.jl): the Sampson residual is
# geometric (model and point; a measurement's covariance never enters), and the
# noise layer `whiten(s_N, N, Σ)` turns it into the first-order Mahalanobis
# residual — exactly the constraint whitened by its propagated covariance.
@testitem "sampson_layers" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: FundMat, HomographyMat, EssentialMat, Conic, Ellipse, Uncertain, CovMat, PrecMat,
        exact, sampson_distance, sampson_residual, sampson_residual_with_normal, mahalanobis_distance, whiten,
        constraint_with_jacobians, pair_covariance

    F = FundMat(SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0))
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    u, v = SVector(310.0, 205.0), SVector(298.0, 211.0)
    Σ₁ = SMatrix{2, 2}(2.0, 0.3, 0.3, 0.5); Σ₂ = SMatrix{2, 2}(0.7, -0.1, -0.1, 1.5)

    @testset "the normal basis is orthonormal, and spans G_x's rows" begin
        for (M, n) in ((F, 1), (H, 2))
            s, N = sampson_residual_with_normal(M, u, v)
            @test s isa SVector{n, Float64} && N isa SMatrix{4, n, Float64}
            @test N' * N ≈ I rtol = 1.0e-12
            _, G_x, _ = constraint_with_jacobians(M, u, v)
            Gm = G_x isa StaticVector ? transpose(G_x) : G_x
            @test norm(Gm * (I - N * N')) < 1.0e-12 * norm(G_x)    # G_x lies in span(N)
            @test sampson_distance(M, u, v) == norm(s)
            @test sampson_residual(M, u => v) == s
        end
    end

    @testset "whiten(s_N, N, Σ) is the constraint whitened by G_x·Σ·G_xᵀ" begin
        Σ = pair_covariance(Σ₁, Σ₂)
        for M in (F, H)
            g, G_x, _ = constraint_with_jacobians(M, u, v)
            Gm = G_x isa StaticVector ? Matrix(transpose(G_x)) : Matrix(G_x)
            ref = cholesky(Symmetric(Gm * Matrix(Σ) * Gm')).L \ (g isa Real ? [g] : Vector(g))
            r = whiten(sampson_residual_with_normal(M, u, v)..., Σ)
            @test Vector(r) ≈ ref rtol = 1.0e-12                # componentwise, not only in norm
            @test whiten(sampson_residual_with_normal(M, u, v)..., I) == sampson_residual(M, u, v)
            @test whiten(sampson_residual_with_normal(M, u, v)..., 4I) ≈ sampson_residual(M, u, v) / 2 rtol = 1.0e-14
        end
    end

    @testset "Sampson is geometric; Mahalanobis reads the measurements' covariance" begin
        a, b = Uncertain(u, CovMat(Σ₁)), Uncertain(v, PrecMat(inv(Σ₂)))
        for M in (F, H)
            @test sampson_distance(M, a, b) == sampson_distance(M, u, v)
            @test mahalanobis_distance(M, u, v) == sampson_distance(M, u, v)      # plain = unit isotropic
            @test mahalanobis_distance(M, a, b) ≈ norm(whiten(sampson_residual_with_normal(M, u, v)..., pair_covariance(Σ₁, Σ₂))) rtol = 1.0e-12
            @test mahalanobis_distance(M, a => b) == mahalanobis_distance(M, a, b)
        end
        @test_throws ArgumentError mahalanobis_distance(F, exact(u), exact(v))
    end

    @testset "a conic and its ellipse agree; an ellipse's centre is degenerate" begin
        e = Ellipse(SVector(320.0, 240.0), 90.0, 40.0, 0.4)
        C = Conic(SMatrix{3, 3, Float64}(VgtkCore.HomEllipseMat(e)))
        for p in (SVector(420.0, 250.0), SVector(300.0, 290.0), SVector(330.0, 230.0))
            @test sampson_distance(C, p) ≈ sampson_distance(e, p) rtol = 1.0e-10
            @test sampson_distance(Conic(-3.0 * C.Q), p) ≈ sampson_distance(C, p) rtol = 1.0e-12     # homogeneous
        end
        @test sampson_distance(e, e.center) == Inf
        @test sampson_distance(C, SVector(e.center...)) == Inf
        # a single point's Mahalanobis distance; an exact single point has no noise
        p = SVector(420.0, 250.0)
        @test mahalanobis_distance(e, Uncertain(p, CovMat(4.0 * SMatrix{2, 2, Float64}(I)))) ≈ sampson_distance(e, p) / 2 rtol = 1.0e-12
        @test_throws ArgumentError mahalanobis_distance(e, exact(p))
    end

    @testset "whiten's own degeneracy: a covariance singular along the normal" begin
        s, N = sampson_residual_with_normal(F, u, v)
        Σ0 = zero(SMatrix{4, 4, Float64})
        @test whiten(s, N, Σ0) == copysign.(Inf, s)
        @test whiten(zero(s), N, Σ0) == zero(s)
        sH, NH = sampson_residual_with_normal(H, u, v)
        @test all(isinf, whiten(sH, NH, Σ0))
        @test all(isnan, whiten(fill(NaN, typeof(s)), N, 2I))
    end

    @testset "no allocations" begin
        f(M, a, b) = sampson_distance(M, a, b)
        g(M, a, b) = mahalanobis_distance(M, a, b)
        h(M, a, b) = sampson_residual_with_normal(M, a, b)
        c(M, a, b) = VgtkCore.constraint(M, a, b)
        f(F, u, v); g(F, u, v); h(H, u, v); f(H, u, v); c(H, u, v)
        @test (@allocated f(F, u, v)) == 0
        @test (@allocated f(H, u, v)) == 0
        @test (@allocated g(H, u, v)) == 0
        @test (@allocated h(H, u, v)) == 0
        @test (@allocated c(H, u, v)) == 0
        @test VgtkCore.constraint(H, u, v) == first(constraint_with_jacobians(H, u, v))
    end
end
