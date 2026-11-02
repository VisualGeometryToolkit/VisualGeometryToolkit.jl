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

# A conic's and an ellipse's Sampson rows: `measurement_gradient_jacobian` against
# ForwardDiff of their constraints' G_x, the exact residual Jacobian against
# ForwardDiff of the residual, ForwardDiff Duals through the whole path, and no
# allocation.
@testitem "sampson_conic_ellipse_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using ForwardDiff: ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Conic, Ellipse, Point2, constraint_with_jacobians, measurement_gradient_jacobian,
        sampson_residual, sampson_residual_with_jacobian

    conic(θ) = Conic(SMatrix{3, 3}(θ[1], θ[2], θ[4], θ[2], θ[3], θ[5], θ[4], θ[5], θ[6]))
    θ₀ = SVector(0.002, 0.0003, 0.003, -0.5, -0.4, 150.0)            # an ellipse's conic, (C₁₁, C₁₂, C₂₂, C₁₃, C₂₃, C₃₃)
    E = Ellipse(Point2(500.0, 400.0), 300.0, 180.0, 0.6)
    Q = VgtkCore._ellipse_form(E)
    θE = SVector(E.center[1], E.center[2], Q[1, 1], Q[1, 2], Q[2, 2])
    ellipse_g(θ, x) = (d = x - SVector(θ[1], θ[2]); dot(d, SMatrix{2, 2}(θ[3], θ[4], θ[4], θ[5]) * d) - 1)
    for u in (SVector(210.0, 180.0), SVector(820.0, 395.0), SVector(505.0, 120.0))
        # the conic: ∂G_x/∂θ and the residual's exact model row
        @test measurement_gradient_jacobian(conic(θ₀), u) ≈
            ForwardDiff.jacobian(θ -> constraint_with_jacobians(conic(θ), u)[2], θ₀) rtol = 1.0e-14
        s, J = sampson_residual_with_jacobian(conic(θ₀), u)
        @test vec(J) ≈ ForwardDiff.gradient(θ -> only(sampson_residual(conic(θ), u)), θ₀) rtol = 1.0e-12
        # the ellipse, in its G_θ's coordinates (c₁, c₂, Q₁₁, Q₁₂, Q₂₂)
        @test constraint_with_jacobians(E, u)[3] ≈ ForwardDiff.gradient(θ -> ellipse_g(θ, u), θE) rtol = 1.0e-13
        @test measurement_gradient_jacobian(E, u) ≈
            ForwardDiff.jacobian(θ -> ForwardDiff.gradient(x -> ellipse_g(θ, x), u), θE) rtol = 1.0e-13
        sE, JE = sampson_residual_with_jacobian(E, u)
        @test vec(JE) ≈ ForwardDiff.gradient(θ -> (d = u - SVector(θ[1], θ[2]); Qθ = SMatrix{2, 2}(θ[3], θ[4], θ[4], θ[5]);
            ellipse_g(θ, u) / norm(2 * Qθ * d)), θE) rtol = 1.0e-12
    end
    # ForwardDiff Duals through the conic's whole Sampson path (residual and row)
    u = SVector(210.0, 180.0)
    d(θ) = (sJ = sampson_residual_with_jacobian(conic(θ), u); only(sJ[1]) + sum(sJ[2]))
    @test ForwardDiff.gradient(d, θ₀) isa SVector{6, Float64}
    @test all(isfinite, ForwardDiff.gradient(d, θ₀))
    # no allocation
    rows(c, x) = sampson_residual_with_jacobian(c, x)
    allocs(c, x) = (rows(c, x); @allocated rows(c, x))
    @test allocs(conic(θ₀), u) == 0
    @test allocs(E, u) == 0
    # a conic's centre is degenerate: the residual's Inf and a zero row
    ctr = SMatrix{2, 2}(θ₀[1], θ₀[2], θ₀[2], θ₀[3]) \ -SVector(θ₀[4], θ₀[5])
    sc, Jc = sampson_residual_with_jacobian(conic(θ₀), ctr)
    @test isinf(only(sc)) && all(iszero, Jc)
end

# A constraint tuple as a least-squares row (`whitened_constraint_rows`): the
# whitened residual and the frozen-whitened model row under ONE factor, the
# degeneracy rule at the tuple's rounding scale — what a predictive scorer reads.
@testitem "whitened_constraint_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: whitened_constraint_rows, whitened_constraint_residual, whitening_factor,
        constraint_with_jacobians, constraint_rounding_scale, whiten, sampson_residual_with_normal, pair_covariance

    F = FundMat(SMatrix{3, 3, Float64}(0.0, 1.0e-6, -2.0e-4, -1.0e-6, 0.0, 3.0e-3, 1.0e-4, -3.0e-3, 0.02))
    u, v = SVector(310.0, 205.0), SVector(330.0, 190.0)
    g, G_x, G_θ = constraint_with_jacobians(F, u, v)
    # plain: the Sampson residual and G_θ/‖G_x‖, the number and the row a scalar constraint gives
    r, J = whitened_constraint_rows((g, G_x, G_θ))
    @test r == whitened_constraint_residual((g, G_x, G_θ))
    @test r ≈ g / norm(G_x) rtol = 1.0e-14
    @test J isa SVector{9}
    @test J ≈ G_θ / norm(G_x) rtol = 1.0e-14
    # with a covariance and the rounding scale: the same factor as whitening_factor
    Σ = pair_covariance(SMatrix{2, 2}(2.0, 0.3, 0.3, 1.0), SMatrix{2, 2}(0.5, 0.0, 0.0, 0.8))
    ρ = constraint_rounding_scale(F, u, v)
    rΣ, JΣ = whitened_constraint_rows((g, G_x, G_θ, Σ, ρ))
    @test rΣ ≈ whitening_factor(G_x, Σ) * g rtol = 1.0e-12
    @test JΣ ≈ whitening_factor(G_x, Σ) * G_θ rtol = 1.0e-12
    # a vector constraint (H's two rows): a vector residual and a 2 × 9 row
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    gH, GxH, GθH = constraint_with_jacobians(H, u, v)
    rH, JH = whitened_constraint_rows((gH, GxH, GθH))
    @test rH ≈ whitening_factor(GxH) * gH rtol = 1.0e-12
    @test JH ≈ whitening_factor(GxH) * GθH rtol = 1.0e-12
    # degenerate at the rounding scale: 0 when the constraint vanishes, ±Inf otherwise; zero rows either way
    Z = zero(SVector{4, Float64}); ρ0 = (; gradient = 1.0, value = 1.0e-3)
    @test whitened_constraint_rows((1.0e-4, Z, G_θ, I, ρ0)) == (0.0, zero(G_θ))
    r∞, J∞ = whitened_constraint_rows((0.7, Z, G_θ, I, ρ0))
    @test r∞ == Inf && all(iszero, J∞)
    # `whiten` with rows: the identity by dispatch on Σ = I, the same factor otherwise
    s, N = sampson_residual_with_normal(F, u, v)
    @test whiten(s, N, I) === s
    @test whiten(s, N, transpose(G_θ / norm(G_x)) |> SMatrix{1, 9}, I) == (s, SMatrix{1, 9}(transpose(G_θ / norm(G_x))))
    rw, Jw = whiten(s, N, SMatrix{1, 9}(transpose(G_θ / norm(G_x))), Σ)
    @test rw == whiten(s, N, Σ)
    @test vec(Jw) ≈ JΣ rtol = 1.0e-12
    allocs(s, N, Σ) = (whiten(s, N, Σ); @allocated whiten(s, N, Σ))
    @test allocs(s, N, Σ) == 0 && allocs(s, N, I) == 0
    rows_allocs(cd) = (whitened_constraint_rows(cd); @allocated whitened_constraint_rows(cd))
    @test rows_allocs((g, G_x, G_θ, Σ, ρ)) == 0
end
