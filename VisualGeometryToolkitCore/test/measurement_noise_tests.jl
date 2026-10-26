# The measurement-noise rule (conventions/measurement-noise.md): a plain point
# is unit isotropic noise, an `Uncertain` point carries its covariance, an
# `exact` point none; a correspondence's constraint covariance is
# G₁Σ₁G₁ᵀ + G₂Σ₂G₂ᵀ.
@testitem "measurement_noise" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: exact, Uncertain, CovMat, PrecMat, cov, pair_covariance, noise_columns,
        sampson_distance, TransferForm, transfer_whitening!, whitening_factor, CspondNormalization,
        source_map, destination_map, implicit_covariance, minimal_manifold, constraint_with_jacobians

    rng = Xoshiro(11)
    H = HomographyMat(SMatrix{3, 3, Float64}(900.0, 20.0, 0.1, -30.0, 950.0, 0.2, 300.0, 200.0, 1.0))  # metres → pixels
    n = 30
    b = [SVector(0.1 * rand(rng), 0.1 * rand(rng)) for _ in 1:n]
    warp(x) = (h = SMatrix{3, 3}(H) * SVector(x[1], x[2], 1.0); SVector(h[1] / h[3], h[2] / h[3]))
    u = [warp(x) + 0.3 * SVector(randn(rng), randn(rng)) for x in b]

    @testset "exact: a zero CovMat" begin
        e = exact(b[1])
        @test e isa Uncertain
        @test geom(e) === b[1]
        @test iszero(cov(e))
        @test exact(SVector(1.0f0, 2.0f0)).attributes isa CovMat{2, Float32}
    end

    @testset "pair_covariance and noise_columns" begin
        @test pair_covariance(I, I) === I
        Σ = SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0)
        @test pair_covariance(Σ, I) == [Σ zeros(2, 2); zeros(2, 2) I(2)]
        @test noise_columns(b, u) === (;)
        nc = noise_columns(exact.(b), u)
        @test keys(nc) == (:Σ₁,) && all(iszero, nc.Σ₁)
        nc2 = noise_columns(b, [Uncertain(x, PrecMat{2}(σ = 0.5)) for x in u])
        @test keys(nc2) == (:Σ₂,) && all(Σ -> Σ ≈ 0.25I, nc2.Σ₂)
        @test_throws ArgumentError noise_columns(exact.(b), exact.(u))
        cs = withattrs(Csponds(b, u); nc...)
        @test pair_covariance(Csponds(b, u), 3) === I
        @test pair_covariance(cs, 3) == [zeros(2, 4); zeros(2, 2) I(2)]
    end

    @testset "Sampson distance: exact source is the transfer error" begin
        for i in 1:n
            e = norm(warp(b[i]) - u[i])
            @test sampson_distance(H, exact(b[i]), u[i]) ≈ e rtol = 1.0e-10
        end
        # plain points: the unit isotropic distance, unchanged — far below the
        # pixel error on a metre source (it is in the source's units)
        @test sampson_distance(H, b[1], u[1]) < 1.0e-2 * norm(warp(b[1]) - u[1])
        @test_throws ArgumentError sampson_distance(H, exact(b[1]), exact(u[1]))
        # analytically, the algebraic constraint's G₂ = −c·I: Σ̃ = c²·I
        g, Gx, _ = constraint_with_jacobians(H, b[1], u[1])
        c = -Gx[1, 3]
        @test whitening_factor(Gx, pair_covariance(zero(SMatrix{2, 2, Float64, 4}), I)) ≈ I / abs(c)
    end

    @testset "transfer whitening: covariances and the plain fast path" begin
        W = Vector{SMatrix{2, 2, Float64, 4}}(undef, n); W0 = similar(W)
        transfer_whitening!(W0, H, b)
        transfer_whitening!(W, H, b, nothing, nothing)
        @test W == W0
        transfer_whitening!(W, H, b, fill(zero(SMatrix{2, 2, Float64, 4}), n), nothing)
        @test all(w == SMatrix{2, 2, Float64, 4}(I) for w in W)
        Σ₁ = [SMatrix{2, 2}(1.0e-6, 2.0e-7, 2.0e-7, 3.0e-6) for _ in 1:n]
        Σ₂ = [SMatrix{2, 2}(2.0, 0.1, 0.1, 0.5) for _ in 1:n]
        transfer_whitening!(W, H, b, Σ₁, Σ₂)
        for i in 1:n
            _, Gx, _ = constraint_with_jacobians(TransferForm(H), b[i], u[i])
            A = Gx[:, SVector(1, 2)]
            L⁻¹ = W[i]
            @test L⁻¹ * (A * Σ₁[i] * A' + Σ₂[i]) * L⁻¹' ≈ I atol = 1.0e-10
        end
        let W = W, b = b, Σ₁ = Σ₁, Σ₂ = Σ₂
            @test (@allocated transfer_whitening!(W, H, b, Σ₁, Σ₂)) == 0
        end
    end

    @testset "normalization moves the noise with the points" begin
        Σ₂ = [SMatrix{2, 2}(1.0 + 0.1i, 0.2, 0.2, 0.7) for i in 1:n]
        cs = withattrs(Csponds(b, u); Σ₂, score = collect(1.0:n))
        N = CspondNormalization(HomographyMat, cs)
        cn = LinearAlgebra.normalize(N, cs)
        A = destination_map(N).linear
        @test all(cn.Σ₂[i] ≈ A * Σ₂[i] * A' for i in 1:n)
        @test cn.score == cs.score                     # other columns forwarded
        @test !hasattr(cn, :Σ₁)
    end

    @testset "solver covariance with an exact side" begin
        ch = minimal_manifold(H)
        Σx = [[zeros(2, 2) zeros(2, 2); zeros(2, 2) Matrix(1.0I, 2, 2)] for _ in 1:n]
        C = implicit_covariance(ch, Csponds(b, u); Σx)
        @test C !== nothing && all(isfinite, C)
    end
end
