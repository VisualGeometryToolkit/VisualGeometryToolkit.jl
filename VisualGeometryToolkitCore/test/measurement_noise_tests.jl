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
        sampson_distance, mahalanobis_distance, TransferForm, whitened_residuals!, whitening_factor, CspondNormalization,
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

    @testset "Mahalanobis distance: exact source is the transfer error" begin
        for i in 1:n
            e = norm(warp(b[i]) - u[i])
            @test mahalanobis_distance(H, exact(b[i]), u[i]) ≈ e rtol = 1.0e-10
        end
        # plain points: the unit isotropic distance, unchanged — far below the
        # pixel error on a metre source (it is in the source's units)
        @test mahalanobis_distance(H, b[1], u[1]) == sampson_distance(H, b[1], u[1]) < 1.0e-2 * norm(warp(b[1]) - u[1])
        # the Sampson distance is geometric: an attached covariance does not enter
        @test sampson_distance(H, exact(b[1]), u[1]) == sampson_distance(H, b[1], u[1])
        @test_throws ArgumentError mahalanobis_distance(H, exact(b[1]), exact(u[1]))
        # analytically, the algebraic constraint's G₂ = −c·I: Σ̃ = c²·I
        g, Gx, _ = constraint_with_jacobians(H, b[1], u[1])
        c = -Gx[1, 3]
        @test whitening_factor(Gx, pair_covariance(zero(SMatrix{2, 2, Float64, 4}), I)) ≈ I / abs(c)
    end

    @testset "whitened batch rows: the noise columns, and the plain path" begin
        r0 = zeros(2n); r = zeros(2n)
        whitened_residuals!(r0, H, Csponds(b, u))                          # no columns: the Sampson residual
        @test r0 ≈ reduce(vcat, [sampson_residual(H, b[i], u[i]) for i in 1:n]) rtol = 1.0e-14
        whitened_residuals!(r, H, withattrs(Csponds(b, u); Σ₁ = fill(zero(SMatrix{2, 2, Float64, 4}), n)))
        # an exact source: the algebraic constraint whitened is the transfer error (r = g/|c|)
        tr = [PerspectiveMap()(SMatrix{3, 3}(H) * SVector(b[i][1], b[i][2], 1.0)) - u[i] for i in 1:n]
        @test [norm(r[2i-1:2i]) for i in 1:n] ≈ norm.(tr) rtol = 1.0e-10
        Σ₁ = [SMatrix{2, 2}(1.0e-6, 2.0e-7, 2.0e-7, 3.0e-6) for _ in 1:n]
        Σ₂ = [SMatrix{2, 2}(2.0, 0.1, 0.1, 0.5) for _ in 1:n]
        whitened_residuals!(r, H, withattrs(Csponds(b, u); Σ₁, Σ₂))
        for i in 1:n
            @test norm(r[2i-1:2i]) ≈ mahalanobis_distance(H, Uncertain(b[i], CovMat(Σ₁[i])), Uncertain(u[i], CovMat(Σ₂[i]))) rtol = 1.0e-12
        end
        xs = withattrs(Csponds(b, u); Σ₁, Σ₂)
        # nothing allocated per row: the kernel launch's own objects only (the same at 5 correspondences as at n)
        allocs(r, H, xs) = (whitened_residuals!(r, H, xs); @allocated whitened_residuals!(r, H, xs))
        @test allocs(r, H, xs) == allocs(zeros(10), H, withattrs(Csponds(b[1:5], u[1:5]); Σ₁ = Σ₁[1:5], Σ₂ = Σ₂[1:5]))
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
