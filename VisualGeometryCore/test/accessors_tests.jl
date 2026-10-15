# Accessors for state downstream packages used to read by field name
# (RobustVisualGeometry's by-name reads, 2026-09-21): each returns what its
# struct was built from, so a field rename moves the getter beside it and
# nothing downstream.
@testitem "accessors" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Rotations
    using VisualGeometryCore
    using VisualGeometryCore: rotation, translation, homography, division_coeff, radial_rows,
        anchor_svd, scale_s2,
        sqrt_weights, center, FixedModels, ConsensusSet, FixedS2,
        ManifoldLmWorkspace, FundMatManifold7, RadialHomography,
        WelschShape

    @testset "EuclideanMap: rotation, translation" begin
        R = RotMatrix(RotXYZ(0.1, 0.2, 0.3)); t = SVector(1.0, 2.0, 3.0)
        E = EuclideanMap(R, t)
        @test rotation(E) == R && translation(E) == t
        @test E(SVector(0.5, -1.0, 2.0)) ≈ rotation(E) * SVector(0.5, -1.0, 2.0) + translation(E)
    end

    @testset "radial models: homography, division_coeff, radial_rows" begin
        H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
        m = RadialHomography(H, -0.2)
        @test homography(m) == H && division_coeff(m) == -0.2 && center(m) == Point2(0.0, 0.0)
        P = SMatrix{2, 3, Float64}(1, 0, 0, 1, 0, 0)
        rc = RadialCamera(P, (5.0, 6.0))
        @test radial_rows(rc) == P && center(rc) == Point2(5.0, 6.0)
    end

    @testset "FundMatManifold7: anchor_svd" begin
        F₀ = SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0)
        U, σ, V = anchor_svd(FundMatManifold7(F₀))
        @test U' * U ≈ I && V' * V ≈ I && 0 ≤ σ ≤ 1
        s = svdvals(F₀)
        @test σ ≈ s[2] / s[1]
    end

    @testset "FixedModels: length and bounds-checked getindex" begin
        fm = FixedModels{3, Float64}(2, (1.0, 2.0, 0.0))
        @test length(fm) == 2 && fm[1] == 1.0 && fm[2] == 2.0
        @test_throws BoundsError fm[3]
        @test collect(fm) == [fm[i] for i in 1:length(fm)]
    end

    @testset "scale_s2: ConsensusSet, DeployedLoss, FixedS2" begin
        @test scale_s2(ConsensusSet(; count = 2, s2 = 0.25, epoch = 5)) == 0.25
        @test scale_s2(deploy(WelschShape(), 0.37^2; scale_multiple = 1.0)) == 0.37^2
        @test scale_s2(FixedS2(0.5)) == 0.5
    end

    @testset "ManifoldLmWorkspace: buffers" begin
        ws = ManifoldLmWorkspace(5, 6)
        @test length(sqrt_weights(ws)) == 5
        sqrt_weights(ws)[2] = 3.0
        @test sqrt_weights(ws)[2] == 3.0          # the buffer itself, not a copy
    end

    @testset "Hyperplane: coeffs" begin
        l = Line2D(Point2(0.0, 0.0), Point2(1.0, 1.0))
        c = VisualGeometryCore.coeffs(l)
        @test c[SOneTo(2)] == normal(l) && c[3] == -offset(l)
    end
end
