# The structured storage of the statistical forms (base/types.jl): a CovMat / PrecMat HOLDS a static matrix — an SMatrix
# (full), an SDiagonal (diagonal) or a ScaledIdentity (λ·I, isotropic) — and reads as its full N × N matrix. It keeps its
# storage under inv and relabel; convert and promote give the coarser storage (isotropic ⊂ diagonal ⊂ full), never a finer;
# propagate through a map gives the full form; a unit change gives an isotropic form's diagonal. Structured arithmetic is the
# matrix's own (`parent`); arithmetic on the form escapes to an SMatrix, as before.
@testitem "diagonal statistical forms" begin
    using Test
    using StaticArrays: SVector, SMatrix, SDiagonal, @SMatrix
    using LinearAlgebra: cholesky, Diagonal
    import Rotations
    using VisualGeometryToolkitCore: CovMat, PrecMat, DiagonalCovMat, DiagonalPrecMat, IsotropicCovMat, IsotropicPrecMat, FullCovMat,
        ScaledIdentity, UnlabeledFrame, Uncertain, EuclideanMap, propagate, relabel, cov, prec, geom, with_unit, mm, m, ImageFrame
    import FlexUnits

    D = CovMat(SDiagonal(4.0, 9.0))
    F = CovMat(@SMatrix [4.0 1.0; 1.0 9.0])
    S = CovMat(ScaledIdentity{2}(0.25))

    @testset "the storage is the held matrix" begin
        @test D isa DiagonalCovMat{2, Float64} && parent(D) === SDiagonal(4.0, 9.0) && isbitstype(typeof(D))
        @test S isa IsotropicCovMat{2, Float64} && parent(S) === ScaledIdentity{2}(0.25) && isbitstype(typeof(S))
        @test F isa FullCovMat{2, Float64} && parent(F) === @SMatrix([4.0 1.0; 1.0 9.0])
        @test D == SMatrix{2, 2}(4.0, 0.0, 0.0, 9.0) && Tuple(D) == (4.0, 0.0, 0.0, 9.0) && D[2, 1] == 0.0 && D[4] == 9.0
        @test S == SMatrix{2, 2}(0.25, 0.0, 0.0, 0.25) && Tuple(S) == (0.25, 0.0, 0.0, 0.25)
        @test CovMat{2, Float32}(SDiagonal(1.0, 2.0)) isa DiagonalCovMat{2, Float32}
        @test CovMat{2}(σ = 0.5) isa IsotropicCovMat{2, Float64} && PrecMat{2}(σ = (0.5, 2.0)) isa DiagonalPrecMat{2, Float64}
        @test CovMat{2, Float64}(4.0, 1.0, 1.0, 9.0) isa FullCovMat{2, Float64}          # entries: the full form, as before
        @test hash(D) == hash(SMatrix(D)) && isequal(D, CovMat(SMatrix(D)))
    end

    @testset "kept: inv, relabel, eltype; structured arithmetic on the matrix" begin
        @test inv(D) isa DiagonalPrecMat{2, Float64} && inv(D) ≈ inv(SMatrix(D)) && inv(inv(D)) ≈ D
        @test inv(S) isa IsotropicPrecMat{2, Float64} && inv(S) ≈ 4.0 * SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
        @test relabel(ImageFrame{:left}, D) isa CovMat{2, Float64, typeof(parent(D)), ImageFrame{:left}}
        @test CovMat{2, Float32}(D) isa DiagonalCovMat{2, Float32} && CovMat{2, Float32}(S) isa IsotropicCovMat{2, Float32}
        @test parent(D) + parent(D) isa Diagonal && parent(S) + parent(S) isa ScaledIdentity && parent(S) + parent(D) isa Diagonal
        @test 2.0 * D isa SMatrix && D + F isa SMatrix                                     # on the form: an SMatrix, as before
        @test cholesky(D).L ≈ SMatrix{2, 2}(2.0, 0.0, 0.0, 3.0)
        u = Uncertain(SVector(1.0, 2.0), D)
        @test cov(u) === D && prec(u) isa DiagonalPrecMat
    end

    @testset "coarser: propagate, convert, promote; never finer" begin
        R = Rotations.RotMatrix{2}(0.3)
        P = propagate(EuclideanMap(R, SVector(0.0, 0.0)), D)
        @test P isa FullCovMat{2, Float64} && P ≈ SMatrix(R) * SMatrix(D) * SMatrix(R)'
        @test propagate(EuclideanMap(R, SVector(0.0, 0.0)), inv(D)) isa PrecMat{2, Float64, SMatrix{2, 2, Float64, 4}}
        @test propagate(EuclideanMap(R, SVector(0.0, 0.0)), S) ≈ SMatrix(S)             # isotropic: a rotation leaves it unchanged
        @test convert(typeof(F), D) isa typeof(F) && convert(typeof(F), D) == D
        @test convert(typeof(D), S) isa typeof(D) && convert(typeof(D), S) == S && convert(typeof(F), S) == S
        @test_throws MethodError convert(typeof(D), F)
        @test_throws MethodError convert(typeof(S), D)
        @test promote_type(typeof(D), typeof(F)) == typeof(F) && promote_type(typeof(S), typeof(D)) == typeof(D)
        @test eltype([D, F]) == typeof(F) && eltype([S, D]) == typeof(D)                  # a literal of mixed storage: one concrete type
        v = Vector{typeof(F)}(); push!(v, D); push!(v, S)
        @test v[1] == D && v[2] == S && eltype(v) == typeof(F)
    end

    @testset "a unit change: the storage kept, an isotropic form's diagonal (an uncertain pose)" begin
        Σ = CovMat(SDiagonal(1.0e-4, 2.0e-4, 3.0e-4, 1.0e-6, 4.0e-6, 9.0e-6))
        X = Uncertain(EuclideanMap(Rotations.RotXYZ(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0) .* m), Σ)
        p = getfield(FlexUnits.ustrip(mm, X), :attributes)
        @test p isa DiagonalCovMat{6, Float64} && p ≈ SDiagonal(1.0e-4, 2.0e-4, 3.0e-4, 1.0, 4.0, 9.0)
        @test cov(with_unit(FlexUnits.ustrip(mm, X), mm)) ≈ Σ
        Xi = Uncertain(geom(X), CovMat(ScaledIdentity{6}(1.0e-6)))
        @test getfield(FlexUnits.ustrip(mm, Xi), :attributes) isa DiagonalCovMat{6, Float64}   # angles and lengths scale apart
    end

    @testset "ScaledIdentity" begin
        @test ScaledIdentity{2}(3.0) + SDiagonal(1.0, 2.0) == SDiagonal(4.0, 5.0) && inv(ScaledIdentity{3}(2.0)) isa ScaledIdentity{3}
        @test 2.0 * ScaledIdentity{2}(3.0) isa ScaledIdentity{2} && ScaledIdentity{2}(3.0) * SMatrix{2, 2}(1.0, 2.0, 3.0, 4.0) isa SMatrix
        @test Diagonal(ScaledIdentity{2}(3.0)) == SDiagonal(3.0, 3.0) && isbitstype(ScaledIdentity{2, Float64})
    end
end
