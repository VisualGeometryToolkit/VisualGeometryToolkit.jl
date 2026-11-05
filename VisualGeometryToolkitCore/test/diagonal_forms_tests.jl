# The diagonal storage of the statistical forms (base/types.jl): a CovMat / PrecMat of N stored entries reads as its full
# N × N matrix, stays diagonal under what keeps it so (inv, a scalar, a sum of diagonal forms, a unit change), becomes the
# full form under what mixes axes (propagate) and under convert / promote (mixed containers), never the reverse.
@testitem "diagonal statistical forms" begin
    using Test
    using StaticArrays: SVector, SMatrix, SDiagonal, @SMatrix
    using LinearAlgebra: cholesky, Diagonal
    import Rotations
    using VisualGeometryToolkitCore: CovMat, PrecMat, DiagonalCovMat, DiagonalPrecMat, UnlabeledFrame, Uncertain, EuclideanMap,
        propagate, relabel, cov, prec, with_unit, mm, m, ImageFrame
    import FlexUnits

    D = CovMat(SDiagonal(4.0, 9.0))
    F = CovMat(@SMatrix [4.0 1.0; 1.0 9.0])
    full_of(x) = CovMat{2, Float64, 4, UnlabeledFrame{2}}

    @testset "storage and reading" begin
        @test D isa DiagonalCovMat{2, Float64} && D isa CovMat{2, Float64, 2} && isbitstype(typeof(D))
        @test D == SMatrix{2, 2}(4.0, 0.0, 0.0, 9.0) && Tuple(D) == (4.0, 0.0, 0.0, 9.0)
        @test D[1, 1] == 4.0 && D[2, 1] == 0.0 && D[1, 2] == 0.0 && D[4] == 9.0 && D[3] == 0.0
        @test getfield(D, :data) == (4.0, 9.0)
        @test CovMat{2, Float32}(SDiagonal(1.0, 2.0)) isa DiagonalCovMat{2, Float32}
        @test_throws ArgumentError CovMat{2, Float64, 3, UnlabeledFrame{2}}((1.0, 2.0, 3.0))
        @test hash(D) == hash(SMatrix(D)) && isequal(D, CovMat(SMatrix(D)))
        # the existing surface is unchanged: the full form is L = N * N
        @test F isa CovMat{2, Float64, 4} && CovMat{2}(σ = 0.5) isa CovMat{2, Float64, 4}
    end

    @testset "kept diagonal: inv, scalars, sums of diagonal forms, relabel, eltype" begin
        Λ = inv(D)
        @test Λ isa DiagonalPrecMat{2, Float64} && Λ ≈ inv(SMatrix(D)) && inv(Λ) ≈ D
        @test 2.0 * D isa DiagonalCovMat && D * 2 isa DiagonalCovMat && D / 2 isa DiagonalCovMat && 2.0 * D ≈ 2 * SMatrix(D)
        @test D + D isa DiagonalCovMat && D + D ≈ 2 * SMatrix(D)
        @test relabel(ImageFrame{:left}, D) isa CovMat{2, Float64, 2, ImageFrame{:left}}
        @test CovMat{2, Float32}(D) isa DiagonalCovMat{2, Float32}
        @test cholesky(D).L ≈ SMatrix{2, 2}(2.0, 0.0, 0.0, 3.0)
        u = Uncertain(SVector(1.0, 2.0), D)
        @test cov(u) === D && prec(u) isa DiagonalPrecMat
    end

    @testset "made full: propagate, mixed sums, convert, promote; never the reverse" begin
        R = Rotations.RotMatrix{2}(0.3)
        P = propagate(EuclideanMap(R, SVector(0.0, 0.0)), D)
        @test P isa CovMat{2, Float64, 4} && P ≈ SMatrix(R) * SMatrix(D) * SMatrix(R)'
        @test propagate(EuclideanMap(R, SVector(0.0, 0.0)), inv(D)) isa PrecMat{2, Float64, 4}
        @test D + F isa CovMat{2, Float64, 4} && F + D isa CovMat{2, Float64, 4} && D + F ≈ SMatrix(D) + SMatrix(F)
        @test convert(typeof(F), D) isa typeof(F) && convert(typeof(F), D) == D
        @test_throws MethodError convert(typeof(D), F)
        @test promote_type(typeof(D), typeof(F)) == typeof(F)
        @test eltype([D, F]) == typeof(F)                                  # a literal of mixed storage is one concrete type
        v = Vector{typeof(F)}(); push!(v, D)
        @test v[1] == D && v[1] isa typeof(F)
    end

    @testset "a unit change keeps the storage (an uncertain pose: rotation block in rad², translation in m²)" begin
        Σ = CovMat(SDiagonal(1.0e-4, 2.0e-4, 3.0e-4, 1.0e-6, 4.0e-6, 9.0e-6))
        X = Uncertain(EuclideanMap(Rotations.RotXYZ(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0) .* m), Σ)
        p = FlexUnits.ustrip(mm, X)
        @test getfield(p, :attributes) isa DiagonalCovMat{6, Float64}
        @test cov(p) ≈ SDiagonal(1.0e-4, 2.0e-4, 3.0e-4, 1.0, 4.0, 9.0)      # the translation block in mm², the rotation's as it was
        @test cov(with_unit(p, mm)) ≈ Σ
    end
end
