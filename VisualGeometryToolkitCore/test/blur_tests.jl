# One Gaussian blur of an imaging model (cameras/blur.jl): the round, axis-aligned and general forms by their storage,
# the moment matrix, the refusals, convolution as the sum of moments, a linear map's J M Jᵀ (full), a sampled kernel's
# second moments, eltype conversion and the kernel laws.
@testitem "blur" begin
    using Test
    using StaticArrays: SVector, SMatrix, SDiagonal, @SMatrix
    using LinearAlgebra: Diagonal, det
    using VisualGeometryToolkitCore: AbstractBlur, GaussianBlur, moment_matrix, convolve, transform_blur, convert_eltype
    import ForwardDiff

    @testset "forms and storage" begin
        r = GaussianBlur(0.8)
        @test r isa AbstractBlur && moment_matrix(r) ≈ SDiagonal(0.64, 0.64) && moment_matrix(r) isa Diagonal
        a = GaussianBlur(0.5, 1.5)
        @test moment_matrix(a) == SDiagonal(0.25, 2.25)
        M = @SMatrix [2.0 0.3; 0.3 1.0]
        g = GaussianBlur(M)
        @test moment_matrix(g) === M && isbitstype(typeof(g)) && isbitstype(typeof(r))
        @test_throws ArgumentError GaussianBlur(@SMatrix [1.0 0.2; 0.1 1.0])     # not symmetric
        @test_throws ArgumentError GaussianBlur(@SMatrix [1.0 2.0; 2.0 1.0])     # indefinite
        @test_throws ArgumentError GaussianBlur(SDiagonal(-1.0, 1.0))
    end

    @testset "convolution adds the moments; diagonal stays diagonal" begin
        a = GaussianBlur(0.5, 1.5); b = GaussianBlur(1.0)
        @test moment_matrix(convolve(a, b)) == SDiagonal(1.25, 3.25) && moment_matrix(convolve(a, b)) isa Diagonal
        g = GaussianBlur(@SMatrix [2.0 0.3; 0.3 1.0])
        @test moment_matrix(convolve(g, a)) ≈ moment_matrix(g) + SMatrix{2, 2}(moment_matrix(a))
        # σ² add: the edge model's σ_print² + σ_cam² for two round blurs
        @test moment_matrix(convolve(GaussianBlur(0.3), GaussianBlur(0.4)))[1, 1] ≈ 0.25
    end

    @testset "through a linear map: J M Jᵀ, full" begin
        a = GaussianBlur(0.5, 1.5)
        J = @SMatrix [0.8 -0.6; 0.6 0.8]
        t = transform_blur(J, a)
        @test moment_matrix(t) isa SMatrix && moment_matrix(t) ≈ J * SMatrix{2, 2}(moment_matrix(a)) * J' && moment_matrix(t)[1, 2] == moment_matrix(t)[2, 1]
        @test det(moment_matrix(t)) ≈ det(SMatrix{2, 2}(moment_matrix(a)))     # a rotation keeps the shape
    end

    @testset "the moments are the sampled kernel's second moments" begin
        M = @SMatrix [2.0 0.3; 0.3 1.0]
        Mi = inv(M)
        h = 0.05; xs = -12:h:12
        w = [exp(-0.5 * (SVector(x, y)' * Mi * SVector(x, y))) for x in xs, y in xs]
        Z = sum(w)
        m2 = sum(w[i, j] * SVector(xs[i], xs[j]) * SVector(xs[i], xs[j])' for i in eachindex(xs), j in eachindex(xs)) / Z
        @test m2 ≈ moment_matrix(GaussianBlur(M)) rtol = 1.0e-6
    end

    @testset "eltype conversion; Duals" begin
        @test eltype(moment_matrix(convert_eltype(Float32, GaussianBlur(0.5, 1.5)))) == Float32
        @test eltype(moment_matrix(convert_eltype(Float32, GaussianBlur(@SMatrix [2.0 0.3; 0.3 1.0])))) == Float32
        d = ForwardDiff.derivative(σ -> moment_matrix(GaussianBlur(σ))[1, 1], 0.7)
        @test d ≈ 1.4
    end
end
