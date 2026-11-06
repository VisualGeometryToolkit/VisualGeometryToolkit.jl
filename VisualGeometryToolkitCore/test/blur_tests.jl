# One Gaussian blur of an imaging model (cameras/blur.jl): the density N(0, Σ) by its covariance — the round,
# axis-aligned and general forms by their storage, the Statistics accessors, the refusals, convolution as the sum of
# covariances, the push-forward through a linear map (propagate: raw J and a framed map), a sampled kernel's second
# moments, units (m² on a print), eltype conversion and Duals.
@testitem "blur" begin
    using Test
    using StaticArrays: SVector, SMatrix, SDiagonal, @SMatrix
    using LinearAlgebra: Diagonal, det
    using Statistics: cov, var, std
    using VisualGeometryToolkitCore: AbstractBlur, GaussianBlur, convolve, propagate, convert_eltype, ScaledIdentity, CovMat,
        EuclideanMap, PixelFrame, UnlabeledFrame, relabel, has_units, with_unit, mm, m
    using FlexUnits: ustrip
    using Rotations: RotMatrix
    import ForwardDiff

    @testset "forms, storage and the Statistics accessors" begin
        r = GaussianBlur(0.8)
        @test r isa AbstractBlur && cov(r) isa CovMat{2} && cov(r) ≈ SDiagonal(0.64, 0.64) && parent(cov(r)) isa ScaledIdentity{2}
        @test std(r) ≈ SVector(0.8, 0.8) && var(r) ≈ SVector(0.64, 0.64)
        @test parent(cov(convolve(r, GaussianBlur(0.6)))) isa ScaledIdentity{2}                     # two round blurs: round
        @test cov(propagate(@SMatrix([0.8 -0.6; 0.6 0.8]), r)) ≈ cov(r)                             # a rotation keeps a round blur
        @test_throws ArgumentError GaussianBlur(ScaledIdentity{2}(-1.0))
        a = GaussianBlur(0.5, 1.5)
        @test cov(a) == SDiagonal(0.25, 2.25) && parent(cov(a)) isa Diagonal && std(a) == SVector(0.5, 1.5)
        M = @SMatrix [2.0 0.3; 0.3 1.0]
        g = GaussianBlur(M)
        @test parent(cov(g)) === M && var(g) == SVector(2.0, 1.0) && isbitstype(typeof(g)) && isbitstype(typeof(r))
        @test GaussianBlur(CovMat(M)) === g
        @test_throws ArgumentError GaussianBlur(@SMatrix [1.0 0.2; 0.1 1.0])     # not symmetric
        @test_throws ArgumentError GaussianBlur(@SMatrix [1.0 2.0; 2.0 1.0])     # indefinite
        @test_throws ArgumentError GaussianBlur(SDiagonal(-1.0, 1.0))
    end

    @testset "convolution adds the covariances; diagonal stays diagonal; frames meet" begin
        a = GaussianBlur(0.5, 1.5); b = GaussianBlur(1.0)
        @test cov(convolve(a, b)) == SDiagonal(1.25, 3.25) && parent(cov(convolve(a, b))) isa Diagonal
        g = GaussianBlur(@SMatrix [2.0 0.3; 0.3 1.0])
        @test cov(convolve(g, a)) ≈ SMatrix{2, 2}(cov(g)) + SMatrix{2, 2}(cov(a))
        # σ² add: the edge model's σ_print² + σ_cam² for two round blurs
        @test cov(convolve(GaussianBlur(0.3), GaussianBlur(0.4)))[1, 1] ≈ 0.25
        P = PixelFrame{:colmap}
        l = GaussianBlur(relabel(P, cov(b)))
        @test cov(convolve(l, a)) isa CovMat{2, Float64, <:Any, P} && cov(convolve(a, l)) isa CovMat{2, Float64, <:Any, P}   # the unlabeled one takes the other's
        @test_throws MethodError convolve(l, GaussianBlur(relabel(PixelFrame{:opencv}, cov(b))))              # two planes
    end

    @testset "through a linear map: the push-forward J Σ Jᵀ, full" begin
        a = GaussianBlur(0.5, 1.5)
        J = @SMatrix [0.8 -0.6; 0.6 0.8]
        t = propagate(J, a)
        @test parent(cov(t)) isa SMatrix && cov(t) ≈ J * SMatrix{2, 2}(cov(a)) * J' && cov(t)[1, 2] == cov(t)[2, 1]
        @test det(cov(t)) ≈ det(SMatrix{2, 2}(cov(a)))     # a rotation keeps the shape
        S = @SMatrix [1.3 0.4; -0.2 0.9]                   # not a rotation: J Σ Jᵀ rounds asymmetric, stored symmetrized
        @test cov(propagate(S, a)) ≈ S * SMatrix{2, 2}(cov(a)) * S' && cov(propagate(S, a))[1, 2] == cov(propagate(S, a))[2, 1]
        # a framed map: the blur's frame meets the map's source, the result is in its destination
        A, B = PixelFrame{:colmap}, PixelFrame{:opencv}
        E = relabel(A, B, EuclideanMap(RotMatrix{2}(J), SVector(0.0, 0.0)))
        f = propagate(E, GaussianBlur(relabel(A, cov(a))))
        @test cov(f) isa CovMat{2, Float64, <:SMatrix, B} && cov(f) ≈ cov(t)
        @test_throws MethodError propagate(E, GaussianBlur(relabel(B, cov(a))))
    end

    @testset "a singular or near-singular map: a rank-deficient covariance, never refused" begin
        using Random: Xoshiro
        rng = Xoshiro(7)
        refused = 0
        for _ in 1:2000
            v, w = randn(rng, SVector{2}), randn(rng, SVector{2})
            J = SMatrix{2, 2}(v * w') + 1.0e-9 * rand(rng) * randn(rng, SMatrix{2, 2})   # rank 1, or a hair from it
            g = GaussianBlur(SMatrix{2, 2}(let A = randn(rng, SMatrix{2, 2}); A * A' end))
            try
                t = propagate(J, g)
                @test cov(t)[1, 2] == cov(t)[2, 1] && cov(t)[1, 1] >= 0 && cov(t)[2, 2] >= 0
            catch e
                e isa ArgumentError || rethrow()
                refused += 1
            end
        end
        @test refused == 0
        t = propagate(@SMatrix([1.0 2.0; 2.0 4.0]), GaussianBlur(0.5, 1.5))           # exactly rank 1
        @test det(cov(t)) ≈ 0 atol = 1.0e-12
        @test cov(t) ≈ @SMatrix([1.0 2.0; 2.0 4.0]) * SDiagonal(0.25, 2.25) * @SMatrix([1.0 2.0; 2.0 4.0])'
        @test_throws ArgumentError GaussianBlur(-0.5)
        @test_throws ArgumentError GaussianBlur(0.5, -1.0)
        @test cov(GaussianBlur(0.0)) == SDiagonal(0.0, 0.0)                              # a zero blur: the identity
    end

    @testset "the covariance is the sampled kernel's second moments" begin
        M = @SMatrix [2.0 0.3; 0.3 1.0]
        Mi = inv(M)
        h = 0.05; xs = -12:h:12
        w = [exp(-0.5 * (SVector(x, y)' * Mi * SVector(x, y))) for x in xs, y in xs]
        Z = sum(w)
        m2 = sum(w[i, j] * SVector(xs[i], xs[j]) * SVector(xs[i], xs[j])' for i in eachindex(xs), j in eachindex(xs)) / Z
        @test m2 ≈ cov(GaussianBlur(M)) rtol = 1.0e-6
    end

    @testset "units: a print's blur in m²" begin
        p = GaussianBlur(0.02mm, 0.03mm)
        @test has_units(p) && !has_units(GaussianBlur(0.5)) && isbitstype(typeof(p)) && parent(cov(p)) isa Diagonal
        @test ustrip(mm, p) isa GaussianBlur && cov(ustrip(mm, p)) ≈ SDiagonal(0.02^2, 0.03^2)
        @test std(p)[1] ≈ 0.02mm
        r = with_unit(ustrip(mm, p), mm)
        @test has_units(r) && cov(ustrip(mm, r)) ≈ cov(ustrip(mm, p)) && parent(cov(r)) isa Diagonal
        @test has_units(GaussianBlur(0.1mm)) && parent(cov(convolve(GaussianBlur(0.1mm), GaussianBlur(0.2mm)))) isa ScaledIdentity{2}
        @test_throws ArgumentError ustrip(mm, GaussianBlur(0.5))
        @test_throws ArgumentError with_unit(p, mm)
    end

    @testset "eltype conversion keeps the storage; Duals" begin
        for b in (GaussianBlur(0.5, 1.5), GaussianBlur(0.5), GaussianBlur(@SMatrix [2.0 0.3; 0.3 1.0]))
            c = convert_eltype(Float32, b)
            @test eltype(cov(c)) == Float32 && typeof(parent(cov(c))).name === typeof(parent(cov(b))).name
        end
        d = ForwardDiff.derivative(σ -> cov(GaussianBlur(σ))[1, 1], 0.7)
        @test d ≈ 1.4
    end
end
