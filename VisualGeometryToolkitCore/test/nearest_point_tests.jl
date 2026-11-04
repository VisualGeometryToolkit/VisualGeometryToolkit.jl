# nearest_point — the least-squares point nearest to N lines.

@testitem "nearest_point" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: m, mm, ustrip

    # the squared distances of x to the lines, weighted
    cost(lines, x, w = ones(length(lines))) =
        sum(w[i] * sum(abs2, (I - normalize(last(l)) * normalize(last(l))') * (x - first(l))) for (i, l) in enumerate(lines))

    rng = Xoshiro(7)
    X = SVector(0.3, -1.2, 4.5)

    @testset "lines through one point meet there" begin
        dirs = [normalize(@SVector(randn(rng, 3))) for _ in 1:5]
        ts = randn(rng, 5)
        lines = [(X + ts[i] * dirs[i]) => 3.7 * dirs[i] for i in 1:5]        # any direction length
        @test norm(nearest_point(lines) - X) < 1.0e-12
        @test norm(nearest_point(csponds(first.(lines), last.(lines))) - X) < 1.0e-12
        two = csponds([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], [SVector(0.0, 0.0, 1.0), SVector(-1.0, 0.0, 1.0)])
        @test nearest_point(two) ≈ SVector(0.0, 0.0, 1.0) atol = 1.0e-14
        # 2-D lines
        l2 = [SVector(0.0, 1.0) => SVector(1.0, 0.0), SVector(2.0, 0.0) => SVector(0.0, 1.0), SVector(0.0, -1.0) => SVector(1.0, 1.0)]
        @test nearest_point(l2) ≈ SVector(2.0, 1.0) atol = 1.0e-12
    end

    @testset "noisy lines: the least-squares minimizer" begin
        lines = map(1:40) do _
            d = normalize(@SVector(randn(rng, 3)))
            (X + 2randn(rng) * d + 0.01 * @SVector(randn(rng, 3))) => d
        end
        x = nearest_point(lines)
        @test norm(x - X) < 0.02
        # stationary: no coordinate perturbation lowers the cost
        for k in 1:3, s in (-1, 1)
            @test cost(lines, x) <= cost(lines, x + s * 1.0e-6 * SVector(ntuple(j -> j == k ? 1.0 : 0.0, 3)))
        end
    end

    @testset "invariance under translating every line" begin
        lines = [(X + randn(rng) * normalize(d) + 0.05 * @SVector(randn(rng, 3))) => d for d in (@SVector(randn(rng, 3)) for _ in 1:12)]
        x = nearest_point(lines)
        for t in (SVector(1.0e3, -2.0e3, 5.0e2), SVector(-0.25, 0.5, 7.0))
            @test norm(nearest_point([(first(l) + t) => last(l) for l in lines]) - (x + t)) < 1.0e-9 * max(1, norm(t))
        end
    end

    @testset "weights" begin
        lines = [SVector(0.0, 0.0, 0.0) => SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0) => SVector(0.0, 0.0, 1.0),
            SVector(0.0, 0.0, 0.0) => SVector(0.0, 1.0, 0.0), SVector(5.0, 5.0, 5.0) => SVector(1.0, 1.0, 0.0)]
        w = [1.0, 2.0, 0.5, 3.0]
        cs = withattrs(csponds(first.(lines), last.(lines)); weights = w)
        x = nearest_point(cs)
        for k in 1:3, s in (-1, 1)
            @test cost(lines, x, w) <= cost(lines, x + s * 1.0e-6 * SVector(ntuple(j -> j == k ? 1.0 : 0.0, 3)), w)
        end
        # a zero weight drops the line; integer weights scale the same answer
        @test nearest_point(withattrs(csponds(first.(lines), last.(lines)); weights = [1.0, 1.0, 1.0, 0.0])) ≈ nearest_point(lines[1:3]) atol = 1.0e-12
        @test nearest_point(withattrs(csponds(first.(lines), last.(lines)); weights = 2w)) ≈ x atol = 1.0e-12
        @test_throws ArgumentError nearest_point(withattrs(csponds(first.(lines), last.(lines)); weights = [1.0, 0.0, 0.0, 0.0]))
        @test_throws ArgumentError nearest_point(withattrs(csponds(first.(lines), last.(lines)); weights = [1.0, -1.0, 1.0, 1.0]))
    end

    @testset "degenerate: parallel lines, a zero direction, too few lines" begin
        d = SVector(0.0, 0.6, 0.8)
        @test_throws ArgumentError nearest_point([SVector(0.0, 0.0, 0.0) => d, SVector(1.0, 0.0, 0.0) => d, SVector(0.0, 2.0, 0.0) => -2d])
        @test_throws ArgumentError nearest_point([SVector(0.0, 0.0, 0.0) => d, SVector(1.0, 0.0, 0.0) => zero(d)])
        @test_throws ArgumentError nearest_point([SVector(0.0, 0.0, 0.0) => d])
    end

    @testset "units: lengths in, metres out" begin
        dirs = [normalize(@SVector(randn(rng, 3))) for _ in 1:4]
        lines = [((X + randn(rng) * dirs[i]) .* mm) => dirs[i] for i in 1:4]
        x = nearest_point(lines)
        @test eltype(x) <: VisualGeometryToolkitCore.Quantity
        @test norm(ustrip.(mm, x) - X) < 1.0e-9
        plain = nearest_point([ustrip.(mm, first(l)) => last(l) for l in lines])
        @test norm(plain - X) < 1.0e-9                                       # plain in, plain out, the caller's unit
    end
end
