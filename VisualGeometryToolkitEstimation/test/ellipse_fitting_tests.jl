# refine_ellipse — the orthogonal-distance ellipse, from the direct_ellipse start.

@testitem "ellipse_fitting" begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using StableRNGs
    using Statistics: mean
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: geom, estimate, retcode
    using VisualGeometryToolkitEstimation

    function _boundary(e, φ)
        s, c = sincos(e.θ)
        return SVector(e.center...) + SMatrix{2, 2}(c, s, -s, c) * SVector(e.a * cos(φ), e.b * sin(φ))
    end
    _points(e, n; σ = 0.0, arc = 2π, rng = StableRNG(1)) =
        [_boundary(e, φ) + σ * randn(rng, SVector{2}) for φ in range(0, arc, n + 1)[1:(end - 1)]]
    _chart(e) = VgtkCore.ellipse_log_shape(e)
    # RMS orthogonal distance of `pts` to `e`, by the fit's own foot point
    function _orms(e, pts)
        c, V, a, b = VgtkEstimation._ellipse_frame(_chart(e))
        return sqrt(mean(p -> first(VgtkEstimation._ellipse_foot(p, c, V, a, b))^2, pts))
    end

    e0 = Ellipse(Point2(120.0, 80.0), 30.0, 12.0, 0.6)

    @testset "the foot point is the closest boundary point" begin
        c, V, a, b = VgtkEstimation._ellipse_frame(_chart(e0))
        rng = StableRNG(2)
        grid = range(0, 2π, 100_001)
        for _ in 1:60
            p = SVector(120.0, 80.0) + 45 * (2 * rand(rng, SVector{2}) .- 1)    # inside, outside, near the centre
            d, n, u = VgtkEstimation._ellipse_foot(p, c, V, a, b)
            @test abs(abs(d) - minimum(φ -> norm(p - _boundary(e0, φ)), grid)) < 1.0e-5
            @test norm(u) ≈ 1                                                  # a boundary parameter
            @test norm(n) ≈ 1
        end
    end

    @testset "noise-free points: the ellipse, to rounding" begin
        r = refine_ellipse(_points(e0, 30))
        @test retcode(r) === :converged
        @test norm(_chart(geom(estimate(r))) - _chart(e0)) < 1.0e-10
        @test r.rms < 1.0e-10
    end

    @testset "noisy points: no worse than the algebraic start, better on an arc" begin
        for arc in (2π, π)
            pts = _points(e0, 60; σ = 0.5, arc)
            r = refine_ellipse(pts)
            @test retcode(r) === :converged
            @test r.rms <= _orms(direct_ellipse(pts), pts) + 1.0e-12
            @test r.rms ≈ _orms(geom(estimate(r)), pts)
            @test r.n == 60 && r.ν == 55
        end
        pts = _points(e0, 60; σ = 0.5, arc = π)
        @test refine_ellipse(pts).rms < 0.97 * _orms(direct_ellipse(pts), pts)
    end

    @testset "a circle: the chart has no singular point there" begin
        circle = Ellipse(Point2(50.0, 40.0), 10.0, 10.0, 0.0)
        r = refine_ellipse(_points(circle, 40; σ = 0.3))
        @test retcode(r) === :converged
        e = geom(estimate(r))
        @test abs(e.a - 10) < 0.3 && abs(e.b - 10) < 0.3
    end

    @testset "an explicit start value" begin
        pts = _points(e0, 40; σ = 0.2)
        r = refine_ellipse(pts, Ellipse(Point2(118.0, 82.0), 25.0, 15.0, 0.4))
        @test retcode(r) === :converged
        @test norm(_chart(geom(estimate(r))) - _chart(geom(estimate(refine_ellipse(pts))))) < 1.0e-6
    end

    @testset "the covariance is calibrated (Monte Carlo NEES ≈ 5)" begin
        nees = map(1:300) do s
            u = estimate(refine_ellipse(_points(e0, 60; σ = 0.5, rng = StableRNG(1000 + s))))
            δ = _chart(geom(u)) - _chart(e0)
            dot(δ, Matrix(VgtkCore.cov(u)) \ δ)
        end
        @test 4.5 < mean(nees) < 5.6
    end

    @testset "argument checks" begin
        @test_throws ArgumentError refine_ellipse(_points(e0, 5))
        @test_throws ArgumentError refine_ellipse([SVector(Float64(i), 2.0i) for i in 1:10])   # no start value
    end
end
