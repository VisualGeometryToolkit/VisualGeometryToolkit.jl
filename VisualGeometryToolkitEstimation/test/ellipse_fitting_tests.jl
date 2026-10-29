# The ellipse refit rows — the points' orthogonal distances (the foot point) —
# and refine_ellipse as plain least squares on outlier-free points: the
# maximum-likelihood ellipse, its covariance the one model covariance.

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
    using VisualGeometryToolkitCore: estimate, converged, GaussianShape, ambient_coordinates
    using VisualGeometryToolkitEstimation

    function _boundary(e, φ)
        s, c = sincos(e.θ)
        return SVector(e.center...) + SMatrix{2, 2}(c, s, -s, c) * SVector(e.a * cos(φ), e.b * sin(φ))
    end
    _points(e, n; σ = 0.0, arc = 2π, rng = StableRNG(1)) =
        [Point2{Float64}(_boundary(e, φ) + σ * randn(rng, SVector{2})) for φ in range(0, arc, n + 1)[1:(end - 1)]]
    _chart(e) = VgtkCore.ellipse_log_shape(e)
    # RMS orthogonal distance of `pts` to `e`, by the rows' own foot point
    function _orms(e, pts)
        c, V, a, b = VgtkEstimation._ellipse_frame(_chart(e))
        return sqrt(mean(p -> first(VgtkEstimation._ellipse_foot(SVector{2, Float64}(p), c, V, a, b))^2, pts))
    end
    # plain least squares over every point: Irls{GaussianShape} at a threshold above every residual
    ls(e, pts) = refine_ellipse(e, pts; local_optimization = Irls{GaussianShape}(), threshold = 1.0e3)

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
        pts = _points(e0, 30)
        r = ls(direct_ellipse(pts), pts)
        @test converged(r) && r.lo_retcode === :converged
        @test norm(_chart(estimate(r)) - _chart(e0)) < 1.0e-9
        @test _orms(estimate(r), pts) < 1.0e-9
        # from the exact ellipse, under the defaults, it comes back
        @test norm(_chart(estimate(refine_ellipse(e0, pts))) - _chart(e0)) < 1.0e-9
    end

    @testset "noisy points: no worse than the algebraic start, better on an arc" begin
        for arc in (2π, π)
            pts = _points(e0, 60; σ = 0.5, arc)
            r = ls(direct_ellipse(pts), pts)
            @test r.lo_retcode === :converged
            @test _orms(estimate(r), pts) <= _orms(direct_ellipse(pts), pts) + 1.0e-12
            @test count(r.inlier_mask) == 60
        end
        pts = _points(e0, 60; σ = 0.5, arc = π)
        @test _orms(estimate(ls(direct_ellipse(pts), pts)), pts) < 0.97 * _orms(direct_ellipse(pts), pts)
    end

    @testset "a circle: the chart has no singular point there" begin
        circle = Ellipse(Point2(50.0, 40.0), 10.0, 10.0, 0.0)
        pts = _points(circle, 40; σ = 0.3)
        r = ls(direct_ellipse(pts), pts)
        @test r.lo_retcode === :converged
        e = estimate(r)
        @test abs(e.a - 10) < 0.3 && abs(e.b - 10) < 0.3
    end

    @testset "another start value, the same minimum" begin
        pts = _points(e0, 40; σ = 0.2)
        r = ls(Ellipse(Point2(118.0, 82.0), 25.0, 15.0, 0.4), pts)
        @test r.lo_retcode === :converged
        @test norm(_chart(estimate(r)) - _chart(estimate(ls(direct_ellipse(pts), pts)))) < 1.0e-6
    end

    @testset "the covariance is calibrated (Monte Carlo NEES ≈ 5)" begin
        # s²·solver_covariance on the inliers, in the ambient coordinates (c, Q)
        nees = map(1:300) do s
            pts = _points(e0, 60; σ = 0.5, rng = StableRNG(1000 + s))
            r = ls(direct_ellipse(pts), pts)
            e = estimate(r)
            Σ = r.sigma^2 * VgtkEstimation.solver_covariance(EllipseProblem(pts), e, findall(r.inlier_mask))
            δ = ambient_coordinates(e) - ambient_coordinates(e0)
            dot(δ, Matrix(Σ) \ δ)
        end
        @test 4.5 < mean(nees) < 5.6
    end

    @testset "too few points" begin
        @test refine_ellipse(e0, _points(e0, 4)) === nothing
    end
end
