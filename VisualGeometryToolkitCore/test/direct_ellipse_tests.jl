# DirectEllipse — the ellipse-specific algebraic fit — and the log-shape chart.

@testitem "DirectEllipse" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ReturnCode, ClosedForm, NoRule

    # the closed form's model (its FitResult's `u`; `nothing` when the points admit no ellipse)
    ellipse_fit(points) = solve(EstimationProblem(Ellipse, points), DirectEllipse()).u

    include(joinpath(pkgdir(VgtkCore), "test", "fixtures", "ellipse_points.jl"))   # `ellipse_boundary_points`: shared with Estimation and the golden records
    _close(e, f; atol) = norm(VgtkCore.ellipse_log_shape(e) - VgtkCore.ellipse_log_shape(f)) < atol

    e0 = Ellipse(Point2(120.0, 80.0), 30.0, 12.0, 0.6)

    @testset "exact points: the ellipse" begin
        @test _close(ellipse_fit(ellipse_boundary_points(MersenneTwister(1), e0, 40)), e0; atol = 1.0e-9)
        @test _close(ellipse_fit(ellipse_boundary_points(MersenneTwister(1), e0, 5)), e0; atol = 1.0e-8)           # the minimal count
        @test _close(ellipse_fit(ellipse_boundary_points(MersenneTwister(1), e0, 30; arc = π / 2)), e0; atol = 1.0e-8)   # a quarter arc
        circle = Ellipse(Point2(-3.0, 7.0), 4.0, 4.0, 0.0)
        @test _close(ellipse_fit(ellipse_boundary_points(MersenneTwister(1), circle, 20)), circle; atol = 1.0e-9)
        # the result record: one closed-form step over every point
        r = solve(EstimationProblem(Ellipse, ellipse_boundary_points(MersenneTwister(1), e0, 40)), DirectEllipse())
        @test r.u == ellipse_fit(ellipse_boundary_points(MersenneTwister(1), e0, 40))
        @test r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm
        @test r.stats.nsteps == 0 && isnan(r.rms) && r.n == 40
    end

    @testset "noisy points still give an ellipse near the truth" begin
        e = ellipse_fit(ellipse_boundary_points(MersenneTwister(1), e0, 60; σ = 0.5))
        @test e isa Ellipse
        @test norm(SVector(e.center...) - SVector(e0.center...)) < 0.5
        @test abs(e.a - e0.a) < 1 && abs(e.b - e0.b) < 1
    end

    @testset "degenerate input" begin
        @test_throws ArgumentError ellipse_fit([SVector(Float64(i), 0.0) for i in 1:4])
        @test ellipse_fit([SVector(Float64(i), 2.0i) for i in 1:10]) === nothing   # collinear
        # no ellipse is a result, not an exception: no model, (InitialFailure, NoRule), no data used
        r = solve(EstimationProblem(Ellipse, [SVector(Float64(i), 2.0i) for i in 1:10]), DirectEllipse())
        @test r.u === nothing && r.retcode === ReturnCode.InitialFailure && r.stop_rule === NoRule && r.n == 0
    end

    @testset "the log-shape chart round-trips and is smooth at a circle" begin
        for e in (e0, Ellipse(Point2(0.0, 0.0), 5.0, 5.0, 0.0), Ellipse(Point2(1.0, 2.0), 9.0, 0.5, 2.9))
            f = VgtkCore.ellipse_from_log_shape(VgtkCore.ellipse_log_shape(e))
            @test SVector(f.center...) ≈ SVector(e.center...)
            @test f.a ≈ e.a && f.b ≈ e.b
            e.a > e.b && @test mod(f.θ - e.θ + π / 2, π) ≈ π / 2              # θ up to the π period
        end
        # a circle has S = log(r)·I whatever its θ: no branch cut in the chart
        m1 = VgtkCore.ellipse_log_shape(Ellipse(Point2(0.0, 0.0), 5.0, 5.0, 0.0))
        m2 = VgtkCore.ellipse_log_shape(Ellipse(Point2(0.0, 0.0), 5.0, 5.0, 1.3))
        @test m1 ≈ m2
    end
end
