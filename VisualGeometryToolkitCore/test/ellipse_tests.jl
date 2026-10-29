# The 2D ellipse as a model: Ellipse5pt (geometry/solvers/ellipse_5pt.jl) exact
# on five points and refusing collinear samples and non-ellipse conics;
# EllipseManifold and the point-on-ellipse constraint
# (geometry/manifolds/ellipse.jl) against ForwardDiff.

@testitem "Ellipse5pt: exact on five points, degenerate and non-ellipse conics refused" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: Ellipse5pt, Ellipse, Point2, solve, sample_size, max_solutions, boundary_point
    rng = Xoshiro(7)
    @test sample_size(Ellipse5pt()) == 5 && max_solutions(Ellipse5pt()) == 1
    for _ in 1:200
        c = SVector(4000rand(rng) - 2000, 4000rand(rng) - 2000); a = 10.0^(3rand(rng)); b = a * (0.2 + 0.8rand(rng))
        e₀ = Ellipse(Point2(c...), a, b, π * rand(rng))
        pts = [boundary_point(e₀, t) for t in 2π * rand(rng) .+ (0:4) .* (2π / 5) .+ 0.3 .* rand(rng, 5)]
        sol = solve(Ellipse5pt(), pts, 1:5)
        @test sol.count == 1
        e = first(sol)
        @test e isa Ellipse{Float64}
        s = norm(c) + a                                       # the coordinates' scale
        @test norm(SVector(e.center...) - c) < 1.0e-9 * s && abs(e.a - a) < 1.0e-9 * s && abs(e.b - b) < 1.0e-9 * s
        @test abs(sin(e.θ - e₀.θ)) < 1.0e-7 * a / (a - b)    # the axis direction, conditioned by the eccentricity
    end
    # a hyperbola (xy = 1), a parabola (y = x²), a line pair: no ellipse
    @test solve(Ellipse5pt(), [Point2(t, 1 / t) for t in (0.5, 1.0, 2.0, -1.0, -3.0)], 1:5).count == 0
    @test solve(Ellipse5pt(), [Point2(t, t^2) for t in (-2.0, -1.0, 0.0, 1.0, 3.0)], 1:5).count == 0
    @test solve(Ellipse5pt(), [Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(2.0, 0.0), Point2(0.0, 1.0), Point2(1.0, 2.0)], 1:5).count == 0
    @test solve(Ellipse5pt(), [Point2(0.0, 0.0), Point2(NaN, 1.0), Point2(3.0, 0.0), Point2(1.0, 1.0), Point2(2.0, 3.0)], 1:5).count == 0
    @test solve(Ellipse5pt(), [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(0.0, -1.0), SVector(√0.5, √0.5)], 1:5)[1].a ≈ 1.0   # any StaticVector{2}
    # the sample gate refuses a sample with a (near-)collinear triple before the
    # solve, at the homography gate's ratio (scale-free: the same verdict at any scale)
    using VisualGeometryToolkitCore: test_sample
    good = [Point2(0.0, 0.0), Point2(100.0, 0.0), Point2(50.0, 1.0), Point2(10.0, 60.0), Point2(80.0, 50.0)]
    for s in (1.0e-3, 1.0, 1.0e4)
        @test !test_sample(Ellipse, s .* [Point2(0.0, 0.0), Point2(100.0, 0.0), Point2(50.0, 0.05), Point2(10.0, 60.0), Point2(80.0, 50.0)], 1:5)   # thin_ratio 1e-5
        @test test_sample(Ellipse, s .* good, 1:5)                                                                                        # thin_ratio ≥ 1e-2
    end
    @test !test_sample(Ellipse, [good[1:4]; good[2:2]], 1:5)                                                                              # a repeated point
end

@testitem "EllipseManifold and the point-on-ellipse constraint" begin
    using Test, StaticArrays, LinearAlgebra
    using ForwardDiff: ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EllipseManifold, Ellipse, Circle, Point2, retract, recenter, retract_jacobian, action_jacobian,
        local_coordinates, tangent_size, model, minimal_manifold, constraint_with_jacobians, ambient_coordinates, ellipse_log_shape
    e = Ellipse(Point2(3.0, -2.0), 5.0, 3.0, 0.7)
    ch = minimal_manifold(e)
    @test ch isa EllipseManifold && tangent_size(ch) == 5 && model(ch) == e
    v = SVector(0.4, -0.3, 0.2, -0.1, 0.05)
    ev = retract(ch, v)
    @test ellipse_log_shape(ev) ≈ ellipse_log_shape(e) + v atol = 1.0e-14
    @test local_coordinates(ch, ev) ≈ v atol = 1.0e-14
    @test model(recenter(ch, v)) == ev
    for w in (v, zero(v))
        @test retract_jacobian(ch, w) ≈ ForwardDiff.jacobian(u -> ambient_coordinates(retract(ch, u)), w) atol = 1.0e-12
    end
    # at a circle, where (a, b, θ) swaps its axes, the chart is smooth (central
    # differences: ForwardDiff through the eigenvectors of a repeated eigenvalue is NaN)
    cc = minimal_manifold(Ellipse(Circle(Point2(1.0, 2.0), 4.0)))
    h = 1.0e-6
    Jc = hcat([(ambient_coordinates(retract(cc, h * I[1:5, k])) - ambient_coordinates(retract(cc, -h * I[1:5, k]))) / 2h for k in 1:5]...)
    @test retract_jacobian(cc) ≈ Jc atol = 1.0e-9
    x = SVector(7.0, 1.0)
    g, G_x, G_θ = constraint_with_jacobians(e, x)
    θ = ambient_coordinates(e)
    gθ(t) = (d = x - SVector(t[1], t[2]); dot(d, SMatrix{2, 2}(t[3], t[4], t[4], t[5]) * d) - 1)
    gx(y) = (d = y - SVector(θ[1], θ[2]); dot(d, SMatrix{2, 2}(θ[3], θ[4], θ[4], θ[5]) * d) - 1)
    @test g ≈ gθ(θ) && G_θ ≈ ForwardDiff.gradient(gθ, θ) && G_x ≈ ForwardDiff.gradient(gx, x)
    @test abs(gx(SVector(e.center...) + 5.0 * SVector(cos(0.7), sin(0.7)))) < 1.0e-14       # the major vertex is on it
    @test action_jacobian(ch, x) ≈ transpose(G_θ) * retract_jacobian(ch) atol = 1.0e-14
    # the Sampson distance is the geometric one to first order: 0.1 off the major vertex
    y = SVector(e.center...) + 5.1 * SVector(cos(0.7), sin(0.7))
    gy, Gy, _ = constraint_with_jacobians(e, y)
    @test gy / norm(Gy) ≈ 0.1 - 0.1^2 / (2 * 5.1) atol = 1.0e-13
    # at a circle it is the circle's constraint scaled by 1/r², the circle's Sampson distance
    c = Circle(Point2(3.0, -2.0), 5.0)
    gc, Gc, _ = constraint_with_jacobians(c, x)
    ge, Ge, _ = constraint_with_jacobians(Ellipse(c), x)
    @test ge ≈ gc / 25 && ge / norm(Ge) ≈ gc / norm(Gc)
end
