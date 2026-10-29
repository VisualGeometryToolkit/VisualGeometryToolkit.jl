# The 2D circle as a model: Circle3pt (geometry/solvers/circle_3pt.jl) exact on
# three points and refusing collinear ones; CircleManifold and the
# point-on-circle constraint (geometry/manifolds/circle.jl) against ForwardDiff.

@testitem "Circle3pt: exact on three points, collinear refused" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: Circle3pt, Circle, Point2, solve, sample_size, max_solutions
    rng = Xoshiro(7)
    @test sample_size(Circle3pt()) == 3 && max_solutions(Circle3pt()) == 1
    for _ in 1:200
        c = SVector(4000rand(rng) - 2000, 4000rand(rng) - 2000); r = 10.0^(4rand(rng) - 1)
        pts = [Point2(c + r * SVector(cos(t), sin(t))) for t in 2π * rand(rng) .+ [0, 2π / 3, 4π / 3] .+ 0.5 .* rand(rng, 3)]
        sol = solve(Circle3pt(), pts, [1, 2, 3])
        @test sol.count == 1
        C = first(sol)
        @test C isa Circle{Float64}
        s = norm(c) + r                                       # the coordinates' scale: the solve is exact to rounding of it
        @test norm(SVector(C.center...) - c) < 1.0e-12 * s && abs(C.r - r) < 1.0e-12 * s
    end
    @test solve(Circle3pt(), [Point2(0.0, 0.0), Point2(1.0, 2.0), Point2(3.0, 6.0)], [1, 2, 3]).count == 0      # collinear
    @test solve(Circle3pt(), [Point2(1.0, 1.0), Point2(1.0, 1.0), Point2(3.0, 0.0)], [1, 2, 3]).count == 0      # a repeated point
    @test solve(Circle3pt(), [Point2(0.0, 0.0), Point2(NaN, 1.0), Point2(3.0, 0.0)], [1, 2, 3]).count == 0
    @test solve(Circle3pt(), [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.0)], [1, 2, 3])[1].r ≈ 1.0     # any StaticVector{2}
    # the sample gate refuses near-collinear triples before the solve, at the
    # homography gate's ratio (scale-free: the same verdict at any scale)
    using VisualGeometryToolkitCore: test_sample, thin_ratio
    for s in (1.0e-3, 1.0, 1.0e4)
        @test !test_sample(Circle, s .* [Point2(0.0, 0.0), Point2(100.0, 0.0), Point2(50.0, 0.05)], [1, 2, 3])   # thin_ratio 1e-5
        @test test_sample(Circle, s .* [Point2(0.0, 0.0), Point2(100.0, 0.0), Point2(50.0, 1.0)], [1, 2, 3])     # thin_ratio 1e-2
    end
    @test thin_ratio(SVector(0.0, 0.0), SVector(100.0, 0.0), SVector(50.0, 1.0)) ≈ 0.01
    @test !test_sample(Circle, [Point2(1.0, 1.0), Point2(1.0, 1.0), Point2(3.0, 0.0)], [1, 2, 3])                 # a repeated point
end

@testitem "CircleManifold and the point-on-circle constraint" begin
    using Test, StaticArrays, LinearAlgebra
    using ForwardDiff: ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CircleManifold, Circle, Point2, retract, recenter, retract_jacobian, action_jacobian,
        local_coordinates, tangent_size, model, minimal_manifold, constraint_with_jacobians, ambient_coordinates
    c = Circle(Point2(3.0, -2.0), 5.0)
    ch = minimal_manifold(c)
    @test ch isa CircleManifold && tangent_size(ch) == 3 && model(ch) == c
    v = SVector(0.4, -0.3, 0.2)
    cv = retract(ch, v)
    @test ambient_coordinates(cv) ≈ SVector(3.4, -2.3, 5exp(0.2)) atol = 1.0e-14
    @test retract_jacobian(ch, v) ≈ ForwardDiff.jacobian(w -> ambient_coordinates(retract(ch, w)), v) atol = 1.0e-14
    @test local_coordinates(ch, cv) ≈ v atol = 1.0e-14
    @test model(recenter(ch, v)) == cv
    x = SVector(7.0, 1.0)
    g, G_x, G_θ = constraint_with_jacobians(c, x)
    θ = ambient_coordinates(c)
    gθ(t) = (x[1] - t[1])^2 + (x[2] - t[2])^2 - t[3]^2
    gx(y) = (y[1] - θ[1])^2 + (y[2] - θ[2])^2 - θ[3]^2
    @test g ≈ gθ(θ) && G_θ ≈ ForwardDiff.gradient(gθ, θ) && G_x ≈ ForwardDiff.gradient(gx, x)
    @test action_jacobian(ch, x) ≈ transpose(G_θ) * retract_jacobian(ch) atol = 1.0e-14
    # the Sampson distance is the geometric one to first order: d = 5 + 0.1 → (d² − r²)/(2d)
    y = SVector(3.0, -2.0) + 5.1 * SVector(0.6, 0.8)
    gy, Gy, _ = constraint_with_jacobians(c, y)
    @test gy / norm(Gy) ≈ 0.1 - 0.1^2 / (2 * 5.1) atol = 1.0e-13
end
