#!/usr/bin/env julia
# Test round-trip conversions for Circle and Ellipse conic matrices

@testitem "conic_roundtrip" begin
    using Test
    using VisualGeometryToolkitCore
    using GeometryBasics: Point2, Circle
    using LinearAlgebra
    using StaticArrays: SMatrix, SVector

    @testset "Conic Round-Trip Conversions" begin

        @testset "Circle → HomCircleMat → Circle" begin
            # Test various circles
            circles = [
                Circle(Point2(0.0, 0.0), 1.0),          # Unit circle at origin
                Circle(Point2(5.0, 3.0), 2.5),          # Offset circle
                Circle(Point2(-10.0, 20.0), 7.3),       # Negative coordinates
                Circle(Point2(100.0, 200.0), 50.0),      # Large circle
            ]

            for circle_orig in circles
                # Forward: Circle → HomCircleMat
                Q = HomCircleMat(circle_orig)
                @test Q isa HomCircleMat{Float64}
                @test is_ellipse(Q)

                # Backward: HomCircleMat → Circle
                circle_recovered = Circle(Q)
                @test circle_recovered isa Circle{Float64}

                # Check round-trip accuracy
                @test isapprox(circle_recovered.center[1], circle_orig.center[1], atol = 1.0e-10)
                @test isapprox(circle_recovered.center[2], circle_orig.center[2], atol = 1.0e-10)
                @test isapprox(circle_recovered.r, circle_orig.r, atol = 1.0e-10)

                # Verify conic equation: points on circle satisfy x'Qx = 0
                θ_samples = range(0, 2π, length = 16)
                for θ in θ_samples
                    p = circle_orig.center .+ circle_orig.r .* [cos(θ), sin(θ)]
                    p_hom = [p[1], p[2], 1.0]
                    residual = abs(dot(p_hom, Q * p_hom))
                    @test residual < 1.0e-12
                end
            end
        end

        @testset "Ellipse → HomEllipseMat → Ellipse" begin
            # Test various ellipses
            ellipses = [
                Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0),         # Axis-aligned at origin
                Ellipse(Point2(5.0, 3.0), 4.0, 2.0, π / 4),         # Rotated 45°
                Ellipse(Point2(-10.0, 20.0), 7.0, 3.5, π / 6),      # Rotated 30°
                Ellipse(Point2(100.0, 200.0), 50.0, 30.0, π / 3),    # Large rotated
            ]

            for ellipse_orig in ellipses
                # Forward: Ellipse → HomEllipseMat
                Q = HomEllipseMat(ellipse_orig)
                @test Q isa HomEllipseMat{Float64}
                @test is_ellipse(Q)

                # Backward: HomEllipseMat → Ellipse
                ellipse_recovered = Ellipse(Q)
                @test ellipse_recovered isa Ellipse{Float64}

                # Check round-trip accuracy
                @test isapprox(ellipse_recovered.center[1], ellipse_orig.center[1], atol = 1.0e-10)
                @test isapprox(ellipse_recovered.center[2], ellipse_orig.center[2], atol = 1.0e-10)
                @test isapprox(ellipse_recovered.a, ellipse_orig.a, atol = 1.0e-10)
                @test isapprox(ellipse_recovered.b, ellipse_orig.b, atol = 1.0e-10)

                # Angle comparison needs to handle wrapping (θ is periodic with period π)
                θ_diff = abs(mod(ellipse_recovered.θ - ellipse_orig.θ, π))
                θ_diff = min(θ_diff, π - θ_diff)  # Handle wraparound
                @test θ_diff < 1.0e-10

                # Verify conic equation: points on ellipse satisfy x'Qx = 0
                θ_samples = range(0, 2π, length = 16)
                for θ in θ_samples
                    # Parametric point on ellipse
                    cos_angle = cos(ellipse_orig.θ)
                    sin_angle = sin(ellipse_orig.θ)
                    x_local = ellipse_orig.a * cos(θ)
                    y_local = ellipse_orig.b * sin(θ)
                    x = ellipse_orig.center[1] + cos_angle * x_local - sin_angle * y_local
                    y = ellipse_orig.center[2] + sin_angle * x_local + cos_angle * y_local
                    p_hom = [x, y, 1.0]
                    residual = abs(dot(p_hom, Q * p_hom))
                    @test residual < 1.0e-11
                end
            end
        end

        @testset "HomEllipseMat(::Circle) delegation" begin
            circle = Circle(Point2(3.0, 4.0), 5.0)

            # HomEllipseMat(circle) should return HomCircleMat
            Q = HomEllipseMat(circle)
            @test Q isa HomCircleMat{Float64}
            @test is_ellipse(Q)

            # Should round-trip correctly
            circle_recovered = Circle(Q)
            @test isapprox(circle_recovered.center[1], circle.center[1], atol = 1.0e-10)
            @test isapprox(circle_recovered.center[2], circle.center[2], atol = 1.0e-10)
            @test isapprox(circle_recovered.r, circle.r, atol = 1.0e-10)
        end

        @testset "Circle → HomCircleMat → Ellipse → Circle" begin
            circle = Circle(Point2(2.0, 3.0), 4.0)

            # Circle → HomCircleMat
            Q = HomCircleMat(circle)
            @test Q isa HomCircleMat{Float64}

            # HomCircleMat → Ellipse (should have a ≈ b ≈ r)
            ellipse = Ellipse(Q)
            @test ellipse isa Ellipse{Float64}
            @test isapprox(ellipse.a, ellipse.b, atol = 1.0e-10)
            @test isapprox(ellipse.a, circle.r, atol = 1.0e-10)
            @test isapprox(ellipse.center[1], circle.center[1], atol = 1.0e-10)
            @test isapprox(ellipse.center[2], circle.center[2], atol = 1.0e-10)

            # Ellipse → HomEllipseMat
            Q2 = HomEllipseMat(ellipse)
            @test Q2 isa HomEllipseMat{Float64}

            # HomEllipseMat → Circle (should work since a ≈ b)
            circle_recovered = Circle(Q2)
            @test circle_recovered isa Circle{Float64}
            @test isapprox(circle_recovered.center[1], circle.center[1], atol = 1.0e-10)
            @test isapprox(circle_recovered.center[2], circle.center[2], atol = 1.0e-10)
            @test isapprox(circle_recovered.r, circle.r, atol = 1.0e-10)
        end

        # One classifier, scale- sign- and translation-invariant: an absolute
        # degeneracy guard called large,
        # distant and Float32 ellipses degenerate, and ordering the block's
        # eigenvalues by value read a negated conic's ratio upside down, so the
        # separate `is_ellipse` called negated line pairs and imaginary ellipses
        # real ones.
        @testset "classify_conic on large, distant, Float32 and negated conics" begin
            for Q in (
                    HomCircleMat(Circle(Point2(0.0, 0.0), 1.0e6)),
                    HomCircleMat(Circle(Point2(1.0e6, 0.0), 10.0)),
                    HomCircleMat(Circle(Point2(1.0e6, 0.0), 1.0)),
                    HomCircleMat(Circle(Point2{Float32}(400.0f0, 0.0f0), 5.0f0)),
                    HomEllipseMat(Ellipse(Point2(0.0, 1.0e4), 100.0, 1.0, 0.0)),
                )
                @test classify_conic(Q) isa EllipseTrait
                @test classify_conic(typeof(Q)(Tuple(-Q))) isa EllipseTrait
                @test is_ellipse(Q)
            end
            # A = I, b = −t, c = |t|² + g: g < 0 real, g = 0 a point, g > 0 imaginary
            conic(t, g; k = 1.0) = HomEllipseMat(k * SMatrix{3, 3}(1.0, 0.0, -t[1], 0.0, 1.0, -t[2], -t[1], -t[2], t[1]^2 + t[2]^2 + g))
            for k in (1.0, -3.0e5), t in ((0.0, 0.0), (2.0e3, -7.0e2))
                @test classify_conic(conic(t, -4.0; k)) isa EllipseTrait
                @test classify_conic(conic(t, 0.0; k)) isa DegenerateTrait
                @test classify_conic(conic(t, 4.0; k)) isa DegenerateTrait
                @test !is_ellipse(conic(t, 4.0; k))
            end
            # a singular block: parallel lines x² = 1, negated — not an ellipse
            lines = HomEllipseMat(-SMatrix{3, 3}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
            @test classify_conic(lines) isa ParabolaTrait
            @test !is_ellipse(lines)
            # an elongated point ellipse (κ = 1e6) far from the origin stays degenerate
            A = SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0e-6)
            t = SVector(3.0e4, -1.2e4)
            b = -A * t
            point = HomEllipseMat(SMatrix{3, 3}(A[1, 1], A[2, 1], b[1], A[1, 2], A[2, 2], b[2], b[1], b[2], dot(t, A * t)))
            @test classify_conic(point) isa DegenerateTrait
            # a thin ellipse far along its minor axis has a well-resolved γ: the noise
            # bound is the rounding of γ's terms, not a multiple of |c| alone
            for θ in (0.0, π / 4)
                @test classify_conic(HomEllipseMat(Ellipse(Point2(0.0, 4000.0), 50.0, 0.1, θ))) isa EllipseTrait
            end
            # no absolute scale anywhere, including the normalization
            Q = HomEllipseMat(Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0))
            @test classify_conic(HomEllipseMat{Float64}(Tuple(1.0e-17 * Q))) isa EllipseTrait
        end
    end
end
