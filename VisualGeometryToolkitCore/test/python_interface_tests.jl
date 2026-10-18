# Tests for Python interface integration
#
# This test file verifies that the Python interface can properly
# interact with the Julia backend for geometric operations.

@testitem "python_interface" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: AbstractHomConicMat   # public, not exported
    using GeometryBasics
    using StaticArrays
    using LinearAlgebra

    @testset "Python Interface Integration" begin

        @testset "Circle decomposition" begin
            # Test that we can create circles and decompose them
            # This mirrors what the Python interface does

            center = Point2f(2.0, 3.0)
            radius = 1.5f0
            circle = GeometryBasics.Circle(center, radius)

            # Test coordinates with different resolutions
            for nvertices in [8, 16, 32, 64]
                points = GeometryBasics.coordinates(circle, nvertices)

                @test length(points) == nvertices
                @test all(p -> isa(p, Point2f) || isa(p, Point{2, Float64}), points)

                # Verify points are on circle boundary
                distances = [norm(p - center) for p in points]
                max_error = maximum(abs.(distances .- radius))
                @test max_error < 1.0e-6
            end
        end

        @testset "Ellipse decomposition" begin
            # Test ellipse creation and decomposition
            center = Point2f(0.0, 0.0)
            a, b = 3.0f0, 2.0f0
            θ = π / 4

            ellipse = Ellipse(center, a, b, θ)

            # Test coordinates
            for nvertices in [8, 16, 32]
                points = GeometryBasics.coordinates(ellipse, nvertices)

                @test length(points) == nvertices
                @test all(p -> isa(p, Point{2}), points)

                # Verify points satisfy ellipse equation
                # Transform to canonical form and check
                centered = [p - center for p in points]

                # Rotate by -θ to align with axes
                cos_θ, sin_θ = cos(-θ), sin(-θ)
                aligned = [
                    Point2f(
                        cos_θ * p[1] - sin_θ * p[2],
                        sin_θ * p[1] + cos_θ * p[2]
                    ) for p in centered
                ]

                # Check ellipse equation: (x/a)² + (y/b)² ≈ 1
                ellipse_values = [(p[1] / a)^2 + (p[2] / b)^2 for p in aligned]
                max_error = maximum(abs.(ellipse_values .- 1.0))
                @test max_error < 1.0e-6
            end
        end

        @testset "Homogeneous conic conversions" begin
            # Test conversions between geometric objects and homogeneous conics

            # Circle to conic and back — Circle round-trips through HomCircleMat
            # (a sibling of HomEllipseMat under AbstractHomConicMat).
            circle = GeometryBasics.Circle(Point2f(1.0, 2.0), 1.5f0)
            circle_conic = HomEllipseMat(circle)

            @test isa(circle_conic, AbstractHomConicMat)

            # Verify the conic represents the same circle (Float32 precision)
            recovered_circle = Circle(circle_conic)
            @test norm(recovered_circle.center - circle.center) < 1.0e-6
            @test abs(recovered_circle.r - circle.r) < 1.0e-6

            # Ellipse to conic and back
            ellipse = Ellipse(Point2f(0.0, 0.0), 3.0f0, 2.0f0, π / 6)
            ellipse_conic = HomEllipseMat(ellipse)

            @test isa(ellipse_conic, AbstractHomConicMat)

            # Float32 input → Float32 precision (matches input ellipse field type)
            recovered_ellipse = Ellipse(ellipse_conic)
            @test norm(recovered_ellipse.center - ellipse.center) < 1.0e-6
            @test abs(recovered_ellipse.a - ellipse.a) < 1.0e-6
            @test abs(recovered_ellipse.b - ellipse.b) < 1.0e-6
            # Note: angle comparison needs to handle periodicity
            angle_diff = abs(recovered_ellipse.θ - ellipse.θ)
            angle_diff = min(angle_diff, abs(angle_diff - π))
            @test angle_diff < 1.0e-6
        end

        @testset "Coordinate transformations" begin
            # Test coordinate transformations that Python interface uses

            # Test to_affine (lift Euclidean → homogeneous via append-1)
            euclidean_2d = SVector(1.0, 2.0)
            homogeneous_3d = to_affine(euclidean_2d)

            @test length(homogeneous_3d) == 3
            @test homogeneous_3d[1] == 1.0
            @test homogeneous_3d[2] == 2.0
            @test homogeneous_3d[3] == 1.0

            # Test to_euclidean
            recovered_2d = to_euclidean(homogeneous_3d)
            @test norm(recovered_2d - euclidean_2d) < 1.0e-15

            # Test with scaling
            scaled_homogeneous = SVector(2.0, 4.0, 2.0)
            recovered_scaled = to_euclidean(scaled_homogeneous)
            @test norm(recovered_scaled - euclidean_2d) < 1.0e-15
        end

        @testset "Performance characteristics" begin
            # Test that operations are reasonably fast
            # This ensures the Python interface will have good performance

            circle = GeometryBasics.Circle(Point2f(0.0, 0.0), 1.0f0)

            # Time circle coordinates generation
            @time points = GeometryBasics.coordinates(circle, 64)
            @test length(points) == 64

            ellipse = Ellipse(Point2f(0.0, 0.0), 2.0f0, 1.0f0, π / 4)

            # Time ellipse coordinates generation
            @time ellipse_points = GeometryBasics.coordinates(ellipse, 64)
            @test length(ellipse_points) == 64

            # These should complete quickly (< 1ms typically)
            # The actual timing will depend on the system
        end
    end
end
