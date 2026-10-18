#!/usr/bin/env julia

@testitem "coordinate_conversion" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using StaticArrays
    using GeometryBasics

    @testset "Coordinate Conversion" begin

        @testset "to_affine function" begin
            # Test 2D to 3D conversion
            v2d = SVector(3.0, 4.0)
            v3d = to_affine(v2d)
            @test v3d == SVector(3.0, 4.0, 1.0)
            @test v3d isa SVector{3, Float64}

            # Test 1D to 2D conversion
            v1d = SVector(5.0)
            v2d_from_1d = to_affine(v1d)
            @test v2d_from_1d == SVector(5.0, 1.0)
            @test v2d_from_1d isa SVector{2, Float64}

            # Test 3D to 4D conversion
            v3d_input = SVector(1.0, 2.0, 3.0)
            v4d = to_affine(v3d_input)
            @test v4d == SVector(1.0, 2.0, 3.0, 1.0)
            @test v4d isa SVector{4, Float64}

            # Test with Point2 — to_affine preserves the input wrapper type
            p2d = Point2(2.0, 3.0)
            p3d = to_affine(p2d)
            @test p3d == SVector(2.0, 3.0, 1.0)
            @test p3d isa Point{3, Float64}

            # Test with different numeric types
            v_int = SVector(1, 2)
            v_homog_int = to_affine(v_int)
            @test v_homog_int == SVector(1, 2, 1)
            @test v_homog_int isa SVector{3, Int}

            # Test with Float32
            v_f32 = SVector(1.0f0, 2.0f0)
            v_homog_f32 = to_affine(v_f32)
            @test v_homog_f32 == SVector(1.0f0, 2.0f0, 1.0f0)
            @test v_homog_f32 isa SVector{3, Float32}
        end

        @testset "to_euclidean function" begin
            # Test 3D to 2D conversion with w=1
            v3d = SVector(3.0, 4.0, 1.0)
            v2d = to_euclidean(v3d)
            @test v2d == SVector(3.0, 4.0)
            @test v2d isa SVector{2, Float64}

            # Test 2D to 1D conversion with w=1
            v2d_input = SVector(5.0, 1.0)
            v1d = to_euclidean(v2d_input)
            @test v1d == SVector(5.0)
            @test v1d isa SVector{1, Float64}

            # Test 4D to 3D conversion with w=1
            v4d = SVector(1.0, 2.0, 3.0, 1.0)
            v3d_result = to_euclidean(v4d)
            @test v3d_result == SVector(1.0, 2.0, 3.0)
            @test v3d_result isa SVector{3, Float64}

            # Test perspective division with w≠1
            v3d_scaled = SVector(6.0, 8.0, 2.0)
            v2d_perspective = to_euclidean(v3d_scaled)
            @test v2d_perspective ≈ SVector(3.0, 4.0)
            @test v2d_perspective isa SVector{2, Float64}

            # Test with different numeric types
            v_int = SVector(4, 6, 2)
            v_eucl_int = to_euclidean(v_int)
            @test v_eucl_int == SVector(2.0, 3.0)  # Note: division creates Float64
            @test v_eucl_int isa SVector{2, Float64}

            # Test with Float32
            v_f32 = SVector(2.0f0, 4.0f0, 2.0f0)
            v_eucl_f32 = to_euclidean(v_f32)
            @test v_eucl_f32 ≈ SVector(1.0f0, 2.0f0)
            @test v_eucl_f32 isa SVector{2, Float32}
        end

        @testset "Roundtrip conversion" begin
            # Test 2D roundtrip
            original_2d = SVector(3.14, 2.71)
            roundtrip_2d = to_euclidean(to_affine(original_2d))
            @test roundtrip_2d ≈ original_2d atol = 1.0e-15

            # Test 1D roundtrip
            original_1d = SVector(42.0)
            roundtrip_1d = to_euclidean(to_affine(original_1d))
            @test roundtrip_1d ≈ original_1d atol = 1.0e-15

            # Test 3D roundtrip
            original_3d = SVector(1.0, -2.5, 3.7)
            roundtrip_3d = to_euclidean(to_affine(original_3d))
            @test roundtrip_3d ≈ original_3d atol = 1.0e-15

            # Test Point2 roundtrip
            original_point = Point2(1.5, -0.8)
            roundtrip_point = Point2(to_euclidean(to_affine(original_point)))
            @test roundtrip_point ≈ original_point atol = 1.0e-15
        end

        @testset "Edge cases" begin
            # Test zero vector
            zero_2d = SVector(0.0, 0.0)
            zero_3d = to_affine(zero_2d)
            @test zero_3d == SVector(0.0, 0.0, 1.0)
            zero_back = to_euclidean(zero_3d)
            @test zero_back == zero_2d

            # Test very small numbers
            tiny_2d = SVector(1.0e-15, -1.0e-15)
            tiny_3d = to_affine(tiny_2d)
            tiny_back = to_euclidean(tiny_3d)
            @test tiny_back ≈ tiny_2d atol = 1.0e-16

            # Test very large numbers
            large_2d = SVector(1.0e10, -1.0e10)
            large_3d = to_affine(large_2d)
            large_back = to_euclidean(large_3d)
            @test large_back ≈ large_2d rtol = 1.0e-15
        end

        @testset "Integration with conic system" begin
            # Test that coordinate conversion works with conic gradient
            ellipse = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)
            Q = HomEllipseMat(ellipse)

            # Test gradient computation using coordinate conversion
            test_pt = Point2(1.0, 0.5)
            grad = gradient(Q, test_pt)
            @test grad isa Vec2{Float64}
            @test length(grad) == 2

            # Verify gradient is computed correctly
            # For ellipse x²/4 + y² = 1, gradient at (1,0.5) should be [0.5, 1.0]
            @test grad ≈ Point2(0.5, 1.0) atol = 1.0e-12

            # Test with different points
            test_pts = [Point2(0.0, 1.0), Point2(2.0, 0.0), Point2(-1.0, -0.5)]
            for pt in test_pts
                grad = gradient(Q, pt)
                @test grad isa Vec2{Float64}
                @test !any(isnan, grad)
                @test !any(isinf, grad)
            end
        end
    end

    using StaticArrays: SVector, SMatrix
    using VisualGeometryToolkitCore: homogeneous_w, maps_within

    @testset "chirality-guarded to_euclidean + maps_within" begin
        v = SVector(2.0, 4.0, 2.0)
        @test to_euclidean(v, 1.0) == SVector(1.0, 2.0)
        @test to_euclidean(v, -1.0) === nothing            # wrong side of the horizon
        @test to_euclidean(SVector(1.0, 1.0, 0.0), 1.0) === nothing
        @test to_euclidean(SVector(1.0, 1.0, NaN), 1.0) === nothing
        H = SMatrix{3, 3}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
        r = VgtkCore.Rect(0.0, 0.0, 10.0, 10.0)
        wr = homogeneous_w(H, SVector(5.0, 5.0))
        @test wr == 1.0
        @test maps_within(H, r, SVector(5.0, 5.0), wr)
        @test maps_within(H, r, SVector(0.0, 10.0), wr)    # closed bounds
        @test !maps_within(H, r, SVector(11.0, 5.0), wr)
        # A tilted plane: w = 1 - 0.2x flips sign past x = 5 — beyond-horizon
        # points must be rejected by the sign guard, not by luck of the ratio.
        Hf = SMatrix{3, 3}(1.0, 0, -0.2, 0, 1.0, 0, 0, 0, 1.0)   # column-major: row 3 = (-0.2, 0, 1), w = 1 - 0.2x
        wr2 = homogeneous_w(Hf, SVector(0.0, 0.0))
        @test wr2 == 1.0
        @test maps_within(Hf, r, SVector(2.0, 1.0), wr2)   # visible side: 2/(0.6) ∈ rect
        @test !maps_within(Hf, r, SVector(6.0, 0.0), wr2)  # w = -0.2: behind the horizon
    end
end
