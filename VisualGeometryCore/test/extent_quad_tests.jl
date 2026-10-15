@testitem "extent_quad" begin
    # `rect_corners` / `is_convex_quad`: the corner order and winding test every
    # extent check walks (`is_orientation_preserving` here, the pose and
    # radial-homography extent gates in RobustVisualGeometry).
    using Test
    using StaticArrays
    using LinearAlgebra: I
    using GeometryBasics: Rect2
    using VisualGeometryCore
    using VisualGeometryCore: rect_corners, is_convex_quad, is_orientation_preserving, HomographyMat

    @testset "rect_corners: counter-clockwise from the origin" begin
        c = rect_corners(Rect2(SA[1.0, 2.0], SA[3.0, 5.0]))
        @test c == (SA[1.0, 2.0], SA[4.0, 2.0], SA[4.0, 7.0], SA[1.0, 7.0])
        @test eltype(first(c)) === Float64
        c32 = rect_corners(Rect2(SA[1.0f0, 2.0f0], SA[3.0f0, 5.0f0]))
        @test eltype(first(c32)) === Float32
    end

    @testset "is_convex_quad" begin
        sq = (SA[0.0, 0.0], SA[1.0, 0.0], SA[1.0, 1.0], SA[0.0, 1.0])
        @test is_convex_quad(sq)
        @test is_convex_quad(reverse(sq))                                   # clockwise winding: still convex
        @test !is_convex_quad((SA[0.0, 0.0], SA[1.0, 1.0], SA[1.0, 0.0], SA[0.0, 1.0]))   # bow-tie
        @test !is_convex_quad((SA[0.0, 0.0], SA[1.0, 0.0], SA[0.2, 0.2], SA[0.0, 1.0]))   # reflex vertex
        @test !is_convex_quad((SA[0.0, 0.0], SA[1.0, 0.0], SA[2.0, 0.0], SA[3.0, 0.0]))   # degenerate
        @test is_convex_quad(map(p -> p + SA[4000.0, 3000.0], sq))           # translation invariant
    end

    @testset "one sign rule: the sample gate's handedness agrees" begin
        _op = VisualGeometryCore._orientation_preserving
        sq = (SA[0.0, 0.0], SA[1.0, 0.0], SA[1.0, 1.0], SA[0.0, 1.0])
        @test _op(sq, sq) && _op(sq, map(p -> SA[p[1], -p[2]], sq))           # a uniform flip is a labelling
        @test !_op(sq, (SA[0.0, 0.0], SA[1.0, 1.0], SA[1.0, 0.0], SA[0.0, 1.0]))   # bow-tie
        line = (SA[0.0, 0.0], SA[1.0, 0.0], SA[2.0, 0.0], SA[3.0, 0.0])
        @test !_op(line, line)                                                # all degenerate: no evidence, false
    end

    @testset "is_orientation_preserving walks them" begin
        rect = Rect2(SA[0.0, 0.0], SA[100.0, 60.0])
        @test is_orientation_preserving(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(I))), rect)
        # A fold through the rectangle (the line x = 50 maps to infinity): bow-tie.
        H = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(1, 0, 0.02, 0, 1, 0, 0, 0, -1)))
        @test !is_orientation_preserving(H, rect)
    end
end
