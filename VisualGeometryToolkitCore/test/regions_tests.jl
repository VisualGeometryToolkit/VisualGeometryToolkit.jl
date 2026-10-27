# Region primitives (primitives/containment.jl, primitives/regions.jl): the
# reverse containment `Rect ∈ Ellipse`, and the lazy intersection `a ∩ b` of a
# Rect, an Ellipse or another intersection, with its bounding box.
@testitem "regions" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RegionIntersection, WholePlane

    grid = [Point2(x, y) for x in range(-20.0, 120.0; length = 57), y in range(-30.0, 90.0; length = 49)]

    @testset "Rect ∈ Ellipse: all four corners inside" begin
        r = Rect(0, 0, 4, 3)
        @test r ∈ Ellipse(Point2(2.0, 1.5), 2.6, 2.6, 0.0)       # corners at 2.5
        @test r ∉ Ellipse(Point2(2.0, 1.5), 2.4, 2.4, 0.0)
        @test r ∉ Ellipse(Point2(2.0, 1.5), 10.0, 1.0, 0.0)      # wide, too flat
        @test r ∈ Ellipse(Point2(2.0, 1.5), 2.6, 2.6, 0.0) && !(Ellipse(Point2(2.0, 1.5), 2.6, 2.6, 0.0) ∈ r)
        @test Rect2(0.0, 0.0, 4.0, 3.0) ∈ Ellipse(Point2(2.0, 1.5), 2.6, 2.6, 0.0)
    end

    @testset "Rect ∩ Ellipse" begin
        r = Rect(0, 0, 100, 60)
        e = Ellipse(Point2(50.0, 30.0), 70.0, 25.0, 0.3)
        d = r ∩ e
        @test d isa RegionIntersection
        @test all(p -> (p ∈ d) == (p ∈ r && p ∈ e), grid)
        @test count(p -> p ∈ d, grid) > 0
        @test e ∩ r isa RegionIntersection
        @test all(p -> (p ∈ (e ∩ r)) == (p ∈ d), grid)
        # the bounding box: the rect clipped to the ellipse's box
        @test Rect2(d) == intersect(Rect2{Float64}(r), Rect2(e))
        @test Rect2(d) == Rect2(0.0, 0.0, 100.0, 60.0)   # the ellipse's box overhangs the rect
        e2 = Ellipse(Point2(80.0, 30.0), 40.0, 10.0, 0.0)  # box x ∈ [40, 120], y ∈ [20, 40]
        @test Rect2(r ∩ e2) ≈ Rect2(40.0, 20.0, 60.0, 20.0)
        @test Rect2(r ∩ e2; snap_to_grid = true) == Rect2(40.0, 20.0, 60.0, 20.0)
        @test bbox(d) == VisualGeometryToolkitCore.SpatialIndexing.Rect(Rect2(d))
    end

    @testset "Ellipse ∩ Ellipse, nested" begin
        e1 = Ellipse(Point2(40.0, 30.0), 50.0, 20.0, 0.0)
        e2 = Ellipse(Point2(70.0, 30.0), 30.0, 30.0, 0.0)
        d = e1 ∩ e2
        @test all(p -> (p ∈ d) == (p ∈ e1 && p ∈ e2), grid)
        @test Rect2(d) == intersect(Rect2(e1), Rect2(e2))
        r = Rect(0, 0, 60, 60)
        n = r ∩ d
        @test all(p -> (p ∈ n) == (p ∈ r && p ∈ e1 && p ∈ e2), grid)
        @test Rect2(n) == intersect(Rect2{Float64}(r), Rect2(d))
    end

    @testset "the whole plane: the identity of ∩" begin
        r = Rect(0, 0, 100, 60)
        e = Ellipse(Point2(50.0, 30.0), 70.0, 25.0, 0.3)
        w = WholePlane()
        @test (r ∩ w) === r && (w ∩ r) === r
        @test (e ∩ w) === e && (w ∩ e) === e
        @test (w ∩ w) === w
        d = r ∩ e
        @test (d ∩ w) === d && (w ∩ d) === d
        @test all(p -> p ∈ w, grid)
        @test r ∈ w
        @test_throws ArgumentError Rect2(w)
    end

    @testset "a rectangle in an intersection: in both parts" begin
        r = Rect(0, 0, 100, 60)
        e = Ellipse(Point2(50.0, 30.0), 70.0, 25.0, 0.0)
        d = r ∩ e
        @test Rect(40, 25, 20, 10) ∈ d                    # inside both
        @test Rect(0, 0, 10, 10) ∉ d                      # in the rect, a corner outside the ellipse
        @test Rect(90, 25, 20, 10) ∉ d                    # past the rect's edge
    end
end
