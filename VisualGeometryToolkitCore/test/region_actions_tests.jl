# A SimilarityMap acting on the regions pixel_mask reads (Rect, RegionIntersection), and pixel_mask
# taking a frame map: a mask on another pixel lattice than its region's (a Bayer plane against the
# sensor), decided in the region's own frame (0.9.2).

@testitem "region actions: a similarity maps a Rect and an intersection" begin
    using VisualGeometryToolkitCore: SimilarityMap, Rect, Ellipse, Point2, RegionIntersection, WholePlane
    using GeometryBasics: origin, widths
    using Rotations: RotMatrix
    using StaticArrays: SVector, SMatrix
    S = SimilarityMap(2.0, RotMatrix{2}(SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)), SVector(0.5, -1.0))
    r = Rect(1.0, 2.0, 3.0, 4.0)
    @test S(r) == Rect(2.5, 3.0, 6.0, 8.0)
    # a quarter turn swaps the widths, exactly
    Q = SimilarityMap(2.0, RotMatrix{2}(SMatrix{2, 2}(0.0, 1.0, -1.0, 0.0)), SVector(0.0, 0.0))
    @test S(r) isa Rect && Q(r) == Rect(-12.0, 2.0, 8.0, 6.0)
    # any other turn is refused: a turned rectangle is no Rect
    @test_throws ArgumentError SimilarityMap(1.0, RotMatrix{2}(0.3), SVector(0.0, 0.0))(r)
    # an intersection maps part by part: p ∈ s ⟺ S(p) ∈ S(s)
    s = Rect(0.0, 0.0, 10.0, 8.0) ∩ Ellipse(Point2(5.0, 4.0), 4.0, 3.0, 0.2)
    Ss = S(s)
    @test Ss isa RegionIntersection
    pts = [Point2(x, y) for x in -1.0:0.25:11.0, y in -1.0:0.25:9.0]   # dyadic: S maps them, and the ellipse, exactly
    @test all((p ∈ s) == (Point2(S(SVector(p...))...) ∈ Ss) for p in pts)
    @test S(Rect(0.0, 0.0, 4.0, 4.0) ∩ WholePlane()) == S(Rect(0.0, 0.0, 4.0, 4.0))
end

@testitem "region actions: pixel_mask through a frame map" begin
    using VisualGeometryToolkitCore: pixel_mask, Rect, Ellipse, Point2, SimilarityMap, LinearMap, Translation, RawMosaic,
        bayer_plane_map, raw_frame_map
    using StaticArrays: SVector
    region = Rect(0.0, 0.0, 20.0, 16.0) ∩ Ellipse(Point2(10.0, 8.0), 7.0, 5.0, 0.0)
    # the identity map is the plain mask
    I2 = SimilarityMap(Translation(0.0, 0.0))
    @test pixel_mask(region, I2, 20, 16) == pixel_mask(region, 20, 16)
    # a mask on a Bayer channel's lattice: each channel pixel's mapped centre tested on the sensor
    m = RawMosaic(fill(UInt16(3000), 16, 20), UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 1), (16, 21), (;))
    for k in 1:4
        S = bayer_plane_map(m, k)
        M = pixel_mask(region, S, 10, 8)
        @test size(M) == (8, 10)
        @test all(M[v, u] == (Point2(S(SVector(u - 0.5, v - 0.5))...) ∈ region) for v in 1:8, u in 1:10)
        @test count(M) > 0
    end
    # a window of the mosaic: its offset
    w = view(m, 3:16, 2:20)
    @test pixel_mask(region, raw_frame_map(w), 19, 14) == pixel_mask(region, 20, 16)[3:16, 2:20]
end
