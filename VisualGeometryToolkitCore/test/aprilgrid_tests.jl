# =============================================================================
# Kalibr's AprilGrid as a PlanarTarget (geometry/cameras/targets.jl): the point
# layout against Kalibr's GridCalibrationTargetAprilgrid, the corner order of a
# detection, the units.
# =============================================================================

@testitem "AprilGrid: Kalibr's point layout and corner order" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AprilGrid, point_index, mm, m, ustrip, points, Canvas, Size2, canvas, has_units

    # Kalibr's documented example target (aprilgrid.yaml in the wiki's "Calibration targets"): tagCols 6, tagRows 6, tagSize 0.088 m,
    # tagSpacing 0.3 (the gap over the tag side). Its points, transcribed from GridCalibrationTargetAprilgrid's constructor: a
    # 2·tagRows × 2·tagCols grid, row by row from the lower-left corner, point(r, c) =
    # ((c ÷ 2)(1 + tagSpacing)·tagSize + (c % 2)·tagSize, (r ÷ 2)(1 + tagSpacing)·tagSize + (r % 2)·tagSize, 0).
    kalibr_point(r, c, rows, cols, size, ratio) =
        SVector((c ÷ 2) * (1 + ratio) * size + (c % 2) * size, (r ÷ 2) * (1 + ratio) * size + (r % 2) * size)
    # and a detection's tag corner → point, transcribed from its computeObservation: baseId = (id ÷ tagCols)·2cols·2 + (id % tagCols)·2,
    # corners k = 0 … 3 at {baseId, baseId + 1, baseId + 2cols + 1, baseId + 2cols} (0-based; cols = 2·tagCols)
    kalibr_index(id, k, tag_cols) = (c2 = 2tag_cols; b = (id ÷ tag_cols) * c2 * 2 + (id % tag_cols) * 2; (b, b + 1, b + c2 + 1, b + c2)[k + 1])

    @testset "the 6 × 6, 88 mm, 0.3 example" begin
        g = AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = 0.088, tag_spacing = 0.3 * 0.088)
        t = PlanarTarget(g)
        ps = points(t)
        @test length(ps) == 144
        @test all(isapprox(ps[r * 12 + c + 1], kalibr_point(r, c, 6, 6, 0.088, 0.3); atol = 1.0e-15) for r in 0:11, c in 0:11)
        # by hand: the first tag's corners, the second tag's first corner, the last point
        @test ps[1] == SVector(0.0, 0.0) && ps[2] ≈ SVector(0.088, 0.0) && ps[13] ≈ SVector(0.0, 0.088) && ps[14] ≈ SVector(0.088, 0.088)
        @test ps[3] ≈ SVector(0.088 * 1.3, 0.0)
        @test ps[144] ≈ SVector(5 * 0.088 * 1.3 + 0.088, 5 * 0.088 * 1.3 + 0.088)
        @test all(point_index(g, id, k) == kalibr_index(id, k, 6) + 1 for id in 0:35, k in 0:3)
        @test [point_index(g, 0, k) for k in 0:3] == [1, 2, 14, 13]
        @test [point_index(g, 7, k) for k in 0:3] == [27, 28, 40, 39]           # row 1, column 1: base 1·12·2 + 1·2 = 26
        # a corner's point: the tag's (min col, min row) corner first, counter-clockwise in the design chart (y up)
        tag7 = [ps[point_index(g, 7, k)] for k in 0:3]
        @test tag7[1] ≈ SVector(0.088 * 1.3, 0.088 * 1.3) && tag7[2] - tag7[1] ≈ SVector(0.088, 0.0) && tag7[4] - tag7[1] ≈ SVector(0.0, 0.088)
        @test sort(vec([point_index(g, id, k) for id in 0:35, k in 0:3])) == 1:144   # every point is one tag's corner, once
    end

    @testset "lengths, a rectangular grid, a page" begin
        # the 1 m AprilGrid of RadiometricCamera's experiments: 8 × 8 tags of 90 mm, 27 mm apart, on a 979 mm page with an 8 mm margin
        # and a 27 mm corner square around the tags (Kalibr's generator): tag 0's first corner 35 mm in from the page's lower-left corner
        g = AprilGrid(; tag_rows = 8, tag_cols = 8, tag_size = 90mm, tag_spacing = 27mm)
        page = Size2(width = 979mm, height = 979mm)
        t = PlanarTarget(g; canvas = Canvas(page, SVector(-35mm, -35mm)))
        ps = points(t)
        @test has_units(t) && length(ps) == 256
        @test ustrip.(mm, ps[point_index(g, 63, 2)]) ≈ SVector(7 * 117 + 90, 7 * 117 + 90)
        @test VgtkCore.Rect(canvas(t)).widths ≈ [0.979, 0.979] && VgtkCore.Rect(canvas(t)).origin ≈ [-0.035, -0.035]   # the design chart in metres
        g2 = AprilGrid(; tag_rows = 2, tag_cols = 3, tag_size = 0.05, tag_spacing = 0.01)
        @test length(points(PlanarTarget(g2))) == 24
        @test all(points(PlanarTarget(g2))[r * 6 + c + 1] ≈ kalibr_point(r, c, 2, 3, 0.05, 0.2) for r in 0:3, c in 0:5)
        @test all(point_index(g2, id, k) == kalibr_index(id, k, 3) + 1 for id in 0:5, k in 0:3)
    end

    @testset "refused" begin
        @test_throws ArgumentError AprilGrid(; tag_rows = 0, tag_cols = 6, tag_size = 0.088, tag_spacing = 0.0264)
        @test_throws ArgumentError AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = -0.088, tag_spacing = 0.0264)
        @test_throws ArgumentError AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = 88mm, tag_spacing = 0.0264)     # one form
        @test_throws ArgumentError AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = 88mm, tag_spacing = 3VgtkCore.px)
        g = AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = 0.088, tag_spacing = 0.0264)
        @test_throws ArgumentError point_index(g, 36, 0)
        @test_throws ArgumentError point_index(g, 0, 4)
    end
end
