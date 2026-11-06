# =============================================================================
# A PlanarTarget records the layout it was built from (geometry/cameras/
# targets.jl): AbstractTargetLayout, layout(t), the layouts' whole designs
# (tag family, ids, star segments), their equality, the unit moves, the
# calibration protocol and resolve, and the layouts' tags.
# =============================================================================

@testitem "target layouts: PlanarTarget(layout) records its layout" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractTargetLayout, AprilGrid, SchoepsPattern, layout, design, points, canvas, has_units,
        mm, m, ustrip, with_unit, with_parameters, parameter_vector, Canvas, Size2, Aspect, Skew, TARGET_LAYOUT_TAGS, tag_of,
        tag_rule_violations

    g = AprilGrid(; tag_rows = 3, tag_cols = 4, tag_size = 90mm, tag_spacing = 27mm)
    p = SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 25mm, apriltags = [(; x = 2, y = 1, width = 2, height = 2, id = 7)])

    @testset "the layouts' whole design, with its defaults" begin
        @test g isa AbstractTargetLayout && p isa AbstractTargetLayout
        @test g.apriltag_family == "tag36h11" && g.first_id == 0
        @test p.star_segments == 16 && p.apriltag_family == "tag36h11" && only(p.apriltags) == (x = 2, y = 1, width = 2, height = 2, id = 7)
        g2 = AprilGrid(; tag_rows = 3, tag_cols = 4, tag_size = 90mm, tag_spacing = 27mm, apriltag_family = "tag25h9", first_id = 12)
        @test g2.apriltag_family == "tag25h9" && g2.first_id == 12 && g2 != g
        @test points(PlanarTarget(g2)) == points(PlanarTarget(g))                   # the family and ids name tags, they move no point
        # tags without ids are numbered in order
        q = SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 1.0,
            apriltags = [(; x = 0, y = 0, width = 1, height = 1), (; x = 5, y = 3, width = 2, height = 2)])
        @test [t.id for t in q.apriltags] == [0, 1]
        # equality and hash are by value (the tag vector included)
        p2 = SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 25mm, apriltags = [(; x = 2, y = 1, width = 2, height = 2, id = 7)])
        @test p2 == p && hash(p2) == hash(p) && p2 !== p
        @test SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 25mm, star_segments = 12, apriltags = p.apriltags) != p
        # refused
        @test_throws ArgumentError AprilGrid(; tag_rows = 3, tag_cols = 4, tag_size = 90mm, tag_spacing = 27mm, first_id = -1)
        @test_throws ArgumentError AprilGrid(; tag_rows = 3, tag_cols = 4, tag_size = 90mm, tag_spacing = 27mm, apriltag_family = "")
        @test_throws ArgumentError SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 1.0, apriltags = p.apriltags, star_segments = 7)
        @test_throws ArgumentError SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 1.0,
            apriltags = [(; x = 0, y = 0, width = 1, height = 1, id = 3), (; x = 5, y = 3, width = 1, height = 1, id = 3)])   # ids distinct
        @test_throws ArgumentError SchoepsPattern(; squares_x = 8, squares_y = 6, square_length = 1.0,
            apriltags = [(; x = 0, y = 0, width = 1, height = 1, id = -1)])
    end

    @testset "recorded, and kept or dropped by what keeps or replaces the points" begin
        for L in (g, p)
            t = PlanarTarget(L)
            @test layout(t) === L && is_resolved(t) && design(t) == points(t)                 # Rigid: the points are the layout's
            @test layout(PlanarTarget(design(t))) === nothing && PlanarTarget(design(t)) != t   # the same points, no layout
            # calibration, canvas and placement as keywords
            c = PlanarTarget(L; calibration = Aspect(0.01) ∘ Skew(0.002), canvas = Canvas(Size2(width = 400mm, height = 300mm), SVector(-10mm, -10mm)))
            @test layout(c) === L && canvas(c) !== nothing && points(c) != points(t)
            # a new calibration keeps the layout; resolve, which supplies other points, drops it
            w = with_parameters(c, 2 .* parameter_vector(c))
            @test layout(w) === L && parameter_vector(w) == 2 .* parameter_vector(c)
            @test layout(VgtkCore.resolve(t, design(t))) === nothing
            # a unit move moves the layout, and the points are the moved layout's
            s = ustrip(m, t)
            @test !has_units(layout(s)) && !has_units(s) && points(s) == points(PlanarTarget(layout(s)))
            back = with_unit(s, m)
            @test back == t && hash(back) == hash(t) && layout(back) == L
            @test ustrip(m, s) == s                                                     # a plain target stays plain
        end
    end

    @testset "the layouts' tags" begin
        @test collect(keys(TARGET_LAYOUT_TAGS)) == ["april_grid", "schoeps_pattern"]
        @test TARGET_LAYOUT_TAGS["april_grid"] === AprilGrid && tag_of(TARGET_LAYOUT_TAGS, g) == "april_grid" && tag_of(TARGET_LAYOUT_TAGS, p) == "schoeps_pattern"
        @test isempty(tag_rule_violations(TARGET_LAYOUT_TAGS))
        @test_throws ArgumentError TARGET_LAYOUT_TAGS["aprilgrid"]
    end
end
