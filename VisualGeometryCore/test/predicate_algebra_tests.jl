# =============================================================================
# Predicate algebra: overlap_area core, derived iou/coverage, geom_rank
# symmetrization, region gate, and the Ellipse matrix-constructor contract.
# =============================================================================

@testitem "predicate_algebra" begin
    using Test
    using LinearAlgebra: Symmetric
    using StaticArrays: SA
    using VisualGeometryCore: ustrip
    using VisualGeometryCore
    # Explicit imports: other suite files do plain `using GeometryBasics`, which
    # also exports `area`/`intersects`/`Circle`, making the bare names ambiguous
    # in the shared test Main.
    using VisualGeometryCore: px, HomEllipseMat, Ellipse, Circle, Rect2, Disc,
        EllipticFrame, Line2D, area, iou, coverage, overlap_area, overlap_method,
        geom_rank, intersects, region, closest_point, distance, moment_matrix,
        AnalyticOverlap, BoundarySampled
    using GeometryBasics: Point2, Point2f, Polygon

    @testset "Predicate algebra" begin
        e1 = Ellipse(Point2(0.0, 0.0), 3.0, 2.0, 0.3)
        e2 = Ellipse(Point2(1.0, 0.5), 2.5, 1.5, 1.1)
        c1 = Circle(Point2(0.0, 0.0), 2.0)
        c2 = Circle(Point2(1.5, 0.0), 2.0)
        r1 = Rect2(0.0, 0.0, 4.0, 3.0)
        quad = Polygon([Point2f(0, 0), Point2f(4, 0), Point2f(4, 3), Point2f(0, 3)])
        small = Ellipse(Point2(1.5, 1.2), 0.5, 0.3, 0.7)

        @testset "overlap_area analytic methods and symmetry" begin
            # Bit-exact symmetry for pairs where the swap is pure re-dispatch
            for (a, b) in [(c1, c2), (c1, e1), (e1, r1), (c1, r1), (r1, Rect2(2.0, 1.0, 4.0, 3.0))]
                @test overlap_area(a, b) == overlap_area(b, a)
                @test overlap_area(a, b) >= 0.0
            end
            # The GTE ellipse-ellipse port is symmetric only to rounding (the
            # algorithm's internal ordering depends on the argument order)
            @test overlap_area(e1, e2) ≈ overlap_area(e2, e1) rtol = 1.0e-12

            # Exact closed-form values
            @test overlap_area(r1, Rect2(2.0, 1.0, 4.0, 3.0)) ≈ 4.0
            @test overlap_area(r1, Rect2(10.0, 10.0, 1.0, 1.0)) == 0.0
            # Circle at the rect corner: exactly a quarter disc
            @test overlap_area(c1, r1) ≈ π * 2.0^2 / 4
            # Containment: overlap equals the smaller area
            @test overlap_area(c1, e1) ≈ area(c1)   # r=2 circle inscribed in (3,2)-ellipse
            @test overlap_area(small, r1) ≈ area(small)

            # Polygon methods agree with the equivalent Rect
            @test overlap_area(e1, quad) ≈ overlap_area(e1, r1)
            @test overlap_area(c1, quad) ≈ overlap_area(c1, r1)
            @test area(quad) ≈ area(r1) ≈ 12.0
        end

        @testset "derived iou and coverage identities" begin
            inter = overlap_area(e1, e2)
            @test iou(e1, e2) ≈ inter / (area(e1) + area(e2) - inter)
            @test iou(e1, e2) ≈ iou(e2, e1) rtol = 1.0e-12
            @test iou(e1, e1) ≈ 1.0
            @test iou(c1, c1) ≈ 1.0
            @test iou(e1, Ellipse(Point2(100.0, 100.0), 1.0, 1.0, 0.0)) == 0.0

            @test coverage(e1, e2) ≈ inter / area(e1)
            @test coverage(e2, e1) ≈ inter / area(e2)
            @test 0.0 <= coverage(e1, e2) <= 1.0
            # Deterministic (the Monte Carlo implementation is gone)
            @test coverage(e1, e2) == coverage(e1, e2)

            # Concentric circles as ellipses: coverage is the exact area ratio
            big = Ellipse(Point2(0.0, 0.0), 2.0, 2.0, 0.0)
            sml = Ellipse(Point2(0.0, 0.0), 1.0, 1.0, 0.0)
            @test coverage(sml, big) ≈ 1.0
            @test coverage(big, sml) ≈ 0.25
            # Mixed-type coverage comes from the same core
            @test coverage(c1, e1) ≈ 1.0        # circle inscribed in ellipse
            @test coverage(small, r1) ≈ 1.0     # ellipse inside rect
        end

        @testset "iou and coverage on conic matrices" begin
            Q1 = HomEllipseMat(e1)
            Q2 = HomEllipseMat(e2)
            @test iou(Q1, Q2) ≈ iou(e1, e2) rtol = 1.0e-9
            @test coverage(Q1, Q2) ≈ coverage(e1, e2) rtol = 1.0e-9
            @test area(Q1) ≈ area(e1)
        end

        @testset "geom_rank symmetrization" begin
            @test geom_rank(typeof(c1)) < geom_rank(typeof(e1)) < geom_rank(typeof(r1)) <
                geom_rank(typeof(quad))
            @test geom_rank(String) == typemax(Int)
            # Unranked pair falls through to MethodError, not infinite recursion
            @test_throws MethodError intersects("a", "b")

            @test intersects(e1, r1) == intersects(r1, e1) == true
            @test intersects(c1, c2) == true
            @test intersects(c1, Circle(Point2(10.0, 0.0), 1.0)) == false
        end

        @testset "overlap_method trait and BoundarySampled fallback" begin
            @test overlap_method(e1, e2) === AnalyticOverlap()
            @test overlap_method(c1, r1) === AnalyticOverlap()
            @test overlap_method(r1, e1) === AnalyticOverlap()

            # Disc pairs have no analytic method: deterministic sampled fallback
            d1 = Disc(Point2(0.0, 0.0), 1.0)
            d2 = Disc(Point2(0.5, 0.0), 1.0)
            @test overlap_method(d1, d2) isa BoundarySampled
            exact = overlap_area(Circle(Point2(0.0, 0.0), 1.0), Circle(Point2(0.5, 0.0), 1.0))
            @test overlap_area(d1, d2) ≈ exact rtol = 0.01
            @test overlap_area(d1, d2) == overlap_area(d1, d2)  # deterministic
        end

        @testset "region gate: units stripped, k·σ contour" begin
            d_px = Disc(Point2(100.0px, 200.0px), 5.0px)
            c = region(d_px, 3.0)
            @test c isa Circle{Float64}
            @test c.center == Point2(100.0, 200.0)
            @test c.r == 15.0
            @test region(d_px) == region(d_px, 1.0)

            ef = EllipticFrame(Point2(10.0, 20.0), SA[2.0 0.5; 0.5 1.0])
            e = region(ef, 2.0)
            @test e isa Ellipse{Float64}
            @test e.a * e.b ≈ 4.0 * abs(2.0 * 1.0 - 0.5 * 0.5)  # k² · det(S)
            # region at k is the k-dilation of the unit region
            u = region(ef)
            @test e.a ≈ 2.0 * u.a && e.b ≈ 2.0 * u.b
        end

        @testset "Ellipse matrix constructor contract" begin
            # Plain matrices are refused: moment-vs-sqrt ambiguity
            @test_throws ArgumentError Ellipse(Point2(0.0, 0.0), [4.0 1.0; 1.0 9.0])
            # Symmetric moment matrix round-trips
            e = Ellipse(Point2(5.0, 7.0), 3.0, 2.0, 0.4)
            e_rt = Ellipse(e.center, Symmetric(moment_matrix(e)))
            @test e_rt.center == e.center
            @test e_rt.a ≈ e.a
            @test e_rt.b ≈ e.b
            @test mod(e_rt.θ - e.θ, π) ≈ 0.0 atol = 1.0e-10
        end

        @testset "closest_point on hyperplane" begin
            line = Line2D(1.0, 0.0, -2.0)  # x = 2
            p = Point2(5.0, 7.0)
            cp = closest_point(line, p)
            @test cp ≈ Point2(2.0, 7.0)
            @test abs(distance(line, p)) ≈ 3.0
            @test abs(distance(line, cp)) < 1.0e-12
            @test side(line, p) == 1
            @test cp ∈ line   # `in` keeps Base's in(item, collection) order
            @test_throws MethodError distance(p, line)
        end

        @testset "allocation-free hot predicates" begin
            p_in = Point2(1.0, 1.0)
            f_in() = p_in in quad
            f_ir() = intersects(e1, r1)
            f_in(); f_ir()
            @test @allocated(f_in()) == 0
            @test @allocated(f_ir()) == 0
        end

        @testset "point-in-polygon correctness" begin
            @test Point2(2.0, 1.5) in quad
            @test !(Point2(5.0, 1.5) in quad)
            # Concave polygon (notched square): notch interior is outside
            notched = Polygon(
                [
                    Point2f(0, 0), Point2f(4, 0), Point2f(4, 4),
                    Point2f(2, 2), Point2f(0, 4),
                ]
            )
            @test Point2(1.0, 1.0) in notched
            @test !(Point2(2.0, 3.0) in notched)
        end
    end
end
