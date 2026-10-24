# =============================================================================
# Unit system: FlexUnits as shipped (px, pd, dpi and pt registered into its
# registry), quantity conversions, Size2, per-type strip, and the unitless
# origin-conversion Translation.
#
# This is the unit-boundary contract of the package (audit #16): units are
# enforced at construction, stripped for compute; dimensions are runtime
# values (ONE concrete quantity type across px/mm/dpi, FlexUnits' base form);
# strip methods are
# defined per-type on VGC-owned types.
# =============================================================================

@testitem "units" begin
    using Test
    using StaticArrays: SA
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: px, dpi, pt, mm, cm, μm, m, inch, ft, ustrip, uconvert,
        dimension, DimensionError, ConversionError,
        @ud_str, uparse, qparse, parse_unit, unit_string, strip_or_bare, Size2, Disc, EllipticFrame, aspect_ratio, area
    using FlexUnits: Dimensions, FixRat32, Quantity, UnitRegistry
    using FlexUnits.UnitRegistry: @u_str
    using GeometryBasics: Point2, Rect2
    using CoordinateTransformations: Translation

    @testset "Unit system" begin
        @testset "FlexUnits' registry: px is a dimensionless count" begin
            @test haskey(UnitRegistry.UNITS, :px) && haskey(UnitRegistry.UNITS, :pd)
            @test dimension(px) == dimension(UnitRegistry.UNITS[:rad])   # the accepted loss
            # the constants are the registry's dynamic units
            @test px == ud"px" && mm == ud"mm" && dpi == ud"dpi" && pt == ud"pt"
            @test uparse("px/inch") == dpi
            @test qparse("300 dpi") == 300.0dpi
            # values are stored in base units: metres for a length, a plain count for pixels
            @test ustrip(10.0mm) ≈ 0.01
            @test ustrip(3px) == 3.0
            # a unit that is wanted is named
            @test ustrip(mm, 10.0mm) ≈ 10.0
            @test ustrip(mm, 1.0inch) ≈ 25.4
            @test ustrip(mm, 72.0pt) ≈ 25.4
            @test ustrip(μm, 1.0mm) ≈ 1000.0
            @test ustrip(mm, 1.0ft) ≈ 304.8
            @test ustrip(mm, 1.0cm) ≈ 10.0
            @test ustrip(m, 1.0mm) ≈ 1.0e-3

            # THE point: px/mm/dpi share one concrete type, as do ud"…" quantities
            @test typeof(10.0mm) === typeof(3.0px) === typeof(300.0dpi) === typeof(1.0ud"mm/px")
            Q = Quantity{Float64, Dimensions{FixRat32}}
            @test 10.0mm isa Q && isbitstype(Q)
            @test isconcretetype(eltype([1.0mm, 2.0px, 3.0dpi]))
            # a quantity–unit product carries its unit; convert takes it to the one type
            @test convert(Q, 5.86μm / px) isa Q
            @test convert(Q, 10u"mm") == 10.0mm

            # Dimension mismatch is a runtime error; a pixel is dimensionless
            @test_throws DimensionError 1.0mm + 1.0px
            @test 3.0px + 1.0 == 4.0
            @test_throws ConversionError uconvert(px, 1.0mm)

            # Fractional exponents survive sqrt
            @test sqrt(4.0mm * 9.0mm) ≈ 6.0mm
            @test sqrt(2.0mm)^2 ≈ 2.0mm
        end

        @testset "logical units and quantity conversions" begin
            # 1 inch at 300 dpi is 300 px; pt is 1/72 inch
            @test ustrip(px, 1.0inch * 300dpi) ≈ 300.0
            @test ustrip(inch, 72.0pt) ≈ 1.0
            @test ustrip(px / mm, 300dpi) ≈ 300 / 25.4
            # density crossings are products, no conversion verb
            @test ustrip(px, 25.4mm * 300dpi) ≈ 300.0
            @test ustrip(mm, 300.0px / 300dpi) ≈ 25.4
            q = 42.0mm
            @test (q * 300dpi) / 300dpi ≈ q

            # Target-unit strip
            @test ustrip(inch, 50.8mm) ≈ 2.0
            @test_throws ConversionError ustrip(px, 1.0mm)

            # Dimension checks are FlexUnits' `dimension`
            @test dimension(300dpi) == dimension(px / m)
            @test dimension(10.0mm) == dimension(m) && dimension(1.0inch) == dimension(m)
            @test dimension(1.0mm / 2px) == dimension(mm / px)
            @test ustrip(10.0mm / 1.0inch) ≈ 10 / 25.4
        end

        @testset "Size2" begin
            s = Size2(width = 800, height = 600)
            @test s.width == 800 && s.height == 600
            @test aspect_ratio(s) ≈ 4 / 3
            @test area(s) == 480_000
            @test Size2(zeros(3, 5)) == Size2(5, 3)  # (width, height) from matrix

            # Unit-carrying Size2 and a broadcast target-unit strip
            sp = Size2(width = 210.0mm, height = 297.0mm)
            @test area(sp) ≈ 210.0mm * 297.0mm
            @test ustrip.(mm, sp) ≈ Size2(210.0, 297.0)
            @test ustrip.(inch, Size2(width = 25.4mm, height = 50.8mm)) ≈ Size2(1.0, 2.0)
            # Broadcasting preserves the Size2 type
            @test ustrip.(mm, sp) isa Size2
            # A mixed-unit-constructed Size2 is CONCRETE (dimensions are values)
            @test Size2(width = 10.0mm, height = 1.0inch) isa Size2{Quantity{Float64, Dimensions{FixRat32}}}
        end

        @testset "unit names from outside: parse_unit and unit_string" begin
            # one registered symbol, resolved by the registry's uparse
            @test parse_unit("mm") == mm && parse_unit("inch") == inch && parse_unit("px") == px
            @test parse_unit("dpi") == dpi && parse_unit("pd") == px && parse_unit("in") == inch
            @test_throws ArgumentError parse_unit("mm/px")                    # an expression needs compound
            @test_throws ArgumentError parse_unit("furlong")
            @test_throws ArgumentError parse_unit("MM")                       # the registry is case-sensitive
            # compound: registered symbols with *, / and integer ^ only, checked before uparse evaluates
            @test parse_unit("mm/px"; compound = true) == mm / px
            @test parse_unit("px/mm"; compound = true) == px / mm
            @test parse_unit("mm^2"; compound = true) == mm^2
            for bad in ("run(`ls`)", "mm + px", "sqrt(mm)", "2mm", "mm^x", "mm/", "")
                @test_throws ArgumentError parse_unit(bad; compound = true)
            end
            # unit_string is FlexUnits' printing of a registered unit, the inverse
            for u in (mm, cm, m, μm, inch, ft, pt, px, dpi)
                @test parse_unit(unit_string(u)) == u
            end
            @test unit_string(mm) == "mm" && unit_string(inch) == "inch"
            @test_throws ArgumentError unit_string(mm / px)                   # no registered name
        end

        @testset "strip_or_bare" begin
            @test strip_or_bare(px, 3.0px) === 3.0 && strip_or_bare(px, 3.0) === 3.0
            @test strip_or_bare(mm, 1.0inch) ≈ 25.4
            @test_throws ConversionError strip_or_bare(px, 1.0mm)
            d = Disc(Point2(1.0px, 2.0px), 3.0px)
            @test strip_or_bare(px, d) == Disc(Point2(1.0, 2.0), 3.0)
            bare = Disc(Point2(1.0, 2.0), 3.0)
            @test strip_or_bare(px, bare) === bare
            @test ustrip.(px, Size2(width = 4.0px, height = 3.0px)) === Size2(4.0, 3.0)
        end

        @testset "per-type ustrip" begin
            d_mm = Disc(Point2(25.4mm, 50.8mm), 2.54mm)
            dm = ustrip(mm, d_mm)
            @test dm isa Disc{Float64} && dm.center ≈ Point2(25.4, 50.8) && dm.σ ≈ 2.54
            di = ustrip(inch, d_mm)
            @test di.center ≈ Point2(1.0, 2.0) && di.σ ≈ 0.1
            @test_throws ConversionError ustrip(px, d_mm)

            # Rect ustrip
            r = Rect2(0.0mm, 0.0mm, 10.0mm, 20.0mm)
            @test ustrip(mm, r).widths ≈ Rect2(0.0, 0.0, 10.0, 20.0).widths
        end

        @testset "region strips units (construction/compute boundary)" begin
            d = Disc(Point2(100.0px, 200.0px), 5.0px)
            c = region(d, 3.0)
            @test eltype(c.center) === Float64
            @test c.r == 15.0
        end

        @testset "origin-conversion frame map is unitless" begin
            T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
            @test T.t == SA[0.5, 0.5]
            @test eltype(T.t) === Float64
            @test T(Point2(1.0, 2.0)) == Point2(1.5, 2.5)
            @test EuclideanMap(PixelFrame{:colmap}, PixelFrame{:colmap})(Point2(1.0, 2.0)) ==
                Point2(1.0, 2.0)
            # Round trip: there and back is identity
            @test EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})(T(Point2(3.0, 4.0))) ==
                Point2(3.0, 4.0)
            @test_throws Exception EuclideanMap(PixelFrame{:vlfeat}, PixelFrame{:julia})  # unknown origin
        end
    end
end

# The feature-coordinate compute boundary (2026-09-17):
# unit-carrying frames stay boundary values, and every predicate strips them
# through ONE check — one dimension, pixels or a length, never mixed with bare
# numbers.
@testitem "units: feature-coordinate compute boundary" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: px, mm, Disc, Point2, Circle, intersects, bbox, overlap_area, ustrip, zhang_intrinsics, Size2, HomographyMat

    dp = Disc(Point2(100.0px, 200.0px), 5.0px)
    dq = Disc(Point2(108.0px, 200.0px), 5.0px)
    fp = Disc(Point2(100.0, 200.0), 5.0)
    fq = Disc(Point2(108.0, 200.0), 5.0)
    @test intersects(dp, dq, 1.0) == intersects(fp, fq, 1.0)
    @test intersects(dp, dq, 0.5) == intersects(fp, fq, 0.5)
    @test bbox(dp; scale = 3.0) == bbox(fp; scale = 3.0)
    @test ustrip(px, dp) == fp
    # a px disc never meets an mm disc, or a bare one, as bare numbers
    dm = Disc(Point2(100.0mm, 200.0mm), 5.0mm)
    @test_throws Exception intersects(dp, dm, 1.0)
    @test_throws ArgumentError intersects(dp, fq, 1.0)
    # the constructor checks the dimensions it is given
    @test_throws Exception Disc(Point2(100.0px, 200.0px), 5.0mm)
    # circles: the same strip, one Float64 answer
    a = overlap_area(Circle(Point2(0.0px, 0.0px), 2.0px), Circle(Point2(1.5px, 0.0px), 2.0px))
    @test a isa Float64
    @test a == overlap_area(Circle(Point2(0.0, 0.0), 2.0), Circle(Point2(1.5, 0.0), 2.0))
    @test overlap_area(Circle(Point2(0.0px, 0.0px), 1.0px), Circle(Point2(9.0px, 0.0px), 1.0px)) === 0.0
    @test overlap_area(Circle(Point2(0.0f0, 0.0f0), 1.0f0), Circle(Point2(9.0f0, 0.0f0), 1.0f0)) isa Float32
    # a frame's circle and a detection's go through the same strip
    @test Circle(dp, 3.0) == Circle(fp, 3.0)
    # a known pitch must be a pitch
    H = HomographyMat{Float64}(ntuple(i -> Float64(i in (1, 5, 9)), 9))
    @test_throws ArgumentError zhang_intrinsics([H], Size2(width = 3.9mm, height = 4.0mm); center = (0.0, 0.0))
end
