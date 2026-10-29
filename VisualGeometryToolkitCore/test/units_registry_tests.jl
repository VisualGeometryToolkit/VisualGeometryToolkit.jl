# =============================================================================
# The unit registry: the stack's own, `StackUnits`, built with FlexUnits'
# RegistryTools over `AffineTransform{Rational{Int}}`, every scale an exact
# rational in base units. No stack quantity type or dimension set; nothing is
# registered into FlexUnits' own registry.
# =============================================================================

@testitem "units registry" begin
    using Test
    import FlexUnits
    using FlexUnits: FixRat32
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: px, pt, dpi, mm, cm, μm, m, inch, ft, ustrip, dimension, parse_unit, unit_string,
        BaseQuantity, base_quantity
    using VisualGeometryToolkitCore.StackUnits: StackUnits

    Q = FlexUnits.Quantity{Float64, FlexUnits.Dimensions{FixRat32}}
    @test BaseQuantity === Q
    @test typeof(1.0mm) == Q
    @test isbitstype(typeof(1.0mm))
    @test typeof(3.0px) == Q && typeof(300.0dpi) == Q

    # the registry's transform type: exact rational scales
    @test valtype(StackUnits.UNITS) === FlexUnits.Units{FlexUnits.Dimensions{FixRat32}, FlexUnits.AffineTransform{Rational{Int}}}
    scales = Dict(
        :m => 1 // 1, :kg => 1 // 1, :s => 1 // 1, :A => 1 // 1, :K => 1 // 1, :cd => 1 // 1, :mol => 1 // 1,
        :km => 1000 // 1, :cm => 1 // 100, :mm => 1 // 1000, :μm => 1 // 1000000, :um => 1 // 1000000,
        :inch => 127 // 5000, :in => 127 // 5000, :ft => 381 // 1250, :pt => 127 // 360000,
        :px => 1 // 1, :pd => 1 // 1, :dpi => 5000 // 127,
    )
    @test Set(keys(StackUnits.UNITS)) == Set(keys(scales))   # exactly the stack's units
    for (k, s) in scales
        u = StackUnits.UNITS[k]
        @test FlexUnits.uscale(u) === s && iszero(FlexUnits.uoffset(u))
    end
    @test dimension(px) == dimension(StackUnits.UNITS[:pd]) == FlexUnits.Dimensions{FixRat32}()
    @test dimension(dpi) == inv(dimension(m))

    # an integer-built length is exact: its stored Float64 is the decimal-nearest
    # value (with FlexUnits' Float64 scales, 142mm was 0.14200000000000002 m)
    @test ustrip(m, 142mm) === 71 // 500
    @test Float64(ustrip(m, 142mm)) === 0.142
    @test ustrip(m, base_quantity(142mm)) === 0.142
    # (`ustrip(m, 142mm) == 0.142` is false: a Rational compares exactly with
    # the binary double 0.142, which is not 71/500)
    # the A5 page's sides
    @test ustrip(m, base_quantity(148mm)) === 0.148 && ustrip(m, base_quantity(210mm)) === 0.21
    @test base_quantity(148mm) isa BaseQuantity
    # every length unit, integer literals: the correctly rounded decimal
    for k in 1:2000
        @test ustrip(base_quantity(k * mm)) === parse(Float64, "$(k)e-3")
        @test ustrip(base_quantity(k * cm)) === parse(Float64, "$(k)e-2")
        @test ustrip(base_quantity(k * μm)) === parse(Float64, "$(k)e-6")
        @test ustrip(base_quantity(k * m)) === Float64(k)
        @test ustrip(base_quantity(k * inch)) === parse(Float64, "$(254k)e-4")    # 25.4 mm
        @test ustrip(base_quantity(k * ft)) === parse(Float64, "$(3048k)e-4")     # 304.8 mm
        @test ustrip(base_quantity(k * pt)) === Float64(big(127k) / 360000)       # inch/72, no decimal
        @test ustrip(base_quantity(k * px)) === Float64(k)
        @test ustrip(base_quantity(k * dpi)) === Float64(big(5000k) / 127)        # px per metre
    end
    # a FLOAT times a rational scale is FlexUnits' promotion, x * Float64(scale):
    # within 1 ulp of the decimal (142.0 × 0.001 = 0.14200000000000002)
    @test abs(ustrip(m, 142.0mm) - 0.142) <= eps(0.142)
    @test all(k -> abs(ustrip(m, (k + 0.0) * mm) - parse(Float64, "$(k)e-3")) <= eps(parse(Float64, "$(k)e-3")), 1:2000)

    @test ustrip(mm, 12.0pt) ≈ 4.2333333333333325
    # 600 px at 300 dpi: two inches. The float path divides by rounded scales
    # (300 × 5000/127, then ÷ 0.001), so it is within 1 ulp of 50.8, not exactly
    # on it (FlexUnits 0.6.13 happened to land on 50.8, 0.6.16 one ulp above);
    # the integer path is exact by construction.
    @test abs(ustrip(mm, (600.0px) / (300.0dpi)) - 50.8) <= eps(50.8)
    @test ustrip(mm, 600px / 300dpi) === 254 // 5

    # parse_unit and unit_string round trip every registered symbol
    for k in keys(StackUnits.UNITS)
        s = String(k)
        @test parse_unit(s) == StackUnits.UNITS[k]
        @test unit_string(parse_unit(s)) == s
    end
    @test parse_unit("pd") == px && parse_unit("in") == inch && parse_unit("um") == μm

    # the stack's former unit layer stays gone
    # (the names are retired, so they are spelled in fragments)
    for name in (Symbol("Vgc", "Quantity"), Symbol("Pixel", "Dimensions"), Symbol("Pixel", "Units"))
        @test !isdefined(VgtkCore, name)
    end

    # the registry is a constant of the precompiled image: a fresh process parses the stack's units
    pkgdir = dirname(dirname(pathof(VgtkCore)))
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir) -e 'using VisualGeometryToolkitCore; VisualGeometryToolkitCore.parse_unit("dpi")'`
    @test success(run(cmd))
end
