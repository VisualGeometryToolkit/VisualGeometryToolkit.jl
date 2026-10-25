# =============================================================================
# The unit registry: FlexUnits as shipped. No stack quantity type, dimension
# set or registry; VisualGeometryToolkitCore registers px, pd, dpi and pt into
# FlexUnits' own registry at `__init__` (registrations made while precompiling
# do not persist).
# =============================================================================

@testitem "units registry" begin
    using Test
    import FlexUnits
    using FlexUnits: FixRat32
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: px, pt, dpi, mm, inch, ustrip, parse_unit, unit_string

    Q = FlexUnits.Quantity{Float64, FlexUnits.Dimensions{FixRat32}}
    @test typeof(1.0mm) == Q
    @test isbitstype(typeof(1.0mm))
    @test typeof(3.0px) == Q && typeof(300.0dpi) == Q

    @test ustrip(mm, 12.0pt) ≈ 4.2333333333333325
    @test ustrip(mm, (600.0px) / (300.0dpi)) == 50.8

    @test parse_unit("dpi") == dpi
    @test parse_unit("pt") == pt
    @test parse_unit("pd") == px
    @test parse_unit("px") == px
    @test parse_unit("mm") == mm
    @test parse_unit("inch") == inch
    @test unit_string(mm) == "mm"

    # the stack's own unit layer is gone
    # (the names are retired, so they are spelled in fragments)
    for name in (Symbol("Vgc", "Quantity"), Symbol("Pixel", "Dimensions"), Symbol("Pixel", "Units"))
        @test !isdefined(VgtkCore, name)
    end

    # the registrations happen at load, not at precompilation: a fresh process
    # parses the stack's units
    pkgdir = dirname(dirname(pathof(VgtkCore)))
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir) -e 'using VisualGeometryToolkitCore; VisualGeometryToolkitCore.parse_unit("dpi")'`
    @test success(run(cmd))
end
