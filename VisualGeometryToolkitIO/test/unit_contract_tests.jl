# =============================================================================
# the unit contract — FlexUnits' quantity / ustrip and the unit names every file uses
# =============================================================================

@testitem "contract" setup = [VGIoFixtures] begin
    # the file boundary is FlexUnits itself: quantity(x, u) in, ustrip(u, q) out
    @test ustrip(mm, 1.0m) ≈ 1000.0
    @test ustrip.(m, [1000.0mm, 500.0mm]) ≈ [1.0, 0.5]
    @test_throws ConversionError ustrip(mm, 3.0px)     # px into a length field
    # a bare number is not a length: a MethodError up to FlexUnits 0.6.14; from 0.6.15 a bare
    # Real is a dimensionless quantity, so it is the ConversionError of the line above
    @test_throws Union{MethodError, ConversionError} ustrip(mm, 3.0)
    @test ustrip(mm, quantity(0.1 + 0.2, mm)) === 0.1 + 0.2   # a file magnitude comes back exactly
    @test ustrip(mm, quantity(1.234, m)) === 1.234 * 1000.0
    # unit names are VisualGeometryToolkitCore's parse_unit / unit_string
    @test parse_unit("mm") == mm
    @test unit_string(parse_unit("um")) == "um"
    @test_throws ArgumentError parse_unit("mm * 2")     # one registered symbol, not an expression
    @test unit_string(m) == "m"
    @test_throws ArgumentError parse_unit("furlong")
    @test_throws ArgumentError UnitContract(world = px)  # px is not a length
    @test UnitContract(m) == UnitContract(world = m) && MM_PX == UnitContract(mm)
    # pixels are the only image unit, so there is no image unit to declare
    @test_throws MethodError UnitContract(world = mm, image = px)
end
