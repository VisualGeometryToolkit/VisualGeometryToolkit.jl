# =============================================================================
# read_length / write_length — decimal-exact file lengths
# =============================================================================

@testitem "lengths" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using FlexUnits: Quantity
    using VisualGeometryToolkitCore: cm, μm, inch, ft, pt, px
    km = VisualGeometryToolkitCore.UnitRegistry.UNITS[:km]
    rng = Xoshiro(1)
    vals = [round(rand(rng) * 10.0^e; digits = d) for e in -1:4, d in 1:6, _ in 1:20_000]
    @test length(vals) == 720_000
    for u in (mm, cm, μm, m, km, inch, ft, pt)
        @test count(v -> repr(VgtkIO.write_length(VgtkIO.read_length(v, u), u)) != repr(v), vals) == 0
    end
    @test ustrip(m, VgtkIO.read_length(13.9, mm)) === 0.0139
    @test VgtkIO.read_length(62.80, mm) == VgtkIO.read_length(62.8, mm)
    @test VgtkIO.read_length(1, m) isa Quantity
    @test ustrip(VgtkIO.read_length(4.3e-5, mm)) === 4.3e-8     # an exponent already in repr(x)
    @test_throws ArgumentError VgtkIO.write_length(VgtkIO.read_length(1.0, mm), px)
    @test_throws ArgumentError VgtkIO.read_length(1.0, px)
    @test JSON.json(VgtkIO.write_length(VgtkIO.read_length(62.8, mm), mm)) == "62.8"
end
