# =============================================================================
# read_length / write_length — decimal-exact file lengths
# =============================================================================

@testitem "lengths" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using FlexUnits: Quantity
    using VisualGeometryToolkitCore: cm, μm, inch, ft, pt, px
    km = parse_unit("km")
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

# IO 0.10.3: a length no number of its unit gives back, not even with its persisted identity, is refused by name,
# never written as the nearest number (whose identity is not the length's). No length unit is such a unit; a unit of
# 3e200 m is (its numbers of a 1e-110 m length are subnormal, on a grid coarser than an identity).
@testitem "lengths: a length no number of its unit has the identity of is refused" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    rng = Xoshiro(5)
    ys = [(1 + rand(rng)) * 1e-110 for _ in 1:200]
    outcome(y) = try
        c = VgtkIO._write_scaled(y, 3e200, 1)
        VgtkIO._persisted(c * 3e200) === VgtkIO._persisted(y) ? :identity : :other
    catch e
        e isa ArgumentError && occursin(repr(y), e.msg) ? :refused : :error
    end
    r = map(outcome, ys)
    @test all(in((:identity, :refused)), r)
    @test count(==(:refused), r) > 0
end
