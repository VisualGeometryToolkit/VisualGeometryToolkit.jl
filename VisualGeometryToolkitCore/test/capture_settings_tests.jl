@testitem "capture settings" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: parse_unit, BaseQuantity, is_constant

    # The per-frame camera settings of a capture (captures.jl): the exposure,
    # the nominal lens settings and the identities of the hardware. Values are
    # math; identity is beside them, never inside a camera model.
    s = parse_unit("s")

    @testset "DeviceIdentity" begin
        d = DeviceIdentity("SONY", "ILME-FX3", nothing)
        @test d.make == "SONY" && d.model == "ILME-FX3" && d.serial === nothing
        @test d == DeviceIdentity(; make = "SONY", model = "ILME-FX3")
        @test DeviceIdentity() == DeviceIdentity(nothing, nothing, nothing)
        @test hash(d) == hash(DeviceIdentity(; make = "SONY", model = "ILME-FX3"))
        @test_throws ArgumentError DeviceIdentity(; model = "")       # unknown is nothing, never ""
    end

    @testset "ExposureSettings" begin
        e = ExposureSettings(0.004s, 8, 100)
        @test e.exposure_time isa BaseQuantity && ustrip(s, e.exposure_time) == 0.004
        @test e.f_number === 8.0 && e.iso === 100.0
        @test ExposureSettings((1 // 250) * s, 8.0, 100.0) == e     # an exact time is stored as its Float64
        @test_throws DimensionError ExposureSettings(0.004, 8, 100)      # a plain number: the unit is not stated
        @test_throws ConversionError ExposureSettings(4.0mm, 8, 100)     # a length is no time
        @test_throws ArgumentError ExposureSettings(0.0s, 8, 100)
        @test_throws ArgumentError ExposureSettings(Inf * s, 8, 100)
        @test_throws ArgumentError ExposureSettings(0.004s, 0, 100)
        @test_throws ArgumentError ExposureSettings(0.004s, 8, -100)
        @test_throws ArgumentError ExposureSettings(0.004s, NaN, 100)
    end

    @testset "LensSettings" begin
        l = LensSettings(14mm)
        @test ustrip(mm, l.focal_length) ≈ 14 && l.focus_distance === nothing
        l2 = LensSettings(14mm, 1.2m)
        @test l2.focus_distance isa BaseQuantity && ustrip(m, l2.focus_distance) == 1.2
        @test l2 == LensSettings(14.0mm, 1200mm)
        @test_throws ArgumentError LensSettings(0mm)
        @test_throws ArgumentError LensSettings(14mm, Inf * m)    # "infinity" or unknown is nothing
        @test_throws ArgumentError LensSettings(14mm, -1m)
        @test_throws DimensionError LensSettings(14.0)
        @test_throws ConversionError LensSettings(14mm, 2.0s)
    end

    @testset "CaptureSettings" begin
        e = ExposureSettings(0.004s, 8, 100)
        l = LensSettings(14mm)
        body, lens = DeviceIdentity(; make = "SONY", model = "ILME-FX3"), DeviceIdentity(; model = "DT 8mm F4 SAM")
        cs = CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 3 => (exposure = e, lens_settings = l)); body, lens)
        @test cs isa AbstractDict{Int, FrameSettings}
        @test length(cs) == 2 && sort!(collect(keys(cs))) == [1, 3] && haskey(cs, 3) && !haskey(cs, 2)
        @test cs[1].exposure == e && cs[3].lens_settings == l
        @test get(cs, 2, nothing) === nothing
        @test cs.body == body && cs.lens == lens
        @test CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 3 => (exposure = e, lens_settings = l)); body, lens) == cs
        @test CaptureSettings(Dict(1 => (exposure = e, lens_settings = l)); body, lens) != cs
        other = CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 3 => (exposure = e, lens_settings = l)); body)
        @test other != cs && !isequal(other, cs) && length(Set([cs, other])) == 2   # the identities count
        @test hash(cs) == hash(CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 3 => (exposure = e, lens_settings = l)); body, lens))
        @test CaptureSettings(Dict{Int, FrameSettings}()).body == DeviceIdentity()
        @test_throws ArgumentError CaptureSettings(Dict(1 => (exposure = e,)))   # a frame states both

        # is_constant: the same exposure time, f-number and ISO in every frame (the lens settings may differ)
        @test is_constant(cs)
        @test is_constant(CaptureSettings(Dict{Int, FrameSettings}()))            # vacuously
        @test is_constant(CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 2 => (exposure = e, lens_settings = LensSettings(20mm)))))
        for e2 in (ExposureSettings(0.008s, 8, 100), ExposureSettings(0.004s, 5.6, 100), ExposureSettings(0.004s, 8, 200))
            @test !is_constant(CaptureSettings(Dict(1 => (exposure = e, lens_settings = l), 2 => (exposure = e2, lens_settings = l))))
        end
    end
end
