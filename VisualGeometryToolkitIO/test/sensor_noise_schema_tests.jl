# vgio/sensor_noise/v1.0 (sensor_noise_schema.jl): Core's SensorNoise on synthetic values — every field present or
# `nothing`, every source, the dark maps and the defect mask as hashed artifacts; exact by type; the refusals.

@testitem "sensor_noise: round trips exact by type, every field present or nothing" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: SensorNoise, DarkFrame, DeviceIdentity, noise_value
    using Random: Xoshiro
    rng = Xoshiro(7)
    s = parse_unit("s")
    W, H = 6, 4
    dark = DarkFrame(rand(rng, Float32, H, W) .* 4f0 .+ 510f0, rand(rng, Float32, H, W) .* 2f0, (1 // 500 * s, 0.8 * s), 16)
    defects = falses(H, W); defects[2, 5] = true; defects[4, 1] = true
    body = DeviceIdentity("Sony", "ILME-FX3", "5012345")
    full = SensorNoise(; body, iso = 100,
        electrons_per_dn = noise_value(2.1, :measured; std_error = 0.01),
        read_noise = noise_value((1.5, 1.25, 1.75, 1.3), :estimated; std_error = (0.01, 0.02, 0.01, 0.02)),
        black = noise_value((512.0, 511.5, 512.25, 511.75), :file),
        dark = noise_value(dark, :measured), defects = noise_value(defects, :measured))
    unknown = SensorNoise(; body = DeviceIdentity("Sony", "ILME-FX3", nothing), iso = 3200.0,
        black = noise_value((512.0, 512.0, 512.0, 512.0), :file), reason = "too few flat pixels (12 of 2000)")
    partial = SensorNoise(; body, iso = 400, electrons_per_dn = noise_value(1.1, :estimated; std_error = 0.05),
        read_noise = noise_value((2.0, 2.0, 2.0, 2.0), :measured), dark = noise_value(dark, :measured))
    mktempdir() do dir
        for (k, n) in enumerate((full, unknown, partial))
            p = joinpath(dir, "noise$(k).json")
            save(p, n)
            @test validate(p) == "sensor_noise"
            b = load(SensorNoise, p)
            @test b == n && typeof(b) == typeof(n)
            for f in fieldnames(SensorNoise)
                @test isequal(getfield(b, f), getfield(n, f)) && typeof(getfield(b, f)) == typeof(getfield(n, f))
            end
        end
        d = JSON.parsefile(joinpath(dir, "noise1.json"))
        @test d["device"] == Dict("body" => Dict("make" => "Sony", "model" => "ILME-FX3", "serial" => "5012345"))
        @test d["electrons_per_dn"] == Dict("value" => 2.1, "source" => "measured", "std_error" => 0.01)
        @test d["black"] == Dict("value" => [512.0, 511.5, 512.25, 511.75], "source" => "file")
        @test d["dark"]["exposure_band"] == [0.002, 0.8] && d["dark"]["size"] == Dict("width" => W, "height" => H)
        @test d["dark"]["offset"]["file"] == "noise1.dark_offset.f32" && filesize(joinpath(dir, "noise1.dark_offset.f32")) == 4W * H
        @test d["defects"]["mask"]["file"] == "noise1.defects.u8" && findall(==(0x01), read(joinpath(dir, "noise1.defects.u8"))) == [(2 - 1) * W + 5, (4 - 1) * W + 1]   # row-major
        u = JSON.parsefile(joinpath(dir, "noise2.json"))
        @test !haskey(u, "electrons_per_dn") && !haskey(u, "dark") && u["reason"] == "too few flat pixels (12 of 2000)"
        @test u["device"]["body"] == Dict("make" => "Sony", "model" => "ILME-FX3")
    end
end

@testitem "sensor_noise: the artifacts are bound by their hash; the table's refusals" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: SensorNoise, DarkFrame, DeviceIdentity, noise_value
    s = parse_unit("s")
    dark = DarkFrame(fill(510f0, 2, 3), fill(1f0, 2, 3), (0.001 * s, 1.0 * s), 4)
    n = SensorNoise(; body = DeviceIdentity("Sony", "ILME-FX3", nothing), iso = 100, electrons_per_dn = noise_value(2.0, :measured),
                    read_noise = noise_value((1.0, 1.0, 1.0, 1.0), :measured), dark = noise_value(dark, :measured))
    mktempdir() do dir
        p = joinpath(dir, "n.json")
        save(p, n)
        good = read(p, String)
        # an artifact changed since: refused by its hash
        a = joinpath(dir, "n.dark_offset.f32")
        bytes = read(a); bytes[1] ⊻= 0x01; write(a, bytes)
        e = try load(SensorNoise, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("SHA-256", e.msg)
        save(p, n)
        refused(f!) = (d = JSON.parse(good; dicttype = Dict{String, Any}); f!(d); write(p, JSON.json(d));
                       try load(SensorNoise, p); false catch e; e isa ArgumentError end)
        @test refused(d -> (d["electrons_per_dn"]["source"] = "guessed"))            # a source outside the three
        @test refused(d -> (d["reason"] = "none"))                                    # a reason beside a known noise
        @test refused(d -> pop!(d, "read_noise"))                                     # an unknown noise without its reason
        @test refused(d -> (d["read_noise"]["value"] = [1.0, 1.0, 1.0]))              # per colour index: four
        @test refused(d -> (d["dark"]["size"]["width"] = 4))                          # the artifact's size
        @test refused(d -> (d["dark"]["offset"]["file"] = "../elsewhere.f32"))         # beside the record only
        @test refused(d -> (d["electrons_per_dn"]["value"] = nothing))                # no null
        @test !refused(identity)
    end
end
