# Shared by sensor_characterisation_schema_tests.jl: Core's `sensor_fixture` (the one way a test builds a sensor
# component) on one body, a dark's synthetic maps, and a write-and-read round trip.
@testsetup module SensorFixtures
    using JSON: JSON
    using VisualGeometryToolkitIO: load, save
    using VisualGeometryToolkitCore: parse_unit
    export SEC, BODY, W, H, dark_maps, char, round_trip, same

    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, DeviceIdentity, SensorSource
    using Random: Xoshiro
    const SEC = parse_unit("s")
    const BODY = DeviceIdentity("Sony", "ILME-FX3", "5012345")
    const W, H = 6, 4
    function dark_maps(seed)
        rng = Xoshiro(seed)
        return (offset = rand(rng, Float32, H, W) .* 4.0f0 .+ 510.0f0, variance = rand(rng, Float32, H, W) .* 2.0f0,
                exposure_band = (1 // 500 * SEC, 0.8 * SEC), frames = 16)
    end
    char(; kw...) = VgtkCore.sensor_fixture(; body = BODY, kw...).characterisation
    # a characterisation written and read back: the value, the file's tree
    function round_trip(dir, name, c)
        p = joinpath(dir, "$(name).json")
        save(p, c)
        return load(SensorCharacterisation, p), JSON.parsefile(p), p
    end
    same(a, b) = a == b && keys(a.entries) == keys(b.entries) &&
        all(typeof(a.entries[k]) == typeof(b.entries[k]) && all(s -> typeof(getfield(a.entries[k], s)) == typeof(getfield(b.entries[k], s)),
                                                                (:black, :noise, :defects)) for k in keys(a.entries))
end
