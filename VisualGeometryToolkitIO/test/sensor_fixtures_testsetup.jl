# Shared by sensor_characterization_schema_tests.jl: Core's `sensor_fixture` (the one way a test builds a sensor
# component) on one body, a dark's synthetic maps, and a write-and-read round trip.
@testsetup module SensorFixtures
    using JSON: JSON
    using VisualGeometryToolkitIO: load, save
    using VisualGeometryToolkitCore: parse_unit
    export SEC, BODY, W, H, dark_maps, char, round_trip, same, V1_FIXTURE, V1_FIXTURE_SHA256, copy_v1_fixture

    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterization, DeviceIdentity, SensorSource
    using Random: Xoshiro
    const SEC = parse_unit("s")
    const BODY = DeviceIdentity("Sony", "ILME-FX3", "5012345")
    const W, H = 6, 4
    function dark_maps(seed)
        rng = Xoshiro(seed)
        return (offset = rand(rng, Float32, H, W) .* 4.0f0 .+ 510.0f0, variance = rand(rng, Float32, H, W) .* 2.0f0,
                exposure_band = (1 // 500 * SEC, 0.8 * SEC), frames = 16)
    end
    char(; kw...) = VgtkCore.sensor_fixture(; body = BODY, kw...).characterization
    # the v1.0 file the registered IO 0.10.1 wrote (test/fixtures/old/), and a copy of it with its rasters in `dir`
    const V1_FIXTURE = joinpath(@__DIR__, "fixtures", "old", "sensor_characterisation_v1.0.json")
    const V1_FIXTURE_SHA256 = "f28e8edc79b3fd7d58e618873a6f28b6d8c395353ca191a6e37226be3f6e9d12"   # its bytes as recorded
    function copy_v1_fixture(dir)
        for f in readdir(dirname(V1_FIXTURE))
            startswith(f, "sensor_characterisation_v1.0") && cp(joinpath(dirname(V1_FIXTURE), f), joinpath(dir, f))
        end
        return joinpath(dir, basename(V1_FIXTURE))
    end
    # a characterisation written and read back: the value, the file's tree
    function round_trip(dir, name, c)
        p = joinpath(dir, "$(name).json")
        save(p, c)
        return load(SensorCharacterization, p), JSON.parsefile(p), p
    end
    same(a, b) = a == b && keys(a.entries) == keys(b.entries) &&
        all(typeof(a.entries[k]) == typeof(b.entries[k]) && all(s -> typeof(getfield(a.entries[k], s)) == typeof(getfield(b.entries[k], s)),
                                                                (:black, :noise, :defects)) for k in keys(a.entries))
end
