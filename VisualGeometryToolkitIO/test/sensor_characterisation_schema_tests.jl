# vgio/sensor_characterisation/v2.0 (sensor_characterisation_schema.jl): Core's SensorCharacterisation, its components
# built by Core's `sensor_fixture` (the one way a test builds one) — each component alone in an entry, partial entries,
# several ISOs, an empty entry; exact by type; a dark's maps as hashed artifacts; the refusals.

@testitem "sensor_characterisation: each component alone in an entry, exact by type" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource
    using VisualGeometryToolkitIO: validate
    import VisualGeometryToolkitIO as VgtkIO
    cases = [
        "channel_black_measured" => char(; black = (512.0, 511.5, 512.25, 511.75), black_std_error = (0.01, 0.02, 0.01, 0.02),
                                         source = SensorSource.measured),
        "channel_black_specified" => char(; black = (512.0, 512.0, 512.0, 512.0)),
        "read_noise" => char(; read_noise = (1.5, 1.25, 1.75, 1.3), read_noise_std_error = (0.01, 0.02, 0.01, 0.02)),
        "poisson_gaussian_measured" => char(; gain = 2.1, gain_std_error = 0.01, read_noise = (1.5, 1.25, 1.75, 1.3),
                                            read_noise_std_error = (0.01, 0.02, 0.01, 0.02), source = SensorSource.measured),
        "poisson_gaussian_estimated" => char(; gain = 1.1, read_noise = (2.0, 2.0, 2.0, 2.0), source = SensorSource.estimated),
        "relative_noise" => char(; relative_noise = 0.01),
        "dark" => char(; black = (512.0, 511.5, 512.25, 511.75), black_std_error = (0.01, 0.02, 0.01, 0.02), dark = dark_maps(7),
                       source = SensorSource.measured),
        "dark_raw_black" => char(; dark = dark_maps(8), source = SensorSource.measured),
        "defects" => char(; defects = (x = [5, 1, 6], y = [2, 4, 4], onset = [1 // 500 * SEC, 0.25 * SEC, 1.3 * SEC], size = (H, W)),
                          source = SensorSource.measured),
    ]
    mktempdir() do dir
        for (name, c) in cases
            b, d, p = round_trip(dir, name, c)
            @test validate(p) == "sensor_characterisation"
            @test same(b, c)
            @test d["device"] == Dict("body" => Dict("make" => "Sony", "model" => "ILME-FX3", "serial" => "5012345"))
            @test length(d["entries"]) == 1 && d["entries"][1]["iso"] == 100
        end
        e(name) = JSON.parsefile(joinpath(dir, "$(name).json"))["entries"][1]
        @test e("channel_black_measured")["black"] == Dict("type" => "channel", "values" => [512.0, 511.5, 512.25, 511.75],
                                                          "std_error" => [0.01, 0.02, 0.01, 0.02], "source" => "measured")
        @test !haskey(e("channel_black_specified")["black"], "std_error") && e("channel_black_specified")["black"]["source"] == "specified"
        @test e("poisson_gaussian_measured")["noise"] == Dict("type" => "poisson_gaussian", "gain" => 2.1, "sigma" => [1.5, 1.25, 1.75, 1.3],
            "std_error" => Dict("gain" => 0.01, "sigma" => [0.01, 0.02, 0.01, 0.02]), "source" => "measured")
        @test e("relative_noise")["noise"] == Dict("type" => "relative", "sigma_rel" => 0.01, "source" => "specified")
        # a slot not stated is absent from the entry (it loads as its absent type)
        @test collect(keys(e("read_noise"))) == ["iso", "noise"] && collect(keys(e("defects"))) == ["iso", "defects"]
        # a dark: its band in seconds, its maps the raster payload body (float32 TIFFs of the raster profile, row 1 on top);
        # its channel black inside it, absent when it is the RAW's
        dk = e("dark")["black"]
        @test dk["type"] == "dark" && dk["exposure_band"] == [0.002, 0.8] && dk["frames"] == 16 && dk["size"] == Dict("width" => W, "height" => H)
        @test dk["black"]["type"] == "channel" && !haskey(e("dark_raw_black")["black"], "black")
        @test dk["offset"]["file"] == "dark.iso100.dark_offset.tiff" &&
              VgtkIO.read_tiff(Float32, joinpath(dir, "dark.iso100.dark_offset.tiff")) == dark_maps(7).offset
        # a defect site [x, y, onset]: 1-based, x the column, onset in seconds, exact
        @test e("defects")["defects"] == Dict("size" => Dict("width" => W, "height" => H), "source" => "measured",
                                              "sites" => [[5, 2, 0.002], [1, 4, 0.25], [6, 4, 1.3]])
    end
end

@testitem "sensor_characterisation: partial entries, several ISOs, an empty entry" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource
    using VisualGeometryToolkitCore: FileBlack, NoNoise, NoDefects, ChannelBlack, PoissonGaussian, DefectList
    defects = (x = [2], y = [3], onset = [0.5 * SEC], size = (H, W))
    partial_bd = char(; iso = 100, black = (512.0, 512.0, 512.0, 512.0), defects, source = SensorSource.measured)
    noise_only = char(; iso = 400, gain = 1.1, read_noise = (2.0, 2.0, 2.0, 2.0), source = SensorSource.estimated)
    empty = char(; iso = 3200.0)
    many = VgtkCore.sensor_fixture(partial_bd, noise_only, empty, char(; iso = 1600, relative_noise = 0.02))
    mktempdir() do dir
        for (name, c) in ("partial" => partial_bd, "noise_only" => noise_only, "empty" => empty, "many" => many)
            b, _, _ = round_trip(dir, name, c)
            @test same(b, c)
        end
        b = load(SensorCharacterisation, joinpath(dir, "many.json"))
        @test sort!(collect(keys(b.entries))) == [100.0, 400.0, 1600.0, 3200.0]
        @test b.entries[100.0].black isa ChannelBlack && b.entries[100.0].noise isa NoNoise && b.entries[100.0].defects isa DefectList
        @test b.entries[400.0].black isa FileBlack && b.entries[400.0].noise isa PoissonGaussian && b.entries[400.0].defects isa NoDefects
        @test b.entries[3200.0].black isa FileBlack && b.entries[3200.0].noise isa NoNoise && b.entries[3200.0].defects isa NoDefects
        d = JSON.parsefile(joinpath(dir, "many.json"))
        @test [x["iso"] for x in d["entries"]] == [100, 400, 1600, 3200]          # in increasing ISO
        @test d["entries"][4] == Dict("iso" => 3200)                              # an empty entry: its ISO alone
    end
end

@testitem "sensor_characterisation: the artifacts are bound by their hash; the refusals" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource
    using VisualGeometryToolkitIO: validate
    c = VgtkCore.sensor_fixture(
        char(; iso = 100, dark = (offset = fill(510.0f0, 2, 3), variance = fill(1.0f0, 2, 3), exposure_band = (0.001 * SEC, 1.0 * SEC), frames = 4),
             gain = 2.0, read_noise = (1.0, 1.0, 1.0, 1.0),
             defects = (x = [1, 3], y = [2, 1], onset = [0.01 * SEC, 0.02 * SEC], size = (2, 3)), source = SensorSource.measured),
        char(; iso = 200, relative_noise = 0.01))
    mktempdir() do dir
        p = joinpath(dir, "n.json")
        save(p, c)
        good = read(p, String)
        # a dark's map changed since: refused by its hash, before it is decoded
        a = joinpath(dir, "n.iso100.dark_offset.tiff")
        bytes = read(a); bytes[1] ⊻= 0x01; write(a, bytes)
        e = try load(SensorCharacterisation, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("corrupted or replaced", e.msg)
        save(p, c)
        refused(f!) = (d = JSON.parse(good; dicttype = Dict{String, Any}); f!(d); write(p, JSON.json(d));
                       try load(SensorCharacterisation, p); false catch e; e isa ArgumentError end)
        e1(d) = d["entries"][1]
        @test !refused(identity)
        @test refused(d -> (e1(d)["defects"]["sites"][1] = [4, 1, 0.01]))            # a defect outside the raw (x ≤ 3)
        @test refused(d -> (e1(d)["defects"]["sites"][1] = [1, 0, 0.01]))            # 1-based
        @test refused(d -> (e1(d)["defects"]["sites"][1] = [1.5, 1, 0.01]))          # a photosite, not a position
        @test refused(d -> (e1(d)["defects"]["sites"][1][3] = 0.0))                  # onset ≤ 0
        @test refused(d -> (e1(d)["defects"]["sites"][2] = e1(d)["defects"]["sites"][1]))   # a photosite twice
        @test refused(d -> (d["entries"][2]["iso"] = 100))                            # an ISO twice
        @test refused(d -> (e1(d)["iso"] = 0))                                        # a nonpositive ISO
        @test refused(d -> (e1(d)["noise"]["source"] = "file"))                       # file: the RAW's own black only
        @test refused(d -> (e1(d)["noise"]["source"] = "guessed"))
        @test refused(d -> (e1(d)["noise"]["type"] = "shot_noise"))                   # not a noise component's type
        @test refused(d -> (e1(d)["noise"]["gain"] = 0.0))
        @test refused(d -> (e1(d)["noise"]["sigma"] = [1.0, 1.0, 1.0]))               # per CFA colour index: four
        @test refused(d -> (d["entries"][2]["noise"]["sigma_rel"] = -0.01))
        # a relative noise is specified only (owner ruling): another source is the constructor's refusal, on load
        for src in ("measured", "estimated")
            d = JSON.parse(good; dicttype = Dict{String, Any}); d["entries"][2]["noise"]["source"] = src; write(p, JSON.json(d))
            e = try load(SensorCharacterisation, p); nothing catch e; e end
            @test e isa ArgumentError && occursin("RelativeNoise: a relative noise is stated by a person", e.msg) && occursin("got source $(src)", e.msg)
        end
        @test refused(d -> (e1(d)["black"]["frames"] = 1))                            # a variance needs 2 frames
        @test refused(d -> (e1(d)["black"]["exposure_band"] = [1.0, 0.001]))
        @test refused(d -> (e1(d)["black"]["size"]["width"] = 4))                     # the raster's size
        @test refused(d -> (e1(d)["black"]["offset"]["file"] = "../elsewhere.tiff"))  # beside the record only
        @test refused(d -> (e1(d)["dark"] = e1(d)["black"]))                          # a dark is the black slot's
        # the retired kind (never shipped, no migration): an unknown kind
        d = JSON.parse(good; dicttype = Dict{String, Any}); d["schema"] = "vgio/sensor_" * "noise/v1.0"; write(p, JSON.json(d))
        e = try validate(p); nothing catch e; e end
        @test e isa ArgumentError && occursin("not a vgio file", e.msg)
    end
end

# v2.0 (IO 0.10.2): the black slot's `Darks`, one Dark per exposure band (each its own raster pair), with the shared
# channel black; a v1.0 file (a single Dark) still loads.
@testitem "sensor_characterisation: Darks — a raster pair per band, the round trip, a stable digest; v1.0 still loads" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource, Darks, Dark, content_digest, sensor_model
    using VisualGeometryToolkitIO: validate
    import VisualGeometryToolkitIO as VgtkIO
    bands = [(1 // 8000 * SEC, 1 // 8000 * SEC), (1 // 500 * SEC, 1 // 500 * SEC), (0.8 * SEC, 0.8 * SEC)]
    darks = [(; dark_maps(10 + k)..., exposure_band = b) for (k, b) in enumerate(bands)]
    c = char(; black = (512.0, 511.5, 512.25, 511.75), black_std_error = (0.01, 0.02, 0.01, 0.02), darks, gain = 2.1,
             read_noise = (1.5, 1.25, 1.75, 1.3), source = SensorSource.measured)
    @test c.entries[100.0].black isa Darks
    mktempdir() do dir
        b, d, p = round_trip(dir, "darks", c)
        @test validate(p) == "sensor_characterisation" && d["schema"] == "vgio/sensor_characterisation/v2.0"
        @test same(b, c) && b.entries[100.0].black isa Darks
        bk = d["entries"][1]["black"]
        @test bk["type"] == "darks" && bk["black"]["type"] == "channel" && length(bk["darks"]) == 3
        @test [x["exposure_band"] for x in bk["darks"]] == [[1 / 8000, 1 / 8000], [0.002, 0.002], [0.8, 0.8]]
        @test [x["offset"]["file"] for x in bk["darks"]] == ["darks.iso100.band$(k).dark_offset.tiff" for k in 1:3]
        @test VgtkIO.read_tiff(Float32, joinpath(dir, "darks.iso100.band2.dark_variance.tiff")) == darks[2].variance
        # a stable identity: every frame's resolved model is the original's, digest and all, and a second save is the same file
        caps = [VgtkCore.sensor_fixture(; body = BODY, exposure = t).capture for t in [first.(bands); 0.01 * SEC]]   # one frame each
        @test all(content_digest(sensor_model(b, cap, 1)) == content_digest(sensor_model(c, cap, 1)) for cap in caps)
        @test sensor_model(b, caps[2], 1).black isa Dark && sensor_model(b, caps[4], 1).reason === VgtkCore.DarkOutOfBand
        p2 = joinpath(dir, "again.json"); save(p2, b)
        @test replace(read(p2, String), "again." => "darks.") == read(p, String)
        # the bands overlapping: refused on read
        good = read(p, String)
        t = JSON.parse(good; dicttype = Dict{String, Any}); t["entries"][1]["black"]["darks"][2]["exposure_band"] = [0.0001, 0.9]
        write(p, JSON.json(t))
        @test (try load(SensorCharacterisation, p); false catch e; e isa ArgumentError end)
    end
end

# The v1.0 file the REGISTERED IO 0.10.1 wrote (test/fixtures/old/, with its two rasters): read through its step, the
# characterisation it held.
@testitem "sensor_characterisation: a v1.0 file IO 0.10.1 wrote reads through its step" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource, Dark, ChannelBlack, PoissonGaussian, DefectList
    using VisualGeometryToolkitIO: validate
    import VisualGeometryToolkitIO as VgtkIO
    src = V1_FIXTURE
    @test JSON.parsefile(src)["schema"] == "vgio/sensor_characterisation/v1.0" && JSON.parsefile(src)["generator"]["version"] == "0.10.1"
    c = load(SensorCharacterisation, src)
    e = c.entries[100.0]
    @test e.black isa Dark && e.black.black isa ChannelBlack && e.noise isa PoissonGaussian && e.defects isa DefectList
    @test e.black.offset == Float32[510 + i + 0.25j for i in 1:4, j in 1:6] && e.black.frames == 16
    @test e.noise.gain == 2.1 && e.defects.x == [5] && e.defects.y == [2]
    @test validate(src) == "sensor_characterisation"
    @test VgtkIO.schema_versions("sensor_characterisation") == ["vgio/sensor_characterisation/v1.0", "vgio/sensor_characterisation/v2.0"]
    @test VgtkIO._written_versions("sensor_characterisation") == [(1, 0), (2, 0)]
    @test_throws ArgumentError VgtkIO.json_schema("sensor_characterisation", (3, 0))
    @test_throws ArgumentError VgtkIO.json_schema("cameras", (5, 0))            # a version no writer emits
    mktempdir() do dir
        # a version the writer emits (no Darks): current, nothing for vgio_migrate to rewrite (IO 0.10.3)
        @test_throws ArgumentError VgtkIO.vgio_migrate(copy_v1_fixture(dir))
        @test !VgtkIO._has_migration("vgio/sensor_characterisation/v1.0")
        # a "v1.0" file holding a darks body is no v1.0 file: refused (IO 0.10.1 refuses it too)
        d = char(; black = (512.0, 511.5, 512.25, 511.75), source = SensorSource.measured,
                 darks = [(; dark_maps(3)..., exposure_band = (0.8 * SEC, 0.8 * SEC))])
        p = joinpath(dir, "fake.json"); save(p, d)
        t = JSON.parse(read(p, String); dicttype = Dict{String, Any}); t["schema"] = "vgio/sensor_characterisation/v1.0"
        write(p, JSON.json(t))
        @test (try load(SensorCharacterisation, p); false catch e; e isa ArgumentError end)
        @test (try validate(p); false catch e; e isa ArgumentError end)
    end
end

# IO 0.10.2's writer (owner ruling (a), 2026-11-14): the kind version follows what the record holds — v2.0 when an
# entry's black is a Darks, else v1.0 exactly as IO 0.10.1 wrote it, the same bytes but the generator's version (and
# the rasters byte for byte), so a characterisation without Darks stays readable by IO 0.10.1.
@testitem "sensor_characterisation: v1.0 unless the record holds a Darks — IO 0.10.1's bytes; v2.0 with Darks" setup = [VGIoFixtures, SensorFixtures] begin
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource, Darks
    import VisualGeometryToolkitIO as VgtkIO
    old = joinpath(@__DIR__, "fixtures", "old")
    src = joinpath(old, "sensor_characterisation_v1.0.json")
    c = load(SensorCharacterisation, src)
    mktempdir() do dir
        p = joinpath(dir, "sensor_characterisation_v1.0.json")
        save(p, c)
        mine = JSON.parsefile(p)
        @test mine["schema"] == "vgio/sensor_characterisation/v1.0"
        # the same bytes as IO 0.10.1's file, the generator's version aside
        @test replace(read(p, String), "\"version\": \"$(pkgversion(VgtkIO))\"" => "\"version\": \"0.10.1\"") == read(src, String)
        for f in ("sensor_characterisation_v1.0.iso100.dark_offset.tiff", "sensor_characterisation_v1.0.iso100.dark_variance.tiff")
            @test read(joinpath(dir, f)) == read(joinpath(old, f))
        end
        @test load(SensorCharacterisation, p) == c
        # with a Darks: v2.0, round trip
        d = char(; black = (512.0, 511.5, 512.25, 511.75), source = SensorSource.measured,
                 darks = [(; dark_maps(3)..., exposure_band = (1 // 500 * SEC, 1 // 500 * SEC)), (; dark_maps(4)..., exposure_band = (0.8 * SEC, 0.8 * SEC))])
        b, t, _ = round_trip(dir, "with_darks", d)
        @test t["schema"] == "vgio/sensor_characterisation/v2.0" && same(b, d)
        # several ISOs: one Darks among them is enough
        both = VisualGeometryToolkitCore.sensor_fixture(c, VisualGeometryToolkitCore.sensor_fixture(; body = c.body, iso = 400, black = (512.0, 512.0, 512.0, 512.0),
            darks = [(; dark_maps(5)..., exposure_band = (0.8 * SEC, 0.8 * SEC))], source = SensorSource.measured).characterisation)
        b2, t2, _ = round_trip(dir, "mixed", both)
        @test t2["schema"] == "vgio/sensor_characterisation/v2.0" && same(b2, both)
    end
end

# IO 0.10.3: a v1.0 file is a version a writer emits (`_written_versions`), so the tree migration finds it current:
# a rerun is idempotent (0.10.2 rewrote it beside itself, then refused the existing `.migrated.json` and migrated that).
@testitem "vgio_migrate_tree: a v1.0 sensor characterisation is current; a rerun is idempotent" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        c = char(; black = (512.0, 511.5, 512.25, 511.75), read_noise = (1.5, 1.25, 1.75, 1.3))
        p = joinpath(dir, "fx3.json")
        save(p, c)
        @test JSON.parsefile(p)["schema"] == "vgio/sensor_characterisation/v1.0"
        before = read(p, String)
        for _ in 1:2
            io = IOBuffer()
            @test (VgtkIO.vgio_migrate_tree(dir; io); true)
            @test occursin("current   $(p): vgio/sensor_characterisation/v1.0", String(take!(io)))
        end
        @test !any(f -> occursin("migrated", f), readdir(dir)) && read(p, String) == before
    end
end

# IO 0.10.3: the v1.0 file form is gone. It wrote the rasters of a Dark into the staging directory, which is removed
# once the file is renamed into place: a file migrated to another directory had no rasters. A v1.0 file is current
# (a version the writer emits): nothing to migrate.
@testitem "vgio_migrate: a v1.0 file holding a Dark is current, never rewritten without its rasters" setup = [VGIoFixtures, SensorFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        a, b = mkpath(joinpath(dir, "a")), mkpath(joinpath(dir, "b"))
        @test_throws r"nothing to migrate" VgtkIO.vgio_migrate(copy_v1_fixture(a); out = joinpath(b, "sc.json"))
        @test isempty(readdir(b))                                          # never a file whose rasters are missing
    end
end

# IO 0.10.3: the smaller rules — a v1 step for every v1 minor read, a non-dict black refused by name, the darks of a
# darks body listed in increasing band (as the writer writes them).
@testitem "sensor_characterisation: v1 without a minor, a non-dict black, darks out of band order" setup = [VGIoFixtures, SensorFixtures, OldVersions] begin
    using VisualGeometryToolkitCore: SensorCharacterisation, SensorSource
    using VisualGeometryToolkitIO: validate
    import VisualGeometryToolkitIO as VgtkIO
    c = load(SensorCharacterisation, V1_FIXTURE)
    mktempdir() do dir
        p = copy_v1_fixture(dir)
        t = json_tree(p)
        # "v1" is v1.0 (`_split_schema`): read through the same step
        t["schema"] = "vgio/sensor_characterisation/v1"
        write_tree(p, t)
        @test load(SensorCharacterisation, p) == c && validate(p) == "sensor_characterisation"
        @test VgtkIO.schema_versions("sensor_characterisation") == ["vgio/sensor_characterisation/v1.0", "vgio/sensor_characterisation/v2.0"]
        # a black that is not an object: an ArgumentError, from the step as from the reader
        t["schema"] = "vgio/sensor_characterisation/v1.0"
        t["entries"][1]["black"] = 3
        @test_throws ArgumentError VgtkIO.migrate_record(t)
        write_tree(p, t)
        @test_throws ArgumentError load(SensorCharacterisation, p)
        # a darks body's darks out of band order: refused (the writer writes them in increasing band)
        d = char(; black = (512.0, 511.5, 512.25, 511.75), source = SensorSource.measured,
                 darks = [(; dark_maps(3)..., exposure_band = (1 // 500 * SEC, 1 // 500 * SEC)), (; dark_maps(4)..., exposure_band = (0.8 * SEC, 0.8 * SEC))])
        q = joinpath(dir, "darks.json")
        save(q, d)
        @test validate(q) == "sensor_characterisation"
        u = json_tree(q)
        reverse!(u["entries"][1]["black"]["darks"])
        write_tree(q, u)
        @test_throws r"increasing" validate(q)
        u["entries"][1]["black"] = 3
        write_tree(q, u)
        @test_throws ArgumentError validate(q)
    end
end
