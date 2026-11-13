# load_raw: the decoder (LibRaw_jll), the display demosaic — of a file and of a mosaic in memory —
# the detection image's refusal of the display image, the black-level rule, the EXIF fields and
# the decode cache. The sensor image the decoder produces (RawMosaic, LinearRaw, their maps) is
# VisualGeometryToolkitCore's since Core 0.9.2, exported by Core only and tested there
# (test/raw_mosaic_tests.jl). CI has no RAW file: a real-file check
# runs when VGTK_RAW_TEST_FILE names one, and is recorded as a skip otherwise.

@testitem "raw: the decoder needs its package" begin
    using VisualGeometryToolkitFeatures
    VGF = VisualGeometryToolkitFeatures
    @test_throws ArgumentError load_raw_mosaic(joinpath(@__DIR__, "no-such-file.ARW"))
    # LibRaw_jll is the one decoder (the rawpy extension is retired)
    @test !isdefined(VGF, Symbol("Raw" * "pyDecoder"))
    if Base.get_extension(VGF, :VisualGeometryToolkitFeaturesLibRawExt) === nothing
        for mode in (:bin2, :demosaic)
            err = try load_raw(@__FILE__; mode) catch e; e end
            @test err isa ArgumentError && occursin("using LibRaw_jll", err.msg)
        end
    else
        @test_skip "VisualGeometryToolkitFeaturesLibRawExt already loaded on this worker"
    end
end

@testitem "raw: the display demosaic keeps the sensor orientation (LibRaw_jll)" begin
    using VisualGeometryToolkitCore: LinearRaw, RawMosaic, raw_frame_map, bayer_plane_map, green_cell_sites, raw_provenance
    using VisualGeometryToolkitFeatures
    using LibRaw_jll: LibRaw_jll, libraw
    VGF = VisualGeometryToolkitFeatures
    X = Base.get_extension(VGF, :VisualGeometryToolkitFeaturesLibRawExt)
    # A synthetic Bayer buffer opened by LibRaw itself (libraw_open_bayer), RGGB, its top-left
    # quadrant dark and the rest bright, with the EXIF orientation set the way a camera records it
    # held upside down (flip 3: 180°) or upright (5: 90° CCW, 6: 90° CW). The display image keeps
    # the sensor's grid and its content where the sensor saw it: a 180° turn keeps H × W, so the
    # content is what tells it apart.
    H, W = 48, 64
    quadrants(H, W) = collect(reinterpret(UInt8, vec(permutedims(UInt16[(i <= H ÷ 2 && j <= W ÷ 2) ? 2000 : 30000 for i in 1:H, j in 1:W]))))
    function demosaic_with_flip(flip, H = H, W = W)
        buf = quadrants(H, W)
        lr = @ccall libraw.libraw_init(0::Cuint)::Ptr{Cvoid}
        try
            rc = @ccall libraw.libraw_open_bayer(lr::Ptr{Cvoid}, buf::Ptr{UInt8}, length(buf)::Cuint, W::Cushort, H::Cushort,
                0::Cushort, 0::Cushort, 0::Cushort, 0::Cushort, 0::Cuchar, 0x94::Cuchar, 0::Cuint, 0::Cuint, 512::Cuint)::Cint
            rc == 0 || error("libraw_open_bayer: $rc")
            (@ccall libraw.libraw_unpack(lr::Ptr{Cvoid})::Cint) == 0 || error("libraw_unpack")
            # the flag, in the live sizes and in the unpacked copy LibRaw's processing restores them from
            s = X._structures(lr)
            for sizes_p in (s.sizes_p, X._fieldptr(X._rawdata_ptr(s.other_p), X.RawData, :sizes))
                X._store_flip!(sizes_p, flip)
            end
            m = X._mosaic(lr, "synthetic.raw")
            # one demosaic per handle: its processing rewrites the live sizes (the flip zeroed), so a second call's
            # mirrored-layout check would refuse the handle
            out = Ref{Any}()
            a = @allocated (out[] = GC.@preserve buf X._demosaiced(lr, "synthetic.raw"))
            rgb, md = out[]
            return m, rgb, md, a
        finally
            @ccall libraw.libraw_close(lr::Ptr{Cvoid})::Cvoid
        end
    end
    # the demosaic copies the visible photosites only once, into its image (6 bytes a pixel: 3 × UInt16), never the
    # mosaic too (2 more: its checks read LibRaw's layout, `_mosaic_layout`). A call also allocates a fixed part that
    # does not grow with the image — the per-channel black table (LibRaw's 4104 cblack entries, 33 kB), the metadata's
    # strings and records — which dominates at 48 × 64 (measured on dgx01: 95 kB against an 18 kB image), so the
    # copies are told apart by the growth: warm (the first call compiles), at 48 × 64 and at 4× the pixels
    demosaic_with_flip(3)
    small = demosaic_with_flip(3); large = demosaic_with_flip(3, 2H, 2W)
    @test (large[4] - small[4]) / (3H * W) < 7                       # bytes per added pixel: the image's 6, not 6 + 2
    for flip in (3, 5, 6)
        m, rgb, md, _ = demosaic_with_flip(flip)
        L = VGF._demosaic_image(rgb, md, 1.0, nothing)
        @test L isa LinearRaw{:demosaic} && L.geometry === nothing
        @test size(L.image) == (H, W, 3)          # the sensor's grid: the flip is never applied
        g = L.image[:, :, 2]
        q = (1:20, 1:28)                          # well inside the dark quadrant
        @test maximum(g[q...]) < minimum(g[29:end, 37:end])   # dark top-left, bright bottom-right
        @test maximum(g[q...]) < minimum(g[1:20, 37:end])     # and bright top-right: no mirror either
        @test maximum(g[q...]) < minimum(g[29:end, 1:28])     # nor bright bottom-left
        # linear: the flat bright part reads its level, (raw − black) / (white − black) to LibRaw's 16 bits
        @test isapprox(g[40, 50], (30000 - m.black[2]) / (m.white - m.black[2]); rtol = 1e-3)
    end
    m, rgb, md = demosaic_with_flip(0)
    L = VGF._demosaic_image(rgb, md, 1.0, nothing)
    # the one demosaic of a mosaic in memory: LibRaw's, on the mosaic's photosites, phase and levels — bit for bit the
    # demosaic of the same buffer opened as a file is, with the mosaic's own metadata
    Lm = load_raw(m; mode = :demosaic)
    @test Lm isa LinearRaw{:demosaic} && Lm.image == L.image && Lm.saturated == L.saturated && Lm.metadata === m.metadata
    @test load_raw(m; mode = :demosaic, gain = 2.0, white_balance = (2.0, 1.0, 0.5)).image ==
          VGF._demosaic_image(rgb, md, 2.0, (2.0, 1.0, 0.5)).image
    # a corrected mosaic (its photosites changed in memory) is displayed through it: brighter photosites, a brighter image
    bright = RawMosaic(parent(m) .+ UInt16(1000), m.pattern, m.black, m.white, m.visible_origin, m.raw_size, m.metadata)
    @test sum(load_raw(bright; mode = :demosaic).image) > sum(Lm.image)
    # a window with an odd start reads another colour phase, and is demosaiced in it
    w = view(m, 2:(H - 1), 2:(W - 1))
    Lw = load_raw(w; mode = :demosaic)
    @test size(Lw.image) == (H - 2, W - 2, 3)
    @test Lw.image[10, 10, 2] ≈ L.image[11, 11, 2] rtol = 1e-2
    @test Lw.image[39, 49, 2] ≈ L.image[40, 50, 2] rtol = 1e-2
    # the cell modes of a mosaic are Core's LinearRaw, and whole levels are what LibRaw holds
    @test load_raw(m; mode = :green_cells).image == LinearRaw(m; mode = :green_cells).image
    half = RawMosaic(parent(m), m.pattern, m.black .+ 0.5, m.white, m.visible_origin, m.raw_size, m.metadata)
    @test_throws ArgumentError load_raw(half; mode = :demosaic)
    @test_throws ArgumentError load_raw(m; mode = :bin2, white_balance = (1.0, 1.0, 1.0))
    # display only: no frame map, no footprint, no sample positions
    @test_throws ArgumentError raw_frame_map(L)
    @test_throws ArgumentError green_cell_sites(L, 1, 1)
    @test_throws ArgumentError bayer_plane_map(L, 1)
    @test raw_provenance(L).mode == "demosaic" && raw_provenance(L).frame_map === nothing
    @test startswith(raw_provenance(L).libraw_version, "0.22")
    # white balance, if asked, is applied in Julia: per channel, after LibRaw's linear output
    Lw = VGF._demosaic_image(rgb, md, 2.0, (2.0, 1.0, 0.5))
    @test Lw.image[:, :, 1] == L.image[:, :, 1] .* 2.0 .* 2.0 && Lw.image[:, :, 3] == L.image[:, :, 3] .* 0.5 .* 2.0
end

@testitem "raw: the display image has no green cells to stabilize" begin
    using VisualGeometryToolkitCore: LinearRaw
    using VisualGeometryToolkitFeatures: variance_stabilize
    display_image = LinearRaw{:demosaic}(zeros(4, 4, 3), falses(4, 4), nothing, 1.0, (;))
    @test_throws ArgumentError variance_stabilize(display_image; electrons_per_dn = 2.5, read_noise = 3.0)
end

@testitem "raw: LibRaw_jll decoder" begin
    using VisualGeometryToolkitCore: LinearRaw
    using VisualGeometryToolkitFeatures
    using LibRaw_jll
    # the version check ran at load; a file LibRaw cannot read fails with its message
    err = try
        load_raw_mosaic(@__FILE__; decoder = LibRawDecoder())
    catch e
        e
    end
    @test err isa ErrorException && occursin("LibRaw cannot open", err.msg)
    # load_raw's cache key names the library and its build
    @test occursin(r"^LibRaw 0\.22\S*, LibRaw_jll 0\.22\.", VisualGeometryToolkitFeatures._decoder_version(LibRawDecoder()))
    path = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isfile(path)
        m = load_raw_mosaic(path; decoder = LibRawDecoder())
        @test all(<(m.white), m.black)
        @test startswith(m.metadata.libraw_version, "0.22")
        L = LinearRaw(m; mode = :bin2)
        @test size(L.image) == (size(m.raw) .÷ 2..., 3)
        @test LinearRaw(m; mode = :green_cells).image[:, :, 1] == L.image[:, :, 2]
        D = load_raw(path; mode = :demosaic)
        @test size(D.image) == (size(m.raw)..., 3)     # the sensor's grid, the visible area, no flip
        @test D.metadata == m.metadata
    else
        @test_skip "VGTK_RAW_TEST_FILE names no RAW file"
    end
end

@testitem "raw: black levels — a spatial pattern is folded or refused, never dropped" begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    adjust = VgtkFeatures._per_channel_black      # the one rule, for every decoder that reads LibRaw's cblack
    filters = 0x94949494                           # RGGB
    cblack(rows, cols, pattern; per_channel = (0, 0, 0, 0)) = begin
        cb = zeros(Int, 4104)                      # LIBRAW_CBLACK_SIZE
        cb[1:4] .= per_channel
        cb[5], cb[6] = rows, cols
        cb[7:(6 + rows * cols)] .= pattern
        cb
    end
    # no pattern: the per-channel levels plus black
    @test adjust(cblack(0, 0, Int[]; per_channel = (1, 2, 3, 4)), 100, filters, "a.ARW") == (101, 102, 103, 104)
    # a 2 × 2 pattern folds into the four channels exactly
    b = adjust(cblack(2, 2, [10, 20, 30, 40]), 100, filters, "a.ARW")
    @test sort(collect(b)) == [110, 120, 130, 140]
    # a uniform 4 × 4 pattern is a constant: it moves into black
    @test adjust(cblack(4, 4, fill(7, 16)), 100, filters, "a.ARW") == (107, 107, 107, 107)
    # a 4 × 4 pattern that varies has no per-channel form: refused, naming the file and the size
    err = try adjust(cblack(4, 4, collect(1:16)), 100, filters, "/data/b.dng"); nothing catch e; e end
    @test err isa ErrorException && occursin("/data/b.dng", err.msg) && occursin("4 × 4", err.msg)
end

@testitem "raw: EXIF as typed settings and identities" begin
    using Test
    using VisualGeometryToolkitFeatures
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitCore: ExposureSettings, LensSettings, DeviceIdentity, CaptureSettings, is_constant,
        parse_unit, mm, m, ustrip
    s = parse_unit("s")
    # a decoder's metadata, as the LibRaw decoder fills it for a Sony FX3 ARW behind an EF adapter
    md = (; datetime_original = "2026:10:29 14:03:12", subsec = "34", exposure_s = Float64(Float32(1 / 250)),
          fnumber = 8.0, focal_mm = 14.0, iso = 100, model = "ILME-FX3", make = "Sony", body_serial = nothing,
          lens_make = nothing, lens_model = "DT 8mm F4 SAM", lens_serial = nothing, focus_distance_m = nothing)
    e = VgtkFeatures.exposure_settings(md)
    @test e == ExposureSettings(Float64(Float32(1 / 250)) * s, 8, 100)        # LibRaw's Float32 shutter, as stored
    @test VgtkFeatures.lens_settings(md) == LensSettings(14.0mm)
    @test VgtkFeatures.body_identity(md) == DeviceIdentity(; make = "Sony", model = "ILME-FX3")
    @test VgtkFeatures.lens_identity(md) == DeviceIdentity(; model = "DT 8mm F4 SAM")
    @test VgtkFeatures.capture_time(md) == "2026-10-29T14:03:12.34"
    @test VgtkFeatures.capture_time(merge(md, (subsec = "",))) == "2026-10-29T14:03:12"
    @test VgtkFeatures.capture_time(merge(md, (datetime_original = nothing,))) === nothing
    # a focus distance the file reports at infinity, 0 or below is absent, never a distance
    for d in (Inf, 0.0, -1.0, NaN)
        @test VgtkFeatures._focus_distance(d) === nothing
        @test VgtkFeatures.lens_settings(merge(md, (focus_distance_m = VgtkFeatures._focus_distance(d),))).focus_distance === nothing
    end
    @test ustrip(m, VgtkFeatures.lens_settings(merge(md, (focus_distance_m = 1.5,))).focus_distance) == 1.5
    # metadata from before these fields (a 0.4.1 decode): the identities are what it states
    old = (; datetime_original = nothing, subsec = "", exposure_s = 0.01, fnumber = 4.0, focal_mm = 20.0, iso = 200, model = "ILME-FX3")
    @test VgtkFeatures.body_identity(old) == DeviceIdentity(; model = "ILME-FX3")
    @test VgtkFeatures.lens_identity(old) == DeviceIdentity()
    # a field the file does not state is named
    err = try VgtkFeatures.exposure_settings(merge(md, (iso = nothing,))) catch x; x end
    @test err isa ArgumentError && occursin("iso", err.msg)

    # EXIF's unknown time (blanks, the colons kept) is no time; SubjectDistance's 0xFFFFFFFF is infinity
    @test VgtkFeatures.capture_time(merge(md, (datetime_original = "    :  :     :  :  ",))) === nothing
    @test VgtkFeatures._focus_distance(4294967295.0) === nothing

    # extract_capture_settings: one body; the stated lens overrides EXIF's claim, warning with both
    md2 = merge(md, (iso = 200,))
    cs = extract_capture_settings(Dict(1 => md, 2 => md2))
    @test cs isa CaptureSettings && cs.lens == DeviceIdentity(; model = "DT 8mm F4 SAM") && !is_constant(cs)
    @test cs[2].exposure.iso == 200.0
    sigma = DeviceIdentity(; make = "SIGMA", model = "14mm F1.8 DG HSM | Art 017")
    cs2 = @test_logs (:warn, r"overrides") extract_capture_settings(Dict(1 => md, 2 => md); lens = sigma)
    @test cs2.lens == sigma && is_constant(cs2)
    @test_logs extract_capture_settings(Dict(1 => md); lens = DeviceIdentity(; model = "DT 8mm F4 SAM", serial = "123"))   # agrees: no warning
    # one body across files: one EXIF's "SONY" and its serial agree with LibRaw's "Sony", which states none
    cs3 = extract_capture_settings(Dict(1 => md, 2 => merge(md, (make = "SONY", body_serial = "5123"))))
    @test cs3.body == DeviceIdentity(; make = "Sony", model = "ILME-FX3", serial = "5123")
    @test_throws ArgumentError extract_capture_settings(Dict(1 => md, 2 => merge(md, (model = "ILCE-7M4",))))
    @test_throws ArgumentError extract_capture_settings(Dict(1 => md, 2 => merge(md, (lens_model = "FE 20mm F1.8 G",))))
    @test_throws ArgumentError extract_capture_settings(Dict{Int, NamedTuple}())
end

@testitem "raw: the metadata alone (LibRaw_jll)" begin
    using Test
    using VisualGeometryToolkitFeatures
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using LibRaw_jll
    @test_throws ArgumentError load_raw_metadata(joinpath(@__DIR__, "no-such-file.ARW"))
    path = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isfile(path)
        md = load_raw_metadata(path; decoder = LibRawDecoder())
        @test md == load_raw_mosaic(path; decoder = LibRawDecoder()).metadata     # the headers give what a decode gives
        @test VgtkFeatures.exposure_settings(md).exposure_time > zero(VgtkFeatures.exposure_settings(md).exposure_time)
        @test md.focus_distance_m === nothing
    else
        @test_skip "VGTK_RAW_TEST_FILE names no RAW file"
    end
end

@testitem "raw: a decode cache — each RAW decoded once per content and settings" begin
    using VisualGeometryToolkitCore: RawMosaic
    # The decoded-RAW cache PanoramaStitching's example kept as code-sum PNGs is
    # the package's one persistent cache (FileCache, JLD2): `load_raw(path; cache)`
    # keys an entry by the file's SHA-256 and the decode settings.
    using VisualGeometryToolkitFeatures
    using VisualGeometryToolkitFeatures: RawDecoder, open_cache, close_cache
    mosaic(raw) = RawMosaic(raw, UInt8[0 1; 3 2], (512.0, 500.0, 520.0, 508.0), 16383.0, (0, 0), size(raw), (; model = "synthetic"))
    using JLD2: JLD2
    struct CountingDecoder <: RawDecoder
        calls::Base.RefValue{Int}
    end
    function VisualGeometryToolkitFeatures._decode_raw(d::CountingDecoder, path::AbstractString)
        d.calls[] += 1
        v = only(reinterpret(UInt16, read(path, 2)))           # the file's content sets the scene
        return mosaic(fill(UInt16(1000 + v), 8, 8))
    end
    dir = mktempdir()
    raw = joinpath(dir, "a.raw"); write(raw, UInt16(7))
    d = CountingDecoder(Ref(0))
    cache = open_cache(joinpath(dir, "decoded.jld2"))
    try
        L1 = load_raw(raw; mode = :bin2, gain = 2, decoder = d, cache)
        L2 = load_raw(raw; mode = :bin2, gain = 2, decoder = d, cache)
        @test d.calls[] == 1                                    # the second is the cached value
        @test L2.image == L1.image && L2.saturated == L1.saturated && L2.metadata == L1.metadata
        @test L1.image == load_raw(raw; mode = :bin2, gain = 2, decoder = d).image   # no cache: the same values
        @test d.calls[] == 2
        load_raw(raw; mode = :bin2, gain = 1, decoder = d, cache)          # another setting: decoded
        load_raw(raw; mode = :green_cells, gain = 2, decoder = d, cache)
        @test d.calls[] == 4
        write(raw, UInt16(9))                                              # another content: decoded
        L3 = load_raw(raw; mode = :bin2, gain = 2, decoder = d, cache)
        @test d.calls[] == 5 && L3.image != L1.image
        load_raw(raw; mode = :bin2, gain = 2, decoder = d, cache = nothing)  # `nothing`: no cache
        @test d.calls[] == 6
    finally
        close_cache(cache)
    end
    # it persists: a reopened cache still holds the entries
    cache = open_cache(joinpath(dir, "decoded.jld2"))
    try
        load_raw(raw; mode = :bin2, gain = 2, decoder = d, cache)
        @test d.calls[] == 6
    finally
        close_cache(cache)
    end
    # the decoder library's version is in the key (`_decoder_version`, LibRaw's for LibRawDecoder): an upgraded
    # library decodes again rather than returning the old library's black levels, white level or crop
    struct VersionedDecoder <: RawDecoder
        calls::Base.RefValue{Int}
        version::Base.RefValue{String}
    end
    VisualGeometryToolkitFeatures._decoder_version(d::VersionedDecoder) = d.version[]
    VisualGeometryToolkitFeatures._decode_raw(d::VersionedDecoder, path::AbstractString) =
        (d.calls[] += 1; mosaic(fill(UInt16(3000), 8, 8)))
    @test VisualGeometryToolkitFeatures._decoder_version(CountingDecoder(Ref(0))) === nothing   # a decoder stating none
    v = VersionedDecoder(Ref(0), Ref("0.22.0-Release"))
    cache = open_cache(joinpath(dir, "versioned.jld2"))
    try
        load_raw(raw; mode = :bin2, decoder = v, cache); load_raw(raw; mode = :bin2, decoder = v, cache)
        @test v.calls[] == 1
        v.version[] = "0.22.1-Release"
        load_raw(raw; mode = :bin2, decoder = v, cache)
        @test v.calls[] == 2
    finally
        close_cache(cache)
    end
end

# The decode hooks' fallbacks are Core's `require_extension`: a decoder whose extension is not loaded names the package
# to load; a decoder whose extension is loaded but has no method for the arguments raises the call's MethodError.
@testitem "raw: a decoder without its extension names the package; with it, the call's MethodError" begin
    using VisualGeometryToolkitFeatures
    const VGF = VisualGeometryToolkitFeatures
    struct UnloadedDecoder <: VGF.RawDecoder end
    VGF._decoder_trigger(::UnloadedDecoder) = "SomeDecoder_jll"
    VGF._decoder_extension(::UnloadedDecoder) = :VisualGeometryToolkitFeaturesSomeDecoderExt
    err = try
        load_raw_mosaic(@__FILE__; decoder = UnloadedDecoder())
    catch e
        e
    end
    @test err isa ArgumentError && occursin("needs `using SomeDecoder_jll`", err.msg)
    @test_throws ArgumentError VGF._decode_demosaic(UnloadedDecoder(), @__FILE__)
    struct LoadedDecoder <: VGF.RawDecoder end                           # its extension (LibRaw's) is loaded below
    VGF._decoder_extension(::LoadedDecoder) = :VisualGeometryToolkitFeaturesLibRawExt
    using LibRaw_jll
    @test Base.get_extension(VGF, :VisualGeometryToolkitFeaturesLibRawExt) !== nothing
    @test_throws MethodError Base.invokelatest(VGF._decode_raw, LoadedDecoder(), @__FILE__)
end
