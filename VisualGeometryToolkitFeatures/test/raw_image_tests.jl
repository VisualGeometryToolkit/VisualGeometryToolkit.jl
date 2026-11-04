# load_raw: the processing on synthetic mosaics (black/white, gain, the 2×2
# binning, partial-saturation propagation, the true sample positions for the
# four Bayer layouts, the green photosites), and the decoders where they can
# run. CI has no RAW file: a real-file check runs when VGTK_RAW_TEST_FILE names
# one, and is recorded as a skip otherwise.

@testsetup module RawFixtures
    using VisualGeometryToolkitFeatures: RawMosaic
    export mosaic, LAYOUTS
    # LibRaw colour indices: 0 R, 1 G, 2 B, 3 the second G
    const LAYOUTS = (
        RGGB = UInt8[0 1; 3 2], BGGR = UInt8[2 1; 3 0],
        GRBG = UInt8[1 0; 2 3], GBRG = UInt8[1 2; 0 3],
    )
    function mosaic(raw::AbstractMatrix{<:Integer}, cell::Matrix{UInt8}; black = (512.0, 500.0, 520.0, 508.0), white = 16383.0)
        H, W = size(raw)
        colors = UInt8[cell[mod1(i, 2), mod1(j, 2)] for i in 1:H, j in 1:W]
        meta = (; model = "synthetic")
        return RawMosaic(Matrix{UInt16}(raw), colors, cell, black, white, (0, 0), (H, W), meta)
    end
end

@testitem "raw: linear values, gain and the 2×2 binning" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, cfa_string
    black = (512.0, 500.0, 520.0, 508.0)
    white = 16383.0
    raw = UInt16[1000 2000 1500 900; 3000 4000 700 16000; 5 6 7 8; 9 10 11 12]   # row 3–4: below black
    m = mosaic(raw, LAYOUTS.RGGB; black, white)
    @test cfa_string(m) == "RGGB"
    lin(v, c) = (v - black[c + 1]) / (white - black[c + 1])
    L = LinearRaw(m; mode = :bin2)
    @test size(L.image) == (2, 2, 3) && L.mode === :bin2 && L.gain == 1.0
    @test all(L.sampled)
    # quad (1, 1): R = raw[1,1], G = mean of raw[1,2] (c 1) and raw[2,1] (c 3), B = raw[2,2]
    @test L.image[1, 1, 1] == lin(1000, 0)
    @test L.image[1, 1, 2] == (lin(2000, 1) + lin(3000, 3)) / 2
    @test L.image[1, 1, 3] == lin(4000, 2)
    @test L.image[2, 1, 1] < 0                                    # noise under the black level is kept
    g = 8.0
    Lg = LinearRaw(m; mode = :bin2, gain = g)
    @test Lg.image[1, 2, 2] == ((lin(900, 1) + lin(700, 3)) / 2) * g   # convert_raw.py's order: mean, then gain
    @test Lg.image[1, 2, 3] == lin(16000, 2) * g && Lg.image[1, 2, 3] > 1   # not clipped
    @test !any(Lg.saturated)                                      # a gain clip is not sensor saturation
    # an odd last row and column are dropped
    @test size(LinearRaw(mosaic(rand(UInt16(600):UInt16(9000), 5, 7), LAYOUTS.RGGB); mode = :bin2).image) == (2, 3, 3)
    @test_throws ArgumentError LinearRaw(m; mode = :fullres)
    @test_throws ArgumentError LinearRaw(m; mode = :bin2, gain = 0)
    @test_throws ArgumentError LinearRaw(m; mode = :bin2, gain = Inf)
end

@testitem "raw: a quad is saturated when any photosite is" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw
    white = 16383.0
    raw = fill(UInt16(3000), 4, 6)
    raw[1, 2] = 16383                     # quad (1, 1): one green at the white level, the other not
    raw[3, 3] = 16383                     # quad (2, 2): its red
    raw[4, 6] = 16390                     # quad (2, 3): above white counts too
    L = LinearRaw(mosaic(raw, LAYOUTS.RGGB; white); mode = :bin2)
    @test L.saturated == BitMatrix([1 0 0; 0 1 1])
    @test 0 < L.image[1, 1, 2] < 1        # the average that looks valid, and is not
    raw[1, 2] = 16382
    @test !LinearRaw(mosaic(raw, LAYOUTS.RGGB; white); mode = :bin2).saturated[1, 1]   # no margin below white
end

@testitem "raw: true sample positions for the four Bayer layouts" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw
    q = 0.25
    expected = Dict(                       # (R, G, B) offsets (Δx, Δy) from the binned pixel centre
        :RGGB => [(-q, -q), (0.0, 0.0), (q, q)],
        :BGGR => [(q, q), (0.0, 0.0), (-q, -q)],
        :GRBG => [(q, -q), (0.0, 0.0), (-q, q)],
        :GBRG => [(-q, q), (0.0, 0.0), (q, -q)],
    )
    raw = UInt16[600 + 100 * (2 * (i - 1) + (j - 1)) for i in 1:2, j in 1:2]   # distinct per site
    for (name, cell) in pairs(LAYOUTS)
        L = LinearRaw(mosaic(raw, cell); mode = :bin2)
        @test Tuple.(L.plane_offsets) == expected[name]
        # the R plane is the photosite the offset points at: (Δy, Δx) = ((di − ½)/2, (dj − ½)/2)
        o = L.plane_offsets[1]
        di, dj = Int(2o[2] + 0.5) + 1, Int(2o[1] + 0.5) + 1
        @test cell[di, dj] == 0
        @test L.image[1, 1, 1] == (raw[di, dj] - 512.0) / (16383.0 - 512.0)
    end
    # a pattern that is not a 2×2 Bayer quad is refused
    bad = mosaic(raw, UInt8[0 0; 2 2])
    @test_throws ArgumentError LinearRaw(bad; mode = :bin2)
end

@testitem "raw: the green photosites on the raw grid" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw
    for cell in (LAYOUTS.RGGB, LAYOUTS.GRBG)
        raw = UInt16[1000 + 37 * i + 11 * j for i in 1:4, j in 1:6]
        raw[findfirst(c -> c == 1, [cell[mod1(i, 2), mod1(j, 2)] for i in 1:4, j in 1:6])] = 16383
        m = mosaic(raw, cell)
        L = LinearRaw(m; mode = :green, gain = 2)
        green = (m.colors .== 1) .| (m.colors .== 3)
        @test size(L.image) == (4, 6, 1) && L.sampled == green
        @test all(L.image[.!green, 1] .== 0)
        @test all(L.image[i, j, 1] == (raw[i, j] - m.black[m.colors[i, j] + 1]) / (m.white - m.black[m.colors[i, j] + 1]) * 2
                  for i in 1:4, j in 1:6 if green[i, j])
        @test count(L.saturated) == 1 && all(L.saturated .<= green)
        @test L.plane_offsets == [zeros(2)]
    end
end

@testitem "raw: a decoder needs its package" begin
    using VisualGeometryToolkitFeatures
    VGF = VisualGeometryToolkitFeatures
    @test_throws ArgumentError load_raw_mosaic(joinpath(@__DIR__, "no-such-file.ARW"))
    for (decoder, ext) in ((LibRawDecoder(), :VisualGeometryToolkitFeaturesLibRawExt), (RawpyDecoder(), :VisualGeometryToolkitFeaturesPythonExt))
        if Base.get_extension(VGF, ext) === nothing
            err = try
                load_raw(@__FILE__; mode = :bin2, decoder)
            catch e
                e
            end
            @test err isa ArgumentError && occursin("using $(VGF._decoder_trigger(decoder))", err.msg)
        else
            @test_skip "$ext already loaded on this worker"
        end
    end
end

@testitem "raw: LibRaw_jll decoder" begin
    using VisualGeometryToolkitFeatures
    using LibRaw_jll
    # the version check ran at load; a file LibRaw cannot read fails with its message
    err = try
        load_raw_mosaic(@__FILE__; decoder = LibRawDecoder())
    catch e
        e
    end
    @test err isa ErrorException && occursin("LibRaw cannot open", err.msg)
    path = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isfile(path)
        m = load_raw_mosaic(path; decoder = LibRawDecoder())
        @test size(m.raw) == size(m.colors) && all(<(m.white), m.black)
        @test startswith(m.metadata.libraw_version, "0.22")
        L = LinearRaw(m; mode = :bin2)
        @test size(L.image) == (size(m.raw) .÷ 2..., 3)
    else
        @test_skip "VGTK_RAW_TEST_FILE names no RAW file"
    end
end

@testitem "raw: LibRaw black levels — a spatial pattern is folded or refused, never dropped" begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using LibRaw_jll
    ext = Base.get_extension(VgtkFeatures, :VisualGeometryToolkitFeaturesLibRawExt)
    filters = 0x94949494                           # RGGB
    cblack(rows, cols, pattern; per_channel = (0, 0, 0, 0)) = begin
        cb = zeros(Int, ext.CBLACK_SIZE)
        cb[1:4] .= per_channel
        cb[5], cb[6] = rows, cols
        cb[7:(6 + rows * cols)] .= pattern
        cb
    end
    # no pattern: the per-channel levels plus black
    @test ext._adjust_bl(cblack(0, 0, Int[]; per_channel = (1, 2, 3, 4)), 100, filters, "a.ARW") == (101, 102, 103, 104)
    # a 2 × 2 pattern folds into the four channels exactly
    b = ext._adjust_bl(cblack(2, 2, [10, 20, 30, 40]), 100, filters, "a.ARW")
    @test sort(collect(b)) == [110, 120, 130, 140]
    # a uniform 4 × 4 pattern is a constant: it moves into black
    @test ext._adjust_bl(cblack(4, 4, fill(7, 16)), 100, filters, "a.ARW") == (107, 107, 107, 107)
    # a 4 × 4 pattern that varies has no per-channel form: refused, naming the file and the size
    err = try ext._adjust_bl(cblack(4, 4, collect(1:16)), 100, filters, "/data/b.dng"); nothing catch e; e end
    @test err isa ErrorException && occursin("/data/b.dng", err.msg) && occursin("4 × 4", err.msg)
end

@testitem "raw: rawpy decoder (PythonCall)" begin
    using VisualGeometryToolkitFeatures
    path = get(ENV, "VGTK_RAW_TEST_FILE", "")
    if isfile(path)
        # Loads Python: run only where a RAW file was supplied. Not in the same
        # process as a LibRaw_jll decode when comparing the two: rawpy's calls can
        # bind to an already-loaded LibRaw_jll library.
        @eval using PythonCall
        m = Base.invokelatest(load_raw_mosaic, path; decoder = RawpyDecoder())
        @test size(m.raw) == size(m.colors) && all(<(m.white), m.black)
        @test startswith(m.metadata.decoder, "rawpy")
    else
        @test_skip "VGTK_RAW_TEST_FILE names no RAW file"
    end
end
