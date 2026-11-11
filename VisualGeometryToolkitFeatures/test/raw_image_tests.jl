# load_raw: the processing on synthetic mosaics (black/white, gain, the 2×2
# binning, partial-saturation propagation, the true sample positions for the
# four Bayer layouts, the green-cell image, its exact frame map and footprint),
# the display demosaic on a synthetic Bayer buffer opened by LibRaw itself, and
# the decoder where it can run. CI has no RAW file: a real-file check runs when
# VGTK_RAW_TEST_FILE names one, and is recorded as a skip otherwise.

@testsetup module RawFixtures
    using VisualGeometryToolkitFeatures: RawMosaic
    export mosaic, scene_mosaic, LAYOUTS, ORIGINS
    # LibRaw colour indices: 0 R, 1 G, 2 B, 3 the second G
    const LAYOUTS = (
        RGGB = UInt8[0 1; 3 2], BGGR = UInt8[2 1; 3 0],
        GRBG = UInt8[1 0; 2 3], GBRG = UInt8[1 2; 0 3],
    )
    function mosaic(raw::AbstractMatrix{<:Integer}, cell::Matrix{UInt8}; black = (512.0, 500.0, 520.0, 508.0), white = 16383.0, origin = (0, 0))
        H, W = size(raw)
        meta = (; model = "synthetic")
        return RawMosaic(Matrix{UInt16}(raw), cell, black, white, origin, (H + origin[1], W + origin[2]), meta)
    end
    # the visible-area origins of every parity
    const ORIGINS = ((0, 0), (1, 0), (0, 1), (1, 1))
    # A mosaic whose every photosite records the scene `f(x, y)` at its own centre, in
    # the visible raw's COLMAP frame (photosite (row, column) at (column − ½, row − ½)).
    scene_mosaic(f, cell, origin; H = 12, W = 16, kw...) =
        mosaic(UInt16[f(j - 0.5, i - 0.5) for i in 1:H, j in 1:W], cell; origin, kw...)
end

@testitem "raw: photosite colours are the pattern tiled from the raw origin" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: RawMosaic, photosite_colors, LinearRaw
    raw = fill(UInt16(3000), 4, 6)
    @test photosite_colors(mosaic(raw, LAYOUTS.RGGB)) == UInt8[LAYOUTS.RGGB[mod1(i, 2), mod1(j, 2)] for i in 1:4, j in 1:6]
    # an odd visible origin shifts the tiling: RGGB from the raw origin reads B G2 / G R from (1, 1)
    m = mosaic(raw, LAYOUTS.RGGB; origin = (1, 1))
    @test photosite_colors(m)[1:2, 1:2] == UInt8[2 3; 1 0]
    @test photosite_colors(mosaic(raw, LAYOUTS.RGGB; origin = (0, 1)))[1:2, 1:2] == UInt8[1 0; 2 3]
    @test size(LinearRaw(m; mode = :bin2).image) == (2, 3, 3)
    # the pattern is validated, not the mosaic: square, at most 4 × 4, indices ≤ 3
    meta = (; model = "synthetic")
    @test_throws ArgumentError RawMosaic(raw, UInt8[0 1 2], (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0), (4, 6), meta)
    @test_throws ArgumentError RawMosaic(raw, zeros(UInt8, 6, 6), (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0), (4, 6), meta)
    @test_throws ArgumentError RawMosaic(raw, UInt8[0 1; 4 2], (0.0, 0.0, 0.0, 0.0), 1.0, (0, 0), (4, 6), meta)
    # a 4 × 4 pattern that is not 2-periodic has no 2×2 quad
    p4 = UInt8[0 1 0 1; 3 2 3 2; 0 1 2 1; 3 2 3 0]
    @test_throws ArgumentError LinearRaw(RawMosaic(raw, p4, (0.0, 0.0, 0.0, 0.0), 16383.0, (0, 0), (4, 6), meta); mode = :bin2)
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
    using VisualGeometryToolkitFeatures: LinearRaw, bayer_plane_map, raw_frame_map
    using VisualGeometryToolkitCore: translation
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
        # a plane's offset is its map read in the cell grid: inv(raw_frame_map) ∘ bayer_plane_map, a translation
        offset(c) = translation(inv(raw_frame_map(L)) ∘ bayer_plane_map(L, c))
        @test [Tuple(offset(c)) for c in 1:3] == expected[name]
        # the R plane is the photosite the offset points at: (Δy, Δx) = ((di − ½)/2, (dj − ½)/2)
        o = offset(1)
        di, dj = Int(2o[2] + 0.5) + 1, Int(2o[1] + 0.5) + 1
        @test cell[di, dj] == 0
        @test L.image[1, 1, 1] == (raw[di, dj] - 512.0) / (16383.0 - 512.0)
    end
    # a pattern that is not a 2×2 Bayer quad is refused
    bad = mosaic(raw, UInt8[0 0; 2 2])
    @test_throws ArgumentError LinearRaw(bad; mode = :bin2)
end

@testitem "raw: each plane's map sends its samples onto their photosites, exactly" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, BayerCellFrame, bayer_plane_map, raw_frame_map, photosite_colors
    using VisualGeometryToolkitCore: SimilarityMap, PixelFrame
    using StaticArrays: SVector
    for cell in LAYOUTS, origin in ORIGINS
        m = mosaic(UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8], cell; origin)
        colors = photosite_colors(m)
        L = LinearRaw(m; mode = :bin2)
        G = LinearRaw(m; mode = :green_cells)
        # one frame per plane; the green plane's map IS the image's frame map
        for (c, C) in enumerate((:R, :G, :B))
            @test bayer_plane_map(L, c) isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, BayerCellFrame{C}, PixelFrame{:colmap}}
        end
        @test bayer_plane_map(L, 2) === raw_frame_map(L)
        @test bayer_plane_map(G, 1) === raw_frame_map(G)
        # pixel (i, j) of plane c sits at its plane frame's (j − ½, i − ½): the map sends it onto the centre of its
        # photosites in the cell, raw COLMAP (column − ½, row − ½) — R's one, B's one, the mean of the two greens
        sites(i, j, want) = [SVector(c - 0.5, r - 0.5) for r in (2i - 1):(2i), c in (2j - 1):(2j) if colors[r, c] in want]
        for i in 1:3, j in 1:4
            p = SVector(j - 0.5, i - 0.5)
            for (c, want) in enumerate(((0x00,), (0x01, 0x03), (0x02,)))
                s = sites(i, j, want)
                @test bayer_plane_map(L, c)(p) == sum(s) / length(s)
                @test inv(bayer_plane_map(L, c))(sum(s) / length(s)) == p   # and back
            end
            @test bayer_plane_map(G, 1)(p) == bayer_plane_map(L, 2)(p)
        end
        # the inverses round-trip, exactly (the offsets are quarters, the scale 2)
        x = SVector(3.375, 1.625)
        @test all(c -> inv(bayer_plane_map(L, c))(bayer_plane_map(L, c)(x)) == x, 1:3)
        @test all(c -> inv(bayer_plane_map(L, c)) ∘ bayer_plane_map(L, c) isa SimilarityMap, 1:3)
        # composing across planes is refused at the junction, as everywhere in Core's frames
        @test_throws MethodError bayer_plane_map(L, 1) ∘ inv(bayer_plane_map(L, 3))
        @test_throws MethodError bayer_plane_map(L, 3) ∘ inv(bayer_plane_map(L, 2))
        # one plane read through another's frame is a map between the two planes, stated as such
        @test inv(bayer_plane_map(L, 1)) ∘ bayer_plane_map(L, 3) isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, BayerCellFrame{:B}, BayerCellFrame{:R}}
        # no plane past the image's
        @test_throws ArgumentError bayer_plane_map(L, 4)
        @test_throws ArgumentError bayer_plane_map(G, 2)
    end
end

@testitem "raw: a bin2 image warped through its plane maps registers its planes (host and device)" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, bayer_plane_map, raw_frame_map
    using VisualGeometryToolkitCore: image_warp, image_warp!, Rect
    # A scene affine in the raw's COLMAP frame, recorded by every photosite at its own centre. Each plane
    # is read back onto the raw's grid through Core's warp of `inv(bayer_plane_map(L, c))`, one warp per
    # plane in one call: bilinear sampling reproduces an affine scene exactly, so wherever a plane's taps
    # are inside the image all three planes read the scene at the same raw point. Read through the cell
    # grid's map alone, R and B are a quarter cell off.
    H, W = 12, 16
    f(x, y) = 4000 + 64 * x + 32 * y
    lin(v) = (v - 500.0) / (16383.0 - 500.0)
    truth = [lin(f(j - 0.5, i - 0.5)) for i in 1:H, j in 1:W]
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    for cell in LAYOUTS, origin in ORIGINS
        L = LinearRaw(scene_mosaic(f, cell, origin; H, W, black = (500.0, 500.0, 500.0, 500.0)); mode = :bin2)
        warps = ntuple(c -> image_warp(Float64, inv(bayer_plane_map(L, c)), Rect(0, 0, W, H)), 3)
        out = image_warp!(fill(NaN, H, W, 3), warps, L.image; fillvalue = NaN)
        seen = trues(H, W)
        for c in 1:3
            seen .&= .!isnan.(out[:, :, c])
        end
        @test count(seen) >= (H - 4) * (W - 4)              # all but a border of the taps' reach
        @test all(c -> maximum(abs.(out[:, :, c][seen] .- truth[seen])) < 1e-14, 1:3)
        cells = ntuple(_ -> image_warp(Float64, inv(raw_frame_map(L)), Rect(0, 0, W, H)), 3)
        off = image_warp!(fill(NaN, H, W, 3), cells, L.image; fillvalue = NaN)
        @test maximum(abs.(off[:, :, 1][seen] .- truth[seen])) > 1e-4
        if cuda !== nothing
            cu = getproperty(cuda, :CuArray)
            d = Base.invokelatest(image_warp!, Base.invokelatest(cu, fill(NaN, H, W, 3)), warps, Base.invokelatest(cu, L.image); fillvalue = NaN)
            @test isequal(Base.invokelatest(Array, d), out) || maximum(abs.(Base.invokelatest(Array, d)[seen, :] .- out[seen, :])) < 1e-12
        end
    end
    cuda === nothing && @test_skip cuda !== nothing      # no functional GPU here: the device half recorded as Broken
end

@testitem "raw: the green-cell image is bin2's green plane" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, raw_frame_map
    for cell in LAYOUTS, origin in ORIGINS
        raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
        m = mosaic(raw, cell; origin)
        B = LinearRaw(m; mode = :bin2, gain = 2)
        G = LinearRaw(m; mode = :green_cells, gain = 2)
        @test size(G.image) == (3, 4, 1) && G.mode === :green_cells && G.gain == 2.0
        @test G.image[:, :, 1] == B.image[:, :, 2]                    # the same bits: one kernel
        @test raw_frame_map(G) === raw_frame_map(B) && G.green_offsets == B.green_offsets
    end
    # a green cell is saturated by its greens alone: a red or blue at the white level is not its sample
    raw = fill(UInt16(3000), 4, 4)
    raw[1, 1] = 16383                      # quad (1, 1)'s red (RGGB)
    raw[3, 4] = 16383                      # quad (2, 2)'s green
    m = mosaic(raw, LAYOUTS.RGGB)
    @test LinearRaw(m; mode = :green_cells).saturated == BitMatrix([0 0; 0 1])
    @test LinearRaw(m; mode = :bin2).saturated == BitMatrix([1 0; 0 1])
    # the raw-grid green photosites (zeros at red and blue) are retired; the demosaic needs the file
    @test_throws ArgumentError LinearRaw(m; mode = Symbol("gre" * "en"))
    err = try LinearRaw(m; mode = :demosaic) catch e; e end
    @test err isa ArgumentError && occursin("load_raw", err.msg)
end

@testitem "raw: the cell frame map is exact for every Bayer phase and origin" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, BayerCellFrame, raw_frame_map
    using VisualGeometryToolkitCore: SimilarityMap, PixelFrame, scale, translation
    using StaticArrays: SVector
    S = raw_frame_map(LinearRaw(mosaic(fill(UInt16(3000), 4, 4), LAYOUTS.RGGB); mode = :green_cells))
    @test S isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, BayerCellFrame{:G}, PixelFrame{:colmap}}
    @test scale(S) == 2 && translation(S) == SVector(0.0, 0.0)
    @test inv(S)(S(SVector(3.25, 1.75))) == SVector(3.25, 1.75)
    # a scene point at a known sub-pixel raw position, (x0, y0): an affine ramp through it in x
    # (and one in y), integer-valued at every photosite centre. The green-cell values are the
    # scene at S(p) for every cell centre p, and the ramp's zero, found in the green-cell image,
    # maps back to x0 (y0) exactly — for every Bayer phase and visible-origin parity.
    x0, y0 = 9.25, 6.75
    black = (512.0, 500.0, 520.0, 500.0)              # the two greens share a black level: the mean stays affine
    lin(v) = (v - 500.0) / (16383.0 - 500.0)
    for cell in LAYOUTS, origin in ORIGINS
        for (axis, f) in ((1, (x, y) -> 4000 + 64 * (x - x0)), (2, (x, y) -> 4000 + 64 * (y - y0)))
            L = LinearRaw(scene_mosaic(f, cell, origin; black); mode = :green_cells)
            Sm = raw_frame_map(L)
            g = L.image[:, :, 1]
            @test all(g[i, j] ≈ lin(f(Sm(SVector(j - 0.5, i - 0.5))...)) for i in axes(g, 1), j in axes(g, 2))
            # the level of the scene at the point, between the two cells that straddle it
            target = lin(4000.0)
            v = axis == 1 ? g[1, :] : g[:, 1]
            k = findfirst(t -> v[t] <= target <= v[t + 1], 1:(length(v) - 1))
            p = k - 0.5 + (target - v[k]) / (v[k + 1] - v[k])   # half-res COLMAP coordinate of the zero
            q = Sm(axis == 1 ? SVector(p, 0.5) : SVector(0.5, p))
            @test q[axis] ≈ (axis == 1 ? x0 : y0) atol = 1e-12
        end
    end
end

@testitem "raw: K moves between the green-cell and raw frames through Core" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, raw_frame_map
    using VisualGeometryToolkitCore: CameraCalibrationMat
    using VisualGeometryToolkitCore.Cameras: reframed_calibration
    using StaticArrays: SMatrix
    S = raw_frame_map(LinearRaw(mosaic(fill(UInt16(3000), 4, 4), LAYOUTS.GBRG; origin = (1, 0)); mode = :green_cells))
    Kh = CameraCalibrationMat(SMatrix{3, 3, Float64}(1234.567, 0, 0, 0, 1230.125, 0, 1520.3125, 1011.71875, 1))
    Kr = reframed_calibration(S, Kh)
    @test (Kr[1, 1], Kr[2, 2], Kr[1, 3], Kr[2, 3]) == (2 * Kh[1, 1], 2 * Kh[2, 2], 2 * Kh[1, 3], 2 * Kh[2, 3])
    @test reframed_calibration(inv(S), Kr) == Kh          # exact both ways
end

@testitem "raw: the green-cell footprint is the cell's two green photosites" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, green_cell_sites, photosite_colors
    using StaticArrays: SVector
    # (Δx, Δy) of the two greens from the cell centre, in raw pixels, for the phase the visible area
    # starts on: the greens are diagonal, so their mean is the cell centre in every phase
    h = 0.5
    anti = (SVector(h, -h), SVector(-h, h))               # R G / G B and B G / G R: the greens on the anti-diagonal
    diag = (SVector(-h, -h), SVector(h, h))               # G R / B G and G B / R G: on the diagonal
    expected = Dict(:RGGB => anti, :BGGR => anti, :GRBG => diag, :GBRG => diag)
    phase = Dict(UInt8[0 1; 3 2] => :RGGB, UInt8[2 1; 3 0] => :BGGR, UInt8[1 0; 2 3] => :GRBG, UInt8[1 2; 0 3] => :GBRG,
                 UInt8[2 3; 1 0] => :BGGR, UInt8[0 3; 1 2] => :RGGB, UInt8[3 2; 0 1] => :GBRG, UInt8[3 0; 2 1] => :GRBG)
    for cell in LAYOUTS, origin in ORIGINS
        raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
        m = mosaic(raw, cell; origin)
        colors = photosite_colors(m)
        L = LinearRaw(m; mode = :green_cells)
        @test Set(L.green_offsets) == Set(expected[phase[colors[1:2, 1:2]]])
        for i in 1:3, j in 1:4
            sites = green_cell_sites(L, i, j)
            @test sum(sites) / 2 == SVector(2j - 1.0, 2i - 1.0)     # the cell centre, raw COLMAP
            # each is a green photosite's centre, (column − ½, row − ½), and the value is their mean
            rc = [(Int(s[2] + 0.5), Int(s[1] + 0.5)) for s in sites]
            @test all(colors[r, c] in (0x01, 0x03) for (r, c) in rc)
            @test L.image[i, j, 1] == sum((raw[r, c] - m.black[colors[r, c] + 1]) / (m.white - m.black[colors[r, c] + 1]) for (r, c) in rc) / 2
        end
    end
end

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
    using VisualGeometryToolkitFeatures
    using VisualGeometryToolkitFeatures: LinearRaw, raw_frame_map, bayer_plane_map, green_cell_sites, raw_provenance
    using LibRaw_jll: LibRaw_jll, libraw
    VGF = VisualGeometryToolkitFeatures
    X = Base.get_extension(VGF, :VisualGeometryToolkitFeaturesLibRawExt)
    # A synthetic Bayer buffer opened by LibRaw itself (libraw_open_bayer), RGGB, its top-left
    # quadrant dark and the rest bright, with the EXIF orientation set the way a camera records it
    # held upside down (flip 3: 180°) or upright (5: 90° CCW, 6: 90° CW). The display image keeps
    # the sensor's grid and its content where the sensor saw it: a 180° turn keeps H × W, so the
    # content is what tells it apart.
    H, W = 48, 64
    data = UInt16[(i <= H ÷ 2 && j <= W ÷ 2) ? 2000 : 30000 for i in 1:H, j in 1:W]
    buf = collect(reinterpret(UInt8, vec(permutedims(data))))
    function demosaic_with_flip(flip)
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
    # the demosaic copies the visible photosites only once, into its image: never the mosaic too (its checks read
    # LibRaw's layout, `_mosaic_layout`); measured warm, on a second handle
    let (m, rgb, _, _) = demosaic_with_flip(3), (_, _, _, a) = demosaic_with_flip(3)
        @test a < sizeof(rgb) + sizeof(m.raw)
    end
    for flip in (3, 5, 6)
        m, rgb, md, _ = demosaic_with_flip(flip)
        L = VGF._demosaic_image(rgb, md, 1.0, nothing)
        @test L isa LinearRaw && L.mode === :demosaic
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

@testitem "raw: provenance records the decoder, the mode, LibRaw and the frame map" setup = [RawFixtures] begin
    using VisualGeometryToolkitFeatures: LinearRaw, RawMosaic, raw_provenance
    m = RawMosaic(fill(UInt16(3000), 4, 4), LAYOUTS.RGGB, (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (4, 4),
        (; model = "synthetic", decoder = "LibRaw_jll", libraw_version = "0.22.0-Release"))
    p = raw_provenance(LinearRaw(m; mode = :green_cells, gain = 3))
    @test p == (decoder = "LibRaw_jll", libraw_version = "0.22.0-Release", mode = "green_cells", gain = 3.0,
                frame_map = (from = "BayerCellFrame{:G}", to = "PixelFrame{:colmap}", scale = 2.0, translation = [0.0, 0.0]))
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

@testitem "raw: a decode cache — each RAW decoded once per content and settings" setup = [RawFixtures] begin
    # The decoded-RAW cache PanoramaStitching's example kept as code-sum PNGs is
    # the package's one persistent cache (FileCache, JLD2): `load_raw(path; cache)`
    # keys an entry by the file's SHA-256 and the decode settings.
    using VisualGeometryToolkitFeatures
    using VisualGeometryToolkitFeatures: RawDecoder, open_cache, close_cache
    using JLD2: JLD2
    struct CountingDecoder <: RawDecoder
        calls::Base.RefValue{Int}
    end
    function VisualGeometryToolkitFeatures._decode_raw(d::CountingDecoder, path::AbstractString)
        d.calls[] += 1
        v = only(reinterpret(UInt16, read(path, 2)))           # the file's content sets the scene
        return mosaic(fill(UInt16(1000 + v), 8, 8), LAYOUTS.RGGB)
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
end
