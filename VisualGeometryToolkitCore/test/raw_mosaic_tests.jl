# The sensor image (moved from VisualGeometryToolkitFeatures' raw_image_tests.jl in 0.9.2, the
# assertions unchanged, only the module they import from): the processing on synthetic mosaics
# (black/white, gain, the 2×2 binning, partial-saturation propagation, the true sample positions
# for the four Bayer layouts, the green-cell image, its exact frame map and footprint), the
# photosite matrix, its windows and its device form.

@testsetup module RawFixtures
    using VisualGeometryToolkitCore: RawMosaic
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
    using VisualGeometryToolkitCore: RawMosaic, photosite_colors, LinearRaw
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

@testitem "raw: a RawMosaic is its photosite matrix" setup = [RawFixtures] begin
    # The wrapper IS an AbstractMatrix: an ordinary array function reads the photosites unchanged, and
    # what it makes is a plain array (the CFA and the metadata stay with the mosaic).
    using VisualGeometryToolkitCore: RawMosaic
    raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
    m = mosaic(raw, LAYOUTS.GRBG; origin = (1, 0))
    @test m isa AbstractMatrix{UInt16} && size(m) == (6, 8) && axes(m) == axes(raw)
    @test parent(m) === m.raw && m == raw && m[2, 3] == raw[2, 3] && m[7] == raw[7]
    @test IndexStyle(m) === IndexLinear()
    @test similar(m) isa Matrix{UInt16} && size(similar(m, Float64, 2, 2)) == (2, 2)
    @test copy(m) == raw && copy(m) isa Matrix{UInt16}
    @test m .+ 1 == raw .+ 1 && (m .+ 1) isa Matrix
    @test sum(m) == sum(raw) && extrema(m) == extrema(raw) && Float64.(m) == Float64.(raw)
    @test m[2, 3:5] == raw[2, 3:5] && m[2, 3:5] isa Vector{UInt16}       # a row is a plain vector
    @test occursin("RawMosaic", sprint(show, MIME"text/plain"(), m)) && occursin("GRBG", sprint(show, m))
    # `==` and `isequal` are the photosite matrices' (an AbstractMatrix's equality): a mosaic of another colour
    # phase, window or metadata with the same photosites is equal; the mosaic's own fields are compared by name
    other = mosaic(raw, LAYOUTS.RGGB; origin = (0, 1))
    @test m == other && isequal(m, other) && hash(m) == hash(other) && m.pattern != other.pattern
    @test m == raw && raw == m && isequal(m, raw) && m != raw .+ 1 && m ≈ raw
end

@testitem "raw: copy and equality of a mosaic read its held array whole" setup = [RawFixtures] begin
    # A held array that refuses element reads (as a device array refuses scalar indexing): copy, ==, isequal and ≈
    # go to the held array's own methods, never element by element through the wrapper (against a bare dense array
    # too: the device item compares a CuArray mosaic with its parent).
    using VisualGeometryToolkitCore: RawMosaic
    struct Whole <: AbstractMatrix{UInt16}
        a::Matrix{UInt16}
    end
    Base.size(x::Whole) = size(x.a)
    Base.getindex(::Whole, ::Int...) = error("an element read")
    Base.copy(x::Whole) = Whole(copy(x.a))
    Base.:(==)(x::Whole, y::Whole) = x.a == y.a
    Base.isequal(x::Whole, y::Whole) = isequal(x.a, y.a)
    Base.isapprox(x::Whole, y::Whole; kw...) = isapprox(x.a, y.a; kw...)
    raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
    wrap(a) = RawMosaic(Whole(a), LAYOUTS.RGGB, (512.0, 500.0, 520.0, 508.0), 16383.0, (0, 0), (6, 8), (; model = "synthetic"))
    m = wrap(raw)
    c = copy(m)
    @test c isa Whole && c.a == raw && c.a !== raw                         # the held array's kind, a copy
    @test m == wrap(copy(raw)) && isequal(m, wrap(copy(raw))) && m != wrap(raw .+ UInt16(1))
    @test isapprox(m, wrap(copy(raw))) && !isapprox(m, wrap(raw .+ UInt16(100)))
end

@testitem "raw: a window of a mosaic keeps its colours and its frame" setup = [RawFixtures] begin
    # A unit-range index (`m[rows, cols]`, a copy) or `view` (no copy) is a mosaic of its own: its
    # colours are the parent's at those photosites, whatever the parity of the window's start (the
    # CFA phase shifts with it), and its frame map is the window's offset into the visible raw, so a
    # cell image made from it maps onto the parent's photosites exactly.
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw, photosite_colors, raw_frame_map, bayer_plane_map, raw_coordinates
    using StaticArrays: SVector
    f(x, y) = 4000 + 64 * x + 32 * y
    lin(v) = (v - 500.0) / (16383.0 - 500.0)
    for cell in LAYOUTS, origin in ORIGINS, (rows, cols) in ((3:10, 2:13), (2:9, 5:16), (1:12, 1:16))
        m = scene_mosaic(f, cell, origin; black = (500.0, 500.0, 500.0, 500.0))
        for w in (m[rows, cols], view(m, rows, cols))
            @test w isa RawMosaic && w == parent(m)[rows, cols]
            @test photosite_colors(w) == photosite_colors(m)[rows, cols]
            @test w.metadata === m.metadata && w.visible_origin == m.visible_origin && w.white == m.white
            @test raw_frame_map(w)(SVector(0.5, 0.5)) == SVector(first(cols) - 0.5, first(rows) - 0.5)
            L = LinearRaw(w; mode = :bin2)
            for i in axes(L.image, 1), j in axes(L.image, 2), c in 1:3
                # the scene is affine: a plane's value is the scene at its photosites' centre
                @test L.image[i, j, c] ≈ lin(f(bayer_plane_map(L, c)(SVector(j - 0.5, i - 0.5))...)) atol = 1e-12
            end
            G = LinearRaw(w; mode = :green_cells)
            @test G.image[2, 2, 1] ≈ lin(f(raw_coordinates(G, SVector(1.5, 1.5))...)) atol = 1e-12
        end
        @test parent(view(m, rows, cols)) isa SubArray                    # a view copies no photosite
        @test m[rows, cols][2:3, 2:5] == view(view(m, rows, cols), 2:3, 2:5)  # windows compose
        @test raw_frame_map(view(view(m, rows, cols), 2:3, 2:5))(SVector(0.5, 0.5)) ==
              SVector(first(cols) + 0.5, first(rows) + 0.5)
    end
    @test_throws BoundsError view(mosaic(fill(UInt16(3000), 4, 4), LAYOUTS.RGGB), 1:5, 1:2)
end

@testitem "raw: raw_coordinates is an image's one sensor-pixel accessor" setup = [RawFixtures] begin
    # Where a position in an image lies on the sensor: the identity on the decoded mosaic, x ↦ 2x on
    # the green-cell grid (a detection on the green cells written as raw pixels unmapped is half the
    # scale). Backed by raw_frame_map; the display demosaic has no frame and is refused, and a plain
    # matrix states no frame at all.
    using VisualGeometryToolkitCore: LinearRaw, raw_coordinates, raw_frame_map
    using StaticArrays: SVector
    m = mosaic(fill(UInt16(3000), 8, 8), LAYOUTS.RGGB; origin = (1, 1))
    x = SVector(3.25, 1.75)
    @test raw_coordinates(m, x) === x
    @test raw_coordinates(m, (3.25, 1.75)) === x && raw_coordinates(m, [3.25, 1.75]) === x
    G = LinearRaw(m; mode = :green_cells)
    @test raw_coordinates(G, x) == 2x == raw_frame_map(G)(x)
    @test raw_coordinates(LinearRaw(m; mode = :bin2), x) == 2x
    @test_throws ArgumentError raw_coordinates(LinearRaw{:demosaic}(zeros(4, 4, 3), falses(4, 4), nothing, 1.0, (;)), x)
    @test_throws MethodError raw_coordinates(zeros(4, 4), x)
end

@testitem "raw: a mosaic on the device — Adapt moves it, the cell kernel reads its parent" setup = [RawFixtures] begin
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw, photosite_colors, raw_coordinates
    using Adapt: adapt
    using CUDA: CUDA
    import KernelAbstractions as KA
    using StaticArrays: SVector
    if CUDA.functional()
        raw = UInt16[1000 + 37 * i + 11 * j for i in 1:12, j in 1:16]
        raw[5, 6] = 16383
        for cell in LAYOUTS, origin in ORIGINS
            m = mosaic(raw, cell; origin)
            d = adapt(CUDA.CuArray, m)
            @test d isa RawMosaic{<:CUDA.CuArray} && d.metadata === m.metadata && photosite_colors(d) == photosite_colors(m)
            @test adapt(Array, d) isa RawMosaic{Matrix{UInt16}} && adapt(Array, d) == m
            @test adapt(CUDA.CUDABackend(), m) isa RawMosaic{<:CUDA.CuArray} && adapt(KA.CPU(), d) isa RawMosaic{Matrix{UInt16}}   # by KA backend
            @test (d .+ 1) isa CUDA.CuArray && Array(d .+ 1) == raw .+ 1 && sum(d) == sum(raw)
            # no scalar read (a test process refuses one): the held array's own copy, == and ≈
            @test copy(d) isa CUDA.CuArray && Array(copy(d)) == raw
            @test d == adapt(CUDA.CuArray, m) && d == parent(d) && isequal(d, parent(d)) && d ≈ parent(d)
            @test contains(sprint(show, MIME"text/plain"(), d), "RawMosaic")   # no scalar read
            for mode in (:green_cells, :bin2), w in (d, view(d, 2:11, 3:16))
                Ld = LinearRaw(w; mode)
                Lh = LinearRaw(adapt(Array, w); mode)
                @test Ld.image isa CUDA.CuArray{Float64, 3} && Array(Ld.image) == Lh.image
                @test Array(Ld.saturated) == Lh.saturated && Ld.geometry == Lh.geometry
                @test raw_coordinates(Ld, SVector(1.5, 2.5)) == raw_coordinates(Lh, SVector(1.5, 2.5))
            end
        end
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end

@testitem "raw: linear values, gain and the 2×2 binning" setup = [RawFixtures] begin
    using VisualGeometryToolkitCore: LinearRaw, cfa_string
    black = (512.0, 500.0, 520.0, 508.0)
    white = 16383.0
    raw = UInt16[1000 2000 1500 900; 3000 4000 700 16000; 5 6 7 8; 9 10 11 12]   # row 3–4: below black
    m = mosaic(raw, LAYOUTS.RGGB; black, white)
    @test cfa_string(m) == "RGGB"
    lin(v, c) = (v - black[c + 1]) / (white - black[c + 1])
    L = LinearRaw(m; mode = :bin2)
    @test size(L.image) == (2, 2, 3) && L isa LinearRaw{:bin2} && L.gain == 1.0
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
    using VisualGeometryToolkitCore: LinearRaw
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
    using VisualGeometryToolkitCore: LinearRaw, bayer_plane_map, raw_frame_map
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
    using VisualGeometryToolkitCore: LinearRaw, BayerCellFrame, bayer_plane_map, raw_frame_map, photosite_colors
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
    using VisualGeometryToolkitCore: LinearRaw, bayer_plane_map, raw_frame_map
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
    using VisualGeometryToolkitCore: LinearRaw, raw_frame_map
    for cell in LAYOUTS, origin in ORIGINS
        raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
        m = mosaic(raw, cell; origin)
        B = LinearRaw(m; mode = :bin2, gain = 2)
        G = LinearRaw(m; mode = :green_cells, gain = 2)
        @test size(G.image) == (3, 4, 1) && G isa LinearRaw{:green_cells} && G.gain == 2.0
        @test G.image[:, :, 1] == B.image[:, :, 2]                    # the same bits: one kernel
        @test raw_frame_map(G) === raw_frame_map(B) && G.geometry.green_offsets == B.geometry.green_offsets
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
    using VisualGeometryToolkitCore: LinearRaw, BayerCellFrame, raw_frame_map
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
    using VisualGeometryToolkitCore: LinearRaw, raw_frame_map
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
    using VisualGeometryToolkitCore: LinearRaw, green_cell_sites, photosite_colors
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
        @test Set(L.geometry.green_offsets) == Set(expected[phase[colors[1:2, 1:2]]])
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

@testitem "raw: provenance records the decoder, the mode, LibRaw and the frame map" setup = [RawFixtures] begin
    using VisualGeometryToolkitCore: LinearRaw, RawMosaic, raw_provenance
    m = RawMosaic(fill(UInt16(3000), 4, 4), LAYOUTS.RGGB, (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (4, 4),
        (; model = "synthetic", decoder = "LibRaw_jll", libraw_version = "0.22.0-Release"))
    p = raw_provenance(LinearRaw(m; mode = :green_cells, gain = 3))
    @test p == (decoder = "LibRaw_jll", libraw_version = "0.22.0-Release", mode = "green_cells", gain = 3.0,
                frame_map = (from = "BayerCellFrame{:G}", to = "PixelFrame{:colmap}", scale = 2.0, translation = [0.0, 0.0]))
end


# -----------------------------------------------------------------------------
# channelview: the photosites channel-first, a view, and the one phase rule (0.9.2)
# -----------------------------------------------------------------------------

@testitem "raw: a mosaic's channelview is its photosites channel-first, without a copy" setup = [RawFixtures] begin
    using VisualGeometryToolkitCore: RawMosaic, channelview, photositeview, bayer_channels, photosite_colors
    raw = UInt16[1000 + 37 * i + 11 * j for i in 1:6, j in 1:8]
    for cell in LAYOUTS, origin in ORIGINS
        m = mosaic(raw, cell; origin)
        v = channelview(m)
        @test size(v) == (4, 3, 4) && eltype(v) == UInt16
        @test v isa Base.ReshapedArray && parent(v) isa PermutedDimsArray      # a view of the held array, no copy
        @test all(v[di + 2dj + 1, i, j] == raw[2i - 1 + di, 2j - 1 + dj] for di in 0:1, dj in 0:1, i in 1:3, j in 1:4)
        # each channel's colour, from the one phase rule: the photosite's own colour index
        names = bayer_channels(m)
        col = photosite_colors(m)
        want = Dict(:R => (0x00,), :G1 => (0x01, 0x03), :G2 => (0x01, 0x03), :B => (0x02,))
        @test all(col[1 + di, 1 + dj] in want[names[di + 2dj + 1]] for di in 0:1, dj in 0:1)
        @test sort(collect(names)) == [:B, :G1, :G2, :R]
        # the inverse view (colorview's role): the held array again, the same memory
        @test photositeview(v) == raw && pointer(photositeview(v)) == pointer(parent(m))
    end
    # it is ImageCore's function: a plain matrix keeps ImageCore's method (a gray array's view is itself)
    import ImageCore
    @test channelview === ImageCore.channelview && channelview(raw) == raw
    # the order is the cell's column-major one: RGGB reads R, G1, G2, B; GRBG G1, B, R, G2
    @test bayer_channels(mosaic(raw, LAYOUTS.RGGB)) == (:R, :G1, :G2, :B)
    @test bayer_channels(mosaic(raw, LAYOUTS.GRBG)) == (:G1, :B, :R, :G2)
    # writing through the view writes the mosaic's held array
    a = copy(raw); m = mosaic(a, LAYOUTS.RGGB)
    channelview(m)[4, 2, 3] = 0x0007
    @test parent(m)[4, 6] == 0x0007
    # an odd last row and column are left out, as by LinearRaw; a window is viewed in place
    m = mosaic(UInt16[i + 100j for i in 1:7, j in 1:9], LAYOUTS.BGGR)
    @test size(channelview(m)) == (4, 3, 4) && channelview(m)[2, 3, 4] == parent(m)[6, 7]
    w = view(m, 2:7, 2:9)
    @test channelview(w)[1, 1, 1] == parent(m)[2, 2] && channelview(w)[4, 3, 4] == parent(m)[7, 9]
    # photositeview of any channel-first array: the mosaic layout
    c = reshape(collect(1.0:48.0), 4, 3, 4)
    p = photositeview(c)
    @test size(p) == (6, 8) && all(p[2i - 1 + di, 2j - 1 + dj] == c[di + 2dj + 1, i, j] for di in 0:1, dj in 0:1, i in 1:3, j in 1:4)
    @test channelview(mosaic(UInt16.(p), LAYOUTS.RGGB)) == UInt16.(c)
    @test_throws ArgumentError photositeview(zeros(3, 2, 2))
    @test_throws ArgumentError channelview(mosaic(fill(UInt16(1), 1, 8), LAYOUTS.RGGB))
    @test_throws ArgumentError bayer_channels(mosaic(raw, UInt8[0 0; 2 2]))
end

@testitem "raw: each channel's map is the one phase rule — every photosite, phase and window parity" setup = [RawFixtures] begin
    # Channel k's map sends its pixel (i, j) onto its photosite's sensor position, as raw_coordinates gives
    # it, for all four Bayer phases × visible-origin parities × both parities of a window's start in
    # rows and columns; the R and B channels' maps ARE the bin2 planes' (one rule, not a copy).
    using VisualGeometryToolkitCore: LinearRaw, BayerCellFrame, channelview, bayer_channels, bayer_plane_map, raw_coordinates,
        photosite_colors, pixel_centre, SimilarityMap, PixelFrame
    using StaticArrays: SVector
    raw = UInt16[1000 + 37 * i + 11 * j for i in 1:12, j in 1:16]
    for cell in LAYOUTS, origin in ORIGINS, r0 in (1, 2), c0 in (1, 2)
        m = view(mosaic(raw, cell; origin), r0:(r0 + 9), c0:(c0 + 13))     # the window's start: either parity
        v = channelview(m)
        names = bayer_channels(m)
        col = photosite_colors(m)
        L = LinearRaw(m; mode = :bin2)
        for k in 1:4
            S = bayer_plane_map(m, k)
            @test S isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, BayerCellFrame{names[k]}, PixelFrame{:colmap}}
            @test isbits(S)
            di, dj = (k - 1) % 2, (k - 1) ÷ 2
            for i in axes(v, 2), j in axes(v, 3)
                r, c = 2i - 1 + di, 2j - 1 + dj
                @test S(pixel_centre(CartesianIndex(i, j))) === raw_coordinates(m, pixel_centre(CartesianIndex(r, c)))
                @test v[k, i, j] == m[r, c]
            end
            names[k] === :R && @test S === bayer_plane_map(L, 1)
            names[k] === :B && @test S === bayer_plane_map(L, 3)
        end
        @test_throws ArgumentError bayer_plane_map(m, 5)
    end
end

@testitem "raw: bayer_plane_map and green_cell_sites are bit-identical on the one rule" setup = [RawFixtures] begin
    # The rewrite onto the one sample-lattice map leaves every map and every site as the 0.9.1 code made
    # them: the plane maps as `raw_frame_map(m) ∘ (cell map ∘ translation)` and the green sites as the
    # cell centre plus the green photosite's offset, written out here as they were.
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: LinearRaw, BayerCellFrame, bayer_plane_map, green_cell_sites, raw_frame_map, relabel,
        SimilarityMap, Translation, scale, pixel_centre
    old_plane(m, C, o) = C === :G ? raw_frame_map(m) ∘ VgtkCore._CELL_FRAME_MAP :
        raw_frame_map(m) ∘ (VgtkCore._CELL_FRAME_MAP ∘ relabel(BayerCellFrame{C}, BayerCellFrame{:G}, SimilarityMap(Translation(o))))
    raw = UInt16[1000 + 37 * i + 11 * j for i in 1:10, j in 1:14]
    for cell in LAYOUTS, origin in ORIGINS, r0 in (1, 2), c0 in (1, 2)
        m = view(mosaic(raw, cell; origin), r0:(r0 + 7), c0:(c0 + 11))
        L = LinearRaw(m; mode = :bin2)
        G = LinearRaw(m; mode = :green_cells)
        cellc = VgtkCore._cell_colors(m, "test")
        site(k) = pixel_centre(CartesianIndex(VgtkCore._QUAD_SITES[k]))
        ks(c) = Tuple(k for k in 1:4 if (cellc[k] == 3 ? 2 : Int(cellc[k]) + 1) == c)
        for (c, C) in enumerate((:R, :G, :B))
            o = sum(site, ks(c)) / (length(ks(c)) * scale(VgtkCore._CELL_FRAME_MAP))
            @test bayer_plane_map(L, c) === old_plane(m, C, o)
        end
        @test bayer_plane_map(G, 1) === old_plane(m, :G, nothing)
        o = G.geometry.green_offsets
        for i in 1:4, j in 1:6
            c = raw_frame_map(G)(pixel_centre(CartesianIndex(i, j)))
            @test green_cell_sites(G, i, j) === (c + o[1], c + o[2])
            # and each is the green channel's map at that pixel, (G1, G2) in the cell's order
            gk = Tuple(k for k in 1:4 if cellc[k] in (0x01, 0x03))
            @test green_cell_sites(G, i, j) === Tuple(bayer_plane_map(m, k)(pixel_centre(CartesianIndex(i, j))) for k in gk)
        end
    end
end

@testitem "raw: the green cells are the mean of the linearised view's two green channels, bit for bit" setup = [RawFixtures] begin
    # ONE linearisation: `linear_value` over `channelview` with `bayer_levels`. The cell kernel's green-cell value is
    # the mean of the two G channels of that view times the gain, and bin2's R and B are its R and B channels.
    using VisualGeometryToolkitCore: LinearRaw, channelview, bayer_channels, bayer_levels, linear_value
    raw = UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:10, j in 1:14]
    raw[3, 4] = 16383
    for cell in LAYOUTS, origin in ORIGINS, gain in (1.0, 3.5)
        m = mosaic(raw, cell; origin)
        v = channelview(m)
        black, span = bayer_levels(m)
        lin = linear_value.(v, reshape(collect(black), 4, 1, 1), reshape(collect(span), 4, 1, 1))
        names = bayer_channels(m)
        g = findall(n -> n in (:G1, :G2), collect(names))
        @test (lin[g[1], :, :] .+ lin[g[2], :, :]) ./ 2 .* gain == LinearRaw(m; mode = :green_cells, gain).image[:, :, 1]
        B = LinearRaw(m; mode = :bin2, gain)
        @test lin[findfirst(==(:R), names), :, :] .* gain == B.image[:, :, 1]
        @test lin[findfirst(==(:B), names), :, :] .* gain == B.image[:, :, 3]
        @test B.geometry.green_dn == (span[g[1]], span[g[2]])
        @test all(linear_value(m.white, black[k], span[k]) == 1 for k in 1:4)
    end
end

@testitem "raw: a mosaic's channelview on the device — a device view, CPU = GPU" setup = [RawFixtures] begin
    # The view of a device mosaic is a view of its device array (no photosite copied; KernelAbstractions sees
    # the device's backend; Adapt moves it whole). A CUDA kernel does not index it through Base's wrappers
    # (benchmark/raw_channelview.jl): the cell kernel reads the held array by the view's index rule, and its
    # output is the host's, bit for bit, as is the view's content.
    using VisualGeometryToolkitCore: RawMosaic, LinearRaw, channelview, photositeview, bayer_plane_map, green_cell_sites
    using Adapt: adapt
    using CUDA: CUDA
    import KernelAbstractions as KA
    innermost(a) = parent(a) === a ? a : innermost(parent(a))
    if CUDA.functional()
        raw = UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:13, j in 1:17]     # odd: a last row and column left out
        raw[5, 6] = 16383
        for cell in LAYOUTS, origin in ORIGINS, (rows, cols) in ((1:13, 1:17), (2:13, 2:17), (2:11, 1:16), (1:12, 1:16))
            h = view(mosaic(raw, cell; origin), rows, cols)
            d = view(adapt(CUDA.CuArray, mosaic(raw, cell; origin)), rows, cols)
            vd = channelview(d)
            @test KA.get_backend(vd) isa CUDA.CUDABackend
            @test pointer(innermost(vd)) == pointer(innermost(parent(d)))      # no photosite copied
            @test adapt(Array, vd) == channelview(h)                            # the same view, moved whole
            @test adapt(Array, photositeview(vd)) == photositeview(channelview(h))
            # the cell kernel on the device: CPU = GPU, bit for bit (and, on the host, the mean of the
            # linearised view's green channels: the item above)
            for mode in (:green_cells, :bin2)
                Ld, Lh = LinearRaw(d; mode, gain = 2.0), LinearRaw(h; mode, gain = 2.0)
                @test Array(Ld.image) == Lh.image && Array(Ld.saturated) == Lh.saturated && Ld.geometry == Lh.geometry
            end
            # the maps are host values, the same for either
            @test all(k -> bayer_plane_map(d, k) === bayer_plane_map(h, k), 1:4)
            @test green_cell_sites(LinearRaw(d; mode = :green_cells), 2, 3) === green_cell_sites(LinearRaw(h; mode = :green_cells), 2, 3)
        end
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end

@testitem "raw: the lazy linear view and saturation flags on the raw grid; a green mask by broadcast" setup = [RawFixtures] begin
    # linearview(m): every photosite's (raw − black[c]) / (white − black[c]), read lazily; channelview of it is the
    # linear values channel-first, and the green-cell image is the mean of its two G channels, bit for bit.
    using VisualGeometryToolkitCore: LinearRaw, LinearRawView, SaturationView, linearview, saturationview, channelview,
        bayer_channels, photosite_colors, green_cell_sites, pixel_centre
    raw = UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:11, j in 1:15]      # odd: a last row and column
    raw[3, 4] = 16383; raw[6, 7] = 16390
    for cell in LAYOUTS, origin in ORIGINS, (rows, cols) in ((1:11, 1:15), (2:11, 2:15), (2:10, 1:14))
        m = view(mosaic(raw, cell; origin), rows, cols)
        lv = linearview(m)
        col = photosite_colors(m)
        @test lv isa LinearRawView && size(lv) == size(m) && parent(lv) === parent(m)
        @test all(lv[i, j] === (Float64(m[i, j]) - m.black[col[i, j] + 1]) / (m.white - m.black[col[i, j] + 1]) for i in axes(m, 1), j in axes(m, 2))
        @test copy(lv) isa Matrix{Float64} && copy(lv) == [lv[i, j] for i in axes(m, 1), j in axes(m, 2)]
        cv = channelview(lv)
        @test size(cv) == size(channelview(m)) && all(cv[k, i, j] === lv[2i - 1 + (k - 1) % 2, 2j - 1 + (k - 1) ÷ 2] for k in 1:4, i in axes(cv, 2), j in axes(cv, 3))
        g = findall(n -> n in (:G1, :G2), collect(bayer_channels(m)))
        @test (cv[g[1], :, :] .+ cv[g[2], :, :]) ./ 2 == LinearRaw(m; mode = :green_cells).image[:, :, 1]
        # the saturation flags: the matching lazy Bool matrix, the one rule; LinearRaw's cell flags are its any over the cell
        sv = saturationview(m)
        @test sv isa SaturationView && size(sv) == size(m) && copy(sv) == (parent(m) .>= m.white)
        L = LinearRaw(m; mode = :bin2)
        @test L.saturated == dropdims(any(channelview(sv); dims = 1); dims = 1)
        # the green mask is a plain broadcast over photosite_colors: its photosites are green_cell_sites' exactly
        green = (col .== 0x01) .| (col .== 0x03)
        G = LinearRaw(m; mode = :green_cells)
        h, w = size(G.image, 1), size(G.image, 2)
        wo = m.window_origin                                       # green_cell_sites are on the visible raw: less the window's offset
        sites = Set((Int(s[2] + 0.5) - wo[1], Int(s[1] + 0.5) - wo[2]) for i in 1:h, j in 1:w for s in green_cell_sites(G, i, j))
        @test sites == Set(Tuple(I) for I in findall(green[1:(2h), 1:(2w)]))
        cs = channelview(sv)
        @test G.saturated == cs[g[1], :, :] .| cs[g[2], :, :]
    end
end

@testitem "raw: the lazy linear view on the device — CPU = GPU" setup = [RawFixtures] begin
    using VisualGeometryToolkitCore: LinearRaw, linearview, saturationview, channelview, bayer_channels
    using Adapt: adapt
    using CUDA: CUDA
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index, @Const
    # a kernel reads the lazy view as it is: each cell's two green photosites, linear, averaged
    @kernel function green_mean_kernel!(out, @Const(lin), greens)
        i, j = @index(Global, NTuple)
        (d1, e1), (d2, e2) = greens
        @inbounds out[i, j] = (lin[2i - 1 + d1, 2j - 1 + e1] + lin[2i - 1 + d2, 2j - 1 + e2]) / 2
    end
    if CUDA.functional()
        raw = UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:13, j in 1:17]
        raw[5, 6] = 16383
        for cell in LAYOUTS, origin in ORIGINS, (rows, cols) in ((1:13, 1:17), (2:13, 2:17), (1:12, 1:16))
            h = view(mosaic(raw, cell; origin), rows, cols)
            d = view(adapt(CUDA.CuArray, mosaic(raw, cell; origin)), rows, cols)
            lh, ld = linearview(h), linearview(d)
            @test KA.get_backend(ld) isa CUDA.CUDABackend
            @test copy(ld) isa CUDA.CuArray{Float64, 2} && Array(copy(ld)) == copy(lh)            # a broadcast on the device
            g = findall(n -> n in (:G1, :G2), collect(bayer_channels(d)))
            site(k) = ((k - 1) % 2, (k - 1) ÷ 2)
            Lg = LinearRaw(h; mode = :green_cells).image[:, :, 1]
            out = CUDA.zeros(Float64, size(Lg))
            green_mean_kernel!(CUDA.CUDABackend())(out, ld, (site(g[1]), site(g[2])); ndrange = size(Lg))
            @test Array(out) == Lg
            cv = adapt(Array, channelview(ld))                                                    # the channel view, moved whole
            @test (cv[g[1], :, :] .+ cv[g[2], :, :]) ./ 2 == Array(LinearRaw(d; mode = :green_cells).image[:, :, 1])
            sd = saturationview(d)
            @test copy(sd) isa CUDA.CuArray{Bool, 2} && Array(copy(sd)) == copy(saturationview(h))   # a broadcast on the device
            @test Array(dropdims(any(channelview(copy(sd)); dims = 1); dims = 1)) == LinearRaw(h; mode = :bin2).saturated
        end
    else
        @test_skip CUDA.functional()
    end
end

@testitem "raw: a photosite mask is one fused broadcast on the device — CPU = GPU" setup = [RawFixtures] begin
    # photosite_colors is lazy and isbits, so a mask over a device mosaic is one broadcast with its arrays, on the
    # device (no host round-trip, no kernel of its own): RC's validity mask — the green photosites, less the ADC floor
    # (raw == 0) and the saturated (raw ≥ white), values under the black level kept — and the linear view, bit for bit
    # the host's, for the four Bayer phases × visible-origin parities × both parities of a window's start.
    using VisualGeometryToolkitCore: PhotositeColors, photosite_colors, saturationview, linearview
    using Adapt: adapt
    using CUDA: CUDA
    raw = UInt16[(i * 7919 + j * 104729) % 16000 + 200 for i in 1:13, j in 1:17]
    raw[5, 6] = 16383; raw[2, 3] = 0; raw[7, 8] = 0; raw[4, 4] = 100                     # saturated, floor, under black
    mask(m) = (c = photosite_colors(m); ((c .== 0x01) .| (c .== 0x03)) .& (parent(m) .!= 0) .& .!saturationview(m))
    @test isbits(photosite_colors(mosaic(raw, LAYOUTS.RGGB)))
    for cell in LAYOUTS, origin in ORIGINS, r0 in (1, 2), c0 in (1, 2)
        rows, cols = r0:13, c0:17
        h = view(mosaic(raw, cell; origin), rows, cols)
        col = photosite_colors(h)
        @test col isa PhotositeColors && collect(col) isa Matrix{UInt8} && collect(col) == [h.pattern[mod(h.visible_origin[1] + h.window_origin[1] + i - 1, 2) + 1, mod(h.visible_origin[2] + h.window_origin[2] + j - 1, 2) + 1] for i in axes(h, 1), j in axes(h, 2)]
        mh = mask(h)
        @test mh == [(col[i, j] in (0x01, 0x03)) && h[i, j] != 0 && h[i, j] < h.white for i in axes(h, 1), j in axes(h, 2)]
        if CUDA.functional()
            d = view(adapt(CUDA.CuArray, mosaic(raw, cell; origin)), rows, cols)
            md = mask(d)
            @test md isa CUDA.CuArray{Bool} && Array(md) == mh
            @test Array(copy(linearview(d))) == copy(linearview(h))
            @test photosite_colors(d) === photosite_colors(h)
        end
    end
    CUDA.functional() || @test_skip CUDA.functional()
end
