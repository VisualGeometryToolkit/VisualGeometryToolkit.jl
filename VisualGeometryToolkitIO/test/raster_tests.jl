# =============================================================================
# vgio/raster/v2.0 (and v1 through its step) and the TIFF codec (tiff.jl): a Float32 raster bit for bit, its
# header's SHA-256 binding, the TIFF profile, and Python's tifffile for neutrality
# =============================================================================

@testsetup module RasterFixtures
    using Random: Xoshiro
    export bits, awkward_values, python_tifffile

    # every Float32 value as its bits: NaN compares by its payload, -0.0 apart from 0.0
    bits(A) = reinterpret(UInt32, A)
    # the values a raster must keep exactly: random normals, ±0, ±Inf, subnormals, extremes, and NaNs of
    # several payloads and both signs (an "unobserved" NaN is any NaN; the TIFF keeps whichever one is written)
    function awkward_values(h, w)
        A = randn(Xoshiro(7), Float32, h, w)
        special = Float32[0.0, -0.0, Inf, -Inf, floatmin(Float32), nextfloat(0.0f0), prevfloat(floatmax(Float32)),
                          reinterpret(Float32, 0x7fc00000), reinterpret(Float32, 0x7f800001), reinterpret(Float32, 0xffc12345),
                          reinterpret(Float32, 0x7fbfffff)]
        A[1:length(special)] .= special
        A[end, end] = NaN32
        return A
    end
    # the pixi `py` environment's python when it has tifffile, else `nothing`
    function python_tifffile()
        python = joinpath(dirname(@__DIR__), ".pixi", "envs", "py", "bin", "python")
        ok = isfile(python) && success(pipeline(`$python -I -c "import tifffile, numpy"`; stdout = devnull, stderr = devnull))
        return ok ? python : nothing
    end
end

@testitem "write_tiff / read_tiff: UInt8 and Float32 bit for bit, every compression and predictor" setup = [VGIoFixtures, RasterFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: dpi
    mktempdir() do dir
        p = joinpath(dir, "a.tiff")
        g = UInt8[(7 * r + 3 * c) % 256 for r in 1:37, c in 1:53]
        for compression in (:none, :lzw, :deflate), predictor in (:none, :horizontal)
            compression === :none && predictor === :horizontal && continue          # refused below: a predictor is a compression's
            VgtkIO.write_tiff(p, g; compression, predictor, rows_per_strip = 8, density = 300dpi, software = "test", icc = UInt8[1, 2, 3])
            @test VgtkIO.read_tiff(UInt8, p) == g
            @test read(p)[1:4] == UInt8['I', 'I', 0x2a, 0x00]                  # little-endian, classic
            # the tags asked for, as libtiff reads them back (BlobBoards' board TIFF: LZW, horizontal, its density)
            VgtkIO._with_tiff(p, "r") do tif
                @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_COMPRESSION, UInt16) == VgtkIO._TIFF_COMPRESSION[compression]
                # (an uncompressed file has no predictor tag: no codec registers it)
                @test something(VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_PREDICTOR, UInt16; defaulted = true), 1) == VgtkIO._TIFF_PREDICTOR[predictor]
                @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_ROWSPERSTRIP, UInt32) == 8
                @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_XRESOLUTION, Float32) == 300 == VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_YRESOLUTION, Float32)
                @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_RESOLUTIONUNIT, UInt16) == 2
                n, data = Ref{UInt32}(0), Ref{Ptr{UInt8}}(C_NULL)               # the ICC profile: (count, bytes)
                @test @ccall(VgtkIO.libtiff.TIFFGetField(tif::Ptr{Cvoid}, VgtkIO._TIFFTAG_ICCPROFILE::UInt32; n::Ptr{UInt32}, data::Ptr{Ptr{UInt8}})::Cint) == 1
                @test unsafe_wrap(Vector{UInt8}, data[], n[]) == UInt8[1, 2, 3]
                s = Ref{Ptr{UInt8}}(C_NULL)
                @test @ccall(VgtkIO.libtiff.TIFFGetField(tif::Ptr{Cvoid}, VgtkIO._TIFFTAG_SOFTWARE::UInt32; s::Ptr{Ptr{UInt8}})::Cint) == 1
                @test unsafe_string(s[]) == "test"
            end
        end
        # without the optional tags, none is written
        VgtkIO.write_tiff(p, g)
        VgtkIO._with_tiff(p, "r") do tif
            @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_XRESOLUTION, Float32) === nothing
            @test VgtkIO._tiff_get(tif, VgtkIO._TIFFTAG_PREDICTOR, UInt16) === nothing
        end
        A = awkward_values(29, 41)
        for compression in (:none, :lzw, :deflate)
            VgtkIO.write_tiff(p, A; compression)
            @test bits(VgtkIO.read_tiff(Float32, p)) == bits(A)
        end
        # the sample type is checked, not converted
        @test_throws "IEEE floating-point" VgtkIO.read_tiff(Float32, (VgtkIO.write_tiff(p, g); p))
        @test_throws "unsigned" VgtkIO.read_tiff(UInt8, (VgtkIO.write_tiff(p, A); p))
        @test_throws "compression is one of" VgtkIO.write_tiff(p, g; compression = :jpeg)
        @test_throws "predictor is one of" VgtkIO.write_tiff(p, A; predictor = :floating_point)
        @test_throws "a predictor is a compression's" VgtkIO.write_tiff(p, g; predictor = :horizontal)
        @test_throws "empty" VgtkIO.write_tiff(p, zeros(UInt8, 0, 4))
        @test_throws ErrorException VgtkIO.write_tiff(joinpath(dir, "no", "such", "dir.tiff"), g)
    end
end

@testitem "vgio/raster: a Float32 raster round-trips bit for bit (NaN included), in a chart or a pixel frame" setup = [VGIoFixtures, RasterFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rect2, sha256_hex
    mktempdir() do dir
        A = awkward_values(48, 64)
        page = (chart = :page, area = Rect2{Float64}(VgtkIO.metres(-105.0, mm), VgtkIO.metres(-148.5, mm), VgtkIO.metres(210.0, mm), VgtkIO.metres(297.0, mm)))
        for (frame, compression, unit) in ((page, :deflate, mm), (page, :none, m), ((chart = :texture, area = page.area), :deflate, mm),
                                           (PixelFrame{:colmap}, :deflate, mm), (PixelFrame{:opencv}, :none, mm))
            p = joinpath(dir, "albedo.json")
            @test VgtkIO.save_raster(p, A; frame, nan = :unobserved, length_unit = unit, compression, provenance = (; note = "t")) == p
            @test sort(readdir(dir)) == ["albedo.json", "albedo.tiff"]           # the header and its payload, nothing staged left
            d = parse_json(read(p, String))
            @test d.schema == "vgio/raster/v2.0" && d.size.width == 64 && d.size.height == 48 && d.element_type == "float32"
            @test d.nan == "unobserved" && d.payload.file == "albedo.tiff" && d.payload.sha256 == sha256_hex(read(joinpath(dir, "albedo.tiff")))
            frame isa NamedTuple && @test d.units.length == unit_string(unit) && d.frame.type == String(frame.chart)
            frame isa NamedTuple || @test d.frame.type == "pixel" && d.units.pixel == VgtkIO.pixel_frame_name(frame)
            frame isa NamedTuple && unit === mm && @test d.frame.area == [-105.0, -148.5, 210.0, 297.0]   # decimal-exact
            @test VgtkIO.validate(p) == "raster"
            r = VgtkIO.load_raster(p)
            @test bits(r.values) == bits(A) && r.nan === :unobserved
            @test r.frame == frame
            # the payload is the TIFF codec's: read alone, the same values
            @test bits(VgtkIO.read_tiff(Float32, joinpath(dir, "albedo.tiff"))) == bits(A)
        end
        # a raster without NaN says so, and one that says so holds none
        B = randn(Float32, 5, 6)
        q = joinpath(dir, "b.json")
        VgtkIO.save_raster(q, B; frame = PixelFrame{:colmap}, nan = :none)
        @test VgtkIO.load_raster(q).nan === :none && VgtkIO.load_raster(q).values == B
        @test_throws "holds no NaN" VgtkIO.save_raster(q, A; frame = PixelFrame{:colmap}, nan = :none)
        @test_throws "nan is one of" VgtkIO.save_raster(q, B; frame = PixelFrame{:colmap}, nan = :missing)
        @test_throws "uncompressed (:none) or Deflate" VgtkIO.save_raster(q, B; frame = PixelFrame{:colmap}, nan = :none, compression = :lzw)
        @test_throws "chart rectangle" VgtkIO.save_raster(q, B; frame = :page, nan = :none)
        @test_throws "chart frame's chart" VgtkIO.save_raster(q, B; frame = (chart = :board, area = page.area), nan = :none)
        VgtkIO.save_raster(q, B; frame = (area = page.area, chart = :page), nan = :none)          # either field order
        @test VgtkIO.load_raster(q).frame == page
        @test_throws "in plain metres" VgtkIO.save_raster(q, B; frame = (chart = :page, area = (0.0, 0.0, 0.21, 0.297)), nan = :none)
        @test_throws UndefKeywordError VgtkIO.save_raster(q, B; nan = :none)            # no default frame
    end
end

@testitem "vgio/raster: the header's SHA-256 binds the payload; the TIFF profile and the header are checked on read" setup = [VGIoFixtures, RasterFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: sha256_hex
    mktempdir() do dir
        A = awkward_values(16, 12)
        p = joinpath(dir, "r.json"); t = joinpath(dir, "r.tiff")
        fresh() = VgtkIO.save_raster(p, A; frame = PixelFrame{:colmap}, nan = :unobserved)
        # a corrupted payload — one bit flipped in the values, or in a tag — is refused by its digest, before it is decoded
        for at in (length(read((fresh(); t))) ÷ 2, 9)
            fresh(); b = read(t); b[at] ⊻= 0x01; write(t, b)
            @test_throws "corrupted or replaced" VgtkIO.load_raster(p)
        end
        # a payload missing, or outside the header's directory
        fresh(); rm(t)
        @test_throws "r.tiff, which is not a file" VgtkIO.load_raster(p)
        # header edits (the payload intact): each refused by name
        function header_refused(f!, why)
            fresh(); d = JSON.parse(read(p, String)); f!(d); write(p, JSON.json(d))
            e = try VgtkIO.load_raster(p); nothing catch err; err end
            return e isa ArgumentError && occursin(why, e.msg)
        end
        @test header_refused(d -> (d["size"]["width"] = 13), "its header states 13 × 16")
        @test header_refused(d -> (d["size"] = [12, 16]), "expected an object")
        @test header_refused(d -> (d["element_type"] = "float64"), "not an element type")
        @test header_refused(d -> (d["element_type"] = "uint8"; d["nan"] = "none"), "BitsPerSample 8")   # the payload is a float32 TIFF
        @test header_refused(d -> (d["element_type"] = "uint8"; d["nan"] = "unobserved"), "a uint8 raster holds no NaN")
        @test header_refused(d -> (d["nan"] = "masked"), "the meaning of a NaN value")
        @test header_refused(d -> (d["payload"]["file"] = "../r.tiff"), "inside the header's directory")
        @test header_refused(d -> (d["payload"]["sha256"] = "abc"), "SHA-256")
        @test header_refused(d -> delete!(d, "frame"), "frame: missing")
        @test header_refused(d -> (d["frame"] = Dict("type" => "page")), "must be \"pixel\"")   # a chart frame without its area
        @test header_refused(d -> (d["frame"] = Dict("type" => "page", "area" => [0, 0, 0, 1])), "positive widths")
        @test header_refused(d -> (d["units"]["pixel"] = "matlab"), "registered pixel frame")
        @test header_refused(d -> (d["frame"] = Dict("type" => "board", "area" => [0, 0, 1, 1])), "a chart")
        @test header_refused(d -> (d["frame"]["bits"] = 32), "unknown key")
        # a TIFF outside the profile, bound by its own digest (so only the profile can refuse it)
        function profile_refused(write!, why)
            fresh(); write!(t)
            d = JSON.parse(read(p, String)); d["payload"]["sha256"] = sha256_hex(read(t)); write(p, JSON.json(d))
            e = try VgtkIO.load_raster(p); nothing catch err; err end
            return e isa ArgumentError && occursin(why, e.msg)
        end
        @test profile_refused(f -> VgtkIO.write_tiff(f, A; compression = :lzw), "Compression 1 (none) or 8 (Deflate)")
        @test profile_refused(f -> VgtkIO.write_tiff(f, A; compression = :deflate, predictor = :horizontal), "Predictor 1")
        @test profile_refused(f -> VgtkIO.write_tiff(f, round.(UInt8, clamp.(abs.(replace(A, NaN32 => 0.0f0)), 0, 255))), "BitsPerSample 32")
        @test profile_refused(f -> (b = read(f); b[1:4] = UInt8['M', 'M', 0x00, 0x2a]; write(f, b)), "little-endian")
        # a NaN in a raster that says it holds none
        fresh(); d = JSON.parse(read(p, String)); d["nan"] = "none"; write(p, JSON.json(d))
        @test_throws "holds 5 NaN" VgtkIO.load_raster(p)
        # the reserved payload key stays refused in every other kind
        @test_throws "which a trajectory file does not hold" VgtkIO._refuse_payload("x.json", "trajectory", VgtkIO._TRAJECTORY_SPEC, Dict(:payload => Dict()))
    end
end

@testitem "vgio/raster: Python's tifffile reads the payload bit for bit, and its own TIFF reads as a raster" setup = [VGIoFixtures, RasterFixtures] begin
    # Neutrality: the payload is a plain TIFF any reader decodes. Runs when the pixi `py` environment is installed
    # (`pixi install -e py`, which brings tifffile); IO's CI does not install it, so there the check is recorded as
    # SKIPPED, and the release gate runs it (the gate installs the environment).
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: sha256_hex
    python = python_tifffile()
    if python === nothing
        @test_skip "tifffile is not installed in the pixi py environment"
    else
        mktempdir() do dir
            A = awkward_values(33, 47)
            for compression in (:deflate, :none)
                p = joinpath(dir, "n.json")
                VgtkIO.save_raster(p, A; frame = PixelFrame{:colmap}, nan = :unobserved, compression)
                # tifffile's array, row-major little-endian float32 bytes, as hex of its SHA-256, and its shape
                out = read(`$python -I -c "import sys, hashlib, tifffile; a = tifffile.imread(sys.argv[1]); assert a.dtype.str == '<f4', a.dtype; print(a.shape[0], a.shape[1], hashlib.sha256(a.tobytes()).hexdigest())" $(joinpath(dir, "n.tiff"))`, String)
                h, w, digest = split(strip(out))
                @test (parse(Int, h), parse(Int, w)) == size(A)
                @test digest == sha256_hex(reinterpret(UInt8, vec(permutedims(A))))      # every value's bits, NaN payloads too
            end
            # tifffile writes a raster-profile TIFF (its own tags carry nothing the header needs): it reads as a raster
            raw = joinpath(dir, "raw.f32"); write(raw, permutedims(A))
            run(`$python -I -c "import sys, numpy, tifffile; a = numpy.fromfile(sys.argv[1], dtype='<f4').reshape(int(sys.argv[2]), int(sys.argv[3])); tifffile.imwrite(sys.argv[4], a, photometric='minisblack', compression='zlib')" $raw 33 47 $(joinpath(dir, "py.tiff"))`)
            hdr = Dict("schema" => "vgio/raster/v1", "size" => Dict("width" => 47, "height" => 33), "element_type" => "float32",
                       "frame" => Dict("pixel_frame" => "colmap"), "nan" => "unobserved",
                       "payload" => Dict("file" => "py.tiff", "sha256" => sha256_hex(read(joinpath(dir, "py.tiff")))))
            write(joinpath(dir, "py.json"), JSON.json(hdr))
            @test bits(VgtkIO.load_raster(joinpath(dir, "py.json")).values) == bits(A)
        end
    end
end

@testitem "vgio/raster: a uint8 raster (a mask) round-trips; a v1 header (IO 0.9.5) reads through its step" setup = [VGIoFixtures, RasterFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rect2, sha256_hex
    mktempdir() do dir
        M = UInt8[(r * c) % 3 == 0 for r in 1:9, c in 1:11]
        p = joinpath(dir, "mask.json")
        VgtkIO.save_raster(p, M; frame = PixelFrame{:colmap}, nan = :none)
        d = parse_json(read(p, String))
        @test d.element_type == "uint8" && d.nan == "none" && VgtkIO.validate(p) == "raster"
        r = VgtkIO.load_raster(p)
        @test r.values == M && eltype(r.values) === UInt8 && r.frame === PixelFrame{:colmap}
        @test_throws "a uint8 raster holds no NaN" VgtkIO.save_raster(p, M; frame = PixelFrame{:colmap}, nan = :unobserved)
        # IO 0.9.5's v1 header beside the same payload: its frame's own length_unit / pixel_frame become the units object
        A = awkward_values(6, 5)
        page = (chart = :page, area = Rect2{Float64}(VgtkIO.metres(-105.0, mm), VgtkIO.metres(-148.5, mm), VgtkIO.metres(210.0, mm), VgtkIO.metres(297.0, mm)))
        for (frame, v1frame) in ((page, Dict("chart" => "page", "length_unit" => "mm", "area" => [-105.0, -148.5, 210.0, 297.0])),
                                 (PixelFrame{:opencv}, Dict("pixel_frame" => "opencv")))
            q = joinpath(dir, "a.json")
            VgtkIO.save_raster(q, A; frame, nan = :unobserved)
            v1 = Dict("schema" => "vgio/raster/v1", "size" => Dict("width" => 5, "height" => 6), "element_type" => "float32",
                      "frame" => v1frame, "nan" => "unobserved",
                      "payload" => Dict("file" => "a.tiff", "sha256" => sha256_hex(read(joinpath(dir, "a.tiff")))))
            old = joinpath(dir, "old.json"); write(old, JSON.json(v1))
            b = VgtkIO.load_raster(old)
            @test bits(b.values) == bits(A) && b.frame == frame && b.nan === :unobserved
        end
    end
end
