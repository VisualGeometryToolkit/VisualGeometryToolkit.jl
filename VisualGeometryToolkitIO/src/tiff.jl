# =============================================================================
# The stack's ONE TIFF codec: libtiff (Libtiff_jll), one sample per pixel, in strips
# =============================================================================
#
# libtiff is the reference TIFF library — the one the print RIPs read with,
# and the one Poppler (BlobBoards' rasterizer) links — so the stack writes and
# reads TIFF through it, here, and nowhere else: `write_tiff` and `read_tiff`
# serve BlobBoards' board TIFFs (8-bit grey, LZW with the horizontal predictor,
# its density and ICC profile in the tags) and vgio/raster/v1's payload
# (Float32, raster_schema.jl) alike. TiffImages.jl is not used: it writes only
# uncompressed strips.
#
# A grey image is one sample per pixel, row by row (a scanline per image row,
# grouped into strips of `rows_per_strip` rows); the file is little-endian
# (libtiff's `"wl"` mode), whatever the host.

# libtiff's tag numbers and values (tiff.h), the ones this codec sets or reads
const _TIFFTAG_IMAGEWIDTH = UInt32(256)
const _TIFFTAG_IMAGELENGTH = UInt32(257)
const _TIFFTAG_BITSPERSAMPLE = UInt32(258)
const _TIFFTAG_COMPRESSION = UInt32(259)
const _TIFFTAG_PHOTOMETRIC = UInt32(262)
const _TIFFTAG_ORIENTATION = UInt32(274)
const _TIFFTAG_SAMPLESPERPIXEL = UInt32(277)
const _TIFFTAG_ROWSPERSTRIP = UInt32(278)
const _TIFFTAG_XRESOLUTION = UInt32(282)
const _TIFFTAG_YRESOLUTION = UInt32(283)
const _TIFFTAG_PLANARCONFIG = UInt32(284)
const _TIFFTAG_RESOLUTIONUNIT = UInt32(296)
const _TIFFTAG_SOFTWARE = UInt32(305)
const _TIFFTAG_PREDICTOR = UInt32(317)
const _TIFFTAG_SAMPLEFORMAT = UInt32(339)
const _TIFFTAG_ICCPROFILE = UInt32(34675)
const _PHOTOMETRIC_MINISWHITE = 0
const _PHOTOMETRIC_MINISBLACK = 1
const _PLANARCONFIG_CONTIG = 1
const _RESUNIT_INCH = 2
const _SAMPLEFORMAT_UINT = 1
const _SAMPLEFORMAT_IEEEFP = 3
# the compressions and predictors a writer may ask for, by name (the codes are tiff.h's)
const _TIFF_COMPRESSION = (none = 1, lzw = 5, deflate = 8)       # deflate: COMPRESSION_ADOBE_DEFLATE (zlib)
const _TIFF_PREDICTOR = (none = 1, horizontal = 2)               # never 3, the floating-point predictor
# a sample type's bits and SampleFormat
_tiff_sample(::Type{UInt8}) = (bits = 8, format = _SAMPLEFORMAT_UINT)
_tiff_sample(::Type{Float32}) = (bits = 32, format = _SAMPLEFORMAT_IEEEFP)

# TIFFSetField is variadic: an integer tag takes an int (C's default promotion of
# uint16/uint32), a resolution a double, a string a char*, the ICC profile (count, bytes).
_tiff_set(tif, tag, v::Integer) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cint)::Cint
_tiff_set(tif, tag, v::Float64) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cdouble)::Cint
_tiff_set(tif, tag, v::String) = @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; v::Cstring)::Cint
_tiff_set(tif, tag, v::Vector{UInt8}) =
    @ccall libtiff.TIFFSetField(tif::Ptr{Cvoid}, tag::UInt32; length(v)::UInt32, v::Ptr{UInt8})::Cint
# A tag's value (`T` its C type: UInt16 or UInt32), or `nothing` when the file has none;
# `defaulted`: libtiff's default for a tag the file leaves out (TIFFGetFieldDefaulted).
function _tiff_get(tif, tag, ::Type{T}; defaulted::Bool = false) where {T}
    v = Ref{T}(0)
    ok = defaulted ? @ccall(libtiff.TIFFGetFieldDefaulted(tif::Ptr{Cvoid}, tag::UInt32; v::Ptr{T})::Cint) :
        @ccall(libtiff.TIFFGetField(tif::Ptr{Cvoid}, tag::UInt32; v::Ptr{T})::Cint)
    return ok == 1 ? v[] : nothing
end

# `f(tif)` on the TIFF at `path` opened in `mode` ("r", or "wl": write little-endian), closed after.
function _with_tiff(f, path::AbstractString, mode::String)
    tif = @ccall libtiff.TIFFOpen(path::Cstring, mode::Cstring)::Ptr{Cvoid}
    tif == C_NULL && error("libtiff could not open $(path) $(mode == "r" ? "for reading" : "for writing")")
    try
        return f(tif)
    finally
        @ccall libtiff.TIFFClose(tif::Ptr{Cvoid})::Cvoid
    end
end

"""
    write_tiff(path, image::AbstractMatrix{T}; compression = :none, predictor = :none, rows_per_strip = 64,
               density = nothing, software = nothing, icc = nothing) -> path

Write a grey image — `T` is `UInt8` (8 bits, unsigned) or `Float32` (32 bits,
IEEE floating point, every value bit for bit, NaN included) — as a
little-endian TIFF of one image, one sample per pixel (min-is-black), in strips
of `rows_per_strip` rows, through libtiff. `image` is (rows, columns), the
first row on top. `compression` is `:none`, `:lzw` or `:deflate` (zlib), all
lossless; `predictor` `:none` or `:horizontal` (row differences, which make a
smooth image compress; a compression's, so not with `:none`). Optional tags: `density`, a pixel density (`300dpi`),
in the resolution tags (pixels per inch); `software`, a string; `icc`, an ICC
profile's bytes, embedded. Nothing else is written.

The stack's one TIFF writer: BlobBoards' board TIFFs (8-bit, LZW with the
horizontal predictor, its density and profile) and the payload of a
`vgio/raster/v1` file ([`save_raster`](@ref): Float32, uncompressed or
Deflate, no tag that means anything) are written by it. A libtiff failure — a
path it cannot open, a full disk — is an error naming the file;
[`read_tiff`](@ref) reads the image back.
"""
function write_tiff(
        path::AbstractString, image::AbstractMatrix{T}; compression::Symbol = :none, predictor::Symbol = :none,
        rows_per_strip::Integer = 64, density::Union{Nothing, Quantity} = nothing, software::Union{Nothing, AbstractString} = nothing,
        icc::Union{Nothing, AbstractVector{UInt8}} = nothing,
    ) where {T <: Union{UInt8, Float32}}
    haskey(_TIFF_COMPRESSION, compression) ||
        throw(ArgumentError("write_tiff: compression is one of $(join(keys(_TIFF_COMPRESSION), ", ")), got :$(compression)"))
    haskey(_TIFF_PREDICTOR, predictor) ||
        throw(ArgumentError("write_tiff: predictor is one of $(join(keys(_TIFF_PREDICTOR), ", ")), got :$(predictor)"))
    (predictor === :none || compression !== :none) ||
        throw(ArgumentError("write_tiff: a predictor is a compression's (LZW or Deflate); an uncompressed TIFF has none"))
    rows_per_strip > 0 || throw(ArgumentError("write_tiff: rows_per_strip is positive, got $(rows_per_strip)"))
    height, width = size(image)
    (height > 0 && width > 0) || throw(ArgumentError("write_tiff: the image is empty ($(height) × $(width))"))
    sample = _tiff_sample(T)
    tags = Pair{UInt32, Any}[
        _TIFFTAG_IMAGEWIDTH => width, _TIFFTAG_IMAGELENGTH => height, _TIFFTAG_BITSPERSAMPLE => sample.bits,
        (T === UInt8 ? () : (_TIFFTAG_SAMPLEFORMAT => sample.format,))...,
        _TIFFTAG_SAMPLESPERPIXEL => 1, _TIFFTAG_PHOTOMETRIC => _PHOTOMETRIC_MINISBLACK, _TIFFTAG_PLANARCONFIG => _PLANARCONFIG_CONTIG,
        _TIFFTAG_COMPRESSION => _TIFF_COMPRESSION[compression],
        (predictor === :none ? () : (_TIFFTAG_PREDICTOR => _TIFF_PREDICTOR[predictor],))...,
        _TIFFTAG_ROWSPERSTRIP => Int(rows_per_strip),
        (density === nothing ? () : (_TIFFTAG_XRESOLUTION => Float64(ustrip(dpi, density)), _TIFFTAG_YRESOLUTION => Float64(ustrip(dpi, density)),
            _TIFFTAG_RESOLUTIONUNIT => _RESUNIT_INCH))...,
        (software === nothing ? () : (_TIFFTAG_SOFTWARE => String(software),))...,
        (icc === nothing ? () : (_TIFFTAG_ICCPROFILE => Vector{UInt8}(icc),))...,
    ]
    _with_tiff(path, "wl") do tif
        for (tag, v) in tags
            _tiff_set(tif, tag, v) == 1 || error("write_tiff: libtiff rejected tag $(tag) = $(repr(v)) for $(path)")
        end
        # libtiff takes scanlines (image rows); a Julia matrix is column-major, so each
        # row is gathered into one contiguous buffer (no transposed copy of the image).
        row = Vector{T}(undef, width)
        for r in 1:height
            copyto!(row, @view image[r, :])
            @ccall(libtiff.TIFFWriteScanline(tif::Ptr{Cvoid}, row::Ptr{T}, (r - 1)::UInt32, 0::UInt16)::Cint) == 1 ||
                error("write_tiff: libtiff failed writing row $(r) of $(path)")
        end
        # TIFFClose returns nothing, yet it writes the last strip and the directory:
        # flush first, so a full disk is an error and not a TIFF without a directory.
        @ccall(libtiff.TIFFFlush(tif::Ptr{Cvoid})::Cint) == 1 || error("write_tiff: libtiff failed to flush $(path) (disk full?)")
    end
    return path
end

"""
    read_tiff(T, path) -> Matrix{T}

The grey image of the TIFF at `path` (its first image), as (rows, columns), the
first row on top, read through libtiff: `T` is `UInt8` for an 8-bit unsigned
image and `Float32` for a 32-bit IEEE floating-point one (every value bit for
bit, NaN included). The file holds one sample per pixel, min-is-black or — an
8-bit image only, as `pdftocairo -gray` may write it — min-is-white, which is
returned inverted (`0xff - v`), so a value is always the grey level with black
0. Any compression and predictor libtiff decodes is read. A file of another
sample type, more than one sample per pixel, or a min-is-white Float32 image is
an `ArgumentError` naming it; a libtiff failure an error. The reader of
[`write_tiff`](@ref)'s files and of [`load_raster`](@ref)'s payload.
"""
read_tiff(::Type{T}, path::AbstractString) where {T <: Union{UInt8, Float32}} = _with_tiff(tif -> _read_tiff(T, tif, path), path, "r")

function _read_tiff(::Type{T}, tif, path) where {T <: Union{UInt8, Float32}}
    width = Int(something(_tiff_get(tif, _TIFFTAG_IMAGEWIDTH, UInt32), 0))
    height = Int(something(_tiff_get(tif, _TIFFTAG_IMAGELENGTH, UInt32), 0))
    (width > 0 && height > 0) || throw(ArgumentError("$(path): a TIFF states its image width and length"))
    bits = _tiff_get(tif, _TIFFTAG_BITSPERSAMPLE, UInt16; defaulted = true)
    samples = _tiff_get(tif, _TIFFTAG_SAMPLESPERPIXEL, UInt16; defaulted = true)
    format = _tiff_get(tif, _TIFFTAG_SAMPLEFORMAT, UInt16; defaulted = true)
    photometric = _tiff_get(tif, _TIFFTAG_PHOTOMETRIC, UInt16)
    want = _tiff_sample(T)
    (bits == want.bits && format == want.format && samples == 1) || throw(
        ArgumentError(
            "$(path): expected one $(want.bits)-bit $(T === UInt8 ? "unsigned" : "IEEE floating-point") sample per pixel " *
                "(read_tiff($(T), …)), got $(samples) sample(s) of $(bits) bits, SampleFormat $(format)"
        )
    )
    invert = photometric == _PHOTOMETRIC_MINISWHITE
    (photometric == _PHOTOMETRIC_MINISBLACK || (invert && T === UInt8)) || throw(
        ArgumentError("$(path): a grey image is min-is-black$(T === UInt8 ? " or min-is-white" : ""); its Photometric is $(repr(photometric))")
    )
    row = Vector{T}(undef, width)
    sizeof(row) == @ccall(libtiff.TIFFScanlineSize(tif::Ptr{Cvoid})::Int) ||
        throw(ArgumentError("$(path): a row is not $(width) contiguous samples (planar or padded rows)"))
    img = Matrix{T}(undef, height, width)
    for r in 1:height
        @ccall(libtiff.TIFFReadScanline(tif::Ptr{Cvoid}, row::Ptr{T}, (r - 1)::UInt32, 0::UInt16)::Cint) == 1 ||
            error("$(path): libtiff could not read row $(r)")
        @views img[r, :] .= row
    end
    T === UInt8 && invert && (img .= .~img)                # min-is-white (8-bit only): 0xff - v
    return img
end
