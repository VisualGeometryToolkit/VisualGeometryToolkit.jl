# =============================================================================
# vgio/raster/v1 — a Float32 raster: a JSON header and its TIFF payload
# =============================================================================
#
# {"schema": "vgio/raster/v1",
#  "size": {"width": W, "height": H},                      the raster's columns and rows
#  "element_type": "float32",
#  "frame": {"chart": "page" | "texture", "length_unit": "mm", "area": [x, y, w, h]}   a chart's rectangle, or
#         | {"pixel_frame": "colmap"},                     an image's pixels
#  "nan": "unobserved" | "none",                           what a NaN value means ("none": the raster holds none)
#  "payload": {"file": "<name>.tiff", "sha256": "<hex>"},  the values: a TIFF of the profile below, beside the header
#  "provenance": {…}}                                      (optional)
#
# The header is a vgio record like any other (validated on read and on the bytes
# before write); the values are the reserved `payload` sidecar (validate.jl), here
# a TIFF, bound by the SHA-256 of its bytes: a payload whose digest is not the
# header's is refused before it is decoded.
#
# THE TIFF PROFILE (checked on write, on the file written, and on read):
#   - baseline TIFF: classic (not BigTIFF), little-endian ("II*\0"), ONE image (one IFD);
#   - one sample per pixel, min-is-black, BitsPerSample 32, SampleFormat 3 (IEEE floating
#     point): every Float32 value bit for bit, NaN (and its payload) included;
#   - Compression 1 (none) or 8 (Deflate, zlib: lossless), Predictor 1 (none): never the
#     floating-point predictor (3), nor the horizontal one;
#   - strips, not tiles; Orientation 1 (row 1 on top, column 1 on the left) when stated;
#   - no meaning in the tags: everything the values mean is in the header, so the writer
#     sets only the structural tags above, and a reader reads only them.
# Its rows are the raster's rows, row 1 on top. In a chart frame the raster covers the
# chart rectangle `area` (the design chart: x right, y up; its lower-left corner and its
# widths, in `length_unit`): row 1 is the rectangle's top edge, column 1 its left edge, each
# value the mean over its cell of `w / W × h / H`. In a pixel frame value [r, c] is the
# image's pixel in row r and column c, its centre where the pixel frame puts it
# (colmap: (c − 0.5, r − 0.5)).
#
# The TIFF codec is the stack's one (tiff.jl, libtiff).

const _RASTER_NAN = ("unobserved", "none")
const _RASTER_CHARTS = ("page", "texture")
const _RASTER_ELEMENT = "float32"
const _TIFF_MAGIC_LE = UInt8['I', 'I', 0x2a, 0x00]      # a classic little-endian TIFF's first four bytes

const _RASTER_CHART_FRAME = _CheckedSpec(
    _Obj(:chart => _Named(in(_RASTER_CHARTS), "a chart, $(join(_RASTER_CHARTS, " or "))"), :length_unit => _LENGTH_UNIT,
         :area => _Vec(_Real(), 4)),
    f -> _area_problem(f[:area], "a chart frame's area"),
)
const _RASTER_SPEC = _Obj(
    Pair{Symbol, _Spec}[
        :schema => _schema_tok("raster"), :size => _Obj(:width => _PosInt(), :height => _PosInt()),
        :element_type => _Tok(_RASTER_ELEMENT),
        :frame => _Either(Pair{Symbol, _Spec}[:chart => _RASTER_CHART_FRAME, :pixel_frame => _Obj(:pixel_frame => _PIXEL_FRAME)]),
        :nan => _Named(in(_RASTER_NAN), "the meaning of a NaN value, $(join(_RASTER_NAN, " or "))"),
        :payload => _Obj(:file => _Named(_is_inside_path, "a relative path inside the header's directory (the calibration manifest's rule)"), :sha256 => _HEX64),
    ],
    _PROVENANCE,
)

# `f(tif)` on the TIFF at `path` once it is of the raster profile: its first bytes (read
# before libtiff opens it: a classic little-endian TIFF), then its directory; a TIFF
# outside the profile is an ArgumentError naming it and the rule.
function _with_raster_tiff(f, path)
    head = open(io -> read(io, 4), path)
    head == _TIFF_MAGIC_LE ||
        throw(ArgumentError("$(path): a raster's TIFF is a classic little-endian TIFF (\"II*\\0\"), not big-endian or BigTIFF"))
    return _with_tiff(path, "r") do tif
        msg = _raster_directory_problem(tif)
        msg === nothing || throw(ArgumentError("$(path): $(msg)"))
        f(tif)
    end
end
function _raster_directory_problem(tif)
    n = @ccall libtiff.TIFFNumberOfDirectories(tif::Ptr{Cvoid})::UInt32
    n == 1 || return "a raster's TIFF holds one image (one IFD), got $(n)"
    @ccall(libtiff.TIFFIsTiled(tif::Ptr{Cvoid})::Cint) == 0 || return "a raster's TIFF is in strips, not tiles"
    tag(t) = _tiff_get(tif, t, UInt16; defaulted = true)
    for (t, name, ok, want) in (
            (_TIFFTAG_SAMPLESPERPIXEL, "SamplesPerPixel", ==(1), "1"),
            (_TIFFTAG_BITSPERSAMPLE, "BitsPerSample", ==(32), "32"),
            (_TIFFTAG_SAMPLEFORMAT, "SampleFormat", ==(_SAMPLEFORMAT_IEEEFP), "3 (IEEE floating point)"),
            (_TIFFTAG_COMPRESSION, "Compression", in((_TIFF_COMPRESSION.none, _TIFF_COMPRESSION.deflate)), "1 (none) or 8 (Deflate)"),
            (_TIFFTAG_PREDICTOR, "Predictor", ==(_TIFF_PREDICTOR.none), "1 (none: no floating-point or horizontal predictor)"),
            (_TIFFTAG_ORIENTATION, "Orientation", ==(1), "1 (row 1 on top, column 1 on the left)"),
        )
        # an uncompressed file has no predictor (no codec registers the tag; asking libtiff for it logs an error): 1
        v = t == _TIFFTAG_PREDICTOR && tag(_TIFFTAG_COMPRESSION) == _TIFF_COMPRESSION.none ? UInt16(1) : tag(t)
        (v !== nothing && ok(v)) || return "a raster's TIFF has $(name) $(want), got $(repr(v))"
    end
    _tiff_get(tif, _TIFFTAG_PHOTOMETRIC, UInt16) == _PHOTOMETRIC_MINISBLACK || return "a raster's TIFF is min-is-black (Photometric 1)"
    return nothing
end
# …the TIFF at `path`, checked and decoded (one open), its size the header's.
function _read_raster_tiff(path, width::Int, height::Int)
    return _with_raster_tiff(path) do tif
        A = _read_tiff(Float32, tif, path)
        size(A) == (height, width) ||
            throw(ArgumentError("$(path): the TIFF is $(size(A, 2)) × $(size(A, 1)) (width × height), its header states $(width) × $(height)"))
        A
    end
end

"""
    save_raster(path, values::AbstractMatrix{Float32}; frame, nan, length_unit = mm, compression = :deflate,
                provenance = (;)) -> path

Write a Float32 raster as a `vgio/raster/v1` file: the JSON header at `path`
and the values beside it, `<name>.tiff` (`path`'s name with its extension
replaced), a TIFF of the raster profile (one image, one 32-bit IEEE
floating-point sample per pixel, little-endian, in strips, `compression`
`:deflate` or `:none`, both lossless, and no predictor), so every value — NaN
included — reads back bit for bit. The header states the raster's `size`
(`{width, height}`: its columns and rows), its element type, its `frame`, what
a NaN means and the SHA-256 of the TIFF's bytes; nothing the values mean is in
the TIFF's tags. `values` is (rows, columns), row 1 on top.

- `frame`, required: a chart rectangle, `(chart = :page, area = Rect2(x, y, w, h))`
  (or `chart = :texture`; the design chart, x right and y up, `area` its
  lower-left corner and widths in plain metres, written in `length_unit`) —
  row 1 its top edge — or a pixel frame, `PixelFrame{:colmap}` — `values[r, c]`
  the image's pixel at row `r`, column `c`.
- `nan`, required: `:unobserved` (a NaN is a value nothing observed) or
  `:none` (the raster holds no NaN: one is refused).

Both files are written through a staging directory beside them and renamed
into place once the TIFF has been checked against the profile and the header
validated ([`write_staged`](@ref)); [`load_raster`](@ref) reads them back.
"""
function save_raster(
        path::AbstractString, values::AbstractMatrix{Float32}; frame, nan::Symbol, length_unit::Units = mm,
        compression::Symbol = :deflate, provenance::NamedTuple = (;),
    )
    String(nan) in _RASTER_NAN || throw(ArgumentError("save_raster: nan is one of :$(join(_RASTER_NAN, ", :")), got :$(nan)"))
    compression in (:none, :deflate) || throw(ArgumentError("save_raster: a raster's TIFF is uncompressed (:none) or Deflate (:deflate), got :$(compression)"))
    nan === :none && any(isnan, values) && throw(ArgumentError("save_raster: nan = :none states the raster holds no NaN; it holds $(count(isnan, values))"))
    height, width = size(values)
    fj = _raster_frame_json(frame, length_unit)
    tiff = first(splitext(basename(path))) * ".tiff"
    tiff == basename(path) && throw(ArgumentError("save_raster: the header is a JSON file beside its payload $(tiff); got the header path $(path)"))
    # the TIFF is renamed into place first, so a header never names a payload that is not there
    write_staged(joinpath(dirname(path), tiff) => tiff, String(path) => basename(path); dry_run = false) do staged, head
        write_tiff(staged, values; compression)
        _with_raster_tiff(Returns(nothing), staged)          # the file written is of the profile
        body = (
            schema = _schema("raster"), size = (; width, height), element_type = _RASTER_ELEMENT, frame = fj,
            nan = String(nan), payload = (file = tiff, sha256 = open(sha256_hex, staged)),
        )
        _write_store(head, _RASTER_SPEC, with_provenance(body, provenance); style = VgioStyle(; length_unit))
    end
    return path
end

_raster_frame_json(F::Type{<:PixelFrame}, _) = (pixel_frame = pixel_frame_name(F),)
function _raster_frame_json(f::NamedTuple, u::Units)
    (Set(keys(f)) == Set((:chart, :area)) && f.chart isa Symbol && f.area isa Rect2{<:Real}) || throw(
        ArgumentError("save_raster: a chart frame is (chart = :page | :texture, area = Rect2(x, y, w, h)), its area in plain metres; got $(repr(f))")
    )
    String(f.chart) in _RASTER_CHARTS || throw(ArgumentError("save_raster: a chart frame's chart is :$(join(_RASTER_CHARTS, " or :")), got :$(f.chart)"))
    a = f.area
    return (chart = String(f.chart), length_unit = unit_string(u), area = [_metres_json(x, u) for x in JSON.lower(VgioStyle(), a)])
end
_raster_frame_json(f, _) = throw(
    ArgumentError("save_raster: frame is a chart rectangle (chart = :page | :texture, area = Rect2(…)) or a pixel frame (PixelFrame{:colmap}); got $(repr(f))")
)

"""
    load_raster(path) -> (; values::Matrix{Float32}, frame, nan::Symbol)

Read a `vgio/raster/v1` file ([`save_raster`](@ref)'s): the header is
validated, the TIFF it names (beside it) must hash to the header's SHA-256 — a
corrupted or replaced payload is an `ArgumentError` naming it, before it is
decoded — and must be of the raster profile and of the header's size. `values`
is (rows, columns), every value as stored, NaN bit for bit; `frame` the
header's, `(chart, area)` with `area` a `Rect2{Float64}` in plain metres, or
the `PixelFrame` type; `nan` `:unobserved` or `:none` (a NaN in a `:none`
raster is refused).
"""
function load_raster(path::AbstractString)
    d = _read_store(path, "raster", _RASTER_SPEC)
    full = _check_bound_file(dirname(abspath(path)), d.payload.file::String, d.payload.sha256::String, path, "the header")
    values = _read_raster_tiff(full, Int(d.size.width::Integer), Int(d.size.height::Integer))
    nan = Symbol(d.nan::String)
    nan === :none && any(isnan, values) && throw(ArgumentError("$(path): its nan is \"none\", yet its payload holds $(count(isnan, values)) NaN"))
    return (; values, frame = _raster_frame_from_json(d.frame), nan)
end

function _raster_frame_from_json(f)
    haskey(f, :pixel_frame) && return pixel_frame_from_name(f.pixel_frame::String)
    u = parse_unit(f.length_unit::String)
    return (chart = Symbol(f.chart::String), area = JSON.lift(VgioStyle(), Rect2, Any[metres(x, u) for x in f.area::Vector{Any}]))
end
