# =============================================================================
# vgio/raster/v2.0 — a raster: a JSON header and its TIFF payload; THE binding of a bulk array to a file
# =============================================================================
#
# {"schema": "vgio/raster/v2.0", <envelope>,
#  "size": {"width": W, "height": H},                      the raster's columns and rows
#  "element_type": "float32" | "uint8",
#  "frame": {"type": "page" | "texture", "area": [x, y, w, h]}   a chart's rectangle (lengths in units.length), or
#         | {"type": "pixel"},                             an image's pixels (in units.pixel)
#  "nan": "unobserved" | "none",                           what a NaN value means ("none": the raster holds none; uint8: "none")
#  "payload": {"file": "<name>.tiff", "sha256": "<hex>"},  the values: a TIFF of the profile below, beside the header
#  "provenance": {…}}                                      (optional)
#
# The header is a vgio record like any other (validated on read and on the bytes
# before write); the values are the reserved `payload` sidecar (validate.jl), here
# a TIFF, bound by the SHA-256 of its bytes: a payload whose digest is not the
# header's is refused before it is decoded. The PAYLOAD BODY `{file, sha256}` + a
# TIFF of the profile is the stack's ONE binding of a bulk array to a file:
# another kind that holds a raster (vgio/sensor_characterisation's dark offset and
# variance maps) embeds the same body (`_raster_payload_json`, `_raster_payload_from_json`).
#
# THE TIFF PROFILE (checked on write, on the file written, and on read):
#   - baseline TIFF: classic (not BigTIFF), little-endian ("II*\0"), ONE image (one IFD);
#   - one sample per pixel, min-is-black: a float32 raster BitsPerSample 32, SampleFormat 3
#     (IEEE floating point: every value bit for bit, NaN and its payload included); a uint8
#     raster BitsPerSample 8, SampleFormat 1 (unsigned integer);
#   - Compression 1 (none) or 8 (Deflate, zlib: lossless), Predictor 1 (none): never the
#     floating-point predictor (3), nor the horizontal one;
#   - strips, not tiles; Orientation 1 (row 1 on top, column 1 on the left) when stated;
#   - no meaning in the tags: everything the values mean is in the header, so the writer
#     sets only the structural tags above, and a reader reads only them.
# Its rows are the raster's rows, row 1 on top (`conventions.grid_layout`, row-major). In a
# chart frame the raster covers the chart rectangle `area` (the design chart: x right, y up;
# its lower-left corner and its widths, in units.length): row 1 is the rectangle's top edge,
# column 1 its left edge, each value the mean over its cell of `w / W × h / H`. In a pixel
# frame value [r, c] is the image's pixel in row r and column c, its centre where
# `units.pixel` puts it (colmap: (c − 0.5, r − 0.5)).
#
# vgio/raster/v1 (IO 0.9.5) reads through its step (`_raster_v1_to_v2`). The TIFF codec is
# the stack's one (tiff.jl, libtiff).

const _RASTER_NAN = ("unobserved", "none")
const _RASTER_CHARTS = ("page", "texture")
# the element types a raster holds, by name
const _RASTER_ELEMENTS = (("float32", Float32), ("uint8", UInt8))
_raster_element_name(::Type{T}) where {T} = first(e for (e, S) in _RASTER_ELEMENTS if S === T)
_raster_element(name::AbstractString) = last(only(e for e in _RASTER_ELEMENTS if e[1] == name))
const _TIFF_MAGIC_LE = UInt8['I', 'I', 0x2a, 0x00]      # a classic little-endian TIFF's first four bytes

const _RASTER_PAYLOAD = _Obj(:file => _Named(_is_inside_path, "a relative path inside the header's directory (the calibration manifest's rule)"), :sha256 => _HEX64)
const _RASTER_FRAME = _CheckedSpec(
    _Obj(Pair{Symbol, _Spec}[:type => _Named(in((_RASTER_CHARTS..., "pixel")), "a chart ($(join(_RASTER_CHARTS, ", "))) or \"pixel\"")],
         Pair{Symbol, _Spec}[:area => _Vec(_Real(), 4)]),
    f -> (f[:type] == "pixel") == haskey(f, :area) ?
        (f[:type] == "pixel" ? "a pixel frame states no area" : "a chart frame's area: missing (a $(f[:type]) frame states its [x, y, w, h])") :
        haskey(f, :area) ? _area_problem(f[:area], "a chart frame's area") : nothing;
    rule = "a chart frame (page, texture) states its area [x, y, w, h], of positive widths; a pixel frame states none",
    schema = [Dict("if" => Dict("properties" => Dict("type" => Dict("const" => "pixel"))),
                   "then" => Dict("not" => Dict("required" => ["area"])), "else" => Dict("required" => ["area"]))],
)
const _RASTER_SPEC = _kind_spec("raster",
    Pair{Symbol, _Spec}[
        :size => _SIZE_INT, :element_type => _Named(in(first.(_RASTER_ELEMENTS)), "an element type ($(join(first.(_RASTER_ELEMENTS), ", ")))"),
        :frame => _RASTER_FRAME,
        :nan => _Named(in(_RASTER_NAN), "the meaning of a NaN value, $(join(_RASTER_NAN, " or "))"),
        :payload => _RASTER_PAYLOAD,
    ];
    check = x -> (x[:element_type] == "uint8" && x[:nan] != "none") ? "a uint8 raster holds no NaN: its nan is \"none\"" : nothing,
    rule = "a uint8 raster's nan is \"none\"",
    schema = [Dict("if" => Dict("properties" => Dict("element_type" => Dict("const" => "uint8"))),
                   "then" => Dict("properties" => Dict("nan" => Dict("const" => "none"))))])

# `f(tif)` on the TIFF at `path` once it is of the raster profile for the element type `T`: its first bytes (read
# before libtiff opens it: a classic little-endian TIFF), then its directory; a TIFF outside the profile is an
# ArgumentError naming it and the rule.
function _with_raster_tiff(f, path, ::Type{T}) where {T}
    head = open(io -> read(io, 4), path)
    head == _TIFF_MAGIC_LE ||
        throw(ArgumentError("$(path): a raster's TIFF is a classic little-endian TIFF (\"II*\\0\"), not big-endian or BigTIFF"))
    return _with_tiff(path, "r") do tif
        msg = _raster_directory_problem(tif, T)
        msg === nothing || throw(ArgumentError("$(path): $(msg)"))
        f(tif)
    end
end
function _raster_directory_problem(tif, ::Type{T}) where {T}
    n = @ccall libtiff.TIFFNumberOfDirectories(tif::Ptr{Cvoid})::UInt32
    n == 1 || return "a raster's TIFF holds one image (one IFD), got $(n)"
    @ccall(libtiff.TIFFIsTiled(tif::Ptr{Cvoid})::Cint) == 0 || return "a raster's TIFF is in strips, not tiles"
    tag(t) = _tiff_get(tif, t, UInt16; defaulted = true)
    sample = _tiff_sample(T)
    for (t, name, ok, want) in (
            (_TIFFTAG_SAMPLESPERPIXEL, "SamplesPerPixel", ==(1), "1"),
            (_TIFFTAG_BITSPERSAMPLE, "BitsPerSample", ==(sample.bits), "$(sample.bits) (a $(_raster_element_name(T)) raster)"),
            (_TIFFTAG_SAMPLEFORMAT, "SampleFormat", ==(sample.format), "$(sample.format) (a $(_raster_element_name(T)) raster)"),
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

# THE PAYLOAD BODY, `{file, sha256}` + its TIFF: the values `A` (a Matrix{Float32} or Matrix{UInt8}, (rows, columns))
# written as `name` in `dir` through the profile and checked against it, and its body; …
function _raster_payload_json(dir::AbstractString, name::AbstractString, A::AbstractMatrix{T}; compression::Symbol = :deflate) where {T <: Union{Float32, UInt8}}
    compression in (:none, :deflate) || throw(ArgumentError("a raster's TIFF is uncompressed (:none) or Deflate (:deflate), got :$(compression)"))
    path = joinpath(dir, name)
    write_tiff(path, Array(A); compression)
    _with_raster_tiff(Returns(nothing), path, T)            # the file written is of the profile
    return (file = String(name), sha256 = open(sha256_hex, path))
end
# …and back: the body's file (beside `owner`, the file that names it), bound by its digest (refused before it is
# decoded), of the profile for `T` and of the size `width × height`.
function _raster_payload_from_json(owner::AbstractString, b, ::Type{T}, width::Int, height::Int) where {T}
    full = _check_bound_file(dirname(abspath(owner)), b.file::String, b.sha256::String, owner, "the header")
    return _with_raster_tiff(full, T) do tif
        A = _read_tiff(T, tif, full)
        size(A) == (height, width) ||
            throw(ArgumentError("$(full): the TIFF is $(size(A, 2)) × $(size(A, 1)) (width × height), its header states $(width) × $(height)"))
        A
    end
end

"""
    save_raster(path, values::AbstractMatrix{Float32 | UInt8}; frame, nan, length_unit = mm, compression = :deflate,
                provenance = (;)) -> path

Write a raster as a `vgio/raster/v2.0` file: the JSON header at `path` and the
values beside it, `<name>.tiff` (`path`'s name with its extension replaced), a
TIFF of the raster profile (one image, one sample per pixel — a 32-bit IEEE
float or an unsigned byte —, little-endian, in strips, `compression`
`:deflate` or `:none`, both lossless, and no predictor), so every value — NaN
included — reads back bit for bit. The header states the raster's `size`
(`{width, height}`: its columns and rows), its element type, its `frame`, what
a NaN means and the SHA-256 of the TIFF's bytes; nothing the values mean is in
the TIFF's tags. `values` is (rows, columns), row 1 on top.

- `frame`, required: a chart rectangle, `(chart = :page, area = Rect2(x, y, w, h))`
  (or `chart = :texture`; the design chart, x right and y up, `area` its
  lower-left corner and widths in plain metres, written in `length_unit`) —
  row 1 its top edge — or a pixel frame, `PixelFrame{:colmap}` — `values[r, c]`
  the image's pixel at row `r`, column `c` (the file's `units.pixel`).
- `nan`, required: `:unobserved` (a NaN is a value nothing observed) or
  `:none` (the raster holds no NaN: one is refused; a `UInt8` raster's).

Both files are written through a staging directory beside them and renamed
into place once the TIFF has been checked against the profile and the header
validated ([`write_staged`](@ref)); [`load_raster`](@ref) reads them back.
"""
function save_raster(
        path::AbstractString, values::AbstractMatrix{T}; frame, nan::Symbol, length_unit::Units = mm,
        compression::Symbol = :deflate, provenance::NamedTuple = (;),
    ) where {T <: Union{Float32, UInt8}}
    String(nan) in _RASTER_NAN || throw(ArgumentError("save_raster: nan is one of :$(join(_RASTER_NAN, ", :")), got :$(nan)"))
    compression in (:none, :deflate) || throw(ArgumentError("save_raster: a raster's TIFF is uncompressed (:none) or Deflate (:deflate), got :$(compression)"))
    nan === :none && T === Float32 && any(isnan, values) && throw(ArgumentError("save_raster: nan = :none states the raster holds no NaN; it holds $(count(isnan, values))"))
    height, width = size(values)
    fj, pf = _raster_frame_json(frame, length_unit)
    tiff = first(splitext(basename(path))) * ".tiff"
    tiff == basename(path) && throw(ArgumentError("save_raster: the header is a JSON file beside its payload $(tiff); got the header path $(path)"))
    # the TIFF is renamed into place first, so a header never names a payload that is not there
    write_staged(joinpath(dirname(path), tiff) => tiff, String(path) => basename(path); dry_run = false) do staged, head
        body = (size = (; width, height), element_type = _raster_element_name(T), frame = fj, nan = String(nan),
                payload = _raster_payload_json(dirname(staged), basename(staged), values; compression))
        _write_kind(head, "raster", body, provenance; style = VgioStyle(; length_unit, pixel_frame = pf))
    end
    return path
end

# A frame as the header's `frame`, and the pixel frame the file declares (a chart's: the canonical one).
_raster_frame_json(F::Type{<:PixelFrame}, _) = ((type = "pixel",), F)
function _raster_frame_json(f::NamedTuple, u::Units)
    (Set(keys(f)) == Set((:chart, :area)) && f.chart isa Symbol && f.area isa Rect2{<:Real}) || throw(
        ArgumentError("save_raster: a chart frame is (chart = :page | :texture, area = Rect2(x, y, w, h)), its area in plain metres; got $(repr(f))")
    )
    String(f.chart) in _RASTER_CHARTS || throw(ArgumentError("save_raster: a chart frame's chart is :$(join(_RASTER_CHARTS, " or :")), got :$(f.chart)"))
    return ((type = String(f.chart), area = [_metres_json(x, u) for x in JSON.lower(VgioStyle(), f.area)]), CANONICAL_PIXEL_FRAME)
end
_raster_frame_json(f, _) = throw(
    ArgumentError("save_raster: frame is a chart rectangle (chart = :page | :texture, area = Rect2(…)) or a pixel frame (PixelFrame{:colmap}); got $(repr(f))")
)

"""
    load_raster(path) -> (; values::Matrix{Float32 | UInt8}, frame, nan::Symbol)

Read a `vgio/raster/v2.0` file ([`save_raster`](@ref)'s; a v1 file through
its step): the header is validated, the TIFF it names (beside it) must hash to
the header's SHA-256 — a corrupted or replaced payload is an `ArgumentError`
naming it, before it is decoded — and must be of the raster profile for its
element type and of the header's size. `values` is (rows, columns), every value
as stored, NaN bit for bit; `frame` the header's, `(chart, area)` with `area`
a `Rect2{Float64}` in plain metres, or the `PixelFrame` type the file
declares; `nan` `:unobserved` or `:none` (a NaN in a `:none` raster is
refused).
"""
function load_raster(path::AbstractString)
    d = _read_store(path, "raster", _RASTER_SPEC)
    values = _raster_payload_from_json(path, d.payload, _raster_element(d.element_type::String), Int(d.size.width::Integer), Int(d.size.height::Integer))
    nan = Symbol(d.nan::String)
    nan === :none && eltype(values) === Float32 && any(isnan, values) &&
        throw(ArgumentError("$(path): its nan is \"none\", yet its payload holds $(count(isnan, values)) NaN"))
    return (; values, frame = _raster_frame_from_json(d.frame, VgioStyle(d)), nan)
end

function _raster_frame_from_json(f, s::VgioStyle)
    f.type::String == "pixel" && return _pixel_frame(s)
    u = s.length_unit
    return (chart = Symbol(f.type::String), area = JSON.lift(VgioStyle(), Rect2, Any[metres(x, u) for x in f.area::Vector{Any}]))
end

# vgio/raster/v1 (IO 0.9.5) → v2.0 (IO 0.10): the envelope (v1's frame stated its own `length_unit` or `pixel_frame`:
# the units object's); the frame `{type, area}` or `{type: "pixel"}`. The payload, its file and its digest, as it is.
function _raster_v1_to_v2(old::AbstractDict)
    f = old["frame"]
    chart = haskey(f, "chart")
    t = migrate_envelope(old, "vgio/raster/v2.0"; length = chart ? f["length_unit"] : "mm", pixel = chart ? "colmap" : f["pixel_frame"])
    t["frame"] = chart ? Dict{String, Any}("type" => f["chart"], "area" => f["area"]) : Dict{String, Any}("type" => "pixel")
    return t
end
migrate_schema(::Val{Symbol("vgio/raster/v1")}, t::AbstractDict) = _raster_v1_to_v2(t)
