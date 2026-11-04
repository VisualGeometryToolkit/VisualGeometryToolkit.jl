# =============================================================================
# rawpy backend of load_raw — PythonCall, decoding only
# =============================================================================
#
# Python opens the file once and hands back, once each, the arrays and levels
# that RadiometricCamera's convert_raw.py read (rawpy's raw_image_visible,
# raw_pattern, black_level_per_channel, white_level, daylight_whitebalance)
# and the EXIF fields it recorded (exifread, on the same file object). The
# photosites cross as one strided copy (PyArray); no per-pixel Python call.
# rawpy's raw_colors_visible is checked against the pattern tiled from the raw
# origin (`photosite_colors`), the colours the mosaic derives. All processing
# is in src/raw_image.jl. rawpy and exifread are declared in the package's
# CondaPkg.toml.
#
# rawpy exposes no spatial black pattern (LibRaw's cblack[4:]) to Python, only
# black_level_per_channel, which drops one; the shared rule that refuses it
# (`_per_channel_black`) needs that array, so this decoder cannot apply it —
# RawpyDecoder's docstring says so.
# =============================================================================

module VisualGeometryToolkitFeaturesPythonExt

using PythonCall: Py, PyArray, pyconvert, pyimport, pyis, pybuiltins, pystr
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures, RawpyDecoder, RawMosaic, photosite_colors

_none(x::Py) = pyis(x, pybuiltins.None)
_str(x::Py) = pyconvert(String, pystr(x))

# convert_raw.py's _ratio: the first value's num/den, else the tag's text as a number
function _ratio(tag::Py)
    try
        v = tag.values[0]
        return pyconvert(Float64, v.num) / pyconvert(Float64, v.den)
    catch
        return tryparse(Float64, _str(tag))
    end
end

# The EXIF fields of the open file `f` (exifread reads from its current position).
function _exif(f::Py)
    tags = pyimport("exifread").process_file(f; details = false)
    tag(k) = (t = tags.get(k); _none(t) ? nothing : t)
    t_dt, t_ss = tag("EXIF DateTimeOriginal"), tag("EXIF SubSecTimeOriginal")
    t_exp, t_fn, t_fl = tag("EXIF ExposureTime"), tag("EXIF FNumber"), tag("EXIF FocalLength")
    t_iso, t_model = tag("EXIF ISOSpeedRatings"), tag("Image Model")
    return (;
        datetime_original = t_dt === nothing ? nothing : _str(t_dt),
        subsec = t_ss === nothing ? "" : _str(t_ss),
        exposure_s = t_exp === nothing ? nothing : _ratio(t_exp),
        fnumber = t_fn === nothing ? nothing : _ratio(t_fn),
        focal_mm = t_fl === nothing ? nothing : _ratio(t_fl),
        iso = t_iso === nothing ? nothing : parse(Int, _str(t_iso)),
        model = t_model === nothing ? nothing : _str(t_model),
    )
end

# The file is opened once: exifread reads it, then rawpy decodes the same file
# object from its start (rawpy.imread takes a file-like object).
function VisualGeometryToolkitFeatures._decode_raw(::RawpyDecoder, path::AbstractString)
    f = pybuiltins.open(String(path), "rb")
    try
        exif = _exif(f)
        f.seek(0)
        return _mosaic(pyimport("rawpy"), f, exif)
    finally
        f.close()
    end
end

function _mosaic(rawpy::Py, f::Py, exif::NamedTuple)
    raw = rawpy.imread(f)
    try
        pattern_py = raw.raw_pattern
        _none(pattern_py) && error("load_raw: not a flat Bayer raw (rawpy returns no raw_pattern)")
        img = Matrix{UInt16}(PyArray{UInt16, 2}(raw.raw_image_visible; copy = false))
        pattern = Matrix{UInt8}(PyArray{UInt8, 2}(pattern_py; copy = false))
        black = pyconvert(Vector{Float64}, raw.black_level_per_channel)
        white = pyconvert(Float64, raw.white_level)
        wb = pyconvert(Vector{Float64}, raw.daylight_whitebalance)
        lin = raw.camera_white_level_per_channel
        s = raw.sizes
        ver = pyconvert(Tuple{Int, Int, Int}, rawpy.libraw_version)
        metadata = (;
            exif...,
            daylight_wb = (wb[1], wb[2], wb[3]),
            linear_max = _none(lin) ? nothing : Tuple(pyconvert(Vector{Float64}, lin)),
            decoder = "rawpy " * pyconvert(String, rawpy.__version__),
            libraw_version = join(ver, '.'),
        )
        m = RawMosaic(img, pattern, (black[1], black[2], black[3], black[4]), white,
            (pyconvert(Int, s.top_margin), pyconvert(Int, s.left_margin)),
            (pyconvert(Int, s.raw_height), pyconvert(Int, s.raw_width)), metadata)
        # rawpy's own per-photosite colours must be the pattern tiled from the raw origin
        PyArray{UInt8, 2}(raw.raw_colors_visible; copy = false) == photosite_colors(m) ||
            error("load_raw: rawpy's raw_colors_visible is not its raw_pattern tiled from the raw origin")
        return m
    finally
        raw.close()
    end
end

end # module
