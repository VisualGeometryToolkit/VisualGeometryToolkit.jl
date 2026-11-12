# =============================================================================
# LibRaw_jll backend of load_raw — a ccall layer over LibRaw's C API
# =============================================================================
#
# Decodes only: open, unpack, copy the visible photosites and the levels out,
# close. All processing is in src/raw_image.jl. The one exception is the
# display demosaic, which is LibRaw's own interpolation (`libraw_dcraw_process`,
# set to linear output; `_demosaiced`): its 16-bit image is handed back, and
# the white balance and gain are applied in Julia.
#
# What the C API's accessors give is read through them (libraw_get_raw_height/
# width, libraw_get_iheight/iwidth, libraw_COLOR, libraw_get_color_maximum,
# libraw_get_pre_mul/cam_mul/rgb_cam, libraw_get_iparams, libraw_get_imgother).
# The rest — the raw_image pointer, the visible-area margins, the black levels
# (color.cblack/black) — has no accessor and lives in `libraw_data_t`. Those
# structures are mirrored below from LibRaw_jll 0.22.2's libraw_types.h,
# field for field (a Julia isbits struct takes the C layout), and only the
# chain of structures between the accessors' pointers and the fields needed:
#
#   libraw_data_t = { image*, sizes, idata, lens, makernotes, shootinginfo,
#                     params, rawparams, progress_flags, process_warnings,
#                     color, other, thumbnail, thumbs_list, rawdata, parent }
#
# `sizes` is at offset 8 (after one pointer); `color` ends where `other` begins
# (the pointer libraw_get_imgother returns), and `rawdata` follows `other`,
# `thumbnail` and `thumbs_list`. The layout is guaranteed by three checks,
# each failing loudly (ErrorException) rather than reading a wrong offset:
#
#   1. at load: LibRaw's version is 0.22.x (`libraw_versionNumber`), the
#      series whose header this mirrors;
#   2. per file: the mirrored offsets agree with the accessors — `idata` sits
#      where libraw_get_iparams points, the mirrored sizes equal
#      libraw_get_raw_height/width and get_iheight/iwidth, and the mirrored
#      `color` reproduces libraw_get_color_maximum, pre_mul, cam_mul and
#      rgb_cam bit for bit (fields spread across the whole structure);
#   3. per file: the `rawdata` copy that unpack makes agrees with the live
#      structures (sizes, filters, maximum, black levels), and its raw_image
#      pointer is the flat Bayer buffer (`raw_alloc`).
#
# A new LibRaw minor version means re-mirroring from its header and moving the
# version check; the compat bound (LibRaw_jll = "0.22.2") keeps Pkg from
# installing one silently.
# =============================================================================

module VisualGeometryToolkitFeaturesLibRawExt

using LibRaw_jll: LibRaw_jll, libraw
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures, LibRawDecoder
using VisualGeometryToolkitCore: RawMosaic, photosite_colors   # the decoded mosaic, and its colour phase (Core's one rule; the in-memory demosaic)

# ---- mirrored structures (libraw_types.h, LibRaw 0.22) ----------------------

const CBLACK_SIZE = 4104          # LIBRAW_CBLACK_SIZE
const THUMBNAIL_MAXCOUNT = 8      # LIBRAW_THUMBNAIL_MAXCOUNT
const XTRANS = 9                  # LIBRAW_XTRANS (filters value)

struct InsetCrop
    cleft::UInt16; ctop::UInt16; cwidth::UInt16; cheight::UInt16
end

struct ImageSizes
    raw_height::UInt16; raw_width::UInt16; height::UInt16; width::UInt16
    top_margin::UInt16; left_margin::UInt16; iheight::UInt16; iwidth::UInt16
    raw_pitch::UInt32
    pixel_aspect::Float64
    flip::Int32
    mask::NTuple{32, Int32}
    raw_aspect::UInt16
    raw_inset_crops::NTuple{2, InsetCrop}
end

struct IParams
    guard::NTuple{4, UInt8}
    make::NTuple{64, UInt8}; model::NTuple{64, UInt8}; software::NTuple{64, UInt8}
    normalized_make::NTuple{64, UInt8}; normalized_model::NTuple{64, UInt8}
    maker_index::UInt32; raw_count::UInt32; dng_version::UInt32; is_foveon::UInt32
    colors::Int32
    filters::UInt32
    xtrans::NTuple{36, Int8}; xtrans_abs::NTuple{36, Int8}
    cdesc::NTuple{5, UInt8}
    xmplen::UInt32
    xmpdata::Ptr{UInt8}
end

struct Ph1
    format::Int32; key_off::Int32; tag_21a::Int32; t_black::Int32
    split_col::Int32; black_col::Int32; split_row::Int32; black_row::Int32
    tag_210::Float32
end

struct DngColor
    parsedfields::UInt32
    illuminant::UInt16
    calibration::NTuple{16, Float32}
    colormatrix::NTuple{12, Float32}
    forwardmatrix::NTuple{12, Float32}
end

struct DngRawOpcode
    len::UInt32
    data::Ptr{Cvoid}
end

struct DngLevels
    parsedfields::UInt32
    dng_cblack::NTuple{CBLACK_SIZE, UInt32}
    dng_black::UInt32
    dng_fcblack::NTuple{CBLACK_SIZE, Float32}
    dng_fblack::Float32
    dng_whitelevel::NTuple{4, UInt32}
    default_crop::NTuple{4, UInt16}
    user_crop::NTuple{4, Float32}
    preview_colorspace::UInt32
    analogbalance::NTuple{4, Float32}
    asshotneutral::NTuple{4, Float32}
    baseline_exposure::Float32
    LinearResponseLimit::Float32
    rawopcodes::NTuple{3, DngRawOpcode}
end

struct P1Color
    romm_cam::NTuple{9, Float32}
end

struct ColorData
    curve::NTuple{65536, UInt16}
    cblack::NTuple{CBLACK_SIZE, UInt32}
    black::UInt32
    data_maximum::UInt32
    maximum::UInt32
    linear_max::NTuple{4, UInt32}
    fmaximum::Float32
    fnorm::Float32
    white::NTuple{64, UInt16}
    cam_mul::NTuple{4, Float32}
    pre_mul::NTuple{4, Float32}
    cmatrix::NTuple{12, Float32}
    ccm::NTuple{12, Float32}
    rgb_cam::NTuple{12, Float32}
    cam_xyz::NTuple{12, Float32}
    phase_one_data::Ph1
    flash_used::Float32
    canon_ev::Float32
    model2::NTuple{64, UInt8}
    UniqueCameraModel::NTuple{64, UInt8}
    LocalizedCameraModel::NTuple{64, UInt8}
    ImageUniqueID::NTuple{64, UInt8}
    RawDataUniqueID::NTuple{17, UInt8}
    OriginalRawFileName::NTuple{64, UInt8}
    profile::Ptr{Cvoid}
    profile_length::UInt32
    black_stat::NTuple{8, UInt32}
    dng_color::NTuple{2, DngColor}
    dng_levels::DngLevels
    WB_Coeffs::NTuple{1024, Int32}
    WBCT_Coeffs::NTuple{320, Float32}
    as_shot_wb_applied::Int32
    P1_color::NTuple{2, P1Color}
    raw_bps::UInt32
    ExifColorSpace::Int32
end

# The leading fields of libraw_lensinfo_t (the pointer libraw_get_lensinfo
# returns, `lens` of libraw_data_t, right after `idata`): only these are read,
# so only they are mirrored; the nested nikon/dng/makernotes parts that follow
# are not.
struct LensInfoPrefix
    MinFocal::Float32; MaxFocal::Float32; MaxAp4MinFocal::Float32; MaxAp4MaxFocal::Float32; EXIF_MaxAp::Float32
    LensMake::NTuple{128, UInt8}; Lens::NTuple{128, UInt8}; LensSerial::NTuple{128, UInt8}; InternalLensSerial::NTuple{128, UInt8}
end

struct GpsInfo
    latitude::NTuple{3, Float32}; longitude::NTuple{3, Float32}; gpstimestamp::NTuple{3, Float32}
    altitude::Float32
    altref::UInt8; latref::UInt8; longref::UInt8; gpsstatus::UInt8; gpsparsed::UInt8
end

struct ImgOther
    iso_speed::Float32; shutter::Float32; aperture::Float32; focal_len::Float32
    timestamp::Int64                       # time_t: 64-bit on every platform LibRaw_jll ships
    shot_order::UInt32
    gpsdata::NTuple{32, UInt32}
    parsed_gps::GpsInfo
    desc::NTuple{512, UInt8}; artist::NTuple{64, UInt8}
    analogbalance::NTuple{4, Float32}
end

struct Thumbnail
    tformat::Cint
    twidth::UInt16; theight::UInt16
    tlength::UInt32
    tcolors::Int32
    thumb::Ptr{UInt8}
end

struct ThumbnailItem
    tformat::Cint
    twidth::UInt16; theight::UInt16; tflip::UInt16
    tlength::UInt32
    tmisc::UInt32
    toffset::Int64
end

struct ThumbnailList
    thumbcount::Int32
    thumblist::NTuple{THUMBNAIL_MAXCOUNT, ThumbnailItem}
end

struct InternalOutputParams
    mix_green::UInt32; raw_color::UInt32; zero_is_bad::UInt32
    shrink::UInt16; fuji_width::UInt16
end

struct RawData
    raw_alloc::Ptr{Cvoid}
    raw_image::Ptr{UInt16}
    color4_image::Ptr{Cvoid}; color3_image::Ptr{Cvoid}
    float_image::Ptr{Cvoid}; float3_image::Ptr{Cvoid}; float4_image::Ptr{Cvoid}
    ph1_cblack::Ptr{Cvoid}; ph1_rblack::Ptr{Cvoid}
    iparams::IParams
    sizes::ImageSizes
    ioparams::InternalOutputParams
    color::ColorData
end

# libraw_processed_image_t up to its pixel data (`data[1]`, at the last field's offset).
struct ProcessedImageHeader
    type::Cint                             # enum LibRaw_image_formats
    height::UInt16; width::UInt16; colors::UInt16; bits::UInt16
    data_size::UInt32
    data::UInt8
end

# One field of a mirrored structure at `p`, without loading the whole structure
# (ColorData alone is ~180 kB).
function _field(p::Ptr{Cvoid}, ::Type{T}, name::Symbol) where {T}
    i = findfirst(==(name), fieldnames(T))::Int
    return unsafe_load(Ptr{fieldtype(T, i)}(p + fieldoffset(T, i)))
end
_fieldptr(p::Ptr{Cvoid}, ::Type{T}, name::Symbol) where {T} = p + fieldoffset(T, findfirst(==(name), fieldnames(T))::Int)
_cstring(t::NTuple{N, UInt8}) where {N} = (k = findfirst(iszero, t); String(UInt8[t[i] for i in 1:(k === nothing ? N : k - 1)]))

# ---- the C API --------------------------------------------------------------

libraw_version() = unsafe_string(@ccall libraw.libraw_version()::Cstring)
libraw_version_number() = @ccall libraw.libraw_versionNumber()::Cint
# load_raw's cache key: the library's own version and the JLL build that ships it (a rebuild can carry patches)
VisualGeometryToolkitFeatures._decoder_version(::LibRawDecoder) = "LibRaw $(libraw_version()), LibRaw_jll $(pkgversion(LibRaw_jll))"

function __init__()
    v = libraw_version_number()
    (v >> 8) == 0x0016 || error("VisualGeometryToolkitFeaturesLibRawExt: its libraw_data_t layout mirrors LibRaw 0.22; the loaded LibRaw_jll is $(libraw_version()). Re-mirror the structures from that version's libraw_types.h.")
    return nothing
end

_check(what, ok::Bool) = ok || error("load_raw (LibRaw_jll $(libraw_version())): the mirrored libraw_data_t layout disagrees with LibRaw's accessors ($what); refusing to read by offset")

function _strerror(code)
    return unsafe_string(@ccall libraw.libraw_strerror(code::Cint)::Cstring)
end

# `f(lr)` on a LibRaw handle with `path` open (its headers parsed), closed after.
function _with_libraw(f, path::AbstractString, who::AbstractString)
    lr = @ccall libraw.libraw_init(0::Cuint)::Ptr{Cvoid}
    lr == C_NULL && error("$who: libraw_init failed")
    try
        rc = @ccall libraw.libraw_open_file(lr::Ptr{Cvoid}, path::Cstring)::Cint
        rc == 0 || error("$who: LibRaw cannot open $path: $(_strerror(rc))")
        return f(lr)
    finally
        @ccall libraw.libraw_close(lr::Ptr{Cvoid})::Cvoid
    end
end

function _unpack(lr::Ptr{Cvoid}, path::AbstractString)
    rc = @ccall libraw.libraw_unpack(lr::Ptr{Cvoid})::Cint
    rc == 0 || error("load_raw: LibRaw cannot unpack $path: $(_strerror(rc))")
    return nothing
end

VisualGeometryToolkitFeatures._decode_raw(::LibRawDecoder, path::AbstractString) = _with_libraw(path, "load_raw") do lr
    _unpack(lr, path)
    return _mosaic(lr, path)
end

VisualGeometryToolkitFeatures._decode_demosaic(::LibRawDecoder, path::AbstractString) = _with_libraw(path, "load_raw") do lr
    _unpack(lr, path)
    return _demosaiced(lr, path)
end

# The display demosaic of a mosaic in memory (a decoded one, or one corrected per photosite): its photosites handed to
# LibRaw as a Bayer buffer (`libraw_open_bayer`, the visible area itself, no margins), its colour phase at its first
# photosite (Core's `photosite_colors`), and its levels written where LibRaw reads them (`color.black`, `cblack[0…3]`,
# `maximum`, in the unpacked `rawdata` copy that `dcraw_process` restores from and in the live structures that the
# mosaic's layout checks compare it with); then the one demosaic, `_demosaiced`. The layout LibRaw then reports — the
# colour phase, the per-channel black levels (`_per_channel_black`), the white level — must be the mosaic's: a check, so
# a buffer LibRaw read otherwise is refused, never demosaiced.
const _OPENBAYER = Dict("RGGB" => 0x94, "BGGR" => 0x16, "GRBG" => 0x61, "GBRG" => 0x49)   # LIBRAW_OPENBAYER_*

VisualGeometryToolkitFeatures._decode_demosaic(::LibRawDecoder, m::RawMosaic) = _demosaic_mosaic(m)

function _demosaic_mosaic(m::RawMosaic)
    H, W = size(m)
    H >= 2 && W >= 2 || throw(ArgumentError("load_raw: a $(H)×$(W) mosaic holds no 2×2 cell to demosaic"))
    letters(c) = "RGBG"[Int(c) + 1]
    cell = photosite_colors(view(m, 1:2, 1:2))
    phase = String([letters(cell[1, 1]), letters(cell[1, 2]), letters(cell[2, 1]), letters(cell[2, 2])])
    code = get(_OPENBAYER, phase, nothing)
    code === nothing && throw(ArgumentError("load_raw: LibRaw demosaics a 2×2 Bayer mosaic; this one's cell reads $phase"))
    whole(x) = isinteger(x) && 0 <= x <= typemax(UInt16)
    all(whole, m.black) && whole(m.white) ||
        throw(ArgumentError("load_raw: LibRaw holds whole black and white levels (codes); this mosaic's are $(m.black) and $(m.white)"))
    common = minimum(m.black)
    buf = collect(reinterpret(UInt8, vec(permutedims(Array(parent(m))))))   # row-major, little-endian UInt16 (LibRaw's Bayer buffer)
    lr = @ccall libraw.libraw_init(0::Cuint)::Ptr{Cvoid}
    lr == C_NULL && error("load_raw: libraw_init failed")
    what = "a $(H)×$(W) mosaic in memory"
    try
        GC.@preserve buf begin
            rc = @ccall libraw.libraw_open_bayer(lr::Ptr{Cvoid}, buf::Ptr{UInt8}, length(buf)::Cuint, W::Cushort, H::Cushort,
                0::Cushort, 0::Cushort, 0::Cushort, 0::Cushort, 0::Cuchar, code::Cuchar, 0::Cuint, 0::Cuint, UInt32(common)::Cuint)::Cint
            rc == 0 || error("load_raw: LibRaw cannot open $what: $(_strerror(rc))")
            _unpack(lr, what)
            s = _structures(lr)
            rcolor_p = _fieldptr(_rawdata_ptr(s.other_p), RawData, :color)
            for color_p in (s.color_p, rcolor_p)
                unsafe_store!(Ptr{UInt32}(_fieldptr(color_p, ColorData, :black)), UInt32(common))
                unsafe_store!(Ptr{UInt32}(_fieldptr(color_p, ColorData, :maximum)), UInt32(m.white))
                cb_p = Ptr{UInt32}(_fieldptr(color_p, ColorData, :cblack))
                for c in 0:3                                         # cblack[c] of colour index c; no spatial pattern
                    unsafe_store!(cb_p, UInt32(m.black[c + 1] - common), c + 1)
                end
                unsafe_store!(cb_p, UInt32(0), 5); unsafe_store!(cb_p, UInt32(0), 6)
            end
            layout = _mosaic_layout(lr, what)
            same(p) = map(c -> c == 0x03 ? 0x01 : c, p)              # LibRaw marks a second green 3, or not
            tile(p) = [p[mod1(i, size(p, 1)), mod1(j, size(p, 2))] for i in 1:2, j in 1:2]
            (same(tile(layout.pattern)) == same(cell) && layout.black == m.black && layout.white == m.white) ||
                error("load_raw: LibRaw read $what as $(layout.pattern) with levels $(layout.black), $(layout.white); " *
                      "the mosaic's are $cell, $(m.black), $(m.white)")
            rgb, _ = _demosaiced(lr, what)
            return rgb, m.metadata
        end
    finally
        @ccall libraw.libraw_close(lr::Ptr{Cvoid})::Cvoid
    end
end

# The file's metadata alone: the headers, no unpack.
VisualGeometryToolkitFeatures._decode_metadata(::LibRawDecoder, path::AbstractString) = _with_libraw(path, "load_raw_metadata") do lr
    s = _structures(lr)
    return _metadata(s.idata_p, s.other_p, s.lens_p, s.color_p)
end

# Locate the live structures and check the mirror against the accessors
# (checks 1–2 of the header comment; they hold once the file is open).
# The live `sizes`: libraw_data_t's second field, after the `image` pointer.
_sizes_ptr(lr::Ptr{Cvoid}) = lr + sizeof(Ptr{Cvoid})

function _structures(lr::Ptr{Cvoid})
    sizes_p = _sizes_ptr(lr)
    idata_p = Ptr{Cvoid}(@ccall libraw.libraw_get_iparams(lr::Ptr{Cvoid})::Ptr{Cvoid})
    other_p = Ptr{Cvoid}(@ccall libraw.libraw_get_imgother(lr::Ptr{Cvoid})::Ptr{Cvoid})
    _check("idata offset", idata_p == sizes_p + sizeof(ImageSizes))
    sizes = unsafe_load(Ptr{ImageSizes}(sizes_p))
    _check("raw_height", Int(sizes.raw_height) == @ccall libraw.libraw_get_raw_height(lr::Ptr{Cvoid})::Cint)
    _check("raw_width", Int(sizes.raw_width) == @ccall libraw.libraw_get_raw_width(lr::Ptr{Cvoid})::Cint)
    _check("iheight", Int(sizes.iheight) == @ccall libraw.libraw_get_iheight(lr::Ptr{Cvoid})::Cint)
    _check("iwidth", Int(sizes.iwidth) == @ccall libraw.libraw_get_iwidth(lr::Ptr{Cvoid})::Cint)
    color_p = other_p - sizeof(ColorData)
    _check("color.maximum", Int(_field(color_p, ColorData, :maximum)) == @ccall libraw.libraw_get_color_maximum(lr::Ptr{Cvoid})::Cint)
    pre = _field(color_p, ColorData, :pre_mul)
    cam = _field(color_p, ColorData, :cam_mul)
    rgb = _field(color_p, ColorData, :rgb_cam)
    for k in 0:3
        _check("pre_mul[$k]", pre[k + 1] === @ccall libraw.libraw_get_pre_mul(lr::Ptr{Cvoid}, k::Cint)::Cfloat)
        _check("cam_mul[$k]", cam[k + 1] === @ccall libraw.libraw_get_cam_mul(lr::Ptr{Cvoid}, k::Cint)::Cfloat)
    end
    for a in 0:2, b in 0:3
        _check("rgb_cam[$a][$b]", rgb[4a + b + 1] === @ccall libraw.libraw_get_rgb_cam(lr::Ptr{Cvoid}, a::Cint, b::Cint)::Cfloat)
    end
    lens_p = Ptr{Cvoid}(@ccall libraw.libraw_get_lensinfo(lr::Ptr{Cvoid})::Ptr{Cvoid})
    _check("lens offset", lens_p == idata_p + sizeof(IParams))
    return (; sizes_p, sizes, idata_p, other_p, color_p, lens_p)
end

# The EXIF fields, as LibRaw parsed them; the white balance and linear limit
# from the colour data at `wb_color_p` (the unpacked `rawdata` copy in a decode,
# the live one when only the headers were read).
function _metadata(idata_p::Ptr{Cvoid}, other_p::Ptr{Cvoid}, lens_p::Ptr{Cvoid}, wb_color_p::Ptr{Cvoid})
    other = unsafe_load(Ptr{ImgOther}(other_p))
    lens = unsafe_load(Ptr{LensInfoPrefix}(lens_p))
    nonempty = VisualGeometryToolkitFeatures._text   # a stripped string, `nothing` when empty
    lin = _field(wb_color_p, ColorData, :linear_max)
    rpre = _field(wb_color_p, ColorData, :pre_mul)
    return (;
        datetime_original = other.timestamp == 0 ? nothing : _strftime(other.timestamp),
        subsec = "",                                  # LibRaw keeps no SubSecTimeOriginal
        exposure_s = other.shutter > 0 ? Float64(other.shutter) : nothing,
        fnumber = other.aperture > 0 ? Float64(other.aperture) : nothing,
        focal_mm = other.focal_len > 0 ? Float64(other.focal_len) : nothing,
        iso = other.iso_speed > 0 ? round(Int, other.iso_speed) : nothing,
        model = nonempty(_cstring(_field(idata_p, IParams, :model))),
        make = nonempty(_cstring(_field(idata_p, IParams, :make))),
        body_serial = nothing,                        # shootinginfo.BodySerial: past the unmirrored makernotes
        lens_make = nonempty(_cstring(lens.LensMake)),
        lens_model = nonempty(_cstring(lens.Lens)),
        lens_serial = nonempty(_cstring(lens.LensSerial)),
        focus_distance_m = nothing,                   # LibRaw parses no focus distance into a common field
        daylight_wb = (Float64(rpre[1]), Float64(rpre[2]), Float64(rpre[3])),
        linear_max = all(>(0), lin) ? Float64.(lin) : nothing,
        decoder = "LibRaw_jll",
        libraw_version = libraw_version(),
    )
end

# The unpacked `rawdata` copy: after `other`, `thumbnail` and `thumbs_list`.
_rawdata_ptr(other_p::Ptr{Cvoid}) = other_p + sizeof(ImgOther) + sizeof(Thumbnail) + sizeof(ThumbnailList)

# The EXIF orientation of a sizes structure (`flip`: 0 none, 3 180°, 5 90° CCW, 6 90° CW).
_store_flip!(sizes_p::Ptr{Cvoid}, flip::Integer) = unsafe_store!(Ptr{Int32}(_fieldptr(sizes_p, ImageSizes, :flip)), Int32(flip))

# The mosaic of an open, unpacked handle: its layout checked (the mirrored structures, a
# flat 2×2 Bayer sensor), then its visible photosites copied out (`_mosaic_layout`'s
# description, the one copy).
function _mosaic(lr::Ptr{Cvoid}, path::AbstractString)
    m = _mosaic_layout(lr, path)
    full = unsafe_wrap(Matrix{UInt16}, m.raw_image, (m.RW, m.RH))     # full[c + 1, r + 1] is photosite (r, c)
    raw = permutedims(view(full, (m.left + 1):(m.left + m.W), (m.top + 1):(m.top + m.H)))   # one copy, out of LibRaw's buffer
    return RawMosaic(raw, m.pattern, Float64.(m.black), m.white, (m.top, m.left), (m.RH, m.RW), m.metadata)
end

# The mosaic's layout, checked, without its photosites: LibRaw's raw buffer and frame, the
# visible area, the colour pattern, the black and white levels and the metadata.
function _mosaic_layout(lr::Ptr{Cvoid}, path::AbstractString)
    (; sizes_p, sizes, idata_p, other_p, color_p, lens_p) = _structures(lr)
    rawdata_p = _rawdata_ptr(other_p)
    rsizes = _field(rawdata_p, RawData, :sizes)
    _check("rawdata.sizes", rsizes == sizes)
    filters = _field(idata_p, IParams, :filters)
    _check("rawdata.iparams.filters", _field(rawdata_p, RawData, :iparams).filters == filters)
    rcolor_p = _fieldptr(rawdata_p, RawData, :color)
    _check("rawdata.color.maximum", _field(rcolor_p, ColorData, :maximum) == _field(color_p, ColorData, :maximum))
    _check("rawdata.color.black", _field(rcolor_p, ColorData, :black) == _field(color_p, ColorData, :black))
    raw_image = _field(rawdata_p, RawData, :raw_image)
    raw_image == C_NULL && error("load_raw: not a flat Bayer raw (no single-channel raw_image: a Foveon, linear DNG or sRAW file)")
    _check("rawdata.raw_alloc", Ptr{Cvoid}(raw_image) == _field(rawdata_p, RawData, :raw_alloc))
    filters == 0 && error("load_raw: a monochrome sensor (no colour filter array) is not supported")
    filters < 1000 && error("load_raw: only 2×2 Bayer patterns are supported (LibRaw filters = $filters: X-Trans or Leaf)")
    sizes.raw_pitch == 2 * sizes.raw_width ||
        error("load_raw: raw_pitch $(sizes.raw_pitch) is not 2 × raw_width $(sizes.raw_width)")

    # -- the visible photosites (LibRaw's raw_image from the margins): row-major in C
    RH, RW = Int(sizes.raw_height), Int(sizes.raw_width)
    top, left = Int(sizes.top_margin), Int(sizes.left_margin)
    H, W = Int(sizes.height), Int(sizes.width)
    top + H <= RH && left + W <= RW || error("load_raw: the visible area exceeds the raw frame")

    # -- colours: LibRaw's COLOR relative to the visible area, read from the raw
    #    origin, reduced 4×4 → 2×2 when it repeats; each photosite's index is
    #    the pattern tiled from the origin (photosite_colors)
    n = 4
    pat = UInt8[(@ccall libraw.libraw_COLOR(lr::Ptr{Cvoid}, (y - top)::Cint, (x - left)::Cint)::Cint) for y in 0:(n - 1), x in 0:(n - 1)]
    q = pat[1:2, 1:2]
    pattern = (pat[1:2, 3:4] == q && pat[3:4, 3:4] == q && pat[3:4, 1:2] == q) ? q : pat

    # -- black levels: per channel (LibRaw's adjust_bl on rawdata.color),
    #    refusing a spatial pattern that would be dropped
    cblack = Vector{Int}(undef, CBLACK_SIZE)
    cb_p = Ptr{UInt32}(_fieldptr(rcolor_p, ColorData, :cblack))
    for k in 1:CBLACK_SIZE
        cblack[k] = unsafe_load(cb_p, k)
    end
    black = VisualGeometryToolkitFeatures._per_channel_black(cblack, Int(_field(rcolor_p, ColorData, :black)), filters, path)
    white = Float64(_field(rcolor_p, ColorData, :maximum))
    metadata = _metadata(idata_p, other_p, lens_p, rcolor_p)
    return (; raw_image, RH, RW, top, left, H, W, pattern, black, white, metadata)
end

# The display demosaic of an open, unpacked handle: LibRaw's AHD interpolation
# with linear output — 16 bits, gamma 1 (both coefficients), no auto-brightening,
# no maximum adjustment, the camera's raw colour (output_color 0), user white
# balance 1 — as `(rgb, metadata)`, `rgb[row, column, channel]` over the visible
# area. The sensor orientation is kept: LibRaw's C API has no setter for
# `params.user_flip`, and `dcraw_process` restores `sizes` (EXIF `flip`
# included) from the unpacked copy, so `sizes.flip` is zeroed after the
# processing and before the image is copied out (`dcraw_make_mem_image` rotates
# by it). The mosaic's checks run first (`_mosaic_layout`: the mirrored layout, a flat
# 2×2 Bayer sensor; no copy of its photosites), and the output must be the visible
# area's size (a non-square pixel aspect or a Fuji rotation would resample it: refused).
const _AHD = 3                   # LibRaw's user_qual for AHD
function _demosaiced(lr::Ptr{Cvoid}, path::AbstractString)
    m = _mosaic_layout(lr, path)
    @ccall libraw.libraw_set_output_bps(lr::Ptr{Cvoid}, 16::Cint)::Cvoid
    @ccall libraw.libraw_set_gamma(lr::Ptr{Cvoid}, 0::Cint, 1.0f0::Cfloat)::Cvoid
    @ccall libraw.libraw_set_gamma(lr::Ptr{Cvoid}, 1::Cint, 1.0f0::Cfloat)::Cvoid
    @ccall libraw.libraw_set_no_auto_bright(lr::Ptr{Cvoid}, 1::Cint)::Cvoid
    @ccall libraw.libraw_set_adjust_maximum_thr(lr::Ptr{Cvoid}, 0.0f0::Cfloat)::Cvoid
    @ccall libraw.libraw_set_output_color(lr::Ptr{Cvoid}, 0::Cint)::Cvoid
    @ccall libraw.libraw_set_demosaic(lr::Ptr{Cvoid}, _AHD::Cint)::Cvoid
    for k in 0:3
        @ccall libraw.libraw_set_user_mul(lr::Ptr{Cvoid}, k::Cint, 1.0f0::Cfloat)::Cvoid
    end
    rc = @ccall libraw.libraw_dcraw_process(lr::Ptr{Cvoid})::Cint
    rc == 0 || error("load_raw: LibRaw cannot demosaic $path: $(_strerror(rc))")
    _store_flip!(_sizes_ptr(lr), 0)                   # the live `sizes`: the sensor's orientation
    err = Ref{Cint}(0)
    img = @ccall libraw.libraw_dcraw_make_mem_image(lr::Ptr{Cvoid}, err::Ptr{Cint})::Ptr{ProcessedImageHeader}
    img == C_NULL && error("load_raw: LibRaw cannot copy out the demosaic of $path: $(_strerror(err[]))")
    try
        h = unsafe_load(img)
        (h.colors == 3 && h.bits == 16) || error("load_raw: LibRaw's demosaic of $path has $(h.colors) colours at $(h.bits) bits, not 3 at 16")
        H, W = m.H, m.W
        (Int(h.height), Int(h.width)) == (H, W) ||
            error("load_raw: LibRaw's demosaic of $path is $(h.height)×$(h.width), not the visible area's $(H)×$(W) (a resampled or rotated output)")
        Int(h.data_size) == 2 * 3 * H * W || error("load_raw: LibRaw's demosaic of $path holds $(h.data_size) bytes, not $(6H * W)")
        data = unsafe_wrap(Array, Ptr{UInt16}(img + fieldoffset(ProcessedImageHeader, 7)), (3, W, H))   # row-major, interleaved RGB
        return permutedims(data, (3, 2, 1)), m.metadata                                                  # one copy, out of LibRaw's buffer
    finally
        @ccall libraw.libraw_dcraw_clear_mem(img::Ptr{ProcessedImageHeader})::Cvoid
    end
end

# LibRaw's local-time rendering of the timestamp it parsed from the EXIF
# DateTimeOriginal with mktime: localtime gives the stored fields back.
_strftime(t::Int64) = Base.Libc.strftime("%Y:%m:%d %H:%M:%S", t)

end # module
