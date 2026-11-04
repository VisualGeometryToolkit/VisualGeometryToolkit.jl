# =============================================================================
# LibRaw_jll backend of load_raw — a ccall layer over LibRaw's C API
# =============================================================================
#
# Decodes only: open, unpack, copy the visible photosites and the levels out,
# close. All processing is in src/raw_image.jl.
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

using LibRaw_jll: libraw
using VisualGeometryToolkitFeatures: VisualGeometryToolkitFeatures, LibRawDecoder, RawMosaic

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

function __init__()
    v = libraw_version_number()
    (v >> 8) == 0x0016 || error("VisualGeometryToolkitFeaturesLibRawExt: its libraw_data_t layout mirrors LibRaw 0.22; the loaded LibRaw_jll is $(libraw_version()). Re-mirror the structures from that version's libraw_types.h.")
    return nothing
end

_check(what, ok::Bool) = ok || error("load_raw (LibRaw_jll $(libraw_version())): the mirrored libraw_data_t layout disagrees with LibRaw's accessors ($what); refusing to read by offset")

function _strerror(code)
    return unsafe_string(@ccall libraw.libraw_strerror(code::Cint)::Cstring)
end

function VisualGeometryToolkitFeatures._decode_raw(::LibRawDecoder, path::AbstractString)
    lr = @ccall libraw.libraw_init(0::Cuint)::Ptr{Cvoid}
    lr == C_NULL && error("load_raw: libraw_init failed")
    try
        rc = @ccall libraw.libraw_open_file(lr::Ptr{Cvoid}, path::Cstring)::Cint
        rc == 0 || error("load_raw: LibRaw cannot open $path: $(_strerror(rc))")
        rc = @ccall libraw.libraw_unpack(lr::Ptr{Cvoid})::Cint
        rc == 0 || error("load_raw: LibRaw cannot unpack $path: $(_strerror(rc))")
        return _mosaic(lr, path)
    finally
        @ccall libraw.libraw_close(lr::Ptr{Cvoid})::Cvoid
    end
end

function _mosaic(lr::Ptr{Cvoid}, path::AbstractString)
    # -- locate the structures and check the mirror against the accessors
    sizes_p = lr + sizeof(Ptr{Cvoid})
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
    rawdata_p = other_p + sizeof(ImgOther) + sizeof(Thumbnail) + sizeof(ThumbnailList)
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

    # -- the visible photosites (rawpy's raw_image_visible): row-major in C
    RH, RW = Int(sizes.raw_height), Int(sizes.raw_width)
    top, left = Int(sizes.top_margin), Int(sizes.left_margin)
    H, W = Int(sizes.height), Int(sizes.width)
    top + H <= RH && left + W <= RW || error("load_raw: the visible area exceeds the raw frame")
    full = unsafe_wrap(Matrix{UInt16}, raw_image, (RW, RH))     # full[c + 1, r + 1] is photosite (r, c)
    raw = permutedims(full[(left + 1):(left + W), (top + 1):(top + H)])

    # -- colours: rawpy's raw_pattern (COLOR relative to the visible area, from
    #    the raw origin), reduced 4×4 → 2×2 when it repeats, tiled from the origin
    n = 4
    pat = UInt8[(@ccall libraw.libraw_COLOR(lr::Ptr{Cvoid}, (y - top)::Cint, (x - left)::Cint)::Cint) for y in 0:(n - 1), x in 0:(n - 1)]
    q = pat[1:2, 1:2]
    pattern = (pat[1:2, 3:4] == q && pat[3:4, 3:4] == q && pat[3:4, 1:2] == q) ? q : pat
    np_ = size(pattern, 1)
    colors = UInt8[pattern[mod(top + i - 1, np_) + 1, mod(left + j - 1, np_) + 1] for i in 1:H, j in 1:W]

    # -- black levels: rawpy's black_level_per_channel (LibRaw's adjust_bl on rawdata.color)
    cblack = Vector{Int}(undef, CBLACK_SIZE)
    cb_p = Ptr{UInt32}(_fieldptr(rcolor_p, ColorData, :cblack))
    for k in 1:CBLACK_SIZE
        cblack[k] = unsafe_load(cb_p, k)
    end
    black = _adjust_bl(cblack, Int(_field(rcolor_p, ColorData, :black)), filters, path)
    white = Float64(_field(rcolor_p, ColorData, :maximum))
    lin = _field(rcolor_p, ColorData, :linear_max)
    rpre = _field(rcolor_p, ColorData, :pre_mul)

    # -- EXIF, as LibRaw parsed it
    other = unsafe_load(Ptr{ImgOther}(other_p))
    model = _cstring(_field(idata_p, IParams, :model))
    metadata = (;
        datetime_original = other.timestamp == 0 ? nothing : _strftime(other.timestamp),
        subsec = "",                                  # LibRaw keeps no SubSecTimeOriginal
        exposure_s = other.shutter > 0 ? Float64(other.shutter) : nothing,
        fnumber = other.aperture > 0 ? Float64(other.aperture) : nothing,
        focal_mm = other.focal_len > 0 ? Float64(other.focal_len) : nothing,
        iso = other.iso_speed > 0 ? round(Int, other.iso_speed) : nothing,
        model = isempty(model) ? nothing : model,
        daylight_wb = (Float64(rpre[1]), Float64(rpre[2]), Float64(rpre[3])),
        linear_max = all(>(0), lin) ? Float64.(lin) : nothing,
        decoder = "LibRaw_jll",
        libraw_version = libraw_version(),
    )
    return RawMosaic(raw, colors, pattern, Float64.(black), white, (top, left), (RH, RW), metadata)
end

# LibRaw's local-time rendering of the timestamp it parsed from the EXIF
# DateTimeOriginal with mktime: localtime gives the stored fields back.
_strftime(t::Int64) = Base.Libc.strftime("%Y:%m:%d %H:%M:%S", t)

# LibRaw::adjust_bl as rawpy's data_helper.h copies it: the per-channel black
# levels cblack[0:3] + black, folding a 2×2 cblack[6+] block into them. A
# spatial black pattern (cblack[4] × cblack[5] at cblack[6+]) that remains
# nonzero after the fold — larger than 2×2, or one LibRaw's fold leaves — has
# no per-channel form: it is refused, naming the file and the pattern's size,
# never dropped (rawpy's black_level_per_channel drops it silently).
function _adjust_bl(cb::Vector{Int}, black::Int, filters::UInt32, path::AbstractString)
    C = copy(cb)                                   # C[k + 1] is cblack[k]
    FC(row, col) = Int((filters >> ((((row << 1) & 14) | (col & 1)) << 1)) & 3)
    if filters > 1000 && (C[5] + 1) ÷ 2 == 1 && (C[6] + 1) ÷ 2 == 1
        clrs = [FC(c ÷ 2, c % 2) for c in 0:3]
        greens = findall(==(1), clrs)
        length(greens) > 1 && (clrs[last(greens)] = 3)
        for c in 0:3
            C[clrs[c + 1] + 1] += C[6 + (c ÷ 2) % C[5] * C[6] + c % 2 % C[6] + 1]
        end
        C[5] = C[6] = 0
    elseif filters <= 1000 && C[5] == 1 && C[6] == 1
        for c in 0:3
            C[c + 1] += C[7]
        end
        C[5] = C[6] = 0
    end
    i = min(C[4], C[1], C[2], C[3])
    for c in 1:4
        C[c] -= i
    end
    black += i
    if C[5] != 0 && C[6] != 0
        nb = C[5] * C[6]
        i = minimum(C[7:(6 + nb)])
        nonz = 0
        for c in 1:nb
            C[6 + c] -= i
            C[6 + c] != 0 && (nonz += 1)
        end
        black += i
        nonz == 0 && (C[5] = C[6] = 0)
    end
    C[5] != 0 && C[6] != 0 && error(
        "load_raw: $path has a $(C[5]) × $(C[6]) spatial black-level pattern (LibRaw cblack[4] × cblack[5]) that " *
            "per-channel black levels cannot hold; such a file is not supported (its pattern would be dropped)"
    )
    return ntuple(c -> C[c] + black, 4)
end

end # module
