# =============================================================================
# Camera RAW input — a Bayer mosaic as exact linear values
# =============================================================================
#
# Two layers, kept apart on purpose:
#
#   decode   — a RAW file to its mosaic (`RawMosaic`): the visible photosites,
#              their colour pattern, the per-channel black levels, the white
#              level and the EXIF fields. A decoder is a weak-dependency
#              extension, chosen by a token: `LibRawDecoder()` (LibRaw_jll,
#              ccall, no Python) or `RawpyDecoder()` (PythonCall + rawpy).
#              Both wrap LibRaw; the extension only decodes.
#   process  — every number after that, here, in Julia: the linear scaling,
#              the 2×2 binning or the green photosites, the saturation flags,
#              the true sample positions (`LinearRaw(mosaic; mode, gain)`).
#
# Ported from RadiometricCamera's scripts/py/convert_raw.py (modes bin2 and
# green, 2026-11-04). Its third mode, `fullres` (LibRaw's linear demosaic), is
# not ported: a demosaic interpolates, and the radiometric chain reads the
# binned or green-only samples. Each mode is one KernelAbstractions kernel
# (on the CPU backend of the host mosaic).
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const

"""
    RawDecoder

A RAW decoder, the token [`load_raw`](@ref) and [`load_raw_mosaic`](@ref)
dispatch on: [`LibRawDecoder`](@ref) or [`RawpyDecoder`](@ref). Each is
implemented by a package extension and needs its trigger package loaded.
"""
abstract type RawDecoder end

"""
    LibRawDecoder()

Decode with LibRaw through LibRaw_jll (`using LibRaw_jll`): a `ccall` layer over
LibRaw's C API, no Python. The extension checks LibRaw's version at load and the
layout of every structure it reads against LibRaw's own accessors, and refuses
(`ErrorException`) a library it was not written for.
"""
struct LibRawDecoder <: RawDecoder end

"""
    RawpyDecoder()

Decode with rawpy and exifread through PythonCall (`using PythonCall`; the
Python packages are declared in the package's `CondaPkg.toml`). Python opens the
file once and returns the arrays once; no per-pixel Python call.

rawpy reports only LibRaw's per-channel black levels (`black_level_per_channel`),
not the spatial black pattern behind them, so where [`LibRawDecoder`](@ref)
refuses a file whose spatial pattern has no per-channel form, this decoder
cannot see it and returns the per-channel levels (the pattern dropped). Use
`LibRawDecoder()` where that matters (DNGs with a black-level pattern).
"""
struct RawpyDecoder <: RawDecoder end

_decoder_trigger(::LibRawDecoder) = "LibRaw_jll"
_decoder_trigger(::RawpyDecoder) = "PythonCall"

# The decode hook: an extension adds the method for its decoder. Without the
# extension loaded this fallback names the package to load.
function _decode_raw(decoder::RawDecoder, path::AbstractString)
    pkg = _decoder_trigger(decoder)
    throw(ArgumentError("load_raw: decoding with $(nameof(typeof(decoder)))() needs `using $pkg` (its package extension is not loaded)"))
end

"""
    RawMosaic

A RAW file decoded and nothing more: the visible area of the sensor, as LibRaw
reports it.

- `raw::Matrix{UInt16}` — the visible photosites, `raw[row, column]`, in the
  sensor's orientation (no EXIF flip): pixel `(row, column)` is the raw's
  COLMAP pixel centred at `(column − ½, row − ½)`.
- `pattern::Matrix{UInt8}` — the smallest repeating colour pattern, square,
  at most 4 × 4, from the full raw frame's origin (rawpy's `raw_pattern`), in
  LibRaw's colour indices: `0` R, `1` G, `2` B, `3` the second G of an RGBG
  pattern. Each photosite's index is [`photosite_colors`](@ref)`(mosaic)`,
  the pattern tiled from that origin.
- `black::NTuple{4, Float64}` — the black level per colour index (the common
  and per-channel parts summed, LibRaw's `adjust_bl`).
- `white::Float64` — the white level (LibRaw's `maximum`): the code at which a
  photosite is saturated.
- `visible_origin::NTuple{2, Int}` — `(top_margin, left_margin)`, the visible
  area's offset into the full raw frame, which is `raw_size` (`(height, width)`).
- `metadata::NamedTuple` — the EXIF fields (`datetime_original`, `subsec`,
  `exposure_s`, `fnumber`, `focal_mm`, `iso`, `model`, `make`, `body_serial`,
  `lens_make`, `lens_model`, `lens_serial`, `focus_distance_m`; `nothing` where
  absent, a non-finite or non-positive focus distance too; typed by
  [`exposure_settings`](@ref), [`lens_settings`](@ref), [`body_identity`](@ref),
  [`lens_identity`](@ref) and [`capture_time`](@ref)),
  `daylight_wb` (LibRaw's daylight white balance, R G B), `linear_max` (the
  camera's declared linear-response limit per channel, `nothing` when the file
  has none), `decoder` and `libraw_version`.
"""
struct RawMosaic{M <: NamedTuple}
    raw::Matrix{UInt16}
    pattern::Matrix{UInt8}
    black::NTuple{4, Float64}
    white::Float64
    visible_origin::NTuple{2, Int}
    raw_size::NTuple{2, Int}
    metadata::M
    function RawMosaic(raw, pattern, black, white, visible_origin, raw_size, metadata::M) where {M <: NamedTuple}
        n = size(pattern, 1)
        (size(pattern, 2) == n && 1 <= n <= 4) ||
            throw(ArgumentError("RawMosaic: the colour pattern must be square, at most 4 × 4, got $(join(size(pattern), '×'))"))
        all(c -> c <= 3, pattern) || throw(ArgumentError("RawMosaic: colour indices are 0 (R), 1 (G), 2 (B) or 3 (G2), got $pattern"))
        all(<(white), black) || throw(ArgumentError("RawMosaic: every black level must be below the white level $white, got $black"))
        return new{M}(raw, pattern, black, white, visible_origin, raw_size, metadata)
    end
end

"""
    photosite_colors(mosaic::RawMosaic) -> Matrix{UInt8}

Each visible photosite's colour index (LibRaw's: `0` R, `1` G, `2` B, `3` the
second G), the size of `mosaic.raw`: the `pattern` tiled from the full raw
frame's origin, so visible photosite `(row, column)` reads
`pattern[mod(top + row − 1, n) + 1, mod(left + column − 1, n) + 1]` with
`(top, left) = visible_origin` — rawpy's `raw_colors_visible`.
"""
photosite_colors(m::RawMosaic) = UInt8[_photosite_color(m, i, j) for i in axes(m.raw, 1), j in axes(m.raw, 2)]

# The colour index of visible photosite (i, j): the pattern tiled from the raw origin.
@inline function _photosite_color(m::RawMosaic, i, j)
    n = size(m.pattern, 1)
    top, left = m.visible_origin
    return @inbounds m.pattern[mod(top + i - 1, n) + 1, mod(left + j - 1, n) + 1]
end

"""
    cfa_string(mosaic::RawMosaic) -> String

The colour pattern as letters, row by row (`"RGGB"` for `[0 1; 3 2]`): LibRaw's
indices read through `"RGBG"`, as convert_raw.py records it.
"""
cfa_string(mosaic::RawMosaic) = join(("RGBG"[Int(c) + 1] for c in permutedims(mosaic.pattern)))

"""
    LinearRaw

A RAW mosaic as linear values: `(raw − black[c]) / (white − black[c]) × gain`,
camera raw colour, white balance 1, `1.0` = the white level (before the gain).
Not clipped: a value above 1 (a gain above 1) or below 0 (noise under the black
level) is kept.

- `image::Array{Float64, 3}` — `image[row, column, plane]`.
- `sampled::BitMatrix` — the pixels that carry a sample (every pixel in
  `:bin2`; the green photosites in `:green`).
- `saturated::BitMatrix` — the pixels whose value includes a photosite at the
  white level (`raw ≥ white`): in `:bin2` a quad is saturated when ANY of its
  four photosites is, so a quad whose two greens straddle the white level is
  flagged rather than averaged into a plausible value.
- `plane_offsets::Vector{SVector{2, Float64}}` — per plane, where its sample
  sits relative to the pixel centre, `(Δx, Δy)` in this image's pixels: the
  sample of pixel `(row, column)`, plane `k`, is at the COLMAP point
  `(column − ½, row − ½) + plane_offsets[k]`.
- `mode`, `gain`, and the mosaic's `metadata`.
"""
struct LinearRaw{M <: NamedTuple}
    image::Array{Float64, 3}
    sampled::BitMatrix
    saturated::BitMatrix
    plane_offsets::Vector{SVector{2, Float64}}
    mode::Symbol
    gain::Float64
    metadata::M
end

const RAW_MODES = (:bin2, :green)

"""
    LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1) -> LinearRaw

The mosaic's linear values, `(raw − black[c]) / (white − black[c]) × gain`,
arranged by `mode` (required: there is no neutral choice between them):

- `:bin2` — 2×2 Bayer quads, `H ÷ 2 × W ÷ 2 × 3` (an odd last row or column is
  dropped): R, the mean of the two greens, B. A COLMAP coordinate of the raw
  scales by exactly ½. The green sample sits at the quad centre (offset
  `(0, 0)`: the two greens are diagonal); R and B sit at their photosites,
  `(±¼, ±¼)` binned pixels from it (`plane_offsets`).
- `:green` — the green photosites alone, `H × W × 1` on the raw's own grid,
  `0` at the red and blue photosites (`sampled` says which), offset `(0, 0)`.

The values are those convert_raw.py writes, in the same Float64 arithmetic and
order, before its clip to `[0, 1]` and 16-bit rounding.

Saturation is `raw ≥ white` with no margin: on the Sony ARW sets this was
written for, codes pile up exactly at the white level (16383) while the
histogram runs smoothly through the camera's declared linear limit (15360,
`metadata.linear_max`), so any margin would be a tuned constant. A caller who
wants that stricter limit thresholds `mosaic.raw` itself.
"""
function LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1)
    mode in RAW_MODES || throw(ArgumentError("LinearRaw: mode must be one of $(RAW_MODES), got :$mode (the full-resolution demosaic is not provided)"))
    isfinite(gain) && gain > 0 || throw(ArgumentError("LinearRaw: the gain must be positive and finite, got $gain"))
    g = Float64(gain)
    return mode === :bin2 ? _bin2(mosaic, g) : _green_sites(mosaic, g)
end

# (raw − black[c]) / (white − black[c]): convert_raw.py's expression, in its
# order (the gain is applied after the binning's mean, as there), with the
# black level and the denominator `white − black[c]` taken from per-colour
# (or per-site) constants computed once — the same two operations on the same
# values, so the same bits.
@inline _linear(x::UInt16, black::Float64, den::Float64) = (Float64(x) - black) / den

# A quad's four photosites in column-major order, k = di + 2dj + 1.
const _QUAD_SITES = ((0, 0), (1, 0), (0, 1), (1, 1))

# Each quad's R, G, B: the planes' photosites (quad-site indices k), their
# linear values' mean times the gain, and whether ANY photosite is at the
# white level. One work item per quad.
@kernel function _bin2_kernel!(img, sat, @Const(raw), planes, black, den, gain::Float64, white::Float64)
    i, j = @index(Global, NTuple)
    r0, c0 = 2i - 1, 2j - 1
    @inbounds begin
        x = ntuple(k -> raw[r0 + _QUAD_SITES[k][1], c0 + _QUAD_SITES[k][2]], Val(4))
        v = ntuple(k -> _linear(x[k], black[k], den[k]), Val(4))
        for p in 1:3
            ks = planes[p]
            img[i, j, p] = (foldl((s, k) -> s + v[k], ks; init = 0.0) / length(ks)) * gain
        end
        sat[i, j] = x[1] >= white || x[2] >= white || x[3] >= white || x[4] >= white
    end
end

function _bin2(m::RawMosaic, gain::Float64)
    H, W = size(m.raw)
    H2, W2 = H ÷ 2, W ÷ 2
    H2 >= 1 && W2 >= 1 || throw(ArgumentError("LinearRaw: a $(H)×$(W) mosaic holds no 2×2 quad"))
    # the quad's colour layout, the same in every quad of a 2×2 Bayer pattern
    # (a pattern that repeats with period 2 from the raw origin)
    n = size(m.pattern, 1)
    (iseven(n) && all(m.pattern[i, j] == m.pattern[mod1(i, 2), mod1(j, 2)] for i in 1:n, j in 1:n)) ||
        throw(ArgumentError("LinearRaw: :bin2 needs a 2×2 Bayer pattern, got the colour pattern $(m.pattern)"))
    cell = ntuple(k -> _photosite_color(m, 1 + _QUAD_SITES[k][1], 1 + _QUAD_SITES[k][2]), Val(4))
    plane(c) = c == 3 ? 2 : Int(c) + 1                         # 0 R, 1 G, 2 B, 3 G2 → planes R, G, B
    planes = ntuple(p -> Tuple(k for k in 1:4 if plane(cell[k]) == p), Val(3))
    map(length, planes) == (1, 2, 1) ||
        throw(ArgumentError("LinearRaw: :bin2 needs one R, two G and one B per quad, got $([cell[1] cell[3]; cell[2] cell[4]])"))
    black = ntuple(k -> m.black[cell[k] + 1], Val(4))
    den = ntuple(k -> m.white - black[k], Val(4))
    img = Array{Float64, 3}(undef, H2, W2, 3)
    sat = Matrix{Bool}(undef, H2, W2)
    planes_static = (planes[1]::NTuple{1, Int}, planes[2]::NTuple{2, Int}, planes[3]::NTuple{1, Int})
    backend = KA.get_backend(img)
    _bin2_kernel!(backend)(img, sat, m.raw, planes_static, black, den, gain, m.white; ndrange = (H2, W2))
    KA.synchronize(backend)
    # photosite (di, dj) of a quad is at raw COLMAP (2j − 1.5 + dj, 2i − 1.5 + di),
    # the binned point (j − ¾ + dj/2, i − ¾ + di/2): ((dj − ½)/2, (di − ½)/2) from the centre
    offset(k) = SVector((_QUAD_SITES[k][2] - 0.5) / 2, (_QUAD_SITES[k][1] - 0.5) / 2)
    offsets = [sum(offset, planes[p]) / length(planes[p]) for p in 1:3]
    return LinearRaw(img, trues(H2, W2), BitMatrix(sat), offsets, :bin2, gain, m.metadata)
end

# Each photosite's green value (0 at a red or a blue one), whether it is
# green, and whether it is at the white level. One work item per photosite.
@kernel function _green_kernel!(img, green, sat, @Const(raw), @Const(pattern), origin, black, den, gain::Float64, white::Float64)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = size(pattern, 1)
        c = pattern[mod(origin[1] + i - 1, n) + 1, mod(origin[2] + j - 1, n) + 1]
        g = c == 0x01 || c == 0x03
        x = raw[i, j]
        green[i, j] = g
        img[i, j, 1] = g ? _linear(x, black[c + 1], den[c + 1]) * gain : 0.0
        sat[i, j] = g && x >= white
    end
end

function _green_sites(m::RawMosaic, gain::Float64)
    H, W = size(m.raw)
    img = Array{Float64, 3}(undef, H, W, 1)
    green = Matrix{Bool}(undef, H, W)
    sat = Matrix{Bool}(undef, H, W)
    den = map(b -> m.white - b, m.black)
    backend = KA.get_backend(img)
    _green_kernel!(backend)(img, green, sat, m.raw, m.pattern, m.visible_origin, m.black, den, gain, m.white; ndrange = (H, W))
    KA.synchronize(backend)
    return LinearRaw(img, BitMatrix(green), BitMatrix(sat), [SVector(0.0, 0.0)], :green, gain, m.metadata)
end

# LibRaw::adjust_bl as rawpy's data_helper.h copies it, the black levels per
# colour index from LibRaw's raw `cblack` array (`cb[k + 1]` is `cblack[k]`:
# four per-channel parts, then a spatial pattern of `cblack[4] × cblack[5]`
# entries at `cblack[6+]`) and its common `black`, folding a 2×2 pattern into
# the four channels. A spatial pattern that remains nonzero after the fold —
# larger than 2×2, or one LibRaw's fold leaves — has no per-channel form: it is
# refused, naming the file and the pattern's size, never dropped (rawpy's
# black_level_per_channel drops it silently). The one rule every decoder that
# can read `cblack` goes through.
function _per_channel_black(cb::AbstractVector{<:Integer}, black::Integer, filters::Integer, path::AbstractString)
    C = Vector{Int}(cb)                            # C[k + 1] is cblack[k]
    black = Int(black)
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

"""
    load_raw_mosaic(path; decoder::RawDecoder = LibRawDecoder()) -> RawMosaic

Decode a camera RAW file (any format LibRaw reads: ARW, CR2, NEF, DNG, …) to
its [`RawMosaic`](@ref). Only flat Bayer sensors (one colour per photosite) are
supported. The decoder's package must be loaded: `using LibRaw_jll` for
[`LibRawDecoder`](@ref), `using PythonCall` for [`RawpyDecoder`](@ref).
"""
function load_raw_mosaic(path::AbstractString; decoder::RawDecoder = LibRawDecoder())
    isfile(path) || throw(ArgumentError("load_raw_mosaic: no file at $path"))
    return _decode_raw(decoder, String(path))::RawMosaic
end

"""
    load_raw(path; mode::Symbol, gain::Real = 1, decoder::RawDecoder = LibRawDecoder()) -> LinearRaw

A camera RAW file as exact linear values: [`load_raw_mosaic`](@ref), then
[`LinearRaw`](@ref)`(mosaic; mode, gain)`. `mode` is `:bin2` (2×2 Bayer quads)
or `:green` (the green photosites on the raw grid). The pixels stay in the
sensor's orientation (no EXIF flip), in the raw's COLMAP frame (scaled by ½ in
`:bin2`).

Rounded to 16 bits (`round(UInt16, clamp(x, 0, 1) * 65535)`), `:bin2` equals
the linear PNG of RadiometricCamera's convert_raw.py (`--mode bin2`), value for
value; unlike that PNG, `saturated` flags the quads that contain a photosite at
the white level.
"""
load_raw(path::AbstractString; mode::Symbol, gain::Real = 1, decoder::RawDecoder = LibRawDecoder()) =
    LinearRaw(load_raw_mosaic(path; decoder); mode, gain)

# -----------------------------------------------------------------------------
# The EXIF fields as Core's typed values: settings and identities
# -----------------------------------------------------------------------------
#
# `metadata` keeps the decoder's fields as it read them; the accessors below
# are the one place they become Core's ExposureSettings, LensSettings and
# DeviceIdentity values, so no fact is held twice.

# The metadata-only hook: a decoder whose library can read the headers without
# the photosites adds a method (LibRaw); otherwise the file is decoded.
_decode_metadata(decoder::RawDecoder, path::AbstractString) = _decode_raw(decoder, path).metadata

"""
    load_raw_metadata(path; decoder::RawDecoder = LibRawDecoder()) -> NamedTuple

The EXIF fields of a camera RAW file, as [`RawMosaic`](@ref)'s `metadata`
holds them, without keeping the photosites: [`LibRawDecoder`](@ref) reads the
headers only (no unpack), [`RawpyDecoder`](@ref) decodes the file. The typed
accessors ([`exposure_settings`](@ref), …) take it as they take a mosaic.
"""
function load_raw_metadata(path::AbstractString; decoder::RawDecoder = LibRawDecoder())
    isfile(path) || throw(ArgumentError("load_raw_metadata: no file at $path"))
    return _decode_metadata(decoder, String(path))::NamedTuple
end

# A focus distance a file reports, in metres: a marker (a lens at "infinity",
# an unset 0, a negative or NaN value, EXIF SubjectDistance's 0xFFFFFFFF) is
# no distance and is `nothing`. The measured Sony ARWs (an FX3, an adapted
# manual-focus lens) report infinity.
const _EXIF_INFINITY = 4294967295   # SubjectDistance's "infinity" (0xFFFFFFFF/1)
_focus_distance(::Nothing) = nothing
_focus_distance(d::Real) = (isfinite(d) && 0 < d < _EXIF_INFINITY) ? Float64(d) : nothing

_raw_metadata(x::RawMosaic) = x.metadata
_raw_metadata(x::NamedTuple) = x
function _stated(md::NamedTuple, k::Symbol, what::AbstractString)
    v = get(md, k, nothing)
    v === nothing && throw(ArgumentError("$(what): the RAW's metadata states no `$(k)` (its EXIF has none)"))
    return v
end
_text(::Nothing) = nothing
_text(s::AbstractString) = (t = strip(s); isempty(t) ? nothing : String(t))

"""
    exposure_settings(x) -> ExposureSettings

The exposure a RAW frame was taken with — `x` a [`RawMosaic`](@ref) or its
`metadata` ([`load_raw_metadata`](@ref)) — as Core's `ExposureSettings`:
`exposure_s` seconds, `fnumber`, `iso`. A field the file does not state is an
`ArgumentError` naming it.
"""
exposure_settings(x::Union{RawMosaic, NamedTuple}) = _exposure_settings(_raw_metadata(x), parse_unit("s"))
# `sec`, the second, parsed once by the caller: Core keeps no public time
# constant, and `parse_unit` cannot run while a package precompiles (FlexUnits'
# parse evaluates into its registry module), so it is parsed at call time.
_exposure_settings(md::NamedTuple, sec) = ExposureSettings(
    _stated(md, :exposure_s, "exposure_settings") * sec,
    _stated(md, :fnumber, "exposure_settings"), _stated(md, :iso, "exposure_settings"),
)

"""
    lens_settings(x) -> LensSettings

The lens settings a RAW frame was taken with, as Core's `LensSettings`: the
focal length the lens reports (`focal_mm`, the NOMINAL value, not a
calibration) and the focus distance where the file states a finite, positive
one (`nothing` otherwise — the FX3's adapted lenses report infinity). A file
without a focal length is an `ArgumentError`.
"""
function lens_settings(x::Union{RawMosaic, NamedTuple})
    md = _raw_metadata(x)
    d = _focus_distance(get(md, :focus_distance_m, nothing))
    return LensSettings(_stated(md, :focal_mm, "lens_settings") * mm, d === nothing ? nothing : d * m)
end

"""
    body_identity(x) -> DeviceIdentity

The camera body a RAW file says took it (`make`, `model`, `body_serial`), as
Core's `DeviceIdentity`; a field the file does not state is `nothing`.
"""
function body_identity(x::Union{RawMosaic, NamedTuple})
    md = _raw_metadata(x)
    return DeviceIdentity(_text(get(md, :make, nothing)), _text(get(md, :model, nothing)), _text(get(md, :body_serial, nothing)))
end

"""
    lens_identity(x) -> DeviceIdentity

The lens a RAW file CLAIMS took it (`lens_make`, `lens_model`,
`lens_serial`). Kept, not trusted: a lens on an adapter is named by what the
adapter reports (the FX3 with Sigma EF lenses claims "DT 8mm F4 SAM"), so a
pipeline that knows its lens states it ([`extract_capture_settings`](@ref)'s `lens`).
"""
function lens_identity(x::Union{RawMosaic, NamedTuple})
    md = _raw_metadata(x)
    return DeviceIdentity(_text(get(md, :lens_make, nothing)), _text(get(md, :lens_model, nothing)), _text(get(md, :lens_serial, nothing)))
end

"""
    capture_time(x) -> Union{String, Nothing}

When a RAW frame was taken, as the file states it: EXIF's
DateTimeOriginal (local time, no zone) with its SubSecTimeOriginal, written
ISO 8601 (`"2026-10-29T14:03:12.34"`); `nothing` when the file states none.
"""
function capture_time(x::Union{RawMosaic, NamedTuple})
    md = _raw_metadata(x)
    dt = get(md, :datetime_original, nothing)
    (dt === nothing || !occursin(r"\d", dt)) && return nothing   # EXIF's unknown time: blanks, the colons kept
    mt = Base.match(r"^(\d{4}):(\d{2}):(\d{2}) (\d{2}:\d{2}:\d{2})$", strip(dt))
    mt === nothing && throw(ArgumentError("capture_time: DateTimeOriginal $(repr(dt)) is not EXIF's YYYY:MM:DD HH:MM:SS"))
    ss = strip(something(get(md, :subsec, ""), ""))
    return "$(mt[1])-$(mt[2])-$(mt[3])T$(mt[4])" * (isempty(ss) ? "" : ".$(ss)")
end

# Two identities agree unless a field both state differs (case aside: LibRaw
# writes a maker as "Sony" where EXIF has "SONY").
_fields(d::DeviceIdentity) = (d.make, d.model, d.serial)
_agrees(a::DeviceIdentity, b::DeviceIdentity) =
    all(((x, y),) -> x === nothing || y === nothing || lowercase(x) == lowercase(y), zip(_fields(a), _fields(b)))
# Identities that agree pairwise, as one: each field the first that states it.
_first_stated(xs) = (i = findfirst(!isnothing, xs); i === nothing ? nothing : xs[i])
_merged(ids) = DeviceIdentity((_first_stated(xs) for xs in zip(map(_fields, ids)...))...)
function _one_identity(ids, what)
    all(a -> all(b -> _agrees(a, b), ids), ids) ||
        throw(ArgumentError("extract_capture_settings: the frames name $(length(unique(ids))) $(what)s, $(unique(ids)); a capture has one"))
    return _merged(ids)
end

"""
    extract_capture_settings(frames; lens = nothing) -> CaptureSettings

The camera settings of a capture from its RAW files' EXIF: `frames` maps a
frame index to a [`RawMosaic`](@ref) or its `metadata`
([`load_raw_metadata`](@ref)); each frame's settings are
[`exposure_settings`](@ref) and [`lens_settings`](@ref). The body is the
frames' [`body_identity`](@ref), which must be one (identities agree when no
field both state differs, case aside; `ArgumentError` otherwise). The lens
is `lens` where the caller states it — a config knows an adapted lens EXIF
misnames — and the frames' one EXIF claim otherwise ([`lens_identity`](@ref),
which must then agree across the frames); a stated lens that disagrees with
EXIF is used, with a warning naming both. EXIF's claim stays readable from
the metadata.
"""
function extract_capture_settings(frames::AbstractDict{<:Integer}; lens::Union{Nothing, DeviceIdentity} = nothing)
    isempty(frames) && throw(ArgumentError("extract_capture_settings: no frames"))
    mds = Dict{Int, NamedTuple}(Int(t) => _raw_metadata(x) for (t, x) in frames)
    ts = sort!(collect(keys(mds)))                      # frame order: the earliest frame stating a field gives it
    body = _one_identity([body_identity(mds[t]) for t in ts], "camera body")
    claims = unique(lens_identity(mds[t]) for t in ts)
    if lens === nothing
        lens = _one_identity(claims, "lens")
    elseif !all(c -> _agrees(lens, c), claims)
        @warn "extract_capture_settings: the stated lens overrides what the RAW files' EXIF claims" stated = lens exif = claims
    end
    sec = parse_unit("s")
    return CaptureSettings(
        Dict(t => (exposure = _exposure_settings(md, sec), lens_settings = lens_settings(md)) for (t, md) in mds); body, lens,
    )
end
