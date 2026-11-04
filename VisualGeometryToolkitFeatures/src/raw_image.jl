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
  `exposure_s`, `fnumber`, `focal_mm`, `iso`, `model`; `nothing` where absent),
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
