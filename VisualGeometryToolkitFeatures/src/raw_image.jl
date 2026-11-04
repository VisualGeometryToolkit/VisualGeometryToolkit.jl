# =============================================================================
# Camera RAW input — a Bayer mosaic as exact linear values
# =============================================================================
#
# Two layers, kept apart on purpose:
#
#   decode   — a RAW file to its mosaic (`RawMosaic`): the visible photosites,
#              their colour indices, the per-channel black levels, the white
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
# binned or green-only samples.
# =============================================================================

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
file and returns the arrays once; no per-pixel Python call.
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
- `colors::Matrix{UInt8}` — each photosite's colour index, LibRaw's: `0` R,
  `1` G, `2` B, `3` the second G of an RGBG pattern.
- `pattern::Matrix{UInt8}` — the smallest repeating colour pattern, from the
  raw image's origin (rawpy's `raw_pattern`).
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
    colors::Matrix{UInt8}
    pattern::Matrix{UInt8}
    black::NTuple{4, Float64}
    white::Float64
    visible_origin::NTuple{2, Int}
    raw_size::NTuple{2, Int}
    metadata::M
    function RawMosaic(raw, colors, pattern, black, white, visible_origin, raw_size, metadata::M) where {M <: NamedTuple}
        size(raw) == size(colors) ||
            throw(DimensionMismatch("RawMosaic: a $(join(size(raw), '×')) raw image and $(join(size(colors), '×')) colour indices"))
        all(c -> c <= 3, colors) || throw(ArgumentError("RawMosaic: colour indices are 0 (R), 1 (G), 2 (B) or 3 (G2)"))
        all(<(white), black) || throw(ArgumentError("RawMosaic: every black level must be below the white level $white, got $black"))
        return new{M}(raw, colors, pattern, black, white, visible_origin, raw_size, metadata)
    end
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
# order (the gain is applied after the binning's mean, as there).
@inline _linear(m::RawMosaic, i, j) =
    (Float64(m.raw[i, j]) - m.black[m.colors[i, j] + 1]) / (m.white - m.black[m.colors[i, j] + 1])

function _bin2(m::RawMosaic, gain::Float64)
    H, W = size(m.raw)
    H2, W2 = H ÷ 2, W ÷ 2
    H2 >= 1 && W2 >= 1 || throw(ArgumentError("LinearRaw: a $(H)×$(W) mosaic holds no 2×2 quad"))
    # the quad's colour layout, the same in every quad of a 2×2 Bayer pattern
    cell = m.colors[1:2, 1:2]
    all(m.colors[i, j] == cell[mod1(i, 2), mod1(j, 2)] for i in 1:min(H, 4), j in 1:min(W, 4)) ||
        throw(ArgumentError("LinearRaw: :bin2 needs a 2×2 Bayer pattern, got colour indices $(m.colors[1:min(H, 4), 1:min(W, 4)])"))
    plane(c) = c == 3 ? 2 : Int(c) + 1                         # 0 R, 1 G, 2 B, 3 G2 → planes R, G, B
    sites = [[(di, dj) for di in 0:1, dj in 0:1 if plane(cell[di + 1, dj + 1]) == p] for p in 1:3]
    all(length(sites[p]) == n for (p, n) in ((1, 1), (2, 2), (3, 1))) ||
        throw(ArgumentError("LinearRaw: :bin2 needs one R, two G and one B per quad, got $(cell)"))
    img = Array{Float64, 3}(undef, H2, W2, 3)
    sat = falses(H2, W2)
    white = m.white
    @inbounds for j in 1:W2, i in 1:H2
        r0, c0 = 2i - 1, 2j - 1
        for p in 1:3
            s = 0.0
            for (di, dj) in sites[p]
                s += _linear(m, r0 + di, c0 + dj)
            end
            img[i, j, p] = (s / length(sites[p])) * gain
        end
        sat[i, j] = m.raw[r0, c0] >= white || m.raw[r0 + 1, c0] >= white ||
            m.raw[r0, c0 + 1] >= white || m.raw[r0 + 1, c0 + 1] >= white
    end
    # photosite (di, dj) of a quad is at raw COLMAP (2j − 1.5 + dj, 2i − 1.5 + di),
    # the binned point (j − ¾ + dj/2, i − ¾ + di/2): ((dj − ½)/2, (di − ½)/2) from the centre
    offsets = [sum(SVector((dj - 0.5) / 2, (di - 0.5) / 2) for (di, dj) in sites[p]) / length(sites[p]) for p in 1:3]
    return LinearRaw(img, trues(H2, W2), sat, offsets, :bin2, gain, m.metadata)
end

function _green_sites(m::RawMosaic, gain::Float64)
    H, W = size(m.raw)
    img = zeros(Float64, H, W, 1)
    green = falses(H, W)
    sat = falses(H, W)
    white = m.white
    @inbounds for j in 1:W, i in 1:H
        c = m.colors[i, j]
        (c == 1 || c == 3) || continue
        green[i, j] = true
        img[i, j, 1] = _linear(m, i, j) * gain
        sat[i, j] = m.raw[i, j] >= white
    end
    return LinearRaw(img, green, sat, [SVector(0.0, 0.0)], :green, gain, m.metadata)
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
