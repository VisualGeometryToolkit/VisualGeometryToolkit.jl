# =============================================================================
# Camera RAW input — the decoder and the display demosaic
# =============================================================================
#
# THE one RAW path of the stack: every consumer reads RAW files through this
# module, and `VisualGeometryToolkitCore.check_raw_decoding` flags any other
# decoding path. Two layers, kept apart on purpose:
#
#   decode   — a RAW file to its mosaic (VisualGeometryToolkitCore's `RawMosaic`):
#              the visible photosites, their colour pattern, the per-channel
#              black levels, the white level and the EXIF fields. The decoder
#              is the weak-dependency extension on LibRaw_jll (`LibRawDecoder()`,
#              ccall, no Python); it only decodes, and for the display demosaic
#              only runs LibRaw's own interpolation (of a file, or of a mosaic
#              in memory).
#   process  — every number after that, in VisualGeometryToolkitCore (moved
#              there in Core 0.9.2 / Features 0.8.0): the colour phase, the
#              photosite views (`channelview`), the linear scaling, the 2×2 Bayer
#              cells (`:bin2`, `:green_cells`), the saturation flags, the true
#              sample positions and the exact frame maps (`LinearRaw(mosaic;
#              mode, gain)`), exported by Core only.
#
# The green-cell image is THE image for geometry and radiometry; `:bin2` adds
# the cell's red and blue; `:demosaic` (LibRaw's interpolation, full
# resolution) is for display figures only and carries no frame map.
# =============================================================================

"""
    RawDecoder

A RAW decoder, the token [`load_raw`](@ref) and [`load_raw_mosaic`](@ref)
dispatch on: [`LibRawDecoder`](@ref), implemented by a package extension that
needs its trigger package loaded.
"""
abstract type RawDecoder end

"""
    LibRawDecoder()

Decode with LibRaw through LibRaw_jll (`using LibRaw_jll`): a `ccall` layer over
LibRaw's C API, no Python. The extension checks LibRaw's version at load and the
layout of every structure it reads against LibRaw's own accessors, and refuses
(`ErrorException`) a library it was not written for. The stack's one decoder.
"""
struct LibRawDecoder <: RawDecoder end

_decoder_trigger(::LibRawDecoder) = "LibRaw_jll"
_decoder_extension(::LibRawDecoder) = :VisualGeometryToolkitFeaturesLibRawExt

# The decode hooks: an extension adds the methods for its decoder. These fallbacks are Core's `require_extension`'s:
# without the extension loaded they name the package to load; with it loaded, the MethodError the call raises.
_require_decoder(f, decoder::RawDecoder, args::Tuple) =
    require_extension(f, args, @__MODULE__, _decoder_extension(decoder)) do
        ArgumentError("load_raw: decoding with $(nameof(typeof(decoder)))() needs `using $(_decoder_trigger(decoder))` (its package extension is not loaded)")
    end
_decode_raw(decoder::RawDecoder, path::AbstractString) = _require_decoder(_decode_raw, decoder, (decoder, path))
# The display demosaic: LibRaw's interpolated, linear, 16-bit camera-RGB image of
# the visible area in the sensor's orientation, and the file's metadata — of a
# file, or of a mosaic in memory (its photosites, colour phase and levels handed
# to LibRaw, its own metadata kept).
_decode_demosaic(decoder::RawDecoder, path::AbstractString) = _require_decoder(_decode_demosaic, decoder, (decoder, path))
_decode_demosaic(decoder::RawDecoder, mosaic::RawMosaic) = _require_decoder(_decode_demosaic, decoder, (decoder, mosaic))
# The decoder library's version, part of `load_raw`'s cache key (a library upgrade can move the black levels, the white
# level or the crop): the extension's (LibRaw's `libraw_version()`); `nothing` for a decoder that states none.
_decoder_version(::RawDecoder) = nothing

# The display image: LibRaw's demosaic (16-bit, linear, camera colour, white
# balance 1) as Float64 in [0, 1], the white balance (per channel, if asked) and
# the gain applied here. A pixel LibRaw clipped (65535 in any channel) is flagged.
function _demosaic_image(rgb::Array{UInt16, 3}, metadata::NamedTuple, gain::Float64,
        white_balance::Union{Nothing, NTuple{3, Real}})
    wb = white_balance === nothing ? (1.0, 1.0, 1.0) : Float64.(white_balance)
    all(w -> isfinite(w) && w > 0, wb) || throw(ArgumentError("load_raw: the white balance must be positive and finite, got $white_balance"))
    img = Array{Float64, 3}(undef, size(rgb))
    for c in 1:3, j in axes(rgb, 2), i in axes(rgb, 1)
        img[i, j, c] = (Float64(rgb[i, j, c]) / 65535) * wb[c] * gain
    end
    sat = BitMatrix(dropdims(any(==(typemax(UInt16)), rgb; dims = 3); dims = 3))
    # no sample positions: every pixel is interpolated
    return LinearRaw{:demosaic}(img, sat, nothing, gain, metadata)
end

# LibRaw::adjust_bl (as LibRaw's data_helper.h writes it), the black levels per
# colour index from LibRaw's raw `cblack` array (`cb[k + 1]` is `cblack[k]`:
# four per-channel parts, then a spatial pattern of `cblack[4] × cblack[5]`
# entries at `cblack[6+]`) and its common `black`, folding a 2×2 pattern into
# the four channels. A spatial pattern that remains nonzero after the fold —
# larger than 2×2, or one LibRaw's fold leaves — has no per-channel form: it is
# refused, naming the file and the pattern's size, never dropped (a per-channel
# reading drops it silently). The decoder's one rule for the black levels.
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
its mosaic, VisualGeometryToolkitCore's `RawMosaic`. Only flat Bayer sensors (one colour per photosite) are
supported. The decoder's package must be loaded: `using LibRaw_jll` for
[`LibRawDecoder`](@ref).
"""
function load_raw_mosaic(path::AbstractString; decoder::RawDecoder = LibRawDecoder())
    isfile(path) || throw(ArgumentError("load_raw_mosaic: no file at $path"))
    return _decode_raw(decoder, String(path))::RawMosaic
end

"""
    load_raw(path; mode::Symbol, gain::Real = 1, noise = nothing, white_balance = nothing,
             decoder::RawDecoder = LibRawDecoder(), cache = nothing) -> LinearRaw
    load_raw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1, noise = nothing, white_balance = nothing,
             decoder::RawDecoder = LibRawDecoder()) -> LinearRaw

A camera RAW file — or a mosaic already in memory, decoded or corrected
(vignetting applied per photosite, say) — as linear values, in the sensor's
orientation (the EXIF flip is never applied, in any mode). `LinearRaw`,
`RawMosaic` and the frame maps are VisualGeometryToolkitCore's (exported by it).
`mode`:

- `:green_cells` or `:bin2` — exact: [`load_raw_mosaic`](@ref) (for a file),
  then Core's `LinearRaw(mosaic; mode, gain, noise)`, on the half-resolution cell grid
  that `raw_frame_map` maps onto the raw. `:green_cells` is THE image
  for geometry and radiometry. Rounded to 16 bits
  (`round(UInt16, clamp(x, 0, 1) * 65535)`), `:bin2` equals the linear PNG of
  RadiometricCamera's convert_raw.py (`--mode bin2`), value for value; unlike
  that PNG, `saturated` flags the cells that contain a photosite at the white
  level.
- `:demosaic` — FOR DISPLAY FIGURES ONLY. LibRaw's demosaic (AHD) of the
  visible area at full resolution: it INTERPOLATES the two colours each
  photosite did not record, so its values are not measured and its
  pixels carry no exact geometry. Evaluation code must not read it; it has no
  frame map (`raw_frame_map` and `green_cell_sites` refuse it).
  LibRaw's output is linear (gamma 1, no auto-brightening, no maximum
  adjustment, 16 bits, the camera's raw colour, white balance 1); the
  `white_balance` (an R G B multiplier triple, e.g. from
  `metadata.daylight_wb`) and then the gain are applied here, in Julia.
  `saturated` flags pixels LibRaw clipped. Of a mosaic in memory it is the same
  demosaic, LibRaw's, on the mosaic's photosites (copied to the host), its colour
  phase and its black and white levels (whole codes, as LibRaw holds them; an
  `ArgumentError` otherwise): a corrected mosaic is displayed through the one
  demosaic, with its own metadata.

`white_balance` is accepted with `:demosaic` only: the exact modes are camera
raw colour, white balance 1.

`noise` (the exact modes only): the frame's resolved sensor noise
(VisualGeometryToolkitCore's `SensorNoise`, from `sensor_noise`), handed to
Core's `LinearRaw`, the stack's one black subtraction: the noise's black level
per channel, its dark frame's per-photosite offset when it holds one, its defect
mask joined to `saturated`. `nothing` (the default) subtracts the file's
per-channel black. The display demosaic is LibRaw's own and takes none.

`cache` (a file only; a [`FileCache`](@ref), `using JLD2` provides it): each RAW is
decoded once per content and settings — the entry is keyed by the file's
SHA-256, `mode`, `gain`, `white_balance`, the decoder's type and its library's
version (LibRaw's, `libraw_version`) ([`cache_key`](@ref)), so an edited file,
another setting or an upgraded LibRaw decodes again and
a hit returns the `LinearRaw` as decoded, value for value. `nothing` (the
default) decodes every call. A `noise` is no key (its dark frame is a map): with
one, cache the mosaic's decode instead, `load_raw(load_raw_mosaic(path); mode, noise)`.
"""
function load_raw(path::AbstractString; mode::Symbol, gain::Real = 1, noise::Union{Nothing, SensorNoise} = nothing,
                  white_balance = nothing, decoder::RawDecoder = LibRawDecoder(), cache = nothing)
    _check_raw_options(mode, white_balance, noise)
    cache === nothing || noise === nothing || throw(ArgumentError(
        "load_raw: a `noise` is no cache key (its dark frame is a map): load the mosaic and give it, load_raw(load_raw_mosaic(path); mode, noise)"))
    cache === nothing && return _load_raw(path, mode, gain, noise, white_balance, decoder)
    isfile(path) || throw(ArgumentError("load_raw: no file at $path"))
    key = cache_key("load_raw", open(sha256_hex, path), mode, Float64(gain), white_balance, string(typeof(decoder)),
                    _decoder_version(decoder))
    return get!(() -> _load_raw(path, mode, gain, noise, white_balance, decoder), cache, key)
end

# The options' rules: the white balance is the display demosaic's, the noise the exact modes'.
function _check_raw_options(mode, white_balance, noise)
    mode === :demosaic || white_balance === nothing ||
        throw(ArgumentError("load_raw: `white_balance` applies to the display demosaic alone; :$mode is camera raw colour, white balance 1"))
    mode === :demosaic && noise !== nothing &&
        throw(ArgumentError("load_raw: the display demosaic is LibRaw's, black level and all; a `noise` applies to :green_cells and :bin2"))
    return nothing
end

function _load_raw(path, mode, gain, noise, white_balance, decoder)
    mode === :demosaic || return LinearRaw(load_raw_mosaic(path; decoder); mode, gain, noise)
    isfile(path) || throw(ArgumentError("load_raw: no file at $path"))
    rgb, metadata = _decode_demosaic(decoder, String(path))
    return _demosaic_image(rgb, metadata, Float64(gain), white_balance)
end

function load_raw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1, noise::Union{Nothing, SensorNoise} = nothing,
                  white_balance = nothing, decoder::RawDecoder = LibRawDecoder())
    _check_raw_options(mode, white_balance, noise)
    mode === :demosaic || return LinearRaw(mosaic; mode, gain, noise)
    rgb, metadata = _decode_demosaic(decoder, mosaic)
    return _demosaic_image(rgb, metadata, Float64(gain), white_balance)
end

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

The EXIF fields of a camera RAW file, as `RawMosaic`'s `metadata`
holds them, without keeping the photosites: [`LibRawDecoder`](@ref) reads the
headers only (no unpack). The typed
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

The exposure a RAW frame was taken with — `x` a `RawMosaic` or its
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
_identity_fields(d::DeviceIdentity) = (d.make, d.model, d.serial)
_agrees(a::DeviceIdentity, b::DeviceIdentity) =
    all(((x, y),) -> x === nothing || y === nothing || lowercase(x) == lowercase(y), zip(_identity_fields(a), _identity_fields(b)))
# Identities that agree pairwise, as one: each field the first that states it.
_first_stated(xs) = (i = findfirst(!isnothing, xs); i === nothing ? nothing : xs[i])
_merged(ids) = DeviceIdentity((_first_stated(xs) for xs in zip(map(_identity_fields, ids)...))...)
function _one_identity(ids, what)
    all(a -> all(b -> _agrees(a, b), ids), ids) ||
        throw(ArgumentError("extract_capture_settings: the frames name $(length(unique(ids))) $(what)s, $(unique(ids)); a capture has one"))
    return _merged(ids)
end

"""
    extract_capture_settings(frames; lens = nothing) -> CaptureSettings

The camera settings of a capture from its RAW files' EXIF: `frames` maps a
frame index to a `RawMosaic` or its `metadata`
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
