# =============================================================================
# Camera RAW input — a Bayer mosaic as exact linear values
# =============================================================================
#
# THE one RAW path of the stack: every consumer reads RAW files through this
# module, and `VisualGeometryToolkitCore.check_raw_decoding` flags any other
# decoding path. Two layers, kept apart on purpose:
#
#   decode   — a RAW file to its mosaic (`RawMosaic`): the visible photosites,
#              their colour pattern, the per-channel black levels, the white
#              level and the EXIF fields. The decoder is the weak-dependency
#              extension on LibRaw_jll (`LibRawDecoder()`, ccall, no Python);
#              it only decodes, and for the display demosaic only runs
#              LibRaw's own interpolation.
#   process  — every number after that, here, in Julia: the linear scaling,
#              the 2×2 Bayer cells (`:bin2`, `:green_cells`), the saturation
#              flags, the true sample positions and the exact frame map of the
#              half-resolution grid (`LinearRaw(mosaic; mode, gain)`).
#
# The green-cell image is THE image for geometry and radiometry; `:bin2` adds
# the cell's red and blue; `:demosaic` (LibRaw's interpolation, full
# resolution) is for display figures only and carries no frame map. Both cell
# modes are one KernelAbstractions kernel (on the CPU backend of the host
# mosaic).
# =============================================================================

using KernelAbstractions: @kernel, @index, @Const
using Adapt: Adapt, adapt

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

# The decode hooks: an extension adds the methods for its decoder. Without the
# extension loaded these fallbacks name the package to load.
_missing_decoder(decoder::RawDecoder) = throw(ArgumentError(
    "load_raw: decoding with $(nameof(typeof(decoder)))() needs `using $(_decoder_trigger(decoder))` (its package extension is not loaded)"
))
_decode_raw(decoder::RawDecoder, path::AbstractString) = _missing_decoder(decoder)
# The display demosaic: LibRaw's interpolated, linear, 16-bit camera-RGB image of
# the visible area in the sensor's orientation, and the file's metadata.
_decode_demosaic(decoder::RawDecoder, path::AbstractString) = _missing_decoder(decoder)

"""
    RawMosaic{A <: AbstractMatrix{UInt16}, M <: NamedTuple} <: AbstractMatrix{UInt16}

A RAW file decoded and nothing more: the visible area of the sensor, as LibRaw
reports it. It IS its photosite matrix: `size`, indexing and `parent` are the
held array's, so an ordinary array or image function reads it unchanged, and
what such a function makes (`similar`, `copy`, a broadcast, a reduction) is a
plain array of the held array's kind — the colour pattern and the metadata stay
with the mosaic. `copy`, `==`, `isequal` and `isapprox` are the held array's
own (no element read through the wrapper, so a device mosaic compares on the
device), and equality is the photosite matrix's: two mosaics with the same
photosites are `==` whatever their colour pattern, window origin or metadata
(compare [`photosite_colors`](@ref), `window_origin` and `metadata` for those),
and a mosaic is `==` to its held array. It is read-only (no `setindex!`): write to `parent(m)`. The held array may be a device array (`adapt(CuArray, m)`, or
`adapt(backend, m)` with a KernelAbstractions backend, moves the photosites and
keeps the rest); a kernel receives `parent(m)`, never the wrapper.

A unit-range index `m[rows, columns]` (a copy) or `view(m, rows, columns)` (no
copy) is a window: a mosaic of its own whose colours are the parent's at those
photosites (the colour phase shifts with an odd start, as a `circshift` of the
pattern would) and whose [`raw_frame_map`](@ref) is its offset into the visible
raw. Any other index is a plain array.

- `raw::A` — the photosites, `raw[row, column]`, in the sensor's orientation
  (no EXIF flip): pixel `(row, column)` is the raw's COLMAP pixel centred at
  `(column − ½, row − ½)` — the visible raw's `PixelFrame{:colmap}`, the frame
  every calibration of the raw is in (a window's pixels are offset by
  `window_origin`).
- `pattern::Matrix{UInt8}` — the smallest repeating colour pattern, square,
  at most 4 × 4, from the full raw frame's origin (LibRaw's `COLOR` read
  from there), in LibRaw's colour indices: `0` R, `1` G, `2` B, `3` the
  second G of an RGBG pattern. Each photosite's index is
  [`photosite_colors`](@ref)`(mosaic)`, the pattern tiled from that origin.
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
- `window_origin::NTuple{2, Int}` — `(rows, columns)`, the offset of `raw[1, 1]`
  from the visible area's first photosite: `(0, 0)` as decoded (the keyword of
  the constructor), a window's start less one otherwise.
"""
struct RawMosaic{A <: AbstractMatrix{UInt16}, M <: NamedTuple} <: AbstractMatrix{UInt16}
    raw::A
    pattern::Matrix{UInt8}
    black::NTuple{4, Float64}
    white::Float64
    visible_origin::NTuple{2, Int}
    raw_size::NTuple{2, Int}
    metadata::M
    window_origin::NTuple{2, Int}
    function RawMosaic(raw::A, pattern, black, white, visible_origin, raw_size, metadata::M;
                       window_origin::NTuple{2, Integer} = (0, 0)) where {A <: AbstractMatrix{UInt16}, M <: NamedTuple}
        n = size(pattern, 1)
        (size(pattern, 2) == n && 1 <= n <= 4) ||
            throw(ArgumentError("RawMosaic: the colour pattern must be square, at most 4 × 4, got $(join(size(pattern), '×'))"))
        all(c -> c <= 3, pattern) || throw(ArgumentError("RawMosaic: colour indices are 0 (R), 1 (G), 2 (B) or 3 (G2), got $pattern"))
        all(<(white), black) || throw(ArgumentError("RawMosaic: every black level must be below the white level $white, got $black"))
        return new{A, M}(raw, pattern, black, white, visible_origin, raw_size, metadata, window_origin)
    end
end

# The photosite matrix: the held array's size, indices and kind. Whatever an array function makes from
# it is the held array's kind (`similar`, a broadcast, a reduction on a device array), never the wrapper.
Base.size(m::RawMosaic) = size(m.raw)
Base.parent(m::RawMosaic) = m.raw
Base.IndexStyle(::Type{<:RawMosaic{A}}) where {A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(m::RawMosaic, i::Int...) = m.raw[i...]
Base.similar(m::RawMosaic, ::Type{T}, dims::Dims) where {T} = similar(m.raw, T, dims)
Base.Broadcast.broadcastable(m::RawMosaic) = m.raw
Base.mapreduce(f, op, m::RawMosaic; kw...) = mapreduce(f, op, m.raw; kw...)
Base.copy(m::RawMosaic) = copy(m.raw)
# Equality is the photosite matrix's (an AbstractMatrix's, so a mosaic is `==` to its held array and `hash` agrees), read
# through the held array's own methods (a device array's, no scalar read; against a dense array, a device array's kind):
# the colour pattern, window origin and metadata are not compared.
for f in (:(==), :isequal)
    @eval Base.$f(a::RawMosaic, b::RawMosaic) = $f(a.raw, b.raw)
    @eval Base.$f(a::RawMosaic, b::DenseArray) = $f(a.raw, b)
    @eval Base.$f(a::DenseArray, b::RawMosaic) = $f(a, b.raw)
end
Base.isapprox(a::RawMosaic, b::RawMosaic; kw...) = isapprox(a.raw, b.raw; kw...)
Base.isapprox(a::RawMosaic, b::DenseArray; kw...) = isapprox(a.raw, b; kw...)
Base.isapprox(a::DenseArray, b::RawMosaic; kw...) = isapprox(a, b.raw; kw...)
Adapt.adapt_structure(to, m::RawMosaic) = _with_raw(m, adapt(to, m.raw), m.window_origin)

# A window (unit ranges in both axes) is a mosaic of the same sensor: the photosites `raw` (the
# window's copy or view) and its offset added to the parent's.
_with_raw(m::RawMosaic, raw, window_origin) =
    RawMosaic(raw, m.pattern, m.black, m.white, m.visible_origin, m.raw_size, m.metadata; window_origin)
function _window(m::RawMosaic, raw, rows::AbstractUnitRange, columns::AbstractUnitRange)
    return _with_raw(m, raw, m.window_origin .+ (first(rows) - 1, first(columns) - 1))
end
Base.getindex(m::RawMosaic, rows::AbstractUnitRange{<:Integer}, columns::AbstractUnitRange{<:Integer}) =
    _window(m, m.raw[rows, columns], rows, columns)
Base.view(m::RawMosaic, rows::AbstractUnitRange{<:Integer}, columns::AbstractUnitRange{<:Integer}) =
    _window(m, view(m.raw, rows, columns), rows, columns)

# One line, no photosite read (a device array is never read element by element).
function Base.show(io::IO, m::RawMosaic)
    print(io, join(size(m), '×'), " RawMosaic (", cfa_string(m), " from the raw origin; visible origin ", m.visible_origin)
    m.window_origin == (0, 0) || print(io, ", window origin ", m.window_origin)
    return print(io, ") of a ", nameof(typeof(m.raw)), "{UInt16}")
end
Base.show(io::IO, ::MIME"text/plain", m::RawMosaic) = show(io, m)

"""
    photosite_colors(mosaic::RawMosaic) -> Matrix{UInt8}

Each photosite's colour index (LibRaw's: `0` R, `1` G, `2` B, `3` the second
G), the size of `mosaic`: the `pattern` tiled from the full raw frame's origin,
so photosite `(row, column)` reads
`pattern[mod(top + row − 1, n) + 1, mod(left + column − 1, n) + 1]` with
`(top, left) = visible_origin .+ window_origin`.
"""
photosite_colors(m::RawMosaic) = UInt8[_photosite_color(m, i, j) for i in axes(m, 1), j in axes(m, 2)]

# The colour index of photosite (i, j): the pattern tiled from the raw origin.
@inline function _photosite_color(m::RawMosaic, i, j)
    n = size(m.pattern, 1)
    top, left = m.visible_origin .+ m.window_origin
    return @inbounds m.pattern[mod(top + i - 1, n) + 1, mod(left + j - 1, n) + 1]
end

"""
    raw_frame_map(m::RawMosaic) -> SimilarityMap{2, …, PixelFrame{:colmap}, PixelFrame{:colmap}}

The map of the mosaic's pixels onto the visible raw's `PixelFrame{:colmap}`:
the identity as decoded, a window's offset (`x ↦ x + (columns, rows)` of its
`window_origin`) for a window.
"""
raw_frame_map(m::RawMosaic) = relabel(PixelFrame{:colmap}, PixelFrame{:colmap},
    SimilarityMap(Translation(SVector{2, Float64}(reverse(m.window_origin)))))

"""
    cfa_string(mosaic::RawMosaic) -> String

The colour pattern as letters, row by row (`"RGGB"` for `[0 1; 3 2]`): LibRaw's
indices read through `"RGBG"`, as convert_raw.py records it.
"""
cfa_string(mosaic::RawMosaic) = join(("RGBG"[Int(c) + 1] for c in permutedims(mosaic.pattern)))

"""
    BayerCellFrame{C} <: Frame{2}

The COLMAP pixel frame of one colour plane `C` (`:R`, `:G` or `:B`) of the
half-resolution grid of 2×2 Bayer cells — the planes of the `:green_cells` and
`:bin2` images of [`LinearRaw`](@ref): pixel `(row, column)` of the plane is
centred at `(column − ½, row − ½)` of its frame. The planes are indexed alike
but sampled at different points of the cell, so each is a frame of its own:
[`bayer_plane_map`](@ref) maps plane `C`'s frame onto the visible raw's
`PixelFrame{:colmap}`, and a composition that joins one plane's frame to
another's is refused (`MethodError`), as at every frame junction of Core.
`BayerCellFrame{:G}` is the cell grid itself (the greens' mean sits at the
cell centre in every phase): [`raw_frame_map`](@ref) is its map.
"""
struct BayerCellFrame{C} <: Frame{2} end

# THE map of the cell grid (the green plane) onto the visible raw, x ↦ 2x, exactly. Cell (i, j)
# holds photosites (2i − 1, 2j − 1) … (2i, 2j) of the visible area, whose
# centres (2j − 1.5, 2i − 1.5) … (2j − 0.5, 2i − 0.5) average to (2j − 1, 2i − 1),
# twice the cell centre (j − ½, i − ½). The cells start at the visible area's
# first photosite whatever the parity of `visible_origin`: an odd margin changes
# only the cell's colour phase (RGGB from the raw origin reads BGGR, GRBG or
# GBRG from an odd one), which the colour lookup carries, not the geometry. In
# the FULL raw frame (`raw_size`) the map would add `(left, top)`; nothing reads
# that frame.
const _CELL_FRAME_MAP = relabel(BayerCellFrame{:G}, PixelFrame{:colmap}, SimilarityMap(LinearMap(UniformScaling(2.0))))

# Plane C's map onto the raw: its translation onto the cell grid (`o`, where the
# plane's samples sit from the cell centre, cell pixels) composed with the cell
# grid's map, so the ×2 is stated once, in `_CELL_FRAME_MAP`. The green plane
# IS the cell grid.
_plane_map(::Val{:G}, o) = _CELL_FRAME_MAP
_plane_map(::Val{C}, o) where {C} = _CELL_FRAME_MAP ∘ relabel(BayerCellFrame{C}, BayerCellFrame{:G}, SimilarityMap(Translation(o)))

"""
    LinearRaw

A RAW image as linear values: `(raw − black[c]) / (white − black[c]) × gain`,
camera raw colour, white balance 1, `1.0` = the white level (before the gain).
Not clipped: a value above 1 (a gain above 1) or below 0 (noise under the black
level) is kept.

- `image::AbstractArray{Float64, 3}` — `image[row, column, plane]`, the planes
  plain arrays on the mosaic's device (an `Array` on the host); a plane's frame
  is asked of the image with its index ([`bayer_plane_map`](@ref)`(L, c)`).
- `saturated::AbstractMatrix{Bool}` (a `BitMatrix` on the host) — the pixels
  whose value includes a photosite at the
  white level (`raw ≥ white`): in `:bin2` a cell is saturated when ANY of its
  four photosites is, so a cell whose two greens straddle the white level is
  flagged rather than averaged into a plausible value; in `:green_cells` when
  either green is.
- `geometry` — where the samples are, `(; plane_maps, green_offsets, green_dn)`:
  - `plane_maps`, per plane, its map onto the visible raw
    ([`bayer_plane_map`](@ref), which reads it);
  - `green_offsets`, the cell's two green photosites, `(Δx, Δy)` from the cell
    centre in RAW pixels ([`green_cell_sites`](@ref));
  - `green_dn`, the two green photosites' `white − black[c]`, the codes that span
    `0 … 1` (before the gain), in the order of `green_offsets`: what a value's
    photon and read noise scale by ([`variance_stabilize`](@ref)).

  `nothing` for `:demosaic`, whose pixels are interpolated, not samples.
- `gain` and the mosaic's `metadata` (the decoder and LibRaw's version among it).

The mode is the type's first parameter, `LinearRaw{:green_cells}`,
`LinearRaw{:bin2}` or `LinearRaw{:demosaic}`: the geometric accessors are
methods of the cell images, and the display image's are its refusals.
[`raw_provenance`](@ref) gives the mode, the gain, the metadata and the frame
map as plain data for a result to record.
"""
struct LinearRaw{Mode, I <: AbstractArray{Float64, 3}, S <: AbstractMatrix{Bool},
                 G <: Union{Nothing, NamedTuple{(:plane_maps, :green_offsets, :green_dn)}}, M <: NamedTuple}
    image::I
    saturated::S
    geometry::G
    gain::Float64
    metadata::M
    # the display image alone holds no geometry
    function LinearRaw{Mode}(image::I, saturated::S, geometry::G, gain, metadata::M) where {Mode, I, S, G, M}
        Mode in RAW_MODES || throw(ArgumentError("LinearRaw: mode must be one of $(RAW_MODES), got :$Mode"))
        (Mode === :demosaic) == (geometry === nothing) ||
            throw(ArgumentError("LinearRaw: a :$Mode image $(Mode === :demosaic ? "holds no" : "needs its") sample geometry"))
        return new{Mode, I, S, G, M}(image, saturated, geometry, gain, metadata)
    end
end

# The saturation flags as held: packed bits on the host, the kernel's Bool array on a device.
_saturation_mask(sat::Array{Bool}) = BitMatrix(sat)
_saturation_mask(sat::AbstractMatrix{Bool}) = sat

_mode(::LinearRaw{Mode}) where {Mode} = Mode

const RAW_MODES = (:green_cells, :bin2, :demosaic)

"""
    LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1) -> LinearRaw

The mosaic's linear values, `(raw − black[c]) / (white − black[c]) × gain`, on
the half-resolution grid of 2×2 Bayer cells (`H ÷ 2 × W ÷ 2`; an odd last row
or column is dropped), arranged by `mode` (required: there is no neutral
choice between them):

- `:green_cells` — THE image for geometry and radiometry, one plane: each cell's
  two green photosites' mean. The greens of a cell are always diagonal, so in
  every Bayer phase their mean sits at the cell centre (offset `(0, 0)`),
  and the cell grid maps onto the raw exactly by [`raw_frame_map`](@ref)
  (`x ↦ 2x` in COLMAP coordinates). The footprint of a value is its two
  photosites, at `(±½, ∓½)` raw pixels from the cell centre for an RGGB or
  BGGR cell, `(±½, ±½)` for GRBG or GBRG ([`green_cell_sites`](@ref)).
- `:bin2` — the same cells with their red and blue, `× 3` planes: R, the mean
  of the two greens (the `:green_cells` image, bit for bit), B. R and B sit at
  their photosites, `(±¼, ±¼)` cell pixels from the centre: each plane's map
  onto the raw is [`bayer_plane_map`](@ref).

`:demosaic` (LibRaw's interpolation, for display figures) needs the file:
[`load_raw`](@ref)`(path; mode = :demosaic)`.

The values are those convert_raw.py writes (`--mode bin2`), in the same Float64
arithmetic and order, before its clip to `[0, 1]` and 16-bit rounding.

Saturation is `raw ≥ white` with no margin: on the Sony ARW sets this was
written for, codes pile up exactly at the white level (16383) while the
histogram runs smoothly through the camera's declared linear limit (15360,
`metadata.linear_max`), so any margin would be a tuned constant. A caller who
wants that stricter limit thresholds `mosaic.raw` itself.
"""
function LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1)
    mode in RAW_MODES || throw(ArgumentError("LinearRaw: mode must be one of $(RAW_MODES), got :$mode"))
    mode === :demosaic && throw(ArgumentError(
        "LinearRaw: the display demosaic is LibRaw's interpolation of the file: `load_raw(path; mode = :demosaic)`"
    ))
    return _cells(mosaic, _checked_gain(gain), mode)
end

function _checked_gain(gain::Real)
    isfinite(gain) && gain > 0 || throw(ArgumentError("LinearRaw: the gain must be positive and finite, got $gain"))
    return Float64(gain)
end

# (raw − black[c]) / (white − black[c]): convert_raw.py's expression, in its
# order (the gain is applied after the cell's mean, as there), with the
# black level and the denominator `white − black[c]` taken from per-colour
# (or per-site) constants computed once — the same two operations on the same
# values, so the same bits.
@inline _linear(x::UInt16, black::Float64, den::Float64) = (Float64(x) - black) / den

# A cell's four photosites in column-major order, k = di + 2dj + 1.
const _QUAD_SITES = ((0, 0), (1, 0), (0, 1), (1, 1))

# Plane p of cell (i, j): the mean of its photosites' (cell-site indices `ks`)
# linear values, times the gain. Unrolled over the planes (a tuple of tuples).
@inline _store_planes!(img, i, j, v, ::Tuple{}, gain, p) = nothing
@inline function _store_planes!(img, i, j, v, planes::Tuple, gain, p)
    ks = first(planes)
    @inbounds img[i, j, p] = (foldl((s, k) -> s + v[k], ks; init = 0.0) / length(ks)) * gain
    return _store_planes!(img, i, j, v, Base.tail(planes), gain, p + 1)
end

# Each cell's planes and whether any photosite its planes read (`sites`) is at
# the white level. One work item per cell; the cell modes differ only in the
# planes they ask for.
@kernel function _cells_kernel!(img, sat, @Const(raw), planes, sites, black, den, gain::Float64, white::Float64)
    i, j = @index(Global, NTuple)
    r0, c0 = 2i - 1, 2j - 1
    @inbounds begin
        x = ntuple(k -> raw[r0 + _QUAD_SITES[k][1], c0 + _QUAD_SITES[k][2]], Val(4))
        v = ntuple(k -> _linear(x[k], black[k], den[k]), Val(4))
        _store_planes!(img, i, j, v, planes, gain, 1)
        sat[i, j] = any(k -> x[k] >= white, sites)
    end
end

function _cells(m::RawMosaic, gain::Float64, mode::Symbol)
    H, W = size(m)
    H2, W2 = H ÷ 2, W ÷ 2
    H2 >= 1 && W2 >= 1 || throw(ArgumentError("LinearRaw: a $(H)×$(W) mosaic holds no 2×2 cell"))
    # the cell's colour layout, the same in every cell of a 2×2 Bayer pattern
    # (a pattern that repeats with period 2 from the raw origin)
    n = size(m.pattern, 1)
    (iseven(n) && all(m.pattern[i, j] == m.pattern[mod1(i, 2), mod1(j, 2)] for i in 1:n, j in 1:n)) ||
        throw(ArgumentError("LinearRaw: :$mode needs a 2×2 Bayer pattern, got the colour pattern $(m.pattern)"))
    cell = ntuple(k -> _photosite_color(m, 1 + _QUAD_SITES[k][1], 1 + _QUAD_SITES[k][2]), Val(4))
    plane(c) = c == 3 ? 2 : Int(c) + 1                         # 0 R, 1 G, 2 B, 3 G2 → planes R, G, B
    planes = ntuple(p -> Tuple(k for k in 1:4 if plane(cell[k]) == p), Val(3))
    map(length, planes) == (1, 2, 1) ||
        throw(ArgumentError("LinearRaw: :$mode needs one R, two G and one B per cell, got $([cell[1] cell[3]; cell[2] cell[4]])"))
    r, g, b = planes[1]::NTuple{1, Int}, planes[2]::NTuple{2, Int}, planes[3]::NTuple{1, Int}
    black = ntuple(k -> m.black[cell[k] + 1], Val(4))
    den = ntuple(k -> m.white - black[k], Val(4))
    # photosite (di, dj) of a cell is at raw COLMAP (2j − 1.5 + dj, 2i − 1.5 + di):
    # (dj − ½, di − ½) raw pixels from the cell centre, half that in cell pixels
    site_offset(k) = SVector(_QUAD_SITES[k][2] - 0.5, _QUAD_SITES[k][1] - 0.5)
    green_offsets = (site_offset(g[1]), site_offset(g[2]))
    used, colours, sites = mode === :bin2 ? ((r, g, b), (:R, :G, :B), (1, 2, 3, 4)) : ((g,), (:G,), g)
    P = length(used)
    # on the photosites' device: the kernel reads the held array, never the wrapper
    raw = parent(m)
    img = similar(raw, Float64, H2, W2, P)
    sat = similar(raw, Bool, H2, W2)
    backend = KA.get_backend(raw)
    _cells_kernel!(backend)(img, sat, raw, used, sites, black, den, gain, m.white; ndrange = (H2, W2))
    KA.synchronize(backend)
    # each plane's samples from the cell centre, in cell pixels (the frame map's
    # scale, the one place it is stated), as the plane's translation onto the grid, and the
    # mosaic's own map onto the visible raw (a window's offset) after it
    offset(ks) = sum(site_offset, ks) / (length(ks) * scale(_CELL_FRAME_MAP))
    maps = map((C, ks) -> raw_frame_map(m) ∘ _plane_map(Val(C), offset(ks)), colours, used)
    geometry = (plane_maps = maps, green_offsets, green_dn = (den[g[1]], den[g[2]]))
    return LinearRaw{mode}(img, _saturation_mask(sat), geometry, gain, m.metadata)
end

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

# The display demosaic carries no geometry: the refusal every geometric accessor shares.
_display_only(what) = ArgumentError(
    "$what: a :demosaic image interpolates and is for display figures only; geometry and radiometry read `mode = :green_cells`"
)
_geometry(L::LinearRaw, what) = L.geometry
_geometry(::LinearRaw{:demosaic}, what) = throw(_display_only(what))

"""
    raw_frame_map(L::LinearRaw) -> SimilarityMap{2, …, BayerCellFrame{:G}, PixelFrame{:colmap}}

THE map of the image's pixel frame (the cell grid, which is its green plane's
frame) onto the visible raw's `PixelFrame{:colmap}`
(the frame of `RawMosaic.raw` and of every calibration of the raw): `x ↦ 2x`,
exactly, for every Bayer phase and either parity of the visible origin (the
cells start at the visible area's first photosite; an odd margin changes only
the cell's colour phase); for an image made from a window of the mosaic, `x ↦ 2x`
then the window's offset. Its inverse is `inv(raw_frame_map(L))`, and a
calibration matrix moves with it through Core,
`reframed_calibration(raw_frame_map(L), K_cells)`, `K_cells` in the cell
image's pixels (`BayerCellFrame{:G}`), giving K in the raw's pixels (and back
with the inverse; K carries no frame label, so which frame it is in is the
caller's statement). A `:demosaic` image is refused (`ArgumentError`): it is for display.
"""
raw_frame_map(L::LinearRaw) = _geometry(L, "raw_frame_map").plane_maps[_green_plane(L)]

# The plane holding the green cells (the cell grid): bin2's second (R, G, B), the green-cell image's one.
_green_plane(::LinearRaw{:bin2}) = 2
_green_plane(::LinearRaw{:green_cells}) = 1

"""
    raw_coordinates(img, x) -> SVector{2, Float64}

Where the point `x` of an image's pixel frame (COLMAP, `(column − ½, row − ½)`
the centre of pixel `(row, column)`; any 2-vector or tuple) lies on the sensor:
in the visible raw's `PixelFrame{:colmap}`, the frame of every calibration of
the raw. THE accessor of an image's sensor-pixel coordinates, `raw_frame_map(img)(x)`:

- a [`RawMosaic`](@ref): the identity as decoded, a window's offset added for a window;
- a [`LinearRaw`](@ref) cell image (`:green_cells`, `:bin2`): the cell grid's
  map, `x ↦ 2x` (and a window's offset) — a detection on the green cells
  reports its position through it, never as raw pixels unmapped (half the scale);
  a position on another plane of a `:bin2` image is
  [`bayer_plane_map`](@ref)`(L, c)(x)`;
- a `:demosaic` image is refused (`ArgumentError`), and a plain matrix, which
  states no frame, has no method.
"""
raw_coordinates(img::Union{RawMosaic, LinearRaw}, x) = raw_frame_map(img)(SVector{2, Float64}(Tuple(x)))

"""
    bayer_plane_map(L::LinearRaw, c::Integer) -> SimilarityMap{2, …, BayerCellFrame{C}, PixelFrame{:colmap}}

The Bayer transform of plane `c` of the image: the map of the plane's own frame
([`BayerCellFrame`](@ref)`{C}`, `C` its colour — `:R`, `:G`, `:B` for
`:bin2`, `:G` for `:green_cells`) onto the visible raw's
`PixelFrame{:colmap}`. It is the plane's translation onto the cell grid (where
its samples sit from the cell centre: `(±¼, ±¼)` cell pixels for R and B, none
for G) composed with [`raw_frame_map`](@ref)`(L)`, so pixel `(row, column)` of
the plane, at `(column − ½, row − ½)` of its frame, goes onto the centre of the
photosites its value is the mean of — R's or B's one, the two greens' midpoint —
exactly, for every Bayer phase and either parity of the visible origin. The
green plane's map IS `raw_frame_map(L)`.

A warp reads plane `c` of an image through `inv(bayer_plane_map(L, c)) ∘ math`,
`math` landing in the raw's pixels, like any other map
(`VisualGeometryToolkitCore.image_warp`; one warp per plane:
`image_warp!(dst, warps, L.image)` with a tuple of them). A composition that
joins two planes' frames is refused (`MethodError`). A plane past the image's,
or a `:demosaic` image, is an `ArgumentError`.
"""
function bayer_plane_map(L::LinearRaw, c::Integer)
    maps = _geometry(L, "bayer_plane_map").plane_maps
    1 <= c <= length(maps) || throw(ArgumentError("bayer_plane_map: a :$(_mode(L)) image has $(length(maps)) plane(s), got plane $c"))
    return maps[c]
end

"""
    green_cell_sites(L::LinearRaw, row::Integer, column::Integer) -> NTuple{2, SVector{2, Float64}}

The footprint of the green-cell value of pixel `(row, column)`: the centres of
its two green photosites in the visible raw's COLMAP frame. They are diagonal,
at `(±½, ∓½)` raw pixels from the cell centre `raw_frame_map(L)((column − ½, row − ½))`
in an RGGB or BGGR cell and `(±½, ±½)` in a GRBG or GBRG one, and the value is
the mean of the two — what a renderer models to predict it exactly. A
`:demosaic` image is refused (`ArgumentError`).
"""
function green_cell_sites(L::LinearRaw, row::Integer, column::Integer)
    o = _geometry(L, "green_cell_sites").green_offsets
    c = raw_frame_map(L)(SVector(column - 0.5, row - 0.5))
    return (c + o[1], c + o[2])
end

"""
    raw_provenance(L::LinearRaw) -> NamedTuple

How the image was made, as plain data a result records: `decoder`,
`libraw_version` (from the mosaic's metadata; `nothing` where it states none),
`mode`, `gain` and `frame_map` (`(; from, to, scale, translation)` of
[`raw_frame_map`](@ref), `nothing` for `:demosaic`).
"""
function raw_provenance(L::LinearRaw)
    return (decoder = get(L.metadata, :decoder, nothing), libraw_version = get(L.metadata, :libraw_version, nothing),
            mode = String(_mode(L)), gain = L.gain, frame_map = _provenance_frame_map(L))
end
_provenance_frame_map(::LinearRaw{:demosaic}) = nothing
function _provenance_frame_map(L::LinearRaw)
    S = raw_frame_map(L)
    return (from = "BayerCellFrame{:G}", to = "PixelFrame{:colmap}", scale = Float64(scale(S)), translation = collect(Float64, translation(S)))
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
its [`RawMosaic`](@ref). Only flat Bayer sensors (one colour per photosite) are
supported. The decoder's package must be loaded: `using LibRaw_jll` for
[`LibRawDecoder`](@ref).
"""
function load_raw_mosaic(path::AbstractString; decoder::RawDecoder = LibRawDecoder())
    isfile(path) || throw(ArgumentError("load_raw_mosaic: no file at $path"))
    return _decode_raw(decoder, String(path))::RawMosaic
end

"""
    load_raw(path; mode::Symbol, gain::Real = 1, white_balance = nothing, decoder::RawDecoder = LibRawDecoder(),
             cache = nothing) -> LinearRaw

A camera RAW file as linear values, in the sensor's orientation (the EXIF
flip is never applied, in any mode). `mode`:

- `:green_cells` or `:bin2` — exact: [`load_raw_mosaic`](@ref), then
  [`LinearRaw`](@ref)`(mosaic; mode, gain)`, on the half-resolution cell grid
  that [`raw_frame_map`](@ref) maps onto the raw. `:green_cells` is THE image
  for geometry and radiometry. Rounded to 16 bits
  (`round(UInt16, clamp(x, 0, 1) * 65535)`), `:bin2` equals the linear PNG of
  RadiometricCamera's convert_raw.py (`--mode bin2`), value for value; unlike
  that PNG, `saturated` flags the cells that contain a photosite at the white
  level.
- `:demosaic` — FOR DISPLAY FIGURES ONLY. LibRaw's demosaic (AHD) of the
  visible area at full resolution: it INTERPOLATES the two colours each
  photosite did not record, so its values are not measurements and its
  pixels carry no exact geometry. Evaluation code must not read it; it has no
  frame map ([`raw_frame_map`](@ref) and [`green_cell_sites`](@ref) refuse it).
  LibRaw's output is linear (gamma 1, no auto-brightening, no maximum
  adjustment, 16 bits, the camera's raw colour, white balance 1); the
  `white_balance` (an R G B multiplier triple, e.g. from
  `metadata.daylight_wb`) and then the gain are applied here, in Julia.
  `saturated` flags pixels LibRaw clipped.

`white_balance` is accepted with `:demosaic` only: the exact modes are camera
raw colour, white balance 1.

`cache` (a [`FileCache`](@ref), `using JLD2` provides it): each RAW is
decoded once per content and settings — the entry is keyed by the file's
SHA-256, `mode`, `gain`, `white_balance` and the decoder's type
([`cache_key`](@ref)), so an edited file or another setting decodes again and
a hit returns the `LinearRaw` as decoded, value for value. `nothing` (the
default) decodes every call.
"""
function load_raw(path::AbstractString; mode::Symbol, gain::Real = 1, white_balance = nothing, decoder::RawDecoder = LibRawDecoder(),
                  cache = nothing)
    mode === :demosaic || white_balance === nothing ||
        throw(ArgumentError("load_raw: `white_balance` applies to the display demosaic alone; :$mode is camera raw colour, white balance 1"))
    cache === nothing && return _load_raw(path, mode, gain, white_balance, decoder)
    isfile(path) || throw(ArgumentError("load_raw: no file at $path"))
    key = cache_key("load_raw", open(sha256_hex, path), mode, Float64(gain), white_balance, string(typeof(decoder)))
    return get!(() -> _load_raw(path, mode, gain, white_balance, decoder), cache, key)
end

function _load_raw(path, mode, gain, white_balance, decoder)
    mode === :demosaic || return LinearRaw(load_raw_mosaic(path; decoder); mode, gain)
    g = _checked_gain(gain)
    isfile(path) || throw(ArgumentError("load_raw: no file at $path"))
    rgb, metadata = _decode_demosaic(decoder, String(path))
    return _demosaic_image(rgb, metadata, g, white_balance)
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

The EXIF fields of a camera RAW file, as [`RawMosaic`](@ref)'s `metadata`
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
