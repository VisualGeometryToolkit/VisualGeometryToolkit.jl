# =============================================================================
# LinearRaw — a Bayer mosaic as exact linear values on its 2×2 cells
# =============================================================================
#
# Every number after the decode: the linear scaling (`_linear_value`, the one
# expression), the 2×2 Bayer cells (`:bin2`, `:green_cells`), the saturation
# flags, the true sample positions and the exact frame map of the
# half-resolution grid (`LinearRaw(mosaic; mode, gain)`). Both cell modes are
# one KernelAbstractions kernel, reading the photosites of `channelview(mosaic)` on the mosaic's
# device. The display demosaic (`:demosaic`, LibRaw's interpolation) is made by
# VisualGeometryToolkitFeatures, which needs the decoder; the type and its
# refusals are here.
# =============================================================================

"""
    LinearRaw

A RAW image as linear values: `(raw − black[c]) / (white − black[c]) × gain`
(each photosite's [`linearview`](@ref) value), camera raw colour, white balance 1,
`1.0` = the white level (before the gain). Not clipped: a value above 1 (a gain
above 1) or below 0 (noise under the black level) is kept.

- `image::AbstractArray{Float64, 3}` — `image[row, column, plane]`, the planes
  plain arrays on the mosaic's device (an `Array` on the host); a plane's frame
  is asked of the image with its index ([`bayer_plane_map`](@ref)`(L, c)`).
- `saturated::AbstractMatrix{Bool}` (a `BitMatrix` on the host) — the MASKED
  pixels: those whose value includes a photosite at the white level
  (`raw ≥ white`) or, with a resolved noise holding a defect mask, a defective
  photosite: in `:bin2` a cell is masked when ANY of its four photosites is, so
  a cell whose two greens straddle the white level is flagged rather than
  averaged into a plausible value; in `:green_cells` when either green is.
  Masked, never interpolated.
- `geometry` — where the samples are, `(; plane_maps, green_offsets, plane_sites)`:
  - `plane_maps`, per plane, its map onto the visible raw
    ([`bayer_plane_map`](@ref), which reads it);
  - `green_offsets`, the cell's two green photosites, `(Δx, Δy)` from the cell
    centre in RAW pixels, in the cell's column-major order (`G1`, `G2` of
    [`bayer_channels`](@ref); [`green_cell_sites`](@ref));
  - `plane_sites`, per plane, the photosites its value is the mean of, each
    `(; color, dn)`: its CFA colour index (LibRaw's `0` R, `1` G, `2` B, `3` the
    second G; the index of its entry in a per-colour [`SensorNoise`](@ref) field,
    `read_noise[color + 1]`) and its `white − black[c]`, the codes that span
    `0 … 1` (before the gain) — what a value's photon and read noise scale by
    ([`pixel_noise`](@ref)). The green plane's are in the order of
    `green_offsets`. `black[c]` is the channel's constant (the resolved noise's,
    else the mosaic's) also when a dark map replaces it per photosite: the span
    is a scale, the same for every photosite of the channel, so a value is still
    `codes / dn` above its own black.

  `nothing` for `:demosaic`, whose pixels are interpolated, not samples.
- `gain` (positive and finite, `ArgumentError` otherwise) and the mosaic's
  `metadata` (the decoder and its library's version among it).

The mode is the type's first parameter, `LinearRaw{:green_cells}`,
`LinearRaw{:bin2}` or `LinearRaw{:demosaic}`: the geometric accessors are
methods of the cell images, and the display image's are its refusals.
[`raw_provenance`](@ref) gives the mode, the gain, the metadata and the frame
map as plain data for a result to record.
"""
struct LinearRaw{Mode, I <: AbstractArray{Float64, 3}, S <: AbstractMatrix{Bool},
                 G <: Union{Nothing, NamedTuple{(:plane_maps, :green_offsets, :plane_sites)}}, M <: NamedTuple}
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
        return new{Mode, I, S, G, M}(image, saturated, geometry, _checked_gain(gain), metadata)
    end
end

# The saturation flags as held: packed bits on the host, the kernel's Bool array on a device.
_saturation_mask(sat::Array{Bool}) = BitMatrix(sat)
_saturation_mask(sat::AbstractMatrix{Bool}) = sat

_mode(::LinearRaw{Mode}) where {Mode} = Mode

const RAW_MODES = (:green_cells, :bin2, :demosaic)

"""
    LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1, noise::Union{Nothing, SensorNoise} = nothing) -> LinearRaw

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

Every value is from the mosaic's [`linearview`](@ref), read channel-first
(`channelview(linearview(mosaic))`, the mosaic's [`channelview`](@ref ImageCore.channelview(::RawMosaic))): the
green-cell value of cell `(i, j)` is the mean of the view's two green channels
there, times the gain, bit for bit. `:demosaic` (LibRaw's interpolation, for
display figures) is VisualGeometryToolkitFeatures': `load_raw(path or mosaic; mode = :demosaic)`.

The values are those convert_raw.py writes (`--mode bin2`), in the same Float64
arithmetic and order, before its clip to `[0, 1]` and 16-bit rounding.

THE black subtraction is here, once, for every caller: `noise`, the frame's
resolved [`SensorNoise`](@ref) ([`sensor_noise`](@ref)), or `nothing`.

- `nothing` (the default): the mosaic's own per-channel black, `mosaic.black`
  — bit for bit the values without the keyword.
- A noise: the per-channel black is `noise.black`'s value when it holds one
  (else the mosaic's), and the span `white − black[c]`; when it holds a dark
  frame, the dark's per-photosite `offset` replaces the channel constant in the
  subtraction (`(raw − offset) / (white − black[c])`; the per-channel scalar is
  the constant-map case of the same kernel path), and its defect mask joins the
  saturation flags (`saturated` is then "masked: saturated or defective"). The
  maps cover the visible raw and are read at the mosaic's `window_origin`
  (`DimensionMismatch` when they do not reach the mosaic's photosites); they
  are moved to the mosaic's device.

Saturation is `raw ≥ white` with no margin: on the Sony ARW sets this was
written for, codes pile up exactly at the white level (16383) while the
histogram runs smoothly through the camera's declared linear limit (15360,
`metadata.linear_max`), so any margin would be a tuned constant. A caller who
wants that stricter limit thresholds `mosaic.raw` itself.
"""
function LinearRaw(mosaic::RawMosaic; mode::Symbol, gain::Real = 1, noise::Union{Nothing, SensorNoise} = nothing)
    mode in RAW_MODES || throw(ArgumentError("LinearRaw: mode must be one of $(RAW_MODES), got :$mode"))
    mode === :demosaic && throw(ArgumentError(
        "LinearRaw: the display demosaic is LibRaw's interpolation, made by VisualGeometryToolkitFeatures: " *
            "`load_raw(path_or_mosaic; mode = :demosaic)`"
    ))
    return _cell_images(mosaic, _checked_gain(gain), mode, noise)
end

function _checked_gain(gain::Real)
    isfinite(gain) && gain > 0 || throw(ArgumentError("LinearRaw: the gain must be positive and finite, got $gain"))
    return Float64(gain)
end

# Plane p of cell (i, j): the mean of its photosites' (channel indices `ks`)
# linear values, times the gain. Unrolled over the planes (a tuple of tuples).
@inline _store_planes!(img, i, j, v, ::Tuple{}, gain, p) = nothing
@inline function _store_planes!(img, i, j, v, planes::Tuple, gain, p)
    ks = first(planes)
    @inbounds img[i, j, p] = (foldl((s, k) -> s + v[k], ks; init = 0.0) / length(ks)) * gain
    return _store_planes!(img, i, j, v, Base.tail(planes), gain, p + 1)
end

# Each cell's planes and whether any photosite its planes read (`sites`) is at
# the white level. One work item per cell, reading the cell's four photosites
# from the held array by the channel view's own index rule: channel k of cell
# (i, j) is photosite (2i − 1 + di, 2j − 1 + dj), (di, dj) = _QUAD_SITES[k] —
# `channelview(m)[k, i, j]`, read without the view's wrappers: in a CUDA kernel
# Base's ReshapedArray indexing runs at a third of this bandwidth over a whole
# mosaic and does not compile over a window (an InvalidIRError;
# benchmark/raw_channelview.jl). The cell
# modes differ only in the planes they ask for. (convert_raw.py's order: the
# gain after the cell's mean.) The black of each photosite is `_black_at` (the
# dark map's offset at its visible-raw index, else the channel's constant; with
# `dark === nothing` the constant itself, bit for bit), and a photosite under the
# defect mask flags its cell as one at the white level does.
@kernel function _cells_kernel!(img, sat, @Const(raw), planes, sites, black, span, gain::Float64, white::Float64,
                                dark, defects, origin)
    i, j = @index(Global, NTuple)
    r0, c0 = 2i - 1, 2j - 1
    @inbounds begin
        x = ntuple(k -> raw[r0 + _QUAD_SITES[k][1], c0 + _QUAD_SITES[k][2]], Val(4))
        b = ntuple(k -> _black_at(dark, black[k], r0 + _QUAD_SITES[k][1] + origin[1], c0 + _QUAD_SITES[k][2] + origin[2]), Val(4))
        v = ntuple(k -> _linear_value(x[k], b[k], span[k]), Val(4))
        _store_planes!(img, i, j, v, planes, gain, 1)
        sat[i, j] = any(k -> _at_white(x[k], white) ||
                             _masked(defects, r0 + _QUAD_SITES[k][1] + origin[1], c0 + _QUAD_SITES[k][2] + origin[2]), sites)
    end
end

function _cell_images(m::RawMosaic, gain::Float64, mode::Symbol, noise)
    H, W = size(m)
    H2, W2 = H ÷ 2, W ÷ 2
    H2 >= 1 && W2 >= 1 || throw(ArgumentError("LinearRaw: a $(H)×$(W) mosaic holds no 2×2 cell"))
    # the cell's colour layout (the one phase rule), the same in every cell of a 2×2 Bayer pattern
    cell = _cell_colors(m, "LinearRaw: :$mode")
    plane(c) = c == 3 ? 2 : Int(c) + 1                         # 0 R, 1 G, 2 B, 3 G2 → planes R, G, B
    planes = ntuple(p -> Tuple(k for k in 1:4 if plane(cell[k]) == p), Val(3))
    r, g, b = planes[1]::NTuple{1, Int}, planes[2]::NTuple{2, Int}, planes[3]::NTuple{1, Int}
    raw = parent(m)
    backend = KA.get_backend(raw)
    black, span, dark, defects = _noise_levels(m, noise, backend)    # THE resolution of the levels (sensor_noise.jl)
    green_offsets = (_site_offset(g[1]), _site_offset(g[2]))
    used, colours, sites = mode === :bin2 ? ((r, g, b), (:R, :G, :B), (1, 2, 3, 4)) : ((g,), (:G,), g)
    P = length(used)
    # on the photosites' device: the kernel reads the held array (the channel view's photosites), never the wrapper
    img = similar(raw, Float64, H2, W2, P)
    sat = similar(raw, Bool, H2, W2)
    OnBackend(backend)(_cells_kernel!)(img, sat, raw, used, sites, black, span, gain, m.white, dark, defects, m.window_origin;
                                       ndrange = (H2, W2))
    KA.synchronize(backend)
    # each plane's samples sit at the mean of its photosites' offsets from the cell centre: the one
    # sample-lattice map (`_sample_map`), after the cell grid's (and the mosaic's own, a window's offset)
    grid = _cell_grid_map(m, mode)
    maps = map((C, ks) -> _sample_map(grid, Val(C), sum(_site_offset, ks) / length(ks)), colours, used)
    plane_sites = map(ks -> map(k -> (color = Int(cell[k]), dn = span[k]), ks), used)
    geometry = (plane_maps = maps, green_offsets, plane_sites)
    return LinearRaw{mode}(img, _saturation_mask(sat), geometry, gain, m.metadata)
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

"""
    raw_frame_map(::Type{LinearRaw{mode}}) -> SimilarityMap{2, …, BayerCellFrame{:G}, PixelFrame{:colmap}}

The frame map of a `mode` image made from a whole mosaic (no window), without
an image: `x ↦ 2x`, exactly, for `:green_cells` and `:bin2` alike (the cell
grid; a map that depends on the mode alone, not on the photograph, its Bayer
phase or its visible origin). THE source of the cell images' map: every
[`LinearRaw`](@ref) is made with this map after its mosaic's own
([`raw_frame_map`](@ref)`(mosaic)`, a window's offset), so
`raw_frame_map(LinearRaw(m; mode)) == raw_frame_map(m) ∘ raw_frame_map(LinearRaw{mode})`
bit for bit, and for an unwindowed mosaic the two are equal. For a frame with no
RAW behind it (a decoded image on the cell grid) whose map is needed, e.g.
`scale(raw_frame_map(LinearRaw{:green_cells})) == 2`. `LinearRaw{:demosaic}` is
refused (`ArgumentError`), as its images are.
"""
raw_frame_map(::Type{<:LinearRaw{:green_cells}}) = _CELL_FRAME_MAP
raw_frame_map(::Type{<:LinearRaw{:bin2}}) = _CELL_FRAME_MAP      # the green plane IS the cell grid in both modes
raw_frame_map(::Type{<:LinearRaw{:demosaic}}) = throw(_display_only("raw_frame_map"))

# The plane holding the green cells (the cell grid): bin2's second (R, G, B), the green-cell image's one.
_green_plane(::LinearRaw{:bin2}) = 2
_green_plane(::LinearRaw{:green_cells}) = 1
_green_plane(::LinearRaw{:demosaic}) = throw(_display_only("green_plane"))

"""
    green_plane(L::LinearRaw) -> AbstractMatrix{Float64}

The image's green cells, a view: each cell's two greens' mean — `L.image`'s plane
whose frame IS the cell grid ([`raw_frame_map`](@ref)`(L)`): a `:green_cells`
image's one plane, a `:bin2` image's second (G). The plane every green-cell
consumer reads (a detector, VisualGeometryToolkitFeatures' `variance_stabilize`);
its photosites' colours and spans are `L.geometry.plane_sites` of that plane, its
noise [`pixel_noise`](@ref)`(L, noise)`. A `:demosaic` image is refused (`ArgumentError`).
"""
green_plane(L::LinearRaw) = view(L.image, :, :, _green_plane(L))

"""
    pixel_noise(L::LinearRaw, noise::SensorNoise) -> NTuple{P, @NamedTuple{α::Float64, σ²::Float64}}

THE noise of a pixel value of `L` under a resolved [`SensorNoise`](@ref), per
plane (`L.image[:, :, p]`): a value `x` of plane `p`, the mean of its `m`
photosites (`L.geometry.plane_sites[p]`, each of colour `c` and span `d`),
times the gain `g`, is Poisson–Gaussian with variance `α·x + σ²`:

    α = g / (m² K) · Σ 1/d,       σ² = g² / m² · Σ σ_r[c]² / d²

(K the gain in electrons per code, σ_r the read noise per colour, the rounding
in it). For a green cell, `α = g/(4K)(1/d₁ + 1/d₂)` and
`σ² = g²/4 (σ_r[c₁]²/d₁² + σ_r[c₂]²/d₂²)`. A noise whose `electrons_per_dn`
or `read_noise` is `nothing` is an `ArgumentError` carrying its `reason`; a
`:demosaic` image (interpolated, no photosites) is refused.
"""
function pixel_noise(L::LinearRaw, noise::SensorNoise)
    sites = _geometry(L, "pixel_noise").plane_sites
    (noise.electrons_per_dn === nothing || noise.read_noise === nothing) && throw(ArgumentError(
        "pixel_noise: the noise has no $(noise.electrons_per_dn === nothing ? "gain" : "read noise"): " *
            something(noise.reason, "the record holds none")))
    K = value_of(noise.electrons_per_dn)
    σr = value_of(noise.read_noise)
    g = L.gain
    return map(sites) do ps
        m = length(ps)
        (α = g / (m^2 * K) * sum(s -> 1 / s.dn, ps), σ² = g^2 / m^2 * sum(s -> σr[s.color + 1]^2 / s.dn^2, ps))
    end
end


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
  [`bayer_plane_map`](@ref)`(L, c)(x)`, and on a channel of a mosaic's
  `channelview(m)` it is `bayer_plane_map(m, k)(x)`;
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
green plane's map IS `raw_frame_map(L)`, and the R and B planes' maps are the
R and B channels' of the mosaic's `channelview` (`bayer_plane_map(m, k)`):
one rule.

A warp reads plane `c` of an image through `inv(bayer_plane_map(L, c)) ∘ math`,
`math` landing in the raw's pixels, like any other map
([`image_warp`](@ref); one warp per plane:
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
its two green photosites in the visible raw's COLMAP frame, `G1` then `G2`
(the cell's column-major order). They are diagonal,
at `(±½, ∓½)` raw pixels from the cell centre `raw_frame_map(L)((column − ½, row − ½))`
in an RGGB or BGGR cell and `(±½, ±½)` in a GRBG or GBRG one, and the value is
the mean of the two — what a renderer models to predict it exactly. Each is the
green channel's [`bayer_plane_map`](@ref) of the mosaic at that pixel. A
`:demosaic` image is refused (`ArgumentError`).
"""
function green_cell_sites(L::LinearRaw, row::Integer, column::Integer)
    o = _geometry(L, "green_cell_sites").green_offsets
    x = pixel_centre(CartesianIndex(row, column))   # Core's one index → COLMAP centre
    grid = raw_frame_map(L)
    return (_sample_map(grid, Val(:G1), o[1])(x), _sample_map(grid, Val(:G2), o[2])(x))
end

"""
    raw_provenance(L::LinearRaw) -> NamedTuple

How the image was made, as plain data a result records: `decoder`,
`libraw_version` (from the mosaic's metadata; `nothing` where it states none),
`mode`, `gain` and `frame_map` (`(; from, to, scale, translation)` of
[`raw_frame_map`](@ref)`(L)`, `nothing` for `:demosaic`). Its `mode` is the
mode's own record's ([`raw_provenance`](@ref)`(LinearRaw{mode})`), and both
records' `frame_map` are written by one rule from the map, so an image of a
whole mosaic records the mode's `frame_map`; a window's records its offset in
`translation`.
"""
function raw_provenance(L::LinearRaw)
    by_mode = raw_provenance(typeof(L))
    return (decoder = get(L.metadata, :decoder, nothing), libraw_version = get(L.metadata, :libraw_version, nothing),
            mode = by_mode.mode, gain = L.gain, frame_map = _provenance_frame_map(L))
end

"""
    raw_provenance(::Type{LinearRaw{mode}}) -> (; mode, frame_map)

The part of [`raw_provenance`](@ref)`(L)` that the mode alone determines, without
an image: `mode` (`"green_cells"`, `"bin2"`, `"demosaic"`) and `frame_map`, the
record of [`raw_frame_map`](@ref)`(LinearRaw{mode})` (`(; from, to, scale,
translation)`; `nothing` for `:demosaic`, which holds no geometry). An image
made from a whole mosaic (no window) records the same `mode` and `frame_map`,
`==` field for field, whatever its Bayer phase or visible origin. For a result
whose frames have no RAW behind them that records the frame they are on.
"""
raw_provenance(::Type{<:LinearRaw{:green_cells}}) =
    (mode = "green_cells", frame_map = _frame_map_record(raw_frame_map(LinearRaw{:green_cells})))
raw_provenance(::Type{<:LinearRaw{:bin2}}) = (mode = "bin2", frame_map = _frame_map_record(raw_frame_map(LinearRaw{:bin2})))
raw_provenance(::Type{<:LinearRaw{:demosaic}}) = (mode = "demosaic", frame_map = nothing)

# THE record of a cell image's frame map (raw_frame_map, BayerCellFrame{:G} → PixelFrame{:colmap}) as plain data: the
# mode's and an image's provenance both write it.
_provenance_frame_map(::LinearRaw{:demosaic}) = nothing
_provenance_frame_map(L::LinearRaw) = _frame_map_record(raw_frame_map(L))
_frame_map_record(S::SimilarityMap) =
    (from = "BayerCellFrame{:G}", to = "PixelFrame{:colmap}", scale = Float64(scale(S)), translation = collect(Float64, translation(S)))
