# =============================================================================
# The sensor image — a Bayer mosaic, its colour phase and its photosite views
# =============================================================================
#
# THE stored RAW image of the stack: `RawMosaic`, an AbstractMatrix of the
# visible photosites that knows its colour pattern, its levels and where it
# sits on the sensor. Decoding a file into one is VisualGeometryToolkitFeatures'
# (its LibRaw_jll extension, `load_raw_mosaic`); everything after the decode —
# the colour phase, the frames, the per-photosite views, the linear values
# (linear_raw.jl) — is here, so every package below Features reads the image
# without the decoder.
#
# ONE phase rule: `_photosite_color` (the pattern tiled from the raw origin,
# shifted by a window's offset), the cell's four sites `_QUAD_SITES` in
# column-major order, and each site's offset from the cell centre
# `_site_offset`. The channels of a mosaic's `channelview`, `bayer_plane_map` (of a mosaic and of
# a `LinearRaw`), `green_cell_sites` and the cell kernel all read them; nothing
# restates an offset or a phase.
#
# ImageCore's conventions, through ImageCore itself (a dependency of Core; the
# CHANGELOG, 0.9.2, lists each adopted and each not, with its reason): the
# photosites channel-first are a METHOD of `ImageCore.channelview` (each 2×2
# cell a pixel, its four photosites the channels, `4 × H/2 × W/2`), a view, not
# a copy, on the mosaic's device. Its inverse is `photositeview`, a name of its
# own: `colorview` builds colorants, and the inverse here is a photosite matrix.
# `rawview` has nothing to unwrap (the mosaic IS its UInt16 codes) and
# `normedview(N0f16, parent(m))` is ImageCore's `code / 65535`, not the linear
# value, which has its own name, `linear_value`.
# =============================================================================

"""
    RawMosaic{A <: AbstractMatrix{UInt16}, M <: NamedTuple} <: AbstractMatrix{UInt16}

A RAW file decoded and nothing more: the visible area of the sensor, as the
decoder reports it (VisualGeometryToolkitFeatures' `load_raw_mosaic`, LibRaw).
It IS its photosite matrix: `size`, indexing and `parent` are the
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
keeps the rest); a kernel receives `parent(m)`, never the wrapper (nor,
on a GPU, the nested view `channelview(m)`: read its photosites by its index rule).

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
- `metadata::NamedTuple` — what the decoder read: the EXIF fields
  (VisualGeometryToolkitFeatures types them: `exposure_settings`, `lens_settings`,
  `body_identity`, `lens_identity`, `capture_time`), the daylight white balance,
  the camera's declared linear-response limit, the decoder and its library's
  version. Core reads none of it except in [`raw_provenance`](@ref).
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
Adapt.adapt_structure(to, m::RawMosaic) = _with_raw(m, Adapt.adapt(to, m.raw), m.window_origin)

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

# THE phase rule: the colour index of photosite (i, j), the pattern tiled from the raw origin.
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

The COLMAP pixel frame of one colour plane `C` of the half-resolution grid of
2×2 Bayer cells: pixel `(row, column)` of the plane is centred at
`(column − ½, row − ½)` of its frame. The planes are indexed alike but sampled
at different points of the cell, so each is a frame of its own:

- `:R`, `:B` — a cell's red and blue photosite (a plane of a `:bin2`
  [`LinearRaw`](@ref), a channel of a mosaic's [`channelview`](@ref ImageCore.channelview(::RawMosaic)));
- `:G1`, `:G2` — a cell's two green photosites, in the cell's column-major
  order (the two green channels of `channelview(m)`);
- `:G` — the mean of the two greens, which sits at the cell centre in every
  phase: the cell grid itself (the `:green_cells` image), whose map is
  [`raw_frame_map`](@ref).

[`bayer_plane_map`](@ref) maps each onto the visible raw's
`PixelFrame{:colmap}`, and a composition that joins one plane's frame to
another's is refused (`MethodError`), as at every frame junction of Core.
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

# A cell's four photosites in column-major order, k = di + 2dj + 1: THE order of channelview's
# channels, of the cell kernel's reads and of a LinearRaw's green offsets.
const _QUAD_SITES = ((0, 0), (1, 0), (0, 1), (1, 1))

# Photosite k of a cell, (di, dj) = _QUAD_SITES[k], is at raw COLMAP (2j − 1.5 + dj, 2i − 1.5 + di):
# (dj − ½, di − ½) raw pixels from the cell centre — the 0-based site's pixel centre (Core's conversion).
@inline _site_offset(k) = pixel_centre(CartesianIndex(_QUAD_SITES[k]))

# The cell grid's map onto the visible raw: x ↦ 2x, then the mosaic's own map (a window's offset).
_cell_grid_map(m::RawMosaic) = raw_frame_map(m) ∘ _CELL_FRAME_MAP

# THE map of a lattice of samples on the cell grid onto the raw: the samples at `o` raw pixels from
# every cell centre (o / 2 in cell pixels: the ×2 stated once, in `_CELL_FRAME_MAP`), as frame
# `BayerCellFrame{C}`, onto the grid, then the grid's map `cells`. The green mean, at the centre, IS the grid.
_sample_map(cells, ::Val{:G}, o) = cells
_sample_map(cells, ::Val{C}, o) where {C} =
    cells ∘ relabel(BayerCellFrame{C}, BayerCellFrame{:G}, SimilarityMap(Translation(o / scale(_CELL_FRAME_MAP))))

# The cell's colour indices at its four sites (the same in every cell of a pattern that repeats with
# period 2 from the raw origin), checked: one R, two G and one B. `what` names the caller in a refusal.
function _cell_colors(m::RawMosaic, what)
    n = size(m.pattern, 1)
    (iseven(n) && all(m.pattern[i, j] == m.pattern[mod1(i, 2), mod1(j, 2)] for i in 1:n, j in 1:n)) ||
        throw(ArgumentError("$what needs a 2×2 Bayer pattern, got the colour pattern $(m.pattern)"))
    cell = ntuple(k -> _photosite_color(m, 1 + _QUAD_SITES[k][1], 1 + _QUAD_SITES[k][2]), Val(4))
    count(==(0x00), cell) == 1 && count(==(0x02), cell) == 1 ||
        throw(ArgumentError("$what needs one R, two G and one B per cell, got $([cell[1] cell[3]; cell[2] cell[4]])"))
    return cell
end

# Channel k's plane name: R, B, and the greens G1, G2 in the cell's column-major order.
function _channel_names(cell::NTuple{4, UInt8})
    greens = 0
    return ntuple(Val(4)) do k
        c = cell[k]
        c == 0x00 ? :R : c == 0x02 ? :B : (greens += 1; greens == 1 ? :G1 : :G2)
    end
end

"""
    bayer_channels(m::RawMosaic) -> NTuple{4, Symbol}

The colour of each channel of `channelview(m)`: `:R`, `:G1`, `:G2`,
`:B`, in the channels' order, the cell's photosites column-major (`(0, 0)`,
`(1, 0)`, `(0, 1)`, `(1, 1)` from the cell's top-left). `(:R, :G1, :G2, :B)`
for an RGGB cell, `(:G1, :B, :R, :G2)` for GRBG; the greens are numbered in
that order. The cell's phase is the mosaic's own at its first photosite, so a
window with an odd start reads another phase (from the same pattern, the one
phase rule of [`photosite_colors`](@ref)). A pattern that is not a 2×2 Bayer
cell (one R, two G, one B) is an `ArgumentError`.
"""
bayer_channels(m::RawMosaic) = _channel_names(_cell_colors(m, "bayer_channels"))

"""
    channelview(m::RawMosaic) -> AbstractArray{UInt16, 3}

ImageCore's `channelview`, for a Bayer mosaic: the photosites channel-first,
`4 × H ÷ 2 × W ÷ 2`, without a copy. The Bayer interpretation of "channels":
each 2×2 cell is a pixel and its four photosites — spatially interleaved, each
at its own position on the sensor — are its channels, so `v[k, i, j]` is
photosite `k` of cell `(i, j)`, the cell's photosites in column-major order
(`k = di + 2dj + 1` for photosite `(2i − 1 + di, 2j − 1 + dj)`): the mosaic's
own phase order, whose colours are [`bayer_channels`](@ref)`(m)` (`R, G1, G2,
B` for an RGGB cell; another phase permutes them, since reordering channels is
not a view). It is built as ImageCore builds a view: a reshape of the photosite
matrix (`H × W → 2 × H/2 × 2 × W/2`), its dimensions permuted
(`→ 2 × 2 × H/2 × W/2`, a `PermutedDimsArray`) and reshaped (`→ 4 × H/2 × W/2`),
views of `parent(m)`: writing through it writes the mosaic's held array, a
device mosaic's view is a view of its device array (KernelAbstractions'
`get_backend` is the device's, `adapt(Array, v)` moves it whole), and a
broadcast or a Cartesian copy over a whole mosaic's view runs on the device. An
odd last row or column (no 2×2 cell) is left out, as in [`LinearRaw`](@ref).

In a GPU kernel read the photosites by the view's index rule instead,
`parent(m)[2i − 1 + di, 2j − 1 + dj]` (or the 4-D `parent(v)[di + 1, dj + 1, i, j]`):
Base's `ReshapedArray` indexing compiles in a CUDA kernel only over a whole
contiguous array, at about a third of the held matrix's bandwidth (509 against
1391 GB/s on a B200, 2848 × 4256), and not at all over a window or an odd-sized
mosaic (an `InvalidIRError`); `benchmark/raw_channelview.jl` measures it, and
[`LinearRaw`](@ref)'s cell kernel reads the held matrix.

A plain `AbstractMatrix{UInt16}` keeps ImageCore's own method (a gray image's
channel view is the array itself): this method is the mosaic's, whose phase
says which photosite is which. Channel `k`'s pixel frame is
[`BayerCellFrame`](@ref)`{C}` (`C` its colour), and
[`bayer_plane_map`](@ref)`(m, k)` maps it onto the visible raw, exactly, for
every Bayer phase and either parity of a window's origin. Its inverse is
[`photositeview`](@ref). The linear value of a photosite is
[`linear_value`](@ref)`(v[k, i, j], black[k], span[k])` with
`(black, span) = `[`bayer_levels`](@ref)`(m)`.
"""
ImageCore.channelview(m::RawMosaic) = _cell_channels(parent(m))

# A photosite matrix's 2×2 cells channel-first (a mosaic's held array, or its linear view): THE view of both.
function _cell_channels(raw::AbstractMatrix)
    H, W = size(raw)
    h, w = H ÷ 2, W ÷ 2
    h >= 1 && w >= 1 || throw(ArgumentError("channelview: a $(H)×$(W) mosaic holds no 2×2 cell"))
    cells = (H, W) == (2h, 2w) ? raw : view(raw, 1:(2h), 1:(2w))
    return reshape(PermutedDimsArray(reshape(cells, 2, h, 2, w), (1, 3, 2, 4)), 4, h, w)
end

"""
    photositeview(v::AbstractArray{T, 3}) -> AbstractMatrix{T}

The inverse of a mosaic's [`channelview`](@ref ImageCore.channelview(::RawMosaic)),
in the role ImageCore's `colorview` plays for an image's (a name of its own:
`colorview` makes colorants of channels, and this makes the photosite matrix of
cells): a channel-first array of 2×2 cells (`4 × h × w`, each column a
cell's photosites in column-major order) as the `2h × 2w` photosite matrix,
without a copy. Of a mosaic's `channelview` it is the mosaic's (even-sized) held array
again; of any other array (per-photosite values computed channel-first, say)
it is a view in the mosaic's layout, `photositeview(v)[2i − 1 + di, 2j − 1 + dj]
== v[di + 2dj + 1, i, j]`. A first dimension other than 4 is an `ArgumentError`.
"""
function photositeview(v::AbstractArray{<:Any, 3})
    size(v, 1) == 4 || throw(ArgumentError("photositeview: a cell has 4 photosites (the first dimension), got $(size(v, 1))"))
    h, w = size(v, 2), size(v, 3)
    return reshape(PermutedDimsArray(reshape(v, 2, 2, h, w), (1, 3, 2, 4)), 2h, 2w)
end
# a mosaic's channelview: its permutation undone, the reshaped photosite matrix itself
photositeview(v::Base.ReshapedArray{T, 3, <:PermutedDimsArray{T, 4, (1, 3, 2, 4), (1, 3, 2, 4)}}) where {T} =
    reshape(parent(parent(v)), 2size(v, 2), 2size(v, 3))

"""
    bayer_plane_map(m::RawMosaic, k::Integer) -> SimilarityMap{2, …, BayerCellFrame{C}, PixelFrame{:colmap}}

The map of channel `k` of `channelview(m)` onto the visible raw's
`PixelFrame{:colmap}`: pixel `(i, j)` of the channel, at `(j − ½, i − ½)` of its
frame [`BayerCellFrame`](@ref)`{C}` (`C = bayer_channels(m)[k]`), goes onto
its photosite's centre, `(2j − 1 + dj − ½, 2i − 1 + di − ½)` plus a window's
offset, exactly, for every Bayer phase and either parity of the visible or
window origin. It is the cell grid's map `x ↦ 2x` after the photosite's
translation from the cell centre (`(±¼, ±¼)` cell pixels): the rule the planes
of a [`LinearRaw`](@ref) share, so the R channel's map IS the R plane's of the
mosaic's `:bin2` image. An isbits value, usable inside a kernel. A channel past
the fourth is an `ArgumentError`.
"""
function bayer_plane_map(m::RawMosaic, k::Integer)
    1 <= k <= 4 || throw(ArgumentError("bayer_plane_map: a mosaic's channelview has 4 channels, got channel $k"))
    return _sample_map(_cell_grid_map(m), Val(bayer_channels(m)[k]), _site_offset(k))
end

"""
    bayer_levels(m::RawMosaic) -> (black::NTuple{4, Float64}, span::NTuple{4, Float64})

The levels of the four channels of `channelview(m)`, in the channels'
order: each channel's black level (`m.black` of its colour index) and its span
`white − black`, the codes that run from `0` to `1` in
[`linear_value`](@ref). The constants the cell kernel of [`LinearRaw`](@ref)
reads, computed once per mosaic.
"""
function bayer_levels(m::RawMosaic)
    cell = _cell_colors(m, "bayer_levels")
    black = ntuple(k -> m.black[cell[k] + 1], Val(4))
    return (black, ntuple(k -> m.white - black[k], Val(4)))
end

"""
    linear_value(code, black::Float64, span::Float64) -> Float64

A photosite's linear value, `(code − black) / span`: `0` at the black level,
`1` at the white level (`span = white − black`), not clipped. THE
linearisation of the stack — convert_raw.py's expression, in its order — which
the cell kernel of [`LinearRaw`](@ref) applies to every photosite it reads;
a per-photosite consumer applies it to a mosaic's `channelview` with
[`bayer_levels`](@ref). Allocation- and branch-free (kernel-lawful).
"""
@inline linear_value(code, black::Float64, span::Float64) = (Float64(code) - black) / span

# THE saturation rule: a photosite is saturated at the white level, `code ≥ white`, with no margin (LinearRaw's flags,
# the green-photosite mask).
@inline _at_white(code, white::Float64) = code >= white

# THE cell site of photosite (i, j): k = di + 2dj + 1 with (di, dj) its parity in the cell grid (_QUAD_SITES[k]),
# the cells starting at the mosaic's first photosite.
@inline _quad_site(i, j) = mod(i - 1, 2) + 2mod(j - 1, 2) + 1

"""
    LinearRawView{A} <: AbstractMatrix{Float64}

The lazy linear view of a mosaic's photosites on the raw grid,
[`linearview`](@ref)`(m)`: element `(i, j)` is [`linear_value`](@ref) of the
held array's code there, with its channel's black level and span, computed when
read. It holds the photosites (`parent`, the mosaic's held array, a device array
too) and the four channels' levels; `similar`, a broadcast and `copy` are the
held array's kind (a broadcast over a device view runs on the device),
`adapt` moves it, and a KernelAbstractions kernel reads it as it is (its
indexing is two parities and one `linear_value`).
"""
struct LinearRawView{A <: AbstractMatrix} <: AbstractMatrix{Float64}
    raw::A
    black::NTuple{4, Float64}
    span::NTuple{4, Float64}
end
Base.size(v::LinearRawView) = size(v.raw)
Base.parent(v::LinearRawView) = v.raw
Base.IndexStyle(::Type{<:LinearRawView}) = IndexCartesian()
Base.@propagate_inbounds function Base.getindex(v::LinearRawView, i::Int, j::Int)
    k = _quad_site(i, j)
    return linear_value(v.raw[i, j], v.black[k], v.span[k])
end
Base.similar(v::LinearRawView, ::Type{T}, dims::Dims) where {T} = similar(v.raw, T, dims)
Base.Broadcast.BroadcastStyle(::Type{<:LinearRawView{A}}) where {A} = Base.Broadcast.BroadcastStyle(A)
Base.copy(v::LinearRawView) = identity.(v)
Adapt.adapt_structure(to, v::LinearRawView) = LinearRawView(Adapt.adapt(to, v.raw), v.black, v.span)
ImageCore.channelview(v::LinearRawView) = _cell_channels(v)

"""
    linearview(m::RawMosaic) -> LinearRawView

The mosaic's linear values on the raw grid, lazily, as ImageCore's
`normedview` reads codes as numbers without a copy: element `(i, j)` is
`(raw − black[c]) / (white − black[c])` of photosite `(i, j)`, `c` its colour
— [`linear_value`](@ref) with [`bayer_levels`](@ref)`(m)`, the one
linearisation, the one [`LinearRaw`](@ref)'s cell kernel applies. Not clipped,
no gain. It composes with ImageCore's `channelview`:
`channelview(linearview(m))` is the linear values channel-first
(`4 × H/2 × W/2`, the channels of `channelview(m)`), so the green-cell value of
cell `(i, j)` is the mean of its two green channels, bit for bit
(`LinearRaw(m; mode = :green_cells).image[i, j, 1]`, at gain 1). An odd last
row or column is in the view (its photosites' colours by the same rule) and
left out of the channel view.

Not `normedview(N0f16, …)` itself, whose `code / 65535` ignores the black
levels, nor a `MappedArrays.mappedarray` of the codes, whose function sees the
value alone: the black level and span depend on the photosite's position (its
colour in the phase), so the view reads its index — one parity rule, the cell
sites' (`k = di + 2dj + 1`).
"""
function linearview(m::RawMosaic)
    black, span = bayer_levels(m)
    return LinearRawView(parent(m), black, span)
end

"""
    green_photosite_mask(m::RawMosaic) -> AbstractMatrix{Bool}

The usable green photosites of a mosaic: `true` at a photosite whose colour is
green (`photosite_colors(m)` 1 or 3, either green of the cell) and whose code is
below the white level — the saturation rule of [`LinearRaw`](@ref)'s flags,
`raw ≥ white` with no margin, applied per photosite. A `BitMatrix` on the
host, a `Bool` array on a device mosaic's device; the size of `m` (an odd last
row or column included).
"""
function green_photosite_mask(m::RawMosaic)
    _cell_colors(m, "green_photosite_mask")
    green = similar(parent(m), Bool, size(m))
    copyto!(green, [c == 0x01 || c == 0x03 for c in photosite_colors(m)])
    green .&= .!_at_white.(parent(m), m.white)
    return green isa Array{Bool} ? BitMatrix(green) : green
end
