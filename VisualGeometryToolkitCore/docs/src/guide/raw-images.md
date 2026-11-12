# Raw images

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The sensor image of the stack lives here, in Core: the decoded mosaic, its
colour phase, its photosite views, its linear values and the exact maps of
every sample onto the sensor. Decoding a RAW file is
VisualGeometryToolkitFeatures' (`load_raw_mosaic`, LibRaw through
`LibRaw_jll`), and so is the display demosaic (`load_raw(path or mosaic; mode
= :demosaic)`). The names below are exported by Core alone (Features
0.8.0 no longer re-exports them): `using VisualGeometryToolkitCore`.

Julia has no convention for raw (Bayer) images: JuliaImages reads none, and
BayeredImages.jl (unregistered) wraps a plain matrix as a `CFAImage <:
AbstractMatrix` with a periodic colour filter array. The stack keeps its own
types as its one raw representation, and takes one idea from BayeredImages:
the raw wrapper IS an `AbstractMatrix`, so an ordinary array or image function
works on it unchanged. There is no dependency on BayeredImages. From ImageCore
(a dependency of Core) it takes its conventions and, where the concept is
the same, its functions: see [ImageCore's conventions](@ref raw-imagecore).

## The representation

A decoded file is a [`RawMosaic`](@ref)`{A} <: AbstractMatrix{UInt16}`: the
photosite matrix `A` (an `Array`, a `CuArray`, or a view of either), its colour
pattern anchored at the full raw frame's origin, the visible area's origin in
that frame, the black and white levels and the decoder's metadata.

- `size`, `getindex`, `IndexStyle` and `parent` are the held array's.
- What an array function makes from it is a plain array of the held array's
  kind: `similar`, `copy`, a broadcast (`m .- black`), a reduction (`sum`,
  `extrema`). The colour pattern and the metadata stay with the mosaic, and
  nothing pretends a derived array still has them.
- A unit-range index `m[rows, columns]` (a copy) or `view(m, rows, columns)`
  (no copy) is a window: a `RawMosaic` of the same sensor whose
  `window_origin` is its offset into the visible area. Its colours are the
  parent's at those photosites, whatever the parity of its start (the phase
  shift BayeredImages makes with a `circshift` of the pattern is the offset
  here, added where the pattern is tiled). Any other index is a plain array.
- On a device: `adapt(CuArray, m)` or `adapt(backend, m)` (a KernelAbstractions
  backend) moves the photosites and keeps the rest. A kernel receives
  `parent(m)` or the mosaic's `channelview`, never the wrapper (its metadata is
  not a device value).

## The photosites channel-first: `channelview`

`channelview(m)` — a method of ImageCore's `channelview` for a mosaic — is
the photosites channel-first, `4 × H/2 × W/2`: each 2×2 Bayer cell is a pixel,
and its four photosites, spatially interleaved, are its channels. Photosite
`(2i − 1 + di, 2j − 1 + dj)` is `v[di + 2dj + 1, i, j]`, the cell's
photosites in column-major order, whose colours are
[`bayer_channels`](@ref)`(m)` — `(:R, :G1, :G2, :B)` for an RGGB cell,
`(:G1, :B, :R, :G2)` for GRBG.

It is built as ImageCore builds its views, with nothing copied: the photosite
matrix reshaped `H × W → 2 × H/2 × 2 × W/2`, its dimensions permuted to
`2 × 2 × H/2 × W/2` (a `PermutedDimsArray`), and reshaped to
`4 × H/2 × W/2`. A view of a device mosaic is a view of its device array:
KernelAbstractions' `get_backend` is the device's, `adapt(Array, v)` moves it
whole, and a broadcast over a whole mosaic's view runs on the device. A GPU
kernel reads the photosites by the view's index rule instead,
`parent(m)[2i − 1 + di, 2j − 1 + dj]`, as the cell kernel of
[`LinearRaw`](@ref) does: measured on a B200 (2848 × 4256, the cell kernel,
`benchmark/raw_channelview.jl`),

| the kernel reads | whole mosaic | window with an odd start |
|---|---|---|
| the view `v[k, i, j]` | 0.197 ms, 509 GB/s | does not compile (`InvalidIRError`) |
| its `PermutedDimsArray`, `parent(v)[di + 1, dj + 1, i, j]` | 0.073 ms, 1371 GB/s | 0.177 ms, 563 GB/s |
| a packed `4 × h × w` copy | 0.070 ms, 1426 GB/s | 0.072 ms, 1378 GB/s |
| the held matrix by the index rule | 0.072 ms, 1391 GB/s | 0.074 ms, 1355 GB/s |

Base's `ReshapedArray` indexing splits each index by multiplicative inverses,
and over a nested wrapper (a window's `SubArray` inside) it does not infer in a
CUDA kernel. Its inverse,
[`photositeview`](@ref), turns a `4 × h × w` array of cells back into the
`2h × 2w` photosite matrix — of a mosaic's view, the held array itself.

Why the channels are in the cell's own order and not always `R, G1, G2, B`:
reordering channels per phase is a gather, not a reshape or a permutation of
dimensions, so it cannot be a view. The colour of each channel is one call
away, `bayer_channels(m)`, from the same phase rule as
[`photosite_colors`](@ref).

Each channel is a lattice of its own, sampled at its photosite's position in
the cell: its frame is [`BayerCellFrame`](@ref)`{C}` (`C` its colour) and
[`bayer_plane_map`](@ref)`(m, k)` its isbits `SimilarityMap` onto the sensor
(`x ↦ 2x` plus the photosite's offset from the cell centre and a window's
offset), exact for every Bayer phase and either parity of the visible or the
window origin. The photosites' linear values channel-first are
`channelview(`[`linearview`](@ref)`(m))` (next section): the one
linearisation, the one the cell kernel applies.

```julia
v = channelview(m)                      # 4 × H/2 × W/2, a view of parent(m)
g1 = findfirst(==(:G1), bayer_channels(m))
S = bayer_plane_map(m, g1)              # G₁'s lattice → the sensor's PixelFrame{:colmap}
x = S(pixel_centre(CartesianIndex(i, j)))   # where G₁ of cell (i, j) sits on the sensor
y = channelview(linearview(m))[g1, i, j]   # its linear value
```

## Linear values on the raw grid: `linearview`

[`linearview`](@ref)`(m)` is the mosaic's linear values on the raw grid,
lazily, as ImageCore's `normedview` reads codes as numbers: element `(i, j)` is
`(raw − black[c]) / (white − black[c])` of photosite `(i, j)`, `c` its colour
(the black and white levels are the mosaic's fields `m.black`, per colour
index, and `m.white`). It is the one public spelling of the linearisation. It composes with `channelview`:
`channelview(linearview(m))` is the linear values channel-first, and the
green-cell image is the mean of its two G channels, bit for bit. A broadcast
over it, and `copy`, run on the mosaic's device, and a kernel reads it as it is.
[`photosite_colors`](@ref)`(m)` is lazy and isbits (the pattern and its
phase, indexed by the one rule), so it fuses into a broadcast with a device
mosaic's arrays and runs on the device. [`saturationview`](@ref)`(m)` is the linear view's `Bool` match: the saturation flags on the
same grid, lazily, by the one rule `raw ≥ white` that [`LinearRaw`](@ref)'s
flags apply. A photosite mask is a plain broadcast over
[`photosite_colors`](@ref) — the usable greens of a radiometric fit are
`((col .== 0x01) .| (col .== 0x03)) .& .!saturationview(m)` with
`col = photosite_colors(m)` — so no mask helper exists.

## Linear values on the cells

A [`LinearRaw`](@ref)`{Mode}` (`:green_cells`, `:bin2`, `:demosaic`) stays a
container of plain planes, `L.image[row, column, plane]`, beside its
saturation flags, geometry, gain and metadata, and is THE public path to the
green-cell image: `LinearRaw(m; mode = :green_cells)` (its `image[:, :, 1]`),
on the mosaic's device. Its cell images are the mosaic's `linearview` read
channel-first: the green-cell value is the mean
of the two green channels, times the gain, bit for bit (a test holds it), and
`:bin2`'s R and B planes are the R and B channels. The planes are not
wrappers that each know their [`BayerCellFrame`](@ref)`{C}`, for three reasons:

1. Every caller reads the planes as one array: a warp reads all of them in one
   call (`image_warp!(dst, warps, L.image)`, a tuple of one warp per plane),
   the decode stage writes `L.image` as one PNG, a detection image takes a view
   of one plane, and a quantile reads `vec(L.image)`. A
   tuple of plane wrappers would split that array or make each caller
   reassemble it.
2. A plane's frame is already one call away and needs nothing new:
   [`bayer_plane_map`](@ref)`(L, c)` and [`raw_frame_map`](@ref)`(L)` (the
   green plane, the cell grid).
3. The saturation flags, the gain and the green code spans belong to the cell,
   not to one plane; per-plane wrappers would hold copies of them or point
   back at the container.

The mosaic, by contrast, is one matrix, and the wrapper costs its callers
nothing.

## One phase rule

Every colour, offset and map above comes from one rule in
`src/raw/raw_mosaic.jl`: the colour of a photosite is the pattern tiled from
the raw origin, shifted by a window's offset ([`photosite_colors`](@ref)); a
cell's four photosites are taken in column-major order; and photosite `k` sits
at its 0-based site's pixel centre from the cell centre, `(dj − ½, di − ½)` raw
pixels. A lattice of samples at offset `o` from every cell centre maps onto the
sensor by the cell grid's `x ↦ 2x` after its translation by `o/2` cell pixels.
`channelview`'s channels, `bayer_plane_map` of a mosaic and of a `LinearRaw`,
[`green_cell_sites`](@ref) and the cell kernel all read that rule; the R
channel's map IS the `:bin2` R plane's.

## Where an image's sensor-pixel coordinates come from

ONE accessor: [`raw_coordinates`](@ref)`(img, x)`, the point `x` of the image's
pixel frame (COLMAP: pixel `(row, column)` centred at `(column − ½, row − ½)`)
in the visible raw's `PixelFrame{:colmap}`, the frame every calibration of the
raw is in. It is `raw_frame_map(img)(x)`:

| image | `raw_frame_map(img)` | `raw_coordinates(img, x)` |
|---|---|---|
| a `RawMosaic` as decoded | the identity | `x` |
| a window of a mosaic | its offset | `x + (columns, rows)` of `window_origin` |
| a `LinearRaw` cell image (`:green_cells`, `:bin2`) | the cell grid's map | `2x` (then a window's offset) |
| a cell mode alone, no image: `LinearRaw{:green_cells}`, `LinearRaw{:bin2}` | the cell grid's map (every whole-mosaic image's, bit for bit) | `2x` |
| a `:demosaic` image | refused (`ArgumentError`) | refused |
| a plain matrix | no method | no method: it states no frame |

A position on another plane of a `:bin2` image is `bayer_plane_map(L, c)(x)`,
and on a channel of a mosaic's `channelview` it is `bayer_plane_map(m, k)(x)`:
the planes are sampled at different points of the cell, so each has its own
frame, and a composition that joins two planes' frames is refused. A mask over
a plane's pixels of a region on the sensor is
[`pixel_mask`](@ref)`(region, bayer_plane_map(m, k), w, h)`: each pixel's
mapped centre tested in the region's own frame. A region moves between frames
by the map itself (`S(rect)`, `S(a ∩ b)`).

## How detectors report positions

A detector works in the pixels of the image it is given and returns COLMAP
pixel centres of that image. A caller that runs it on a raw image and writes
positions as raw pixels maps them through the image first:

```julia
L = load_raw(path; mode = :green_cells)        # VisualGeometryToolkitFeatures
xs = detect(L.image[:, :, 1])                  # any detector: COLMAP centres of the cell grid
raw_xs = raw_coordinates.(Ref(L), xs)          # the sensor's pixels
```

Writing the cell-grid positions as raw pixels unmapped puts them at half the
scale; RadiometricCamera's chrome detection did exactly that with green-cell
coordinates. A calibration matrix moves the same way:
`reframed_calibration(raw_frame_map(L), K_cells)`.

## How the frame survives a file round trip

A file of a decoded image (a PNG of the cells) holds pixels and no frame, and a
plain matrix read back from it has no `raw_coordinates`. The stage that writes
it records the frame with it: [`raw_provenance`](@ref)`(L)` gives the decoder,
the mode, the gain and the frame map as plain data (`from`, `to`, `scale`,
`translation`), for the stage's record (RadiometricCamera's `conversion.json`).
The stage that reads the file rebuilds the map from that record and maps every
position it writes through it.

## [ImageCore's conventions](@id raw-imagecore)

ImageCore is a dependency of Core, and the sensor image follows it where the
concept is the same, by its own functions:

| ImageCore | here | why |
|---|---|---|
| `channelview`, channel-first `C × H × W` | a method, `channelview(::RawMosaic)`, `4 × H/2 × W/2` | the same concept: the channels of each pixel, first, as a view |
| views, not copies | `channelview(m)`, `photositeview`, windows by `view` | the same rule |
| `colorview` (the inverse of `channelview`) | [`photositeview`](@ref), a name of its own | `colorview(C, A)` makes colorants of channels; the inverse here makes the photosite matrix of cells, no colorant |
| `rawview` | not extended | a mosaic IS its raw `UInt16` codes: there is nothing to unwrap |
| `normedview(N0f16, A)` | usable on `parent(m)` as ImageCore defines it | it is `code / 65535`; the linear value subtracts each channel's black level and divides by its span, so it has its own view, [`linearview`](@ref) |

`LinearRaw{:bin2}`'s image stays `H × W × 3` (planes last) in this release:
moving it channel-first is a breaking change to a stored layout, left for a
breaking release (Core's CHANGELOG, 0.9.2).
