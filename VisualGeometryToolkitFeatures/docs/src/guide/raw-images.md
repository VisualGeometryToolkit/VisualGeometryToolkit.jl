# Raw images

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

Julia has no convention for raw (Bayer) images: JuliaImages reads none, and
BayeredImages.jl (unregistered) wraps a plain matrix as a `CFAImage <:
AbstractMatrix` with a periodic colour filter array. The stack keeps its own
types, in this package, as its one raw representation, and takes one idea from
BayeredImages: the raw wrapper IS an `AbstractMatrix`, so an ordinary array or
image function works on it unchanged. There is no dependency on BayeredImages.

## The representation

A decoded file is a [`RawMosaic`](@ref)`{A} <: AbstractMatrix{UInt16}`: the
photosite matrix `A` (an `Array`, a `CuArray`, or a view of either), its colour
pattern anchored at the full raw frame's origin, the visible area's origin in
that frame, the black and white levels and the EXIF metadata.

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
  backend) moves the photosites and keeps the rest. A kernel receives `parent(m)`,
  never the wrapper (its metadata is not a device value): the cell kernel of
  [`LinearRaw`](@ref) runs on the photosites' device and leaves its planes and
  saturation flags there.

A [`LinearRaw`](@ref)`{Mode}` (`:green_cells`, `:bin2`, `:demosaic`) stays a
container of plain planes, `L.image[row, column, plane]`, beside its
saturation flags, geometry, gain and metadata. Its planes are not wrappers
that each know their [`BayerCellFrame`](@ref)`{C}`, for three reasons:

1. Every caller reads the planes as one array: a warp reads all of them in one
   call (`image_warp!(dst, warps, L.image)`, a tuple of one warp per plane),
   the decode stage writes `L.image` as one PNG, `variance_stabilize` takes a
   view of one plane, and a quantile reads `vec(L.image)`. A tuple of plane
   wrappers would split that array or make each caller reassemble it.
2. A plane's frame is already one call away and needs nothing new:
   [`bayer_plane_map`](@ref)`(L, c)` and [`raw_frame_map`](@ref)`(L)` (the
   green plane, the cell grid).
3. The saturation flags, the gain and the green code spans belong to the cell,
   not to one plane; per-plane wrappers would hold copies of them or point
   back at the container.

The mosaic, by contrast, is one matrix, and the wrapper costs its callers
nothing.

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
| a `:demosaic` image | refused (`ArgumentError`) | refused |
| a plain matrix | no method | no method: it states no frame |

A position on another plane of a `:bin2` image is
`bayer_plane_map(L, c)(x)`: the planes are sampled at different points of the
cell, so each has its own frame, and a composition that joins two planes'
frames is refused.

## How detectors report positions

A detector works in the pixels of the image it is given and returns COLMAP
pixel centres of that image. A caller that runs it on a raw image and writes
positions as raw pixels maps them through the image first:

```julia
L = load_raw(path; mode = :green_cells)
xs = detect(L.image[:, :, 1])                  # any detector: COLMAP centres of the cell grid
raw_xs = raw_coordinates.(Ref(L), xs)          # the sensor's pixels
```

Writing the cell-grid positions as raw pixels unmapped puts them at half the
scale; RadiometricCamera's chrome detection did exactly that with green-cell
coordinates. A calibration matrix moves the same way, through Core:
`reframed_calibration(raw_frame_map(L), K_cells)`.

## How the frame survives a file round trip

A file of a decoded image (a PNG of the cells) holds pixels and no frame, and a
plain matrix read back from it has no `raw_coordinates`. The stage that writes
it records the frame with it: [`raw_provenance`](@ref)`(L)` gives the decoder,
the mode, the gain and the frame map as plain data (`from`, `to`, `scale`,
`translation`), for the stage's record (RadiometricCamera's `conversion.json`).
The stage that reads the file rebuilds the map from that record and maps every
position it writes through it.

Next step: RadiometricCamera's decode stage records `raw_provenance(L)` per
photograph (today its record holds the size and gain, not the map), and a
reader of a provenance record returns the `SimilarityMap` it states, so a
stage reading the PNG asks `raw_coordinates` of the decoded record rather than
assuming a scale.
