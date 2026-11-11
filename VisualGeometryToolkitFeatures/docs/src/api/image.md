# Image Input

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

The image-input helpers, the linear-radiance PNG reader (`load_linear_image`)
and the camera RAW reader (`load_raw`).

## Camera RAW

The stack's one RAW path: every consumer reads RAW files through this module,
and VisualGeometryToolkitCore's `check_raw_decoding` flags any other decoding
path. `load_raw(path; mode, gain)` reads a Bayer RAW file in the sensor's
orientation (the EXIF flip is never applied). The decoder is
[`LibRawDecoder`](@ref) (`using LibRaw_jll`: LibRaw 0.22 through `ccall`, no
Python). Everything after the decode is Julia code here.

| mode | image | use |
|---|---|---|
| `:green_cells` | each 2×2 Bayer cell's two greens, averaged; half resolution, one plane | geometry and radiometry |
| `:bin2` | the same cells with their red and blue (R, the green-cell plane, B) | colour radiometry |
| `:demosaic` | LibRaw's AHD demosaic, full resolution, linear | display figures only |

The cell modes are exact: `(raw − black[c]) / (white − black[c]) × gain`, with
saturation flags and each plane's true sample position. Their pixel frame,
the cell grid [`BayerCellFrame`](@ref)`{:G}`, maps onto the visible raw's
`PixelFrame{:colmap}` by [`raw_frame_map`](@ref), `x ↦ 2x` exactly, for every
Bayer phase and either parity of the visible origin. Its inverse is `inv`, and
a calibration matrix moves with it through Core's
`reframed_calibration(raw_frame_map(L), K)`. Each plane is a frame of its own,
`BayerCellFrame{C}`, and [`bayer_plane_map`](@ref)`(L, c)` is its map onto the
raw: the plane's translation onto the cell grid (`(±¼, ±¼)` cells for R and
B, none for G) composed with `raw_frame_map(L)`. A warp reads plane `c`
through `inv(bayer_plane_map(L, c)) ∘ math` like any other map, all planes in
one call (`image_warp!(dst, warps, L.image)`, a tuple of one warp per plane);
a composition that joins two planes' frames is refused. A
green-cell value is the mean of two diagonal photosites, at `(±½, ∓½)` raw
pixels from the cell centre for an RGGB or BGGR cell and `(±½, ±½)` for GRBG or
GBRG; [`green_cell_sites`](@ref) gives them, so a renderer can model the
footprint exactly. The demosaic interpolates. It has no frame map, so the
geometric accessors refuse it. [`raw_provenance`](@ref) records the decoder,
LibRaw's version, the mode and the frame map.

The EXIF fields become VisualGeometryToolkitCore's typed values through
[`exposure_settings`](@ref), [`lens_settings`](@ref), [`body_identity`](@ref),
[`lens_identity`](@ref) and [`capture_time`](@ref), from a mosaic or from
[`load_raw_metadata`](@ref) (the headers only); [`extract_capture_settings`](@ref)
gathers a capture's frames into a `CaptureSettings`. A lens's EXIF name is a
claim: on an adapter (the FX3 with Sigma EF lenses) it names another lens, so a
pipeline that knows its lens states it.

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["raw_image.jl"]
```

## The detection image of a RAW

[`variance_stabilize`](@ref) maps the green cells to the generalized Anscombe
transform for Poisson–Gaussian noise: the noise has σ ≈ 1 at every level, so
ASIFT on it ([`asift_detect`](@ref) of a [`StabilizedImage`](@ref)) thresholds
its DoG at `noise_sigmas` noise σ, in a dim interior as in a bright window, and
masks the saturated pixels. A descriptor still reads a display rendering of the
same cells: both are pixel-wise, so the positions coincide.

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["variance_stabilization.jl"]
```

## Image input and the linear PNG reader

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["image_input.jl", "linear_image.jl"]
```
