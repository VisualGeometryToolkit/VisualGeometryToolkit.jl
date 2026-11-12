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
Python). It produces VisualGeometryToolkitCore's `RawMosaic`, and every number
after the decode is Core's: the mosaic's colour phase and its photosites
channel-first (`channelview`), the linear values on the 2×2 cells
(`LinearRaw`) and the exact maps onto the sensor (`raw_frame_map`,
`bayer_plane_map`, `green_cell_sites`, `raw_coordinates`). Those names are
re-exported here, the same bindings; their reference and design are Core's
"Camera RAW" API page and "Raw images" guide, and this package's
[Raw images](@ref) page says what is left here.

| mode | image | use |
|---|---|---|
| `:green_cells` | each 2×2 Bayer cell's two greens, averaged; half resolution, one plane | geometry and radiometry |
| `:bin2` | the same cells with their red and blue (R, the green-cell plane, B) | colour radiometry |
| `:demosaic` | LibRaw's AHD demosaic, full resolution, linear | display figures only |

The cell modes are exact: `(raw − black[c]) / (white − black[c]) × gain`, with
saturation flags and each plane's true sample position, on the cell grid that
maps onto the visible raw's `PixelFrame{:colmap}` by `x ↦ 2x` exactly. The
demosaic interpolates. It has no frame map, so the geometric accessors refuse
it. It takes a file or a mosaic in memory, `load_raw(mosaic; mode =
:demosaic)`: a mosaic corrected per photosite (vignetting, say) is displayed
through the same LibRaw demosaic, on its photosites, phase and levels.

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

[`variance_stabilize`](@ref) maps the green cells (Core's `green_cells(L)`) to the generalized Anscombe
transform for Poisson–Gaussian noise: the noise has σ ≈ 1 at every level, so
ASIFT on it ([`asift_detect`](@ref) of a [`StabilizedImage`](@ref)) thresholds
its DoG at `noise_sigmas` noise σ, in a dim interior as in a bright window, and
masks the saturated pixels. A descriptor still reads a display rendering of the
same cells: both are pixel-wise, so the positions coincide.

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["variance_stabilization.jl"]
```

## Image input, the linear PNG reader and the display encoding

A linear image is shown through one encoding, [`srgb_image`](@ref): exposure,
a white balance per plane, an optional tone curve, clamp, the sRGB transfer
curve ([`linear_to_srgb`](@ref), the inverse of [`srgb_to_linear`](@ref)) and
8 or 16 bits, a NaN pixel the background.

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["image_input.jl", "linear_image.jl"]
```
