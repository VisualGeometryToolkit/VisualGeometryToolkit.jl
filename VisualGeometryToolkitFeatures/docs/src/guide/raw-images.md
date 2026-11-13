# Raw images

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

The sensor image — the decoded mosaic (`RawMosaic`), its colour phase, its
photosites channel-first (a method of ImageCore's `channelview`), its linear
values on the 2×2 cells (`LinearRaw`) and the exact maps of every plane and
channel onto the sensor (`raw_frame_map`, `bayer_plane_map`,
`green_cell_sites`, `raw_coordinates`) — is VisualGeometryToolkitCore's since
Core 0.9.2 and Features 0.8.0, so every package below Features reads the image
without the decoder. Its design is Core's "Raw images" guide. Core alone exports
those names (`using VisualGeometryToolkitCore`); this package no longer
re-exports them (0.8.0).

What stays here is what needs LibRaw:

- **The decoder.** [`load_raw_mosaic`](@ref)`(path)` decodes a file with
  [`LibRawDecoder`](@ref) (the `LibRaw_jll` extension: `ccall`, no Python) into
  Core's `RawMosaic`: the visible photosites in the sensor's orientation, the
  colour pattern from the raw origin, the per-channel black levels (LibRaw's
  `adjust_bl`; a spatial black pattern that has no per-channel form is
  refused) and the white level, and the EXIF fields
  ([`exposure_settings`](@ref), [`lens_settings`](@ref), …;
  [`load_raw_metadata`](@ref) reads them without the photosites).
- **The one RAW entry.** [`load_raw`](@ref)`(path; mode, gain)` is the decode
  followed by Core's `LinearRaw(mosaic; mode, gain)` for the exact cell modes,
  with an optional decode cache ([`FileCache`](@ref)).
- **The display demosaic.** `load_raw(path; mode = :demosaic)` is LibRaw's
  AHD interpolation of the visible area at full resolution, linear, for display
  figures only (it has no frame map). `load_raw(mosaic; mode = :demosaic)` is
  the same demosaic of a mosaic in memory: its photosites are handed to LibRaw
  as a Bayer buffer with its colour phase and levels, so a mosaic corrected per
  photosite (vignetting applied, say) is displayed through the one demosaic,
  never a second one. LibRaw holds whole levels, so a mosaic whose black or
  white level is not a whole code is refused.
- **The sensor model.** The frame's `SensorModel` (VisualGeometryToolkitCore's
  `sensor_model`, a lookup in the body's characterisation) goes to
  [`load_raw`](@ref)'s `sensor_model`: Core's one black subtraction reads its
  black, joins its defects to `saturated`, and under a Poisson–Gaussian noise
  returns each value with its variance.
- **The detection image.** [`variance_stabilize`](@ref)`(L, sensor_model)` of
  the green cells, which needs the model's gain.

## How detectors report positions

A detector works in the pixels of the image it is given and returns COLMAP
pixel centres of that image; positions on the green cells go to the sensor
through the image's map:

```julia
using VisualGeometryToolkitFeatures            # load_raw
using VisualGeometryToolkitCore                # raw_coordinates, LinearRaw, …
L = load_raw(path; mode = :green_cells)
xs = detect(L.image[:, :, 1])                  # any detector: COLMAP centres of the cell grid
raw_xs = raw_coordinates.(Ref(L), xs)          # the sensor's pixels
```

Writing the cell-grid positions as raw pixels unmapped puts them at half the
scale; RadiometricCamera's chrome detection did exactly that with green-cell
coordinates.
