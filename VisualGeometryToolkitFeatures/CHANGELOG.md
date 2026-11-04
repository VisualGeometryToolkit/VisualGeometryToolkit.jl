# VisualGeometryToolkitFeatures changelog

## 0.4.1 (unreleased)

- **New `load_raw(path; mode, gain = 1, decoder = LibRawDecoder())`**: a camera
  RAW file (any Bayer format LibRaw reads) as exact linear values,
  `(raw − black[c]) / (white − black[c]) × gain`, camera raw colour, white
  balance 1, in the sensor's orientation (no EXIF flip; the raw's COLMAP frame).
  Ported from RadiometricCamera's `convert_raw.py`. Modes `:bin2` (2×2 Bayer
  quads: R, the mean of the two greens, B; half size) and `:green` (the green
  photosites alone on the raw grid). Its `fullres` demosaic is not provided.
  Returns a `LinearRaw`: `image`, `sampled`, `saturated` (a quad is saturated
  when ANY of its photosites is at the white level, `raw ≥ white`, no margin),
  `plane_offsets` (each plane's true sample position: the green at the quad
  centre, R and B `(±¼, ±¼)` binned pixels from it) and the EXIF `metadata`.
  Rounded to 16 bits, `:bin2` equals `convert_raw.py --mode bin2`'s linear PNG
  value for value.
- **New `load_raw_mosaic`, `RawMosaic`, `LinearRaw(mosaic; mode, gain)`**: the
  decode and the processing apart; the processing is all Julia.
- **Two decoders, both weak dependencies**: `LibRawDecoder()` (new extension
  `VisualGeometryToolkitFeaturesLibRawExt` on LibRaw_jll 0.22.2: a `ccall`
  layer whose mirrored `libraw_data_t` layout is checked against LibRaw's
  accessors on every file, refusing a LibRaw other than 0.22) and
  `RawpyDecoder()` (new extension `VisualGeometryToolkitFeaturesPythonExt` on
  PythonCall; rawpy and exifread declared in the new `CondaPkg.toml`). Without
  the extension loaded, `load_raw` names the package to load.
