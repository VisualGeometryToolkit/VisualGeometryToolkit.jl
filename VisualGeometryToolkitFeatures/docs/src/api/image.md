# Image Input

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

The image-input helpers, the linear-radiance PNG reader (`load_linear_image`)
and the camera RAW reader (`load_raw`).

## Camera RAW

`load_raw(path; mode, gain, decoder)` reads a Bayer RAW file as exact linear
values, `(raw − black[c]) / (white − black[c]) × gain`, in the sensor's
orientation. The decode is an extension, chosen by a token:

| decoder | load | backend |
|---|---|---|
| [`LibRawDecoder`](@ref) (default) | `using LibRaw_jll` | LibRaw 0.22 through `ccall`, no Python |
| [`RawpyDecoder`](@ref) | `using PythonCall` | rawpy and exifread (CondaPkg) |

Everything after the decode — the `:bin2` quads or the `:green` photosites, the
per-quad saturation flags and each plane's true sample position — is the same
Julia code for both. On 21 Sony ARW frames the two decoders return identical
photosites, colour indices, black and white levels and visible area; their EXIF
differs in precision only: LibRaw stores exposure, f-number and focal length as
`Float32` (`0.01666666753590107` s for exifread's `1/60`) and keeps no
sub-second time (`subsec = ""`).

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

## Image input and the linear PNG reader

```@autodocs
Modules = [VisualGeometryToolkitFeatures]
Pages = ["image_input.jl", "linear_image.jl"]
```
