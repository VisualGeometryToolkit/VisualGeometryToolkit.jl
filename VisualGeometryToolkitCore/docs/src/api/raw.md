# Camera RAW API

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The sensor image: a decoded Bayer RAW as [`RawMosaic`](@ref) (an
`AbstractMatrix{UInt16}` of its photosites that knows its colour phase, its
levels and where it sits on the sensor), its photosites channel-first as a
method of ImageCore's `channelview` (a view, on the mosaic's device), and
[`LinearRaw`](@ref), its exact linear values on the 2×2 cells, with the maps
of every plane and channel onto the sensor. Decoding a file is
VisualGeometryToolkitFeatures' (`load_raw_mosaic`, LibRaw), and so is the
display demosaic. A cell mode's frame map is also read without an image,
`raw_frame_map(LinearRaw{:green_cells})`, and its provenance record,
`raw_provenance(LinearRaw{:green_cells})`, for a frame with no RAW behind it. See the
[Raw images](../guide/raw-images.md) guide.

## The mosaic and its photosite views

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["raw/raw_mosaic.jl"]
```

## Linear values on the 2×2 cells

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["raw/linear_raw.jl"]
```

## Sensor noise

A body's Poisson–Gaussian noise at an ISO with the source of every value, the one
resolver that decides what a frame uses when a record is missing, and the two
estimators (a frame's photon transfer, a record's dark frames). `LinearRaw`'s
kernel subtracts the resolved black and dark offset (`noise`).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["raw/sensor_noise.jl"]
```
