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

## The sensor model: the black level, the noise and the defects

Three levels: the geometric `Sensor` (part of a camera), a
[`SensorCharacterisation`](@ref) (ONE body's components per ISO, held beside
the camera like [`CaptureSettings`](@ref)) and a [`SensorModel`](@ref) (ONE
frame's black, noise and defect components). Each slot's component is a type,
absence included ([`FileBlack`](@ref), [`NoNoise`](@ref), [`NoDefects`](@ref)),
and each present one states its source (`SensorSource`: measured, estimated,
specified, file). The resolver [`sensor_model`](@ref) is a lookup: the frame's
body and ISO pick an entry, and with none the absent model `SensorModel()`
linearises bit for bit as the RAW alone. [`LinearRaw`](@ref)`(mosaic; mode,
sensor_model)` is THE one black subtraction; under a [`PoissonGaussian`](@ref)
or [`RelativeNoise`](@ref) its image carries each value's variance. The
estimators build a characterisation's components from frames, offline:
[`photon_transfer`](@ref), [`master_dark`](@ref), [`defect_onsets`](@ref).

```@autodocs
Modules = [VisualGeometryToolkitCore]
Private = false
Pages = ["raw/sensor_model.jl", "raw/sensor_estimators.jl"]
```

```@docs
VisualGeometryToolkitCore.SensorSource
```
