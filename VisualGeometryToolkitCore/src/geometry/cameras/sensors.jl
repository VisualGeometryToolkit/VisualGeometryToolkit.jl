"""
    Sensor{TR, TP}

Image sensor specification with resolution and pixel pitch.

# Fields
- `resolution::Size2{TR}`: Pixel count (width × height in px)
- `pitch::Size2{TP}`: Pixel pitch (physical size per pixel, e.g., μm/px)

# Constructors
```julia
Sensor(Size2(width=1920px, height=1200px), 5.86μm)
Sensor(Size2(width=1920, height=1200), 5.86μm)  # Integers → px
```

# See Also
- [`CMOS_SENSORS`](@ref): Pre-defined sensor database
- [`aspect_ratio`](@ref)
"""
struct Sensor{TR <: Quantity, TP <: Quantity}
    resolution::Size2{TR}   # in px (logical width/height)
    pitch::Size2{TP}        # a length per pixel (e.g., μm/px)

    # Inner constructor — the dims enforcement boundary (px resolution,
    # length/px pitch).
    function Sensor{TR, TP}(resolution::Size2{TR}, pitch::Size2{TP}) where {TR <: Quantity, TP <: Quantity}
        all(c -> dimension(c) == dimension(px), resolution) ||
            throw(ArgumentError("resolution must be in pixels, got $resolution"))
        all(c -> dimension(c) == dimension(mm / px), pitch) ||
            throw(ArgumentError("pitch must be a length per pixel, got $pitch"))
        return new{TR, TP}(resolution, pitch)
    end
end

# Constructors: any FlexUnits quantity in, stored in FlexUnits' base form
# `Quantity{Float64, Dimensions{FixRat32}}` (`5.86μm / px` is a unit-carrying
# quantity; `convert` takes it to the base form).
Sensor(res::Size2{<:Integer}, pitch::Quantity) = Sensor(Size2(width = res.width * px, height = res.height * px), pitch)
Sensor(res::Size2{<:Quantity}, pitch::Quantity) = Sensor(res, Size2(width = pitch / px, height = pitch / px))
function Sensor(res::Size2{<:Quantity}, pitch::Size2{<:Quantity})
    Q = Quantity{Float64, Dimensions{FixRat32}}
    return Sensor{Q, Q}(Size2{Q}(res), Size2{Q}(pitch))
end

# Accessors

"""
    aspect_ratio(sensor) -> Float64

Width-to-height ratio of the sensor resolution.

```julia
s = sensor("Sony", "IMX174")
aspect_ratio(s)  # 1.6 (1920/1200)
```
"""
aspect_ratio(s::Sensor) = s.resolution.width / s.resolution.height

@inline Base.Tuple(sensor::Sensor) = (sensor.resolution, sensor.pitch)

"""
    CMOS_SENSORS

Database of common CMOS image sensors indexed by manufacturer and model.

```julia
CMOS_SENSORS["Sony"]["IMX174"]      # 1920×1200 @ 5.86μm
CMOS_SENSORS["Sony"]["IMX250"]      # 2448×2048 @ 3.45μm
CMOS_SENSORS["OmniVision"]["OV9281"] # 1280×800 @ 3.0μm
```
"""
const CMOS_SENSORS = ImmutableDict(
    "Sony" => ImmutableDict(
        # Industrial sensors
        "IMX249" => Sensor(Size2(width = 1920px, height = 1200px), 5.86μm),
        "IMX174" => Sensor(Size2(width = 1920px, height = 1200px), 5.86μm),
        "IMX250" => Sensor(Size2(width = 2448px, height = 2048px), 3.45μm),
        # Cinema cameras (FX3/A7S III use IMX510AQL)
        "FX3" => Sensor(Size2(width = 4240px, height = 2832px), 8.4μm),
        "IMX510" => Sensor(Size2(width = 4240px, height = 2832px), 8.4μm)  # alias for FX3/A7S III
    ),
    "Nikon" => ImmutableDict(
        # Full-frame DSLRs (Sony-manufactured sensor)
        "D780" => Sensor(Size2(width = 6048px, height = 4024px), 5.94μm)
    ),
    "Canon" => ImmutableDict(
        # APS-C DSLRs (Canon in-house sensor)
        "EOS 77D" => Sensor(Size2(width = 6000px, height = 4000px), 3.72μm)
    ),
    "OmniVision" => ImmutableDict(
        # Global shutter sensors
        "OV9281" => Sensor(Size2(width = 1280px, height = 800px), 3.0μm),
        "OV2311" => Sensor(Size2(width = 1600px, height = 1300px), 3.0μm)
    ),
    "ON Semiconductor" => ImmutableDict(
        # Industrial global shutter
        "XGS 5000" => Sensor(Size2(width = 2592px, height = 2048px), 3.2μm),
        "XGS 12000" => Sensor(Size2(width = 4096px, height = 3072px), 3.2μm)
    )
)

"""
    sensor(manufacturer, model) -> Sensor

Look up a sensor from the CMOS_SENSORS database.

```julia
sensor("Sony", "IMX174")  # equivalent to CMOS_SENSORS["Sony"]["IMX174"]
```
"""
sensor(manufacturer::AbstractString, model::AbstractString) = CMOS_SENSORS[manufacturer][model]

# =============================================================================
# Pixel Convention Definitions
# =============================================================================

# NOTE: `INTRINSICS_COORDINATE_OFFSET` (the corner-offset table behind the
# pixel-origin conventions) moved to transforms/coord_maps.jl in 0.7 — it is
# frame-level data backing `EuclideanMap(PixelFrame{A}, PixelFrame{B})`, and
# the typed `image_warp` bridges need it before the Cameras submodule loads.
# It is internal to the top module: neither exported nor imported into Cameras.

# =============================================================================
# Rect Constructor for Sensor
# =============================================================================

"""
    Rect(sensor::Sensor) -> Rect

Create a Rect representing the sensor bounds in COLMAP convention.

Uses COLMAP convention where pixel centers are at (0.5, 0.5), (1.5, 0.5), etc.
The rect origin is at (0, 0) and extends to (width, height).

# Returns
Rect with origin at (0, 0) and widths equal to sensor resolution

# Examples
```julia
sensor = Sensor(Size2(width=1920px, height=1200px), 5.86μm)
rect = Rect(sensor)  # Rect{2, Int}: origin=(0, 0), widths=(1920, 1200)

# The extent a warp resamples into
w = image_warp(transform, rect)
```
"""
function Rect(sensor::Sensor)
    w = Int(ustrip(px, sensor.resolution.width))
    h = Int(ustrip(px, sensor.resolution.height))
    return Rect(0, 0, w, h)
end

# NOTE (2026-07-26, found by JET): a `Rect2(sensor::Sensor, image_origin::Symbol)`
# method used to live here, forwarding to `Rect(sensor, image_origin)`. No 2-arg
# `Rect(::Sensor, ::Symbol)` has ever existed — only the 1-arg `Rect(::Sensor)`
# above — so the method raised on every call, and its docstring promised an
# origin-dependent rect the codebase never implemented (`Rect(::Sensor)` is
# COLMAP-convention unconditionally). Removed rather than made to silently
# ignore its second argument. Nothing called it.
