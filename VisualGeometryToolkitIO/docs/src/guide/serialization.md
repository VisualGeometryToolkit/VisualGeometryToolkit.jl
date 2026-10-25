# Quantities, Frames and Detections

```@meta
CurrentModule = VisualGeometryToolkitIO
```

VisualGeometryToolkitIO owns the stack's quantity JSON format. A quantity in a
JSON record is a **number in the unit the record declares**, and the record
declares its units once, inline: `"units": {"world": "mm", "density": "dpi"}`,
a [`UnitContract`](@ref). A length is a number in the world unit, a density
(pixels per length) a number in the density unit, and a pixel count a number
of pixels. The unit is not repeated beside every value.

What a number *is* — a length, a pixel count or a density — is the record's to
say, field by field, because a pixel is dimensionless and a number alone cannot
tell a pixel count from an angle or a ratio. [`write_quantity`](@ref) and
[`read_quantity`](@ref) convert one quantity:

```jldoctest serialization
julia> using VisualGeometryToolkitCore, VisualGeometryToolkitIO

julia> import JSON

julia> import VisualGeometryToolkitIO as VgtkIO

julia> VgtkIO.write_quantity(62.8mm, MM_PX), VgtkIO.write_quantity(640px, MM_PX), VgtkIO.write_quantity(300dpi, MM_PX)
(62.8, 640, 300)

julia> VgtkIO.read_quantity(0.0628, :length, UnitContract(world = m)) == 62.8mm
true

julia> VgtkIO.json_value(MM_PX)   # the record's declaration
(world = "mm", density = "dpi")
```

A length crosses through [`read_length`](@ref) and [`write_length`](@ref), so a
millimetre number reads and writes back to its own text. A pixel count and a
density are written as integers when they are integral. A quantity of any
other dimension has no declared unit and is an `ArgumentError`.

## Sizes, feature frames and detections

[`json_value`](@ref)`(x, contract)` is the plain value a record holds for one
of VisualGeometryToolkitCore's values, which `JSON.json` writes, and
[`from_json`](@ref)`(T, d, contract; kind)` reads it back from a parsed object.
These are explicit functions, not methods of JSON.jl's `lower` and `lift`.

| Value | JSON object |
|---|---|
| `Size2` | `{"width": w, "height": h}` |
| `Disc` | `{"center": [x, y], "σ": s}` |
| `OrientedDisc` | `{"center": [x, y], "σ": s, "θ": angle}` |
| `EllipticFrame` | `{"center": [x, y], "S": [S11, S21, S12, S22]}` |
| `AffineFrame` | `{"center": [x, y], "A": [A11, A21, A12, A22]}` |

Matrices are written as their four entries in column order. A `Size2` or a
`Disc` may hold plain numbers or quantities of ONE kind; its reader declares
which with `kind` (`:plain`, `:length` or `:pixel`). The other frames hold
plain numbers.

```jldoctest serialization
julia> s = JSON.json(VgtkIO.json_value(Size2(width = 210.0mm, height = 297.0mm), MM_PX))
"{\"width\":210.0,\"height\":297.0}"

julia> VgtkIO.from_json(Size2, JSON.parse(s), MM_PX; kind = :length) == Size2(width = 210.0mm, height = 297.0mm)
true
```

A detection — an `Attributed` feature frame, VisualGeometryToolkitCore's
`AbstractDetection`, which VisualGeometryToolkitFeatures' `Detection` is — is
one JSON object. Its frame is under `frame`, and every metadata field is under
its own name; the frame's type is read off its keys. A detection with an
`Uncertain` frame (from `uncertainty = FeatureCovariance()`) also has `cov`, the
covariance's entries in column order, or `null` when the covariance is not
finite (a keypoint the detector could not refine); reading it back makes the
frame `Uncertain` again. On reading, the documented metadata fields
(`response`, `edge_score`, `laplacian_scale_score`) come first and as
`Float64`, and the other fields follow, sorted by name, as they were parsed.
Metadata has no declared kind, so a quantity in it is refused on writing.

```jldoctest serialization
julia> d = Attributed(Disc(Point2(10.5, 20.25), 3.0), (response = 0.25, laplacian_scale_score = 1.0));

julia> s = JSON.json(VgtkIO.json_value(d, MM_PX))
"{\"frame\":{\"center\":[10.5,20.25],\"σ\":3.0},\"response\":0.25,\"laplacian_scale_score\":1.0}"

julia> VgtkIO.from_json(AbstractDetection, JSON.parse(s), MM_PX) == d
true
```

With VisualGeometryToolkitFeatures loaded, `from_json(Detection, obj, contract)`
reads the same object.

## The legacy `{value, unit}` form

Files written before this format — VisualGeometryToolkitFeatures' JSON.jl hooks
and BlobBoards' board files — hold each quantity as a `{"value": v, "unit": u}`
object. It is read wherever a number in the declared unit is, by
`read_quantity` and every `from_json`, as `v` times the unit: any unit name
VisualGeometryToolkitCore's `parse_unit` reads, compound ones such as `"mm/px"`
included, whatever the contract declares. A declared kind is checked against
the unit's dimension. Nothing writes the legacy form.

```jldoctest serialization
julia> VgtkIO.read_quantity(JSON.parse("""{"value": 2.5, "unit": "mm"}"""), :length, MM_PX) == 2.5mm
true

julia> VgtkIO.from_json(Size2, JSON.parse("""{"width": 3.0, "height": {"value": 4.0, "unit": "mm"}}"""), MM_PX) ==
           Size2(width = 3.0, height = 4.0mm)
true
```

[`ScalarOrQuantity`](@ref) is the type of a field that may hold a plain number
or a quantity.
