# Serialization

```@meta
CurrentModule = VisualGeometryToolkitFeatures
```

Detections, their feature frames, image sizes and unit-carrying quantities are
written and read with [JSON.jl](https://github.com/JuliaIO/JSON.jl) 1.x. The
package adds JSON.jl's two hooks for each type: `JSON.lower` for writing and
`JSON.lift` for reading. Loading VisualGeometryToolkitFeatures is enough;
there is nothing to call.

```jldoctest serialization
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore, JSON

julia> d = Detection(Disc(Point2(10.5, 20.25), 3.0), (response = 0.25, laplacian_scale_score = 1.0));

julia> s = JSON.json(d)
"{\"frame\":{\"center\":[10.5,20.25],\"σ\":3.0},\"response\":0.25,\"laplacian_scale_score\":1.0}"

julia> JSON.parse(s, Detection) == d
true
```

## The format

A detection is one JSON object. Its frame is under `frame`, and every metadata
field is under its own name. The frame's type is read off its keys:

| Frame | JSON object |
|---|---|
| `Disc` | `{"center": [x, y], "σ": s}` |
| `OrientedDisc` | `{"center": [x, y], "σ": s, "θ": angle}` |
| `EllipticFrame` | `{"center": [x, y], "S": [S11, S21, S12, S22]}` |
| `AffineFrame` | `{"center": [x, y], "A": [A11, A21, A12, A22]}` |

Matrices are written as their four entries in column order. A detection with
a covariance (from `uncertainty = FeatureCovariance()`) also has `cov`, the
3×3 covariance's entries in column order, or `null` when the covariance is not
finite (a keypoint the detector could not refine). Reading it back makes the
frame `Uncertain` again.

On reading, the documented metadata fields (`response`, `edge_score`,
`laplacian_scale_score`) come first and as `Float64`, and any other fields
follow, sorted by name. The result does not depend on the key order of the
file.

## Quantities and sizes

A quantity is written as `{"value": v, "unit": u}` in a canonical unit: `mm`
for a length, `px` for a pixel count, `dpi` for a density and `mm/px` for a
pitch. Pixel counts and densities are written as integers when they are
integer-valued. Any unit name that VisualGeometryToolkitCore's `parse_unit`
reads is accepted on reading. A `Size2` is `{"width": …, "height": …}`, each a
plain number or a quantity.

```jldoctest serialization
julia> JSON.json(Size2(width = 640px, height = 480px))
"{\"width\":{\"value\":640,\"unit\":\"px\"},\"height\":{\"value\":480,\"unit\":\"px\"}}"

julia> JSON.parse("""{"width": 3.0, "height": {"value": 4.0, "unit": "mm"}}""", Size2) ==
           Size2(width = 3.0, height = 4.0mm)
true
```

## Reading from a parsed object

`JSON.parse(str, T)` reads a type directly. Code that has already parsed a
document, and holds a `JSON.Object` (or any dictionary with string keys), calls
the reading hook on the part it needs: `JSON.lift(Detection, obj)`,
`JSON.lift(Size2, obj)`. [`ScalarOrQuantity`](@ref) is the type of a value that
may be a plain number or a quantity:

```jldoctest serialization
julia> import VisualGeometryToolkitFeatures as VgtkFeatures

julia> obj = JSON.parse("""{"scale": {"value": 2.5, "unit": "mm"}, "count": 3}""");

julia> JSON.lift(VgtkFeatures.ScalarOrQuantity, obj["scale"]) == 2.5mm
true

julia> JSON.lift(VgtkFeatures.ScalarOrQuantity, obj["count"])
3
```
