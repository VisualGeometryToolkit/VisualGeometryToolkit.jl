# Quantities, Frames and Detections

```@meta
CurrentModule = VisualGeometryToolkitIO
DocTestSetup = :(using VisualGeometryToolkitCore, VisualGeometryToolkitIO; import VisualGeometryToolkitIO as VgtkIO)
```

Every number a vgio record holds is in a unit its file's `units` object
declares — the envelope every file opens with (`"units": {"length": "mm",
"pixel": "colmap", "density": "dpi", "length_roles": {…}, …}`) — and no field
states a unit of its own. A value is written by `JSON.lower(::VgioStyle, x)`
and read by `JSON.lift(::VgioStyle, T, x)`, on the style that carries those
units ([`VgioStyle`](@ref)); a writer that leaves a quantity in a record gets
it written in the declared unit.

## What a number is: the field tag

A pixel is dimensionless, so `640` cannot say whether it is 640 px, 640 mm or a
plain 640. A field says it by its FIELD TAGS, StructUtils' fourth argument of
`lift` (and `lower`): `(; quantity = :length | :pixel | :density[, role])`. A
lift without tags reads plain numbers. A length is decimal-exact in
`units.length`, or in the role its tag names (`units.length_roles`):

```jldoctest
julia> import JSON

julia> using FlexUnits: Quantity

julia> s = VgtkIO.VgioStyle(; length_unit = mm);

julia> JSON.lower(s, 62.8mm), JSON.lower(s, 640px), JSON.lower(s, 300dpi)
(62.8, 640, 300)

julia> JSON.lift(s, Quantity, 62.8, (; quantity = :length)) == VgtkIO.read_length(62.8, mm)
true

julia> JSON.json(Size2(width = 640px, height = 480px); style = s)
"{\"width\":640,\"height\":480}"

julia> JSON.lift(s, Size2, JSON.parse("{\"width\": 640, \"height\": 480}"), (; quantity = :pixel)) == Size2(width = 640px, height = 480px)
true
```

The `{"value": v, "unit": u}` form, and UnitContract's own layout
(`{"world", "density", "lengths"}`), are never a current record's. A record of
an earlier version that holds them is read by its owner's version step;
[`migrate_units`](@ref) gives such a record's 0.10 `units` object.

## The frame codec

Every VisualGeometryToolkitCore `FeatureFrame` is a case of ONE codec: `{"center": [x, y]}` and
its shape — a `Disc`'s `sigma`, an `OrientedDisc`'s `sigma` and `angle`, an
`EllipticFrame`'s `shape_matrix`, an `AffineFrame`'s `affine_matrix` (rows); a
`PointFrame` has none. An uncertain frame adds its `covariance` over the
frame's `frame_dof` parameters, centre first, in the storage it holds (a
`PointFrame`'s is the uncertain point's 2×2):

```jldoctest
julia> import JSON

julia> s = VgtkIO.VgioStyle();

julia> u = Uncertain(PointFrame(10.5, 20.25), CovMat{2, Float64}(0.5, 0.0, 0.0, 0.25));

julia> JSON.json(u; style = s)
"{\"center\":[10.5,20.25],\"covariance\":[[0.5,0.0],[0.0,0.25]]}"

julia> JSON.lift(s, Uncertain{PointFrame{Float64}}, JSON.parse(JSON.json(u; style = s))) == u
true
```

## Detections

A detector's output per image is a `vgio/detection/v1.0` file
(`save(path, Dict(image => detections))`,
`load(Dict{String, <:AbstractVector{<:AbstractDetection}}, path)`): each
detection is its frame through the codec above and its attributes, and the
file declares once the type every detection has (frame type, number type,
covariance storage, each attribute's type), so it reads back exact by type. A
detection without a covariance omits it and states its `covariance_status`
attribute, the producer's reason: there is no `null`, and no default.
