# =============================================================================
# JSON serialization (JSON.jl 1.x) for the geometry types detections carry
# =============================================================================
#
# Moved from VisualGeometryToolkitCore. Each type gets the two JSON.jl hooks:
#   - writing: `JSON.lower(::JSON.JSONStyle, x)` returns a plain value JSON.jl writes
#   - reading: `JSON.lift(::Type{T}, obj)` builds `T` from a parsed JSON object
#     (a `JSON.Object`, any `AbstractDict` with String keys, or a NamedTuple)
# and `StructUtils.structlike(::Type{T}) = false` (what JSON.jl's `@nonstruct`
# declares), so that `JSON.parse(str, T)` hands `lift` the whole object instead
# of building `T` from its fields. Covered:
#   - FlexUnits quantities ({value, unit} objects, `lower_quantity`) and
#     ScalarOrQuantity (a number or one)
#   - Size2 (width/height with optional units)
#   - Disc (unit-carrying center/scale), OrientedDisc, EllipticFrame, AffineFrame
# The Detection methods are in detection_types.jl.
#
# The lower methods take `JSON.JSONStyle` rather than the style-free form because
# JSON.jl's own `lower(::JSONStyle, ::AbstractVector)` is more specific than a
# style-free method, and Size2 is a vector.

using JSON: JSON
using StructUtils: StructUtils
using FlexUnits: Quantity
using VisualGeometryToolkitCore: ustrip, dimension, parse_unit, unit_string, mm, px, dpi

# =============================================================================
# JSON field access
# =============================================================================

# A parsed JSON object is an AbstractDict with String keys (JSON.Object); a
# NamedTuple stands in for one when a caller builds the input by hand.
_hasfield(x::AbstractDict, key::String) = haskey(x, key)
_hasfield(x::NamedTuple, key::String) = haskey(x, Symbol(key))

_getfield(x::AbstractDict, key::String) = x[key]
_getfield(x::NamedTuple, key::String) = getproperty(x, Symbol(key))
_getfield(x, key::String, default) = _hasfield(x, key) ? _getfield(x, key) : default

_keys(x::AbstractDict) = (String(k) for k in keys(x))
_keys(x::NamedTuple) = (String(k) for k in keys(x))

const _JSONObject = Union{AbstractDict, NamedTuple}

# =============================================================================
# Writing a quantity: the unit comes from the field
# =============================================================================
#
# A pixel is dimensionless (FlexUnits' dimensions are the seven SI ones), so
# a quantity's dimension no longer says whether it is a pixel count, an angle
# or a ratio: the dimension cannot choose "px". The unit a value is written in
# comes from what its FIELD holds instead — a size or a position (`Size2`'s
# width and height, a `Disc`'s center and σ) is a length or a pixel count, so
# a dimensionless value there is pixels — and a value whose field admits two
# units of its dimension, or none, is refused with an `ArgumentError` naming
# the field. A bare quantity, written with no field to say what it holds, is
# a length (mm) or a density (dpi); a dimensionless one is refused.
#
# The bytes are those of the writer this replaces, which drew the line by a
# pixel dimension: a length is written in mm as a Float64; a pixel count and a
# density snap to an integer when they are integral (`640`, `300`). BlobBoards'
# board hashes are computed over these bytes, and the golden test in
# test/vgf_tests.jl pins them. A length per pixel (a pitch, `mm / px`) is a
# length now, written in mm: the old writer raised a MethodError on one.
#
# PARSER: any unit name VisualGeometryToolkitCore's `parse_unit` reads — registered
# unit symbols and their products, quotients and powers (`mm/px`), the stack's
# one guarded `uparse` — so every string the legacy Unitful writer ever
# produced (observed corpus: mm, px, dpi, pd) loads, and old board JSONs keep
# their stored config hash (UIDs printed on physical boards stay valid:
# readers never re-hash).

# The units a field of each kind may be written in, by its semantics.
const _SIZE_UNITS = (mm, px)            # a size or a position: a length or a pixel count
const _BARE_UNITS = (mm, dpi)           # a quantity with no field: a length or a density

# A length is written as a Float64; anything else (a pixel count, a density)
# as an integer when it is integral, as the writer this replaces did.
_written_value(v, u) = dimension(u) == dimension(mm) ? Float64(v) : _snap_integer(v)
_snap_integer(v::Integer) = Int(v)
_snap_integer(v::AbstractFloat) = isinteger(v) ? Int(v) : v

"""
    lower_quantity(q, field::Symbol, units) -> (value = …, unit = …)

The `{"value": …, "unit": …}` object of the quantity `q` held in `field`,
written in the one unit of `units` (a tuple of FlexUnits units: the units the
field may hold) whose dimension is `q`'s, under its registered name
(`unit_string`). A plain number is returned as it is. A quantity whose
dimension matches no unit of `units`, or more than one, is an
`ArgumentError` naming `field`: a pixel is dimensionless, so a field that
admits `px` and another dimensionless unit cannot tell them apart.

A length is written as a `Float64`, any other value as an integer when it is
integral (`640`, `300`).

A serializer of a type whose fields hold quantities calls this with each
field's units: `Size2` and `Disc` write theirs with `(mm, px)`, so a
dimensionless width is pixels.

```jldoctest
julia> using JSON, VisualGeometryToolkitCore

julia> import VisualGeometryToolkitFeatures as VgtkFeatures

julia> VgtkFeatures.lower_quantity(640px, :width, (mm, px))
(value = 640, unit = "px")

julia> VgtkFeatures.lower_quantity(2.5mm, :width, (mm, px))
(value = 2.5, unit = "mm")
```

Public, not exported.
"""
lower_quantity(q::Real, ::Symbol, units::Tuple) = q
function lower_quantity(q::Quantity, field::Symbol, units::Tuple)
    d = dimension(q)
    matches = filter(u -> dimension(u) == d, units)
    isempty(matches) && throw(
        ArgumentError(
            "field `$field`: a quantity of dimension $(repr(d)) is none of the units it holds " *
                "($(join(unit_string.(units), ", ")))"
        )
    )
    length(matches) == 1 || throw(
        ArgumentError(
            "field `$field`: a quantity of dimension $(repr(d)) is ambiguous between " *
                "$(join(unit_string.(matches), " and ")) (a pixel is dimensionless)"
        )
    )
    u = only(matches)
    return (value = _written_value(ustrip(u, q), u), unit = unit_string(u))
end

# =============================================================================
# ScalarOrQuantity
# =============================================================================

"""
    ScalarOrQuantity

The type of a JSON value that holds either a plain number or a quantity: an
alias of `Number`, since a FlexUnits quantity is a `Number`. Read one with
`JSON.lift(ScalarOrQuantity, x)`: a number comes back as it is, and a
`{"value": …, "unit": …}` object comes back as a quantity in that unit.

```jldoctest
julia> using JSON, VisualGeometryToolkitCore

julia> import VisualGeometryToolkitFeatures as VgtkFeatures

julia> JSON.lift(VgtkFeatures.ScalarOrQuantity, 5)
5

julia> JSON.lift(VgtkFeatures.ScalarOrQuantity, JSON.parse(\"\"\"{"value": 2.5, "unit": "mm"}\"\"\")) == 2.5mm
true

julia> JSON.json(2.5mm)
"{\\"value\\":2.5,\\"unit\\":\\"mm\\"}"
```

Public, not exported.
"""
const ScalarOrQuantity = Number

# A plain number lifts to itself through StructUtils' default `lift`; the
# {value, unit} object form is added here.
JSON.lift(::Type{ScalarOrQuantity}, data::AbstractDict) = JSON.lift(Quantity, data)

# =============================================================================
# Quantity Serialization ({value, unit} JSON objects)
# =============================================================================

# A bare quantity — one JSON.jl reaches with no field to say what it holds
# (inside a NamedTuple, a Dict or a metadata field) — is a length or a density.
function JSON.lower(::JSON.JSONStyle, q::Quantity)
    dimension(q) == dimension(px) && throw(
        ArgumentError(
            "a dimensionless quantity ($q) has no unit to be written in: a pixel count, an angle " *
                "and a ratio share its dimension. Write the field that holds it with " *
                "`VisualGeometryToolkitFeatures.lower_quantity(q, field, (px,))`"
        )
    )
    return lower_quantity(q, :quantity, _BARE_UNITS)
end

StructUtils.structlike(::Type{<:Quantity}) = false

function JSON.lift(::Type{<:Quantity}, obj::_JSONObject)
    u = parse_unit(String(_getfield(obj, "unit")); compound = true)
    return _getfield(obj, "value") * u
end

# =============================================================================
# Size2 Serialization
# =============================================================================

JSON.lower(::JSON.JSONStyle, s::Size2) =
    (width = lower_quantity(s.width, :width, _SIZE_UNITS), height = lower_quantity(s.height, :height, _SIZE_UNITS))

# Size2 is a StaticArrays FieldVector; on the reading side it is an object.
StructUtils.structlike(::Type{<:Size2}) = false
StructUtils.arraylike(::Type{<:Size2}) = false

function JSON.lift(::Type{<:Size2}, x::_JSONObject)
    width = JSON.lift(ScalarOrQuantity, _getfield(x, "width"))
    height = JSON.lift(ScalarOrQuantity, _getfield(x, "height"))
    return Size2(width = width, height = height)
end

# =============================================================================
# Feature Frame Serialization
# =============================================================================
#
# The wire format of the frames: `center` as [x, y], a 2×2 matrix (`S`, `A`) as
# its four entries in column order, scales and angles as numbers.

_json_center(x) = (c = _getfield(x, "center"); Point2{Float64}(c[1], c[2]))
_json_mat2(v) = SMatrix{2, 2, Float64, 4}(v[1], v[2], v[3], v[4])
_json_vec(M) = collect(Float64, vec(M))

StructUtils.structlike(::Type{<:Union{Disc, OrientedDisc, EllipticFrame, AffineFrame}}) = false

# Disc carries units: its center and σ may be quantities, a length or a pixel count.
JSON.lower(::JSON.JSONStyle, d::Disc) = (
    center = (lower_quantity(d.center[1], :center, _SIZE_UNITS), lower_quantity(d.center[2], :center, _SIZE_UNITS)),
    σ = lower_quantity(d.σ, :σ, _SIZE_UNITS),
)

function JSON.lift(::Type{<:Disc}, x::_JSONObject)
    center_data = _getfield(x, "center")
    center = Point2(
        JSON.lift(ScalarOrQuantity, center_data[1]),
        JSON.lift(ScalarOrQuantity, center_data[2])
    )
    σ = JSON.lift(ScalarOrQuantity, _getfield(x, "σ"))
    return Disc(center, σ)
end

JSON.lower(::JSON.JSONStyle, f::OrientedDisc) = (center = (f.center[1], f.center[2]), σ = f.σ, θ = f.θ)
JSON.lower(::JSON.JSONStyle, f::EllipticFrame) = (center = (f.center[1], f.center[2]), S = _json_vec(f.S))
JSON.lower(::JSON.JSONStyle, f::AffineFrame) = (center = (f.center[1], f.center[2]), A = _json_vec(f.A))

JSON.lift(::Type{<:OrientedDisc}, x::_JSONObject) =
    OrientedDisc{Float64}(_json_center(x), Float64(_getfield(x, "σ")), Float64(_getfield(x, "θ")))
JSON.lift(::Type{<:EllipticFrame}, x::_JSONObject) =
    EllipticFrame{Float64}(_json_center(x), _json_mat2(_getfield(x, "S")))
JSON.lift(::Type{<:AffineFrame}, x::_JSONObject) =
    AffineFrame{Float64}(_json_center(x), _json_mat2(_getfield(x, "A")))
