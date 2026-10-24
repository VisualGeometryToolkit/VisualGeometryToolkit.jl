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
#   - VgcQuantity ({value, unit} objects) and ScalarOrQuantity (a number or one)
#   - Size2 (width/height with optional units)
#   - Disc (unit-carrying center/scale), OrientedDisc, EllipticFrame, AffineFrame
# The Detection methods are in detection_types.jl.
#
# The lower methods take `JSON.JSONStyle` rather than the style-free form because
# JSON.jl's own `lower(::JSONStyle, ::AbstractVector)` is more specific than a
# style-free method, and Size2 is a vector.

using JSON: JSON
using StructUtils: StructUtils
using VisualGeometryToolkitCore: VgcQuantity, ustrip, dimension, parse_unit, mm, px, dpi

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
# Dimension Traits for Compile-Time Dispatch
# =============================================================================

"""Trait types for dimension-based serialization dispatch."""
abstract type DimTrait end

"""Logical dimensions (pixel exponent ≠ 0): counts, densities. Snap to Int when integer-valued."""
struct LogicalDim <: DimTrait end

"""Physical dimensions (pure length): always serialize as Float64."""
struct PhysicalDim <: DimTrait end

# Dimensions are runtime values: a quantity (or unit) is "logical" whenever its
# px exponent is nonzero (px, dpi, and pitches), physical otherwise (pure
# lengths). The legacy Unitful writer drew the same line, so px, dpi and μm/px
# all snap to integers when they can.
dim_trait(q) = iszero(dimension(q).px) ? PhysicalDim() : LogicalDim()

# =============================================================================
# Value Canonicalization
# =============================================================================

"""
    canon_value(v, trait::DimTrait)

Canonicalize a numeric value for JSON serialization based on dimension trait.
- `LogicalDim`: snap to Int if exactly integer-valued
- `PhysicalDim`: convert to Float64
"""
canon_value(v, ::LogicalDim) = snap_integer(v)
canon_value(v, ::PhysicalDim) = Float64(v)

"""Snap to Int if the value is exactly integer-valued."""
snap_integer(v::Integer) = Int(v)
snap_integer(v::AbstractFloat) = isinteger(v) ? Int(v) : v
snap_integer(v) = v

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
JSON.lift(::Type{ScalarOrQuantity}, data::AbstractDict) = JSON.lift(VgcQuantity, data)

# =============================================================================
# Quantity Serialization ({value, unit} JSON objects)
# =============================================================================
#
# WRITER: canonical units only — length→"mm", pixel→"px", density→"dpi",
# pitch→"mm/px".  Deterministic output; new files hash at creation.
# PARSER: any unit name VisualGeometryToolkitCore's `parse_unit` reads — registered
# unit symbols and their products, quotients and powers (`mm/px`), the stack's
# one guarded `uparse` — so every string the legacy Unitful writer ever
# produced (observed corpus: mm, px, dpi, pd) loads, and old board JSONs keep
# their stored config hash (UIDs printed on physical boards stay valid:
# readers never re-hash).
# The writer keeps its own table: FlexUnits prints a registered unit as its
# symbol, but a compound one (`mm/px`) as its internal form, no name.

const _CANONICAL_UNITS = (("mm", mm), ("px", px), ("dpi", dpi), ("mm/px", mm / px))

function _canonical_unit(q)
    for (s, u) in _CANONICAL_UNITS
        dimension(q) == dimension(u) && return (s, u)
    end
    throw(ArgumentError("no canonical serialization unit for dimensions $(dimension(q))"))
end

function JSON.lower(::JSON.JSONStyle, q::VgcQuantity)
    unit_str, u = _canonical_unit(q)
    return (value = canon_value(ustrip(u, q), dim_trait(q)), unit = unit_str)
end

StructUtils.structlike(::Type{<:VgcQuantity}) = false

function JSON.lift(::Type{<:VgcQuantity}, obj::_JSONObject)
    u = parse_unit(String(_getfield(obj, "unit")); compound = true)
    v = canon_value(_getfield(obj, "value"), dim_trait(u))
    return v * u
end

# =============================================================================
# Size2 Serialization
# =============================================================================

JSON.lower(::JSON.JSONStyle, s::Size2) = (width = s.width, height = s.height)

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

# Disc carries units: its center and σ may be quantities.
JSON.lower(::JSON.JSONStyle, d::Disc) = (center = (d.center[1], d.center[2]), σ = d.σ)

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
