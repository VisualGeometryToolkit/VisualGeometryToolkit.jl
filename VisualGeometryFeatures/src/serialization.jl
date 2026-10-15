# =============================================================================
# JSON3/StructTypes Serialization for Geometry Types
# =============================================================================
#
# Moved from VisualGeometryCore. Defines JSON serialization for:
#   - VgcQuantity (dimension-aware canonicalization, {value, unit} JSON)
#   - Size2 (width/height with optional units)
#   - ScalarOrQuantity (dispatch for Number vs Quantity)
#   - Disc (feature frame with unit-carrying center/scale)
#   - OrientedDisc, EllipticFrame, AffineFrame (simple struct serialization)
#
# Also provides _getfield/_hasfield helpers for unified JSON field access.

using JSON3: JSON3
using StructTypes: StructTypes
using VisualGeometryCore: VgcQuantity, ustrip, dimension, parse_unit, mm, px, dpi

# =============================================================================
# JSON Field Access Helpers
# =============================================================================

# Unified JSON field access across Dict, JSON3.Object, and NamedTuple.
# JSON3.Object and NamedTuple use Symbol keys; Dict uses String keys.
_hasfield(x::Dict, key::String) = haskey(x, key)
_hasfield(x, key::String) = haskey(x, Symbol(key))

_getfield(x::Dict, key::String) = x[key]
_getfield(x, key::String) = getproperty(x, Symbol(key))
_getfield(x, key::String, default) = _hasfield(x, key) ? _getfield(x, key) : default

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
# lengths).  This matches the legacy Unitful rule, which keyed on the
# LogicalUnits dimension appearing anywhere in the dimension product — so px,
# dpi, AND μm/px were all LogicalDim there too.
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

# A FlexUnits quantity is a Number, so the old
# Union{Number, Quantity} collapses; the alias survives for dispatch and
# call-site readability.
const ScalarOrQuantity = Number

# Construct ScalarOrQuantity from parsed JSON data
function StructTypes.construct(::Type{ScalarOrQuantity}, data::Number)
    return data
end

# The {value, unit} object form. The second argument is typed on purpose here and
# in the quantity and Size2 methods below: untyped, it overlapped StructTypes' own
# construct(::Type{T}, ::T), (::Type{<:Number}, ::String) and (::Type{T}, ::Nothing),
# so those calls threw an ambiguity error instead of dispatching.
StructTypes.construct(::Type{ScalarOrQuantity}, data::AbstractDict) =
    StructTypes.construct(VgcQuantity, data)

# =============================================================================
# Quantity Serialization ({value, unit} JSON objects)
# =============================================================================
#
# WRITER: canonical units only — length→"mm", pixel→"px", density→"dpi",
# pitch→"mm/px".  Deterministic output; new files hash at creation.
# PARSER: any unit name VisualGeometryCore's `parse_unit` reads — registered
# unit symbols and their products, quotients and powers (`mm/px`), the stack's
# one guarded `uparse` — so every string the legacy Unitful writer ever
# produced (observed corpus: mm, px, dpi, pd) loads, and old board JSONs keep
# their stored config hash (UIDs printed on physical boards stay valid —
# load_bundle never re-hashes).
# The writer keeps its own table: FlexUnits prints a registered unit as its
# symbol, but a compound one (`mm/px`) as its internal form, no name.

const _CANONICAL_UNITS = (("mm", mm), ("px", px), ("dpi", dpi), ("mm/px", mm / px))

function _canonical_unit(q)
    for (s, u) in _CANONICAL_UNITS
        dimension(q) == dimension(u) && return (s, u)
    end
    throw(ArgumentError("no canonical serialization unit for dimensions $(dimension(q))"))
end

StructTypes.StructType(::Type{<:VgcQuantity}) = StructTypes.CustomStruct()

function StructTypes.lower(q::VgcQuantity)
    unit_str, u = _canonical_unit(q)
    return (
        value = canon_value(ustrip(u, q), dim_trait(q)),
        unit = unit_str,
    )
end

function StructTypes.construct(::Type{<:VgcQuantity}, obj::AbstractDict)
    unit_str = obj["unit"]
    u = parse_unit(String(unit_str); compound = true)
    v = canon_value(obj["value"], dim_trait(u))
    return v * u
end

# =============================================================================
# Size2 Serialization
# =============================================================================

StructTypes.StructType(::Type{<:Size2}) = StructTypes.CustomStruct()

StructTypes.lower(s::Size2) = (width = s.width, height = s.height)

function StructTypes.construct(::Type{Size2}, x::Union{AbstractDict, NamedTuple})
    width = StructTypes.construct(ScalarOrQuantity, _getfield(x, "width"))
    height = StructTypes.construct(ScalarOrQuantity, _getfield(x, "height"))
    return Size2(width = width, height = height)
end

function StructTypes.construct(::Type{Size2{T}}, x::Union{AbstractDict, NamedTuple}) where {T}
    return StructTypes.construct(Size2, x)
end

# =============================================================================
# Feature Frame Serialization
# =============================================================================

# Disc uses CustomStruct to properly handle unit-carrying quantities
StructTypes.StructType(::Type{<:Disc}) = StructTypes.CustomStruct()

StructTypes.lower(d::Disc) = (center = d.center, σ = d.σ)

function StructTypes.construct(::Type{Disc}, x::Union{Dict{String, Any}, JSON3.Object, NamedTuple})
    center_data = _getfield(x, "center")
    center = Point2(
        StructTypes.construct(ScalarOrQuantity, center_data[1]),
        StructTypes.construct(ScalarOrQuantity, center_data[2])
    )
    σ = StructTypes.construct(ScalarOrQuantity, _getfield(x, "σ"))
    return Disc(center, σ)
end

# Other frame types use simpler serialization (typically no units)
StructTypes.StructType(::Type{<:OrientedDisc}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:EllipticFrame}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:AffineFrame}) = StructTypes.Struct()

# ...and are READ from a JSON object here. `Struct()` lowers `center` as [x, y]
# and a 2×2 matrix as its four entries in column order, but it cannot construct
# from the object `Detection`'s reader hands over: detections on these frames
# (every ASIFT detection is an EllipticFrame) could be written but not read.
_json_center(x) = (c = _getfield(x, "center"); Point2{Float64}(c[1], c[2]))
_json_mat2(v) = SMatrix{2, 2, Float64, 4}(v[1], v[2], v[3], v[4])

StructTypes.construct(::Type{OrientedDisc}, x::Union{Dict{String, Any}, JSON3.Object, NamedTuple}) =
    OrientedDisc{Float64}(_json_center(x), Float64(_getfield(x, "σ")), Float64(_getfield(x, "θ")))
StructTypes.construct(::Type{EllipticFrame}, x::Union{Dict{String, Any}, JSON3.Object, NamedTuple}) =
    EllipticFrame{Float64}(_json_center(x), _json_mat2(_getfield(x, "S")))
StructTypes.construct(::Type{AffineFrame}, x::Union{Dict{String, Any}, JSON3.Object, NamedTuple}) =
    AffineFrame{Float64}(_json_center(x), _json_mat2(_getfield(x, "A")))

