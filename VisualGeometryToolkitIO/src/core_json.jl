# =============================================================================
# The quantity JSON format — numbers in the declared unit
# =============================================================================
#
# A quantity in a JSON record is a NUMBER in the unit its record declares
# (`"units": {"world": "mm", "density": "dpi"}`, a `UnitContract`): a length
# in the world unit, a density in the density unit, a pixel count as a number
# of pixels. The unit is not repeated beside every value. What a number IS —
# a length, a pixel count, a density — is the record's to say, field by field
# (`read_quantity(x, kind, contract)`, the required `kind` keyword of
# `json_value` and `from_json`, never defaulted: a default would hand a
# length or a pixel count back as a bare number),
# because a pixel is dimensionless and a number alone cannot say it.
#
# ONE FORMAT. A `{"value": v, "unit": u}` object (the form
# VisualGeometryToolkitFeatures' JSON.lower hooks wrote) is not read: a file
# holding one is migrated once by its owner (vgio_migrate, migrate.jl).
#
# LENGTH ROLES. A record whose lengths are declared in more than one unit (a
# board's canvas in mm and its pattern in pt) names each further unit as a
# ROLE of its contract, `UnitContract(world = mm, lengths = (pattern = pt,))`,
# declared `"units": {"world": "mm", "density": "dpi", "lengths": {"pattern":
# "pt"}}`; a length field read or written in a role names it (`role =
# :pattern`), and `:world` is the default.
#
# The codecs of VisualGeometryToolkitCore's sizes, feature frames and
# detections are explicit functions, `json_value(x, contract)` and
# `json_value(x, contract; kind)` and
# `from_json(T, d, contract; kind)`, not methods of JSON.jl's `lower`/`lift`
# (those are JSON.jl's functions on Core's types: piracy).

"""
    ScalarOrQuantity

The type of a JSON-held value that is either a plain number or a quantity: an
alias of `Number`, since a FlexUnits quantity is a `Number`. A field of this
type is read with [`read_quantity`](@ref) when it holds a quantity, and kept
as it is when it holds a plain number.

Public, not exported.
"""
const ScalarOrQuantity = Number

# -----------------------------------------------------------------------------
# One quantity
# -----------------------------------------------------------------------------

const _QUANTITY_KINDS = (:length, :pixel, :density)

# The dimension a kind's quantities have (px is dimensionless).
function _kind_dimension(kind::Symbol)
    kind === :length && return dimension(m)
    kind === :pixel && return dimension(px)
    kind === :density && return dimension(dpi)
    throw(ArgumentError("a quantity's kind is one of $(join(repr.(_QUANTITY_KINDS), ", ")), got $(repr(kind))"))
end

# The kind of a quantity, from its dimension.
function _quantity_kind(q::Quantity)
    d = dimension(q)
    d == dimension(m) && return :length
    d == dimension(px) && return :pixel
    d == dimension(dpi) && return :density
    throw(
        ArgumentError(
            "a quantity of dimension $(repr(d)) ($q) has no declared unit: a record declares lengths, " *
                "pixel counts and densities only"
        )
    )
end

# A pixel count or a density is written as an integer when it is integral
# (`640`, `300`), as the legacy writer did; a length is the Float64 `write_length` returns.
_snap_integer(v::AbstractFloat) = (isinteger(v) && abs(v) < maxintfloat(Float64)) ? Int(v) : v

"""
    write_quantity(q::Quantity, contract::UnitContract; role = :world) -> Real

The number a JSON record holds for the quantity `q` under `contract`: a length
in the unit of the contract's length `role` (`:world` is the default role, a
named role's otherwise; through [`write_length`](@ref), decimal-exact), a pixel
count (a dimensionless quantity: px is dimensionless) as its number of pixels,
a density in `contract.density`. A pixel count or a density is an integer when
it is integral. An `ArgumentError` for a quantity of any other dimension, a
role the contract does not declare, or a role given for a quantity that is not
a length.

```jldoctest
julia> VgtkIO.write_quantity(62.8mm, MM_PX)
62.8

julia> VgtkIO.write_quantity(640px, MM_PX), VgtkIO.write_quantity(300dpi, MM_PX)
(640, 300)
```
"""
function write_quantity(q::Quantity, contract::UnitContract; role::Symbol = :world)
    kind = _quantity_kind(q)
    _check_role(kind, role)
    kind === :length && return write_length(q, role_unit(contract, role))
    kind === :pixel && return _snap_integer(Float64(ustrip(px, q)))
    return _snap_integer(Float64(ustrip(contract.density, q)))
end
_check_role(kind::Symbol, role::Symbol) = (kind === :length || role === :world) || throw(
    ArgumentError("a length role ($(repr(role))) is for lengths; a $(kind) quantity has none")
)

# A parsed JSON object, any dictionary with string keys, or a NamedTuple.
const _JSONObject = Union{AbstractDict, NamedTuple}
_has_key(x::AbstractDict, key::String) = haskey(x, key) || haskey(x, Symbol(key))
_has_key(x::NamedTuple, key::String) = haskey(x, Symbol(key))
_get_key(x::AbstractDict, key::String) = haskey(x, key) ? x[key] : x[Symbol(key)]
_get_key(x::NamedTuple, key::String) = getproperty(x, Symbol(key))
_key_names(x::_JSONObject) = (String(k) for k in keys(x))

# A quantity field holds a number; anything else is refused, an object naming
# the retired `{value, unit}` form.
_refuse_quantity(x::_JSONObject, what) = throw(
    ArgumentError(
        "$(what) holds an object, $(repr(x)); a quantity is a number in the record's declared unit " *
            "(the {value, unit} form is not read: its owner migrates such a file once, with vgio_migrate)"
    )
)
_refuse_quantity(x, what) = throw(ArgumentError("$(what) holds a number, got $(repr(x))"))
_number(x, what) = (x isa Real && !(x isa Bool)) ? x : _refuse_quantity(x, what)

"""
    read_quantity(x, kind::Symbol, contract::UnitContract; role = :world) -> Quantity

The quantity a JSON record holds as `x`, a NUMBER in a field of `kind` —
`:length`, `:pixel` or `:density` — under `contract`: a length in the unit of
the contract's length `role` (`:world`, the default),
through [`read_length`](@ref); a pixel count in pixels; a density in
`contract.density`. An `ArgumentError` for an unknown kind, a role the
contract does not declare (or one given for a pixel count or a density), and
a value that is not a number (a `{"value": v, "unit": u}` object is not read).

```jldoctest
julia> VgtkIO.read_quantity(2.5, :length, MM_PX) == 2.5mm
true

julia> VgtkIO.read_quantity(300, :density, MM_PX) == 300dpi
true
```
"""
function read_quantity(x, kind::Symbol, contract::UnitContract; role::Symbol = :world)
    _kind_dimension(kind)
    _check_role(kind, role)
    v = _number(x, "a $(kind) field")
    kind === :length && return read_length(v, role_unit(contract, role))
    kind === :pixel && return Float64(v) * px
    return Float64(v) * contract.density
end

# A field of a codec: `:plain` (plain numbers) or a quantity kind.
function _read_field(x, kind::Symbol, contract::UnitContract; role::Symbol = :world)
    kind === :plain || return read_quantity(x, kind, contract; role)
    return _number(x, "a plain field")
end

_write_field(x::Real, ::UnitContract; role::Symbol = :world) = x
_write_field(q::Quantity, contract::UnitContract; role::Symbol = :world) = write_quantity(q, contract; role)

# A value is written under the kind its caller declares, the one its reader
# will declare: a number of any other kind (a width in px under :length, a
# plain number under :pixel) is refused, so nothing is written as a number
# that reads back as something else.
_kind_of(::Real) = :plain
_kind_of(q::Quantity) = _quantity_kind(q)
const _VALUE_KINDS = (:plain, :length, :pixel)
function _check_kind(what, kind::Symbol, xs...)
    kind in _VALUE_KINDS ||
        throw(ArgumentError("$(what): the kind is one of $(join(repr.(_VALUE_KINDS), ", ")), got $(repr(kind))"))
    for x in xs
        _kind_of(x) === kind || throw(
            ArgumentError("$(what) holds $(x), a $(_kind_of(x)) value, but is written as $(repr(kind))")
        )
    end
    return nothing
end

# -----------------------------------------------------------------------------
# The contract's own declaration
# -----------------------------------------------------------------------------

"""
    json_value(x, contract::UnitContract; kind, role = :world)
    json_value(contract::UnitContract)

The plain value (a NamedTuple of numbers, arrays and strings, which JSON.jl's
`JSON.json` writes) a JSON record holds for `x` under `contract`, its
quantities as numbers in the declared units ([`write_quantity`](@ref)); read
it back with [`from_json`](@ref). Defined for VisualGeometryToolkitCore's
`Size2` (`{"width": w, "height": h}`), `Disc` (`{"center": [x, y], "σ": s}`),
`OrientedDisc` (`… "θ": angle`), `EllipticFrame` (`{"center": …, "S": [S11,
S21, S12, S22]}`, column order), `AffineFrame` (`… "A": …`) and a detection —
an `Attributed` feature frame, VisualGeometryToolkitCore's
`AbstractDetection` — written as its frame under `frame`, an `Uncertain`
frame's covariance entries (column order) under `cov` (`null` when not
finite), and each metadata field under its own name.

`kind`, required, declares what the numbers of the value are — `:plain`
(plain numbers), `:length` or `:pixel` — the same `kind` its reader gives
`from_json`. A value holding a number of another kind, and a quantity in a
detection's metadata (it has no declared kind), are an `ArgumentError`. The
`OrientedDisc`, `EllipticFrame` and `AffineFrame` hold plain numbers only
(`kind = :plain`). The lengths of a `Size2` or a `Disc` are written in the
unit of the contract's length `role` (`:world` by default).

`json_value(contract)` is the record's declaration of its units, `(world =
"mm", density = "dpi")` for [`MM_PX`](@ref), with `lengths = (pattern =
"pt", …)` when the contract declares further length roles (registered unit
names; an `ArgumentError` for a unit without one).

```jldoctest
julia> import JSON

julia> JSON.json(VgtkIO.json_value(Size2(width = 640px, height = 480px), MM_PX; kind = :pixel))
"{\\"width\\":640,\\"height\\":480}"

julia> JSON.json(VgtkIO.json_value(Disc(Point2(1.0mm, 2.0mm), 3.0mm), UnitContract(world = m); kind = :length))
"{\\"center\\":[0.001,0.002],\\"σ\\":0.003}"

julia> VgtkIO.json_value(MM_PX)
(world = "mm", density = "dpi")

julia> VgtkIO.json_value(UnitContract(world = mm, lengths = (pattern = pt,)))
(world = "mm", density = "dpi", lengths = (pattern = "pt",))
```
"""
function json_value(c::UnitContract)
    # The file layout, known here alone: `world` top level, the other roles under `lengths`.
    base = (world = unit_string(role_unit(c)), density = unit_string(c.density))
    rest = _other_roles(c.lengths)
    isempty(rest) && return base
    return merge(base, (lengths = map(unit_string, rest),))
end

"""
    from_json(T, d, contract::UnitContract; kind, role = :world)
    from_json(UnitContract, d)

The value of type `T` a JSON record holds as `d` (a parsed `JSON.Object`, any
dictionary with string keys, or a NamedTuple), the inverse of
[`json_value`](@ref): `T` is `Size2`, `Disc`, `OrientedDisc`,
`EllipticFrame`, `AffineFrame` or a detection type (VisualGeometryToolkitCore's
`AbstractDetection`, or VisualGeometryToolkitFeatures' `Detection`). `kind`,
required and never defaulted, declares what the numbers of a `Size2` or a
`Disc` are — `:plain` (plain numbers), `:length`, `:pixel` — and is read
through [`read_quantity`](@ref) (a length in the unit of the contract's
length `role`); the other frames hold plain numbers (`kind = :plain`). Every
quantity is a number: a `{"value": v, "unit": u}` object is refused. A
detection's frame type is read off its keys
(`A`, `S`, `θ`, else a `Disc`); its documented metadata fields (`response`,
`edge_score`, `laplacian_scale_score`) come first, as Float64, and the others
follow sorted by name, each as it was parsed; a `cov` makes the frame
`Uncertain` again.

`from_json(UnitContract, d)` reads a record's `"units"` declaration; a
declaration without a `density` has dpi, one without `lengths` no further
length roles.

```jldoctest
julia> import JSON

julia> VgtkIO.from_json(Size2, JSON.parse("{\\"width\\": 640, \\"height\\": 480}"), MM_PX; kind = :pixel) ==
           Size2(width = 640px, height = 480px)
true

julia> VgtkIO.from_json(UnitContract, JSON.parse("{\\"world\\": \\"m\\"}")) == UnitContract(world = m)
true
```
"""
function from_json(::Type{UnitContract}, d::_JSONObject)
    world = parse_unit(String(_get_key(d, "world")))
    density = _has_key(d, "density") ? parse_unit(String(_get_key(d, "density"))) : dpi
    _has_key(d, "lengths") || return UnitContract(world, density)
    ls = _get_key(d, "lengths")
    ls isa _JSONObject || throw(ArgumentError("a units declaration's `lengths` is an object of role => unit name, got $(repr(ls))"))
    lengths = NamedTuple(Symbol(k) => parse_unit(String(_get_key(ls, String(k)))) for k in _key_names(ls))
    haskey(lengths, :world) && throw(ArgumentError(
        "a units declaration names `world` twice: at its top level and under `lengths` (the file holds `world` top level only)"))
    return UnitContract(world, density; lengths)
end

# -----------------------------------------------------------------------------
# Size2 and the feature frames
# -----------------------------------------------------------------------------

function json_value(s::Size2, contract::UnitContract; kind::Symbol, role::Symbol = :world)
    _check_kind("a Size2", kind, s.width, s.height)
    return (width = _write_field(s.width, contract; role), height = _write_field(s.height, contract; role))
end

function from_json(::Type{<:Size2}, d::_JSONObject, contract::UnitContract; kind::Symbol, role::Symbol = :world)
    width = _read_field(_get_key(d, "width"), kind, contract; role)
    height = _read_field(_get_key(d, "height"), kind, contract; role)
    return Size2(width = width, height = height)
end

function json_value(f::Disc, contract::UnitContract; kind::Symbol, role::Symbol = :world)
    _check_kind("a Disc", kind, f.center[1], f.center[2], f.σ)
    return (center = (_write_field(f.center[1], contract; role), _write_field(f.center[2], contract; role)), σ = _write_field(f.σ, contract; role))
end

function from_json(::Type{<:Disc}, d::_JSONObject, contract::UnitContract; kind::Symbol, role::Symbol = :world)
    c = _get_key(d, "center")
    center = Point2(_read_field(c[1], kind, contract; role), _read_field(c[2], kind, contract; role))
    return Disc(center, _read_field(_get_key(d, "σ"), kind, contract; role))
end

_center_value(f) = (Float64(f.center[1]), Float64(f.center[2]))
_matrix_value(M) = collect(Float64, vec(M))
_center_from(d) = (c = _get_key(d, "center"); Point2{Float64}(c[1], c[2]))
_matrix_from(v) = SMatrix{2, 2, Float64, 4}(v[1], v[2], v[3], v[4])

_plain_frame_kind(T, kind) = kind === :plain ||
    throw(ArgumentError("a $(T) holds plain numbers (pixels and radians); its kind is :plain, got $(repr(kind))"))

json_value(f::OrientedDisc{<:Real}, ::UnitContract; kind::Symbol) =
    (_plain_frame_kind(OrientedDisc, kind); (center = _center_value(f), σ = f.σ, θ = f.θ))
json_value(f::EllipticFrame{<:Real}, ::UnitContract; kind::Symbol) =
    (_plain_frame_kind(EllipticFrame, kind); (center = _center_value(f), S = _matrix_value(f.S)))
json_value(f::AffineFrame{<:Real}, ::UnitContract; kind::Symbol) =
    (_plain_frame_kind(AffineFrame, kind); (center = _center_value(f), A = _matrix_value(f.A)))

function from_json(::Type{<:OrientedDisc}, d::_JSONObject, ::UnitContract; kind::Symbol)
    _plain_frame_kind(OrientedDisc, kind)
    return OrientedDisc{Float64}(_center_from(d), Float64(_get_key(d, "σ")), Float64(_get_key(d, "θ")))
end
function from_json(::Type{<:EllipticFrame}, d::_JSONObject, ::UnitContract; kind::Symbol)
    _plain_frame_kind(EllipticFrame, kind)
    return EllipticFrame{Float64}(_center_from(d), _matrix_from(_get_key(d, "S")))
end
function from_json(::Type{<:AffineFrame}, d::_JSONObject, ::UnitContract; kind::Symbol)
    _plain_frame_kind(AffineFrame, kind)
    return AffineFrame{Float64}(_center_from(d), _matrix_from(_get_key(d, "A")))
end

# -----------------------------------------------------------------------------
# Detections: an Attributed feature frame, plain or Uncertain
# -----------------------------------------------------------------------------

# Metadata has no declared kind: a quantity there cannot be written as a number.
_holds_quantity(::Quantity) = true
_holds_quantity(x::AbstractArray) = any(_holds_quantity, x)
_holds_quantity(x::Union{Tuple, NamedTuple}) = any(_holds_quantity, x)
_holds_quantity(_) = false
function _metadata_value(meta::NamedTuple)
    for (k, v) in pairs(meta)
        _holds_quantity(v) && throw(
            ArgumentError(
                "detection metadata `$(k)` holds a quantity ($(v)); metadata has no declared kind, " *
                    "so write it as a plain number in a unit its name states"
            )
        )
    end
    return meta
end

function json_value(d::Attributed{<:FeatureFrame}, contract::UnitContract; kind::Symbol)
    meta = _metadata_value(getfield(d, :attributes))
    return (frame = json_value(getfield(d, :value), contract; kind), pairs(meta)...)
end

# An uncertain frame writes its covariance as `cov`, the matrix's entries in
# column order — or `null` when it is not finite (an unrefined keypoint's),
# which JSON cannot hold. A frame of `kind = :length` (every parameter a
# length: a Disc's center and σ) holds its covariance in base units, m², and
# writes it in the declared length unit squared, as it writes its center; any
# other kind's entries are written as they are (pixels are dimensionless).
_cov_entry_json(x::Float64, contract::UnitContract, kind::Symbol) = kind === :length ? _write_power(x, role_unit(contract), 2) : x
_cov_entry_from_json(x::Real, contract::UnitContract, kind::Symbol) = kind === :length ? _read_power(x, role_unit(contract), 2) : Float64(x)
function json_value(d::Attributed{<:Uncertain{<:FeatureFrame, <:CovMat}}, contract::UnitContract; kind::Symbol)
    u = getfield(d, :value)
    meta = _metadata_value(getfield(d, :attributes))
    Σ = cov(u)
    c = all(isfinite, Σ) ? Float64[_cov_entry_json(Float64(x), contract, kind) for x in vec(Σ)] : nothing
    return (frame = json_value(geom(u), contract; kind), cov = c, pairs(meta)...)
end

# The metadata fields every detector writes as Float64, in their canonical order.
const _FLOAT_METADATA = (:response, :edge_score, :laplacian_scale_score)

function from_json(::Type{<:AbstractDetection}, d::_JSONObject, contract::UnitContract; kind::Symbol)
    frame_data = _get_key(d, "frame")
    fr = if _has_key(frame_data, "A")
        from_json(AffineFrame, frame_data, contract; kind)
    elseif _has_key(frame_data, "S")
        from_json(EllipticFrame, frame_data, contract; kind)
    elseif _has_key(frame_data, "θ")
        from_json(OrientedDisc, frame_data, contract; kind)
    else
        from_json(Disc, frame_data, contract; kind)
    end
    # Every key but `frame` and `cov` is metadata: the documented fields first,
    # as Float64 (a writer may print 1.0 as 1), then the others sorted by name,
    # so the result does not depend on the key order of the input.
    present = Symbol[Symbol(k) for k in _key_names(d) if k ∉ ("frame", "cov")]
    names = (filter(in(present), collect(_FLOAT_METADATA))..., sort!(setdiff(present, _FLOAT_METADATA))...)
    vals = Tuple(n in _FLOAT_METADATA ? Float64(_get_key(d, String(n))) : _get_key(d, String(n)) for n in names)
    meta = NamedTuple{names}(vals)
    _has_key(d, "cov") || return Attributed(fr, meta)
    c = _get_key(d, "cov")
    N = _cov_size(fr)
    (c === nothing || length(c) == N * N) ||
        throw(ArgumentError("a $(nameof(typeof(fr))) detection's `cov` has $(N * N) entries ($(N)×$(N)), got $(length(c))"))
    Σ = c === nothing ? fill(NaN, N, N) : reshape(Float64[_cov_entry_from_json(x, contract, kind) for x in c], N, N)
    return Attributed(Uncertain(fr, CovMat{N, Float64, N * N}(Tuple(Σ))), meta)
end
# The size of a frame's covariance, its number of parameters — the frame's
# type decides it, also for a `null` covariance, which states no size.
_cov_size(::Disc) = 3            # center, σ
_cov_size(::OrientedDisc) = 4    # center, σ, θ
_cov_size(::EllipticFrame) = 5   # center, S (symmetric)
_cov_size(::AffineFrame) = 6     # center, A
