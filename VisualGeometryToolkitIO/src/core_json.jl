# =============================================================================
# Quantities, sizes and feature frames on the style — one codec, `JSON.lower` / `JSON.lift`
# =============================================================================
#
# Every number a record holds is in a unit the file's `units` object declares
# (the envelope, validate.jl; the style carries it, style.jl), never in one of
# its own: a length in `units.length`, or in a ROLE of `units.length_roles` (a
# board's pattern in pt), decimal-exact through `write_length`/`read_length`; a
# pixel count as its number of pixels; a density in `units.density`; an angle
# in radians. A value is written by `JSON.lower(::VgioStyle, x)` and read by
# `JSON.lift(::VgioStyle, T, x)` — StructUtils' functions, on IO's own style
# (no piracy) — as every VGC value a record embeds.
#
# WHAT A NUMBER IS, on read, is the field's to say: a pixel is dimensionless, so
# `640` cannot say whether it is 640 px, 640 mm or a plain 640. A field says it
# by its FIELD TAGS, StructUtils' fourth argument of `lower`/`lift`:
#
#   (; quantity = :length | :pixel | :density[, role = <a length role>])
#
# A lift without tags reads plain numbers (a frame's pixel coordinates, an image
# size in pixels counted as numbers). A writer lowers a quantity by its
# dimension (a length in the role its tags name, `:world` — `units.length` —
# otherwise); a tagged writer's value is checked against its tag, so nothing is
# written as a number its reader's tag would read as something else.
#
# THE FRAME CODEC is one codec over VisualGeometryToolkitCore's FeatureFrame
# interface (conventions/feature-frames.md, persisted-geometry.md): a frame is
# its `center` and its shape, each type a case of one method pair; an uncertain
# frame adds its `covariance` over the frame's `frame_dof` parameters, centre
# first, in the storage it holds (transform_schema.jl's `_cov_json`, the one
# covariance encoding). PointFrame is the 2-DOF case: `{"center"}` and the
# uncertain point's 2×2. A detection file (`vgio/detection`, detection_schema.jl)
# is built of these.
#
# UnitContract's own layout (`{"world", "density", "lengths"}`) is retired:
# a record's units are the envelope's. `migrate_units` gives the units object of
# a record that declared that layout (an owner's version step).

# -----------------------------------------------------------------------------
# One quantity
# -----------------------------------------------------------------------------

const _QUANTITY_TAGS = (:length, :pixel, :density)

# The quantity tag of a value, from its dimension (`:plain` for a plain number).
_quantity_of(::Real) = :plain
function _quantity_of(q::Quantity)
    d = dimension(q)
    d == dimension(m) && return :length
    d == dimension(px) && return :pixel
    d == dimension(dpi) && return :density
    throw(ArgumentError("a quantity of dimension $(repr(d)) ($q) has no declared unit: a record declares lengths, pixel counts and densities"))
end

# A field's tags: what its numbers are, and the length role.
function _tag_quantity(tags::NamedTuple)
    q = get(tags, :quantity, :plain)
    (q === :plain || q in _QUANTITY_TAGS) ||
        throw(ArgumentError("a field's `quantity` tag is one of :length, :pixel, :density, got $(repr(q))"))
    role = get(tags, :role, :world)
    (role === :world || q === :length) ||
        throw(ArgumentError("a length role ($(repr(role))) is for lengths; a field of $(repr(q)) has none"))
    return q, role
end

"""
    snap_integer(v) -> Real

A file number as a record writes it: an integral value as an `Int` (`640`,
`300`: a pixel count or a density), anything else as it is — a float that is
not integral or lies past Float64's exact integers, an integer past `Int`'s
range, a `Bool` (a flag is not a number: `true` stays `true`), a `Rational`
that is not integral as its `Float64`. A length is the Float64
[`write_length`](@ref) returns and is not snapped.
"""
snap_integer(v::AbstractFloat) = (isinteger(v) && abs(v) < maxintfloat(Float64)) ? Int(v) : v
snap_integer(v::Integer) = typemin(Int) <= v <= typemax(Int) ? Int(v) : v
snap_integer(v::Bool) = v   # a flag, not a number
snap_integer(v::Rational) = isinteger(v) ? Int(v) : Float64(v)
snap_integer(v) = v

# A value as the number the style declares for it, under the field's tags: a
# plain number as it is (its tag `:plain`); a quantity by its dimension, which
# the tag (when it states one) must match.
function _number_json(s::VgioStyle, x, tags::NamedTuple)
    q, role = _tag_quantity(tags)
    k = _quantity_of(x)
    (haskey(tags, :quantity) || k === :plain) && k !== q && throw(
        ArgumentError("a field of $(repr(q)) holds $(x), a $(k) value: it would read back as something else")
    )
    k === :plain && return x isa AbstractFloat ? Float64(x) : x   # a Float32 exactly, as its Float64
    k === :length && return role === :world ? _leaf_length(x, s.length_unit) : write_length(x, role_unit(s.contract, role))
    k === :pixel && return snap_integer(Float64(ustrip(px, x)))
    return snap_integer(Float64(ustrip(s.contract.density, x)))
end
# …and back: the number `x` a field of these tags holds.
function _number_from_json(s::VgioStyle, x, tags::NamedTuple)
    q, role = _tag_quantity(tags)
    v = _number(x, q === :plain ? "a plain field" : "a $(q) field")
    q === :plain && return v
    q === :length && return read_length(v, role_unit(s.contract, role))
    q === :pixel && return Float64(v) * px
    return Float64(v) * s.contract.density
end

# A quantity field holds a number; anything else is refused, an object naming
# the retired `{value, unit}` form.
_refuse_quantity(x::Union{AbstractDict, NamedTuple}, what) = throw(
    ArgumentError(
        "$(what) holds an object, $(repr(x)); a quantity is a number in the file's declared unit " *
            "(the {value, unit} form is read only where an earlier schema version held it, by its owner's " *
            "version step on load; vgio_migrate rewrites such a file)"
    )
)
_refuse_quantity(x, what) = throw(ArgumentError("$(what) holds a number, got $(repr(x))"))
_number(x, what) = (x isa Real && !(x isa Bool)) ? x : _refuse_quantity(x, what)

# A key of a parsed object (a `JSON.Object`, a Dict with String or Symbol keys, a NamedTuple).
_get_key(x::AbstractDict, key::String) = haskey(x, key) ? x[key] : x[Symbol(key)]
_get_key(x::NamedTuple, key::String) = getproperty(x, Symbol(key))
_has_key(x::AbstractDict, key::String) = haskey(x, key) || haskey(x, Symbol(key))
_has_key(x::NamedTuple, key::String) = haskey(x, Symbol(key))

# A quantity, by its dimension (a length in the `:world` role, or the role its tags name).
JSON.lower(s::VgioStyle, q::Quantity) = _number_json(s, q, (;))
JSON.lower(s::VgioStyle, q::Quantity, tags::NamedTuple) = _number_json(s, q, tags)
JSON.lift(s::VgioStyle, ::Type{<:Quantity}, x, tags::NamedTuple) = (
    haskey(tags, :quantity) || throw(ArgumentError("a quantity's field states what it is: tags (; quantity = :length | :pixel | :density)"));
    _number_from_json(s, x, tags)
)

# -----------------------------------------------------------------------------
# Size2: {"width", "height"}
# -----------------------------------------------------------------------------

JSON.lower(s::VgioStyle, sz::Size2, tags::NamedTuple = (;)) =
    (width = _number_json(s, sz.width, tags), height = _number_json(s, sz.height, tags))
JSON.lift(s::VgioStyle, ::Type{<:Size2}, x, tags::NamedTuple = (;)) =
    Size2(width = _number_from_json(s, _get_key(x, "width"), tags), height = _number_from_json(s, _get_key(x, "height"), tags))

# -----------------------------------------------------------------------------
# The retired UnitContract layout: an owner's version step
# -----------------------------------------------------------------------------

"""
    migrate_units(declaration) -> Dict{String, Any}

The IO 0.10 `units` object (the envelope's: `length`, `time`, `angle`, `pixel`,
`density`, `radiance`, `length_roles`) of a record that declared its units in
the retired UnitContract layout, `{"world": "mm", "density": "dpi", "lengths":
{"pattern": "pt"}}` (a BlobBoards board before IO 0.10): `world` is `length`,
the further roles `length_roles`, a missing `density` dpi (the layout's own
rule), and what the layout never declared is what its records meant — colmap
pixels, seconds, radians, a relative radiance. For an owner's version step
([`migrate_schema`](@ref)); a 0.10 record states no units of its own
([`write_record`](@ref) writes the style's). An `ArgumentError` for a unit
name that is not registered or of the wrong dimension, and for `world` named
again under `lengths`.
"""
function migrate_units(d::Union{AbstractDict, NamedTuple})
    world = parse_unit(String(_get_key(d, "world")))
    density = _has_key(d, "density") ? parse_unit(String(_get_key(d, "density"))) : dpi
    roles = _has_key(d, "lengths") ? _get_key(d, "lengths") : Dict{String, Any}()
    roles isa Union{AbstractDict, NamedTuple} ||
        throw(ArgumentError("a units declaration's `lengths` is an object of role => unit name, got $(repr(roles))"))
    ls = NamedTuple(Symbol(k) => parse_unit(String(v)) for (k, v) in pairs(roles))
    haskey(ls, :world) && throw(ArgumentError("a units declaration names `world` twice: at its top level and under `lengths`"))
    return _lower(_units_json(VgioStyle(UnitContract(density, merge((world = world,), ls)))))
end

# -----------------------------------------------------------------------------
# THE frame codec: every FeatureFrame a case
# -----------------------------------------------------------------------------

# The tag of each frame type in a file (the tag rule: snake_case of its name).
const _FRAME_TAGS = VgtkCore.TagRegistry(:feature_frame, "",
    "point_frame" => PointFrame, "disc" => Disc, "oriented_disc" => OrientedDisc,
    "elliptic_frame" => EllipticFrame, "affine_frame" => AffineFrame)

# A frame's shape after its centre, per type: its keys and values (a matrix in rows), and back.
_shape_json(::PointFrame, n) = (;)
_shape_json(f::Disc, n) = (sigma = n(f.σ),)
_shape_json(f::OrientedDisc, n) = (sigma = n(f.σ), angle = _angle_json(f.θ))
_shape_json(f::EllipticFrame, n) = (shape_matrix = _rows_json(f.S, n),)
_shape_json(f::AffineFrame, n) = (affine_matrix = _rows_json(f.A, n),)
_rows_json(M, n) = [[n(M[1, 1]), n(M[1, 2])], [n(M[2, 1]), n(M[2, 2])]]
_angle_json(θ::Real) = Float64(θ)   # radians (`units.angle`)

_frame_from(::Type{<:PointFrame}, ::Type{T}, c, x, n) where {T} = PointFrame{T}(c)
_frame_from(::Type{<:Disc}, ::Type{T}, c, x, n) where {T} = Disc{T}(c, n(_get_key(x, "sigma")))
_frame_from(::Type{<:OrientedDisc}, ::Type{T}, c, x, n) where {T} =
    OrientedDisc{T}(c, n(_get_key(x, "sigma")), T(_number(_get_key(x, "angle"), "an angle")))
_frame_from(::Type{<:EllipticFrame}, ::Type{T}, c, x, n) where {T} = EllipticFrame{T}(c, _rows_from(_get_key(x, "shape_matrix"), T, n))
_frame_from(::Type{<:AffineFrame}, ::Type{T}, c, x, n) where {T} = AffineFrame{T}(c, _rows_from(_get_key(x, "affine_matrix"), T, n))
_rows_from(r, ::Type{T}, n) where {T} = SMatrix{2, 2, T, 4}(n(r[1][1]), n(r[2][1]), n(r[1][2]), n(r[2][2]))

# The element type a lift builds: the type's own parameter, Float64 for a plain
# frame named without one, a quantity's for a tagged one.
_frame_eltype(::Type{<:FeatureFrame{T}}, tags) where {T} = T
_frame_eltype(::Type{<:FeatureFrame}, tags) = first(_tag_quantity(tags)) === :plain ? Float64 : BaseQuantity

"""
    JSON.lower(s::VgioStyle, f::FeatureFrame[, tags])
    JSON.lower(s::VgioStyle, u::Uncertain{<:FeatureFrame})
    JSON.lift(s::VgioStyle, F::Type{<:FeatureFrame}, x[, tags])
    JSON.lift(s::VgioStyle, Uncertain{F}, x)

THE frame codec, one over VisualGeometryToolkitCore's FeatureFrame interface:
a frame is `{"center": [x, y]}` and its shape — a `PointFrame` nothing, a
`Disc` `"sigma"`, an `OrientedDisc` `"sigma"` and `"angle"` (radians), an
`EllipticFrame` `"shape_matrix"` and an `AffineFrame` `"affine_matrix"`, each
in rows. Its numbers are plain (pixel coordinates in the file's pixel frame) or
quantities under the field's tags (`(; quantity = :length[, role])`). The lift
builds `F`'s element type (`PointFrame{Float32}` reads Float32; a type without
one plain Float64).

An uncertain frame adds its `"covariance"`: the frame's `frame_dof` ×
`frame_dof` covariance over its parameters, the centre `(x, y)` first, in the
storage it holds (rows, `{"diagonal": […]}`, `{"isotropic": λ}`); a
`PointFrame`'s is the uncertain point's 2×2. Its entries are the frame's own
numbers' (px², never converted): an uncertain frame holds plain numbers. A
covariance of another size, a non-finite one (a detection states why it has
none: `vgio/detection`'s `covariance_status`) and a labeled one are an
`ArgumentError`. A precision form is written as its covariance and reads back
as one.

```jldoctest
julia> import JSON

julia> s = VgtkIO.VgioStyle();

julia> JSON.json(OrientedDisc(1.0, 2.0, 3.0, 0.5); style = s)
"{\\"center\\":[1.0,2.0],\\"sigma\\":3.0,\\"angle\\":0.5}"

julia> u = Uncertain(PointFrame(10.5, 20.25), CovMat{2, Float64}(0.5, 0.0, 0.0, 0.25));

julia> JSON.json(u; style = s)
"{\\"center\\":[10.5,20.25],\\"covariance\\":[[0.5,0.0],[0.0,0.25]]}"

julia> JSON.lift(s, Uncertain{PointFrame{Float64}}, JSON.parse(JSON.json(u; style = s))) == u
true
```
"""
function JSON.lower(s::VgioStyle, f::FeatureFrame, tags::NamedTuple = (;))
    n = x -> _number_json(s, x, tags)
    return merge((center = [n(f.center[1]), n(f.center[2])],), _shape_json(f, n))
end
function JSON.lift(s::VgioStyle, ::Type{F}, x, tags::NamedTuple = (;)) where {F <: FeatureFrame}
    T = _frame_eltype(F, tags)
    n = v -> T(_number_from_json(s, v, tags))
    c = _get_key(x, "center")
    return _frame_from(F, T, Point2{T}(n(c[1]), n(c[2])), x, n)
end

function JSON.lower(s::VgioStyle, u::Uncertain{<:FeatureFrame})
    f = geom(u)
    Σ = cov(u)
    f isa FeatureFrame{<:Real} ||
        throw(ArgumentError("an uncertain frame's covariance is in its own numbers' units (px²): its frame holds plain numbers, got $(typeof(f))"))
    _frame_cov_size(f, Σ)
    _covariance_label(Σ) <: UnlabeledFrame ||
        throw(ArgumentError("a frame's covariance is unlabeled in a file (its frame is the detection's); got one labeled $(_covariance_label(Σ))"))
    all(isfinite, Σ) || throw(ArgumentError(
        "a frame's covariance is finite; a detection without one omits it and states its `covariance_status` (vgio/detection)"))
    return merge(JSON.lower(s, f), (covariance = _cov_json(Σ),))
end
JSON.lift(s::VgioStyle, ::Type{<:Uncertain{F}}, x) where {F <: FeatureFrame} = _uncertain_frame(JSON.lift(s, F, x), _get_key(x, "covariance"))
# The frame with the covariance `c` (a wire form) over its parameters.
function _uncertain_frame(f::FeatureFrame, c)
    Σ = _cov_matrix(c isa NamedTuple ? _lower(c) : c, frame_dof(f))
    return Uncertain(f, CovMat(Σ))
end
_covariance_label(::CovMat{N, T, M, C}) where {N, T, M, C} = C
_frame_cov_size(f, Σ) = size(Σ, 1) == frame_dof(f) || throw(ArgumentError(
    "a $(nameof(typeof(f)))'s covariance is over its $(frame_dof(f)) parameters (frame_dof: the centre (x, y) first, then its shape), got $(size(Σ, 1))×$(size(Σ, 2))"))
