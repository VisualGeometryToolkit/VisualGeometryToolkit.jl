# =============================================================================
# Units — FlexUnits as it ships, plus four registered units
# =============================================================================
#
# No stack quantity type, dimension set or registry: values are FlexUnits'
# standard base-form quantities, `Quantity{T, Dimensions{FixRat32}}` (a bits
# type), and unit names resolve through FlexUnits' own registry,
# `FlexUnits.UnitRegistry`. The stack adds four units to that registry in
# `__init__` (a registration made while precompiling does not persist):
#
# - `px`  — a pixel count, DIMENSIONLESS (FlexUnits has the seven SI
#   dimensions only, so a pixel is indistinguishable from a radian; pixel
#   coordinates keep their safety through `PixelFrame` tags);
# - `pd`  — the board JSONs' print dot, equal to `px`;
# - `dpi` — `px/inch`, a density of dimension 1/m;
# - `pt`  — the typographic point, `inch/72`, a length.
#
# Values are stored in base units: metres for a length. A bare `ustrip(q)` of a
# length is in metres; code that wants millimetres names the unit,
# `ustrip(mm, q)`. See conventions/units.md.

# The unit constants: dynamic units over `Dimensions{FixRat32}`, values fixed
# when the package is precompiled — no registry lookup at load. The SI ones are
# FlexUnits' own; the stack's four are built from them.
const mm = UnitRegistry.UNITS[:mm]
const cm = UnitRegistry.UNITS[:cm]
const μm = UnitRegistry.UNITS[:μm]
const m = UnitRegistry.UNITS[:m]
const inch = UnitRegistry.UNITS[:inch]
const ft = UnitRegistry.UNITS[:ft]
const px = Units("px" => UnitRegistry.UNITS[:NoDims])
const pt = Units("pt" => (1 / 72) * inch)
const dpi = Units("dpi" => px / inch)
const _pd = Units("pd" => px)   # the board JSONs' unit name for a pixel (print dot), equal to px

# The stack's units, registered into FlexUnits' registry by `__init__`.
const _STACK_UNITS = (px, _pd, dpi, pt)

# Register the stack's units into `FlexUnits.UnitRegistry`, skipping one
# already there (a second load, or another package that registered the same
# unit). The registry is permanent: a name already bound to a different unit is
# FlexUnits' `PermanentDictError`.
function _register_stack_units()
    for u in _STACK_UNITS
        name = string(u)
        haskey(UnitRegistry.UNITS, Symbol(name)) || UnitRegistry.register_unit(name => u)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Unit names from outside: the one guarded parse, and its inverse
# -----------------------------------------------------------------------------
#
# FlexUnits' `uparse` evaluates what it parses, so a string from a file, a QR
# code or Python never reaches it unchecked. This is the stack's ONE check:
# every package reads a unit name through `parse_unit` and writes one through
# `unit_string` (FlexUnits' own printing, a registered unit's symbol).

"""
    parse_unit(s::AbstractString; compound = false) -> Units

The unit a serialized name `s` spells, resolved by FlexUnits' registry
(`FlexUnits.UnitRegistry`, with the stack's `px`, `pd`, `dpi` and `pt`) and
its `uparse` — guarded, since `uparse` evaluates its parse. By default
`s` must be ONE registered unit symbol (`"mm"`, `"inch"`, `"px"`, `"dpi"`, …);
`compound = true` also admits a product, quotient or power of them (`"mm/px"`,
`"px/mm"`, `"mm^2"`), checked on the parsed expression before anything is
evaluated. Anything else is an `ArgumentError`, never a default.
"""
function parse_unit(s::AbstractString; compound::Bool = false)
    ok = haskey(UnitRegistry.UNITS, Symbol(s)) ||
        (compound && _unit_expr_ok(try Meta.parse(String(s); raise = true) catch; nothing end))
    ok || throw(
        ArgumentError(
            "unknown unit name $(repr(String(s))); a unit name is " *
                (compound ? "registered units (mm, inch, px, dpi, …) joined by *, / and integer ^" : "one registered unit (mm, cm, m, um, inch, px, dpi, …)")
        )
    )
    u = UnitRegistry.uparse(String(s))
    u isa Units || throw(ArgumentError("$(repr(String(s))) is not a unit"))
    return u
end
# A parsed unit expression of registered symbols, `*`, `/` and integer powers
# (nothing evaluated: `Meta.parse` only builds the expression).
_unit_expr_ok(ex::Symbol) = haskey(UnitRegistry.UNITS, ex)
_unit_expr_ok(ex::Expr) =
    ex.head === :call && length(ex.args) == 3 && (
    (ex.args[1] in (:*, :/) && _unit_expr_ok(ex.args[2]) && _unit_expr_ok(ex.args[3])) ||
        (ex.args[1] === :^ && _unit_expr_ok(ex.args[2]) && ex.args[3] isa Integer)
)
_unit_expr_ok(_) = false

"""
    unit_string(u::Units) -> String

The serialized name of a registered unit: FlexUnits' own printing, `string(u)`,
which for a registry unit is its symbol. A unit the registry has no symbol for
(`mm/px`, `1000mm`) is an `ArgumentError`: its printing is no name
[`parse_unit`](@ref) reads back. The inverse of `parse_unit`.
"""
function unit_string(u::Units)
    s = string(u)
    k = Symbol(s)
    (haskey(UnitRegistry.UNITS, k) && UnitRegistry.UNITS[k] == u) ||
        throw(ArgumentError("unit $s has no registered name (it is no single unit of FlexUnits' registry)"))
    return s
end

"""
    strip_or_bare(u::Units, x)

`x` as plain numbers in unit `u`: a quantity, or a struct of quantities with a
`ustrip(u, ·)` method (`Size2`, the feature frames), stripped in `u`; plain
numbers returned as they are. For the producers that emit either (the ASIFT
frames are plain `Float64`, the detectors' are `px`). A quantity of another
dimension is FlexUnits' `ConversionError`.
"""
strip_or_bare(::Units, x::Real) = x
strip_or_bare(u::Units, x::Quantity) = ustrip(u, x)

# -----------------------------------------------------------------------------
# Plain or unitful: the layering's one predicate
# -----------------------------------------------------------------------------
#
# Public values that carry physical lengths (poses, target points, rigs,
# captures) hold FlexUnits' base-form quantities, metres; the inner loop holds
# plain numbers. `has_units` tells the two forms apart — each carrier adds a
# method beside its type — and `ustrip(u, x)` / `with_unit(x, u)` move between
# them: `ustrip` gives the plain form in `u`, `with_unit` reads a plain value as
# being in `u` and stores it in metres. Each refuses a value already in the
# form it produces (`ArgumentError`), so a strip is never applied twice.

"""
    has_units(x) -> Bool

Whether `x` holds FlexUnits quantities — a physical value (a pose's
translation, a target's points, a rig's mounts, in metres) — rather than plain
numbers, the inner loop's form. Defined for numbers, arrays, and every carrier
of a physical length (`EuclideanMap`, `Uncertain` values, targets, mounts,
rigs, cameras, trajectories, target poses and 3D target points).
"""
has_units(::Real) = false
has_units(::Quantity) = true
has_units(::AbstractArray{<:Real}) = false
has_units(::AbstractArray{<:Quantity}) = true
has_units(x::AbstractArray) = any(has_units, x)

"""
    with_unit(x, u::Units)

The inverse of `ustrip(u, x)` on the stack's carriers: `x`'s plain lengths
read as being in the unit `u` and stored as FlexUnits base-form quantities
(metres) — an `Uncertain` pose's covariance rescaled with them. An
`ArgumentError` when `x` already has units ([`has_units`](@ref)). A plain
`EuclideanMap` takes its unit by construction instead,
`EuclideanMap(rotation(E), translation(E) .* u)`.
"""
function with_unit end

# A length unit's factor from metres: `ustrip(u, 1 m)` (1000 for mm). A unit of
# another dimension is FlexUnits' `ConversionError`.
_metre_factor(u::Units) = ustrip(u, Quantity(1.0, dimension(m)))

# One length vector as plain metres, refusing plain numbers where a physical
# value is meant (a unitful pose meeting a plain point): the strip at a public
# boundary. FlexUnits' `DimensionError`, as the arithmetic would raise.
_strip_metres(x::StaticVector{N, <:Quantity}) where {N} = map(q -> ustrip(m, q), x)
_strip_metres(x::AbstractVector{<:Quantity}) = ustrip.(m, x)
_strip_metres(x::AbstractVector) = throw(DimensionError(x, m))

# -----------------------------------------------------------------------------
# Pixel-sized values: the length check that survives a dimensionless px
# -----------------------------------------------------------------------------
#
# A pixel is dimensionless in FlexUnits, so `px` cannot be told from a radian
# or a ratio. What can still be refused is a LENGTH (or any other dimension)
# where a pixel count is meant: every public function that takes a pixel-sized
# value (a resolution, an image size, a pixel focal length or principal point)
# sends it through `_check_pixels` or `_strip_pixels`, which accept a plain
# number or a dimensionless quantity and raise FlexUnits' `DimensionError`
# otherwise.

# `x` itself when it is a plain number or a dimensionless quantity.
_check_pixels(x::Real) = x
function _check_pixels(q::Quantity)
    dimension(q) == dimension(px) || throw(DimensionError(q, px))
    return q
end
# `x` as a plain number of pixels.
_strip_pixels(x::Real) = x
_strip_pixels(q::Quantity) = ustrip(px, _check_pixels(q))
