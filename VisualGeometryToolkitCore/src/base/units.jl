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
