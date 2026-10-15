# =============================================================================
# Units — FlexUnits with a pixel dimension
# =============================================================================
#
# The stack's whole unit system is FlexUnits' documented custom-dimension
# recipe: one dimension struct (the SI base dimensions plus `px`) and one
# registry module built with `FlexUnits.RegistryTools`. Parsing (`u"…"`,
# `ud"…"`, `uparse`, `qparse`), printing, conversion and dimension checks are
# FlexUnits'. Nothing here parses, prints or compares dimensions by hand.
#
# Values are stored in base units: metres for a length, pixels for a pixel
# count. A bare `ustrip(q)` is therefore in metres for a length; code that wants
# millimetres names the unit, `ustrip(mm, q)`. See conventions/units.md.

"""
    PixelDimensions{P} <: FlexUnits.AbstractDimensions{P}

The stack's dimension set: the seven SI base dimensions plus a logical `px`.
FlexUnits registers a unit per field, so `px` is a unit of its own dimension;
a density is `px/m` (`dpi`), a pitch `m/px`, and `px + mm` is a dimension
error.
"""
@kwdef struct PixelDimensions{P} <: AbstractDimensions{P}
    m::P = zero(FixRat32)
    kg::P = zero(FixRat32)
    s::P = zero(FixRat32)
    A::P = zero(FixRat32)
    K::P = zero(FixRat32)
    cd::P = zero(FixRat32)
    mol::P = zero(FixRat32)
    px::P = zero(FixRat32)
end
PixelDimensions(args::Real...) = PixelDimensions{FixRat32}(args...)

"""
    PixelUnits

The unit registry over [`PixelDimensions`](@ref): FlexUnits' defaults, plus
`dpi = px/inch`, the typographic `pt = inch/72`, and `pd` (print dot), the
board JSONs' unit name for a pixel, equal to `px`. It provides `@u_str`
(static units), `@ud_str` (dynamic units), `uparse` and `qparse`.
"""
module PixelUnits
    using FlexUnits: FlexUnits, RegistryTools, AffineTransform, FixRat32, Units, dimvaltype
    using FlexUnits.RegistryTools: PermanentDict, register_unit!, registry_defaults!,
        @generate_registry_exports, @generate_unit_simplifier
    import ..PixelDimensions
    public UNITS
    const UNITS = PermanentDict{Symbol, Units{PixelDimensions{FixRat32}, AffineTransform{Float64}}}()
    registry_defaults!(UNITS)
    register_unit!(UNITS, "dpi" => UNITS[:px] / UNITS[:inch])
    register_unit!(UNITS, "pt" => (1 / 72) * UNITS[:inch])
    register_unit!(UNITS, "pd" => UNITS[:px])   # pd: the board JSONs' unit name for a pixel (print dot), equal to px
    const PREFERRED_UNITS = [UNITS[u] for u in [:F, :H, :T, :Ω, :V, :W, :J, :Pa, :N, :C, :L]]
    @generate_unit_simplifier(PREFERRED_UNITS)
    @generate_registry_exports(UNITS)
end
using .PixelUnits

"""
    VgcQuantity{T}

The type of a dynamic quantity over [`PixelDimensions`](@ref): the type of
`10mm`, `3px`, `300dpi` and every `ud"…"` quantity. Dimensions are runtime
values, so quantities of any dimension share this one concrete type, and a
struct holding quantities needs no unit type parameter.
"""
const VgcQuantity{T} = Quantity{T, PixelDimensions{FixRat32}}

# The unit constants (dynamic units, so `10mm isa VgcQuantity{Float64}`).
const px = ud"px"
const mm = ud"mm"
const cm = ud"cm"
const μm = ud"μm"
const m = ud"m"
const inch = ud"inch"
const ft = ud"ft"
const pt = ud"pt"
const dpi = ud"dpi"

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

The unit a serialized name `s` spells, resolved by the [`PixelUnits`](@ref)
registry's `uparse` — guarded, since `uparse` evaluates its parse. By default
`s` must be ONE registered unit symbol (`"mm"`, `"inch"`, `"px"`, `"dpi"`, …);
`compound = true` also admits a product, quotient or power of them (`"mm/px"`,
`"px/mm"`, `"mm^2"`), checked on the parsed expression before anything is
evaluated. Anything else is an `ArgumentError`, never a default.
"""
function parse_unit(s::AbstractString; compound::Bool = false)
    ok = haskey(PixelUnits.UNITS, Symbol(s)) ||
        (compound && _unit_expr_ok(try Meta.parse(String(s); raise = true) catch; nothing end))
    ok || throw(
        ArgumentError(
            "unknown unit name $(repr(String(s))); a unit name is " *
                (compound ? "registered units (mm, inch, px, dpi, …) joined by *, / and integer ^" : "one registered unit (mm, cm, m, um, inch, px, dpi, …)")
        )
    )
    u = uparse(String(s))
    u isa Units || throw(ArgumentError("$(repr(String(s))) is not a unit"))
    return u
end
# A parsed unit expression of registered symbols, `*`, `/` and integer powers
# (nothing evaluated: `Meta.parse` only builds the expression).
_unit_expr_ok(ex::Symbol) = haskey(PixelUnits.UNITS, ex)
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
    (haskey(PixelUnits.UNITS, k) && PixelUnits.UNITS[k] == u) ||
        throw(ArgumentError("unit $s has no registered name (it is no single unit of the PixelUnits registry)"))
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
