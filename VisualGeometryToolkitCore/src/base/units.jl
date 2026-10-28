# =============================================================================
# Units — FlexUnits, with the stack's own registry of exact scales
# =============================================================================
#
# No stack quantity type or dimension set: values are FlexUnits' standard
# base-form quantities, `Quantity{T, Dimensions{FixRat32}}` (a bits type).
# Unit NAMES resolve through the stack's own registry, `StackUnits`, built
# with FlexUnits' `RegistryTools` exactly as FlexUnits documents for a
# registry of another transform type ("Extending FlexUnits", the
# `RationalRegistry` recipe): a `PermanentDict` of
# `Units{Dimensions{FixRat32}, AffineTransform{Rational{Int}}}`, filled by
# `register_unit!` and exposed by `@generate_registry_exports` (`uparse`,
# `qparse`, `@u_str`, `@ud_str`).
#
# Why a registry of our own: FlexUnits' default registry holds Float64 scales,
# so `142mm` was `142 × 0.001` = 0.14200000000000002 m, one ulp off the
# decimal 0.142. With an exact scale, `142mm` is the rational 71//500 m and
# its Float64 is the decimal-nearest 0.142. The scales below are written as
# exact rationals and handed to `register_unit!` as data; `registry_defaults!`
# is NOT called, because in a Rational registry it converts its Float64
# scales (`1e-3`) to their binary values (1152921504606847//1152921504606846976).
#
# What stays inexact, on FlexUnits' side: a FLOAT times a rational scale is
# promoted to Float64 (`142.0mm` is `142.0 * 0.001`, within 1 ulp of the
# decimal), and so is the inverse (`ustrip(mm, q)` divides by `0.001`). The
# stack accepts that (VisualGeometryToolkitIO's file lengths keep their
# decimal-exact pair for it) until FlexUnits applies a rational scale to a
# float as `x * num / den`.
#
# A pixel is DIMENSIONLESS (FlexUnits' seven SI dimensions only, so a pixel is
# indistinguishable from a radian; pixel coordinates keep their safety through
# `PixelFrame` tags). Values are stored in base units: metres for a length. A
# bare `ustrip(q)` of a length is in metres; code that wants millimetres names
# the unit, `ustrip(mm, q)`. A quantity built from an integer (`10mm`) has an
# exact Rational value; a struct stores the Float64 base form, through
# [`base_quantity`](@ref). See conventions/units.md.

"""
    StackUnits

The stack's unit registry: FlexUnits' `RegistryTools` over
`Units{Dimensions{FixRat32}, AffineTransform{Rational{Int}}}`, every scale an
exact rational in base units. It holds the units the stack reads and writes
and nothing else:

| unit | scale (base units) | |
|---|---|---|
| `m`, `kg`, `s`, `A`, `K`, `cd`, `mol` | 1 | the SI base units |
| `km`, `cm`, `mm`, `μm` (`um`) | 1000, 1//100, 1//1000, 1//1000000 | m |
| `inch` (`in`), `ft`, `pt` | 127//5000, 381//1250, 127//360000 | m (the international inch; `pt = inch/72`) |
| `px`, `pd` | 1, dimensionless | a pixel count; `pd` the board JSONs' print dot |
| `dpi` | 5000//127 | `px/inch`, of dimension 1/m |

Its generated `uparse`, `qparse` and `@ud_str` are VisualGeometryToolkitCore's;
a name from outside is read with [`parse_unit`](@ref), never `uparse`.
"""
module StackUnits
using FlexUnits.RegistryTools: RegistryTools, PermanentDict, register_unit!, @generate_registry_exports,
    Units, Dimensions, AffineTransform, FixRat32

const UNITS = PermanentDict{Symbol, Units{Dimensions{FixRat32}, AffineTransform{Rational{Int}}}}()

# The SI base units (scale 1), then the stack's units as exact ratios of them.
for (name, dims) in (
        "m" => Dimensions(m = 1), "kg" => Dimensions(kg = 1), "s" => Dimensions(s = 1),
        "A" => Dimensions(A = 1), "K" => Dimensions(K = 1), "cd" => Dimensions(cd = 1),
        "mol" => Dimensions(mol = 1),
    )
    register_unit!(UNITS, name => dims)
end
register_unit!(UNITS, "km" => 1000 * UNITS[:m])
register_unit!(UNITS, "cm" => (1 // 100) * UNITS[:m])
register_unit!(UNITS, "mm" => (1 // 1000) * UNITS[:m])
register_unit!(UNITS, "μm" => (1 // 1000000) * UNITS[:m])
register_unit!(UNITS, "um" => UNITS[:μm])
register_unit!(UNITS, "inch" => (127 // 5000) * UNITS[:m])   # the international inch, 25.4 mm
register_unit!(UNITS, "in" => UNITS[:inch])
register_unit!(UNITS, "ft" => 12 * UNITS[:inch])
register_unit!(UNITS, "pt" => (1 // 72) * UNITS[:inch])      # the typographic point
register_unit!(UNITS, "px" => Dimensions())                  # a pixel count, dimensionless
register_unit!(UNITS, "pd" => UNITS[:px])                    # the board JSONs' print dot, equal to px
register_unit!(UNITS, "dpi" => UNITS[:px] / UNITS[:inch])

@generate_registry_exports(UNITS)
end

# The unit constants: the registry's dynamic units, fixed when the package is
# precompiled.
const mm = StackUnits.UNITS[:mm]
const cm = StackUnits.UNITS[:cm]
const μm = StackUnits.UNITS[:μm]
const m = StackUnits.UNITS[:m]
const inch = StackUnits.UNITS[:inch]
const ft = StackUnits.UNITS[:ft]
const px = StackUnits.UNITS[:px]
const pt = StackUnits.UNITS[:pt]
const dpi = StackUnits.UNITS[:dpi]

"""
    BaseQuantity

FlexUnits' concrete stored form of a base quantity, computed once as
`typeof(ubase(1.0mm))` (today `Quantity{Float64, Dimensions{FixRat32}}`): the
type a struct field holds, so a container of mixed dimensions is concrete. It
is NOT a stack quantity type — public signatures take any FlexUnits
`Quantity` and convert with [`base_quantity`](@ref) — and it is named here
only so the stack spells FlexUnits' stored form once instead of restating its
parameters (see conventions/units.md).
"""
const BaseQuantity = typeof(ubase(1.0mm))

"""
    base_quantity(q::Quantity) -> Quantity

`q` in FlexUnits' base form with a floating-point value: `float(ubase(q))`.
The ONE conversion a constructor applies to a quantity it stores. A quantity
built from an integer (`142mm`, `3px`) holds an exact `Rational`
(71//500 m); its stored form is the correctly rounded Float64, 0.142 m, the
decimal-nearest value. A float value keeps its type (`Float64`, `Float32`, a
ForwardDiff `Dual`), so `base_quantity(1.0mm)` is `ubase(1.0mm)`. A
`Rational` never reaches a struct field or a kernel.
"""
base_quantity(q::Quantity) = float(ubase(q))

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

The unit a serialized name `s` spells, resolved by the stack's registry
([`StackUnits`](@ref VisualGeometryToolkitCore.StackUnits)) and its generated `uparse` — guarded, since `uparse` evaluates its parse. By default
`s` must be ONE registered unit symbol (`"mm"`, `"inch"`, `"px"`, `"dpi"`, …);
`compound = true` also admits a product, quotient or power of them (`"mm/px"`,
`"px/mm"`, `"mm^2"`), checked on the parsed expression before anything is
evaluated. Anything else is an `ArgumentError`, never a default.
"""
function parse_unit(s::AbstractString; compound::Bool = false)
    ok = haskey(StackUnits.UNITS, Symbol(s)) ||
        (compound && _unit_expr_ok(try Meta.parse(String(s); raise = true) catch; nothing end))
    ok || throw(
        ArgumentError(
            "unknown unit name $(repr(String(s))); a unit name is " *
                (compound ? "registered units (mm, inch, px, dpi, …) joined by *, / and integer ^" : "one registered unit (mm, cm, m, um, inch, px, dpi, …)")
        )
    )
    u = StackUnits.uparse(String(s))
    u isa Units || throw(ArgumentError("$(repr(String(s))) is not a unit"))
    return u
end
# A parsed unit expression of registered symbols, `*`, `/` and integer powers
# (nothing evaluated: `Meta.parse` only builds the expression).
_unit_expr_ok(ex::Symbol) = haskey(StackUnits.UNITS, ex)
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
    (haskey(StackUnits.UNITS, k) && StackUnits.UNITS[k] == u) ||
        throw(ArgumentError("unit $s has no registered name (it is no single unit of the stack's registry)"))
    return s
end

"""
    strip_or_bare(u::Units, x)

`x` as plain numbers in unit `u`: a quantity, or a struct of quantities with a
`ustrip(u, ·)` method (`Size2`, the feature frames), stripped in `u`; plain
numbers returned as they are. A stripped quantity is a float (`float` of an
exact `Rational` from `3px`). For the producers that emit either (the ASIFT
frames are plain `Float64`, the detectors' are `px`). A quantity of another
dimension is FlexUnits' `ConversionError`.
"""
strip_or_bare(::Units, x::Real) = x
strip_or_bare(u::Units, x::Quantity) = float(ustrip(u, x))

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
`EuclideanMap` takes its unit this way too, its rotation kept exactly
(construction, `EuclideanMap(rotation(E), translation(E) .* u)`, re-stores a
non-`RotMatrix` rotation as one).
"""
function with_unit end

# A length unit's factor from metres: `ustrip(u, 1 m)` (1000 for mm). A unit of
# another dimension is FlexUnits' `ConversionError`.
_metre_factor(u::Units) = ustrip(u, Quantity(1.0, dimension(m)))   # Float64: a float value over the rational scale

"""
    strip_metres(x) -> plain numbers

A length, or a vector of lengths, as plain metres: the one strip at a public
boundary that takes physical values and computes on plain numbers (a pose's
translation, a 3D point, a threshold given beside physical data). It refuses
plain numbers where a length is meant — a unit the caller did not state — with
FlexUnits' `DimensionError`, as the arithmetic would raise; a quantity of
another dimension is FlexUnits' `ConversionError`. A static vector stays
static. Values are stored in metres, so the strip is exact; a quantity built
from an integer (`2m`, exactly rational) comes out as its correctly rounded
float.
"""
strip_metres(q::Quantity) = float(ustrip(m, q))
strip_metres(x::Real) = throw(DimensionError(x, m))
strip_metres(x::StaticVector{N, <:Quantity}) where {N} = map(strip_metres, x)
strip_metres(x::AbstractVector{<:Quantity}) = strip_metres.(x)
strip_metres(x::AbstractVector) = throw(DimensionError(x, m))

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
_strip_pixels(q::Quantity) = float(ustrip(px, _check_pixels(q)))
