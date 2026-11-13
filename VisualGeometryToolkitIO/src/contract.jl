# =============================================================================
# Unit contract — the checked boundary between file magnitudes and quantities
# =============================================================================
#
# A file holds a length as a number in its declared unit; memory holds it as
# FlexUnits' base-form quantity, metres. Every length crosses through the one
# pair `read_length(x, u)` / `write_length(q, u)` (lengths.jl: decimal-exact
# for a power-of-ten unit, dimension-checked), never by multiplying by a
# scale here; a writer refuses plain numbers where a length is meant. Unit
# names are VisualGeometryToolkitCore's `parse_unit` (the stack's one guarded
# `uparse`) and `unit_string` (FlexUnits' printing of a registered unit).

# The roles of a table beyond `world`, in order.
_other_roles(lengths::NamedTuple) = NamedTuple{filter(!=(:world), keys(lengths))}(lengths)

"""
    UnitContract(world, density = dpi; lengths = (;))
    UnitContract(; world = mm, density = dpi, lengths = (;))
    UnitContract(density, lengths::NamedTuple)

`world`, `density` and every role's unit are FlexUnits dynamic units (`mm`,
`pt`, `dpi`, or `parse_unit` of a name).

Declares the units a file's magnitudes are expressed in: `lengths`, a table of
LENGTH ROLES, each a named length unit; and `density`, a unit of pixels per
length (dimension 1/length, since a pixel is dimensionless), for print and
sensor densities. The role `world` is REQUIRED and is the default role: the
unit of translations, 3D points and every other length. Further roles
(`pattern = pt`) serve a record whose lengths are declared in more than one
unit — a board's canvas in mm and its pattern in pt, each written
decimal-exactly in its own unit. A field read or written in a role names it
([`read_quantity`](@ref)`(x, :length, c; role = :pattern)`). Pixel counts need
no declaration: pixels are the only image unit.

In memory the contract is that table, `world` first:
`UnitContract(mm, dpi; lengths = (pattern = pt,)).lengths == (world = mm,
pattern = pt)`. The first two forms are conveniences that merge `world` into
the table (the keyword form also reads a `world` already in `lengths`; naming
it twice is an error); the third is the canonical one and refuses a table
without `world`.

The FILE layout is separate and unchanged: the serialiser alone knows it.
Inline fields in vgio JSON records hold `"units": {"world": "mm", "density":
"dpi", "lengths": {"pattern": "pt"}}` — `world` top level, the other roles
under `lengths` — written by [`json_value`](@ref)`(contract)` and read by
`from_json(UnitContract, d)`; a `vgio_units.json` sidecar for COLMAP model
directories holds world and density only. Files stay self-describing. The
roles are a CLOSED set, the ones the contract declares: an `ArgumentError` for
a missing `world`, a role that is not a length, a role named `density`, a role
name that is not snake_case, a `density` that is not pixels per length, and
(on use) a role the contract does not declare.
"""
struct UnitContract{R <: NamedTuple}
    density::typeof(mm)   # a FlexUnits dynamic unit; dimension 1/length (px is dimensionless)
    lengths::R            # role => length unit, a closed set the record declares; `world` is required and first

    function UnitContract(density::Units, lengths::NamedTuple)
        haskey(lengths, :world) ||
            throw(ArgumentError("the units declare no `world` length role (the default role: translations, 3D points); roles given: $(keys(lengths))"))
        dimension(density) == dimension(dpi) ||
            throw(ArgumentError("the density unit must be pixels per length (like dpi), got $(density)"))
        for (r, u) in pairs(lengths)
            r === :density && throw(ArgumentError("`density` is the contract's pixels-per-length unit, not a length role; name another"))
            name = String(r)
            (occursin(r"^[a-z][a-z0-9_]*$", name) && VgtkCore.snake_case(name) == name) ||
                throw(ArgumentError("a length role is named in snake_case (`world`, `pattern`, `ruler_tick`), got $(repr(name))"))
            (u isa typeof(mm) && dimension(u) == dimension(m)) ||
                throw(ArgumentError("the length role $(r) must be a length unit, got $(u)"))
        end
        rest = _other_roles(lengths)   # `world` first, whatever the caller's order
        ls = merge((world = lengths.world,), rest)
        ls = NamedTuple{keys(ls)}(Tuple(typeof(mm)[u for u in values(ls)]))
        return new{typeof(ls)}(density, ls)
    end
end

function UnitContract(world::Units, density::Units = dpi; lengths::NamedTuple = (;))
    haskey(lengths, :world) && throw(ArgumentError("`world` is named twice: as the first argument and in `lengths`"))
    return UnitContract(density, merge((world = world,), lengths))
end

function UnitContract(; world = nothing, density = dpi, lengths::NamedTuple = (;))
    if world === nothing
        haskey(lengths, :world) || (world = mm)   # the stack's default world, as MM_PX
    elseif haskey(lengths, :world)
        throw(ArgumentError("`world` is named twice: as a keyword and in `lengths`"))
    end
    return world === nothing ? UnitContract(density, lengths) : UnitContract(world, density; lengths)
end

"""
    role_unit(contract::UnitContract, role::Symbol = :world) -> Units

The unit `contract` declares for the lengths of `role`: `contract.lengths[role]`;
an `ArgumentError` listing the declared roles for a role it does not declare.
Every read of a contract's length unit goes through it (the unit of
`:world`, the default, included).
"""
function role_unit(c::UnitContract, role::Symbol = :world)
    haskey(c.lengths, role) || throw(
        ArgumentError("the units declare no length role $(repr(role)); declared: $(join(keys(c.lengths), ", "))")
    )
    return c.lengths[role]
end

"""The stack-canonical contract: world in millimeters (image values are pixels)."""
const MM_PX = UnitContract()

"""
    UnknownScale()

The contract of a model whose world has NO unit: a structure-from-motion
reconstruction without metric control (PhotoTourism, any plain COLMAP run) is
fixed only up to a similarity, so its translations and 3D points are numbers
in the reconstruction's own arbitrary scale. Declaring `mm` or `m` for them
would be false, and nothing declares this either unless someone says so: the
caller states it (`load(ColmapModel, dir; contract = UnknownScale())`), or
the model's `vgio_units.json` sidecar does (`"world": "unknown"`, written by
`save` for such a model). It is never a default.

Under it the wire values stay what they are, plain numbers
(`ColmapImage.tvec`, `ColmapPoint3D.xyz`), and anything that needs a length
unit refuses: `rigid_transform(image, contract)` (a metric `EuclideanMap`)
and `colmap_image(pose, …; contract)` (a metric pose written into the model)
throw an `ArgumentError`. What does not depend on scale is read from the wire
as usual: the rotation `qvec`, and translation directions.
"""
struct UnknownScale end

# The world unit a metric conversion needs. It is the ONE place an unknown
# scale refuses, so every metric use of a contract goes through it rather than
# reading the contract's `world` role directly.
_world_unit(c::UnitContract) = role_unit(c)
_world_unit(::UnknownScale) = throw(
    ArgumentError(
        "the world scale is unknown (UnknownScale: a structure-from-motion gauge) — " *
            "its translations are plain numbers with no length unit, so no metric pose or " *
            "length can be formed from them; read the unitless wire values (ColmapImage.qvec, " *
            ".tvec) instead, or state the model's real unit with a UnitContract if it has one"
    )
)

