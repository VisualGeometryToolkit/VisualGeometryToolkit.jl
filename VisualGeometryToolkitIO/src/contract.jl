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

"""
    UnitContract(; world = mm, density = dpi, lengths = (;))
    UnitContract(world, density = dpi; lengths = (;))

Declares the units a file's magnitudes are expressed in: `world`, a length
unit, for translations, 3D points and every other length; `density`, a unit
of pixels per length (dimension 1/length, since a pixel is dimensionless), for
print and sensor densities; and `lengths`, further LENGTH ROLES, each a named
length unit (`lengths = (pattern = pt,)`), for a record whose lengths are
declared in more than one unit — a board's canvas in mm and its pattern in pt,
each written decimal-exactly in its own unit. A field read or written in a
role names it ([`read_quantity`](@ref)`(x, :length, c; role = :pattern)`);
`:world` is the default role and names `world`. Pixel counts need no
declaration: pixels are the only image unit. Persisted with the data it
governs — inline fields in vgio JSON records (`"units": {"world": "mm",
"density": "dpi", "lengths": {"pattern": "pt"}}`, written by
[`json_value`](@ref)`(contract)` and read by `from_json(UnitContract, d)`), a
`vgio_units.json` sidecar for COLMAP model directories (world and density
only) — so files stay self-describing. The roles are a CLOSED set, the ones
the contract declares: an `ArgumentError` for a `world` or a role that is not
a length, a role named `world` or `density`, a role name that is not
snake_case, a `density` that is not pixels per length, and (on use) a role
the contract does not declare.
"""
struct UnitContract{R <: NamedTuple}
    world::typeof(mm)     # a FlexUnits dynamic unit
    density::typeof(mm)   # likewise; dimension 1/length (px is dimensionless)
    lengths::R            # role => length unit beyond `world`, a closed set the record declares (empty for most)

    function UnitContract(world::Units, density::Units = dpi; lengths::NamedTuple = (;))
        dimension(world) == dimension(m) ||
            throw(ArgumentError("the world unit must be a length, got $(world)"))
        dimension(density) == dimension(dpi) ||
            throw(ArgumentError("the density unit must be pixels per length (like dpi), got $(density)"))
        for (r, u) in pairs(lengths)
            r === :world && throw(ArgumentError("`world` is the contract's own length role; name another"))
            r === :density && throw(ArgumentError("`density` is the contract's pixels-per-length unit, not a length role; name another"))
            name = String(r)
            (occursin(r"^[a-z][a-z0-9_]*$", name) && VgtkCore.snake_case(name) == name) ||
                throw(ArgumentError("a length role is named in snake_case (`pattern`, `ruler_tick`), got $(repr(name))"))
            (u isa typeof(mm) && dimension(u) == dimension(m)) ||
                throw(ArgumentError("the length role $(r) must be a length unit, got $(u)"))
        end
        ls = NamedTuple{keys(lengths)}(Tuple(typeof(mm)[u for u in values(lengths)]))
        return new{typeof(ls)}(world, density, ls)
    end
end

UnitContract(; world = mm, density = dpi, lengths::NamedTuple = (;)) = UnitContract(world, density; lengths)

"""
    role_unit(contract::UnitContract, role::Symbol = :world) -> Units

The unit `contract` declares for the lengths of `role`: `world` for `:world`,
the named role's otherwise; an `ArgumentError` for a role it does not declare.
"""
function role_unit(c::UnitContract, role::Symbol = :world)
    role === :world && return c.world
    haskey(c.lengths, role) || throw(
        ArgumentError("the units declare no length role $(repr(role)) (world$(join((", $(k)" for k in keys(c.lengths)))))")
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
# reading `contract.world`.
_world_unit(c::UnitContract) = c.world
_world_unit(::UnknownScale) = throw(
    ArgumentError(
        "the world scale is unknown (UnknownScale: a structure-from-motion gauge) — " *
            "its translations are plain numbers with no length unit, so no metric pose or " *
            "length can be formed from them; read the unitless wire values (ColmapImage.qvec, " *
            ".tvec) instead, or state the model's real unit with a UnitContract if it has one"
    )
)

# Sidecar for formats we can't extend in-band (COLMAP model directories).
const UNITS_SIDECAR = "vgio_units.json"

# A sidecar's `world` is a unit name, or this word for an UnknownScale model —
# not a unit, so `parse_unit` (and every older reader) refuses it loudly.
const _UNKNOWN_SCALE_NAME = "unknown"
_world_name(c::UnitContract) = unit_string(c.world)
_world_name(::UnknownScale) = _UNKNOWN_SCALE_NAME

# {"schema": "vgio/units/v2", "world": "mm" | "m" | … | "unknown"}
# The one declaration a COLMAP model directory needs beyond COLMAP's own
# format, which fixes everything else (pixel centres at 0.5, world → camera
# poses, Hamilton w-x-y-z quaternions, the per-model parameter order).
# A `density` is optional: absent, it is dpi, and a contract whose density is
# dpi writes none, so a sidecar holds what it held before densities existed.
const _UNITS_SPEC = _Obj(
    Pair{Symbol, _Spec}[
        :schema => _schema_tok("units"),
        :world => _Named(s -> s == _UNKNOWN_SCALE_NAME || _is_length_name(s), "a registered length unit, or \"unknown\""),
    ],
    Pair{Symbol, _Spec}[:density => _DENSITY_UNIT],
)

_sidecar_body(c::UnitContract) = c.density == dpi ? (schema = _schema("units"), world = _world_name(c)) :
    (schema = _schema("units"), world = _world_name(c), density = unit_string(c.density))
_sidecar_body(c::UnknownScale) = (schema = _schema("units"), world = _world_name(c))

function save_units_sidecar(dir::AbstractString, contract::Union{UnitContract, UnknownScale})
    (contract isa UnitContract && !isempty(contract.lengths)) && throw(
        ArgumentError("a COLMAP model's units sidecar declares its world and density only; this contract has length roles $(keys(contract.lengths))")
    )
    return _write_store(joinpath(dir, UNITS_SIDECAR), _UNITS_SPEC, _sidecar_body(contract))
end

"""
    load_units_sidecar(dir) -> Union{UnitContract, UnknownScale, Nothing}

Read the `vgio_units.json` sidecar of a COLMAP model directory; `nothing` if
absent (a model produced by other tools — the caller must then declare units
explicitly). `"world": "unknown"` is an [`UnknownScale`](@ref) model. A v1
sidecar is refused by name: nothing reads a retired version. A sidecar
without a `density` has dpi.
"""
function load_units_sidecar(dir::AbstractString)
    path = joinpath(dir, UNITS_SIDECAR)
    isfile(path) || return nothing
    d = _read_store(path, "units", _UNITS_SPEC)
    world = d.world::String
    world == _UNKNOWN_SCALE_NAME && return UnknownScale()
    density = haskey(d, :density) ? parse_unit(d.density::String) : dpi
    return UnitContract(parse_unit(world), density)
end
