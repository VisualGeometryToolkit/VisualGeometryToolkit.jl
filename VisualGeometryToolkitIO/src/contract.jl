# =============================================================================
# Unit contract — the checked boundary between file magnitudes and quantities
# =============================================================================
#
# Dimensioned numbers cross between disk and memory through FlexUnits itself:
# a read attaches the file's unit with `quantity(x, u)`, a write strips in the
# file's unit with `ustrip(u, q)`, both dimension-checked at runtime (a px
# quantity into a mm field is a `ConversionError`; a bare number has no
# `ustrip(u, ·)` method). `quantity` keeps the file unit on the value, so a
# magnitude read into the same unit comes back exactly (FlexUnits composes the
# two unit transforms instead of storing the value in metres; `x * u` does
# not). A single-argument `ustrip` (base units) is never used here. Unit names
# are VisualGeometryToolkitCore's `parse_unit` (the stack's one guarded `uparse`) and
# `unit_string` (FlexUnits' printing of a registered unit).

"""
    UnitContract(; world = mm)
    UnitContract(world)

Declares the unit a file's world magnitudes — translations and 3D points —
are expressed in (a length unit). Image magnitudes need no declaration:
pixels are the only image unit. Persisted with the data it governs — inline
fields in vgio JSON records, a `vgio_units.json` sidecar for COLMAP model
directories — so files stay self-describing.
"""
struct UnitContract
    world::typeof(mm)   # a dynamic unit of the PixelUnits registry

    function UnitContract(world::Units)
        dimension(world) == dimension(m) ||
            throw(ArgumentError("the world unit must be a length, got $(world)"))
        return new(world)
    end
end

UnitContract(; world = mm) = UnitContract(world)

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
const _UNITS_SPEC = _Obj(
    :schema => _schema_tok("units"),
    :world => _Named(s -> s == _UNKNOWN_SCALE_NAME || _is_length_name(s), "a registered length unit, or \"unknown\""),
)

save_units_sidecar(dir::AbstractString, contract::Union{UnitContract, UnknownScale}) =
    _write_store(joinpath(dir, UNITS_SIDECAR), _UNITS_SPEC, (schema = _schema("units"), world = _world_name(contract)))

"""
    load_units_sidecar(dir) -> Union{UnitContract, UnknownScale, Nothing}

Read the `vgio_units.json` sidecar of a COLMAP model directory; `nothing` if
absent (a model produced by other tools — the caller must then declare units
explicitly). `"world": "unknown"` is an [`UnknownScale`](@ref) model. A v1
sidecar is refused by name: nothing reads a retired version.
"""
function load_units_sidecar(dir::AbstractString)
    path = joinpath(dir, UNITS_SIDECAR)
    isfile(path) || return nothing
    world = _read_store(path, "units", _UNITS_SPEC).world::String
    world == _UNKNOWN_SCALE_NAME && return UnknownScale()
    return UnitContract(world = parse_unit(world))
end
