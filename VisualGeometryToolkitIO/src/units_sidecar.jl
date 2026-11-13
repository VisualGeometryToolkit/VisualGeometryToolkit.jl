# =============================================================================
# vgio/units — the units sidecar of a COLMAP model directory
# =============================================================================

# Sidecar for formats we can't extend in-band (COLMAP model directories).
const UNITS_SIDECAR = "vgio_units.json"

# A sidecar's `world` is a unit name, or this word for an UnknownScale model —
# not a unit, so `parse_unit` (and every older reader) refuses it loudly.
const _UNKNOWN_SCALE_NAME = "unknown"
_world_name(c::UnitContract) = unit_string(role_unit(c))
_world_name(::UnknownScale) = _UNKNOWN_SCALE_NAME

# {"schema": "vgio/units/v3.0", <envelope>}: the sidecar IS the envelope — its
# `units` object declares the model's length unit (`length`, or "unknown" for an
# UnknownScale model) and density; COLMAP's own format fixes everything else
# (pixel centres at 0.5, world → camera poses, Hamilton w-x-y-z quaternions, the
# per-model parameter order), so `units.pixel` is "colmap". A model's sidecar
# declares no length roles.
const _UNITS_SPEC = _kind_spec("units", Pair{Symbol, _Spec}[];
    units = _units_spec(_Named(s -> s == _UNKNOWN_SCALE_NAME || _is_length_name(s), "a registered length unit, or \"unknown\"")),
    check = x -> isempty(x[:units][:length_roles]) ? (x[:units][:pixel] == "colmap" ? nothing : "a COLMAP model's pixels are colmap's") :
        "a COLMAP model's units sidecar declares no length roles")

function save_units_sidecar(dir::AbstractString, contract::Union{UnitContract, UnknownScale})
    (contract isa UnitContract && length(contract.lengths) > 1) && throw(
        ArgumentError("a COLMAP model's units sidecar declares its world and density only; this contract has length roles $(keys(contract.lengths))")
    )
    style = contract isa UnitContract ? VgioStyle(contract) : VgioStyle()
    path = joinpath(dir, UNITS_SIDECAR)
    units = merge(_units_json(style), (length = _world_name(contract),))
    record = (schema = _schema("units"), generator = _generator_json(), units = units, conventions = _CONVENTIONS_JSON)
    text = _json_text(record, style)
    _validate(_UNITS_SPEC, _parse_vgio_text(text, path), path, "")
    write(path, text)
    return nothing
end

"""
    load_units_sidecar(dir) -> Union{UnitContract, UnknownScale, Nothing}

Read the `vgio_units.json` sidecar of a COLMAP model directory; `nothing` if
absent (a model produced by other tools — the caller must then declare units
explicitly). `"length": "unknown"` is an [`UnknownScale`](@ref) model. A v2
sidecar (`world`, an optional `density`: dpi when absent, as v2 read it) is read
through its version step; a v1 one is refused by name.
"""
function load_units_sidecar(dir::AbstractString)
    path = joinpath(dir, UNITS_SIDECAR)
    isfile(path) || return nothing
    d = _read_store(path, "units")
    world = d.units.length::String
    world == _UNKNOWN_SCALE_NAME && return UnknownScale()
    return UnitContract(parse_unit(world), parse_unit(d.units.density::String))
end
