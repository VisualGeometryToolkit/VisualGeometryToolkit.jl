# =============================================================================
# vgio/generic_camera/v2 — the Schöps et al. grid models, serialized
# =============================================================================
#
# One model per file, field names mirroring the reference implementation's
# YAML tags (puzzlepaint/camera_calibration) so the two formats translate
# one-to-one:
#
# {"schema": "vgio/generic_camera/v2",
#  "pixel_frame": "colmap",                        VGC's grid models ARE in it
#  "length_unit": "mm",                            non-central only (point_grid)
#  "type": "CentralGenericModel" | "NoncentralGenericModel",
#  "width": …, "height": …,
#  "calibration_min_x": …, "calibration_min_y": …,
#  "calibration_max_x": …, "calibration_max_y": …,
#  "grid_width": …, "grid_height": …, "grid_layout": "row_major",
#  "direction_grid": [[x, y, z], …],               entry y·grid_width + x
#  "point_grid":     [[x, y, z], …],               non-central only
#  "provenance": {…}}                               (optional)
#
# The pixel frame is COLMAP's because VGC's generic models are defined in it —
# the reference's convention (origin at the top-left corner of the top-left
# pixel) is exactly that frame — so the declaration can only be "colmap", and a
# reader refuses any other rather than shifting a grid it cannot shift. The
# grids are the model's ~10,000 parameters (~200 KB of JSON at the paper's grid
# size), still configuration-adjacent and diffable. A rig whose sensor carries
# a generic model embeds this BODY (no schema or declarations — the rig
# declares them) where a parametric sensor embeds a camera body; it carries its
# own width/height, so a bare `Camera` sensor suffices.

const _GRID = _Vec(_XYZ)
const _GENERIC_TYPES = ("CentralGenericModel", "NoncentralGenericModel")
const _GENERIC_BODY_FIELDS = Pair{Symbol, _Spec}[
    :type => _Named(in(_GENERIC_TYPES), "a generic model type ($(join(_GENERIC_TYPES, ", ")))"),
    :width => _PosInt(), :height => _PosInt(),
    :calibration_min_x => _Int(), :calibration_min_y => _Int(),
    :calibration_max_x => _Int(), :calibration_max_y => _Int(),
    :grid_width => _PosInt(), :grid_height => _PosInt(),
    :grid_layout => _Tok(_GRID_LAYOUT),
    :direction_grid => _GRID,
]

# The body's own check, shared by the standalone file and a rig sensor: a
# point grid exactly when the model is non-central, and grids of the declared
# size.
function _generic_check(x)
    noncentral = x[:type] == "NoncentralGenericModel"
    noncentral == haskey(x, :point_grid) ||
        return noncentral ? "a NoncentralGenericModel needs its point_grid" : "a CentralGenericModel has no point_grid"
    n = x[:grid_width] * x[:grid_height]
    length(x[:direction_grid]) == n ||
        return "direction_grid has $(length(x[:direction_grid])) entries, expected grid_width × grid_height = $(n)"
    (noncentral && length(x[:point_grid]) != n) &&
        return "point_grid has $(length(x[:point_grid])) entries, expected grid_width × grid_height = $(n)"
    return nothing
end

const _GENERIC_BODY = _Checked(_Obj(_GENERIC_BODY_FIELDS, Pair{Symbol, _Spec}[:point_grid => _GRID]), _generic_check)
const _GENERIC_CAMERA_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("generic_camera"),
            :pixel_frame => _Tok(pixel_frame_name(CANONICAL_PIXEL_FRAME)),
            _GENERIC_BODY_FIELDS...,
        ],
        Pair{Symbol, _Spec}[:length_unit => _LENGTH_UNIT, :point_grid => _GRID, _PROVENANCE...],
    ),
    x -> something(
        _generic_check(x),
        (x[:type] == "NoncentralGenericModel") == haskey(x, :length_unit) ? nothing :
            "a non-central model declares the length_unit of its point_grid, and only it does",
        Some(nothing),
    ),
)

_grid_json(g::Matrix{<:Vec3}) =
    [[Float64(v[1]), Float64(v[2]), Float64(v[3])] for gy in 1:size(g, 1) for v in (g[gy, :]...,)]

# A grid back from its row-major entries, each scaled by `scale` (millimetres
# per file unit; 1 for a direction grid).
function _grid_from_json(a, gw::Int, gh::Int; unit::Units = mm, is_points::Bool = false)
    g = Matrix{Vec3{Float64}}(undef, gh, gw)
    for (i, v) in enumerate(a)
        gy, gx = divrem(i - 1, gw)
        xyz = Float64[Float64(c::Real) for c in v::Vector{Any}]
        g[gy + 1, gx + 1] = Vec3{Float64}((is_points ? ustrip.(mm, quantity.(xyz, unit)) : xyz)...)
    end
    return g
end

_generic_body(m::CentralGenericModel) = (
    type = "CentralGenericModel",
    width = m.width, height = m.height,
    calibration_min_x = m.calib_min_x, calibration_min_y = m.calib_min_y,
    calibration_max_x = m.calib_max_x, calibration_max_y = m.calib_max_y,
    grid_width = size(m.grid, 2), grid_height = size(m.grid, 1), grid_layout = _GRID_LAYOUT,
    direction_grid = _grid_json(m.grid),
)
# The point grid is written in millimetres, VGC's in-memory length unit.
_generic_body(m::NoncentralGenericModel) = (
    type = "NoncentralGenericModel",
    width = m.width, height = m.height,
    calibration_min_x = m.calib_min_x, calibration_min_y = m.calib_min_y,
    calibration_max_x = m.calib_max_x, calibration_max_y = m.calib_max_y,
    grid_width = size(m.directions, 2), grid_height = size(m.directions, 1), grid_layout = _GRID_LAYOUT,
    direction_grid = _grid_json(m.directions),
    point_grid = _grid_json(m.points),
)

# A validated body back to the model; `unit` is the length unit the enclosing
# file declares (read only for a non-central point grid).
function _generic_from_body(d, unit::Units)
    gw, gh = Int(d.grid_width::Integer), Int(d.grid_height::Integer)
    dirs = _grid_from_json(d.direction_grid::Vector{Any}, gw, gh)
    args = (
        Int(d.width::Integer), Int(d.height::Integer),
        Int(d.calibration_min_x::Integer), Int(d.calibration_min_y::Integer),
        Int(d.calibration_max_x::Integer), Int(d.calibration_max_y::Integer),
    )
    d.type == "CentralGenericModel" && return CentralGenericModel(args..., dirs)
    return NoncentralGenericModel(args..., dirs, _grid_from_json(d.point_grid::Vector{Any}, gw, gh; unit, is_points = true))
end

"""
    save(path, m::CentralGenericModel; provenance = (;))
    save(path, m::NoncentralGenericModel; provenance = (;))

Write a generic grid camera model as a `vgio/generic_camera/v2` file (field
names mirror the reference implementation's YAML tags; grids row-major, in
the COLMAP pixel frame; a non-central point grid in millimetres).
"""
function save(
        path::AbstractString, m::Union{CentralGenericModel, NoncentralGenericModel};
        provenance::NamedTuple = (;)
    )
    head = (schema = _schema("generic_camera"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME))
    unit = m isa NoncentralGenericModel ? (length_unit = unit_string(mm),) : (;)
    _write_store(path, _GENERIC_CAMERA_SPEC, _with_provenance(merge(head, unit, _generic_body(m)), provenance))
    return nothing
end

"""
    load(CentralGenericModel, path) -> CentralGenericModel
    load(NoncentralGenericModel, path) -> NoncentralGenericModel

Read a `vgio/generic_camera/v2` file. The typed load is an assertion of
intent: loading a central record as non-central (or vice versa) errors.
"""
load(::Type{T}, path::AbstractString) where {T <: Union{CentralGenericModel, NoncentralGenericModel}} =
    _typed_generic(T, _load_generic(path), path)

function _load_generic(path::AbstractString)
    d = _read_store(path, "generic_camera", _GENERIC_CAMERA_SPEC)
    return _generic_from_body(d, haskey(d, :length_unit) ? parse_unit(d.length_unit::String) : mm)
end

_typed_generic(::Type{T}, m, path) where {T} = m isa T ? m : throw(
    ArgumentError("$(path) holds a $(nameof(typeof(m))), not a $(nameof(T))")
)
