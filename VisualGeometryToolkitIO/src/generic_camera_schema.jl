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
#  "nominal_intrinsics": [fx, fy, cx, cy, skew],   (optional) the pinhole the model reports with, px
#  "domain": {"ellipse": [cx, cy, a, b, θ]},       (optional) the source's imaged region, px; absent = the whole plane
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
# own width/height, so a bare `PosedCamera` sensor suffices.
#
# The point grid is a plain model's numbers: VGC's grid models hold
# `Vec3{T}`, so the points are plain METRES in memory — the unit VGC strips a
# physical camera's extrinsics in to apply its model — and the file's
# `length_unit` (mm unless the writer is given another) on disk, through
# `read_length` / `write_length`.
#
# A model built from a source camera (Core's `init = :quasi_interpolate`)
# carries that camera's pinhole, `nominal_intrinsics`, and its imaged region,
# `domain` — an `Ellipse` (an image circle) or the whole plane. Neither is
# derivable from the grid, so both are stored when present (the same five
# numbers as a camera's `intrinsics`, focal lengths positive; the ellipse's
# centre, semi-axes — positive, the major first — and angle), and read back bit
# for bit; a domain of any other region type is refused by the writer. Schöps' YAML carries neither: its models read with
# none and the whole plane.

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
    # the domain ellipse as Core's `Ellipse` holds it — semi-axes positive, the
    # major first — so it reads back as the numbers the file holds; the nominal
    # pinhole's focal lengths positive (K maps the image the right way up, K⁻¹
    # is finite)
    if haskey(x, :domain)
        e = x[:domain][:ellipse]
        (e[3] > 0 && e[4] > 0) || return "the domain ellipse's semi-axes are positive; got a = $(e[3]), b = $(e[4])"
        e[3] >= e[4] || return "the domain ellipse is stored major semi-axis first (a ≥ b, θ the angle of a); got a = $(e[3]) < b = $(e[4])"
    end
    if haskey(x, :nominal_intrinsics)
        v = x[:nominal_intrinsics]
        (v[1] > 0 && v[2] > 0) || return "nominal_intrinsics' focal lengths fx, fy are positive; got fx = $(v[1]), fy = $(v[2])"
    end
    return nothing
end

const _GENERIC_DOMAIN = _Obj(:ellipse => _Vec(_Real(), 5))
const _GENERIC_OPTIONAL = Pair{Symbol, _Spec}[:nominal_intrinsics => _Vec(_Real(), 5), :domain => _GENERIC_DOMAIN]
const _GENERIC_BODY = _Checked(_Obj(_GENERIC_BODY_FIELDS, Pair{Symbol, _Spec}[:point_grid => _GRID, _GENERIC_OPTIONAL...]), _generic_check)
const _GENERIC_CAMERA_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("generic_camera"),
            :pixel_frame => _Tok(pixel_frame_name(CANONICAL_PIXEL_FRAME)),
            _GENERIC_BODY_FIELDS...,
        ],
        Pair{Symbol, _Spec}[:length_unit => _LENGTH_UNIT, :point_grid => _GRID, _GENERIC_OPTIONAL..., _PROVENANCE...],
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
# a point grid (plain metres) in the file's length unit
_point_grid_json(g::Matrix{<:Vec3}, unit::Units) =
    [Float64[_write_power(Float64(v[k]), unit, 1) for k in 1:3] for gy in 1:size(g, 1) for v in (g[gy, :]...,)]

# A grid back from its row-major entries: a direction grid as stored, a point
# grid read in the file's length unit into plain metres.
function _grid_from_json(a, gw::Int, gh::Int; unit::Units = mm, is_points::Bool = false)
    g = Matrix{Vec3{Float64}}(undef, gh, gw)
    for (i, v) in enumerate(a)
        gy, gx = divrem(i - 1, gw)
        xyz = Float64[is_points ? _read_power(c::Real, unit, 1) : Float64(c::Real) for c in v::Vector{Any}]
        g[gy + 1, gx + 1] = Vec3{Float64}(xyz...)
    end
    return g
end

# The optional nominal pinhole and domain of a model (empty when it has none).
function _generic_extras(m)
    n = nominal_intrinsics(m)
    nom = n === nothing ? (;) : (K = n.K;
        (nominal_intrinsics = Float64[K[1, 1], K[2, 2], K[1, 3], K[2, 3], K[1, 2]],))
    d = m.domain
    dom = d isa WholePlane ? (;) : d isa Ellipse ?
        (domain = (ellipse = Float64[d.center[1], d.center[2], d.a, d.b, d.θ],),) :
        throw(ArgumentError("save: a generic model's domain is stored as an Ellipse or the whole plane; got a $(nameof(typeof(d)))"))
    return merge(nom, dom)
end
function _generic_extras_from(d)
    nominal = haskey(d, :nominal_intrinsics) ? (v = Float64.(d.nominal_intrinsics::Vector{Any});
        LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(v[1], 0, 0, v[5], v[2], 0, v[3], v[4], 1)))) : nothing
    domain = haskey(d, :domain) ? (e = Float64.(d.domain.ellipse::Vector{Any}); Ellipse(Point2{Float64}(e[1], e[2]), e[3], e[4], e[5])) :
        WholePlane()
    return (; nominal, domain)
end

_generic_body(m::CentralGenericModel, u::Units = mm) = merge(_generic_body_core(m, u), _generic_extras(m))
_generic_body(m::NoncentralGenericModel, u::Units = mm) = merge(_generic_body_core(m, u), _generic_extras(m))
_generic_body_core(m::CentralGenericModel, ::Units = mm) = (
    type = "CentralGenericModel",
    width = m.width, height = m.height,
    calibration_min_x = m.calib_min_x, calibration_min_y = m.calib_min_y,
    calibration_max_x = m.calib_max_x, calibration_max_y = m.calib_max_y,
    grid_width = size(m.grid, 2), grid_height = size(m.grid, 1), grid_layout = _GRID_LAYOUT,
    direction_grid = _grid_json(m.grid),
)
# The point grid (plain metres) is written in the file's `unit`.
_generic_body_core(m::NoncentralGenericModel, unit::Units = mm) = (
    type = "NoncentralGenericModel",
    width = m.width, height = m.height,
    calibration_min_x = m.calib_min_x, calibration_min_y = m.calib_min_y,
    calibration_max_x = m.calib_max_x, calibration_max_y = m.calib_max_y,
    grid_width = size(m.directions, 2), grid_height = size(m.directions, 1), grid_layout = _GRID_LAYOUT,
    direction_grid = _grid_json(m.directions),
    point_grid = _point_grid_json(m.points, unit),
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
    # the controls as stored (a model's controls may be off the sphere — a quasi-interpolated start, a tangent-plane optimizer's
    # result — and a re-normalization would change its spline; a unit grid reads back unit either way)
    x = _generic_extras_from(d)
    d.type == "CentralGenericModel" && return CentralGenericModel(args..., dirs; unit_controls = false, x...)
    return NoncentralGenericModel(args..., dirs, _grid_from_json(d.point_grid::Vector{Any}, gw, gh; unit, is_points = true);
                                  unit_controls = false, x...)
end

"""
    save(path, m::CentralGenericModel; provenance = (;))
    save(path, m::NoncentralGenericModel; length_unit = mm, provenance = (;))

Write a generic grid camera model as a `vgio/generic_camera/v2` file (field
names mirror the reference implementation's YAML tags; grids row-major, in
the COLMAP pixel frame; a non-central point grid — plain metres in memory —
in `length_unit`).
"""
function save(
        path::AbstractString, m::Union{CentralGenericModel, NoncentralGenericModel};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    head = (schema = _schema("generic_camera"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME))
    unit = m isa NoncentralGenericModel ? (length_unit = unit_string(length_unit),) : (;)
    _write_store(path, _GENERIC_CAMERA_SPEC, _with_provenance(merge(head, unit, _generic_body(m, length_unit)), provenance))
    return nothing
end

"""
    load(CentralGenericModel, path) -> CentralGenericModel
    load(NoncentralGenericModel, path; file_unit = nothing) -> NoncentralGenericModel

Read a `vgio/generic_camera/v2` file (a non-central point grid into plain
metres), or the reference implementation's own `intrinsics0.yaml` (Schöps et
al.'s camera_calibration; the form is decided by the file's content, see
schoeps.jl). The typed load is an assertion of intent: loading a central record as
non-central (or vice versa) errors.

The YAML is in [`SCHOEPS_PIXEL_FRAME`](@ref), which is the COLMAP frame the
models are defined in: its numbers are read as stored. A non-central YAML does
not record the length unit of its `point_grid`, so the caller states it,
`file_unit` (a length unit, e.g. `m`); a vgio file declares its own, and a
`file_unit` beside it, or beside a central model (which has no lengths), is
refused.
"""
function load(
        ::Type{T}, path::AbstractString; file_unit::Union{Nothing, Units} = nothing
    ) where {T <: Union{CentralGenericModel, NoncentralGenericModel}}
    if _file_form(path) !== :json
        (T === CentralGenericModel && file_unit !== nothing) &&
            throw(ArgumentError("$(path): a CentralGenericModel has no lengths; `file_unit` does not apply"))
        return _load_schoeps_model(T, path, file_unit)
    end
    file_unit === nothing ||
        throw(ArgumentError("$(path): a vgio file declares its own length_unit; `file_unit` is for a Schöps YAML"))
    return _typed_generic(T, _load_generic(path), path)
end

function _load_generic(path::AbstractString)
    d = _read_store(path, "generic_camera", _GENERIC_CAMERA_SPEC)
    return _generic_from_body(d, haskey(d, :length_unit) ? parse_unit(d.length_unit::String) : mm)
end

_typed_generic(::Type{T}, m, path) where {T} = m isa T ? m : throw(
    ArgumentError("$(path) holds a $(nameof(typeof(m))), not a $(nameof(T))")
)
