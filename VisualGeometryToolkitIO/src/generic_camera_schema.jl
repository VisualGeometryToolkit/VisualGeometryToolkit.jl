# =============================================================================
# vgio/generic_camera/v3 — the Schöps et al. grid models, serialized
# =============================================================================
#
# One model per file, field names mirroring the reference implementation's
# YAML tags (puzzlepaint/camera_calibration) so the two formats translate
# one-to-one:
#
# {"schema": "vgio/generic_camera/v3",
#  "pixel_frame": "colmap",                        VGC's grid models ARE in it
#  "length_unit": "mm",                            non-central only (point_grid)
#  "type": "CentralGenericModel" | "NoncentralGenericModel",
#  "width": …, "height": …,
#  "calibration_min_x": …, "calibration_min_y": …,
#  "calibration_max_x": …, "calibration_max_y": …,
#  "grid_width": …, "grid_height": …, "grid_layout": "row_major",
#  "present": [[row, [[col_lo, col_hi], …]], …],   (optional) the present nodes: runs per lattice row, 0-based, inclusive; absent = every node
#  "direction_grid": [[x, y, z], …],               the present nodes, row-major
#  "point_grid":     [[x, y, z], …],               non-central only, the same nodes
#  "nominal_intrinsics": [fx, fy, cx, cy, skew],   (optional) the pinhole the model reports with, px
#  "domain": {"ellipse": [cx, cy, a, b, θ]},       (optional) the source's imaged region, px; absent = the whole plane
#  "provenance": {…}}                               (optional)
#
# The pixel frame is COLMAP's because VGC's generic models are defined in it —
# the reference's convention (origin at the top-left corner of the top-left
# pixel) is exactly that frame — so the declaration can only be "colmap", and a
# reader refuses any other rather than shifting a grid it cannot shift. The
# grids are the model's parameters (~10,000 nodes, ~200 KB of JSON at the
# paper's grid size, less the nodes a model does not hold), still
# configuration-adjacent and diffable. A rig whose sensor carries a generic
# model embeds this BODY (no schema or declarations — the rig declares them)
# where a parametric sensor embeds a camera body; it carries its own
# width/height, so a bare `PosedCamera` sensor suffices.
#
# THE PRESENT NODES (v3, 2026-11). A model holds the nodes of its lattice that
# its imaged region reaches (Core's `present_nodes`); the others are absent and
# have no value. `present` lists the held nodes as runs per lattice row — every
# row that holds one, rows increasing, its runs `[col_lo, col_hi]` increasing
# and apart (any shape: convex, cut by the frame, with holes) — and the grids
# hold exactly those nodes, row by row. A model reads back with NaN in its
# absent slots, which no evaluation reads. A body carries `present` only for a
# model that does not hold every node, and a body without it holds every node:
# the standalone file's and a rig sensor's alike, so rig/v6's bodies still read
# as they are and a `vgio/generic_camera/v2` file is a v3 record as it stands
# (its version step changes only the schema).
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

# The runs of present nodes per lattice row, `[[row, [[lo, hi], …]], …]`:
# each row once, rows increasing, its runs nonempty, increasing and apart (a
# gap of at least one node), every index a non-negative integer. The lattice's
# bounds and the node count are the body's check (`_generic_check`).
struct _RowRuns <: _Spec end
function _validate(::_RowRuns, x, file, at)
    x isa AbstractVector || _refuse(file, at, "expected an array of [row, [[col_lo, col_hi], …]] entries")
    isint(v) = v isa Integer && !(v isa Bool) && v >= 0
    last_row = -1
    for (i, e) in enumerate(x)
        a = "$(at)[$(i)]"
        (e isa AbstractVector && length(e) == 2 && isint(e[1]) && e[2] isa AbstractVector && !isempty(e[2])) ||
            _refuse(file, a, "expected [row, [[col_lo, col_hi], …]], a row index and its nonempty runs")
        e[1] > last_row || _refuse(file, a, "rows are listed once, increasing; row $(e[1]) follows row $(last_row)")
        last_row = e[1]
        last_hi = -2
        for (j, r) in enumerate(e[2])
            (r isa AbstractVector && length(r) == 2 && isint(r[1]) && isint(r[2]) && r[1] <= r[2]) ||
                _refuse(file, "$(a)[2][$(j)]", "expected a run [col_lo, col_hi] of non-negative integers, col_lo ≤ col_hi")
            r[1] > last_hi + 1 ||
                _refuse(file, "$(a)[2][$(j)]", "a row's runs are increasing and apart; [$(r[1]), $(r[2])] follows a run ending at $(last_hi)")
            last_hi = r[2]
        end
    end
    return nothing
end

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

# The number of nodes a body's runs hold (every node of the lattice without them).
_present_count(x) = haskey(x, :present) ?
    sum(r[2] - r[1] + 1 for e in x[:present] for r in e[2]; init = 0) : x[:grid_width] * x[:grid_height]

# The body's own check, shared by the standalone file and a rig sensor: a
# point grid exactly when the model is non-central, runs inside the lattice,
# and grids of the held nodes' count.
function _generic_check(x)
    noncentral = x[:type] == "NoncentralGenericModel"
    noncentral == haskey(x, :point_grid) ||
        return noncentral ? "a NoncentralGenericModel needs its point_grid" : "a CentralGenericModel has no point_grid"
    if haskey(x, :present)
        isempty(x[:present]) && return "`present` holds no node"
        for e in x[:present]
            e[1] < x[:grid_height] || return "`present` names row $(e[1]) of a lattice of grid_height = $(x[:grid_height]) rows"
            last(e[2])[2] < x[:grid_width] ||
                return "`present` names column $(last(e[2])[2]) of a lattice of grid_width = $(x[:grid_width]) columns"
        end
    end
    n = _present_count(x)
    what = haskey(x, :present) ? "the $(n) nodes `present` holds" : "grid_width × grid_height = $(n)"
    length(x[:direction_grid]) == n ||
        return "direction_grid has $(length(x[:direction_grid])) entries, expected $(what)"
    (noncentral && length(x[:point_grid]) != n) &&
        return "point_grid has $(length(x[:point_grid])) entries, expected $(what)"
    # the domain ellipse as Core's `Ellipse` holds it — semi-axes positive, the
    # major first — so it reads back as the numbers the file holds; the nominal
    # pinhole's focal lengths positive (K maps the image the right way up, K⁻¹
    # is finite)
    if haskey(x, :domain)
        p = _ellipse_record_problem(x[:domain][:ellipse], "the domain ellipse")
        p === nothing || return p
    end
    if haskey(x, :nominal_intrinsics)
        v = x[:nominal_intrinsics]
        (v[1] > 0 && v[2] > 0) || return "nominal_intrinsics' focal lengths fx, fy are positive; got fx = $(v[1]), fy = $(v[2])"
    end
    return nothing
end

const _GENERIC_DOMAIN = _Obj(:ellipse => _Vec(_Real(), 5))
const _GENERIC_OPTIONAL = Pair{Symbol, _Spec}[:nominal_intrinsics => _Vec(_Real(), 5), :domain => _GENERIC_DOMAIN]
const _GENERIC_PRESENT = Pair{Symbol, _Spec}[:present => _RowRuns()]
const _GENERIC_CAMERA_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("generic_camera"),
            :pixel_frame => _Tok(pixel_frame_name(CANONICAL_PIXEL_FRAME)),
            _GENERIC_BODY_FIELDS...,
        ],
        Pair{Symbol, _Spec}[:length_unit => _LENGTH_UNIT, :point_grid => _GRID, _GENERIC_OPTIONAL..., _GENERIC_PRESENT..., _PROVENANCE...],
    ),
    x -> something(
        _generic_check(x),
        (x[:type] == "NoncentralGenericModel") == haskey(x, :length_unit) ? nothing :
            "a non-central model declares the length_unit of its point_grid, and only it does",
        Some(nothing),
    ),
)

# The present nodes of a `[gy, gx]` mask as runs per row, 0-based and inclusive.
function _present_runs(P::AbstractMatrix{Bool})
    rows = Any[]
    for gy in 1:size(P, 1)
        runs = Any[]
        gx = 1
        while gx <= size(P, 2)
            if P[gy, gx]
                lo = gx
                while gx < size(P, 2) && P[gy, gx + 1]
                    gx += 1
                end
                push!(runs, [lo - 1, gx - 1])
            end
            gx += 1
        end
        isempty(runs) || push!(rows, Any[gy - 1, runs])
    end
    return rows
end
# The runs back to a `[gy, gx]` mask of a `gh × gw` lattice.
function _present_from_runs(runs, gw::Int, gh::Int)
    P = falses(gh, gw)
    for e in runs, r in e[2]::Vector{Any}
        P[(e[1]::Integer) + 1, ((r[1]::Integer) + 1):((r[2]::Integer) + 1)] .= true
    end
    return P
end

# The held nodes of a grid, row by row: a direction grid as stored, a point
# grid (plain metres) in the file's length unit.
_grid_json(g::Matrix{<:Vec3}, P::AbstractMatrix{Bool}) =
    [[Float64(g[gy, gx][k]) for k in 1:3] for gy in 1:size(g, 1) for gx in 1:size(g, 2) if P[gy, gx]]
_point_grid_json(g::Matrix{<:Vec3}, P::AbstractMatrix{Bool}, unit::_LeafUnit) =
    [Float64[_leaf_power(Float64(g[gy, gx][k]), unit, 1) for k in 1:3] for gy in 1:size(g, 1) for gx in 1:size(g, 2) if P[gy, gx]]

# A grid back from its held nodes' row-major entries: a direction grid as
# stored, a point grid read in the file's length unit into plain metres; NaN in
# the absent slots (no value: no evaluation reads one).
function _grid_from_json(a, P::AbstractMatrix{Bool}; unit::Units = mm, is_points::Bool = false)
    g = fill(Vec3{Float64}(NaN, NaN, NaN), size(P))
    for (I, v) in zip(findall(permutedims(P)), a)    # the held nodes in row-major order
        xyz = Float64[is_points ? _read_power(c::Real, unit, 1) : Float64(c::Real) for c in v::Vector{Any}]
        g[I[2], I[1]] = Vec3{Float64}(xyz...)
    end
    return g
end

# The optional nominal pinhole (a K's five numbers, style.jl) and domain (an
# ellipse's minimal form) of a model, empty when it has none — the record's and
# the identity tree's (`_id_grid_model`) alike.
function _generic_extras(s::VgioStyle, m)
    n = nominal_intrinsics(m)
    nom = n === nothing ? (;) : (nominal_intrinsics = _k_values(n.K),)
    d = m.domain
    dom = d isa WholePlane ? (;) : d isa Ellipse ? (domain = (ellipse = JSON.lower(s, d),),) : _refuse_domain(s, d)
    return merge(nom, dom)
end
# the caller's context: the identity tree's style (`_PERSISTED`) is content_id's, any other a writer's
_refuse_domain(s::VgioStyle, d) = throw(
    ArgumentError(
        s.length_unit isa _Persisted ?
            "content_id: a generic model's domain is an Ellipse or the whole plane; got a $(nameof(typeof(d)))" :
            "save: a generic model's domain is stored as an Ellipse or the whole plane; got a $(nameof(typeof(d)))"
    )
)
_generic_extras_from(s::VgioStyle, d) = (
    nominal = haskey(d, :nominal_intrinsics) ? LogicalIntrinsics(_k_from_values(d.nominal_intrinsics::Vector{Any})) : nothing,
    domain = haskey(d, :domain) ? JSON.lift(s, Ellipse, d.domain.ellipse::Vector{Any}) : WholePlane(),
)

const _GenericModel = Union{CentralGenericModel, NoncentralGenericModel}

# A model's body on IO's style, the standalone file's and a rig sensor's:
# `present` for a model that does not hold every node, a non-central point grid
# (plain metres) in the style's length form (the file's unit; the identity
# tree's persisted metres).
function JSON.lower(s::VgioStyle, m::_GenericModel)
    P = present_nodes(m)
    dirs = m isa CentralGenericModel ? m.grid : m.directions
    return merge(
        (
            type = String(nameof(typeof(m))),
            width = m.width, height = m.height,
            calibration_min_x = m.calib_min_x, calibration_min_y = m.calib_min_y,
            calibration_max_x = m.calib_max_x, calibration_max_y = m.calib_max_y,
            grid_width = size(dirs, 2), grid_height = size(dirs, 1), grid_layout = _GRID_LAYOUT,
        ),
        all(P) ? (;) : (present = _present_runs(P),),
        (direction_grid = _grid_json(dirs, P),),
        m isa NoncentralGenericModel ? (point_grid = _point_grid_json(m.points, P, s.length_unit),) : (;),
        _generic_extras(s, m),
    )
end

# A validated body back to the model; the style's length unit is the one the
# enclosing file declares (read only for a non-central point grid).
function JSON.lift(s::VgioStyle, ::Type{_GenericModel}, d::_Record)
    unit = s.length_unit
    gw, gh = Int(d.grid_width::Integer), Int(d.grid_height::Integer)
    P = haskey(d, :present) ? _present_from_runs(d.present::Vector{Any}, gw, gh) : trues(gh, gw)
    dirs = _grid_from_json(d.direction_grid::Vector{Any}, P)
    args = (
        Int(d.width::Integer), Int(d.height::Integer),
        Int(d.calibration_min_x::Integer), Int(d.calibration_min_y::Integer),
        Int(d.calibration_max_x::Integer), Int(d.calibration_max_y::Integer),
    )
    # the controls as stored (a model's controls may be off the sphere — a quasi-interpolated start, a tangent-plane optimizer's
    # result — and a re-normalization would change its spline; a unit grid reads back unit either way)
    x = _generic_extras_from(s, d)
    d.type == "CentralGenericModel" && return CentralGenericModel(args..., dirs; unit_controls = false, x..., present = P)
    return NoncentralGenericModel(args..., dirs, _grid_from_json(d.point_grid::Vector{Any}, P; unit, is_points = true);
                                  unit_controls = false, x..., present = P)
end

"""
    save(path, m::CentralGenericModel; provenance = (;))
    save(path, m::NoncentralGenericModel; length_unit = mm, provenance = (;))

Write a generic grid camera model as a `vgio/generic_camera/v3` file (field
names mirror the reference implementation's YAML tags; the nodes the model
holds, `present` as runs per lattice row and the grids row-major over them, in
the COLMAP pixel frame; a non-central point grid — plain metres in memory —
in `length_unit`).
"""
function save(
        path::AbstractString, m::Union{CentralGenericModel, NoncentralGenericModel};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    head = (schema = _schema("generic_camera"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME))
    unit = m isa NoncentralGenericModel ? (length_unit = unit_string(length_unit),) : (;)
    s = VgioStyle(; length_unit)
    _write_store(path, _GENERIC_CAMERA_SPEC, _with_provenance(merge(head, unit, JSON.lower(s, m)), provenance); style = s)
    return nothing
end

"""
    load(CentralGenericModel, path) -> CentralGenericModel
    load(NoncentralGenericModel, path; file_unit = nothing) -> NoncentralGenericModel

Read a `vgio/generic_camera/v3` file (a non-central point grid into plain
metres; the absent nodes NaN, which no evaluation reads) or a
`vgio/generic_camera/v2` one (every node present), or the reference
implementation's own `intrinsics0.yaml` (Schöps et al.'s camera_calibration; the form is decided by the file's content, see
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
    return JSON.lift(VgioStyle(; length_unit = haskey(d, :length_unit) ? parse_unit(d.length_unit::String) : mm), _GenericModel, d::_Record)
end

_typed_generic(::Type{T}, m, path) where {T} = m isa T ? m : throw(
    ArgumentError("$(path) holds a $(nameof(typeof(m))), not a $(nameof(T))")
)
