# =============================================================================
# vgio/noncentral_spline/v1 — the non-central grid camera of Schöps et al.
# =============================================================================
#
# One `BoundedCamera{<:NoncentralSplineModel}` per file: the non-central grid
# camera (Schöps, Larsson, Pollefeys, Sattler, CVPR 2020), its two splines — the
# directions and the line points of its observation lines — laid over one
# rectangle of the K-normalized plane x = K⁻¹u:
#
# {"schema": "vgio/noncentral_spline/v1", "pixel_frame": "colmap", "length_unit": "mm",
#  "type": "noncentral_spline",
#  "width": …, "height": …,
#  "K": {"fx": …, "fy": …, "cx": …, "cy": …, "skew": 0.0},   the gauge pinhole, in the file's pixel frame
#  "spline": {"area": [x, y, w, h], "grid_size": [gw, gh],
#             "present": [[row, [[col_lo, col_hi], …]], …],  (optional) the held nodes; absent = every node
#             "directions": [[x, y, z], …],                  the held nodes, row by row
#             "points": [[x, y, z], …]},                     the same nodes, in length_unit
#  "pitch": [w, h],                                          (optional) the sensor's pixel pitch, in length_unit
#  "lens_coverage": {"ellipse": [cx, cy, a, b, θ]},          (optional; or circle, rect) where light reaches the sensor
#  "provenance": {…}}                                        (optional)
#
# The CENTRAL grid camera is not this kind: it is a `CameraModel` with a
# `SplineLens`, stored as any camera (a vgio/cameras record whose lens is the
# `{"spline": …}` record, camera_schema.jl). A central `vgio/generic_camera/v2`
# or `v3` file reads as the camera store of its one camera — `load(BoundedCamera,
# path)` — and a non-central one through its step to this kind (migrate.jl; the
# kind was the old grid kind's v4 before it was renamed, unreleased); neither is
# rewritten (owner, 2026-11-09): `vgio_migrate` writes a central one's camera
# store, a non-central one's record, to a new file beside it. A record naming the
# central type is refused, pointing at that read.
#
# The spline record is the lens record's (`_spline_json`): the held nodes (Core's
# presence mask) as runs per lattice row and the controls over them; the absent
# nodes read back as NaN, which no evaluation reads. The point controls are plain
# METRES in memory — the unit VGC strips a physical camera's extrinsics in — and
# the file's `length_unit` on disk (`read_length` / `write_length`). A rig sensor
# that is a non-central grid camera embeds this BODY (no schema or declarations:
# the rig declares them).

# The record's type: a token of the format, never a Julia type's name (a renamed type would otherwise force a new schema).
const _NONCENTRAL_TYPE = "noncentral_spline"
const _GRID = _Vec(_XYZ)
const _NONCENTRAL_SPLINE = _Obj(
    Pair{Symbol, _Spec}[:area => _Vec(_Real(), 4), :grid_size => _Vec(_PosInt(), 2), :directions => _GRID, :points => _GRID],
    Pair{Symbol, _Spec}[:present => _RowRuns()],
)
const _NONCENTRAL_BODY_FIELDS = Pair{Symbol, _Spec}[
    :type => _Named(
        ==(_NONCENTRAL_TYPE),
        "\"$(_NONCENTRAL_TYPE)\" — a central grid camera is a camera with a spline lens, a vgio/cameras record; a central " *
            "vgio/generic_camera/v2 or v3 file reads through its migration, load(BoundedCamera, path)",
    ),
    :width => _PosInt(), :height => _PosInt(), :K => _K_SPEC, :spline => _NONCENTRAL_SPLINE,
]
const _NONCENTRAL_BODY_OPTIONAL = Pair{Symbol, _Spec}[:pitch => _PITCH, :lens_coverage => _LENS_COVERAGE]

# The body's own check, the standalone file's and a rig sensor's: its lattice
# (`_lattice_problem`: two control arrays of the held nodes' count) and its lens
# coverage. Its K, a gauge (skew 0, positive focal lengths), is Core's one check
# of a spline camera's K, when the model is built.
function _noncentral_check(x)
    p = _lattice_problem(x[:spline], (:directions, :points))
    p === nothing || return p
    return _lens_coverage_problem(get(x, :lens_coverage, nothing), "the non-central camera")
end
const _NONCENTRAL_BODY = _CheckedSpec(_Obj(_NONCENTRAL_BODY_FIELDS, _NONCENTRAL_BODY_OPTIONAL), _noncentral_check)
const _NONCENTRAL_SPLINE_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("noncentral_spline"), :pixel_frame => _PIXEL_FRAME, :length_unit => _LENGTH_UNIT,
            _NONCENTRAL_BODY_FIELDS...,
        ],
        Pair{Symbol, _Spec}[_NONCENTRAL_BODY_OPTIONAL..., _PROVENANCE...],
    ),
    _noncentral_check,
)

# A non-central camera's body on IO's style, the standalone file's and a rig
# sensor's: its extent, K (in the style's pixel frame) and spline record, the
# point controls (plain metres) in the style's length form (the file's unit; the
# identity tree's persisted metres) — or `points`, a migration's own numbers as
# its file holds them.
function _model_body(s::VgioStyle, m::NoncentralSplineModel, ext::Size2; points = nothing)
    P = _held_mask(m.present, size(m.directions.controls))
    sp = _spline_json(m.directions, P)
    pts = points === nothing ? _controls_json(m.points.controls, P, x -> _leaf_power(x, s.length_unit, 1)) : points
    return (
        type = _NONCENTRAL_TYPE, width = Int(ext.width), height = Int(ext.height), K = _k_record(m.intrinsics.K, _pixel_frame(s)),
        spline = merge((area = sp.area, grid_size = sp.grid_size), haskey(sp, :present) ? (present = sp.present,) : (;),
                       (directions = sp.controls, points = pts)),
    )
end

# …and back: the model and its extent, from a validated body whose K is in the
# style's pixel frame and whose points are in its length unit.
function _noncentral_from_body(s::VgioStyle, b::Record)
    sp = b.spline
    L = _lattice_from_json(sp.grid_size, haskey(sp, :present) ? sp.present : nothing, sp.area)
    D = ChartSpline(L.area, _controls_from_json(sp.directions::Vector{Any}, L.P, Val(3)))
    X = ChartSpline(L.area, _controls_from_json(sp.points::Vector{Any}, L.P, Val(3), x -> _read_power(x, s.length_unit, 1)))
    m = NoncentralSplineModel(LogicalIntrinsics(_k_from_record(b.K, _pixel_frame(s))), D, X; L.present)
    return m, _extent_from_body(b)
end

"""
    save(path, camera::BoundedCamera{<:NoncentralSplineModel}; length_unit = mm, provenance = (;))

Write a non-central grid camera (Schöps et al.) as a `vgio/noncentral_spline/v1`
file: its extent, gauge `K` and the two splines over the held nodes (`present`
as runs per lattice row, the controls row by row), the line points — plain
metres in memory — in `length_unit`; a physical sensor's pitch and the camera's
lens coverage when it has them. A central grid camera is a `CameraModel` with a
`SplineLens`: save it in a camera store (`save(path, Dict(:name => camera))`).
"""
function save(
        path::AbstractString, camera::BoundedCamera{<:NoncentralSplineModel};
        length_unit::Units = mm, provenance::NamedTuple = (;)
    )
    s = VgioStyle(; length_unit)
    head = (schema = _schema("noncentral_spline"), pixel_frame = pixel_frame_name(CANONICAL_PIXEL_FRAME), length_unit = unit_string(length_unit))
    body = merge(head, _bounded_body(s, camera))
    _write_store(path, _NONCENTRAL_SPLINE_SPEC, _with_provenance(body, provenance); style = s)
    return nothing
end

# A noncentral_spline record, or an old grid file (vgio/generic_camera, the kind's
# former name) that is not a central v2/v3 one: `load(BoundedCamera, path)` reads it,
# an old file through its version steps.
_is_noncentral_grid(d) =
    d isa AbstractDict && (v = _split_schema(get(d, :schema, nothing)); v !== nothing &&
        (v.kind == "noncentral_spline" || (v.kind == _FORMER_KIND["noncentral_spline"] && !_is_central_grid(d))))

function _load_noncentral(raw, path::AbstractString)
    d = _checked_store(raw, path, "noncentral_spline", _NONCENTRAL_SPLINE_SPEC)
    u = parse_unit(d.length_unit::String)
    s = VgioStyle(; length_unit = u, pixel_frame = pixel_frame_from_name(d.pixel_frame::String))
    return _bounded_from_body(s, d)
end
