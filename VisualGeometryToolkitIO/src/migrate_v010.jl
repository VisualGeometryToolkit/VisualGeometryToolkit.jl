# =============================================================================
# The IO 0.10 steps: every kind's previous version → its 0.10 major version
# =============================================================================
#
# IO 0.10 moved every kind up a major version (validate.jl, "THE VERSION OF A
# KIND"). Each kind's previous version has ONE step here, `migrate_schema(::Val{
# <its 0.9 schema>}, tree)`, which carries a record over exactly:
#
# - the ENVELOPE: the top-level declarations (`length_unit`, `pixel_frame`,
#   `rotation_encoding`, `transform_convention`, `point_index_base`,
#   `time_unit`) move into the `units` and `conventions` objects; what a 0.9
#   record left implicit is stated as its version meant it (a record with no
#   length declares millimetres and holds no length; time in seconds, angles in
#   radians, density in dpi, radiance relative); the `generator` is this
#   package (the record is the migration's);
# - a stored rotation with w < 0, which every 0.9 reader flipped (q and -q are
#   one rotation), is written w ≥ 0, as that reader read it; the 0.10 reader
#   refuses one;
# - the NAMES: `t` → `frame_index` (the frame join key), `camera_optical` →
#   `camera` (the frame mirrors its type), a camera's `calibration` →
#   `calibration_record`, the target body tags `points`/`planar` →
#   `point_target`/`planar_target`, `l_max` → `max_degree`, the identity's;
# - the SHAPES: every Size2 a `{width, height}` object (an image size, a pitch,
#   a lattice's `grid_size`), the lattice `present` runs 1-based (+1), the one
#   lattice record (`controls: {direction, point}`), one covariance key.
#
# No number moves but a w < 0 quaternion's signs and the +1 of an index. Each
# step's own comment says what else it moves.

const _V010_DECLARATIONS = ("length_unit", "pixel_frame", "rotation_encoding", "transform_convention", "point_index_base", "time_unit")

# A 0.9 record's envelope as the 0.10 one: the record's other keys kept, its
# declarations moved into `units` and `conventions`, its schema `to`. `length`
# and `pixel` are what the record declares (or, absent, what its version meant:
# no length held, the colmap pixels of every 0.9 record without a pixel frame).
function _envelope_v010(old::AbstractDict, to::AbstractString; length = get(old, "length_unit", "mm"), pixel = get(old, "pixel_frame", "colmap"))
    t = Dict{String, Any}(
        "schema" => String(to),
        "generator" => _lower(_generator_json()),
        "units" => Dict{String, Any}("length" => length, "time" => _TIME_UNIT, "angle" => _ANGLE_UNIT, "pixel" => pixel,
                                     "density" => "dpi", "radiance" => _RADIANCE_UNIT, "length_roles" => Dict{String, Any}()),
        "conventions" => _lower(_CONVENTIONS_JSON),
    )
    for (k, v) in old
        (k == "schema" || k in _V010_DECLARATIONS) || (t[String(k)] = v)
    end
    return _rotations_v010!(t)
end

# A record's file-level keys only (its schema, declarations and provenance), not its body: the envelope a
# record that becomes a store's entry (an old grid file's body) leaves behind.
_file_head(t::AbstractDict) = Dict{String, Any}(k => v for (k, v) in t if k in ("schema", "provenance", _V010_DECLARATIONS...))

# Every stored rotation `[w, x, y, z]` with w < 0 as the 0.9 reader read it, -q.
function _rotations_v010!(x::AbstractDict)
    for (k, v) in x
        if k == "rotation" && v isa AbstractVector && length(v) == 4 && all(e -> e isa Real, v) && v[1] < 0
            x[k] = Any[-e for e in v]
        else
            _rotations_v010!(v)
        end
    end
    return x
end
_rotations_v010!(x::AbstractVector) = (foreach(_rotations_v010!, x); x)
_rotations_v010!(x) = x

# A key renamed in a dict (a copy; absent, the dict as it is).
_renamed_key(d::AbstractDict, from::AbstractString, to::AbstractString) =
    haskey(d, from) ? Dict{String, Any}((k == from ? to : String(k)) => v for (k, v) in d) : Dict{String, Any}(d)
# The frame join key of every frame of a record, `t` → `frame_index`.
_frames_v010(t::AbstractDict) = haskey(t, "frames") ? merge(t, Dict{String, Any}("frames" => Any[_renamed_key(f, "t", "frame_index") for f in t["frames"]])) : t
# A frame name of the grammar as 0.10 names it (`camera_optical` → `camera`).
_frame_name_v010(s) = s == "camera_optical" ? "camera" : s
function _frame_names_v010(t::AbstractDict, keys_ = ("from", "to", "frame"))
    n = Dict{String, Any}(t)
    for k in keys_
        haskey(n, k) && (n[k] = _frame_name_v010(n[k]))
    end
    return n
end
_size_v010(v::AbstractVector) = Dict{String, Any}("width" => v[1], "height" => v[2])
_runs_v010(runs) = Any[Any[e[1] + 1, Any[Any[r[1] + 1, r[2] + 1] for r in e[2]]] for e in runs]

# --- the camera bodies --------------------------------------------------------

# A 0.9 lattice (`{area, grid_size: [gw, gh], present?: 0-based runs, <arrays>}`) as
# the 0.10 lattice record, its arrays `names` (0.9 key => 0.10 control name).
function _lattice_v010(sp::AbstractDict, names)
    l = Dict{String, Any}("area" => sp["area"], "grid_size" => _size_v010(sp["grid_size"]),
                          "controls" => Dict{String, Any}(to => sp[from] for (from, to) in names))
    haskey(sp, "present") && (l["present"] = _runs_v010(sp["present"]))
    return l
end
# A 0.9 lens record: a spline lens's lattice and its forward spline; any other as it stands.
function _lens_v010(l::AbstractDict)
    haskey(l, "spline") || return l
    sp = l["spline"]
    n = _lattice_v010(sp, ("controls" => "direction",))
    haskey(sp, "forward") && (n["forward"] = _lattice_v010(sp["forward"], ("controls" => "point",)))
    return Dict{String, Any}("spline" => n)
end
# A 0.9 bounded camera body (a store entry's, a rig sensor's, a non-central
# record's, an owner's embedded one) as the 0.10 body: `image_size`, the pitch an
# object, the lattice record, a store entry's `calibration_record`. A body
# already in the 0.10 form (an old grid body's conversion builds the current
# body) is returned as it is.
function _camera_body_v010(b::AbstractDict)
    haskey(b, "image_size") && return Dict{String, Any}(b)
    n = Dict{String, Any}()
    for (k, v) in b
        if k in ("width", "height")
            continue
        elseif k == "pitch"
            n["pitch"] = _size_v010(v)
        elseif k == "lens"
            n["lens"] = _lens_v010(v)
        elseif k == "spline"         # a non-central body's
            n["spline"] = _lattice_v010(v, ("directions" => "direction", "points" => "point"))
        elseif k == "calibration"
            n["calibration_record"] = v
        else
            n[String(k)] = v
        end
    end
    n["image_size"] = Dict{String, Any}("width" => b["width"], "height" => b["height"])
    return n
end

# vgio/cameras/v5 → v6.0 (IO 0.10): the envelope; every camera's body
# (`_camera_body_v010`). The store holds any bounded camera from v6.0 on, so a
# v5 store (central cameras only) is a v6.0 one as it stands.
function _cameras_v5_to_v6(old::AbstractDict)
    t = _envelope_v010(old, "vgio/cameras/v6.0")
    t["cameras"] = Any[_camera_body_v010(c) for c in old["cameras"]]
    return t
end
migrate_schema(::Val{Symbol("vgio/cameras/v5")}, t::AbstractDict) = _cameras_v5_to_v6(t)

# A vgio/noncentral_spline/v1 record (its own kind before IO 0.10) as the
# vgio/cameras/v6.0 store of its one camera, named `name` (its file's basename):
# no tree step can name a store's camera, so the reader (`_one_camera_store`) and
# the file migration call this with the name.
function _noncentral_store_v6(old::AbstractDict, name::AbstractString)
    t = _envelope_v010(old, "vgio/cameras/v6.0")
    body = Dict{String, Any}(k => v for (k, v) in old if !(k in ("schema", "provenance", _V010_DECLARATIONS...)))
    for k in keys(body)
        delete!(t, k)
    end
    t["cameras"] = Any[merge(Dict{String, Any}("name" => String(name)), _camera_body_v010(body))]
    return t
end
