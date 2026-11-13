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

# A 0.9 record's envelope as the 0.10 one: the record's other keys kept (a
# copy: a step never moves its input's numbers), its declarations moved into
# `units` and `conventions`, its schema `to`. `length`, `pixel` and `density` are
# what the record declares (or, absent, what its version meant: no length held,
# the colmap pixels of every 0.9 record without a pixel frame, a units sidecar's
# dpi).
function _envelope_v010(old::AbstractDict, to::AbstractString; length = get(old, "length_unit", "mm"),
                        pixel = get(old, "pixel_frame", "colmap"), density = "dpi")
    t = Dict{String, Any}(
        "schema" => String(to),
        "generator" => _lower(_generator_json()),
        "units" => Dict{String, Any}("length" => length, "time" => _TIME_UNIT, "angle" => _ANGLE_UNIT, "pixel" => pixel,
                                     "density" => density, "radiance" => _RADIANCE_UNIT, "length_roles" => Dict{String, Any}()),
        "conventions" => _lower(_CONVENTIONS_JSON),
    )
    for (k, v) in old
        (k == "schema" || k in _V010_DECLARATIONS) || (t[String(k)] = _lower(v))
    end
    return _rotations_v010!(t)
end

# A record's file-level keys only (its schema, declarations and provenance), not its body: the envelope a
# record that becomes a store's entry (an old grid file's body) leaves behind.
_file_head(t::AbstractDict) = Dict{String, Any}(k => v for (k, v) in t if k in ("schema", "provenance", _V010_DECLARATIONS...))

# Every stored rotation `[w, x, y, z]` with w < 0 as the 0.9 reader read it, -q
# (the free-form `provenance` is never read, so never moved).
function _rotations_v010!(x::AbstractDict)
    for (k, v) in x
        if k == "rotation" && v isa AbstractVector && length(v) == 4 && all(e -> e isa Real, v) && v[1] < 0
            x[k] = Any[-e for e in v]
        elseif k != "provenance"
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

# --- the target bodies --------------------------------------------------------

# A 0.9 target body as the 0.10 one: the tags by the tag rule (`points` →
# `point_target`, `planar` → `planar_target`); a point target's points in the
# one point vocabulary — each `{"xy" | "xyz": […]}`, its `covariance` beside it
# (0.9's parallel `point_covariances`, entry for entry); a planar body (its
# design, calibration, board, canvas and placement) as it stands.
function _target_body_v010(b::AbstractDict)
    t = Dict{String, Any}(b)
    ty = get(b, "type", nothing)
    if ty == "planar"
        t["type"] = _PLANAR_TARGET_TAG
    elseif ty == "points"
        t["type"] = _POINT_TARGET_TAG
        pts = b["points"]
        covs = pop!(t, "point_covariances", nothing)
        (covs === nothing || length(covs) == length(pts)) ||
            _not_written_by("an IO 0.9 points target", "has one `point_covariances` entry per point ($(length(covs)) for $(length(pts)) points)")
        t["points"] = Any[
            merge(Dict{String, Any}((length(p) == 2 ? "xy" : "xyz") => p), covs === nothing ? Dict{String, Any}() : Dict{String, Any}("covariance" => covs[i]))
                for (i, p) in enumerate(pts)
        ]
    else
        _not_written_by("an IO 0.9 target body", "is of type $(repr(ty)), not \"points\" or \"planar\"")
    end
    return t
end

# vgio/rig/v9 → v10.0 (IO 0.10): the envelope (a rig with sensors declared
# its `pixel_frame`, one without none: colmap); the frame's name; every
# sensor's body (`_camera_body_v010`) and every target's (`_target_body_v010`).
# Its identity is re-minted (`content_id`, JCS over the 0.10 identity tree):
# `vgio_migrate_tree` maps the 0.9 one (`_v09_id`) to it.
function _rig_v9_to_v10(old::AbstractDict)
    t = _envelope_v010(old, "vgio/rig/v10.0")
    t["frame"] = _frame_name_v010(t["frame"])
    haskey(t, "sensors") && (t["sensors"] = Any[merge(s, Dict{String, Any}("camera" => _camera_body_v010(s["camera"]))) for s in t["sensors"]])
    haskey(t, "targets") && (t["targets"] = Any[merge(x, Dict{String, Any}("target" => _target_body_v010(x["target"]))) for x in t["targets"]])
    return t
end
migrate_schema(::Val{Symbol("vgio/rig/v9")}, t::AbstractDict) = _rig_v9_to_v10(t)

# vgio/target/v5 → v6.0: the envelope and the body; its `calibration_record` as it stands.
function _target_v5_to_v6(old::AbstractDict)
    t = _envelope_v010(old, "vgio/target/v6.0")
    t["target"] = _target_body_v010(t["target"])
    return t
end
migrate_schema(::Val{Symbol("vgio/target/v5")}, t::AbstractDict) = _target_v5_to_v6(t)

# --- the pose and capture kinds -------------------------------------------------

# A 0.9 `time_origin` is an ISO 8601 date-time WITHOUT a zone; 0.10 states its
# UTC offset (instants written on different machines compare). The offset is
# not in the file, and no published file holds a `time_origin` (the stored
# records of 2026-11 hold none), so none is guessed: such a file is refused by
# name, naming its producer — the rule for a version no published file holds
# (IO's CLAUDE.md, "Schema versions").
function _time_origin_v010(t::AbstractDict, s::AbstractString)
    o = get(t, "time_origin", nothing)
    (o === nothing || (o isa AbstractString && occursin(_TIME_ORIGIN_RE, o))) && return t
    throw(ArgumentError(
        "a $(s) record's time_origin $(repr(o)) states no UTC offset, which every record from IO 0.10 on states (instants " *
            "compare across machines); the offset is not in the file and is not guessed, and no published file holds a " *
            "time_origin: re-export the file from its producer with the offset (2026-05-27T09:30:00+02:00, or …Z)"))
end

# vgio/transforms/v4 → v5.0: the envelope; each record's frame names.
function _transforms_v4_to_v5(old::AbstractDict)
    t = _envelope_v010(old, "vgio/transforms/v5.0")
    t["transforms"] = Any[_frame_names_v010(r, ("from", "to")) for r in t["transforms"]]
    return t
end
migrate_schema(::Val{Symbol("vgio/transforms/v4")}, t::AbstractDict) = _transforms_v4_to_v5(t)

# vgio/trajectory/v4 → v5.0: the envelope; `from`/`to`; `t` → `frame_index`; a
# `time_origin` states its offset (`_time_origin_v010`).
_trajectory_v4_to_v5(old::AbstractDict) =
    _frames_v010(_frame_names_v010(_time_origin_v010(_envelope_v010(old, "vgio/trajectory/v5.0"), "vgio/trajectory/v4"), ("from", "to")))
migrate_schema(::Val{Symbol("vgio/trajectory/v4")}, t::AbstractDict) = _trajectory_v4_to_v5(t)

# vgio/target_poses/v2 → v3.0: the envelope; `to`; `t` → `frame_index`; the time origin.
_target_poses_v2_to_v3(old::AbstractDict) =
    _frames_v010(_frame_names_v010(_time_origin_v010(_envelope_v010(old, "vgio/target_poses/v3.0"), "vgio/target_poses/v2"), ("to",)))
migrate_schema(::Val{Symbol("vgio/target_poses/v2")}, t::AbstractDict) = _target_poses_v2_to_v3(t)

# vgio/target_homographies/v1 → v2.0: the envelope (its `pixel_frame` the
# units' `pixel`); `t` → `frame_index`; the time origin. Its `targets` name the
# definitions by their 0.9 identities: `vgio_migrate_tree` repoints them.
_target_homographies_v1_to_v2(old::AbstractDict) =
    _frames_v010(_time_origin_v010(_envelope_v010(old, "vgio/target_homographies/v2.0"), "vgio/target_homographies/v1"))
migrate_schema(::Val{Symbol("vgio/target_homographies/v1")}, t::AbstractDict) = _target_homographies_v1_to_v2(t)

# --- the observation kinds ------------------------------------------------------
#
# vgio/target_points2d/v2, vgio/target_points3d/v2 and vgio/target_detections/v2
# → vgio/observations/v3.0 (the one observation kind; Core 0.10's
# `TargetObservations`, one container for pixels and 3D points): the envelope
# (`point_index_base`, 1 in every 0.9 file, is the conventions' `index_base`;
# a 3D file's `frame` named as 0.10 names it); `entity` "target_point"; per
# frame (`t` → `frame_index`) one SET per target — its point indices, its
# points in the one point vocabulary (`{"xy" | "xyz", "covariance"?}`) and its
# `weights` column where it has one. A points2d point's `weight` (every one has
# one) becomes the set's `weights` column, a detections set's `covariance` (a
# list, one per point) each point's own. The sets of a frame are listed in the
# order the file first names their targets, each set's points in file order.
# A record a `TargetObservations` cannot hold is refused naming the frame, the
# target and the index: a (target, index) observed twice in one frame (a set is
# keyed by its indices), and plain and `Uncertain` points in one file (one
# container holds one point type).

# The observation sets of one 0.9 frame, `points` its per-point records (`point(p)`
# the 0.10 point of a record, `weight(p)` its weight or `nothing`), grouped by target.
function _observation_sets_v010(points, f, point, weight)
    sets = Dict{String, Dict{String, Any}}()
    order = String[]
    for p in points
        id = String(p["target"])
        s = get!(sets, id) do
            push!(order, id)
            Dict{String, Any}("target" => id, "index" => Any[], "points" => Any[])
        end
        i = p["index"]
        i in s["index"] && _duplicate_observation(f, id, i)
        push!(s["index"], i)
        push!(s["points"], point(p))
        w = weight(p)
        w === nothing || push!(get!(s, "weights", Any[]), w)
    end
    return Any[sets[id] for id in order]
end
_duplicate_observation(f, id, i) = throw(ArgumentError(
    "frame $(f["t"]) observes point $(i) of target $(id) twice; an observations set holds one observation per (target, index) " *
        "in a frame (VisualGeometryToolkitCore's TargetObservations), so this file is not carried over: re-export it with one per point"))

# Every point of a record (its observation sets) states a covariance, or none does.
function _check_one_point_type(t::AbstractDict, s::AbstractString)
    seen = Dict{Bool, Any}()       # covariance stated => the first (frame, target, index) that does / does not
    for f in t["frames"], r in f["observations"], (k, p) in enumerate(r["points"])
        get!(seen, haskey(p, "covariance"), (f["frame_index"], r["target"], r["index"][k]))
    end
    length(seen) == 2 || return t
    (fp, tp, ip), (fu, tu, iu) = seen[false], seen[true]
    throw(ArgumentError(
        "a $(s) file holds plain and Uncertain points: frame $(fp) target $(tp) index $(ip) is plain, frame $(fu) target $(tu) " *
            "index $(iu) states a covariance; one TargetObservations holds one point type, so this file is not carried over: " *
            "re-export it with every point's covariance (exact where known) or none"))
end

function _observations_v010(old::AbstractDict, s::AbstractString, frame_sets)
    t = _time_origin_v010(_envelope_v010(old, "vgio/observations/v3.0"), s)
    haskey(t, "frame") && (t["frame"] = _frame_name_v010(t["frame"]))
    t["entity"] = _TARGET_POINT_ENTITY
    t["frames"] = Any[_observation_frame_v010(f, frame_sets) for f in t["frames"]]
    return _check_one_point_type(t, s)
end
function _observation_frame_v010(f::AbstractDict, frame_sets)
    n = Dict{String, Any}(k => v for (k, v) in f if !(k in ("t", "points", "detections")))
    n["frame_index"] = f["t"]
    n["observations"] = frame_sets(f)
    return n
end
_with_covariance(p::AbstractDict, k::AbstractString) =
    merge(Dict{String, Any}(k => p[k]), haskey(p, "covariance") ? Dict{String, Any}("covariance" => p["covariance"]) : Dict{String, Any}())
_points2d_v2_to_observations(old::AbstractDict) = _observations_v010(old, "vgio/target_points2d/v2",
    f -> _observation_sets_v010(f["points"], f, p -> _with_covariance(p, "xy"), p -> p["weight"]))
_points3d_v2_to_observations(old::AbstractDict) = _observations_v010(old, "vgio/target_points3d/v2",
    f -> _observation_sets_v010(f["points"], f, p -> _with_covariance(p, "xyz"), p -> nothing))
# A detections set is already a set: its `points` arrays in the point
# vocabulary, its per-point `covariance` list onto its points; checked as the
# points of the other two are.
function _detections_v2_to_observations(old::AbstractDict)
    key = haskey(old, "frame") ? "xyz" : "xy"
    expand(r) = (covs = get(r, "covariance", nothing);
        Any[merge(Dict{String, Any}("target" => r["target"], "index" => i, key => r["points"][k]),
                  covs === nothing ? Dict{String, Any}() : Dict{String, Any}("covariance" => covs[k]),
                  haskey(r, "weights") ? Dict{String, Any}("weight" => r["weights"][k]) : Dict{String, Any}())
            for (k, i) in enumerate(r["index"])])
    return _observations_v010(old, "vgio/target_detections/v2",
        f -> _observation_sets_v010(reduce(vcat, (expand(r) for r in f["detections"]); init = Any[]), f,
                                    p -> _with_covariance(p, key), p -> get(p, "weight", nothing)))
end
migrate_schema(::Val{Symbol("vgio/target_points2d/v2")}, t::AbstractDict) = _points2d_v2_to_observations(t)
migrate_schema(::Val{Symbol("vgio/target_points3d/v2")}, t::AbstractDict) = _points3d_v2_to_observations(t)
migrate_schema(::Val{Symbol("vgio/target_detections/v2")}, t::AbstractDict) = _detections_v2_to_observations(t)

# --- capture settings, the manifest, the radiometric kinds, the units sidecar -----

# vgio/capture_settings/v1 → v2.0: the envelope (`time_unit` "s" the units'
# `time`); the body's and the lens's identities (`make`, `model`, `serial`)
# under `device`, the ONE device-identity encoding, their `source`s under
# `identity_sources`, the lens's `exif` claim as `exif_lens`; `t` → `frame_index`.
function _capture_settings_v1_to_v2(old::AbstractDict)
    t = _frames_v010(_envelope_v010(old, "vgio/capture_settings/v2.0"))
    body, lens = pop!(t, "body"), pop!(t, "lens")
    identity(x) = Dict{String, Any}(k => v for (k, v) in x if k in ("make", "model", "serial"))
    t["device"] = Dict{String, Any}("body" => identity(body), "lens" => identity(lens))
    t["identity_sources"] = Dict{String, Any}("body" => body["source"], "lens" => lens["source"])
    haskey(lens, "exif") && (t["exif_lens"] = lens["exif"])
    return t
end
migrate_schema(::Val{Symbol("vgio/capture_settings/v1")}, t::AbstractDict) = _capture_settings_v1_to_v2(t)

# vgio/calibration/v2 → v3.0: the envelope; `kind` → `result_kind`, a file's
# `kind` → `schema` (the files' schemas as they stand: a listed file is bound by
# its bytes, its hash and schema the file's own); `metrics` is closed (name →
# finite number), so an entry that is not a number — text, a list, a nested
# object, which 0.9's free-form `metrics` held — moves, as it stands, under
# `provenance.metrics` (the one free-form object), where nothing reads it.
function _calibration_v2_to_v3(old::AbstractDict)
    t = _renamed_key(_envelope_v010(old, "vgio/calibration/v3.0"), "kind", "result_kind")
    t["files"] = Any[_renamed_key(f, "kind", "schema") for f in t["files"]]
    if haskey(t, "metrics")
        ms = pop!(t, "metrics")
        numbers = Dict{String, Any}(k => v for (k, v) in ms if VgtkCore.is_finite_real(v))
        rest = Dict{String, Any}(k => v for (k, v) in ms if !haskey(numbers, k))
        isempty(numbers) || (t["metrics"] = numbers)
        if !isempty(rest)
            prov = get!(t, "provenance", Dict{String, Any}())
            haskey(prov, "metrics") && _not_written_by("a vgio/calibration/v2 manifest whose `metrics` hold non-numbers",
                "holds no `provenance.metrics` (the step moves those entries there)")
            prov["metrics"] = rest
        end
    end
    return t
end
migrate_schema(::Val{Symbol("vgio/calibration/v2")}, t::AbstractDict) = _calibration_v2_to_v3(t)

# vgio/vignetting/v1 → v2.0: the envelope (v1's `pixel_frame`, always colmap);
# a radial profile states its chart: absent, the linear chart v1 read it in
# (IO 0.8's body); present — IO 0.9 wrote a non-linear profile's `chart` inside
# v1 without a version bump (the review's I1) — as it stands. A log field's
# spline is a body too.
_vignetting_body_v010(b::AbstractDict) =
    b["type"] == "log" ? merge(b, Dict{String, Any}("spline" => _vignetting_body_v010(b["spline"]))) :
    (b["type"] == "radial" && !haskey(b, "chart")) ? merge(b, Dict{String, Any}("chart" => "linear")) : b
function _vignetting_v1_to_v2(old::AbstractDict)
    t = _envelope_v010(old, "vgio/vignetting/v2.0")
    t["vignetting"] = _vignetting_body_v010(t["vignetting"])
    return t
end
migrate_schema(::Val{Symbol("vgio/vignetting/v1")}, t::AbstractDict) = _vignetting_v1_to_v2(t)

# vgio/environment_light/v1 → v2.0: the envelope (its coefficients' radiance
# "relative", as v1 left them in the caller's units); `l_max` → `max_degree`, Core's name.
function _environment_light_v1_to_v2(old::AbstractDict)
    t = _envelope_v010(old, "vgio/environment_light/v2.0")
    t["map"] = _renamed_key(t["map"], "l_max", "max_degree")
    return t
end
migrate_schema(::Val{Symbol("vgio/environment_light/v1")}, t::AbstractDict) = _environment_light_v1_to_v2(t)

# vgio/units/v2 → v3.0: the sidecar IS the envelope; v2's `world` (a length unit
# or "unknown") its `units.length`, its `density` (absent: dpi, as v2 read it)
# its `units.density`.
function _units_v2_to_v3(old::AbstractDict)
    t = _envelope_v010(old, "vgio/units/v3.0"; length = old["world"], density = get(old, "density", "dpi"))
    delete!(t, "world")
    delete!(t, "density")
    return t
end
migrate_schema(::Val{Symbol("vgio/units/v2")}, t::AbstractDict) = _units_v2_to_v3(t)

# =============================================================================
# THE FROZEN IO 0.9 IDENTITY: the content_id IO 0.9 minted for a rig or a target
# =============================================================================
#
# IO 0.10 re-minted every identity ONCE (the owner's ruling, 2026-11-13): JCS
# over the 0.10 identity tree, lengths normalised in metres (rig_schema.jl,
# `content_id`). Observation files written before name their rigs and targets
# by the 0.9 identity, so `vgio_migrate_tree` needs it for every rig and target
# file it migrates, to map it to the new one. It is computed HERE, from the
# value (loaded by the 0.10 reader through the steps, which move no number), as
# IO 0.9.3 computed it (rig_schema.jl at 2f994d2b, `_identity_tree`, `_id_*`,
# canonical.jl) — FROZEN: nothing here may follow a later change of the live
# writers or of `canonical_json`, or the old identities stop matching.
#
# IO 0.9's tree, from the value:
#   rig     {"frame" (0.9's frame names: `camera_optical`), "sensors"? [{"id", "camera", "extrinsics"}] (rig order),
#            "targets"? [{"id", "target", "pose"}] (by id)}
#   pose    {"rotation" [w, x, y, z], "translation", "covariance"?}
#   camera  {"intrinsics" [fx, fy, cx, cy, skew] (canonical px), "extent" [w, h], "pitch"? [w, h],
#            "lens" (VGC's lens record, a composition's stages under "chain" in application order;
#                    a SplineLens's 0.9 spline record {"area", "grid_size" [gw, gh], "present"? (0-based runs),
#                    "controls", "forward"?}) | "noncentral_spline" {area, grid_size, present?, directions, points},
#            "lens_coverage"?}
#   target  {"kind": "points", "points", "point_covariances"?}
#         | {"kind": "planar", "reference" | "nominal" (a layout target's design points), "calibration",
#            "board"? {"size" [w, h], "origin"}, "canvas"? {"size" [w, h], "origin", "texture"? {…, "size" [w, h]}},
#            "placement"?}
#   calibration {"kind", "parameters"?, "nodes"?, a composition {"kind": "calibration_chain", "members"
#               (application order)}, "covariance"? (permuted to that order)}
# every length the metre value a MILLIMETRE file holds for it and reads back as
# (0.9's `_persisted`, `_persisted_v09`), every covariance entry likewise at its
# length power; serialized by 0.9's canonical JSON (`_canonical_json_v09`: keys
# sorted bytewise, an integral float printed as an integer, any other by Julia's
# shortest round-trip `print`).
#
# It is built from the 0.10 writer's record of the value in millimetres (whose
# numbers are exactly the millimetre numbers 0.9's `_persisted` read back), each
# body taken back to its 0.9 shape (`_record_v09`, the inverse of the 0.10 steps
# above), each length leaf read back from its millimetre number (`_leaf_v09`).
# The test pins it on the identities IO 0.9 pinned (migrate_identity_tests.jl).

# 0.9's persisted form of a base-unit number of length power `n` (lengths.jl at IO 0.9.3).
_persisted_v09(y::Real, n::Int = 1) = _read_power(_write_power(Float64(y), mm, n), mm, n)
# …the same, of the millimetre number `x` a millimetre file holds for it.
_leaf_v09(x::Real, n::Int) = _read_power(x, mm, n)
const _V09_MM = VgioStyle(; length_unit = mm)

# IO 0.9's canonical JSON (canonical.jl at IO 0.9.3): keys sorted bytewise, an
# integral Float64 printed as an integer, any other by Julia's `print`.
function _canonical_json_v09(io::IO, x)
    if x isa Float64
        isfinite(x) || throw(ArgumentError("non-finite number in canonical tree"))
        (isinteger(x) && abs(x) <= 9.007199254740992e15) ? print(io, Int64(x)) : print(io, x)
    elseif x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            _canonical_json_v09(io, v)
        end
        print(io, ']')
    elseif x isa AbstractDict
        print(io, '{')
        for (i, k) in enumerate(sort!(collect(String, keys(x))))
            i > 1 && print(io, ',')
            _canonical_string(io, k)
            print(io, ':')
            _canonical_json_v09(io, x[k])
        end
        print(io, '}')
    else
        canonical_json(io, x)          # null, a Bool, an Int64, a string: printed alike
    end
    return nothing
end
_hash_v09(tree) = sha256_hex(sprint(_canonical_json_v09, tree))

# --- the 0.9 shapes of a 0.10 record (`leaf(x, n)` each millimetre number of length power n) ---

_frame_name_v09(s) = s == "camera" ? "camera_optical" : s
_size_v09(d::AbstractDict) = Any[d["width"], d["height"]]
_runs_v09(runs) = Any[Any[e[1] - 1, Any[Any[r[1] - 1, r[2] - 1] for r in e[2]]] for e in runs]
_lengths_v09(v, leaf) = Any[leaf(x, 1) for x in v]
function _cov_v09(c, dims, leaf)
    c isa AbstractVector && return Any[Any[leaf(c[i][j], dims[i] + dims[j]) for j in eachindex(c[i])] for i in eachindex(c)]
    haskey(c, "diagonal") && return Dict{String, Any}("diagonal" => Any[leaf(x, 2 * dims[i]) for (i, x) in enumerate(c["diagonal"])])
    return Dict{String, Any}("isotropic" => leaf(c["isotropic"], 2 * first(dims)))
end
function _pose_v09(p::AbstractDict, leaf)
    q = Dict{String, Any}("rotation" => p["rotation"], "translation" => _lengths_v09(p["translation"], leaf))
    haskey(p, "covariance") && (q["covariance"] = _cov_v09(p["covariance"], _POSE_COV_DIMS, leaf))
    return q
end
# a lattice record as 0.9's spline record, its control arrays `names` (0.10 name => 0.9 key)
function _lattice_v09(l::AbstractDict, names)
    d = Dict{String, Any}("area" => l["area"], "grid_size" => _size_v09(l["grid_size"]))
    haskey(l, "present") && (d["present"] = _runs_v09(l["present"]))
    for (from, to) in names
        d[to] = l["controls"][from]
    end
    return d
end
function _lens_v09(l::AbstractDict)
    haskey(l, "spline") || return l
    sp = l["spline"]
    n = _lattice_v09(sp, ("direction" => "controls",))
    haskey(sp, "forward") && (n["forward"] = _lattice_v09(sp["forward"], ("point" => "controls",)))
    return Dict{String, Any}("spline" => n)
end
function _camera_body_v09(b::AbstractDict, leaf)
    n = Dict{String, Any}()
    for (k, v) in b
        if k == "image_size"
            n["width"], n["height"] = v["width"], v["height"]
        elseif k == "pitch"
            n["pitch"] = Any[leaf(v["width"], 1), leaf(v["height"], 1)]
        elseif k == "lens"
            n["lens"] = _lens_v09(v)
        elseif k == "spline"         # a non-central body's
            n["spline"] = merge(_lattice_v09(v, ("direction" => "directions",)),
                                Dict{String, Any}("points" => Any[_lengths_v09(p, leaf) for p in v["controls"]["point"]]))
        else
            n[k] = v
        end
    end
    return n
end
_calibration_mask_v09(r::AbstractDict) = Int[x for x in length_mask(planar_kind(TARGET_CALIBRATION_TAGS[r["type"]], length(get(r, "nodes", ()))))]
function _calibration_v09(c::AbstractDict, leaf)
    out = Dict{String, Any}(c)
    if c["type"] == _COMPOSITION_TYPE
        out[_COMPOSE_KEY] = Any[_calibration_v09(r, leaf) for r in c[_COMPOSE_KEY]]
        mask = reduce(vcat, (_calibration_mask_v09(r) for r in c[_COMPOSE_KEY]); init = Int[])
    else
        mask = _calibration_mask_v09(c)
        haskey(c, "nodes") && (out["nodes"] = Any[_lengths_v09(p, leaf) for p in c["nodes"]])
        haskey(c, "parameters") && (out["parameters"] = Any[mask[i] == 1 ? leaf(x, 1) : x for (i, x) in enumerate(c["parameters"])])
    end
    haskey(c, "covariance") && (out["covariance"] = _cov_v09(c["covariance"], mask, leaf))
    return out
end
_rect_v09(r::AbstractDict, leaf) = merge(r, Dict{String, Any}(
    "size" => Dict{String, Any}("width" => leaf(r["size"]["width"], 1), "height" => leaf(r["size"]["height"], 1)),
    "origin" => _lengths_v09(r["origin"], leaf)))
# A 0.10 target body as 0.9's, `t` its value (a layout target's design points
# where `layout_points`: 0.9's identity held them, not the layout).
function _target_body_v09(b::AbstractDict, t, leaf, layout_points::Bool)
    n = Dict{String, Any}(b)
    if b["type"] == _POINT_TARGET_TAG
        n["type"] = "points"
        n["points"] = Any[_lengths_v09(haskey(p, "xy") ? p["xy"] : p["xyz"], leaf) for p in b["points"]]
        if !isempty(b["points"]) && haskey(first(b["points"]), "covariance")
            d = ntuple(_ -> 1, length(first(n["points"])))
            n["point_covariances"] = Any[_cov_v09(p["covariance"], d, leaf) for p in b["points"]]
        end
        return n
    end
    n["type"] = "planar"
    haskey(b, "nominal") && (n["nominal"] = Any[_lengths_v09(p, leaf) for p in b["nominal"]])
    if layout_points && haskey(b, "layout")
        delete!(n, "layout")
        n["nominal"] = Any[Any[leaf(write_length(c, mm), 1) for c in p] for p in design(t)]
    end
    placed = haskey(b, "placement")
    n["calibration"] = _calibration_v09(b["calibration"], placed ? (x, k) -> x : leaf)   # a placed one: chart numbers
    haskey(b, "board") && (n["board"] = _rect_v09(b["board"], leaf))
    if haskey(b, "canvas")
        cv = _rect_v09(b["canvas"], leaf)
        haskey(cv, "texture") && (cv["texture"] = _rect_v09(cv["texture"], leaf))
        n["canvas"] = cv
    end
    placed && (n["placement"] = Dict{String, Any}("origin" => _lengths_v09(b["placement"]["origin"], leaf), "side" => leaf(b["placement"]["side"], 1)))
    return n
end
# The 0.10 writer's millimetre record of a value, as plain JSON.
_mm_record(x) = _lower(JSON.parse(JSON.json(x; style = _V09_MM)))
# A rig's body as IO 0.9's rig record held it (its frame, sensors and targets;
# no envelope), every millimetre number through `leaf`.
function _rig_record_v09(rig::Rig, leaf, layout_points::Bool)
    b = _mm_record(_rig_body(rig, _V09_MM))
    r = Dict{String, Any}("frame" => _frame_name_v09(b["frame"]))
    haskey(b, "sensors") && (r["sensors"] = Any[
        Dict{String, Any}("id" => s["id"], "camera" => _camera_body_v09(s["camera"], leaf), "extrinsics" => _pose_v09(s["extrinsics"], leaf))
            for s in b["sensors"]])
    if haskey(b, "targets")
        tr = target_rig(rig)
        r["targets"] = Any[
            Dict{String, Any}("id" => x["id"], "target" => _target_body_v09(x["target"], tr[Symbol(x["id"])].target, leaf, layout_points),
                              "pose" => _pose_v09(x["pose"], leaf))
                for x in b["targets"]]
    end
    return r
end

# --- the 0.9 identity tree (IO 0.9.3's `_id_*`) -----------------------------------

function _id_kind_v09!(d::AbstractDict)
    d["kind"] = pop!(d, "type")
    haskey(d, "calibration") && _id_kind_v09!(d["calibration"])
    foreach(_id_kind_v09!, get(d, _COMPOSE_KEY, ()))
    return d
end
_id_size_v09!(r) = (r["size"] = _size_v09(r["size"]); r)
function _id_target_v09(b::AbstractDict)
    d = Dict{String, Any}(b)
    haskey(d, "board") && (d["board"] = _id_size_v09!(Dict{String, Any}(d["board"])))
    if haskey(d, "canvas")
        cv = _id_size_v09!(Dict{String, Any}(d["canvas"]))
        haskey(cv, "texture") && (cv["texture"] = _id_size_v09!(Dict{String, Any}(cv["texture"])))
        d["canvas"] = cv
    end
    _id_kind_v09!(d)
    haskey(d, "calibration") && (d["calibration"] = _composition_v3_to_v2(d["calibration"], "kind"))   # application order, as v5
    return d
end
function _id_camera_v09(c::AbstractDict)
    d = Dict{String, Any}("intrinsics" => Any[Float64(c["K"][k]) for k in ("fx", "fy", "cx", "cy", "skew")],
                          "extent" => Any[c["width"], c["height"]])
    haskey(c, "pitch") && (d["pitch"] = c["pitch"])
    haskey(c, "type") ? (d["noncentral_spline"] = c["spline"]) : (d["lens"] = _lens_v4_to_v3(c["lens"]))
    haskey(c, "lens_coverage") && (d["lens_coverage"] = c["lens_coverage"])
    return d
end
function _identity_tree_v09(rig::Rig)
    r = _rig_record_v09(rig, _leaf_v09, true)
    d = Dict{String, Any}("frame" => r["frame"])
    haskey(r, "sensors") && (d["sensors"] = Any[
        Dict{String, Any}("id" => s["id"], "camera" => _id_camera_v09(s["camera"]), "extrinsics" => s["extrinsics"]) for s in r["sensors"]])
    haskey(r, "targets") && (d["targets"] = Any[
        Dict{String, Any}("id" => x["id"], "target" => _id_target_v09(x["target"]), "pose" => x["pose"]) for x in r["targets"]])
    return d
end
_identity_tree_v09(t::AbstractTarget) = _id_target_v09(_target_body_v09(_mm_record(JSON.lower(_V09_MM, t)), t, _leaf_v09, true))

"""
    _v09_id(x) -> String

The content identity IO 0.9 minted for a rig or an unposed target (`x` a
`Rig`, an `AbstractTarget`, an `Attributed` one by its target): FROZEN (see
above). An observation file written before IO 0.10 names its definitions by it.
"""
_v09_id(x::Union{Rig, AbstractTarget}) = _hash_v09(_identity_tree_v09(x))
_v09_id(t::Attributed{<:AbstractTarget}) = _v09_id(t.value)
