# =============================================================================
# export/correspondences/v2 — an EXPORT: a session's 2D–3D pairs in the shape
# external calibrators take (write-only, like TUM / EuRoC / BOP / TRC)
# =============================================================================
#
# The observations (VGC `TargetObservations{F}`, pixels) and the targets (a
# vgio/rig file, a VGC `Rig` of targets) are COLMAP's points2D → points3D
# decomposition: a pixel and a reference into a target's point list. Feeding a
# third-party calibrator — cv::calibrateCamera, MATLAB's
# estimateCameraParameters, the python side — means resolving the references
# into point arrays and moving the pixels into that library's frame. This is
# that resolution, once, so no driver script redoes the pixel-origin
# arithmetic. It is not a VGC value of its own (it is derived from two that
# are), so VGIo writes it and never reads it back: its schema is not a `vgio/`
# kind (no reader), `export/correspondences/v2`, and its file states its units
# in the one `units` object every file has.
#
# ONE ROW PER (FRAME, TARGET): object points are in their target's own frame,
# so the points of two boards seen in one frame are two views of two planar
# targets, never one array.
#
# {"schema": "export/correspondences/v2", <envelope: units.pixel the calibrator's frame, units.length>,
#  "rows": [{"frame_index": 1, "image": "image_1.png", "sensor": "cam1", "target": "a1f2",
#            "object_points": [[x, y, z], …], "image_points": [[u, v], …]}],
#  "provenance": {…}}                                               (optional)

const _CORRESPONDENCES_SCHEMA = "export/correspondences/v2"
const _CORRESPONDENCES_SPEC = _CheckedSpec(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _Tok(_CORRESPONDENCES_SCHEMA), :generator => _GENERATOR, :units => _UNITS_OBJECT, :conventions => _CONVENTIONS_OBJECT,
            :rows => _Vec(
                _CheckedSpec(
                    _Obj(
                        :frame_index => _Int(), :image => _NonEmpty(), :sensor => _NonEmpty(), :target => _NonEmpty(),
                        :object_points => _Vec(_XYZ), :image_points => _Vec(_XY),
                    ),
                    r -> length(r[:object_points]) == length(r[:image_points]) ? nothing :
                        "object_points and image_points must pair one to one",
                )
            ),
        ],
        _PROVENANCE,
    ),
    x -> _unique_check([(r[:frame_index], r[:target]) for r in x[:rows]], "(frame_index, target) rows"),
)

# One (frame, target) row: object points as plain METRES (read from the
# targets' lengths), image points in pixels; the file holds the object points in
# its `length_unit`.
const _CorrespondenceRow = @NamedTuple{
    t::Int, image::String, sensor::String, target::Symbol,
    object_points::Matrix{Float64}, image_points::Matrix{Float64},
}

# The rows of `obs` resolved against `rigs`, image points moved into `G`.
function _correspondence_rows(::Type{G}, obs::TargetObservations{F}, rigs) where {G <: PixelFrame, F <: PixelFrame}
    _check_resolves(obs, rigs)
    points = Dict{Symbol, Vector}()
    for rig in rigs, id in ids(rig)
        is_resolved(rig[id].target) || continue        # not observed: `_check_resolves` refused an observed one
        pts = VgtkCore.points(rig[id].target)
        if haskey(points, id)
            points[id] == pts || throw(ArgumentError("target $id carries different points in two rigs"))
        else
            points[id] = pts
        end
    end
    shift = EuclideanMap(F, G)
    out = _CorrespondenceRow[]
    for t in sort!(collect(keys(obs))), id in sort!(collect(keys(obs[t])))
        cs = obs[t][id]
        n = length(cs)
        X = Matrix{Float64}(undef, n, 3)
        U = Matrix{Float64}(undef, n, 2)
        for (i, (j, xy)) in enumerate(zip(sources(cs), targets(cs)))
            p = geom(points[id][j])
            has_units(p) || _refuse_plain("save_correspondences: target $(id)'s points")
            X[i, 1] = ustrip(m, p[1]); X[i, 2] = ustrip(m, p[2]); X[i, 3] = length(p) >= 3 ? ustrip(m, p[3]) : 0.0
            u = shift(SVector{2, Float64}(geom(xy)))
            U[i, 1] = u[1]; U[i, 2] = u[2]
        end
        push!(out, (; t, image = get(obs.images, t, ""), sensor = get(obs.sensors, t, ""), target = id, object_points = X, image_points = U))
    end
    return out
end

function _write_correspondences(
        path::AbstractString, ::Type{G}, rows; length_unit::Units = mm, provenance::NamedTuple = (;)
    ) where {G <: PixelFrame}
    style = VgioStyle(; length_unit, pixel_frame = G)
    body = (
        schema = _CORRESPONDENCES_SCHEMA, _envelope_tail(style)...,
        rows = [
            (
                frame_index = r.t, image = r.image, sensor = r.sensor, target = String(r.target),
                object_points = [
                    Float64[_write_power(r.object_points[i, k], length_unit, 1) for k in 1:3] for i in axes(r.object_points, 1)
                ],
                image_points = [collect(Float64, view(r.image_points, i, :)) for i in axes(r.image_points, 1)],
            ) for r in rows
        ],
    )
    text = _json_text(with_provenance(body, provenance), style)
    _validate(_CORRESPONDENCES_SPEC, _parse_vgio_text(text, path), path, "")
    write(path, text)
    return path
end

"""
    save_correspondences([to_frame = F,] path, obs::TargetObservations{F}, targets...; resolve = nothing, length_unit = mm, provenance = (;)) -> path

Export a session's 2D–3D correspondences — pixel observations `obs` of target
points — as one `export/correspondences/v2` file, one row per (frame, target)
in frame then target order: `object_points` N×3 in the target's own frame (the
target's points on its surface: z = 0 for a planar calibration, the height a
`HeightField` gives otherwise; in `length_unit`, from the targets' lengths —
plain points are an `ArgumentError`) and `image_points` N×2, moved into
`to_frame` — the external calibrator's convention, declared in the file's
`units.pixel`: `PixelFrame{:opencv}` for `cv::calibrateCamera` (the top-left
pixel's centre at (0, 0)), `PixelFrame{:julia}` for MATLAB's (1, 1); the
observations' own frame by default. A type, it leads.

`targets` is one or more rigs or rig file paths (one per connected
component; a target present in several must carry the same points in each;
`resolve` resolves a referenced planar design read from a path); when `obs`
names its targets (`obs.targets`), the given ones must be exactly those, by
[`content_id`](@ref). Every (target, index) must resolve, and every frame must
name its image and sensor. The export is write-only: no reader reads it.
"""
save_correspondences(path::AbstractString, obs::TargetObservations{F}, targets...; kwargs...) where {F <: PixelFrame} =
    save_correspondences(F, path, obs, targets...; kwargs...)

function save_correspondences(
        ::Type{G}, path::AbstractString, obs::TargetObservations{<:PixelFrame}, targets...;
        resolve = nothing, length_unit::Units = mm, provenance::NamedTuple = (;)
    ) where {G <: PixelFrame}
    isempty(targets) && throw(ArgumentError("save_correspondences needs at least one targets rig or file"))
    list = collect(Any, targets)
    if !isempty(obs.targets)
        given = sort!(_target_hashes(list))
        sort(obs.targets) == given || throw(
            ArgumentError("the observations resolve against the targets $(sort(obs.targets)); the targets given are $(given)")
        )
    end
    return _write_correspondences(path, G, _correspondence_rows(G, obs, _target_rigs(list; resolve)); length_unit, provenance)
end
