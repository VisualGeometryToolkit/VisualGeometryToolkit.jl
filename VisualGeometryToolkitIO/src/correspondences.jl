# =============================================================================
# correspondences/v2 — an EXPORT: a session's 2D–3D pairs in the shape external
# calibrators take (write-only, like TUM / EuRoC / BOP / TRC)
# =============================================================================
#
# The measurements (target_points2d/v1, VGC `TargetPoints2D`) and the targets
# (rig/v5, a VGC `Rig` of targets) are COLMAP's points2D → points3D
# decomposition: a pixel and a reference into a target's point list. Feeding a
# third-party calibrator — cv::calibrateCamera, MATLAB's
# estimateCameraParameters, the python side — means resolving the references
# into point arrays and moving the pixels into that library's frame. This is
# that resolution, once, so no driver script redoes the pixel-origin
# arithmetic. It is not a VGC value of its own (it is derived from two that
# are), so VGIo writes it and never reads it back as a kind.
#
# ONE ROW PER (FRAME, TARGET): object points are in their target's own frame,
# so the points of two boards seen in one frame are two views of two planar
# targets, never one array — v1 concatenated them, which a calibrator fed a
# multi-board frame could not tell apart.
#
# {"schema": "vgio/correspondences/v2", "pixel_frame": "opencv", "length_unit": "mm",
#  "rows": [{"t": 1, "image": "image_1.png", "sensor": "cam1", "target": "a1f2",
#            "object_points": [[x, y, z], …], "image_points": [[u, v], …]}],
#  "provenance": {…}}                                               (optional)

const _CORRESPONDENCES_SPEC = _Checked(
    _Obj(
        Pair{Symbol, _Spec}[
            :schema => _schema_tok("correspondences"), :pixel_frame => _PIXEL_FRAME, :length_unit => _LENGTH_UNIT,
            :rows => _Vec(
                _Checked(
                    _Obj(
                        :t => _Int(), :image => _NonEmpty(), :sensor => _NonEmpty(), :target => _NonEmpty(),
                        :object_points => _Vec(_XYZ), :image_points => _Vec(_XY),
                    ),
                    r -> length(r[:object_points]) == length(r[:image_points]) ? nothing :
                        "object_points and image_points must pair one to one",
                )
            ),
        ],
        _PROVENANCE,
    ),
    x -> _unique_check([(r[:t], r[:target]) for r in x[:rows]], "(t, target) rows"),
)

# One (frame, target) row: object points as plain METRES (read from the
# targets' lengths), image points in pixels; the file holds the object points in
# its `length_unit`.
const _CorrespondenceRow = @NamedTuple{
    t::Int, image::String, sensor::String, target::Symbol,
    object_points::Matrix{Float64}, image_points::Matrix{Float64},
}

# The rows of `obs` resolved against `rigs`, image points moved into `G`.
function _correspondence_rows(::Type{G}, obs::TargetPoints2D{F}, rigs) where {G <: PixelFrame, F <: PixelFrame}
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
    for t in sort!(collect(keys(obs)))
        by_target = Dict{Symbol, Vector{TargetPoint2D}}()
        for o in obs[t]
            push!(get!(() -> TargetPoint2D[], by_target, o.target), o)
        end
        for id in sort!(collect(keys(by_target)))
            os = by_target[id]
            X = Matrix{Float64}(undef, length(os), 3)
            U = Matrix{Float64}(undef, length(os), 2)
            for (i, o) in enumerate(os)
                p = geom(points[id][o.index])
                has_units(p) || _refuse_plain("save_correspondences: target $(id)'s points")
                X[i, 1] = ustrip(m, p[1]); X[i, 2] = ustrip(m, p[2]); X[i, 3] = length(p) >= 3 ? ustrip(m, p[3]) : 0.0
                u = shift(SVector{2, Float64}(geom(o.xy)))
                U[i, 1] = u[1]; U[i, 2] = u[2]
            end
            push!(out, (; t, image = get(obs.images, t, ""), sensor = get(obs.sensors, t, ""), target = id, object_points = X, image_points = U))
        end
    end
    return out
end

function _write_correspondences(
        path::AbstractString, ::Type{G}, rows; length_unit::Units = mm, provenance::NamedTuple = (;)
    ) where {G <: PixelFrame}
    body = (
        schema = _schema("correspondences"), pixel_frame = pixel_frame_name(G), length_unit = unit_string(length_unit),
        rows = [
            (
                t = r.t, image = r.image, sensor = r.sensor, target = String(r.target),
                object_points = [
                    Float64[_write_power(r.object_points[i, k], length_unit, 1) for k in 1:3] for i in axes(r.object_points, 1)
                ],
                image_points = [collect(Float64, view(r.image_points, i, :)) for i in axes(r.image_points, 1)],
            ) for r in rows
        ],
    )
    _write_store(path, _CORRESPONDENCES_SPEC, _with_provenance(body, provenance))
    return path
end

"""
    save_correspondences([to_frame = F,] path, obs::TargetPoints2D{F}, targets...; resolve = nothing, length_unit = mm, provenance = (;)) -> path

Export a session's 2D–3D correspondences as one `vgio/correspondences/v2`
file, one row per (frame, target) in frame then target order:
`object_points` N×3 in the target's own frame (a board's plane points with
z = 0, in `length_unit`, from the targets' lengths — plain points are an
`ArgumentError`) and `image_points` N×2, moved into `to_frame` — the
external calibrator's convention, declared in the file: `PixelFrame{:opencv}`
for `cv::calibrateCamera` (the top-left pixel's centre at (0, 0)),
`PixelFrame{:julia}` for MATLAB's (1, 1); the measurements' own frame by
default. A type, it leads.

`targets` is one or more rigs or `vgio/rig/v5` paths (one per connected
component; a target present in several must carry the same points in each;
`resolve` resolves a referenced planar design read from a path); when `obs` names its targets (`obs.targets`), the given ones must be
exactly those, by [`content_id`](@ref). Every (target, index) must resolve,
and every frame must name its image and sensor.
"""
save_correspondences(path::AbstractString, obs::TargetPoints2D{F}, targets...; kwargs...) where {F <: PixelFrame} =
    save_correspondences(F, path, obs, targets...; kwargs...)

function save_correspondences(
        ::Type{G}, path::AbstractString, obs::TargetPoints2D, targets...;
        resolve = nothing, length_unit::Units = mm, provenance::NamedTuple = (;)
    ) where {G <: PixelFrame}
    isempty(targets) && throw(ArgumentError("save_correspondences needs at least one targets rig or file"))
    list = collect(Any, targets)
    if !isempty(obs.targets)
        given = sort!(_target_hashes(list))
        sort(obs.targets) == given || throw(
            ArgumentError("the measurements resolve against the targets $(sort(obs.targets)); the targets given are $(given)")
        )
    end
    return _write_correspondences(path, G, _correspondence_rows(G, obs, _target_rigs(list; resolve)); length_unit, provenance)
end

# The file back, for the migration's verification only (correspondences is an
# export; no kind reads it): its frame and its rows, object points in plain metres.
function _read_correspondences(path::AbstractString)
    d = _read_store(path, "correspondences", _CORRESPONDENCES_SPEC)
    unit = parse_unit(d.length_unit::String)
    rows = _CorrespondenceRow[
        (
            t = Int(r.t::Integer), image = r.image::String, sensor = r.sensor::String, target = Symbol(r.target::String),
            object_points = _point_matrix(r.object_points::Vector{Any}, 3, p -> [_read_power(x, unit, 1) for x in p]),
            image_points = _point_matrix(r.image_points::Vector{Any}, 2, identity),
        ) for r in d.rows::Vector{Any}
    ]
    return pixel_frame_from_name(d.pixel_frame::String), rows
end

function _point_matrix(a, n::Int, f)
    M = Matrix{Float64}(undef, length(a), n)
    for (i, v) in enumerate(a)
        M[i, :] = f(Float64[Float64(c::Real) for c in v::Vector{Any}])
    end
    return M
end
